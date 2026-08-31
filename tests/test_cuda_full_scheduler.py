from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
LLAMA_LOGITS = ROOT / ".cache" / "authorities" / "llama-evidence" / "llama.f32le.bin"


def test_full_scheduler_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_scheduler_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_full_scheduler.json").read_text())
    tolerances = json.loads(
        (ROOT / "pins" / "scalar_oracle_tolerances.json").read_text()
    )["authorities"]["llama_vs_quartz"]["taps"]
    assert contract["model"]["bytes"] == 18_973_870_432
    assert contract["model"]["bound_tensors"] == 851
    assert contract["schedule"]["layers"] == 64
    assert contract["schedule"]["gdn_layers"] == 48
    assert contract["schedule"]["attention_layers"] == 16
    assert contract["schedule"]["expected_greedy"] == [3_649, 1_277]
    assert contract["admission"]["logits"] == {
        "maximum_absolute_error": tolerances["global.logits"]["admission"][
            "maximum_absolute_error"
        ],
        "maximum_rms_error": tolerances["global.logits"]["admission"][
            "maximum_root_mean_square_error"
        ],
        "minimum_cosine": tolerances["global.logits"]["admission"][
            "minimum_cosine_similarity"
        ],
        "nonfinite": 0,
    }
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert all(case["passed"] for case in fixture["scalar_logit_comparisons"])
    assert all(case["passed"] for case in fixture["scalar_tap_comparisons"])
    assert fixture["greedy"] == [3_649, 1_277]
    assert fixture["llama_same_gguf_comparisons"][0]["envelope_passed"]
    assert not fixture["llama_same_gguf_comparisons"][1]["envelope_passed"]
    assert all(case["greedy_equal"] for case in fixture["llama_same_gguf_comparisons"])
    chapter = (ROOT / "docs" / "46-cuda-full-scheduler.md").read_text().casefold()
    for term in [
        "resident",
        "memory map",
        "48 gdn",
        "16 attention",
        "residual accumulator",
        "scratch",
        "248,320 logits",
        "greedy",
        "negative",
        "proof boundary",
    ]:
        assert term in chapter


def test_full_scheduler_matches_scalar_taps_logits_and_greedy() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists() or not LLAMA_LOGITS.exists():
        pytest.skip("pinned model and llama.cpp full-logit evidence are required")
    common = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        IMAGE,
    ]
    build = subprocess.run(
        [*common, "make", "build/qw38-cuda-full-scheduler-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-full-scheduler-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            ".cache/authorities/llama-evidence/llama.f32le.bin",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    logits = [line for line in lines if line.startswith("scheduler_compare=logits ")]
    assert len(logits) == 2
    assert all(
        "nonfinite=0" in line and "first_over_absolute=none" in line for line in logits
    )
    greedy = [line for line in lines if line.startswith("scheduler_greedy=")]
    assert greedy == [
        "scheduler_greedy=row_0 cuda=3649 scalar=3649 equal=true",
        "scheduler_greedy=row_1 cuda=1277 scalar=1277 equal=true",
    ]
    taps = [
        line
        for line in lines
        if line.startswith(
            (
                "scheduler_compare=layer_0_residual ",
                "scheduler_compare=layer_3_residual ",
                "scheduler_compare=layer_63_residual ",
                "scheduler_compare=final_norm ",
            )
        )
    ]
    assert len(taps) == 8
    assert all(
        "nonfinite=0" in line and "first_over_absolute=none" in line for line in taps
    )
    run_line = next(line for line in lines if line.startswith("scheduler_run=full"))
    fields = dict(field.split("=", 1) for field in run_line.split())
    assert fields["layers"] == "64"
    assert fields["gdn_layers"] == "48"
    assert fields["attention_layers"] == "16"
    assert fields["frontier"] == "2"
    assert fields["model_bytes"] == "18973870432"
    assert float(fields["token0_ms"]) > 0.0
    assert float(fields["token1_ms"]) > 0.0
    assert "scheduler_admission=scalar passed=true" in lines
    assert "scheduler_admission=scalar_taps passed=true" in lines
    assert "scheduler_admission=llama_same_gguf passed=false" in lines
    assert "status=passed" in lines
