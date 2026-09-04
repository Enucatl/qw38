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


def test_prompt_scheduler_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "cuda_prompt_scheduler_contract.json").read_text()
    )
    fixture = json.loads((ROOT / "fixtures" / "cuda_prompt_scheduler.json").read_text())
    assert contract["prompt_chunk_rows"] == 64
    assert contract["admission"]["state_byte_equal_to_tokenwise"]
    assert fixture["boundary_case"]["chunks"] == [64, 1]
    assert fixture["boundary_case"]["state_byte_equal"]
    assert fixture["boundary_case"]["last_outputs_byte_equal"]
    assert fixture["boundary_case"]["speedup"] > 1.0
    assert fixture["cancellation_case"]["committed_state_equal_to_empty"]
    raw = ROOT / fixture["benchmark_smoke"]["raw_result"]
    assert (
        hashlib.sha256(raw.read_bytes()).hexdigest()
        == (fixture["benchmark_smoke"]["raw_result_sha256"])
    )
    chapter = (ROOT / "docs" / "62-cuda-full-prefill.md").read_text().casefold()
    for term in [
        "layer-major",
        "token-major",
        "64",
        "mmq",
        "q8_0",
        "candidate",
        "committed",
        "cancellation",
        "scratch",
        "decode",
        "proof boundary",
    ]:
        assert term in chapter


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned GGUF is required")
def test_chunked_full_prompt_is_exact_atomic_and_faster() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
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
        [*common, "make", "build/qw38-cuda-prompt-scheduler-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-prompt-scheduler-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    chunk = next(line for line in lines if line.startswith("prompt_chunk="))
    fields = dict(field.split("=", 1) for field in chunk.split())
    assert fields["prompt_chunk"] == "tokens_65"
    assert fields["rows_per_chunk"] == "64"
    assert fields["evaluated"] == "65"
    assert fields["frontier"] == "65"
    assert fields["state_exact"] == "true"
    assert fields["outputs_exact"] == "true"
    assert float(fields["speedup"]) > 1.0
    assert (
        "prompt_cancel=before_commit status=cancelled frontier=0 passed=true" in lines
    )
    assert "prompt_workspace_bytes=186306144" in lines
    assert "status=passed" in lines
