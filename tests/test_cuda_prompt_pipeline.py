from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
PREFIX = "QW38_PROMPT_PIPELINE_RESULT="
CONTRACT = ROOT / "pins/cuda_prompt_pipeline_contract.json"
FIXTURE = ROOT / "fixtures/cuda_prompt_pipeline.json"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == {
        "schema_version",
        "task",
        "status",
        "device",
        "compute_capability",
        "driver",
        "runtime",
        "toolkit",
        "pinned_image",
        "measurement_utc",
        "threads",
        "semantic",
        "equality",
        "cancellation",
        "launch",
        "counters",
        "timing",
        "proof_limit",
    }
    assert result["schema_version"] == 1 and result["task"] == "OPT-011"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert result["threads"] == contract["threads"]
    assert "component-only" in result["proof_limit"]
    assert "Nsight" in result["proof_limit"]
    assert "end-to-end" in result["proof_limit"]
    assert set(result["semantic"]) == set(contract["semantic_predicates"])
    assert all(type(value) is bool and value for value in result["semantic"].values())
    assert result["equality"]["row_counts"] == contract["row_counts"]
    assert result["equality"]["state_hidden_logits_exact"] == [True, True, True, True]
    cancellation = result["cancellation"]
    assert cancellation["polls"] == contract["cancel_poll"]
    assert cancellation["frontier"] == 0
    assert cancellation["scatter_kernel_launches"] == 0
    assert cancellation["outputs_unchanged"] is True
    assert cancellation["state_equals_empty"] is True
    launch = result["launch"]
    assert launch["fused_embedding_64"]["kernel_nodes"] == 1
    assert launch["fused_embedding_4096"]["kernel_nodes"] == 1
    assert launch["fused_embedding_64"]["grid"][1] == 64
    assert launch["fused_embedding_4096"]["grid"][1] == 4096
    assert launch["fused_embedding_64"]["grid"][0] == contract["embedding_grid_x"]
    assert launch["fused_embedding_4096"]["grid"][0] == contract["embedding_grid_x"]
    assert launch["fused_embedding_64"]["block"] == [256, 1, 1]
    assert launch["unfused_embedding_64"]["kernel_nodes"] == 64
    assert launch["fused_residual_norm"]["kernel_nodes"] == 1
    assert launch["fused_residual_norm"]["grid"][0] == 64
    assert launch["unfused_residual_norm"]["kernel_nodes"] == 2
    assert launch["fused_scatter_64"]["kernel_nodes"] == 1
    assert launch["fused_scatter_4096"]["kernel_nodes"] == 1
    assert launch["fused_scatter_64"]["grid"][1] == contract["scatter_grid_y"]
    assert launch["fused_scatter_4096"]["grid"][1] == contract["scatter_grid_y"]
    assert launch["fused_scatter_64"]["grid"][0] >= contract["scatter_min_grid_x_64"]
    assert (
        launch["fused_scatter_4096"]["grid"][0] >= contract["scatter_min_grid_x_4096"]
    )
    assert launch["unfused_scatter_64"]["kernel_nodes"] == 16
    assert result["counters"]["fused"] == contract["fused_counters"]
    unfused = result["counters"]["unfused"]
    for key, value in contract["unfused_counters"].items():
        assert unfused[key] == value
    timing = result["timing"]
    assert timing["warmups"] == contract["timing_warmups"]
    assert timing["samples"] == contract["timing_samples"]
    assert len(timing["fused_ms"]) == len(timing["unfused_ms"]) == 30
    fused_mean = sum(timing["fused_ms"]) / 30
    unfused_mean = sum(timing["unfused_ms"]) / 30
    assert math.isclose(fused_mean, timing["fused_mean_ms"], rel_tol=5e-7)
    assert math.isclose(unfused_mean, timing["unfused_mean_ms"], rel_tol=5e-7)
    assert timing["fused_mean_ms"] < timing["unfused_mean_ms"]


def _common() -> list[str]:
    return [
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


def _build_and_run() -> dict[str, Any]:
    nvcc = [
        *_common(),
        "nvcc",
        "-std=c++17",
        "-O2",
        "-arch=sm_120",
        "--expt-relaxed-constexpr",
        "--fmad=false",
        "-Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread",
        "-Iinclude",
        "-Isrc",
        "-Ithird_party/utf8proc",
        "-Icuda",
        "-DQW38_DIAGNOSTIC_TRACE",
        "cuda/prompt_pipeline_test.cu",
        "build/full_scheduler.trace.cuda.o",
        "build/scheduler_primitives.cuda.o",
        "build/quant_mmv.cuda.o",
        "build/gdn_step.cuda.o",
        "build/attention_decode.cuda.o",
        "build/diagnostic/status.o",
        "build/diagnostic/sha256.o",
        "build/diagnostic/model.o",
        "build/diagnostic/tokenizer.o",
        "build/diagnostic/template.o",
        "build/diagnostic/quant.o",
        "build/diagnostic/tensor.o",
        "build/diagnostic/conversion.o",
        "build/diagnostic/projection.o",
        "build/diagnostic/weights.o",
        "build/diagnostic/mixer.o",
        "build/diagnostic/scheduler.o",
        "build/diagnostic/scalar_runtime.o",
        "build/diagnostic/gdn.o",
        "build/diagnostic/attention.o",
        "build/diagnostic/engine.o",
        "build/diagnostic/diagnostic_trace.o",
        "build/utf8proc.o",
        "-o",
        "build/qw38-cuda-prompt-pipeline-test",
    ]
    commands = [
        [*_common(), "make", "build/qw38-cuda-prompt-scheduler-test"],
        nvcc,
        [
            *_common(),
            "./build/qw38-cuda-prompt-pipeline-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    ]
    outputs = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(completed.stdout)
    records = [
        json.loads(line.removeprefix(PREFIX))
        for line in outputs[-1].splitlines()
        if line.startswith(PREFIX)
    ]
    assert len(records) == 1
    validate_result(records[0])
    return records[0]


def test_prompt_pipeline_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))


def test_prompt_pipeline_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"].pop("fused_mean_below_unfused"),
        lambda x: x["launch"]["fused_embedding_64"].__setitem__("kernel_nodes", 2),
        lambda x: x["launch"]["fused_scatter_64"]["grid"].__setitem__(1, 15),
        lambda x: x["launch"]["fused_scatter_4096"]["grid"].__setitem__(0, 1),
        lambda x: x["counters"]["fused"].__setitem__("async_d2h_copies", 0),
        lambda x: x["counters"]["unfused"].__setitem__("device_synchronizes", 0),
        lambda x: x["cancellation"].__setitem__("polls", 7),
        lambda x: x["timing"].__setitem__(
            "fused_mean_ms", x["timing"]["unfused_mean_ms"]
        ),
        lambda x: x.__setitem__("driver", "placeholder"),
        lambda x: x.__setitem__("proof_limit", "end-to-end speedup from Nsight"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_prompt_pipeline_native_smoke() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")
    _build_and_run()


def _regenerate() -> None:
    result = _build_and_run()
    fd, temporary = tempfile.mkstemp(
        dir=FIXTURE.parent, prefix=f".{FIXTURE.name}.", text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, FIXTURE)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate-fixture"]:
        raise SystemExit("usage: test_cuda_prompt_pipeline.py --regenerate-fixture")
    _regenerate()
