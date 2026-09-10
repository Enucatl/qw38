from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
PREFIX = "QW38_PROMPT_GRAPH_RESULT="
CONTRACT = ROOT / "pins/cuda_prompt_graph_contract.json"
FIXTURE = ROOT / "fixtures/cuda_prompt_graph.json"
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
        "semantic",
        "graphs",
        "launch",
        "fail_closed",
        "equality",
        "fallback",
        "cancellation",
        "counters",
        "proof_limit",
    }
    assert result["schema_version"] == 1 and result["task"] == "OPT-012"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert "component-only" in result["proof_limit"]
    assert "whole-chunk" in result["proof_limit"]
    assert "Nsight" in result["proof_limit"]
    assert "end-to-end" in result["proof_limit"]
    assert "128K quality" in result["proof_limit"]
    assert set(result["semantic"]) == set(contract["semantic_predicates"])
    assert all(type(value) is bool and value for value in result["semantic"].values())
    graphs_4096 = result["graphs"]["capacity_4096"]
    graphs_65 = result["graphs"]["capacity_65"]
    assert (
        graphs_4096["decode_graph_count"]
        == contract["capacity_4096"]["decode_graph_count"]
    )
    assert (
        graphs_4096["prompt_graph_count"]
        == contract["capacity_4096"]["prompt_graph_count"]
    )
    assert (
        graphs_4096["prompt_graph_rows"]
        == contract["capacity_4096"]["prompt_graph_rows"]
    )
    assert graphs_4096["graph_count"] == contract["capacity_4096"]["graph_count"]
    assert graphs_4096["allocated_bytes"] > 0
    assert graphs_65 == contract["capacity_65"]
    launch = result["launch"]
    assert launch["kernel_nodes"] >= contract["min_kernel_nodes"]
    assert launch["has_rowwise_residual_add_norm"] is True
    assert launch["grid"][0] == contract["rowwise_grid_x"]
    assert launch["block"][0] == contract["rowwise_block_x"]
    fail_closed = result["fail_closed"]
    assert fail_closed["mismatch"] is True
    assert fail_closed["unfused"] is True
    assert fail_closed["mismatch_frontier"] == 0
    equality = result["equality"]
    assert equality["state_equals"] is True
    assert equality["hidden_memcmp"] is True
    assert equality["logits_memcmp"] is True
    assert equality["frontier"] == 4096
    assert isinstance(equality["graph_ms"], (int, float)) and equality["graph_ms"] > 0.0
    assert (
        isinstance(equality["ordinary_ms"], (int, float))
        and equality["ordinary_ms"] > 0.0
    )
    fallback = result["fallback"]
    assert fallback["state_hidden_logits_exact"] is True
    assert fallback["rows"] == contract["fallback_rows"]
    cancellation = result["cancellation"]
    assert cancellation["polls"] == contract["cancel_poll"]
    assert cancellation["frontier"] == 0
    assert cancellation["scatter_kernel_launches"] == 0
    assert cancellation["prompt_graph_launches"] == 8
    assert cancellation["outputs_unchanged"] is True
    assert cancellation["state_equals_empty"] is True
    assert result["counters"]["graph"] == contract["graph_counters"]
    assert result["counters"]["ordinary"] == contract["ordinary_counters"]


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
        "cuda/prompt_graph_test.cu",
        "build/full_scheduler.trace.cuda.o",
        "build/scheduler_primitives.cuda.o",
        "build/quant_mmv.cuda.o",
        "build/q4k_decode_dots.cuda.o",
        "build/q8_decode_dots.cuda.o",
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
        "build/qw38-cuda-prompt-graph-test",
    ]
    commands = [
        [*_common(), "make", "build/qw38-cuda-graph-test"],
        nvcc,
        [
            *_common(),
            "./build/qw38-cuda-prompt-graph-test",
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


def test_prompt_graph_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))


def test_prompt_graph_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"].pop("graph_ordinary_exact_4096"),
        lambda x: x["graphs"]["capacity_4096"].__setitem__("graph_count", 64),
        lambda x: x["graphs"]["capacity_65"].__setitem__("prompt_graph_count", 64),
        lambda x: x["launch"].__setitem__("kernel_nodes", 5),
        lambda x: x["launch"]["grid"].__setitem__(0, 64),
        lambda x: x["fail_closed"].__setitem__("mismatch_frontier", 1),
        lambda x: x["equality"].__setitem__("state_equals", False),
        lambda x: x["cancellation"].__setitem__("prompt_graph_launches", 7),
        lambda x: x["counters"]["graph"].__setitem__("prompt_graph_launches", 0),
        lambda x: x["counters"]["ordinary"].__setitem__(
            "fused_residual_norm_kernel_launches", 0
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


def test_prompt_graph_native_smoke() -> None:
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
        raise SystemExit("usage: test_cuda_prompt_graph.py --regenerate-fixture")
    _regenerate()
