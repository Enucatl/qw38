from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _make_target_prerequisites(makefile: str, target: str) -> set[str]:
    for line in makefile.splitlines():
        if line.startswith(f"{target}:"):
            return set(line.split(":", 1)[1].split("|", 1)[0].split())
    raise AssertionError(f"missing Make target: {target}")


def test_cuda_diagnostic_object_graph_uses_matching_trace_variants() -> None:
    makefile = (ROOT / "Makefile").read_text()
    diagnostic_targets = {
        "$(CUDA_BUILD_DIR)/qw38-eval-diagnostic": {
            "$(BUILD_DIR)/checkpoint.trace.cuda.o",
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
        "$(BUILD_DIR)/qw38-cuda-prefix-sync-test": {
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
        "$(BUILD_DIR)/qw38-cuda-prompt-scheduler-test": {
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
        "$(BUILD_DIR)/qw38-cuda-atomic-eval-test": {
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
        "$(BUILD_DIR)/qw38-cuda-checkpoint-test": {
            "$(BUILD_DIR)/checkpoint.trace.cuda.o",
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
        "$(BUILD_DIR)/qw38-cuda-memory-fit-test": {
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
        "$(BUILD_DIR)/qw38-cuda-timing-test": {
            "$(BUILD_DIR)/checkpoint.trace.cuda.o",
            "$(BUILD_DIR)/full_scheduler.trace.cuda.o",
        },
    }
    for target, expected in diagnostic_targets.items():
        prerequisites = _make_target_prerequisites(makefile, target)
        assert expected <= prerequisites
        assert "$(BUILD_DIR)/full_scheduler.cuda.o" not in prerequisites
        assert "$(BUILD_DIR)/checkpoint.cuda.o" not in prerequisites

    release_eval = _make_target_prerequisites(makefile, "$(CUDA_BUILD_DIR)/qw38-eval")
    diagnostic_engine_objects = makefile.split("CUDA_DIAGNOSTIC_ENGINE_OBJECTS :=", 1)[
        1
    ].splitlines()[0]
    assert "$(CUDA_BUILD_DIR)/engine.trace.o" in diagnostic_engine_objects
    assert "$(CUDA_BUILD_DIR)/engine.o" not in diagnostic_engine_objects
    assert "$(BUILD_DIR)/checkpoint.cuda.o" in release_eval
    assert "$(BUILD_DIR)/full_scheduler.cuda.o" in release_eval
    assert "$(CUDA_BUILD_DIR)/engine.trace.o" not in release_eval
    assert "$(BUILD_DIR)/checkpoint.trace.cuda.o" not in release_eval
    assert "$(BUILD_DIR)/full_scheduler.trace.cuda.o" not in release_eval

    engine_trace_rule = makefile.split("$(CUDA_BUILD_DIR)/engine.trace.o:", 1)[1]
    assert "-DQW38_CUDA_RUNTIME -DQW38_DIAGNOSTIC_TRACE" in engine_trace_rule
    checkpoint_trace_rule = makefile.split("$(BUILD_DIR)/checkpoint.trace.cuda.o:", 1)[
        1
    ]
    assert "-DQW38_DIAGNOSTIC_TRACE" in checkpoint_trace_rule


def test_cuda_trace_contract_links_scalar_contract_and_fixture() -> None:
    contract = json.loads((ROOT / "pins/cuda_trace_contract.json").read_text())
    scalar = json.loads((ROOT / contract["scalar_contract"]).read_text())
    fixture = json.loads((ROOT / "fixtures/cuda_trace.json").read_text())
    expected = {
        (0, "layer_residual"): [5120],
        (3, "layer_residual"): [5120],
        (63, "layer_residual"): [5120],
        (64, "final_norm"): [5120],
        (64, "logits"): [248320],
    }
    assert {
        (item["layer"], item["tap"]): item["shape"] for item in contract["filters"]
    } == expected
    global_taps = {item["name"]: item["shape"] for item in scalar["global_taps"]}
    assert global_taps["final_norm"] == [5120]
    assert global_taps["logits"] == [248320]
    assert all(item["passed"] for item in fixture["filters"])
    assert fixture["failure_path"]["frontier_after_failure"] == 0


def test_normal_scheduler_source_has_no_trace_entry_point() -> None:
    source = (ROOT / "cuda/full_scheduler.cu").read_text()
    assert "execute_token_traced" in source
    guarded = source.split("#ifdef QW38_DIAGNOSTIC_TRACE", 1)[1]
    assert "execute_token_traced" in guarded
