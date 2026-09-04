from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
