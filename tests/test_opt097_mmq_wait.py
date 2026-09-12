"""Host tests for OPT-097 MMQ split X/Y wait schedule."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt097_mmq_wait_contract.json"
ITERATION = ROOT / "pins/opt097_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt097_mmq_wait.json"
REPORT = ROOT / "evidence/optimization/opt097-mmq-wait/REPORT.md"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt097_mmq_wait_test.cu"
MMQ = ROOT / "cuda/quant_mmq_mma.cuh"
MMV_H = ROOT / "cuda/quant_mmv.h"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-097")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-097"
    assert contract["claims_throughput"] is True
    assert contract["control_wait"] == "joined_wait"
    assert contract["candidate_wait"] == "split_xy_wait"
    assert iteration["task"] == "OPT-097"
    assert iteration["target"] == "build/qw38-cuda-opt097-mmq-wait-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt097-diagnostics"
    assert fixture["control_wait"] == "joined_wait"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt097-mmq-wait-test" in makefile
    assert "cuda-opt097-diagnostics" in makefile


def test_source_wires_split_wait_selector() -> None:
    mmq = MMQ.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "UseSplitXYWait" in mmq
    assert "mmq_cp_async_wait_group<1>" in mmq
    assert "q4_i128_j128_fma_async_x_split_wait" in mmq
    assert "kSelectedMmqSplitXYWait = false" in mmq
    assert "split_xy_wait" in header
    assert "MmqSplitXYWaitOverrideScope" in header
    assert "joined_wait" in native
    assert "split_xy_wait" in native
    assert "corr_k5120" in native
    assert "corr_k17408" in native
    assert "fallback_m17" in native


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-097")
    assert iteration["claims_throughput"] is True
    assert iteration["modes"]["feedback"]["tier_sequence"] == [
        "eligibility",
        "parity",
        "mmq",
    ]
    assert loop_product(iteration["workloads"]["eligibility"]) == 1
    assert loop_product(iteration["workloads"]["parity"]) == 28
    assert loop_product(iteration["workloads"]["mmq"]) == 16


def test_eligibility_helper() -> None:
    from tools.opt097_mmq_wait import mmq_wait_eligible

    opt090 = _json(ROOT / "fixtures/opt090_decode_attribution.json")
    gate = mmq_wait_eligible(opt090)
    assert gate["eligible"] is True
    assert gate["verdict"] == "proceed"
    assert float(gate["estimated_removable_wait_ms"]) >= 5.0
