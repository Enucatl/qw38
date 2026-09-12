"""Host tests for OPT-105 64x128 double raw-X MMQ admission."""

from __future__ import annotations

import json
from pathlib import Path

from tools.opt105_mmq_double_x import (
    CANDIDATE_ID,
    CANDIDATE_KERNEL,
    CONTROL_ID,
    CONTROL_KERNEL,
    MIN_SAVING_MS,
    double_x_eligible,
)
from tools.run_optimization_task import load_contract, loop_product, workload_for_mode

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt105_mmq_double_x_contract.json"
ITERATION = ROOT / "pins/opt105_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt105_mmq_double_x.json"
REPORT = ROOT / "evidence/optimization/opt105-mmq-double-x/REPORT.md"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt105_mmq_double_x_test.cu"
MMQ = ROOT / "cuda/quant_mmq_mma.cuh"
MMV_H = ROOT / "cuda/quant_mmv.h"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-105")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-105"
    assert contract["claims_throughput"] is True
    assert contract["control_tile"] == CONTROL_ID
    assert contract["candidate_tile"] == CANDIDATE_ID
    assert contract["control_kernel"] == CONTROL_KERNEL
    assert contract["candidate_kernel"] == CANDIDATE_KERNEL
    assert contract["selected_mmq_double_x"] is False
    assert iteration["task"] == "OPT-105"
    assert iteration["target"] == "build/qw38-cuda-opt105-mmq-double-x-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt105-diagnostics"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt105-mmq-double-x-test" in makefile
    assert "cuda-opt105-diagnostics" in makefile


def test_source_wires_double_x_selector() -> None:
    mmq = MMQ.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "UseAsyncX2" in mmq
    assert "kMmqRawX2Stages" in mmq
    assert "q4_i64_j128_fma_async_x2" in mmq
    assert "kSelectedMmqDoubleX = false" in mmq
    assert "kSelectedMmqSplitXYWait = false" in mmq
    assert "MmqDoubleXOverrideScope" in header
    assert "async_x2" in header
    assert "i64_j128_x2" in native
    assert "corr_k5120" in native
    assert "corr_k17408" in native
    assert "fallback_m17" in native
    assert "graph_eager" in native
    assert "complete64_" in native


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-105")
    assert iteration["claims_throughput"] is True
    assert iteration["modes"]["feedback"]["tier_sequence"] == [
        "eligibility",
        "parity",
        "mmq",
    ]
    assert (
        loop_product(
            workload_for_mode(iteration["workloads"]["eligibility"], "feedback")
        )
        == 1
    )
    assert (
        loop_product(workload_for_mode(iteration["workloads"]["parity"], "feedback"))
        == 36
    )
    assert (
        loop_product(workload_for_mode(iteration["workloads"]["mmq"], "feedback")) == 16
    )


def test_eligibility_helper() -> None:
    gate = double_x_eligible()
    assert gate["eligible"] is True
    assert gate["verdict"] == "proceed"
    assert float(gate["min_saving_ms"]) >= MIN_SAVING_MS
    assert gate["candidate_kernel"] == CANDIDATE_KERNEL


def test_measured_reject_retains_128x128() -> None:
    fixture = _json(FIXTURE)
    assert fixture["keep"] is False
    assert fixture["selected_mmq_double_x"] is False
    assert fixture["claims_throughput"] is False
    assert fixture["independent_verdicts"]["kernel_parity_pass"] is True
    assert fixture["independent_verdicts"]["performance_pass"] is False
    assert fixture["independent_verdicts"]["production_kept"] is True
    assert "reject" in str(fixture["status"])
    saving = float(fixture["mmq"]["complete_ffn_saving_ms"])
    assert saving < MIN_SAVING_MS
    report = REPORT.read_text(encoding="utf-8")
    assert "639.904419" in report
    assert (REPORT.parent / "REJECTION.md").is_file()
