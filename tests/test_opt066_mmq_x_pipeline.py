"""Host tests for OPT-066 packed Q4 X-prefetch pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt066_mmq_x_pipeline_contract.json"
ITERATION = ROOT / "pins/opt066_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt066_mmq_x_pipeline.json"
REPORT = ROOT / "evidence/optimization/opt066-mmq-x-pipeline/REPORT.md"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt066_mmq_x_pipeline_test.cu"
MMQ = ROOT / "cuda/quant_mmq_mma.cuh"
MMV_H = ROOT / "cuda/quant_mmv.h"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-066")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-066"
    assert contract["claims_throughput"] is False
    assert contract["selected_mmq_pipeline_path"] == "fma_async"
    assert contract["control_tile"] == "i128_j128"
    assert contract["candidate_variant"] == "fma_async_x"
    assert iteration["task"] == "OPT-066"
    assert iteration["target"] == "build/qw38-cuda-opt066-mmq-x-pipeline-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt066-diagnostics"
    assert fixture["control_tile"] == "i128_j128"
    assert fixture["keep"] is True
    assert fixture["installed"] is True
    assert fixture["selected_mmq_async_x"] is True
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt066-mmq-x-pipeline-test" in makefile
    assert "cuda-opt066-diagnostics" in makefile


def test_source_wires_x_ring_on_opt065_tile() -> None:
    mmq = MMQ.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "UseAsyncX" in mmq
    assert "mmq_load_raw_q4_async" in mmq
    assert "quality_unpack_q4_kb" in mmq
    assert "kSelectedMmqAsyncX" in mmq
    assert "q4_i128_j128_fma_async_x" in mmq
    assert "mmq_x_pipeline_kernel_attributes" in header
    assert "set_mmq_async_x_override" in header
    assert "MmqAsyncXOverrideScope" in header
    assert "async_x" in header
    assert "control" in native
    assert "candidate" in native
    assert "i128_j128" in native
    assert "corr_k256" in native
    assert "corr_k512" in native
    assert "corr_k768" in native
    assert "tail_m129" in native
    assert "stale" in native
    assert 'kSelectedMmqPipelinePath[] = "fma_async"' in mmq
    assert "kSelectedMmqAsyncX = true" in mmq


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-066")
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["feedback"]["tier_sequence"] == [
        "smoke",
        "correctness",
    ]
    assert iteration["modes"]["acceptance"]["tier_sequence"] == [
        "smoke",
        "correctness",
        "screen",
    ]
    assert loop_product(iteration["workloads"]["smoke"]) == 2
    assert loop_product(iteration["workloads"]["correctness"]) == 8
    assert loop_product(iteration["workloads"]["screen"]) == 16
    assert loop_product(iteration["workloads"]["acceptance"]) == 8
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "packed q4" in proof
    assert "no q6" in proof
    assert "no stream-k" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_report_records_pipeline_and_decision() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-066" in text
    assert "i128_j128" in text
    assert "fma_async" in text
    assert "stage" in text
    assert "occupancy" in text
    assert "8.614" in text
    assert "9.360" in text
    assert "keep" in text
    assert "ldgsts" in text
    assert "18448" in text or "94736" in text
