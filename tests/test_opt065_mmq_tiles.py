"""Host tests for OPT-065 MMQ tile retune with the active pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt065_mmq_tiles_contract.json"
ITERATION = ROOT / "pins/opt065_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt065_mmq_tiles.json"
REPORT = ROOT / "evidence/optimization/opt065-mmq-tiles/REPORT.md"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt065_mmq_tiles_test.cu"
MMQ = ROOT / "cuda/quant_mmq_mma.cuh"
MMV_H = ROOT / "cuda/quant_mmv.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-065")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-065"
    assert contract["claims_throughput"] is False
    assert contract["selected_mmq_pipeline_path"] == "fma_async"
    assert contract["control_tile"] == "i128_j128"
    assert contract["tiles"] == ["i128_j128", "i64_j128", "i128_j64", "i64_j64"]
    assert iteration["task"] == "OPT-065"
    assert iteration["target"] == "build/qw38-cuda-opt065-mmq-tiles-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt065-diagnostics"
    assert fixture["selected_mmq_pipeline_path"] == "fma_async"
    assert fixture["control_tile"] == "i128_j128"
    assert fixture["screen_winner_gate_up"] == "i128_j128"
    assert fixture["screen_winner_down"] == "i128_j128"
    assert fixture["installed"] is False
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt065-mmq-tiles-test" in makefile
    assert "cuda-opt065-diagnostics" in makefile


def test_source_wires_four_pipeline_tiles() -> None:
    mmq = MMQ.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert 'kSelectedMmqPipelinePath[] = "fma_async"' in mmq
    assert "mmq_q4_pipeline_tile" in mmq
    assert "launch_q4_pipeline_aligned_ij" in mmq
    assert "query_q4_pipeline_kernel" in mmq
    assert "record_q4_tile_dispatch" in mmq
    assert "g_ffn_tile_override" in mmq
    assert "struct MmqTileDispatch" in header
    assert "mmq_pipeline_kernel_attributes" in header
    assert "set_ffn_tile_override" in header
    assert "selected_ffn_gate_quality_i()" in scheduler
    assert "selected_ffn_down_prompt_tile()" in scheduler
    assert "i64_j128" in native
    assert "i128_j64" in native
    assert "i64_j64" in native
    assert "fma_async" in native
    assert "launch_quant_mmq_mma_y_pipeline" in mmq


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-065")
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
    assert loop_product(iteration["workloads"]["screen"]) == 32
    assert loop_product(iteration["workloads"]["acceptance"]) == 8
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "fma/async" in proof
    assert "no q6" in proof
    assert "no stream-k" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_report_records_tiles_and_pipeline() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-065" in text
    assert "i128_j128" in text
    assert "i64_j128" in text
    assert "i128_j64" in text
    assert "i64_j64" in text
    assert "fma_async" in text or "fma/async" in text
    assert "pipeline" in text
    assert "128" in text
    assert "retained" in text
    assert "2.865" in text or "2.86" in text
