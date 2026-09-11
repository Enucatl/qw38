"""Host tests for OPT-064 Q8 decode row grouping."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt064_q8_rows_contract.json"
ITERATION = ROOT / "pins/opt064_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt064_q8_rows.json"
REPORT = ROOT / "evidence/optimization/opt064-q8-rows/REPORT.md"
MAKEFILE = ROOT / "Makefile"
PATH_H = ROOT / "cuda/q8_decode_path.cuh"
DOTS_H = ROOT / "cuda/q8_decode_dots.cuh"
DOTS = ROOT / "cuda/q8_decode_dots.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
NATIVE = ROOT / "cuda/opt064_q8_rows_test.cu"
MMV_H = ROOT / "cuda/quant_mmv.h"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-064")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-064"
    assert contract["claims_throughput"] is False
    assert contract["selected_q8_decode_path"] == "dp4a_q8_1"
    assert contract["control_layout"] == "r1_w4"
    assert contract["layouts"] == ["r1_w4", "r2_w2", "r4_w1", "r8_w1"]
    assert iteration["task"] == "OPT-064"
    assert iteration["target"] == "build/qw38-cuda-opt064-q8-rows-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt064-diagnostics"
    assert fixture["selected_q8_decode_path"] == "dp4a_q8_1"
    assert fixture["screen_winner"] == "r2_w2"
    assert fixture["installed"] is True
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt064-q8-rows-test" in makefile
    assert "cuda-opt064-diagnostics" in makefile


def test_source_wires_row_grouping() -> None:
    path = PATH_H.read_text(encoding="utf-8")
    dots_h = DOTS_H.read_text(encoding="utf-8")
    dots = DOTS.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert 'kSelectedQ8DecodePath[] = "dp4a_q8_1"' in path
    assert "kSelectedQ8DecodeRowsSkinny = 2" in path
    assert "kSelectedQ8DecodeLayoutWarpsSkinny = 2" in path
    assert "kSelectedQ8DecodeWarpsSkinny = 4" in path
    assert "q8_decode_layout_for_rows" in path
    assert "template <int RowsPerCta, int WarpsPerRow>" in dots_h
    assert "warp / WarpsPerRow" in dots_h
    assert "warp % WarpsPerRow" in dots_h
    assert "if constexpr (WarpsPerRow == 1)" in dots_h
    assert "get_int_b2" in dots_h
    assert "aligned int*" in dots_h
    assert "launch_coop<1, 4>" in dots
    assert "launch_coop<2, 2>" in dots
    assert "launch_coop<4, 1>" in dots
    assert "launch_coop<8, 1>" in dots
    assert "unsigned int rows_per_cta" in header
    assert "q8_decode_layout_for_rows" in scheduler
    assert "layout.rows_per_cta" in scheduler
    assert "invalidate_q8_decode_staging" in scheduler
    assert "layout.rows_per_cta" in replay
    assert "r1_w4" in native
    assert "pointer_identity_not_data" in native
    assert "SoA" not in dots_h


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-064")
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
    assert loop_product(iteration["workloads"]["smoke"]) == 4
    assert loop_product(iteration["workloads"]["correctness"]) == 24
    assert loop_product(iteration["workloads"]["screen"]) == 48
    assert loop_product(iteration["workloads"]["acceptance"]) == 26
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "no persistent soa copy" in proof
    assert "no opt-024" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_report_records_layouts_and_staging() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-064" in text
    assert "r1_w4" in text
    assert "r2_w2" in text
    assert "rotating" in text
    assert "2.676" in text or "2.67" in text
    assert "staging" in text
    assert "dp4a_q8_1" in text
