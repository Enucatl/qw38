"""Host tests for OPT-063 paired integer gate/up + SwiGLU."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt063_integer_ffn_contract.json"
ITERATION = ROOT / "pins/opt063_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt063_integer_ffn.json"
OPT062 = ROOT / "fixtures/opt062_q4_admission.json"
REPORT = ROOT / "evidence/optimization/opt063-integer-ffn/REPORT.md"
MAKEFILE = ROOT / "Makefile"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PATH = ROOT / "cuda/ffn_decode_path.cuh"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
DOTS_H = ROOT / "cuda/q4k_decode_dots.cuh"
DOTS = ROOT / "cuda/q4k_decode_dots.cu"
NATIVE = ROOT / "cuda/opt063_integer_ffn_test.cu"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
MMV_H = ROOT / "cuda/quant_mmv.h"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-063")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-063"
    assert contract["claims_throughput"] is False
    assert contract["opt062_installed"] is False
    assert contract["paired_integer_installed"] is False
    assert contract["selected_q4_decode_path"] == "packed"
    assert contract["selected_ffn_decode_path"] == "paired_staged"
    assert "paired_integer" in contract["legal_ffn_decode_paths"]
    assert iteration["task"] == "OPT-063"
    assert iteration["target"] == "build/qw38-cuda-opt063-integer-ffn-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt063-diagnostics"
    assert fixture["selected_q4_decode_path"] == "packed"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt063-integer-ffn-test" in makefile
    assert "cuda-opt063-diagnostics" in makefile


def test_opt062_admitted_integer_not_installed() -> None:
    opt062 = _json(OPT062)
    assert opt062["status"] == "wired"
    assert opt062["installed"] is False
    assert opt062["screen_winner"] == "integer_q8"


def test_source_wires_paired_integer_fusion() -> None:
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    dots_h = DOTS_H.read_text(encoding="utf-8")
    dots = DOTS.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    execute = scheduler[scheduler.find("cudaError_t execute_ffn") :]
    execute = execute[: execute.find("\ncudaError_t ", 1)]
    assert 'kSelectedQ4DecodePath[] = "packed"' in q4
    assert 'kSelectedFfnDecodePath[] = "paired_staged"' in ffn
    assert "kLegalFfnDecodePathPairedInteger" in ffn
    assert "ffn_decode_uses_paired_integer" in ffn
    assert "kStagingQ8Fp32UnfusedTrace" in ffn
    assert "kQ4LaunchVariantPairedIntegerQ8" in q4
    assert "load_q8_pack" in dots_h
    assert "q4k_coop_gate_up_swiglu" in dots_h
    assert "admitted_swiglu" in dots_h
    assert "launch_q4k_coop_gate_up_swiglu_prequant_q8" in dots
    assert "launch_q4k_coop_gate_up_swiglu_prequant_q8" in header
    assert "launch_q4k_coop_gate_up_swiglu_prequant_q8" in execute
    assert "ffn_decode_uses_paired_integer" in execute
    assert "kStagingQ8Fp32UnfusedTrace" in execute
    assert "q8_decode_staged_activation_ = nullptr" in execute
    assert "launch_q4k_coop_gate_up_swiglu_prequant_q8" in replay
    assert "opt062_installed=false" in native
    assert "concatenat" not in dots_h.casefold()


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-063")
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
    assert loop_product(iteration["workloads"]["smoke"]) == 6
    assert loop_product(iteration["workloads"]["correctness"]) == 8
    assert loop_product(iteration["workloads"]["screen"]) == 8
    assert loop_product(iteration["workloads"]["acceptance"]) == 26
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "no prompt mmq" in proof
    assert "no 1/2/4/8 warp sweep" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_report_records_fusion_and_pin() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-063" in text
    assert "paired_integer" in text or "paired integer" in text
    assert "packed" in text
    assert "swiglu" in text
    assert "q8block" in text or "q8" in text
    assert "opt-062" in text
    assert "no throughput" in text or "claims_throughput" in text
