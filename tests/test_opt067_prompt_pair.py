"""Host tests for OPT-067 paired prompt gate/up + BF16 SwiGLU."""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt067_prompt_pair_contract.json"
ITERATION = ROOT / "pins/opt067_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt067_prompt_pair.json"
REPORT = ROOT / "evidence/optimization/opt067-prompt-pair/REPORT.md"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt067_prompt_pair_test.cu"
MMQ = ROOT / "cuda/quant_mmq_mma.cuh"
MMV_H = ROOT / "cuda/quant_mmv.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-067")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-067"
    assert contract["claims_throughput"] is False
    assert contract["control_tile"] == "i128_j128"
    assert contract["candidate_tile"] == "i64_j64"
    assert iteration["task"] == "OPT-067"
    assert iteration["target"] == "build/qw38-cuda-opt067-prompt-pair-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt067-diagnostics"
    assert fixture["control_tile"] == "i128_j128"
    assert fixture["candidate_tile"] == "i64_j64"
    assert fixture["opt061_threshold_ms"] == 5.0
    assert fixture["keep"] is False
    assert fixture["installed"] is False
    assert fixture["selected_ffn_prompt_pair_path"] == "off"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt067-prompt-pair-test" in makefile
    assert "cuda-opt067-diagnostics" in makefile


def test_source_wires_paired_kernel_and_selector() -> None:
    mmq = MMQ.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "quality_mma_q4_accumulate_half" in mmq
    assert "quant_mmq_mma_q4_paired_swiglu_kernel" in mmq
    assert "kSelectedFfnPromptPairPath" in mmq
    assert "launch_q4_mmq_paired_gate_up_swiglu_bf16" in header
    assert "ffn_prompt_uses_paired" in header
    assert "FfnPromptPairOverrideScope" in header
    assert "ffn_prompt_uses_paired()" in scheduler
    assert "i64_j64" in native
    assert "i128_j128" in native
    assert "graph_eager" in native
    assert 'kSelectedFfnPath[] = "shared_y_swiglu_q8"' in mmq
    assert 'kSelectedFfnPromptPairPath[] = "off"' in mmq


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-067")
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
    assert loop_product(iteration["workloads"]["correctness"]) == 20
    assert loop_product(iteration["workloads"]["screen"]) == 8
    assert loop_product(iteration["workloads"]["acceptance"]) == 8
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "5ms" in proof
    assert "i64" in proof
    assert "no stream-k" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_report_records_threshold_and_decision() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-067" in text
    assert "5" in text
    assert "i64" in text
    assert "i128" in text
    assert "occupancy" in text
    assert "8.517" in text
    assert "11.276" in text
    assert "keep" in text or "no-go" in text or "reject" in text
