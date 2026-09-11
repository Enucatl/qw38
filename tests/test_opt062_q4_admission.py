"""Host tests for OPT-062 cooperative Q4 on every decode FFN leg."""

from __future__ import annotations

import json
from pathlib import Path

from tools.production_numerics_v2 import lookup_admission
from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt062_q4_admission_contract.json"
ITERATION = ROOT / "pins/opt062_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt062_q4_admission.json"
MANIFEST = ROOT / "pins/opt059_admission_manifest.json"
OPT046 = ROOT / "fixtures/opt046_q4_decode.json"
REPORT = ROOT / "evidence/optimization/opt062-q4-admission/REPORT.md"
MAKEFILE = ROOT / "Makefile"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PATH = ROOT / "cuda/ffn_decode_path.cuh"
MMV = ROOT / "cuda/quant_mmv.cu"
MMV_H = ROOT / "cuda/quant_mmv.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
NATIVE = ROOT / "cuda/opt062_q4_admission_test.cu"
DOTS = ROOT / "cuda/q4k_decode_dots.cu"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-062")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-062"
    assert contract["claims_throughput"] is False
    assert contract["opt046_installed"] is False
    assert contract["half_scale_skipped"] is True
    assert contract["selected_q4_decode_path"] == "packed"
    assert contract["selected_ffn_decode_path"] == "paired_staged"
    assert contract["q8block_prequant_never_reinterprets_q8_1"] is True
    assert iteration["task"] == "OPT-062"
    assert iteration["target"] == "build/qw38-cuda-opt062-q4-admission-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt062-diagnostics"
    assert fixture["selected_q4_decode_path"] == "packed"
    assert fixture["half_scale_skipped"] is True
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt062-q4-admission-test" in makefile
    assert "cuda-opt062-diagnostics" in makefile


def test_opt059_half_scale_is_not_v2_admitted() -> None:
    manifest = _json(MANIFEST)
    quartz = lookup_admission(
        manifest,
        family="q4_k",
        staging="quartz_q8_1_sum_q",
        variant="integer_dp4a_q8_1",
    )
    llama = lookup_admission(
        manifest,
        family="q4_k",
        staging="llama_q8_1_sum_x",
        variant="integer_dp4a_q8_1_sum_x",
    )
    q8 = lookup_admission(
        manifest,
        family="q4_k",
        staging="q8_fp32",
        variant="integer_dp4a_q8",
    )
    assert quartz is not None and quartz["v2_admitted"] is False
    assert llama is not None and llama["v2_admitted"] is False
    assert q8 is not None and q8["testing_admitted"] is True
    assert q8["currently_selected"] is False


def test_source_wires_integer_q8block_on_all_ffn_legs() -> None:
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    mmv = MMV.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    dots = DOTS.read_text(encoding="utf-8")
    execute = scheduler[scheduler.find("cudaError_t execute_ffn") :]
    execute = execute[: execute.find("\ncudaError_t ", 1)]
    prequant = mmv[mmv.find("cudaError_t launch_quant_mmv_prequant") :]
    prequant = prequant[: prequant.find("\ncudaError_t ", 1)]
    assert 'kSelectedQ4DecodePath[] = "packed"' in q4
    assert "kSelectedQ4DecodeWarpsPerRow = 4" in q4
    assert "q4_decode_uses_integer_q8block" in q4
    assert "record_q4_launch_variant" in q4
    assert "FfnDecodeDispatchRecord" in ffn
    assert "captured_in_graph" in ffn
    assert "launch_q4k_coop_mmv_prequant_q8" in header
    assert "q4_decode_uses_integer_q8block" in prequant
    assert "Never reinterpret" in prequant or "Never reinterpret" in header
    assert "q4_decode_uses_integer_q8block" in execute
    assert "launch_quant_mmv_prequant" in execute
    assert "matrix_vector(layer.ffn_down" in execute
    assert "record_ffn_decode_dispatch" in execute
    assert "cudaStreamIsCapturing" in execute
    assert "q4_decode_uses_integer_q8block" in replay
    assert "launch_quant_mmv_prequant" in replay
    assert "launch_q4k_coop_mmv_prequant_q8" in dots
    assert "kQ4LaunchVariantCoopQ8Prequant" in dots
    assert "opt046_installed=false" in native
    assert "half_scale_skipped" in native
    assert "__dp4a" not in mmv
    opt046 = _json(OPT046)
    assert opt046["status"] == "rejected"
    assert opt046["selected_q4_decode_path"] == "packed"


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-062")
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
    assert loop_product(iteration["workloads"]["smoke"]) == 8
    assert loop_product(iteration["workloads"]["correctness"]) == 12
    assert loop_product(iteration["workloads"]["screen"]) == 8
    assert loop_product(iteration["workloads"]["acceptance"]) == 26
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "no prompt mmq" in proof
    assert "half-scale skipped" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_report_records_wiring_and_skip() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-062" in text
    assert "packed" in text
    assert "paired_staged" in text
    assert "q8block" in text
    assert "half-scale" in text or "half_scale" in text
    assert "opt-059" in text
    assert "gate" in text and "down" in text
    assert "no throughput" in text or "claims_throughput" in text
