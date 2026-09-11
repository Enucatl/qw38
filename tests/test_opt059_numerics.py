"""Host tests for OPT-059 v2 GPU numerics policy and admission mapping."""

from __future__ import annotations

import copy
import json
import math

import pytest

from tools.production_numerics import (
    GGUF_SHA,
    LLAMA_REV,
    PRODUCTION_PPL_RATIO,
    RECURRENCE_INCREMENTAL_NLL,
    ROOT,
    STRICT_CUD001_ABS,
    STRICT_CUD001_RMS,
)
from tools.production_numerics_v2 import (
    CALIBRATION_LAYERS,
    CALIBRATION_TOKEN_POSITIONS,
    HELD_OUT_LAYERS,
    HELD_OUT_TOKEN_POSITIONS,
    INT32_OVERFLOW_MAX,
    MAX_PRODUCTION_FP64_DOTS,
    Q4K_GROUP_INT_PRODUCT_MAX,
    STAGING_LLAMA_SUM_X,
    STAGING_QUARTZ_SUM_Q,
    V2_SCHEMA,
    admit_top1_swap,
    any_optimized_selected,
    engine_summary_path,
    family_v2_budgets,
    freeze_case_ids,
    integer_lost_in_half,
    lookup_admission,
    opt046_independent_verdict,
    parse_opt059_payload,
    staged_rounding_guard,
    staging_semantics,
    teacher_forced_margin,
    unrepresented_fallback,
    v2_abs_rms_ceiling,
    v2_cosine_ceiling,
    validate_v2_contract,
)
from tools.run_optimization_task import load_contract, loop_product

CONTRACT = ROOT / "pins/production_numerics_v2_contract.json"
MANIFEST = ROOT / "pins/opt059_admission_manifest.json"
ITERATION = ROOT / "pins/opt059_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt059_gpu_numerics.json"
V1_CONTRACT = ROOT / "pins/production_numerics_contract.json"
REPORT = ROOT / "evidence/optimization/opt059-gpu-numerics/REPORT.md"
MAKEFILE = ROOT / "Makefile"
MMV = ROOT / "cuda/quant_mmv.cu"
HEADER = ROOT / "cuda/production_numerics.h"
NATIVE = ROOT / "cuda/opt059_numerics_test.cu"
EXPORT = ROOT / "tools/llama_authority/projection_export.cpp"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_v2_contracts_exist_and_validate() -> None:
    assert CONTRACT.is_file()
    assert MANIFEST.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    contract = _contract()
    fixture = _fixture()
    validate_v2_contract(contract, fixture)
    assert contract["schema"] == V2_SCHEMA
    assert contract["task"] == "OPT-059"
    assert contract["claims_performance_improvement"] is False
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["gguf_sha256"] == GGUF_SHA
    assert contract["engine_summary_path"] == "mixed"
    assert contract["unrepresented_shapes"] == "strict"
    assert contract["opt046_installed"] is False
    assert contract["quality"]["nll_ppl_ratio"] == PRODUCTION_PPL_RATIO
    assert (
        contract["quality"]["recurrence_incremental_nll"] == RECURRENCE_INCREMENTAL_NLL
    )
    assert fixture["opt046_installed"] is False
    assert fixture["opt042_numeric_reject_preserved"] is True
    assert fixture["llama_gpu_sidecars"]["not_cpu_replica"] is True
    assert fixture["llama_gpu_sidecars"]["status"] == (
        "probe_exported_production_k_unadmitted"
    )


def test_v1_history_stays_strict() -> None:
    v1 = json.loads(V1_CONTRACT.read_text(encoding="utf-8"))
    assert v1["path_selection"]["selected"] == "strict"
    assert v1["path_selection"]["optimized_admitted"] is False
    assert v1["task"] == "OPT-044"


def test_case_ids_are_frozen_and_disjoint() -> None:
    contract = _contract()
    frozen = freeze_case_ids()
    cal = [row["id"] for row in contract["case_ids"]["calibration"]]
    held = [row["id"] for row in contract["case_ids"]["held_out"]]
    assert cal == [row["id"] for row in frozen["calibration"]]
    assert held == [row["id"] for row in frozen["held_out"]]
    assert contract["case_ids"]["id_manifest_sha256"] == frozen["id_manifest_sha256"]
    assert set(cal).isdisjoint(set(held))
    assert contract["case_ids"]["calibration_layers"] == list(CALIBRATION_LAYERS)
    assert contract["case_ids"]["held_out_layers"] == list(HELD_OUT_LAYERS)
    assert contract["case_ids"]["calibration_token_positions"] == list(
        CALIBRATION_TOKEN_POSITIONS
    )
    assert contract["case_ids"]["held_out_token_positions"] == list(
        HELD_OUT_TOKEN_POSITIONS
    )
    assert contract["case_ids"]["opt043_preprojection_insufficient"] is True
    assert "swiglu_down_input" in contract["case_ids"]["required_extra_captures"]
    assert "final_norm" in contract["case_ids"]["required_extra_captures"]
    assert contract["case_ids"]["max_production_fp64_dots_per_phase"] == (
        MAX_PRODUCTION_FP64_DOTS
    )


def test_budget_immutability_rejects_mutated_ceilings() -> None:
    contract = _contract()
    fixture = _fixture()
    mutated = copy.deepcopy(contract)
    first = next(iter(mutated["families"].values()))
    first["vs_fp64_original_bf16"]["abs"]["ceiling"] *= 2
    with pytest.raises(ValueError, match="v2 budget mutated"):
        validate_v2_contract(mutated, fixture)
    speedup = copy.deepcopy(contract)
    speedup["claims_performance_improvement"] = True
    with pytest.raises(ValueError, match="speedup"):
        validate_v2_contract(speedup, fixture)


def test_missing_held_out_or_gpu_reference_is_unadmitted() -> None:
    contract = _contract()
    fixture = _fixture()
    missing = copy.deepcopy(contract)
    missing["case_ids"]["held_out"] = []
    with pytest.raises(ValueError, match="held-out case IDs mutated"):
        validate_v2_contract(missing, fixture)
    budget = v2_abs_rms_ceiling(STRICT_CUD001_ABS, None)
    assert budget["admitted"] is False
    assert budget["path"] == "strict"
    assert budget["ceiling"] == STRICT_CUD001_ABS
    cosine = v2_cosine_ceiling(None)
    assert cosine["admitted"] is False
    for family in contract["families"].values():
        if family["vs_fp64_original_bf16"]["abs"]["llama_gpu_error"] is None:
            assert family["coverage"] == "unadmitted"
            assert family["vs_fp64_original_bf16"]["abs"]["path"] == "strict"


def test_zero_vector_uses_abs_and_norm_not_cosine() -> None:
    budgets = family_v2_budgets(
        {"max_abs": 0.0, "rms": 0.0, "one_minus_cosine": 1.0, "candidate_norm": 0.0},
        family="q4_k",
        ident="zero_probe",
        activation_source="zero",
        nonfinite=0,
    )
    policy = budgets["vs_fp64_original_bf16"]["zero_vector"]
    assert policy["rule"] == "explicit_norm_and_abs"
    assert policy["cosine_not_used"] is True
    assert (
        budgets["vs_fp64_original_bf16"]["one_minus_cosine"]["cosine_not_used"] is True
    )


def test_nonfinite_and_overflow_stay_strict() -> None:
    nan_budget = v2_abs_rms_ceiling(STRICT_CUD001_ABS, float("nan"), nonfinite=1)
    inf_budget = v2_abs_rms_ceiling(STRICT_CUD001_ABS, math.inf, nonfinite=0)
    assert nan_budget["admitted"] is False
    assert inf_budget["admitted"] is False
    assert nan_budget["path"] == inf_budget["path"] == "strict"
    assert Q4K_GROUP_INT_PRODUCT_MAX == 15 * 127 * 32
    assert Q4K_GROUP_INT_PRODUCT_MAX < INT32_OVERFLOW_MAX
    guard = staged_rounding_guard(fp32_scale_reductions=8, sum_abs_products=1.0)
    assert guard["waives_total_error"] is False
    assert guard["int32_overflow_maximum"] == INT32_OVERFLOW_MAX
    bad_total = v2_abs_rms_ceiling(STRICT_CUD001_ABS, 0.05)
    assert bad_total["ceiling"] > STRICT_CUD001_ABS
    assert guard["conservative_abs_bound"] < bad_total["ceiling"]
    rms = v2_abs_rms_ceiling(STRICT_CUD001_RMS, None)
    assert rms["ceiling"] == STRICT_CUD001_RMS


def test_unadmitted_shape_falls_back_to_strict() -> None:
    manifest = _manifest()
    assert lookup_admission(manifest, family="q5_k") is None
    fallback = unrepresented_fallback()
    assert fallback["production_dispatch"] == "strict"
    assert fallback["v2_admitted"] is False
    q4 = lookup_admission(
        manifest, family="q4_k", columns=5120, staging="q8_fp32", variant="packed_fp32"
    )
    assert q4 is not None
    assert q4["currently_selected"] is True
    assert q4["production_dispatch"] == "strict"
    q8 = lookup_admission(
        manifest,
        family="q8_0",
        columns=5120,
        staging=STAGING_QUARTZ_SUM_Q,
        variant="dp4a_q8_1",
    )
    assert q8 is not None
    assert q8["currently_selected"] is True
    assert q8["production_dispatch"] == "optimized"
    assert q8["v2_admitted"] is False
    assert engine_summary_path(manifest) == "mixed"
    assert any_optimized_selected(manifest) is True


def test_q8_1_staging_semantics_label_sum_q_versus_sum_x() -> None:
    staging = staging_semantics()
    assert staging["quartz_q8_1_sum_q"]["stores"] == "half(sum(integer quants))"
    assert staging["llama_gpu_q8_1_sum_x"]["stores"] == "half(sum(original x))"
    assert staging["opt044_host_replica"]["not_gpu_authority"] is True
    assert staging["pinned_q4_mmv"]["uses_stored_sum"] is False
    assert integer_lost_in_half(2047) is False
    assert integer_lost_in_half(-2047) is False
    assert integer_lost_in_half(2049) is True
    assert integer_lost_in_half(-2049) is True
    assert staging["units_lost_when_stored_as_half"]["2049"] is True
    dots = NATIVE.read_text(encoding="utf-8")
    header = (ROOT / "cuda/q4k_decode_dots.cuh").read_text(encoding="utf-8")
    assert "quantize_bf16_q8_1_sum_x" in dots
    assert "quartz_q8_1_sum_q" in header
    assert STAGING_LLAMA_SUM_X in json.dumps(_manifest())


def test_opt046_is_not_installed_and_fails_historical_cud001() -> None:
    fixture = _fixture()
    opt046 = json.loads((ROOT / "fixtures/opt046_q4_decode.json").read_text())
    verdict = opt046_independent_verdict(
        opt046, STRICT_CUD001_ABS, llama_gpu_error=None
    )
    assert verdict["installed"] is False
    assert verdict["production_path"] == "packed"
    assert verdict["meets_historical_cud001"] is False
    assert verdict["meets_v2_original"] is False
    assert verdict["keep_strict_q4_dispatch"] is True
    assert fixture["opt046_independent_verdict"]["installed"] is False
    pin = Q4_PATH.read_text(encoding="utf-8")
    assert 'kSelectedQ4DecodePath[] = "packed"' in pin
    native = NATIVE.read_text(encoding="utf-8")
    assert "opt046_installed=false" in native
    assert "Q4DecodePathScope" in native


def test_source_admission_mapping_replaces_always_false_stub() -> None:
    mmv = MMV.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    cmake = (ROOT / "tools/llama_authority/CMakeLists.txt").read_text(encoding="utf-8")
    assert 'kSelectedProductionNumericsPath[] = "strict"' in mmv
    assert "kProductionAdmission[]" in mmv
    assert (
        "production_numerics_optimized_admitted() noexcept { return false; }" not in mmv
    )
    assert "kLegalProductionNumericsPathMixed" in header
    assert "unrepresented_production_numerics_path" in header
    assert "qw38-cuda-opt059-numerics-test" in makefile
    assert "cuda-opt059-diagnostics" in makefile
    assert "maybe_capture_down_input" in scheduler
    assert "down_captured" in scheduler
    assert "qw38-llama-projection-export" in cmake
    assert EXPORT.is_file()
    assert "ggml_backend_synchronize" in EXPORT.read_text(encoding="utf-8")
    assert "cudaDeviceSynchronize" in EXPORT.read_text(encoding="utf-8")


def test_iteration_contract_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-059")
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["feedback"]["tier_sequence"] == ["smoke", "q4"]
    assert iteration["modes"]["acceptance"]["tier_sequence"] == [
        "smoke",
        "correctness",
        "calibrate",
    ]
    assert loop_product(iteration["workloads"]["smoke"]) == 6
    assert loop_product(iteration["workloads"]["q4"]) == 4
    assert loop_product(iteration["workloads"]["correctness"]) == 12
    assert loop_product(iteration["workloads"]["calibrate"]) == 1
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "no p/d performance oracles" in proof
    assert "no opt-046 install" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_native_payload_parser_and_teacher_forced_margin() -> None:
    payload = parse_opt059_payload(
        "noise\nQW38_OPT059_NUMERICS_RESULT="
        '{"schema_version":1,"task":"OPT-059","status":"passed",'
        '"opt046_installed":false,"engine_path":"mixed"}\n'
    )
    assert payload["task"] == "OPT-059"
    assert payload["engine_path"] == "mixed"
    assert payload["opt046_installed"] is False
    with pytest.raises(ValueError, match="missing OPT-059"):
        parse_opt059_payload("no payload here")
    pending = teacher_forced_margin(None)
    assert pending["status"] == "unadmitted_until_reference_logits"
    frozen = teacher_forced_margin(0.25)
    assert frozen["per_position_margin"] == pytest.approx(0.5)
    assert admit_top1_swap(1, 1, 2, 0.1, 0.5) is True
    assert admit_top1_swap(2, 1, 2, 0.1, 0.5) is True
    assert admit_top1_swap(3, 1, 2, 0.1, 0.5) is False
    assert admit_top1_swap(2, 1, 2, 0.9, 0.5) is False


def test_report_records_policy_and_opt046_verdict() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-059" in text
    assert "half(sum(integer quants))" in text or "half(sum(q))" in text
    assert "half(sum(original x))" in text or "half(sum(original_x))" in text
    assert "mixed" in text
    assert "unadmitted" in text
    assert "opt-046" in text
    assert "packed" in text
    assert "1.25" in text
    assert "claims_performance_improvement" in text or "no speedup" in text
