"""Host tests for OPT-072 rejection inventory and Q8_1 consumer audit."""

from __future__ import annotations

from pathlib import Path

from tools.opt072_rejection_review import (
    CONTRACT,
    DOTS,
    FIXTURE,
    INTEGER_SUM_CASES,
    ITERATION,
    Q4_PATH,
    REPORT,
    STAGING_LLAMA_SUM_X,
    STAGING_QUARTZ_SUM_Q,
    build_register,
    case_workloads,
    covered_tasks,
    expected_tasks,
    half_sum_q,
    half_sum_x,
    integer_sum_audit,
    load_json,
    load_register,
    may_admit_by_numerical_headroom,
    pairing_legal,
    quants_for_sum,
    run_phase,
    validate_q8_1_audit,
    validate_register,
)
from tools.production_numerics_v2 import integer_lost_in_half
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contracts_fixture_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-072")
    fixture = load_json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-072"
    assert contract["claims_throughput"] is False
    assert contract["no_candidate_fitted_epsilon"] is True
    assert contract["no_copy_synthetic_m17_k256_to_production_k"] is True
    assert contract["opt068_rescued_by_tolerance"] is False
    assert contract["q4_route"] == ["OPT-074", "OPT-075"]
    assert contract["opt064_066_owner"] == "OPT-070"
    assert contract["independent_legacy_candidate"] == "half-scale-q8-1"
    assert iteration["task"] == "OPT-072"
    assert iteration["host_only"] is True
    assert iteration["skip_gpu_setup"] is True
    assert iteration["skip_compile"] is True
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["case_ids"] == [
        "inventory",
        "q4-original-failure",
        "half-scale-q8-1",
        "review",
    ]
    assert fixture["task"] == "OPT-072"
    assert fixture["claims_throughput"] is False
    path = Q4_PATH.read_text(encoding="utf-8")
    dots = DOTS.read_text(encoding="utf-8")
    assert "kQ81ConsumerExpects" in path
    assert "kQ81IllegalPairing" in path
    assert "quartz_q8_1_sum_q" in path
    assert "half(sum(integer quants))" in dots or "half(sum(q))" in dots
    assert "quantize_bf16_q8_1_sum_x" in dots
    assert ITERATION.is_file()
    assert ROOT.joinpath("tools/opt072_rejection_review.py").is_file()


def test_register_covers_opt001_through_opt069() -> None:
    register = load_register()
    built = build_register()
    assert covered_tasks(register) == expected_tasks()
    assert len(register["tasks_covered"]) == 69
    assert register["entry_count"] == len(register["entries"])
    assert register["entry_count"] == built["entry_count"]
    ids = {row["id"] for row in register["entries"]}
    assert "OPT-042:integer_dp4a_q8block" in ids
    assert "OPT-046:integer_q8_w4" in ids
    assert "OPT-046:historical_scheduler_nan" in ids
    assert "OPT-046:integer_q8_1_w4_half_scale" in ids
    assert "OPT-018:association_rule_historical" in ids
    assert "OPT-062:integer_q8_complete_ffn" in ids
    assert "OPT-062:half_scale_skipped" in ids
    assert "OPT-063:paired_integer" in ids
    assert "OPT-068:scoped O3/FMA codegen" in ids
    payload = validate_register(register)
    assert payload["success"] is True
    assert payload["missing_tasks"] == []
    assert payload["incomplete_reevaluation"] == []


def test_nonfinite_and_layout_cannot_be_admitted_by_headroom() -> None:
    register = load_register()
    nan = next(
        row
        for row in register["entries"]
        if row["id"] == "OPT-046:historical_scheduler_nan"
    )
    assert nan["kind"] == "layout_nonfinite_state"
    assert nan["classification"] == "real_correctness_bug"
    assert nan["nonfinite"] == 248320
    assert may_admit_by_numerical_headroom(nan) is False
    assert nan["admit_by_numerical_headroom"] is False
    opt068 = next(row for row in register["entries"] if row["task"] == "OPT-068")
    assert opt068["disposition"] == "no_repeat"
    assert opt068["reevaluation"]["status"] == (
        "performance_loss_not_rescued_by_tolerance"
    )


def test_q4_q8_1_sum_cases_distinguish_sum_q_sum_x_and_int32() -> None:
    for total in INTEGER_SUM_CASES:
        row = integer_sum_audit(total)
        assert row["exact_int32_sum_q"] == total
        assert row["half_sum_q"] == half_sum_q(quants_for_sum(total))
        if abs(total) == 2047:
            assert integer_lost_in_half(total) is False
            assert row["sum_q_consumer_matches_exact"] is True
            if abs(total) == 2049:
                assert integer_lost_in_half(total) is True
                assert row["sum_q_consumer_matches_exact"] is False
                assert abs(row["q4_consumer_from_sum_q"]) in {2048, 2050}
        if abs(total) == 4064:
            assert integer_lost_in_half(total) is False
            assert row["sum_q_consumer_matches_exact"] is True
    scaled = integer_sum_audit(2047, scale=1.0 / 127.0)
    assert scaled["half_sum_x"] != scaled["half_sum_q"]
    assert pairing_legal(STAGING_LLAMA_SUM_X, STAGING_QUARTZ_SUM_Q) is False
    assert pairing_legal(STAGING_QUARTZ_SUM_Q, STAGING_QUARTZ_SUM_Q) is True
    activations = [float(q) / 127.0 for q in quants_for_sum(4064)]
    assert half_sum_x(activations) != half_sum_q(quants_for_sum(4064))
    payload = validate_q8_1_audit()
    assert payload["success"] is True


def test_q4_candidates_route_and_v2_does_not_copy_synthetic_error() -> None:
    register = load_register()
    opt046 = next(
        row for row in register["entries"] if row["id"] == "OPT-046:integer_q8_w4"
    )
    v2 = opt046["v2_comparison"]
    assert v2["copied_synthetic_m17_k256_to_production_k"] is False
    assert v2["relaxation_changes_eligibility"] is False
    assert v2["llama_gpu_error_production_k"] is None
    assert v2["faster_implementation_if_admitted"] == "integer_q8 / integer_q8_w4"
    assert opt046["next_owner"] == "OPT-075"
    assert opt046["historical_30_sample_oracle"] is False
    assert opt046["candidate_fitted_epsilon"] is False
    opt064 = next(
        row for row in register["entries"] if row["id"] == "OPT-064:r2_w2_keep_evidence"
    )
    opt066 = next(
        row
        for row in register["entries"]
        if row["id"] == "OPT-066:fma_async_x_keep_evidence"
    )
    assert opt064["next_owner"] == "OPT-070"
    assert opt066["next_owner"] == "OPT-070"
    half = register["independent_legacy_candidate"]
    assert half["case_id"] == "half-scale-q8-1"
    assert half["covered_by_opt075_079"] is False
    assert half["screen_executed"] is False


def test_iteration_loop_products_and_dry_run_case_ids() -> None:
    iteration = load_contract("OPT-072")
    assert loop_product(iteration["workloads"]["inventory"]) == 69
    assert loop_product(iteration["workloads"]["q4-original-failure"]) == 4
    assert loop_product(iteration["workloads"]["half-scale-q8-1"]) == 4
    assert loop_product(iteration["workloads"]["half-scale-q8-1-screen"]) == 8
    assert loop_product(iteration["workloads"]["review"]) == 69
    q4 = iteration["workloads"]["q4-original-failure"]
    assert q4["candidates"] == 2
    assert q4["cases"] == 2
    assert q4["sampled_rows"] == 16
    assert q4["sampled_vectors"] == 4
    screen = iteration["workloads"]["half-scale-q8-1-screen"]
    assert screen["warmups"] == 1
    assert screen["samples"] == 3
    assert screen["candidates"] == 2
    plan = describe_plan("OPT-072", "feedback", iteration, None)
    assert "case_ids=inventory,q4-original-failure,half-scale-q8-1,review" in plan
    assert "case_id_count=4" in plan
    assert "inventory:" in plan
    assert "product=69" in plan
    q4_plan = describe_plan("OPT-072", "feedback", iteration, "q4-original-failure")
    assert "q4-original-failure:" in q4_plan
    assert "product=4" in q4_plan
    assert "loop_product_total=4" in q4_plan
    half_plan = describe_plan("OPT-072", "feedback", iteration, "half-scale-q8-1")
    assert "product=4" in half_plan
    assert "candidates=2" in half_plan
    assert "cases=2" in half_plan
    review_plan = describe_plan("OPT-072", "acceptance", iteration, "review")
    assert "product=69" in review_plan
    workloads = case_workloads(iteration)
    assert workloads["inventory"]["product"] == 69
    assert workloads["half-scale-q8-1-screen"]["product"] == 8


def test_host_phases_and_report() -> None:
    inventory = run_phase("inventory")
    assert inventory["success"] is True
    assert inventory["gpu_work"] is False
    q4 = run_phase("q4-original-failure")
    assert q4["success"] is True
    assert q4["gpu_work"] is False
    assert q4["original_failure"]["shape"] == "q4_k_17x256"
    assert q4["historical_30_sample_oracle"] is False
    review = run_phase("review")
    assert review["success"] is True
    assert review["acceptance"] == "review_completeness"
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-072" in text
    assert "half(sum(q))" in text
    assert "half(sum(x))" in text
    assert "opt-075" in text
    assert "opt-074" in text
    assert "opt-070" in text
    assert "2049" in text
    assert "no throughput" in text or "claims_throughput" in text
    proof = " ".join(load_contract("OPT-072")["proof_limit"]).casefold()
    assert "register audit has no gpu work" in proof
