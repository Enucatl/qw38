"""Host tests for OPT-075 Q4 production admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt075_q4_production_admission import (
    CONFIGS,
    CONTROL_ID,
    CONTRACT,
    FIXTURE,
    HISTORICAL_HYPOTHESES,
    ITERATION,
    LEGACY_ABS,
    LEGACY_NEAR_MISS_ABS,
    REPORT,
    TINY_CASES,
    WARPS_PER_ROW,
    classify_numeric,
    decide_verdict,
    dispatch_matches,
    expected_dispatch,
    family_plan,
    opt073_quality,
    opt074_q4_budgets,
    pick_survivor,
    shared_production_references,
    tiny_cases,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PATH = ROOT / "cuda/ffn_decode_path.cuh"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
OPT062 = ROOT / "cuda/opt062_q4_admission_test.cu"
OPT063 = ROOT / "cuda/opt063_integer_ffn_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-075")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-075"
    assert contract["claims_throughput"] is False
    assert contract["warps_per_row"] == 4
    assert contract["half_scale_forbidden"] is True
    assert len(contract["configurations"]) == 3
    assert iteration["task"] == "OPT-075"
    assert iteration["diagnostics_make_target"] == "cuda-opt075-diagnostics"
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert fixture["shipping_q4_decode"] == "packed"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt075-diagnostics" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "qw38-cuda-opt062-q4-admission-test" in makefile
    assert "qw38-cuda-opt063-integer-ffn-test" in makefile
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    assert "apply_q4_decode_ident" in q4
    assert "apply_ffn_decode_ident" in ffn
    assert "--q4-decode" in replay
    assert "--ffn-decode" in replay
    assert "record_ffn_decode_dispatch" in replay
    assert "invalidate_q8_decode_staging" in replay
    assert "q4-ab" in probe
    assert "recapture_after_selector" in probe
    assert "override_before_capture_applied" in probe
    assert "graphs.create" in probe


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-075")
    q4 = iteration["workloads"]["q4"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(q4, "feedback")) == 12
    assert loop_product(workload_for_mode(q4, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 1
    assert workload_for_mode(q4, "feedback")["tier"] == "screen"
    assert workload_for_mode(q4, "acceptance")["tier"] == "acceptance"
    plan = family_plan("q4", "feedback")
    assert plan["candidates"] == 3
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    assert plan["engine_pairs"] == 1
    accept = family_plan("q4", "acceptance")
    assert accept["candidates"] == 2
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-075", "feedback", iteration, "q4")
    assert "phase=q4" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "half-scale" in proof
    assert "hypotheses" in proof
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300


def test_tiny_cases_zero_cancellation_and_tail_guards() -> None:
    cases = tiny_cases()
    assert {(row["rows"], row["columns"]) for row in cases} == {(17, 256), (33, 512)}
    opt062 = OPT062.read_text(encoding="utf-8")
    opt063 = OPT063.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    assert "17" in opt062 and "256" in opt062
    assert '"zero"' in opt062 or '"zero"' in replay
    assert "33" in opt063 and "512" in opt063
    for spec in TINY_CASES:
        assert spec["rows"] % 32 != 0
        assert "zero" in spec["patterns"]
        assert "cancellation" in spec["patterns"]
        assert "tail" in spec["patterns"]
    for row in cases:
        assert row["tail_guard"] is True
        assert row["zero"] is True
        assert row["cancellation"] is True
        assert row["shared_across_candidates"] is True


def test_production_references_decoded_once_and_shared() -> None:
    refs = shared_production_references()
    assert refs["sampled_rows"] == 16
    assert refs["captures_per_case"] == 4
    assert refs["decode_once"] is True
    assert refs["shared_across_candidates"] is True
    assert refs["identity_shared"] is True
    assert set(refs["roles"]) == {"gate", "up", "down"}
    first = refs["references"]["gate"]["decoded_values"]
    for config_id in refs["candidate_ids"]:
        assert refs["references"]["gate"]["decoded_values"] == first
        assert config_id in refs["candidate_ids"]
    gate_rows = refs["references"]["gate"]["sample_rows"]
    down_rows = refs["references"]["down"]["sample_rows"]
    assert len(gate_rows) == 16
    assert len(down_rows) == 16
    assert gate_rows[0] == 0
    assert down_rows[-1] == 5119
    assert first is refs["references"]["gate"]["decoded_values"]


def test_legacy_near_miss_can_fail_while_opt074_production_passes() -> None:
    coverage = opt074_q4_budgets()
    gate = coverage["budgets"]["q4_gate_up_k5120"]["ceilings"]["abs"]["ceiling"]
    assert coverage["all_admitted"] is False
    assert coverage["unfinished_reference_freeze_is_not_numerical_rejection"] is True
    classified = classify_numeric(
        original_abs=LEGACY_NEAR_MISS_ABS,
        staged_abs=LEGACY_NEAR_MISS_ABS,
        total_abs=LEGACY_NEAR_MISS_ABS,
        abs_ceiling=float(gate),
        nonfinite=0,
    )
    assert classified["legacy_pass"] is False
    assert classified["production_pass"] is True
    assert classified["near_miss_legacy_only"] is True
    assert LEGACY_NEAR_MISS_ABS > LEGACY_ABS
    tight = classify_numeric(
        original_abs=LEGACY_NEAR_MISS_ABS,
        staged_abs=LEGACY_NEAR_MISS_ABS,
        total_abs=LEGACY_NEAR_MISS_ABS,
        abs_ceiling=LEGACY_ABS,
        nonfinite=0,
    )
    assert tight["production_pass"] is False


def test_dispatch_uses_launch_variants_not_frozen_selector_strings() -> None:
    packed = expected_dispatch(CONFIGS[0])
    separate = expected_dispatch(CONFIGS[1])
    paired = expected_dispatch(CONFIGS[2])
    assert packed["gate_variant"] == "q4k_gate_up_swiglu_prequant"
    assert separate["gate_variant"] == "q4k_coop_mmv_prequant_q8"
    assert paired["gate_variant"] == "q4k_coop_gate_up_swiglu_prequant_q8"
    assert separate["down_variant"] == "q4k_coop_mmv_q8"
    assert packed["warps_per_row"] == WARPS_PER_ROW
    observed = {
        "gate_variant": "q4k_coop_mmv_prequant_q8",
        "up_variant": "q4k_coop_mmv_prequant_q8",
        "down_variant": "q4k_coop_mmv_q8",
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": 4,
    }
    assert dispatch_matches(observed, separate) is True
    wrong = dict(observed)
    wrong["gate_variant"] = "packed"
    assert dispatch_matches(wrong, separate) is False
    replay = REPLAY.read_text(encoding="utf-8")
    q4 = Q4_PATH.read_text(encoding="utf-8")
    assert "last_q4_launch_variant" in replay
    assert "kQ4LaunchVariantPairedStaged" in q4
    assert "kQ4LaunchVariantCoopQ8Prequant" in q4
    assert "kQ4LaunchVariantPairedIntegerQ8" in q4
    source = replay + q4
    assert 'candidate == "packed"' not in source
    assert "frozen selector" not in source.casefold()


def test_unadmitted_opt074_is_missing_evidence_not_numerical_rejection() -> None:
    coverage = opt074_q4_budgets()
    quality = opt073_quality()
    assert quality["quality_v3_engine_non_regression"] == "pass"
    assert quality["quality_v3_absolute"] == "fail"
    stats = {
        "positive": True,
        "mean_diff_ms": 1.0,
        "ci95_high": 1.2,
        "ci95_low": 0.8,
    }
    numeric = classify_numeric(
        original_abs=7.6e-6,
        staged_abs=0.0,
        total_abs=7.6e-6,
        abs_ceiling=0.011,
        nonfinite=0,
    )
    verdict = decide_verdict(
        component=stats,
        engine={
            "control_ms": [100.0] * 5,
            "control_mean_ms": 100.0,
            "regression_upper_ms": 0.1,
        },
        coverage=coverage,
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="integer_q8_paired",
    )
    assert verdict["verdict"] == "retain_packed"
    assert verdict["production_kept"] is False
    assert verdict["shipping_unchanged"] is True
    assert "opt074_coverage_unadmitted_missing_evidence" in verdict["reasons"]
    assert verdict["unfinished_reference_freeze_is_not_numerical_rejection"] is True
    assert verdict["retain_reason"] == "missing_evidence"
    assert verdict["verdict"] != "numerical_rejection"


def test_negative_saving_and_quality_block_retain_packed() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    coverage = {"all_admitted": True}
    numeric = classify_numeric(
        original_abs=0.0, staged_abs=0.0, total_abs=0.0, abs_ceiling=0.01
    )
    slower = decide_verdict(
        component={
            "positive": False,
            "mean_diff_ms": -1.0,
            "ci95_high": -0.2,
            "ci95_low": -1.5,
        },
        engine={
            "control_ms": [50.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.0,
        },
        coverage=coverage,
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="integer_q8_paired",
    )
    assert slower["verdict"] == "performance_rejected"
    assert slower["retain_reason"] == "performance"
    blocked = decide_verdict(
        component={"positive": True, "mean_diff_ms": 1.0, "ci95_high": 1.2},
        engine={
            "control_ms": [50.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.0,
        },
        coverage=coverage,
        quality={"quality_v3_engine_non_regression": "fail"},
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="integer_q8_paired",
    )
    assert blocked["verdict"] == "quality_blocked"
    keep = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 1.0,
            "ci95_high": 1.2,
            "ci95_low": 0.8,
        },
        engine={
            "control_ms": [50.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.01,
        },
        coverage=coverage,
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="integer_q8_paired",
    )
    assert keep["verdict"] == "keep"
    assert keep["production_kept"] is True


def test_pick_survivor_is_the_faster_integer_against_packed() -> None:
    survivor = pick_survivor(
        {
            CONTROL_ID: {"mean_ms": 16.0},
            "integer_q8_separate": {"mean_ms": 12.5},
            "integer_q8_paired": {"mean_ms": 11.2},
        }
    )
    assert survivor == "integer_q8_paired"
    assert survivor != CONTROL_ID


def test_nonfinite_and_dispatch_failure_are_hard_stops() -> None:
    with pytest.raises(AssertionError, match="nonfinite"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 1.0},
            engine=None,
            coverage={"all_admitted": True},
            quality={"quality_v3_engine_non_regression": "pass"},
            numeric={"nonfinite": 1, "production_pass": False},
            dispatch_ok=True,
            mode="acceptance",
            survivor="integer_q8_paired",
        )
    with pytest.raises(AssertionError, match="dispatch"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 1.0},
            engine=None,
            coverage={"all_admitted": True},
            quality={"quality_v3_engine_non_regression": "pass"},
            numeric={"nonfinite": 0, "production_pass": True},
            dispatch_ok=False,
            mode="acceptance",
            survivor="integer_q8_paired",
        )


def test_screen_keep_is_rejected_and_counts_must_match() -> None:
    iteration = load_contract("OPT-075")
    q4 = workload_for_mode(iteration["workloads"]["q4"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 3,
            "observed_shapes": 1,
            "observed_tier": "screen",
            "pairs": 3,
            "sample_ids": [0, 1, 2],
            "acceptance_executed": False,
            "keep": True,
        }
    )
    screen_keep = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q4",
        workload=q4,
        stdout="QW38_OPT075_NATIVE_COUNTS=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    assert screen_keep["result_class"] == "screen_only_keep"
    observed = parse_native_observation("QW38_OPT075_RESULT=" + stdout)
    assert observed["observed_candidates"] == 3
    assert observed["keep"] is True


def test_report_names_hypotheses_and_versioned_quality() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-075" in text
    assert "packed" in text
    assert "integer_q8" in text
    assert "opt-074" in text
    assert "opt-073" in text
    assert "1.26x" in text
    assert "1.40x" in text
    assert "hypotheses" in text
    assert "claims_throughput" in text
    fixture = _json(FIXTURE)
    assert fixture["claims_throughput"] is False
    assert fixture["historical_screens_are_hypotheses"] == list(HISTORICAL_HYPOTHESES)
    assert fixture["production_kept"] is False
