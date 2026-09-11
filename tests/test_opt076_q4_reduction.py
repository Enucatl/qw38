"""Host tests for OPT-076 Q4 packed-load / late-reduction admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt076_q4_reduction import (
    CONFIGS,
    CONTROL_ID,
    CONTRACT,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    REPORT,
    TINY_CASES,
    decide_verdict,
    dispatch_matches,
    expected_dispatch,
    family_plan,
    opt073_quality,
    opt074_q4_budgets,
    pick_survivor,
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
DOTS = ROOT / "cuda/q4k_decode_dots.cuh"
DOTS_CU = ROOT / "cuda/q4k_decode_dots.cu"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
NATIVE = ROOT / "cuda/opt076_q4_reduction_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-076")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-076"
    assert contract["claims_throughput"] is False
    assert contract["half_scale_forbidden"] is True
    assert contract["k_loop_shuffle_removed"] is True
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert len(contract["configurations"]) == 3
    assert iteration["task"] == "OPT-076"
    assert iteration["diagnostics_make_target"] == "cuda-opt076-diagnostics"
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert fixture["shipping_q4_decode"] == "packed"
    assert fixture["opt075_unaffected"] is True
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt076-diagnostics" in makefile
    assert "qw38-cuda-opt076-q4-reduction-test" in makefile
    q4 = Q4_PATH.read_text(encoding="utf-8")
    dots = DOTS.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "integer_q8_late" in q4
    assert "kQ4LaunchVariantCoopQ8Late" in q4
    assert 'kSelectedQ4DecodePath[] = "packed"' in q4
    assert "vec_dot_q4k_q8block_late" in dots
    assert "__dp4a" in dots
    assert "0x0F0F0F0F" in dots
    late_fn = dots.split("vec_dot_q4k_q8block_late")[1].split("q4k_coop_mmv_late")[0]
    assert "__shfl_xor_sync" not in late_fn
    assert "q4k_coop_mmv_late" in dots
    assert "q4k_coop_gate_up_swiglu_late" in dots
    assert "--q4-warps" in replay
    assert "--q4-warps" in probe
    assert "integer_q8_late" in replay
    assert "M1_K256" in native
    assert "misaligned" in native
    assert "17408" in native and "5120" in native
    _ = DOTS_CU.read_text(encoding="utf-8")


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-076")
    q4 = iteration["workloads"]["q4"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(q4, "feedback")) == 12
    assert loop_product(workload_for_mode(q4, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 1
    plan = family_plan("q4", "feedback")
    assert plan["candidates"] == 3
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    accept = family_plan("q4", "acceptance")
    assert accept["candidates"] == 2
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-076", "feedback", iteration, "q4")
    assert "phase=q4" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "half-scale" in proof
    assert "two and four warps" in proof
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300


def test_tiny_cases_cover_groups_tails_and_misalignment() -> None:
    cases = tiny_cases()
    shapes = {(row["rows"], row["columns"]) for row in cases}
    assert shapes == {(1, 256), (1, 512), (17, 256), (17, 512), (33, 256), (33, 512)}
    native = NATIVE.read_text(encoding="utf-8")
    assert "group0p" in native and "group7p" in native
    assert "cancellation" in native
    assert "misaligned" in native
    for spec in TINY_CASES:
        if spec["rows"] in {17, 33}:
            assert spec["rows"] % 32 != 0
    for row in cases:
        assert row["groups"] == 8
        assert row["zero"] is True
        assert row["cancellation"] is True
        assert row["misalignment_fallback"] is True


def test_dispatch_uses_late_launch_variants() -> None:
    packed = expected_dispatch(CONFIGS[0])
    late2 = expected_dispatch(CONFIGS[1])
    late4 = expected_dispatch(CONFIGS[2])
    assert packed["gate_variant"] == "q4k_gate_up_swiglu_prequant"
    assert late2["gate_variant"] == "q4k_coop_gate_up_swiglu_late_prequant_q8"
    assert late4["down_variant"] == "q4k_coop_mmv_late_q8"
    assert late2["warps_per_row"] == 2
    assert late4["warps_per_row"] == 4
    observed = {
        "gate_variant": late2["gate_variant"],
        "up_variant": late2["up_variant"],
        "down_variant": late2["down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": 2,
    }
    assert dispatch_matches(observed, late2) is True
    wrong = dict(observed)
    wrong["warps_per_row"] = 4
    assert dispatch_matches(wrong, late2) is False
    q4 = Q4_PATH.read_text(encoding="utf-8")
    assert "kQ4LaunchVariantPairedIntegerQ8Late" in q4
    assert "kLegalQ4DecodePathIntegerQ8Late" in q4


def test_unadmitted_opt074_is_missing_evidence_not_numerical_rejection() -> None:
    coverage = opt074_q4_budgets()
    quality = opt073_quality()
    assert quality["quality_v3_engine_non_regression"] == "pass"
    stats = {
        "positive": True,
        "mean_diff_ms": 1.0,
        "ci95_high": 1.2,
        "ci95_low": 0.8,
    }
    numeric = {
        "nonfinite": 0,
        "production_pass": True,
        "legacy_pass": True,
    }
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
        survivor="late_w4",
    )
    assert verdict["verdict"] == "retain_packed"
    assert verdict["production_kept"] is False
    assert "opt074_coverage_unadmitted_missing_evidence" in verdict["reasons"]
    assert verdict["unfinished_reference_freeze_is_not_numerical_rejection"] is True
    assert verdict["retain_reason"] == "missing_evidence"


def test_negative_saving_and_below_threshold_reject() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    coverage = {"all_admitted": True}
    numeric = {"nonfinite": 0, "production_pass": True}
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
        survivor="late_w2",
    )
    assert slower["verdict"] == "performance_rejected"
    small = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 0.05,
            "ci95_high": 0.08,
            "ci95_low": 0.02,
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
        survivor="late_w4",
    )
    assert small["verdict"] == "performance_rejected"
    assert "saving_below_0_10_ms" in small["reasons"]
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
        survivor="late_w4",
    )
    assert keep["verdict"] == "keep"
    assert keep["production_kept"] is True


def test_pick_survivor_is_the_faster_late_against_packed() -> None:
    survivor = pick_survivor(
        {
            CONTROL_ID: {"mean_ms": 16.0},
            "late_w2": {"mean_ms": 12.5},
            "late_w4": {"mean_ms": 11.2},
        }
    )
    assert survivor == "late_w4"
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
            survivor="late_w4",
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
            survivor="late_w4",
        )


def test_screen_keep_is_rejected_and_counts_must_match() -> None:
    iteration = load_contract("OPT-076")
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
        stdout="QW38_OPT076_NATIVE_COUNTS=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    assert screen_keep["result_class"] == "screen_only_keep"
    observed = parse_native_observation("QW38_OPT076_RESULT=" + stdout)
    assert observed["observed_candidates"] == 3
    assert observed["keep"] is True


def test_report_and_source_forbid_half_scale_and_grid() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "opt-076" in text
    assert "packed" in text
    dots = DOTS.read_text(encoding="utf-8")
    body = dots.split("vec_dot_q4k_q8block_late")[1].split(
        "__device__ __forceinline__ __nv_bfloat16 admitted_swiglu"
    )[0]
    assert "UseQ81" not in body
    iteration = load_contract("OPT-076")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "no warps/rows/staging grid" in proof
    assert "q8/q6" in proof
    fixture = _json(FIXTURE)
    assert fixture["claims_throughput"] is False
    assert fixture["shipping_q4_decode"] == "packed"
