"""Host tests for OPT-135 target_guard_v2 keep policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from tools.performance_keep_policy import (
    POLICY_ID,
    POLICY_PATH,
    KeepPolicyError,
    complete_records,
    default_candidate_contract,
    evaluate,
    freeze_hash,
    load_policy,
    records_from_opt130,
    self_check,
    validate_opt_in_contract,
)
from tools.run_optimization_task import (
    SetupError,
    load_contract,
    validate_opt_in_keep_policy,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/opt135_target_guard_policy.json"
REPORT = ROOT / "evidence/optimization/opt135-target-guard-policy/REPORT.md"
OPT130 = ROOT / "fixtures/opt130_dense_attention.json"
T_CRIT = 1.8331129326536335


def _cases() -> dict[str, dict[str, Any]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["cases"]}


def independent_interval(ratios: Sequence[float]) -> tuple[float, float, float]:
    xs = [math.log(float(ratio)) for ratio in ratios]
    mean_x = sum(xs) / float(len(xs))
    geo = math.exp(mean_x)
    var = sum((item - mean_x) ** 2 for item in xs) / float(len(xs) - 1)
    se = math.sqrt(var / float(len(xs))) if var > 0.0 else 0.0
    if se == 0.0:
        return geo, geo, geo
    return (
        geo,
        math.exp(mean_x - T_CRIT * se),
        math.exp(mean_x + T_CRIT * se),
    )


def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    records = dict(case.get("records") or {})
    if case.get("source") == "opt130":
        loaded = records_from_opt130()
        loaded.update(records)
        records = loaded
    return evaluate(case["contract"], records)


def test_policy_pin_and_report_exist() -> None:
    policy = load_policy()
    assert policy["policy_id"] == POLICY_ID
    assert policy["acceptance_pairs"] == 10
    assert policy["t_critical_df9_one_sided_95"] == T_CRIT
    assert policy["target_l_exclusive_min"] == 1.0
    assert policy["guard_l_min"] == 0.98
    assert policy["not_two_sided_95_ci"] is True
    assert policy["historical_opt130_admission"] == "reject"
    assert POLICY_PATH.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    text = REPORT.read_text(encoding="utf-8")
    assert "target_guard_v2" in text
    assert "reject" in text
    assert "OPT-130" in text


def test_self_check_fixture_matches() -> None:
    result = self_check(FIXTURE, policy_path=POLICY_PATH)
    assert result["ok"] is True
    assert result["mismatches"] == 0


@pytest.mark.parametrize(
    ("case_id", "verdict"),
    [
        ("target_gain_neutral_guard_keep", "keep"),
        ("guard_ci_crosses_one_above_floor", "keep"),
        ("target_win_guard_loss_reject", "reject"),
        ("target_ci_straddles_one", "inconclusive"),
        ("guard_ci_straddles_floor", "inconclusive"),
        ("guard_equality_at_floor_keep", "keep"),
        ("target_l_exactly_one", "inconclusive"),
        ("target_all_ratios_one_reject", "reject"),
        ("target_g_1_019_keep_not_material", "keep"),
        ("decode_request_guard_fail", "reject"),
        ("aggregate_hides_failed_target", "reject"),
        ("p95_over_limit_reject", "reject"),
        ("p95_boundary_keep", "keep"),
        ("missing_quality_incomplete", "incomplete"),
        ("nan_rate_incomplete", "incomplete"),
        ("zero_rate_incomplete", "incomplete"),
        ("duplicate_sample_incomplete", "incomplete"),
        ("unpaired_incomplete", "incomplete"),
        ("mixed_capacity_incomplete", "incomplete"),
        ("wrong_pair_count_incomplete", "incomplete"),
        ("opt130_counterfactual_d2048_target", "reject"),
    ],
)
def test_fixture_verdicts(case_id: str, verdict: str) -> None:
    result = _run_case(_cases()[case_id])
    assert result["verdict"] == verdict
    assert result["shipping_impact"] == 0
    assert result["interval_label"] == "one_sided_95_bounds"


def test_neutral_guard_keep_even_when_guard_l_le_one() -> None:
    result = _run_case(_cases()["target_gain_neutral_guard_keep"])
    guard = result["metrics"]["d128.decode_only"]
    target = result["metrics"]["d2048.decode_only"]
    assert target["decision"] == "pass"
    assert target["L"] > 1.0
    assert guard["L"] <= 1.0
    assert guard["L"] >= 0.98
    assert guard["decision"] == "pass"
    assert result["verdict"] == "keep"


def test_independent_neutral_failed_and_inconclusive_intervals() -> None:
    cases = _cases()
    keep = _run_case(cases["target_gain_neutral_guard_keep"])
    geo, lower, upper = independent_interval([1.02] * 10)
    target = keep["metrics"]["d2048.decode_only"]
    assert target["g"] == pytest.approx(geo)
    assert target["L"] == pytest.approx(lower)
    assert target["U"] == pytest.approx(upper)
    assert lower > 1.0

    guard_geo, guard_l, guard_u = independent_interval([1.0] * 10)
    guard = keep["metrics"]["d128.complete_request"]
    assert guard["g"] == pytest.approx(guard_geo)
    assert guard["L"] == pytest.approx(guard_l)
    assert guard["U"] == pytest.approx(guard_u)
    assert guard_l <= 1.0
    assert guard_l >= 0.98

    failed = _run_case(cases["target_win_guard_loss_reject"])
    _, failed_l, failed_u = independent_interval([0.97] * 10)
    lost = failed["metrics"]["d128.decode_only"]
    assert lost["L"] == pytest.approx(failed_l)
    assert lost["U"] == pytest.approx(failed_u)
    assert failed_u < 0.98
    assert lost["decision"] == "reject"
    assert failed["verdict"] == "reject"

    split = [0.99] * 5 + [1.01] * 5
    _, straddle_l, straddle_u = independent_interval(split)
    straddled = _run_case(cases["target_ci_straddles_one"])
    target_s = straddled["metrics"]["d2048.decode_only"]
    assert target_s["L"] == pytest.approx(straddle_l)
    assert target_s["U"] == pytest.approx(straddle_u)
    assert straddle_l <= 1.0 < straddle_u
    assert straddled["verdict"] == "inconclusive"


def test_guard_floor_straddle_and_equality() -> None:
    split = [0.97] * 5 + [0.99] * 5
    _, lower, upper = independent_interval(split)
    result = _run_case(_cases()["guard_ci_straddles_floor"])
    guard = result["metrics"]["d128.decode_only"]
    assert guard["L"] == pytest.approx(lower)
    assert guard["U"] == pytest.approx(upper)
    assert lower < 0.98 <= upper
    assert result["verdict"] == "inconclusive"
    equal = _run_case(_cases()["guard_equality_at_floor_keep"])
    assert equal["metrics"]["d128.decode_only"]["L"] == pytest.approx(0.98)
    assert equal["verdict"] == "keep"


def test_material_2pct_is_not_a_keep_threshold() -> None:
    result = _run_case(_cases()["target_g_1_019_keep_not_material"])
    target = result["metrics"]["d2048.decode_only"]
    assert result["verdict"] == "keep"
    assert target["g"] == pytest.approx(1.019)
    assert target["L"] > 1.0
    assert target["material_2pct"] is False


def test_p95_boundary_and_violation() -> None:
    over = _run_case(_cases()["p95_over_limit_reject"])
    assert over["verdict"] == "reject"
    assert any("p95" in reason for reason in over["reasons"])
    boundary = _run_case(_cases()["p95_boundary_keep"])
    assert boundary["verdict"] == "keep"
    ratios = boundary["metrics"]["d2048.decode_only"]["p95_ratios"]
    assert ratios
    assert max(float(item) for item in ratios) == pytest.approx(1.05)


def test_unknown_policy_and_hash_fail_closed() -> None:
    unknown = _cases()["unknown_policy_id"]
    with pytest.raises(KeepPolicyError, match="unknown keep policy id"):
        evaluate(unknown["contract"], unknown.get("records") or {"pairs": []})
    hashed = _cases()["frozen_hash_mismatch"]
    with pytest.raises(KeepPolicyError, match="hash mismatch"):
        evaluate(hashed["contract"], hashed["records"])


def test_opt130_counterfactual_preserves_reject() -> None:
    result = _run_case(_cases()["opt130_counterfactual_d2048_target"])
    assert OPT130.is_file()
    historical = json.loads(OPT130.read_text(encoding="utf-8"))
    assert historical["verdict"] == "reject"
    assert result["verdict"] == "reject"
    assert result["historical_verdict"] == "reject"
    assert result["counterfactual"] is True
    d2048 = result["metrics"]["d2048.decode_only"]
    d128 = result["metrics"]["d128.decode_only"]
    p4096 = result["metrics"]["p4096.prefill"]
    assert d2048["decision"] == "pass"
    assert d2048["L"] > 1.0
    assert d2048["material_2pct"] is True
    assert d128["decision"] == "pass"
    assert d128["L"] >= 0.98
    assert p4096["decision"] == "pass"
    raw = records_from_opt130()
    decode_pairs = [
        row
        for row in raw["pairs"]
        if row["workload"] == "d2048" and row["metric"] == "decode_only"
    ]
    ratios = [
        float(row["candidate_rate"]) / float(row["control_rate"])
        for row in decode_pairs
    ]
    geo, lower, upper = independent_interval(ratios)
    assert d2048["g"] == pytest.approx(geo)
    assert d2048["L"] == pytest.approx(lower)
    assert d2048["U"] == pytest.approx(upper)


def test_legacy_iteration_contracts_unchanged() -> None:
    contract = load_contract("OPT-057")
    assert "keep_policy_id" not in contract
    validate_opt_in_keep_policy("OPT-057", contract)


def test_schema_and_hash_helpers() -> None:
    contract = default_candidate_contract()
    digest = freeze_hash(contract)
    assert len(digest) == 64
    validate_opt_in_contract(contract)
    with pytest.raises(KeepPolicyError, match="unknown keep policy id"):
        validate_opt_in_contract({"policy_id": "other"})
    overlapping = default_candidate_contract(
        guards=[{"workload": "d2048", "kind": "decode"}]
    )
    with pytest.raises(KeepPolicyError, match="not disjoint"):
        validate_opt_in_contract(overlapping)
    with pytest.raises(SetupError, match="unknown keep policy id"):
        validate_opt_in_keep_policy("OPT-999", {"keep_policy_id": "nope"})


def test_evaluate_rejects_quality_fail() -> None:
    contract = default_candidate_contract()
    records = complete_records(target_decode=[1.02] * 10, quality=False)
    result = evaluate(contract, records)
    assert result["verdict"] == "reject"
    assert any("quality_fail" in reason for reason in result["reasons"])


def test_nan_rate_is_incomplete() -> None:
    contract = default_candidate_contract()
    records = complete_records(target_decode=[1.02] * 10)
    records["pairs"][0]["candidate_rate"] = math.nan
    result = evaluate(contract, records)
    assert result["verdict"] == "incomplete"
