"""Host tests for OPT-091 successor quality gate. GPU-free."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools.opt089_q4_promotion import (
    CANDIDATE_ID,
    CONTROL_ID,
    FIXTURE as OPT089_FIXTURE,
)
from tools.opt091_quality_tradeoff import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    QUALITY_CONTRACT_ID,
    REPORT,
    STRICT_CONTRACT_ID,
    AdmissionError,
    authenticated_opt089_measured,
    build_ppl_ratios,
    concession_route,
    contract_payload,
    family_plan,
    policy_record,
    regression_release_quality_pass,
    run,
    strict_fail_solely_on_ppl_window,
    timed_phase_record,
)
from tools.quality.compare import evaluate_ppl_contract
from tools.quality.suite import (
    PPL_RATIO_MAX,
    QUALITY_CONTRACT_SPECS,
    evaluate_quality_contracts,
    ppl_1024_spans,
    quality_contract_spec,
    recurrence_nll,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _measured(
    held_nll: float = 1.7872306470098762,
    *,
    recurrence: float = 0.0,
    functional_failures: int = 0,
    greedy_mismatch: bool = False,
) -> dict[str, Any]:
    return {
        "cases": [{"name": "held_out_wikitext_1024", "mean_nll": held_nll}],
        "recurrence_incremental_nll": recurrence,
        "new_functional_failures": functional_failures,
        "new_greedy_mismatch": greedy_mismatch,
    }


def test_contracts_makefile_and_iteration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-091")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert contract["task"] == "OPT-091"
    assert contract["quality_contract_id"] == QUALITY_CONTRACT_ID
    assert contract["strict_quality_contract_id"] == STRICT_CONTRACT_ID
    assert contract["default_ppl_ratio_max"] == 1.01
    assert contract["candidate_ppl_ratio_max"] == 1.015
    assert contract["aggregate_ppl_ratio_max"] == 1.015
    assert contract["eligible_candidate"] == "late_w4"
    assert contract["anchors"] == ["opt084_frozen", "opt088_authenticated"]
    assert contract["historical_gate_relabel"] is False
    assert contract["claims_throughput"] is False
    assert iteration["host_only"] is True
    assert iteration["skip_compile"] is True
    assert iteration["diagnostics_make_target"] == "cuda-opt091-diagnostics"
    assert iteration["case_ids"] == [
        "policy",
        "quality",
        "q4",
        "d128",
        "d2048",
        "decision",
    ]
    assert "cuda-opt091-diagnostics" in makefile
    assert "qw38-cuda-opt082-kernel-parity-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    policy = iteration["workloads"]["policy"]
    assert loop_product(workload_for_mode(policy, "feedback")) == 1
    described = describe_plan("OPT-091", "feedback", iteration, "policy")
    assert "phase=policy" in described
    assert "historical_oracles=none" in described


def test_quality_contract_specs_and_helpers() -> None:
    strict = quality_contract_spec(STRICT_CONTRACT_ID)
    successor = quality_contract_spec(QUALITY_CONTRACT_ID)
    assert strict["ppl_ratio_max"] == 1.01
    assert successor["candidate_ppl_ratio_max"] == 1.015
    assert PPL_RATIO_MAX == 1.01
    default_span = ppl_1024_spans()
    successor_span = ppl_1024_spans(QUALITY_CONTRACT_ID)
    assert default_span["proposed_ppl_ratio_max"] == 1.01
    assert successor_span["proposed_ppl_ratio_max"] == 1.015
    default_rec = recurrence_nll()
    successor_rec = recurrence_nll(QUALITY_CONTRACT_ID)
    assert default_rec["proposed_max"] == 0.02
    assert successor_rec["proposed_max"] == 0.02
    assert evaluate_ppl_contract(
        {"opt088_authenticated": 1.005, "opt084_frozen": 1.004},
        strict,
    )["pass"] is True
    assert evaluate_ppl_contract(
        {"opt088_authenticated": 1.016, "opt084_frozen": 1.012},
        strict,
    )["pass"] is False
    assert evaluate_ppl_contract(
        {"opt088_authenticated": 1.014, "opt084_frozen": 1.013},
        successor,
    )["pass"] is True
    assert evaluate_ppl_contract(
        {"opt088_authenticated": 1.016, "opt084_frozen": 1.016},
        successor,
    )["pass"] is False


def test_evaluate_quality_contracts_negative_examples() -> None:
    strict_fail = evaluate_quality_contracts(
        ratios={"opt088_authenticated": 1.016, "opt084_frozen": 1.016},
        recurrence_incremental_nll=0.0,
        contract_ids=[STRICT_CONTRACT_ID, QUALITY_CONTRACT_ID],
    )
    assert strict_fail["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False
    assert strict_fail["contracts"][QUALITY_CONTRACT_ID]["model_quality_pass"] is False
    incomplete = evaluate_quality_contracts(
        ratios={"opt088_authenticated": 1.005, "opt084_frozen": 1.005},
        recurrence_incremental_nll=0.0,
        incomplete=True,
    )
    assert incomplete["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False
    recurrence = evaluate_quality_contracts(
        ratios={"opt088_authenticated": 1.005, "opt084_frozen": 1.005},
        recurrence_incremental_nll=0.021,
    )
    assert recurrence["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False
    changed = evaluate_quality_contracts(
        ratios={"opt088_authenticated": 1.005, "opt084_frozen": 1.005},
        recurrence_incremental_nll=0.0,
        changed_inherited_answer=True,
    )
    assert changed["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False
    greedy = evaluate_quality_contracts(
        ratios={"opt088_authenticated": 1.005, "opt084_frozen": 1.005},
        recurrence_incremental_nll=0.0,
        greedy_mismatch=True,
    )
    assert greedy["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False


def test_concession_window_and_ppl_only_rejects() -> None:
    control_nll = 1.7878782710057632
    measured = _measured(control_nll + math.log(1.012))
    evidence = {
        "contracts": evaluate_quality_contracts(
            ratios=build_ppl_ratios(measured),
            recurrence_incremental_nll=0.0,
            contract_ids=[STRICT_CONTRACT_ID, QUALITY_CONTRACT_ID],
        ),
        "ratios": build_ppl_ratios(measured),
        "recurrence_incremental_nll": 0.0,
        "functional_failures": 0,
        "greedy_mismatch": False,
        "changed_inherited_answer": False,
        "incomplete": False,
        "strict_model_quality_pass": False,
        "successor_model_quality_pass": True,
    }
    assert strict_fail_solely_on_ppl_window(evidence) is True
    route = concession_route(evidence, parity_pass=True, performance_pass=False)
    assert route["concession_eligible"] is True
    assert route["concession_used"] is False
    assert route["reason"] == "performance_gate_not_met"
    parity_fail = concession_route(evidence, parity_pass=False, performance_pass=True)
    assert parity_fail["concession_used"] is False
    assert parity_fail["reason"] == "parity_or_same_math_failure"


def test_opt089_authenticated_policy_no_concession() -> None:
    opt089 = authenticated_opt089_measured()
    measured = opt089["measured"]
    ratios = build_ppl_ratios(measured)
    assert ratios["opt088_authenticated"] <= 1.01
    assert ratios["opt084_frozen"] <= 1.01
    policy = policy_record(mode="feedback", measured=measured, parity_pass=True)
    assert policy["strict_model_quality_pass"] is True
    assert policy["successor_model_quality_pass"] is True
    assert policy["regression_release_quality_pass"] is True
    assert policy["concession_used"] is False
    assert policy["claims_throughput"] is False
    assert policy["production_kept"] is False
    assert policy["timed_phases_status"] == "not_applicable"
    assert policy["quality_contract_id"] == QUALITY_CONTRACT_ID
    assert policy["independent_verdicts"][CANDIDATE_ID]["model_quality_pass"] is True
    assert policy["independent_verdicts"][CONTROL_ID]["production_kept"] is False


def test_timed_phases_not_applicable_without_concession() -> None:
    policy = policy_record(mode="feedback")
    for phase in ("q4", "d128", "d2048", "decision"):
        record = timed_phase_record(phase, "acceptance", policy)
        assert record["status"] == "not_applicable"
        assert record["gpu_work"] is False
        assert record["claims_throughput"] is False


def test_host_policy_writes_fixture_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt091_quality_tradeoff as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt091_quality_tradeoff.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    result = mod.run("feedback", "policy", tmp_path / "run")
    assert result["concession_used"] is False
    assert result["claims_throughput"] is False
    assert (tmp_path / "opt091_quality_tradeoff.json").is_file()
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "opt091_late_w4_v1" in report
    assert "concession_used=False" in report


def test_committed_fixture_matches_opt089_strict_pass() -> None:
    fixture = _json(FIXTURE) if FIXTURE.is_file() else {}
    if not fixture:
        pytest.skip("fixture not yet materialized")
    opt089 = _json(OPT089_FIXTURE)
    assert opt089["independent_verdicts"][CANDIDATE_ID]["model_quality_pass"] is True
    assert fixture["concession_used"] is False
    assert fixture["claims_throughput"] is False
    assert fixture["strict_model_quality_pass"] is True
    assert fixture["successor_model_quality_pass"] is True
    assert fixture["timed_phases_status"] == "not_applicable"
    assert fixture["quality_contract_id"] == QUALITY_CONTRACT_ID


def test_selector_mismatch_cache_fails_closed(tmp_path: Path) -> None:
    bad = {
        "measured": _measured(),
        "incomplete": True,
        "parity_pass": True,
    }
    policy = policy_record(mode="feedback", synthetic=bad)
    assert policy["regression_release_quality_pass"] is False
    assert policy["concession_used"] is False


def test_family_plan_policy_feedback() -> None:
    plan = family_plan("policy", "feedback")
    assert plan["phase"] == "policy"
    assert plan["tier"] == "correctness"


def test_contract_payload_matches_specs() -> None:
    contract = contract_payload()
    assert contract["quality_contract_id"] in QUALITY_CONTRACT_SPECS
    assert contract["strict_quality_contract_id"] in QUALITY_CONTRACT_SPECS
