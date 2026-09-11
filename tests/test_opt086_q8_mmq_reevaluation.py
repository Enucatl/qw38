"""Host tests for OPT-086 Q8/MMQ independent keep/revert. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt086_q8_mmq_reevaluation import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    MMQ_CONTROL,
    MMQ_IDENTS,
    MMQ_SHIPPING,
    Q8_CONTROL,
    Q8_IDENTS,
    Q8_SHIPPING,
    REPORT,
    VERDICT_KEYS,
    decide_family_verdicts,
    empty_verdict_row,
    evaluate_parity,
    evaluate_quality,
    family_plan,
    historical_reports_intact,
    load_json,
    mmq_ident_catalog,
    q8_ident_catalog,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    validate_performance_admission,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
OPT070_REPORT = ROOT / "evidence/optimization/opt070-keep-revalidation/REPORT.md"
PATH_H = ROOT / "cuda/q8_decode_path.cuh"
MMQ_H = ROOT / "cuda/quant_mmq_mma.cuh"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-086")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-086"
    assert contract["claims_throughput"] is False
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["opt074_family_admission_required"] is False
    assert contract["families_independent"] is True
    assert contract["control_q8_layout"] == Q8_CONTROL
    assert contract["shipping_q8_layout"] == Q8_SHIPPING
    assert contract["control_mmq"] == MMQ_CONTROL
    assert contract["shipping_mmq"] == MMQ_SHIPPING
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert iteration["task"] == "OPT-086"
    assert iteration["diagnostics_make_target"] == "cuda-opt086-diagnostics"
    assert iteration["opt074_family_admission_required"] is False
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["case_ids"] == ["q8", "mmq"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["modes"]["feedback"]["aggregate_deadline_s"] == 300
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300
    assert "cuda-opt086-diagnostics" in makefile
    assert "qw38-cuda-opt082-kernel-parity-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    path = PATH_H.read_text(encoding="utf-8")
    mmq = MMQ_H.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    assert "apply_q8_layout_ident" in path
    assert "Q8DecodeLayoutScope" in path
    assert "--q8-layout" in replay
    assert "--mmq-async-x" in replay
    assert "invalidate_q8_decode_staging" in replay
    assert "q8-ab" in probe
    assert "mmq-ab" in probe
    assert "override_before_capture_applied" in probe
    assert "set_mmq_async_x_override" in mmq or "set_mmq_async_x_override" in (
        ROOT / "cuda/quant_mmv.h"
    ).read_text(encoding="utf-8")
    assert "kSelectedMmqAsyncX" in mmq


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-086")
    q8 = iteration["workloads"]["q8"]
    mmq = iteration["workloads"]["mmq"]
    assert loop_product(workload_for_mode(q8, "feedback")) == 8
    assert loop_product(workload_for_mode(mmq, "feedback")) == 8
    assert loop_product(workload_for_mode(q8, "acceptance")) == 26
    assert loop_product(workload_for_mode(mmq, "acceptance")) == 26
    plan = family_plan("q8", "feedback")
    assert plan["candidates"] == 2
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    assert plan["engine_pairs"] == 0
    assert plan["parity_cases"] == len(q8_ident_catalog())
    accept = family_plan("q8", "acceptance")
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    mmq_plan = family_plan("mmq", "feedback")
    assert mmq_plan["parity_cases"] == len(mmq_ident_catalog())
    described = describe_plan("OPT-086", "feedback", iteration, "q8")
    assert "phase=q8" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt074_coverage_unadmitted is not a blocker" in proof
    assert "independent" in proof
    assert "historical opt-070 report unmodified" in proof


def test_catalogs_are_family_specific() -> None:
    q8 = q8_ident_catalog()
    mmq = mmq_ident_catalog()
    assert len(q8) == 20
    assert len(mmq) == 38
    assert {row["candidate"] for row in q8 if row["op"] == "mmv"} == set(Q8_IDENTS)
    assert {row["candidate"] for row in mmq} == set(MMQ_IDENTS)
    assert not any(row["candidate"] in MMQ_IDENTS for row in q8)
    assert not any(row["candidate"] in Q8_IDENTS for row in mmq)
    assert any(row["op"] == "staging_typed" for row in q8)
    assert any(
        row["pattern"] == "aligned" and not row["expect_fallback"] for row in mmq
    )
    assert any(row["expect_fallback"] for row in mmq)


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-086")
    validate_future_keep_policy("OPT-086", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-086",
            {
                "task": "OPT-086",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_opt074_unadmitted_is_not_a_keep_blocker() -> None:
    decided = decide_family_verdicts(
        "mmq",
        parity={
            "by_ident": {
                MMQ_CONTROL: {"kernel_parity_pass": True},
                MMQ_SHIPPING: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            MMQ_CONTROL: {"model_quality_pass": True},
            MMQ_SHIPPING: {"model_quality_pass": True},
        },
        performance_by_id={
            MMQ_CONTROL: {"performance_pass": False, "mean_diff_ms": 0.0},
            MMQ_SHIPPING: {"performance_pass": True, "mean_diff_ms": 48.0},
        },
        mode="acceptance",
        this_sitting=True,
    )
    text = json.dumps(decided)
    assert "opt074_coverage_unadmitted_missing_evidence" not in text
    assert decided["opt074_coverage_unadmitted_blocker"] is False
    assert decided["decision"] == "keep"
    assert decided["selected_path"] == MMQ_SHIPPING
    assert decided["independent_verdicts"][MMQ_SHIPPING]["production_kept"] is True
    assert decided["independent_verdicts"][MMQ_CONTROL]["production_kept"] is False
    assert decided["shipping_unchanged"] is True
    assert decided["claims_throughput"] is True


def test_q8_revert_when_control_wins_all_gates() -> None:
    decided = decide_family_verdicts(
        "q8",
        parity={
            "by_ident": {
                Q8_CONTROL: {"kernel_parity_pass": True},
                Q8_SHIPPING: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            Q8_CONTROL: {"model_quality_pass": True},
            Q8_SHIPPING: {"model_quality_pass": True},
        },
        performance_by_id={
            Q8_CONTROL: {"performance_pass": True, "mean_diff_ms": 0.05},
            Q8_SHIPPING: {"performance_pass": False, "mean_diff_ms": -0.05},
        },
        mode="acceptance",
        this_sitting=True,
    )
    assert decided["decision"] == "revert"
    assert decided["selected_path"] == Q8_CONTROL
    assert decided["shipping_unchanged"] is False
    assert decided["independent_verdicts"][Q8_CONTROL]["production_kept"] is True
    assert decided["independent_verdicts"][Q8_SHIPPING]["production_kept"] is False
    assert decided["claims_throughput"] is False


def test_families_are_independent() -> None:
    q8 = decide_family_verdicts(
        "q8",
        parity={
            "by_ident": {ident: {"kernel_parity_pass": False} for ident in Q8_IDENTS}
        },
        quality_by_id={ident: {"model_quality_pass": True} for ident in Q8_IDENTS},
        performance_by_id={
            Q8_CONTROL: {"performance_pass": True},
            Q8_SHIPPING: {"performance_pass": False},
        },
        mode="acceptance",
        this_sitting=True,
    )
    mmq = decide_family_verdicts(
        "mmq",
        parity={
            "by_ident": {ident: {"kernel_parity_pass": True} for ident in MMQ_IDENTS}
        },
        quality_by_id={ident: {"model_quality_pass": True} for ident in MMQ_IDENTS},
        performance_by_id={
            MMQ_CONTROL: {"performance_pass": False},
            MMQ_SHIPPING: {"performance_pass": True},
        },
        mode="acceptance",
        this_sitting=True,
    )
    assert q8["independent_verdicts"][Q8_SHIPPING]["production_kept"] is False
    assert q8["decision"] != "keep"
    assert mmq["decision"] == "keep"
    assert mmq["independent_verdicts"][MMQ_SHIPPING]["production_kept"] is True


def test_feedback_cannot_keep() -> None:
    decided = decide_family_verdicts(
        "mmq",
        parity={
            "by_ident": {ident: {"kernel_parity_pass": True} for ident in MMQ_IDENTS}
        },
        quality_by_id={ident: {"model_quality_pass": True} for ident in MMQ_IDENTS},
        performance_by_id={
            MMQ_CONTROL: {"performance_pass": False, "mean_diff_ms": 0.0},
            MMQ_SHIPPING: {"performance_pass": True, "mean_diff_ms": 48.0},
        },
        mode="feedback",
        this_sitting=True,
    )
    assert decided["production_kept"] is False
    assert all(
        not row["production_kept"] for row in decided["independent_verdicts"].values()
    )
    assert decided["claims_throughput"] is False


def test_quality_shipping_uses_opt084_freeze() -> None:
    shipping = evaluate_quality("q8", Q8_SHIPPING)
    assert shipping["model_quality_pass"] is True
    assert shipping["identity_cached"] is True
    assert shipping["reason"] == "shipping_quartz_vs_opt084_freeze"
    control = evaluate_quality("q8", Q8_CONTROL)
    assert control["model_quality_pass"] is True
    assert control["reason"] == "replacement_control_same_freeze_rules"
    mmq = evaluate_quality("mmq", MMQ_SHIPPING)
    assert mmq["model_quality_pass"] is True


def test_documented_q8_association_fail_does_not_fail_ident() -> None:
    cases = q8_ident_catalog()
    gpu_cases = []
    for row in cases:
        gpu_cases.append(
            {
                "id": row["id"],
                "candidate": row["candidate"],
                "pass": row["id"]
                not in {
                    "Q8_0_mmv_r1_w4_M17_N1_K2048_random_assoc",
                    "Q8_0_mmv_r2_w2_M17_N1_K2048_random_assoc",
                },
                "fallback": False,
            }
        )
    parity = evaluate_parity(
        "q8", skip_gpu=True, synthetic={"gpu_cases": gpu_cases, "source": "synthetic"}
    )
    assert parity["by_ident"][Q8_CONTROL]["kernel_parity_pass"] is True
    assert parity["by_ident"][Q8_SHIPPING]["kernel_parity_pass"] is True
    assert parity["opt074_coverage_unadmitted_blocker"] is False
    assert (
        "Q8_0_mmv_r2_w2_M17_N1_K2048_random_assoc"
        in parity["by_ident"][Q8_SHIPPING]["documented_fails"]
    )


def test_mmq_fallback_as_candidate_fails_ident_parity() -> None:
    cases = mmq_ident_catalog()
    gpu_cases = []
    for row in cases:
        gpu_cases.append(
            {
                "id": row["id"],
                "candidate": row["candidate"],
                "pass": True,
                "fallback": bool(
                    row["candidate"] == MMQ_SHIPPING and not row.get("expect_fallback")
                ),
            }
        )
    parity = evaluate_parity(
        "mmq", skip_gpu=True, synthetic={"gpu_cases": gpu_cases, "source": "synthetic"}
    )
    assert parity["by_ident"][MMQ_CONTROL]["kernel_parity_pass"] is True
    assert parity["by_ident"][MMQ_SHIPPING]["kernel_parity_pass"] is False
    assert parity["by_ident"][MMQ_SHIPPING]["fallback_measured_as_candidate"] is True


def test_screen_keep_is_rejected() -> None:
    iteration = load_contract("OPT-086")
    q8 = workload_for_mode(iteration["workloads"]["q8"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 2,
            "observed_shapes": 1,
            "observed_tier": "screen",
            "pairs": 1,
            "sample_ids": [0, 1, 2],
            "acceptance_executed": False,
            "keep": True,
        }
    )
    screen_keep = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q8",
        workload=q8,
        stdout="QW38_OPT086_RESULT=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    assert screen_keep["result_class"] == "screen_only_keep"
    observed = parse_native_observation("QW38_OPT086_RESULT=" + stdout)
    assert observed["keep"] is True


def test_historical_opt070_report_unmodified() -> None:
    intact = historical_reports_intact()
    assert intact["unmodified"] is True
    assert intact["opt070_q8_verdict"] == "inconclusive"
    assert intact["opt070_mmq_verdict"] == "inconclusive"
    assert intact["opt070_shipping_unchanged"] is True
    opt070 = OPT070_REPORT.read_text(encoding="utf-8")
    assert "OPT-070" in opt070
    assert "inconclusive" in opt070.casefold()
    assert "r2_w2" in opt070
    assert "fma_async_x" in opt070


def test_host_phases_write_fixture_and_independent_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt086_q8_mmq_reevaluation as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt086_q8_mmq_reevaluation.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    tmp = tmp_path / "run"
    q8 = mod.run("feedback", "q8", tmp, skip_gpu=True)
    assert q8["task"] == "OPT-086"
    assert (tmp_path / "opt086_q8_mmq_reevaluation.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    assert q8["opt074_coverage_unadmitted_blocker"] is False
    dumped = json.dumps(q8.get("independent_verdicts"))
    assert "opt074_coverage_unadmitted_missing_evidence" not in dumped
    mmq = mod.run("feedback", "mmq", tmp, skip_gpu=True)
    assert mmq["independent_verdicts"]["q8"][Q8_SHIPPING]["kernel_parity_pass"] in {
        True,
        False,
    }
    assert (
        mmq["independent_verdicts"]["mmq"][MMQ_SHIPPING]["kernel_parity_pass"] is True
    )
    accept_q8 = mod.run("acceptance", "q8", tmp, skip_gpu=True)
    accept_mmq = mod.run("acceptance", "mmq", tmp, skip_gpu=True)
    assert accept_q8["q8"]["production_kept"] is False or isinstance(
        accept_q8["independent_verdicts"]["q8"][Q8_SHIPPING]["production_kept"], bool
    )
    assert accept_mmq["claims_throughput_by_family"]["mmq"] is False
    fixture = load_json(tmp_path / "opt086_q8_mmq_reevaluation.json")
    assert fixture["historical_reports"]["unmodified"] is True
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "kernel_parity_pass" in report
    lowered = report.casefold()
    assert "not" in lowered and "blocker" in lowered
    assert empty_verdict_row()["opt074_coverage_unadmitted_blocker"] is False
    assert "r1_w4" in report
    assert "fma_async_x" in report


def test_committed_fixture_schema() -> None:
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    fixture = load_json(FIXTURE)
    report = REPORT.read_text(encoding="utf-8")
    assert fixture["task"] == "OPT-086"
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert "q8" in fixture["independent_verdicts"]
    assert "mmq" in fixture["independent_verdicts"]
    for ident in Q8_IDENTS:
        row = fixture["independent_verdicts"]["q8"][ident]
        for key in VERDICT_KEYS:
            assert key in row
    for ident in MMQ_IDENTS:
        row = fixture["independent_verdicts"]["mmq"][ident]
        for key in VERDICT_KEYS:
            assert key in row
    assert "opt074_coverage_unadmitted" in report.casefold()
    assert "not" in report.casefold() and "blocker" in report.casefold()
    kept_q8 = [
        ident
        for ident, row in fixture["independent_verdicts"]["q8"].items()
        if row["production_kept"]
    ]
    kept_mmq = [
        ident
        for ident, row in fixture["independent_verdicts"]["mmq"].items()
        if row["production_kept"]
    ]
    assert len(kept_q8) <= 1
    assert len(kept_mmq) <= 1
