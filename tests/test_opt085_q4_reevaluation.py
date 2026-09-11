"""Host tests for OPT-085 Q4 decode candidate re-evaluation. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt085_q4_reevaluation import (
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    Q4_IDENTS,
    REPORT,
    VERDICT_KEYS,
    AdmissionError,
    config_by_id,
    decide_independent_verdicts,
    dispatch_matches,
    empty_verdict_row,
    evaluate_parity,
    evaluate_quality,
    expected_dispatch,
    family_plan,
    historical_reports_intact,
    load_json,
    q4_ident_catalog,
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
OPT075_REPORT = ROOT / "evidence/optimization/opt075-q4-production-admission/REPORT.md"
OPT076_REPORT = ROOT / "evidence/optimization/opt076-q4-reduction/REPORT.md"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-085")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-085"
    assert contract["claims_throughput"] is False
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["opt074_family_admission_required"] is False
    assert contract["warps_per_row"] == 4
    assert contract["half_scale_forbidden"] is True
    assert contract["at_most_one_q4_path"] is True
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        "packed_paired_staged",
        "integer_q8_paired",
        "late_w4",
    ]
    assert iteration["task"] == "OPT-085"
    assert iteration["diagnostics_make_target"] == "cuda-opt085-diagnostics"
    assert iteration["opt074_family_admission_required"] is False
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["case_ids"] == ["parity", "q4", "quality"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt085-diagnostics" in makefile
    assert "qw38-cuda-opt082-kernel-parity-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    q4 = Q4_PATH.read_text(encoding="utf-8")
    assert "integer_q8" in q4
    assert "integer_q8_late" in q4
    assert 'kSelectedQ4DecodePath[] = "packed"' in q4


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-085")
    parity = iteration["workloads"]["parity"]
    q4 = iteration["workloads"]["q4"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(parity, "feedback")) == len(
        q4_ident_catalog()
    )
    assert loop_product(workload_for_mode(q4, "feedback")) == 12
    assert loop_product(workload_for_mode(q4, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 7
    assert family_plan("parity", "feedback")["cases"] == len(q4_ident_catalog())
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
    described = describe_plan("OPT-085", "feedback", iteration, "parity")
    assert "phase=parity" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt074_coverage_unadmitted is not a blocker" in proof
    assert "half-scale" in proof
    assert "at most one q4 production path" in proof


def test_q4_ident_catalog_covers_three_idents_only() -> None:
    cases = q4_ident_catalog()
    assert {row["candidate"] for row in cases} == set(Q4_IDENTS)
    assert not any(row["candidate"] in {"fma_async", "fma_async_x"} for row in cases)
    for ident in Q4_IDENTS:
        assert any(row["candidate"] == ident and row["op"] == "mmv" for row in cases)
    assert any(row["class_name"] == "same_math_equivalence" for row in cases)
    assert any(row["op"] == "fused_gate_up" for row in cases)
    assert any(row["op"] == "eager_vs_captured" for row in cases)


def test_dispatch_uses_launch_variants_not_selector_strings() -> None:
    packed = expected_dispatch(CONFIGS[0])
    integer = expected_dispatch(CONFIGS[1])
    late = expected_dispatch(CONFIGS[2])
    assert packed["gate_variant"] == "q4k_gate_up_swiglu_prequant"
    assert integer["gate_variant"] == "q4k_coop_gate_up_swiglu_prequant_q8"
    assert late["gate_variant"] == "q4k_coop_gate_up_swiglu_late_prequant_q8"
    assert late["down_variant"] == "q4k_coop_mmv_late_q8"
    assert integer["warps_per_row"] == 4
    assert late["warps_per_row"] == 4
    observed = {
        "gate_variant": integer["gate_variant"],
        "up_variant": integer["up_variant"],
        "down_variant": integer["down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": 4,
    }
    assert dispatch_matches(observed, integer) is True
    wrong = dict(observed)
    wrong["gate_variant"] = "packed"
    assert dispatch_matches(wrong, integer) is False


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-085")
    validate_future_keep_policy("OPT-085", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-085",
            {
                "task": "OPT-085",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_opt074_unadmitted_is_not_a_keep_blocker() -> None:
    parity = {
        "by_ident": {
            "packed": {"kernel_parity_pass": True},
            "integer_q8": {"kernel_parity_pass": True},
            "integer_q8_late": {"kernel_parity_pass": True},
        }
    }
    quality = {
        CONTROL_ID: {"model_quality_pass": True},
        "integer_q8_paired": {"model_quality_pass": True},
        "late_w4": {"model_quality_pass": True},
    }
    performance = {
        CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
        "integer_q8_paired": {"performance_pass": True, "mean_diff_ms": 4.6},
        "late_w4": {"performance_pass": True, "mean_diff_ms": 6.6},
    }
    decided = decide_independent_verdicts(
        parity=parity,
        quality_by_id=quality,
        performance_by_id=performance,
        mode="acceptance",
    )
    text = json.dumps(decided)
    assert "opt074_coverage_unadmitted_missing_evidence" not in text
    assert decided["opt074_coverage_unadmitted_blocker"] is False
    assert decided["selected_path"] == "late_w4"
    assert decided["independent_verdicts"]["late_w4"]["production_kept"] is True
    assert (
        decided["independent_verdicts"]["integer_q8_paired"]["production_kept"] is False
    )
    assert decided["independent_verdicts"][CONTROL_ID]["production_kept"] is False
    kept = [
        cid
        for cid, row in decided["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert kept == ["late_w4"]


def test_independent_verdicts_and_retain_packed_without_perf() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                "packed": {"kernel_parity_pass": True},
                "integer_q8": {"kernel_parity_pass": True},
                "integer_q8_late": {"kernel_parity_pass": False},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            "integer_q8_paired": {"model_quality_pass": False, "incomplete": True},
            "late_w4": {"model_quality_pass": False},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            "integer_q8_paired": {"performance_pass": True, "mean_diff_ms": 4.6},
            "late_w4": {"performance_pass": False},
        },
        mode="acceptance",
    )
    packed = decided["independent_verdicts"][CONTROL_ID]
    integer = decided["independent_verdicts"]["integer_q8_paired"]
    assert packed["kernel_parity_pass"] is True
    assert packed["model_quality_pass"] is True
    assert packed["performance_pass"] is False
    assert packed["production_kept"] is True
    assert integer["performance_pass"] is True
    assert integer["model_quality_pass"] is False
    assert integer["production_kept"] is False
    assert decided["selected_path"] == CONTROL_ID
    assert decided["shipping_unchanged"] is True
    assert decided["production_kept"] is False


def test_feedback_cannot_keep() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {ident: {"kernel_parity_pass": True} for ident in Q4_IDENTS}
        },
        quality_by_id={row["id"]: {"model_quality_pass": True} for row in CONFIGS},
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            "integer_q8_paired": {"performance_pass": True, "mean_diff_ms": 4.6},
            "late_w4": {"performance_pass": True, "mean_diff_ms": 6.6},
        },
        mode="feedback",
    )
    assert decided["production_kept"] is False
    assert all(
        not row["production_kept"] for row in decided["independent_verdicts"].values()
    )


def test_parity_fail_rejects_candidate_before_quality() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                "packed": {"kernel_parity_pass": True},
                "integer_q8": {"kernel_parity_pass": False},
                "integer_q8_late": {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            "integer_q8_paired": {"model_quality_pass": True},
            "late_w4": {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            "integer_q8_paired": {"performance_pass": True, "mean_diff_ms": 9.0},
            "late_w4": {"performance_pass": True, "mean_diff_ms": 1.0},
        },
        mode="acceptance",
    )
    assert (
        decided["independent_verdicts"]["integer_q8_paired"]["production_kept"] is False
    )
    assert decided["selected_path"] == "late_w4"


def test_fallback_as_candidate_fails_ident_parity() -> None:
    cases = q4_ident_catalog()
    gpu_cases = []
    for row in cases:
        gpu_cases.append(
            {
                "id": row["id"],
                "candidate": row["candidate"],
                "pass": True,
                "fallback": bool(
                    row["candidate"] == "integer_q8" and not row.get("expect_fallback")
                ),
            }
        )
    with pytest.raises(AdmissionError, match="packed Q4_K kernel parity failed"):
        evaluate_parity(
            skip_gpu=True,
            synthetic={
                "gpu_cases": [
                    {
                        "id": row["id"],
                        "candidate": row["candidate"],
                        "pass": row["candidate"] != "packed",
                        "fallback": False,
                    }
                    for row in cases
                ]
            },
        )
    parity = evaluate_parity(
        skip_gpu=True,
        synthetic={"gpu_cases": gpu_cases, "source": "synthetic"},
    )
    assert parity["by_ident"]["packed"]["kernel_parity_pass"] is True
    assert parity["by_ident"]["integer_q8"]["kernel_parity_pass"] is False
    assert parity["by_ident"]["integer_q8"]["fallback_measured_as_candidate"] is True
    assert parity["opt074_coverage_unadmitted_blocker"] is False
    assert parity["opt074_family_admission_required"] is False


def test_packed_quality_uses_opt084_freeze() -> None:
    packed = evaluate_quality(config_by_id(CONTROL_ID))
    assert packed["model_quality_pass"] is True
    assert packed["identity_cached"] is True
    assert packed["quartz_baseline_regression_status"] == "pass"
    integer = evaluate_quality(config_by_id("integer_q8_paired"))
    assert integer["model_quality_pass"] is False
    assert integer["incomplete"] is True
    assert integer["reason"] == "candidate_specific_quality_not_measured"
    injected = evaluate_quality(
        config_by_id("late_w4"),
        synthetic={
            "quality_by_id": {
                "late_w4": {"model_quality_pass": True, "regression": False}
            }
        },
    )
    assert injected["model_quality_pass"] is True


def test_screen_keep_is_rejected() -> None:
    iteration = load_contract("OPT-085")
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
            "pairs": 1,
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
        stdout="QW38_OPT085_RESULT=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    assert screen_keep["result_class"] == "screen_only_keep"
    observed = parse_native_observation("QW38_OPT085_RESULT=" + stdout)
    assert observed["keep"] is True


def test_historical_opt075_076_reports_unmodified() -> None:
    intact = historical_reports_intact()
    assert intact["unmodified"] is True
    assert intact["opt075_production_kept"] is False
    assert intact["opt076_production_kept"] is False
    assert intact["opt075_shipping"] == "packed"
    assert intact["opt076_shipping"] == "packed"
    opt075 = OPT075_REPORT.read_text(encoding="utf-8")
    opt076 = OPT076_REPORT.read_text(encoding="utf-8")
    assert "OPT-075" in opt075
    assert "OPT-076" in opt076
    assert "opt074" in opt075.casefold() or "packed" in opt075.casefold()


def test_host_phases_write_fixture_and_retain_packed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt085_q4_reevaluation as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt085_q4_reevaluation.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    tmp = tmp_path / "run"
    parity = mod.run("feedback", "parity", tmp, skip_gpu=True)
    assert parity["task"] == "OPT-085"
    assert (tmp_path / "opt085_q4_reevaluation.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    assert parity["opt074_coverage_unadmitted_blocker"] is False
    dumped = json.dumps(parity.get("independent_verdicts"))
    assert "opt074_coverage_unadmitted_missing_evidence" not in dumped
    quality = mod.run("acceptance", "quality", tmp, skip_gpu=True)
    assert quality["independent_verdicts"][CONTROL_ID]["model_quality_pass"] is True
    q4 = mod.run("feedback", "q4", tmp, skip_gpu=True)
    assert q4["production_kept"] is False
    assert q4["claims_throughput"] is False
    fixture = load_json(tmp_path / "opt085_q4_reevaluation.json")
    assert fixture["shipping_q4_decode"] == "packed"
    assert fixture["shipping_ffn_decode"] == "paired_staged"
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "kernel_parity_pass" in report
    lowered = report.casefold()
    assert "not" in lowered and "blocker" in lowered
    assert empty_verdict_row()["opt074_coverage_unadmitted_blocker"] is False
    assert fixture["historical_reports"]["unmodified"] is True
    assert quality["mode"] == "acceptance"
    committed = load_json(FIXTURE)
    assert committed["task"] == "OPT-085"
    assert committed.get("q4", {}).get("this_sitting") is True


def test_committed_fixture_has_independent_verdicts() -> None:
    fixture = load_json(FIXTURE)
    report = REPORT.read_text(encoding="utf-8")
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert fixture["claims_throughput"] is False
    assert fixture["shipping_q4_decode"] == "packed"
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert kept in ([], [CONTROL_ID])
    assert fixture["independent_verdicts"][CONTROL_ID]["kernel_parity_pass"] is True
    assert fixture["independent_verdicts"][CONTROL_ID]["model_quality_pass"] is True
    assert "late_w4" in fixture["independent_verdicts"]
    assert "this_sitting" in report or fixture["q4"]["this_sitting"] is True
    assert "opt074_coverage_unadmitted" in report.casefold()
    assert "not" in report.casefold() and "blocker" in report.casefold()
