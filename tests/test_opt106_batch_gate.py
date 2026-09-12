"""Host tests for the OPT-106 combined post-098 outcome gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt073_quality_policy import QualityPolicyError
from tools.opt080_batch_gate import MARGIN, plus5_gap_ms, parity_gap_ms
from tools.opt106_batch_gate import (
    CONTRACT,
    EXPECTED_PATHS,
    FAMILIES,
    ITERATION,
    OPT098_CONTRACT,
    OPT098_FIXTURE,
    OPT098_REPORT,
    BatchGateError,
    audit_dependencies,
    concession_active,
    evaluate_combination_quality,
    family_kernel_parity,
    frozen_combined_config,
    load_contract,
    opt098_artifacts_unmodified,
    post098_control_paths,
    source_paths,
    three_outcomes,
    validate_batch_result,
    validate_report_agrees,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract as load_iteration,
    loop_product,
    validate_future_keep_policy,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_fields() -> dict[str, Any]:
    opt098 = _json(OPT098_FIXTURE)
    vs_base = copy.deepcopy(opt098["quartz_vs_baseline_quality_delta"])
    vs_llama = copy.deepcopy(opt098["quartz_vs_llama_quality_delta"])
    recurrence = copy.deepcopy(opt098["recurrence_state_status"])
    return {
        "model_quality_pass": True,
        "absolute_quality_status": "fail",
        "quartz_baseline_regression_status": "pass",
        "quartz_vs_baseline_quality_delta": vs_base,
        "quartz_vs_llama_quality_delta": vs_llama,
        "recurrence_state_status": recurrence,
        "single_boolean": None,
    }


def _synthetic_measured() -> dict[str, Any]:
    freeze = frozen_combined_config()
    opt098 = _json(OPT098_FIXTURE)
    quality = _quality_fields()
    from tools.opt080_batch_gate import compute_gate

    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-106",
        "status": "measured",
        "measurement_utc": "2026-09-12T18:00:00Z",
        "device": opt098["device"],
        "compute_capability": "12.0",
        "power_limit_w": 400.0,
        "telemetry": copy.deepcopy(opt098["telemetry"]),
        "llama_revision": opt098["llama_revision"],
        "gguf_sha256": opt098["gguf_sha256"],
        "source_revision": "synthetic",
        "source_state": "dirty",
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        "post098_control": freeze["post098_control"],
        "post106_selected": freeze["post106_selected"],
        "combined_production_paths": freeze["combined_production_paths"],
        "selected_equals_control": True,
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "kernel_parity_pass": freeze["kernel_parity_pass"],
        "matched_family_attribution": freeze["matched_family_attribution"],
        **quality,
        "performance_pass": False,
        "production_kept": True,
        "release_eligible": True,
        "p": copy.deepcopy(opt098["p"]),
        "d128": copy.deepcopy(opt098["d128"]),
        "d2048": copy.deepcopy(opt098["d2048"]),
        "opt016": copy.deepcopy(opt098["opt016"]),
        "quality": copy.deepcopy(opt098["quality"]),
        "gaps": copy.deepcopy(opt098["gaps"]),
        "state_isolation": copy.deepcopy(opt098["state_isolation"]),
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "preflight_is_release_evidence": False,
        "opt074_coverage_unadmitted_blocker": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": " ".join(load_contract()["proof_limit"]),
        "report_path": "evidence/optimization/opt106-batch-gate/REPORT.md",
    }
    result["p"]["quartz"]["attribution"] = None
    result["p"]["quartz"]["instrumented"] = False
    result["d128"]["quartz"]["attribution"] = None
    result["d2048"]["quartz"]["attribution"] = None
    result["opt016"]["gate_passed"] = False
    result["gate"] = compute_gate(result)
    result["outcomes"] = three_outcomes(result)
    result["performance_pass"] = False
    return result


def test_contracts_iteration_plan_and_freeze() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-106")
    freeze = frozen_combined_config()
    paths = source_paths()
    control = post098_control_paths()
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-106"
    assert contract["strict_ppl_ratio_max"] == 1.01
    assert contract["concession_active"] is False
    assert contract["selected_equals_control"] is True
    assert contract["reuse_historical_llama"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt106-diagnostics"
    assert "freeze" in iteration["modes"]["feedback"]["tier_sequence"]
    assert "report" in iteration["modes"]["acceptance"]["tier_sequence"]
    assert "quality" in iteration["modes"]["release"]["workloads"]
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt106-diagnostics" in makefile
    assert paths["q4_decode"] == "integer_q8_late"
    assert paths["ffn_decode"] == "paired_integer"
    assert paths["q8_grouping"] == "grouped_r1_w4"
    assert paths["q8_device_layout"] == "raw_gguf"
    assert paths["q6_device_layout"] == "raw_gguf"
    assert paths["mmq_double_x"] is False
    assert control["q4_decode"] == "integer_q8_late"
    assert control["ffn_decode"] == "paired_integer"
    for key, value in EXPECTED_PATHS.items():
        assert freeze["combined_production_paths"][key] == value
    assert freeze["selected_equals_control"] is True
    assert (
        freeze["post106_selected"]["q4_decode"]
        == freeze["post098_control"]["q4_decode"]
    )
    plan = describe_plan("OPT-106", "feedback", iteration, "preflight")
    assert "phase=preflight" in plan
    assert loop_product(iteration["workloads"]["preflight"]) == 4


def test_opt098_artifacts_remain_unmodified() -> None:
    hashes = opt098_artifacts_unmodified()
    contract = load_contract()
    assert hashes["report"] == contract["opt098_historical_hashes"]["report"]
    assert hashlib.sha256(OPT098_REPORT.read_bytes()).hexdigest() == hashes["report"]
    assert hashlib.sha256(OPT098_FIXTURE.read_bytes()).hexdigest() == hashes["fixture"]
    assert (
        hashlib.sha256(OPT098_CONTRACT.read_bytes()).hexdigest() == hashes["contract"]
    )
    opt098 = _json(OPT098_FIXTURE)
    assert opt098["task"] == "OPT-098"


def test_dependency_audit_and_family_parity() -> None:
    loaded = audit_dependencies()
    assert set(loaded) == {
        "OPT-100",
        "OPT-101",
        "OPT-102",
        "OPT-103",
        "OPT-104",
        "OPT-105",
    }
    parity = family_kernel_parity(loaded)
    for name in FAMILIES:
        assert parity[name] is True
    assert parity["all"] is True


def test_concession_inactive_uses_strict_ppl() -> None:
    assert concession_active() is False
    contract = load_contract()
    assert contract["concession_active"] is False


def test_rejects_quality_reduced_to_one_boolean() -> None:
    fake = _synthetic_measured()
    fake["quality"] = True
    with pytest.raises(AssertionError, match="quality reduced to one boolean"):
        validate_batch_result(fake)


def test_rejects_opt074_unadmitted_as_blocker() -> None:
    iteration = load_iteration("OPT-106")
    validate_future_keep_policy("OPT-106", iteration)
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-106",
            {
                "task": "OPT-106",
                "opt074_family_admission_required": True,
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_absolute_task_fail_does_not_block_without_regression() -> None:
    inputs = _json(V2_INPUTS)
    functional = {
        "quartz": [
            {
                "case": name,
                "text": "A"
                if name == "task_arithmetic"
                else inputs["cases"][name]["expected"],
                "actual": "A"
                if name == "task_arithmetic"
                else inputs["cases"][name]["expected"],
                "expected": inputs["cases"][name]["expected"],
                "pass": name != "task_arithmetic",
            }
            for name in inputs["cases"]
            if name.startswith("task_")
        ]
    }
    held = {
        "cases": [
            {
                "name": "held_out_wikitext_1024",
                "scored": 32,
                "mean_nll": 3.325543138013308,
            }
        ]
    }
    evaluated = evaluate_combination_quality(
        held=held, functional=functional, inputs=inputs
    )
    assert evaluated["model_quality_pass"] is True
    assert evaluated["absolute_quality_status"] == "fail"
    assert evaluated["status"] == "pass"
    assert evaluated["quartz_vs_baseline_quality_delta"]["ppl_ratio_max"] == 1.01


def test_new_regression_blocks_release() -> None:
    inputs = _json(V2_INPUTS)
    functional = {
        "quartz": [
            {
                "case": name,
                "text": "D",
                "actual": "D",
                "expected": inputs["cases"][name]["expected"],
                "pass": False,
            }
            for name in inputs["cases"]
            if name.startswith("task_")
        ]
    }
    held = {
        "cases": [
            {
                "name": "held_out_wikitext_1024",
                "scored": 32,
                "mean_nll": 3.325543138013308,
            }
        ]
    }
    try:
        evaluated = evaluate_combination_quality(
            held=held, functional=functional, inputs=inputs
        )
    except (BatchGateError, QualityPolicyError):
        return
    assert evaluated["model_quality_pass"] is False
    assert evaluated["status"] == "quality_blocked"


def test_parity_and_plus5_millisecond_arithmetic() -> None:
    quartz = 3046.23218
    llama = 3170.927662
    parity = parity_gap_ms(quartz, llama, 4096)
    plus5 = plus5_gap_ms(quartz, llama, 4096)
    tq = 4096 * 1000.0 / quartz
    tl = 4096 * 1000.0 / llama
    assert parity == pytest.approx(tq - tl)
    assert plus5 == pytest.approx(tq - tl / MARGIN)
    assert parity != pytest.approx(plus5)


def test_does_not_relabel_opt056_or_opt016() -> None:
    fake = _synthetic_measured()
    assert fake["gate"]["passed"] is False
    assert fake["outcomes"]["opt056_plus5"]["pass"] is False
    fake["opt056_gate_passed"] = True
    with pytest.raises(AssertionError, match="OPT-056|historical|gate.passed"):
        validate_batch_result(fake)


def test_rejects_selector_leakage() -> None:
    leaked = _synthetic_measured()
    leaked["combined_production_paths"]["q8_device_layout"] = "aligned_soa"
    with pytest.raises(AssertionError, match="selector|freeze|combined|aligned|raw"):
        validate_batch_result(leaked)
    mmq = _synthetic_measured()
    mmq["combined_production_paths"]["mmq_double_x"] = True
    with pytest.raises(AssertionError, match="selector|freeze|combined|double"):
        validate_batch_result(mmq)


def test_three_outcomes_are_independent() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    outcomes = honest["outcomes"]
    assert "internal_improvement_with_quality" in outcomes
    assert "llama_parity" in outcomes
    assert "opt056_plus5" in outcomes
    assert outcomes["internal_improvement_with_quality"][
        "control_p_tok_s"
    ] == pytest.approx(3046.23218)
    assert outcomes["internal_improvement_with_quality"]["pass"] is False
    assert outcomes["llama_parity"]["pass"] is False
    assert outcomes["opt056_plus5"]["pass"] is False
    assert honest["model_quality_pass"] is True
    assert honest["performance_pass"] is False
    assert honest["release_eligible"] is True
    assert honest["selected_equals_control"] is True


def test_memory_fit_failure_is_honest_not_lost_state() -> None:
    honest = _synthetic_measured()
    honest["state_isolation"]["memory_fit"]["ok"] = False
    honest["outcomes"] = three_outcomes(honest)
    assert honest["outcomes"]["internal_improvement_with_quality"]["pass"] is False
    validate_batch_result(honest)


def test_synthetic_measured_validates() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    validate_report_agrees(
        honest,
        "OPT-098 remains historical\n`gate.passed` is False\n",
    )
