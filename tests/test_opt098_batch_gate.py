"""Host tests for the OPT-098 combined post-088 outcome gate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools.opt073_quality_policy import QualityPolicyError
from tools.opt080_batch_gate import MARGIN, plus5_gap_ms, parity_gap_ms
from tools.opt098_batch_gate import (
    CONTRACT,
    EXPECTED_PATHS,
    FAMILIES,
    ITERATION,
    OPT088_CONTRACT,
    OPT088_FIXTURE,
    OPT088_REPORT,
    STRICT_PPL_RATIO_MAX,
    BatchGateError,
    audit_dependencies,
    concession_active,
    evaluate_combination_quality,
    family_kernel_parity,
    frozen_combined_config,
    load_contract,
    opt088_artifacts_unmodified,
    post088_control_paths,
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
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
MAKEFILE = ROOT / "Makefile"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_fields(held_nll: float = 3.325543138013308) -> dict[str, Any]:
    baseline = 3.325543138013308
    return {
        "model_quality_pass": True,
        "absolute_quality_status": "fail",
        "quartz_baseline_regression_status": "pass",
        "quartz_vs_baseline_quality_delta": {
            "held_out_32_mean_nll": held_nll,
            "baseline_held_out_32_mean_nll": baseline,
            "held_out_ppl_ratio": math.exp(held_nll - baseline),
            "delta_nll": held_nll - baseline,
            "pass": True,
            "status": "pass",
            "ppl_ratio_max": STRICT_PPL_RATIO_MAX,
            "single_boolean": None,
            "inspectable": True,
            "known_baseline_defect_does_not_reject": True,
            "inherited_absolute_task_fail": True,
            "new_functional_regression": False,
        },
        "quartz_vs_llama_quality_delta": {
            "quartz_held_out_mean_nll": held_nll,
            "llama_held_out_mean_nll": 1.788091893045086,
            "delta_quartz_minus_llama": held_nll - 1.788091893045086,
            "inspectable": True,
            "authoritative_local_comparison": "quartz+llama",
            "single_boolean": None,
        },
        "recurrence_state_status": {
            "status": "pass",
            "incremental_nll": 0.0,
            "incremental_nll_max": 0.02,
        },
        "single_boolean": None,
    }


def _synthetic_measured() -> dict[str, Any]:
    freeze = frozen_combined_config()
    opt056 = _json(OPT056)
    p = copy.deepcopy(opt056["p"])
    p["quartz"]["attribution"] = None
    p["quartz"]["instrumented"] = False
    d128 = copy.deepcopy(opt056["d128"])
    d2048 = copy.deepcopy(opt056["d2048"])
    d128["quartz"]["attribution"] = None
    d2048["quartz"]["attribution"] = None
    p_q = float(p["quartz"]["mean_tok_s"])
    p_l = float(p["llama_cpp"]["avg_ts"])
    d128_q = float(d128["quartz"]["mean_tok_s"])
    d128_l = float(d128["llama_cpp"]["mean_tok_s"])
    d2048_q = float(d2048["quartz"]["mean_tok_s"])
    d2048_l = float(d2048["llama_cpp"]["mean_tok_s"])
    proof = " ".join(load_contract()["proof_limit"])
    quality = _quality_fields()
    from tools.opt080_batch_gate import compute_gate

    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-098",
        "status": "measured",
        "measurement_utc": "2026-09-12T10:30:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "power_limit_w": 400.0,
        "telemetry": {
            "device": "NVIDIA GeForce RTX 5090",
            "device_substring": "RTX 5090",
            "power_limit_w": 400.0,
        },
        "llama_revision": opt056["llama_revision"],
        "gguf_sha256": opt056["gguf_sha256"],
        "source_revision": "synthetic",
        "source_state": "dirty",
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        "post088_control": freeze["post088_control"],
        "post098_selected": freeze["post098_selected"],
        "combined_production_paths": freeze["combined_production_paths"],
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "kernel_parity_pass": freeze["kernel_parity_pass"],
        **quality,
        "performance_pass": False,
        "production_kept": True,
        "release_eligible": True,
        "p": p,
        "d128": d128,
        "d2048": d2048,
        "opt016": copy.deepcopy(opt056["opt016"]),
        "quality": {
            "suite": "opt083",
            "model_quality_pass": True,
            "absolute_quality_status": "fail",
            "quartz_vs_baseline_quality_delta": quality[
                "quartz_vs_baseline_quality_delta"
            ],
            "quartz_vs_llama_quality_delta": quality["quartz_vs_llama_quality_delta"],
            "quality_v2": {
                "suite": "quality-v2",
                "status": "fail",
                "all": False,
                "wikitext_nll": {"pass": True, "mean_nll": 1.5252005926497396},
                "held_out_wikitext_1024": {
                    "pass": True,
                    "mean_nll": 1.7878782710057632,
                },
                "recurrence": {"pass": True, "incremental_nll": -0.00831433},
                "tasks": {"pass": False, "count": 8},
                "llama_reference": {
                    "wikitext_nll": {"mean_nll": 1.52508},
                    "held_out_wikitext_1024": {"mean_nll": 1.78826},
                },
            },
            "quality_v3_replaces_v2": False,
            "relaxes_opt056": False,
            "single_boolean": None,
        },
        "gaps": {
            "p": {
                "parity_ms": parity_gap_ms(p_q, p_l, 4096),
                "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
            },
            "d128": {
                "parity_ms": parity_gap_ms(d128_q, d128_l, 256),
                "plus5_ms": plus5_gap_ms(d128_q, d128_l, 256),
            },
            "d2048": {
                "parity_ms": parity_gap_ms(d2048_q, d2048_l, 256),
                "plus5_ms": plus5_gap_ms(d2048_q, d2048_l, 256),
            },
        },
        "state_isolation": copy.deepcopy(opt056["state_isolation"]),
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "preflight_is_release_evidence": False,
        "opt074_coverage_unadmitted_blocker": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": proof,
        "report_path": "evidence/optimization/opt098-batch-gate/REPORT.md",
    }
    result["opt016"]["gate_passed"] = False
    result["gate"] = compute_gate(result)
    result["outcomes"] = three_outcomes(result)
    result["performance_pass"] = False
    return result


def test_contracts_iteration_plan_and_freeze() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-098")
    freeze = frozen_combined_config()
    paths = source_paths()
    control = post088_control_paths()
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-098"
    assert contract["strict_ppl_ratio_max"] == 1.01
    assert contract["concession_active"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt098-diagnostics"
    assert "freeze" in iteration["modes"]["feedback"]["tier_sequence"]
    assert "report" in iteration["modes"]["acceptance"]["tier_sequence"]
    assert "quality" in iteration["modes"]["release"]["workloads"]
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt098-diagnostics" in makefile
    assert paths["q4_decode"] == "integer_q8_late"
    assert paths["ffn_decode"] == "paired_integer"
    assert paths["q8_grouping"] == "grouped_r1_w4"
    assert control["q4_decode"] == "packed"
    assert control["ffn_decode"] == "paired_staged"
    for key, value in EXPECTED_PATHS.items():
        assert freeze["combined_production_paths"][key] == value
    plan = describe_plan("OPT-098", "feedback", iteration, "preflight")
    assert "phase=preflight" in plan
    assert loop_product(iteration["workloads"]["preflight"]) == 4


def test_opt088_artifacts_remain_unmodified() -> None:
    hashes = opt088_artifacts_unmodified()
    contract = load_contract()
    assert hashes["report"] == contract["opt088_historical_hashes"]["report"]
    assert hashlib.sha256(OPT088_REPORT.read_bytes()).hexdigest() == hashes["report"]
    assert hashlib.sha256(OPT088_FIXTURE.read_bytes()).hexdigest() == hashes["fixture"]
    assert (
        hashlib.sha256(OPT088_CONTRACT.read_bytes()).hexdigest() == hashes["contract"]
    )
    opt088 = _json(OPT088_FIXTURE)
    assert opt088["task"] == "OPT-088"


def test_dependency_audit_and_family_parity() -> None:
    loaded = audit_dependencies()
    assert set(loaded) == set(
        (
            "OPT-089",
            "OPT-090",
            "OPT-091",
            "OPT-092",
            "OPT-093",
            "OPT-094",
            "OPT-095",
            "OPT-096",
            "OPT-097",
        )
    )
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
    iteration = load_iteration("OPT-098")
    validate_future_keep_policy("OPT-098", iteration)
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-098",
            {
                "task": "OPT-098",
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
    quartz = 2808.49609
    llama = 3263.516321
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
    leaked["combined_production_paths"]["q4_decode"] = "packed"
    with pytest.raises(AssertionError, match="selector|freeze|combined|packed"):
        validate_batch_result(leaked)
    q8 = _synthetic_measured()
    q8["combined_production_paths"]["q8_grouping"] = "separate"
    with pytest.raises(AssertionError, match="selector|freeze|combined|grouped"):
        validate_batch_result(q8)


def test_three_outcomes_are_independent() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    outcomes = honest["outcomes"]
    assert "internal_improvement_with_quality" in outcomes
    assert "llama_parity" in outcomes
    assert "opt056_plus5" in outcomes
    assert outcomes["internal_improvement_with_quality"][
        "control_p_tok_s"
    ] == pytest.approx(2956.4502)
    assert honest["model_quality_pass"] is True
    assert honest["performance_pass"] is False
    assert honest["release_eligible"] is True


def test_synthetic_measured_validates() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    validate_report_agrees(
        honest,
        "OPT-088 remains historical\n`gate.passed` is False\n",
    )
