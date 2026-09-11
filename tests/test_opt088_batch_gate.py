"""Host tests for the OPT-088 combined post-reset production gate."""

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
from tools.opt088_batch_gate import (
    CONTRACT,
    EXPECTED_PATHS,
    FAMILIES,
    FIXTURE,
    INDEPENDENT_FIELDS,
    ITERATION,
    OPT080_CONTRACT,
    OPT080_FIXTURE,
    OPT080_REPORT,
    REPORT,
    BatchGateError,
    audit_dependencies,
    combination_oracle_policy,
    diagnostic_performance_plan,
    evaluate_combination_quality,
    family_kernel_parity,
    frozen_combined_config,
    load_contract,
    opt080_artifacts_unmodified,
    remaining_latency_budget,
    source_paths,
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
    from tools.opt080_batch_gate import compute_gate, three_outcomes

    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-088",
        "status": "measured",
        "measurement_utc": "2026-09-11T22:40:00Z",
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
        "combined_production_paths": freeze["combined_production_paths"],
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "batch_size": freeze["batch_size"],
        "workspace_bytes": freeze["workspace_bytes"],
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
        "report_path": "evidence/optimization/opt088-batch-gate/REPORT.md",
    }
    result["opt016"]["gate_passed"] = False
    result["gate"] = compute_gate(result)
    result["outcomes"] = three_outcomes(result)
    result["performance_pass"] = False
    return result


def test_contracts_iteration_plan_and_freeze() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-088")
    freeze = frozen_combined_config()
    paths = source_paths()
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert REPORT.is_file()
    assert FIXTURE.is_file()
    assert contract["task"] == "OPT-088"
    assert contract["relaxes_opt056"] is False
    assert contract["relaxes_opt016"] is False
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["opt074_family_admission_required"] is False
    assert contract["preflight_is_release_evidence"] is False
    assert contract["independent_fields"] == list(INDEPENDENT_FIELDS)
    assert contract["families"] == list(FAMILIES)
    assert "OPT-086 r1_w4 Q8 revert" in contract["keeps"]
    assert iteration["target"] == "build/qw38-cuda-optimization-engine-probe"
    assert iteration["diagnostics_make_target"] == "cuda-opt088-diagnostics"
    assert iteration["opt074_family_admission_required"] is False
    assert iteration["modes"]["feedback"]["tier_sequence"] == ["preflight"]
    assert iteration["modes"]["release"]["historical_oracles"] is True
    assert "smoke" not in iteration["modes"]["release"]
    assert loop_product(iteration["workloads"]["preflight"]) == 4
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt088-diagnostics" in makefile
    assert "qw38-cuda-prefill-2k-parity-test" in makefile
    assert paths["q8_layout"] == "r1_w4"
    assert paths["mmq_async_x"] is True
    assert paths["q4_decode"] == "packed"
    assert paths["attention_pipeline"] == "kv_once"
    for key, value in EXPECTED_PATHS.items():
        assert freeze["combined_production_paths"][key] == value
    assert freeze["candidate_decisions"]["OPT-086"]["q8_verdict"] == "revert"
    assert freeze["candidate_decisions"]["OPT-087"]["no_additional_reopen"] is True
    plan = describe_plan("OPT-088", "release", iteration, None)
    assert "engines=" in plan
    assert "d128:" in plan
    assert "warmups=3" in plan
    assert "replicates=30" in plan
    assert "historical_oracles=" in plan
    preflight = describe_plan("OPT-088", "feedback", iteration, "preflight")
    assert "phase=preflight" in preflight
    assert "historical_oracles=none" in preflight


def test_opt080_artifacts_remain_unmodified() -> None:
    hashes = opt080_artifacts_unmodified()
    contract = load_contract()
    assert hashes["report"] == contract["opt080_historical_hashes"]["report"]
    assert hashlib.sha256(OPT080_REPORT.read_bytes()).hexdigest() == hashes["report"]
    assert hashlib.sha256(OPT080_FIXTURE.read_bytes()).hexdigest() == hashes["fixture"]
    assert (
        hashlib.sha256(OPT080_CONTRACT.read_bytes()).hexdigest() == hashes["contract"]
    )
    opt080 = _json(OPT080_FIXTURE)
    assert opt080["task"] == "OPT-080"
    assert opt080["opt056_gate_passed"] is False


def test_dependency_audit_and_family_parity() -> None:
    loaded = audit_dependencies()
    assert set(loaded) == {"OPT-079", "OPT-085", "OPT-086", "OPT-087"}
    parity = family_kernel_parity(loaded)
    for name in FAMILIES:
        assert parity[name] is True
    assert parity["all"] is True
    assert parity["opt074_coverage_unadmitted_blocker"] is False


def test_rejects_incomplete_family_parity() -> None:
    fake = _synthetic_measured()
    del fake["kernel_parity_pass"]["q8"]
    with pytest.raises(AssertionError, match="incomplete family parity"):
        validate_batch_result(fake)
    missing = _synthetic_measured()
    missing["kernel_parity_pass"] = True
    with pytest.raises(AssertionError, match="incomplete family parity"):
        validate_batch_result(missing)


def test_rejects_quality_reduced_to_one_boolean() -> None:
    fake = _synthetic_measured()
    fake["quality"] = True
    with pytest.raises(AssertionError, match="quality reduced to one boolean"):
        validate_batch_result(fake)
    collapsed = _synthetic_measured()
    collapsed["quartz_vs_baseline_quality_delta"] = {"pass": True}
    collapsed["quality"] = {"pass": True, "model_quality_pass": True}
    with pytest.raises(AssertionError, match="quality reduced to one boolean"):
        validate_batch_result(collapsed)


def test_rejects_opt074_unadmitted_as_blocker() -> None:
    iteration = load_iteration("OPT-088")
    validate_future_keep_policy("OPT-088", iteration)
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-088",
            {
                "task": "OPT-088",
                "opt074_family_admission_required": True,
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )
    fake = _synthetic_measured()
    fake["opt074_coverage_unadmitted_blocker"] = True
    with pytest.raises(AssertionError, match="opt074_coverage_unadmitted"):
        validate_batch_result(fake)


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
    assert evaluated["quartz_vs_baseline_quality_delta"]["delta_nll"] == pytest.approx(
        0.0
    )
    policy = combination_oracle_policy(True)
    assert policy["release_eligible"] is True
    assert policy["run_oracles"] is True


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
        policy = combination_oracle_policy(False)
        assert policy["release_eligible"] is False
        return
    assert evaluated["model_quality_pass"] is False
    assert evaluated["status"] == "quality_blocked"
    policy = combination_oracle_policy(False)
    assert policy["run_oracles"] is False
    assert policy["release_eligible"] is False


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
    fake = _synthetic_measured()
    fake["gaps"]["p"]["parity_ms"] = fake["gaps"]["p"]["plus5_ms"]
    with pytest.raises(AssertionError, match="parity gap labeled|wrong parity"):
        validate_batch_result(fake)


def test_does_not_relabel_opt056_or_opt016() -> None:
    fake = _synthetic_measured()
    assert fake["gate"]["passed"] is False
    assert fake["outcomes"]["opt056_plus5"]["pass"] is False
    fake["opt056_gate_passed"] = True
    with pytest.raises(AssertionError, match="OPT-056|historical"):
        validate_batch_result(fake)
    fake = _synthetic_measured()
    fake["opt016"]["gate_passed"] = True
    fake["opt016_gate_passed"] = True
    with pytest.raises(AssertionError, match="OPT-016"):
        validate_batch_result(fake)
    honest = _synthetic_measured()
    validate_batch_result(honest)
    assert honest["opt056_gate_passed"] is False
    assert honest["opt016"]["gate_passed"] is False


def test_rejects_historical_relabeling_in_report() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    with pytest.raises(AssertionError, match="contradictory report"):
        validate_report_agrees(honest, "`gate.passed` is True")
    relabel = _synthetic_measured()
    relabel["opt056_gate_passed"] = True
    with pytest.raises(AssertionError, match="historical gate relabeling|OPT-056"):
        validate_report_agrees(relabel, "`gate.passed` is False")


def test_rejects_selector_leakage_and_r2_w2() -> None:
    leaked = _synthetic_measured()
    leaked["combined_production_paths"]["q8_layout"] = "r2_w2"
    with pytest.raises(AssertionError, match="selector|freeze|combined|r1_w4"):
        validate_batch_result(leaked)
    q4 = _synthetic_measured()
    q4["combined_production_paths"]["q4_decode"] = "integer_q8_1"
    with pytest.raises(AssertionError, match="leak|freeze|selector|combined"):
        validate_batch_result(q4)


def test_measured_fixture_and_report_agree() -> None:
    fixture = _json(FIXTURE)
    assert fixture["status"] == "measured"
    assert fixture["kernel_parity_pass"]["q4"] is True
    assert fixture["kernel_parity_pass"]["q8"] is True
    assert fixture["kernel_parity_pass"]["mmq"] is True
    assert fixture["kernel_parity_pass"]["kv_once"] is True
    assert fixture["model_quality_pass"] is True
    assert fixture["performance_pass"] is False
    assert fixture["production_kept"] is True
    assert fixture["release_eligible"] is True
    assert fixture["opt056_gate_passed"] is False
    assert fixture["owns_opt016_parity_gate"] is False
    assert fixture["gate"]["passed"] is False
    assert fixture["outcomes"]["opt056_plus5"]["pass"] is False
    assert fixture["combined_production_paths"]["q8_layout"] == "r1_w4"
    assert fixture["combined_production_paths"]["q4_decode"] == "packed"
    assert fixture["combined_production_paths"]["attention_pipeline"] == "kv_once"
    validate_batch_result(fixture)
    validate_report_agrees(fixture)
    report = REPORT.read_text(encoding="utf-8")
    assert "OPT-080 remains historical" in report
    assert "`gate.passed` is False" in report
    assert "OPT-056 remains" in report


def test_independent_fields_are_distinct() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    assert honest["model_quality_pass"] is True
    assert honest["performance_pass"] is False
    assert honest["release_eligible"] is True
    assert honest["kernel_parity_pass"]["q4"] is True
    assert honest["quartz_vs_baseline_quality_delta"]["delta_nll"] == pytest.approx(0.0)
    assert honest["quartz_vs_llama_quality_delta"]["inspectable"] is True
    plan = diagnostic_performance_plan({"status": "quality_blocked"})
    assert plan["release_eligible"] is False
    assert plan["opt056_gate_passed"] is False
    budget = remaining_latency_budget()
    assert budget["tok_s_delta_vs_opt069"] == 0.0
    assert budget["p"]["quartz_tok_s"] == pytest.approx(2914.65698)
