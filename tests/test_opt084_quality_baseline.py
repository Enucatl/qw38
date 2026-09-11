"""Host tests for OPT-084 shipping quality baseline freeze. GPU-free."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.opt084_quality_baseline import (
    CONTRACT,
    FIXTURE,
    FREEZE_SELECTORS,
    FROZEN_ACCEPTANCE,
    ITERATION,
    NLL_CASES,
    PROOF,
    REPORT,
    historical_reclassification,
    load_json,
    quartz_vs_baseline,
    refuse_historical_relabel,
    require_complete_suite,
    require_shipping_selectors,
    run_phase,
)
from tools.quality.errors import QualityFrameworkError
from tools.quality.quality_mode import (
    QUALITY_FLAG,
    QUALITY_SHORTCUTS,
    SHIPPING_SELECTORS,
)
from tools.quality.remote import API_KEY_ENV
from tools.quality.suite import HELD_OUT_TARGETS, SUITE_CLASSES
from tools.run_optimization_task import describe_plan, load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
OPT073 = ROOT / "fixtures/opt073_quality_policy.json"
LEDGER = ROOT / "implementation_ledger.md"


def test_contracts_iteration_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-084")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert contract["task"] == "OPT-084"
    assert contract["claims_throughput"] is False
    assert contract["candidate_installed"] is False
    assert contract["quality_mode"]["flag"] == QUALITY_FLAG
    assert contract["quality_mode"]["disables"] == list(QUALITY_SHORTCUTS)
    assert contract["quality_mode"]["precision"]["nvccflags"] == "-O2 --fmad=false"
    assert contract["shipping_selectors"]["q4_decode"] == "packed"
    assert contract["shipping_selectors"]["q4_staging"] == "paired_staged"
    assert contract["shipping_selectors"]["q8_decode"] == "r2_w2"
    assert contract["shipping_selectors"]["prompt_mmq"] == "fma_async_x"
    assert contract["shipping_selectors"]["prompt_mmq_tile"] == "i128_j128"
    assert contract["shipping_selectors"]["prompt_attention"] == "kv_once"
    assert contract["shipping_selectors"]["decode_gdn"] == "sequential"
    assert contract["shipping_selectors"]["decode_attention"] == "warp_query"
    assert contract["shipping_selectors"]["prompt_pair"] == "off"
    assert contract["suite_classes"] == list(SUITE_CLASSES)
    assert contract["frozen_acceptance"]["ppl_ratio_max"] == 1.01
    assert contract["frozen_acceptance"]["recurrence_incremental_nll_max"] == 0.02
    assert contract["opt056_remains_blocked"] is True
    assert contract["opt016_remains_blocked"] is True
    assert contract["historical_failures_erased"] is False
    assert iteration["claims_throughput"] is False
    assert iteration["candidate_installed"] is False
    assert iteration["skip_compile"] is True
    assert iteration["case_ids"] == ["llama", "quartz", "baseline"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt084-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert require_shipping_selectors() == dict(SHIPPING_SELECTORS)
    assert FREEZE_SELECTORS["nvccflags"] == "-O2 --fmad=false"
    assert FROZEN_ACCEPTANCE["primary_candidate_gate"] == (
        "quartz_vs_shipping_baseline_regression"
    )


def test_selector_mismatch_fails_closed() -> None:
    with pytest.raises(QualityFrameworkError, match="shipping selector mismatch"):
        require_shipping_selectors({"q4_decode": "integer_q8_paired"})
    with pytest.raises(QualityFrameworkError, match="shipping selector mismatch"):
        require_shipping_selectors({"prompt_pair": "on"})
    with pytest.raises(QualityFrameworkError, match="shipping selector mismatch"):
        require_shipping_selectors({"nvccflags": "-O3 --fmad=true"})


def test_remote_scorer_invocation_refused_in_pytest() -> None:
    with pytest.raises(QualityFrameworkError, match="cannot run in pytest"):
        run_phase("baseline", openrouter=True)
    os.environ.pop(API_KEY_ENV, None)
    llama = run_phase("llama", skip_gpu=True)
    assert llama["openrouter"] == "disabled"


def test_incomplete_suite_rejected() -> None:
    with pytest.raises(QualityFrameworkError, match="incomplete suite"):
        require_complete_suite({"classes": {}, "openrouter_not_invoked": True})
    with pytest.raises(QualityFrameworkError, match="incomplete suite"):
        require_complete_suite(
            {
                "classes": {name: {} for name in SUITE_CLASSES},
                "openrouter_not_invoked": True,
            }
        )


def test_historical_opt056_073_failures_cannot_be_relabeled() -> None:
    opt056 = json.loads(OPT056.read_text(encoding="utf-8"))
    opt073 = json.loads(OPT073.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")
    assert opt056["quality"]["production_optimization"]["tasks"]["pass"] is False
    assert opt073["does_not_replace_opt056"] is True
    assert "| OPT-056 |" in ledger and "blocked" in ledger
    assert "| OPT-016 |" in ledger
    dual = {
        "quartz": {
            "quality_v2_all": False,
            "absolute_task_accuracy": {"status": "fail"},
            "engine_non_regression": {"status": "pass"},
        }
    }
    honest = historical_reclassification(
        {
            "quartz": {
                "quality_v2_all": False,
                "absolute_task_accuracy": {"status": "fail", "pass": False},
                "engine_non_regression": {"status": "pass", "pass": True},
            }
        }
    )
    assert honest["opt056_tasks_pass"] is False
    assert honest["historical_failures_erased"] is False
    relabel = dict(honest)
    relabel["opt056_tasks_pass"] = True
    with pytest.raises(QualityFrameworkError, match="OPT-056"):
        refuse_historical_relabel(relabel, dual)
    erased = dict(honest)
    erased["historical_failures_erased"] = True
    with pytest.raises(QualityFrameworkError, match="erase historical"):
        refuse_historical_relabel(erased, dual)
    v2_pass = {
        "quartz": {
            "quality_v2_all": True,
            "absolute_task_accuracy": {"status": "fail"},
            "engine_non_regression": {"status": "pass"},
        }
    }
    with pytest.raises(QualityFrameworkError, match="quality-v2"):
        refuse_historical_relabel(honest, v2_pass)


def test_iteration_loop_products_and_dry_run_phases() -> None:
    iteration = load_contract("OPT-084")
    assert loop_product(iteration["workloads"]["llama"]) == 7168
    assert loop_product(iteration["workloads"]["quartz"]) == 7168
    assert loop_product(iteration["workloads"]["baseline"]) == 7
    plan = describe_plan("OPT-084", "feedback", iteration, None)
    assert "case_ids=llama,quartz,baseline" in plan
    assert "historical_oracles=none" in plan
    llama = describe_plan("OPT-084", "feedback", iteration, "llama")
    assert "phase=llama" in llama
    assert "engines=['llama.cpp']" in llama or "engines=llama.cpp" in llama
    assert "product=7168" in llama
    assert "held_out_targets=32" in llama
    quartz = describe_plan("OPT-084", "feedback", iteration, "quartz")
    assert "phase=quartz" in quartz
    assert "product=7168" in quartz
    baseline = describe_plan("OPT-084", "acceptance", iteration, "baseline")
    assert "phase=baseline" in baseline
    assert "product=7" in baseline
    assert "held_out_targets=32" in baseline


def test_host_phases_write_fixture_and_freeze_zero_delta() -> None:
    llama = run_phase("llama", skip_gpu=True)
    assert llama["success"] is True
    assert llama["claims_throughput"] is False
    assert llama["candidate_installed"] is False
    assert llama["identity_cached"] is True
    quartz = run_phase("quartz", skip_gpu=True)
    assert quartz["success"] is True
    assert quartz["shipping_selectors"]["prompt_attention"] == "kv_once"
    baseline = run_phase("baseline", skip_gpu=True)
    assert baseline["success"] is True
    assert baseline["absolute_quality_status"] == "fail"
    assert baseline["quartz_baseline_regression_status"] == "pass"
    assert baseline["zero_delta"] is True
    assert baseline["historical_failures_erased"] is False
    assert baseline["candidate_installed"] is False
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    fixture = load_json(FIXTURE)
    assert fixture["claims_throughput"] is False
    assert fixture["candidate_installed"] is False
    assert fixture["absolute_quality_status"] == "fail"
    assert fixture["quartz_baseline_regression_status"] == "pass"
    assert fixture["quartz_vs_baseline"]["zero_delta"] is True
    assert fixture["quartz_vs_baseline"]["delta_nll"] == pytest.approx(0.0)
    assert fixture["opt056_tasks_pass"] is False
    assert fixture["opt056_remains_blocked"] is True
    assert fixture["opt016_remains_blocked"] is True
    assert fixture["single_boolean_refused"] is True
    assert fixture["openrouter"]["status"] == "disabled"
    teacher = fixture["suite"]["classes"]["teacher_forced_continuation"]
    assert len(teacher["examples"]) == len(NLL_CASES)
    for example in teacher["examples"]:
        assert example["llama"]["target_tokens"] > 0
        assert example["quartz"]["target_tokens"] > 0
        assert example["llama"]["finite"] is True
        assert example["quartz"]["finite"] is True
        assert example["llama"]["nonfinite_count"] == 0
        assert example["quartz"]["nonfinite_count"] == 0
    held = fixture["suite"]["classes"]["held_out_32_alarm"]
    assert held["kernel_admission"] is False
    assert held["held_out_targets"] == HELD_OUT_TARGETS
    dual = fixture["suite"]["classes"]["opt073_dual_verdict"]["quartz"]
    assert dual["quality_v2_all"] is False
    assert dual["absolute_task_accuracy"]["status"] == "fail"
    assert dual["engine_non_regression"]["status"] == "pass"
    text = REPORT.read_text(encoding="utf-8")
    for item in PROOF:
        assert item in text
    assert "--quality" in text
    assert "zero-delta" in text
    regression = quartz_vs_baseline(fixture["quartz_scores"])
    assert regression["pass"] is True
    assert os.environ.get("PYTEST_CURRENT_TEST")
    with pytest.raises(QualityFrameworkError, match="cannot run in pytest"):
        run_phase("baseline", openrouter=True)
