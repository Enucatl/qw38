"""Host tests for OPT-083 ds4-style Qwen quality framework. GPU-free."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tools.opt058_quality_baseline import NO_THINKING_SUFFIX
from tools.opt083_quality_framework import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    PROOF,
    REPORT,
    load_json,
    run_phase,
)
from tools.quality.compare import compare_engine_records, refuse_single_boolean
from tools.quality.errors import QualityFrameworkError
from tools.quality.identity import (
    GGUF_SHA,
    LLAMA_ADAPTER,
    QUARTZ_NATIVE,
    VOCAB_SIZE,
    case_plan,
    scoring_identity,
)
from tools.quality.qwen_fixtures import DS4_DATASETS_NOT_APPLICABLE, QWEN_CONTINUATIONS
from tools.quality.quality_mode import (
    QUALITY_DELTA,
    QUALITY_FLAG,
    QUALITY_SHORTCUTS,
    apply_quality_mode,
    quality_argv,
)
from tools.quality.remote import API_KEY_ENV, OPENROUTER_MODEL, openrouter_status
from tools.quality.scoring import greedy_token, teacher_forced_nll
from tools.quality.suite import HELD_OUT_TARGETS, SUITE_CLASSES
from tools.run_llama_quality_reference import parse_oracle_stdout
from tools.run_optimization_task import describe_plan, load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
LEDGER = ROOT / "implementation_ledger.md"


def test_contracts_iteration_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-083")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert contract["task"] == "OPT-083"
    assert contract["claims_throughput"] is False
    assert contract["quality_mode"]["flag"] == QUALITY_FLAG
    assert contract["quality_mode"]["disables"] == list(QUALITY_SHORTCUTS)
    assert contract["quality_mode"]["unknown_shortcut"] == "fail_closed"
    assert contract["quality_mode"]["precision"]["nvccflags"] == "-O2 --fmad=false"
    assert contract["suite_classes"] == list(SUITE_CLASSES)
    assert contract["openrouter"]["default"] == "off"
    assert contract["opt056_remains_blocked"] is True
    assert contract["opt016_remains_blocked"] is True
    assert contract["same_model_comparison_with_ds4"] is False
    assert contract["native_reuse"] == [LLAMA_ADAPTER, QUARTZ_NATIVE]
    assert iteration["host_only"] is True
    assert iteration["skip_compile"] is True
    assert iteration["case_ids"] == ["identity", "quality-flag", "framework"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["claims_throughput"] is False
    assert "cuda-opt083-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert CONTRACT.is_file()
    assert ITERATION.is_file()


def test_scoring_identity_and_llama_control_parser() -> None:
    logits = [[0.0, 1.0, 3.0, -1.0], [2.0, 0.5, -2.0, 0.0]]
    targets = [2, 0]
    quartz = teacher_forced_nll(logits, targets, case_id="id0", engine="quartz")
    llama = teacher_forced_nll(logits, targets, case_id="id0", engine="llama")
    assert quartz["nll"] == pytest.approx(llama["nll"])
    assert quartz["avg_nll"] == pytest.approx(llama["avg_nll"])
    assert quartz["first_match"] == 1
    assert quartz["greedy_lcp"] == 2
    assert quartz["target_tokens"] == 2
    assert math.isfinite(quartz["perplexity"])
    stdout = (
        "step\tid0\t0\t2\t-0.1\t2\t3.0\t1\t1.0\t2.0\n"
        "step\tid0\t1\t0\t-0.2\t0\t2.0\t1\t0.5\t1.5\n"
    )
    parsed = parse_oracle_stdout(stdout, ("id0",), 0)
    assert parsed["id0"]["steps"][0]["target_token"] == 2
    assert parsed["id0"]["mean_nll"] == pytest.approx(0.15)
    case = QWEN_CONTINUATIONS[0]
    quartz_plan = case_plan(
        case_id=str(case["id"]),
        user=str(case["user"]),
        context=case["context"],
        targets=case["targets"],
        engine="quartz",
    )
    llama_plan = case_plan(
        case_id=str(case["id"]),
        user=str(case["user"]),
        context=case["context"],
        targets=case["targets"],
        engine="llama",
    )
    identity = scoring_identity(quartz_plan, llama_plan)
    assert identity["identical_tokenization"] is True
    assert identity["identical_scoring_definition"] is True
    assert quartz_plan["rendered"].endswith(NO_THINKING_SUFFIX)
    assert quartz_plan["vocab_size"] == VOCAB_SIZE
    assert quartz_plan["gguf_sha256"] == GGUF_SHA


def test_quality_flag_delta_and_unknown_shortcut_fails_closed() -> None:
    contract = load_json(CONTRACT)
    applied = apply_quality_mode(enabled=True)
    assert applied["argv"] == [QUALITY_FLAG]
    assert quality_argv(enabled=True) == ["--quality"]
    assert applied["enabled_shortcuts"] == []
    assert set(applied["disabled_shortcuts"]) == set(QUALITY_SHORTCUTS)
    assert applied["selectors"]["nvccflags"] == "-O2 --fmad=false"
    assert applied["same_math_equivalence_not_different_kernel"] is True
    assert contract["quality_mode"]["flag"] == QUALITY_DELTA["flag"]
    assert contract["quality_mode"]["selector"] == QUALITY_DELTA["selector"]
    assert contract["quality_mode"]["disables"] == QUALITY_DELTA["disables"]
    off = apply_quality_mode(enabled=False)
    assert "diagnostic_fallback" in off["enabled_shortcuts"]
    with pytest.raises(QualityFrameworkError, match="unknown quality shortcut"):
        apply_quality_mode(enabled=True, requested_shortcuts=("made_up_bypass",))


def test_remote_scorer_default_off_and_missing_key_is_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    disabled = openrouter_status(enabled=False)
    assert disabled["status"] == "disabled"
    assert disabled["success"] is True
    assert disabled["authoritative"] is False
    assert disabled["model"] == OPENROUTER_MODEL
    assert disabled["network"] is False
    missing = openrouter_status(enabled=True)
    assert missing["status"] == "skipped_missing_credentials"
    assert missing["success"] is True
    assert missing["network"] is False
    with pytest.raises(QualityFrameworkError, match="cannot run in pytest"):
        openrouter_status(enabled=True, allow_network=True)


def test_refuse_single_boolean_and_inspectable_deltas() -> None:
    left = {
        "a": {
            "target_tokens": 2,
            "nll": 1.0,
            "avg_nll": 0.5,
            "first_match": 1,
            "greedy_lcp": 2,
        }
    }
    right = {
        "a": {
            "target_tokens": 2,
            "nll": 1.2,
            "avg_nll": 0.6,
            "first_match": 1,
            "greedy_lcp": 1,
        }
    }
    compared = compare_engine_records(left, right)
    assert compared["cases"] == 1
    assert compared["examples"][0]["delta_right_minus_left"] == pytest.approx(0.2)
    assert compared["single_boolean"] is None
    with pytest.raises(QualityFrameworkError, match="one boolean"):
        refuse_single_boolean(compared)


def test_incomplete_vocab_and_greedy_tie_break() -> None:
    assert greedy_token([1.0, 3.0, 3.0]) == 1
    with pytest.raises(QualityFrameworkError, match="incomplete vocabulary"):
        teacher_forced_nll(
            [[0.0, 1.0]],
            [0],
            case_id="short",
            engine="quartz",
            vocab_size=8,
        )
    with pytest.raises(QualityFrameworkError, match="outside vocab"):
        teacher_forced_nll([[0.0, 1.0]], [9], case_id="oob", engine="quartz")


def test_iteration_loop_products_and_dry_run_phases() -> None:
    iteration = load_contract("OPT-083")
    assert loop_product(iteration["workloads"]["identity"]) == 8
    assert loop_product(iteration["workloads"]["quality-flag"]) == 12
    assert loop_product(iteration["workloads"]["framework"]) == 28
    plan = describe_plan("OPT-083", "feedback", iteration, None)
    assert "case_ids=identity,quality-flag,framework" in plan
    assert "host_only=true" in plan
    assert "historical_oracles=none" in plan
    identity = describe_plan("OPT-083", "feedback", iteration, "identity")
    assert "phase=identity" in identity
    assert "engines=" in identity
    assert "product=8" in identity
    flag = describe_plan("OPT-083", "feedback", iteration, "quality-flag")
    assert "phase=quality-flag" in flag
    assert "product=12" in flag
    framework = describe_plan("OPT-083", "acceptance", iteration, "framework")
    assert "phase=framework" in framework
    assert "held_out_targets=32" in framework
    assert "product=28" in framework


def test_host_phases_write_fixture_and_preserve_legacy_gates() -> None:
    identity = run_phase("identity")
    assert identity["success"] is True
    assert identity["llama_control_reused"] is True
    assert identity["claims_throughput"] is False
    flag = run_phase("quality-flag")
    assert flag["success"] is True
    assert flag["unknown_shortcut_fail_closed"] is True
    assert flag["argv"] == ["--quality"]
    framework = run_phase("framework")
    assert framework["success"] is True
    assert framework["opt084_baseline_not_run"] is True
    assert framework["openrouter"] == "disabled"
    assert framework["suite_classes"] == list(SUITE_CLASSES)
    assert framework["held_out_targets"] == HELD_OUT_TARGETS
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    fixture = load_json(FIXTURE)
    opt056 = json.loads(OPT056.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")
    assert fixture["claims_throughput"] is False
    assert fixture["single_boolean_refused"] is True
    assert fixture["opt056_remains_blocked"] is True
    assert fixture["opt016_remains_blocked"] is True
    assert fixture["opt056_tasks_pass"] is False
    assert opt056["quality"]["production_optimization"]["tasks"]["pass"] is False
    assert "| OPT-056 |" in ledger and "blocked" in ledger
    assert "| OPT-016 |" in ledger
    assert fixture["ds4_datasets_not_applicable"] == list(DS4_DATASETS_NOT_APPLICABLE)
    dual = fixture["suite"]["classes"]["opt073_dual_verdict"]["quartz"]
    assert dual["quality_v2_all"] is False
    assert dual["absolute_task_accuracy"]["status"] == "fail"
    held = fixture["suite"]["classes"]["held_out_32_alarm"]
    assert held["kernel_admission"] is False
    assert held["held_out_targets"] == 32
    text = REPORT.read_text(encoding="utf-8")
    for item in PROOF:
        assert item in text
    assert "--quality" in text
    assert "not applicable" in text.lower() or "not applicable" in text
    assert os.environ.get("PYTEST_CURRENT_TEST")
    with pytest.raises(QualityFrameworkError, match="cannot run in pytest"):
        run_phase("framework", openrouter=True)
