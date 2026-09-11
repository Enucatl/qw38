"""Host tests for OPT-073 quality-v3 dual-verdict policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.opt058_quality_baseline import (
    NLL_CASES,
    NO_THINKING_SUFFIX,
    TASK_NAMES,
    parse_generated_answer,
    render_no_thinking_user_turn,
)
from tools.opt073_quality_policy import (
    ARITHMETIC_DIAGNOSTICS,
    CONTRACT,
    FIXTURE,
    FROZEN_ARITHMETIC_LOGITS,
    ITERATION,
    QualityPolicyError,
    REPORT,
    audit_v2_functional_cases,
    both_engines_arithmetic,
    evaluate_preflight_quality,
    freeze_arithmetic_diagnostics,
    load_json,
    oracle_policy,
    parse_numeric_answer,
    quality_v3_verdicts,
    require_selected_quality_verdicts,
    retained_generated,
    retained_nll_bundle,
    run_phase,
    validate_parsed_functional_answers,
)
from tools.run_optimization_task import describe_plan, load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
TEMPLATE = ROOT / "src/template.cpp"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"


def _nll_passing() -> tuple[dict, dict, dict]:
    inputs = load_json(V2_INPUTS)
    nll = {
        name: {
            "mean_nll": 1.0,
            "steps": [{"log_probability": -1.0}]
            * (1024 if "wikitext" in name else 128),
        }
        for name in NLL_CASES
    }
    authority = {"cases": {name: {"mean_nll": 1.0} for name in NLL_CASES}}
    return nll, authority, inputs


def _correct_generated() -> dict:
    return {
        name: {"text": expected}
        for name, expected in zip(TASK_NAMES, "BACDBACD", strict=True)
    }


def test_contracts_iteration_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-073")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert contract["task"] == "OPT-073"
    assert contract["claims_throughput"] is False
    assert contract["quality_v3"]["replaces_opt056"] is False
    assert contract["quality_v3"]["replaces_opt016"] is False
    assert contract["arithmetic"]["independently_correct_value"] == 42
    assert contract["arithmetic"]["independently_correct_choice_original_v2"] == "B"
    assert contract["arithmetic"]["wrong_authority_choice"] == "A"
    assert contract["arithmetic"]["wrong_authority_value"] == 41
    assert iteration["host_only"] is True
    assert iteration["skip_compile"] is True
    assert iteration["case_ids"] == ["arithmetic", "preflight", "quality"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt073-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert 'enable_thinking ? "<think>\\n" : "<think>\\n\\n</think>\\n\\n"' in template
    assert CONTRACT.is_file()
    assert ITERATION.is_file()


def test_original_ab_answers_and_whitespace_extra_text() -> None:
    assert parse_generated_answer("B", "B")["pass"] is True
    assert parse_generated_answer("  B\n", "B")["pass"] is True
    wrong = parse_generated_answer("A", "B")
    assert wrong["pass"] is False
    assert wrong["actual"] == "A"
    assert wrong["reason"] == "wrong_choice"
    assert parse_generated_answer("A", "B")["expected"] == "B"
    empty = parse_generated_answer("  \n", "B")
    extra = parse_generated_answer("B please", "B")
    second = parse_generated_answer("B\nC", "B")
    assert empty["reason"] == "empty"
    assert extra["reason"] == "extra_text"
    assert second["reason"] == "second_answer"
    numeric = parse_numeric_answer("42", "42")
    assert numeric["pass"] is True
    assert parse_numeric_answer(" 42\n", "42")["pass"] is True
    assert parse_numeric_answer("41", "42")["reason"] == "wrong_integer"
    assert parse_numeric_answer("42 extra", "42")["reason"] == "extra_text"
    assert parse_numeric_answer("", "42")["reason"] == "empty"


def test_v2_arithmetic_audit_preserves_b42_and_template() -> None:
    inputs = load_json(V2_INPUTS)
    audit = audit_v2_functional_cases(inputs)
    assert audit["rendering_objectively_wrong"] is False
    assert audit["arithmetic"]["independently_correct"] == "B"
    assert audit["arithmetic"]["independently_correct_value"] == 42
    assert audit["arithmetic"]["wrong_authority_choice"] == "A"
    assert audit["arithmetic"]["wrong_authority_value"] == 41
    arithmetic = inputs["cases"]["task_arithmetic"]
    rendered = render_no_thinking_user_turn(str(arithmetic["instruction"]))
    assert arithmetic["rendered"] == rendered
    assert arithmetic["rendered"].endswith(NO_THINKING_SUFFIX)
    for name in TASK_NAMES:
        row = next(item for item in audit["cases"] if item["case"] == name)
        assert row["assistant_answer_boundary"] is True
        assert row["tokenizer_id_count"] > 0
        assert row["objective_defects"] == []


def test_wrong_tokenizer_or_template_is_an_objective_defect() -> None:
    inputs = load_json(V2_INPUTS)
    broken = json.loads(json.dumps(inputs))
    broken["cases"]["task_arithmetic"]["rendered"] = "bare completion without template"
    broken["cases"]["task_arithmetic"]["context"] = []
    audit = audit_v2_functional_cases(broken)
    row = next(item for item in audit["cases"] if item["case"] == "task_arithmetic")
    assert "rendered_mismatch" in row["objective_defects"]
    assert "missing_assistant_boundary" in row["objective_defects"]
    assert "missing_tokenizer_ids" in row["objective_defects"]
    assert audit["rendering_objectively_wrong"] is True


def test_four_arithmetic_diagnostics_frozen_before_measurement() -> None:
    frozen = freeze_arithmetic_diagnostics()
    assert [row["id"] for row in frozen] == [
        "original_v2",
        "permuted_options",
        "direct_numeric",
        "independent_paraphrase",
    ]
    assert len(ARITHMETIC_DIAGNOSTICS) == 4
    original = frozen[0]
    permuted = frozen[1]
    numeric = frozen[2]
    paraphrase = frozen[3]
    assert original["expected"] == "B"
    assert original["independently_correct_value"] == 42
    assert "A. 41" in original["user"]
    assert "B. 42" in original["user"]
    assert permuted["expected"] == "C"
    assert "C. 42" in permuted["user"]
    assert numeric["parser"] == "numeric"
    assert numeric["expected"] == "42"
    assert "Reply with the integer only." in numeric["instruction"]
    assert paraphrase["expected"] == "B"
    for row in frozen:
        assert row["rendered"].endswith(NO_THINKING_SUFFIX)
        assert row["logit_masking"] is False
        assert row["max_new_tokens"] == 16
        assert row["scoring"] == "free_running_text"


def test_both_engines_emit_a_and_that_is_not_kernel_blame() -> None:
    engines = both_engines_arithmetic()
    assert engines["quartz"]["actual"] == "A"
    assert engines["llama"]["actual"] == "A"
    assert engines["both_emit_A"] is True
    assert engines["independently_correct"] == "B"
    assert engines["independently_correct_value"] == 42
    assert engines["kernel_or_scorer_bug"] is False
    assert engines["authority_item_miss"] is True


def test_missing_authority_is_incomplete_not_pass() -> None:
    nll, _authority, inputs = _nll_passing()
    generated = _correct_generated()
    verdict = quality_v3_verdicts(nll, None, inputs, generated=generated)
    assert verdict["status"] == "incomplete"
    assert verdict["all"] is False
    assert verdict["absolute_task_accuracy"]["status"] == "incomplete"
    assert verdict["engine_non_regression"]["status"] == "incomplete"


def test_quality_v3_dual_verdict_and_near_tie_cannot_waive() -> None:
    nll, authority, inputs = _nll_passing()
    generated = _correct_generated()
    passing = quality_v3_verdicts(nll, authority, inputs, generated=generated)
    assert passing["absolute_task_accuracy"]["pass"] is True
    assert passing["engine_non_regression"]["pass"] is True
    generated["task_arithmetic"] = {
        "text": "A",
        "next_token": 32,
        "token": 32,
        "runner_up_token": 33,
        "logit": FROZEN_ARITHMETIC_LOGITS["authority_logit"],
        "runner_up_logit": FROZEN_ARITHMETIC_LOGITS["authority_runner_up_logit"],
    }
    dual = quality_v3_verdicts(
        nll,
        authority,
        inputs,
        generated=generated,
        engine_logits={"task_arithmetic": generated["task_arithmetic"]},
    )
    assert dual["quality_v2"]["all"] is False
    assert dual["absolute_task_accuracy"]["pass"] is False
    assert dual["engine_non_regression"]["pass"] is True
    assert dual["engine_non_regression"]["known_miss"]["status"] == "authority_behavior"
    assert dual["all"] is False
    near = dict(generated)
    near["task_arithmetic"] = {
        "text": "A",
        "next_token": 32,
        "token": 32,
        "runner_up_token": 33,
        "logit": 1.000001,
        "runner_up_logit": 1.0,
    }
    near_tie = quality_v3_verdicts(
        nll, authority, inputs, generated=near, engine_logits=near
    )
    assert near_tie["absolute_task_accuracy"]["pass"] is False
    assert near_tie["absolute_task_accuracy"][
        "near_tie_cannot_waive_wrong_semantic_answer"
    ]
    other = dict(generated)
    other["task_arithmetic"] = {"text": "C", "next_token": 34}
    waived = quality_v3_verdicts(nll, authority, inputs, generated=other)
    assert waived["engine_non_regression"]["pass"] is False
    assert waived["engine_non_regression"]["known_miss"]["status"] == (
        "other_wrong_choice_not_waived"
    )
    wrong_ref = dict(generated)
    wrong_ref["task_sort"] = {"text": "A"}
    regress = quality_v3_verdicts(nll, authority, inputs, generated=wrong_ref)
    assert regress["engine_non_regression"]["pass"] is False
    assert regress["engine_non_regression"]["other_wrong_choices_waived"] is False


def test_skipped_quality_phases_and_missing_answers_fail_closed() -> None:
    inputs = load_json(V2_INPUTS)
    with pytest.raises(QualityPolicyError, match="skipped quality"):
        require_selected_quality_verdicts({"selected_quality_verdicts": {}})
    with pytest.raises(QualityPolicyError, match="missing"):
        validate_parsed_functional_answers([], inputs)
    with pytest.raises(QualityPolicyError, match="missing"):
        evaluate_preflight_quality(None, {"cases": []}, inputs)
    held_ok = {"cases": [{"scored": 32, "mean_nll": 1.5}]}
    functional = {
        "quartz": [{"case": "task_arithmetic", "text": "A"}],
        "quartz_output_tokens": 16,
    }
    with pytest.raises(QualityPolicyError, match="missing"):
        evaluate_preflight_quality(functional, held_ok, inputs)
    stop = oracle_policy(False, diagnostic_performance=False)
    assert stop["run_oracles"] is False
    assert stop["release_eligible"] is False
    assert "failed required quality" in str(stop["stop_reason"])
    diagnostic = oracle_policy(False, diagnostic_performance=True)
    assert diagnostic["run_oracles"] is True
    assert diagnostic["release_eligible"] is False
    assert diagnostic["keep_claims_allowed"] is False
    assert diagnostic["retain_quality_failure"] is True


def test_iteration_loop_products_and_dry_run_phases() -> None:
    iteration = load_contract("OPT-073")
    assert loop_product(iteration["workloads"]["arithmetic"]) == 128
    assert loop_product(iteration["workloads"]["preflight"]) == 40
    assert loop_product(iteration["workloads"]["quality"]) == 12288
    plan = describe_plan("OPT-073", "feedback", iteration, None)
    assert "case_ids=arithmetic,preflight,quality" in plan
    assert "host_only=true" in plan
    assert "historical_oracles=none" in plan
    arithmetic = describe_plan("OPT-073", "feedback", iteration, "arithmetic")
    assert "phase=arithmetic" in arithmetic
    assert "engines=" in arithmetic
    assert "product=128" in arithmetic
    preflight = describe_plan("OPT-073", "feedback", iteration, "preflight")
    assert "held_out_targets=32" in preflight
    assert "functional_cases=8" in preflight
    quality = describe_plan("OPT-073", "acceptance", iteration, "quality")
    assert "phase=quality" in quality
    assert "product=12288" in quality


def test_host_phases_write_fixture_and_preserve_legacy_gates() -> None:
    arithmetic = run_phase("arithmetic")
    assert arithmetic["success"] is True
    assert arithmetic["kernel_blame"] is False
    assert arithmetic["prompt_rewrite"] is False
    preflight = run_phase("preflight")
    assert preflight["success"] is True
    assert preflight["skipped_quality_phases_fail_closed"] is True
    assert preflight["release_gate"]["run_oracles"] is False
    assert preflight["opt056_quality_requirement_met"] is False
    quality = run_phase("quality")
    assert quality["success"] is True
    assert quality["quality_v2"]["all"] is False
    assert quality["quality_v3"]["absolute_task_accuracy"] == "fail"
    assert quality["quality_v3"]["engine_non_regression"] == "pass"
    assert quality["does_not_replace_opt056"] is True
    assert quality["universally_passing_absolute_baseline"] is False
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    fixture = load_json(FIXTURE)
    opt056 = json.loads(OPT056.read_text(encoding="utf-8"))
    assert fixture["does_not_replace_opt056"] is True
    assert fixture["does_not_replace_opt016"] is True
    assert fixture["legacy"]["opt056_tasks_pass"] is False
    assert opt056["quality"]["production_optimization"]["tasks"]["pass"] is False
    text = REPORT.read_text(encoding="utf-8")
    for item in load_json(CONTRACT)["proof_limit"]:
        assert item in text
    assert "B=42" in text
    assert "A=41" in text
    nll, authority = retained_nll_bundle()
    quartz, llama = retained_generated()
    assert "wikitext_nll" in nll
    assert "task_arithmetic" in quartz
    assert llama["task_arithmetic"]["actual"] == "A"
    assert authority["cases"]["wikitext_nll"]["mean_nll"]
