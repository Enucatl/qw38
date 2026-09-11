"""Host tests for the OPT-058 finite scheduler and v2 quality baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.opt058_quality_baseline import (
    NLL_CASES,
    TASK_NAMES,
    TOKEN_271,
    V2_REQUIRED_CASES,
    inspect_tap,
    parse_generated_answer,
    quality_v2_verdicts,
    render_no_thinking_user_turn,
    require_finite_tap,
    sha256_tokens,
)
from tools.run_llama_quality_reference import CASES as LLAMA_SEVEN
from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt058_quality_baseline_contract.json"
OPT057 = ROOT / "pins/opt057_iteration_contract.json"
QUALITY = ROOT / "pins/quality_contract.json"
OPT044 = ROOT / "pins/production_numerics_contract.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt058_quality_baseline_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
REPORT = ROOT / "evidence/optimization/opt058-quality-baseline/REPORT.md"
FIXTURE = ROOT / "fixtures/opt058_quality_baseline.json"
LEGACY_REF = ROOT / "fixtures/quality_llama_reference.json"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_parser_accepts_whitespace_only_around_a_single_choice() -> None:
    parsed = parse_generated_answer("  B\n", "B")
    assert parsed["pass"] is True
    assert parsed["actual"] == "B"
    assert parsed["reason"] == "exact_choice"
    assert parsed["used_teacher_forcing"] is False


def test_parser_rejects_empty_extra_text_and_second_answer() -> None:
    empty = parse_generated_answer("   \n", "A")
    extra = parse_generated_answer("B please", "B")
    shortcut = parse_generated_answer("The answer is B", "B")
    second = parse_generated_answer("B\nC", "B")
    wrong = parse_generated_answer("A", "B")
    assert empty["pass"] is False and empty["reason"] == "empty"
    assert extra["pass"] is False and extra["reason"] == "extra_text"
    assert shortcut["pass"] is False and shortcut["reason"] == "extra_text"
    assert second["pass"] is False and second["reason"] == "second_answer"
    assert wrong["pass"] is False and wrong["reason"] == "wrong_choice"


def test_finite_versus_nonfinite_injected_tap() -> None:
    finite = inspect_tap([1.0, 2.0, 3.0])
    assert finite["finite"] is True
    assert finite["first_nonfinite"] is None
    require_finite_tap([0.0, -1.5], name="ok")
    nonfinite = inspect_tap([1.0, math.nan, math.inf])
    assert nonfinite["finite"] is False
    assert nonfinite["first_nonfinite"] == 1
    assert nonfinite["finite_count"] == 1
    with pytest.raises(ValueError, match="nonfinite tap"):
        require_finite_tap([1.0, math.nan], name="logits")


def test_missing_authority_or_case_is_incomplete() -> None:
    inputs = {"cases": {name: {"expected": "A"} for name in V2_REQUIRED_CASES}}
    missing_auth = quality_v2_verdicts({"wikitext_nll": {}}, None, inputs)
    assert missing_auth["status"] == "incomplete"
    assert missing_auth["all"] is False
    authority = {
        "cases": {
            name: {"mean_nll": 1.0, "steps": [{"log_probability": -1.0}] * 1024}
            for name in NLL_CASES
        }
    }
    missing_case = quality_v2_verdicts({}, authority, inputs)
    assert missing_case["status"] == "incomplete"
    assert "wikitext_nll" in missing_case["missing_cases"]


def test_teacher_forcing_is_not_a_generated_answer() -> None:
    inputs = {"cases": {name: {"expected": "B"} for name in V2_REQUIRED_CASES}}
    nll = {
        name: {
            "mean_nll": 1.0,
            "steps": [{"log_probability": -1.0}]
            * (1024 if "wikitext" in name else 128),
        }
        for name in NLL_CASES
    }
    for name in ("wikitext_nll", "held_out_wikitext_1024"):
        nll[name]["steps"] = [{"log_probability": -1.0}] * 1024
    authority = {"cases": {name: {"mean_nll": 1.0} for name in NLL_CASES}}
    generated = {name: {"greedy_token": 33} for name in TASK_NAMES}
    verdict = quality_v2_verdicts(nll, authority, inputs, generated=generated)
    assert verdict["used_teacher_forcing_for_answers"] is True
    assert verdict["tasks"]["pass"] is False
    assert verdict["all"] is False
    text_answers = {name: {"text": "B"} for name in TASK_NAMES}
    passing = quality_v2_verdicts(nll, authority, inputs, generated=text_answers)
    assert passing["tasks"]["pass"] is True
    assert passing["used_teacher_forcing_for_answers"] is False


def test_no_thinking_template_has_assistant_answer_boundary() -> None:
    rendered = render_no_thinking_user_turn("Hello")
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert rendered.startswith("<|im_start|>user\nHello<|im_end|>\n")


def test_contracts_makefile_and_native_target_declare_opt058() -> None:
    quality = _contract()
    iteration = load_contract("OPT-058")
    opt057 = json.loads(OPT057.read_text(encoding="utf-8"))
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert quality["task"] == "OPT-058"
    assert quality["claims_throughput"] is False
    assert quality["logit_masking"] is False
    assert quality["max_new_tokens"] == 16
    assert quality["missing_held_out_authority"] == "incomplete"
    assert quality["thresholds"]["nll_ppl_ratio"] == 1.01
    assert quality["thresholds"]["recurrence_incremental_nll"] == 0.02
    assert set(TASK_NAMES).issubset(set(quality["cases"]))
    assert "held_out_wikitext_1024" in quality["cases"]
    assert iteration["modes"]["feedback"]["tier_sequence"] == [
        "smoke",
        "scheduler",
        "functional",
    ]
    assert loop_product(iteration["workloads"]["scheduler"]) == 6
    assert loop_product(iteration["workloads"]["functional"]) == 256
    assert iteration["modes"]["acceptance"]["warm_repetitions"] == 0
    assert opt057["quality_baseline"]["task"] == "OPT-058"
    assert opt057["quality_baseline"]["immutable"] is True
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "cuda-opt058-diagnostics" in makefile
    assert "execute_token_traced_bundle" in native
    assert "execute_token_traced_bundle" in scheduler
    assert "single_traversal_taps" in native


def test_legacy_quality_and_opt056_verdicts_are_preserved() -> None:
    legacy = json.loads(QUALITY.read_text(encoding="utf-8"))
    assert legacy["thresholds"]["nll_ppl_ratio"] == 1.05
    assert list(LLAMA_SEVEN) == [
        "wikitext_nll",
        "continuation_2048",
        "continuation_4096",
        "continuation_6144",
        "continuation_8192",
        "recurrence_short",
        "recurrence_long",
    ]
    assert LEGACY_REF.is_file()
    opt056 = json.loads(OPT056.read_text(encoding="utf-8"))
    tasks = opt056["quality"]["production_optimization"]["tasks"]
    assert tasks["pass"] is False
    assert tasks["count"] == 8


def test_held_out_span_matches_opt044_offsets_and_hash() -> None:
    quality = _contract()
    opt044 = json.loads(OPT044.read_text(encoding="utf-8"))
    held = quality["held_out_wikitext_1024"]
    reference = opt044["quality_suite"]["held_out_wikitext_1024"]
    assert held["stream_start"] == reference["stream_start"] == 16385
    assert held["stream_end"] == reference["stream_end"] == 17409
    assert held["continuation_sha256"] == reference["continuation_sha256"]
    assert held["target_count"] == 1024


def test_v2_inputs_use_chat_template_and_keep_held_out_hash() -> None:
    if not V2_INPUTS.is_file():
        pytest.skip("v2 inputs are frozen by tools/freeze_quality_v2_inputs.py")
    record = json.loads(V2_INPUTS.read_text(encoding="utf-8"))
    quality_inputs = json.loads(
        (ROOT / "fixtures/quality_inputs.json").read_text(encoding="utf-8")
    )
    assert record["schema"] == "qw38.quality-inputs-v2"
    assert record["chat_template"] == "no_thinking"
    assert record["logit_masking"] is False
    assert record["token_271"] == TOKEN_271
    held = record["cases"]["held_out_wikitext_1024"]
    assert held["continuation_sha256"] == sha256_tokens(held["continuation"])
    assert (
        held["continuation_sha256"]
        == _contract()["held_out_wikitext_1024"]["continuation_sha256"]
    )
    original = quality_inputs["cases"]["task_arithmetic"]["context"]
    templated = record["cases"]["task_arithmetic"]["context"]
    assert templated != original
    assert record["cases"]["task_arithmetic"]["rendered"].endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert record["cases"]["wikitext_nll"]["scoring"] == "teacher_forced_nll"
    assert record["cases"]["task_arithmetic"]["scoring"] == "free_running_text"


def test_checked_in_fixture_and_report_when_present() -> None:
    quality = _contract()
    if not FIXTURE.is_file() or not REPORT.is_file():
        pytest.skip("OPT-058 fixture/report are written by the GPU freeze")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    text = REPORT.read_text(encoding="utf-8")
    for key in load_contract("OPT-058")["required_fixture_keys"]:
        assert key in fixture, key
    assert fixture["claims_throughput"] is False
    assert fixture["task"] == "OPT-058"
    for item in quality["proof_limit"]:
        assert item in text
    assert "tok/s" not in text.lower() or "no throughput" in text.lower()
