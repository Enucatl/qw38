"""OPT-073 versioned quality-v3 dual-verdict policy.

Audits v2 functional arithmetic without rewriting 17+25 truth. Quality-v2 and
quality-v3 are reported side by side. OPT-056 and OPT-016 are not relabeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt058_quality_baseline import (  # noqa: E402
    NLL_CASES,
    NO_THINKING_SUFFIX,
    TASK_NAMES,
    original_task_instruction,
    parse_generated_answer,
    quality_v2_verdicts,
    render_no_thinking_user_turn,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
CONTRACT = ROOT / "pins/opt073_quality_policy_contract.json"
ITERATION = ROOT / "pins/opt073_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt073_quality_policy.json"
REPORT = ROOT / "evidence/optimization/opt073-quality-policy/REPORT.md"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
V2_LLAMA = ROOT / "pins/production_quality_v2_llama_reference.json"
OPT058_FIXTURE = ROOT / "fixtures/opt058_quality_baseline.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
QUALITY_CONTRACT = ROOT / "pins/quality_contract.json"
FUNCTIONAL_V2 = ROOT / (
    "evidence/optimization/opt069-batch-gate/quality-v2-functional.json"
)
NLL_V2 = ROOT / "evidence/optimization/opt069-batch-gate/quality-v2-nll.json"
PREFLIGHT_FUNCTIONAL = ROOT / (
    "evidence/optimization/opt069-batch-gate/preflight-functional.json"
)
PREFLIGHT_HELD = ROOT / (
    "evidence/optimization/opt069-batch-gate/preflight-held-out-32.json"
)
NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
LLAMA_ORACLE = ".cache/authorities/llama-build/bin/qw38-llama-quality-oracle"
MAX_NEW_TOKENS = 16
CHOICE_TOKENS = {"A": 32, "B": 33, "C": 34, "D": 35}
TOKEN_CHOICES = {value: key for key, value in CHOICE_TOKENS.items()}

# Frozen from retained pinned-llama first-token logits on original v2
# task_arithmetic (quality-v2-functional.json llama_generate[0]).
FROZEN_ARITHMETIC_LOGITS = {
    "case": "task_arithmetic",
    "prompt": "original_v2",
    "authority_answer": "A",
    "authority_value": 41,
    "independently_correct": "B",
    "independently_correct_value": 42,
    "authority_token": 32,
    "authority_runner_up_token": 33,
    "authority_logit": 26.5466595,
    "authority_runner_up_logit": 24.3310528,
    "authority_margin": 2.2156067,
    "source": (
        "evidence/optimization/opt069-batch-gate/quality-v2-functional.json"
        "#llama_generate[task_arithmetic][0]"
    ),
}

ARITHMETIC_DIAGNOSTICS: tuple[dict[str, Any], ...] = (
    {
        "id": "original_v2",
        "user": "What is 17 + 25?\nA. 41\nB. 42\nC. 43\nD. 44",
        "parser": "choice",
        "expected": "B",
        "independently_correct_value": 42,
        "wrong_authority_choice": "A",
        "wrong_authority_value": 41,
        "instruction_kind": "letter",
    },
    {
        "id": "permuted_options",
        "user": "What is 17 + 25?\nA. 44\nB. 43\nC. 42\nD. 41",
        "parser": "choice",
        "expected": "C",
        "independently_correct_value": 42,
        "instruction_kind": "letter",
        "note": "fixed permutation frozen before measurement",
    },
    {
        "id": "direct_numeric",
        "user": "What is 17 + 25?",
        "parser": "numeric",
        "expected": "42",
        "independently_correct_value": 42,
        "instruction_kind": "integer",
        "instruction": "What is 17 + 25?\nReply with the integer only.",
    },
    {
        "id": "independent_paraphrase",
        "user": "Add seventeen and twenty-five.\nA. 41\nB. 42\nC. 43\nD. 44",
        "parser": "choice",
        "expected": "B",
        "independently_correct_value": 42,
        "instruction_kind": "letter",
        "note": "one independent paraphrase frozen before measurement",
    },
)

RATIONALE = (
    "Pinned llama and Quartz both emit A (41) on the authenticated no-thinking "
    "v2 17+25 item whose independently correct answer is B (42). Rendering, "
    "tokenizer IDs, assistant boundary, termination, and the A/B/C/D parser "
    "match src/template.cpp enable_thinking=false; this is an authority item "
    "miss, not a kernel or scorer defect. Quality-v3 keeps absolute arithmetic "
    "accuracy failed while allowing internal candidate admission when every "
    "reference-correct functional item stays correct and the known miss is "
    "either independently corrected or preserved as authority A within the "
    "frozen first-token identity and logit/margin snapshot. Other wrong "
    "choices are not waived. Near-ties cannot pass absolute accuracy. "
    "Quality-v2, QLT-001, OPT-056, and OPT-016 remain unchanged historical "
    "conditions. Missing evidence is incomplete, never pass."
)

PROOF = (
    "no throughput claim",
    "no arithmetic kernel change",
    "B=42 remains the correct 17+25 answer",
    "A=41 remains wrong",
    "both engines failing is not kernel blame",
    "no prompt rewrite without an objective defect",
    "no token mask",
    "no teacher-forced functional answer",
    "no selecting only paraphrases that pass",
    "quality-v3 does not replace OPT-056 or OPT-016",
    "missing evidence is incomplete",
    "failed required quality stops release before long timing",
    "diagnostic performance is not a release or keep",
)


class QualityPolicyError(AssertionError):
    """Inadmissible quality-policy evidence or a fail-closed gate."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def letter_instruction(user: str) -> str:
    return original_task_instruction(user)


def numeric_instruction(user: str) -> str:
    return user.strip() + "\nReply with the integer only."


def parse_numeric_answer(text: str, expected: str) -> dict[str, Any]:
    """Accept exactly the expected integer after stripping whitespace."""
    stripped = text.strip()
    record: dict[str, Any] = {
        "raw": text,
        "stripped": stripped,
        "expected": expected,
        "pass": False,
        "reason": "",
        "actual": None,
        "used_teacher_forcing": False,
        "parser": "numeric",
    }
    if stripped == "":
        record["reason"] = "empty"
        return record
    if not stripped.lstrip("+-").isdigit() or stripped.count("-") > 1:
        record["reason"] = "extra_text"
        return record
    if stripped[0] in "+-" and (len(stripped) == 1 or not stripped[1:].isdigit()):
        record["reason"] = "extra_text"
        return record
    record["actual"] = str(int(stripped))
    if record["actual"] == str(int(expected)):
        record["pass"] = True
        record["reason"] = "exact_integer"
    else:
        record["reason"] = "wrong_integer"
    return record


def diagnostic_instruction(row: Mapping[str, Any]) -> str:
    if row.get("instruction"):
        return str(row["instruction"])
    if row["instruction_kind"] == "integer":
        return numeric_instruction(str(row["user"]))
    return letter_instruction(str(row["user"]))


def freeze_arithmetic_diagnostics() -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for row in ARITHMETIC_DIAGNOSTICS:
        instruction = diagnostic_instruction(row)
        rendered = render_no_thinking_user_turn(instruction)
        if not rendered.endswith(NO_THINKING_SUFFIX):
            raise QualityPolicyError(
                f"{row['id']} missing no-thinking assistant boundary"
            )
        frozen.append(
            {
                **row,
                "instruction": instruction,
                "rendered": rendered,
                "rendered_sha256": sha256_text(rendered),
                "max_new_tokens": MAX_NEW_TOKENS,
                "logit_masking": False,
                "scoring": "free_running_text",
                "chat_template": "no_thinking",
                "assistant_answer_boundary": NO_THINKING_SUFFIX,
            }
        )
    return frozen


def audit_v2_case(name: str, case: Mapping[str, Any]) -> dict[str, Any]:
    instruction = str(case["instruction"])
    rendered = str(case["rendered"])
    expected_render = render_no_thinking_user_turn(instruction)
    defects: list[str] = []
    if rendered != expected_render:
        defects.append("rendered_mismatch")
    if not rendered.endswith(NO_THINKING_SUFFIX):
        defects.append("missing_assistant_boundary")
    if case.get("scoring") != "free_running_text":
        defects.append("scoring_not_free_running")
    if int(case.get("max_new_tokens", 0)) > MAX_NEW_TOKENS:
        defects.append("max_new_tokens")
    context = list(case.get("context") or [])
    if not context:
        defects.append("missing_tokenizer_ids")
    return {
        "case": name,
        "user": case.get("user"),
        "expected": case.get("expected"),
        "instruction": instruction,
        "rendered": rendered,
        "rendered_matches_template": rendered == expected_render,
        "assistant_answer_boundary": rendered.endswith(NO_THINKING_SUFFIX),
        "tokenizer_id_count": len(context),
        "max_new_tokens": case.get("max_new_tokens", MAX_NEW_TOKENS),
        "scoring": case.get("scoring"),
        "objective_defects": defects,
        "prompt_rewrite_allowed": bool(defects),
    }


def audit_v2_functional_cases(inputs: Mapping[str, Any]) -> dict[str, Any]:
    cases = inputs["cases"]
    rows = [audit_v2_case(name, cases[name]) for name in TASK_NAMES]
    arithmetic = cases["task_arithmetic"]
    user = str(arithmetic["user"])
    if "17 + 25" not in user and "17+25" not in user.replace(" ", ""):
        raise QualityPolicyError("task_arithmetic is not 17+25")
    if arithmetic["expected"] != "B":
        raise QualityPolicyError("independently correct 17+25 answer must stay B")
    if "A. 41" not in user or "B. 42" not in user:
        raise QualityPolicyError("original A=41 / B=42 options missing")
    defects = [row["case"] for row in rows if row["objective_defects"]]
    return {
        "cases": rows,
        "objective_defects": defects,
        "rendering_objectively_wrong": bool(defects),
        "arithmetic": {
            "user": user,
            "independently_correct": "B",
            "independently_correct_value": 42,
            "wrong_authority_choice": "A",
            "wrong_authority_value": 41,
            "expected_unchanged": arithmetic["expected"] == "B",
        },
        "chat_template": inputs.get("chat_template"),
        "assistant_answer_boundary": inputs.get("assistant_answer_boundary"),
        "logit_masking": inputs.get("logit_masking", False),
        "max_new_tokens": inputs.get("max_new_tokens", MAX_NEW_TOKENS),
    }


def choice_from_token(token: Any) -> str | None:
    if token is None:
        return None
    return TOKEN_CHOICES.get(int(token))


def parse_functional_row(
    row: Mapping[str, Any], expected: str, *, parser: str = "choice"
) -> dict[str, Any]:
    if "greedy_token" in row and "text" not in row and "stripped" not in row:
        raise QualityPolicyError("teacher-forced functional answer is not admitted")
    text = str(row.get("text", row.get("stripped", row.get("raw", ""))))
    if parser == "numeric":
        parsed = parse_numeric_answer(text, expected)
    else:
        parsed = parse_generated_answer(text, expected)
    parsed["next_token"] = row.get("next_token")
    parsed["top_two"] = row.get("top_two")
    parsed["case"] = row.get("case")
    parsed["engine"] = row.get("engine")
    return parsed


def validate_parsed_functional_answers(
    rows: Sequence[Mapping[str, Any]] | None,
    inputs: Mapping[str, Any],
    *,
    names: Sequence[str] = TASK_NAMES,
) -> list[dict[str, Any]]:
    if not rows:
        raise QualityPolicyError("parsed functional answers missing")
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = str(row.get("case") or "")
        if name:
            by_name[name] = row
    parsed_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in names:
        row = by_name.get(name)
        if row is None:
            missing.append(name)
            continue
        text = row.get("text", row.get("stripped"))
        if text is None:
            missing.append(name)
            continue
        expected = str(inputs["cases"][name]["expected"])
        parsed_rows.append(parse_functional_row(row, expected))
    if missing:
        raise QualityPolicyError(
            "parsed functional answers missing for " + ",".join(missing)
        )
    return parsed_rows


def first_llama_logits(
    llama_generate: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    first: dict[str, dict[str, Any]] = {}
    for row in llama_generate or []:
        name = str(row.get("case") or "")
        if name and name not in first:
            first[name] = dict(row)
    return first


def _logit_snapshot(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    token = row.get("token", row.get("next_token"))
    runner = row.get("runner_up_token")
    top_two = row.get("top_two")
    if runner is None and isinstance(top_two, (list, tuple)) and len(top_two) > 1:
        runner = top_two[1]
        if token is None:
            token = top_two[0]
    logit = row.get("logit")
    runner_logit = row.get("runner_up_logit")
    margin = None
    if _finite(logit) and _finite(runner_logit):
        margin = float(logit) - float(runner_logit)
    return {
        "token": None if token is None else int(token),
        "runner_up_token": None if runner is None else int(runner),
        "logit": None if not _finite(logit) else float(logit),
        "runner_up_logit": None if not _finite(runner_logit) else float(runner_logit),
        "margin": margin,
        "choice": choice_from_token(token),
    }


def _authority_behavior(
    parsed: Mapping[str, Any], snapshot: Mapping[str, Any] | None
) -> dict[str, Any]:
    frozen = FROZEN_ARITHMETIC_LOGITS
    actual = parsed.get("actual")
    next_token = parsed.get("next_token")
    if next_token is None and snapshot is not None:
        next_token = snapshot.get("token")
    independently_correct = actual == frozen["independently_correct"]
    authority_choice = actual == frozen["authority_answer"]
    token_ok = next_token in (None, frozen["authority_token"])
    other_wrong = actual not in {
        frozen["independently_correct"],
        frozen["authority_answer"],
        None,
    }
    logits_ok = True
    logits_incomplete = False
    if snapshot is None or snapshot.get("logit") is None:
        logits_incomplete = True
    else:
        if snapshot.get("token") not in (
            frozen["authority_token"],
            CHOICE_TOKENS[frozen["independently_correct"]],
        ):
            logits_ok = False
        if snapshot.get("token") == frozen["authority_token"]:
            runner = snapshot.get("runner_up_token")
            if runner not in (None, frozen["authority_runner_up_token"]):
                logits_ok = False
            margin = snapshot.get("margin")
            if margin is not None and margin < 0:
                logits_ok = False
            if not _finite(snapshot.get("logit")):
                logits_ok = False
    if independently_correct:
        status = "independently_correct"
        admitted = True
    elif other_wrong or parsed.get("reason") in {
        "empty",
        "extra_text",
        "second_answer",
        "wrong_integer",
    }:
        status = "other_wrong_choice_not_waived"
        admitted = False
    elif authority_choice and token_ok and logits_ok:
        status = "authority_behavior"
        admitted = True
    else:
        status = "authority_behavior_mismatch"
        admitted = False
    return {
        "admitted": admitted,
        "status": status,
        "independently_correct": independently_correct,
        "authority_choice": authority_choice,
        "logits_incomplete": logits_incomplete,
        "near_tie_cannot_pass_absolute": True,
    }


def quality_v3_verdicts(
    results: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    inputs: Mapping[str, Any],
    *,
    generated: Mapping[str, Mapping[str, Any]] | None = None,
    engine_logits: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    v2 = quality_v2_verdicts(results, authority, inputs, generated=generated)
    verdict: dict[str, Any] = {
        "suite": "quality-v3",
        "version": 3,
        "status": "incomplete",
        "all": False,
        "does_not_replace_opt056": True,
        "does_not_replace_opt016": True,
        "internal_candidate_admission": True,
        "rationale": RATIONALE,
        "quality_v2": {
            "status": v2.get("status"),
            "all": v2.get("all"),
            "tasks": v2.get("tasks"),
        },
        "frozen_arithmetic_logits": dict(FROZEN_ARITHMETIC_LOGITS),
        "missing_authority": list(v2.get("missing_authority") or []),
        "missing_cases": list(v2.get("missing_cases") or []),
    }
    if v2.get("status") == "incomplete":
        verdict["absolute_task_accuracy"] = {"pass": False, "status": "incomplete"}
        verdict["engine_non_regression"] = {"pass": False, "status": "incomplete"}
        return verdict
    nll_ok = all(
        bool(v2[name]["pass"])
        for name in ("wikitext_nll", "held_out_wikitext_1024", "recurrence")
    )
    if generated is None:
        verdict["absolute_task_accuracy"] = {
            "pass": False,
            "status": "incomplete",
            "reason": "generated_answers_required",
        }
        verdict["engine_non_regression"] = {
            "pass": False,
            "status": "incomplete",
            "reason": "generated_answers_required",
        }
        verdict["status"] = "incomplete"
        return verdict
    miss = None
    reference_correct_ok = True
    other_wrong = False
    task_rows: dict[str, Any] = {}
    for name in TASK_NAMES:
        expected = str(inputs["cases"][name]["expected"])
        row = generated.get(name) or {}
        parsed = (
            row
            if "reason" in row and "expected" in row
            else parse_functional_row(row, expected)
        )
        snapshot = _logit_snapshot((engine_logits or {}).get(name) or row)
        task_rows[name] = {"parser": parsed, "logits": snapshot}
        if name == "task_arithmetic":
            miss = _authority_behavior(parsed, snapshot)
            task_rows[name]["miss"] = miss
            continue
        if not parsed.get("pass"):
            reference_correct_ok = False
            other_wrong = True
    absolute_ok = bool(v2.get("tasks", {}).get("pass")) and nll_ok
    engine_ok = (
        nll_ok
        and reference_correct_ok
        and bool(miss and miss["admitted"])
        and not other_wrong
    )
    verdict["wikitext_nll"] = v2["wikitext_nll"]
    verdict["held_out_wikitext_1024"] = v2["held_out_wikitext_1024"]
    verdict["recurrence"] = v2["recurrence"]
    verdict["tasks"] = task_rows
    verdict["absolute_task_accuracy"] = {
        "pass": absolute_ok,
        "status": "pass" if absolute_ok else "fail",
        "nll_pass": nll_ok,
        "functional_independently_correct": bool(v2.get("tasks", {}).get("pass")),
        "task_arithmetic_must_be_B_42": True,
        "near_tie_cannot_waive_wrong_semantic_answer": True,
    }
    verdict["engine_non_regression"] = {
        "pass": engine_ok,
        "status": "pass" if engine_ok else "fail",
        "nll_pass": nll_ok,
        "reference_correct_items_remain_correct": reference_correct_ok,
        "known_miss": miss,
        "other_wrong_choices_waived": False,
    }
    verdict["all"] = absolute_ok and engine_ok
    verdict["status"] = "pass" if verdict["all"] else "fail"
    verdict["selected_internal_quality"] = {
        "name": "quality-v3-engine-non-regression-plus-nll",
        "pass": engine_ok,
        "does_not_replace_opt056": True,
    }
    return verdict


def oracle_policy(
    quality_v2_all: bool | None, *, diagnostic_performance: bool = False
) -> dict[str, Any]:
    if quality_v2_all is None:
        return {
            "run_oracles": False,
            "release_eligible": False,
            "keep_claims_allowed": False,
            "stop_reason": "skipped quality phase",
            "diagnostic_performance": diagnostic_performance,
        }
    if quality_v2_all:
        return {
            "run_oracles": True,
            "release_eligible": True,
            "keep_claims_allowed": True,
            "stop_reason": None,
            "diagnostic_performance": False,
        }
    if diagnostic_performance:
        return {
            "run_oracles": True,
            "release_eligible": False,
            "keep_claims_allowed": False,
            "retain_quality_failure": True,
            "stop_reason": None,
            "diagnostic_performance": True,
        }
    return {
        "run_oracles": False,
        "release_eligible": False,
        "keep_claims_allowed": False,
        "stop_reason": "failed required quality stops release before long timing",
        "diagnostic_performance": False,
    }


def require_selected_quality_verdicts(record: Mapping[str, Any]) -> None:
    required = (
        "parsed_functional_answers",
        "held_out_32",
        "quality_v2",
        "quality_v3_absolute",
        "quality_v3_engine_non_regression",
    )
    selected = record.get("selected_quality_verdicts") or {}
    missing = [name for name in required if name not in selected]
    if missing:
        raise QualityPolicyError("skipped quality phases: " + ",".join(missing))
    if selected.get("parsed_functional_answers") in (None, False, "skipped"):
        raise QualityPolicyError("parsed functional answers skipped")
    if selected.get("held_out_32") in (None, False, "skipped"):
        raise QualityPolicyError("held-out quality phase skipped")
    for name in (
        "quality_v2",
        "quality_v3_absolute",
        "quality_v3_engine_non_regression",
    ):
        if selected.get(name) in (None, "skipped"):
            raise QualityPolicyError(f"skipped quality phase {name}")


def evaluate_preflight_quality(
    functional: Mapping[str, Any] | None,
    held: Mapping[str, Any] | None,
    inputs: Mapping[str, Any],
    *,
    nll: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
    generated_override: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if functional is None:
        raise QualityPolicyError("preflight functional sidecar missing")
    if held is None:
        raise QualityPolicyError("preflight held-out sidecar missing")
    parsed = validate_parsed_functional_answers(functional.get("quartz"), inputs)
    cases = held.get("cases") or []
    if not cases or int(cases[0].get("scored", 0)) != 32:
        raise QualityPolicyError(
            "preflight held-out must score 32 teacher-forced targets"
        )
    if not _finite(cases[0].get("mean_nll")):
        raise QualityPolicyError("preflight held-out NLL is nonfinite")
    generated = generated_override or {
        str(item["case"]): item for item in parsed if item.get("case")
    }
    if len(generated) != 8:
        generated = {
            name: parsed[index]
            for index, name in enumerate(TASK_NAMES)
            if index < len(parsed)
        }
        for item in parsed:
            if item.get("case"):
                generated[str(item["case"])] = item
    v2: dict[str, Any]
    v3: dict[str, Any]
    if nll is None or authority is None:
        v2 = {
            "suite": "quality-v2",
            "status": "incomplete",
            "all": False,
            "tasks": {
                "pass": all(item["pass"] for item in parsed),
                "count": len(parsed),
            },
        }
        v3 = {
            "suite": "quality-v3",
            "status": "incomplete",
            "absolute_task_accuracy": {"pass": False, "status": "incomplete"},
            "engine_non_regression": {"pass": False, "status": "incomplete"},
        }
        full = False
    else:
        v2 = quality_v2_verdicts(nll, authority, inputs, generated=generated)
        logits = {
            name: item for name, item in generated.items() if isinstance(item, Mapping)
        }
        v3 = quality_v3_verdicts(
            nll,
            authority,
            inputs,
            generated=generated,
            engine_logits=logits,
        )
        full = True
    selected = {
        "parsed_functional_answers": True,
        "held_out_32": True,
        "quality_v2": v2.get("status"),
        "quality_v3_absolute": (v3.get("absolute_task_accuracy") or {}).get("status"),
        "quality_v3_engine_non_regression": (v3.get("engine_non_regression") or {}).get(
            "status"
        ),
    }
    payload = {
        "parsed": parsed,
        "quality_v2": v2,
        "quality_v3": v3,
        "selected_quality_verdicts": selected,
        "full_nll": full,
        "held_out_targets": 32,
        "functional_output_tokens": functional.get("quartz_output_tokens"),
        "status": (
            "passed"
            if v2.get("all")
            else "quality_blocked"
            if (not all(item["pass"] for item in parsed))
            or v2.get("status") not in {None, "incomplete"}
            else "incomplete"
        ),
        "is_release_evidence": False,
        "opt056_quality_requirement_met": bool(v2.get("all")),
    }
    require_selected_quality_verdicts(payload)
    return payload


def retained_nll_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    nll_live = load_json(NLL_V2)
    authority = load_json(V2_LLAMA)
    fixture = load_json(OPT058_FIXTURE)
    results: dict[str, Any] = {}
    quartz = fixture["quality_v2"]["quartz"]
    for name in NLL_CASES:
        mean = float(quartz[name]["quartz_mean_nll"])
        steps = 1024 if "wikitext" in name else 128
        if name in {"wikitext_nll", "held_out_wikitext_1024"}:
            steps = 1024
        results[name] = {
            "mean_nll": mean,
            "steps": [{"log_probability": 0.0}] * steps,
        }
    if nll_live.get("cases"):
        for row in nll_live["cases"]:
            name = row["name"]
            if name in results:
                results[name]["mean_nll"] = float(row["mean_nll"])
                scored = int(row.get("scored") or 0)
                if scored:
                    results[name]["steps"] = [{"log_probability": 0.0}] * scored
    return results, authority


def retained_generated() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    functional = load_json(FUNCTIONAL_V2)
    inputs = load_json(V2_INPUTS)
    quartz_rows = []
    for row in functional.get("quartz") or []:
        expected = str(inputs["cases"][row["case"]]["expected"])
        parsed = parse_functional_row(row, expected)
        parsed["case"] = row["case"]
        parsed["engine"] = "quartz"
        quartz_rows.append(parsed)
    llama_first = first_llama_logits(functional.get("llama_generate"))
    llama_rows: dict[str, dict[str, Any]] = {}
    for name, row in llama_first.items():
        choice = choice_from_token(row.get("token"))
        expected = str(inputs["cases"][name]["expected"])
        parsed = parse_generated_answer(choice or "", expected)
        parsed["next_token"] = row.get("token")
        parsed["top_two"] = [row.get("token"), row.get("runner_up_token")]
        parsed["logit"] = row.get("logit")
        parsed["runner_up_logit"] = row.get("runner_up_logit")
        parsed["runner_up_token"] = row.get("runner_up_token")
        parsed["case"] = name
        parsed["engine"] = "llama"
        llama_rows[name] = parsed
    quartz = {str(row["case"]): row for row in quartz_rows}
    return quartz, llama_rows


def both_engines_arithmetic() -> dict[str, Any]:
    quartz, llama = retained_generated()
    quartz_row = quartz["task_arithmetic"]
    llama_row = llama["task_arithmetic"]
    return {
        "quartz": quartz_row,
        "llama": llama_row,
        "both_emit_A": quartz_row.get("actual") == "A"
        and llama_row.get("actual") == "A",
        "independently_correct": "B",
        "independently_correct_value": 42,
        "kernel_or_scorer_bug": False,
        "authority_item_miss": True,
    }


def build_fixture() -> dict[str, Any]:
    inputs = load_json(V2_INPUTS)
    audit = audit_v2_functional_cases(inputs)
    diagnostics = freeze_arithmetic_diagnostics()
    nll, authority = retained_nll_bundle()
    quartz, llama = retained_generated()
    engines = both_engines_arithmetic()
    v2_q = quality_v2_verdicts(nll, authority, inputs, generated=quartz)
    v3_q = quality_v3_verdicts(
        nll, authority, inputs, generated=quartz, engine_logits=quartz
    )
    v2_l = quality_v2_verdicts(nll, authority, inputs, generated=llama)
    v3_l = quality_v3_verdicts(
        nll, authority, inputs, generated=llama, engine_logits=llama
    )
    opt056 = load_json(OPT056)
    qlt = load_json(QUALITY_CONTRACT)
    preflight = evaluate_preflight_quality(
        load_json(PREFLIGHT_FUNCTIONAL),
        load_json(PREFLIGHT_HELD),
        inputs,
        nll=nll,
        authority=authority,
        generated_override=quartz,
    )
    release = oracle_policy(bool(v2_q.get("all")), diagnostic_performance=False)
    diagnostic = oracle_policy(bool(v2_q.get("all")), diagnostic_performance=True)
    return {
        "schema_version": 1,
        "task": "OPT-073",
        "status": "quality_policy",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "native_reuse": [NATIVE, LLAMA_ORACLE],
        "v2_audit": audit,
        "arithmetic_diagnostics": diagnostics,
        "both_engines_arithmetic": engines,
        "frozen_arithmetic_logits": dict(FROZEN_ARITHMETIC_LOGITS),
        "quality_v2": {"quartz": v2_q, "llama": v2_l},
        "quality_v3": {"quartz": v3_q, "llama": v3_l},
        "preflight": {
            "status": preflight["status"],
            "opt056_quality_requirement_met": preflight[
                "opt056_quality_requirement_met"
            ],
            "is_release_evidence": False,
            "held_out_targets": 32,
            "selected_quality_verdicts": preflight["selected_quality_verdicts"],
        },
        "release_gate": release,
        "diagnostic_performance": diagnostic,
        "legacy": {
            "opt056_tasks_pass": opt056["quality"]["production_optimization"]["tasks"][
                "pass"
            ],
            "opt056_all": opt056["quality"]["production_optimization"]["all"],
            "qlt001_threshold": (qlt.get("thresholds") or {}).get(
                "nll_ppl_ratio", 1.05
            ),
            "historical_only": True,
            "not_relabeled": True,
        },
        "does_not_replace_opt056": True,
        "does_not_replace_opt016": True,
        "proof_limit": list(PROOF),
        "rationale": RATIONALE,
        "report_path": "evidence/optimization/opt073-quality-policy/REPORT.md",
    }


def write_report(fixture: Mapping[str, Any]) -> None:
    arithmetic = fixture["both_engines_arithmetic"]
    v2q = fixture["quality_v2"]["quartz"]
    v3q = fixture["quality_v3"]["quartz"]
    text = f"""# OPT-073 — Separate engine quality from the authority's arithmetic miss

Status: **quality policy frozen**. `claims_throughput: false`. No arithmetic
kernel change and no speedup are claimed. Authority remains llama.cpp
`{LLAMA_REV}` and GGUF SHA-256 `{GGUF_SHA}`.

## Why this sitting exists

OPT-058 established valid no-thinking functional prompts. Both engines still
miss `task_arithmetic`. OPT-069 preflight counted functional tokens and then
release continued into timed oracles after quality-v2 failed. This increment
authenticates the prompts/scorer, freezes a versioned dual-verdict policy, and
fail-closes release on required quality.

## Arithmetic audit (17 + 25)

Human-verifiable truth written first: **17 + 25 = 42**. On the original v2
options, **B = 42 is correct** and **A = 41 is wrong**. C=43 and D=44 are also
wrong. The parser still accepts exactly one of A/B/C/D after stripping
whitespace and rejects empty output, extra text, a second answer, and
teacher-forced greedy tokens.

Authenticated original v2 rendering matches
`src/template.cpp` `render_user_turn(..., enable_thinking=false)` and ends at
`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`. Tokenizer IDs are the frozen
v2 context. Max new tokens remain 16. Logits are not masked.

Retained two-engine original v2 result:

| Engine | Parsed | First token | Independently correct |
|---|---|---:|---|
| Quartz | {arithmetic["quartz"].get("actual")} ({arithmetic["quartz"].get("reason")}) | {arithmetic["quartz"].get("next_token")} | B / 42 |
| llama.cpp | {arithmetic["llama"].get("actual")} ({arithmetic["llama"].get("reason")}) | {arithmetic["llama"].get("next_token")} | B / 42 |

Both engines emit A. That **does not** prove a Quartz kernel or scorer bug.
No objective rendering/parser defect was found, so the v2 prompt is not
rewritten. Four arithmetic diagnostics were frozen before any candidate
measurement: original v2, one fixed option permutation (42 moves to C), direct
numeric completion under the same template, and one independent paraphrase.
Every diagnostic is retained. Passing paraphrases were not selected.

Frozen llama first-token snapshot on original v2: token 32 (A) logit
{FROZEN_ARITHMETIC_LOGITS["authority_logit"]}, runner-up 33 (B) logit
{FROZEN_ARITHMETIC_LOGITS["authority_runner_up_logit"]}, margin
{FROZEN_ARITHMETIC_LOGITS["authority_margin"]}.

## Quality-v3 dual verdict

Rationale: {fixture["rationale"]}

| Suite | Quartz status | Absolute accuracy | Engine non-regression |
|---|---|---|---|
| quality-v2 | {v2q.get("status")} all={v2q.get("all")} | n/a (v2 requires B on 17+25) | n/a |
| quality-v3 | {v3q.get("status")} all={v3q.get("all")} | {v3q["absolute_task_accuracy"]["status"]} | {v3q["engine_non_regression"]["status"]} |

Absolute arithmetic accuracy stays **failed**. Internal candidate admission may
use quality-v3 engine non-regression plus both 1024-target PPL ratios <= 1.01
and recurrence drift <= 0.02. This does **not** replace OPT-056 or OPT-016.
QLT-001 and the original eight-task OPT-056 functional fail remain historical.

Same-input near-ties cannot waive a known wrong semantic answer. Other wrong
choices (C, D, empty, extra text) on the missed item are not waived. Missing
NLL, authority, or generated answers yield **incomplete**, not pass.

## Release preflight

OPT-069 now parses all eight functional answers and records each required
selected quality verdict. Token count alone is not sufficient. A release that
would claim OPT-056 stops while quality-v2 `all` is false, before P/D/2K
oracles. `--diagnostic-performance` is an explicit non-release mode: timing may
run, the quality failure is retained, and keep/release claims are prohibited.

Current retained preflight: status `{fixture["preflight"]["status"]}`;
OPT-056 quality requirement met =
{fixture["preflight"]["opt056_quality_requirement_met"]}.
Default release oracles: run={fixture["release_gate"]["run_oracles"]}
eligible={fixture["release_gate"]["release_eligible"]}.

## Proof limit

- no throughput claim
- no arithmetic kernel change
- B=42 remains the correct 17+25 answer
- A=41 remains wrong
- both engines failing is not kernel blame
- no prompt rewrite without an objective defect
- no token mask
- no teacher-forced functional answer
- no selecting only paraphrases that pass
- quality-v3 does not replace OPT-056 or OPT-016
- missing evidence is incomplete
- failed required quality stops release before long timing
- diagnostic performance is not a release or keep
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def run_phase(phase: str, run_dir: Path | None = None) -> dict[str, Any]:
    inputs = load_json(V2_INPUTS)
    if phase == "arithmetic":
        audit = audit_v2_functional_cases(inputs)
        diagnostics = freeze_arithmetic_diagnostics()
        engines = both_engines_arithmetic()
        payload: dict[str, Any] = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "audit": audit,
            "arithmetic_diagnostics": [
                {
                    k: row[k]
                    for k in ("id", "expected", "parser", "independently_correct_value")
                }
                for row in diagnostics
            ],
            "diagnostic_count": len(diagnostics),
            "engines": ["quartz", "llama.cpp"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "both_engines_arithmetic": engines,
            "kernel_blame": False,
            "prompt_rewrite": False,
            "claims_throughput": False,
        }
        payload["success"] = (
            not audit["rendering_objectively_wrong"]
            and audit["arithmetic"]["expected_unchanged"]
            and engines["both_emit_A"]
            and engines["independently_correct"] == "B"
        )
    elif phase == "preflight":
        nll, authority = retained_nll_bundle()
        quartz, _llama = retained_generated()
        evaluated = evaluate_preflight_quality(
            load_json(PREFLIGHT_FUNCTIONAL),
            load_json(PREFLIGHT_HELD),
            inputs,
            nll=nll,
            authority=authority,
            generated_override=quartz,
        )
        skipped = False
        try:
            require_selected_quality_verdicts({"selected_quality_verdicts": {}})
        except QualityPolicyError:
            skipped = True
        release = oracle_policy(False, diagnostic_performance=False)
        diagnostic = oracle_policy(False, diagnostic_performance=True)
        payload = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "held_out_targets": 32,
            "functional_cases": 8,
            "preflight": evaluated,
            "skipped_quality_phases_fail_closed": skipped,
            "release_gate": release,
            "diagnostic_performance": diagnostic,
            "opt056_quality_requirement_met": evaluated[
                "opt056_quality_requirement_met"
            ],
            "is_release_evidence": False,
            "claims_throughput": False,
        }
        payload["success"] = (
            skipped
            and release["run_oracles"] is False
            and diagnostic["keep_claims_allowed"] is False
            and evaluated["selected_quality_verdicts"]["parsed_functional_answers"]
            is True
        )
    elif phase == "quality":
        fixture = build_fixture()
        write_json(FIXTURE, fixture)
        write_report(fixture)
        v2 = fixture["quality_v2"]["quartz"]
        v3 = fixture["quality_v3"]["quartz"]
        payload = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "quality_v2": {"status": v2["status"], "all": v2["all"]},
            "quality_v3": {
                "status": v3["status"],
                "all": v3["all"],
                "absolute_task_accuracy": v3["absolute_task_accuracy"]["status"],
                "engine_non_regression": v3["engine_non_regression"]["status"],
            },
            "does_not_replace_opt056": True,
            "does_not_replace_opt016": True,
            "universally_passing_absolute_baseline": False,
            "claims_throughput": False,
            "fixture": "fixtures/opt073_quality_policy.json",
            "report": "evidence/optimization/opt073-quality-policy/REPORT.md",
        }
        payload["success"] = (
            v2["all"] is False
            and v3["absolute_task_accuracy"]["status"] == "fail"
            and v3["engine_non_regression"]["status"] == "pass"
            and fixture["does_not_replace_opt056"] is True
        )
    else:
        raise ValueError(f"unknown OPT-073 phase {phase}")
    payload["task"] = "OPT-073"
    payload["phase"] = phase
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt073-result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("arithmetic", "preflight", "quality"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase(args.phase, args.run_dir)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
