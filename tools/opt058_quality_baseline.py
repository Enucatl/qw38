"""OPT-058 finite scheduler and v2 quality baseline helpers.

Teacher-forced NLL scoring and free-running functional answers stay separate.
Missing held-out llama authority is incomplete, never a pass by omission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOKEN_271 = 271
CHOICES = ("A", "B", "C", "D")
MAX_NEW_TOKENS = 16
PRODUCTION_PPL_RATIO = 1.01
RECURRENCE_INCREMENTAL_NLL = 0.02
NO_THINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TASK_NAMES = (
    "task_arithmetic",
    "task_python_len",
    "task_inference",
    "task_minutes",
    "task_sort",
    "task_json",
    "task_reading",
    "task_sequence",
)
NLL_CASES = (
    "wikitext_nll",
    "held_out_wikitext_1024",
    "recurrence_short",
    "recurrence_long",
)
V2_REQUIRED_CASES = NLL_CASES + TASK_NAMES

_SECOND_ANSWER = re.compile(r"[ABCD].*[ABCD]", re.DOTALL)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_tokens(tokens: Sequence[int]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
    return sha256_bytes(payload)


def decode_token_271(tokenizer: Any) -> str:
    """Pinned tokenizer decoding of the OPT-056 greedy token."""
    return tokenizer.decode([TOKEN_271], skip_special_tokens=False)


def render_no_thinking_user_turn(user: str) -> str:
    """Match src/template.cpp render_user_turn(..., enable_thinking=false)."""
    query = user.strip()
    if not query:
        raise ValueError("user turn cannot be empty")
    return f"<|im_start|>user\n{query}<|im_end|>\n{NO_THINKING_SUFFIX}"


def parse_generated_answer(text: str, expected: str) -> dict[str, Any]:
    """Accept exactly one of A/B/C/D after stripping leading/trailing whitespace.

    Rejects empty output, extra words, a second answer, and first-letter shortcuts.
    Does not inspect logits or mask the vocabulary.
    """
    if expected not in CHOICES:
        raise ValueError(f"expected answer must be one of {CHOICES}")
    stripped = text.strip()
    record: dict[str, Any] = {
        "raw": text,
        "stripped": stripped,
        "expected": expected,
        "pass": False,
        "reason": "",
        "actual": None,
        "used_teacher_forcing": False,
    }
    if stripped == "":
        record["reason"] = "empty"
        return record
    if stripped in CHOICES:
        record["actual"] = stripped
        if stripped == expected:
            record["pass"] = True
            record["reason"] = "exact_choice"
        else:
            record["reason"] = "wrong_choice"
        return record
    if _SECOND_ANSWER.search(stripped):
        record["reason"] = "second_answer"
        return record
    record["reason"] = "extra_text"
    return record


def inspect_tap(values: Sequence[float]) -> dict[str, Any]:
    """Finite versus nonfinite summary for an injected or captured tap."""
    finite = 0
    first_nonfinite: int | None = None
    for index, value in enumerate(values):
        number = float(value)
        if math.isfinite(number):
            finite += 1
        elif first_nonfinite is None:
            first_nonfinite = index
    payload = json.dumps([float(v) for v in values], separators=(",", ":")).encode(
        "ascii"
    )
    return {
        "count": len(values),
        "finite_count": finite,
        "nonfinite_count": len(values) - finite,
        "first_nonfinite": first_nonfinite,
        "finite": first_nonfinite is None and len(values) > 0,
        "sha256": sha256_bytes(payload),
    }


def require_finite_tap(values: Sequence[float], *, name: str) -> dict[str, Any]:
    summary = inspect_tap(values)
    summary["name"] = name
    if not summary["finite"]:
        raise ValueError(
            f"nonfinite tap {name}: first_index={summary['first_nonfinite']} "
            f"finite={summary['finite_count']}/{summary['count']}"
        )
    return summary


def _mean_nll(record: Mapping[str, Any]) -> float:
    if "mean_nll" in record:
        return float(record["mean_nll"])
    steps = record.get("steps") or []
    if not steps:
        raise ValueError("NLL record has no steps")
    total = 0.0
    for step in steps:
        total += -float(step["log_probability"])
    return total / len(steps)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def quality_v2_verdicts(
    results: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    inputs: Mapping[str, Any],
    *,
    generated: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require every named case and its llama reference. Missing authority is incomplete."""
    verdict: dict[str, Any] = {
        "suite": "quality-v2",
        "status": "incomplete",
        "all": False,
        "missing_authority": [],
        "missing_cases": [],
        "used_teacher_forcing_for_answers": False,
    }
    named = list(inputs.get("cases", ()))
    if not named:
        named = list(V2_REQUIRED_CASES)
    authority_cases = (authority or {}).get("cases") or {}
    for name in named:
        if name in NLL_CASES and name not in results:
            verdict["missing_cases"].append(name)
        if name in NLL_CASES and name not in authority_cases:
            verdict["missing_authority"].append(name)
    if verdict["missing_cases"] or verdict["missing_authority"] or authority is None:
        verdict["status"] = "incomplete"
        verdict["all"] = False
        return verdict

    def ppl_ratio(name: str, expected_steps: int) -> dict[str, Any]:
        actual = results[name]
        auth = authority_cases[name]
        steps = actual.get("steps") or []
        ratio = math.exp(_mean_nll(actual) - float(auth["mean_nll"]))
        passed = (
            len(steps) == expected_steps
            and _finite(ratio)
            and ratio <= PRODUCTION_PPL_RATIO
        )
        return {
            "pass": passed,
            "ppl_ratio": ratio,
            "steps": len(steps),
        }

    verdict["wikitext_nll"] = ppl_ratio("wikitext_nll", 1024)
    verdict["held_out_wikitext_1024"] = ppl_ratio("held_out_wikitext_1024", 1024)
    short = results["recurrence_short"]
    long = results["recurrence_long"]
    short_a = authority_cases["recurrence_short"]
    long_a = authority_cases["recurrence_long"]
    drift = (_mean_nll(long) - float(long_a["mean_nll"])) - (
        _mean_nll(short) - float(short_a["mean_nll"])
    )
    verdict["recurrence"] = {
        "pass": _finite(drift) and drift <= RECURRENCE_INCREMENTAL_NLL,
        "incremental_nll": drift,
    }
    if generated is None:
        verdict["tasks"] = {
            "pass": False,
            "count": 8,
            "reason": "generated_answers_required",
        }
        verdict["status"] = "fail"
        verdict["all"] = False
        return verdict
    task_rows = []
    for name in TASK_NAMES:
        expected = str(inputs["cases"][name]["expected"])
        row = generated.get(name) or {}
        if "greedy_token" in row and "text" not in row and "stripped" not in row:
            parsed = {
                "pass": False,
                "reason": "teacher_forced_answer_rejected",
                "used_teacher_forcing": True,
                "expected": expected,
            }
            verdict["used_teacher_forcing_for_answers"] = True
        else:
            text = str(row.get("text", row.get("stripped", "")))
            parsed = parse_generated_answer(text, expected)
        task_rows.append(parsed)
        verdict[name] = parsed
    verdict["tasks"] = {
        "pass": all(item["pass"] for item in task_rows),
        "count": len(task_rows),
    }
    checks = [
        verdict["wikitext_nll"]["pass"],
        verdict["held_out_wikitext_1024"]["pass"],
        verdict["recurrence"]["pass"],
        verdict["tasks"]["pass"],
    ]
    verdict["all"] = all(checks)
    verdict["status"] = "pass" if verdict["all"] else "fail"
    return verdict


def original_task_instruction(question: str) -> str:
    return question + "\nReply with exactly one letter: A, B, C, or D."


def build_v2_inputs(
    quality_inputs: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Chat-templated functional prompts plus copied NLL spans. Does not rewrite QLT-001."""
    cases_in = quality_inputs["cases"]
    held_context = list(cases_in["recurrence_long"]["context"][:1])
    held_continuation = list(cases_in["recurrence_long"]["context"][1:1025])
    cases: dict[str, Any] = {
        "wikitext_nll": {
            "context": list(cases_in["wikitext_nll"]["context"]),
            "continuation": list(cases_in["wikitext_nll"]["continuation"]),
            "scoring": "teacher_forced_nll",
        },
        "held_out_wikitext_1024": {
            "context": held_context,
            "continuation": held_continuation,
            "scoring": "teacher_forced_nll",
            "stream_start": 16385,
            "stream_end": 17409,
            "continuation_sha256": sha256_tokens(held_continuation),
            "source": "fixtures/quality_inputs.json#cases.recurrence_long.context[1:1025]",
        },
        "recurrence_short": {
            "context": list(cases_in["recurrence_short"]["context"]),
            "continuation": list(cases_in["recurrence_short"]["continuation"]),
            "scoring": "teacher_forced_nll",
        },
        "recurrence_long": {
            "context": list(cases_in["recurrence_long"]["context"]),
            "continuation": list(cases_in["recurrence_long"]["continuation"]),
            "scoring": "teacher_forced_nll",
        },
    }
    for name in TASK_NAMES:
        source = cases_in[name]
        question = str(source["user"])
        instruction = original_task_instruction(question)
        rendered = render_no_thinking_user_turn(instruction)
        ids = list(tokenizer.encode(rendered, add_special_tokens=False).ids)
        cases[name] = {
            "user": question,
            "instruction": instruction,
            "rendered": rendered,
            "expected": source["expected"],
            "context": ids,
            "max_new_tokens": MAX_NEW_TOKENS,
            "scoring": "free_running_text",
            "original_context": list(source["context"]),
            "original_continuation": list(source["continuation"]),
        }
    return {
        "schema": "qw38.quality-inputs-v2",
        "version": 1,
        "chat_template": "no_thinking",
        "assistant_answer_boundary": NO_THINKING_SUFFIX,
        "dataset_sha256": quality_inputs.get("dataset_sha256"),
        "tokenizer_sha256": quality_inputs.get("tokenizer_sha256"),
        "max_new_tokens": MAX_NEW_TOKENS,
        "logit_masking": False,
        "cases": cases,
    }


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def bundle_cases(
    inputs: Mapping[str, Any],
    path: Path,
    cases: Sequence[str],
) -> None:
    """Write a QW38Q bundle for an explicit case list. Does not mutate the old seven-case path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"QW38Q\1\0\0")
        handle.write(struct.pack("<I", len(cases)))
        for name in cases:
            case = inputs["cases"][name]
            encoded = name.encode("ascii")
            context = list(case["context"])
            continuation = list(case.get("continuation") or [0])
            if not continuation:
                continuation = [0]
            handle.write(struct.pack("<H", len(encoded)))
            handle.write(encoded)
            handle.write(struct.pack("<II", len(context), len(continuation)))
            handle.write(struct.pack("<%dI" % len(context), *context))
            handle.write(struct.pack("<%dI" % len(continuation), *continuation))


def top_two(logits: Sequence[float]) -> tuple[int, float, int, float]:
    first = 0
    second = 1 if len(logits) > 1 else 0
    for index, value in enumerate(logits):
        number = float(value)
        if number > float(logits[first]) or (
            number == float(logits[first]) and index < first
        ):
            second = first
            first = index
        elif index != first and (
            number > float(logits[second])
            or (number == float(logits[second]) and index < second)
        ):
            second = index
    return first, float(logits[first]), second, float(logits[second])


def functional_record(
    *,
    engine: str,
    name: str,
    tokens: Sequence[int],
    logits0: Sequence[float] | None,
    text: str,
    expected: str,
    tokenizer_271: str,
) -> dict[str, Any]:
    parsed = parse_generated_answer(text, expected)
    first = second = first_logit = second_logit = None
    if logits0:
        first, first_logit, second, second_logit = top_two(logits0)
    return {
        "engine": engine,
        "case": name,
        "tokens": list(tokens)[:MAX_NEW_TOKENS],
        "text": text,
        "token_271_decoded": tokenizer_271,
        "next_token": int(tokens[0]) if tokens else None,
        "top_two": {
            "token": first,
            "logit": first_logit,
            "runner_up_token": second,
            "runner_up_logit": second_logit,
        },
        "parser": parsed,
        "scoring": "free_running_text",
    }


def merge_engine_functional(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        merged[str(row["case"])] = dict(row)
    return merged
