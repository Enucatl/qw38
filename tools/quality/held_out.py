"""OPT-116 held-out free-running admission cases. Distinct from calibration."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError

ADMISSION_ROLE = "admission"
CALIBRATION_ROLE = "calibration"
CASE_CLASSES: tuple[str, ...] = (
    "extraction_long_context",
    "instruction_constraints",
    "arithmetic_reasoning",
    "code_executable",
    "tool_call_json",
    "multi_turn_prose",
)
MIN_PER_CLASS = 4
MIN_CASES = 32
SAMPLE_SUBSET = 12
GREEDY_SAMPLER: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 0,
}
SAMPLE_SAMPLER: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "seeds": (11, 29, 47),
}
MAX_NEW_TOKENS = 64
PROSE_MAX_NEW_TOKENS = 128


def _case(
    *,
    case_id: str,
    suite_class: str,
    prompt: str,
    validator: str,
    expected: str | None = None,
    turns: Sequence[Mapping[str, str]] | None = None,
    retrieval_positions: Sequence[int] | None = None,
    executable: Mapping[str, Any] | None = None,
    json_schema: Mapping[str, Any] | None = None,
    instruction: Mapping[str, Any] | None = None,
    rubric: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
    role: str = ADMISSION_ROLE,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "class": suite_class,
        "role": role,
        "prompt": prompt,
        "turns": [dict(turn) for turn in turns] if turns else None,
        "validator": validator,
        "expected": expected,
        "retrieval_positions": list(retrieval_positions or ()),
        "executable": dict(executable) if executable else None,
        "json_schema": dict(json_schema) if json_schema else None,
        "instruction": dict(instruction) if instruction else None,
        "rubric": rubric,
        "teacher_forced_answers": False,
        "max_new_tokens": max_new_tokens,
        "consumes_own_sampled_tokens": True,
    }


def _extraction_cases() -> tuple[dict[str, Any], ...]:
    early = (
        "Record dump. Token 16: CITY=Bruges. Later notes mention Paris, "
        "Oslo, and Lima. Ignore the later cities. Which city is recorded "
        "at the earliest dump token? Reply with the city name only."
    )
    mid = (
        "Prefix filler about weather. Midway code ALPHA-9173 is assigned "
        "to crate 12. Later codes are BETA-1001 and GAMMA-0007. What is "
        "crate 12's code? Reply with the code only."
    )
    late = (
        "A ledger lists: start=red, middle=green, near-end=indigo, "
        "final=gold. What color is recorded near the end but not last? "
        "One word."
    )
    two = (
        "Needle A at position 32 is river=Thames. Needle B at position "
        "8000 is year=1666. Distractors: Seine 1851. Report river then "
        "year as 'river year'."
    )
    listed = (
        "Inventory in order: copper, tin, zinc, nickel, lead. What is "
        "the third item? One word."
    )
    distract = (
        "True id of the package is QW-4401. A nearby sticker wrongly "
        "says QW-4410. What is the true package id? Reply with the id."
    )
    return (
        _case(
            case_id="extract_early_city",
            suite_class="extraction_long_context",
            prompt=early,
            validator="exact_text",
            expected="Bruges",
            retrieval_positions=(16,),
        ),
        _case(
            case_id="extract_mid_code",
            suite_class="extraction_long_context",
            prompt=mid,
            validator="exact_text",
            expected="ALPHA-9173",
            retrieval_positions=(4096,),
        ),
        _case(
            case_id="extract_late_not_last",
            suite_class="extraction_long_context",
            prompt=late,
            validator="exact_text",
            expected="indigo",
            retrieval_positions=(8000,),
        ),
        _case(
            case_id="extract_two_needles",
            suite_class="extraction_long_context",
            prompt=two,
            validator="exact_text",
            expected="Thames 1666",
            retrieval_positions=(32, 8000),
        ),
        _case(
            case_id="extract_list_index",
            suite_class="extraction_long_context",
            prompt=listed,
            validator="exact_text",
            expected="zinc",
            retrieval_positions=(64, 128, 192),
        ),
        _case(
            case_id="extract_ignore_distractor",
            suite_class="extraction_long_context",
            prompt=distract,
            validator="exact_text",
            expected="QW-4401",
            retrieval_positions=(120,),
        ),
    )


def _instruction_cases() -> tuple[dict[str, Any], ...]:
    return (
        _case(
            case_id="instr_three_words",
            suite_class="instruction_constraints",
            prompt="Reply with exactly three English words and nothing else.",
            validator="word_count",
            instruction={"words": 3},
        ),
        _case(
            case_id="instr_json_only",
            suite_class="instruction_constraints",
            prompt='Reply with only this JSON object: {"ok": true}',
            validator="exact_json",
            expected='{"ok": true}',
        ),
        _case(
            case_id="instr_no_greeting",
            suite_class="instruction_constraints",
            prompt=(
                "Name the chemical symbol for water. Do not greet. Do not "
                "use the word hello. Reply with the formula only."
            ),
            validator="forbidden_and_exact",
            expected="H2O",
            instruction={"forbidden": ["hello", "hi ", "hey"]},
        ),
        _case(
            case_id="instr_uppercase",
            suite_class="instruction_constraints",
            prompt="Reply with the word READY in uppercase and nothing else.",
            validator="exact_text",
            expected="READY",
        ),
        _case(
            case_id="instr_two_bullets",
            suite_class="instruction_constraints",
            prompt=(
                "List exactly two fruits as markdown bullets, one fruit "
                "per line, no intro or outro."
            ),
            validator="bullet_count",
            instruction={"bullets": 2},
        ),
        _case(
            case_id="instr_answer_prefix",
            suite_class="instruction_constraints",
            prompt="Reply in the exact form ANSWER: 9",
            validator="exact_text",
            expected="ANSWER: 9",
        ),
    )


def _arithmetic_cases() -> tuple[dict[str, Any], ...]:
    return (
        _case(
            case_id="arith_add",
            suite_class="arithmetic_reasoning",
            prompt="Compute 47 + 18. Reply with the integer only.",
            validator="exact_text",
            expected="65",
        ),
        _case(
            case_id="arith_mul",
            suite_class="arithmetic_reasoning",
            prompt="Compute 12 * 11. Reply with the integer only.",
            validator="exact_text",
            expected="132",
        ),
        _case(
            case_id="arith_percent",
            suite_class="arithmetic_reasoning",
            prompt="What is 15% of 80? Reply with the integer only.",
            validator="exact_text",
            expected="12",
        ),
        _case(
            case_id="arith_compare",
            suite_class="arithmetic_reasoning",
            prompt=(
                "Which is larger, 2/3 or 3/5? Reply with the larger "
                "fraction in lowest terms."
            ),
            validator="exact_text",
            expected="2/3",
        ),
        _case(
            case_id="arith_minutes",
            suite_class="arithmetic_reasoning",
            prompt=(
                "A meeting starts at 09:20 and lasts 95 minutes. What is "
                "the end time in 24-hour HH:MM?"
            ),
            validator="exact_text",
            expected="10:55",
        ),
    )


def _code_cases() -> tuple[dict[str, Any], ...]:
    return (
        _case(
            case_id="code_len",
            suite_class="code_executable",
            prompt=(
                "Write a Python expression or function body that returns "
                "len([1, 2, 3]). Output only executable Python."
            ),
            validator="python_exec",
            executable={"expect": 3},
        ),
        _case(
            case_id="code_sum",
            suite_class="code_executable",
            prompt=(
                "Write Python that binds result = sum(range(5)). Output "
                "only executable Python."
            ),
            validator="python_exec",
            executable={"expect": 10, "name": "result"},
        ),
        _case(
            case_id="code_reverse",
            suite_class="code_executable",
            prompt=(
                "Write Python that binds result = 'abcd'[::-1]. Output "
                "only executable Python."
            ),
            validator="python_exec",
            executable={"expect": "dcba", "name": "result"},
        ),
        _case(
            case_id="code_max",
            suite_class="code_executable",
            prompt=(
                "Write Python that binds result = max(3, 9, 4). Output "
                "only executable Python."
            ),
            validator="python_exec",
            executable={"expect": 9, "name": "result"},
        ),
        _case(
            case_id="code_even",
            suite_class="code_executable",
            prompt=(
                "Write Python that binds result to the list of even "
                "integers from 0 through 6 inclusive. Output only "
                "executable Python."
            ),
            validator="python_exec",
            executable={"expect": [0, 2, 4, 6], "name": "result"},
        ),
    )


def _tool_cases() -> tuple[dict[str, Any], ...]:
    weather = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "get_weather"},
            "arguments": {
                "type": "object",
                "required": ["city"],
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    search = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "search"},
            "arguments": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    }
    add = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "add"},
            "arguments": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        },
    }
    nested = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "book_flight"},
            "arguments": {
                "type": "object",
                "required": ["route"],
                "properties": {
                    "route": {
                        "type": "object",
                        "required": ["from", "to"],
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                        },
                    }
                },
            },
        },
    }
    status = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "set_status"},
            "arguments": {
                "type": "object",
                "required": ["status"],
                "properties": {
                    "status": {"enum": ["open", "closed"]},
                },
            },
        },
    }
    return (
        _case(
            case_id="tool_weather_city",
            suite_class="tool_call_json",
            prompt=(
                "Emit a JSON tool call named get_weather with argument "
                'city="Oslo". JSON only.'
            ),
            validator="json_schema",
            expected='{"name": "get_weather", "arguments": {"city": "Oslo"}}',
            json_schema=weather,
        ),
        _case(
            case_id="tool_search_query",
            suite_class="tool_call_json",
            prompt=(
                "Emit a JSON tool call named search with argument "
                'query="gdn state". JSON only.'
            ),
            validator="json_schema",
            expected='{"name": "search", "arguments": {"query": "gdn state"}}',
            json_schema=search,
        ),
        _case(
            case_id="tool_add_args",
            suite_class="tool_call_json",
            prompt=(
                "Emit a JSON tool call named add with integer arguments "
                "a=2 and b=5. JSON only."
            ),
            validator="json_schema",
            expected='{"name": "add", "arguments": {"a": 2, "b": 5}}',
            json_schema=add,
        ),
        _case(
            case_id="tool_nested_route",
            suite_class="tool_call_json",
            prompt=(
                "Emit a JSON tool call named book_flight with nested "
                'route from="HEL" to="ARN". JSON only.'
            ),
            validator="json_schema",
            expected=(
                '{"name": "book_flight", "arguments": '
                '{"route": {"from": "HEL", "to": "ARN"}}}'
            ),
            json_schema=nested,
        ),
        _case(
            case_id="tool_enum_status",
            suite_class="tool_call_json",
            prompt=(
                "Emit a JSON tool call named set_status with argument "
                'status="closed". JSON only.'
            ),
            validator="json_schema",
            expected='{"name": "set_status", "arguments": {"status": "closed"}}',
            json_schema=status,
        ),
    )


def _prose_cases() -> tuple[dict[str, Any], ...]:
    return (
        _case(
            case_id="prose_continue_story",
            suite_class="multi_turn_prose",
            prompt="",
            turns=(
                {
                    "role": "user",
                    "content": "Start a two-sentence story about a lighthouse.",
                },
                {
                    "role": "assistant",
                    "content": "The lighthouse kept a cold watch over the reef.",
                },
                {
                    "role": "user",
                    "content": (
                        "Continue in two sentences. Keep the lighthouse. "
                        "Do not restart the story."
                    ),
                },
            ),
            validator="rubric_prose",
            rubric=True,
            max_new_tokens=PROSE_MAX_NEW_TOKENS,
        ),
        _case(
            case_id="prose_summarize_then_ask",
            suite_class="multi_turn_prose",
            prompt="",
            turns=(
                {
                    "role": "user",
                    "content": (
                        "Context: The ferry to Naxos leaves at 07:10 and "
                        "returns at 19:40. Summarize in one sentence."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The Naxos ferry departs at 07:10 and comes back at 19:40."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Using only that context, say when it returns. One sentence."
                    ),
                },
            ),
            validator="rubric_prose",
            rubric=True,
            expected="19:40",
            max_new_tokens=PROSE_MAX_NEW_TOKENS,
        ),
        _case(
            case_id="prose_correct_then_explain",
            suite_class="multi_turn_prose",
            prompt="",
            turns=(
                {"role": "user", "content": "Is 21 a prime number?"},
                {"role": "assistant", "content": "No."},
                {
                    "role": "user",
                    "content": "Explain why in two short sentences.",
                },
            ),
            validator="rubric_prose",
            rubric=True,
            expected="21",
            max_new_tokens=PROSE_MAX_NEW_TOKENS,
        ),
        _case(
            case_id="prose_two_speakers",
            suite_class="multi_turn_prose",
            prompt="",
            turns=(
                {
                    "role": "user",
                    "content": "Write one line of dialogue from Maya.",
                },
                {"role": "assistant", "content": 'Maya: "The kettle is on."'},
                {
                    "role": "user",
                    "content": (
                        "Reply with one line from Noah that refers to the "
                        "kettle. Prefix with Noah:"
                    ),
                },
            ),
            validator="rubric_prose",
            rubric=True,
            expected="Noah",
            max_new_tokens=PROSE_MAX_NEW_TOKENS,
        ),
        _case(
            case_id="prose_remember_constraint",
            suite_class="multi_turn_prose",
            prompt="",
            turns=(
                {
                    "role": "user",
                    "content": (
                        "Call the cat Pepper. Do not rename it. Confirm "
                        "the name in three words."
                    ),
                },
                {"role": "assistant", "content": "The cat Pepper."},
                {
                    "role": "user",
                    "content": (
                        "In one sentence, say what the cat is named. Use "
                        "the agreed name."
                    ),
                },
            ),
            validator="rubric_prose",
            rubric=True,
            expected="Pepper",
            max_new_tokens=PROSE_MAX_NEW_TOKENS,
        ),
    )


def admission_cases() -> tuple[dict[str, Any], ...]:
    cases = (
        *_extraction_cases(),
        *_instruction_cases(),
        *_arithmetic_cases(),
        *_code_cases(),
        *_tool_cases(),
        *_prose_cases(),
    )
    if len(cases) < MIN_CASES:
        raise QualityFrameworkError(
            f"held-out admission needs {MIN_CASES} cases, got {len(cases)}"
        )
    counts: dict[str, int] = {name: 0 for name in CASE_CLASSES}
    for case in cases:
        counts[str(case["class"])] += 1
    short = [name for name, count in counts.items() if count < MIN_PER_CLASS]
    if short:
        raise QualityFrameworkError("held-out class under-covered: " + ",".join(short))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise QualityFrameworkError("duplicate held-out case id")
    return cases


def calibration_cases() -> tuple[dict[str, Any], ...]:
    """Development-only prompts. Must not be used to tune admission."""
    return (
        _case(
            case_id="cal_extract_city",
            suite_class="extraction_long_context",
            prompt="Dev only. Capital of Italy in one word?",
            validator="exact_text",
            expected="Rome",
            role=CALIBRATION_ROLE,
        ),
        _case(
            case_id="cal_instr_one_word",
            suite_class="instruction_constraints",
            prompt="Dev only. Reply with the single word yes.",
            validator="exact_text",
            expected="yes",
            role=CALIBRATION_ROLE,
        ),
        _case(
            case_id="cal_arith_one_plus_one",
            suite_class="arithmetic_reasoning",
            prompt="Dev only. 1+1 integer only.",
            validator="exact_text",
            expected="2",
            role=CALIBRATION_ROLE,
        ),
        _case(
            case_id="cal_code_one",
            suite_class="code_executable",
            prompt="Dev only. Python that binds result = 1.",
            validator="python_exec",
            executable={"expect": 1, "name": "result"},
            role=CALIBRATION_ROLE,
        ),
        _case(
            case_id="cal_tool_ping",
            suite_class="tool_call_json",
            prompt="Dev only. JSON tool ping.",
            validator="json_schema",
            expected='{"name": "ping", "arguments": {}}',
            json_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"const": "ping"}},
            },
            role=CALIBRATION_ROLE,
        ),
        _case(
            case_id="cal_prose_hello",
            suite_class="multi_turn_prose",
            prompt="Dev only. Write one calm sentence.",
            validator="rubric_prose",
            rubric=True,
            role=CALIBRATION_ROLE,
        ),
    )


def sample_subset_ids(cases: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    pool = list(cases or admission_cases())
    selected: list[str] = []
    by_class: dict[str, list[str]] = {name: [] for name in CASE_CLASSES}
    for case in pool:
        by_class[str(case["class"])].append(str(case["id"]))
    for name in CASE_CLASSES:
        selected.extend(by_class[name][:2])
    if len(selected) < SAMPLE_SUBSET:
        raise QualityFrameworkError("sample subset smaller than 12")
    return selected[:SAMPLE_SUBSET]


def split_hash(cases: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "id": case["id"],
            "class": case["class"],
            "role": case["role"],
            "prompt": case.get("prompt"),
            "turns": case.get("turns"),
            "expected": case.get("expected"),
            "validator": case["validator"],
        }
        for case in cases
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise QualityFrameworkError("no JSON object in generated text")
    return json.loads(stripped[start : end + 1])


def _extract_python(text: str) -> str:
    stripped = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


def _schema_ok(instance: Any, schema: Mapping[str, Any]) -> bool:
    if schema.get("const") is not None:
        return instance == schema["const"]
    if "enum" in schema:
        return instance in list(schema["enum"])
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(instance, dict):
            return False
        required = list(schema.get("required") or [])
        if any(key not in instance for key in required):
            return False
        properties = schema.get("properties") or {}
        return all(
            _schema_ok(instance[key], spec)
            for key, spec in properties.items()
            if key in instance
        )
    if expected_type == "string":
        return isinstance(instance, str)
    if expected_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    return True


def validate_generation(case: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Objective validators. Rubric prose is scored separately."""
    kind = str(case["validator"])
    stripped = text.strip()
    passed = False
    reason = "unscored"
    if kind == "exact_text":
        passed = _normalize_text(stripped) == _normalize_text(str(case["expected"]))
        reason = "exact_match" if passed else "mismatch"
    elif kind == "exact_json":
        try:
            passed = json.loads(stripped) == json.loads(str(case["expected"]))
            reason = "json_match" if passed else "json_mismatch"
        except json.JSONDecodeError:
            passed = False
            reason = "invalid_json"
    elif kind == "word_count":
        words = [token for token in stripped.split() if token]
        want = int((case.get("instruction") or {})["words"])
        passed = len(words) == want
        reason = "word_count" if passed else f"word_count={len(words)}"
    elif kind == "bullet_count":
        bullets = [
            line
            for line in stripped.splitlines()
            if line.strip().startswith(("-", "*"))
        ]
        want = int((case.get("instruction") or {})["bullets"])
        passed = len(bullets) == want
        reason = "bullet_count" if passed else f"bullets={len(bullets)}"
    elif kind == "forbidden_and_exact":
        lowered = stripped.lower()
        forbidden = list((case.get("instruction") or {}).get("forbidden") or [])
        hit = [word for word in forbidden if word in lowered]
        exact = _normalize_text(stripped) == _normalize_text(str(case["expected"]))
        passed = exact and not hit
        reason = "ok" if passed else ("forbidden" if hit else "mismatch")
    elif kind == "json_schema":
        try:
            parsed = _extract_json(stripped)
            schema = case.get("json_schema") or {}
            passed = _schema_ok(parsed, schema)
            if passed and case.get("expected"):
                passed = parsed == json.loads(str(case["expected"]))
            reason = "schema_ok" if passed else "schema_fail"
        except (json.JSONDecodeError, QualityFrameworkError, ValueError):
            passed = False
            reason = "invalid_json"
    elif kind == "python_exec":
        try:
            code = _extract_python(stripped)
            tree = ast.parse(code)
            if any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                for node in ast.walk(tree)
            ):
                raise QualityFrameworkError("imports forbidden")
            safe_builtins = {
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "range": range,
                "list": list,
                "reversed": reversed,
                "int": int,
                "str": str,
            }
            env: dict[str, Any] = {}
            exec(
                compile(tree, "<opt116>", "exec"), {"__builtins__": safe_builtins}, env
            )
            name = str((case.get("executable") or {}).get("name") or "result")
            if name not in env:
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    value = eval(
                        compile(ast.Expression(tree.body[0].value), "<opt116>", "eval"),
                        {"__builtins__": safe_builtins},
                        env,
                    )
                    env[name] = value
            passed = env.get(name) == (case.get("executable") or {}).get("expect")
            reason = "exec_ok" if passed else "exec_mismatch"
        except (SyntaxError, QualityFrameworkError, ValueError, TypeError):
            passed = False
            reason = "exec_error"
    elif kind == "rubric_prose":
        passed = True
        reason = "deferred_to_rubric"
    else:
        raise QualityFrameworkError(f"unknown validator {kind}")
    return {
        "id": case["id"],
        "pass": passed,
        "reason": reason,
        "validator": kind,
        "objective": kind != "rubric_prose",
        "text": text,
    }


def refuse_missing_case(
    cases: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = [str(case["id"]) for case in cases if str(case["id"]) not in scores]
    if missing:
        raise QualityFrameworkError("missing case " + ",".join(missing))
