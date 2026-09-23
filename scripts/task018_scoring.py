"""Deterministic answer extractors selected by EVAL-01."""

from __future__ import annotations

import re
from dataclasses import dataclass

ASCII_WS = " \t\n\r\v\f"


@dataclass(frozen=True)
class Grade:
    """Represent one exact extraction and grading outcome."""

    valid: bool
    extracted: str | None
    correct: bool
    reason: str | None


def _final_answer_line(text: str) -> str | None:
    """Return the required last nonempty `Answer: ` line, if well formed."""
    lines = [line.strip(ASCII_WS) for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    if not nonempty:
        return None
    final = nonempty[-1]
    if not final.startswith("Answer: "):
        return None
    return final[len("Answer: ") :]


def grade_l12(text: str, expected: str) -> Grade:
    """Grade an L12 anchor after trimming ASCII edge whitespace only."""
    actual = text.strip(ASCII_WS)
    return Grade(True, actual, actual == expected, None)


def grade_retrieval(text: str, expected: str) -> Grade:
    """Grade a synthetic retrieval answer after ASCII edge trimming."""
    actual = text.strip(ASCII_WS)
    return Grade(True, actual, actual == expected, None)


def expand_compsec_key(key: str) -> set[int]:
    """Expand DS4's accepted comma-separated COMPSEC line/range keys."""
    result: set[int] = set()
    for item in key.split(","):
        value = item.strip(ASCII_WS)
        if re.fullmatch(r"[0-9]+(?:-[0-9]+)?", value) is None:
            raise ValueError(f"invalid COMPSEC key item: {value}")
        if "-" in value:
            first, last = map(int, value.split("-", maxsplit=1))
            if first > last:
                raise ValueError(f"descending COMPSEC key range: {value}")
            result.update(range(first, last + 1))
        else:
            result.add(int(value))
    if not result:
        raise ValueError("empty COMPSEC key")
    return result


def grade_c92(text: str, source: str, key: str, choice_count: int = 0) -> Grade:
    """Apply the fixed final-line grader for the declared C92 family."""
    answer = _final_answer_line(text)
    if answer is None:
        return Grade(False, None, False, "missing_or_malformed_final_answer_line")
    if source in {"GPQA Diamond", "SuperGPQA"}:
        if (
            re.fullmatch(r"[A-Z]", answer) is None
            or ord(answer) - ord("A") >= choice_count
        ):
            return Grade(False, answer, False, "invalid_option_letter")
        return Grade(True, answer, answer == key, None)
    if source == "AIME2025":
        if re.fullmatch(r"[0-9]+", answer) is None or int(answer) > 999:
            return Grade(False, answer, False, "invalid_aime_integer")
        return Grade(True, answer, int(answer) == int(key), None)
    if source == "COMPSEC":
        if re.fullmatch(r"[0-9]+(?:[ \t]*,[ \t]*[0-9]+)*", answer) is None:
            return Grade(False, answer, False, "invalid_compsec_line_list")
        numbers = [int(part.strip(ASCII_WS)) for part in answer.split(",")]
        if len(set(numbers)) != len(numbers):
            return Grade(False, answer, False, "duplicate_compsec_line")
        accepted = expand_compsec_key(key)
        if 0 in numbers and accepted != {0}:
            return Grade(True, answer, False, None)
        return Grade(True, answer, set(numbers) <= accepted, None)
    raise ValueError(f"unknown C92 source: {source}")
