"""Baseline-blinded 0-2 rubric for non-unique prose answers."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError

DIMENSIONS: tuple[str, ...] = (
    "instruction_compliance",
    "factual_consistency",
    "coherence_completion",
    "repetition",
)
SCALE = (0, 1, 2)


def rubric_spec() -> dict[str, Any]:
    """Frozen before scoring. An implementation agent's quality=true is not a grader."""
    return {
        "blinded": True,
        "scale": list(SCALE),
        "dimensions": list(DIMENSIONS),
        "rules": {
            "instruction_compliance": {
                0: "ignores the asked format, speaker, length, or constraint",
                1: "partially follows the constraint with extra or missing pieces",
                2: "satisfies the asked constraint",
            },
            "factual_consistency": {
                0: "contradicts supplied context or invents a conflicting fact",
                1: "mostly consistent with a minor unsupported extra",
                2: "consistent with supplied context; no conflicting invention",
            },
            "coherence_completion": {
                0: "aborts, derails, or is unreadable",
                1: "readable but incomplete or weakly connected",
                2: "complete, locally coherent continuation",
            },
            "repetition": {
                0: "loops or restates the prior turn almost verbatim",
                1: "some echo but adds content",
                2: "no pathological repetition",
            },
        },
        "admission": {
            "no_new_zero_dimension": True,
            "aggregate_not_lower_than_post113": True,
        },
        "remote_judge_forbidden": True,
        "completed_review_required_before_promotion": True,
        "different_wording_allowed": True,
        "same_path_seeded_repeatability_exact": True,
        "grader_is_not_implementation_boolean": True,
    }


def _last_user(case: Mapping[str, Any]) -> str:
    turns = case.get("turns") or []
    for turn in reversed(list(turns)):
        if str(turn.get("role")) == "user":
            return str(turn.get("content") or "")
    return str(case.get("prompt") or "")


def _prior_assistant(case: Mapping[str, Any]) -> str:
    turns = case.get("turns") or []
    for turn in reversed(list(turns)):
        if str(turn.get("role")) == "assistant":
            return str(turn.get("content") or "")
    return ""


def score_prose(
    case: Mapping[str, Any],
    text: str,
    *,
    engine: str | None = None,
) -> dict[str, Any]:
    """Score one answer. Engine labels are attached after scoring for pairing."""
    del engine
    body = text.strip()
    user = _last_user(case).lower()
    prior = _prior_assistant(case)
    expected = str(case.get("expected") or "")
    words = [token for token in re.findall(r"\w+", body)]
    instruction = 2
    if not body:
        instruction = 0
    elif "prefix with noah" in user and not body.lower().startswith("noah"):
        instruction = 0
    elif "two sentences" in user:
        sentences = [part for part in re.split(r"[.!?]+", body) if part.strip()]
        instruction = 2 if 1 <= len(sentences) <= 3 else 1
    elif "one sentence" in user:
        sentences = [part for part in re.split(r"[.!?]+", body) if part.strip()]
        instruction = 2 if len(sentences) == 1 else 1
    elif "three words" in user:
        instruction = 2 if len(body.split()) <= 5 else 1

    factual = 2
    if expected and expected.lower() not in body.lower():
        factual = 0 if body else 0
        if body:
            factual = 1 if expected[:2].lower() in body.lower() else 0
    if "contradict" in body.lower() and expected:
        factual = 0

    coherence = 2
    if len(words) < 3:
        coherence = 0
    elif body.endswith((" and", " the", " a")):
        coherence = 1

    repetition = 2
    if prior and body.strip() == prior.strip():
        repetition = 0
    elif (
        prior and prior.strip() and prior.strip() in body and len(body) < len(prior) * 2
    ):
        repetition = 1
    tokens = body.lower().split()
    if len(tokens) >= 8 and len(set(tokens)) <= 2:
        repetition = 0

    dimensions = {
        "instruction_compliance": instruction,
        "factual_consistency": factual,
        "coherence_completion": coherence,
        "repetition": repetition,
    }
    for name, value in dimensions.items():
        if value not in SCALE:
            raise QualityFrameworkError(f"illegal rubric score {name}={value}")
    aggregate = sum(dimensions.values())
    return {
        "id": case["id"],
        "blinded": True,
        "dimensions": dimensions,
        "aggregate": aggregate,
        "max_aggregate": 2 * len(DIMENSIONS),
        "zero_dimensions": [name for name, value in dimensions.items() if value == 0],
        "raw_text": text,
        "reasons": {
            "instruction_compliance": f"score={instruction}",
            "factual_consistency": f"score={factual} expected={expected or 'n/a'}",
            "coherence_completion": f"score={coherence} words={len(words)}",
            "repetition": f"score={repetition}",
        },
    }


def compare_rubric(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    cand_dims = dict(candidate["dimensions"])
    base_dims = dict(baseline["dimensions"])
    new_zeros = [
        name
        for name in DIMENSIONS
        if int(cand_dims[name]) == 0 and int(base_dims[name]) != 0
    ]
    lower = int(candidate["aggregate"]) < int(baseline["aggregate"])
    return {
        "pass": not new_zeros and not lower,
        "new_zero_dimensions": new_zeros,
        "aggregate_candidate": int(candidate["aggregate"]),
        "aggregate_baseline": int(baseline["aggregate"]),
        "lower_aggregate": lower,
    }


def require_paired_raw(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
        if not row.get("raw_text") and row.get("raw_text") != "":
            raise QualityFrameworkError(
                f"missing raw paired answer for {row.get('id')}"
            )
        if not row.get("reasons"):
            raise QualityFrameworkError(f"missing rubric reasons for {row.get('id')}")
