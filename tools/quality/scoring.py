"""Teacher-forced continuation NLL, matching ds4 score_official semantics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def greedy_token(logits: Sequence[float]) -> int:
    """Argmax with lower-index tie-break, matching OPT-058 top_two."""
    if not logits:
        raise QualityFrameworkError("empty logits")
    best = 0
    best_value = float(logits[0])
    for index, value in enumerate(logits):
        number = float(value)
        if not math.isfinite(number):
            raise QualityFrameworkError(f"nonfinite logit at {index}")
        if number > best_value or (number == best_value and index < best):
            best = index
            best_value = number
    return best


def log_softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise QualityFrameworkError("empty logits")
    finite = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in finite):
        raise QualityFrameworkError("nonfinite logits")
    peak = max(finite)
    shifted = [value - peak for value in finite]
    log_z = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_z for value in shifted]


def teacher_forced_nll(
    logits_per_step: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    case_id: str,
    engine: str,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    """Score a known continuation token-by-token. Lower NLL is better."""
    if len(logits_per_step) != len(targets):
        raise QualityFrameworkError(
            f"{case_id}: logits steps {len(logits_per_step)} != targets {len(targets)}"
        )
    if not targets:
        raise QualityFrameworkError(f"{case_id}: empty continuation")
    nll = 0.0
    steps: list[dict[str, Any]] = []
    greedy_lcp = 0
    still_matching = True
    first_match = False
    for position, (logits, target) in enumerate(
        zip(logits_per_step, targets, strict=True)
    ):
        if vocab_size is not None and len(logits) != vocab_size:
            raise QualityFrameworkError(
                f"{case_id}: incomplete vocabulary at {position}: "
                f"{len(logits)} != {vocab_size}"
            )
        if not 0 <= int(target) < len(logits):
            raise QualityFrameworkError(
                f"{case_id}: target {target} outside vocab {len(logits)}"
            )
        probs = log_softmax(logits)
        greedy = greedy_token(logits)
        log_probability = probs[int(target)]
        if not _finite(log_probability):
            raise QualityFrameworkError(
                f"{case_id}: nonfinite logprob at target token {position}"
            )
        nll += -log_probability
        if position == 0:
            first_match = greedy == int(target)
        if still_matching and greedy == int(target):
            greedy_lcp += 1
        else:
            still_matching = False
        runner = 1 if len(logits) > 1 else 0
        if runner == greedy:
            runner = 0 if greedy != 0 else min(1, len(logits) - 1)
        runner_logit = float(logits[runner]) if logits else None
        for index, value in enumerate(logits):
            number = float(value)
            if index == greedy:
                continue
            if number > runner_logit or (number == runner_logit and index < runner):
                runner = index
                runner_logit = number
        steps.append(
            {
                "position": position,
                "target_token": int(target),
                "log_probability": log_probability,
                "nll": -log_probability,
                "greedy_token": greedy,
                "greedy_logit": float(logits[greedy]),
                "runner_up_token": runner,
                "runner_up_logit": float(runner_logit),
                "margin": float(logits[greedy]) - float(runner_logit),
            }
        )
    mean = nll / len(targets)
    return {
        "id": case_id,
        "engine": engine,
        "scoring": "teacher_forced_nll",
        "target_tokens": len(targets),
        "nll": nll,
        "avg_nll": mean,
        "mean_nll": mean,
        "perplexity": math.exp(mean),
        "first_match": int(first_match),
        "greedy_lcp": greedy_lcp,
        "vocab_size": len(logits_per_step[0]),
        "finite": True,
        "steps": steps,
    }


def mean_nll_from_record(record: Mapping[str, Any]) -> float:
    if "mean_nll" in record:
        return float(record["mean_nll"])
    if "avg_nll" in record:
        return float(record["avg_nll"])
    steps = record.get("steps") or []
    if not steps:
        raise QualityFrameworkError("NLL record has no steps")
    total = 0.0
    for step in steps:
        total += -float(step["log_probability"])
    return total / len(steps)


def ppl_ratio(actual: Mapping[str, Any], authority_mean_nll: float) -> float:
    return math.exp(mean_nll_from_record(actual) - float(authority_mean_nll))


def recurrence_incremental_nll(
    short: Mapping[str, Any],
    long: Mapping[str, Any],
    short_authority: float,
    long_authority: float,
) -> float:
    return (mean_nll_from_record(long) - float(long_authority)) - (
        mean_nll_from_record(short) - float(short_authority)
    )
