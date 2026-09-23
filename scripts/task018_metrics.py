"""Reference metric math and deterministic paired bootstrap for EVAL-01."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

VOCAB_SIZE = 248_320
BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class PositionMetrics:
    """Store full-vocabulary teacher-forced metrics for one target position."""

    nll_reference: float
    nll_candidate: float
    kl_reference_to_candidate: float
    top1_agreement: bool
    top20_overlap_fraction: float


@dataclass(frozen=True)
class CaseMetric:
    """Store token-weighted and case-weighted paired evaluation values."""

    case_id: str
    family: str
    nll_reference_sum: float
    nll_candidate_sum: float
    target_count: int
    correct_reference: int
    correct_candidate: int


def _log_softmax_fp32(logits: np.ndarray) -> np.ndarray:
    """Compute stable full-vector log-softmax with FP32 operations."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("logits must be one finite FP32 vocabulary vector")
    maximum = np.max(values)
    shifted = np.subtract(values, maximum, dtype=np.float32)
    exponential = np.exp(shifted, dtype=np.float32)
    denominator = np.sum(exponential, dtype=np.float32)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("invalid log-softmax normalizer")
    log_denominator = np.log(denominator, dtype=np.float32)
    return np.subtract(shifted, log_denominator, dtype=np.float32)


def score_position(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    target_token: int,
    *,
    require_qw38_vocabulary: bool = True,
) -> PositionMetrics:
    """Score target NLL, full-vocabulary KL, and deterministic top-k overlap."""
    reference = np.asarray(reference_logits, dtype=np.float32)
    candidate = np.asarray(candidate_logits, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired logits must have equal one-dimensional shapes")
    if require_qw38_vocabulary and reference.size != VOCAB_SIZE:
        raise ValueError(f"logit vocabulary size {reference.size} != {VOCAB_SIZE}")
    if not 0 <= target_token < reference.size:
        raise ValueError("target token is outside the vocabulary")
    logp_reference = _log_softmax_fp32(reference)
    logp_candidate = _log_softmax_fp32(candidate)
    nll_reference = float(np.float64(-logp_reference[target_token]))
    nll_candidate = float(np.float64(-logp_candidate[target_token]))
    probability_reference = np.exp(logp_reference, dtype=np.float32)
    terms = np.multiply(
        probability_reference,
        np.subtract(logp_reference, logp_candidate, dtype=np.float32),
        dtype=np.float32,
    )
    kl = float(np.sum(terms, dtype=np.float64))
    ids = np.arange(reference.size, dtype=np.int64)
    reference_top20 = np.lexsort((ids, -reference))[:20]
    candidate_top20 = np.lexsort((ids, -candidate))[:20]
    top1_reference = int(reference_top20[0])
    top1_candidate = int(candidate_top20[0])
    overlap = len(set(reference_top20.tolist()) & set(candidate_top20.tolist()))
    return PositionMetrics(
        nll_reference=nll_reference,
        nll_candidate=nll_candidate,
        kl_reference_to_candidate=kl,
        top1_agreement=top1_reference == top1_candidate,
        top20_overlap_fraction=overlap / 20.0,
    )


def _draw_index(n: int, group: str, replicate: int, draw: int) -> int:
    """Draw one index with the policy's SHA-256 rejection sampler."""
    if n < 1:
        raise ValueError("cannot resample an empty case family")
    limit = ((1 << 64) // n) * n
    retry = 0
    while True:
        message = f"qw38-language-v1|{group}|{replicate}|{draw}|{retry}".encode()
        value = int.from_bytes(hashlib.sha256(message).digest()[:8], "little")
        if value < limit:
            return value % n
        retry += 1


def nearest_rank(values: Sequence[float], probability: float) -> float:
    """Select a nearest-rank quantile without interpolating or rounding."""
    if not values or not 0 < probability <= 1:
        raise ValueError("nearest-rank percentile needs values and p in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return float(ordered[rank - 1])


def paired_case_bootstrap(
    rows: Sequence[CaseMetric],
    *,
    metric_group: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Bootstrap token-weighted NLL delta and paired accuracy loss."""
    if replicates < 1 or not rows:
        raise ValueError("bootstrap needs cases and a positive replicate count")
    families: dict[str, list[CaseMetric]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.case_id in seen:
            raise ValueError(f"duplicate case id: {row.case_id}")
        seen.add(row.case_id)
        if row.target_count < 1:
            raise ValueError(f"case has no scored targets: {row.case_id}")
        families[row.family].append(row)
    for family in families.values():
        family.sort(key=lambda row: row.case_id)

    deltas: list[float] = []
    losses: list[float] = []
    for replicate in range(replicates):
        reference_sum = candidate_sum = 0.0
        token_count = 0
        reference_correct = candidate_correct = case_count = 0
        for family_name in sorted(families):
            family = families[family_name]
            n = len(family)
            for draw in range(n):
                row = family[
                    _draw_index(n, f"{metric_group}/{family_name}", replicate, draw)
                ]
                reference_sum += row.nll_reference_sum
                candidate_sum += row.nll_candidate_sum
                token_count += row.target_count
                reference_correct += row.correct_reference
                candidate_correct += row.correct_candidate
                case_count += 1
        deltas.append((candidate_sum - reference_sum) / token_count)
        losses.append((reference_correct - candidate_correct) / case_count)
    return {
        "nll_delta_low_95": nearest_rank(deltas, 0.025),
        "nll_delta_high_95": nearest_rank(deltas, 0.975),
        "accuracy_loss_low_95": nearest_rank(losses, 0.025),
        "accuracy_loss_high_95": nearest_rank(losses, 0.975),
        "replicates": float(replicates),
    }
