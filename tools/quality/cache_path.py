"""Cache-reading NLL/generation probes. Decode path only; never all-prefill."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError

CAPACITY = 131072
SCORED_TOKENS = 32
GDN_DECODE_UPDATES = 2048
CACHE_DECODE_PATH = "compressed_cache_decode"
FORBIDDEN_PATH = "uncached_all_prefill"
PREFIXES: tuple[dict[str, Any], ...] = (
    {
        "id": "cache_8192",
        "prefix_tokens": 8192,
        "scored_or_generated": SCORED_TOKENS,
        "capacity": CAPACITY,
        "retrieval_positions": (16, 4096, 8000),
    },
    {
        "id": "cache_32768",
        "prefix_tokens": 32768,
        "scored_or_generated": SCORED_TOKENS,
        "capacity": CAPACITY,
        "retrieval_positions": (64, 16384, 32000),
    },
    {
        "id": "cache_131040_plus_32",
        "prefix_tokens": 131040,
        "scored_or_generated": SCORED_TOKENS,
        "capacity": CAPACITY,
        "retrieval_positions": (32, 65536, 120000, 131000),
    },
)
GDN_CHUNK_SPLITS: tuple[int, ...] = (512, 1024, 2048)
GRAPH_EAGER = "exact_equivalence_same_path"
PPL_RATIO_MAX = 1.01


def cache_contract() -> dict[str, Any]:
    return {
        "capacity": CAPACITY,
        "path": CACHE_DECODE_PATH,
        "forbidden_path": FORBIDDEN_PATH,
        "prefixes": [dict(row) for row in PREFIXES],
        "dispersed_retrieval": True,
        "recent_window_only": False,
        "ppl_ratio_max_vs_post113": PPL_RATIO_MAX,
        "pinned_llama_published_separately": True,
        "teacher_forced_and_free_running": True,
        "nll_must_use_actual_compressed_cache_decode": True,
    }


def gdn_stress_contract() -> dict[str, Any]:
    return {
        "decode_updates": GDN_DECODE_UPDATES,
        "chunk_splits": list(GDN_CHUNK_SPLITS),
        "prompt_to_decode": True,
        "graph_eager": GRAPH_EAGER,
        "cancellation": True,
        "divergent_prefix": True,
        "save_restore": True,
        "seeded_sampler_continuation": True,
        "matched_token_histories": True,
        "recurrence_nll": True,
        "finite_state": True,
        "free_running_stability": True,
        "record_numerical_drift_and_output_quality": True,
        "uses_actual_candidate_representation": True,
    }


def require_cache_path(record: Mapping[str, Any]) -> dict[str, Any]:
    path = str(record.get("path") or "")
    if path == FORBIDDEN_PATH or str(record.get("scoring_path") or "") == (
        FORBIDDEN_PATH
    ):
        raise QualityFrameworkError(
            "uncached all-prefill cannot validate cache quantization"
        )
    if path not in {"", CACHE_DECODE_PATH} and path != CACHE_DECODE_PATH:
        raise QualityFrameworkError(f"unknown cache scoring path {path}")
    if record.get("path") and path != CACHE_DECODE_PATH:
        raise QualityFrameworkError(
            "cache NLL must run the compressed-cache decode path"
        )
    if record.get("all_prefill") is True:
        raise QualityFrameworkError(
            "uncached all-prefill cannot validate cache quantization"
        )
    if "prefixes" not in record:
        prefixes = [dict(row) for row in PREFIXES]
    else:
        prefixes = list(record.get("prefixes") or [])
    ids = {str(row["id"]) for row in prefixes}
    expected = {str(row["id"]) for row in PREFIXES}
    missing = sorted(expected - ids)
    if missing:
        raise QualityFrameworkError("missing long-cache execution " + ",".join(missing))
    for row in prefixes:
        positions = list(row.get("retrieval_positions") or ())
        prefix = int(row["prefix_tokens"])
        if not positions:
            raise QualityFrameworkError(f"{row['id']}: retrieval positions required")
        if max(int(pos) for pos in positions) > prefix:
            raise QualityFrameworkError(f"{row['id']}: retrieval past prefix")
        if positions == [prefix - 1] or (
            len(positions) == 1 and int(positions[0]) >= prefix - 32
        ):
            raise QualityFrameworkError(
                f"{row['id']}: retrieval must not be only the recent window"
            )
        if int(row.get("capacity") or CAPACITY) != CAPACITY:
            raise QualityFrameworkError(f"{row['id']}: capacity must be {CAPACITY}")
        if int(row["prefix_tokens"]) + int(row["scored_or_generated"]) > CAPACITY:
            raise QualityFrameworkError(f"{row['id']}: exceeds capacity {CAPACITY}")
    return {
        "path": CACHE_DECODE_PATH,
        "pass": True,
        "prefixes": [dict(row) for row in prefixes],
        "forbidden_rejected": True,
    }


def refuse_fabricated_aggregate(
    spans: Sequence[Mapping[str, Any]],
    claimed: Mapping[str, Any],
) -> None:
    if not spans:
        raise QualityFrameworkError("fabricated aggregate: no cache spans")
    measured = [float(row["ppl_ratio"]) for row in spans if "ppl_ratio" in row]
    if claimed.get("ppl_ratio") is not None and measured:
        reported = float(claimed["ppl_ratio"])
        peak = max(measured)
        if (
            abs(reported - peak) > 1e-12
            and abs(reported - (sum(measured) / len(measured))) > 1e-12
        ):
            raise QualityFrameworkError("fabricated aggregate ppl_ratio")
    if claimed.get("missing_case_ignored") is True:
        raise QualityFrameworkError("fabricated aggregate: missing case ignored")
    if claimed.get("path") == FORBIDDEN_PATH:
        raise QualityFrameworkError(
            "uncached all-prefill cannot validate cache quantization"
        )


def synthetic_cache_span(
    *,
    span_id: str,
    prefix_tokens: int,
    scored: int,
    candidate_nll: float,
    control_nll: float,
    path: str = CACHE_DECODE_PATH,
    retrieval_positions: Sequence[int],
) -> dict[str, Any]:
    if path != CACHE_DECODE_PATH:
        raise QualityFrameworkError(
            "uncached all-prefill cannot validate cache quantization"
        )
    ratio = __import__("math").exp(float(candidate_nll) - float(control_nll))
    return {
        "id": span_id,
        "prefix_tokens": prefix_tokens,
        "scored_or_generated": scored,
        "capacity": CAPACITY,
        "path": path,
        "retrieval_positions": list(retrieval_positions),
        "candidate_mean_nll": float(candidate_nll),
        "post113_mean_nll": float(control_nll),
        "ppl_ratio": ratio,
        "ppl_ratio_max": PPL_RATIO_MAX,
        "pass": ratio <= PPL_RATIO_MAX,
        "finite": True,
        "all_prefill": False,
    }


def synthetic_gdn_stress(*, finite: bool = True, drift: float = 0.0) -> dict[str, Any]:
    if not finite:
        raise QualityFrameworkError("nonfinite GDN state")
    return {
        "decode_updates": GDN_DECODE_UPDATES,
        "chunk_splits": list(GDN_CHUNK_SPLITS),
        "matched_token_histories": True,
        "recurrence_incremental_nll": float(drift),
        "finite_state": True,
        "free_running_stable": True,
        "graph_eager_exact": True,
        "cancellation_exercised": True,
        "save_restore_exact": True,
        "divergent_prefix_exercised": True,
        "seeded_sampler_continuation_exact": True,
        "numerical_drift": float(drift),
        "output_quality_recorded": True,
        "pass": finite and abs(float(drift)) <= 0.02,
    }
