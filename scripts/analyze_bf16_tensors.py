#!/usr/bin/env python3
"""Exact BF16 payload statistics for the Qwen3.8-27B Transformers authority.

Streams every safetensor payload in 8 MiB chunks, accumulates a 65,536-bin
histogram plus optional directional reductions, and emits the TASK-05
analysis JSON object. Stdlib only; family mapping is imported from TASK-01.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import statistics
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_bf16_checkpoint import (  # noqa: E402
    AUTHORITY,
    BF16,
    LANGUAGE_LAYER_RE,
    LEVEL1_BY_LEVEL2,
    LEVEL2_IDS,
    MissingCheckpoint,
    _parse_safetensor_header,
    family_id_for,
    first_json_fence,
    inventory_checkpoint,
)

CHUNK_BYTES = 8 * 1024 * 1024
HIST_BINS = 65536
BF16_EXP_MASK = 0x7F80
BF16_MAX_FINITE_MAG = 0x7F7F
ZERO_NEG_U16 = 0x8000
INDEX_NAME = "model.safetensors.index.json"

EXPECTED_N_TENSORS = 1199
EXPECTED_N_SHARDS = 18
EXPECTED_N_PARAMETERS = 27_781_427_952
EXPECTED_N_BYTES = 55_562_855_904
EXPECTED_N_LANGUAGE_MTP_TENSORS = 866
EXPECTED_N_VISION_TENSORS = 333
EXPECTED_N_LANGUAGE_MTP_PARAMETERS = 27_320_697_856
EXPECTED_N_VISION_PARAMETERS = 460_730_096
EXPECTED_VISION_BLOCKS = 324

PAYLOAD_POLICY = "full_stream_exact"
BF16_DECODE = "u16_le_shift16_ieee_float32"
PERCENTILE_METHOD = "nearest_rank_ceil"
AXIS_CONVENTION = "d_out_d_in_transformers"
SCOPE_DETAIL = "language_mtp"
SCOPE_COARSE = "vision"

PERCENTILES: list[int | float] = [50, 90, 99, 99.9, 99.99]
OUTLIER_MULTIPLES: list[int] = [6, 10]
PERCENTILE_KEYS: tuple[tuple[int | float, str], ...] = (
    (50, "p50_abs"),
    (90, "p90_abs"),
    (99, "p99_abs"),
    (99.9, "p99_9_abs"),
    (99.99, "p99_99_abs"),
)

MAJOR_64: frozenset[str] = frozenset(
    {
        "input_layernorm",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
)
MAJOR_48: frozenset[str] = frozenset(
    {
        "linear_attn.A_log",
        "linear_attn.conv1d",
        "linear_attn.dt_bias",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.norm",
        "linear_attn.out_proj",
    }
)
MAJOR_16: frozenset[str] = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
    }
)
MAJOR_FAMILIES: frozenset[str] = MAJOR_64 | MAJOR_48 | MAJOR_16
MAJOR_RANK2: frozenset[str] = frozenset(
    {
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "linear_attn.conv1d",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    }
)

LAYERS_64: frozenset[int] = frozenset(range(64))
LAYERS_48: frozenset[int] = frozenset(i for i in range(64) if i % 4 != 3)
LAYERS_16: frozenset[int] = frozenset(i for i in range(64) if i % 4 == 3)

CANONICAL_SENTENCE = (
    "These measurements describe BF16 source-weight distributions. "
    "They are not quantization-bit, grouping, or quality recommendations."
)
REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Method",
    "Evidence policy",
    "Global distributions",
    "Family-level distributions",
    "Layer comparison",
    "Embeddings and lm_head",
    "Directional scale variation",
    "Outlier structure",
    "MTP",
    "Deferred vision",
    "Non-conclusions",
    "Machine-checkable analysis",
)
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
QUALITY_PHRASES: tuple[str, ...] = (
    "should be 4-bit",
    "should be 8-bit",
    "quantization winner",
    "recommend 4-bit",
    "recommend 8-bit",
)
BANNER_SUBSTRING = "unverified"

TOP_KEYS: tuple[str, ...] = (
    "authority",
    "n_tensors",
    "n_shards",
    "dtype",
    "n_parameters",
    "n_bytes",
    "payload_policy",
    "bf16_decode",
    "percentile_method",
    "percentiles",
    "outlier_multiples",
    "chunk_bytes",
    "axis_convention",
    "scope_detail",
    "scope_coarse",
    "n_language_mtp_tensors",
    "n_vision_tensors",
    "n_language_mtp_parameters",
    "n_vision_parameters",
    "pooled_all",
    "pooled_language_mtp",
    "pooled_vision",
    "families",
)
GLOBAL_STATS_KEYS: tuple[str, ...] = (
    "n",
    "min",
    "max",
    "absmax",
    "mean",
    "mean_abs",
    "rms",
    "std",
    "n_zero",
    "fraction_zero",
    "p50_abs",
    "p90_abs",
    "p99_abs",
    "p99_9_abs",
    "p99_99_abs",
    "n_out_6x",
    "frac_out_6x",
    "n_out_10x",
    "frac_out_10x",
    "n_nonfinite",
)
FAMILY_KEYS: tuple[str, ...] = (
    "level1",
    "n_tensors",
    "n_parameters",
    "detail",
    "pooled",
    "unweighted_mean_absmax",
    "unweighted_mean_rms",
    "weighted_mean_absmax",
    "weighted_mean_rms",
    "layer_comparison",
    "directional",
)
LAYER_COMPARISON_BASE_KEYS: tuple[str, ...] = (
    "n_layers",
    "layer_index",
    "absmax",
    "rms",
    "absmax_min",
    "absmax_min_layer",
    "absmax_median",
    "absmax_max",
    "absmax_max_layer",
    "rms_min",
    "rms_min_layer",
    "rms_median",
    "rms_max",
    "rms_max_layer",
)
LAYER_COMPARISON_DIR_KEYS: tuple[str, ...] = (
    "row_absmax_max_over_median",
    "col_absmax_max_over_median",
    "max_row_absmax_max_over_median",
    "max_row_absmax_max_over_median_layer",
    "max_col_absmax_max_over_median",
    "max_col_absmax_max_over_median_layer",
)
OCC1_DIRECTIONAL_KEYS: tuple[str, ...] = (
    "applicable",
    "n_rows",
    "n_cols",
    "row_absmax_min",
    "row_absmax_median",
    "row_absmax_max",
    "row_rms_min",
    "row_rms_median",
    "row_rms_max",
    "col_absmax_min",
    "col_absmax_median",
    "col_absmax_max",
    "col_rms_min",
    "col_rms_median",
    "col_rms_max",
    "row_absmax_max_over_median",
    "row_rms_max_over_median",
    "col_absmax_max_over_median",
    "col_rms_max_over_median",
)
MAJOR_DIRECTIONAL_KEYS: tuple[str, ...] = (
    "applicable",
    "n_rows",
    "n_cols",
    "unweighted_mean_row_absmax_max_over_median",
    "unweighted_mean_col_absmax_max_over_median",
    "unweighted_mean_row_rms_max_over_median",
    "unweighted_mean_col_rms_max_over_median",
)
GLOBAL_INT_KEYS: frozenset[str] = frozenset(
    {"n", "n_zero", "n_out_6x", "n_out_10x", "n_nonfinite"}
)


class AnalysisMismatch(Exception):
    """Raised when documented analysis markdown disagrees with a live run."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def _canon(value: float) -> float:
    """Return ``value`` rounded through the canonical JSON float format.

    Args:
        value: Finite Python float.

    Returns:
        ``float`` parsed from a 12-digit scientific representation.
    """

    return float(format(value, ".12e"))


def _canon_opt(value: float | None) -> float | None:
    """Canonicalize ``value``, preserving JSON ``null``.

    Args:
        value: Finite float or ``None``.

    Returns:
        Canonical float or ``None``.
    """

    if value is None:
        return None
    return _canon(value)


def _is_json_int(value: Any) -> bool:
    """Return True when ``value`` is a JSON integer (not bool or float)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _is_vision_family(family_id: str) -> bool:
    """Return True when ``family_id`` is a vision level-2 id."""

    return LEVEL1_BY_LEVEL2[family_id] == "vision"


def _build_lut() -> list[float]:
    """Build the 65,536-entry BF16→float64 lookup table.

    Returns:
        List ``lut`` where ``lut[u]`` is the IEEE binary32 value of BF16 bits
        ``u``, stored as a Python float.

    Raises:
        AssertionError: If known finite codes do not decode as expected.
    """

    lut = [0.0] * HIST_BINS
    pack_u32 = struct.pack
    unpack_f32 = struct.unpack
    for u in range(HIST_BINS):
        lut[u] = unpack_f32("<f", pack_u32("<I", u << 16))[0]
    assert lut[0] == 0.0
    assert lut[0x3F80] == 1.0
    assert lut[0x4000] == 2.0
    assert lut[0x3F00] == 0.5
    assert math.copysign(1.0, lut[ZERO_NEG_U16]) < 0.0
    return lut


def _zero_hist(hist: array.array) -> None:
    """Set every bin of ``hist`` to zero.

    Args:
        hist: Length-``HIST_BINS`` unsigned histogram.
    """

    for i in range(HIST_BINS):
        hist[i] = 0


def _add_hist(dst: array.array, src: array.array) -> None:
    """Add ``src`` counts into ``dst`` in place.

    Args:
        dst: Accumulator histogram.
        src: Histogram to merge.
    """

    for i in range(HIST_BINS):
        count = src[i]
        if count:
            dst[i] += count


def _hist_total(hist: array.array) -> int:
    """Return the sum of all histogram bins.

    Args:
        hist: Signed-BF16 histogram.

    Returns:
        Total element count stored in ``hist``.
    """

    total = 0
    for i in range(HIST_BINS):
        total += hist[i]
    return total


def _n_nonfinite(hist: array.array) -> int:
    """Count Inf/NaN BF16 codes (exponent 255) in ``hist``.

    Args:
        hist: Signed-BF16 histogram.

    Returns:
        Number of non-finite elements.
    """

    n = 0
    for mag in range(BF16_EXP_MASK, 0x8000):
        n += hist[mag]
        n += hist[mag | ZERO_NEG_U16]
    return n


def _ratio(maximum: float, median: float) -> float | None:
    """Return ``maximum / median`` with the locked zero-median rule.

    Args:
        maximum: Vector maximum.
        median: Vector median.

    Returns:
        Ratio, ``1.0`` when both are zero, or ``None`` when only median is zero.
    """

    if median == 0.0:
        if maximum == 0.0:
            return 1.0
        return None
    return maximum / median


def _mean_opt(values: list[float | None]) -> float | None:
    """Return the unweighted mean of non-null floats.

    Args:
        values: Ratio list that may contain ``None``.

    Returns:
        Mean of present values, or ``None`` when none are present.
    """

    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _percentile_abs(
    mag_hist: list[int],
    lut: list[float],
    n: int,
    p: int | float,
) -> float:
    """Nearest-rank |w| percentile from a magnitude histogram.

    Args:
        mag_hist: Counts for finite magnitude codes ``0 .. 0x7F7F``.
        lut: BF16 lookup table.
        n: Finite element count (equals ``numel`` after the non-finite assert).
        p: Percentile in ``(0, 100]``.

    Returns:
        Non-negative LUT value at the nearest-rank magnitude bin.

    Raises:
        AssertionError: If ``n`` is not positive or the walk does not fill.
    """

    assert n > 0, "percentile of empty tensor"
    rank = min(n, max(1, math.ceil(p / 100.0 * n)))
    cumulative = 0
    for mag in range(BF16_MAX_FINITE_MAG + 1):
        cumulative += mag_hist[mag]
        if cumulative >= rank:
            return lut[mag]
    raise AssertionError(f"percentile walk exhausted before rank {rank} of {n}")


def _global_stats(hist: array.array, lut: list[float]) -> dict[str, Any]:
    """Compute locked GlobalStats from a completed histogram.

    Args:
        hist: Signed 65,536-bin BF16 histogram.
        lut: Process-global BF16 LUT.

    Returns:
        GlobalStats object with canonical floats and integer counts.

    Raises:
        AssertionError: If any Inf/NaN codes are present or ``n`` is 0.
    """

    n = _hist_total(hist)
    n_nonfinite = _n_nonfinite(hist)
    assert n_nonfinite == 0, f"n_nonfinite={n_nonfinite} != 0"
    assert n > 0, "empty histogram"

    n_zero = hist[0] + hist[ZERO_NEG_U16]
    min_v: float | None = None
    max_v: float | None = None
    absmax = 0.0
    sum_w = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    mag_hist = [0] * (BF16_MAX_FINITE_MAG + 1)

    for u in range(HIST_BINS):
        count = hist[u]
        if not count:
            continue
        if (u & BF16_EXP_MASK) == BF16_EXP_MASK:
            continue
        v = lut[u]
        a = -v if v < 0.0 else v
        if min_v is None or v < min_v:
            min_v = v
        if max_v is None or v > max_v:
            max_v = v
        if a > absmax:
            absmax = a
        count_f = float(count)
        sum_w += v * count_f
        sum_abs += a * count_f
        sum_sq += (v * v) * count_f
        mag_hist[u & 0x7FFF] += count

    assert min_v is not None and max_v is not None
    mean = sum_w / n
    mean_abs = sum_abs / n
    rms = math.sqrt(sum_sq / n)
    std = math.sqrt(max(0.0, sum_sq / n - mean * mean))

    stats: dict[str, Any] = {
        "n": int(n),
        "min": _canon(min_v),
        "max": _canon(max_v),
        "absmax": _canon(absmax),
        "mean": _canon(mean),
        "mean_abs": _canon(mean_abs),
        "rms": _canon(rms),
        "std": _canon(std),
        "n_zero": int(n_zero),
        "fraction_zero": _canon(n_zero / n),
    }
    percentiles: dict[str, float] = {}
    for p, key in PERCENTILE_KEYS:
        percentiles[key] = _percentile_abs(mag_hist, lut, n, p)
        stats[key] = _canon(percentiles[key])

    p50 = percentiles["p50_abs"]
    n_out_6x = 0
    n_out_10x = 0
    thresh6 = 6.0 * p50
    thresh10 = 10.0 * p50
    for mag in range(BF16_MAX_FINITE_MAG + 1):
        count = mag_hist[mag]
        if not count:
            continue
        if p50 == 0.0:
            if mag != 0:
                n_out_6x += count
                n_out_10x += count
            continue
        a = lut[mag]
        if a > thresh6:
            n_out_6x += count
        if a > thresh10:
            n_out_10x += count

    stats["n_out_6x"] = int(n_out_6x)
    stats["frac_out_6x"] = _canon(n_out_6x / n)
    stats["n_out_10x"] = int(n_out_10x)
    stats["frac_out_10x"] = _canon(n_out_10x / n)
    stats["n_nonfinite"] = 0
    return stats


def _directional_axes(family_id: str, shape: list[int]) -> tuple[int, int] | None:
    """Return ``(d_out, d_in)`` when directional stats apply.

    Args:
        family_id: TASK-01 level-2 family id.
        shape: Tensor shape from the safetensor header.

    Returns:
        Row/column counts, or ``None`` when directional stats are not applicable.
    """

    if _is_vision_family(family_id):
        return None
    if family_id == "linear_attn.conv1d":
        assert len(shape) == 3, f"{family_id} shape {shape} is not rank-3"
        assert shape[1] == 1, f"{family_id} axis-1 {shape[1]} != 1"
        return int(shape[0]), int(shape[2])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None


def _vector_summary(values: list[float]) -> tuple[float, float, float]:
    """Return min, median, and max of a short float vector.

    Args:
        values: Row or column reduction vector.

    Returns:
        ``(min, median, max)`` using ``statistics.median``.

    Raises:
        AssertionError: If ``values`` is empty.
    """

    assert values, "empty directional vector"
    return min(values), float(statistics.median(values)), max(values)


def _directional_from_vectors(
    row_absmax: list[float],
    row_sumsq: array.array,
    col_absmax: list[float],
    col_sumsq: array.array,
    n_rows: int,
    n_cols: int,
) -> dict[str, Any]:
    """Build an occupancy-1 directional object from running vectors.

    Args:
        row_absmax: Per-row absolute maxima.
        row_sumsq: Per-row sum of squares.
        col_absmax: Per-column absolute maxima.
        col_sumsq: Per-column sum of squares.
        n_rows: ``d_out``.
        n_cols: ``d_in``.

    Returns:
        Directional object with ``applicable: true`` and canonical floats.
    """

    row_rms = [math.sqrt(row_sumsq[r] / n_cols) for r in range(n_rows)]
    col_rms = [math.sqrt(col_sumsq[c] / n_rows) for c in range(n_cols)]
    row_absmax_min, row_absmax_median, row_absmax_max = _vector_summary(row_absmax)
    row_rms_min, row_rms_median, row_rms_max = _vector_summary(row_rms)
    col_absmax_min, col_absmax_median, col_absmax_max = _vector_summary(col_absmax)
    col_rms_min, col_rms_median, col_rms_max = _vector_summary(col_rms)
    return {
        "applicable": True,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "row_absmax_min": _canon(row_absmax_min),
        "row_absmax_median": _canon(row_absmax_median),
        "row_absmax_max": _canon(row_absmax_max),
        "row_rms_min": _canon(row_rms_min),
        "row_rms_median": _canon(row_rms_median),
        "row_rms_max": _canon(row_rms_max),
        "col_absmax_min": _canon(col_absmax_min),
        "col_absmax_median": _canon(col_absmax_median),
        "col_absmax_max": _canon(col_absmax_max),
        "col_rms_min": _canon(col_rms_min),
        "col_rms_median": _canon(col_rms_median),
        "col_rms_max": _canon(col_rms_max),
        "row_absmax_max_over_median": _canon_opt(
            _ratio(row_absmax_max, row_absmax_median)
        ),
        "row_rms_max_over_median": _canon_opt(_ratio(row_rms_max, row_rms_median)),
        "col_absmax_max_over_median": _canon_opt(
            _ratio(col_absmax_max, col_absmax_median)
        ),
        "col_rms_max_over_median": _canon_opt(_ratio(col_rms_max, col_rms_median)),
    }


def _stream_histogram(
    handle: Any,
    n_bytes: int,
    hist: array.array,
) -> None:
    """Read ``n_bytes`` of BF16 payload into ``hist`` without directional work.

    Args:
        handle: Binary file positioned at the tensor payload start.
        n_bytes: Exact payload size in bytes.
        hist: Histogram to increment (not cleared).

    Raises:
        AssertionError: If the payload is truncated or not 2-byte aligned.
    """

    remaining = n_bytes
    read = handle.read
    chunk_bytes = CHUNK_BYTES
    while remaining:
        nread = chunk_bytes if remaining >= chunk_bytes else remaining
        chunk = read(nread)
        assert len(chunk) == nread, "truncated tensor payload"
        remaining -= nread
        assert nread % 2 == 0, "payload chunk is not 2-byte aligned"
        mv = memoryview(chunk).cast("H")
        for u in mv:
            hist[u] += 1
        mv.release()


def _stream_directional(
    handle: Any,
    n_bytes: int,
    hist: array.array,
    lut: list[float],
    row_absmax: list[float],
    row_sumsq: array.array,
    col_absmax: list[float],
    col_sumsq: array.array,
    n_rows: int,
    n_cols: int,
) -> None:
    """Read a rank-2 payload, filling histogram and row/column reductions.

    Args:
        handle: Binary file positioned at the tensor payload start.
        n_bytes: Exact payload size in bytes.
        hist: Histogram to increment (not cleared).
        lut: BF16 lookup table.
        row_absmax: Per-row absmax list (initialized to 0).
        row_sumsq: Per-row sum-of-squares array (initialized to 0).
        col_absmax: Per-column absmax list (initialized to 0).
        col_sumsq: Per-column sum-of-squares array (initialized to 0).
        n_rows: ``d_out``.
        n_cols: ``d_in``.

    Raises:
        AssertionError: On truncated payload, misalignment, or index overrun.
    """

    remaining = n_bytes
    read = handle.read
    chunk_bytes = CHUNK_BYTES
    row = 0
    col = 0
    d_in = n_cols
    while remaining:
        nread = chunk_bytes if remaining >= chunk_bytes else remaining
        chunk = read(nread)
        assert len(chunk) == nread, "truncated tensor payload"
        remaining -= nread
        assert nread % 2 == 0, "payload chunk is not 2-byte aligned"
        mv = memoryview(chunk).cast("H")
        for u in mv:
            hist[u] += 1
            v = lut[u]
            a = -v if v < 0.0 else v
            if a > row_absmax[row]:
                row_absmax[row] = a
            if a > col_absmax[col]:
                col_absmax[col] = a
            vv = v * v
            row_sumsq[row] += vv
            col_sumsq[col] += vv
            col += 1
            if col == d_in:
                col = 0
                row += 1
        mv.release()
    assert row == n_rows and col == 0, (
        f"directional walk ended at row={row} col={col} "
        f"expected row={n_rows} col=0"
    )


def _extrema(
    layers: list[int],
    values: list[float],
    *,
    want_max: bool,
) -> tuple[float, int]:
    """Return the extremum and the lowest layer index that attains it.

    Args:
        layers: Layer indices aligned with ``values``.
        values: Per-layer statistics.
        want_max: If True, return the maximum; otherwise the minimum.

    Returns:
        ``(extremum, layer_index)``.

    Raises:
        AssertionError: If the lists are empty or misaligned.
    """

    assert layers and len(layers) == len(values)
    best = values[0]
    best_layer = layers[0]
    for layer, value in zip(layers, values):
        if want_max:
            if value > best:
                best = value
                best_layer = layer
        elif value < best:
            best = value
            best_layer = layer
    return best, best_layer


def _max_ratio_layer(
    layers: list[int],
    ratios: list[float | None],
) -> tuple[float | None, int | None]:
    """Return the maximum ratio and the lowest layer that attains it.

    Args:
        layers: Layer indices aligned with ``ratios``.
        ratios: Per-layer max/median ratios (JSON ``null`` allowed).

    Returns:
        ``(max_ratio, layer)``, both ``None`` when every ratio is ``None``.
    """

    best: float | None = None
    best_layer: int | None = None
    for layer, ratio in zip(layers, ratios):
        if ratio is None:
            continue
        if best is None or ratio > best:
            best = ratio
            best_layer = layer
    return best, best_layer


def _empty_family_acc(family_id: str) -> dict[str, Any]:
    """Return a streaming accumulator for one level-2 family.

    Args:
        family_id: Level-2 identifier.

    Returns:
        Accumulator with a fresh histogram and empty record list.
    """

    return {
        "family_id": family_id,
        "level1": LEVEL1_BY_LEVEL2[family_id],
        "hist": array.array("Q", [0]) * HIST_BINS,
        "n_tensors": 0,
        "n_parameters": 0,
        "records": [],
    }


def _layer_index_for(name: str) -> int | None:
    """Return the language layer index encoded in ``name``, if any.

    Args:
        name: Full checkpoint tensor name.

    Returns:
        Integer layer index, or ``None`` when the name is not a language layer.
    """

    match = LANGUAGE_LAYER_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1))


def _build_layer_comparison(
    family_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the layer-comparison object for a major family.

    Args:
        family_id: Major-family level-2 id.
        records: Per-tensor records with ``layer_index``, ``absmax``, ``rms``.

    Returns:
        Layer-comparison mapping.

    Raises:
        AssertionError: If occupancy or layer-index sets disagree with TASK-01.
    """

    ordered = sorted(records, key=lambda rec: rec["layer_index"])
    layers = [int(rec["layer_index"]) for rec in ordered]
    absmax = [float(rec["absmax"]) for rec in ordered]
    rms = [float(rec["rms"]) for rec in ordered]
    layer_set = set(layers)
    n_layers = len(layers)
    if family_id in MAJOR_64:
        assert layer_set == LAYERS_64, f"{family_id} layers {sorted(layer_set)}"
        assert n_layers == 64
    elif family_id in MAJOR_48:
        assert layer_set == LAYERS_48, f"{family_id} layers {sorted(layer_set)}"
        assert n_layers == 48
    else:
        assert family_id in MAJOR_16
        assert layer_set == LAYERS_16, f"{family_id} layers {sorted(layer_set)}"
        assert n_layers == 16

    absmax_min, absmax_min_layer = _extrema(layers, absmax, want_max=False)
    absmax_max, absmax_max_layer = _extrema(layers, absmax, want_max=True)
    rms_min, rms_min_layer = _extrema(layers, rms, want_max=False)
    rms_max, rms_max_layer = _extrema(layers, rms, want_max=True)
    comparison: dict[str, Any] = {
        "n_layers": int(n_layers),
        "layer_index": layers,
        "absmax": [_canon(value) for value in absmax],
        "rms": [_canon(value) for value in rms],
        "absmax_min": _canon(absmax_min),
        "absmax_min_layer": int(absmax_min_layer),
        "absmax_median": _canon(float(statistics.median(absmax))),
        "absmax_max": _canon(absmax_max),
        "absmax_max_layer": int(absmax_max_layer),
        "rms_min": _canon(rms_min),
        "rms_min_layer": int(rms_min_layer),
        "rms_median": _canon(float(statistics.median(rms))),
        "rms_max": _canon(rms_max),
        "rms_max_layer": int(rms_max_layer),
    }
    if family_id in MAJOR_RANK2:
        row_ratios = [rec["directional"]["row_absmax_max_over_median"] for rec in ordered]
        col_ratios = [rec["directional"]["col_absmax_max_over_median"] for rec in ordered]
        max_row, max_row_layer = _max_ratio_layer(layers, row_ratios)
        max_col, max_col_layer = _max_ratio_layer(layers, col_ratios)
        comparison["row_absmax_max_over_median"] = [
            _canon_opt(value) for value in row_ratios
        ]
        comparison["col_absmax_max_over_median"] = [
            _canon_opt(value) for value in col_ratios
        ]
        comparison["max_row_absmax_max_over_median"] = _canon_opt(max_row)
        comparison["max_row_absmax_max_over_median_layer"] = (
            None if max_row_layer is None else int(max_row_layer)
        )
        comparison["max_col_absmax_max_over_median"] = _canon_opt(max_col)
        comparison["max_col_absmax_max_over_median_layer"] = (
            None if max_col_layer is None else int(max_col_layer)
        )
    return comparison


def _family_directional(
    family_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the family-level directional object.

    Args:
        family_id: Level-2 identifier.
        records: Per-tensor records for the family.

    Returns:
        Directional mapping (``applicable`` false, occupancy-1 full stats, or
        major-family unweighted-mean ratios).
    """

    if _is_vision_family(family_id):
        for rec in records:
            assert rec["directional"] is None
        return {"applicable": False}

    directional_records = [rec for rec in records if rec["directional"] is not None]
    if not directional_records:
        return {"applicable": False}

    if family_id in MAJOR_FAMILIES:
        assert family_id in MAJOR_RANK2
        n_rows = directional_records[0]["directional"]["n_rows"]
        n_cols = directional_records[0]["directional"]["n_cols"]
        for rec in directional_records:
            spec = rec["directional"]
            assert spec["applicable"] is True
            assert spec["n_rows"] == n_rows
            assert spec["n_cols"] == n_cols
        return {
            "applicable": True,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "unweighted_mean_row_absmax_max_over_median": _canon_opt(
                _mean_opt(
                    [
                        rec["directional"]["row_absmax_max_over_median"]
                        for rec in directional_records
                    ]
                )
            ),
            "unweighted_mean_col_absmax_max_over_median": _canon_opt(
                _mean_opt(
                    [
                        rec["directional"]["col_absmax_max_over_median"]
                        for rec in directional_records
                    ]
                )
            ),
            "unweighted_mean_row_rms_max_over_median": _canon_opt(
                _mean_opt(
                    [
                        rec["directional"]["row_rms_max_over_median"]
                        for rec in directional_records
                    ]
                )
            ),
            "unweighted_mean_col_rms_max_over_median": _canon_opt(
                _mean_opt(
                    [
                        rec["directional"]["col_rms_max_over_median"]
                        for rec in directional_records
                    ]
                )
            ),
        }

    assert len(records) == 1, (
        f"{family_id} occupancy {len(records)} is not 1 for full directional"
    )
    spec = records[0]["directional"]
    assert spec is not None and spec["applicable"] is True
    return spec


def dumps_analysis(analysis: dict[str, Any]) -> str:
    """Pretty-print the analysis JSON object.

    Args:
        analysis: Object produced by :func:`analyze_checkpoint`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(analysis, indent=2) + "\n"


def _shard_names(checkpoint: Path) -> list[str]:
    """Return sorted unique shard filenames from the safetensor index.

    Args:
        checkpoint: Checkpoint directory.

    Returns:
        Sorted shard file names.

    Raises:
        MissingCheckpoint: If the index file is absent.
        AssertionError: If the index JSON is malformed.
    """

    index_path = checkpoint / INDEX_NAME
    if not index_path.is_file():
        raise MissingCheckpoint(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert isinstance(index, dict), "index JSON is not an object"
    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict), "weight_map is not an object"
    return sorted(set(weight_map.values()))


def analyze_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Stream every BF16 payload and return the TASK-05 analysis object.

    Args:
        checkpoint: Directory containing config, index, and 18 shards.

    Returns:
        Analysis JSON object with locked key order.

    Raises:
        MissingCheckpoint: If the directory, index, config, or a shard is absent.
        AssertionError: If inventory, dtype, occupancy, or non-finite checks fail.
    """

    assert sys.byteorder == "little", f"byteorder {sys.byteorder!r} is not little"
    inventory = inventory_checkpoint(checkpoint)
    assert inventory["authority"] == AUTHORITY
    assert inventory["n_tensors"] == EXPECTED_N_TENSORS
    assert inventory["n_shards"] == EXPECTED_N_SHARDS
    assert inventory["dtype"] == BF16
    assert inventory["n_parameters"] == EXPECTED_N_PARAMETERS
    assert inventory["n_bytes"] == EXPECTED_N_BYTES

    lut = _build_lut()
    hist = array.array("Q", [0]) * HIST_BINS
    hist_all = array.array("Q", [0]) * HIST_BINS
    hist_language_mtp = array.array("Q", [0]) * HIST_BINS
    hist_vision = array.array("Q", [0]) * HIST_BINS
    families = {family_id: _empty_family_acc(family_id) for family_id in LEVEL2_IDS}

    shard_names = _shard_names(checkpoint)
    assert len(shard_names) == EXPECTED_N_SHARDS
    n_tensors_seen = 0
    n_parameters_seen = 0
    n_bytes_seen = 0

    for shard_name in shard_names:
        shard_path = checkpoint / shard_name
        if not shard_path.is_file():
            raise MissingCheckpoint(shard_path)
        print(f"shard {shard_name}", file=sys.stderr, flush=True)
        header = _parse_safetensor_header(shard_path)
        header.pop("__metadata__", None)
        with shard_path.open("rb") as handle:
            raw_len = handle.read(8)
            assert len(raw_len) == 8, f"{shard_path}: truncated header length"
            header_len = struct.unpack("<Q", raw_len)[0]
            payload_base = 8 + header_len
            items = []
            for name, spec in header.items():
                assert isinstance(spec, dict), f"{name}: header entry is not an object"
                offsets = spec["data_offsets"]
                start = int(offsets[0])
                end = int(offsets[1])
                items.append((start, end, name, spec))
            items.sort(key=lambda item: item[0])
            for start, end, name, spec in items:
                dtype = spec["dtype"]
                shape = [int(dim) for dim in spec["shape"]]
                n_parameters = 1
                for dim in shape:
                    n_parameters *= dim
                n_bytes = end - start
                assert dtype == BF16, f"{name}: dtype {dtype!r} is not {BF16}"
                assert n_bytes == n_parameters * 2, (
                    f"{name}: byte size {n_bytes} != numel {n_parameters} * 2"
                )
                family_id = family_id_for(name)
                n_tensors_seen += 1
                print(
                    f"{n_tensors_seen}/{EXPECTED_N_TENSORS} {name}",
                    file=sys.stderr,
                    flush=True,
                )
                handle.seek(payload_base + start)
                _zero_hist(hist)
                axes = _directional_axes(family_id, shape)
                directional: dict[str, Any] | None
                if axes is None:
                    _stream_histogram(handle, n_bytes, hist)
                    directional = None
                else:
                    n_rows, n_cols = axes
                    row_absmax = [0.0] * n_rows
                    col_absmax = [0.0] * n_cols
                    row_sumsq = array.array("d", [0.0]) * n_rows
                    col_sumsq = array.array("d", [0.0]) * n_cols
                    _stream_directional(
                        handle,
                        n_bytes,
                        hist,
                        lut,
                        row_absmax,
                        row_sumsq,
                        col_absmax,
                        col_sumsq,
                        n_rows,
                        n_cols,
                    )
                    directional = _directional_from_vectors(
                        row_absmax,
                        row_sumsq,
                        col_absmax,
                        col_sumsq,
                        n_rows,
                        n_cols,
                    )
                assert _hist_total(hist) == n_parameters, (
                    f"{name}: histogram n {_hist_total(hist)} != numel {n_parameters}"
                )
                assert _n_nonfinite(hist) == 0, f"{name}: non-finite BF16 values"
                absmax_raw, rms_raw = _raw_absmax_rms(hist, lut)
                acc = families[family_id]
                acc["n_tensors"] += 1
                acc["n_parameters"] += n_parameters
                _add_hist(acc["hist"], hist)
                _add_hist(hist_all, hist)
                if _is_vision_family(family_id):
                    _add_hist(hist_vision, hist)
                    assert directional is None
                else:
                    _add_hist(hist_language_mtp, hist)
                acc["records"].append(
                    {
                        "name": name,
                        "layer_index": _layer_index_for(name),
                        "n": n_parameters,
                        "absmax": absmax_raw,
                        "rms": rms_raw,
                        "directional": directional,
                    }
                )
                n_parameters_seen += n_parameters
                n_bytes_seen += n_bytes

    assert n_tensors_seen == EXPECTED_N_TENSORS
    assert n_parameters_seen == EXPECTED_N_PARAMETERS
    assert n_bytes_seen == EXPECTED_N_BYTES
    assert _hist_total(hist_all) == EXPECTED_N_PARAMETERS
    assert _hist_total(hist_language_mtp) == EXPECTED_N_LANGUAGE_MTP_PARAMETERS
    assert _hist_total(hist_vision) == EXPECTED_N_VISION_PARAMETERS

    _assert_family_occupancy(families, inventory)
    families_json = _build_families_json(families, lut, inventory)

    n_language_mtp_tensors = sum(
        families[family_id]["n_tensors"]
        for family_id in LEVEL2_IDS
        if not _is_vision_family(family_id)
    )
    n_vision_tensors = sum(
        families[family_id]["n_tensors"]
        for family_id in LEVEL2_IDS
        if _is_vision_family(family_id)
    )
    n_language_mtp_parameters = sum(
        families[family_id]["n_parameters"]
        for family_id in LEVEL2_IDS
        if not _is_vision_family(family_id)
    )
    n_vision_parameters = sum(
        families[family_id]["n_parameters"]
        for family_id in LEVEL2_IDS
        if _is_vision_family(family_id)
    )
    assert n_language_mtp_tensors == EXPECTED_N_LANGUAGE_MTP_TENSORS
    assert n_vision_tensors == EXPECTED_N_VISION_TENSORS
    assert n_language_mtp_tensors + n_vision_tensors == EXPECTED_N_TENSORS
    assert n_language_mtp_parameters == EXPECTED_N_LANGUAGE_MTP_PARAMETERS
    assert n_vision_parameters == EXPECTED_N_VISION_PARAMETERS
    assert n_language_mtp_parameters + n_vision_parameters == EXPECTED_N_PARAMETERS

    return {
        "authority": AUTHORITY,
        "n_tensors": int(EXPECTED_N_TENSORS),
        "n_shards": int(EXPECTED_N_SHARDS),
        "dtype": BF16,
        "n_parameters": int(EXPECTED_N_PARAMETERS),
        "n_bytes": int(EXPECTED_N_BYTES),
        "payload_policy": PAYLOAD_POLICY,
        "bf16_decode": BF16_DECODE,
        "percentile_method": PERCENTILE_METHOD,
        "percentiles": list(PERCENTILES),
        "outlier_multiples": list(OUTLIER_MULTIPLES),
        "chunk_bytes": int(CHUNK_BYTES),
        "axis_convention": AXIS_CONVENTION,
        "scope_detail": SCOPE_DETAIL,
        "scope_coarse": SCOPE_COARSE,
        "n_language_mtp_tensors": int(n_language_mtp_tensors),
        "n_vision_tensors": int(n_vision_tensors),
        "n_language_mtp_parameters": int(n_language_mtp_parameters),
        "n_vision_parameters": int(n_vision_parameters),
        "pooled_all": _global_stats(hist_all, lut),
        "pooled_language_mtp": _global_stats(hist_language_mtp, lut),
        "pooled_vision": _global_stats(hist_vision, lut),
        "families": families_json,
    }


def _raw_absmax_rms(hist: array.array, lut: list[float]) -> tuple[float, float]:
    """Return pre-canonical absmax and rms from ``hist``.

    Args:
        hist: Completed family or tensor histogram.
        lut: BF16 lookup table.

    Returns:
        ``(absmax, rms)`` before canonical rounding.
    """

    n = _hist_total(hist)
    absmax = 0.0
    sum_sq = 0.0
    for u in range(HIST_BINS):
        count = hist[u]
        if not count:
            continue
        if (u & BF16_EXP_MASK) == BF16_EXP_MASK:
            continue
        v = lut[u]
        a = -v if v < 0.0 else v
        if a > absmax:
            absmax = a
        sum_sq += (v * v) * float(count)
    return absmax, math.sqrt(sum_sq / n)


def _assert_family_occupancy(
    families: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    """Assert live occupancy against TASK-01 inventory and locked counts.

    Args:
        families: Streaming family accumulators.
        inventory: Header inventory object.

    Raises:
        AssertionError: On occupancy or parameter-count mismatch.
    """

    vision_total = 0
    for family_id in LEVEL2_IDS:
        acc = families[family_id]
        n = acc["n_tensors"]
        inv = inventory["families"][family_id]
        assert n == inv["n_tensors"], (
            f"{family_id} n_tensors {n} != inventory {inv['n_tensors']}"
        )
        assert acc["n_parameters"] == inv["n_parameters"], (
            f"{family_id} n_parameters {acc['n_parameters']} != "
            f"inventory {inv['n_parameters']}"
        )
        if family_id.startswith("linear_attn."):
            assert n == 48, f"{family_id} occupancy {n} != 48"
        elif family_id.startswith("self_attn."):
            assert n == 16, f"{family_id} occupancy {n} != 16"
        elif family_id.startswith("mlp.") or family_id in {
            "input_layernorm",
            "post_attention_layernorm",
        }:
            assert n == 64, f"{family_id} occupancy {n} != 64"
        elif family_id.startswith("mtp."):
            assert n == 1, f"{family_id} occupancy {n} != 1"
        elif family_id in {"embed", "final_norm", "lm_head"}:
            assert n == 1, f"{family_id} occupancy {n} != 1"
        if _is_vision_family(family_id):
            vision_total += n
        if family_id == "vision.blocks":
            assert n == EXPECTED_VISION_BLOCKS, (
                f"vision.blocks occupancy {n} != {EXPECTED_VISION_BLOCKS}"
            )
    assert vision_total == EXPECTED_N_VISION_TENSORS


def _build_families_json(
    families: dict[str, Any],
    lut: list[float],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the ``families`` object in ``LEVEL2_IDS`` order.

    Args:
        families: Streaming accumulators.
        lut: BF16 lookup table.
        inventory: Header inventory (parameter/occupancy authority).

    Returns:
        Mapping from every level-2 id to its family JSON object.
    """

    out: dict[str, Any] = {}
    for family_id in LEVEL2_IDS:
        acc = families[family_id]
        records = acc["records"]
        n_tensors = int(acc["n_tensors"])
        n_parameters = int(acc["n_parameters"])
        detail = "coarse" if _is_vision_family(family_id) else "full"
        pooled = _global_stats(acc["hist"], lut)
        absmax_list = [float(rec["absmax"]) for rec in records]
        rms_list = [float(rec["rms"]) for rec in records]
        unweighted_mean_absmax = sum(absmax_list) / n_tensors
        unweighted_mean_rms = sum(rms_list) / n_tensors
        weighted_mean_absmax = (
            sum(rec["absmax"] * rec["n"] for rec in records) / n_parameters
        )
        weighted_mean_rms = (
            sum(rec["rms"] * rec["n"] for rec in records) / n_parameters
        )
        if family_id in MAJOR_FAMILIES:
            layer_comparison = _build_layer_comparison(family_id, records)
        else:
            layer_comparison = None
            if not _is_vision_family(family_id):
                assert n_tensors == 1, (
                    f"{family_id} occupancy {n_tensors} expected 1 "
                    "when layer_comparison is null"
                )
        directional = _family_directional(family_id, records)
        applicable = directional.get("applicable") is True
        if _is_vision_family(family_id):
            assert directional == {"applicable": False}, family_id
        elif family_id in MAJOR_RANK2:
            assert applicable, family_id
        elif family_id in MAJOR_FAMILIES:
            assert directional == {"applicable": False}, family_id
        elif records[0]["directional"] is None:
            assert directional == {"applicable": False}, family_id
        else:
            assert applicable, family_id

        out[family_id] = {
            "level1": acc["level1"],
            "n_tensors": n_tensors,
            "n_parameters": n_parameters,
            "detail": detail,
            "pooled": pooled,
            "unweighted_mean_absmax": _canon(unweighted_mean_absmax),
            "unweighted_mean_rms": _canon(unweighted_mean_rms),
            "weighted_mean_absmax": _canon(weighted_mean_absmax),
            "weighted_mean_rms": _canon(weighted_mean_rms),
            "layer_comparison": layer_comparison,
            "directional": directional,
        }
        inv = inventory["families"][family_id]
        assert out[family_id]["n_tensors"] == inv["n_tensors"]
        assert out[family_id]["n_parameters"] == inv["n_parameters"]
        assert out[family_id]["level1"] == inv["level1"]
    assert list(out) == list(LEVEL2_IDS)
    return out


def _diff_values(path: str, live: Any, documented: Any) -> list[str]:
    """Recursively diff ``documented`` against ``live``.

    Args:
        path: Dotted path for error messages.
        live: Live analysis value.
        documented: Documented value from the JSON fence.

    Returns:
        Human-readable difference strings.
    """

    diffs: list[str] = []
    if isinstance(live, dict):
        if not isinstance(documented, dict):
            diffs.append(f"{path}: documented is {type(documented).__name__}, not object")
            return diffs
        live_keys = list(live)
        doc_keys = list(documented)
        if doc_keys != live_keys:
            extra = [key for key in doc_keys if key not in live]
            missing = [key for key in live_keys if key not in documented]
            if extra:
                diffs.append(f"{path} extra keys: {extra}")
            if missing:
                diffs.append(f"{path} missing keys: {missing}")
            if not extra and not missing:
                diffs.append(f"{path} key order {doc_keys} != {live_keys}")
        for key in live_keys:
            if key not in documented:
                continue
            diffs.extend(_diff_values(f"{path}.{key}", live[key], documented[key]))
        return diffs
    if isinstance(live, list):
        if not isinstance(documented, list):
            diffs.append(f"{path}: documented is {type(documented).__name__}, not array")
            return diffs
        if len(documented) != len(live):
            diffs.append(f"{path}: length documented={len(documented)} live={len(live)}")
            return diffs
        for index, (live_item, doc_item) in enumerate(zip(live, documented)):
            diffs.extend(_diff_values(f"{path}[{index}]", live_item, doc_item))
        return diffs
    if live != documented:
        diffs.append(f"{path}: documented={documented!r} live={live!r}")
    return diffs


def diff_analysis(live: dict[str, Any], documented: dict[str, Any]) -> list[str]:
    """Compare documented analysis JSON against a live object.

    Args:
        live: Fresh script analysis.
        documented: Object parsed from a markdown JSON fence.

    Returns:
        Human-readable difference strings; empty when the documents match.
    """

    diffs = _diff_values("$", live, documented)
    for key in GLOBAL_INT_KEYS:
        pooled = documented.get("pooled_all")
        if isinstance(pooled, dict) and key in pooled and not _is_json_int(pooled[key]):
            if key in GLOBAL_INT_KEYS:
                diffs.append(f"pooled_all.{key}: expected int, got {pooled[key]!r}")
    return diffs


def check_analysis(live: dict[str, Any], analysis_path: Path) -> None:
    """Check a markdown analysis file against a live full-payload object.

    Args:
        live: Fresh analysis object.
        analysis_path: Markdown path containing the JSON fence.

    Raises:
        AnalysisMismatch: On heading, banner, JSON, or phrase mismatches.
        OSError: If the file cannot be read.
    """

    text = analysis_path.read_text(encoding="utf-8")
    differences: list[str] = []

    headings = HEADING_RE.findall(text)
    if headings != list(REQUIRED_HEADINGS):
        differences.append(
            "heading mismatch:\n"
            f"  documented: {headings}\n"
            f"  required:   {list(REQUIRED_HEADINGS)}"
        )

    if BANNER_SUBSTRING not in text:
        differences.append("draft unverified banner substring not present")

    if CANONICAL_SENTENCE not in text:
        differences.append("canonical sentence not present verbatim")

    try:
        documented = first_json_fence(text)
    except (AssertionError, json.JSONDecodeError) as exc:
        differences.append(f"json fence: {exc}")
        documented = None

    if documented is not None:
        differences.extend(diff_analysis(live, documented))

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: forbidden token {match.group(0)!r}")

    for phrase in QUALITY_PHRASES:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            line = text.count("\n", 0, index) + 1
            differences.append(f"line {line}: forbidden quality phrase {phrase!r}")
            start = index + len(phrase)

    if differences:
        raise AnalysisMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the BF16 tensor analyzer.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Stream Qwen3.8-27B BF16 Transformers checkpoint payloads and "
            "emit exact family and layer distribution statistics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory (config.json, index, and 18 shards).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write analysis JSON to stdout (default when not checking).",
    )
    parser.add_argument(
        "--check-analysis",
        type=Path,
        default=None,
        dest="check_analysis",
        help="Markdown path whose first json fence must match a live analysis.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the BF16 tensor-analysis CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing path.
    """

    args = _parse_args(argv)
    try:
        analysis = analyze_checkpoint(args.checkpoint)
        if args.check_analysis is not None:
            if not args.check_analysis.is_file():
                print(
                    f"analysis file not found: {args.check_analysis}",
                    file=sys.stderr,
                )
                return 1
            check_analysis(analysis, args.check_analysis)
    except MissingCheckpoint as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except AnalysisMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"assert failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        missing = Path(getattr(exc, "filename", args.checkpoint))
        print(str(missing), file=sys.stderr)
        return 2

    if args.json or args.check_analysis is None:
        sys.stdout.write(dumps_analysis(analysis))
    return 0


if __name__ == "__main__":
    sys.exit(main())
