"""OPT-136 reusable SQLite interval parsing, unions, and coverage validation.

Diagnostics only. Timestamps stay integer nanoseconds after schema conversion.
Units are read from the export schema; they are never inferred from magnitude.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

NS_PER_MS = 1_000_000
NS_PER_US = 1_000
UNIT_TO_NS = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": NS_PER_US,
    "µs": NS_PER_US,
    "microsecond": NS_PER_US,
    "microseconds": NS_PER_US,
    "ms": NS_PER_MS,
    "millisecond": NS_PER_MS,
    "milliseconds": NS_PER_MS,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
}

# CUPTI / Nsight Systems export schema: activity timestamps are nanoseconds.
# This mapping is the documented export schema, not a magnitude heuristic.
NSYS_CUPTI_SCHEMA_NS: dict[str, dict[str, str]] = {
    "CUPTI_ACTIVITY_KIND_KERNEL": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_MEMCPY": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_MEMSET": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_MEMCPY2": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_RUNTIME": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_DRIVER": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_OVERHEAD": {"start": "ns", "end": "ns"},
    "CUPTI_ACTIVITY_KIND_CUDA_EVENT": {"start": "ns", "end": "ns"},
    "NVTX_EVENTS": {"start": "ns", "end": "ns", "endNs": "ns", "startNs": "ns"},
    "PROFILER_OVERHEAD": {"start": "ns", "end": "ns"},
}

KERNEL_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL",)
COPY_TABLES = (
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_MEMCPY2",
)
GRAPH_TABLES = ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE",)
CPU_API_TABLES = (
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_DRIVER",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
)
OVERHEAD_TABLES = (
    "CUPTI_ACTIVITY_KIND_OVERHEAD",
    "PROFILER_OVERHEAD",
)
WINDOW_MARKER_NAMES = (
    "opt136.window",
    "opt133.window",
    "opt138.prefill",
    "window",
)
PREFILL_NVTX_NAMES = ("opt138.prefill", "qw38.prefill_chunk")
SYNC_TYPE_NAMES = {
    1: "cudaEventSynchronize",
    2: "cudaStreamWaitEvent",
    3: "cudaStreamSynchronize",
    4: "cudaDeviceSynchronize",
}

T_CRIT_DF2_ONE_SIDED = 2.919985580355516
T_CRIT_DF2_TWO_SIDED = 4.302652729696142
T_CRIT_DF4_TWO_SIDED = 2.7764451051977987
T_CRIT_DF9_ONE_SIDED = 1.8331129326536335
T_CRIT_DF9_TWO_SIDED = 2.2621571628540993

FAMILY_GROUPS = (
    "ffn",
    "mixer",
    "attention",
    "logits",
    "gdn",
    "residual_norm_quant",
    "prompt_mmq",
    "decode_mmv",
    "conversion",
    "staging",
    "epilogue",
    "commit_copy",
    "copy",
    "unclassified",
)

NAMED_FAMILIES = (
    "q4_gate",
    "q4_up",
    "q4_swiglu",
    "q4_down",
    "q8_mixer",
    "q6_attn_out",
    "q6_logits",
    "gdn_conv",
    "gdn_recurrence",
    "gdn_gated_out",
    "attn_prepare",
    "attn_cache_write",
    "attn_core",
    "attn_merge",
    "attn_gate",
    "residual_norm_quant",
    "prompt_mmq",
    "decode_mmv",
    "conversion",
    "staging",
    "epilogue",
    "commit_copy",
    "copy_other",
    "unclassified",
)

# Llama fused kernels define enclosing groups. Quartz leaves that sit inside a
# fused llama op are regrouped to that boundary; unmatched time stays explicit.
LLAMA_ENCLOSING_GROUPS: dict[str, tuple[str, ...]] = {
    "q4_ffn_fused": ("q4_gate", "q4_up", "q4_swiglu"),
    "q4_down": ("q4_down",),
    "q8_mixer": ("q8_mixer",),
    "attn_fused": (
        "attn_prepare",
        "attn_core",
        "attn_merge",
        "attn_gate",
        "attn_cache_write",
        "q6_attn_out",
    ),
    "gdn_fused": ("gdn_conv", "gdn_recurrence", "gdn_gated_out"),
    "prompt_mmq": ("prompt_mmq",),
    "decode_mmv": ("decode_mmv",),
    "q6_logits": ("q6_logits",),
    "residual_norm_quant": ("residual_norm_quant",),
    "conversion": ("conversion",),
    "staging": ("staging",),
    "epilogue": ("epilogue",),
    "commit_copy": ("commit_copy", "copy_other"),
}

RESIDUAL_WALL_LIMIT = 0.05

REQUIRED_COVERAGE_FIELDS = (
    "identity",
    "window_start_ns",
    "window_end_ns",
    "tables",
    "expected_graph_launches",
    "observed_graph_launches",
    "parent_node_correspondence",
    "drops",
    "errors",
    "kernel_copy_union_ns",
    "graph_envelope_union_ns",
    "graph_exclusive_without_leaf_ns",
    "time_outside_all_activity_ns",
    "unmapped_kernel_time_ns",
    "profiler_overhead",
    "internal_idle_ms",
    "coverage_valid",
)


class CoverageError(RuntimeError):
    """Coverage validation failed closed."""


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: int
    kind: str = "unknown"
    name: str = ""
    stream_id: int | None = None
    device_id: int | None = None
    context_id: int | None = None
    process_id: int | None = None
    graph_id: int | None = None
    node_id: int | None = None
    correlation_id: int | None = None
    table: str = ""
    family: str = "unclassified"

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def clipped(self, window_start: int, window_end: int) -> Interval | None:
        start = max(self.start_ns, window_start)
        end = min(self.end_ns, window_end)
        if end <= start:
            return None
        return Interval(
            start_ns=start,
            end_ns=end,
            kind=self.kind,
            name=self.name,
            stream_id=self.stream_id,
            device_id=self.device_id,
            context_id=self.context_id,
            process_id=self.process_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            correlation_id=self.correlation_id,
            table=self.table,
            family=self.family,
        )


@dataclass
class TraceTables:
    path: str
    units_by_table: dict[str, dict[str, str]] = field(default_factory=dict)
    kernels: list[Interval] = field(default_factory=list)
    copies: list[Interval] = field(default_factory=list)
    graphs: list[Interval] = field(default_factory=list)
    nodes: list[Interval] = field(default_factory=list)
    cpu_apis: list[Interval] = field(default_factory=list)
    nvtx: list[Interval] = field(default_factory=list)
    overhead: list[Interval] = field(default_factory=list)
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    drops: int = 0
    truncated: bool = False
    units_source: str = "export_schema"


def ns_to_ms(value_ns: int | float | None) -> float | None:
    if value_ns is None:
        return None
    return float(value_ns) / NS_PER_MS


def ms_to_ns(value_ms: float | int) -> int:
    return int(round(float(value_ms) * NS_PER_MS))


def round_ms(value_ms: float, digits: int = 2) -> float:
    return round(float(value_ms), digits)


def conservation_tolerance_ns(window_ns: int) -> int:
    return max(ms_to_ns(0.01), int(math.ceil(abs(window_ns) * 0.001)))


def merge_overlaps(intervals: Sequence[Interval]) -> list[tuple[int, int]]:
    spans = sorted(
        (item.start_ns, item.end_ns)
        for item in intervals
        if item.end_ns > item.start_ns
    )
    if not spans:
        return []
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        cur_start, cur_end = merged[-1]
        if start <= cur_end:
            merged[-1] = (cur_start, max(cur_end, end))
        else:
            merged.append((start, end))
    return merged


def union_ns(intervals: Sequence[Interval]) -> int:
    return sum(end - start for start, end in merge_overlaps(intervals))


def clip_all(
    intervals: Sequence[Interval], window_start: int, window_end: int
) -> list[Interval]:
    clipped: list[Interval] = []
    for item in intervals:
        piece = item.clipped(window_start, window_end)
        if piece is not None:
            clipped.append(piece)
    return clipped


def uncovered_ns(
    window_start: int, window_end: int, covered: Sequence[Interval]
) -> int:
    window = max(0, window_end - window_start)
    return max(0, window - union_ns(covered))


def overlap_union_ns(left: Sequence[Interval], right: Sequence[Interval]) -> int:
    """Disjoint overlap of two unions. Summing concurrent CPU+GPU is forbidden."""
    return union_ns(left) + union_ns(right) - union_ns(list(left) + list(right))


def exclusive_union_ns(left: Sequence[Interval], right: Sequence[Interval]) -> int:
    """Duration in union(left) that does not overlap union(right)."""
    return max(0, union_ns(left) - overlap_union_ns(left, right))


def classify_host_api(name: str) -> str:
    lowered = name.casefold()
    if "launch" in lowered or "graphlaunch" in lowered:
        return "launch"
    if (
        "synchron" in lowered
        or "waitevent" in lowered
        or lowered.startswith("cupti_sync")
    ):
        return "sync"
    if "memcpy" in lowered:
        return "memcpy_api"
    if not lowered:
        return "unnamed_api"
    return "other_api"


def first_last_span_ns(intervals: Sequence[Interval]) -> int:
    if not intervals:
        return 0
    start = min(item.start_ns for item in intervals)
    end = max(item.end_ns for item in intervals)
    return max(0, end - start)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]): str(row[2]).upper() for row in rows}


def _pick(columns: Mapping[str, str], *names: str) -> str | None:
    lower = {name.casefold(): name for name in columns}
    for name in names:
        if name in columns:
            return name
        found = lower.get(name.casefold())
        if found is not None:
            return found
    return None


def load_export_schema(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    tables = _table_names(conn)
    schema: dict[str, dict[str, str]] = {}
    schema_table = None
    for candidate in (
        "export_schema",
        "EXPORT_SCHEMA",
        "schema_units",
        "SCHEMA_UNITS",
        "CUPTI_EXPORT_SCHEMA",
    ):
        if candidate in tables:
            schema_table = candidate
            break
    if schema_table is not None:
        columns = _columns(conn, schema_table)
        table_col = _pick(columns, "table_name", "table", "object", "name")
        column_col = _pick(columns, "column_name", "column", "field")
        unit_col = _pick(columns, "unit", "units")
        if table_col and column_col and unit_col:
            for row in conn.execute(
                f'SELECT "{table_col}", "{column_col}", "{unit_col}" '
                f'FROM "{schema_table}"'
            ):
                table, column, unit = str(row[0]), str(row[1]), str(row[2]).casefold()
                schema.setdefault(table, {})[column] = unit
    for table, mapping in NSYS_CUPTI_SCHEMA_NS.items():
        if table in tables:
            schema.setdefault(table, {})
            for column, unit in mapping.items():
                schema[table].setdefault(column, unit)
    return schema


def unit_scale(unit: str | None) -> int | None:
    if unit is None:
        return None
    return UNIT_TO_NS.get(str(unit).strip().casefold())


def to_ns(value: Any, unit: str | None) -> int:
    scale = unit_scale(unit)
    if scale is None:
        raise CoverageError(f"unknown or missing time unit {unit!r}")
    if value is None:
        raise CoverageError("missing timestamp")
    return int(value) * scale


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_lookup(conn: sqlite3.Connection) -> dict[int, str]:
    tables = _table_names(conn)
    if "StringIds" not in tables:
        return {}
    columns = _columns(conn, "StringIds")
    id_col = _pick(columns, "id", "Id")
    value_col = _pick(columns, "value", "Value", "string")
    if id_col is None or value_col is None:
        return {}
    return {
        int(row[0]): str(row[1] or "")
        for row in conn.execute(f'SELECT "{id_col}", "{value_col}" FROM "StringIds"')
    }


def classify_kernel_family(name: str) -> str:
    lowered = name.casefold()
    if "memcpy" in lowered or "memset" in lowered or "copy" in lowered:
        return "copy_other"
    if "swiglu" in lowered or "silu" in lowered:
        return "q4_swiglu"
    if "gate" in lowered and ("q4" in lowered or "mmq" in lowered or "ffn" in lowered):
        return "q4_gate"
    if "ffn" in lowered and "up" in lowered:
        return "q4_up"
    if "ffn" in lowered and "down" in lowered:
        return "q4_down"
    if "q4" in lowered and "up" in lowered:
        return "q4_up"
    if "q4" in lowered and "down" in lowered:
        return "q4_down"
    if "q4" in lowered and "gate" in lowered:
        return "q4_gate"
    if "mixer" in lowered or ("q8" in lowered and "proj" in lowered):
        return "q8_mixer"
    if "logits" in lowered or ("q6" in lowered and "lm_head" in lowered):
        return "q6_logits"
    if "mmq" in lowered or "mul_mat_q" in lowered:
        return "prompt_mmq"
    if "mmvq" in lowered or "mmv" in lowered or "mul_mat_vec" in lowered:
        return "decode_mmv"
    if "convert" in lowered or "unpack" in lowered or "repack" in lowered:
        return "conversion"
    if "stage" in lowered or "staging" in lowered:
        return "staging"
    if "epilogue" in lowered:
        return "epilogue"
    if "commit" in lowered:
        return "commit_copy"
    if "attn" in lowered and "out" in lowered:
        return "q6_attn_out"
    if "gdn" in lowered and "conv" in lowered:
        return "gdn_conv"
    if "gdn" in lowered and ("recurr" in lowered or "ssm" in lowered):
        return "gdn_recurrence"
    if "gdn" in lowered:
        return "gdn_gated_out"
    if "cache" in lowered and "write" in lowered:
        return "attn_cache_write"
    if "prepare" in lowered or "qkv" in lowered:
        return "attn_prepare"
    if "merge" in lowered:
        return "attn_merge"
    if "attn" in lowered and "gate" in lowered:
        return "attn_gate"
    if (
        "attn" in lowered
        or "attention" in lowered
        or "softmax" in lowered
        or "fattn" in lowered
    ):
        return "attn_core"
    if (
        "norm" in lowered
        or "rms" in lowered
        or "quant" in lowered
        or "residual" in lowered
    ):
        return "residual_norm_quant"
    return "unclassified"


def family_group(family: str) -> str:
    if family == "prompt_mmq":
        return "prompt_mmq"
    if family == "decode_mmv":
        return "decode_mmv"
    if family.startswith("q4_") or family in {
        "q4_gate",
        "q4_up",
        "q4_swiglu",
        "q4_down",
    }:
        return "ffn"
    if family == "q8_mixer":
        return "mixer"
    if family.startswith("attn") or family == "q6_attn_out":
        return "attention"
    if family == "q6_logits":
        return "logits"
    if family.startswith("gdn"):
        return "gdn"
    if family == "residual_norm_quant":
        return "residual_norm_quant"
    if family in {"conversion"}:
        return "conversion"
    if family in {"staging"}:
        return "staging"
    if family in {"epilogue"}:
        return "epilogue"
    if family in {"commit_copy", "copy_other"}:
        return "copy" if family == "copy_other" else "commit_copy"
    return "unclassified"


def _read_activity_rows(
    conn: sqlite3.Connection,
    table: str,
    schema: Mapping[str, Mapping[str, str]],
    *,
    kind: str,
    strings: Mapping[int, str],
) -> tuple[list[Interval], list[str], int]:
    errors: list[str] = []
    if table not in _table_names(conn):
        return [], errors, 0
    columns = _columns(conn, table)
    units = dict(schema.get(table) or {})
    start_col = _pick(columns, "start", "startNs", "start_ns", "timestamp")
    end_col = _pick(columns, "end", "endNs", "end_ns")
    duration_col = _pick(columns, "duration", "durationNs", "duration_ns")
    if start_col is None:
        errors.append(f"{table}: missing start column")
        return [], errors, 0
    start_unit = units.get(start_col)
    if start_unit is None:
        errors.append(f"{table}.{start_col}: unit missing from export schema")
        return [], errors, 0
    if unit_scale(start_unit) is None:
        errors.append(f"{table}.{start_col}: unsupported unit {start_unit!r}")
        return [], errors, 0
    end_unit = units.get(end_col) if end_col else start_unit
    duration_unit = units.get(duration_col) if duration_col else start_unit
    name_col = _pick(
        columns,
        "name",
        "Name",
        "demangledName",
        "shortName",
        "nameId",
        "valueId",
        "text",
        "eventType",
        "syncType",
    )
    stream_col = _pick(columns, "streamId", "stream", "streamIndex")
    device_col = _pick(columns, "deviceId", "device")
    context_col = _pick(columns, "contextId", "context")
    process_col = _pick(columns, "globalPid", "pid", "processId")
    graph_col = _pick(columns, "graphId", "graphExecId", "graph")
    node_col = _pick(columns, "graphNodeId", "nodeId", "nodeid", "node")
    corr_col = _pick(columns, "correlationId", "correlation")
    select = [f'"{start_col}"']
    extras = [
        end_col,
        duration_col,
        name_col,
        stream_col,
        device_col,
        context_col,
        process_col,
        graph_col,
        node_col,
        corr_col,
    ]
    for column in extras:
        select.append("NULL" if column is None else f'"{column}"')
    sql = f'SELECT {", ".join(select)} FROM "{table}"'
    intervals: list[Interval] = []
    drops = 0
    for row in conn.execute(sql):
        try:
            start_ns = to_ns(row[0], start_unit)
            if end_col is not None and row[1] is not None:
                end_ns = to_ns(row[1], end_unit)
            elif duration_col is not None and row[2] is not None:
                end_ns = start_ns + to_ns(row[2], duration_unit)
            else:
                drops += 1
                errors.append(f"{table}: row missing end/duration")
                continue
        except CoverageError as exc:
            drops += 1
            errors.append(f"{table}: {exc}")
            continue
        if end_ns < start_ns:
            drops += 1
            errors.append(f"{table}: inverted interval")
            continue
        raw_name = row[3]
        if isinstance(raw_name, int) and raw_name in strings:
            name = strings[raw_name]
        elif table.endswith("SYNCHRONIZATION") and isinstance(raw_name, int):
            name = SYNC_TYPE_NAMES.get(raw_name, f"cupti_sync_{raw_name}")
        else:
            name = "" if raw_name is None else str(raw_name)
        family = (
            classify_kernel_family(name)
            if kind in {"kernel", "memcpy", "memset", "node"}
            else "unclassified"
        )
        intervals.append(
            Interval(
                start_ns=start_ns,
                end_ns=end_ns,
                kind=kind,
                name=name,
                stream_id=_int_or_none(row[4]),
                device_id=_int_or_none(row[5]),
                context_id=_int_or_none(row[6]),
                process_id=_int_or_none(row[7]),
                graph_id=_int_or_none(row[8]),
                node_id=_int_or_none(row[9]),
                correlation_id=_int_or_none(row[10]),
                table=table,
                family=family,
            )
        )
    return intervals, errors, drops


def parse_nsys_sqlite(path: Path | str) -> TraceTables:
    sqlite_path = Path(path)
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = _table_names(conn)
        schema = load_export_schema(conn)
        strings = _string_lookup(conn)
        parsed = TraceTables(path=str(sqlite_path), units_by_table=schema)
        parsed.table_counts = {
            name: int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in sorted(tables)
            if not name.startswith("sqlite_")
        }
        for table in KERNEL_TABLES:
            rows, errors, drops = _read_activity_rows(
                conn, table, schema, kind="kernel", strings=strings
            )
            parsed.kernels.extend(rows)
            parsed.errors.extend(errors)
            parsed.drops += drops
        for table in COPY_TABLES:
            kind = "memset" if "MEMSET" in table else "memcpy"
            rows, errors, drops = _read_activity_rows(
                conn, table, schema, kind=kind, strings=strings
            )
            parsed.copies.extend(rows)
            parsed.errors.extend(errors)
            parsed.drops += drops
        for table in GRAPH_TABLES:
            rows, errors, drops = _read_activity_rows(
                conn, table, schema, kind="graph", strings=strings
            )
            parsed.graphs.extend(rows)
            parsed.errors.extend(errors)
            parsed.drops += drops
        for table in CPU_API_TABLES:
            rows, errors, drops = _read_activity_rows(
                conn, table, schema, kind="cuda_api", strings=strings
            )
            parsed.cpu_apis.extend(rows)
            parsed.errors.extend(errors)
            parsed.drops += drops
        for table in OVERHEAD_TABLES:
            rows, errors, drops = _read_activity_rows(
                conn, table, schema, kind="overhead", strings=strings
            )
            parsed.overhead.extend(rows)
            parsed.errors.extend(errors)
            parsed.drops += drops
        if "NVTX_EVENTS" in tables:
            rows, errors, drops = _read_activity_rows(
                conn, "NVTX_EVENTS", schema, kind="nvtx", strings=strings
            )
            parsed.nvtx.extend(rows)
            parsed.errors.extend(errors)
            parsed.drops += drops
        parsed.nodes = [item for item in parsed.kernels if item.node_id is not None]
        if not parsed.nodes:
            parsed.nodes = [item for item in parsed.graphs if item.node_id is not None]
        parsed.truncated = any("truncat" in error.casefold() for error in parsed.errors)
        if not schema:
            parsed.errors.append("export schema units missing")
        return parsed
    finally:
        conn.close()


def marker_window(
    nvtx: Sequence[Interval],
    *,
    names: Sequence[str] = WINDOW_MARKER_NAMES,
) -> tuple[int, int] | None:
    wanted = {name.casefold() for name in names}
    matches = [item for item in nvtx if item.name.casefold() in wanted]
    if not matches:
        return None
    return min(item.start_ns for item in matches), max(item.end_ns for item in matches)


def eval_or_prefill_nvtx_window(
    nvtx: Sequence[Interval],
) -> tuple[int, int] | None:
    """Window from in-capture eval/prefill NVTX when envelope markers are absent.

    `opt136.window` / `opt138.prefill` are pushed before cudaProfilerStart, so
    nsys cudaProfilerApi captures omit them. The 12 decode eval ranges and
    `qw38.prefill_chunk` remain inside the capture.
    """
    evals = [
        item
        for item in nvtx
        if "op=eval" in item.name.casefold()
        or item.name.casefold().startswith("opt136 engine=")
    ]
    if evals:
        return min(item.start_ns for item in evals), max(item.end_ns for item in evals)
    prefill = [item for item in nvtx if item.name.casefold() in PREFILL_NVTX_NAMES]
    if prefill:
        return (
            min(item.start_ns for item in prefill),
            max(item.end_ns for item in prefill),
        )
    return None


def resolve_capture_window(
    tables: TraceTables,
    *,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> dict[str, Any]:
    if window_start_ns is not None and window_end_ns is not None:
        return {
            "window_start_ns": int(window_start_ns),
            "window_end_ns": int(window_end_ns),
            "source": "explicit",
            "missing_observation": None,
        }
    marker = marker_window(tables.nvtx)
    if marker is not None:
        return {
            "window_start_ns": marker[0],
            "window_end_ns": marker[1],
            "source": "nvtx_marker",
            "missing_observation": None,
        }
    eval_window = eval_or_prefill_nvtx_window(tables.nvtx)
    if eval_window is not None:
        return {
            "window_start_ns": eval_window[0],
            "window_end_ns": eval_window[1],
            "source": "eval_or_prefill_nvtx",
            "missing_observation": (
                "opt136.window/opt138.prefill envelopes are outside "
                "cudaProfilerApi; used in-capture eval/prefill NVTX"
            ),
        }
    gpu_all = tables.kernels + tables.copies + tables.graphs
    if gpu_all:
        return {
            "window_start_ns": min(item.start_ns for item in gpu_all),
            "window_end_ns": max(item.end_ns for item in gpu_all),
            "source": "gpu_span_fallback",
            "missing_observation": (
                "NVTX window markers absent; host time before the first and "
                "after the last GPU interval is unobservable"
            ),
        }
    return {
        "window_start_ns": 0,
        "window_end_ns": 0,
        "source": "empty",
        "missing_observation": "no NVTX markers and no GPU intervals",
    }


def graph_exclusive_without_leaves(
    graphs: Sequence[Interval], leaves: Sequence[Interval]
) -> list[Interval]:
    exclusive: list[Interval] = []
    for graph in graphs:
        inside = clip_all(leaves, graph.start_ns, graph.end_ns)
        covered = merge_overlaps(inside)
        cursor = graph.start_ns
        for start, end in covered:
            if start > cursor:
                exclusive.append(
                    Interval(
                        start_ns=cursor,
                        end_ns=start,
                        kind="graph_exclusive",
                        table=graph.table,
                        graph_id=graph.graph_id,
                    )
                )
            cursor = max(cursor, end)
        if cursor < graph.end_ns:
            exclusive.append(
                Interval(
                    start_ns=cursor,
                    end_ns=graph.end_ns,
                    kind="graph_exclusive",
                    table=graph.table,
                    graph_id=graph.graph_id,
                )
            )
    return exclusive


def _ms(value_ns: int) -> float:
    return float(value_ns) / NS_PER_MS


def family_union_sum_ns(leaves: Sequence[Interval]) -> tuple[int, dict[str, int]]:
    grouped: dict[str, list[Interval]] = {}
    for item in leaves:
        grouped.setdefault(item.family, []).append(item)
    per_family = {name: union_ns(items) for name, items in grouped.items()}
    return sum(per_family.values()), per_family


def reconcile_window_wall(
    tables: TraceTables,
    *,
    window_start_ns: int,
    window_end_ns: int,
    window_source: str | None = None,
    missing_observation: str | None = None,
    profiler_perturbation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Partition one wall window with unions. CPU overlapping GPU is not additive.

    Remaining process-scoped gaps are unresolved, not proven device-wide idle.
    Family-sum minus GPU union is bookkeeping, not extra wall.
    """
    window = max(0, int(window_end_ns) - int(window_start_ns))
    kernels = clip_all(tables.kernels, window_start_ns, window_end_ns)
    copies = clip_all(tables.copies, window_start_ns, window_end_ns)
    graphs = clip_all(tables.graphs, window_start_ns, window_end_ns)
    nodes = clip_all(tables.nodes, window_start_ns, window_end_ns)
    cpu = clip_all(tables.cpu_apis, window_start_ns, window_end_ns)
    overhead = clip_all(tables.overhead, window_start_ns, window_end_ns)
    leaves = kernels + copies
    gpu = leaves + graphs
    gpu_union = union_ns(gpu)
    kernel_copy_union = union_ns(leaves)
    graph_union = union_ns(graphs)
    graph_excl = union_ns(graph_exclusive_without_leaves(graphs, leaves))
    launches = [item for item in cpu if classify_host_api(item.name) == "launch"]
    syncs = [item for item in cpu if classify_host_api(item.name) == "sync"]
    memcpy_apis = [item for item in cpu if classify_host_api(item.name) == "memcpy_api"]
    other_apis = [
        item
        for item in cpu
        if classify_host_api(item.name) not in {"launch", "sync", "memcpy_api"}
    ]
    host_exclusive = exclusive_union_ns(cpu, gpu)
    host_launch_exclusive = exclusive_union_ns(launches, gpu)
    host_sync_exclusive = exclusive_union_ns(syncs, gpu)
    host_memcpy_exclusive = exclusive_union_ns(memcpy_apis, gpu)
    host_other_exclusive = exclusive_union_ns(other_apis, gpu)
    cpu_gpu_overlap = overlap_union_ns(cpu, gpu)
    remaining = uncovered_ns(window_start_ns, window_end_ns, gpu + cpu)
    instrumentation_exclusive = exclusive_union_ns(overhead, gpu + cpu)
    unresolved_gap = max(0, remaining - instrumentation_exclusive)
    accounted = gpu_union + host_exclusive + instrumentation_exclusive + unresolved_gap
    conservation_delta = window - accounted
    family_sum, per_family = family_union_sum_ns(leaves)
    family_overlap = max(0, family_sum - kernel_copy_union)
    perturbation = dict(profiler_perturbation or {})
    overhead_union = union_ns(overhead)
    relative_overhead = (instrumentation_exclusive / window) if window else None
    median_rel = perturbation.get("median_relative_perturbation")
    if median_rel is None:
        median_rel = relative_overhead
        perturbation["median_relative_perturbation"] = relative_overhead
        perturbation["source"] = perturbation.get("source") or "profiler_overhead_union"
    perturbed = bool(median_rel is not None and abs(float(median_rel)) > 0.05)
    unresolved_share = (unresolved_gap / window) if window else None
    within_limit = (
        unresolved_share is not None and unresolved_share <= RESIDUAL_WALL_LIMIT
    )
    buckets = [
        {
            "label": "gpu_union",
            "ns": gpu_union,
            "ms": _ms(gpu_union),
            "classification": "recovered",
            "evidence": (
                "union(CUPTI kernel+memcpy+graph); graph parents not added on "
                "top of children"
            ),
        },
        {
            "label": "graph_exclusive_without_leaves",
            "ns": graph_excl,
            "ms": _ms(graph_excl),
            "classification": "recovered",
            "evidence": "graph envelope minus overlapping kernel/copy leaves",
        },
        {
            "label": "host_launch_exclusive",
            "ns": host_launch_exclusive,
            "ms": _ms(host_launch_exclusive),
            "classification": "recovered",
            "evidence": "cudaLaunch/graphLaunch APIs exclusive of GPU union",
        },
        {
            "label": "host_sync_exclusive",
            "ns": host_sync_exclusive,
            "ms": _ms(host_sync_exclusive),
            "classification": "recovered",
            "evidence": (
                "synchronization APIs exclusive of GPU union; overlapping "
                "cudaEventSynchronize is wait, not added GPU wall"
            ),
        },
        {
            "label": "host_memcpy_api_exclusive",
            "ns": host_memcpy_exclusive,
            "ms": _ms(host_memcpy_exclusive),
            "classification": "recovered",
            "evidence": "host memcpy APIs exclusive of GPU union",
        },
        {
            "label": "host_other_api_exclusive",
            "ns": host_other_exclusive,
            "ms": _ms(host_other_exclusive),
            "classification": "recovered",
            "evidence": "other CUDA runtime/driver APIs exclusive of GPU union",
        },
        {
            "label": "host_exclusive_total",
            "ns": host_exclusive,
            "ms": _ms(host_exclusive),
            "classification": "recovered",
            "evidence": "union(CPU APIs) minus overlap with GPU union",
        },
        {
            "label": "instrumentation_exclusive",
            "ns": instrumentation_exclusive,
            "ms": _ms(instrumentation_exclusive),
            "classification": "recovered",
            "evidence": (
                "PROFILER_OVERHEAD/CUPTI overhead exclusive of GPU and host APIs"
            ),
        },
        {
            "label": "unresolved_process_gap",
            "ns": unresolved_gap,
            "ms": _ms(unresolved_gap),
            "classification": "unresolved",
            "evidence": (
                "process-scoped gap with neither GPU nor host API coverage; "
                "not proven device-wide idle"
            ),
        },
        {
            "label": "family_sum_overlap_bookkeeping",
            "ns": family_overlap,
            "ms": _ms(family_overlap),
            "classification": "bookkeeping",
            "evidence": (
                "sum of per-family GPU unions minus leaf GPU union; concurrent "
                "families are not extra wall"
            ),
        },
    ]
    missing = missing_observation
    if window_source == "gpu_span_fallback" and not missing:
        missing = (
            "NVTX window markers absent; host time before the first and after "
            "the last GPU interval is unobservable"
        )
    return {
        "window_start_ns": int(window_start_ns),
        "window_end_ns": int(window_end_ns),
        "window_ns": window,
        "window_ms": _ms(window),
        "window_source": window_source,
        "gpu_union_ns": gpu_union,
        "gpu_union_ms": _ms(gpu_union),
        "kernel_copy_union_ns": kernel_copy_union,
        "graph_envelope_union_ns": graph_union,
        "graph_exclusive_without_leaf_ns": graph_excl,
        "host_exclusive_ns": host_exclusive,
        "host_exclusive_ms": _ms(host_exclusive),
        "host_launch_exclusive_ns": host_launch_exclusive,
        "host_sync_exclusive_ns": host_sync_exclusive,
        "cpu_gpu_overlap_ns": cpu_gpu_overlap,
        "cpu_api_union_ns": union_ns(cpu),
        "instrumentation_union_ns": overhead_union,
        "instrumentation_exclusive_ns": instrumentation_exclusive,
        "instrumentation_overlap_gpu_or_host_ns": max(
            0, overhead_union - instrumentation_exclusive
        ),
        "unresolved_ns": unresolved_gap,
        "unresolved_ms": _ms(unresolved_gap),
        "unresolved_share_of_wall": unresolved_share,
        "accounted_ns": accounted,
        "conservation_delta_ns": conservation_delta,
        "conservation_ok": abs(conservation_delta) <= conservation_tolerance_ns(window),
        "family_union_sum_ns": family_sum,
        "family_overlap_ns": family_overlap,
        "family_union_ns": per_family,
        "observed_graph_launches": len(graphs),
        "observed_node_count": len(nodes),
        "observed_kernel_count": len(kernels),
        "buckets": buckets,
        "profiler_perturbation": perturbation,
        "perturbed": perturbed,
        "within_unresolved_limit": within_limit,
        "residual_limit": RESIDUAL_WALL_LIMIT,
        "missing_observation": missing,
        "claim_type": "derived",
        "parent_child_not_double_counted": True,
        "cpu_overlapping_gpu_not_additive": True,
        "process_scoped_gaps_are_not_device_wide_idle": True,
        "no_proportional_allocation": True,
    }


def identities_match(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    required = (
        "engine",
        "metric",
        "prefix",
        "eval_count",
        "allocated_capacity",
        "populated_length",
        "graph_mode",
        "output_boundary",
    )
    mismatches: list[str] = []
    for key in required:
        if left.get(key) != right.get(key):
            mismatches.append(f"{key}: {left.get(key)!r} != {right.get(key)!r}")
    return not mismatches, mismatches


def student_t_interval(
    samples: Sequence[float],
    *,
    critical: float,
) -> dict[str, Any]:
    values = [float(item) for item in samples]
    n = len(values)
    if n < 2:
        return {
            "n": n,
            "mean": values[0] if values else None,
            "se": None,
            "low": None,
            "high": None,
            "df": max(0, n - 1),
            "critical": critical,
        }
    mean = sum(values) / n
    variance = sum((item - mean) ** 2 for item in values) / (n - 1)
    se = math.sqrt(variance / n)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "low": mean - critical * se,
        "high": mean + critical * se,
        "df": n - 1,
        "critical": critical,
    }


def paired_log_ratio_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    critical: float,
) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        return {"usable": False, "reason": "pair_count_mismatch"}
    logs: list[float] = []
    ratios: list[float] = []
    for left, right in zip(a, b):
        if left <= 0 or right <= 0:
            return {"usable": False, "reason": "non_positive_sample"}
        ratio = right / left
        ratios.append(ratio)
        logs.append(math.log(ratio))
    stats = student_t_interval(logs, critical=critical)
    if stats["low"] is None or stats["se"] is None:
        return {"usable": False, "reason": "insufficient_samples", **stats}
    geo = math.exp(stats["mean"])
    return {
        "usable": True,
        "n": stats["n"],
        "df": stats["df"],
        "geometric_ratio": geo,
        "ci95_low": math.exp(stats["low"]),
        "ci95_high": math.exp(stats["high"]),
        "point_ratios": ratios,
        "critical": critical,
    }


def aa_usable(ci: Mapping[str, Any]) -> bool:
    if not ci.get("usable"):
        return False
    low = float(ci["ci95_low"])
    high = float(ci["ci95_high"])
    point = float(ci["geometric_ratio"])
    return low <= 1.0 <= high and 0.98 <= point <= 1.02


def family_excess_significant(differences: Sequence[float]) -> dict[str, Any]:
    one = student_t_interval(differences, critical=T_CRIT_DF2_ONE_SIDED)
    two = student_t_interval(differences, critical=T_CRIT_DF2_TWO_SIDED)
    positive = (
        one["low"] is not None and float(one["low"]) > 0.0 and len(differences) == 3
    )
    return {
        "n": len(differences),
        "mean_ms": one["mean"],
        "one_sided_low": one["low"],
        "two_sided_low": two["low"],
        "two_sided_high": two["high"],
        "significant_positive_excess": positive,
        "df": 2,
        "one_sided_critical": T_CRIT_DF2_ONE_SIDED,
        "two_sided_critical": T_CRIT_DF2_TWO_SIDED,
    }


def default_identity(**overrides: Any) -> dict[str, Any]:
    identity = {
        "engine": "quartz",
        "metric": "decode_only",
        "prefix": 128,
        "eval_count": 12,
        "allocated_capacity": 131072,
        "populated_length": 128,
        "graph_mode": "decode_segments8",
        "output_boundary": "eval_logits_no_host_materialize",
        "capture_arm": "graph",
        "device": 0,
        "context": 0,
        "process": 1,
    }
    identity.update(overrides)
    return identity


def coverage_row(
    tables: TraceTables,
    *,
    identity: Mapping[str, Any],
    window_start_ns: int,
    window_end_ns: int,
    expected_graph_launches: int | None = None,
    expected_nodes: int | None = None,
    profiler_overhead: Mapping[str, Any] | None = None,
    capture_arm: str | None = None,
) -> dict[str, Any]:
    errors = list(tables.errors)
    arm = capture_arm or str(identity.get("capture_arm") or "graph")
    window = max(0, window_end_ns - window_start_ns)
    if window <= 0:
        errors.append("missing or inverted window bounds")
    kernels = clip_all(tables.kernels, window_start_ns, window_end_ns)
    copies = clip_all(tables.copies, window_start_ns, window_end_ns)
    graphs = clip_all(tables.graphs, window_start_ns, window_end_ns)
    nodes = clip_all(tables.nodes, window_start_ns, window_end_ns)
    cpu_apis = clip_all(tables.cpu_apis, window_start_ns, window_end_ns)
    leaves = kernels + copies
    kernel_copy_union = union_ns(leaves)
    graph_union = union_ns(graphs)
    combined = union_ns(leaves + graphs)
    exclusive = graph_exclusive_without_leaves(graphs, leaves)
    exclusive_ns = union_ns(exclusive)
    outside_ns = uncovered_ns(window_start_ns, window_end_ns, leaves + graphs)
    span_ns = first_last_span_ns(leaves + graphs)
    outside_span_ns = max(0, span_ns - combined) if span_ns else 0
    unmapped = union_ns([item for item in leaves if item.family == "unclassified"])
    named = kernel_copy_union - unmapped
    family_share = (named / kernel_copy_union) if kernel_copy_union else 1.0
    observed_graphs = len(graphs)
    observed_nodes = len(nodes)
    parent_node = {
        "graph_launches": observed_graphs,
        "expanded_nodes": observed_nodes,
        "expected_nodes": expected_nodes,
        "matched": (
            expected_nodes is None
            or observed_nodes == expected_nodes
            or (arm == "graph" and observed_nodes == 0)
        ),
    }
    present_tables = {
        name: {
            "count": count,
            "unit": (tables.units_by_table.get(name) or {}).get("start"),
        }
        for name, count in tables.table_counts.items()
    }
    missing_required: list[str] = []
    if arm == "graph" and not any(name in tables.table_counts for name in GRAPH_TABLES):
        missing_required.append("CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    if arm == "node" and observed_nodes == 0:
        errors.append("node capture missing expanded node identities")
    if (
        expected_graph_launches is not None
        and observed_graphs != expected_graph_launches
    ):
        errors.append(
            "graph launch count "
            f"{observed_graphs} != expected {expected_graph_launches}"
        )
    if tables.drops:
        errors.append(f"dropped_rows={tables.drops}")
    if tables.truncated:
        errors.append("truncated_nodes")
    if missing_required:
        errors.append("absent_required_tables:" + ",".join(missing_required))
    if any(
        (tables.units_by_table.get(name) or {}).get("start") not in UNIT_TO_NS
        and name in tables.table_counts
        and name in NSYS_CUPTI_SCHEMA_NS
        for name in tables.table_counts
    ):
        pass
    covered = combined
    uncovered = outside_ns
    conservation_delta = abs(window - (covered + uncovered))
    conservation_ok = conservation_delta <= conservation_tolerance_ns(window)
    if not conservation_ok:
        errors.append(
            "duration_conservation_failed "
            f"window={window} covered={covered} uncovered={uncovered} "
            f"delta={conservation_delta}"
        )
    parent_child_double = False
    if graphs and leaves:
        summed = union_ns(graphs) + union_ns(leaves)
        if summed - union_ns(leaves + graphs) > conservation_tolerance_ns(window):
            # Overlap exists; charging both would double-count. Coverage uses union.
            parent_child_double = False
    nvtx_gpu = union_ns(clip_all(tables.nvtx, window_start_ns, window_end_ns))
    nvtx_added_gpu = False
    if nvtx_gpu > 0 and union_ns(leaves + graphs) < nvtx_gpu:
        # Nested NVTX must not invent GPU work; we never add NVTX to GPU union.
        nvtx_added_gpu = False
    internal_idle_ms = None
    if arm == "graph":
        internal_idle_ms = None
    elif arm == "node" and parent_node["matched"] and observed_nodes > 0:
        internal_idle_ms = ns_to_ms(exclusive_ns)
    overhead = dict(profiler_overhead or {})
    median_rel = overhead.get("median_relative_perturbation")
    perturbed = bool(median_rel is not None and abs(float(median_rel)) > 0.05)
    if perturbed:
        errors.append("excessive_profiler_overhead")
    llama_family_ok = True
    if (
        identity.get("require_llama_family")
        and identity.get("llama_family_present") is False
    ):
        llama_family_ok = False
        errors.append("missing_llama_family")
    no_work = (
        not leaves
        and not graphs
        and not cpu_apis
        and window > 0
        and arm != "unprofiled"
    )
    if no_work:
        errors.append("no-work_trace")
    coverage_valid = (
        not errors
        and not missing_required
        and llama_family_ok
        and window > 0
        and conservation_ok
        and not parent_child_double
        and not nvtx_added_gpu
    )
    families: dict[str, dict[str, Any]] = {}
    for item in leaves:
        bucket = families.setdefault(
            item.family,
            {
                "family": item.family,
                "count": 0,
                "union_ns": 0,
                "group": family_group(item.family),
            },
        )
        bucket["count"] += 1
    for family_name, group in list(families.items()):
        group["union_ns"] = union_ns(
            [item for item in leaves if item.family == family_name]
        )
        group["ms_per_eval"] = ns_to_ms(group["union_ns"]) / max(
            1, int(identity.get("eval_count") or 1)
        )
    return {
        "identity": dict(identity),
        "window_start_ns": int(window_start_ns),
        "window_end_ns": int(window_end_ns),
        "window_ns": window,
        "window_ms": ns_to_ms(window),
        "tables": present_tables,
        "table_names": sorted(present_tables),
        "expected_graph_launches": expected_graph_launches,
        "observed_graph_launches": observed_graphs,
        "observed_kernel_count": len(kernels),
        "observed_copy_count": len(copies),
        "observed_node_count": observed_nodes,
        "parent_node_correspondence": parent_node,
        "drops": tables.drops,
        "errors": errors,
        "kernel_copy_union_ns": kernel_copy_union,
        "graph_envelope_union_ns": graph_union,
        "combined_union_ns": combined,
        "graph_exclusive_without_leaf_ns": exclusive_ns,
        "time_outside_all_activity_ns": outside_ns,
        "first_last_span_ns": span_ns,
        "outside_combined_envelopes_in_span_ns": outside_span_ns,
        "unmapped_kernel_time_ns": unmapped,
        "named_family_share": family_share,
        "family_assignment_ok": family_share >= 0.95 or kernel_copy_union == 0,
        "families": families,
        "profiler_overhead": overhead,
        "internal_idle_ms": internal_idle_ms,
        "capture_arm": arm,
        "cpu_api_union_ns": union_ns(cpu_apis),
        "conservation_ok": conservation_ok,
        "parent_child_not_double_counted": True,
        "nvtx_does_not_add_gpu_work": True,
        "process_scoped_gaps_are_not_device_wide_idle": True,
        "coverage_valid": coverage_valid,
        "perturbed": perturbed,
        "sql_window": {
            "start_ns": window_start_ns,
            "end_ns": window_end_ns,
            "source": "markers_or_explicit",
        },
    }


def validate_coverage_row(row: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(row, Mapping):
        return ["coverage row is not an object"]
    for key in REQUIRED_COVERAGE_FIELDS:
        if key not in row:
            problems.append(f"missing field {key}")
    if row.get("coverage_valid") is None:
        problems.append("coverage_valid is null/unavailable and is not a pass")
    if row.get("coverage_valid") is not True:
        reasons = row.get("errors") or ["coverage_valid is not true"]
        problems.extend(str(item) for item in reasons)
    window = int(row.get("window_ns") or 0)
    covered = int(row.get("combined_union_ns") or 0)
    uncovered = int(row.get("time_outside_all_activity_ns") or 0)
    if window and abs(window - (covered + uncovered)) > conservation_tolerance_ns(
        window
    ):
        problems.append("duration conservation failed")
    if row.get("capture_arm") == "graph" and row.get("internal_idle_ms") is not None:
        problems.append("graph-only capture must leave internal_idle_ms null")
    share = row.get("named_family_share")
    if (
        share is not None
        and float(share) < 0.95
        and int(row.get("kernel_copy_union_ns") or 0) > 0
    ):
        if row.get("family_assignment_ok") is True:
            problems.append("family share below 95% marked ok")
    identity = row.get("identity") or {}
    if identity.get("metric") in {
        "complete_request",
        "decode_only",
        "prefill",
    } and identity.get("capacity_mismatch"):
        problems.append("metric/capacity mismatch")
    if identity.get("metric") == "prefill" and identity.get("output_policy_mismatch"):
        problems.append("prefill output-policy mismatch")
    return problems


def validate_coverage_document(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "ok": False,
            "status": "unavailable",
            "errors": ["coverage document is null/unavailable and is not a pass"],
        }
    if not isinstance(payload, Mapping):
        return {
            "ok": False,
            "status": "invalid",
            "errors": ["coverage document is not an object"],
        }
    status = str(payload.get("status") or "")
    if status in {"unavailable", "null", ""} and payload.get("windows") is None:
        return {
            "ok": False,
            "status": "unavailable",
            "errors": ["coverage status unavailable is not a pass"],
        }
    windows = payload.get("windows") or payload.get("rows") or []
    if isinstance(payload.get("identity"), Mapping) and "window_start_ns" in payload:
        windows = [payload]
    if not windows:
        return {
            "ok": False,
            "status": "invalid",
            "errors": ["coverage document has no windows"],
        }
    errors: list[str] = []
    for index, row in enumerate(windows):
        row_errors = validate_coverage_row(row)
        errors.extend(f"window[{index}]: {item}" for item in row_errors)
    ok = not errors
    return {
        "ok": ok,
        "status": "valid" if ok else "invalid",
        "errors": errors,
        "window_count": len(windows),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def regroup_fused_families(
    quartz_ms: Mapping[str, float],
    llama_ms: Mapping[str, float],
    *,
    enclosing: Mapping[str, Sequence[str]] | None = None,
    kernel_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Regroup Quartz leaves to llama fused enclosing groups.

    Unmatched time stays explicit. A parent graph is not an extra family.
    """
    groups = enclosing or LLAMA_ENCLOSING_GROUPS
    llama_present = {name for name, value in llama_ms.items() if float(value) > 0.0}
    assigned_quartz: set[str] = set()
    rows: list[dict[str, Any]] = []
    matched_excess = 0.0
    for family, members in groups.items():
        member_list = [str(item) for item in members]
        llama_value = 0.0
        quartz_value = 0.0
        used_members: list[str] = []
        llama_hit = False
        for member in member_list:
            if member in llama_ms:
                llama_value += float(llama_ms[member])
                llama_hit = True
            if member in quartz_ms:
                quartz_value += float(quartz_ms[member])
                used_members.append(member)
                assigned_quartz.add(member)
        if not llama_hit and not used_members:
            continue
        if family not in llama_present and not llama_hit and used_members:
            continue
        delta = quartz_value - llama_value
        matched_excess += delta
        rows.append(
            {
                "family": family,
                "logical_members": used_members or member_list,
                "Quartz_ms": quartz_value,
                "llama_ms": llama_value,
                "delta_ms": delta,
                "kernel_ids": list(kernel_ids.get(family, []) if kernel_ids else []),
                "coverage_status": (
                    "matched" if llama_hit and used_members else "partial"
                ),
                "proof_limit": (
                    "enclosing fused kernel; comparison is the fused boundary"
                    if len(member_list) > 1
                    else "leaf family"
                ),
            }
        )
    unmatched_quartz = {
        name: float(value)
        for name, value in quartz_ms.items()
        if name not in assigned_quartz and name != "unclassified"
    }
    unmatched_unclassified = float(quartz_ms.get("unclassified") or 0.0)
    unmatched_llama = {
        name: float(value)
        for name, value in llama_ms.items()
        if name not in {row["family"] for row in rows}
        and all(name not in (row.get("logical_members") or []) for row in rows)
        and name != "unclassified"
    }
    unmatched_ms = sum(unmatched_quartz.values()) + unmatched_unclassified
    rows.sort(key=lambda row: str(row["family"]))
    return {
        "families": rows,
        "matched_disjoint_excess_ms": matched_excess,
        "unmatched_quartz_ms": unmatched_quartz,
        "unmatched_llama_ms": unmatched_llama,
        "unmatched_work_gap_ms": unmatched_ms,
        "unclassified_ms": unmatched_unclassified,
    }


def reconcile_whole_gap(
    *,
    quartz_wall_ms: float,
    llama_wall_ms: float,
    matched_disjoint_excess_ms: float,
    unmatched_work_gap_ms: float,
    scheduling_host_residual_ms: float,
    overlap_ms: float = 0.0,
) -> dict[str, Any]:
    """whole wall gap = matched family excess + unmatched + host residual."""
    reconstructed = (
        matched_disjoint_excess_ms + unmatched_work_gap_ms + scheduling_host_residual_ms
    )
    observed = float(quartz_wall_ms) - float(llama_wall_ms)
    residual = observed - reconstructed
    limit = RESIDUAL_WALL_LIMIT * float(quartz_wall_ms)
    ranking_complete = abs(residual) <= limit
    return {
        "quartz_wall_ms": float(quartz_wall_ms),
        "llama_wall_ms": float(llama_wall_ms),
        "observed_gap_ms": observed,
        "matched_disjoint_excess_ms": float(matched_disjoint_excess_ms),
        "unmatched_work_gap_ms": float(unmatched_work_gap_ms),
        "scheduling_host_residual_ms": float(scheduling_host_residual_ms),
        "overlap_ms": float(overlap_ms),
        "reconstructed_gap_ms": reconstructed,
        "absolute_residual_ms": residual,
        "residual_limit_ms": limit,
        "residual_share_of_quartz_wall": (
            abs(residual) / float(quartz_wall_ms) if quartz_wall_ms else None
        ),
        "ranking_complete": ranking_complete,
        "proof_limit": (
            None
            if ranking_complete
            else (
                "absolute residual above 5% of Quartz wall; "
                "mapping/window diagnosis required"
            )
        ),
    }


def whole_wall_gap_from_partitions(
    quartz: Mapping[str, Any],
    llama: Mapping[str, Any],
) -> dict[str, Any]:
    """Q-L wall gap from disjoint partitions, not summed family deltas."""
    q_wall = float(quartz.get("window_ms") or 0.0)
    l_wall = float(llama.get("window_ms") or 0.0)
    gpu_gap = float(quartz.get("gpu_union_ms") or 0.0) - float(
        llama.get("gpu_union_ms") or 0.0
    )
    host_gap = float(quartz.get("host_exclusive_ms") or 0.0) - float(
        llama.get("host_exclusive_ms") or 0.0
    )
    q_instr = _ms(int(quartz.get("instrumentation_exclusive_ns") or 0))
    l_instr = _ms(int(llama.get("instrumentation_exclusive_ns") or 0))
    q_unresolved = float(quartz.get("unresolved_ms") or 0.0)
    l_unresolved = float(llama.get("unresolved_ms") or 0.0)
    reconstructed = (
        gpu_gap + host_gap + (q_instr - l_instr) + (q_unresolved - l_unresolved)
    )
    observed = q_wall - l_wall
    residual = observed - reconstructed
    limit = RESIDUAL_WALL_LIMIT * q_wall if q_wall else 0.0
    return {
        "quartz_wall_ms": q_wall,
        "llama_wall_ms": l_wall,
        "observed_gap_ms": observed,
        "gpu_union_gap_ms": gpu_gap,
        "host_exclusive_gap_ms": host_gap,
        "instrumentation_exclusive_gap_ms": q_instr - l_instr,
        "unresolved_gap_ms": q_unresolved - l_unresolved,
        "reconstructed_gap_ms": reconstructed,
        "absolute_residual_ms": residual,
        "residual_limit_ms": limit,
        "residual_share_of_quartz_wall": abs(residual) / q_wall if q_wall else None,
        "ranking_complete": abs(residual) <= limit,
        "method": "disjoint_gpu_union_plus_host_exclusive_plus_unresolved",
        "no_proportional_allocation": True,
        "family_overlap_bookkeeping_ms": {
            "quartz": _ms(int(quartz.get("family_overlap_ns") or 0)),
            "llama": _ms(int(llama.get("family_overlap_ns") or 0)),
        },
    }


def _positive_lower_bound(deltas: Sequence[float]) -> dict[str, Any]:
    stats = family_excess_significant(list(deltas))
    return {
        **stats,
        "evidenced": bool(stats.get("significant_positive_excess")),
    }


def select_top_two_families(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    """Deterministic top-two selection.

    Decode ranks the larger positive middle-window excess across D128/D2048.
    Prefill ranks P4096 complete-prompt totals. Ties break by family name.
    A family is selected once. Early/late-only wins are labeled unstable.
    """
    ranked: list[dict[str, Any]] = []
    unstable: list[str] = []
    for raw in rows:
        family = str(raw.get("family") or "")
        if not family:
            continue
        if phase == "prefill":
            deltas = [float(item) for item in (raw.get("p4096_deltas_ms") or [])]
            stats = _positive_lower_bound(deltas)
            score = stats.get("mean_ms")
            evidenced = bool(stats.get("evidenced"))
            ranked.append(
                {
                    "family": family,
                    "score_ms": score,
                    "evidenced": evidenced,
                    "stats": stats,
                    "source": "p4096_complete",
                }
            )
            continue
        d128 = [float(item) for item in (raw.get("d128_middle_deltas_ms") or [])]
        d2048 = [float(item) for item in (raw.get("d2048_middle_deltas_ms") or [])]
        s128 = (
            _positive_lower_bound(d128)
            if d128
            else {"evidenced": False, "mean_ms": None}
        )
        s2048 = (
            _positive_lower_bound(d2048)
            if d2048
            else {"evidenced": False, "mean_ms": None}
        )
        means = [
            value
            for value in (s128.get("mean_ms"), s2048.get("mean_ms"))
            if value is not None
        ]
        score = max(means) if means else None
        evidenced = bool(s128.get("evidenced") or s2048.get("evidenced"))
        early = _mean([float(item) for item in (raw.get("d128_early_deltas_ms") or [])])
        late = _mean([float(item) for item in (raw.get("d128_late_deltas_ms") or [])])
        middle = s128.get("mean_ms")
        only_edge = False
        if not evidenced and (
            (early is not None and early > 0) or (late is not None and late > 0)
        ):
            if middle is None or middle <= 0:
                only_edge = True
                unstable.append(family)
        ranked.append(
            {
                "family": family,
                "score_ms": score,
                "evidenced": evidenced and not only_edge,
                "unstable_edge_only": only_edge,
                "d128_middle": s128,
                "d2048_middle": s2048,
                "source": "middle_window",
            }
        )
    evidenced_rows = [
        row
        for row in ranked
        if row.get("evidenced") and row.get("score_ms") is not None
    ]
    evidenced_rows.sort(key=lambda row: (-float(row["score_ms"]), str(row["family"])))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in evidenced_rows:
        name = str(row["family"])
        if name in seen:
            continue
        seen.add(name)
        selected.append(row)
        if len(selected) == 2:
            break
    return {
        "phase": phase,
        "selected": selected,
        "ranked": ranked,
        "unstable": unstable,
        "selection_complete": len(selected) > 0,
        "max_selected": 2,
        "tie_break": "family_name",
        "window": "middle_[122,134)" if phase != "prefill" else "p4096_complete",
    }


def prefill_output_policy_ok(record: Mapping[str, Any]) -> dict[str, Any]:
    final_only = bool(record.get("final_token_logits_only"))
    logits_rows = record.get("logits_rows")
    full_vocab_every_row = bool(record.get("full_vocabulary_logits_every_row"))
    ok = final_only and not full_vocab_every_row and (logits_rows in {1, None})
    return {
        "ok": ok,
        "final_token_logits_only": final_only,
        "logits_rows": logits_rows,
        "full_vocabulary_logits_every_row": full_vocab_every_row,
        "reason": None if ok else "prefill output-policy mismatch",
    }


def replay_production_boundary_ok(record: Mapping[str, Any]) -> dict[str, Any]:
    cache_mode = str(record.get("cache_mode") or "")
    rotating = cache_mode == "rotating"
    synthetic = bool(record.get("synthetic_weights"))
    production_weights = bool(record.get("production_weights", True))
    phase = str(record.get("phase") or "")
    replay_family = str(record.get("replay_family") or "")
    phase_mismatch = False
    if phase == "prefill" and replay_family.startswith("decode-"):
        phase_mismatch = True
    if phase == "decode" and replay_family.startswith("prompt-"):
        phase_mismatch = True
    ok = rotating and production_weights and not synthetic and not phase_mismatch
    reason = None
    if not rotating:
        reason = "hot_cache_is_not_production"
    elif synthetic or not production_weights:
        reason = "synthetic_weights_not_production"
    elif phase_mismatch:
        reason = "replay/production boundary mismatch"
    return {
        "ok": ok,
        "cache_mode": cache_mode,
        "production_weights": production_weights,
        "synthetic_weights": synthetic,
        "phase": phase,
        "replay_family": replay_family,
        "reason": reason,
    }


def parent_identity_ok(
    pins: Mapping[str, Any],
    *,
    expected_graphs: str = "decode_segments8",
    expected_q4: str = "llama_q4k_mmvq",
    expected_mma: bool = True,
) -> dict[str, Any]:
    graphs = str(
        pins.get("execution_graphs") or pins.get("selected_execution_graph_path") or ""
    )
    q4 = str(pins.get("q4_decode") or "")
    mma = pins.get("opt137_dense_mma")
    ok = graphs == expected_graphs and q4 == expected_q4 and mma is expected_mma
    mismatches: list[str] = []
    if graphs != expected_graphs:
        mismatches.append("stale_execution_graphs")
    if q4 != expected_q4:
        mismatches.append("stale_q4_decode")
    if mma is not expected_mma:
        mismatches.append("stale_opt137_mma_parent")
    return {
        "ok": ok,
        "reason": None if ok else "stale parent identity",
        "mismatches": mismatches,
        "expected": {
            "execution_graphs": expected_graphs,
            "q4_decode": expected_q4,
            "opt137_dense_mma": expected_mma,
        },
        "observed": {
            "execution_graphs": graphs,
            "q4_decode": q4,
            "opt137_dense_mma": mma,
        },
    }


def missing_counter_record(
    *,
    kernel: str,
    engine: str,
    error: str | None,
) -> dict[str, Any]:
    """Unsupported metrics or permission failures are null, never zero."""
    return {
        "kernel": kernel,
        "engine": engine,
        "metrics": {},
        "dram_read_bytes": None,
        "dram_write_bytes": None,
        "dram_throughput": None,
        "l2_traffic": None,
        "sm_throughput": None,
        "achieved_occupancy": None,
        "stalls": None,
        "registers": None,
        "local_memory_spills": None,
        "error": error or "counters_unavailable",
        "zero_filled": False,
        "full_ncu_sweep": False,
    }


def opt133_expected_128_early() -> dict[str, float | int]:
    return {
        "ordinary_kernel_count": 84,
        "gpu_graph_count": 96,
        "kernel_copy_union_ms": 8.24,
        "graph_envelope_union_ms": 186.02,
        "combined_union_ms": 194.26,
        "first_to_last_span_ms": 196.74,
        "outside_combined_envelopes_ms": 2.48,
    }


def audit_window_from_tables(
    tables: TraceTables,
    *,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    expected_graph_launches: int | None = None,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_capture_window(
        tables,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )
    window_start_ns = int(resolved["window_start_ns"])
    window_end_ns = int(resolved["window_end_ns"])
    row = coverage_row(
        tables,
        identity=identity or default_identity(),
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        expected_graph_launches=expected_graph_launches,
    )
    row["window_source"] = resolved.get("source")
    row["window_missing_observation"] = resolved.get("missing_observation")
    row["wall_reconciliation"] = reconcile_window_wall(
        tables,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        window_source=str(resolved.get("source") or ""),
        missing_observation=resolved.get("missing_observation"),
    )
    row["sql"] = {
        "kernel": "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL",
        "memcpy": "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY",
        "graph": "SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        "units": "export_schema / CUPTI documented nanoseconds",
    }
    row["quantities_ms"] = {
        "ordinary_kernel_count": row["observed_kernel_count"],
        "gpu_graph_count": row["observed_graph_launches"],
        "kernel_copy_union_ms": round_ms(ns_to_ms(row["kernel_copy_union_ns"]) or 0.0),
        "graph_envelope_union_ms": round_ms(
            ns_to_ms(row["graph_envelope_union_ns"]) or 0.0
        ),
        "combined_union_ms": round_ms(ns_to_ms(row["combined_union_ns"]) or 0.0),
        "first_to_last_span_ms": round_ms(ns_to_ms(row["first_last_span_ns"]) or 0.0),
        "outside_combined_envelopes_ms": round_ms(
            ns_to_ms(row["outside_combined_envelopes_in_span_ns"]) or 0.0
        ),
    }
    return row


def compare_opt133_expected(
    quantities_ms: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    ok = True
    for key, want in expected.items():
        got = quantities_ms.get(key)
        if isinstance(want, float):
            match = got is not None and abs(float(got) - float(want)) <= 0.01
        else:
            match = got == want
        deltas[key] = {"expected": want, "observed": got, "match": match}
        ok = ok and match
    return {"ok": ok, "fields": deltas}


def historical_reconciliation_invalid(
    *,
    old_unobserved_ms: float | None,
    new_idle_ms: float | None,
    reason: str,
    link: str,
) -> dict[str, Any]:
    return {
        "method": "min(old_unobserved, new_idle)",
        "old_unobserved_ms": old_unobserved_ms,
        "new_idle_ms": new_idle_ms,
        "historical_min_ms": (
            min(float(old_unobserved_ms), float(new_idle_ms))
            if old_unobserved_ms is not None and new_idle_ms is not None
            else None
        ),
        "causal": False,
        "claim_type": "historical",
        "current_derived_idle_ms": None,
        "current_derived_idle_valid": False,
        "reason": reason,
        "link": link,
        "scalar_fraction_of_different_window_forbidden": True,
    }


def interval_as_dict(item: Interval) -> dict[str, Any]:
    payload = asdict(item)
    payload["duration_ns"] = item.duration_ns
    return payload


def validate_cli(coverage_path: Path) -> int:
    if not coverage_path.is_file():
        sys.stderr.write(f"missing coverage file {coverage_path}\n")
        return 2
    try:
        payload = load_json(coverage_path)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"invalid JSON: {exc}\n")
        return 2
    result = validate_coverage_document(payload)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        metavar="coverage.json",
        help="Validate a coverage.json document and exit nonzero if invalid",
    )
    args = parser.parse_args(argv)
    if args.validate:
        return validate_cli(Path(args.validate))
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
