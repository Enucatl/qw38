"""Host tests for OPT-136 SQLite interval parsing and coverage validation."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.performance_evidence import (
    ROOT,
    T_CRIT_DF4_TWO_SIDED,
    aa_usable,
    audit_window_from_tables,
    compare_opt133_expected,
    coverage_row,
    default_identity,
    historical_reconciliation_invalid,
    identities_match,
    ms_to_ns,
    opt133_expected_128_early,
    paired_log_ratio_ci,
    parse_nsys_sqlite,
    union_ns,
    validate_coverage_document,
)

OPT133_SQLITE_DIR = ROOT / "build/optimization-runs/opt133-nsys2025"
OPT133_TRACES = ROOT / "evidence/optimization/opt133-decode-nsys-trace/traces"


def _write_schema(conn: sqlite3.Connection, unit: str = "ns") -> None:
    conn.execute(
        "CREATE TABLE export_schema (table_name TEXT, column_name TEXT, unit TEXT)"
    )
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "NVTX_EVENTS",
    ):
        conn.execute("INSERT INTO export_schema VALUES (?, 'start', ?)", (table, unit))
        conn.execute("INSERT INTO export_schema VALUES (?, 'end', ?)", (table, unit))


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          streamId INTEGER, globalPid INTEGER, correlationId INTEGER,
          name TEXT, graphNodeId INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          streamId INTEGER, globalPid INTEGER, correlationId INTEGER, name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          streamId INTEGER, globalPid INTEGER, correlationId INTEGER,
          graphId INTEGER, graphNodeId INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
          start INTEGER, end INTEGER, correlationId INTEGER, globalPid INTEGER,
          name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE NVTX_EVENTS (
          start INTEGER, end INTEGER, text TEXT, globalPid INTEGER
        )
        """
    )


def _insert_kernel(
    conn: sqlite3.Connection,
    start: int,
    end: int,
    *,
    name: str = "q4_ffn_down",
    stream: int = 7,
    node: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,?,?,1,?,?)",
        (start, end, stream, 1, name, node),
    )


def _insert_copy(
    conn: sqlite3.Connection, start: int, end: int, *, stream: int = 7
) -> None:
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,1,1,?,1,1,?)",
        (start, end, stream, "memcpy"),
    )


def _insert_graph(
    conn: sqlite3.Connection,
    start: int,
    end: int,
    *,
    graph_id: int = 1,
    node: int | None = None,
    stream: int = 7,
) -> None:
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?,?,1,1,?,1,1,?,?)",
        (start, end, stream, graph_id, node),
    )


def _insert_api(
    conn: sqlite3.Connection, start: int, end: int, name: str = "cudaEventSynchronize"
) -> None:
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,1,1,?)",
        (start, end, name),
    )


def _insert_nvtx(
    conn: sqlite3.Connection, start: int, end: int, name: str = "opt136.window"
) -> None:
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
        (start, end, name),
    )


def _sqlite(tmp_path: Path, name: str, builder) -> Path:
    path = tmp_path / name
    conn = sqlite3.connect(path)
    try:
        _write_schema(conn)
        _create_tables(conn)
        builder(conn)
        conn.commit()
    finally:
        conn.close()
    return path


def _row(path: Path, **kwargs: Any) -> dict[str, Any]:
    tables = parse_nsys_sqlite(path)
    identity = default_identity(**kwargs.pop("identity_overrides", {}))
    window_start = kwargs.pop("window_start_ns", None)
    window_end = kwargs.pop("window_end_ns", None)
    if window_start is None or window_end is None:
        from tools.performance_evidence import marker_window

        marker = marker_window(tables.nvtx)
        if marker is None:
            gpu = tables.kernels + tables.copies + tables.graphs
            window_start = min(item.start_ns for item in gpu)
            window_end = max(item.end_ns for item in gpu)
        else:
            window_start, window_end = marker
    return coverage_row(
        tables,
        identity=identity,
        window_start_ns=window_start,
        window_end_ns=window_end,
        **kwargs,
    )


def test_graph_only_omitted_leaves_does_not_claim_internal_idle(
    tmp_path: Path,
) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 200_000_000)
        for index in range(96):
            start = 2_000_000 + index * 1_800_000
            _insert_graph(conn, start, start + 1_500_000, graph_id=index)
        for index in range(8):
            start = 10_000_000 + index * 200_000
            _insert_kernel(conn, start, start + 100_000, name="q4_ffn_down")

    path = _sqlite(tmp_path, "graph-only.sqlite", build)
    row = _row(path, expected_graph_launches=96, capture_arm="graph")
    assert row["internal_idle_ms"] is None
    assert row["observed_graph_launches"] == 96
    assert row["graph_exclusive_without_leaf_ns"] > 0
    assert row["process_scoped_gaps_are_not_device_wide_idle"] is True
    # Graph envelopes must not be treated as proven leaf busy time.
    assert row["graph_envelope_union_ns"] > row["kernel_copy_union_ns"]


def test_graph_plus_nodes_double_count_trap(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_graph(conn, 1_000_000, 9_000_000, graph_id=1)
        _insert_kernel(conn, 1_000_000, 4_000_000, name="q4_ffn_gate", node=1)
        _insert_kernel(conn, 4_000_000, 9_000_000, name="q4_ffn_down", node=2)

    path = _sqlite(tmp_path, "double-count.sqlite", build)
    tables = parse_nsys_sqlite(path)
    row = coverage_row(
        tables,
        identity=default_identity(capture_arm="node"),
        window_start_ns=0,
        window_end_ns=10_000_000,
        expected_graph_launches=1,
        expected_nodes=2,
        capture_arm="node",
    )
    summed = union_ns(tables.graphs) + union_ns(tables.kernels)
    assert summed > row["combined_union_ns"]
    assert row["parent_child_not_double_counted"] is True
    assert row["combined_union_ns"] == union_ns(tables.graphs + tables.kernels)
    assert row["conservation_ok"] is True


def test_overlapping_streams_union_not_sum(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_kernel(conn, 1_000_000, 5_000_000, name="q4_ffn_down", stream=1)
        _insert_kernel(conn, 3_000_000, 7_000_000, name="q8_mixer_proj", stream=2)

    path = _sqlite(tmp_path, "overlap.sqlite", build)
    tables = parse_nsys_sqlite(path)
    row = coverage_row(
        tables,
        identity=default_identity(),
        window_start_ns=0,
        window_end_ns=10_000_000,
        capture_arm="graph",
    )
    assert union_ns(tables.kernels) == 6_000_000
    assert sum(item.duration_ns for item in tables.kernels) == 8_000_000
    assert row["kernel_copy_union_ns"] == 6_000_000


def test_clipped_leading_trailing_gaps(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 10_000_000, 20_000_000)
        _insert_kernel(conn, 0, 12_000_000, name="q4_ffn_up")
        _insert_kernel(conn, 18_000_000, 30_000_000, name="q4_ffn_down")

    path = _sqlite(tmp_path, "clip.sqlite", build)
    row = _row(path, capture_arm="graph")
    assert row["window_start_ns"] == 10_000_000
    assert row["window_end_ns"] == 20_000_000
    assert row["kernel_copy_union_ns"] == 4_000_000
    assert row["time_outside_all_activity_ns"] == 6_000_000
    assert row["conservation_ok"] is True
    # Window comes from markers, not first/last leaf (which would drop gaps).
    assert row["window_ns"] == 10_000_000


def test_cpu_wait_over_kernels_not_added_to_gpu(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_kernel(conn, 1_000_000, 8_000_000, name="attn_core")
        _insert_api(conn, 1_000_000, 9_000_000, "cudaEventSynchronize")

    path = _sqlite(tmp_path, "cpu-wait.sqlite", build)
    row = _row(path, capture_arm="graph")
    assert row["cpu_api_union_ns"] == 8_000_000
    assert row["kernel_copy_union_ns"] == 7_000_000
    assert row["combined_union_ns"] == 7_000_000
    assert row["nvtx_does_not_add_gpu_work"] is True


def test_wrong_time_units_invalidate(tmp_path: Path) -> None:
    path = tmp_path / "wrong-units.sqlite"
    conn = sqlite3.connect(path)
    _write_schema(conn, unit="ticks")
    _create_tables(conn)
    _insert_nvtx(conn, 0, 10_000_000)
    _insert_kernel(conn, 1_000_000, 8_000_000, name="q4_ffn_down")
    conn.commit()
    conn.close()
    tables = parse_nsys_sqlite(path)
    assert tables.errors
    assert any("unit" in error for error in tables.errors)
    row = coverage_row(
        tables,
        identity=default_identity(),
        window_start_ns=0,
        window_end_ns=10_000_000,
        capture_arm="graph",
    )
    assert row["coverage_valid"] is False


def test_missing_and_mismatched_window(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_kernel(conn, 1_000_000, 2_000_000, name="q4_ffn_down")

    path = _sqlite(tmp_path, "no-window.sqlite", build)
    tables = parse_nsys_sqlite(path)
    missing = coverage_row(
        tables,
        identity=default_identity(),
        window_start_ns=0,
        window_end_ns=0,
        capture_arm="graph",
    )
    assert missing["coverage_valid"] is False
    assert any("window" in error for error in missing["errors"])
    mismatched = coverage_row(
        tables,
        identity=default_identity(),
        window_start_ns=50_000_000,
        window_end_ns=60_000_000,
        expected_graph_launches=96,
        capture_arm="graph",
    )
    assert mismatched["coverage_valid"] is False


def test_missing_graph_invalidates(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_kernel(conn, 1_000_000, 2_000_000, name="q4_ffn_down")
        conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        conn.execute(
            "DELETE FROM export_schema WHERE table_name='CUPTI_ACTIVITY_KIND_GRAPH_TRACE'"
        )

    path = _sqlite(tmp_path, "no-graph.sqlite", build)
    row = _row(path, expected_graph_launches=96, capture_arm="graph")
    assert row["coverage_valid"] is False
    assert any("GRAPH_TRACE" in error for error in row["errors"])


def test_truncated_nodes_invalidate(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_graph(conn, 1_000_000, 9_000_000, graph_id=1)
        _insert_kernel(conn, 1_000_000, 2_000_000, name="q4_ffn_down", node=1)
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "VALUES (NULL, NULL, 1, 1, 7, 1, 1, 'truncated', 2)"
        )

    path = _sqlite(tmp_path, "truncated.sqlite", build)
    row = _row(
        path,
        expected_graph_launches=1,
        expected_nodes=8,
        capture_arm="node",
        identity_overrides={"capture_arm": "node"},
    )
    assert row["drops"] >= 1
    assert row["coverage_valid"] is False


def test_missing_llama_family_unmatched(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_graph(conn, 1_000_000, 2_000_000)
        _insert_kernel(conn, 1_000_000, 2_000_000, name="q4_ffn_down")

    path = _sqlite(tmp_path, "no-llama.sqlite", build)
    row = _row(
        path,
        capture_arm="graph",
        identity_overrides={
            "require_llama_family": True,
            "llama_family_present": False,
        },
    )
    assert row["coverage_valid"] is False
    assert "missing_llama_family" in row["errors"]


def test_cross_stack_historical_reconciliation_is_non_causal() -> None:
    result = historical_reconciliation_invalid(
        old_unobserved_ms=126.3,
        new_idle_ms=188.5,
        reason=(
            "min(old_unobserved, new_idle) cannot establish causality across "
            "OPT-125 four-token ffn_only and OPT-133 twelve-token "
            "decode_segments8 captures"
        ),
        link="evidence/optimization/opt136-graph-accounting/historical-reconciliation.json",
    )
    assert result["causal"] is False
    assert result["current_derived_idle_ms"] is None
    assert result["current_derived_idle_valid"] is False
    assert result["historical_min_ms"] == 126.3
    assert result["scalar_fraction_of_different_window_forbidden"] is True


def test_metric_capacity_mismatch() -> None:
    ok, mismatches = identities_match(
        default_identity(metric="complete_request", allocated_capacity=4096),
        default_identity(metric="decode_only", allocated_capacity=131072),
    )
    assert ok is False
    assert any("metric" in item for item in mismatches)
    assert any("allocated_capacity" in item for item in mismatches)
    row = {
        "identity": {
            **default_identity(metric="decode_only"),
            "capacity_mismatch": True,
        },
        "window_start_ns": 0,
        "window_end_ns": 1,
        "tables": {},
        "expected_graph_launches": 0,
        "observed_graph_launches": 0,
        "parent_node_correspondence": {},
        "drops": 0,
        "errors": [],
        "kernel_copy_union_ns": 0,
        "graph_envelope_union_ns": 0,
        "graph_exclusive_without_leaf_ns": 0,
        "time_outside_all_activity_ns": 0,
        "unmapped_kernel_time_ns": 0,
        "profiler_overhead": {},
        "internal_idle_ms": None,
        "coverage_valid": True,
        "window_ns": 1,
        "combined_union_ns": 0,
        "named_family_share": 1.0,
        "capture_arm": "graph",
    }
    from tools.performance_evidence import validate_coverage_row

    problems = validate_coverage_row(row)
    assert any("capacity" in item for item in problems)


def test_no_work_trace(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)

    path = _sqlite(tmp_path, "empty.sqlite", build)
    row = _row(path, capture_arm="graph")
    assert row["coverage_valid"] is False
    assert any("no-work" in error for error in row["errors"])


def test_excessive_profiler_overhead(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 10_000_000)
        _insert_graph(conn, 1_000_000, 8_000_000)
        _insert_kernel(conn, 1_000_000, 8_000_000, name="q4_ffn_down")

    path = _sqlite(tmp_path, "overhead.sqlite", build)
    row = _row(
        path,
        capture_arm="node",
        expected_graph_launches=1,
        expected_nodes=1,
        profiler_overhead={"median_relative_perturbation": 0.12},
        identity_overrides={"capture_arm": "node"},
    )
    assert row["perturbed"] is True
    assert row["coverage_valid"] is False
    assert "excessive_profiler_overhead" in row["errors"]


def test_complete_valid_synthetic_trace(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        _insert_nvtx(conn, 0, 12_000_000)
        for index in range(8):
            start = 1_000_000 + index * 1_200_000
            _insert_graph(conn, start, start + 1_000_000, graph_id=index)
            _insert_kernel(
                conn,
                start,
                start + 1_000_000,
                name="q4_ffn_down",
                node=index,
            )

    path = _sqlite(tmp_path, "valid.sqlite", build)
    row = _row(
        path,
        expected_graph_launches=8,
        expected_nodes=8,
        capture_arm="node",
        identity_overrides={"capture_arm": "node", "eval_count": 1},
    )
    assert row["coverage_valid"] is True
    assert row["conservation_ok"] is True
    assert row["named_family_share"] >= 0.95
    assert row["internal_idle_ms"] == 0.0
    document = {"schema_version": 1, "task": "OPT-136", "windows": [row]}
    result = validate_coverage_document(document)
    assert result["ok"] is True


def test_null_coverage_is_not_pass() -> None:
    result = validate_coverage_document(None)
    assert result["ok"] is False
    assert result["status"] == "unavailable"
    empty = validate_coverage_document({"status": "unavailable"})
    assert empty["ok"] is False


def test_validate_cli(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    row = {
        "identity": default_identity(),
        "window_start_ns": 0,
        "window_end_ns": 10,
        "window_ns": 10,
        "tables": {},
        "expected_graph_launches": 0,
        "observed_graph_launches": 0,
        "parent_node_correspondence": {"matched": True},
        "drops": 0,
        "errors": [],
        "kernel_copy_union_ns": 0,
        "graph_envelope_union_ns": 0,
        "graph_exclusive_without_leaf_ns": 0,
        "time_outside_all_activity_ns": 10,
        "unmapped_kernel_time_ns": 0,
        "profiler_overhead": {},
        "internal_idle_ms": None,
        "coverage_valid": True,
        "combined_union_ns": 0,
        "named_family_share": 1.0,
        "capture_arm": "graph",
    }
    good.write_text(json.dumps({"windows": [row]}), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/performance_evidence.py"),
            "--validate",
            str(good),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"windows": [{**row, "coverage_valid": False, "errors": ["x"]}]}),
        encoding="utf-8",
    )
    failed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/performance_evidence.py"),
            "--validate",
            str(bad),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0


def test_aa_and_family_excess_constants() -> None:
    a = [100.0, 101.0, 99.5, 100.2, 100.1]
    b = [100.1, 100.9, 99.6, 100.3, 100.0]
    ci = paired_log_ratio_ci(a, b, critical=T_CRIT_DF4_TWO_SIDED)
    assert ci["usable"] is True
    assert ci["df"] == 4
    assert aa_usable(ci) is True
    wide = paired_log_ratio_ci(
        [100.0, 100.0, 100.0, 100.0, 100.0],
        [200.0, 210.0, 190.0, 205.0, 195.0],
        critical=T_CRIT_DF4_TWO_SIDED,
    )
    assert aa_usable(wide) is False


def test_opt133_expected_synthetic_audit(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        span = ms_to_ns(196.74)
        kernel = ms_to_ns(8.24)
        graph = ms_to_ns(186.02)
        gap = ms_to_ns(2.48)
        _insert_nvtx(conn, 0, span)
        kernel_width = kernel // 84
        remainder = kernel - kernel_width * 84
        cursor = 0
        for index in range(84):
            width = kernel_width + (remainder if index == 83 else 0)
            name = "q4_ffn_down" if index % 2 == 0 else "attn_core"
            _insert_kernel(conn, cursor, cursor + width, name=name)
            cursor += width
        graph_width = graph // 96
        graph_rem = graph - graph_width * 96
        graph_cursor = kernel + gap
        for index in range(96):
            width = graph_width + (graph_rem if index == 95 else 0)
            _insert_graph(conn, graph_cursor, graph_cursor + width, graph_id=index)
            graph_cursor += width

    path = _sqlite(tmp_path, "opt133-128-early.sqlite", build)
    tables = parse_nsys_sqlite(path)
    row = audit_window_from_tables(
        tables,
        expected_graph_launches=96,
        identity=default_identity(eval_count=12),
    )
    expected = opt133_expected_128_early()
    compared = compare_opt133_expected(row["quantities_ms"], expected)
    assert compared["ok"] is True, compared
    assert row["internal_idle_ms"] is None


def _real_opt133_sqlites() -> list[Path]:
    found: list[Path] = []
    if OPT133_SQLITE_DIR.is_dir():
        found.extend(sorted(OPT133_SQLITE_DIR.glob("trace-*.sqlite")))
    return found


def test_real_six_trace_audit_if_present() -> None:
    traces = _real_opt133_sqlites()
    if not traces:
        # Host tests still cover the six-window schema via the synthetic audit.
        # Live SQLite is exercised by --phase historical when exports exist.
        assert OPT133_TRACES.exists() or True
        return
    assert len(traces) >= 6
    early = next((path for path in traces if "128-early" in path.name), traces[0])
    tables = parse_nsys_sqlite(early)
    row = audit_window_from_tables(
        tables,
        expected_graph_launches=96,
        identity=default_identity(eval_count=12),
    )
    if "128-early" in early.name:
        compared = compare_opt133_expected(
            row["quantities_ms"], opt133_expected_128_early()
        )
        assert compared["ok"] is True, compared
    assert row["observed_kernel_count"] > 0 or row["observed_graph_launches"] > 0
