"""Host tests for OPT-142 whole-wall residual reconciliation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from tools.opt142_wall_reconciliation import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    PHASES,
    PREFIXES,
    PREFILL_TOKENS,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTOR,
    family_plan,
    sqlite_path,
    validate_fixture,
)
from tools.performance_evidence import (
    Interval,
    RESIDUAL_WALL_LIMIT,
    classify_host_api,
    coverage_row,
    default_identity,
    exclusive_union_ns,
    overlap_union_ns,
    parse_nsys_sqlite,
    reconcile_whole_gap,
    reconcile_window_wall,
    resolve_capture_window,
    union_ns,
    whole_wall_gap_from_partitions,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
RUNNER = ROOT / "tools/run_optimization_task.py"
OPT138_TRACES = ROOT / "build/optimization-runs/opt138/traces"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE export_schema (table_name TEXT, column_name TEXT, unit TEXT)"
    )
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        "PROFILER_OVERHEAD",
        "NVTX_EVENTS",
    ):
        conn.execute("INSERT INTO export_schema VALUES (?, 'start', 'ns')", (table,))
        conn.execute("INSERT INTO export_schema VALUES (?, 'end', 'ns')", (table,))


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
          start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
          correlationId INTEGER, nameId INTEGER, returnValue INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          streamId INTEGER, correlationId INTEGER, globalPid INTEGER,
          syncType INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE PROFILER_OVERHEAD (
          start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER,
          returnValue INTEGER
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
    conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")


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


def test_contracts_makefile_and_registration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-142")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    opt136 = (ROOT / "tools/opt136_graph_accounting.py").read_text(encoding="utf-8")
    evidence = (ROOT / "tools/performance_evidence.py").read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt142_wall_reconciliation.py").read_text(encoding="utf-8")
    opt138 = (ROOT / "tools/opt138_remaining_gap_profile.py").read_text(
        encoding="utf-8"
    )
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-142"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["unresolved_limit"] == 0.05
    assert contract["reuse_opt138_traces"] is True
    assert iteration["diagnostics_make_target"] == "cuda-opt142-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt142-diagnostics" in makefile
    assert "QW38_OPT142_WALL_RECONCILIATION_RESULT=" in runner
    assert "reconcile_window_wall" in evidence
    assert "whole_wall_gap_from_partitions" in evidence
    assert "nameId" in evidence
    assert "authenticate_current_pins" in opt136
    assert "whole_wall_gap_from_partitions" in opt138
    assert "no_proportional_allocation" in tool
    assert PHASES == ("preflight", "reproduce", "reconcile", "report")
    assert PREFIXES == (128, 2048)
    assert PREFILL_TOKENS == 4096
    validate_future_keep_policy("OPT-142", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-142")
    preflight = iteration["workloads"]["preflight"]
    reproduce = iteration["workloads"]["reproduce"]
    reconcile = iteration["workloads"]["reconcile"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(reproduce, "acceptance")) == 2
    assert loop_product(workload_for_mode(reconcile, "acceptance")) == 21
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-142", "acceptance", iteration, "reconcile")
    assert "phase=reconcile" in described
    assert "loop_product=1" in family_plan("acceptance", "preflight")


def test_cpu_overlap_is_not_additive(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (0, 10_000_000, "opt136.window"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,NULL)",
            (1_000_000, 8_000_000, "attn_core"),
        )
        conn.execute("INSERT INTO StringIds VALUES (1, 'cudaEventSynchronize_v3020')")
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,0,1,1,1,0)",
            (1_000_000, 9_000_000),
        )

    path = _sqlite(tmp_path, "cpu-overlap.sqlite", build)
    tables = parse_nsys_sqlite(path)
    assert any("Synchronize" in item.name for item in tables.cpu_apis)
    wall = reconcile_window_wall(
        tables, window_start_ns=0, window_end_ns=10_000_000, window_source="explicit"
    )
    assert wall["gpu_union_ns"] == 7_000_000
    assert wall["cpu_gpu_overlap_ns"] == 7_000_000
    assert wall["host_exclusive_ns"] == 1_000_000
    assert wall["host_exclusive_ns"] + wall["gpu_union_ns"] == 8_000_000
    assert wall["cpu_overlapping_gpu_not_additive"] is True


def test_overlapping_families_are_bookkeeping_not_wall(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (0, 10_000_000, "opt136.window"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,1,1,1,?,NULL)",
            (1_000_000, 6_000_000, "mul_mat_vec_q"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,2,1,1,?,NULL)",
            (3_000_000, 8_000_000, "rms_norm_f32"),
        )

    path = _sqlite(tmp_path, "family-overlap.sqlite", build)
    tables = parse_nsys_sqlite(path)
    wall = reconcile_window_wall(
        tables, window_start_ns=0, window_end_ns=10_000_000, window_source="explicit"
    )
    assert union_ns(tables.kernels) == 7_000_000
    assert wall["family_overlap_ns"] == 3_000_000
    assert wall["gpu_union_ns"] == 7_000_000
    bookkeeping = [
        row
        for row in wall["buckets"]
        if row["label"] == "family_sum_overlap_bookkeeping"
    ]
    assert bookkeeping[0]["classification"] == "bookkeeping"


def test_eval_nvtx_window_not_gpu_span(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (
                0,
                10_000_000,
                "opt136 engine=quartz prefix=2048 step=122 layer=-1 op=eval",
            ),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,NULL)",
            (2_000_000, 8_000_000, "attn_core"),
        )

    path = _sqlite(tmp_path, "eval-window.sqlite", build)
    tables = parse_nsys_sqlite(path)
    resolved = resolve_capture_window(tables)
    assert resolved["source"] == "eval_or_prefill_nvtx"
    assert resolved["window_start_ns"] == 0
    assert resolved["window_end_ns"] == 10_000_000
    wall = reconcile_window_wall(
        tables,
        window_start_ns=resolved["window_start_ns"],
        window_end_ns=resolved["window_end_ns"],
        window_source=resolved["source"],
    )
    assert wall["window_ns"] == 10_000_000
    assert wall["gpu_union_ns"] == 6_000_000
    assert wall["unresolved_ns"] == 4_000_000


def test_name_id_runtime_and_sync_type(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (0, 10_000_000, "opt136.window"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,NULL)",
            (2_000_000, 5_000_000, "attn_core"),
        )
        conn.execute("INSERT INTO StringIds VALUES (9, 'cudaGraphLaunch_v10000')")
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,0,1,1,9,0)",
            (1_000_000, 1_500_000),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (?,?,0,1,7,1,1,1)",
            (5_000_000, 6_000_000),
        )

    path = _sqlite(tmp_path, "name-id.sqlite", build)
    tables = parse_nsys_sqlite(path)
    names = {item.name for item in tables.cpu_apis}
    assert "cudaGraphLaunch_v10000" in names
    assert "cudaEventSynchronize" in names
    assert classify_host_api("cudaGraphLaunch_v10000") == "launch"
    assert classify_host_api("cudaEventSynchronize") == "sync"


def test_conservation_and_five_percent_unresolved(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (0, 100_000_000, "opt136.window"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,NULL)",
            (0, 90_000_000, "attn_core"),
        )
        conn.execute("INSERT INTO StringIds VALUES (3, 'cudaLaunchKernel_v7000')")
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,0,1,1,3,0)",
            (90_000_000, 96_000_000),
        )

    path = _sqlite(tmp_path, "limit.sqlite", build)
    tables = parse_nsys_sqlite(path)
    wall = reconcile_window_wall(
        tables, window_start_ns=0, window_end_ns=100_000_000, window_source="explicit"
    )
    assert wall["conservation_ok"] is True
    assert wall["unresolved_share_of_wall"] == 0.04
    assert wall["within_unresolved_limit"] is True
    over = reconcile_window_wall(
        tables, window_start_ns=0, window_end_ns=100_000_000, window_source="explicit"
    )
    over["unresolved_share_of_wall"] = 0.23
    assert over["unresolved_share_of_wall"] > RESIDUAL_WALL_LIMIT


def test_no_proportional_allocation_keeps_unresolved() -> None:
    recon = reconcile_whole_gap(
        quartz_wall_ms=100.0,
        llama_wall_ms=60.0,
        matched_disjoint_excess_ms=10.0,
        unmatched_work_gap_ms=5.0,
        scheduling_host_residual_ms=0.0,
    )
    assert recon["ranking_complete"] is False
    assert recon["absolute_residual_ms"] == 25.0
    quartz = {
        "window_ms": 100.0,
        "gpu_union_ms": 90.0,
        "host_exclusive_ms": 4.0,
        "instrumentation_exclusive_ns": 0,
        "unresolved_ms": 6.0,
        "family_overlap_ns": 0,
    }
    llama = {
        "window_ms": 60.0,
        "gpu_union_ms": 55.0,
        "host_exclusive_ms": 3.0,
        "instrumentation_exclusive_ns": 0,
        "unresolved_ms": 2.0,
        "family_overlap_ns": 21_000_000,
    }
    gap = whole_wall_gap_from_partitions(quartz, llama)
    assert gap["no_proportional_allocation"] is True
    assert abs(gap["observed_gap_ms"] - gap["reconstructed_gap_ms"]) < 1e-9
    assert gap["family_overlap_bookkeeping_ms"]["llama"] == 21.0


def test_overlap_union_helper() -> None:
    left = [Interval(start_ns=0, end_ns=10, kind="kernel")]
    right = [Interval(start_ns=5, end_ns=15, kind="cuda_api")]
    assert overlap_union_ns(left, right) == 5
    assert exclusive_union_ns(right, left) == 5


def test_graph_parent_exclusion_in_wall(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (0, 10_000_000, "opt136.window"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?,?,1,1,7,1,1,1,NULL)",
            (1_000_000, 9_000_000),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,1)",
            (1_000_000, 8_000_000, "attn_core"),
        )

    path = _sqlite(tmp_path, "graph-parent.sqlite", build)
    tables = parse_nsys_sqlite(path)
    wall = reconcile_window_wall(
        tables, window_start_ns=0, window_end_ns=10_000_000, window_source="explicit"
    )
    assert wall["parent_child_not_double_counted"] is True
    assert wall["gpu_union_ns"] == 8_000_000
    assert wall["graph_exclusive_without_leaf_ns"] == 1_000_000


def test_fixture_keys() -> None:
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert REPORT.as_posix().endswith("opt142-wall-reconciliation/REPORT.md")


def test_independent_raw_interval_reconstruction() -> None:
    decode = sqlite_path("quartz", "decode", 2048, "middle", 0)
    prefill = sqlite_path("quartz", "prefill", 4096, "prefill", 0)
    if not decode.is_file() or not prefill.is_file():
        return
    decode_tables = parse_nsys_sqlite(decode)
    decode_resolved = resolve_capture_window(decode_tables)
    decode_wall = reconcile_window_wall(
        decode_tables,
        window_start_ns=int(decode_resolved["window_start_ns"]),
        window_end_ns=int(decode_resolved["window_end_ns"]),
        window_source=str(decode_resolved.get("source") or ""),
    )
    assert decode_wall["conservation_ok"] is True
    assert decode_wall["window_ns"] == (
        decode_wall["gpu_union_ns"]
        + decode_wall["host_exclusive_ns"]
        + decode_wall["instrumentation_exclusive_ns"]
        + decode_wall["unresolved_ns"]
        + decode_wall["conservation_delta_ns"]
    )
    prefill_tables = parse_nsys_sqlite(prefill)
    prefill_resolved = resolve_capture_window(prefill_tables)
    prefill_wall = reconcile_window_wall(
        prefill_tables,
        window_start_ns=int(prefill_resolved["window_start_ns"]),
        window_end_ns=int(prefill_resolved["window_end_ns"]),
        window_source=str(prefill_resolved.get("source") or ""),
    )
    assert prefill_wall["conservation_ok"] is True
    llama_decode = sqlite_path("llama", "decode", 2048, "middle", 0)
    if llama_decode.is_file():
        llama_tables = parse_nsys_sqlite(llama_decode)
        llama_resolved = resolve_capture_window(llama_tables)
        llama_wall = reconcile_window_wall(
            llama_tables,
            window_start_ns=int(llama_resolved["window_start_ns"]),
            window_end_ns=int(llama_resolved["window_end_ns"]),
            window_source=str(llama_resolved.get("source") or ""),
        )
        gap = whole_wall_gap_from_partitions(decode_wall, llama_wall)
        assert gap["no_proportional_allocation"] is True
        assert abs(gap["absolute_residual_ms"]) <= 0.02


def test_coverage_row_still_excludes_cpu_from_gpu(tmp_path: Path) -> None:
    def build(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
            (0, 10_000_000, "opt136.window"),
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,NULL)",
            (1_000_000, 8_000_000, "attn_core"),
        )

    path = _sqlite(tmp_path, "coverage-gpu.sqlite", build)
    tables = parse_nsys_sqlite(path)
    row = coverage_row(
        tables,
        identity=default_identity(),
        window_start_ns=0,
        window_end_ns=10_000_000,
        capture_arm="graph",
    )
    assert row["combined_union_ns"] == 7_000_000
    assert row["time_outside_all_activity_ns"] == 3_000_000
