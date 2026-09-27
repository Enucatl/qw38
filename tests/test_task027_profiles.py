"""Check transient allocation accounting and exclusive kernel attribution."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task027_profiles import (
    allocation_peak,
    category,
    compare_telemetry,
    range_costs,
    telemetry,
)


def test_range_costs_overlap_boundaries_and_correlation() -> None:
    """Overlapping calls and kernels must not double count elapsed coverage."""
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1,'cudaGraphLaunch'),(2,'nested'),(3,'cudaMemcpyAsync'),(4,'other_thread');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10,90,7,1,1),(20,50,7,2,2),(0,5,7,3,3),(0,100,8,4,4);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (30,80,1),(60,90,1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER,end INTEGER,bytes INTEGER,copyKind INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0,20,16,1,3);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER,end INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (95,105);
    """)
    try:
        row = range_costs(db, 0, 100, 7)
    finally:
        db.close()
    assert row["cpu_cuda_api_union_ms"] == 85 / 1e6
    assert row["host_outside_cuda_api_ms"] == 15 / 1e6
    assert row["kernel_sum_ms"] == 80 / 1e6
    assert row["kernel_union_ms"] == 60 / 1e6
    assert row["gpu_activity_union_ms"] == 85 / 1e6
    assert row["window_without_gpu_activity_ms"] == 15 / 1e6
    assert row["kernel_launch_correlation"] == {"cudaGraphLaunch": 2}
    assert row["gpu_transfers"]["1:cudaMemcpyAsync"]["bytes"] == 16
    assert "other_thread" not in row["cpu_cuda_api"]


def test_transient_peak_and_invalid_lifetime() -> None:
    """A temporary allocation affects the peak even when freed before readout."""
    events = [
        (0, 1, 100, 50, 2, 0),
        (1, 1, 200, 80, 2, 0),
        (2, 1, 200, 80, 2, 1),
        (3, 1, 100, 50, 2, 1),
    ]
    assert allocation_peak(events) == {
        "peak_tracked_device_bytes": 130,
        "retained_at_process_exit_bytes": 0,
    }
    with pytest.raises(ValueError, match="unmatched"):
        allocation_peak(events[1:])
    with pytest.raises(ValueError, match="duplicate"):
        allocation_peak(events[:1] * 2)
    assert (
        category("unpack_tile_kernel", 5, "qw38", False)
        == "weight_unpack_or_activation_packing"
    )
    assert category("unpack_tile_kernel", 5, "qw38", True) == "head"
    assert category("attention_prefill_scan_kernel", 5, "qw38", False) == "attention"
    assert category("gemv_kernel", 256, "qw38", False) == "projections"
    assert category("mmq", 256, "qw38", False) == "projections"
    assert category("mmq_j128", 40, "qw38", False) == "projections"


def telemetry_trace(path: Path) -> None:
    """Build repeated names, cloned graph nodes and a recaptured graph bucket."""
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1,'cudaLaunchKernel'),(2,'cudaGraphLaunch'),(3,'same_kernel'),(4,'cudaMemcpyAsync');
        CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
        INSERT INTO NVTX_EVENTS VALUES
          (0,800,7,'perf01_measured',NULL),
          (10,90,7,'qw38:phase name=prefill tokens=8 prefix=0',NULL),
          (20,60,7,'qw38:layer layer=0 kind=gdn m=8',NULL),
          (25,55,7,'qw38:op name=mlp',NULL),
          (30,50,7,'qw38:op name=projection role=gate m=8 n=16 k=32',NULL),
          (100,300,7,'qw38:phase name=decode position=8 populated=8 bucket=16 graph=build',NULL),
          (105,150,7,'qw38:phase name=graph_capture',NULL),
          (106,116,7,'qw38:layer layer=0 kind=gdn m=1',NULL),
          (108,115,7,'qw38:op name=projection role=gate m=1 n=16 k=32',NULL),
          (117,127,7,'qw38:layer layer=1 kind=attention m=1',NULL),
          (118,125,7,'qw38:op name=projection role=up m=1 n=16 k=32',NULL),
          (190,270,7,'qw38:host name=graph_first_launch',NULL),
          (270,290,7,'qw38:op name=readback',NULL),
          (400,500,7,'qw38:phase name=decode position=9 populated=9 bucket=16 graph=replay',NULL),
          (405,470,7,'qw38:host name=graph_replay',NULL),
          (600,750,7,'qw38:phase name=decode position=16 populated=16 bucket=32 graph=build',NULL),
          (605,650,7,'qw38:phase name=graph_capture',NULL),
          (606,616,7,'qw38:layer layer=0 kind=gdn m=1',NULL),
          (608,615,7,'qw38:op name=projection role=down m=1 n=32 k=16',NULL);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
          (35,40,7,1,1),(109,112,7,2,1),(119,122,7,3,1),(195,198,7,4,2),
          (275,278,7,5,4),(410,415,7,6,2),(609,612,7,7,1),(690,695,7,8,2);
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS (start INTEGER,end INTEGER,globalTid INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER);
        INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES
          (110,110,7,1,NULL),(120,120,7,2,NULL),
          (130,130,7,11,NULL),(131,131,7,11,1),(132,132,7,12,2),
          (610,610,7,3,NULL),(630,630,7,13,3);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,correlationId INTEGER,graphNodeId INTEGER,shortName INTEGER,registersPerThread INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
          (70,85,1,NULL,3,32),(200,240,4,11,3,64),(220,260,4,12,3,64),
          (420,450,6,11,3,64),(440,465,6,12,3,64),(700,720,8,13,3,32);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER,end INTEGER,bytes INTEGER,copyKind INTEGER,correlationId INTEGER,graphNodeId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (278,288,64,2,5,NULL);
    """)
    db.close()


def test_telemetry_causal_graph_identity_and_accounting(tmp_path: Path) -> None:
    """Repeated names retain layer, role and current replay position/bucket."""
    path = tmp_path / "candidate.sqlite"
    telemetry_trace(path)
    report = telemetry(path)
    rows = [r for r in report["operators"] if r["kernel_count"]]
    replay = [r for r in rows if r["context"].get("graph") == "replay"]
    assert len(replay) == 2
    assert {r["context"]["layer"] for r in replay} == {"0", "1"}
    assert {r["context"]["role"] for r in replay} == {"gate", "up"}
    assert all(r["context"]["position"] == "9" for r in replay)
    assert all(r["context"]["populated"] == "9" for r in replay)
    assert all(r["context"]["stage"] == "decode" for r in replay)
    assert all(r["dispatch"][0]["mapping"] == "graph_node" for r in replay)
    later = next(r for r in rows if r["context"].get("bucket") == "32")
    assert later["context"]["role"] == "down"
    assert later["context"]["n"] == "32"
    prefill = next(r for r in rows if r["context"]["phase"] == "prefill")
    assert prefill["context"]["operator_path"] == "mlp/projection"
    assert prefill["kernel_sum_ms"] == 15 / 1e6
    assert report["totals"]["kernel_sum_ms"] == 170 / 1e6
    assert report["totals"]["kernel_union_ms"] == 140 / 1e6
    assert report["totals"]["gpu_activity_union_ms"] == 150 / 1e6
    assert report["totals"]["gpu_overlap_ms"] == 30 / 1e6
    assert report["totals"]["transfer_bytes"] == 64
    first = next(r for r in report["windows"] if r.get("graph") == "build")
    assert first["kernel_union_ms"] == 60 / 1e6
    assert first["host_wall_ms"] == 200 / 1e6
    assert sum(r["kernel_count"] for r in rows) == report["totals"]["kernel_count"]
    assert sum(r["kernel_sum_ms"] for r in rows) == pytest.approx(
        report["totals"]["kernel_sum_ms"]
    )
    assert report["graph_mapping"]["replayed_activities"] == 5
    assert report["unmapped_inference"] == []


def test_telemetry_existing_setup_ranges_retain_api_costs(tmp_path: Path) -> None:
    """Legacy setup boundaries contribute named host/API diagnostics."""
    path = tmp_path / "candidate.sqlite"
    telemetry_trace(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (0,9,7,'qw38:phase name=setup',NULL)"
        )
        db.execute("INSERT INTO NVTX_EVENTS VALUES (0,8,7,'cuda_initialization',NULL)")
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES (1,7,7,'unrelated_legacy_label',NULL)"
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,6,7,99,4)")
    report = telemetry(path)
    startup = next(w for w in report["windows"] if w["name"] == "cuda_initialization")
    assert startup["kind"] == "host"
    assert startup["host_wall_ms"] == 8 / 1e6
    assert startup["cpu_cuda_api_union_ms"] == 5 / 1e6
    assert startup["host_outside_cuda_api_ms"] == pytest.approx(3 / 1e6)
    assert not any(w["name"] == "unrelated_legacy_label" for w in report["windows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM CUDA_GRAPH_NODE_EVENTS WHERE graphNodeId=1",
        "UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET correlationId=999",
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=999 WHERE graphNodeId=12",
        "DELETE FROM NVTX_EVENTS WHERE text LIKE 'qw38:op%'",
    ],
)
def test_telemetry_rejects_missing_owners(tmp_path: Path, mutation: str) -> None:
    """Missing node/operator/copy metadata fails instead of guessing by name."""
    path = tmp_path / "candidate.sqlite"
    telemetry_trace(path)
    with sqlite3.connect(path) as db:
        db.execute(mutation)
    with pytest.raises(ValueError, match="unmapped inference"):
        telemetry(path)
    assert telemetry(path, strict=False)["unmapped_inference"]


def test_telemetry_comparison_keeps_host_union_and_sum_separate(tmp_path: Path) -> None:
    """A changed family reconciles to kernel sum without inventing wall savings."""
    path = tmp_path / "candidate.sqlite"
    telemetry_trace(path)
    records = [
        {
            "kind": "setup",
            "capacity_requested": 32,
            "capacity_allocated": 32,
            "chunk": 8,
            "graph_mode": "graph",
            "profiling_enabled": True,
            "telemetry_schema": 1,
            "manifest_digest": "parent",
        },
        {
            "kind": "sample",
            "row": "request-8",
            "final_position": 17,
            "steps_ms": [0.1, 0.1, 0.1],
            "total_ms": 0.8,
            "ttft_ms": 0.08,
            "tail_ms": 0.72,
        },
    ]
    path.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in records))
    baseline = telemetry(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=end+5 WHERE start=700")
    records[0]["manifest_digest"] = "candidate"
    path.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in records))
    current = telemetry(path)
    result = compare_telemetry(current, baseline)
    assert result["host_phase_deltas"]["request"]["delta_ms"] == 0
    assert result["host_phase_deltas"]["outside_inference_phases"][
        "current_ms"
    ] == pytest.approx(270 / 1e6)
    assert result["measured_gpu_deltas"]["kernel_sum_ms"]["delta_ms"] == pytest.approx(
        5 / 1e6
    )
    assert sum(r["delta_ms"] for r in result["family_kernel_deltas"]) == pytest.approx(
        5 / 1e6
    )
    assert result["measured_gpu_deltas"]["gpu_overlap_ms"]["delta_ms"] == 0
    for field, value in (("capacity_requested", 64), ("profiling_enabled", False)):
        changed = json.loads(json.dumps(current))
        changed["benchmark_records"][0][field] = value
        with pytest.raises(ValueError, match=field):
            compare_telemetry(changed, baseline)
    current["benchmark_records"][1]["steps_ms"].pop()
    with pytest.raises(ValueError, match="step count"):
        compare_telemetry(current, baseline)
