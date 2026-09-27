"""Check transient allocation accounting and exclusive kernel attribution."""

import sys
import sqlite3
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task027_profiles import allocation_peak, category, range_costs


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
