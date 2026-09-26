# /// script
# requires-python = "==3.12.*"
# ///
"""Analyze the existing single-run traces without launching more inference."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(".cache/task027/single-run")


def allocation_peak(events: list[tuple]) -> dict:
    """Count unique live device allocations, including temporary allocations."""
    live = {}
    total = peak = 0
    for _, context, address, size, kind, operation in events:
        if kind not in (2, 5):
            continue
        key = (context, address, kind)
        if operation == 0:
            if key in live:
                raise ValueError("duplicate live device allocation")
            live[key] = size
            total += size
            peak = max(peak, total)
        elif operation == 1:
            if live.pop(key, None) != size:
                raise ValueError("unmatched allocation/free evidence")
            total -= size
        else:
            raise ValueError("unknown allocation event")
    return {"peak_tracked_device_bytes": peak, "retained_at_process_exit_bytes": total}


def category(name: str, grid_x: int, engine: str, head: bool) -> str:
    """Assign each GPU kernel once, separating conversion from contractions."""
    if (
        head
        or (engine == "qw38" and name == "decode_mmv_kernel" and grid_x == 31040)
        or (engine == "llama" and name == "mul_mat_vec_q" and grid_x == 248320)
    ):
        return "head"
    if "unpack_tile" in name or "quantize" in name:
        return "weight_unpack_or_activation_packing"
    if "attention" in name or "flash_attn" in name or "fattn" in name:
        return "attention"
    if (
        "gdn_" in name
        or "gated_delta" in name
        or "ssm_conv" in name
        or "conv_history" in name
    ):
        return "gdn"
    if (
        "mmv" in name
        or "mma" in name
        or "gemm" in name.lower()
        or "mul_mat" in name
        or "splitK" in name
    ):
        return "projections"
    return "normalization_epilogue_other"


def trace(path: Path) -> dict:
    """Extract same-run kernels, CPU API costs, transfers and memory peaks."""
    engine = path.stem.rsplit("-", 1)[1]
    records = [
        json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()
    ]
    setup, sample, _ = records
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    ranges = db.execute(
        "SELECT start,end FROM NVTX_EVENTS WHERE text='perf01_measured'"
    ).fetchall()
    if len(ranges) != 1:
        raise ValueError("expected exactly one measured range")
    start, end = ranges[0]
    phase_end = start + round(sample["ttft_ms"] * 1e6)
    # The candidate prefill readout is the final single-row hidden RMS followed
    # by head unpack/GEMM/epilogue. No layer projections follow that RMS.
    head_start = None
    if engine == "qw38" and sample["ttft_ms"]:
        head_start = db.execute(
            "SELECT max(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
            "WHERE k.start>=? AND k.end<=? AND s.value='hidden_rms_kernel' AND k.gridX=1",
            (start, phase_end),
        ).fetchone()[0]
        if head_start is None:
            raise ValueError("missing prefill head boundary")
    kernel_ns = defaultdict(int)
    kernel_count = defaultdict(int)
    phases = {"prefill": defaultdict(int), "decode": defaultdict(int)}
    names = defaultdict(int)
    union_ns = 0
    union_end = start
    for begin, finish, name, grid in db.execute(
        "SELECT k.start,k.end,s.value,k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.shortName WHERE k.start>=? AND k.end<=? ORDER BY k.start",
        (start, end),
    ):
        kind = category(
            name,
            grid,
            engine,
            head_start is not None and head_start <= begin < phase_end,
        )
        duration = finish - begin
        kernel_ns[kind] += duration
        kernel_count[kind] += 1
        phases["prefill" if sample["ttft_ms"] and begin < phase_end else "decode"][
            kind
        ] += duration
        names[name] += duration
        union_ns += max(0, finish - max(begin, union_end))
        union_end = max(union_end, finish)
    api = {}
    for name, count, duration in db.execute(
        "SELECT s.value,count(*),sum(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId WHERE r.start>=? AND r.end<=? GROUP BY s.value",
        (start, end),
    ):
        api[name] = {"calls": count, "cpu_ms": duration / 1e6}
    transfers = {}
    for kind, count, size, duration in db.execute(
        "SELECT copyKind,count(*),sum(bytes),sum(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE start>=? AND end<=? GROUP BY copyKind",
        (start, end),
    ):
        transfers[str(kind)] = {"calls": count, "bytes": size, "gpu_ms": duration / 1e6}
    events = db.execute(
        "SELECT start,contextId,address,bytes,memKind,memoryOperationType "
        "FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"
    ).fetchall()
    memory = allocation_peak(events)
    resident = setup["initial_free_bytes"] - sample["free_bytes"]
    memory.update(
        observed_resident_growth_bytes=resident,
        free_after_run_bytes=sample["free_bytes"],
        reserve_bytes=2147483648,
        reserve_satisfied=sample["free_bytes"] >= 2147483648,
        categories_from_adapter={
            key: setup[key]
            for key in ("model_device_bytes", "persistent_bytes", "scratch_bytes")
            if key in setup
        },
        scope="peak tracks CUDA allocation events across the entire process including setup/transients; resident growth is cudaMemGetInfo after the run and also includes allocator/context overhead, not an exact allocation sum",
    )
    db.close()
    return {
        "row": sample["row"],
        "engine": engine,
        "nvtx_window_ms": (end - start) / 1e6,
        "host_measured_ms": sample["total_ms"],
        "kernel_ms": {
            key: value / 1e6
            for key, value in sorted(kernel_ns.items(), key=lambda item: -item[1])
        },
        "kernel_calls": dict(kernel_count),
        "kernel_busy_union_ms": union_ns / 1e6,
        "phase_kernel_ms": {
            phase: {key: value / 1e6 for key, value in costs.items()}
            for phase, costs in phases.items()
        },
        "top_kernel_ms": dict(
            sorted(
                ((key, value / 1e6) for key, value in names.items()),
                key=lambda item: -item[1],
            )[:20]
        ),
        "cpu_cuda_api": api,
        "gpu_transfers": transfers,
        "memory": memory,
        "accounting": "each kernel assigned once; graph parent durations excluded; CPU waits/launches and transfers are separate diagnostics, never added to GPU kernel time or host latency; phase boundary uses recorded host TTFT relative to NVTX start",
    }


def main() -> None:
    """Summarize all twelve existing traces and compare memory by workload."""
    paths = sorted(ROOT.glob("*.sqlite"))
    if len(paths) != 12:
        raise ValueError("twelve single-run traces required")
    traces = {path.stem: trace(path) for path in paths}
    if any(not row["memory"]["reserve_satisfied"] for row in traces.values()):
        raise ValueError("GPU memory reserve not satisfied")
    memory_ratios = {}
    for key, row in traces.items():
        if row["engine"] != "qw38":
            continue
        other = traces[key.removesuffix("qw38") + "llama"]
        memory_ratios[row["row"]] = {
            "llama_over_qw38_peak_allocation_ratio": other["memory"][
                "peak_tracked_device_bytes"
            ]
            / row["memory"]["peak_tracked_device_bytes"],
            "llama_over_qw38_observed_resident_ratio": other["memory"][
                "observed_resident_growth_bytes"
            ]
            / row["memory"]["observed_resident_growth_bytes"],
        }
    output = {"traces": traces, "memory_ratios": memory_ratios}
    (ROOT / "profiles.json").write_text(json.dumps(output, indent=2) + "\n")
    for key, row in traces.items():
        print(
            key,
            row["kernel_ms"],
            row["memory"]["peak_tracked_device_bytes"],
            flush=True,
        )


if __name__ == "__main__":
    main()
