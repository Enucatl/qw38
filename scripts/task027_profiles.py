# /// script
# requires-python = "==3.12.*"
# ///
"""Analyze the existing single-run traces without launching more inference."""

from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(".cache/task027/single-run")


def interval_union(intervals: list[tuple[int, int]]) -> int:
    """Return covered nanoseconds without double counting overlapping intervals.

    Args:
        intervals: Nonnegative trace start/end timestamp pairs.

    Returns:
        Total duration covered by at least one interval.
    """
    total = end = 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def range_costs(db: sqlite3.Connection, start: int, end: int, tid: int) -> dict:
    """Describe one NVTX window, retaining CPU/GPU overlap and API correlation.

    GPU totals describe activity inside the window, not work causally charged
    to setup stages. Activities crossing a boundary are clipped for unions.
    API sums can include nested calls; only the union is subtracted from wall
    time. Uncovered host time is unclassified, not necessarily CPU execution.

    Args:
        db: Nsight SQLite connection for one benchmark process.
        start: Inclusive NVTX start timestamp in nanoseconds.
        end: Exclusive NVTX end timestamp in nanoseconds.
        tid: NVTX range's global thread identifier.

    Returns:
        CPU/API and GPU costs, correlations, transfers and uncovered intervals.
    """
    api = defaultdict(lambda: {"calls": 0, "cpu_ms": 0.0})
    api_intervals = []
    for begin, finish, name in db.execute(
        "SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId "
        "WHERE r.start<? AND r.end>? AND r.globalTid=?",
        (end, start, tid),
    ):
        begin, finish = max(begin, start), min(finish, end)
        api[name]["calls"] += 1
        api[name]["cpu_ms"] += (finish - begin) / 1e6
        api_intervals.append((begin, finish))
    kernels = db.execute(
        "SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start<? AND end>? ORDER BY start",
        (end, start),
    ).fetchall()
    kernel_intervals = [(max(a, start), min(b, end)) for a, b, _ in kernels]
    correlations = dict(
        db.execute(
            "SELECT r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId WHERE r.globalTid=?",
            (tid,),
        )
    )
    launches = defaultdict(int)
    for _, _, correlation in kernels:
        launches[correlations.get(correlation, "unmatched")] += 1
    transfers = defaultdict(lambda: {"calls": 0, "bytes": 0, "gpu_ms": 0.0})
    activity_intervals = list(kernel_intervals)
    for begin, finish, size, kind, correlation in db.execute(
        "SELECT start,end,bytes,copyKind,correlationId "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?",
        (end, start),
    ):
        begin, finish = max(begin, start), min(finish, end)
        key = f"{kind}:{correlations.get(correlation, 'unmatched')}"
        transfers[key]["calls"] += 1
        transfers[key]["bytes"] += size
        transfers[key]["gpu_ms"] += (finish - begin) / 1e6
        activity_intervals.append((begin, finish))
    memset_intervals = [
        (max(a, start), min(b, end))
        for a, b in db.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start<? AND end>?",
            (end, start),
        )
    ]
    activity_intervals.extend(memset_intervals)
    api_union = interval_union(api_intervals)
    gpu_union = interval_union(activity_intervals)
    return {
        "start_ns": start,
        "end_ns": end,
        "nvtx_ms": (end - start) / 1e6,
        "cpu_cuda_api": dict(api),
        "cpu_cuda_api_union_ms": api_union / 1e6,
        "host_outside_cuda_api_ms": (end - start - api_union) / 1e6,
        "kernel_count": len(kernels),
        "kernel_sum_ms": sum(b - a for a, b in kernel_intervals) / 1e6,
        "kernel_union_ms": interval_union(kernel_intervals) / 1e6,
        "kernel_launch_correlation": dict(launches),
        "gpu_transfers": dict(transfers),
        "gpu_memset_ms": sum(b - a for a, b in memset_intervals) / 1e6,
        "gpu_activity_union_ms": gpu_union / 1e6,
        "window_without_gpu_activity_ms": (end - start - gpu_union) / 1e6,
        "first_kernel_after_start_ms": (kernel_intervals[0][0] - start) / 1e6
        if kernels
        else None,
        "last_kernel_before_end_ms": (end - max(b for _, b in kernel_intervals)) / 1e6
        if kernels
        else None,
    }


def decode_attribution(path: Path) -> dict:
    """Extract startup ranges and the first eight decode steps of a saved run.

    Args:
        path: SQLite trace path with a sibling benchmark JSONL file.

    Returns:
        Host measurements and matching setup/token range diagnostics.

    Raises:
        ValueError: Startup ranges repeat or eight complete steps are missing.
    """
    records = [
        json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()
    ]
    setup = next(row for row in records if row["kind"] == "setup")
    sample = next(row for row in records if row["kind"] == "sample")
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        startup = {}
        for name in (
            "cuda_initialization",
            "runtime_create",
            "model_load_upload",
            "session_create",
            "plan_bind",
            "initial_prefill",
        ):
            rows = db.execute(
                "SELECT start,end,globalTid FROM NVTX_EVENTS WHERE text=?", (name,)
            ).fetchall()
            if len(rows) > 1:
                raise ValueError(f"duplicate startup range: {name}")
            if rows:
                startup[name] = range_costs(db, *rows[0])
        tokens = []
        for start, end, tid in db.execute(
            "SELECT start,end,globalTid FROM NVTX_EVENTS WHERE text='decode_token' ORDER BY start LIMIT 8"
        ).fetchall():
            row = range_costs(db, start, end, tid)
            row["detail"] = [
                r[0]
                for r in db.execute(
                    "SELECT text FROM NVTX_EVENTS WHERE start>=? AND end<=? AND globalTid=? AND text LIKE 'decode_token position=%'",
                    (start, end, tid),
                )
            ]
            tokens.append(row)
        if len(tokens) != 8 or len(sample["steps_ms"]) < 8:
            raise ValueError("eight completed decode steps required")
        return {
            "path": str(path),
            "setup": setup,
            "sample": sample,
            "startup_ranges": startup,
            "first_eight_decode": tokens,
            "accounting": "CPU API sums may nest; GPU sums may overlap. Unions cover the NVTX window. CPU waits overlap GPU execution. Uncovered host time is unclassified. Transfer bytes count intersecting operations in full. Never add these diagnostics to host latency.",
        }
    finally:
        db.close()


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
    if "unpack_tile" in name or "quantize" in name or name == "pack_kernel":
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
        name == "device_kernel"  # CUTLASS NVFP4/FP8; confirmed by full kernel name.
        or name in {"mmq", "mmq_j128"}  # Q4_K x Q8 integer prefill contractions.
        or "mmv" in name
        or "gemv" in name
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


def marker_context(rows: list[dict], points: list[tuple]) -> dict:
    """Resolve nested, thread-local NVTX metadata at launch or node creation.

    Args:
        rows: Parsed ranges, with start/end/tid, marker kind and fields.
        points: (key, timestamp, thread) tuples to resolve.

    Returns:
        Metadata by point key; operator paths preserve fused semantic parents.
    """
    ranges = defaultdict(list)
    queries = defaultdict(list)
    result = {}
    for row in rows:
        ranges[row["tid"]].append(row)
    for key, stamp, tid in points:
        queries[tid].append((stamp, key))
    for tid, query in queries.items():
        ordered = sorted(ranges[tid], key=lambda r: (r["start"], -r["end"]))
        active = []
        index = 0
        for stamp, key in sorted(query):
            while index < len(ordered) and ordered[index]["start"] <= stamp:
                active.append(ordered[index])
                index += 1
            active = [r for r in active if stamp < r["end"]]
            fields = {}
            operators = []
            hosts = []
            for row in active:
                data = row["fields"].copy()
                name = data.pop("name", None)
                if row["kind"] == "phase":
                    fields["stage"] = name
                    if name in ("prefill", "decode") or "phase" not in fields:
                        fields["phase"] = name
                elif row["kind"] == "op":
                    operators.append(name)
                elif row["kind"] == "host":
                    hosts.append(name)
                fields.update(data)
            if operators:
                fields["operator"] = operators[-1]
                fields["operator_path"] = "/".join(operators)
            if hosts:
                fields["host"] = "/".join(hosts)
            result[key] = fields
    return result


def telemetry(path: Path, strict: bool = True) -> dict:
    """Attribute production work through launch and actual graph-node identity.

    Args:
        path: Nsight Systems SQLite export with qw38 NVTX markers.
        strict: Reject missing phase/operator ownership of inference activity.

    Returns:
        Lifecycle windows, exclusive operator costs, dispatch and memory data.

    Raises:
        ValueError: Required metadata or causal graph attribution is missing.
    """
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}

        def read(table: str) -> list[dict]:
            """Read an optional activity table from this exporter version."""
            return (
                [dict(r) for r in db.execute(f"SELECT * FROM {table}")]
                if table in tables
                else []
            )

        strings = {r["id"]: r["value"] for r in read("StringIds")}
        markers = []
        measured = []
        measured_markers = []
        for row in read("NVTX_EVENTS"):
            label = row.get("text") or strings.get(row.get("textId"), "")
            if label in (
                "cuda_initialization",
                "runtime_create",
                "model_load_upload",
                "session_create",
                "plan_bind",
            ):
                label = f"qw38:host name={label}"
            if label == "perf01_measured" and row["end"] is not None:
                measured.append((row["start"], row["end"]))
                measured_markers.append(
                    {
                        "start": row["start"],
                        "end": row["end"],
                        "tid": row["globalTid"],
                        "kind": "measured",
                        "fields": {"name": label},
                    }
                )
            if not label.startswith("qw38:"):
                continue
            if row["end"] is None:
                raise ValueError(f"incomplete telemetry range: {label}")
            words = shlex.split(label)
            kind = words[0].removeprefix("qw38:")
            fields = dict(word.split("=", 1) for word in words[1:])
            if kind not in ("phase", "op", "layer", "host"):
                raise ValueError(f"unknown telemetry marker: {label}")
            if kind != "layer" and not fields.get("name"):
                raise ValueError(f"missing telemetry name: {label}")
            markers.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "tid": row["globalTid"],
                    "kind": kind,
                    "fields": fields,
                }
            )
        if not markers or not any(
            r["kind"] == "phase" and r["fields"]["name"] in ("prefill", "decode")
            for r in markers
        ):
            raise ValueError(
                "missing qw38 inference phase markers; recapture with telemetry enabled"
            )
        apis = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            for row in read(table):
                row["name"] = strings.get(row["nameId"], str(row["nameId"]))
                row["source"] = table
                apis.append(row)
        nodes = read("CUDA_GRAPH_NODE_EVENTS")
        points = [(("api", i), r["start"], r["globalTid"]) for i, r in enumerate(apis)]
        points += [
            (("node", i), r["start"], r["globalTid"]) for i, r in enumerate(nodes)
        ]
        contexts = marker_context(markers, points)
        correlations = defaultdict(list)
        for i, api in enumerate(apis):
            api["context"] = contexts[("api", i)]
            correlations[api["correlationId"]].append(api)
        originals = {}
        node_context = {}
        for i, node in enumerate(nodes):
            node_id = node["graphNodeId"]
            if node.get("originalGraphNodeId"):
                originals[node_id] = node["originalGraphNodeId"]
            context = contexts[("node", i)]
            if context.get("operator"):
                node_context[node_id] = context

        def captured_context(node_id: int) -> dict:
            """Follow explicit clone links back to the original captured node."""
            seen = set()
            while node_id in originals:
                if node_id in seen:
                    raise ValueError("cyclic CUDA graph clone identity")
                seen.add(node_id)
                node_id = originals[node_id]
            return node_context.get(node_id, {})

        activities = []
        unmapped = []
        inference_windows = measured + [
            (r["start"], r["end"])
            for r in markers
            if r["kind"] == "phase" and r["fields"]["name"] in ("prefill", "decode")
        ]
        for kind, table in (
            ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
        ):
            for row in read(table):
                matches = correlations.get(row.get("correlationId"), [])
                # CUPTI correlation IDs are process-local, not thread-local.
                if row.get("globalPid") is not None:
                    matches = [
                        a
                        for a in matches
                        if a["globalTid"] is not None
                        and (a["globalTid"] >> 24) == (row["globalPid"] >> 24)
                    ]
                matches = [a for a in matches if a["start"] <= row["start"]]
                api = max(
                    matches, key=lambda a: (len(a["context"]), a["start"]), default=None
                )
                context = dict(api["context"]) if api else {}
                graph_node = row.get("graphNodeId")
                mapping = "launch"
                if graph_node:
                    captured = captured_context(graph_node)
                    # Shapes/layers/operators come from capture. Execution phase,
                    # position and bucket always come from this graph launch.
                    context = {**captured, **context}
                    for key in (
                        "operator",
                        "operator_path",
                        "layer",
                        "kind",
                        "m",
                        "n",
                        "k",
                        "schedule",
                    ):
                        if key in captured:
                            context[key] = captured[key]
                    mapping = "graph_node"
                    if not captured:
                        context.pop("operator", None)
                row.update(
                    kind=kind,
                    context=context,
                    mapping=mapping,
                    api=api["name"] if api else None,
                )
                row["name"] = strings.get(
                    row.get("demangledName"), strings.get(row.get("shortName"), kind)
                )
                inference = context.get("phase") in ("prefill", "decode") or any(
                    a <= row["start"] < b for a, b in inference_windows
                )
                if inference and (
                    context.get("phase") not in ("prefill", "decode")
                    or not context.get("operator")
                    or (graph_node and not api)
                ):
                    unmapped.append(
                        {
                            "kind": kind,
                            "name": row["name"],
                            "start_ns": row["start"],
                            "graph_node_id": graph_node,
                            "correlation_id": row.get("correlationId"),
                            "context": context,
                        }
                    )
                activities.append(row)
        if strict and unmapped:
            raise ValueError(
                f"{len(unmapped)} unmapped inference activities; first: {json.dumps(unmapped[0])}"
            )

        def activity_costs(rows: list[dict]) -> dict:
            """Keep overlapping device sums distinct from elapsed coverage."""
            spans = [(r["start"], r["end"]) for r in rows]
            kernels = [(r["start"], r["end"]) for r in rows if r["kind"] == "kernel"]
            total = sum(b - a for a, b in spans)
            union = interval_union(spans)
            return {
                "activity_count": len(rows),
                "kernel_count": len(kernels),
                "kernel_sum_ms": sum(b - a for a, b in kernels) / 1e6,
                "kernel_union_ms": interval_union(kernels) / 1e6,
                "gpu_activity_sum_ms": total / 1e6,
                "gpu_activity_union_ms": union / 1e6,
                "gpu_overlap_ms": (total - union) / 1e6,
                "transfer_bytes": sum(
                    r.get("bytes", 0) for r in rows if r["kind"] == "memcpy"
                ),
                "memcpy_ms": sum(
                    r["end"] - r["start"] for r in rows if r["kind"] == "memcpy"
                )
                / 1e6,
                "memset_ms": sum(
                    r["end"] - r["start"] for r in rows if r["kind"] == "memset"
                )
                / 1e6,
            }

        groups = defaultdict(list)
        api_groups = defaultdict(list)
        for row in activities:
            groups[json.dumps(row["context"], sort_keys=True)].append(row)
        for api in apis:
            if api["context"]:
                api_groups[json.dumps(api["context"], sort_keys=True)].append(api)
        phase_sums = defaultdict(int)
        for row in activities:
            if row["kind"] == "kernel":
                phase_sums[row["context"].get("phase")] += row["end"] - row["start"]
        operators = []
        resource_keys = (
            "registersPerThread",
            "staticSharedMemory",
            "dynamicSharedMemory",
            "localMemoryPerThread",
            "localMemoryTotal",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
        )
        for key in groups.keys() | api_groups.keys():
            rows = groups[key]
            context = json.loads(key)
            costs = activity_costs(rows)
            dispatch = defaultdict(list)
            for row in rows:
                identity = {
                    "name": row["name"],
                    "kind": row["kind"],
                    "launch_api": row["api"],
                    "mapping": row["mapping"],
                    "copy_kind": row.get("copyKind"),
                    "resources": {k: row[k] for k in resource_keys if k in row},
                }
                dispatch[json.dumps(identity, sort_keys=True)].append(row)
            calls = defaultdict(list)
            for api in api_groups[key]:
                calls[api["name"]].append((api["start"], api["end"]))
            costs["cpu_cuda_api"] = {
                name: {
                    "calls": len(spans),
                    "sum_ms": sum(b - a for a, b in spans) / 1e6,
                    "union_ms": interval_union(spans) / 1e6,
                }
                for name, spans in calls.items()
            }
            costs["phase_kernel_share_percent"] = (
                100 * costs["kernel_sum_ms"] * 1e6 / phase_sums[context.get("phase")]
                if phase_sums[context.get("phase")]
                else 0.0
            )
            operators.append(
                {
                    "context": context,
                    **costs,
                    "dispatch": [
                        {**json.loads(identity), **activity_costs(items)}
                        for identity, items in dispatch.items()
                    ],
                }
            )
        operators.sort(key=lambda r: -r["gpu_activity_sum_ms"])

        def index_intervals(rows: list[dict]) -> tuple:
            """Index starts and prefix maximum ends for overlap-safe slicing."""
            ordered = sorted(rows, key=lambda r: r["start"])
            ends = []
            for row in ordered:
                ends.append(max(row["end"], ends[-1] if ends else 0))
            return ordered, [r["start"] for r in ordered], ends

        def intersect(index: tuple, start: int, end: int) -> list[dict]:
            """Return intervals intersecting a half-open measurement window."""
            rows, starts, ends = index
            return [
                r
                for r in rows[bisect_right(ends, start) : bisect_left(starts, end)]
                if r["end"] > start
            ]

        activity_index = index_intervals(activities)
        thread_apis = defaultdict(list)
        for api in apis:
            thread_apis[api["globalTid"]].append(api)
        api_indexes = {tid: index_intervals(rows) for tid, rows in thread_apis.items()}
        windows = []
        for marker in markers + measured_markers:
            if marker["kind"] not in ("phase", "host", "measured"):
                continue
            start, end = marker["start"], marker["end"]
            spans = [
                (max(start, a["start"]), min(end, a["end"]))
                for a in intersect(
                    api_indexes.get(marker["tid"], ([], [], [])), start, end
                )
            ]
            clipped = [
                {**a, "start": max(start, a["start"]), "end": min(end, a["end"])}
                for a in intersect(activity_index, start, end)
            ]
            costs = activity_costs(clipped)
            wall = (end - start) / 1e6
            windows.append(
                {
                    "kind": marker["kind"],
                    **marker["fields"],
                    "start_ns": start,
                    "end_ns": end,
                    "overlaps_measured": any(
                        a < end and b > start for a, b in measured
                    ),
                    "host_wall_ms": wall,
                    "cpu_cuda_api_union_ms": interval_union(spans) / 1e6,
                    "host_outside_cuda_api_ms": wall - interval_union(spans) / 1e6,
                    "window_without_gpu_activity_ms": wall
                    - costs["gpu_activity_union_ms"],
                    **costs,
                }
            )
        for window in windows:
            if window["kind"] == "measured":
                spans = [
                    (
                        max(window["start_ns"], r["start"]),
                        min(window["end_ns"], r["end"]),
                    )
                    for r in markers
                    if r["kind"] == "phase"
                    and r["fields"]["name"] in ("prefill", "decode")
                    and r["start"] < window["end_ns"]
                    and r["end"] > window["start_ns"]
                ]
                window["host_inference_phase_union_ms"] = interval_union(spans) / 1e6
                window["host_outside_inference_phases_ms"] = (
                    window["host_wall_ms"] - interval_union(spans) / 1e6
                )
        families = defaultdict(list)
        inference = []
        measured_activities = []
        for row in activities:
            context = row["context"]
            if context.get("phase") in ("prefill", "decode"):
                inference.append(row)
                family = context.get("operator_path", "unmapped").split("/")[0]
                families[
                    (context["phase"], context.get("graph", "eager"), family)
                ].append(row)
            if any(start <= row["start"] < end for start, end in measured):
                measured_activities.append(row)
        family_costs = [
            {"phase": phase, "graph": graph, "family": family, **activity_costs(rows)}
            for (phase, graph, family), rows in families.items()
        ]
        family_costs.sort(key=lambda r: -r["kernel_sum_ms"])
        measured_families = defaultdict(list)
        for row in measured_activities:
            context = row["context"]
            family = context.get("operator_path", "unmapped").split("/")[0]
            measured_families[
                (context.get("phase"), context.get("graph", "eager"), family)
            ].append(row)
        memory_events = read("CUDA_GPU_MEMORY_USAGE_EVENTS")
        memory = (
            allocation_peak(
                sorted(
                    [
                        (
                            r["start"],
                            r["contextId"],
                            r["address"],
                            r["bytes"],
                            r["memKind"],
                            r["memoryOperationType"],
                        )
                        for r in memory_events
                    ],
                    key=lambda r: r[0],
                )
            )
            if memory_events
            else {"available": False}
        )
        jsonl = path.with_suffix(".jsonl")
        records = (
            [
                json.loads(line)
                for line in jsonl.read_text().splitlines()
                if line.strip()
            ]
            if jsonl.exists()
            else []
        )
        return {
            "schema_version": 1,
            "path": str(path),
            "benchmark_records": records,
            "totals": activity_costs(activities),
            "inference_totals": activity_costs(inference),
            "measured_totals": activity_costs(measured_activities)
            if measured
            else None,
            "family_costs": family_costs,
            "measured_family_costs": [
                {
                    "phase": phase,
                    "graph": graph,
                    "family": family,
                    **activity_costs(rows),
                }
                for (phase, graph, family), rows in measured_families.items()
            ],
            "windows": windows,
            "operators": operators,
            "memory": memory,
            "graph_mapping": {
                "creation_events": len(nodes),
                "clone_links": len(originals),
                "captured_operator_nodes": len(node_context),
                "replayed_activities": sum(
                    a["mapping"] == "graph_node" for a in activities
                ),
            },
            "unmapped_inference": unmapped,
            "accounting": "totals cover all traced activity; inference_totals cover causal prefill/decode; measured_totals select activity starts in perf01_measured and exclude initial population for decode-only benchmarks. Operator rows exclusively own complete causal activities; parent operator paths are labels, never extra costs. Window GPU/API unions are timestamp-clipped and may overlap nested windows; never sum windows. Kernel/activity sums may overlap; unions measure occupied time. Host outside CUDA APIs is unclassified, not CPU execution. CUDA event intervals in benchmark records are elapsed intervals, not kernel time. Captured graph metadata supplies operator identity; actual launch supplies execution phase/token. Transfer bytes count intersecting operations in full.",
        }
    finally:
        db.close()


def compare_telemetry(current: dict, baseline: dict) -> dict:
    """Compare matched measured requests, preserving host/GPU timing semantics.

    Args:
        current: Candidate telemetry report.
        baseline: Earlier telemetry report from the same workload/profile mode.

    Returns:
        Current-minus-baseline host phase and exclusive family kernel changes.

    Raises:
        ValueError: Workload, capacity, step count or profiling modes differ.
    """

    def record(report: dict, kind: str) -> dict:
        """Require the single-run benchmark record used for comparison."""
        rows = [r for r in report["benchmark_records"] if r.get("kind") == kind]
        if len(rows) != 1:
            raise ValueError(f"comparison requires one {kind} benchmark record")
        return rows[0]

    setup = record(current, "setup")
    old_setup = record(baseline, "setup")
    sample = record(current, "sample")
    old_sample = record(baseline, "sample")
    for name in (
        "capacity_requested",
        "capacity_allocated",
        "chunk",
        "graph_mode",
        "profiling_enabled",
        "telemetry_schema",
    ):
        if name not in setup or name not in old_setup or setup[name] != old_setup[name]:
            raise ValueError(f"incompatible telemetry comparison: {name}")
    for name in ("row", "final_position"):
        if (
            name not in sample
            or name not in old_sample
            or sample[name] != old_sample[name]
        ):
            raise ValueError(f"incompatible telemetry comparison: {name}")
    if len(sample["steps_ms"]) != len(old_sample["steps_ms"]):
        raise ValueError("incompatible telemetry comparison: decode step count")
    if not current.get("measured_totals") or not baseline.get("measured_totals"):
        raise ValueError("comparison requires perf01_measured windows")

    def delta(new: float, old: float) -> dict:
        """Retain absolute costs alongside their signed change."""
        return {"current_ms": new, "baseline_ms": old, "delta_ms": new - old}

    def phases(report: dict) -> dict:
        """Sum measured phase host walls without adding nested chunk ranges."""
        result = defaultdict(float)
        for row in report["windows"]:
            if (
                row["kind"] == "phase"
                and row["name"] in ("prefill", "decode")
                and row["overlaps_measured"]
            ):
                result[row["name"]] += row["host_wall_ms"]
            elif row["kind"] == "measured":
                result["request"] += row["host_wall_ms"]
                result["outside_inference_phases"] += row[
                    "host_outside_inference_phases_ms"
                ]
        return result

    new_phases, old_phases = phases(current), phases(baseline)
    phase_changes = {
        name: delta(new_phases[name], old_phases[name])
        for name in new_phases.keys() | old_phases.keys()
    }
    new_families = {
        (r["phase"], r["graph"], r["family"]): r
        for r in current["measured_family_costs"]
    }
    old_families = {
        (r["phase"], r["graph"], r["family"]): r
        for r in baseline["measured_family_costs"]
    }
    families = []
    for key in new_families.keys() | old_families.keys():
        new, old = new_families.get(key, {}), old_families.get(key, {})
        families.append(
            {
                "phase": key[0],
                "graph": key[1],
                "family": key[2],
                **delta(new.get("kernel_sum_ms", 0), old.get("kernel_sum_ms", 0)),
                "memcpy": delta(new.get("memcpy_ms", 0), old.get("memcpy_ms", 0)),
                "memset": delta(new.get("memset_ms", 0), old.get("memset_ms", 0)),
            }
        )
    families.sort(key=lambda r: -abs(r["delta_ms"]))
    return {
        "baseline_path": baseline["path"],
        "host_phase_deltas": phase_changes,
        "benchmark_host_deltas": {
            name: delta(sample[name], old_sample[name])
            for name in ("total_ms", "ttft_ms", "tail_ms")
            if name in sample and name in old_sample
        },
        "family_kernel_deltas": families,
        "measured_gpu_deltas": {
            name: delta(
                current["measured_totals"][name], baseline["measured_totals"][name]
            )
            for name in (
                "kernel_sum_ms",
                "kernel_union_ms",
                "gpu_activity_union_ms",
                "gpu_overlap_ms",
                "memcpy_ms",
                "memset_ms",
            )
        },
        "accounting": "Deltas are candidate minus baseline. Family kernel sums are exclusive contributions; their sum reconciles to kernel_sum_ms, not request wall time. GPU unions/overlap and host wrapper gaps remain separate measured changes. Different artifacts/precision are permitted; caller must validate token identity and execution environment.",
    }


def telemetry_main(arguments: list[str]) -> None:
    """Emit machine-readable reports and an optional compact stderr summary."""
    parser = argparse.ArgumentParser(description=telemetry.__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--summary", action="store_true")
    options = parser.parse_args(arguments)
    reports = [telemetry(path) for path in options.paths]
    print(json.dumps(reports, indent=2))
    if options.summary:
        for report in reports:
            totals = report["measured_totals"] or report["inference_totals"]
            print(
                f"{report['path']}: {totals['kernel_count']} kernels, kernel sum {totals['kernel_sum_ms']:.3f} ms, GPU union {totals['gpu_activity_union_ms']:.3f} ms, overlap {totals['gpu_overlap_ms']:.3f} ms; unmapped={len(report['unmapped_inference'])}",
                file=sys.stderr,
            )
            for row in report["family_costs"][:10]:
                print(
                    f"  {row['phase']} {row['graph']} {row['family']}: {row['kernel_sum_ms']:.3f} ms",
                    file=sys.stderr,
                )


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
    if len(sys.argv) > 2 and sys.argv[1] == "--telemetry":
        telemetry_main(sys.argv[2:])
    elif len(sys.argv) > 2 and sys.argv[1] == "--decode-attribution":
        print(
            json.dumps(
                [decode_attribution(Path(path)) for path in sys.argv[2:]], indent=2
            )
        )
    else:
        main()
