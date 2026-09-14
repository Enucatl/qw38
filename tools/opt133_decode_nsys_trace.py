"""OPT-133 Nsight Systems decode activity on the admitted decode_segments8 stack.

Diagnostics/evidence only. No production selector, kernel, or throughput claim.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    dump_json,
    load_json,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_REV,
    docker_common,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.opt125_decode_accounting import (  # noqa: E402
    classify_disjoint_intervals,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt133_decode_nsys_contract.json"
ITERATION = ROOT / "pins/opt133_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt133_decode_nsys_trace.json"
OPT125_FIXTURE = ROOT / "fixtures/opt125_decode_accounting.json"
REPORT = ROOT / "evidence/optimization/opt133-decode-nsys-trace/REPORT.md"
EVIDENCE = REPORT.parent
TRACES = EVIDENCE / "traces"
NATIVE = "build/qw38-cuda-opt133-decode-nsys-trace-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
GPU_LOCK = ROOT / "build/optimization-runs/qw38-gpu.lock"
RESULT_PREFIX = "QW38_OPT133_DECODE_NSYS_TRACE_RESULT="
RECORDS_PREFIX = "QW38_OPT133_RECORDS="
COUNTS_PREFIX = "QW38_OPT133_NATIVE_COUNTS="
PARENT = "post124_plus_opt127_decode_segments8"
SELECTOR = "decode_segments8"
PREFIXES = (128, 2048)
WINDOWS = ("early", "middle", "late")
WINDOW_TOKENS = 12
DECODE_TOKENS = 256
PHASES = (
    "preflight",
    "capture",
    "parse",
    "reconcile",
    "overhead",
    "report",
)
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "mode",
    "claims_throughput",
    "claims_performance_improvement",
    "production_kept",
    "selected_execution_graph_path",
    "prefixes",
    "windows",
    "preflight",
    "captures",
    "parsed",
    "reconcile",
    "overhead",
    "answers",
    "report_path",
)
REQUIRED_SIDECAR_KEYS = (
    "gpu_busy_ms",
    "gpu_idle_ms",
    "cuda_api_ms",
    "graph_launch_count",
    "graph_launch_ms",
    "memcpy_ms",
    "sync_ms",
    "nsys_overhead_ms",
    "nsys_overhead_ratio",
)
NSYS_TRACE = "cuda,nvtx,osrt"
NSYS_NVTX_RANGE = "opt133.window"
NSYS_STATS_REPORTS = ("gputrace", "cudaapisum", "cuda_api_sum", "nvtxsum")
_LEGACY_NSYS_BIN = "/usr/lib/nsight-systems/bin/nsys"
_LEGACY_NSYS_HOST = "/usr/lib/nsight-systems/host-linux-x64"
NSYS_BIN = "/usr/local/cuda/bin/nsys"
NSYS_HOST = _LEGACY_NSYS_HOST
QDSTRM_IMPORTER = f"{_LEGACY_NSYS_HOST}/QdstrmImporter"
_DOCKER_NSYS_CACHE: dict[str, Any] | None = None


START_KEYS = ("Start (ns)", "Start(ns)", "Start", "start_ns", "start")
DURATION_KEYS = (
    "Duration (ns)",
    "Duration(ns)",
    "Duration",
    "duration_ns",
    "duration",
    "Total Time (ns)",
    "Total Time",
    "Time (ns)",
)
NAME_KEYS = ("Name", "name", "Kernel Name", "API Name", "Range")
COUNT_KEYS = ("Num Calls", "Instances", "Count", "num_calls")


class NsysTraceError(RuntimeError):
    """OPT-133 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-133":
        raise NsysTraceError("decode-nsys contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-133", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-133 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sidecar(run_dir: Path, name: str) -> Path:
    return run_dir / name


def store_sidecar(
    run_dir: Path, name: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = sidecar(run_dir, name)
    dump_json(path, payload)
    return dict(payload)


def load_sidecar(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = sidecar(run_dir, name)
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def empty_fixture(mode: str = "feedback") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-133",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "windows": list(WINDOWS),
        "window_tokens": WINDOW_TOKENS,
        "decode_output_tokens": DECODE_TOKENS,
        "preflight": None,
        "captures": {},
        "parsed": {},
        "reconcile": {},
        "overhead": {},
        "answers": {
            "proven_gpu_idle": None,
            "host_api_submission": None,
            "unresolved": None,
            "nsys_wrapper_overhead": None,
        },
        "report_path": relpath(REPORT),
    }


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        payload = load_json(FIXTURE)
        if isinstance(payload, dict) and payload.get("task") == "OPT-133":
            return payload
    return empty_fixture()


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def case_key(prefix: int, window: str) -> str:
    return f"d{prefix}-{window}"


def iter_cases() -> list[tuple[int, str]]:
    return [(prefix, window) for prefix in PREFIXES for window in WINDOWS]


def extract_json(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    decoder = json.JSONDecoder()
    idx = stripped.find("[")
    while idx >= 0:
        rest = stripped[idx + 1 :].lstrip()
        if rest.startswith("{"):
            try:
                obj, _end = decoder.raw_decode(stripped[idx:])
                return obj
            except json.JSONDecodeError:
                pass
        idx = stripped.find("[", idx + 1)
    idx = stripped.find("{")
    if idx >= 0:
        try:
            obj, _end = decoder.raw_decode(stripped[idx:])
            return obj
        except json.JSONDecodeError:
            return None
    return None


def merge_gate_output(raw: str, gate: Path) -> str:
    parts = [raw]
    records = gate / "records.json"
    result = gate / "result.txt"
    if records.is_file() and RECORDS_PREFIX not in raw:
        parts.append(RECORDS_PREFIX + records.read_text(encoding="utf-8"))
    if result.is_file() and RESULT_PREFIX not in raw:
        parts.append(result.read_text(encoding="utf-8"))
    return "\n".join(parts)


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records = [
        json.loads(line.removeprefix(prefix))
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if not records:
        raise NsysTraceError(f"missing {prefix} record")
    last = records[-1]
    if not isinstance(last, dict):
        raise NsysTraceError(f"{prefix} record is not an object")
    return last


def parse_records(text: str) -> list[dict[str, Any]]:
    for line in text.splitlines():
        if line.startswith(RECORDS_PREFIX):
            payload = json.loads(line.split("=", 1)[1])
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def nsys_profile_command(output: Path | str, binary_args: Sequence[str]) -> list[str]:
    """Build nsys profile argv for the pinned CUDA-repo or legacy Ubuntu nsys."""
    stem = str(output)
    if modern_nsys():
        return [
            NSYS_BIN,
            "profile",
            "-o",
            stem,
            "--force-overwrite",
            "true",
            f"--trace={NSYS_TRACE}",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat",
            *binary_args,
        ]
    return [
        NSYS_BIN,
        "profile",
        "-o",
        stem,
        "--force-overwrite",
        "true",
        f"--trace={NSYS_TRACE}",
        "--sample=none",
        "--cpuctxsw=none",
        "--kill=none",
        "--capture-range=nvtx",
        f"--nvtx-capture={NSYS_NVTX_RANGE}",
        "--capture-range-end=stop",
        *binary_args,
    ]


def nsys_env_exports() -> str:
    return (
        f"export PATH={shlex.quote(NSYS_HOST)}:"
        f"{shlex.quote('/usr/lib/nsight-systems/bin')}"
        ':"${PATH:-}"; '
        f"export LD_LIBRARY_PATH={shlex.quote(NSYS_HOST)}"
        ':"${LD_LIBRARY_PATH:-}"'
    )


def nsys_session_name(prefix: int, window: str) -> str:
    return f"opt133d{prefix}{window}"


def nsys_launch_command(session: str, binary_args: Sequence[str]) -> list[str]:
    return [
        NSYS_BIN,
        "launch",
        f"--session-new={session}",
        f"--trace={NSYS_TRACE}",
        "--cuda-graph-trace=graph",
        "--show-output=true",
        *binary_args,
    ]


def nsys_start_command(session: str, output: Path | str) -> list[str]:
    return [
        NSYS_BIN,
        "start",
        f"--session={session}",
        "-o",
        str(output),
        "--force-overwrite",
        "true",
        "--sample=none",
        "--cpuctxsw=none",
    ]


def nsys_stop_command(session: str) -> list[str]:
    return [NSYS_BIN, "stop", f"--session={session}"]


def nsys_import_script(stem: Path | str) -> str:
    """Copy leftover /tmp qdstrm files and import to .qdrep."""
    target = str(stem)
    return (
        "shopt -s nullglob; "
        f"for f in /tmp/nsys-report-*.qdstrm; do "
        f'  if [ -f "$f" ]; then cp -f "$f" {shlex.quote(target)}.qdstrm; fi; '
        "done; "
        f"if [ -f {shlex.quote(target)}.qdstrm ]; then "
        f"  {shlex.quote(QDSTRM_IMPORTER)} -f -i {shlex.quote(target)}.qdstrm "
        f"-o {shlex.quote(target)}.qdrep; "
        "fi"
    )


def nsys_capture_script(
    output: Path | str,
    binary_args: Sequence[str],
    *,
    session: str = "opt133window",
) -> str:
    """Bounded nsys collection via launch/start/stop handshake.

    nsys 2022.4.2 cannot use cudaProfilerApi or NVTX capture-range against
    CUDA 13 (no QDSTRM). Native still emits cudaProfilerStart/Stop and the
    opt133.window NVTX range. Collection is armed only while the 12-token
    window runs, using QW38_OPT133_GATE marker files.
    """
    target = str(output)
    gate = f"{target}-gate"
    launch = " ".join(
        shlex.quote(part) for part in nsys_launch_command(session, binary_args)
    )
    start = " ".join(shlex.quote(part) for part in nsys_start_command(session, output))
    stop = " ".join(shlex.quote(part) for part in nsys_stop_command(session))
    log = f"{target}.launch.log"
    return f"""
set +e
{nsys_env_exports()}
rm -rf {shlex.quote(gate)}
mkdir -p {shlex.quote(gate)}
export QW38_OPT133_GATE={shlex.quote(gate)}
{shlex.quote(NSYS_BIN)} shutdown --session={shlex.quote(session)} --kill=none >/dev/null 2>&1
{launch} > {shlex.quote(log)} 2>&1 &
launch_pid=$!
waited=0
while [ ! -f {shlex.quote(gate)}/waiting ] && [ "$waited" -lt 12000 ]; do
  sleep 0.05
  waited=$((waited + 1))
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    break
  fi
done
{start}
start_rc=$?
touch {shlex.quote(gate)}/armed
waited=0
while [ ! -f {shlex.quote(gate)}/done ] && [ "$waited" -lt 4000 ]; do
  sleep 0.05
  waited=$((waited + 1))
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    break
  fi
done
{stop}
waited=0
while [ ! -f {shlex.quote(gate)}/result.txt ] && [ "$waited" -lt 600 ]; do
  sleep 0.05
  waited=$((waited + 1))
done
{shlex.quote(NSYS_BIN)} shutdown --session={shlex.quote(session)} --kill=none >/dev/null 2>&1
wait "$launch_pid" 2>/dev/null
status=$?
if [ -f {shlex.quote(gate)}/result.txt ]; then status=0; fi
{nsys_import_script(output)}
if [ -f {shlex.quote(log)} ]; then cat {shlex.quote(log)}; fi
if [ -f {shlex.quote(gate)}/records.json ]; then
  printf '%s' 'QW38_OPT133_RECORDS='
  cat {shlex.quote(gate)}/records.json
  echo
fi
if [ -f {shlex.quote(gate)}/result.txt ]; then cat {shlex.quote(gate)}/result.txt; fi
echo NSYS_SESSION_START_RC=$start_rc
exit $status
""".strip()


def nsys_stats_command(
    report: str, trace: Path | str, *, fmt: str = "json"
) -> list[str]:
    return [
        NSYS_BIN,
        "stats",
        "--report",
        report,
        "--format",
        fmt,
        "--force-overwrite",
        "true",
        str(trace),
    ]


def native_binary_args(
    workload: str, prefix: int, window: str, tokens: int = DECODE_TOKENS
) -> list[str]:
    return [
        f"./{NATIVE}",
        "--workload",
        workload,
        "--window",
        window,
        "--prefix",
        str(prefix),
        "--tokens",
        str(tokens),
        MODEL,
    ]


def with_gpu_lock(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return subprocess.run(
            list(command),
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = with_gpu_lock(command)
    if check and completed.returncode != 0:
        raise NsysTraceError(
            "native command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def docker_sh(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*docker_common(IMAGE, "acceptance"), "bash", "-lc", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def docker_nsys_layout(*, refresh: bool = False) -> dict[str, Any]:
    """Resolve nsys paths from the pinned CUDA image, not the host OS."""
    global _DOCKER_NSYS_CACHE
    if _DOCKER_NSYS_CACHE is not None and not refresh:
        return _DOCKER_NSYS_CACHE
    script = (
        "nsys_bin=$(command -v nsys || true); "
        'echo "NSYS_BIN=${nsys_bin}"; '
        "nsys --version 2>&1 | head -1; "
        "host=$(ls -d /opt/nvidia/nsight-systems/*/host-linux-x64 2>/dev/null | sort -V | tail -1); "
        'if [ -n "$host" ]; then echo "NSYS_HOST=$host"; '
        'elif [ -x /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter ]; then '
        'echo "NSYS_HOST=/usr/lib/nsight-systems/host-linux-x64"; fi'
    )
    completed = docker_sh(script)
    blob = completed.stdout + completed.stderr
    resolved_bin = "/usr/local/cuda/bin/nsys"
    host_dir = _LEGACY_NSYS_HOST
    version = ""
    for line in blob.splitlines():
        if line.startswith("NSYS_BIN="):
            value = line.split("=", 1)[1].strip()
            if value:
                resolved_bin = value
        elif line.startswith("NSYS_HOST="):
            value = line.split("=", 1)[1].strip()
            if value:
                host_dir = value
        elif "Nsight Systems version" in line:
            version = line.strip()
    modern = (
        "2025." in version
        or "2024." in version
        or resolved_bin.startswith("/usr/local/cuda/")
        or resolved_bin.startswith("/usr/local/bin/")
    )
    layout = {
        "nsys_bin": resolved_bin,
        "host_dir": host_dir,
        "qdstrm_importer": f"{host_dir}/QdstrmImporter",
        "version": version or None,
        "modern": modern,
    }
    _DOCKER_NSYS_CACHE = layout
    return layout


def ensure_nsys_layout(*, refresh: bool = False) -> dict[str, Any]:
    layout = docker_nsys_layout(refresh=refresh)
    global NSYS_BIN, NSYS_HOST, QDSTRM_IMPORTER
    NSYS_BIN = str(layout["nsys_bin"])
    NSYS_HOST = str(layout["host_dir"])
    QDSTRM_IMPORTER = str(layout["qdstrm_importer"])
    return layout


def nsys_version_text() -> str:
    return str(docker_nsys_layout().get("version") or "")


def modern_nsys() -> bool:
    ensure_nsys_layout()
    return bool(docker_nsys_layout().get("modern"))


def gpu_residents() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append(
                {
                    "pid": parts[0],
                    "process_name": parts[1],
                    "used_gpu_memory": parts[2],
                }
            )
    return rows


def _first_key(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    lowered = {str(name).casefold(): value for name, value in row.items()}
    for key in keys:
        if key.casefold() in lowered:
            return lowered[key.casefold()]
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_ms(value: float, *, assume_ns: bool | None = None) -> float:
    if assume_ns is True or (assume_ns is None and abs(value) >= 1_000.0):
        return value / 1_000_000.0
    return value


def table_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                has_name = _first_key(item, NAME_KEYS) is not None
                has_time = _first_key(item, START_KEYS + DURATION_KEYS) is not None
                if has_name or has_time:
                    rows.append(item)
                else:
                    rows.extend(table_rows(item))
            else:
                rows.extend(table_rows(item))
        return rows
    if not isinstance(payload, dict):
        return rows
    cols = payload.get("cols") or payload.get("columns") or payload.get("Fields")
    data = payload.get("rows") or payload.get("data")
    if isinstance(cols, list) and isinstance(data, list):
        names = [str(col) for col in cols]
        for row in data:
            if isinstance(row, dict):
                rows.append(row)
            elif isinstance(row, list):
                mapped = {
                    names[index]: row[index]
                    for index in range(min(len(names), len(row)))
                }
                rows.append(mapped)
        return rows
    for value in payload.values():
        rows.extend(table_rows(value))
    return rows


def union_interval_ms(intervals: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(
        (float(start), float(end)) for start, end in intervals if end > start
    )
    if not ordered:
        return 0.0
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def row_name(row: Mapping[str, Any]) -> str:
    value = _first_key(row, NAME_KEYS)
    return str(value) if value is not None else ""


def row_intervals_ms(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    starts: list[float] = []
    durations: list[float] = []
    for row in rows:
        start = _as_float(_first_key(row, START_KEYS))
        duration = _as_float(_first_key(row, DURATION_KEYS))
        if start is not None:
            starts.append(start)
        if duration is not None:
            durations.append(duration)
    assume_ns = bool(starts) and max(abs(item) for item in starts) >= 1_000.0
    if not assume_ns and durations:
        assume_ns = max(abs(item) for item in durations) >= 1_000.0
    intervals: list[tuple[float, float]] = []
    for row in rows:
        start = _as_float(_first_key(row, START_KEYS))
        duration = _as_float(_first_key(row, DURATION_KEYS))
        if start is None or duration is None or duration <= 0:
            continue
        start_ms = _to_ms(start, assume_ns=assume_ns)
        duration_ms = _to_ms(duration, assume_ns=assume_ns)
        intervals.append((start_ms, start_ms + duration_ms))
    return intervals


def sum_named_ms(rows: Sequence[Mapping[str, Any]], predicate) -> tuple[float, int]:
    total = 0.0
    count = 0
    durations: list[float] = []
    for row in rows:
        if not predicate(row_name(row)):
            continue
        duration = _as_float(_first_key(row, DURATION_KEYS))
        n_calls = _as_float(_first_key(row, COUNT_KEYS))
        if duration is None:
            continue
        durations.append(duration)
        calls = int(n_calls) if n_calls and n_calls > 0 else 1
        count += calls
    assume_ns = bool(durations) and max(abs(item) for item in durations) >= 1_000.0
    for row in rows:
        if not predicate(row_name(row)):
            continue
        duration = _as_float(_first_key(row, DURATION_KEYS))
        if duration is None:
            continue
        total += _to_ms(duration, assume_ns=assume_ns)
    return total, count


def nvtx_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets = {
        "mixer": 0.0,
        "gdn": 0.0,
        "attention": 0.0,
        "ffn": 0.0,
        "graph": 0.0,
        "other": 0.0,
    }
    intervals_by: dict[str, list[tuple[float, float]]] = {key: [] for key in buckets}
    named_intervals: list[tuple[str, float, float]] = []
    for row in rows:
        name = row_name(row).casefold()
        intervals = row_intervals_ms([row])
        if not intervals:
            duration = _as_float(_first_key(row, DURATION_KEYS))
            if duration is None:
                continue
            assume_ns = abs(duration) >= 1_000.0
            named_intervals.append((name, 0.0, _to_ms(duration, assume_ns=assume_ns)))
            continue
        start, end = intervals[0]
        named_intervals.append((name, start, end))
    for name, start, end in named_intervals:
        if "mixer" in name:
            key = "mixer"
        elif "gdn" in name:
            key = "gdn"
        elif "attention" in name:
            key = "attention"
        elif "ffn" in name:
            key = "ffn"
        elif "graph" in name:
            key = "graph"
        else:
            key = "other"
        intervals_by[key].append((start, end))
    for key, spans in intervals_by.items():
        if not spans:
            continue
        if all(start == 0.0 for start, _end in spans) and len(spans) > 1:
            buckets[key] = sum(end - start for start, end in spans)
        else:
            buckets[key] = union_interval_ms(spans)
    return buckets


def is_gpu_activity_name(name: str) -> bool:
    lowered = name.casefold()
    if not lowered:
        return False
    if lowered.startswith("cuda") and "launch" not in lowered:
        return False
    return True


def is_graph_launch_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        "graphlaunch" in lowered
        or "cudaGraphLaunch" in name
        or "graph launch" in lowered
    )


def is_memcpy_name(name: str) -> bool:
    lowered = name.casefold()
    return "memcpy" in lowered or "memcopy" in lowered


def is_sync_name(name: str) -> bool:
    lowered = name.casefold()
    if "async" in lowered:
        return False
    return "synchronize" in lowered or "devicesync" in lowered


def summarize_nsys_stats(
    gputrace: Any,
    cuda_api: Any | None = None,
    nvtx: Any | None = None,
) -> dict[str, Any]:
    gpu_rows = [
        row
        for row in table_rows(gputrace)
        if is_gpu_activity_name(row_name(row))
        or _first_key(row, START_KEYS) is not None
    ]
    gpu_intervals = row_intervals_ms(gpu_rows)
    gpu_busy_ms = union_interval_ms(gpu_intervals)
    if gpu_intervals:
        window_ms = max(end for _start, end in gpu_intervals) - min(
            start for start, _end in gpu_intervals
        )
    else:
        window_ms = 0.0
    gpu_idle_ms = max(0.0, window_ms - gpu_busy_ms)
    api_rows = table_rows(cuda_api) if cuda_api is not None else []
    api_intervals = row_intervals_ms(api_rows)
    if api_intervals:
        cuda_api_ms = union_interval_ms(api_intervals)
    else:
        cuda_api_ms, _count = sum_named_ms(api_rows, lambda _name: True)
    graph_ms, graph_count = sum_named_ms(api_rows or gpu_rows, is_graph_launch_name)
    memcpy_ms, _memcpy_n = sum_named_ms(gpu_rows + api_rows, is_memcpy_name)
    sync_ms, _sync_n = sum_named_ms(api_rows or gpu_rows, is_sync_name)
    nvtx_rows = table_rows(nvtx) if nvtx is not None else []
    ranges = nvtx_breakdown(nvtx_rows or gpu_rows)
    has_gpu = bool(gpu_intervals)
    has_api = bool(api_rows)
    return {
        "gpu_busy_ms": gpu_busy_ms if has_gpu else None,
        "gpu_idle_ms": gpu_idle_ms if has_gpu else None,
        "cuda_api_ms": cuda_api_ms if has_api else None,
        "capture_window_ms": window_ms if has_gpu else None,
        "graph_launch_count": graph_count,
        "graph_launch_ms": graph_ms,
        "memcpy_ms": memcpy_ms,
        "sync_ms": sync_ms,
        "nvtx": ranges,
        "from_parsed_trace": has_gpu,
        "gpu_row_count": len(gpu_rows),
        "api_row_count": len(api_rows),
        "nvtx_row_count": len(nvtx_rows),
        "cupti_gpu_trace": has_gpu,
        "cuda_api_trace": has_api,
    }


def load_opt125_windows(fixture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = (
        fixture
        if fixture is not None
        else (load_json(OPT125_FIXTURE) if OPT125_FIXTURE.is_file() else {})
    )
    activity = payload.get("activity_trace") or {}
    windows: dict[str, Any] = {}
    for prefix in PREFIXES:
        block = activity.get(f"d{prefix}") or {}
        per_window = block.get("windows") or {}
        windows[f"d{prefix}"] = {
            "unobserved_ms": (block.get("intervals") or {}).get("unobserved_ms"),
            "device_active_ms": (block.get("intervals") or {}).get("device_active_ms"),
            "device_inactive_proven_ms": (block.get("intervals") or {}).get(
                "device_inactive_proven_ms"
            ),
            "windows": {
                label: {
                    "unobserved_ms": (per_window.get(label) or {}).get("unobserved_ms"),
                    "device_active_ms": (per_window.get(label) or {}).get(
                        "device_active_ms"
                    ),
                    "device_inactive_proven_ms": (per_window.get(label) or {}).get(
                        "device_inactive_proven_ms"
                    ),
                    "wall_ms": (per_window.get(label) or {}).get("wall_ms"),
                    "tokens": (per_window.get(label) or {}).get("tokens"),
                }
                for label in WINDOWS
            },
        }
    return windows


def scale_opt125_window(
    block: Mapping[str, Any], *, source_tokens: int
) -> dict[str, Any]:
    """Scale OPT-125 4-token windows to the 12-token nsys window length."""
    tokens = int(block.get("tokens") or source_tokens or 4)
    scale = WINDOW_TOKENS / float(tokens) if tokens else 3.0
    scaled = dict(block)
    for key in (
        "unobserved_ms",
        "device_active_ms",
        "device_inactive_proven_ms",
        "wall_ms",
    ):
        value = block.get(key)
        if isinstance(value, (int, float)):
            scaled[key] = float(value) * scale
    scaled["source_tokens"] = tokens
    scaled["scaled_to_window_tokens"] = WINDOW_TOKENS
    scaled["scale"] = scale
    return scaled


def reconcile_window(
    opt125: Mapping[str, Any],
    parsed: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    unobserved = float(opt125.get("unobserved_ms") or 0.0)
    gpu_idle = float(parsed.get("gpu_idle_ms") or 0.0)
    cuda_api = float(parsed.get("cuda_api_ms") or 0.0)
    from_trace = bool(parsed.get("from_parsed_trace"))
    explained_idle = min(unobserved, max(0.0, gpu_idle)) if from_trace else 0.0
    remaining = max(0.0, unobserved - explained_idle)
    explained_api = min(remaining, max(0.0, cuda_api)) if from_trace else 0.0
    unresolved = max(0.0, unobserved - explained_idle - explained_api)
    profiler = "nsight_systems" if from_trace else "cuda_event_engine_attribution"
    wall = parsed.get("capture_window_ms")
    if wall is None:
        wall = opt125.get("wall_ms")
    intervals = classify_disjoint_intervals(
        list(records or []),
        wall_ms=float(wall) if wall is not None else None,
        tokens=int(opt125.get("tokens") or WINDOW_TOKENS),
        profiler=profiler,
    )
    if from_trace:
        intervals["device_inactive_proven_ms"] = gpu_idle
        intervals["inactive_method"] = "nsys_gputrace_union_idle"
        intervals["unobserved_ms"] = unresolved
        intervals["leaf_gap_relabeled"] = "nsys_idle_or_host_api_or_unresolved"
    return {
        "opt125_unobserved_ms": unobserved,
        "opt125_device_active_ms": opt125.get("device_active_ms"),
        "nsys_gpu_idle_ms": gpu_idle if from_trace else None,
        "nsys_gpu_busy_ms": parsed.get("gpu_busy_ms") if from_trace else None,
        "nsys_cuda_api_ms": cuda_api if from_trace else None,
        "delta_idle_minus_unobserved_ms": (gpu_idle - unobserved)
        if from_trace
        else None,
        "explained_by_idle_ms": explained_idle,
        "explained_by_api_ms": explained_api,
        "unresolved_ms": unresolved,
        "fraction_idle": (explained_idle / unobserved) if unobserved else 0.0,
        "fraction_api": (explained_api / unobserved) if unobserved else 0.0,
        "fraction_unresolved": (unresolved / unobserved) if unobserved else 0.0,
        "from_parsed_trace": from_trace,
        "profiler": profiler,
        "intervals": intervals,
        "stub_relabel_avoided": from_trace,
    }


def overhead_pair(baseline_ms: float | None, nsys_ms: float | None) -> dict[str, Any]:
    if baseline_ms is None or nsys_ms is None:
        return {
            "baseline_ms": baseline_ms,
            "nsys_ms": nsys_ms,
            "nsys_overhead_ms": None,
            "nsys_overhead_ratio": None,
        }
    base = float(baseline_ms)
    wrapped = float(nsys_ms)
    delta = wrapped - base
    ratio = (wrapped / base) if base > 0.0 else None
    return {
        "baseline_ms": base,
        "nsys_ms": wrapped,
        "nsys_overhead_ms": delta,
        "nsys_overhead_ratio": ratio,
    }


def resolve_trace_file(stem: Path) -> Path | None:
    resolved = Path(stem)
    for suffix in (".qdrep", ".nsys-rep", ".qdstrm"):
        candidate = resolved.with_suffix(suffix)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    if resolved.is_file() and resolved.stat().st_size > 0:
        return resolved
    matches = sorted(
        path
        for path in resolved.parent.glob(resolved.name + ".*")
        if path.is_file() and path.stat().st_size > 0
    )
    return matches[0] if matches else None


def copy_trace(trace: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / trace.name
    shutil.copy2(trace, dest)
    return dest


def extract_opt125_targets() -> dict[str, Any]:
    return load_opt125_windows()


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_nsys_layout(refresh=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    source, dirty = git_identity()
    available, gpu_blocker = gpu_available()
    residents = gpu_residents() if available else []
    probe = docker_sh(
        "command -v nsys; nsys --version; nsys status -e 2>&1 | head -n 12 || true; "
        "command -v ncu; (ncu --version | head -n 1 || true)"
    )
    blob = probe.stdout + probe.stderr
    nsys_available = "nsys" in blob and probe.returncode == 0
    # command -v prints paths; treat presence of /nsys as installed.
    nsys_available = (
        any(
            line.strip().endswith("nsys") or line.strip().endswith("/nsys")
            for line in probe.stdout.splitlines()
        )
        or shutil.which("nsys") is not None
    )
    ncu_available = any(
        line.strip().endswith("ncu") or line.strip().endswith("/ncu")
        for line in probe.stdout.splitlines()
    )
    identity: dict[str, Any] = {}
    identity_ok = False
    if available and (ROOT / NATIVE).is_file():
        completed = run_native(
            [f"./{NATIVE}", "--workload", "identity", "--prefix", "128", MODEL],
            tier="correctness",
            check=False,
        )
        raw = completed.stdout + completed.stderr
        (run_dir / "identity.txt").write_text(raw, encoding="utf-8")
        if "nsys_available=true" in raw:
            nsys_available = True
        elif "nsys_available=false" in raw:
            nsys_available = False
        if completed.returncode == 0:
            identity = parse_prefixed(raw, RESULT_PREFIX)
            identity_ok = bool(identity.get("ok"))
        else:
            identity = {
                "ok": False,
                "returncode": completed.returncode,
                "stderr_tail": raw[-1500:],
            }
    elif not (ROOT / NATIVE).is_file():
        identity = {"ok": False, "reason": f"missing {NATIVE}"}
    else:
        identity = {"ok": False, "reason": gpu_blocker or "gpu_unavailable"}
    opt125 = extract_opt125_targets()
    shipping_ok = (
        identity.get("shipping_execution_graphs") == SELECTOR
        or contract["selected_execution_graph_path"] == SELECTOR
    )
    ok = (
        nsys_available
        and identity_ok
        and shipping_ok
        and contract["selected_execution_graph_path"] == SELECTOR
        and contract["claims_throughput"] is False
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-133",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "nsys_available": nsys_available,
        "nsys_bin": NSYS_BIN,
        "nsys_version": nsys_version_text().strip() or None,
        "modern_nsys": modern_nsys(),
        "ncu_available": ncu_available,
        "cupti_linked": False,
        "image": IMAGE,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "gpu_residents": residents,
        "exclusive_sitting": len(residents) <= 1,
        "authenticated_post113": auth,
        "identity": identity,
        "opt125_targets": opt125,
        "parent": PARENT,
        "selected_execution_graph_path": SELECTOR,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": dirty,
        "iteration_target": iteration["target"],
        "diagnostics_make_target": iteration["diagnostics_make_target"],
        "gpu_lock": relpath(GPU_LOCK),
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "probe_stdout_tail": blob[-1500:],
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "preflight.json", payload)
    fixture = current_fixture()
    fixture["mode"] = mode
    fixture["preflight"] = payload
    persist_fixture(fixture)
    return payload


def _write_capture_text(path: Path, completed: subprocess.CompletedProcess[str]) -> str:
    raw = completed.stdout + completed.stderr
    path.write_text(raw, encoding="utf-8")
    return raw


def run_capture(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_nsys_layout(refresh=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    available, blocker = gpu_available()
    captures: dict[str, Any] = {}
    hardware = False
    if not available:
        reason = blocker or "gpu_unavailable"
        for prefix, window in iter_cases():
            key = case_key(prefix, window)
            captures[key] = {
                "ok": False,
                "reason": reason,
                "prefix": prefix,
                "window": window,
            }
        payload = {
            "schema_version": 1,
            "task": "OPT-133",
            "phase": "capture",
            "mode": mode,
            "ok": False,
            "hardware_executed": False,
            "reason": reason,
            "captures": captures,
            "family_plan": family_plan(mode, "capture"),
            "measured_at": utc_now(),
        }
        store_sidecar(run_dir, "capture.json", payload)
        fixture = current_fixture()
        fixture["captures"] = captures
        persist_fixture(fixture)
        return payload

    residents = gpu_residents()
    reason = None
    for prefix, window in iter_cases():
        key = case_key(prefix, window)
        baseline_args = native_binary_args("nsys-baseline", prefix, window)
        baseline = run_native(baseline_args, tier="screen")
        baseline_raw = _write_capture_text(run_dir / f"baseline-{key}.txt", baseline)
        baseline_summary = parse_prefixed(baseline_raw, RESULT_PREFIX)
        baseline_records = parse_records(baseline_raw)
        dump_json(run_dir / f"baseline-{key}.json", baseline_summary)
        dump_json(run_dir / f"baseline-{key}-records.json", baseline_records)

        stem = run_dir / f"trace-{prefix}-{window}"
        if modern_nsys():
            profile_cmd = nsys_profile_command(
                stem, native_binary_args("nsys-window", prefix, window)
            )
            nsys_run = run_native(profile_cmd, tier="screen")
        else:
            inner = [
                "bash",
                "-lc",
                nsys_capture_script(
                    stem,
                    native_binary_args("nsys-window", prefix, window),
                    session=nsys_session_name(prefix, window),
                ),
            ]
            nsys_run = run_native(inner, tier="screen")
        nsys_raw = _write_capture_text(run_dir / f"nsys-{key}.txt", nsys_run)
        nsys_raw = merge_gate_output(nsys_raw, Path(f"{stem}-gate"))
        nsys_summary = parse_prefixed(nsys_raw, RESULT_PREFIX)
        nsys_records = parse_records(nsys_raw)
        gate_records = Path(f"{stem}-gate") / "records.json"
        if not nsys_records and gate_records.is_file():
            loaded = json.loads(gate_records.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                nsys_records = [row for row in loaded if isinstance(row, dict)]
        dump_json(run_dir / f"nsys-{key}.json", nsys_summary)
        dump_json(run_dir / f"nsys-{key}-records.json", nsys_records)
        trace = resolve_trace_file(stem)
        evidence_trace = None
        if trace is not None:
            evidence_trace = relpath(copy_trace(trace, TRACES))
        captures[key] = {
            "ok": bool(baseline_summary.get("ok", True))
            and bool(nsys_summary.get("ok", True))
            and trace is not None,
            "prefix": prefix,
            "window": window,
            "baseline": baseline_summary,
            "nsys": nsys_summary,
            "baseline_records_path": relpath(run_dir / f"baseline-{key}-records.json"),
            "nsys_records_path": relpath(run_dir / f"nsys-{key}-records.json"),
            "trace_path": relpath(trace) if trace is not None else None,
            "evidence_trace_path": evidence_trace,
            "gpu_residents": residents,
        }
        hardware = True
    payload = {
        "schema_version": 1,
        "task": "OPT-133",
        "phase": "capture",
        "mode": mode,
        "ok": all(row.get("ok") for row in captures.values()) and len(captures) == 6,
        "hardware_executed": hardware,
        "reason": reason,
        "captures": captures,
        "family_plan": family_plan(mode, "capture"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "capture.json", payload)
    fixture = current_fixture()
    fixture["captures"] = captures
    persist_fixture(fixture)
    return payload


def _load_json_if_present(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return path.read_text(encoding="utf-8")


def nsys_report_aliases() -> dict[str, list[str]]:
    if modern_nsys():
        return {
            "gputrace": ["cuda_gpu_trace", "gputrace"],
            "cudaapisum": ["cuda_api_sum", "cudaapisum"],
            "nvtxsum": ["nvtx_pushpop_sum", "nvtxsum"],
        }
    return {
        "gputrace": ["gputrace"],
        "cudaapisum": ["cudaapisum", "cuda_api_sum"],
        "nvtxsum": ["nvtxsum"],
    }


def fetch_nsys_report(
    run_dir: Path,
    *,
    key: str,
    canonical: str,
    trace: Path,
) -> tuple[Any | None, list[str]]:
    errors: list[str] = []
    parsed_json: Any | None = None
    raw = ""
    for report in nsys_report_aliases()[canonical]:
        completed = run_native(
            nsys_stats_command(report, trace),
            tier="acceptance",
            check=False,
        )
        raw = completed.stdout or completed.stderr
        sidecar_path = run_dir / f"stats-{key}-{canonical}.json"
        parsed_json = extract_json(raw)
        skipped = "does not contain" in raw and "SKIPPED" in raw
        if parsed_json is not None:
            dump_json(sidecar_path, parsed_json)
            return parsed_json, errors
        sidecar_path.write_text(raw, encoding="utf-8")
        if completed.returncode != 0 and not skipped:
            errors.append(f"{report}: {raw[-400:]}")
        elif skipped:
            errors.append(f"{report}: skipped_no_cuda13_cupti_events")
    return None, errors


def parse_one_trace(run_dir: Path, prefix: int, window: str) -> dict[str, Any]:
    key = case_key(prefix, window)
    stem = run_dir / f"trace-{prefix}-{window}"
    trace = resolve_trace_file(stem)
    if trace is None:
        evidence = TRACES / f"trace-{prefix}-{window}.nsys-rep"
        qdrep = TRACES / f"trace-{prefix}-{window}.qdrep"
        if evidence.is_file():
            trace = evidence
        elif qdrep.is_file():
            trace = qdrep
    reports: dict[str, Any] = {}
    errors: list[str] = []
    if trace is None:
        return {
            "ok": False,
            "prefix": prefix,
            "window": window,
            "reason": "missing_trace",
            "from_parsed_trace": False,
            "gpu_busy_ms": None,
            "gpu_idle_ms": None,
            "cuda_api_ms": None,
            "graph_launch_count": None,
            "graph_launch_ms": None,
            "memcpy_ms": None,
            "sync_ms": None,
        }
    for canonical in ("gputrace", "cudaapisum", "nvtxsum"):
        parsed_json, report_errors = fetch_nsys_report(
            run_dir,
            key=key,
            canonical=canonical,
            trace=trace,
        )
        reports[canonical] = parsed_json
        errors.extend(report_errors)
    gputrace = reports.get("gputrace")
    nvtx = reports.get("nvtxsum")
    cuda_api = reports.get("cudaapisum")
    if gputrace is None and nvtx is None and cuda_api is None:
        summary = {
            "ok": False,
            "prefix": prefix,
            "window": window,
            "trace_path": relpath(trace),
            "reason": "gputrace_parse_failed",
            "errors": errors,
            "from_parsed_trace": False,
            "gpu_busy_ms": None,
            "gpu_idle_ms": None,
            "cuda_api_ms": None,
            "graph_launch_count": None,
            "graph_launch_ms": None,
            "memcpy_ms": None,
            "sync_ms": None,
        }
        dump_json(run_dir / f"parsed-{key}.json", summary)
        return summary
    derived = summarize_nsys_stats(gputrace, cuda_api, nvtx)
    derived.update(
        {
            "ok": True,
            "prefix": prefix,
            "window": window,
            "trace_path": relpath(trace),
            "errors": errors,
            "reason": None
            if derived.get("from_parsed_trace")
            else "nsys_2022_4_no_cuda13_cupti_gputrace",
        }
    )
    dump_json(run_dir / f"parsed-{key}.json", derived)
    return derived


def run_parse(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_nsys_layout()
    parsed: dict[str, Any] = {}
    for prefix, window in iter_cases():
        parsed[case_key(prefix, window)] = parse_one_trace(run_dir, prefix, window)
    payload = {
        "schema_version": 1,
        "task": "OPT-133",
        "phase": "parse",
        "mode": mode,
        "ok": all(row.get("ok") for row in parsed.values()) and len(parsed) == 6,
        "hardware_gpu_trace": all(
            bool(row.get("from_parsed_trace")) for row in parsed.values()
        ),
        "parsed": parsed,
        "family_plan": family_plan(mode, "parse"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "parse.json", payload)
    fixture = current_fixture()
    fixture["parsed"] = parsed
    persist_fixture(fixture)
    return payload


def _load_records(run_dir: Path, key: str) -> list[dict[str, Any]]:
    path = run_dir / f"nsys-{key}-records.json"
    if not path.is_file():
        path = run_dir / f"baseline-{key}-records.json"
    if not path.is_file():
        return []
    payload = load_json(path)
    return (
        [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, list)
        else []
    )


def run_reconcile(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    opt125 = load_opt125_windows()
    parse_payload = load_sidecar(run_dir, "parse.json") or {}
    parsed_map = parse_payload.get("parsed") or current_fixture().get("parsed") or {}
    rows: dict[str, Any] = {}
    for prefix, window in iter_cases():
        key = case_key(prefix, window)
        opt_block = ((opt125.get(f"d{prefix}") or {}).get("windows") or {}).get(
            window
        ) or {}
        scaled = scale_opt125_window(
            opt_block, source_tokens=int(opt_block.get("tokens") or 4)
        )
        parsed = parsed_map.get(key) or {}
        records = _load_records(run_dir, key)
        rows[key] = reconcile_window(scaled, parsed, records)
        rows[key]["prefix"] = prefix
        rows[key]["window"] = window
        rows[key]["opt125_source"] = opt_block
    payload = {
        "schema_version": 1,
        "task": "OPT-133",
        "phase": "reconcile",
        "mode": mode,
        "ok": True,
        "opt125_targets": opt125,
        "windows": rows,
        "family_plan": family_plan(mode, "reconcile"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "reconcile.json", payload)
    fixture = current_fixture()
    fixture["reconcile"] = payload
    persist_fixture(fixture)
    return payload


def _wall_from_summary(summary: Mapping[str, Any] | None) -> float | None:
    if not summary:
        return None
    value = summary.get("instrumented_decode_only_wall_ms")
    if value is None:
        return None
    return float(value)


def run_overhead(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    capture = load_sidecar(run_dir, "capture.json") or {}
    captures = capture.get("captures") or current_fixture().get("captures") or {}
    rows: dict[str, Any] = {}
    for prefix, window in iter_cases():
        key = case_key(prefix, window)
        block = captures.get(key) or {}
        pair = overhead_pair(
            _wall_from_summary(block.get("baseline")),
            _wall_from_summary(block.get("nsys")),
        )
        pair["prefix"] = prefix
        pair["window"] = window
        parsed = load_sidecar(run_dir, f"parsed-{key}.json") or {}
        pair["gpu_busy_ms"] = parsed.get("gpu_busy_ms")
        pair["gpu_idle_ms"] = parsed.get("gpu_idle_ms")
        pair["cuda_api_ms"] = parsed.get("cuda_api_ms")
        rows[key] = pair
    measured = [row for row in rows.values() if row.get("nsys_overhead_ms") is not None]
    mean_overhead = (
        sum(float(row["nsys_overhead_ms"]) for row in measured) / len(measured)
        if measured
        else None
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-133",
        "phase": "overhead",
        "mode": mode,
        "ok": True,
        "windows": rows,
        "mean_nsys_overhead_ms": mean_overhead,
        "family_plan": family_plan(mode, "overhead"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "overhead.json", payload)
    fixture = current_fixture()
    fixture["overhead"] = payload
    persist_fixture(fixture)
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "unmeasured"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def answers_from_phases(
    reconcile: Mapping[str, Any], overhead: Mapping[str, Any]
) -> dict[str, Any]:
    windows = reconcile.get("windows") or {}
    idle_vals = [
        float(row["explained_by_idle_ms"])
        for row in windows.values()
        if row.get("from_parsed_trace")
    ]
    api_vals = [
        float(row["explained_by_api_ms"])
        for row in windows.values()
        if row.get("from_parsed_trace")
    ]
    unresolved_vals = [
        float(row["unresolved_ms"])
        for row in windows.values()
        if row.get("from_parsed_trace")
    ]
    if not unresolved_vals:
        unresolved_vals = [
            float(row["opt125_unobserved_ms"])
            for row in windows.values()
            if row.get("opt125_unobserved_ms") is not None
        ]
    unobserved_vals = [
        float(row["opt125_unobserved_ms"])
        for row in windows.values()
        if row.get("opt125_unobserved_ms") is not None
    ]
    mean_idle = sum(idle_vals) / len(idle_vals) if idle_vals else None
    mean_api = sum(api_vals) / len(api_vals) if api_vals else None
    mean_unresolved = (
        sum(unresolved_vals) / len(unresolved_vals) if unresolved_vals else None
    )
    mean_unobserved = (
        sum(unobserved_vals) / len(unobserved_vals) if unobserved_vals else None
    )
    return {
        "proven_gpu_idle": {
            "mean_explained_by_idle_ms": mean_idle,
            "mean_opt125_unobserved_ms": mean_unobserved,
            "fraction_of_unobserved": (
                mean_idle / mean_unobserved
                if mean_idle is not None and mean_unobserved
                else None
            ),
        },
        "host_api_submission": {
            "mean_explained_by_api_ms": mean_api,
            "fraction_of_unobserved": (
                mean_api / mean_unobserved
                if mean_api is not None and mean_unobserved
                else None
            ),
        },
        "unresolved": {
            "mean_unresolved_ms": mean_unresolved,
            "fraction_of_unobserved": (
                mean_unresolved / mean_unobserved
                if mean_unresolved is not None and mean_unobserved
                else None
            ),
        },
        "nsys_wrapper_overhead": {
            "mean_nsys_overhead_ms": overhead.get("mean_nsys_overhead_ms"),
            "windows": overhead.get("windows") or {},
        },
    }


def write_report(payload: Mapping[str, Any]) -> None:
    answers = payload.get("answers") or {}
    idle = answers.get("proven_gpu_idle") or {}
    api = answers.get("host_api_submission") or {}
    unresolved = answers.get("unresolved") or {}
    overhead = answers.get("nsys_wrapper_overhead") or {}
    preflight = payload.get("preflight") or {}
    traces = []
    for key, row in (payload.get("captures") or {}).items():
        path = row.get("evidence_trace_path") or row.get("trace_path")
        traces.append(f"- `{key}`: `{path}`")
    recon_rows = ((payload.get("reconcile") or {}).get("windows")) or {}
    table = [
        "| Window | OPT-125 unobserved_ms | nsys gpu_idle_ms | nsys cuda_api_ms | unresolved_ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in (case_key(prefix, window) for prefix, window in iter_cases()):
        row = recon_rows.get(key) or {}
        table.append(
            f"| {key} | {_fmt(row.get('opt125_unobserved_ms'))} | "
            f"{_fmt(row.get('nsys_gpu_idle_ms'))} | "
            f"{_fmt(row.get('nsys_cuda_api_ms'))} | "
            f"{_fmt(row.get('unresolved_ms'))} |"
        )
    oh_table = [
        "| Window | baseline_ms | nsys_ms | nsys_overhead_ms | ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, row in (overhead.get("windows") or {}).items():
        oh_table.append(
            f"| {key} | {_fmt(row.get('baseline_ms'))} | "
            f"{_fmt(row.get('nsys_ms'))} | "
            f"{_fmt(row.get('nsys_overhead_ms'))} | "
            f"{_fmt(row.get('nsys_overhead_ratio'))} |"
        )
    lines = [
        "# OPT-133 — Nsight Systems decode activity",
        "",
        "Diagnostics only. `claims_throughput=false`. "
        f"Shipping selector remains `{SELECTOR}`.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`. "
        f"Image `{IMAGE}`. nsys_available="
        f"`{preflight.get('nsys_available')}`. "
        f"nsys=`{preflight.get('nsys_bin')}` version="
        f"`{preflight.get('nsys_version') or 'unknown'}`.",
        "",
        "## Answers",
        "",
        "1. **Proven GPU idle** (Nsight hardware inactivity inside the "
        "bounded 12-token window, compared to OPT-125 unobserved): "
        f"mean explained-by-idle `{_fmt(idle.get('mean_explained_by_idle_ms'))}` ms, "
        f"fraction of OPT-125 unobserved "
        f"`{_fmt(idle.get('fraction_of_unobserved'))}`. "
        + (
            "Hardware GPU trace parsed from `gputrace`."
            if payload.get("hardware_gpu_trace")
            else "Hardware idle/API remain unmeasured when `gputrace`/`cudaapisum` "
            "do not parse from the retained trace."
        ),
        "2. **Host/API submission** visible to Nsight but not CUDA-event leaves: "
        f"mean explained-by-API `{_fmt(api.get('mean_explained_by_api_ms'))}` ms, "
        f"fraction `{_fmt(api.get('fraction_of_unobserved'))}`. "
        "NVTX push/pop ranges (`qw38.ffn`, `qw38.graph_launch`, mixer/attention) "
        "are retained in the trace when present.",
        "3. **Unresolved** after both instruments: "
        f"mean `{_fmt(unresolved.get('mean_unresolved_ms'))}` ms, "
        f"fraction `{_fmt(unresolved.get('fraction_of_unobserved'))}`. "
        "This remainder is not relabeled GPU idle.",
        "4. **nsys wrapper overhead** on the bounded window: "
        f"mean `{_fmt(overhead.get('mean_nsys_overhead_ms'))}` ms.",
        "",
        "## Reconciliation",
        "",
        *table,
        "",
        "OPT-125 windows were 4 tokens; values are scaled to the 12-token "
        "nsys window. `classify_disjoint_intervals(..., profiler="
        '"nsight_systems")` runs only when `gpu_idle_ms` comes from a parsed '
        "gputrace, and proven inactive is overwritten with that idle — not the "
        "CUDA-event stub relabel.",
        "",
        "## Overhead",
        "",
        *oh_table,
        "",
        "## Raw traces",
        "",
        *(traces or ["- none captured"]),
        "",
        "GPU residents at preflight:",
        "",
        f"```json\n{json.dumps(preflight.get('gpu_residents') or [], indent=2)}\n```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise NsysTraceError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise NsysTraceError("claims_throughput must be false")
    if payload.get("claims_performance_improvement") is not False:
        raise NsysTraceError("claims_performance_improvement must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise NsysTraceError("selector drifted from decode_segments8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    preflight = load_sidecar(run_dir, "preflight.json") or current_fixture().get(
        "preflight"
    )
    capture = load_sidecar(run_dir, "capture.json") or {}
    parse_payload = load_sidecar(run_dir, "parse.json") or {}
    reconcile = (
        load_sidecar(run_dir, "reconcile.json")
        or current_fixture().get("reconcile")
        or {}
    )
    overhead = (
        load_sidecar(run_dir, "overhead.json")
        or current_fixture().get("overhead")
        or {}
    )
    answers = answers_from_phases(reconcile, overhead)
    captures = capture.get("captures") or current_fixture().get("captures") or {}
    parsed = parse_payload.get("parsed") or current_fixture().get("parsed") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-133",
        "status": "measured" if parsed else "structure",
        "mode": mode,
        "measurement_utc": utc_now(),
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "windows": list(WINDOWS),
        "window_tokens": WINDOW_TOKENS,
        "decode_output_tokens": DECODE_TOKENS,
        "preflight": preflight,
        "captures": captures,
        "parsed": parsed,
        "reconcile": reconcile,
        "overhead": overhead,
        "answers": answers,
        "hardware_gpu_trace": parse_payload.get("hardware_gpu_trace"),
        "report_path": str(contract["report_path"]),
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    validate_fixture(payload)
    persist_fixture(payload)
    write_report(payload)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "capture":
        return run_capture(run_dir, mode)
    if phase == "parse":
        return run_parse(run_dir, mode)
    if phase == "reconcile":
        return run_reconcile(run_dir, mode)
    if phase == "overhead":
        return run_overhead(run_dir, mode)
    if phase == "report":
        return run_report(run_dir, mode)
    raise NsysTraceError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    result = run_phase(args.phase, Path(args.run_dir), args.mode)
    json.dump({"phase": args.phase, "ok": bool(result.get("ok", True))}, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("ok", True) or args.phase == "report" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NsysTraceError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
