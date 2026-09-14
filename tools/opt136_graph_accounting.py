"""OPT-136 graph-aware decode accounting harness.

Diagnostics only. Production selectors and arithmetic stay unchanged.
Phases: historical, preflight, baseline, capture, analyze, report.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_REV,
    docker_common,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools import opt133_decode_nsys_trace as opt133_nsys  # noqa: E402
from tools.performance_evidence import (  # noqa: E402
    T_CRIT_DF4_TWO_SIDED,
    T_CRIT_DF9_TWO_SIDED,
    aa_usable,
    audit_window_from_tables,
    compare_opt133_expected,
    default_identity,
    historical_reconciliation_invalid,
    opt133_expected_128_early,
    paired_log_ratio_ci,
    parse_nsys_sqlite,
    validate_coverage_document,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt136_graph_accounting_contract.json"
ITERATION = ROOT / "pins/opt136_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt136_graph_accounting.json"
EVIDENCE = ROOT / "evidence/optimization/opt136-graph-accounting"
REPORT = EVIDENCE / "REPORT.md"
NATIVE = "build/qw38-cuda-opt136-matched-decode-profile-test"
LLAMA_BIN = ".cache/authorities/llama-build-opt136/bin/qw38-llama-opt136-decode-profile"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
GPU_LOCK = ROOT / "build/optimization-runs/qw38-gpu.lock"
RESULT_PREFIX = "QW38_OPT136_GRAPH_ACCOUNTING_RESULT="
COUNTS_PREFIX = "QW38_OPT136_NATIVE_COUNTS="
PARENT = "post124_plus_opt127_decode_segments8"
SELECTOR = "decode_segments8"
PREFIXES = (128, 2048, 8192, 32768)
WINDOWS = ("early", "middle", "late")
WINDOW_STEPS = ((0, 12), (122, 134), (244, 256))
WINDOW_TOKENS = 12
DECODE_TOKENS = 256
CAPACITY = 131072
ENGINES = ("quartz", "llama")
ARMS = ("unprofiled", "graph", "node")
REPETITIONS = 3
BASELINE_WARMUPS = 3
BASELINE_PAIRS = 10
AA_WARMUPS = 3
AA_PAIRS = 5
CHILD_TIMEOUT_S = 300
PHASES = (
    "historical",
    "preflight",
    "baseline",
    "capture",
    "analyze",
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
    "historical",
    "preflight",
    "baseline",
    "captures",
    "analyze",
    "answers",
    "report_path",
)
OPT133_SQLITE = ROOT / "build/optimization-runs/opt133-nsys2025"
OPT133_TRACES = ROOT / "evidence/optimization/opt133-decode-nsys-trace/traces"
OPT133_FIXTURE = ROOT / "fixtures/opt133_decode_nsys_trace.json"
OPT133_REPORT = ROOT / "evidence/optimization/opt133-decode-nsys-trace/REPORT.md"
OPT133_DOSSIER = ROOT / "tasks/OPT-133.md"
OPT125_REPORT = ROOT / "evidence/optimization/opt125-decode-accounting/REPORT.md"
OPT125_DOSSIER = ROOT / "tasks/OPT-125.md"
OPT132_REPORT = ROOT / "evidence/optimization/opt132-combined-decode/REPORT.md"
OPT132_DOSSIER = ROOT / "tasks/OPT-132.md"
CORRECTION_DATE = "2026-09-14"
IDLE_INVALID_REASON = (
    "OPT-133 subtracted ordinary kernel/copy union from the window and labeled "
    "the remainder GPU idle while CUPTI_ACTIVITY_KIND_GRAPH_TRACE envelopes "
    "were omitted from cuda_gpu_trace. Graph-internal activity is unknown "
    "until node tracing covers it. min(old_unobserved, new_idle) is not causal."
)
IDLE_LINK = (
    "evidence/optimization/opt136-graph-accounting/historical-reconciliation.json"
)


class GraphAccountingError(RuntimeError):
    """OPT-136 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-136":
        raise GraphAccountingError("graph-accounting contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-136", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-136 mode={mode} phase={family} "
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


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def empty_fixture(mode: str = "feedback") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-136",
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
        "decode_eval_tokens": DECODE_TOKENS,
        "allocated_capacity": CAPACITY,
        "historical": None,
        "preflight": None,
        "baseline": None,
        "captures": {},
        "analyze": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        payload = load_json(FIXTURE)
        if isinstance(payload, dict) and payload.get("task") == "OPT-136":
            return payload
    return empty_fixture()


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise GraphAccountingError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise GraphAccountingError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise GraphAccountingError("selector drifted from decode_segments8")


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


def with_gpu_lock(
    command: Sequence[str], *, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str]:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR, 0x1A4)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        residents = gpu_residents()
        (GPU_LOCK.parent / "opt136-residents-before.json").write_text(
            json.dumps(residents, indent=2) + "\n", encoding="utf-8"
        )
        return subprocess.run(
            list(command),
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            list(command),
            124,
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or ""),
            (
                exc.stderr.decode("utf-8", "replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            + f"\ntimeout after {timeout_s}s",
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *list(args)]


def nsys_profile_command(
    output: Path | str,
    binary_args: Sequence[str],
    *,
    graph_trace: str,
) -> list[str]:
    stem = str(output)
    command = [
        opt133_nsys.NSYS_BIN,
        "profile",
        "-o",
        stem,
        "--force-overwrite",
        "true",
        "--trace=cuda,nvtx,osrt",
        f"--cuda-graph-trace={graph_trace}",
        "--sample=none",
        "--cpuctxsw=none",
    ]
    if opt133_nsys.modern_nsys():
        command.extend(
            ["--capture-range=cudaProfilerApi", "--capture-range-end=repeat"]
        )
    else:
        command.extend(
            [
                "--kill=none",
                "--capture-range=nvtx",
                "--nvtx-capture=opt136.window",
                "--capture-range-end=stop",
            ]
        )
    command.extend(binary_args)
    return command


def nsys_export_sqlite(
    trace: Path, sqlite_path: Path
) -> subprocess.CompletedProcess[str]:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        opt133_nsys.ensure_nsys_layout()
    except Exception as exc:  # noqa: BLE001
        return subprocess.CompletedProcess(
            ["nsys", "export"], 1, "", f"nsys_layout: {exc}"
        )
    rel_trace = relpath(Path(trace).resolve())
    rel_sqlite = relpath(Path(sqlite_path).resolve())
    return subprocess.run(
        [
            *docker_common(IMAGE, "acceptance"),
            opt133_nsys.NSYS_BIN,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            rel_sqlite,
            rel_trace,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CHILD_TIMEOUT_S,
    )


def capture_trace_files(stem: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    direct = Path(str(stem) + ".nsys-rep")
    if direct.is_file() and direct.stat().st_size > 0:
        found.append(("all", direct))
    for index, name in enumerate(WINDOWS, start=1):
        path = Path(f"{stem}.{index}.nsys-rep")
        if path.is_file() and path.stat().st_size > 0:
            found.append((name, path))
    return found


def export_capture_traces(stem: Path) -> dict[str, str]:
    exported: dict[str, str] = {}
    for window, trace in capture_trace_files(stem):
        sqlite_path = trace.with_name(trace.name.replace(".nsys-rep", ".sqlite"))
        if not sqlite_path.is_file():
            result = nsys_export_sqlite(trace, sqlite_path)
            if result.returncode != 0 or not sqlite_path.is_file():
                continue
        exported[window] = relpath(sqlite_path)
    return exported


def authenticate_current_pins() -> dict[str, Any]:
    header = (ROOT / "cuda/execution_graph_path.cuh").read_text(encoding="utf-8")
    q4 = (ROOT / "cuda/q4k_decode_path.cuh").read_text(encoding="utf-8")
    graphs_ok = 'kSelectedExecutionGraphPath[] = "decode_segments8"' in header
    q4_ok = 'kSelectedQ4DecodePath[] = "llama_q4k_mmvq"' in q4
    return {
        "ok": graphs_ok and q4_ok,
        "execution_graphs": SELECTOR,
        "q4_decode": "llama_q4k_mmvq",
        "ffn_only_prerequisite_not_used": True,
        "mismatches": []
        if graphs_ok and q4_ok
        else ["current pins are not decode_segments8 + llama_q4k_mmvq"],
    }


def opt133_windows() -> list[tuple[int, str]]:
    return [
        (128, "early"),
        (128, "middle"),
        (128, "late"),
        (2048, "early"),
        (2048, "middle"),
        (2048, "late"),
    ]


def find_opt133_sqlite(prefix: int, window: str) -> Path | None:
    name = f"trace-{prefix}-{window}.sqlite"
    for folder in (
        OPT133_SQLITE,
        ROOT / "build/optimization-runs/opt133",
        EVIDENCE / "opt133-copies",
    ):
        candidate = folder / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_opt133_trace(prefix: int, window: str) -> Path | None:
    for folder in (OPT133_TRACES, ROOT / "build/optimization-runs/opt133"):
        for suffix in (".nsys-rep", ".qdrep"):
            candidate = folder / f"trace-{prefix}-{window}{suffix}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def copy_trace_readonly(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return dest


def append_unique_block(path: Path, heading: str, body: str) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if heading in text:
        return
    path.write_text(text.rstrip() + "\n\n" + body.rstrip() + "\n", encoding="utf-8")


def apply_opt133_parser_correction() -> dict[str, Any]:
    """Keep raw OPT-133 numbers; invalidate derived idle without rewriting arrays."""
    from tools import opt133_decode_nsys_trace as opt133

    fixture = opt133.current_fixture()
    historical = fixture.get("opt136_historical_idle") or {}
    if not historical:
        parsed = fixture.get("parsed") or {}
        windows = {}
        for key, row in parsed.items() if isinstance(parsed, dict) else []:
            if not isinstance(row, dict):
                continue
            windows[key] = {
                "gpu_idle_ms": row.get("gpu_idle_ms"),
                "gpu_busy_ms": row.get("gpu_busy_ms"),
                "cuda_api_ms": row.get("cuda_api_ms"),
            }
        answers = fixture.get("answers") or {}
        historical = {
            "parsed_gpu_idle_ms": windows,
            "answers_proven_gpu_idle": answers.get("proven_gpu_idle"),
            "preserved_raw_sample_arrays": True,
        }
        fixture["opt136_historical_idle"] = historical
    correction = {
        "date": CORRECTION_DATE,
        "gpu_idle_ms": None,
        "device_inactive_proven_ms": None,
        "idle_claim_valid": False,
        "causal_reconciliation": False,
        "reason": IDLE_INVALID_REASON,
        "link": IDLE_LINK,
        "historical_values_preserved_under": "opt136_historical_idle",
    }
    fixture["opt136_correction"] = correction
    answers = dict(fixture.get("answers") or {})
    proven = dict(answers.get("proven_gpu_idle") or {})
    if "historical_mean_explained_by_idle_ms" not in proven:
        proven["historical_mean_explained_by_idle_ms"] = proven.get(
            "mean_explained_by_idle_ms"
        )
    proven["mean_explained_by_idle_ms"] = None
    proven["valid"] = False
    proven["reason"] = IDLE_INVALID_REASON
    proven["link"] = IDLE_LINK
    answers["proven_gpu_idle"] = proven
    fixture["answers"] = answers
    opt133.persist_fixture(fixture)
    return {"fixture": relpath(OPT133_FIXTURE), "correction": correction}


def correction_markdown() -> str:
    return f"""## OPT-136 dated correction ({CORRECTION_DATE})

Derived GPU idle from OPT-133 `cuda_gpu_trace` is **invalid**. The capture used
`--cuda-graph-trace=graph`, but the GPU sum omitted
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`. Subtracting ordinary kernel/copy union from
the window does not prove hardware idle; graph-internal activity is unknown
until node tracing covers those envelopes. Cross-run
`min(old_unobserved, new_idle)` is **not causal**.

Raw OPT-133 sample arrays and historical admission are preserved. Current
derived idle fields are null. Evidence:
[`{IDLE_LINK}`](../../{IDLE_LINK}).

Do not treat graph envelopes as continuous busy time, and do not replace the
old ~188 ms idle claim with a claim of ~194 ms continuous hardware busy time.
"""


def append_report_corrections() -> list[str]:
    updated: list[str] = []
    heading = f"## OPT-136 dated correction ({CORRECTION_DATE})"
    body = correction_markdown()
    for path in (
        OPT133_REPORT,
        OPT125_REPORT,
        OPT132_REPORT,
        OPT133_DOSSIER,
        OPT125_DOSSIER,
        OPT132_DOSSIER,
    ):
        if path.is_file():
            append_unique_block(path, heading, body)
            updated.append(relpath(path))
    return updated


def run_historical(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    copies = run_dir / "opt133-trace-copies"
    sql_dir = run_dir / "opt133-sqlite"
    sql_dir.mkdir(parents=True, exist_ok=True)
    windows: dict[str, Any] = {}
    sql_log: list[str] = []
    blockers: list[str] = []
    for prefix, window in opt133_windows():
        key = f"d{prefix}-{window}"
        sqlite_path = find_opt133_sqlite(prefix, window)
        trace = find_opt133_trace(prefix, window)
        exported = False
        if sqlite_path is None and trace is not None:
            try:
                opt133_nsys.ensure_nsys_layout()
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"{key}: nsys_layout: {exc}")
                sqlite_path = None
            else:
                copied = copy_trace_readonly(trace, copies)
                sqlite_path = sql_dir / f"trace-{prefix}-{window}.sqlite"
                export = nsys_export_sqlite(copied, sqlite_path)
                exported = export.returncode == 0 and sqlite_path.is_file()
                if not exported:
                    blockers.append(
                        f"{key}: nsys export failed: "
                        f"{(export.stderr or export.stdout)[-400:]}"
                    )
                    sqlite_path = None
        record: dict[str, Any] = {
            "prefix": prefix,
            "window": window,
            "sqlite": relpath(sqlite_path) if sqlite_path else None,
            "trace": relpath(trace) if trace else None,
            "exported": exported,
        }
        if sqlite_path is None:
            record["ok"] = False
            record["reason"] = "sqlite_and_nsys_rep_unavailable"
            blockers.append(f"{key}: no sqlite or nsys-rep")
            windows[key] = record
            continue
        tables = parse_nsys_sqlite(sqlite_path)
        expected_graphs = 96
        row = audit_window_from_tables(
            tables,
            expected_graph_launches=expected_graphs,
            identity=default_identity(
                prefix=prefix,
                eval_count=WINDOW_TOKENS,
                populated_length=prefix,
                capture_arm="graph",
            ),
        )
        compared = None
        if prefix == 128 and window == "early":
            compared = compare_opt133_expected(
                row["quantities_ms"], opt133_expected_128_early()
            )
        record.update(
            {
                "ok": True,
                "quantities_ms": row["quantities_ms"],
                "coverage": row,
                "expected_128_early": compared,
                "sql": row.get("sql"),
                "table_counts": tables.table_counts,
                "errors": tables.errors,
            }
        )
        sql_log.append(
            f"{key}: SELECT start,end FROM KERNEL/MEMCPY/GRAPH_TRACE; "
            f"kernels={row['observed_kernel_count']} graphs={row['observed_graph_launches']}"
        )
        windows[key] = record
    opt125 = {}
    if (ROOT / "fixtures/opt125_decode_accounting.json").is_file():
        opt125 = load_json(ROOT / "fixtures/opt125_decode_accounting.json")
    unobserved = None
    activity = (opt125.get("activity_trace") or {}).get("d128") or {}
    unobserved = (activity.get("intervals") or {}).get("unobserved_ms")
    early = (windows.get("d128-early") or {}).get("quantities_ms") or {}
    historical_idle = None
    if OPT133_FIXTURE.is_file():
        parsed = (load_json(OPT133_FIXTURE).get("parsed") or {}).get("d128-early") or {}
        historical_idle = parsed.get("gpu_idle_ms")
    reconciliation = {
        "schema_version": 1,
        "task": "OPT-136",
        "phase": "historical",
        "date": CORRECTION_DATE,
        "windows": windows,
        "opt133_d128_early_expected": opt133_expected_128_early(),
        "opt133_d128_early_observed": early,
        "idle_invalidation": historical_reconciliation_invalid(
            old_unobserved_ms=float(unobserved) if unobserved is not None else 126.3,
            new_idle_ms=float(historical_idle)
            if historical_idle is not None
            else 188.5,
            reason=IDLE_INVALID_REASON,
            link=IDLE_LINK,
        ),
        "graph_envelopes_are_not_continuous_busy": True,
        "sql": sql_log,
        "blockers": blockers,
        "ok": not blockers and bool(windows.get("d128-early", {}).get("ok")),
    }
    parser = apply_opt133_parser_correction()
    reports = append_report_corrections()
    reconciliation["parser_correction"] = parser
    reconciliation["appended_reports"] = reports
    dump_json(EVIDENCE / "historical-reconciliation.json", reconciliation)
    dump_json(run_dir / "historical.json", reconciliation)
    return store_sidecar(run_dir, "historical.json", reconciliation)


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    available, gpu_blocker = gpu_available()
    residents = gpu_residents() if available else []
    source, dirty = git_identity()
    pins = authenticate_current_pins()
    nsys_layout = {}
    nsys_available = False
    try:
        opt133_nsys.ensure_nsys_layout()
        nsys_layout = opt133_nsys.docker_nsys_layout()
        nsys_available = bool(
            nsys_layout.get("nsys_bin") or Path(opt133_nsys.NSYS_BIN).is_file()
        )
    except Exception as exc:  # noqa: BLE001
        nsys_layout = {"error": str(exc)}
    make = (
        subprocess.run(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt136-diagnostics"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT_S,
        )
        if available
        else subprocess.CompletedProcess(["make"], 1, "", gpu_blocker)
    )
    native_path = ROOT / NATIVE
    native_hash = sha256_file(native_path)
    llama_script = ROOT / "tools/llama_authority/build_opt136_profile.sh"
    llama = {"ok": False, "reason": "not_started"}
    if llama_script.is_file() and available:
        built = subprocess.run(
            ["bash", str(llama_script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT_S,
        )
        llama = {
            "ok": built.returncode == 0 and (ROOT / LLAMA_BIN).is_file(),
            "returncode": built.returncode,
            "stdout_tail": (built.stdout or "")[-1500:],
            "stderr_tail": (built.stderr or "")[-1500:],
            "hash": sha256_file(ROOT / LLAMA_BIN),
        }
    elif not available:
        llama = {"ok": False, "reason": gpu_blocker or "gpu_unavailable"}
    identity = {}
    if available and native_path.is_file():
        completed = with_gpu_lock(
            native_command(
                [
                    f"./{NATIVE}",
                    "--workload",
                    "identity",
                    "--prefix",
                    "128",
                    "--capacity",
                    str(CAPACITY),
                    MODEL,
                ],
                tier="correctness",
            ),
            timeout_s=CHILD_TIMEOUT_S,
        )
        (run_dir / "identity.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        identity = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_tail": (completed.stdout or "")[-2000:],
        }
    hashes = {
        "quartz_native": native_hash,
        "llama_profile": llama.get("hash"),
        "gguf_sha256": GGUF_SHA,
        "llama_revision": LLAMA_REV,
    }
    blocked = []
    if not available:
        blocked.append(gpu_blocker or "gpu_unavailable")
    if not nsys_available:
        blocked.append("nsys_unavailable")
    if not native_path.is_file():
        blocked.append("quartz_native_missing")
    if not identity.get("ok"):
        blocked.append("quartz_identity_failed")
    if not llama.get("ok"):
        blocked.append("llama_profile_missing")
    ok = (
        available
        and nsys_available
        and native_path.is_file()
        and pins["ok"]
        and bool(identity.get("ok"))
        and bool(llama.get("ok"))
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-136",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "gpu_residents": residents,
        "exclusive_sitting": len(residents) <= 1,
        "nsys_available": nsys_available,
        "nsys_bin": nsys_layout.get("nsys_bin") or opt133_nsys.NSYS_BIN,
        "nsys_host": nsys_layout.get("host_dir") or opt133_nsys.NSYS_HOST,
        "nsys_version": (opt133_nsys.nsys_version_text() or "").strip() or None,
        "nsys_layout": nsys_layout,
        "image": IMAGE,
        "qdstrm_importer": nsys_layout.get("qdstrm_importer")
        or opt133_nsys.QDSTRM_IMPORTER,
        "native": NATIVE,
        "llama_bin": LLAMA_BIN,
        "hashes": hashes,
        "make_returncode": make.returncode,
        "make_stderr_tail": (make.stderr or "")[-2000:],
        "llama_build": llama,
        "identity": identity,
        "authenticated_current_pins": pins,
        "source": source,
        "dirty": dirty,
        "capacity": CAPACITY,
        "prefixes": list(PREFIXES),
        "eval_tokens": DECODE_TOKENS,
        "token_generator": "(42 + index * 997) % 248320",
        "gpu_lock": relpath(GPU_LOCK),
        "claims_throughput": False,
        "blocked": blocked,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def _blocked_gpu_phase(
    phase: str, run_dir: Path, mode: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-136",
        "phase": phase,
        "mode": mode,
        "ok": False,
        "status": "incomplete",
        "blocked": True,
        "reason": preflight.get("blocked") or ["gpu_or_nsys_unavailable"],
        "gpu_available": preflight.get("gpu_available"),
        "nsys_available": preflight.get("nsys_available"),
        "hashes": preflight.get("hashes"),
        "claims_throughput": False,
        "family_plan": family_plan(
            mode, phase if phase in load_iteration()["workloads"] else "report"
        ),
        "measured_at": utc_now(),
    }
    if phase == "baseline":
        dump_json(EVIDENCE / "baseline.json", payload)
    if phase == "capture":
        dump_json(
            EVIDENCE / "trace-manifest.json",
            {
                "status": "incomplete",
                "blocked": True,
                "reason": payload["reason"],
                "captures": {},
                "commands": [],
            },
        )
    return store_sidecar(run_dir, f"{phase}.json", payload)


def quartz_args(
    *,
    workload: str,
    prefix: int,
    attribution: bool,
) -> list[str]:
    return [
        f"./{NATIVE}",
        "--workload",
        workload,
        "--prefix",
        str(prefix),
        "--tokens",
        str(DECODE_TOKENS),
        "--capacity",
        str(CAPACITY),
        "--attribution",
        "on" if attribution else "off",
        MODEL,
    ]


def llama_args(*, workload: str, prefix: int) -> list[str]:
    return [
        LLAMA_BIN,
        MODEL,
        "--workload",
        workload,
        "--prefix",
        str(prefix),
        "--tokens",
        str(DECODE_TOKENS),
        "--ctx",
        str(CAPACITY),
    ]


def run_baseline(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("baseline", run_dir, mode, preflight)
    results: dict[str, Any] = {}
    # GPU work is executed only when preflight succeeded. Alternating arms are
    # never loaded together: each child process is a separate docker run.
    for prefix in PREFIXES:
        for warmup in range(BASELINE_WARMUPS):
            for engine, args in (
                (
                    "quartz",
                    quartz_args(
                        workload="unprofiled", prefix=prefix, attribution=False
                    ),
                ),
                ("llama", llama_args(workload="unprofiled", prefix=prefix)),
            ):
                key = f"warmup-d{prefix}-{engine}-{warmup}"
                binary = ROOT / (NATIVE if engine == "quartz" else LLAMA_BIN)
                if not binary.is_file():
                    results[key] = {"ok": False, "reason": f"missing {binary}"}
                    continue
                completed = with_gpu_lock(
                    native_command(args, tier="acceptance"),
                    timeout_s=CHILD_TIMEOUT_S,
                )
                results[key] = {
                    "ok": completed.returncode == 0,
                    "returncode": completed.returncode,
                    "stdout_tail": (completed.stdout or "")[-1000:],
                }
                sys.stderr.write(f"OPT-136 baseline {key} rc={completed.returncode}\n")
                sys.stderr.flush()
        quartz_ms: list[float] = []
        llama_ms: list[float] = []
        for sample in range(BASELINE_PAIRS):
            for engine, args, bucket in (
                (
                    "quartz",
                    quartz_args(
                        workload="unprofiled", prefix=prefix, attribution=False
                    ),
                    quartz_ms,
                ),
                ("llama", llama_args(workload="unprofiled", prefix=prefix), llama_ms),
            ):
                key = f"pair-d{prefix}-{engine}-{sample}"
                binary = ROOT / (NATIVE if engine == "quartz" else LLAMA_BIN)
                if not binary.is_file():
                    results[key] = {"ok": False, "reason": f"missing {binary}"}
                    continue
                completed = with_gpu_lock(
                    native_command(args, tier="acceptance"),
                    timeout_s=CHILD_TIMEOUT_S,
                )
                wall = None
                for line in (completed.stdout or "").splitlines():
                    if line.startswith(RESULT_PREFIX):
                        try:
                            parsed = json.loads(line[len(RESULT_PREFIX) :])
                            wall = parsed.get("decode_only_ms")
                        except json.JSONDecodeError:
                            wall = None
                if wall is not None:
                    bucket.append(float(wall))
                results[key] = {
                    "ok": completed.returncode == 0,
                    "decode_only_ms": wall,
                    "returncode": completed.returncode,
                }
                sys.stderr.write(
                    f"OPT-136 baseline {key} rc={completed.returncode} ms={wall}\n"
                )
                sys.stderr.flush()
        results[f"d{prefix}"] = {
            "quartz_decode_only_ms": quartz_ms,
            "llama_decode_only_ms": llama_ms,
            "n_pairs": min(len(quartz_ms), len(llama_ms)),
        }
    aa: dict[str, Any] = {}
    for prefix in (128, 2048):
        a: list[float] = []
        b: list[float] = []
        for warmup in range(AA_WARMUPS):
            with_gpu_lock(
                native_command(
                    quartz_args(
                        workload="unprofiled", prefix=prefix, attribution=False
                    ),
                    tier="acceptance",
                ),
                timeout_s=CHILD_TIMEOUT_S,
            )
        for sample in range(AA_PAIRS):
            first = with_gpu_lock(
                native_command(
                    quartz_args(
                        workload="unprofiled", prefix=prefix, attribution=False
                    ),
                    tier="acceptance",
                ),
                timeout_s=CHILD_TIMEOUT_S,
            )
            second = with_gpu_lock(
                native_command(
                    quartz_args(
                        workload="unprofiled", prefix=prefix, attribution=False
                    ),
                    tier="acceptance",
                ),
                timeout_s=CHILD_TIMEOUT_S,
            )
            for completed, bucket in ((first, a), (second, b)):
                for line in (completed.stdout or "").splitlines():
                    if line.startswith(RESULT_PREFIX):
                        parsed = json.loads(line[len(RESULT_PREFIX) :])
                        if parsed.get("decode_only_ms"):
                            bucket.append(float(parsed["decode_only_ms"]))
        ci = (
            paired_log_ratio_ci(a, b, critical=T_CRIT_DF4_TWO_SIDED)
            if a and b
            else {"usable": False}
        )
        aa[f"d{prefix}"] = {"a": a, "b": b, "ci": ci, "usable": aa_usable(ci)}
    payload = {
        "schema_version": 1,
        "task": "OPT-136",
        "phase": "baseline",
        "mode": mode,
        "ok": True,
        "results": results,
        "aa": aa,
        "hashes": preflight.get("hashes"),
        "claims_throughput": False,
        "metric": "decode_only_ms",
        "family_plan": family_plan(mode, "baseline"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "baseline.json", payload)
    return store_sidecar(run_dir, "baseline.json", payload)


def run_capture(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("capture", run_dir, mode, preflight)
    traces = run_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    captures: dict[str, Any] = {}
    commands: list[str] = []
    for prefix in PREFIXES:
        for engine in ENGINES:
            for arm in ARMS:
                for rep in range(REPETITIONS):
                    key = f"{engine}-d{prefix}-{arm}-r{rep}"
                    stem = traces / key
                    if arm == "unprofiled":
                        workload = "unprofiled"
                        graph_trace = None
                    elif arm == "graph":
                        workload = "graph-capture"
                        graph_trace = "graph"
                    else:
                        workload = "node-capture"
                        graph_trace = "node"
                    if engine == "quartz":
                        args = quartz_args(
                            workload=workload,
                            prefix=prefix,
                            attribution=False,
                        )
                    else:
                        args = llama_args(workload=workload, prefix=prefix)
                    if graph_trace:
                        command = nsys_profile_command(
                            stem, args, graph_trace=graph_trace
                        )
                    else:
                        command = list(args)
                    commands.append(" ".join(shlex.quote(part) for part in command))
                    binary = ROOT / (NATIVE if engine == "quartz" else LLAMA_BIN)
                    if not binary.is_file():
                        captures[key] = {"ok": False, "reason": f"missing {binary}"}
                        continue
                    wrapped = native_command(command, tier="acceptance")
                    completed = with_gpu_lock(wrapped, timeout_s=CHILD_TIMEOUT_S)
                    captures[key] = {
                        "ok": completed.returncode == 0,
                        "returncode": completed.returncode,
                        "command": command,
                        "stdout_tail": (completed.stdout or "")[-1500:],
                        "stderr_tail": (completed.stderr or "")[-1500:],
                    }
                    sys.stderr.write(
                        f"OPT-136 capture {key} rc={completed.returncode}\n"
                    )
                    sys.stderr.flush()
                    exported = export_capture_traces(stem)
                    if exported:
                        captures[key]["sqlite_by_window"] = exported
                        captures[key]["sqlite"] = next(iter(exported.values()))
    payload = {
        "schema_version": 1,
        "task": "OPT-136",
        "phase": "capture",
        "mode": mode,
        "ok": any(row.get("ok") for row in captures.values()),
        "captures": captures,
        "nsys_commands": commands,
        "hashes": preflight.get("hashes"),
        "three_windows_in_one_process": True,
        "window_steps": [list(item) for item in WINDOW_STEPS],
        "family_plan": family_plan(mode, "capture"),
        "measured_at": utc_now(),
    }
    dump_json(
        EVIDENCE / "trace-manifest.json", {"captures": captures, "commands": commands}
    )
    return store_sidecar(run_dir, "capture.json", payload)


def run_analyze(run_dir: Path, mode: str) -> dict[str, Any]:
    capture = load_sidecar(run_dir, "capture.json") or {}
    baseline = load_sidecar(run_dir, "baseline.json") or {}
    historical = load_sidecar(run_dir, "historical.json") or {}
    windows: list[dict[str, Any]] = []
    family_gaps: dict[str, Any] = {}
    if historical.get("windows"):
        for key, row in historical["windows"].items():
            coverage = row.get("coverage")
            if coverage:
                windows.append(coverage)
    captures = capture.get("captures") or {}
    traces_dir = run_dir / "traces"
    for key, row in list(captures.items()):
        stem = traces_dir / key
        exported = dict(row.get("sqlite_by_window") or {})
        if not exported:
            exported = export_capture_traces(stem)
            if exported:
                row["sqlite_by_window"] = exported
                row["sqlite"] = next(iter(exported.values()))
                captures[key] = row
        engine = key.split("-", 1)[0]
        arm = "graph"
        if "-node-" in key:
            arm = "node"
        elif "-unprofiled-" in key:
            arm = "unprofiled"
        prefix = 128
        for part in key.split("-"):
            if part.startswith("d") and part[1:].isdigit():
                prefix = int(part[1:])
                break
        for window, sqlite_rel in exported.items():
            path = ROOT / sqlite_rel
            if not path.is_file():
                continue
            tables = parse_nsys_sqlite(path)
            coverage = audit_window_from_tables(
                tables,
                identity=default_identity(
                    capture_arm=arm,
                    engine=engine,
                    prefix=prefix,
                    populated_length=prefix,
                    eval_count=WINDOW_TOKENS,
                ),
                expected_graph_launches=(
                    96 if arm == "graph" and engine == "quartz" else None
                ),
            )
            coverage["capture_key"] = key
            coverage["window_name"] = window
            windows.append(coverage)
            family_gaps[f"{key}-{window}"] = coverage.get("families")
    dump_json(
        sidecar(run_dir, "capture.json"),
        {**capture, "captures": captures},
    )
    dump_json(
        EVIDENCE / "trace-manifest.json",
        {
            "captures": captures,
            "commands": capture.get("nsys_commands") or [],
            "three_windows_in_one_process": True,
        },
    )
    if windows:
        document: dict[str, Any] = {
            "schema_version": 1,
            "task": "OPT-136",
            "status": "valid"
            if all(row.get("coverage_valid") for row in windows)
            else "incomplete",
            "windows": windows,
        }
    else:
        document = {
            "schema_version": 1,
            "task": "OPT-136",
            "status": "unavailable",
            "windows": None,
            "reason": "historical sqlite and GPU captures unavailable",
        }
    validation = validate_coverage_document(document)
    document["validation"] = validation
    dump_json(EVIDENCE / "coverage.json", document)
    dump_json(EVIDENCE / "family-gaps.json", family_gaps)
    payload = {
        "schema_version": 1,
        "task": "OPT-136",
        "phase": "analyze",
        "mode": mode,
        "ok": bool(validation.get("ok")) or bool(historical.get("ok")),
        "coverage_validation": validation,
        "window_count": len(windows),
        "family_gaps": family_gaps,
        "baseline_aa": (baseline.get("aa") or {}),
        "historical_ok": historical.get("ok"),
        "family_plan": family_plan(mode, "analyze"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "analyze.json", payload)


def matched_ratio_placeholder(baseline: Mapping[str, Any]) -> dict[str, Any]:
    ratios: dict[str, Any] = {}
    results = baseline.get("results") or {}
    for prefix in PREFIXES:
        block = results.get(f"d{prefix}") or {}
        quartz = block.get("quartz_decode_only_ms") or []
        llama = block.get("llama_decode_only_ms") or []
        if len(quartz) >= 2 and len(llama) >= 2:
            q = sum(quartz) / len(quartz)
            ell = sum(llama) / len(llama)
            tok_q = DECODE_TOKENS * 1000.0 / q if q else None
            tok_l = DECODE_TOKENS * 1000.0 / ell if ell else None
            ratio = (tok_q / tok_l) if tok_q and tok_l else None
            ci = paired_log_ratio_ci(quartz, llama, critical=T_CRIT_DF9_TWO_SIDED)
            ratios[f"d{prefix}"] = {
                "metric": "decode_only",
                "quartz_mean_ms": q,
                "llama_mean_ms": ell,
                "quartz_tok_s": tok_q,
                "llama_tok_s": tok_l,
                "ratio_quartz_over_llama": ratio,
                "ci": ci,
                "status": "measured" if ratio else "incomplete",
            }
        else:
            ratios[f"d{prefix}"] = {
                "metric": "decode_only",
                "status": "incomplete",
                "reason": "baseline_pairs_missing",
            }
    return ratios


def write_report(payload: Mapping[str, Any]) -> None:
    historical = payload.get("historical") or {}
    preflight = payload.get("preflight") or {}
    baseline = payload.get("baseline") or {}
    analyze = payload.get("analyze") or {}
    answers = payload.get("answers") or {}
    early = ((historical.get("windows") or {}).get("d128-early") or {}).get(
        "quantities_ms"
    ) or {}
    lines = [
        "# OPT-136 — Graph accounting and matched production decode",
        "",
        "Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`. Image `{IMAGE}`.",
        f"nsys `{preflight.get('nsys_version')}`. GPU available `{preflight.get('gpu_available')}`.",
        "",
        "## Historical OPT-133 sqlite audit",
        "",
        "Raw timestamps remain integer nanoseconds. Units come from the export schema.",
        "",
        "| Quantity | Expected d128-early | Observed |",
        "|---|---:|---:|",
    ]
    expected = opt133_expected_128_early()
    for key, want in expected.items():
        got = early.get(key)
        lines.append(f"| {key} | {want} | {got} |")
    lines.extend(
        [
            "",
            "Graph envelopes are not leaf kernel time. Internal idle on graph-only",
            "captures is null. `min(old_unobserved, new_idle)` is non-causal.",
            "",
            f"SQL: `{((historical.get('windows') or {}).get('d128-early') or {}).get('sql')}`.",
            "",
            "## Coverage",
            "",
            f"validation ok=`{(analyze.get('coverage_validation') or {}).get('ok')}`",
            f"windows=`{analyze.get('window_count')}`",
            "",
            "## Matched decode ratios (identity-matched decode_only, capacity 131072)",
            "",
        ]
    )
    ratios = answers.get("matched_decode_ratios") or {}
    for key, row in ratios.items():
        lines.append(
            f"- {key}: status=`{row.get('status')}` quartz_tok_s=`{row.get('quartz_tok_s')}` "
            f"llama_tok_s=`{row.get('llama_tok_s')}` ratio=`{row.get('ratio_quartz_over_llama')}`"
        )
    lines.extend(
        [
            "",
            "## Contradictions",
            "",
            "- OPT-108/129 component times come from different sittings than this",
            "  matched decode probe; they are not used as current denominators.",
            "- OPT-115/132 long-context rates used mixed request vs decode-only",
            "  identities in older reports; this harness does not promise to",
            "  reproduce OPT-132 ratios under the new matched output/capacity boundary.",
            "",
            "## Status",
            "",
            f"historical_ok=`{historical.get('ok')}` preflight_ok=`{preflight.get('ok')}`",
            f"baseline_ok=`{baseline.get('ok')}` gpu_blocked=`{bool(preflight.get('blocked'))}`",
            "",
            "Unknown causal attribution may remain explicitly unknown. Missing",
            "required captures/coverage are blocked, not done.",
            "",
        ]
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    historical = load_sidecar(run_dir, "historical.json") or {}
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    baseline = load_sidecar(run_dir, "baseline.json") or {}
    capture = load_sidecar(run_dir, "capture.json") or {}
    analyze = load_sidecar(run_dir, "analyze.json") or {}
    blocked = bool(preflight.get("blocked")) or not preflight.get("ok")
    historical_ok = bool(historical.get("ok"))
    coverage_ok = bool((analyze.get("coverage_validation") or {}).get("ok"))
    status = (
        "measured"
        if historical_ok and coverage_ok and not blocked
        else ("incomplete" if blocked or not coverage_ok else "historical")
    )
    answers = {
        "matched_decode_ratios": matched_ratio_placeholder(baseline),
        "idle_claim_valid": False,
        "idle_reason": IDLE_INVALID_REASON,
        "opt137_attention_excess": "use_this_harness_complete_attention_not_0.113_or_0.029",
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-136",
        "status": status,
        "mode": mode,
        "measurement_utc": utc_now(),
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "windows": list(WINDOWS),
        "historical": historical,
        "preflight": preflight,
        "baseline": baseline,
        "captures": capture.get("captures") or {},
        "analyze": analyze,
        "answers": answers,
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "family_plan": family_plan(mode, "report"),
        "gpu_phases_blocked": blocked,
    }
    validate_fixture(payload)
    persist_fixture(payload)
    write_report(payload)
    if not (EVIDENCE / "baseline.json").is_file():
        dump_json(
            EVIDENCE / "baseline.json",
            baseline
            or {
                "schema_version": 1,
                "task": "OPT-136",
                "status": "incomplete",
                "blocked": True,
                "reason": "baseline_phase_not_measured",
            },
        )
    if not (EVIDENCE / "coverage.json").is_file():
        dump_json(
            EVIDENCE / "coverage.json",
            {
                "schema_version": 1,
                "task": "OPT-136",
                "status": "unavailable",
                "windows": None,
            },
        )
    if not (EVIDENCE / "family-gaps.json").is_file():
        dump_json(EVIDENCE / "family-gaps.json", {})
    if not (EVIDENCE / "trace-manifest.json").is_file():
        dump_json(
            EVIDENCE / "trace-manifest.json",
            {
                "status": "incomplete",
                "blocked": True,
                "captures": payload.get("captures") or {},
                "commands": [],
            },
        )
    if not (EVIDENCE / "historical-reconciliation.json").is_file() and historical:
        dump_json(EVIDENCE / "historical-reconciliation.json", historical)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "historical":
        return run_historical(run_dir, mode)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "baseline":
        return run_baseline(run_dir, mode)
    if phase == "capture":
        return run_capture(run_dir, mode)
    if phase == "analyze":
        return run_analyze(run_dir, mode)
    if phase == "report":
        return run_report(run_dir, mode)
    raise GraphAccountingError(f"unknown phase {phase}")


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
    if args.phase in {"historical", "report", "analyze"}:
        return 0 if result.get("ok", True) or args.phase == "report" else 1
    if args.phase in {"preflight", "baseline", "capture"} and result.get("blocked"):
        return 0
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphAccountingError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
