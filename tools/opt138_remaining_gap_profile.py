"""OPT-138 remaining short-decode and prefill gap profiler.

Thin orchestrator over the OPT-136 parser/capture harness. Diagnostics only.
Phases: preflight, baseline, capture, families, counters, replay, report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt080_batch_gate import IMAGE, LLAMA_REV, docker_common  # noqa: E402
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools import opt136_graph_accounting as opt136  # noqa: E402
from tools.performance_evidence import (  # noqa: E402
    audit_window_from_tables,
    default_identity,
    missing_counter_record,
    ns_to_ms,
    parent_identity_ok,
    parse_nsys_sqlite,
    prefill_output_policy_ok,
    reconcile_whole_gap,
    regroup_fused_families,
    replay_production_boundary_ok,
    select_top_two_families,
    validate_coverage_document,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)
from tools import opt139_counter_identity as opt139  # noqa: E402

CONTRACT = ROOT / "pins/opt138_remaining_gap_contract.json"
ITERATION = ROOT / "pins/opt138_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt138_remaining_gap_profile.json"
EVIDENCE = ROOT / "evidence/optimization/opt138-remaining-gap"
REPORT = EVIDENCE / "REPORT.md"
REPLAY_BIN = "build/qw38-cuda-component-replay"
MODEL = opt136.MODEL
NATIVE = opt136.NATIVE
LLAMA_BIN = opt136.LLAMA_BIN
GPU_LOCK = opt136.GPU_LOCK
CAPACITY = opt136.CAPACITY
DECODE_TOKENS = opt136.DECODE_TOKENS
WINDOW_TOKENS = opt136.WINDOW_TOKENS
WINDOWS = opt136.WINDOWS
PREFIXES = (128, 2048)
PREFILL_TOKENS = 4096
ENGINES = opt136.ENGINES
ARMS = opt136.ARMS
REPETITIONS = 3
BASELINE_WARMUPS = 3
BASELINE_PAIRS = 10
CHILD_TIMEOUT_S = 300
# prompt-ffn NCU application replay profiles every layer/kernel in the rotating
# OPT-138 protocol; 300s expires around replay pass 3 of ~10.
COUNTER_TIMEOUT_BY_REPLAY: dict[str, int] = {
    "prompt-ffn": 1200,
    "prompt-attention": 1200,
}
AGGREGATE_DEADLINE_S = 7200
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
SELECTOR = "decode_segments8"
PHASES = (
    "preflight",
    "baseline",
    "capture",
    "families",
    "counters",
    "replay",
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
    "prefill_tokens",
    "windows",
    "preflight",
    "baseline",
    "captures",
    "families",
    "counters",
    "replay",
    "answers",
    "report_path",
)
NCU_NEEDLES = (
    "dram__bytes_read",
    "dram__bytes_write",
    "dram__bytes",
    "dram__throughput",
    "lts__t_sectors",
    "lts__t_sector_hit_rate",
    "sm__pipe_tensor",
    "sm__throughput",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_issue_stalled_long_scoreboard",
    "smsp__warps_issue_stalled_barrier",
    "launch__registers_per_thread",
    "launch__local_memory",
    "smsp__sass_lmem_total_bytes",
)
NCU_NEEDLE_ALIASES = {
    "dram__bytes_read": "dram__bytes_op_read",
    "dram__bytes_write": "dram__bytes_op_write",
    "sm__pipe_tensor": "sm__pipe_tensor_cycles_active",
    "dram__throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
}
FAMILY_TO_REPLAY = {
    "q4_ffn_fused": "decode-ffn",
    "q4_down": "decode-ffn",
    "decode_mmv": "decode-ffn",
    "q8_mixer": "decode-mixer",
    "gdn_fused": "decode-gdn",
    "attn_fused": "decode-attention",
    "attn_core": "decode-attention",
    "prompt_mmq": "prompt-ffn",
    "q6_logits": "decode-q6",
    "residual_norm_quant": "decode-mixer",
}


def replay_family_for(phase: str, family: str) -> str | None:
    """Map a ranked family onto the production-boundary replay workload."""
    if family in {"attn_core", "attn_fused"}:
        if phase == "prefill":
            return "prompt-attention"
        if phase == "decode":
            return "decode-attention"
    return FAMILY_TO_REPLAY.get(family)


FAMILY_SOURCES = {
    "q4_ffn_fused": {
        "quartz": "cuda/q4k_decode_path.cuh",
        "llama": "ggml-cuda/mmvq / fused gate-up",
    },
    "q4_down": {
        "quartz": "cuda/q4k_decode_path.cuh",
        "llama": "ggml-cuda q4_K mmvq down",
    },
    "decode_mmv": {
        "quartz": "cuda/q4k_decode_path.cuh",
        "llama": "ggml-cuda mul_mat_vec_q",
    },
    "prompt_mmq": {
        "quartz": "cuda/q4_prompt_mmq / quant_mmq_mma.cuh",
        "llama": "ggml-cuda mul_mat_q",
    },
    "q8_mixer": {
        "quartz": "cuda/q8_decode_path.cuh",
        "llama": "ggml-cuda q8 mixer projections",
    },
    "attn_fused": {
        "quartz": "cuda/attention_decode.cu",
        "llama": "ggml-cuda fattn / vec-dot attention",
    },
    "attn_core": {
        "quartz": "cuda/attention_decode.cu",
        "llama": "ggml-cuda fattn",
    },
    "gdn_fused": {
        "quartz": "cuda/gdn_step.cu",
        "llama": "ggml-cuda ssm/recurrent fused",
    },
    "q6_logits": {
        "quartz": "cuda/q6k_decode_path.cuh",
        "llama": "ggml-cuda q6_K lm_head",
    },
}


class RemainingGapError(RuntimeError):
    """OPT-138 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-138":
        raise RemainingGapError("remaining-gap contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-138", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-138 mode={mode} phase={family} "
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
        "task": "OPT-138",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "llama_revision": LLAMA_REV,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "windows": list(WINDOWS),
        "preflight": None,
        "baseline": None,
        "captures": {},
        "families": None,
        "counters": None,
        "replay": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        payload = load_json(FIXTURE)
        if isinstance(payload, dict) and payload.get("task") == "OPT-138":
            return payload
    return empty_fixture()


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise RemainingGapError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise RemainingGapError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise RemainingGapError("selector drifted from decode_segments8")


def authenticate_opt138_pins() -> dict[str, Any]:
    pins = opt136.authenticate_current_pins()
    identity = parent_identity_ok(
        {
            "execution_graphs": pins.get("execution_graphs"),
            "q4_decode": pins.get("q4_decode"),
            "opt137_dense_mma": pins.get("opt137_dense_mma"),
        }
    )
    pins["parent_identity"] = identity
    pins["ok"] = bool(pins.get("ok")) and bool(identity.get("ok"))
    if not identity.get("ok"):
        pins.setdefault("mismatches", [])
        pins["mismatches"].extend(identity.get("mismatches") or [])
    return pins


def parse_result(text: str) -> dict[str, Any] | None:
    parsed = opt136.parse_prefixed_json(text, opt136.RESULT_PREFIX_OPT138)
    if parsed:
        return parsed
    return opt136.parse_prefixed_json(text, opt136.RESULT_PREFIX)


def _blocked_gpu_phase(
    phase: str, run_dir: Path, mode: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
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
        "family_plan": family_plan(mode, phase if phase in PHASES else "report"),
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


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    available, gpu_blocker = gpu_available()
    residents = opt136.gpu_residents() if available else []
    pins = authenticate_opt138_pins()
    nsys_layout: dict[str, Any] = {}
    nsys_available = False
    try:
        from tools import opt133_decode_nsys_trace as opt133_nsys

        opt133_nsys.ensure_nsys_layout()
        nsys_layout = opt133_nsys.docker_nsys_layout()
        nsys_available = bool(
            nsys_layout.get("nsys_bin") or Path(opt133_nsys.NSYS_BIN).is_file()
        )
    except Exception as exc:  # noqa: BLE001
        nsys_layout = {"error": str(exc)}
    make = (
        subprocess.run(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt138-diagnostics"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT_S,
        )
        if available
        else subprocess.CompletedProcess(["make"], 1, "", gpu_blocker or "")
    )
    native_path = ROOT / NATIVE
    native_hash = opt136.sha256_file(native_path)
    llama_script = ROOT / "tools/llama_authority/build_opt136_profile.sh"
    llama: dict[str, Any] = {"ok": False, "reason": "not_started"}
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
            "hash": opt136.sha256_file(ROOT / LLAMA_BIN),
        }
    elif not available:
        llama = {"ok": False, "reason": gpu_blocker or "gpu_unavailable"}
    identity: dict[str, Any] = {}
    if available and native_path.is_file():
        completed = opt136.with_gpu_lock(
            opt136.native_command(
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
        parsed = parse_result(completed.stdout or "")
        identity = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "parsed": parsed,
            "opt137_dense_mma": (parsed or {}).get("opt137_dense_mma"),
            "stdout_tail": (completed.stdout or "")[-2000:],
        }
    blocked: list[str] = []
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
    if not pins.get("ok"):
        blocked.append("stale_parent_identity")
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
        "task": "OPT-138",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "gpu_residents": residents,
        "exclusive_sitting": len(residents) <= 1,
        "nsys_available": nsys_available,
        "nsys_layout": nsys_layout,
        "image": IMAGE,
        "native": NATIVE,
        "replay_bin": REPLAY_BIN,
        "llama_bin": LLAMA_BIN,
        "hashes": {
            "quartz_native": native_hash,
            "llama_profile": llama.get("hash"),
            "replay": opt136.sha256_file(ROOT / REPLAY_BIN),
        },
        "make_returncode": make.returncode,
        "make_stderr_tail": (make.stderr or "")[-2000:],
        "llama_build": llama,
        "identity": identity,
        "authenticated_current_pins": pins,
        "parent": PARENT,
        "capacity": CAPACITY,
        "prefixes": list(PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "eval_tokens": DECODE_TOKENS,
        "token_generator": "(42 + index * 997) % 248320",
        "gpu_lock": relpath(GPU_LOCK),
        "claims_throughput": False,
        "blocked": blocked,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def _engine_args(engine: str, *, kind: str, prefix: int | None = None) -> list[str]:
    if kind == "prefill":
        workload_map = {
            "unprofiled": "prefill-unprofiled",
            "graph": "prefill-graph-capture",
            "node": "prefill-node-capture",
        }
        if engine == "quartz":
            return opt136.quartz_prefill_args(workload=workload_map["unprofiled"])
        return opt136.llama_prefill_args(workload=workload_map["unprofiled"])
    assert prefix is not None
    if engine == "quartz":
        return opt136.quartz_args(
            workload="unprofiled", prefix=prefix, attribution=False
        )
    return opt136.llama_args(workload="unprofiled", prefix=prefix)


def _capture_args(engine: str, *, kind: str, arm: str, prefix: int | None) -> list[str]:
    if kind == "prefill":
        names = {
            "unprofiled": "prefill-unprofiled",
            "graph": "prefill-graph-capture",
            "node": "prefill-node-capture",
        }
        workload = names[arm]
        if engine == "quartz":
            return opt136.quartz_prefill_args(workload=workload)
        return opt136.llama_prefill_args(workload=workload)
    names = {
        "unprofiled": "unprofiled",
        "graph": "graph-capture",
        "node": "node-capture",
    }
    assert prefix is not None
    if engine == "quartz":
        return opt136.quartz_args(workload=names[arm], prefix=prefix, attribution=False)
    return opt136.llama_args(workload=names[arm], prefix=prefix)


def run_baseline(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("baseline", run_dir, mode, preflight)
    results: dict[str, Any] = {}
    workloads: list[tuple[str, str, int | None]] = [
        ("d128", "decode", 128),
        ("d2048", "decode", 2048),
        ("p4096", "prefill", None),
    ]
    for key, kind, prefix in workloads:
        quartz_ms: list[float] = []
        llama_ms: list[float] = []
        for warmup in range(BASELINE_WARMUPS):
            for engine in ENGINES:
                args = _engine_args(engine, kind=kind, prefix=prefix)
                label = f"warmup-{key}-{engine}-{warmup}"
                completed = opt136.with_gpu_lock(
                    opt136.native_command(args, tier="acceptance"),
                    timeout_s=CHILD_TIMEOUT_S,
                )
                results[label] = {
                    "ok": completed.returncode == 0,
                    "returncode": completed.returncode,
                }
                sys.stderr.write(
                    f"OPT-138 baseline {label} rc={completed.returncode}\n"
                )
        for sample in range(BASELINE_PAIRS):
            for engine, bucket in (("quartz", quartz_ms), ("llama", llama_ms)):
                args = _engine_args(engine, kind=kind, prefix=prefix)
                label = f"pair-{key}-{engine}-{sample}"
                completed = opt136.with_gpu_lock(
                    opt136.native_command(args, tier="acceptance"),
                    timeout_s=CHILD_TIMEOUT_S,
                )
                parsed = parse_result(completed.stdout or "")
                metric = "prefill_ms" if kind == "prefill" else "decode_only_ms"
                wall = None if parsed is None else parsed.get(metric)
                if wall is not None:
                    bucket.append(float(wall))
                policy = None
                if kind == "prefill" and parsed is not None:
                    policy = prefill_output_policy_ok(parsed)
                results[label] = {
                    "ok": completed.returncode == 0,
                    metric: wall,
                    "returncode": completed.returncode,
                    "output_policy": policy,
                }
                sys.stderr.write(
                    f"OPT-138 baseline {label} rc={completed.returncode} ms={wall}\n"
                )
        results[key] = {
            "kind": kind,
            "metric": "prefill_ms" if kind == "prefill" else "decode_only_ms",
            "quartz_ms": quartz_ms,
            "llama_ms": llama_ms,
            "n_pairs": min(len(quartz_ms), len(llama_ms)),
        }
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
        "phase": "baseline",
        "mode": mode,
        "ok": True,
        "results": results,
        "hashes": preflight.get("hashes"),
        "claims_throughput": False,
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
    jobs: list[tuple[str, str, str, int | None, str]] = []
    for prefix in PREFIXES:
        for engine in ENGINES:
            for arm in ARMS:
                jobs.append(
                    (f"{engine}-d{prefix}-{arm}", engine, "decode", prefix, arm)
                )
    for engine in ENGINES:
        for arm in ARMS:
            jobs.append((f"{engine}-p4096-{arm}", engine, "prefill", None, arm))
    for stem_key, engine, kind, prefix, arm in jobs:
        for rep in range(REPETITIONS):
            key = f"{stem_key}-r{rep}"
            stem = traces / key
            args = _capture_args(engine, kind=kind, arm=arm, prefix=prefix)
            nvtx = "opt138.prefill" if kind == "prefill" else "opt136.window"
            if arm == "unprofiled":
                command = list(args)
            else:
                command = opt136.nsys_profile_command(
                    stem,
                    args,
                    graph_trace="graph" if arm == "graph" else "node",
                    nvtx_capture=nvtx,
                )
            commands.append(" ".join(shlex.quote(part) for part in command))
            completed = opt136.with_gpu_lock(
                opt136.native_command(command, tier="acceptance"),
                timeout_s=CHILD_TIMEOUT_S,
            )
            windows = opt136.PREFILL_WINDOWS if kind == "prefill" else None
            exported = {}
            if arm != "unprofiled":
                exported = opt136.export_capture_traces(stem, windows=windows)
            parsed = parse_result(completed.stdout or "")
            captures[key] = {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "engine": engine,
                "kind": kind,
                "arm": arm,
                "prefix": prefix if prefix is not None else PREFILL_TOKENS,
                "rep": rep,
                "command": command,
                "sqlite_by_window": exported,
                "parsed": parsed,
                "stdout_tail": (completed.stdout or "")[-1200:],
                "stderr_tail": (completed.stderr or "")[-800:],
            }
            sys.stderr.write(f"OPT-138 capture {key} rc={completed.returncode}\n")
            sys.stderr.flush()
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
        "phase": "capture",
        "mode": mode,
        "ok": any(row.get("ok") for row in captures.values()),
        "captures": captures,
        "nsys_commands": commands,
        "hashes": preflight.get("hashes"),
        "family_plan": family_plan(mode, "capture"),
        "measured_at": utc_now(),
    }
    dump_json(
        EVIDENCE / "trace-manifest.json",
        {"captures": captures, "commands": commands},
    )
    return store_sidecar(run_dir, "capture.json", payload)


def _family_ms(coverage: Mapping[str, Any]) -> dict[str, float]:
    families = coverage.get("families") or {}
    out: dict[str, float] = {}
    if isinstance(families, Mapping):
        for name, row in families.items():
            if isinstance(row, Mapping):
                union_ns = row.get("union_ns")
                if union_ns is not None:
                    out[str(name)] = float(ns_to_ms(int(union_ns)))
    return out


def _largest_kernel(coverage: Mapping[str, Any], family: str) -> str | None:
    # coverage families do not carry kernel names; keep explicit unknown.
    families = coverage.get("families") or {}
    row = families.get(family) if isinstance(families, Mapping) else None
    if isinstance(row, Mapping):
        ids = row.get("kernel_ids") or row.get("names")
        if isinstance(ids, list) and ids:
            return str(ids[0])
    return None


def run_families(run_dir: Path, mode: str) -> dict[str, Any]:
    capture = load_sidecar(run_dir, "capture.json") or {}
    captures = capture.get("captures") or {}
    windows: list[dict[str, Any]] = []
    decode_rows: dict[str, dict[str, Any]] = {}
    prefill_rows: dict[str, dict[str, Any]] = {}
    family_gaps: dict[str, Any] = {}
    traces_dir = run_dir / "traces"

    def ensure_export(key: str, row: dict[str, Any]) -> dict[str, str]:
        exported = dict(row.get("sqlite_by_window") or {})
        if exported:
            return exported
        kind = str(row.get("kind") or "decode")
        stem = traces_dir / key
        windows_arg = opt136.PREFILL_WINDOWS if kind == "prefill" else None
        exported = opt136.export_capture_traces(stem, windows=windows_arg)
        if exported:
            row["sqlite_by_window"] = exported
            captures[key] = row
        return exported

    node_keys = [key for key, row in captures.items() if row.get("arm") == "node"]
    for key in node_keys:
        row = captures[key]
        exported = ensure_export(key, row)
        engine = str(row.get("engine") or key.split("-", 1)[0])
        kind = str(row.get("kind") or "decode")
        prefix = int(row.get("prefix") or 128)
        for window, sqlite_rel in exported.items():
            path = ROOT / sqlite_rel
            if not path.is_file():
                continue
            tables = parse_nsys_sqlite(path)
            metric = "prefill" if kind == "prefill" else "decode_only"
            eval_count = PREFILL_TOKENS if kind == "prefill" else WINDOW_TOKENS
            arm = "node"
            if not tables.nodes and tables.kernels:
                arm = "leaves" if not tables.graphs else "graph"
            coverage = audit_window_from_tables(
                tables,
                identity=default_identity(
                    capture_arm=arm,
                    engine=engine,
                    prefix=prefix if kind == "decode" else PREFILL_TOKENS,
                    populated_length=prefix if kind == "decode" else PREFILL_TOKENS,
                    eval_count=eval_count,
                    metric=metric,
                    output_boundary="eval_logits_no_host_materialize",
                ),
            )
            coverage["capture_key"] = key
            coverage["window_name"] = window
            coverage["kind"] = kind
            coverage["requested_arm"] = "node"
            if arm != "node":
                coverage["node_identities_absent_used_kernel_leaves"] = True
            windows.append(coverage)
            family_ms = _family_ms(coverage)
            bucket_key = f"{engine}-{kind}-d{prefix}-{window}-r{row.get('rep')}"
            family_gaps[bucket_key] = family_ms
            target = prefill_rows if kind == "prefill" else decode_rows
            for family, value in family_ms.items():
                entry = target.setdefault(
                    family,
                    {
                        "family": family,
                        "d128_middle_deltas_ms": [],
                        "d2048_middle_deltas_ms": [],
                        "d128_early_deltas_ms": [],
                        "d128_late_deltas_ms": [],
                        "d2048_early_deltas_ms": [],
                        "d2048_late_deltas_ms": [],
                        "p4096_deltas_ms": [],
                        "quartz": {},
                        "llama": {},
                    },
                )
                engine_map = entry["quartz"] if engine == "quartz" else entry["llama"]
                loc = f"{prefix}-{window}-r{row.get('rep')}"
                engine_map[loc] = value

    def paired_deltas(
        entry: Mapping[str, Any], prefix: int, window: str
    ) -> list[float]:
        deltas: list[float] = []
        for rep in range(REPETITIONS):
            loc = f"{prefix}-{window}-r{rep}"
            q = (entry.get("quartz") or {}).get(loc)
            ell = (entry.get("llama") or {}).get(loc)
            if q is None or ell is None:
                continue
            deltas.append(float(q) - float(ell))
        return deltas

    decode_input: list[dict[str, Any]] = []
    for family, entry in decode_rows.items():
        decode_input.append(
            {
                "family": family,
                "d128_middle_deltas_ms": paired_deltas(entry, 128, "middle"),
                "d2048_middle_deltas_ms": paired_deltas(entry, 2048, "middle"),
                "d128_early_deltas_ms": paired_deltas(entry, 128, "early"),
                "d128_late_deltas_ms": paired_deltas(entry, 128, "late"),
                "d2048_early_deltas_ms": paired_deltas(entry, 2048, "early"),
                "d2048_late_deltas_ms": paired_deltas(entry, 2048, "late"),
            }
        )
    prefill_input: list[dict[str, Any]] = []
    for family, entry in prefill_rows.items():
        prefill_input.append(
            {
                "family": family,
                "p4096_deltas_ms": paired_deltas(entry, PREFILL_TOKENS, "prefill"),
            }
        )

    # Also regroup a representative pair for the report table.
    def representative(kind: str, prefix: int, window: str) -> dict[str, Any]:
        q_key = f"quartz-{'p4096' if kind == 'prefill' else 'd' + str(prefix)}-node-r0"
        l_key = f"llama-{'p4096' if kind == 'prefill' else 'd' + str(prefix)}-node-r0"
        q_cov = next(
            (
                item
                for item in windows
                if item.get("capture_key") == q_key
                and item.get("window_name") == window
            ),
            None,
        )
        l_cov = next(
            (
                item
                for item in windows
                if item.get("capture_key") == l_key
                and item.get("window_name") == window
            ),
            None,
        )
        if not q_cov or not l_cov:
            return {"status": "incomplete", "reason": "paired_node_capture_missing"}
        grouped = regroup_fused_families(_family_ms(q_cov), _family_ms(l_cov))
        q_wall = q_cov.get("window_ms")
        l_wall = l_cov.get("window_ms")
        recon = None
        if q_wall is not None and l_wall is not None:
            recon = reconcile_whole_gap(
                quartz_wall_ms=float(q_wall),
                llama_wall_ms=float(l_wall),
                matched_disjoint_excess_ms=float(
                    grouped.get("matched_disjoint_excess_ms") or 0.0
                ),
                unmatched_work_gap_ms=float(
                    grouped.get("unmatched_work_gap_ms") or 0.0
                ),
                scheduling_host_residual_ms=0.0,
            )
            recon["metric_identity"] = (
                "prefill_complete" if kind == "prefill" else "decode_window"
            )
        return {
            "regrouped": grouped,
            "reconciliation": recon,
            "quartz_window_ms": q_wall,
            "llama_window_ms": l_wall,
        }

    decode_selection = select_top_two_families(decode_input, phase="decode")
    prefill_selection = select_top_two_families(prefill_input, phase="prefill")
    if windows:
        document: dict[str, Any] = {
            "schema_version": 1,
            "task": "OPT-138",
            "status": "valid"
            if all(row.get("coverage_valid") for row in windows)
            else "incomplete",
            "windows": windows,
        }
    else:
        document = {
            "schema_version": 1,
            "task": "OPT-138",
            "status": "unavailable",
            "windows": None,
            "reason": "GPU captures unavailable",
        }
    validation = validate_coverage_document(document)
    document["validation"] = validation
    dump_json(EVIDENCE / "coverage.json", document)
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
        "phase": "families",
        "mode": mode,
        "ok": bool(validation.get("ok")) or bool(decode_input or prefill_input),
        "coverage_validation": validation,
        "window_count": len(windows),
        "decode_selection": decode_selection,
        "prefill_selection": prefill_selection,
        "decode_d128_middle": representative("decode", 128, "middle"),
        "decode_d2048_middle": representative("decode", 2048, "middle"),
        "prefill_p4096": representative("prefill", PREFILL_TOKENS, "prefill"),
        "family_gaps": family_gaps,
        "family_plan": family_plan(mode, "families"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "family-gaps.json", payload)
    dump_json(
        sidecar(run_dir, "capture.json"),
        {**capture, "captures": captures},
    )
    return store_sidecar(run_dir, "families.json", payload)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _parse_ncu_metric_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('"') and '","' in line:
            name = line[1 : line.index('","')]
            if name.lower() != "metric name":
                names.append(name)
            continue
        if not line or line.startswith("==") or line.startswith("Device "):
            continue
        if line.startswith("Metric Name") or set(line) <= {"-", " "}:
            continue
        token = line.split(maxsplit=1)[0] if line.split() else ""
        if "__" in token:
            names.append(token)
    return names


def _select_ncu_metrics(names: Sequence[str]) -> list[str]:
    available = set(names)
    selected: list[str] = []
    for needle in NCU_NEEDLES:
        candidate = NCU_NEEDLE_ALIASES.get(needle, needle)
        if candidate in available:
            selected.append(candidate)
            continue
        if needle in available:
            if candidate != needle:
                selected.append(candidate)
            else:
                selected.append(needle)
            continue
        base = candidate.split(".", 1)[0]
        if base in available and candidate not in selected:
            selected.append(candidate)
    return selected


def _docker_ncu(*extra: str) -> list[str]:
    base = docker_common(IMAGE, "acceptance")
    return [*base[:-1], "-e", "HOME=/tmp", base[-1], *extra]


def counter_timeout_s(replay_family: str) -> int:
    return COUNTER_TIMEOUT_BY_REPLAY.get(replay_family, CHILD_TIMEOUT_S)


def _query_ncu_metrics() -> dict[str, Any]:
    command = [
        *_docker_ncu(),
        "bash",
        "-lc",
        "command -v ncu && ncu --query-metrics --csv",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CHILD_TIMEOUT_S,
    )
    text = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0 or "ncu" not in text:
        host = subprocess.run(
            ["bash", "-lc", "HOME=/tmp command -v ncu && ncu --query-metrics --csv"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHILD_TIMEOUT_S,
        )
        text = (host.stdout or "") + (host.stderr or "")
        completed = host
    names = _parse_ncu_metric_names(text)
    selected = _select_ncu_metrics(names)
    available = completed.returncode == 0 and bool(selected or "ncu" in text)
    permission = "ERR_NVGPUCTRPERM" in text
    error = None
    if permission:
        error = "ERR_NVGPUCTRPERM"
    elif not available:
        error = "ncu_unavailable"
    return {
        "ncu_available": available and not permission,
        "permission_denied": permission,
        "queried": True,
        "full_ncu_sweep": False,
        "selected_metrics": selected[:48],
        "raw_name_count": len(names),
        "error": error,
    }


def _selected_families(families: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for phase, key in (
        ("decode", "decode_selection"),
        ("prefill", "prefill_selection"),
    ):
        selected = ((families.get(key) or {}).get("selected") or [])[:2]
        for row in selected:
            name = str(row.get("family") or "")
            if name:
                out.append((phase, name))
    return out


def run_counters(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    families = load_sidecar(run_dir, "families.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("counters", run_dir, mode, preflight)
    ncu = _query_ncu_metrics()
    selected = _selected_families(families)
    kernels: list[dict[str, Any]] = []
    for phase, family in selected:
        for engine in ENGINES:
            kernel_id = f"{engine}:{family}"
            if not ncu.get("ncu_available"):
                kernels.append(
                    missing_counter_record(
                        kernel=kernel_id,
                        engine=engine,
                        error=str(ncu.get("error") or "ncu_unavailable"),
                    )
                )
                continue
            replay_family = replay_family_for(phase, family)
            if replay_family is None:
                kernels.append(
                    missing_counter_record(
                        kernel=kernel_id,
                        engine=engine,
                        error="no_replay_mapping_for_family",
                    )
                )
                continue
            if phase == "prefill" and replay_family.startswith("decode-"):
                kernels.append(
                    {
                        **missing_counter_record(
                            kernel=kernel_id,
                            engine=engine,
                            error=opt139.OPT140_INELIGIBLE,
                        ),
                        "family": family,
                        "phase": phase,
                        "replay_family": replay_family,
                        "supported_mechanism": None,
                        "candidate": None,
                    }
                )
                continue
            metrics = ",".join(ncu.get("selected_metrics") or [])
            if engine == "llama":
                kernels.append(
                    missing_counter_record(
                        kernel=kernel_id,
                        engine=engine,
                        error="llama_ncu_kernel_identity_unresolved",
                    )
                )
                continue
            # Collection is separately timed and is not a throughput sample.
            if not metrics:
                kernels.append(
                    missing_counter_record(
                        kernel=kernel_id,
                        engine=engine,
                        error="no_supported_metrics",
                    )
                )
                continue
            expected_kernel = None
            if family == "attn_core" and phase == "decode":
                expected_kernel = opt139.launch_for_path(
                    opt139.decode_attention_vec128_path_for_position(128)
                )
            if family == "attn_core" and phase == "prefill":
                expected_kernel = opt139.LAUNCH_PREFILL_ATTN
            protocol_flag = (
                "--opt140-protocol"
                if replay_family == "prompt-attention"
                else "--opt138-protocol"
            )
            collect = [
                *_docker_ncu(),
                "ncu",
                "--csv",
                "--metrics",
                metrics,
                "--target-processes",
                "all",
                "--replay-mode",
                "application",
            ]
            if expected_kernel:
                collect.extend(
                    [
                        "--kernel-name",
                        f"regex:{opt139.ncu_kernel_regex(expected_kernel)}",
                        "--launch-count",
                        "1",
                    ]
                )
            collect.extend(
                [
                    f"./{REPLAY_BIN}",
                    MODEL,
                    "--workload",
                    replay_family,
                    "--cache-mode",
                    "rotating",
                    protocol_flag,
                    "--warmups",
                    "0",
                    "--samples",
                    "1",
                ]
            )
            timeout_s = counter_timeout_s(replay_family)
            raw_dir = EVIDENCE / "ncu-raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{engine}-{phase}-{family}.ncu.txt"
            completed = opt136.with_gpu_lock(collect, timeout_s=timeout_s)
            blob = (completed.stdout or "") + (completed.stderr or "")
            raw_path.write_text(blob, encoding="utf-8")
            timed_out = "timeout after" in blob
            if completed.returncode != 0 or "ERR_NVGPUCTRPERM" in blob or timed_out:
                kernels.append(
                    {
                        **missing_counter_record(
                            kernel=kernel_id,
                            engine=engine,
                            error=blob[-800:] or f"ncu_rc={completed.returncode}",
                        ),
                        "command": collect,
                        "replay_mode": "application",
                        "selected_metrics": ncu.get("selected_metrics"),
                        "timeout_s": timeout_s,
                        "raw_artifact": relpath(raw_path),
                    }
                )
                continue
            kernels.append(
                opt139.counter_record_from_ncu_blob(
                    blob,
                    kernel_id=kernel_id,
                    engine=engine,
                    family=family,
                    phase=phase,
                    replay_family=replay_family,
                    expected_kernel=expected_kernel,
                    workload=phase,
                    selected_metrics=ncu.get("selected_metrics") or [],
                    command=collect,
                    timeout_s=timeout_s,
                    raw_artifact=relpath(raw_path),
                )
            )
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
        "phase": "counters",
        "mode": mode,
        "ok": True,
        "ncu": ncu,
        "selected_families": selected,
        "kernels": kernels,
        "full_ncu_sweep": False,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "counters"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "counter-evidence.json", payload)
    return store_sidecar(run_dir, "counters.json", payload)


def run_replay(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    families = load_sidecar(run_dir, "families.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("replay", run_dir, mode, preflight)
    selected = _selected_families(families)
    rounds: list[dict[str, Any]] = []
    for phase, family in selected:
        replay_family = replay_family_for(phase, family)
        record = {
            "phase": phase,
            "family": family,
            "replay_family": replay_family or "unmapped",
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "warmups": 1,
            "samples": 3,
        }
        boundary = replay_production_boundary_ok(record)
        record["boundary"] = boundary
        if replay_family is None or not boundary.get("ok"):
            record["ok"] = False
            record["reason"] = boundary.get("reason") or "unmapped_family"
            rounds.append(record)
            continue
        protocol_flag = (
            "--opt140-protocol"
            if replay_family == "prompt-attention"
            else "--opt138-protocol"
        )
        command = [
            f"./{REPLAY_BIN}",
            MODEL,
            "--workload",
            replay_family,
            "--cache-mode",
            "rotating",
            protocol_flag,
            "--warmups",
            "1",
            "--samples",
            "3",
            "--evidence-dir",
            relpath(EVIDENCE / "replay"),
        ]
        completed = opt136.with_gpu_lock(
            opt136.native_command(command, tier="acceptance"),
            timeout_s=counter_timeout_s(replay_family),
        )
        prefix = (
            "QW38_OPT140_PREFILL_ATTENTION_REPLAY_RESULT="
            if replay_family == "prompt-attention"
            else "QW38_OPT138_REPLAY_RESULT="
        )
        parsed = opt136.parse_prefixed_json(completed.stdout or "", prefix)
        record.update(
            {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "command": command,
                "parsed": parsed,
                "stdout_tail": (completed.stdout or "")[-1500:],
                "stderr_tail": (completed.stderr or "")[-800:],
            }
        )
        rounds.append(record)
        sys.stderr.write(f"OPT-138 replay {family} rc={completed.returncode}\n")
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
        "phase": "replay",
        "mode": mode,
        "ok": True,
        "rounds": rounds,
        "protocol": {
            "warmups": 1,
            "samples": 3,
            "cache_mode": "rotating",
            "accept_hot_as_production": False,
        },
        "claims_throughput": False,
        "family_plan": family_plan(mode, "replay"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "replay.json", payload)


def _experiment_entry(
    phase: str,
    selection: Mapping[str, Any],
    counters: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    selected = list(selection.get("selected") or [])
    top = selected[0] if selected else None
    if top is None:
        return {
            "phase": phase,
            "selected_family": None,
            "measured_excess_ms": None,
            "uncertainty": None,
            "candidate": None,
            "supported_mechanism": None,
            "resolving_measurement": (
                "need matched node-level family excess with a positive "
                "one-sided 95% lower bound (df=2)"
            ),
            "evidence_links": [
                "evidence/optimization/opt138-remaining-gap/family-gaps.json",
                "evidence/optimization/opt138-remaining-gap/coverage.json",
            ],
        }
    family = str(top.get("family"))
    stats = top.get("stats") or top.get("d2048_middle") or top.get("d128_middle") or {}
    kernel_rows = [
        row for row in (counters.get("kernels") or []) if row.get("family") == family
    ]
    supported = None
    for row in kernel_rows:
        if row.get("error"):
            continue
        admission = opt139.admit_supported_mechanism(
            {
                "phase": phase,
                "engine": row.get("engine") or "quartz",
                "workload": row.get("workload") or phase,
                "replay_family": row.get("replay_family")
                or replay_family_for(phase, family)
                or "",
                "kernel": row.get("target_kernel") or row.get("kernel"),
                "expected_kernel": row.get("expected_kernel"),
                "identity": row.get("identity"),
                "replay_boundary": row.get("replay_boundary")
                or (replay.get("rounds") and None),
                "dram_throughput": row.get("dram_throughput"),
                "sm_throughput": row.get("sm_throughput"),
                "dram_read_bytes": row.get("dram_read_bytes"),
                "dram_write_bytes": row.get("dram_write_bytes"),
                "l2_traffic": row.get("l2_traffic"),
                "tensor_activity": row.get("tensor_activity"),
                "achieved_occupancy": row.get("achieved_occupancy"),
                "stalls": row.get("stalls"),
                "registers": row.get("registers"),
                "local_memory_spills": row.get("local_memory_spills"),
                "source_observations": row.get("source_observations"),
                "sass_observations": row.get("sass_observations"),
                "candidate": row.get("candidate"),
            }
        )
        if admission.get("ok") and admission.get("supported_mechanism"):
            supported = admission.get("supported_mechanism")
            break
    replay_row = next(
        (
            row
            for row in (replay.get("rounds") or [])
            if row.get("family") == family and row.get("phase") == phase
        ),
        None,
    )
    sources = FAMILY_SOURCES.get(family, {"quartz": None, "llama": None})
    if supported is None:
        return {
            "phase": phase,
            "selected_family": family,
            "measured_excess_ms": stats.get("mean_ms") or top.get("score_ms"),
            "uncertainty": {
                "one_sided_low": stats.get("one_sided_low"),
                "df": 2,
                "n": stats.get("n"),
            },
            "quartz_source_function": sources.get("quartz"),
            "llama_source_function": sources.get("llama"),
            "supported_mechanism": None,
            "affected_files": [sources.get("quartz")] if sources.get("quartz") else [],
            "candidate": None,
            "target_workloads": ["D128", "D2048"] if phase == "decode" else ["P4096"],
            "guards_under_opt135": {
                "policy": "target_guard_v2",
                "decode_guards": ["D128", "D2048", "P4096"]
                if phase == "decode"
                else ["D128", "D2048"],
            },
            "expected_benefit": {
                "kind": "labeled_estimate",
                "value": None,
                "label": "unknown_until_mechanism",
            },
            "disconfirming_experiment": (
                "repeat node-level family accounting after any candidate; "
                "a non-positive lower bound rejects the family as the priority"
            ),
            "quality_state_gates": "N/A diagnostics; any later keep requires OPT-058 NLL",
            "resolving_measurement": (
                "targeted Nsight Compute metrics on the enclosing fused kernel "
                "plus rotating real-input replay; occupancy-only is insufficient"
            ),
            "replay": None
            if replay_row is None
            else {"ok": replay_row.get("ok"), "reason": replay_row.get("reason")},
            "evidence_links": [
                "evidence/optimization/opt138-remaining-gap/family-gaps.json",
                "evidence/optimization/opt138-remaining-gap/counter-evidence.json",
                "evidence/optimization/opt138-remaining-gap/coverage.json",
            ],
        }
    return {
        "phase": phase,
        "selected_family": family,
        "measured_excess_ms": stats.get("mean_ms") or top.get("score_ms"),
        "uncertainty": {
            "one_sided_low": stats.get("one_sided_low"),
            "df": 2,
            "n": stats.get("n"),
        },
        "quartz_source_function": sources.get("quartz"),
        "llama_source_function": sources.get("llama"),
        "supported_mechanism": supported,
        "affected_files": [sources.get("quartz")] if sources.get("quartz") else [],
        "candidate": {
            "design": (
                f"one bounded experiment on {family} at the llama fused boundary"
            ),
            "not_an_automatic_ladder": True,
        },
        "target_workloads": ["D128", "D2048"] if phase == "decode" else ["P4096"],
        "guards_under_opt135": {
            "policy": "target_guard_v2",
            "target_l_exclusive_min": 1.0,
            "guard_l_min": 0.98,
        },
        "expected_benefit": {
            "kind": "labeled_estimate",
            "value": stats.get("mean_ms") or top.get("score_ms"),
            "label": "family_excess_not_end_to_end_tok_s",
        },
        "disconfirming_experiment": (
            "paired node-level excess lower bound <= 0 after the candidate"
        ),
        "quality_state_gates": "N/A diagnostics; later keep requires OPT-058 NLL",
        "evidence_links": [
            "evidence/optimization/opt138-remaining-gap/family-gaps.json",
            "evidence/optimization/opt138-remaining-gap/counter-evidence.json",
        ],
    }


def write_report(payload: Mapping[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    baseline = payload.get("baseline") or {}
    families = payload.get("families") or {}
    answers = payload.get("answers") or {}
    lines = [
        "# OPT-138 — Remaining short-decode and prefill excess versus llama",
        "",
        "Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`. Image `{IMAGE}`.",
        f"Parent `{PARENT}`. GPU available `{preflight.get('gpu_available')}`.",
        "",
        "## Identity",
        "",
        f"pins ok=`{((preflight.get('authenticated_current_pins') or {}).get('ok'))}`",
        f"opt137 MMA=`{((preflight.get('authenticated_current_pins') or {}).get('opt137_dense_mma'))}`",
        "",
        "## Baseline Quartz vs llama",
        "",
    ]
    results = baseline.get("results") or {}
    for key in ("d128", "d2048", "p4096"):
        block = results.get(key) or {}
        q = _mean(block.get("quartz_ms") or [])
        ell = _mean(block.get("llama_ms") or [])
        metric = block.get("metric")
        tok_q = tok_l = None
        if key != "p4096" and q and ell:
            tok_q = DECODE_TOKENS * 1000.0 / q
            tok_l = DECODE_TOKENS * 1000.0 / ell
        elif key == "p4096" and q and ell:
            tok_q = PREFILL_TOKENS * 1000.0 / q
            tok_l = PREFILL_TOKENS * 1000.0 / ell
        lines.append(
            f"- {key} metric=`{metric}` quartz_ms=`{q}` llama_ms=`{ell}` "
            f"quartz_tok_s=`{tok_q}` llama_tok_s=`{tok_l}`"
        )
    lines.extend(
        [
            "",
            "## Family ranking",
            "",
            f"decode selected=`{(families.get('decode_selection') or {}).get('selected')}`",
            f"prefill selected=`{(families.get('prefill_selection') or {}).get('selected')}`",
            f"coverage ok=`{(families.get('coverage_validation') or {}).get('ok')}`",
            "",
            "## Next experiments",
            "",
            f"decode=`{(answers.get('next_experiments') or {}).get('decode', {}).get('selected_family')}`",
            f"prefill=`{(answers.get('next_experiments') or {}).get('prefill', {}).get('selected_family')}`",
            "",
            "## Status",
            "",
            f"status=`{payload.get('status')}` blocked=`{payload.get('gpu_phases_blocked')}`",
            "",
            "Unsupported counters leave causal explanation explicitly unknown.",
            "Missing trace coverage is blocked. Shipping throughput delta is N/A.",
            "",
        ]
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    baseline = load_sidecar(run_dir, "baseline.json") or {}
    capture = load_sidecar(run_dir, "capture.json") or {}
    families = load_sidecar(run_dir, "families.json") or {}
    counters = load_sidecar(run_dir, "counters.json") or {}
    replay = load_sidecar(run_dir, "replay.json") or {}
    blocked = bool(preflight.get("blocked")) or not preflight.get("ok")
    coverage_ok = bool((families.get("coverage_validation") or {}).get("ok"))
    status = (
        "measured"
        if coverage_ok and not blocked
        else ("incomplete" if blocked or not coverage_ok else "structure")
    )
    next_experiments = {
        "decode": _experiment_entry(
            "decode",
            families.get("decode_selection") or {},
            counters,
            replay,
        ),
        "prefill": _experiment_entry(
            "prefill",
            families.get("prefill_selection") or {},
            counters,
            replay,
        ),
    }
    dump_json(EVIDENCE / "next-experiments.json", next_experiments)
    if not (EVIDENCE / "counter-evidence.json").is_file():
        dump_json(
            EVIDENCE / "counter-evidence.json",
            counters
            or {
                "schema_version": 1,
                "task": "OPT-138",
                "status": "incomplete",
                "kernels": [],
            },
        )
    if not (EVIDENCE / "coverage.json").is_file():
        dump_json(
            EVIDENCE / "coverage.json",
            {
                "schema_version": 1,
                "task": "OPT-138",
                "status": "unavailable",
                "windows": None,
            },
        )
    if not (EVIDENCE / "baseline.json").is_file():
        dump_json(
            EVIDENCE / "baseline.json",
            baseline
            or {
                "schema_version": 1,
                "task": "OPT-138",
                "status": "incomplete",
            },
        )
    if not (EVIDENCE / "family-gaps.json").is_file():
        dump_json(EVIDENCE / "family-gaps.json", families or {})
    if not (EVIDENCE / "trace-manifest.json").is_file():
        dump_json(
            EVIDENCE / "trace-manifest.json",
            {
                "status": "incomplete",
                "captures": capture.get("captures") or {},
                "commands": [],
            },
        )
    answers = {
        "next_experiments": next_experiments,
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "matched_rates": {},
    }
    results = baseline.get("results") or {}
    for key, tokens in (
        ("d128", DECODE_TOKENS),
        ("d2048", DECODE_TOKENS),
        ("p4096", PREFILL_TOKENS),
    ):
        block = results.get(key) or {}
        q = _mean(block.get("quartz_ms") or [])
        ell = _mean(block.get("llama_ms") or [])
        answers["matched_rates"][key] = {
            "quartz_ms": q,
            "llama_ms": ell,
            "quartz_tok_s": (tokens * 1000.0 / q) if q else None,
            "llama_tok_s": (tokens * 1000.0 / ell) if ell else None,
            "metric": block.get("metric"),
        }
    payload = {
        "schema_version": 1,
        "task": "OPT-138",
        "status": status,
        "mode": mode,
        "measurement_utc": utc_now(),
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "windows": list(WINDOWS),
        "preflight": preflight,
        "baseline": baseline,
        "captures": capture.get("captures") or {},
        "families": families,
        "counters": counters,
        "replay": replay,
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
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if phase == "preflight":
        result = run_preflight(run_dir, mode)
    elif phase == "baseline":
        result = run_baseline(run_dir, mode)
    elif phase == "capture":
        result = run_capture(run_dir, mode)
    elif phase == "families":
        result = run_families(run_dir, mode)
    elif phase == "counters":
        result = run_counters(run_dir, mode)
    elif phase == "replay":
        result = run_replay(run_dir, mode)
    elif phase == "report":
        result = run_report(run_dir, mode)
    else:
        raise RemainingGapError(f"unknown phase {phase}")
    elapsed = time.time() - started
    result = dict(result)
    result["phase_elapsed_s"] = elapsed
    if elapsed > AGGREGATE_DEADLINE_S:
        result["aggregate_deadline_exceeded"] = True
    return result


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
    if args.phase in {"preflight", "baseline", "capture"} and result.get("blocked"):
        return 0
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemainingGapError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
