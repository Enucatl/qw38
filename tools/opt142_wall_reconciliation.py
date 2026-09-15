"""OPT-142 whole-wall profiling residual reconciliation.

Diagnostics only. Reuses authenticated OPT-138 nsys captures. No production
scheduling, selector, or kernel changes.
Phases: preflight, reproduce, reconcile, report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools import opt136_graph_accounting as opt136  # noqa: E402
from tools.performance_evidence import (  # noqa: E402
    RESIDUAL_WALL_LIMIT,
    audit_window_from_tables,
    default_identity,
    ns_to_ms,
    parse_nsys_sqlite,
    regroup_fused_families,
    reconcile_whole_gap,
    reconcile_window_wall,
    resolve_capture_window,
    validate_coverage_document,
    whole_wall_gap_from_partitions,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt142_wall_reconciliation_contract.json"
ITERATION = ROOT / "pins/opt142_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt142_wall_reconciliation.json"
EVIDENCE = ROOT / "evidence/optimization/opt142-wall-reconciliation"
REPORT = EVIDENCE / "REPORT.md"
OPT138_EVIDENCE = ROOT / "evidence/optimization/opt138-remaining-gap"
OPT138_FIXTURE = ROOT / "fixtures/opt138_remaining_gap_profile.json"
OPT138_TRACES = ROOT / "build/optimization-runs/opt138/traces"
OPT138_RUN = ROOT / "build/optimization-runs/opt138"
SELECTOR = "decode_segments8"
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
PREFIXES = (128, 2048)
WINDOWS = ("early", "middle", "late")
PREFILL_TOKENS = 4096
WINDOW_TOKENS = 12
REPETITIONS = 3
CAPACITY = 131072
PHASES = ("preflight", "reproduce", "reconcile", "report")
CHILD_TIMEOUT_S = 300
AGGREGATE_DEADLINE_S = 7200
RESULT_PREFIX = "QW38_OPT142_WALL_RECONCILIATION_RESULT="
REQUIRED_WINDOWS = (
    ("decode", 128, "early"),
    ("decode", 128, "middle"),
    ("decode", 128, "late"),
    ("decode", 2048, "early"),
    ("decode", 2048, "middle"),
    ("decode", 2048, "late"),
    ("prefill", 4096, "prefill"),
)
REPRODUCE_WINDOWS = (
    ("decode", 2048, "middle", 0),
    ("prefill", 4096, "prefill", 0),
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
    "reproduce",
    "reconcile",
    "answers",
    "report_path",
)


class WallReconciliationError(RuntimeError):
    """OPT-142 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-142":
        raise WallReconciliationError("wall-reconciliation contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-142", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-142 mode={mode} phase={family} "
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
        "task": "OPT-142",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "windows": list(WINDOWS),
        "preflight": None,
        "reproduce": None,
        "reconcile": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise WallReconciliationError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise WallReconciliationError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise WallReconciliationError("selector drifted from decode_segments8")


def sqlite_path(engine: str, kind: str, prefix: int, window: str, rep: int) -> Path:
    if kind == "prefill":
        return OPT138_TRACES / f"{engine}-p4096-node-r{rep}.sqlite"
    index = WINDOWS.index(window) + 1
    return OPT138_TRACES / f"{engine}-d{prefix}-node-r{rep}.{index}.sqlite"


def opt138_recorded_hashes() -> dict[str, Any]:
    for path in (
        OPT138_RUN / "preflight.json",
        OPT138_FIXTURE,
    ):
        if not path.is_file():
            continue
        payload = load_json(path)
        preflight = payload.get("preflight") if "preflight" in payload else payload
        if isinstance(preflight, Mapping):
            hashes = preflight.get("hashes") or {}
            if hashes:
                return dict(hashes)
    return {}


def capture_identity(
    engine: str, kind: str, prefix: int, window: str, rep: int
) -> dict[str, Any]:
    metric = "prefill" if kind == "prefill" else "decode_only"
    eval_count = PREFILL_TOKENS if kind == "prefill" else WINDOW_TOKENS
    populated = PREFILL_TOKENS if kind == "prefill" else prefix
    return default_identity(
        capture_arm="node",
        engine=engine,
        prefix=populated,
        populated_length=populated,
        eval_count=eval_count,
        metric=metric,
        output_boundary="eval_logits_no_host_materialize",
    )


def audit_opt138_sqlite(
    path: Path, *, engine: str, kind: str, prefix: int, window: str
) -> dict[str, Any]:
    tables = parse_nsys_sqlite(path)
    coverage = audit_window_from_tables(
        tables,
        identity=capture_identity(engine, kind, prefix, window, 0),
    )
    resolved = resolve_capture_window(tables)
    wall = coverage.get("wall_reconciliation") or reconcile_window_wall(
        tables,
        window_start_ns=int(resolved["window_start_ns"]),
        window_end_ns=int(resolved["window_end_ns"]),
        window_source=str(resolved.get("source") or ""),
        missing_observation=resolved.get("missing_observation"),
    )
    if kind == "prefill" and engine == "llama" and not tables.nodes and tables.kernels:
        coverage["node_identities_absent_used_kernel_leaves"] = True
        wall["node_identities_absent_used_kernel_leaves"] = True
    return {
        "path": relpath(path),
        "coverage": coverage,
        "wall": wall,
        "resolved_window": resolved,
        "tables_errors": list(tables.errors),
        "kernel_count": len(tables.kernels),
        "cpu_count": len(tables.cpu_apis),
        "overhead_count": len(tables.overhead),
        "nvtx_count": len(tables.nvtx),
    }


def family_ms_from_coverage(coverage: Mapping[str, Any]) -> dict[str, float]:
    families = coverage.get("families") or {}
    out: dict[str, float] = {}
    if isinstance(families, Mapping):
        for name, row in families.items():
            if isinstance(row, Mapping) and row.get("union_ns") is not None:
                out[str(name)] = float(ns_to_ms(int(row["union_ns"])))
    return out


def legacy_family_gap(
    quartz: Mapping[str, Any], llama: Mapping[str, Any]
) -> dict[str, Any]:
    grouped = regroup_fused_families(
        family_ms_from_coverage(quartz.get("coverage") or {}),
        family_ms_from_coverage(llama.get("coverage") or {}),
    )
    q_wall = float((quartz.get("wall") or {}).get("window_ms") or 0.0)
    l_wall = float((llama.get("wall") or {}).get("window_ms") or 0.0)
    recon = reconcile_whole_gap(
        quartz_wall_ms=q_wall,
        llama_wall_ms=l_wall,
        matched_disjoint_excess_ms=float(
            grouped.get("matched_disjoint_excess_ms") or 0.0
        ),
        unmatched_work_gap_ms=float(grouped.get("unmatched_work_gap_ms") or 0.0),
        scheduling_host_residual_ms=0.0,
    )
    recon["method"] = "opt138_family_sum_host_unmeasured"
    return {"regrouped": grouped, "reconciliation": recon}


def compact_window_row(row: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "kind": row.get("kind"),
        "prefix": row.get("prefix"),
        "window": row.get("window"),
        "rep": row.get("rep"),
        "key": row.get("key"),
        "quartz_unresolved_share": row.get("quartz_unresolved_share"),
        "quartz_within_limit": row.get("quartz_within_limit"),
        "disjoint_wall_gap": row.get("disjoint_wall_gap"),
        "legacy_family_sum": {
            "reconciliation": (
                (row.get("legacy_family_sum") or {}).get("reconciliation")
            )
        },
    }
    for engine in ("quartz", "llama"):
        block = row.get(engine) or {}
        wall = block.get("wall") or {}
        compact[engine] = {
            "ok": block.get("ok"),
            "path": block.get("path"),
            "window_source": wall.get("window_source"),
            "missing_observation": wall.get("missing_observation"),
            "window_ms": wall.get("window_ms"),
            "gpu_union_ms": wall.get("gpu_union_ms"),
            "host_exclusive_ms": wall.get("host_exclusive_ms"),
            "unresolved_ms": wall.get("unresolved_ms"),
            "unresolved_share_of_wall": wall.get("unresolved_share_of_wall"),
            "family_overlap_ms": ns_to_ms(int(wall.get("family_overlap_ns") or 0)),
            "perturbed": wall.get("perturbed"),
            "node_identities_absent_used_kernel_leaves": wall.get(
                "node_identities_absent_used_kernel_leaves"
            ),
        }
    return compact


def compact_phase(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    out = dict(payload)
    windows = out.get("windows")
    if isinstance(windows, list) and windows and isinstance(windows[0], Mapping):
        if "wall" in (windows[0].get("quartz") or {}):
            out["windows"] = [compact_window_row(row) for row in windows]
    traces = out.get("traces")
    if isinstance(traces, list):
        out["trace_count"] = len(traces)
        out["traces"] = traces[:4]
    return out


def perturbation_from_twins(
    engine: str, kind: str, prefix: int, rep: int
) -> dict[str, Any]:
    capture = (
        load_json(OPT138_RUN / "capture.json")
        if (OPT138_RUN / "capture.json").is_file()
        else {}
    )
    captures = capture.get("captures") or {}
    if kind == "prefill":
        node_key = f"{engine}-p4096-node-r{rep}"
        unp_key = f"{engine}-p4096-unprofiled-r{rep}"
        metric = "prefill_ms"
    else:
        node_key = f"{engine}-d{prefix}-node-r{rep}"
        unp_key = f"{engine}-d{prefix}-unprofiled-r{rep}"
        metric = "decode_only_ms"
    node = ((captures.get(node_key) or {}).get("parsed") or {}).get(metric)
    unp = ((captures.get(unp_key) or {}).get("parsed") or {}).get(metric)
    if not node or not unp or float(unp) == 0:
        return {
            "usable": False,
            "reason": "matched unprofiled twin missing",
            "identity": "whole_request_native_chrono",
        }
    rel = (float(node) - float(unp)) / float(unp)
    return {
        "usable": True,
        "identity": "whole_request_native_chrono",
        "metric": metric,
        "profiled_ms": float(node),
        "unprofiled_ms": float(unp),
        "median_relative_perturbation": rel,
        "source": "opt138_unprofiled_twin",
        "acceptable": abs(rel) <= RESIDUAL_WALL_LIMIT,
    }


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    available, gpu_blocker = gpu_available()
    pins = opt136.authenticate_current_pins()
    recorded = opt138_recorded_hashes()
    current = {
        "quartz_native": opt136.sha256_file(ROOT / opt136.NATIVE),
        "llama_profile": opt136.sha256_file(ROOT / opt136.LLAMA_BIN),
    }
    sitting_matches_capture = bool(recorded) and current.get(
        "quartz_native"
    ) == recorded.get("quartz_native")
    traces = []
    missing = []
    for engine in ("quartz", "llama"):
        for kind, prefix, window in REQUIRED_WINDOWS:
            for rep in range(REPETITIONS):
                path = sqlite_path(engine, kind, prefix, window, rep)
                row = {
                    "engine": engine,
                    "kind": kind,
                    "prefix": prefix,
                    "window": window,
                    "rep": rep,
                    "path": relpath(path) if path.exists() else str(path),
                    "present": path.is_file() and path.stat().st_size > 0,
                    "bytes": path.stat().st_size if path.is_file() else 0,
                }
                traces.append(row)
                if not row["present"]:
                    missing.append(row["path"])
    payload = {
        "schema_version": 1,
        "task": "OPT-142",
        "phase": "preflight",
        "mode": mode,
        "ok": pins.get("ok") is True and not missing,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "authenticated_current_pins": pins,
        "opt138_recorded_hashes": recorded,
        "current_binary_hashes": current,
        "sitting_binary_matches_opt138_capture": sitting_matches_capture,
        "reuse_opt138_traces": True,
        "reuse_reason": (
            "OPT-142 explains the residual inside the authenticated OPT-138 "
            "captures. Recapture would mix a drifted sitting binary into that "
            "identity."
        ),
        "trace_root": relpath(OPT138_TRACES),
        "traces": traces,
        "missing_traces": missing,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
        "claims_throughput": False,
        "production_kept": True,
    }
    if missing:
        payload["blocked"] = True
        payload["missing_observation"] = "retained OPT-138 sqlite missing: " + ",".join(
            missing
        )
    return store_sidecar(run_dir, "preflight.json", payload)


def _legacy_and_wall(engine_rows: Mapping[str, Any]) -> dict[str, Any]:
    quartz = engine_rows.get("quartz") or {}
    llama = engine_rows.get("llama") or {}
    legacy = None
    wall_gap = None
    if quartz and llama:
        legacy = legacy_family_gap(quartz, llama)
        wall_gap = whole_wall_gap_from_partitions(
            quartz.get("wall") or {}, llama.get("wall") or {}
        )
    return {"legacy_family_sum": legacy, "disjoint_wall_gap": wall_gap}


def run_reproduce(run_dir: Path, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for kind, prefix, window, rep in REPRODUCE_WINDOWS:
        pair: dict[str, Any] = {
            "kind": kind,
            "prefix": prefix,
            "window": window,
            "rep": rep,
        }
        for engine in ("quartz", "llama"):
            path = sqlite_path(engine, kind, prefix, window, rep)
            if not path.is_file():
                pair[engine] = {"ok": False, "missing": relpath(path)}
                continue
            audited = audit_opt138_sqlite(
                path, engine=engine, kind=kind, prefix=prefix, window=window
            )
            audited["ok"] = True
            audited["perturbation"] = perturbation_from_twins(engine, kind, prefix, rep)
            pair[engine] = audited
        pair.update(_legacy_and_wall(pair))
        rows.append(pair)
    payload = {
        "schema_version": 1,
        "task": "OPT-142",
        "phase": "reproduce",
        "mode": mode,
        "ok": all(
            (row.get("quartz") or {}).get("ok") and (row.get("llama") or {}).get("ok")
            for row in rows
        ),
        "note": (
            "Reproduced D2048 middle and P4096 from retained OPT-138 sqlite "
            "before using the new wall partition as the acceptance equation."
        ),
        "windows": rows,
        "family_plan": family_plan(mode, "reproduce"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "reproduce.json", payload)
    return store_sidecar(run_dir, "reproduce.json", payload)


def run_reconcile(run_dir: Path, mode: str) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    quartz_failures: list[str] = []
    coverage_rows: list[dict[str, Any]] = []
    for kind, prefix, window in REQUIRED_WINDOWS:
        for rep in range(REPETITIONS):
            pair: dict[str, Any] = {
                "kind": kind,
                "prefix": prefix,
                "window": window,
                "rep": rep,
                "key": f"{kind}-d{prefix}-{window}-r{rep}",
            }
            for engine in ("quartz", "llama"):
                path = sqlite_path(engine, kind, prefix, window, rep)
                if not path.is_file():
                    pair[engine] = {"ok": False, "missing": str(path)}
                    continue
                audited = audit_opt138_sqlite(
                    path, engine=engine, kind=kind, prefix=prefix, window=window
                )
                audited["ok"] = True
                twin = perturbation_from_twins(engine, kind, prefix, rep)
                audited["perturbation"] = twin
                wall = dict(audited["wall"])
                wall["unprofiled_twin"] = twin
                audited["wall"] = wall
                pair[engine] = audited
                if engine == "quartz":
                    cov = dict(audited["coverage"])
                    cov["capture_key"] = (
                        f"quartz-{'p4096' if kind == 'prefill' else 'd' + str(prefix)}"
                        f"-node-r{rep}"
                    )
                    cov["window_name"] = window
                    coverage_rows.append(cov)
                    share = wall.get("unresolved_share_of_wall")
                    perturbed = bool(wall.get("perturbed"))
                    if share is None or share > RESIDUAL_WALL_LIMIT:
                        quartz_failures.append(pair["key"])
                    elif perturbed:
                        quartz_failures.append(pair["key"])
            pair.update(_legacy_and_wall(pair))
            q_share = ((pair.get("quartz") or {}).get("wall") or {}).get(
                "unresolved_share_of_wall"
            )
            pair["quartz_unresolved_share"] = q_share
            pair["quartz_within_limit"] = (
                q_share is not None and q_share <= RESIDUAL_WALL_LIMIT
            )
            windows.append(pair)
    document = {
        "schema_version": 1,
        "task": "OPT-142",
        "status": "valid" if coverage_rows and not quartz_failures else "incomplete",
        "windows": coverage_rows,
    }
    validation = validate_coverage_document(document)
    document["validation"] = validation
    dump_json(EVIDENCE / "coverage.json", document)
    dump_json(EVIDENCE / "per-window.json", {"windows": windows})
    missing_obs = None
    if quartz_failures:
        failed = [row for row in windows if row["key"] in quartz_failures]
        details = []
        for row in failed:
            wall = (row.get("quartz") or {}).get("wall") or {}
            details.append(
                f"{row['key']}: unresolved_share="
                f"{wall.get('unresolved_share_of_wall')!r} "
                f"window_source={wall.get('window_source')!r} "
                f"missing={wall.get('missing_observation')!r} "
                f"perturbed={wall.get('perturbed')!r}"
            )
        missing_obs = (
            "absolute unresolved residual >5% of Quartz wall or excessive "
            "profiler perturbation on: " + "; ".join(details)
        )
    payload = {
        "schema_version": 1,
        "task": "OPT-142",
        "phase": "reconcile",
        "mode": mode,
        "ok": not quartz_failures and bool(validation.get("ok")),
        "coverage_validation": validation,
        "window_count": len(windows),
        "quartz_failures": quartz_failures,
        "missing_observation": missing_obs,
        "residual_limit": RESIDUAL_WALL_LIMIT,
        "windows": windows,
        "family_plan": family_plan(mode, "reconcile"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "reconcile.json", payload)


def _engine_wall(row: Mapping[str, Any], engine: str = "quartz") -> dict[str, Any]:
    block = row.get(engine) or {}
    if not isinstance(block, Mapping):
        return {}
    wall = block.get("wall")
    if isinstance(wall, Mapping):
        return dict(wall)
    return dict(block)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(payload: Mapping[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    reproduce = payload.get("reproduce") or {}
    reconcile = payload.get("reconcile") or {}
    answers = payload.get("answers") or {}
    lines = [
        "# OPT-142 — Reconcile remaining whole-wall profiling residual",
        "",
        "Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.",
        "No production scheduling, selector, or kernel changes.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`.",
        f"Parent `{PARENT}`. Selector `{SELECTOR}`.",
        "",
        "## Identity",
        "",
        f"pins ok=`{(preflight.get('authenticated_current_pins') or {}).get('ok')}`",
        "sitting binary matches OPT-138 capture="
        f"`{preflight.get('sitting_binary_matches_opt138_capture')}`",
        f"reuse OPT-138 traces=`{preflight.get('reuse_opt138_traces')}`",
        "",
        "Current sitting Quartz binary hash may differ from the OPT-138 capture",
        "hash. This task reuses the authenticated OPT-138 traces because the",
        "question is the residual inside those captures, not a new sitting.",
        "",
        "## OPT-138 residual reproduced (family-sum method, host unmeasured)",
        "",
    ]
    for row in reproduce.get("windows") or []:
        legacy = ((row.get("legacy_family_sum") or {}).get("reconciliation")) or {}
        wall = row.get("disjoint_wall_gap") or {}
        lines.extend(
            [
                f"### {row.get('kind')} prefix={row.get('prefix')} "
                f"window={row.get('window')} r{row.get('rep')}",
                "",
                "- family-sum residual share of Quartz wall: "
                f"`{_fmt(legacy.get('residual_share_of_quartz_wall'))}` "
                f"(ranking_complete=`{legacy.get('ranking_complete')}`)",
                "- disjoint wall-gap residual share: "
                f"`{_fmt(wall.get('residual_share_of_quartz_wall'))}` "
                f"(ranking_complete=`{wall.get('ranking_complete')}`)",
                "- Quartz unresolved share: "
                f"`{_fmt(_engine_wall(row).get('unresolved_share_of_wall'))}`",
                "- Llama family-overlap bookkeeping ms: "
                f"`{_fmt(((wall.get('family_overlap_bookkeeping_ms') or {}).get('llama')))}`",
                f"- Quartz window source: `{_engine_wall(row).get('window_source')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-window Quartz wall partition",
            "",
            "| Window | Wall ms | GPU union ms | Host exclusive ms | Unresolved ms | Unresolved share | Source | Within 5% |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in reconcile.get("windows") or []:
        if (row.get("quartz") or {}).get("ok") is not True:
            continue
        wall = _engine_wall(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("key")),
                    _fmt(wall.get("window_ms")),
                    _fmt(wall.get("gpu_union_ms")),
                    _fmt(wall.get("host_exclusive_ms")),
                    _fmt(wall.get("unresolved_ms")),
                    _fmt(wall.get("unresolved_share_of_wall")),
                    str(wall.get("window_source")),
                    str(row.get("quartz_within_limit")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Host exclusive is CUDA runtime/driver/sync APIs that do not overlap",
            "GPU union. Overlapping `cudaEventSynchronize` is wait duration, not",
            "added GPU wall. Remaining process-scoped gaps are **unresolved**,",
            "not proven device-wide idle. Family-sum overlap is bookkeeping.",
            "",
            "Llama P4096 node traces have no `NVTX_EVENTS` table, so those llama",
            "windows use GPU-span fallback. Host time before the first and after",
            "the last llama GPU interval is unobservable there. Quartz P4096 uses",
            "`qw38.prefill_chunk`. The acceptance gate is Quartz unresolved share.",
            "",
            "## Status",
            "",
            f"status=`{payload.get('status')}`",
            f"blocked=`{answers.get('blocked')}`",
            f"missing observation=`{answers.get('missing_observation')}`",
            f"unresolved limit=`{RESIDUAL_WALL_LIMIT}` of Quartz wall",
            "",
            "Family rankings from OPT-138 remain separately qualified. This task",
            "does not redistribute unexplained time across families and does not",
            "widen the 5% tolerance.",
            "",
        ]
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    reproduce = load_sidecar(run_dir, "reproduce.json") or {}
    reconcile = load_sidecar(run_dir, "reconcile.json") or {}
    failures = list(reconcile.get("quartz_failures") or [])
    missing = reconcile.get("missing_observation")
    if not preflight.get("ok"):
        missing = missing or preflight.get("missing_observation") or "preflight failed"
        failures = failures or ["preflight"]
    blocked = bool(failures or missing or not reconcile.get("ok"))
    status = "blocked" if blocked else "measured"
    answers = {
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "blocked": blocked,
        "missing_observation": missing,
        "quartz_failures": failures,
        "unresolved_limit": RESIDUAL_WALL_LIMIT,
        "whole_wall_ranking_complete": not blocked,
        "family_rankings_separately_qualified": True,
        "no_proportional_allocation": True,
        "no_production_scheduling_change": True,
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-142",
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
        "preflight": compact_phase(preflight),
        "reproduce": compact_phase(reproduce),
        "reconcile": compact_phase(reconcile),
        "answers": answers,
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "family_plan": family_plan(mode, "report"),
    }
    validate_fixture(payload)
    persist_fixture(payload)
    write_report(
        {
            **payload,
            "preflight": preflight,
            "reproduce": reproduce,
            "reconcile": reconcile,
        }
    )
    dump_json(EVIDENCE / "answers.json", answers)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if phase == "preflight":
        result = run_preflight(run_dir, mode)
    elif phase == "reproduce":
        result = run_reproduce(run_dir, mode)
    elif phase == "reconcile":
        result = run_reconcile(run_dir, mode)
    elif phase == "report":
        result = run_report(run_dir, mode)
    else:
        raise WallReconciliationError(f"unknown phase {phase}")
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
    json.dump(
        {
            "phase": args.phase,
            "ok": bool(result.get("ok", True)),
            **({"status": result.get("status")} if result.get("status") else {}),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.stdout.write(
        RESULT_PREFIX
        + json.dumps({"phase": args.phase, "ok": bool(result.get("ok", True))})
        + "\n"
    )
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WallReconciliationError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
