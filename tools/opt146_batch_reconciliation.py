"""OPT-146 final batch reconciliation after the OPT-137 stack.

Diagnostics only. Reuses OPT-136/138 capture, OPT-142 wall accounting, and
OPT-135 paired statistics. No production selector, kernel, or scheduling
change. Phases: preflight, throughput, capture, reconcile, families,
experiments, report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
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
from tools import opt138_remaining_gap_profile as opt138  # noqa: E402
from tools import opt142_wall_reconciliation as opt142  # noqa: E402
from tools.performance_evidence import (  # noqa: E402
    RESIDUAL_WALL_LIMIT,
    T_CRIT_DF9_TWO_SIDED,
    audit_window_from_tables,
    default_identity,
    paired_log_ratio_ci,
    parse_nsys_sqlite,
    regroup_fused_families,
    reconcile_whole_gap,
    reconcile_window_wall,
    resolve_capture_window,
    select_top_two_families,
    student_t_interval,
    validate_coverage_document,
    whole_wall_gap_from_partitions,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt146_batch_reconciliation_contract.json"
ITERATION = ROOT / "pins/opt146_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt146_batch_reconciliation.json"
EVIDENCE = ROOT / "evidence/optimization/opt146-batch-reconciliation"
REPORT = EVIDENCE / "REPORT.md"
RESULT_PREFIX = "QW38_OPT146_BATCH_RECONCILIATION_RESULT="
NATIVE = opt136.NATIVE
LLAMA_BIN = opt136.LLAMA_BIN
GPU_LOCK = opt136.GPU_LOCK
MODEL = opt136.MODEL
CAPACITY = opt136.CAPACITY
DECODE_TOKENS = opt136.DECODE_TOKENS
WINDOW_TOKENS = opt136.WINDOW_TOKENS
WINDOWS = opt136.WINDOWS
DECODE_PREFIXES = (128, 2048, 8192, 32768)
PROFILE_PREFIXES = (128, 2048)
PREFILL_TOKENS = 4096
ENGINES = opt136.ENGINES
ARMS = opt136.ARMS
REPETITIONS = 3
BASELINE_WARMUPS = 3
BASELINE_PAIRS = 10
CHILD_TIMEOUT_S = 300
LONG_TIMEOUT_S = 1200
AGGREGATE_DEADLINE_S = 7200
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
SELECTOR = "decode_segments8"
MMA_THRESHOLD = 8192
PHASES = (
    "preflight",
    "throughput",
    "capture",
    "reconcile",
    "families",
    "experiments",
    "report",
)
THROUGHPUT_WORKLOADS = (
    ("d128", "decode", 128),
    ("d2048", "decode", 2048),
    ("d8192", "decode", 8192),
    ("d32768", "decode", 32768),
    ("p4096", "prefill", None),
)
REQUIRED_WINDOWS = opt142.REQUIRED_WINDOWS
BATCH_EXPERIMENTS = (
    (
        "OPT-143",
        ROOT / "fixtures/opt143_short_attention.json",
        ROOT / "evidence/optimization/opt143-short-attention/freeze.json",
        "decode",
        "attn_core",
    ),
    (
        "OPT-144",
        ROOT / "fixtures/opt144_prefill_attention.json",
        ROOT / "evidence/optimization/opt144-prefill-attention/freeze.json",
        "prefill",
        "attn_core",
    ),
    (
        "OPT-145",
        ROOT / "fixtures/opt145_secondary_family.json",
        ROOT / "evidence/optimization/opt145-secondary-family/freeze.json",
        "decode",
        "residual_norm_quant",
    ),
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
    "allocated_capacity",
    "decode_prefixes",
    "prefill_tokens",
    "preflight",
    "throughput",
    "captures",
    "reconcile",
    "families",
    "experiments",
    "answers",
    "report_path",
)


class BatchReconciliationError(RuntimeError):
    """OPT-146 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-146":
        raise BatchReconciliationError("batch-reconciliation contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-146", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-146 mode={mode} phase={family} "
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
        "task": "OPT-146",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "llama_revision": LLAMA_REV,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "allocated_capacity": CAPACITY,
        "decode_prefixes": list(DECODE_PREFIXES),
        "profile_prefixes": list(PROFILE_PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "windows": list(WINDOWS),
        "preflight": None,
        "throughput": None,
        "captures": {},
        "reconcile": None,
        "families": None,
        "experiments": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": 0,
        "candidate_measured_delta": 0,
    }


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        payload = load_json(FIXTURE)
        if isinstance(payload, dict) and payload.get("task") == "OPT-146":
            return payload
    return empty_fixture()


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise BatchReconciliationError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise BatchReconciliationError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise BatchReconciliationError("selector drifted from decode_segments8")
    if int(payload.get("allocated_capacity") or 0) != CAPACITY:
        raise BatchReconciliationError("allocated_capacity must be 131072")


def child_timeout_s(kind: str, prefix: int | None) -> int:
    if kind == "decode" and prefix is not None and prefix >= 8192:
        return LONG_TIMEOUT_S
    return CHILD_TIMEOUT_S


def tok_s(tokens: int, wall_ms: float | None) -> float | None:
    if wall_ms is None or float(wall_ms) <= 0:
        return None
    return float(tokens) * 1000.0 / float(wall_ms)


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(item) for item in values) / len(values)


def sqlite_path(
    traces: Path, engine: str, kind: str, prefix: int, window: str, rep: int
) -> Path:
    if kind == "prefill":
        direct = traces / f"{engine}-p4096-node-r{rep}.sqlite"
        indexed = traces / f"{engine}-p4096-node-r{rep}.1.sqlite"
        return direct if direct.is_file() else indexed
    index = WINDOWS.index(window) + 1
    return traces / f"{engine}-d{prefix}-node-r{rep}.{index}.sqlite"


def opt137_historical_rates() -> dict[str, Any]:
    path = ROOT / "fixtures/opt137_long_attention.json"
    if not path.is_file():
        return {"ok": False, "reason": "opt137_fixture_missing"}
    payload = load_json(path)
    return {
        "ok": True,
        "claim_type": "historical",
        "production_kept": bool(payload.get("production_kept")),
        "verdict": payload.get("verdict"),
        "tok_s_deltas": payload.get("tok_s_deltas") or {},
        "usable_as_paired_gain": False,
        "note": (
            "OPT-137 candidate rates are historical. They are not a paired "
            "gain versus the OPT-146 sitting unless binary identity matches."
        ),
    }


def freeze_opt137_control(
    pins: Mapping[str, Any], hashes: Mapping[str, Any]
) -> dict[str, Any]:
    identity_ok = bool(pins.get("ok"))
    return {
        "reconstructable": identity_ok,
        "method": "current_production_is_opt137_kept_stack",
        "parent": PARENT,
        "selector": SELECTOR,
        "opt137_dense_mma": bool(pins.get("opt137_dense_mma")),
        "q4_decode": pins.get("q4_decode"),
        "execution_graphs": pins.get("execution_graphs"),
        "mma_threshold": MMA_THRESHOLD,
        "allocated_capacity": CAPACITY,
        "production_pins_changed_after_opt137": False,
        "same_sitting_as_final": True,
        "paired_gpu_rounds_versus_distinct_control_binary": False,
        "historical_rates_as_paired_gain": False,
        "reason": (
            "OPT-143/144/145 did not keep. Production pins remain the OPT-137 "
            "kept stack. A distinct pre-keep control cannot be reconstructed "
            "without changing production pins."
        ),
        "hashes": dict(hashes),
        "historical": opt137_historical_rates(),
        "claim_type": "measured" if identity_ok else "incomplete",
    }


def summarize_experiment(
    task: str,
    fixture_path: Path,
    freeze_path: Path,
    phase: str,
    family: str,
) -> dict[str, Any]:
    if not fixture_path.is_file():
        return {
            "task": task,
            "ok": False,
            "status": "incomplete",
            "reason": f"missing {relpath(fixture_path)}",
        }
    fixture = load_json(fixture_path)
    freeze = load_json(freeze_path) if freeze_path.is_file() else {}
    verdict = str(fixture.get("verdict") or freeze.get("verdict") or "unknown")
    selected = (
        freeze.get("selected_family")
        or freeze.get("family")
        or fixture.get("selected_family")
        or family
    )
    return {
        "task": task,
        "ok": True,
        "status": "no_opportunity" if verdict == "no_opportunity" else verdict,
        "verdict": verdict,
        "keep": False,
        "no_keep": verdict != "keep",
        "candidate": fixture.get("candidate"),
        "selected_family": selected,
        "unselected_family": freeze.get("unselected_family")
        or fixture.get("unselected_family"),
        "phase": phase,
        "supported_mechanism": freeze.get("supported_mechanism"),
        "production_kept": bool(fixture.get("production_kept")),
        "shipping_delta": 0,
        "candidate_measured_delta": None,
        "quality": "nll_not_required_no_arithmetic_change",
        "excess_ms": freeze.get("excess_ms"),
        "admission_reason": freeze.get("admission_reason"),
        "opt137_mma_retained": freeze.get("opt137_mma_retained", True),
        "claims_throughput": False,
        "report_path": fixture.get("report_path"),
        "freeze_path": relpath(freeze_path) if freeze_path.is_file() else None,
    }


def experiment_outcomes() -> dict[str, Any]:
    rows = [
        summarize_experiment(task, fixture, freeze, phase, family)
        for task, fixture, freeze, phase, family in BATCH_EXPERIMENTS
    ]
    all_no_opportunity = all(row.get("verdict") == "no_opportunity" for row in rows)
    any_keep = any(row.get("keep") for row in rows)
    return {
        "schema_version": 1,
        "task": "OPT-146",
        "ok": all(row.get("ok") for row in rows)
        and all_no_opportunity
        and not any_keep,
        "all_no_opportunity": all_no_opportunity,
        "any_keep": any_keep,
        "parent_retained": True,
        "shipping_delta": 0,
        "experiments": rows,
        "claim_type": "measured",
    }


def net_batch_gains(
    control: Mapping[str, Any], outcomes: Mapping[str, Any]
) -> dict[str, Any]:
    shipping = 0
    return {
        "shipping_delta": shipping,
        "candidate_measured_delta": 0,
        "identity": "production_pins_unchanged_after_opt137",
        "opt137_control": {
            "reconstructable": bool(control.get("reconstructable")),
            "method": control.get("method"),
            "historical_rates_as_paired_gain": False,
        },
        "batch_keeps": 0,
        "batch_no_opportunity": list(
            row.get("task") for row in (outcomes.get("experiments") or [])
        ),
        "claims_throughput": False,
        "claim_type": "derived",
        "formula": "shipping_delta = 0 because no keep after OPT-137",
        "units": "tok/s shipping change versus OPT-137 starting stack",
        "note": (
            "Individual OPT-143/144/145 candidate speedups were never kept "
            "and are not additive. Net batch shipping delta is 0."
        ),
    }


def next_experiment_spec(
    phase: str,
    selection: Mapping[str, Any],
    outcomes: Mapping[str, Any],
) -> dict[str, Any]:
    selected = list(selection.get("selected") or [])
    top = selected[0] if selected else None
    family = None if top is None else str(top.get("family"))
    stats = {}
    if isinstance(top, Mapping):
        stats = (
            top.get("stats") or top.get("d2048_middle") or top.get("d128_middle") or {}
        )

    def _matches(row: Mapping[str, Any]) -> bool:
        if not family:
            return False
        if row.get("unselected_family") == family:
            return True
        return row.get("selected_family") == family and row.get("phase") == phase

    prior = next(
        (row for row in (outcomes.get("experiments") or []) if _matches(row)),
        None,
    )
    sources = opt138.FAMILY_SOURCES.get(family or "", {"quartz": None, "llama": None})
    resolving = (
        "need matched node-level family excess with a positive "
        "one-sided 95% lower bound (df=2) and a source-grounded mechanism"
    )
    if family is None:
        resolving = (
            "need matched node-level family excess with a positive "
            "one-sided 95% lower bound (df=2)"
        )
    elif prior and prior.get("verdict") == "no_opportunity":
        resolving = (
            f"{prior.get('task')} already measured no_opportunity for "
            f"{family}: {prior.get('admission_reason') or 'null mechanism'}. "
            "A later experiment requires a new named source/SASS or fused-"
            "boundary identity, not a rerun of occupancy/DRAM ranking."
        )
    return {
        "phase": phase,
        "selected_family": family,
        "measured_excess_ms": None
        if top is None
        else (stats.get("mean_ms") or top.get("score_ms")),
        "uncertainty": {
            "one_sided_low": stats.get("one_sided_low"),
            "df": 2,
            "n": stats.get("n"),
        }
        if family
        else None,
        "quartz_source_function": sources.get("quartz"),
        "llama_source_function": sources.get("llama"),
        "supported_mechanism": None,
        "candidate": None,
        "prior_experiment": None if prior is None else prior.get("task"),
        "prior_verdict": None if prior is None else prior.get("verdict"),
        "target_workloads": ["D128", "D2048"] if phase == "decode" else ["P4096"],
        "guards_under_opt135": {
            "policy": "target_guard_v2",
            "decode_guards": ["D128", "D2048", "D8192", "D32768", "P4096"],
        },
        "expected_benefit": {
            "kind": "labeled_estimate",
            "value": None,
            "label": "unknown_until_mechanism",
        },
        "resolving_measurement": resolving,
        "quality_state_gates": "N/A diagnostics; any later keep requires OPT-058 NLL",
        "evidence_links": [
            "evidence/optimization/opt146-batch-reconciliation/family-gaps.json",
            "evidence/optimization/opt146-batch-reconciliation/coverage.json",
            "evidence/optimization/opt143-short-attention/freeze.json",
            "evidence/optimization/opt144-prefill-attention/freeze.json",
            "evidence/optimization/opt145-secondary-family/freeze.json",
        ],
    }


def parse_result(text: str) -> dict[str, Any] | None:
    return opt138.parse_result(text)


def _blocked_gpu_phase(
    phase: str, run_dir: Path, mode: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
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
    return store_sidecar(run_dir, f"{phase}.json", payload)


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    available, gpu_blocker = gpu_available()
    residents = opt136.gpu_residents() if available else []
    pins = opt138.authenticate_opt138_pins()
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
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt146-diagnostics"],
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
            "capacity": (parsed or {}).get("capacity"),
            "stdout_tail": (completed.stdout or "")[-2000:],
        }
    hashes = {
        "quartz_native": native_hash,
        "llama_profile": llama.get("hash"),
        "replay": opt136.sha256_file(ROOT / "build/qw38-cuda-component-replay"),
    }
    control = freeze_opt137_control(pins, hashes)
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
    if (
        identity.get("capacity") not in (None, CAPACITY)
        and int(identity.get("capacity") or 0) != CAPACITY
    ):
        blocked.append("capacity_mismatch")
    ok = (
        available
        and nsys_available
        and native_path.is_file()
        and pins["ok"]
        and bool(identity.get("ok"))
        and bool(llama.get("ok"))
        and not blocked
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
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
        "llama_bin": LLAMA_BIN,
        "hashes": hashes,
        "make_returncode": make.returncode,
        "make_stderr_tail": (make.stderr or "")[-2000:],
        "llama_build": llama,
        "identity": identity,
        "authenticated_current_pins": pins,
        "opt137_control": control,
        "parent": PARENT,
        "selector": SELECTOR,
        "capacity": CAPACITY,
        "decode_prefixes": list(DECODE_PREFIXES),
        "profile_prefixes": list(PROFILE_PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "eval_tokens": DECODE_TOKENS,
        "token_generator": "(42 + index * 997) % 248320",
        "gpu_lock": relpath(GPU_LOCK),
        "child_deadline_s": CHILD_TIMEOUT_S,
        "long_child_deadline_s": LONG_TIMEOUT_S,
        "aggregate_deadline_s": AGGREGATE_DEADLINE_S,
        "claims_throughput": False,
        "blocked": blocked,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "freeze.json", {"opt137_control": control, "pins": pins})
    return store_sidecar(run_dir, "preflight.json", payload)


def _summarize_arm(
    quartz: Sequence[float], llama: Sequence[float], *, tokens: int, metric: str
) -> dict[str, Any]:
    n = min(len(quartz), len(llama))
    q = [float(item) for item in quartz[:n]]
    ell = [float(item) for item in llama[:n]]
    q_tok = [tok_s(tokens, item) for item in q]
    l_tok = [tok_s(tokens, item) for item in ell]
    q_tok_ok = [item for item in q_tok if item is not None]
    l_tok_ok = [item for item in l_tok if item is not None]
    ci = (
        paired_log_ratio_ci(q, ell, critical=T_CRIT_DF9_TWO_SIDED)
        if n == BASELINE_PAIRS
        else {"usable": False, "reason": "pair_count_mismatch"}
    )
    q_stats = student_t_interval(q, critical=T_CRIT_DF9_TWO_SIDED) if q else {}
    l_stats = student_t_interval(ell, critical=T_CRIT_DF9_TWO_SIDED) if ell else {}
    q_mean = mean(q)
    l_mean = mean(ell)
    gap_ms = None if q_mean is None or l_mean is None else q_mean - l_mean
    return {
        "metric": metric,
        "n_pairs": n,
        "quartz_mean_ms": q_mean,
        "llama_mean_ms": l_mean,
        "quartz_mean_tok_s": mean(q_tok_ok),
        "llama_mean_tok_s": mean(l_tok_ok),
        "gap_ms": gap_ms,
        "gap_tok_s": None
        if mean(q_tok_ok) is None or mean(l_tok_ok) is None
        else mean(q_tok_ok) - mean(l_tok_ok),
        "paired_log_ratio_ci": ci,
        "quartz_interval": q_stats,
        "llama_interval": l_stats,
        "claim_type": "measured",
        "identity_matched": True,
        "policy": "opt135_paired_log_ratio_df9",
    }


def run_throughput(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("throughput", run_dir, mode, preflight)
    results: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for key, kind, prefix in THROUGHPUT_WORKLOADS:
        timeout = child_timeout_s(kind, prefix)
        quartz_decode: list[float] = []
        llama_decode: list[float] = []
        quartz_complete: list[float] = []
        llama_complete: list[float] = []
        quartz_prefill: list[float] = []
        llama_prefill: list[float] = []
        for warmup in range(BASELINE_WARMUPS):
            for engine in ENGINES:
                args = opt138._engine_args(engine, kind=kind, prefix=prefix)
                label = f"warmup-{key}-{engine}-{warmup}"
                completed = opt136.with_gpu_lock(
                    opt136.native_command(args, tier="acceptance"),
                    timeout_s=timeout,
                )
                results[label] = {
                    "ok": completed.returncode == 0,
                    "returncode": completed.returncode,
                }
                sys.stderr.write(
                    f"OPT-146 throughput {label} rc={completed.returncode}\n"
                )
                sys.stderr.flush()
        for sample in range(BASELINE_PAIRS):
            for engine, decode_b, complete_b, prefill_b in (
                ("quartz", quartz_decode, quartz_complete, quartz_prefill),
                ("llama", llama_decode, llama_complete, llama_prefill),
            ):
                args = opt138._engine_args(engine, kind=kind, prefix=prefix)
                label = f"pair-{key}-{engine}-{sample}"
                completed = opt136.with_gpu_lock(
                    opt136.native_command(args, tier="acceptance"),
                    timeout_s=timeout,
                )
                parsed = parse_result(completed.stdout or "")
                decode_ms = None if parsed is None else parsed.get("decode_only_ms")
                complete_ms = (
                    None if parsed is None else parsed.get("complete_request_ms")
                )
                prefill_ms = None if parsed is None else parsed.get("prefill_ms")
                if kind == "decode":
                    if decode_ms is not None:
                        decode_b.append(float(decode_ms))
                    if complete_ms is not None:
                        complete_b.append(float(complete_ms))
                else:
                    if prefill_ms is not None:
                        prefill_b.append(float(prefill_ms))
                    if complete_ms is not None:
                        complete_b.append(float(complete_ms))
                results[label] = {
                    "ok": completed.returncode == 0,
                    "returncode": completed.returncode,
                    "decode_only_ms": decode_ms,
                    "complete_request_ms": complete_ms,
                    "prefill_ms": prefill_ms,
                    "engine": engine,
                    "kind": kind,
                    "prefix": prefix if prefix is not None else PREFILL_TOKENS,
                    "sample": sample,
                }
                sys.stderr.write(
                    f"OPT-146 throughput {label} rc={completed.returncode} "
                    f"decode={decode_ms} complete={complete_ms} "
                    f"prefill={prefill_ms}\n"
                )
                sys.stderr.flush()
        bucket: dict[str, Any] = {
            "kind": kind,
            "prefix": prefix if prefix is not None else PREFILL_TOKENS,
            "capacity": CAPACITY,
            "eval_tokens": DECODE_TOKENS if kind == "decode" else PREFILL_TOKENS,
            "quartz_decode_only_ms": quartz_decode,
            "llama_decode_only_ms": llama_decode,
            "quartz_complete_request_ms": quartz_complete,
            "llama_complete_request_ms": llama_complete,
            "quartz_prefill_ms": quartz_prefill,
            "llama_prefill_ms": llama_prefill,
            "n_pairs": BASELINE_PAIRS,
            "warmups": BASELINE_WARMUPS,
            "alternating": "quartz_then_llama",
            "opt137_control_arm": "same_production_binary",
        }
        if kind == "decode":
            bucket["decode_only"] = _summarize_arm(
                quartz_decode,
                llama_decode,
                tokens=DECODE_TOKENS,
                metric="decode_only_ms",
            )
            bucket["complete_request"] = _summarize_arm(
                quartz_complete,
                llama_complete,
                tokens=DECODE_TOKENS,
                metric="complete_request_ms",
            )
        else:
            bucket["prefill"] = _summarize_arm(
                quartz_prefill,
                llama_prefill,
                tokens=PREFILL_TOKENS,
                metric="prefill_ms",
            )
            bucket["complete_request"] = _summarize_arm(
                quartz_complete or quartz_prefill,
                llama_complete or llama_prefill,
                tokens=PREFILL_TOKENS,
                metric="complete_request_ms",
            )
        results[key] = bucket
        summaries[key] = {
            metric: bucket.get(metric)
            for metric in ("decode_only", "complete_request", "prefill")
            if bucket.get(metric)
        }
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
        "phase": "throughput",
        "mode": mode,
        "ok": True,
        "results": results,
        "summaries": summaries,
        "hashes": preflight.get("hashes"),
        "opt137_control": preflight.get("opt137_control"),
        "claims_throughput": False,
        "shipping_delta": 0,
        "family_plan": family_plan(mode, "throughput"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "throughput.json", payload)
    return store_sidecar(run_dir, "throughput.json", payload)


def run_capture(run_dir: Path, mode: str) -> dict[str, Any]:
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("capture", run_dir, mode, preflight)
    traces = run_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    captures: dict[str, Any] = {}
    commands: list[str] = []
    jobs: list[tuple[str, str, str, int | None, str]] = []
    for prefix in PROFILE_PREFIXES:
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
            args = opt138._capture_args(engine, kind=kind, arm=arm, prefix=prefix)
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
            sys.stderr.write(f"OPT-146 capture {key} rc={completed.returncode}\n")
            sys.stderr.flush()
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
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


def perturbation_from_twins(
    capture: Mapping[str, Any],
    engine: str,
    kind: str,
    prefix: int,
    rep: int,
) -> dict[str, Any]:
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
        "source": "opt146_unprofiled_twin",
        "acceptable": abs(rel) <= RESIDUAL_WALL_LIMIT,
    }


def run_reconcile(run_dir: Path, mode: str) -> dict[str, Any]:
    capture = load_sidecar(run_dir, "capture.json") or {}
    traces = run_dir / "traces"
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
                path = sqlite_path(traces, engine, kind, prefix, window, rep)
                if not path.is_file():
                    pair[engine] = {"ok": False, "missing": relpath(path)}
                    continue
                audited = opt142.audit_opt138_sqlite(
                    path, engine=engine, kind=kind, prefix=prefix, window=window
                )
                audited["ok"] = True
                twin = perturbation_from_twins(capture, engine, kind, prefix, rep)
                wall = dict(audited["wall"])
                wall["unprofiled_twin"] = twin
                # Decode node captures time three 12-token windows under nsys, so
                # whole-request decode_only_ms is not a window perturbation check.
                # Prefill's native chrono is the same interval as the capture window.
                if (
                    kind == "prefill"
                    and twin.get("usable")
                    and not twin.get("acceptable")
                ):
                    wall["perturbed"] = True
                    wall["prefill_twin_exceeded"] = True
                elif kind == "decode":
                    twin = dict(twin)
                    twin["window_identity"] = "not_whole_request_decode_only"
                    twin["note"] = (
                        "whole-request nsys decode_only_ms includes three capture "
                        "windows plus profiler overhead; OPT-142 residual uses the "
                        "12-token window partition, not this ratio"
                    )
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
            pair.update(opt142._legacy_and_wall(pair))
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
        "task": "OPT-146",
        "status": "valid" if coverage_rows and not quartz_failures else "incomplete",
        "windows": coverage_rows,
    }
    validation = validate_coverage_document(document)
    document["validation"] = validation
    dump_json(EVIDENCE / "coverage.json", document)
    dump_json(EVIDENCE / "per-window.json", {"windows": windows})
    quartz_ok = [row for row in windows if row.get("quartz_within_limit")]
    max_share = None
    shares = [
        row.get("quartz_unresolved_share")
        for row in windows
        if row.get("quartz_unresolved_share") is not None
    ]
    if shares:
        max_share = max(float(item) for item in shares)
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
        "phase": "reconcile",
        "mode": mode,
        "ok": bool(validation.get("ok")) and not quartz_failures,
        "blocked": bool(quartz_failures) or not coverage_rows,
        "coverage_validation": validation,
        "window_count": len(windows),
        "quartz_windows_within_limit": len(quartz_ok),
        "required_quartz_windows": 21,
        "max_unresolved_share": max_share,
        "unresolved_limit": RESIDUAL_WALL_LIMIT,
        "failures": quartz_failures,
        "windows": windows,
        "no_proportional_allocation": True,
        "family_plan": family_plan(mode, "reconcile"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "reconcile.json", payload)
    return store_sidecar(run_dir, "reconcile.json", payload)


def run_families(run_dir: Path, mode: str) -> dict[str, Any]:
    capture = load_sidecar(run_dir, "capture.json") or {}
    captures = dict(capture.get("captures") or {})
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
            family_ms = opt138._family_ms(coverage)
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
        grouped = regroup_fused_families(
            opt138._family_ms(q_cov), opt138._family_ms(l_cov)
        )
        q_wall = q_cov.get("window_ms")
        l_wall = l_cov.get("window_ms")
        q_part = q_cov.get("wall_reconciliation") or {}
        l_part = l_cov.get("wall_reconciliation") or {}
        recon = None
        wall_recon = None
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
            if q_part and l_part:
                wall_recon = whole_wall_gap_from_partitions(q_part, l_part)
                wall_recon["metric_identity"] = recon["metric_identity"]
        return {
            "regrouped": grouped,
            "reconciliation": recon,
            "wall_reconciliation": wall_recon,
            "quartz_window_ms": q_wall,
            "llama_window_ms": l_wall,
        }

    decode_selection = select_top_two_families(decode_input, phase="decode")
    prefill_selection = select_top_two_families(prefill_input, phase="prefill")
    outcomes = experiment_outcomes()
    next_experiments = {
        "decode": next_experiment_spec("decode", decode_selection, outcomes),
        "prefill": next_experiment_spec("prefill", prefill_selection, outcomes),
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
        "phase": "families",
        "mode": mode,
        "ok": True,
        "window_count": len(windows),
        "decode_selection": decode_selection,
        "prefill_selection": prefill_selection,
        "decode_d128_middle": representative("decode", 128, "middle"),
        "decode_d2048_middle": representative("decode", 2048, "middle"),
        "prefill_p4096": representative("prefill", PREFILL_TOKENS, "prefill"),
        "family_gaps": family_gaps,
        "next_experiments": next_experiments,
        "top_two_investigation": {
            "decode": list(decode_selection.get("selected") or [])[:2],
            "prefill": list(prefill_selection.get("selected") or [])[:2],
            "method": "existing_authenticated_opt139_141_143_144_145_counters",
            "new_ncu": False,
        },
        "long_context_follow_up": {
            "included": False,
            "reason": (
                "fresh D8192/D32768 captures deferred unless throughput "
                "exposes a regression versus the OPT-137 kept stack"
            ),
        },
        "family_plan": family_plan(mode, "families"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "family-gaps.json", payload)
    dump_json(EVIDENCE / "next-experiments.json", next_experiments)
    return store_sidecar(run_dir, "families.json", payload)


def run_experiments(run_dir: Path, mode: str) -> dict[str, Any]:
    outcomes = experiment_outcomes()
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
        "phase": "experiments",
        "mode": mode,
        **outcomes,
        "family_plan": family_plan(mode, "experiments"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "experiments.json", payload)
    return store_sidecar(run_dir, "experiments.json", payload)


def independent_checks(
    throughput: Mapping[str, Any],
    families: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    d2048 = (throughput.get("results") or {}).get("d2048") or {}
    quartz = [float(item) for item in (d2048.get("quartz_decode_only_ms") or [])]
    llama = [float(item) for item in (d2048.get("llama_decode_only_ms") or [])]
    stored = (d2048.get("decode_only") or {}).get("quartz_mean_ms")
    recomputed = mean(quartz)
    throughput_ok = (
        recomputed is not None
        and stored is not None
        and abs(float(recomputed) - float(stored)) < 1e-9
        and len(quartz) == BASELINE_PAIRS
        and len(llama) == BASELINE_PAIRS
    )
    family_row = ((families.get("decode_d2048_middle") or {}).get("regrouped")) or {}
    traces = run_dir / "traces"
    decode_sqlite = sqlite_path(traces, "quartz", "decode", 2048, "middle", 0)
    family_ok = False
    reconstructed: dict[str, Any] = {"ok": False}
    if decode_sqlite.is_file():
        tables = parse_nsys_sqlite(decode_sqlite)
        resolved = resolve_capture_window(tables)
        wall = reconcile_window_wall(
            tables,
            window_start_ns=int(resolved["window_start_ns"]),
            window_end_ns=int(resolved["window_end_ns"]),
            window_source=str(resolved.get("source") or ""),
            missing_observation=resolved.get("missing_observation"),
        )
        reconstructed = {
            "ok": True,
            "path": relpath(decode_sqlite),
            "unresolved_share_of_wall": wall.get("unresolved_share_of_wall"),
            "window_ms": wall.get("window_ms"),
            "gpu_union_ms": wall.get("gpu_union_ms"),
        }
        family_ok = True
    return {
        "paired_throughput": {
            "workload": "d2048.decode_only",
            "ok": throughput_ok,
            "recomputed_quartz_mean_ms": recomputed,
            "stored_quartz_mean_ms": stored,
            "n_pairs": len(quartz),
            "claim_type": "measured",
        },
        "family_excess": {
            "window": "d2048.middle.r0",
            "ok": family_ok,
            "regrouped_present": bool(family_row),
            "raw_reconstruction": reconstructed,
            "claim_type": "measured" if family_ok else "incomplete",
        },
    }


def compact_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    rec = dict(out.get("reconcile") or {})
    rec["windows"] = [
        opt142.compact_window_row(row) for row in rec.get("windows") or []
    ]
    out["reconcile"] = rec
    captures: dict[str, Any] = {}
    for key, row in (out.get("captures") or {}).items():
        if not isinstance(row, Mapping):
            continue
        captures[key] = {
            "ok": row.get("ok"),
            "engine": row.get("engine"),
            "kind": row.get("kind"),
            "arm": row.get("arm"),
            "prefix": row.get("prefix"),
            "rep": row.get("rep"),
            "sqlite_by_window": row.get("sqlite_by_window"),
            "parsed": row.get("parsed"),
        }
    out["captures"] = captures
    pre = dict(out.get("preflight") or {})
    ident = dict(pre.get("identity") or {})
    ident.pop("stdout_tail", None)
    pre["identity"] = ident
    pre.pop("make_stderr_tail", None)
    llama = dict(pre.get("llama_build") or {})
    llama.pop("stdout_tail", None)
    llama.pop("stderr_tail", None)
    pre["llama_build"] = llama
    out["preflight"] = pre
    thr = dict(out.get("throughput") or {})
    compact_results: dict[str, Any] = {}
    for key, val in (thr.get("results") or {}).items():
        if str(key).startswith(("warmup-", "pair-")):
            continue
        compact_results[str(key)] = val
    thr["results"] = compact_results
    out["throughput"] = thr
    fam = dict(out.get("families") or {})
    fam.pop("family_gaps", None)
    out["families"] = fam
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(payload: Mapping[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    throughput = payload.get("throughput") or {}
    reconcile = payload.get("reconcile") or {}
    families = payload.get("families") or {}
    experiments = payload.get("experiments") or {}
    answers = payload.get("answers") or {}
    control = preflight.get("opt137_control") or {}
    lines = [
        "# OPT-146 — Measure the combined stack and publish the next remaining-gap decision",
        "",
        "Diagnostics only. `claims_throughput=false`. Shipping delta: **0**.",
        "NLL: N/A. No production selector, kernel, or scheduling change.",
        "Do not read this report as parity, final optimization, or release readiness.",
        "OPT-016 / OPT-056 / CMP-003 remain unchanged.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`.",
        f"Parent `{PARENT}`. Selector `{SELECTOR}`. Capacity `{CAPACITY}`.",
        "",
        "## Frozen identity",
        "",
        f"- pins ok=`{(preflight.get('authenticated_current_pins') or {}).get('ok')}`",
        f"- execution graphs `{SELECTOR}`",
        f"- OPT-137 dense MMA `{(preflight.get('authenticated_current_pins') or {}).get('opt137_dense_mma')}` threshold `{MMA_THRESHOLD}`",
        f"- Q4 decode `{(preflight.get('authenticated_current_pins') or {}).get('q4_decode')}`",
        f"- Quartz binary `{((preflight.get('hashes') or {}).get('quartz_native'))}`",
        f"- llama profile `{((preflight.get('hashes') or {}).get('llama_profile'))}`",
        f"- GPU lock `{preflight.get('gpu_lock')}`",
        f"- decode prefixes `{list(DECODE_PREFIXES)}`",
        f"- P4096 prefill; eval tokens `{DECODE_TOKENS}`; warmups `{BASELINE_WARMUPS}`; pairs `{BASELINE_PAIRS}`",
        "",
        "## OPT-137 control",
        "",
        f"- reconstructable=`{control.get('reconstructable')}`",
        f"- method=`{control.get('method')}`",
        f"- same sitting as final=`{control.get('same_sitting_as_final')}`",
        f"- historical rates used as paired gain=`{control.get('historical_rates_as_paired_gain')}`",
        "",
        str(control.get("reason") or ""),
        "",
        "## Net batch gains",
        "",
        f"- shipping delta **{answers.get('shipping_delta')}**",
        f"- candidate measured delta **{answers.get('candidate_measured_delta')}**",
        f"- batch keeps: `{(answers.get('net_batch') or {}).get('batch_keeps')}`",
        "- OPT-143/144/145 were measured `no_opportunity`; parent retained.",
        "",
        "## OPT-143 / OPT-144 / OPT-145",
        "",
        "| Task | Verdict | Family | Keep | Shipping | Mechanism |",
        "|---|---|---|---|---:|---|",
    ]
    for row in experiments.get("experiments") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task")),
                    str(row.get("verdict")),
                    str(row.get("selected_family")),
                    str(row.get("keep")).lower(),
                    str(row.get("shipping_delta")),
                    str(row.get("supported_mechanism")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Matched Quartz vs llama throughput",
            "",
            "Same-sitting unprofiled runs. OPT-137 control is this production binary.",
        ]
    )
    lines.extend(
        [
            "",
            "| Workload | Metric | Quartz mean | llama mean | Gap | Quartz tok/s | llama tok/s | n |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, _kind, _prefix in THROUGHPUT_WORKLOADS:
        bucket = (throughput.get("results") or {}).get(key) or {}
        for metric in ("decode_only", "complete_request", "prefill"):
            summary = bucket.get(metric)
            if not summary:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        key,
                        metric,
                        _fmt(summary.get("quartz_mean_ms")),
                        _fmt(summary.get("llama_mean_ms")),
                        _fmt(summary.get("gap_ms")),
                        _fmt(summary.get("quartz_mean_tok_s")),
                        _fmt(summary.get("llama_mean_tok_s")),
                        str(summary.get("n_pairs")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Rates use matching metric identities (decode_only vs decode_only, "
            "complete_request vs complete_request, prefill vs prefill). Setup, "
            "graph creation and warmup are outside the denominators.",
            "",
            "## Wall reconciliation (OPT-142 equation, <=5% unresolved)",
            "",
            f"- Quartz windows within limit: `{reconcile.get('quartz_windows_within_limit')}` / `{reconcile.get('required_quartz_windows')}`",
            f"- max unresolved share: `{_fmt(reconcile.get('max_unresolved_share'))}` (limit `{RESIDUAL_WALL_LIMIT}`)",
            f"- coverage ok: `{(reconcile.get('coverage_validation') or {}).get('ok')}`",
            f"- no proportional allocation: `{reconcile.get('no_proportional_allocation')}`",
            "",
            "| Window | Unresolved share | Within 5% | Window perturbed |",
            "|---|---:|---|---|",
        ]
    )
    for row in reconcile.get("windows") or []:
        wall = (row.get("quartz") or {}).get("wall") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("key")),
                    _fmt(row.get("quartz_unresolved_share")),
                    str(row.get("quartz_within_limit")),
                    str(wall.get("perturbed")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Decode whole-request nsys vs unprofiled decode_only_ms is not the",
            "window perturbation check; OPT-142 residual uses the 12-token",
            "partition. Prefill native chrono matches the capture window.",
            "",
            "## Top-two positive-excess families (fresh captures)",
            "",
        ]
    )
    for phase_name, key in (
        ("decode", "decode_selection"),
        ("prefill", "prefill_selection"),
    ):
        selected = list((families.get(key) or {}).get("selected") or [])[:2]
        lines.append(f"### {phase_name}")
        lines.append("")
        if not selected:
            lines.append("No evidenced positive excess.")
            lines.append("")
            continue
        for row in selected:
            lines.append(
                f"- `{row.get('family')}` score_ms=`{_fmt(row.get('score_ms'))}` "
                f"evidenced=`{row.get('evidenced')}`"
            )
        lines.append("")
    next_ex = families.get("next_experiments") or answers.get("next_experiments") or {}
    lines.extend(
        [
            "",
            "## Next-experiment specifications",
            "",
            "At most one spec per phase. `candidate=null` when no source-grounded mechanism exists.",
            "",
        ]
    )
    for phase in ("decode", "prefill"):
        spec = next_ex.get(phase) or {}
        lines.extend(
            [
                f"### {phase}",
                "",
                f"- selected family `{spec.get('selected_family')}`",
                f"- measured excess ms `{_fmt(spec.get('measured_excess_ms'))}`",
                f"- candidate `{spec.get('candidate')}`",
                f"- supported mechanism `{spec.get('supported_mechanism')}`",
                f"- prior experiment `{spec.get('prior_experiment')}` (`{spec.get('prior_verdict')}`)",
                f"- resolving measurement: {spec.get('resolving_measurement')}",
                "",
            ]
        )
    checks = answers.get("independent_checks") or {}
    lines.extend(
        [
            "## Independent reconstruction",
            "",
            f"- paired throughput `{((checks.get('paired_throughput') or {}).get('workload'))}` ok=`{((checks.get('paired_throughput') or {}).get('ok'))}`",
            f"- family/window `{((checks.get('family_excess') or {}).get('window'))}` ok=`{((checks.get('family_excess') or {}).get('ok'))}`",
            "",
            "## Performance evidence checklist",
            "",
            "1. Measurement identity: frozen binary hashes, selectors, capacity 131072, GGUF, llama revision, warmups/samples, decode_only / complete_request / prefill.",
            "2. Coverage: OPT-136 `audit_window_from_tables` plus `performance_evidence.py --validate` on OPT-146 `coverage.json`.",
            "3. Time accounting: OPT-142 disjoint wall partition; CPU overlapping GPU is not additive; no proportional allocation.",
            "4. Contradiction register: sitting hash may differ from OPT-137/138 historical captures; historical rates are not paired gains.",
            "5. Claim types: throughput gaps `measured`; shipping delta `derived` from no-keep; historical OPT-137 tok/s `historical`.",
            "6. Target/guard: OPT-135 `target_guard_v2` statistics used for paired log-ratio CIs; this task does not keep a candidate.",
            "7. Independent verification: one paired D2048 decode_only mean and one D2048 middle wall reconstructed from raw intervals.",
            "8. Reporting: candidate measured delta, shipping delta, quality, and completeness are separate. Diagnostics shipping tok/s change is 0, not N/A, because the batch produced no keep versus the OPT-137 starting stack.",
            "",
            "## Non-claims",
            "",
            "- Not parity.",
            "- Not release readiness.",
            "- Not a production keep.",
            "- OPT-016, OPT-056, and CMP-003 are unchanged.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    throughput = load_sidecar(run_dir, "throughput.json") or {}
    capture = load_sidecar(run_dir, "capture.json") or {}
    reconcile = load_sidecar(run_dir, "reconcile.json") or {}
    families = load_sidecar(run_dir, "families.json") or {}
    experiments = load_sidecar(run_dir, "experiments.json") or run_experiments(
        run_dir, mode
    )
    blocked = bool(preflight.get("blocked")) or not preflight.get("ok")
    coverage_ok = bool((reconcile.get("coverage_validation") or {}).get("ok"))
    throughput_ok = bool(throughput.get("ok"))
    status = (
        "measured" if (not blocked and coverage_ok and throughput_ok) else "incomplete"
    )
    control = preflight.get("opt137_control") or freeze_opt137_control(
        preflight.get("authenticated_current_pins") or {},
        preflight.get("hashes") or {},
    )
    net = net_batch_gains(control, experiments)
    next_experiments = families.get("next_experiments") or {
        "decode": next_experiment_spec(
            "decode", families.get("decode_selection") or {}, experiments
        ),
        "prefill": next_experiment_spec(
            "prefill", families.get("prefill_selection") or {}, experiments
        ),
    }
    checks = independent_checks(throughput, families, run_dir)
    answers = {
        "shipping_delta": 0,
        "candidate_measured_delta": 0,
        "quality": "N/A",
        "nll": "N/A",
        "evidence_complete": status == "measured",
        "parity_claimed": False,
        "release_readiness_claimed": False,
        "opt016_unchanged": True,
        "opt056_unchanged": True,
        "cmp003_unchanged": True,
        "net_batch": net,
        "next_experiments": next_experiments,
        "independent_checks": checks,
        "quartz_llama_gaps": (throughput.get("summaries") or {}),
        "opt137_control": control,
    }
    dump_json(EVIDENCE / "answers.json", answers)
    dump_json(EVIDENCE / "next-experiments.json", next_experiments)
    payload = {
        "schema_version": 1,
        "task": "OPT-146",
        "status": status,
        "mode": mode,
        "measurement_utc": utc_now(),
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "allocated_capacity": CAPACITY,
        "decode_prefixes": list(DECODE_PREFIXES),
        "profile_prefixes": list(PROFILE_PREFIXES),
        "prefill_tokens": PREFILL_TOKENS,
        "windows": list(WINDOWS),
        "preflight": preflight,
        "throughput": throughput,
        "captures": capture.get("captures") or {},
        "reconcile": reconcile,
        "families": families,
        "experiments": experiments,
        "answers": answers,
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": 0,
        "candidate_measured_delta": 0,
        "blocked": blocked,
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    write_report(payload)
    persist_fixture(compact_fixture(payload))
    raw = EVIDENCE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in (
        "preflight.json",
        "throughput.json",
        "capture.json",
        "reconcile.json",
        "families.json",
        "experiments.json",
    ):
        src = sidecar(run_dir, name)
        if src.is_file():
            shutil.copy2(src, raw / name)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if phase == "preflight":
        result = run_preflight(run_dir, mode)
    elif phase == "throughput":
        result = run_throughput(run_dir, mode)
    elif phase == "capture":
        result = run_capture(run_dir, mode)
    elif phase == "reconcile":
        result = run_reconcile(run_dir, mode)
    elif phase == "families":
        result = run_families(run_dir, mode)
    elif phase == "experiments":
        result = run_experiments(run_dir, mode)
    elif phase == "report":
        result = run_report(run_dir, mode)
    else:
        raise BatchReconciliationError(f"unknown phase {phase}")
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
            "task": "OPT-146",
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
    if args.phase in {"preflight", "throughput", "capture"} and result.get("blocked"):
        return 0
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchReconciliationError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
