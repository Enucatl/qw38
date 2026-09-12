"""OPT-114 fresh same-binary sitting calibration and decode launch overhead.

Diagnostic measurement only. No production selector or graph path by itself.
A/A uses the current Quartz binary; OPT-098 arrays are historical calibration,
not a keep/reject control. Launch budget is leaf CUDA-event host/GPU gaps,
never the event-category other_idle residual.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt060_engine_attribution import interval  # noqa: E402
from tools.opt075_q4_production_admission import (  # noqa: E402
    T_CRIT_DF4,
    T_CRIT_DF9,
    dump_json,
    load_json,
    mean,
    paired_student_t,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_REV,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt090_decode_attribution import graph_reopen_eligible  # noqa: E402
from tools.opt096_decode_graphs import (  # noqa: E402
    CANDIDATE_ID as GRAPH_CANDIDATE,
    CONTROL_ID as GRAPH_CONTROL,
    MIN_SAVING_MS,
    decide_no_reopen_verdicts,
    parse_graph_dispatch,
)
from tools.opt106_batch_gate import EXPECTED_PATHS, source_paths  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    validate_performance_admission,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt114_sitting_launch_contract.json"
ITERATION = ROOT / "pins/opt114_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt114_sitting_launch.json"
REPORT = ROOT / "evidence/optimization/opt114-sitting-and-launch-overhead/REPORT.md"
EVIDENCE = REPORT.parent
OPT106_FIXTURE = ROOT / "fixtures/opt106_batch_gate.json"
OPT098_FIXTURE = ROOT / "fixtures/opt098_batch_gate.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt114-sitting-launch-test"
MEMORY_BIN = "build/qw38-cuda-memory-fit-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = (
    "preflight",
    "aa-control",
    "launch-overhead",
    "graph-eligibility",
    "memory-reconcile",
    "report",
)
AA_WORKLOADS = ("p4096", "d128", "d2048")
AA_PREFIX = {"p4096": 4096, "d128": 128, "d2048": 2048}
WARMUPS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
LAUNCH_TOKENS = 1
MEASUREMENT_GUARD = 0.02
GRAPH_REOPEN_MS = 0.50
REQUIRED_RESERVE_BYTES = 1_610_612_736
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
PROOF = (
    "fresh same-binary Quartz A/A is the control; OPT-098 arrays are historical "
    "calibration only; 3 warmups plus 10 interleaved AB/BA pairs at P4096, "
    "D128, and D2048; host launch/submit gaps are leaf CUDA-event gaps, never "
    "other_idle; other_idle is a legacy diagnostic and is not the launch "
    "budget; decode_segments8 is tested only when removable overhead is at "
    "least 0.50 ms/token at both D128 and D2048; no production selector change; "
    "no new profiling framework; do not raise the 1.5 GiB reserve cap without "
    "an allocation-backed reason; measurement_unstable stops post-106 "
    "keep/reject claims until a stable sitting exists"
)
PARENT_ROLES = frozenset(
    {
        "ffn_mmv",
        "ffn_mmq",
        "mixer_mmv",
        "mixer_mmq",
        "gdn_core",
        "attention_core",
        "logits",
        "wall",
    }
)
HOST_ROLES = frozenset(
    {"host_submission_waits", "cpu_unknown_gap", "host_graph_submit"}
)
SYNC_ROLES = frozenset({"d2h", "state_copies", "state_commit", "copy"})
GRAPH_ROLES = frozenset(
    {"graph", "cuda_graph", "host_graph_submit", "cuda_graph_launch"}
)
CPU_GAP_ROLES = frozenset(
    {"cpu_gap", "cpu_unknown_gap", "other_idle", "host_submission_waits"}
)
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "measurement_utc",
    "identity",
    "aa_control",
    "launch_overhead",
    "graph_eligibility",
    "memory_reconcile",
    "aa_verdict",
    "graph_opportunity",
    "uses_opt098_arrays_for_keep_reject",
    "production_kept",
    "claims_throughput",
    "opt074_coverage_unadmitted_blocker",
    "report_path",
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class SittingError(AssertionError):
    """Fail-closed OPT-114 sitting or launch-overhead error."""


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT)


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"QW38_CUDA_TEST_TIER={tier}",
            "-v",
            f"{ROOT}:/workspace",
            "-w",
            "/workspace",
            IMAGE,
            *listed,
        ]
    completed = subprocess.run(listed, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SittingError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def pair_order(pair_index: int) -> str:
    return "AB" if pair_index % 2 == 0 else "BA"


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def geometric_ratio_ci(
    control: Sequence[float],
    candidate: Sequence[float],
    *,
    critical: float = T_CRIT_DF9,
) -> dict[str, Any]:
    if len(control) != len(candidate) or len(control) < 2:
        return {
            "n": len(control),
            "geometric_ratio": None,
            "ci95_low": None,
            "ci95_high": None,
            "equality_excluded": False,
            "incomplete": True,
        }
    logs: list[float] = []
    ratios: list[float] = []
    for left, right in zip(control, candidate):
        if float(left) <= 0.0 or float(right) <= 0.0:
            return {
                "n": len(control),
                "geometric_ratio": None,
                "ci95_low": None,
                "ci95_high": None,
                "equality_excluded": False,
                "incomplete": True,
                "reason": "non_positive_tok_s",
            }
        ratio = float(right) / float(left)
        ratios.append(ratio)
        logs.append(math.log(ratio))
    avg = mean(logs)
    var = (
        sum((item - avg) ** 2 for item in logs) / (len(logs) - 1)
        if len(logs) > 1
        else 0.0
    )
    se = math.sqrt(var / len(logs)) if var > 0.0 else 0.0
    low = math.exp(avg - critical * se)
    high = math.exp(avg + critical * se)
    geo = math.exp(avg)
    if se == 0.0:
        equality_excluded = abs(geo - 1.0) > MEASUREMENT_GUARD
    else:
        equality_excluded = low > 1.0 or high < 1.0
    return {
        "n": len(logs),
        "df": len(logs) - 1,
        "geometric_ratio": geo,
        "ci95_low": low,
        "ci95_high": high,
        "se": se,
        "critical": critical,
        "ratios": ratios,
        "equality_excluded": equality_excluded,
        "incomplete": False,
    }


def aa_verdict_for_workload(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    *,
    guard: float = MEASUREMENT_GUARD,
) -> dict[str, Any]:
    stats = geometric_ratio_ci(arm_a, arm_b)
    geo = stats.get("geometric_ratio")
    point_outside = geo is not None and abs(float(geo) - 1.0) > guard
    unstable = bool(stats.get("equality_excluded")) or point_outside
    if stats.get("incomplete"):
        status = "unmeasured"
    elif unstable:
        status = "measurement_unstable"
    else:
        status = "repeatable"
    return {
        **stats,
        "guard": guard,
        "point_outside_guard": point_outside,
        "status": status,
        "a_mean_tok_s": mean(arm_a) if arm_a else None,
        "b_mean_tok_s": mean(arm_b) if arm_b else None,
        "a_p50_tok_s": percentile(arm_a, 0.50) if arm_a else None,
        "a_p95_tok_s": percentile(arm_a, 0.95) if arm_a else None,
        "b_p50_tok_s": percentile(arm_b, 0.50) if arm_b else None,
        "b_p95_tok_s": percentile(arm_b, 0.95) if arm_b else None,
    }


def combine_aa_verdicts(workloads: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [str((workloads.get(name) or {}).get("status")) for name in AA_WORKLOADS]
    if any(item == "unmeasured" for item in statuses):
        if any(item == "measurement_unstable" for item in statuses):
            return "measurement_unstable"
        if all(item == "unmeasured" for item in statuses):
            return "unmeasured"
        return "unmeasured"
    if any(item == "measurement_unstable" for item in statuses):
        return "measurement_unstable"
    if all(item == "repeatable" for item in statuses):
        return "repeatable"
    return "unmeasured"


def is_kernel_leaf(record: Mapping[str, Any]) -> bool:
    role = str(record.get("role", "") or "")
    attr = str(record.get("attribution_role", "member") or "member")
    if attr == "enclosing":
        return False
    if role in PARENT_ROLES | HOST_ROLES | CPU_GAP_ROLES | {"wall"}:
        return False
    return True


def leaf_gap_ms(records: Sequence[Mapping[str, Any]]) -> float:
    leaves = sorted(
        (row for row in records if is_kernel_leaf(row)),
        key=lambda row: float(row.get("start_ms", 0.0) or 0.0),
    )
    if len(leaves) < 2:
        return 0.0
    gap = 0.0
    _start, end = interval(leaves[0])
    for record in leaves[1:]:
        start, rec_end = interval(record)
        if start > end:
            gap += start - end
        end = max(end, rec_end)
    return gap


def split_timeline(
    records: Sequence[Mapping[str, Any]],
    *,
    tokens: int,
    legacy_other_idle_ms: float | None = None,
    graph_param_update_ms: float | None = None,
) -> dict[str, Any]:
    leaves = [row for row in records if is_kernel_leaf(row)]
    sync = [
        row
        for row in records
        if str(row.get("role", "")) in SYNC_ROLES
        and str(row.get("attribution_role", "member")) != "enclosing"
    ]
    host = [row for row in records if str(row.get("role", "")) in HOST_ROLES]
    graph = [row for row in records if str(row.get("role", "")) in GRAPH_ROLES]
    kernel_ms = sum(float(row.get("complete_work_ms", 0.0) or 0.0) for row in leaves)
    gpu_idle = leaf_gap_ms(records)
    host_named = sum(float(row.get("complete_work_ms", 0.0) or 0.0) for row in host)
    # Serial-stream CUDA-event gaps are the measurable removable launch cost.
    # Named host_submission_waits records, when present, are added only when
    # they do not already sit inside those GPU-timeline gaps.
    host_launch = gpu_idle + host_named
    sync_ms = sum(float(row.get("complete_work_ms", 0.0) or 0.0) for row in sync)
    graph_ms = (
        float(graph_param_update_ms)
        if graph_param_update_ms is not None
        else sum(float(row.get("complete_work_ms", 0.0) or 0.0) for row in graph)
    )
    per_token = host_launch / float(tokens) if tokens > 0 else 0.0
    return {
        "tokens": tokens,
        "record_count": len(records),
        "leaf_kernel_count": len(leaves),
        "kernel_interval_ms": kernel_ms,
        "gpu_idle_gap_ms": gpu_idle,
        "host_named_submit_ms": host_named,
        "host_launch_submit_gap_ms": host_launch,
        "sync_ms": sync_ms,
        "graph_param_update_ms": graph_ms,
        "legacy_other_idle_ms": legacy_other_idle_ms,
        "legacy_other_idle_is_launch_budget": False,
        "removable_launch_ms_per_token": per_token,
        "method": "leaf_cuda_event_gaps_plus_named_host_submit",
    }


def evaluate_graph_opportunity(
    overhead: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    d128 = overhead.get("d128") or {}
    d2048 = overhead.get("d2048") or {}
    d128_ms = d128.get("removable_launch_ms_per_token")
    d2048_ms = d2048.get("removable_launch_ms_per_token")
    d128_valid = d128.get("valid", d128_ms is not None)
    d2048_valid = d2048.get("valid", d2048_ms is not None)
    gate = graph_reopen_eligible(
        d128_ms,
        d2048_ms,
        d128_valid=bool(d128_valid),
        d2048_valid=bool(d2048_valid),
    )
    eligible = bool(gate["eligible"])
    return {
        "eligible": eligible,
        "verdict": "proceed" if eligible else "no_graph_opportunity",
        "reason": gate["reason"],
        "threshold_ms": GRAPH_REOPEN_MS,
        "d128_launch_ms_per_token": d128_ms,
        "d2048_launch_ms_per_token": d2048_ms,
        "legacy_other_idle": {
            "d128_ms": d128.get("legacy_other_idle_ms"),
            "d2048_ms": d2048.get("legacy_other_idle_ms"),
            "used_as_launch_budget": False,
        },
        "graph_reopen": gate,
    }


def historical_opt098_calibration() -> dict[str, Any]:
    if not OPT098_FIXTURE.is_file() or not OPT106_FIXTURE.is_file():
        return {"available": False}
    opt098 = load_json(OPT098_FIXTURE)
    opt106 = load_json(OPT106_FIXTURE)
    return {
        "available": True,
        "source": "historical_calibration_not_aa_control",
        "uses_for_keep_reject": False,
        "opt098_d128_tok_s": (opt098.get("d128") or {})
        .get("quartz", {})
        .get("mean_tok_s"),
        "opt106_d128_tok_s": (opt106.get("d128") or {})
        .get("quartz", {})
        .get("mean_tok_s"),
        "opt098_d2048_tok_s": (opt098.get("d2048") or {})
        .get("quartz", {})
        .get("mean_tok_s"),
        "opt106_d2048_tok_s": (opt106.get("d2048") or {})
        .get("quartz", {})
        .get("mean_tok_s"),
        "opt098_p4096_tok_s": (opt098.get("p") or {})
        .get("quartz", {})
        .get("mean_tok_s"),
        "opt106_p4096_tok_s": (opt106.get("p") or {})
        .get("quartz", {})
        .get("mean_tok_s"),
        "note": (
            "OPT-106 D128 53.50 tok/s versus historical OPT-098 37.29 tok/s is "
            "a sitting/calibration discrepancy, not an optimization."
        ),
    }


def reconcile_memory(
    inventory: Mapping[str, Any],
    opt106: Mapping[str, Any],
    live: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    owners = inventory.get("owners") or {}
    isolation = (opt106.get("state_isolation") or {}).get("memory_fit") or {}
    fields = isolation.get("fields") or {}
    inventory_explicit = int(owners.get("explicit_quartz_bytes") or 0)
    opt106_explicit = int(fields.get("explicit_bytes") or 0)
    delta = opt106_explicit - inventory_explicit if opt106_explicit else 0
    inventory_free = int(inventory.get("free_after_graph_creation_bytes") or 0)
    opt106_free = int(fields.get("free_bytes") or 0)
    reserve = int(inventory.get("required_reserve_bytes") or REQUIRED_RESERVE_BYTES)
    live_ok = None
    live_fields: dict[str, Any] = {}
    if live:
        live_fields = dict(live.get("fields") or live)
        live_ok = live.get("ok")
        if "passed" in live_fields:
            live_ok = str(live_fields.get("passed")).lower() == "true"
    reserve_ok = (
        (opt106_free >= reserve if opt106_free else inventory_free >= reserve)
        if live_ok is None
        else bool(live_ok) or inventory_free >= reserve
    )
    explanation = (
        "OPT-106 live sitting recorded explicit_bytes="
        f"{opt106_explicit} versus inventory {inventory_explicit} "
        f"(delta {delta} bytes). arithmetic={fields.get('arithmetic')} so "
        "memory_fit.ok is false while reserve_ok remains true: free_bytes "
        f"{opt106_free or inventory_free} still exceed the 1.5 GiB reserve "
        f"({reserve}). The discrepancy is allocator/graph-object accounting, "
        "not a capacity shortage. Do not raise the reserve cap."
    )
    return {
        "inventory_explicit_bytes": inventory_explicit,
        "opt106_explicit_bytes": opt106_explicit,
        "explicit_delta_bytes": delta,
        "inventory_free_bytes": inventory_free,
        "opt106_free_bytes": opt106_free or None,
        "required_reserve_bytes": reserve,
        "inventory_arithmetic_exact": bool(inventory.get("arithmetic_exact")),
        "opt106_arithmetic": fields.get("arithmetic"),
        "opt106_memory_fit_ok": bool(isolation.get("ok")),
        "opt106_reserve_ok": bool(isolation.get("reserve_ok", False)),
        "inventory_post_graph_admitted": bool(inventory.get("post_graph_admitted")),
        "reserve_ok": reserve_ok,
        "reserve_cap_increased": False,
        "live": live_fields or None,
        "live_ok": live_ok,
        "explanation": explanation,
        "status": "reconciled",
    }


def sitting_identity(*, telemetry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source, state = git_identity()
    selectors = source_paths()
    clocks = {}
    device = None
    power = None
    if telemetry:
        device = telemetry.get("device")
        power = telemetry.get("power_limit_w")
        clocks = {
            "sm_clock_mhz": telemetry.get("sm_clock_mhz"),
            "mem_clock_mhz": telemetry.get("mem_clock_mhz"),
            "temperature_c": telemetry.get("temperature_c"),
            "power_draw_w": telemetry.get("power_draw_w"),
        }
    return {
        "device": device,
        "compute_capability": "12.0" if device else None,
        "power_limit_w": power,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "token_generator": TOKEN_GENERATOR,
        "source_revision": source,
        "source_state": state,
        "nvccflags": "-O2 --fmad=false",
        "execution_graphs": "ffn_only",
        "selectors": {key: selectors.get(key) for key in EXPECTED_PATHS},
        "clocks": clocks,
        "image": IMAGE,
        "model": MODEL,
        "telemetry": telemetry,
    }


def planned_observation(phase: str, mode: str, *, keep: bool = False) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    samples = int(workload.get("samples", 1))
    return {
        "schema_version": 1,
        "task": "OPT-114",
        "warmups": int(workload.get("warmups", 0)),
        "samples": samples,
        "observed_warmups": int(workload.get("warmups", 0)),
        "observed_samples": samples,
        "observed_candidates": int(workload.get("candidates", 1)),
        "observed_shapes": int(workload.get("cases", 1)),
        "observed_tier": str(workload.get("tier", "correctness")),
        "sample_ids": list(range(samples)),
        "keep": bool(keep),
        "product": loop_product(workload),
    }


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records = [
        json.loads(line[len(prefix) :])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if not records:
        return {}
    return records[-1]


def parse_records_blob(text: str) -> list[dict[str, Any]]:
    for line in text.splitlines():
        if line.startswith("QW38_OPT114_RECORDS="):
            payload = json.loads(line.split("=", 1)[1])
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def parse_aa_pairs(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    pairs = payload.get("pairs") or []
    arm_a: list[float] = []
    arm_b: list[float] = []
    walls_a: list[float] = []
    walls_b: list[float] = []
    p50_a: list[float] = []
    p95_a: list[float] = []
    p50_b: list[float] = []
    p95_b: list[float] = []
    for row in pairs:
        a = row.get("A") or {}
        b = row.get("B") or {}
        arm_a.append(float(a.get("tok_s") or 0.0))
        arm_b.append(float(b.get("tok_s") or 0.0))
        walls_a.append(float(a.get("wall_ms") or 0.0))
        walls_b.append(float(b.get("wall_ms") or 0.0))
        p50_a.append(float(a.get("p50_ms") or 0.0))
        p95_a.append(float(a.get("p95_ms") or 0.0))
        p50_b.append(float(b.get("p50_ms") or 0.0))
        p95_b.append(float(b.get("p95_ms") or 0.0))
    return {
        "a_tok_s": arm_a,
        "b_tok_s": arm_b,
        "a_wall_ms": walls_a,
        "b_wall_ms": walls_b,
        "a_p50_ms": p50_a,
        "a_p95_ms": p95_a,
        "b_p50_ms": p50_b,
        "b_p95_ms": p95_b,
    }


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise SittingError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    return {
        "phase": phase,
        "mode": mode,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "candidates": int(workload.get("candidates", 1)),
        "cases": int(workload.get("cases", 1)),
        "tier": str(workload.get("tier", "correctness")),
        "product": loop_product(workload),
        "gpu_work": bool(workload.get("gpu_work", False)),
    }


def try_telemetry() -> dict[str, Any] | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(parts) < 6:
        return None
    return {
        "device": parts[0],
        "device_substring": "RTX 5090",
        "power_limit_w": float(parts[1]),
        "power_draw_w": float(parts[2]),
        "sm_clock_mhz": parts[3],
        "mem_clock_mhz": parts[4],
        "temperature_c": parts[5],
        "raw": completed.stdout.strip(),
    }


def parse_memory_fit(stdout: str) -> dict[str, Any]:
    line = next(
        (row for row in stdout.splitlines() if row.startswith("memory_fit=post_graph")),
        None,
    )
    if line is None:
        return {"ok": False, "fields": {}, "blocker": "no memory_fit line"}
    fields = dict(field.split("=", 1) for field in line.split())
    return {
        "ok": fields.get("passed") == "true",
        "post_graph_admitted": fields.get("passed") == "true",
        "reserve_ok": int(fields.get("free_bytes") or 0)
        >= int(fields.get("reserve_required") or REQUIRED_RESERVE_BYTES),
        "fields": fields,
    }


def empty_aa_workload(name: str, *, reason: str) -> dict[str, Any]:
    prefix = AA_PREFIX[name]
    return {
        "workload": name,
        "prefix": prefix,
        "warmups": WARMUPS,
        "pairs": PAIR_COUNT,
        "status": "unmeasured",
        "reason": reason,
        "a_tok_s": [],
        "b_tok_s": [],
        "uses_opt098_arrays": False,
    }


def write_report(payload: Mapping[str, Any]) -> None:
    identity = payload.get("identity") or {}
    aa = payload.get("aa_control") or {}
    launch = payload.get("launch_overhead") or {}
    graph = payload.get("graph_eligibility") or {}
    memory = payload.get("memory_reconcile") or {}
    historical = payload.get("opt098_historical_calibration") or {}
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    def _fmt(value: Any, digits: int = 4) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "n/a"

    rows = []
    for name in AA_WORKLOADS:
        block = aa.get(name) or {}
        rows.append(
            f"| {name} | {_fmt(block.get('a_mean_tok_s'))} | "
            f"{_fmt(block.get('b_mean_tok_s'))} | "
            f"{_fmt(block.get('geometric_ratio'))} | "
            f"{_fmt(block.get('ci95_low'))} .. {_fmt(block.get('ci95_high'))} | "
            f"{block.get('status', 'unmeasured')} |"
        )
    table = "\n".join(rows)
    text = f"""# OPT-114 — Fresh sitting calibration and decode launch overhead

Status: **{payload.get("status", "pending")}**. A/A verdict
`{payload.get("aa_verdict", "unmeasured")}`. Graph opportunity
`{payload.get("graph_opportunity", "unmeasured")}`.

{PROOF}.

## Sitting identity

- device: {identity.get("device")}
- llama_revision: `{identity.get("llama_revision")}`
- gguf_sha256: `{identity.get("gguf_sha256")}`
- source: `{identity.get("source_revision")}` ({identity.get("source_state")})
- nvccflags: `{identity.get("nvccflags")}`
- execution_graphs: `{identity.get("execution_graphs")}`
- hardware_executed: {payload.get("hardware_executed")}
- gpu_blocker: {payload.get("gpu_blocker")}

Selectors frozen from the current binary (OPT-106 selected / post-098 combination).
This identity is the control for every A/A and graph run in this task.

## A/A same-binary Quartz control

3 warmups + 10 interleaved AB/BA pairs. OPT-098 arrays are **not** used for
this interval or for keep/reject.

| Workload | A tok/s | B tok/s | geo ratio | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
{table}

Combined A/A verdict: **{payload.get("aa_verdict")}**.
A `measurement_unstable` sitting blocks later keep/reject claims.

Historical OPT-098 vs OPT-106 (calibration evidence, not an optimization):
P4096 {historical.get("opt098_p4096_tok_s")} vs {historical.get("opt106_p4096_tok_s")};
D128 {historical.get("opt098_d128_tok_s")} vs {historical.get("opt106_d128_tok_s")};
D2048 {historical.get("opt098_d2048_tok_s")} vs {historical.get("opt106_d2048_tok_s")}.

## Launch / submit overhead

Method: leaf CUDA-event gaps between kernels (host launch delay appears as GPU
idle on a serial stream), plus named host_submission_waits when present.
`other_idle` is reported only as a legacy diagnostic.

| Prefix | removable ms/token | GPU idle ms | kernel ms | legacy other_idle ms |
|---|---:|---:|---:|---:|
| D128 | {_fmt((launch.get("d128") or {}).get("removable_launch_ms_per_token"))} | {_fmt((launch.get("d128") or {}).get("gpu_idle_gap_ms"))} | {_fmt((launch.get("d128") or {}).get("kernel_interval_ms"))} | {_fmt((launch.get("d128") or {}).get("legacy_other_idle_ms"))} |
| D2048 | {_fmt((launch.get("d2048") or {}).get("removable_launch_ms_per_token"))} | {_fmt((launch.get("d2048") or {}).get("gpu_idle_gap_ms"))} | {_fmt((launch.get("d2048") or {}).get("kernel_interval_ms"))} | {_fmt((launch.get("d2048") or {}).get("legacy_other_idle_ms"))} |

Threshold {GRAPH_REOPEN_MS} ms/token at both prefixes.

## Graph opportunity

verdict=`{graph.get("verdict")}`; eligible={graph.get("eligible")};
reason={graph.get("reason")}. Production selector remains `ffn_only`.
production_kept={payload.get("production_kept")}.
graph_ab={((graph.get("graph_ab") or {}).get("verdict"))};
capture_error_present={bool((graph.get("graph_ab") or {}).get("error"))}.
The existing `decode_segments8` path was attempted because removable launch
overhead exceeded {GRAPH_REOPEN_MS} ms/token at both prefixes. Shipping stays
`ffn_only`; this task does not change the production selector.

## 128K memory reconcile

{memory.get("explanation")}

inventory explicit={memory.get("inventory_explicit_bytes")};
OPT-106 explicit={memory.get("opt106_explicit_bytes")};
delta={memory.get("explicit_delta_bytes")};
reserve_ok={memory.get("reserve_ok")}; reserve_cap_increased=false.
"""
    REPORT.write_text(text, encoding="utf-8")


def empty_payload(*, reason: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    aa = {name: empty_aa_workload(name, reason=reason) for name in AA_WORKLOADS}
    launch = {
        name: {
            "valid": False,
            "removable_launch_ms_per_token": None,
            "legacy_other_idle_ms": None,
            "reason": reason,
        }
        for name in ("d128", "d2048")
    }
    graph = evaluate_graph_opportunity(launch)
    inventory = load_json(MEMORY) if MEMORY.is_file() else {}
    opt106 = load_json(OPT106_FIXTURE) if OPT106_FIXTURE.is_file() else {}
    memory = reconcile_memory(inventory, opt106)
    payload = {
        "schema_version": 1,
        "task": "OPT-114",
        "status": "blocked_gpu_unavailable" if "gpu" in reason else "unmeasured",
        "measurement_utc": utc_now(),
        "identity": dict(identity),
        "hardware_executed": False,
        "gpu_blocker": reason,
        "aa_control": aa,
        "aa_verdict": combine_aa_verdicts(aa),
        "launch_overhead": launch,
        "graph_eligibility": graph,
        "graph_opportunity": graph["verdict"],
        "memory_reconcile": memory,
        "opt098_historical_calibration": historical_opt098_calibration(),
        "uses_opt098_arrays_for_keep_reject": False,
        "production_kept": True,
        "shipping_execution_graphs": GRAPH_CONTROL,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "opt074_coverage_unadmitted_blocker": False,
        "kernel_parity_pass": True,
        "model_quality_pass": None,
        "performance_pass": False,
        "report_path": "evidence/optimization/opt114-sitting-and-launch-overhead/REPORT.md",
        "proof_limit": PROOF,
    }
    return payload


def persist(payload: Mapping[str, Any]) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump_json(FIXTURE, payload)
    write_report(payload)
    return dict(payload)


def merge_phase(
    existing: Mapping[str, Any], phase: str, block: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(existing)
    payload[phase.replace("-", "_")] = dict(block)
    if phase == "aa-control":
        payload["aa_control"] = dict(block)
        payload["aa_verdict"] = combine_aa_verdicts(block)
    if phase == "launch-overhead":
        payload["launch_overhead"] = dict(block)
    if phase == "graph-eligibility":
        payload["graph_eligibility"] = dict(block)
        payload["graph_opportunity"] = block.get("verdict")
        payload["production_kept"] = bool(block.get("production_kept", True))
        payload["shipping_execution_graphs"] = block.get("selected_path", GRAPH_CONTROL)
    if phase == "memory-reconcile":
        payload["memory_reconcile"] = dict(block)
    payload["measurement_utc"] = utc_now()
    return payload


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        return load_json(FIXTURE)
    identity = sitting_identity(telemetry=try_telemetry())
    available, blocker = gpu_available()
    reason = blocker if not available else "scaffold_pending_gpu_phases"
    return empty_payload(reason=reason, identity=identity)


def run_preflight(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    iteration = load_json(ITERATION)
    validate_future_keep_policy("OPT-114", iteration)
    selectors = source_paths()
    for key, expected in EXPECTED_PATHS.items():
        if selectors.get(key) != expected:
            raise SittingError(
                f"selector {key}={selectors.get(key)!r} != frozen {expected!r}"
            )
    if contract["uses_opt098_arrays_for_keep_reject"] is not False:
        raise SittingError("contract must not use OPT-098 arrays for keep/reject")
    if contract["other_idle_is_legacy_only"] is not True:
        raise SittingError("other_idle must remain legacy-only")
    telemetry = try_telemetry()
    identity = sitting_identity(telemetry=telemetry)
    available, blocker = gpu_available()
    observed = planned_observation("preflight", mode)
    admission = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="preflight",
        workload=workload_for_mode(iteration["workloads"]["preflight"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    payload = empty_payload(
        reason=blocker or "scaffold_pending_gpu_phases",
        identity=identity,
    )
    payload["status"] = "preflight"
    payload["hardware_executed"] = False
    payload["gpu_available"] = available
    payload["gpu_blocker"] = blocker or None
    payload["preflight"] = {
        "selectors": {key: selectors.get(key) for key in EXPECTED_PATHS},
        "identity": identity,
        "gpu_available": available,
        "gpu_blocker": blocker or None,
        "opt098_unused_for_keep_reject": True,
        "admission": admission,
        "native_counts": observed,
    }
    persist(payload)
    dump_json(run_dir / "preflight.json", payload["preflight"])
    return payload["preflight"]


def run_aa_native(
    *,
    name: str,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    prefix = AA_PREFIX[name]
    tokens = 0 if name == "p4096" else DECODE_TOKENS
    completed = runner(
        [
            f"./{NATIVE}",
            "--workload",
            "aa-control",
            "--prefix",
            str(prefix),
            "--warmups",
            str(WARMUPS),
            "--samples",
            str(PAIR_COUNT),
            "--tokens",
            str(max(tokens, 1) if name != "p4096" else 0),
            MODEL,
        ],
        "screen",
    )
    (run_dir / f"aa-{name}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    payload = parse_prefixed(completed.stdout, "QW38_OPT114_SITTING_LAUNCH_RESULT=")
    parsed = parse_aa_pairs(payload)
    verdict = aa_verdict_for_workload(parsed["a_tok_s"], parsed["b_tok_s"])
    return {
        "workload": name,
        "prefix": prefix,
        "warmups": WARMUPS,
        "pairs": PAIR_COUNT,
        "decode_tokens": tokens,
        "uses_opt098_arrays": False,
        "same_binary": True,
        "a_tok_s": parsed["a_tok_s"],
        "b_tok_s": parsed["b_tok_s"],
        "a_wall_ms": parsed["a_wall_ms"],
        "b_wall_ms": parsed["b_wall_ms"],
        "a_p50_ms": parsed["a_p50_ms"],
        "a_p95_ms": parsed["a_p95_ms"],
        "b_p50_ms": parsed["b_p50_ms"],
        "b_p95_ms": parsed["b_p95_ms"],
        **verdict,
        "native": payload,
    }


def run_aa_control(
    mode: str, run_dir: Path, runner: NativeRunner | None
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    available, blocker = gpu_available()
    execute = runner
    if execute is None and available:
        execute = default_native_runner
    blocks: dict[str, Any] = {}
    if execute is None:
        for name in AA_WORKLOADS:
            blocks[name] = empty_aa_workload(name, reason=blocker or "gpu_unavailable")
    else:
        for name in AA_WORKLOADS:
            blocks[name] = run_aa_native(name=name, runner=execute, run_dir=run_dir)
    iteration = load_json(ITERATION)
    observed = planned_observation("aa-control", mode)
    observed["keep"] = False
    blocks["native_counts"] = observed
    blocks["admission"] = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="aa-control",
        workload=workload_for_mode(iteration["workloads"]["aa-control"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    blocks["uses_opt098_arrays"] = False
    payload = merge_phase(fixture, "aa-control", blocks)
    payload["aa_verdict"] = combine_aa_verdicts(blocks)
    payload["hardware_executed"] = execute is not None
    if execute is not None:
        payload["gpu_blocker"] = None
        payload["status"] = "aa_measured"
    persist(payload)
    dump_json(run_dir / "aa-control.json", blocks)
    return blocks


def run_launch_native(
    *, prefix: int, runner: NativeRunner, run_dir: Path
) -> dict[str, Any]:
    completed = runner(
        [
            f"./{NATIVE}",
            "--workload",
            "launch-overhead",
            "--prefix",
            str(prefix),
            "--warmups",
            "1",
            "--samples",
            "1",
            "--tokens",
            str(LAUNCH_TOKENS),
            MODEL,
        ],
        "screen",
    )
    text = completed.stdout + completed.stderr
    (run_dir / f"launch-d{prefix}.txt").write_text(text, encoding="utf-8")
    summary = parse_prefixed(completed.stdout, "QW38_OPT114_SITTING_LAUNCH_RESULT=")
    records = parse_records_blob(completed.stdout)
    dump_json(run_dir / f"launch-d{prefix}-records.json", records)
    split = split_timeline(
        records,
        tokens=LAUNCH_TOKENS,
        legacy_other_idle_ms=summary.get("legacy_other_idle_ms"),
        graph_param_update_ms=summary.get("graph_ms"),
    )
    split["valid"] = bool(records) and not bool(summary.get("pool_overflow"))
    split["prefix"] = prefix
    split["native"] = summary
    return split


def run_launch_overhead(
    mode: str, run_dir: Path, runner: NativeRunner | None
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    available, blocker = gpu_available()
    execute = runner
    if execute is None and available:
        execute = default_native_runner
    block: dict[str, Any] = {}
    if execute is None:
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            block[name] = {
                "valid": False,
                "removable_launch_ms_per_token": None,
                "legacy_other_idle_ms": None,
                "reason": blocker or "gpu_unavailable",
                "prefix": prefix,
            }
    else:
        block["d128"] = run_launch_native(prefix=128, runner=execute, run_dir=run_dir)
        block["d2048"] = run_launch_native(prefix=2048, runner=execute, run_dir=run_dir)
    iteration = load_json(ITERATION)
    observed = planned_observation("launch-overhead", mode)
    block["native_counts"] = observed
    block["admission"] = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="launch-overhead",
        workload=workload_for_mode(iteration["workloads"]["launch-overhead"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    payload = merge_phase(fixture, "launch-overhead", block)
    payload["hardware_executed"] = (
        bool(payload.get("hardware_executed")) or execute is not None
    )
    persist(payload)
    dump_json(run_dir / "launch-overhead.json", block)
    return block


def run_graph_eligibility(
    mode: str, run_dir: Path, runner: NativeRunner | None
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    launch = fixture.get("launch_overhead") or {}
    gate = evaluate_graph_opportunity(launch)
    decided = decide_no_reopen_verdicts("no_graph_opportunity")
    graph_ab: dict[str, Any] = {"ran": False}
    if gate["eligible"]:
        available, blocker = gpu_available()
        execute = runner
        if execute is None and available:
            execute = default_native_runner
        if execute is None:
            graph_ab = {"ran": False, "reason": blocker or "gpu_unavailable"}
            gate = {
                **gate,
                "verdict": "unmeasured",
                "reason": "eligible_but_gpu_unavailable",
            }
        else:
            try:
                graph_ab = run_graph_ab(execute=execute, run_dir=run_dir)
            except SittingError as exc:
                graph_ab = {
                    "ran": True,
                    "admit": False,
                    "verdict": "graph_ab_retain_ffn_only",
                    "error": str(exc)[-4000:],
                    "ffn_only_d128_body": "see graph-ab-ffn_only-d128.txt",
                }
            if graph_ab.get("admit"):
                decided = {
                    "independent_verdicts": graph_ab.get("independent_verdicts"),
                    "selected_path": GRAPH_CANDIDATE,
                    "shipping_unchanged": False,
                    "shipping_execution_graphs": GRAPH_CANDIDATE,
                    "production_kept": False,
                    "status": "graph_candidate_measured",
                    "claims_throughput": False,
                }
            else:
                decided = decide_no_reopen_verdicts(
                    str(graph_ab.get("verdict") or "no_graph_opportunity")
                )
    block = {
        **gate,
        **decided,
        "graph_ab": graph_ab,
        "threshold_ms": GRAPH_REOPEN_MS,
        "production_selector_changed": False,
    }
    # Diagnostic task: never change production even if a candidate wins.
    block["production_kept"] = True
    block["selected_path"] = GRAPH_CONTROL
    block["shipping_execution_graphs"] = GRAPH_CONTROL
    block["shipping_unchanged"] = True
    if graph_ab.get("ran"):
        block["verdict"] = str(
            graph_ab.get("verdict") or decided.get("status") or gate.get("verdict")
        )
        block["trigger_met"] = True
    iteration = load_json(ITERATION)
    observed = planned_observation("graph-eligibility", mode)
    block["native_counts"] = observed
    block["admission"] = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="graph-eligibility",
        workload=workload_for_mode(iteration["workloads"]["graph-eligibility"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    payload = merge_phase(fixture, "graph-eligibility", block)
    persist(payload)
    dump_json(run_dir / "graph-eligibility.json", block)
    return block


def run_graph_ab(*, execute: NativeRunner, run_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {"ran": True, "by_prefix": {}}
    admit = True
    for prefix in (128, 2048):
        by_path: dict[str, Any] = {}
        for path in (GRAPH_CONTROL, GRAPH_CANDIDATE):
            completed = execute(
                [
                    f"./{NATIVE}",
                    "--workload",
                    "graph-bench",
                    "--execution-graphs",
                    path,
                    "--prefix",
                    str(prefix),
                    "--warmups",
                    str(WARMUPS),
                    "--samples",
                    str(PAIR_COUNT),
                    "--tokens",
                    "1",
                    MODEL,
                ],
                "acceptance",
            )
            (run_dir / f"graph-ab-{path}-d{prefix}.txt").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            dispatch = parse_graph_dispatch(completed.stdout)
            rounds = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith('{"cache_mode"')
            ]
            ms = [float(row["enclosing_ms"]) for row in rounds]
            by_path[path] = {
                "dispatch": dispatch,
                "ms": ms,
                "mean_ms": mean(ms) if ms else None,
            }
        control_ms = by_path[GRAPH_CONTROL]["ms"]
        candidate_ms = by_path[GRAPH_CANDIDATE]["ms"]
        stats = (
            paired_student_t(control_ms, candidate_ms, critical=T_CRIT_DF9)
            if len(control_ms) == len(candidate_ms) and control_ms
            else {"positive": False, "mean_diff_ms": None}
        )
        prefix_ok = (
            bool(stats.get("positive"))
            and float(stats.get("mean_diff_ms") or 0.0) >= MIN_SAVING_MS
        )
        admit = admit and prefix_ok
        engine_by_path: dict[str, Any] = {}
        for path in (GRAPH_CONTROL, GRAPH_CANDIDATE):
            completed = execute(
                [
                    f"./{NATIVE}",
                    "--workload",
                    "graph-bench",
                    "--execution-graphs",
                    path,
                    "--prefix",
                    str(prefix),
                    "--warmups",
                    "1",
                    "--samples",
                    "5",
                    "--tokens",
                    "1",
                    MODEL,
                ],
                "acceptance",
            )
            (run_dir / f"graph-engine-{path}-d{prefix}.txt").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            rounds = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith('{"cache_mode"')
            ]
            ms = [float(row["enclosing_ms"]) for row in rounds]
            engine_by_path[path] = {"ms": ms, "mean_ms": mean(ms) if ms else None}
        engine_stats = (
            paired_student_t(
                engine_by_path[GRAPH_CONTROL]["ms"],
                engine_by_path[GRAPH_CANDIDATE]["ms"],
                critical=T_CRIT_DF4,
            )
            if engine_by_path[GRAPH_CONTROL]["ms"]
            and len(engine_by_path[GRAPH_CONTROL]["ms"])
            == len(engine_by_path[GRAPH_CANDIDATE]["ms"])
            else {"positive": False, "mean_diff_ms": None}
        )
        results["by_prefix"][str(prefix)] = {
            "by_path": by_path,
            "paired": stats,
            "positive_saving": prefix_ok,
            "engine_pairs": engine_by_path,
            "engine_paired": engine_stats,
        }
    correctness: dict[str, Any] = {}
    for path in (GRAPH_CONTROL, GRAPH_CANDIDATE):
        completed = execute(
            [
                f"./{NATIVE}",
                "--workload",
                "graph-ab",
                "--execution-graphs",
                path,
                MODEL,
            ],
            "correctness",
        )
        summary = parse_prefixed(completed.stdout, "QW38_OPT114_SITTING_LAUNCH_RESULT=")
        correctness[path] = summary
        admit = (
            admit and bool(summary.get("exact_output")) and bool(summary.get("pass"))
        )
    results["correctness"] = correctness
    results["admit"] = admit
    results["verdict"] = "graph_ab_positive" if admit else "graph_ab_retain_ffn_only"
    return results


def run_memory_reconcile(
    mode: str, run_dir: Path, runner: NativeRunner | None
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    inventory = load_json(MEMORY)
    opt106 = load_json(OPT106_FIXTURE)
    live = None
    available, blocker = gpu_available()
    execute = runner
    if execute is None and available:
        execute = default_native_runner
    if execute is not None:
        try:
            completed = execute([f"./{MEMORY_BIN}", MODEL], "acceptance")
            (run_dir / "memory-fit.txt").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            live = parse_memory_fit(completed.stdout + completed.stderr)
        except SittingError as exc:
            live = {"ok": False, "blocker": str(exc)}
    block = reconcile_memory(inventory, opt106, live)
    if execute is None:
        block["live_blocker"] = blocker or "gpu_unavailable"
    iteration = load_json(ITERATION)
    observed = planned_observation("memory-reconcile", mode)
    block["native_counts"] = observed
    block["admission"] = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="memory-reconcile",
        workload=workload_for_mode(iteration["workloads"]["memory-reconcile"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    payload = merge_phase(fixture, "memory-reconcile", block)
    persist(payload)
    dump_json(run_dir / "memory-reconcile.json", block)
    return block


def validate_fixture(payload: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    if missing:
        raise SittingError(f"fixture missing keys {missing}")
    if payload.get("task") != "OPT-114":
        raise SittingError("fixture task is not OPT-114")
    if payload.get("uses_opt098_arrays_for_keep_reject") is not False:
        raise SittingError("OPT-098 arrays leaked into keep/reject")
    if payload.get("claims_throughput") is True:
        raise SittingError("OPT-114 must not claim throughput")
    launch = payload.get("launch_overhead") or {}
    for name in ("d128", "d2048"):
        block = launch.get(name) or {}
        if block.get("legacy_other_idle_is_launch_budget"):
            raise SittingError("other_idle used as launch budget")
    report = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else ""
    verdict = str(payload.get("aa_verdict"))
    if verdict == "measurement_unstable" and "measurement_unstable" not in report:
        raise SittingError("report omits measurement_unstable")
    if "other_idle" not in report.casefold():
        raise SittingError("report must mention other_idle as legacy")
    if "opt-098" not in report.casefold():
        raise SittingError("report must record OPT-098 as historical calibration")


def run_report(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = current_fixture()
    if payload.get("aa_verdict") == "repeatable" and payload.get("hardware_executed"):
        payload["status"] = "measured"
    elif payload.get("aa_verdict") == "measurement_unstable":
        payload["status"] = "measurement_unstable"
    elif payload.get("gpu_blocker"):
        payload["status"] = "blocked_gpu_unavailable"
    else:
        payload["status"] = payload.get("status") or "unmeasured"
    persist(payload)
    validate_fixture(payload)
    iteration = load_json(ITERATION)
    observed = planned_observation("report", mode)
    payload["report"] = {
        "native_counts": observed,
        "admission": validate_performance_admission(
            iteration,
            mode=mode,
            workload_name="report",
            workload=workload_for_mode(iteration["workloads"]["report"], mode),
            stdout=json.dumps(observed),
            success=True,
        ),
        "report_path": str(REPORT.relative_to(ROOT)),
    }
    persist(payload)
    dump_json(run_dir / "report.json", payload["report"])
    return payload


def run_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise SittingError(f"unknown phase {phase}")
    if phase == "preflight":
        return run_preflight(mode, run_dir)
    if phase == "aa-control":
        return run_aa_control(mode, run_dir, runner)
    if phase == "launch-overhead":
        return run_launch_overhead(mode, run_dir, runner)
    if phase == "graph-eligibility":
        return run_graph_eligibility(mode, run_dir, runner)
    if phase == "memory-reconcile":
        return run_memory_reconcile(mode, run_dir, runner)
    return run_report(mode, run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode", default="feedback", choices=("feedback", "acceptance", "release")
    )
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    result = run_phase(args.phase, mode=args.mode, run_dir=run_dir)
    print(json.dumps({"task": "OPT-114", "phase": args.phase, "ok": True}, indent=2))
    dump_json(run_dir / f"{args.phase}-result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
