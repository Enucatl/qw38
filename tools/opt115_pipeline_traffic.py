"""OPT-115 complete-request memory traffic audit and conditional bounds.

Diagnostic measurement only. No production selector, kernel, or throughput
claim. Fresh post113_selected is the authenticated control. Pinned llama
cc83d7b4824f73cfdda4dfbb47ee39804f71b328 is the benchmark authority;
inspection HEAD is recorded separately and is not silently the pin.
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

from tools.opt060_engine_attribution import (  # noqa: E402
    chargeable_records,
    family_table,
    interval,
    stream_aware_totals,
    union_ms,
)
from tools.opt075_q4_production_admission import (  # noqa: E402
    T_CRIT_DF9,
    dump_json,
    load_json,
    mean,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_REV,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt113_coupled_stack_gate import (  # noqa: E402
    source_paths as post113_source_paths,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    validate_performance_admission,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt115_pipeline_traffic_contract.json"
ITERATION = ROOT / "pins/opt115_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
REPORT = ROOT / "evidence/optimization/opt115-pipeline-traffic/REPORT.md"
EVIDENCE = REPORT.parent
OPT113_FIXTURE = ROOT / "fixtures/opt113_coupled_stack_gate.json"
OPT113_CONTRACT = ROOT / "pins/opt113_coupled_stack_gate_contract.json"
OPT114_FIXTURE = ROOT / "fixtures/opt114_sitting_launch.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
TENSOR_INVENTORY = ROOT / "pins/tensor_inventory.json"
NATIVE = "build/qw38-cuda-opt115-pipeline-traffic-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = (
    "preflight",
    "request-trace",
    "long-context",
    "traffic-timeline",
    "ranking",
    "report",
)
AA_WORKLOADS = ("p4096", "d128", "d2048")
AA_PREFIX = {"p4096": 4096, "d128": 128, "d2048": 2048}
LONG_PROBES = ("d8192", "d32768", "d131040")
LONG_PREFIX = {"d8192": 8192, "d32768": 32768, "d131040": 131040}
WARMUPS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
SESSION_LOOP_TOKENS = 32
LONG_DECODE_TOKENS = 32
CAPACITY_128K = 131072
MEASUREMENT_GUARD = 0.02
PEAK_BANDWIDTH_GB_S = 1792.0
REQUIRED_RESERVE_BYTES = 1_610_612_736
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
LLAMA_INSPECTION_HEAD = "1945e092030f8668ff93382799502d01490e564d"
DS4_INSPECTION_HEAD = "c238077a87186381bf626cc531bccffe1fef79e7"
RESIDUAL_WIDTH = 5120
VOCABULARY_SIZE = 248320
MODEL_LAYERS = 64
GDN_LAYERS = 48
ATTENTION_LAYERS = 16
GDN_CONVOLUTION_VALUES = 40960
GDN_RECURRENT_VALUES = 786432
ATTENTION_KV_WIDTH = 1024
ATTENTION_HEAD_WIDTH = 256
KV_HEADS = 4
QUERY_HEADS = 24
Q4K_BLOCK_VALUES = 256
Q4K_BLOCK_BYTES = 144
TRAFFIC_PHASES = (
    "weights",
    "activations",
    "kv",
    "gdn_state",
    "transfers",
    "logits",
    "barriers",
    "graphs",
)
RANKED_TASKS = (
    "OPT-117",
    "OPT-118",
    "OPT-119",
    "OPT-120",
    "OPT-121",
    "OPT-122",
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
    "post113_authenticated",
    "request_trace",
    "long_context",
    "traffic_map",
    "timeline",
    "bandwidth_bounds",
    "source_comparison",
    "ranked_opportunities",
    "shared_keep_protocol",
    "production_kept",
    "claims_throughput",
    "claims_performance_improvement",
    "report_path",
)
PROOF = (
    "diagnostics only: no production selector or kernel change; fresh "
    "post113_selected is the authenticated control; pinned llama "
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328 is the benchmark authority; "
    "3 warmups plus 10 interleaved AB/BA pairs at P4096, D128, and D2048; "
    "long-context D8192/D32768/D131040+32 are separately labeled probes in a "
    "131072-capacity session; 1792 GB/s is listed RTX 5090 peak, not observed "
    "application bandwidth; event unions distinguish GPU busy/idle, host "
    "overlap, sync and launch submission; no throughput or optimality claim"
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class TrafficError(AssertionError):
    """Fail-closed OPT-115 pipeline-traffic error."""


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT)


def shared_keep_protocol() -> dict[str, Any]:
    return dict(load_contract()["shared_keep_protocol"])


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
        raise TrafficError(
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
    if role in PARENT_ROLES | HOST_ROLES | CPU_GAP_ROLES | SYNC_ROLES | GRAPH_ROLES | {
        "wall"
    }:
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


def classify_timeline(
    records: Sequence[Mapping[str, Any]],
    *,
    wall_ms: float | None = None,
    tokens: int = 1,
) -> dict[str, Any]:
    """Union-based GPU busy/idle/host/sync/launch split. No overlapping sums."""
    charged = chargeable_records(records)
    leaves = [row for row in charged if is_kernel_leaf(row)]
    sync = [
        row
        for row in charged
        if str(row.get("role", "")) in SYNC_ROLES
        and str(row.get("attribution_role", "member")) != "enclosing"
    ]
    host = [row for row in charged if str(row.get("role", "")) in HOST_ROLES]
    graph = [row for row in charged if str(row.get("role", "")) in GRAPH_ROLES]
    gpu_busy = union_ms(leaves)
    gpu_idle = leaf_gap_ms(charged)
    sync_union = union_ms(sync)
    host_union = union_ms(host)
    graph_union = union_ms(graph)
    classified = union_ms(leaves + sync + host + graph)
    wall = float(wall_ms) if wall_ms is not None else classified + gpu_idle
    unclassified = max(0.0, wall - classified - gpu_idle)
    coverage = (classified + gpu_idle) / wall if wall > 0.0 else 0.0
    totals = stream_aware_totals(charged)
    families = family_table(charged)
    return {
        "tokens": tokens,
        "record_count": len(records),
        "charged_count": len(charged),
        "leaf_kernel_count": len(leaves),
        "gpu_busy_union_ms": gpu_busy,
        "gpu_idle_gap_ms": gpu_idle,
        "host_overlap_union_ms": host_union,
        "sync_union_ms": sync_union,
        "graph_union_ms": graph_union,
        "launch_submission_ms": gpu_idle + host_union,
        "classified_union_ms": classified,
        "unclassified_ms": unclassified,
        "wall_ms": wall,
        "interval_coverage": coverage,
        "overlap_not_summed": True,
        "sum_exceeds_wall": bool(totals.get("sum_exceeds_wall")),
        "stream_aware": totals,
        "families": families,
        "method": "event_union_plus_leaf_gaps",
    }


def authenticate_post113(
    selectors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = dict(selectors or post113_source_paths())
    contract = load_json(OPT113_CONTRACT)
    expected = dict(contract["post113_selected"])
    mismatches: dict[str, Any] = {}
    for key, value in expected.items():
        if paths.get(key) != value:
            mismatches[key] = {"actual": paths.get(key), "expected": value}
    fixture = load_json(OPT113_FIXTURE) if OPT113_FIXTURE.is_file() else {}
    selected = dict(fixture.get("post113_selected") or {})
    authenticated = not mismatches
    if paths.get("q4_decode") != "llama_q4k_mmvq":
        authenticated = False
        mismatches.setdefault(
            "q4_decode",
            {"actual": paths.get("q4_decode"), "expected": "llama_q4k_mmvq"},
        )
    if paths.get("attention_pipeline") != "opt111_base":
        authenticated = False
    if paths.get("decode_attention") != "hybrid_crossover":
        authenticated = False
    if paths.get("gdn_decode") != "sequential":
        authenticated = False
    if paths.get("execution_graphs") != "ffn_only":
        authenticated = False
    return {
        "ok": authenticated,
        "selectors": paths,
        "expected": expected,
        "opt113_fixture_selected": selected,
        "mismatches": mismatches,
        "llama_q4k_mmvq": paths.get("q4_decode") == "llama_q4k_mmvq",
        "opt111_base": paths.get("attention_pipeline") == "opt111_base",
        "hybrid_crossover": paths.get("decode_attention") == "hybrid_crossover",
        "sequential_gdn": paths.get("gdn_decode") == "sequential",
        "ffn_only_graphs": paths.get("execution_graphs") == "ffn_only",
        "opt113_numbers_historical_only": True,
    }


def inventory_bytes(inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(inventory or load_json(TENSOR_INVENTORY))
    by_dtype: dict[str, int] = {}
    by_role: dict[str, int] = {}
    by_phase: dict[str, int] = {
        "weights": 0,
        "norms": 0,
        "embedding": 0,
        "output_head": 0,
    }
    for tensor in payload.get("tensors") or []:
        dtype = str(tensor.get("dtype") or "unknown")
        role = str(tensor.get("role") or "unknown")
        size = int(tensor.get("storage_bytes") or 0)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + size
        by_role[role] = by_role.get(role, 0) + size
        if role == "token_embedding":
            by_phase["embedding"] += size
        elif role == "output_projection":
            by_phase["output_head"] += size
        elif dtype == "F32" and "norm" in role:
            by_phase["norms"] += size
        else:
            by_phase["weights"] += size
    return {
        "tensor_count": int(
            payload.get("tensor_count") or len(payload.get("tensors") or [])
        ),
        "by_dtype_bytes": by_dtype,
        "by_role_bytes": by_role,
        "by_phase_bytes": by_phase,
        "q4_k_bytes": int(by_dtype.get("Q4_K") or 0),
        "q8_0_bytes": int(by_dtype.get("Q8_0") or 0),
        "q6_k_bytes": int(by_dtype.get("Q6_K") or 0),
        "f32_bytes": int(by_dtype.get("F32") or 0),
        "units": "bytes",
    }


def embedding_row_bytes() -> int:
    blocks = (RESIDUAL_WIDTH + Q4K_BLOCK_VALUES - 1) // Q4K_BLOCK_VALUES
    return blocks * Q4K_BLOCK_BYTES


def kv_bytes(prefix: int) -> int:
    return ATTENTION_LAYERS * int(prefix) * ATTENTION_KV_WIDTH * 2 * 2


def gdn_state_bytes() -> dict[str, int]:
    conv = GDN_LAYERS * GDN_CONVOLUTION_VALUES * 4
    recurrent = GDN_LAYERS * GDN_RECURRENT_VALUES * 4
    return {
        "convolution_bytes": conv,
        "recurrent_bytes": recurrent,
        "total_bytes": conv + recurrent,
        "candidate_plus_committed_capacity_note": (
            "decode swaps candidate/committed pointers; do not charge a full "
            "duplicate copy on every token unless measured"
        ),
    }


def compulsory_decode_bytes(*, prefix: int) -> dict[str, Any]:
    """Tensor-derived compulsory DRAM reads/writes for one decode token."""
    inv = inventory_bytes()
    embed_row = embedding_row_bytes()
    streamed_weights = (
        int(inv["q4_k_bytes"])
        - int((inv.get("by_role_bytes") or {}).get("token_embedding") or 0)
        + int(inv["q8_0_bytes"])
        + int(inv["q6_k_bytes"])
        + int(inv["f32_bytes"])
        + embed_row
    )
    gdn = gdn_state_bytes()
    kv_read = kv_bytes(prefix)
    kv_append = kv_bytes(1)
    logits = VOCABULARY_SIZE * 4
    hidden = RESIDUAL_WIDTH * 4
    # Sequential GDN: read committed recurrent + write candidate, then swap.
    gdn_rw = int(gdn["recurrent_bytes"]) + int(gdn["convolution_bytes"])
    activations = (
        MODEL_LAYERS * RESIDUAL_WIDTH * 4  # residual / norm working set
        + MODEL_LAYERS * 17408 * 4  # FFN intermediate (one of gate/up)
    )
    graphs = 0
    barriers = 0
    transfers = logits + hidden
    phases = {
        "weights": streamed_weights,
        "activations": activations,
        "kv": kv_read + kv_append,
        "gdn_state": gdn_rw,
        "transfers": transfers,
        "logits": logits,
        "barriers": barriers,
        "graphs": graphs,
    }
    total = sum(phases.values())
    return {
        "scope": "one_decode_token",
        "prefix": prefix,
        "units": "bytes",
        "phases": phases,
        "bytes_per_token": total,
        "bytes_per_request_one_token": total,
        "embedding_row_bytes": embed_row,
        "embedding_full_table_excluded": True,
        "kv_read_bytes": kv_read,
        "kv_append_bytes": kv_append,
        "gdn": gdn,
        "logits_d2h_bytes": logits,
        "hidden_d2h_bytes": hidden,
        "source": "tensor_inventory_plus_architecture_constants",
        "measured": False,
    }


def compulsory_prefill_bytes(*, tokens: int, microbatch: int = 4096) -> dict[str, Any]:
    inv = inventory_bytes()
    tiles = max(1, (int(tokens) + microbatch - 1) // microbatch)
    streamed_weights = (
        int(inv["q4_k_bytes"])
        - int((inv.get("by_role_bytes") or {}).get("token_embedding") or 0)
        + int(inv["q8_0_bytes"])
        + int(inv["q6_k_bytes"])
        + int(inv["f32_bytes"])
    ) * tiles
    embed = embedding_row_bytes() * int(tokens)
    kv_write = kv_bytes(tokens)
    gdn = gdn_state_bytes()
    # Prompt GDN scans the chunk; charge recurrent+conv once per tile plus carry.
    gdn_scan = int(gdn["total_bytes"]) * tiles
    activations = int(tokens) * RESIDUAL_WIDTH * 4 * MODEL_LAYERS
    logits = VOCABULARY_SIZE * 4
    phases = {
        "weights": streamed_weights + embed,
        "activations": activations,
        "kv": kv_write,
        "gdn_state": gdn_scan,
        "transfers": logits + RESIDUAL_WIDTH * 4,
        "logits": logits,
        "barriers": 0,
        "graphs": 0,
    }
    return {
        "scope": "prefill_request",
        "tokens": tokens,
        "microbatch": microbatch,
        "tiles": tiles,
        "units": "bytes",
        "phases": phases,
        "bytes_per_request": sum(phases.values()),
        "bytes_per_token": sum(phases.values()) / float(tokens) if tokens else 0.0,
        "source": "tensor_inventory_plus_architecture_constants",
        "measured": False,
    }


def bytes_to_peak_ms(
    byte_count: int | float, *, gb_s: float = PEAK_BANDWIDTH_GB_S
) -> float:
    if gb_s <= 0.0:
        return float("inf")
    return (float(byte_count) / (gb_s * 1.0e9)) * 1000.0


def bandwidth_bounds(
    phases: Mapping[str, int | float],
    *,
    wall_ms: float | None = None,
    measured_bandwidth_gb_s: float | None = None,
    prefix: int | None = None,
) -> dict[str, Any]:
    peak = PEAK_BANDWIDTH_GB_S
    sustainable = measured_bandwidth_gb_s
    per_phase_peak = {
        name: {
            "bytes": float(value),
            "peak_ms": bytes_to_peak_ms(value, gb_s=peak),
            "sustainable_ms": (
                bytes_to_peak_ms(value, gb_s=sustainable) if sustainable else None
            ),
        }
        for name, value in phases.items()
    }
    independent_sum_ms = sum(item["peak_ms"] for item in per_phase_peak.values())
    critical_path_ms = max(
        (item["peak_ms"] for item in per_phase_peak.values()), default=0.0
    )
    # Weight reads and kernel compute share the busy interval; launch idle is serial.
    serial_ms = per_phase_peak.get("barriers", {}).get("peak_ms", 0.0)
    overlap_aware_ms = max(critical_path_ms, serial_ms)
    assumptions = [
        "RTX 5090 listed peak 1792 GB/s from NVIDIA Blackwell architecture paper",
        "peak is an optimistic bound, not observed application bandwidth",
        "compulsory reads assume no L2 reuse across layers and no write-allocate traffic beyond listed writes",
        "embedding lookup charges one Q4_K row, not the full table",
        "GDN decode charges one recurrent+conv pass with pointer-swap commit, not a full 8 GiB KV copy",
        "overlapping costs use max/critical-path, not a sum of independent peak bounds",
        "fixed-format Q4_K/Q8_0/Q6_K; quality-gated lower-bit models are separate OPT-121/122 candidates",
    ]
    if sustainable is None:
        assumptions.append(
            "measured sustainable bandwidth unavailable; sustainable_ms is null"
        )
    return {
        "prefix": prefix,
        "peak_bandwidth_gb_s": peak,
        "measured_sustainable_bandwidth_gb_s": sustainable,
        "per_phase": per_phase_peak,
        "independent_sum_peak_ms": independent_sum_ms,
        "independent_sum_not_a_bound": True,
        "critical_path_peak_ms": critical_path_ms,
        "overlap_aware_peak_ms": overlap_aware_ms,
        "wall_ms": wall_ms,
        "assumptions": assumptions,
        "units": {"bytes": "bytes", "time": "milliseconds", "bandwidth": "GB/s"},
        "claims_throughput": False,
    }


def inspect_repo_revision(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "path": str(path),
            "present": False,
            "revision": None,
            "state": "missing",
        }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "path": str(path),
        "present": True,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "state": "dirty" if dirty.stdout.strip() else "clean",
    }


def _file_has(path: Path, needles: Sequence[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False, "found": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [needle for needle in needles if needle in text]
    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "found": found,
        "missing": [needle for needle in needles if needle not in found],
    }


def source_comparison_matrix() -> dict[str, Any]:
    llama = ROOT.parent / "llama.cpp"
    ds4 = ROOT.parent / "ds4"
    llama_id = inspect_repo_revision(llama)
    ds4_id = inspect_repo_revision(ds4)
    mechanisms = {
        "OPT-117_cuda_graphs": {
            "task": "OPT-117",
            "quartz": {
                "file": "cuda/full_scheduler.cu",
                "functions": [
                    "capture_decode_segment_graph",
                    "enqueue_decode_layer_eager",
                ],
                "physical_dtype": "existing production kernels",
                "notes": "ffn_only ships; decode_segments8 capture failed in OPT-114",
            },
            "llama": _file_has(
                llama / "ggml/src/ggml-cuda/ggml-cuda.cu",
                [
                    "ggml_cuda_graph_evaluate_and_capture",
                    "cudaGraphLaunch",
                    "ggml_cuda_graph_check_compability",
                ],
            ),
            "ds4": {"present": False, "notes": "not the Qwen graph authority"},
            "transferability": (
                "llama CUDA-graph capture of the decoded cgraph is the "
                "mechanism analogue; Quartz must capture its own scheduler "
                "FFN/attention/GDN launches, not import ggml graphs"
            ),
            "actual_consumers": "decode token body after graph capture",
        },
        "OPT-118_transfers": {
            "task": "OPT-118",
            "quartz": {
                "file": "cuda/full_scheduler.cu",
                "functions": ["execute_token D2H logits/hidden", "greedy_sample"],
                "physical_dtype": "FP32 logits 248320",
                "bytes_per_token": VOCABULARY_SIZE * 4,
            },
            "llama": _file_has(
                llama / "src/models/qwen35.cpp",
                ["llama_model_qwen35", "tok_embd"],
            ),
            "ds4": {
                "present": ds4_id["present"],
                "notes": "host sampling analogue only",
            },
            "transferability": (
                "lazy host logits is a Quartz API change; llama keeps outputs "
                "on device until requested. Not a drop-in kernel."
            ),
            "actual_consumers": "Session::sample / Session::logits / CLI poll",
        },
        "OPT-119_activations": {
            "task": "OPT-119",
            "quartz": {
                "file": "cuda/scheduler_primitives.cu",
                "notes": "norm, quantize, residual, SwiGLU staging",
            },
            "llama": _file_has(
                llama / "ggml/src/ggml-cuda/ggml-cuda.cu",
                ["ggml_cuda_try_fuse", "mul_mat"],
            ),
            "ds4": {"present": ds4_id["present"]},
            "transferability": (
                "producer/consumer fusion is architecture-specific; Q8_1 vs "
                "internal Q8Block are not interchangeable"
            ),
            "actual_consumers": "next projection / FFN down / KV append",
        },
        "OPT-120_packed_kv": {
            "task": "OPT-120",
            "quartz": {
                "physical_dtype": "BF16 K/V, 16 layers, 4 KV heads, 256 width",
                "bytes_at_128k": kv_bytes(CAPACITY_128K),
            },
            "llama": _file_has(
                llama / "ggml/src/ggml-cuda/fattn-vec.cuh",
                ["flash_attn_ext_vec", "type_K", "type_V", "q8_1"],
            ),
            "ds4": _file_has(
                ds4 / "ds4_cuda.cu",
                [
                    "fp8_kv_quantize_row",
                    "fp8_kv_quantize_store_rows_kernel",
                    "indexer_hadamard_fp4_kernel",
                ],
            ),
            "transferability": (
                "llama fattn-vec can consume quantized K/V types in-registers. "
                "ds4 fp8_kv_quantize_row quantizes numerically then writes "
                "floats/half into raw cache (sizeof(float) storage); not a "
                "ready packed 8/4-bit resident cache. DeepSeek indexer/MoE is "
                "not Qwen 48-GDN/16-attention."
            ),
            "actual_consumers": "prompt and decode attention over the live prefix",
        },
        "OPT-121_weight_bytes": {
            "task": "OPT-121",
            "quartz": {
                "physical_dtype": "Q4_K FFN, Q8_0 mixer, Q6_K output/attn out",
                "streamed_decode_bytes": inventory_bytes()["q4_k_bytes"]
                + inventory_bytes()["q8_0_bytes"]
                + inventory_bytes()["q6_k_bytes"],
            },
            "llama": {
                "files": [
                    "ggml/src/ggml-cuda/mmq.cuh",
                    "ggml/src/ggml-cuda/mmvq.cuh",
                ],
                "present": (llama / "ggml/src/ggml-cuda/mmq.cuh").is_file(),
            },
            "ds4": {
                "files": ["cuda/mmq/mmq.cuh", "cuda/mmq/ds4_repack.cu", "ds4_cuda.cu"],
                "present": (ds4 / "cuda/mmq").is_dir(),
                "notes": (
                    "MXFP4 path preserves an already-MXFP4 source; requantizing "
                    "this Q4_K_M artifact is a different approximation"
                ),
            },
            "transferability": (
                "use llama/ds4 packed consumers as implementation references; "
                "do not relabel Q4_K bits as FP4"
            ),
            "actual_consumers": "decode MMVQ and prompt MMQ of each role",
        },
        "OPT-122_gdn_state": {
            "task": "OPT-122",
            "quartz": {
                "physical_dtype": "FP32 recurrent + conv rings",
                "bytes": gdn_state_bytes(),
            },
            "llama": _file_has(
                llama / "src/llama-memory-recurrent.h",
                ["llama_memory_recurrent", "ssm"],
            ),
            "ds4": {"present": False, "notes": "DeepSeek MLA/indexer, not Qwen GDN"},
            "transferability": (
                "BF16/block-scaled recurrent storage is a Quartz numerical "
                "experiment; llama recurrent memory is a different layout"
            ),
            "actual_consumers": "gdn_step recurrence, checkpoint, prompt carry",
        },
    }
    return {
        "pinned_llama_revision": LLAMA_REV,
        "llama_inspection_head_expected": LLAMA_INSPECTION_HEAD,
        "llama_inspection_head_actual": llama_id.get("revision"),
        "llama_inspection_is_not_pin": llama_id.get("revision") != LLAMA_REV,
        "ds4_inspection_head_expected": DS4_INSPECTION_HEAD,
        "ds4_inspection_head_actual": ds4_id.get("revision"),
        "llama": llama_id,
        "ds4": ds4_id,
        "mechanisms": mechanisms,
        "architecture_warning": (
            "DeepSeek compression/indexer/MoE is not Qwen's 48-GDN/16-attention"
        ),
    }


def _opt114_launch_ms() -> dict[str, Any]:
    if not OPT114_FIXTURE.is_file():
        return {"available": False}
    payload = load_json(OPT114_FIXTURE)
    launch = payload.get("launch_overhead") or {}
    return {
        "available": True,
        "historical_not_this_sitting": True,
        "d128_ms": (launch.get("d128") or {}).get("removable_launch_ms_per_token"),
        "d2048_ms": (launch.get("d2048") or {}).get("removable_launch_ms_per_token"),
        "status": payload.get("graph_opportunity"),
        "note": (
            "OPT-114 launch numbers are a hypothesis to revalidate on post113; "
            "wall minus leaf-event sums alone cannot identify causal removable overhead"
        ),
    }


def rank_opportunities(
    *,
    traffic: Mapping[str, Any],
    timeline: Mapping[str, Any],
    bounds: Mapping[str, Any],
) -> dict[str, Any]:
    decode_d128 = traffic.get("d128") or compulsory_decode_bytes(prefix=128)
    decode_d2048 = traffic.get("d2048") or compulsory_decode_bytes(prefix=2048)
    decode_long = traffic.get("d131040") or compulsory_decode_bytes(prefix=131040)
    d128_phases = dict(decode_d128.get("phases") or {})
    d2048_phases = dict(decode_d2048.get("phases") or {})
    long_phases = dict(decode_long.get("phases") or {})
    weight_ms = bytes_to_peak_ms(d128_phases.get("weights") or 0)
    kv_d128_ms = bytes_to_peak_ms(d128_phases.get("kv") or 0)
    kv_d2048_ms = bytes_to_peak_ms(d2048_phases.get("kv") or 0)
    kv_long_ms = bytes_to_peak_ms(long_phases.get("kv") or 0)
    gdn_ms = bytes_to_peak_ms(d128_phases.get("gdn_state") or 0)
    logits_ms = bytes_to_peak_ms(d128_phases.get("logits") or 0)
    act_ms = bytes_to_peak_ms(d128_phases.get("activations") or 0)
    launch_hist = _opt114_launch_ms()
    d128_tl = (timeline.get("d128") or {}) if timeline else {}
    d2048_tl = (timeline.get("d2048") or {}) if timeline else {}
    launch_d128 = d128_tl.get("launch_submission_ms")
    launch_d2048 = d2048_tl.get("launch_submission_ms")
    if launch_d128 is None:
        launch_d128 = launch_hist.get("d128_ms")
    if launch_d2048 is None:
        launch_d2048 = launch_hist.get("d2048_ms")

    def _entry(
        task: str,
        *,
        verdict: str,
        removable_bytes: int,
        peak_ms: float,
        uncertainty: str,
        primary: str,
        rationale: str,
    ) -> dict[str, Any]:
        return {
            "task": task,
            "verdict": verdict,
            "removable_bytes_upper": removable_bytes,
            "peak_ms_upper": peak_ms,
            "uncertainty": uncertainty,
            "primary_workload": primary,
            "rationale": rationale,
            "go": verdict.startswith("go"),
        }

    rows = [
        _entry(
            "OPT-117",
            verdict="go",
            removable_bytes=0,
            peak_ms=float(launch_d128 or 0.0),
            uncertainty=(
                "launch ms is GPU-idle leaf gaps this sitting when measured, "
                "else OPT-114 historical hypothesis; capture failure is not a "
                "measured graph loss"
            ),
            primary="d128/d2048",
            rationale=(
                "decode-segment graphs target launch/submit gaps. Bytes are not "
                "the savings; time is. Revalidate post113 gaps before treating "
                "OPT-114 ~11 ms/token as causal."
            ),
        ),
        _entry(
            "OPT-118",
            verdict="go",
            removable_bytes=int(d128_phases.get("transfers") or 0),
            peak_ms=logits_ms,
            uncertainty=(
                "PCIe byte bound is small; host-sample/D2H sync latency may "
                "exceed the copy bound and is not proven zero"
            ),
            primary="complete sync/sample/eval request",
            rationale=(
                f"logits D2H {VOCABULARY_SIZE * 4} B/token "
                f"({logits_ms:.4f} ms at peak). Inventory, do not skip: sync "
                "and sampling can sit on the request critical path."
            ),
        ),
        _entry(
            "OPT-119",
            verdict="go",
            removable_bytes=int(d128_phases.get("activations") or 0),
            peak_ms=act_ms,
            uncertainty=(
                "activation bytes are architecture-derived; fusion may not "
                "remove the compulsory dense weight sweep"
            ),
            primary="prefill (P4096) then decode guard",
            rationale=(
                "decode activation traffic is far below weights. Prefill "
                "microbatch activations are the plausible slice."
            ),
        ),
        _entry(
            "OPT-120",
            verdict="go",
            removable_bytes=int(long_phases.get("kv") or 0),
            peak_ms=kv_long_ms,
            uncertainty=(
                "D128/D2048 KV peak bounds are small; 128K populated-cache "
                "reads are the material term. Capacity is not traffic."
            ),
            primary="long-context decode",
            rationale=(
                f"KV peak bound D128 {kv_d128_ms:.4f} ms, D2048 {kv_d2048_ms:.4f} "
                f"ms, D131040 {kv_long_ms:.3f} ms. Headline D128/D2048 must not "
                "be claimed from 128K capacity savings."
            ),
        ),
        _entry(
            "OPT-121",
            verdict="go",
            removable_bytes=int(d128_phases.get("weights") or 0),
            peak_ms=weight_ms,
            uncertainty=(
                "peak-bandwidth lower bound on weight time; L2 reuse and "
                "compute can only make wall longer, not shorter than this "
                "optimistic bound"
            ),
            primary="d128/d2048",
            rationale=(
                f"streamed packed weights dominate compulsory decode bytes "
                f"({weight_ms:.3f} ms at 1792 GB/s peak). Largest byte sink."
            ),
        ),
        _entry(
            "OPT-122",
            verdict="go_conditional",
            removable_bytes=int(d128_phases.get("gdn_state") or 0),
            peak_ms=gdn_ms,
            uncertainty=(
                "upper bound assumes one recurrent+conv pass per token with "
                "pointer-swap commit; extra rereads would raise the bound. "
                "Missing counter is not a measured zero."
            ),
            primary="complete request after OPT-118",
            rationale=(
                f"GDN state {gdn_state_bytes()['total_bytes']} B resident; "
                f"one-pass traffic peak bound {gdn_ms:.3f} ms. Material only "
                "if measured amplification or long-scan traffic exceeds A/A "
                "resolution; otherwise no_material_opportunity remains open "
                "until OPT-122 measures it."
            ),
        ),
    ]
    ranked = sorted(rows, key=lambda row: float(row["peak_ms_upper"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return {
        "ranked": ranked,
        "method": "peak_ms_upper_then_request_relevance",
        "claims_throughput": False,
        "opt114_launch_historical": launch_hist,
        "bounds_ref": {
            "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
            "overlap_aware": (bounds or {}).get("overlap_aware_peak_ms"),
        },
    }


def sitting_identity(*, telemetry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source, state = git_identity()
    selectors = post113_source_paths()
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
        "llama_inspection_head": LLAMA_INSPECTION_HEAD,
        "ds4_inspection_head": DS4_INSPECTION_HEAD,
        "gguf_sha256": GGUF_SHA,
        "token_generator": TOKEN_GENERATOR,
        "source_revision": source,
        "source_state": state,
        "nvccflags": "-O2 --fmad=false",
        "execution_graphs": "ffn_only",
        "selectors": selectors,
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
        "task": "OPT-115",
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
        if line.startswith("QW38_OPT115_RECORDS="):
            payload = json.loads(line.split("=", 1)[1])
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def parse_aa_pairs(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    pairs = payload.get("pairs") or []
    parsed: dict[str, list[float]] = {
        "a_tok_s": [],
        "b_tok_s": [],
        "a_wall_ms": [],
        "b_wall_ms": [],
        "a_p50_ms": [],
        "a_p95_ms": [],
        "b_p50_ms": [],
        "b_p95_ms": [],
    }
    for row in pairs:
        a = row.get("A") or {}
        b = row.get("B") or {}
        parsed["a_tok_s"].append(float(a.get("tok_s") or 0.0))
        parsed["b_tok_s"].append(float(b.get("tok_s") or 0.0))
        parsed["a_wall_ms"].append(float(a.get("wall_ms") or 0.0))
        parsed["b_wall_ms"].append(float(b.get("wall_ms") or 0.0))
        parsed["a_p50_ms"].append(float(a.get("p50_ms") or 0.0))
        parsed["a_p95_ms"].append(float(a.get("p95_ms") or 0.0))
        parsed["b_p50_ms"].append(float(b.get("p50_ms") or 0.0))
        parsed["b_p95_ms"].append(float(b.get("p95_ms") or 0.0))
    return parsed


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise TrafficError(f"unknown phase {phase}")
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
        "same_binary": True,
    }


def empty_long_probe(name: str, *, reason: str) -> dict[str, Any]:
    return {
        "probe": name,
        "prefix": LONG_PREFIX[name],
        "capacity": CAPACITY_128K,
        "decode_tokens": LONG_DECODE_TOKENS,
        "status": "unmeasured",
        "reason": reason,
        "populated_cache": True,
        "capacity_alone_is_not_traffic": True,
    }


def architecture_traffic_map() -> dict[str, Any]:
    prefill = compulsory_prefill_bytes(tokens=4096)
    d128 = compulsory_decode_bytes(prefix=128)
    d2048 = compulsory_decode_bytes(prefix=2048)
    d8192 = compulsory_decode_bytes(prefix=8192)
    d32768 = compulsory_decode_bytes(prefix=32768)
    d131040 = compulsory_decode_bytes(prefix=131040)
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    return {
        "units": "bytes",
        "inventory": inventory_bytes(),
        "memory_ledger": {
            "resident_model_bytes": (memory.get("owners") or {}).get(
                "resident_model_bytes"
            ),
            "attention_kv_bytes": (memory.get("owners") or {}).get(
                "attention_kv_bytes"
            ),
            "gdn_state_bytes": (memory.get("owners") or {}).get("gdn_state_bytes"),
            "workspace_bytes": (memory.get("owners") or {}).get("workspace_bytes"),
            "required_reserve_bytes": memory.get("required_reserve_bytes"),
            "post_graph_admitted": memory.get("post_graph_admitted"),
        },
        "p4096": prefill,
        "d128": d128,
        "d2048": d2048,
        "d8192": d8192,
        "d32768": d32768,
        "d131040": d131040,
        "phases": list(TRAFFIC_PHASES),
        "measured": False,
        "note": (
            "tensor-derived compulsory traffic; GPU phases overlay measured "
            "timeline counters when available"
        ),
    }


def empty_payload(*, reason: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    traffic = architecture_traffic_map()
    bounds = {
        name: bandwidth_bounds(
            dict((traffic.get(name) or {}).get("phases") or {}),
            prefix=AA_PREFIX.get(name) or LONG_PREFIX.get(name),
        )
        for name in (*AA_WORKLOADS, *LONG_PROBES)
    }
    timeline = {
        name: {
            "valid": False,
            "reason": reason,
            "interval_coverage": 0.0,
            "unclassified_ms": None,
        }
        for name in ("d128", "d2048")
    }
    ranking = rank_opportunities(
        traffic=traffic, timeline=timeline, bounds=bounds["d128"]
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-115",
        "status": "blocked_gpu_unavailable" if "gpu" in reason else "unmeasured",
        "measurement_utc": utc_now(),
        "identity": dict(identity),
        "hardware_executed": False,
        "gpu_blocker": reason,
        "post113_authenticated": authenticate_post113(identity.get("selectors")),
        "request_trace": {
            name: empty_aa_workload(name, reason=reason) for name in AA_WORKLOADS
        },
        "aa_verdict": "unmeasured",
        "session_loop": {"status": "unmeasured", "reason": reason},
        "long_context": {
            name: empty_long_probe(name, reason=reason) for name in LONG_PROBES
        },
        "traffic_map": traffic,
        "timeline": timeline,
        "bandwidth_bounds": bounds,
        "source_comparison": source_comparison_matrix(),
        "ranked_opportunities": ranking,
        "shared_keep_protocol": shared_keep_protocol(),
        "production_kept": True,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "opt074_coverage_unadmitted_blocker": False,
        "report_path": "evidence/optimization/opt115-pipeline-traffic/REPORT.md",
        "proof_limit": PROOF,
    }
    return payload


def persist(payload: Mapping[str, Any]) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump_json(FIXTURE, payload)
    write_report(payload)
    return dict(payload)


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        return load_json(FIXTURE)
    identity = sitting_identity(telemetry=try_telemetry())
    available, blocker = gpu_available()
    reason = blocker if not available else "scaffold_pending_gpu_phases"
    return empty_payload(reason=reason, identity=identity)


def merge_phase(
    existing: Mapping[str, Any], phase: str, block: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(existing)
    key = phase.replace("-", "_")
    payload[key] = dict(block)
    if phase == "request-trace":
        payload["request_trace"] = dict(block)
        payload["aa_verdict"] = combine_aa_verdicts(block)
        if "session_loop" in block:
            payload["session_loop"] = block["session_loop"]
    if phase == "long-context":
        payload["long_context"] = dict(block)
    if phase == "traffic-timeline":
        payload["timeline"] = dict(block.get("timeline") or block)
        if block.get("traffic_overlay"):
            traffic = dict(payload.get("traffic_map") or architecture_traffic_map())
            traffic["measured_overlay"] = block["traffic_overlay"]
            payload["traffic_map"] = traffic
    if phase == "ranking":
        payload["ranked_opportunities"] = dict(
            block.get("ranked_opportunities") or block
        )
        if block.get("bandwidth_bounds"):
            payload["bandwidth_bounds"] = block["bandwidth_bounds"]
        if block.get("source_comparison"):
            payload["source_comparison"] = block["source_comparison"]
        if block.get("traffic_map"):
            payload["traffic_map"] = block["traffic_map"]
    payload["measurement_utc"] = utc_now()
    return payload


def write_report(payload: Mapping[str, Any]) -> None:
    identity = payload.get("identity") or {}
    auth = payload.get("post113_authenticated") or {}
    trace = payload.get("request_trace") or {}
    long_ctx = payload.get("long_context") or {}
    traffic = payload.get("traffic_map") or {}
    bounds = payload.get("bandwidth_bounds") or {}
    ranking = payload.get("ranked_opportunities") or {}
    protocol = payload.get("shared_keep_protocol") or {}
    comparison = payload.get("source_comparison") or {}
    timeline = payload.get("timeline") or {}
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    def _fmt(value: Any, digits: int = 4) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "n/a"

    aa_rows = []
    for name in AA_WORKLOADS:
        block = trace.get(name) or {}
        aa_rows.append(
            f"| {name} | {_fmt(block.get('a_mean_tok_s'))} | "
            f"{_fmt(block.get('b_mean_tok_s'))} | "
            f"{_fmt(block.get('geometric_ratio'))} | "
            f"{block.get('status', 'unmeasured')} |"
        )
    long_rows = []
    for name in LONG_PROBES:
        block = long_ctx.get(name) or {}
        long_rows.append(
            f"| {name} | {block.get('prefix')} | {block.get('decode_tokens')} | "
            f"{_fmt(block.get('tok_s'))} | {_fmt(block.get('ttft_ms'))} | "
            f"{_fmt(block.get('p50_ms'))} | {_fmt(block.get('p95_ms'))} | "
            f"{block.get('status', 'unmeasured')} |"
        )
    d128 = traffic.get("d128") or {}
    phases = d128.get("phases") or {}
    byte_rows = []
    for name in TRAFFIC_PHASES:
        byte_rows.append(
            f"| {name} | {int(phases.get(name) or 0)} | bytes | "
            f"{_fmt(bytes_to_peak_ms(phases.get(name) or 0))} |"
        )
    rank_rows = []
    for row in ranking.get("ranked") or []:
        rank_rows.append(
            f"| {row.get('rank')} | {row.get('task')} | {row.get('verdict')} | "
            f"{int(row.get('removable_bytes_upper') or 0)} | "
            f"{_fmt(row.get('peak_ms_upper'))} | {row.get('primary_workload')} |"
        )
    d128_tl = timeline.get("d128") or {}
    d2048_tl = timeline.get("d2048") or {}
    text = f"""# OPT-115 — Complete-request memory traffic and conditional bounds

Status: **{payload.get("status", "pending")}**. A/A verdict
`{payload.get("aa_verdict", "unmeasured")}`. Hardware executed
`{payload.get("hardware_executed")}`.

{PROOF}.

## Sitting identity

- device: {identity.get("device")}
- llama_revision (pinned authority): `{identity.get("llama_revision")}`
- llama inspection HEAD (not the pin): `{identity.get("llama_inspection_head")}`
- ds4 inspection HEAD: `{identity.get("ds4_inspection_head")}`
- gguf_sha256: `{identity.get("gguf_sha256")}`
- source: `{identity.get("source_revision")}` ({identity.get("source_state")})
- nvccflags: `{identity.get("nvccflags")}`
- execution_graphs: `{identity.get("execution_graphs")}`
- gpu_blocker: {payload.get("gpu_blocker")}

post113_authenticated=`{auth.get("ok")}`; Q4=`{auth.get("llama_q4k_mmvq")}`;
prompt attention=`{auth.get("opt111_base")}`; decode attention
`{auth.get("hybrid_crossover")}`; GDN `{auth.get("sequential_gdn")}`;
graphs `{auth.get("ffn_only_graphs")}`. OPT-113 tok/s numbers are historical
observations, not this sitting's paired control.

## A/A same-binary post113 control

3 warmups + 10 interleaved AB/BA pairs. Combined verdict
**{payload.get("aa_verdict")}**.

| Workload | A tok/s | B tok/s | geo ratio | Verdict |
|---|---:|---:|---:|---|
{chr(10).join(aa_rows)}

Session loop (sync → greedy sample → eval): 
status=`{(payload.get("session_loop") or {}).get("status")}`;
TTFT_ms=`{_fmt((payload.get("session_loop") or {}).get("ttft_ms"))}`;
p50=`{_fmt((payload.get("session_loop") or {}).get("p50_ms"))}`;
p95=`{_fmt((payload.get("session_loop") or {}).get("p95_ms"))}`.
CLI/server cancellation poll is `EvalControl::poll` at layer/chunk boundaries;
benchmark `execute_token`/`sync_tokens` paths omit the public Session poll
unless an `EvalControl` is supplied.

## Long-context probes

Capacity 131072. Capacity alone is not evidence of reading a populated cache.
These probes are separately labeled and are not D128/D2048 wins.

| Probe | prefix | outputs | tok/s | TTFT ms | p50 ms | p95 ms | status |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(long_rows)}

## Bytes / lifetime (decode D128 tensor-derived compulsory)

Units are bytes. Peak ms uses 1792 GB/s listed peak.

| Phase | bytes/token | units | peak_ms |
|---|---:|---|---:|
{chr(10).join(byte_rows)}

Resident 128K ledger: model
{(traffic.get("memory_ledger") or {}).get("resident_model_bytes")} B, KV
{(traffic.get("memory_ledger") or {}).get("attention_kv_bytes")} B, GDN
{(traffic.get("memory_ledger") or {}).get("gdn_state_bytes")} B, workspace
{(traffic.get("memory_ledger") or {}).get("workspace_bytes")} B, reserve
{(traffic.get("memory_ledger") or {}).get("required_reserve_bytes")} B.

## Timeline coverage

Method: CUDA-event unions; overlapping intervals are not summed.

| Prefix | GPU busy ms | GPU idle ms | host union ms | unclassified ms | coverage |
|---|---:|---:|---:|---:|---:|
| D128 | {_fmt(d128_tl.get("gpu_busy_union_ms"))} | {_fmt(d128_tl.get("gpu_idle_gap_ms"))} | {_fmt(d128_tl.get("host_overlap_union_ms"))} | {_fmt(d128_tl.get("unclassified_ms"))} | {_fmt(d128_tl.get("interval_coverage"))} |
| D2048 | {_fmt(d2048_tl.get("gpu_busy_union_ms"))} | {_fmt(d2048_tl.get("gpu_idle_gap_ms"))} | {_fmt(d2048_tl.get("host_overlap_union_ms"))} | {_fmt(d2048_tl.get("unclassified_ms"))} | {_fmt(d2048_tl.get("interval_coverage"))} |

## Bandwidth bounds

D128 overlap-aware peak bound
`{_fmt((bounds.get("d128") or {}).get("overlap_aware_peak_ms"))}` ms.
Independent phase sums are **not** a bound.
Assumptions: listed 1792 GB/s peak; embedding is one Q4_K row; GDN decode
does not copy the 8 GiB KV cache; measured sustainable bandwidth is
`{(bounds.get("d128") or {}).get("measured_sustainable_bandwidth_gb_s")}`.

## Pinned-source comparison

Pinned llama `{comparison.get("pinned_llama_revision")}`. Inspection llama HEAD
`{comparison.get("llama_inspection_head_actual")}` is_not_pin=
`{comparison.get("llama_inspection_is_not_pin")}`. ds4 HEAD
`{comparison.get("ds4_inspection_head_actual")}`.
{comparison.get("architecture_warning")}

## Ranked OPT-117–122 opportunities

| Rank | Task | Verdict | removable bytes upper | peak ms upper | primary |
|---|---|---|---:|---:|---|
{chr(10).join(rank_rows)}

No keep. production_kept={payload.get("production_kept")}.
claims_throughput=false. claims_performance_improvement=false.

## Shared keep protocol (frozen for OPT-117–123)

parent=`{protocol.get("parent")}`; quality_against=`{protocol.get("quality_against")}`;
screen {protocol.get("screen_warmups")} warmups / {protocol.get("screen_pairs")} pairs;
acceptance {protocol.get("acceptance_warmups")} / {protocol.get("acceptance_pairs")};
decode outputs {protocol.get("decode_output_tokens")}; ratio lower bound
{protocol.get("throughput_ratio_lower_bound")}; non-target >=
{protocol.get("non_target_point_ratio_min")}; decode p95 <=
{protocol.get("decode_p95_ratio_max")}× parent. Memory-only is not a
throughput keep.
"""
    REPORT.write_text(text, encoding="utf-8")


def run_preflight(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    iteration = load_json(ITERATION)
    validate_future_keep_policy("OPT-115", iteration)
    contract = load_contract()
    if contract["claims_throughput"] is not False:
        raise TrafficError("contract must not claim throughput")
    if contract["claims_performance_improvement"] is not False:
        raise TrafficError("contract must not claim performance improvement")
    if contract["production_selector_change"] is not False:
        raise TrafficError("OPT-115 must not change production selectors")
    auth = authenticate_post113()
    if not auth["ok"]:
        raise TrafficError(f"post113_selected mismatch {auth['mismatches']}")
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
    payload["post113_authenticated"] = auth
    payload["preflight"] = {
        "identity": identity,
        "post113_authenticated": auth,
        "gpu_available": available,
        "gpu_blocker": blocker or None,
        "shared_keep_protocol": shared_keep_protocol(),
        "admission": admission,
        "native_counts": observed,
    }
    persist(payload)
    dump_json(run_dir / "preflight.json", payload["preflight"])
    return payload["preflight"]


def run_aa_native(*, name: str, runner: NativeRunner, run_dir: Path) -> dict[str, Any]:
    prefix = AA_PREFIX[name]
    tokens = 0 if name == "p4096" else DECODE_TOKENS
    completed = runner(
        [
            f"./{NATIVE}",
            "--workload",
            "request-trace",
            "--prefix",
            str(prefix),
            "--warmups",
            str(WARMUPS),
            "--samples",
            str(PAIR_COUNT),
            "--tokens",
            str(tokens if name == "p4096" else max(tokens, 1)),
            MODEL,
        ],
        "screen",
    )
    (run_dir / f"aa-{name}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    payload = parse_prefixed(completed.stdout, "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=")
    parsed = parse_aa_pairs(payload)
    verdict = aa_verdict_for_workload(parsed["a_tok_s"], parsed["b_tok_s"])
    return {
        "workload": name,
        "prefix": prefix,
        "warmups": WARMUPS,
        "pairs": PAIR_COUNT,
        "decode_tokens": tokens,
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


def run_session_loop_native(*, runner: NativeRunner, run_dir: Path) -> dict[str, Any]:
    completed = runner(
        [
            f"./{NATIVE}",
            "--workload",
            "session-loop",
            "--prefix",
            "4096",
            "--tokens",
            str(SESSION_LOOP_TOKENS),
            MODEL,
        ],
        "screen",
    )
    (run_dir / "session-loop.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    payload = parse_prefixed(completed.stdout, "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=")
    payload["status"] = "measured" if payload else "unmeasured"
    return payload or {"status": "unmeasured", "reason": "no_native_payload"}


def run_request_trace(
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
        blocks["session_loop"] = {
            "status": "unmeasured",
            "reason": blocker or "gpu_unavailable",
        }
    else:
        for name in AA_WORKLOADS:
            try:
                blocks[name] = run_aa_native(name=name, runner=execute, run_dir=run_dir)
            except TrafficError as exc:
                blocks[name] = empty_aa_workload(name, reason=str(exc)[:2000])
        try:
            blocks["session_loop"] = run_session_loop_native(
                runner=execute, run_dir=run_dir
            )
        except TrafficError as exc:
            blocks["session_loop"] = {
                "status": "unmeasured",
                "reason": str(exc)[:2000],
            }
    iteration = load_json(ITERATION)
    observed = planned_observation("request-trace", mode)
    observed["keep"] = False
    blocks["native_counts"] = observed
    blocks["admission"] = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="request-trace",
        workload=workload_for_mode(iteration["workloads"]["request-trace"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    payload = merge_phase(fixture, "request-trace", blocks)
    payload["aa_verdict"] = combine_aa_verdicts(blocks)
    payload["hardware_executed"] = execute is not None
    if execute is not None:
        payload["gpu_blocker"] = None
        payload["status"] = "request_traced"
    persist(payload)
    dump_json(run_dir / "request-trace.json", blocks)
    return blocks


def run_long_native(
    *, name: str, runner: NativeRunner, run_dir: Path
) -> dict[str, Any]:
    prefix = LONG_PREFIX[name]
    completed = runner(
        [
            f"./{NATIVE}",
            "--workload",
            "long-context",
            "--prefix",
            str(prefix),
            "--capacity",
            str(CAPACITY_128K),
            "--warmups",
            "1",
            "--samples",
            "1",
            "--tokens",
            str(LONG_DECODE_TOKENS),
            MODEL,
        ],
        "screen",
    )
    (run_dir / f"long-{name}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    payload = parse_prefixed(completed.stdout, "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=")
    return {
        "probe": name,
        "prefix": prefix,
        "capacity": CAPACITY_128K,
        "decode_tokens": LONG_DECODE_TOKENS,
        "populated_cache": True,
        "capacity_alone_is_not_traffic": True,
        "status": "measured" if payload else "unmeasured",
        "tok_s": payload.get("tok_s"),
        "wall_ms": payload.get("wall_ms"),
        "ttft_ms": payload.get("ttft_ms"),
        "p50_ms": payload.get("p50_ms"),
        "p95_ms": payload.get("p95_ms"),
        "prefill_wall_ms": payload.get("prefill_wall_ms"),
        "native": payload,
    }


def run_long_context(
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
        for name in LONG_PROBES:
            blocks[name] = empty_long_probe(name, reason=blocker or "gpu_unavailable")
    else:
        for name in LONG_PROBES:
            try:
                blocks[name] = run_long_native(
                    name=name, runner=execute, run_dir=run_dir
                )
            except TrafficError as exc:
                probe = empty_long_probe(name, reason=str(exc)[:2000])
                probe["status"] = "blocked"
                blocks[name] = probe
    iteration = load_json(ITERATION)
    observed = planned_observation("long-context", mode)
    observed["keep"] = False
    blocks["native_counts"] = observed
    blocks["admission"] = validate_performance_admission(
        iteration,
        mode=mode,
        workload_name="long-context",
        workload=workload_for_mode(iteration["workloads"]["long-context"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    payload = merge_phase(fixture, "long-context", blocks)
    payload["hardware_executed"] = (
        bool(payload.get("hardware_executed")) or execute is not None
    )
    persist(payload)
    dump_json(run_dir / "long-context.json", blocks)
    return blocks


def run_timeline_native(
    *, prefix: int, runner: NativeRunner, run_dir: Path
) -> dict[str, Any]:
    completed = runner(
        [
            f"./{NATIVE}",
            "--workload",
            "traffic-timeline",
            "--prefix",
            str(prefix),
            "--warmups",
            "1",
            "--samples",
            "1",
            "--tokens",
            "1",
            MODEL,
        ],
        "screen",
    )
    text = completed.stdout + completed.stderr
    (run_dir / f"timeline-d{prefix}.txt").write_text(text, encoding="utf-8")
    summary = parse_prefixed(completed.stdout, "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=")
    records = parse_records_blob(completed.stdout)
    dump_json(run_dir / f"timeline-d{prefix}-records.json", records)
    split = classify_timeline(
        records,
        wall_ms=summary.get("wall_ms"),
        tokens=1,
    )
    split["valid"] = bool(records) and not bool(summary.get("pool_overflow"))
    split["prefix"] = prefix
    split["uninstrumented_wall_ms"] = summary.get("uninstrumented_wall_ms")
    split["instrumented_wall_ms"] = summary.get("instrumented_wall_ms")
    split["profiler_perturbation_ms"] = summary.get("profiler_perturbation_ms")
    split["native"] = summary
    return split


def run_traffic_timeline(
    mode: str, run_dir: Path, runner: NativeRunner | None
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    available, blocker = gpu_available()
    execute = runner
    if execute is None and available:
        execute = default_native_runner
    timeline: dict[str, Any] = {}
    overlay: dict[str, Any] = {}
    if execute is None:
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            timeline[name] = {
                "valid": False,
                "reason": blocker or "gpu_unavailable",
                "prefix": prefix,
                "interval_coverage": 0.0,
            }
    else:
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            try:
                timeline[name] = run_timeline_native(
                    prefix=prefix, runner=execute, run_dir=run_dir
                )
                native = timeline[name].get("native") or {}
                overlay[name] = {
                    "logits_d2h_bytes": native.get("logits_d2h_bytes"),
                    "hidden_d2h_bytes": native.get("hidden_d2h_bytes"),
                    "kv_append_bytes": native.get("kv_append_bytes"),
                    "gdn_state_bytes": native.get("gdn_state_bytes"),
                    "measured": True,
                }
            except TrafficError as exc:
                timeline[name] = {
                    "valid": False,
                    "reason": str(exc)[:2000],
                    "prefix": prefix,
                    "interval_coverage": 0.0,
                }
    iteration = load_json(ITERATION)
    observed = planned_observation("traffic-timeline", mode)
    observed["keep"] = False
    block = {
        "timeline": timeline,
        "traffic_overlay": overlay,
        "native_counts": observed,
        "admission": validate_performance_admission(
            iteration,
            mode=mode,
            workload_name="traffic-timeline",
            workload=workload_for_mode(
                iteration["workloads"]["traffic-timeline"], mode
            ),
            stdout=json.dumps(observed),
            success=True,
        ),
        **timeline,
    }
    payload = merge_phase(fixture, "traffic-timeline", block)
    payload["hardware_executed"] = (
        bool(payload.get("hardware_executed")) or execute is not None
    )
    persist(payload)
    dump_json(run_dir / "traffic-timeline.json", block)
    return block


def run_ranking(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    traffic = architecture_traffic_map()
    existing = fixture.get("traffic_map") or {}
    if existing.get("measured_overlay"):
        traffic["measured_overlay"] = existing["measured_overlay"]
    timeline = fixture.get("timeline") or {}
    bounds = {
        name: bandwidth_bounds(
            dict((traffic.get(name) or {}).get("phases") or {}),
            wall_ms=(
                (timeline.get(name) or {}).get("wall_ms")
                if name in ("d128", "d2048")
                else None
            ),
            prefix=AA_PREFIX.get(name) or LONG_PREFIX.get(name),
        )
        for name in (*AA_WORKLOADS, *LONG_PROBES)
    }
    comparison = source_comparison_matrix()
    ranking = rank_opportunities(
        traffic=traffic, timeline=timeline, bounds=bounds["d128"]
    )
    iteration = load_json(ITERATION)
    observed = planned_observation("ranking", mode)
    block = {
        "traffic_map": traffic,
        "bandwidth_bounds": bounds,
        "source_comparison": comparison,
        "ranked_opportunities": ranking,
        "shared_keep_protocol": shared_keep_protocol(),
        "native_counts": observed,
        "admission": validate_performance_admission(
            iteration,
            mode=mode,
            workload_name="ranking",
            workload=workload_for_mode(iteration["workloads"]["ranking"], mode),
            stdout=json.dumps(observed),
            success=True,
        ),
    }
    payload = merge_phase(fixture, "ranking", block)
    persist(payload)
    dump_json(run_dir / "ranking.json", block)
    return block


def validate_fixture(payload: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    if missing:
        raise TrafficError(f"fixture missing keys {missing}")
    if payload.get("task") != "OPT-115":
        raise TrafficError("fixture task is not OPT-115")
    if payload.get("claims_throughput") is True:
        raise TrafficError("OPT-115 must not claim throughput")
    if payload.get("claims_performance_improvement") is True:
        raise TrafficError("OPT-115 must not claim performance improvement")
    protocol = payload.get("shared_keep_protocol") or {}
    if protocol.get("parent") != "post113_selected":
        raise TrafficError("keep protocol parent must be post113_selected")
    traffic = payload.get("traffic_map") or {}
    d128 = traffic.get("d128") or {}
    phases = d128.get("phases") or {}
    for name in TRAFFIC_PHASES:
        if name not in phases:
            raise TrafficError(f"traffic map missing phase {name}")
        if int(phases[name]) < 0:
            raise TrafficError(f"negative bytes in {name}")
    for name, block in (payload.get("timeline") or {}).items():
        if not isinstance(block, Mapping):
            continue
        coverage = block.get("interval_coverage")
        if coverage is not None and float(coverage) > 1.0 + 1e-6:
            raise TrafficError(f"{name} interval coverage exceeds 1")
        if block.get("overlap_not_summed") is False:
            raise TrafficError("overlapping intervals were summed")
    ranked = (payload.get("ranked_opportunities") or {}).get("ranked") or []
    names = [row.get("task") for row in ranked]
    for task in RANKED_TASKS:
        if task not in names:
            raise TrafficError(f"missing ranked task {task}")
    report = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else ""
    if "1792" not in report:
        raise TrafficError("report must record 1792 GB/s peak assumption")
    if "post113" not in report.casefold():
        raise TrafficError("report must authenticate post113")
    if "claims_throughput=false" not in report.casefold().replace(" ", ""):
        if "claims_throughput=false" not in report:
            raise TrafficError("report must deny throughput claims")


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
        raise TrafficError(f"unknown phase {phase}")
    if phase == "preflight":
        return run_preflight(mode, run_dir)
    if phase == "request-trace":
        return run_request_trace(mode, run_dir, runner)
    if phase == "long-context":
        return run_long_context(mode, run_dir, runner)
    if phase == "traffic-timeline":
        return run_traffic_timeline(mode, run_dir, runner)
    if phase == "ranking":
        return run_ranking(mode, run_dir)
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
    print(json.dumps({"task": "OPT-115", "phase": args.phase, "ok": True}, indent=2))
    dump_json(run_dir / f"{args.phase}-result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
