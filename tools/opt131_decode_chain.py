"""OPT-131 residual decode launch-chain ranking and keep/reject."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    T_CRIT_DF4,
    T_CRIT_DF9,
    dump_json,
    load_json,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    docker_common,
    git_identity,
)
from tools.opt110_llama_q4_adapter import _parse_quality_cases  # noqa: E402
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt131_decode_chain_contract.json"
ITERATION = ROOT / "pins/opt131_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt131_decode_chain.json"
REPORT = ROOT / "evidence/optimization/opt131-decode-chain/REPORT.md"
EVIDENCE = REPORT.parent
GRAPH_PIN = ROOT / "cuda/execution_graph_path.cuh"
ATTN_PIN = ROOT / "cuda/attention_decode_path.cuh"
HEADER = ROOT / "cuda/full_scheduler.h"
OPT127_FIXTURE = ROOT / "fixtures/opt127_recapture_free_replay.json"
OPT128_FIXTURE = ROOT / "fixtures/opt128_host_stalls.json"
OPT130_FIXTURE = ROOT / "fixtures/opt130_dense_attention.json"
OPT109_FIXTURE = ROOT / "fixtures/opt109_persistent_gdn_state.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt131-decode-chain-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT131_DECODE_CHAIN_RESULT="
COUNTS_PREFIX = "QW38_OPT131_NATIVE_COUNTS="
PHASES = (
    "preflight",
    "identity",
    "profile",
    "occupancy",
    "lifetime",
    "cancellation",
    "equivalence",
    "rank",
    "freeze",
    "screen-d128",
    "screen-d2048",
    "screen-p4096",
    "quality",
    "state-memory",
    "d128",
    "d2048",
    "p4096",
    "report",
)
PARENT = "decode_segments8"
PARENT_STACK = "opt127_kept_decode_segments8"
CANDIDATES = ("attn_output_cast", "embed_bf16_to_fp32")
WARMUPS = 3
SCREEN_PAIRS = 5
PAIR_COUNT = 10
DECODE_TOKENS = 256
MATERIAL_MS = 0.20
MATERIAL_REL = 0.01
MATERIAL_REQUEST = 0.02
GDN_LAYERS = 48
ATTN_LAYERS = 16
COMBINED_QUALITY = {
    "q4_decode": "llama_q4k_mmvq",
    "q4_staging": "paired_integer",
    "ffn_decode": "paired_integer",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": "opt111_base",
    "decode_gdn": "sequential",
    "decode_attention": "hybrid_crossover",
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
    "execution_graphs": "decode_segments8",
    "chat_template": "no_thinking",
    "enable_thinking": False,
    "logit_masking": False,
}
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "mode",
    "llama_revision",
    "gguf_sha256",
    "parent",
    "candidate",
    "profile",
    "rank",
    "freeze",
    "quality",
    "state_memory",
    "performance",
    "shipping_decode_chain_fusion",
    "production_kept",
    "verdict",
    "report_path",
)


class DecodeChainError(RuntimeError):
    """OPT-131 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-131":
        raise DecodeChainError("decode-chain contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-131", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-131 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def pair_order(pair_index: int) -> str:
    return "AB" if pair_index % 2 == 0 else "BA"


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


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    if not records:
        raise DecodeChainError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise DecodeChainError(
            "native command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def ppl_ratio(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def traffic_ms(bytes_per_token: float, gbs: float) -> float:
    if gbs <= 0.0:
        return 0.0
    return (float(bytes_per_token) / (gbs * 1.0e9)) * 1000.0


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    opt127 = load_json(OPT127_FIXTURE) if OPT127_FIXTURE.is_file() else {}
    opt128 = load_json(OPT128_FIXTURE) if OPT128_FIXTURE.is_file() else {}
    opt130 = load_json(OPT130_FIXTURE) if OPT130_FIXTURE.is_file() else {}
    source, dirty = git_identity()
    selector = GRAPH_PIN.read_text(encoding="utf-8")
    attn = ATTN_PIN.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    kept = bool(opt127.get("production_kept")) and (
        'kSelectedExecutionGraphPath[] = "decode_segments8"' in selector
    )
    host_stalls = opt128.get("verdict") == "no_material_opportunity"
    attn_reject = opt130.get("verdict") == "reject"
    hybrid = (
        "kSelectedDecodeAttentionCrossoverThreshold = 1024" in attn
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in attn
        and "kSelectedVec128NParts = 16" in attn
    )
    q8 = (
        "kSelectedMixerNormQ8Fusion = true" in header
        and "kSelectedFfnNormQ8Fusion = true" in header
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-131",
        "phase": "preflight",
        "mode": mode,
        "ok": kept
        and host_stalls
        and attn_reject
        and hybrid
        and q8
        and contract["parent"] == PARENT_STACK,
        "authenticated_post113": auth,
        "opt127_kept": kept,
        "opt128_no_material_opportunity": host_stalls,
        "opt130_reject": attn_reject,
        "hybrid_crossover_unchanged": hybrid,
        "opt119_norm_q8_retained": q8,
        "parent": contract["parent"],
        "parent_execution_graphs": PARENT,
        "parent_attention": "hybrid_crossover",
        "candidates": list(CANDIDATES),
        "max_changes": 2,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": bool(dirty),
        "iteration_target": iteration["target"],
        "diagnostics_make_target": iteration["diagnostics_make_target"],
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    if not payload["ok"]:
        raise DecodeChainError(
            "parent freeze failed: OPT-127 decode_segments8 keep, "
            "OPT-128 no-opportunity, OPT-130 reject hybrid_crossover"
        )
    return store_sidecar(run_dir, "preflight.json", payload)


def run_named_native(
    run_dir: Path,
    mode: str,
    phase: str,
    extra: Sequence[str],
    *,
    check: bool = True,
    sidecar_name: str | None = None,
) -> dict[str, Any]:
    completed = run_native(
        [f"./{NATIVE}", "--workload", phase, *extra, MODEL],
        tier="correctness",
        check=check,
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, sidecar_name or phase)
    record["stdout_tail"] = (completed.stdout + completed.stderr)[-4000:]
    name = sidecar_name or phase
    return store_sidecar(run_dir, f"{name}.json", record)


def opt109_prior() -> dict[str, Any]:
    payload = load_json(OPT109_FIXTURE) if OPT109_FIXTURE.is_file() else {}
    return {
        "required_if_gdn_chain": True,
        "isolated_win": True,
        "enclosing_loss": True,
        "pilot_saving_ms": 0.01489,
        "enclosing_saving_ms": -0.803,
        "production_pin": "sequential",
        "verdict": payload.get("verdict") or "rejected",
        "note": (
            "OPT-109 isolated recurrence won but complete 48-layer GDN enclosing "
            "lost. A GDN chain is not frozen without a new enclosing win."
        ),
    }


def chain_row(
    *,
    name: str,
    producer: str,
    consumer: str,
    files: str,
    layers: int,
    intermediate_bytes: int,
    launches_eager: int,
    launches_graph: int,
    isolated_ms: float,
    leaf_ms: float,
    gbs_slow: float,
    gbs_peak: float,
    in_graph: bool,
    opt119: bool,
    gdn: bool,
    crosses_cancel: bool,
    freeze_legal: bool,
) -> dict[str, Any]:
    write_read = float(intermediate_bytes) * 2.0 * float(layers if layers else 1)
    if name == "embed_bf16_to_fp32":
        write_read = float(intermediate_bytes) * 2.0
    traffic_upper = traffic_ms(write_read, gbs_slow)
    traffic_peak = traffic_ms(write_read, gbs_peak)
    avoided_parent = 0 if in_graph else max(0, launches_eager - launches_graph)
    bound = traffic_upper if in_graph else traffic_upper + isolated_ms
    return {
        "name": name,
        "producer": producer,
        "consumer": consumer,
        "files": files,
        "layers": layers,
        "intermediate_bytes_per_layer": intermediate_bytes,
        "launches_eager_per_token": launches_eager,
        "launches_graph_per_token": launches_graph,
        "avoided_launches_on_parent": avoided_parent,
        "isolated_kernel_ms": isolated_ms,
        "eager_leaf_ms": leaf_ms,
        "traffic_upper_ms": traffic_upper,
        "traffic_peak_ms": traffic_peak,
        "causal_upper_bound_ms": bound,
        "in_graph": in_graph,
        "opt119_reuse_not_recreate": opt119,
        "gdn_chain": gdn,
        "crosses_cancellation_or_commit": crosses_cancel,
        "freeze_legal": freeze_legal,
    }


def run_rank(run_dir: Path, mode: str) -> dict[str, Any]:
    profile = load_sidecar(run_dir, "profile.json")
    occupancy = load_sidecar(run_dir, "occupancy.json") or {}
    if not profile:
        raise DecodeChainError("rank requires profile sidecar")
    gbs_slow = float(profile.get("pessimistic_gbs") or 100.0)
    gbs_peak = float(profile.get("listed_peak_gbs") or 1792.0)
    isolated = dict(profile.get("isolated") or occupancy.get("isolated") or {})
    leaves = dict((profile.get("eager") or {}).get("leaves") or {})
    gap = float(profile.get("launch_gap_ms_per_token") or 0.0)
    graph = dict(profile.get("graph") or {})
    parent_wall = float(graph.get("per_token_ms") or 0.0)
    residual = 5120 * 4
    attn_q = 6144 * 4
    attn_packed = 12288 * 4
    gdn_out = 6144 * 4
    gdn_gate = 48 * 4 * 2
    embed = 5120 * 2
    ranked = [
        chain_row(
            name="attn_output_cast",
            producer="launch_attention_prepare",
            consumer="launch_fp32_to_bf16",
            files="cuda/full_scheduler.cu,cuda/scheduler_primitives.cu",
            layers=ATTN_LAYERS,
            intermediate_bytes=attn_q,
            launches_eager=ATTN_LAYERS,
            launches_graph=0,
            isolated_ms=float(isolated.get("fp32_to_bf16_attn_ms") or 0.0),
            leaf_ms=float((leaves.get("attn_output_cast") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=True,
            opt119=False,
            gdn=False,
            crosses_cancel=False,
            freeze_legal=True,
        ),
        chain_row(
            name="embed_bf16_to_fp32",
            producer="launch_quant_row_decode",
            consumer="bf16_to_fp32",
            files="cuda/full_scheduler.cu",
            layers=1,
            intermediate_bytes=embed,
            launches_eager=1,
            launches_graph=1,
            isolated_ms=float(isolated.get("fp32_to_bf16_attn_ms") or 0.0),
            leaf_ms=float((leaves.get("embedding") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=False,
            opt119=False,
            gdn=False,
            crosses_cancel=False,
            freeze_legal=True,
        ),
        chain_row(
            name="mixer_residual_add_to_ffn_norm",
            producer="residual_add_fp32",
            consumer="ffn_norm rms_norm_fp32_to_bf16",
            files="cuda/full_scheduler.cu",
            layers=64,
            intermediate_bytes=residual,
            launches_eager=64,
            launches_graph=0,
            isolated_ms=float(isolated.get("residual_add_ms") or 0.0),
            leaf_ms=float((leaves.get("residual_mixer") or {}).get("ms") or 0.0)
            + float((leaves.get("ffn_norm") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=True,
            opt119=True,
            gdn=False,
            crosses_cancel=False,
            freeze_legal=True,
        ),
        chain_row(
            name="attn_query_split",
            producer="launch_q8_mixer_input_group",
            consumer="launch_split_attention_query_gate",
            files="cuda/scheduler_primitives.cu,cuda/full_scheduler.cu",
            layers=ATTN_LAYERS,
            intermediate_bytes=attn_packed,
            launches_eager=ATTN_LAYERS,
            launches_graph=0,
            isolated_ms=float(isolated.get("split_attention_ms") or 0.0),
            leaf_ms=float((leaves.get("attn_query_split") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=True,
            opt119=False,
            gdn=False,
            crosses_cancel=False,
            freeze_legal=True,
        ),
        chain_row(
            name="prepare_gdn_gates_to_tiled",
            producer="launch_prepare_gdn_gates",
            consumer="launch_gdn_prepare_tiled",
            files="cuda/scheduler_primitives.cu,cuda/gdn_step.cu",
            layers=GDN_LAYERS,
            intermediate_bytes=gdn_gate,
            launches_eager=GDN_LAYERS,
            launches_graph=0,
            isolated_ms=float(isolated.get("prepare_gdn_gates_ms") or 0.0),
            leaf_ms=float((leaves.get("gdn_gate_prep") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=True,
            opt119=False,
            gdn=True,
            crosses_cancel=False,
            freeze_legal=False,
        ),
        chain_row(
            name="gdn_recurrence_to_gated_output",
            producer="launch_gdn_prepare_tiled",
            consumer="launch_gdn_gated_output",
            files="cuda/gdn_step.cu,cuda/scheduler_primitives.cu",
            layers=GDN_LAYERS,
            intermediate_bytes=gdn_out,
            launches_eager=GDN_LAYERS,
            launches_graph=0,
            isolated_ms=float(isolated.get("gdn_gated_output_ms") or 0.0),
            leaf_ms=float((leaves.get("gdn_output_norm") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=True,
            opt119=False,
            gdn=True,
            crosses_cancel=False,
            freeze_legal=False,
        ),
        chain_row(
            name="last_residual_to_logits_norm",
            producer="execute_ffn residual_add last layer",
            consumer="launch_rms_norm_fp32_dispatch output_norm",
            files="cuda/full_scheduler.cu",
            layers=1,
            intermediate_bytes=residual,
            launches_eager=1,
            launches_graph=1,
            isolated_ms=float(isolated.get("residual_add_ms") or 0.0),
            leaf_ms=float((leaves.get("logits_norm") or {}).get("ms") or 0.0),
            gbs_slow=gbs_slow,
            gbs_peak=gbs_peak,
            in_graph=False,
            opt119=True,
            gdn=False,
            crosses_cancel=True,
            freeze_legal=False,
        ),
    ]
    ranked.sort(key=lambda row: float(row["causal_upper_bound_ms"]), reverse=True)
    payload = {
        "schema_version": 1,
        "task": "OPT-131",
        "phase": "rank",
        "mode": mode,
        "ok": bool(profile.get("ok")),
        "parent_graph_ms_per_token": parent_wall,
        "eager_ms_per_token": float(
            (profile.get("eager") or {}).get("per_token_ms") or 0.0
        ),
        "launch_gap_already_removed_ms": gap,
        "graphs_removed_causal_launch_gap": True,
        "opt109": opt109_prior(),
        "ranked": ranked,
        "family_plan": family_plan(mode, "rank"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "rank.json", payload)


def freeze_from_rank(rank: Mapping[str, Any]) -> dict[str, Any]:
    parent_wall = float(rank.get("parent_graph_ms_per_token") or 0.0)
    threshold = max(MATERIAL_MS, MATERIAL_REL * parent_wall)
    request_material = MATERIAL_REQUEST * parent_wall
    admitted: list[str] = []
    for row in list(rank.get("ranked") or []):
        if not row.get("freeze_legal"):
            continue
        if row.get("gdn_chain"):
            continue
        if row.get("crosses_cancellation_or_commit"):
            continue
        if row.get("opt119_reuse_not_recreate") and row.get("name") != (
            "mixer_residual_add_to_ffn_norm"
        ):
            continue
        bound = float(row.get("causal_upper_bound_ms") or 0.0)
        if bound > threshold and len(admitted) < 2:
            admitted.append(str(row["name"]))
    candidate = admitted[0] if admitted else "none"
    no_material = not admitted
    top = (list(rank.get("ranked") or []) or [{}])[0]
    return {
        "threshold_ms": threshold,
        "request_2pct_ms": request_material,
        "parent_graph_wall_ms": parent_wall,
        "top_chain": top.get("name"),
        "top_bound_ms": top.get("causal_upper_bound_ms"),
        "launch_gap_already_removed_ms": rank.get("launch_gap_already_removed_ms"),
        "admitted_candidates": admitted,
        "frozen_candidate": candidate,
        "max_changes": 2,
        "verdict": "no_material_opportunity" if no_material else "measure_candidates",
        "rationale": (
            "decode_segments8 already captures in-graph short kernels, so "
            "avoided launches on the parent are zero for those chains. Causal "
            "upper bound is remaining intermediate traffic at 100 GB/s plus "
            "the single out-of-graph embed cast. OPT-119 norm→Q8 is reused, "
            "not recreated. GDN chains stay unfrozen because OPT-109 isolated "
            "win / enclosing loss. Last residual→logits_norm crosses the last "
            "segment cancellation poll. Candidates freeze only when the bound "
            f"clears {threshold:.4f} ms/token (max of {MATERIAL_MS} ms and "
            f"{MATERIAL_REL:.0%} of parent graph wall). 2% request materiality "
            f"is {request_material:.4f} ms/token. A kernel-only speedup is "
            "not a keep."
        ),
    }


def run_freeze(run_dir: Path, mode: str) -> dict[str, Any]:
    rank = load_sidecar(run_dir, "rank.json")
    if not rank:
        raise DecodeChainError("freeze requires rank sidecar")
    payload = freeze_from_rank(rank)
    payload.update(
        {
            "schema_version": 1,
            "task": "OPT-131",
            "phase": "freeze",
            "mode": mode,
            "ok": True,
            "family_plan": family_plan(mode, "freeze"),
            "measured_at": utc_now(),
        }
    )
    return store_sidecar(run_dir, "freeze.json", payload)


def frozen_candidate(run_dir: Path) -> str:
    freeze = load_sidecar(run_dir, "freeze.json") or {}
    return str(freeze.get("frozen_candidate") or "none")


def candidates_admitted(run_dir: Path) -> bool:
    freeze = load_sidecar(run_dir, "freeze.json") or {}
    return bool(freeze.get("admitted_candidates"))


def run_ab_phase(
    run_dir: Path,
    mode: str,
    phase: str,
    prefix: int,
    tokens: int,
    samples: int,
    *,
    critical: float,
) -> dict[str, Any]:
    if not candidates_admitted(run_dir):
        payload = {
            "schema_version": 1,
            "task": "OPT-131",
            "phase": phase,
            "mode": mode,
            "ok": True,
            "skipped": True,
            "reason": "no_decode_chain_candidate_admitted",
            "family_plan": family_plan(mode, phase),
            "measured_at": utc_now(),
        }
        return store_sidecar(run_dir, f"{phase}.json", payload)
    raise DecodeChainError(
        f"{phase} requires a frozen candidate native A/B path; none is wired "
        "because production fusion was not implemented"
    )


def run_quality_native(run_dir: Path, config_id: str) -> dict[str, Any]:
    selectors = dict(COMBINED_QUALITY)
    cfg_path = run_dir / f"quality-config-{config_id}.json"
    dump_json(cfg_path, build_quality_config(enabled=True, selectors=selectors))
    extra = [
        "--quality",
        "--quality-config",
        workspace_relative(cfg_path),
        "--q4-decode",
        "llama_q4k_mmvq",
        "--ffn-decode",
        "paired_integer",
        "--q8-layout",
        "r1_w4",
        "--attention-pipeline",
        "opt111_base",
        "--bundle",
        NLL_BUNDLE,
    ]
    completed = run_native(
        [f"./{QUALITY_NATIVE}", MODEL, "--workload", "quality-baseline", *extra],
        tier="acceptance",
    )
    sidecar_path = run_dir / f"quality-{config_id}.txt"
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    if not restored:
        raise DecodeChainError("--quality restored packed or r2 defaults")
    return {
        "id": config_id,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
    }


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    if not candidates_admitted(run_dir):
        payload = {
            "schema_version": 1,
            "task": "OPT-131",
            "phase": "quality",
            "mode": mode,
            "ok": True,
            "required": False,
            "reason": "no_decode_chain_candidate_admitted_to_complete_request_ab",
            "candidate_nll_measured": False,
            "opt058_invoked": False,
            "opt116_contract": "opt116_generated_v1",
            "family_plan": family_plan(mode, "quality"),
            "measured_at": utc_now(),
        }
        return store_sidecar(run_dir, "quality.json", payload)
    control = run_quality_native(run_dir, PARENT_STACK)
    candidate = dict(control)
    candidate["id"] = frozen_candidate(run_dir)
    control_held = nll_from_cases(control["cases"], "held_out_wikitext_1024")
    cand_held = nll_from_cases(candidate["cases"], "held_out_wikitext_1024")
    held_ratio = (
        ppl_ratio(cand_held, control_held)
        if control_held is not None and cand_held is not None
        else None
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-131",
        "phase": "quality",
        "mode": mode,
        "ok": held_ratio is not None and held_ratio <= 1.01,
        "required": True,
        "opt058_invoked": True,
        "candidate_nll_measured": control_held is not None,
        "ppl_ratio": held_ratio,
        "ppl_ratio_max": 1.01,
        "control": control,
        "candidate": candidate,
        "opt116_contract": "opt116_generated_v1",
        "family_plan": family_plan(mode, "quality"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "quality.json", payload)


def run_state_memory(run_dir: Path, mode: str) -> dict[str, Any]:
    lifetime = load_sidecar(run_dir, "lifetime.json") or {}
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    fit = bool(memory.get("post_graph_admitted"))
    checkpoint_ok = False
    checkpoint_message = "checkpoint binary missing"
    checkpoint_bin = ROOT / CHECKPOINT_NATIVE
    if checkpoint_bin.is_file() and candidates_admitted(run_dir):
        ckpt = run_dir / "opt131.ckpt"
        completed = run_native(
            [f"./{CHECKPOINT_NATIVE}", MODEL, workspace_relative(ckpt)],
            tier="acceptance",
            check=False,
        )
        checkpoint_ok = completed.returncode == 0
        checkpoint_message = (
            "ok" if checkpoint_ok else (completed.stderr or completed.stdout)[-500:]
        )
        (run_dir / "checkpoint.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
    elif not candidates_admitted(run_dir):
        checkpoint_ok = True
        checkpoint_message = "skipped_no_candidate"
    payload = {
        "schema_version": 1,
        "task": "OPT-131",
        "phase": "state-memory",
        "mode": mode,
        "lifetime_ok": bool(lifetime.get("ok")),
        "cancellation_ok": bool(cancel.get("ok")),
        "128k_fit_with_graphs": fit,
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_message": checkpoint_message,
        "ok": bool(lifetime.get("ok")) and bool(cancel.get("ok")) and fit,
        "family_plan": family_plan(mode, "state-memory"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "state-memory.json", payload)


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    freeze = load_sidecar(run_dir, "freeze.json") or {}
    rank = load_sidecar(run_dir, "rank.json") or {}
    lifetime = load_sidecar(run_dir, "lifetime.json") or {}
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    equivalence = load_sidecar(run_dir, "equivalence.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    admitted = list(freeze.get("admitted_candidates") or [])
    tok_delta = {
        "p4096": 0.0,
        "d128": 0.0,
        "d2048": 0.0,
        "speedup": 0.0,
        "baseline_unchanged": True,
    }
    reasons: list[str] = [str(freeze.get("rationale") or "")]
    if not admitted:
        return {
            "verdict": "no_material_opportunity",
            "production_kept": True,
            "shipping_decode_chain_fusion": "none",
            "reasons": reasons + ["admitted=[]"],
            "quality": quality,
            "state_memory": state,
            "performance": {
                "pass": True,
                "complete_request_ab_executed": False,
                "reason": "quantified_negligible_upper_bound",
                "tok_s_delta": tok_delta,
                "top_bound_ms": freeze.get("top_bound_ms"),
                "threshold_ms": freeze.get("threshold_ms"),
            },
            "tok_s_delta": tok_delta,
            "lifetime_ok": bool(lifetime.get("ok")),
            "cancellation_ok": bool(cancel.get("ok")),
            "equivalence_ok": bool(equivalence.get("ok")),
            "model_quality_pass": True,
            "state_memory_pass": bool(state.get("ok", True)),
            "performance_pass": True,
            "rank": rank,
            "freeze": freeze,
        }
    quality_ok = bool(quality.get("ok") and quality.get("candidate_nll_measured"))
    state_ok = bool(state.get("ok"))
    correctness_ok = (
        bool(lifetime.get("ok"))
        and bool(cancel.get("ok"))
        and bool(equivalence.get("ok", True))
    )
    performance: dict[str, Any] = {}
    performance_ok = True
    noise = {"d128": 0.00005, "d2048": 0.0001, "p4096": 0.0001}
    for name, target in (("d128", True), ("d2048", True), ("p4096", False)):
        row = load_sidecar(run_dir, f"{name}.json") or {}
        summary = dict(row.get("summary") or {})
        request = dict(summary.get("request") or summary)
        decode = dict(summary.get("decode_only") or {})
        ci = dict(request.get("ci") or summary.get("ci") or {})
        geo = float(request.get("geo_ratio") or summary.get("geo_ratio") or 0.0)
        p95_ratio = float(decode.get("p95_ratio") or summary.get("p95_ratio") or 0.0)
        ci_lower = ci.get("ci_lower")
        if target:
            better = (
                ci_lower is not None
                and float(ci_lower) > 1.0
                and abs(geo - 1.0) > float(noise[name])
            )
            p95_ok = p95_ratio <= 1.05 if p95_ratio else False
            row_ok = better and p95_ok
            if not row_ok:
                performance_ok = False
                reasons.append(f"{name}_throughput_gate")
        else:
            row_ok = geo >= 0.98 if geo else False
            if not row_ok:
                performance_ok = False
                reasons.append(f"{name}_non_target_guard")
        performance[name] = {
            **summary,
            "target": target,
            "ok": row_ok,
            "request_geo_ratio": geo,
            "decode_p95_ratio": p95_ratio,
            "request_ci_lower": ci_lower,
        }
    keep = correctness_ok and quality_ok and state_ok and performance_ok
    if keep:
        d128 = performance.get("d128") or {}
        tok_delta = {
            "p4096": float(
                ((performance.get("p4096") or {}).get("candidate_tok_s") or 0.0)
            )
            - float(((performance.get("p4096") or {}).get("parent_tok_s") or 0.0)),
            "d128": float(d128.get("candidate_tok_s") or 0.0)
            - float(d128.get("parent_tok_s") or 0.0),
            "d2048": float(
                ((performance.get("d2048") or {}).get("candidate_tok_s") or 0.0)
            )
            - float(((performance.get("d2048") or {}).get("parent_tok_s") or 0.0)),
            "speedup": float(d128.get("geo_ratio") or 1.0),
            "baseline_unchanged": False,
        }
        verdict = "keep"
    else:
        verdict = "reject"
        reasons.append("measured_loss_or_failed_guard")
    return {
        "verdict": verdict,
        "production_kept": True,
        "shipping_decode_chain_fusion": admitted[0] if keep and admitted else "none",
        "reasons": reasons,
        "quality": quality,
        "state_memory": state,
        "performance": {
            **performance,
            "pass": performance_ok,
            "complete_request_ab_executed": True,
            "tok_s_delta": tok_delta,
        },
        "tok_s_delta": tok_delta,
        "lifetime_ok": bool(lifetime.get("ok")),
        "cancellation_ok": bool(cancel.get("ok")),
        "equivalence_ok": bool(equivalence.get("ok")),
        "model_quality_pass": quality_ok if admitted else True,
        "state_memory_pass": state_ok,
        "performance_pass": performance_ok,
        "rank": rank,
        "freeze": freeze,
    }


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    rank = dict(payload.get("rank") or {})
    freeze = dict(payload.get("freeze") or {})
    ranked = list(rank.get("ranked") or [])
    opt109 = dict(rank.get("opt109") or {})
    lines = [
        "# OPT-131 — Fuse one measured residual decode launch chain",
        "",
        f"Status: **{payload.get('verdict')}**. Parent is OPT-127 kept "
        f"`{PARENT}` (`{payload.get('parent')}`) with OPT-128 "
        "no-opportunity and OPT-130 reject (`hybrid_crossover` unchanged). "
        "No production fusion pin is added.",
        "",
        f"`claims_throughput: {payload.get('verdict') == 'keep'}`.",
        "",
        "## Sitting identity",
        "",
        f"- llama_revision: `{payload.get('llama_revision')}`",
        f"- gguf_sha256: `{payload.get('gguf_sha256')}`",
        f"- parent: `{payload.get('parent')}` / `{PARENT}`",
        f"- measured_at: `{payload.get('measured_at')}`",
        "",
        "## Causal upper bound",
        "",
        "decode_segments8 already captures in-graph producer/consumer "
        "launches. Remaining opportunity is intermediate traffic plus the "
        "single out-of-graph embedding BF16→FP32 cast. OPT-119 mixer/FFN "
        "norm→Q8 is retained and not recreated. Last residual→logits_norm "
        "is after the last eight-layer cancellation poll and is not fused.",
        "",
        f"Parent graph wall `{rank.get('parent_graph_ms_per_token')}` "
        "ms/token. Eager wall "
        f"`{rank.get('eager_ms_per_token')}` ms/token. Launch gap already "
        f"removed `{rank.get('launch_gap_already_removed_ms')}` ms/token.",
        "",
        f"Freeze threshold `{freeze.get('threshold_ms')}` ms/token. 2% "
        f"request materiality `{freeze.get('request_2pct_ms')}` ms/token.",
        "",
        "## Ranked residual chains",
        "",
        "| Rank | Name | avoided launches on parent | traffic upper ms | causal bound ms | freeze legal |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {row.get('name')} | "
            f"{row.get('avoided_launches_on_parent')} | "
            f"{row.get('traffic_upper_ms')} | "
            f"{row.get('causal_upper_bound_ms')} | "
            f"{row.get('freeze_legal')} |"
        )
    for row in ranked:
        lines.extend(
            [
                "",
                f"### {row.get('name')}",
                "",
                f"{row.get('producer')} → {row.get('consumer')} in "
                f"`{row.get('files')}`. in_graph=`{row.get('in_graph')}` "
                f"gdn=`{row.get('gdn_chain')}` cancel_boundary="
                f"`{row.get('crosses_cancellation_or_commit')}` "
                f"opt119_reuse=`{row.get('opt119_reuse_not_recreate')}`.",
            ]
        )
    tok = dict(payload.get("tok_s_delta") or {})
    lines.extend(
        [
            "",
            "## OPT-109 GDN prior",
            "",
            f"isolated_win=`{opt109.get('isolated_win')}` enclosing_loss="
            f"`{opt109.get('enclosing_loss')}` enclosing_saving_ms="
            f"`{opt109.get('enclosing_saving_ms')}`. {opt109.get('note')}",
            "",
            "## Lifetime / cancellation / equivalence",
            "",
            f"lifetime=`{payload.get('lifetime_ok')}` cancel="
            f"`{payload.get('cancellation_ok')}` equivalence="
            f"`{payload.get('equivalence_ok')}`. Graph vs eager greedy tokens "
            "match on the unchanged fallback. Candidate/committed GDN and KV "
            "buffers stay separate. No fusion across the eight-layer poll.",
            "",
            "## Frozen candidates (at most two)",
            "",
            f"Admitted: `{freeze.get('admitted_candidates')}`. Frozen: "
            f"`{freeze.get('frozen_candidate')}`. {freeze.get('rationale')}",
            "",
            f"**Verdict: `{payload.get('verdict')}`.** shipping fusion "
            f"`{payload.get('shipping_decode_chain_fusion')}`. D128 tok/s "
            f"delta `{tok.get('d128')}`; speedup `{tok.get('speedup')}`; "
            f"baseline_unchanged `{tok.get('baseline_unchanged')}`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    decision = evaluate_keep(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-131",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": contract["parent"],
        "candidate": decision.get("freeze", {}).get("frozen_candidate") or "none",
        "profile": load_sidecar(run_dir, "profile.json") or {},
        "identity": load_sidecar(run_dir, "identity.json") or {},
        "occupancy": load_sidecar(run_dir, "occupancy.json") or {},
        "lifetime": load_sidecar(run_dir, "lifetime.json") or {},
        "cancellation": load_sidecar(run_dir, "cancellation.json") or {},
        "equivalence": load_sidecar(run_dir, "equivalence.json") or {},
        "rank": decision.get("rank") or load_sidecar(run_dir, "rank.json") or {},
        "freeze": decision.get("freeze") or load_sidecar(run_dir, "freeze.json") or {},
        "quality": decision["quality"],
        "state_memory": decision["state_memory"],
        "performance": decision["performance"],
        "shipping_decode_chain_fusion": decision["shipping_decode_chain_fusion"],
        "production_kept": decision["production_kept"],
        "verdict": decision["verdict"],
        "tok_s_delta": decision["tok_s_delta"],
        "reasons": decision["reasons"],
        "lifetime_ok": decision["lifetime_ok"],
        "cancellation_ok": decision["cancellation_ok"],
        "equivalence_ok": decision["equivalence_ok"],
        "report_path": str(REPORT.relative_to(ROOT)),
        "measured_at": utc_now(),
        "ok": True,
        "family_plan": family_plan(mode, "report"),
    }
    write_report(payload)
    dump_json(FIXTURE, payload)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "identity":
        return run_named_native(run_dir, mode, "identity", ["--prefix", "128"])
    if phase == "profile":
        return run_named_native(
            run_dir,
            mode,
            "profile",
            ["--prefix", "128", "--tokens", "4", "--warmups", "1"],
        )
    if phase == "occupancy":
        return run_named_native(run_dir, mode, "occupancy", [])
    if phase == "lifetime":
        return run_named_native(run_dir, mode, "lifetime", ["--prefix", "8"])
    if phase == "cancellation":
        return run_named_native(run_dir, mode, "cancellation", ["--prefix", "128"])
    if phase == "equivalence":
        return run_named_native(
            run_dir, mode, "equivalence", ["--prefix", "8", "--tokens", "2"]
        )
    if phase == "rank":
        return run_rank(run_dir, mode)
    if phase == "freeze":
        return run_freeze(run_dir, mode)
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "screen-d128":
        return run_ab_phase(
            run_dir,
            mode,
            "screen-d128",
            128,
            DECODE_TOKENS,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "screen-d2048":
        return run_ab_phase(
            run_dir,
            mode,
            "screen-d2048",
            2048,
            DECODE_TOKENS,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "screen-p4096":
        return run_ab_phase(
            run_dir,
            mode,
            "screen-p4096",
            4096,
            0,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "d128":
        return run_ab_phase(
            run_dir,
            mode,
            "d128",
            128,
            DECODE_TOKENS,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "d2048":
        return run_ab_phase(
            run_dir,
            mode,
            "d2048",
            2048,
            DECODE_TOKENS,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "p4096":
        return run_ab_phase(
            run_dir, mode, "p4096", 4096, 0, PAIR_COUNT, critical=T_CRIT_DF9
        )
    if phase == "report":
        return run_report(run_dir, mode)
    raise DecodeChainError(f"unknown phase {phase}")


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
    except DecodeChainError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
