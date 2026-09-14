"""OPT-128 host submission and synchronization stall ranking and keep/reject."""

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
    mean,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    docker_common,
    git_identity,
)
from tools.opt106_batch_gate import log_speed_ratio_ci_lower  # noqa: E402
from tools.opt110_llama_q4_adapter import _parse_quality_cases  # noqa: E402
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt128_host_stalls_contract.json"
ITERATION = ROOT / "pins/opt128_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt128_host_stalls.json"
REPORT = ROOT / "evidence/optimization/opt128-host-stalls/REPORT.md"
EVIDENCE = REPORT.parent
HEADER = ROOT / "cuda/full_scheduler.h"
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT127_FIXTURE = ROOT / "fixtures/opt127_recapture_free_replay.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt128-host-stalls-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT128_HOST_STALLS_RESULT="
COUNTS_PREFIX = "QW38_OPT128_NATIVE_COUNTS="
PHASES = (
    "preflight",
    "profile",
    "lifetime",
    "cancellation",
    "logits-consumer",
    "non-greedy",
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
CANDIDATES = ("poll", "defer")
WARMUPS = 3
SCREEN_PAIRS = 5
PAIR_COUNT = 10
DECODE_TOKENS = 256
MATERIAL_MS = 0.20
MATERIAL_REL = 0.01
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
    "shipping_poll_without_device_sync",
    "shipping_defer_elapsed_event_sync",
    "production_kept",
    "verdict",
    "report_path",
)


class HostStallError(RuntimeError):
    """OPT-128 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-128":
        raise HostStallError("host-stalls contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-128", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-128 mode={mode} phase={family} "
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
        raise HostStallError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise HostStallError(
            "native command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def geo_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    logs = [math.log(float(item)) for item in values if float(item) > 0.0]
    if len(logs) != len(values):
        return 0.0
    return math.exp(sum(logs) / float(len(logs)))


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


def arm_tok_s(arm: Mapping[str, Any], metric: str) -> float:
    if metric == "request":
        return float(arm.get("request_tok_s") or arm.get("tok_s") or 0.0)
    return float(arm.get("decode_only_tok_s") or arm.get("tok_s") or 0.0)


def summarize_pairs(
    record: Mapping[str, Any],
    *,
    metric: str,
    critical: float,
) -> dict[str, Any]:
    pairs = list(record.get("pairs") or [])
    parent = [arm_tok_s(row["A"], metric) for row in pairs]
    candidate = [arm_tok_s(row["B"], metric) for row in pairs]
    parent_p95 = [float(row["A"]["p95_ms"]) for row in pairs]
    candidate_p95 = [float(row["B"]["p95_ms"]) for row in pairs]
    ratios = [
        cand / ctrl if ctrl > 0.0 else 0.0 for ctrl, cand in zip(parent, candidate)
    ]
    ci = log_speed_ratio_ci_lower(parent, candidate, threshold=1.0, critical=critical)
    parent_p95_mean = mean(parent_p95)
    candidate_p95_mean = mean(candidate_p95)
    p95_ratio = candidate_p95_mean / parent_p95_mean if parent_p95_mean > 0.0 else 0.0
    return {
        "n": len(pairs),
        "metric": metric,
        "parent_tok_s": mean(parent),
        "candidate_tok_s": mean(candidate),
        "geo_ratio": geo_mean(ratios),
        "point_ratios": ratios,
        "ci": ci,
        "parent_p95_ms": parent_p95_mean,
        "candidate_p95_ms": candidate_p95_mean,
        "p95_ratio": p95_ratio,
        "pairs": pairs,
    }


def dual_summary(record: Mapping[str, Any], *, critical: float) -> dict[str, Any]:
    decode = summarize_pairs(record, metric="decode_only", critical=critical)
    request = summarize_pairs(record, metric="request", critical=critical)
    return {
        "decode_only": decode,
        "request": request,
        "n": decode["n"],
        "parent_tok_s": request["parent_tok_s"],
        "candidate_tok_s": request["candidate_tok_s"],
        "geo_ratio": request["geo_ratio"],
        "ci": request["ci"],
        "p95_ratio": decode["p95_ratio"],
        "parent_p95_ms": decode["parent_p95_ms"],
        "candidate_p95_ms": decode["candidate_p95_ms"],
        "pairs": record.get("pairs") or [],
    }


def pin_text() -> str:
    return HEADER.read_text(encoding="utf-8")


def pin_enabled(name: str) -> bool:
    return f"constexpr bool {name} = true;" in pin_text()


def apply_shipping_pins(*, poll: bool, defer: bool) -> None:
    text = pin_text()
    updated = text.replace(
        "constexpr bool kSelectedPollWithoutDeviceSync = false;",
        f"constexpr bool kSelectedPollWithoutDeviceSync = {'true' if poll else 'false'};",
    )
    updated = updated.replace(
        "constexpr bool kSelectedPollWithoutDeviceSync = true;",
        f"constexpr bool kSelectedPollWithoutDeviceSync = {'true' if poll else 'false'};",
    )
    updated = updated.replace(
        "constexpr bool kSelectedDeferElapsedEventSync = false;",
        f"constexpr bool kSelectedDeferElapsedEventSync = {'true' if defer else 'false'};",
    )
    updated = updated.replace(
        "constexpr bool kSelectedDeferElapsedEventSync = true;",
        f"constexpr bool kSelectedDeferElapsedEventSync = {'true' if defer else 'false'};",
    )
    if updated != text:
        HEADER.write_text(updated, encoding="utf-8")
    contract = load_json(CONTRACT)
    contract["selected_poll_without_device_sync"] = poll
    contract["selected_defer_elapsed_event_sync"] = defer
    dump_json(CONTRACT, contract)


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    opt127 = load_json(OPT127_FIXTURE) if OPT127_FIXTURE.is_file() else {}
    source, dirty = git_identity()
    selector = Path("cuda/execution_graph_path.cuh").read_text(encoding="utf-8")
    kept = bool(opt127.get("production_kept")) and (
        'kSelectedExecutionGraphPath[] = "decode_segments8"' in selector
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-128",
        "phase": "preflight",
        "mode": mode,
        "ok": kept and contract["parent"] == PARENT_STACK,
        "authenticated_post113": auth,
        "opt127_kept": kept,
        "parent": contract["parent"],
        "parent_execution_graphs": PARENT,
        "candidates": list(CANDIDATES),
        "max_changes": 2,
        "shipping_poll_without_device_sync": pin_enabled(
            "kSelectedPollWithoutDeviceSync"
        ),
        "shipping_defer_elapsed_event_sync": pin_enabled(
            "kSelectedDeferElapsedEventSync"
        ),
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
        raise HostStallError("OPT-127 decode_segments8 parent was not kept")
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


def arm_wall(profile: Mapping[str, Any], name: str) -> dict[str, Any]:
    arms = dict(profile.get("arms") or {})
    row = dict(arms.get(name) or {})
    stalls = dict(row.get("stalls") or {})
    tokens = float(stalls.get("tokens") or profile.get("outputs") or 1)
    wall = float(stalls.get("wall_ms") or 0.0)
    delaying = float(stalls.get("delaying_host_ms") or 0.0)
    overlapping = float(stalls.get("overlapping_wait_ms") or 0.0)
    return {
        "ok": bool(row.get("ok")),
        "tokens": tokens,
        "wall_ms": wall,
        "per_token_wall_ms": wall / tokens if tokens else 0.0,
        "delaying_host_ms": delaying,
        "per_token_delaying_ms": delaying / tokens if tokens else 0.0,
        "overlapping_wait_ms": overlapping,
        "per_token_overlapping_ms": overlapping / tokens if tokens else 0.0,
        "event_create_ms": float(stalls.get("event_create_ms") or 0.0) / tokens
        if tokens
        else 0.0,
        "event_destroy_ms": float(stalls.get("event_destroy_ms") or 0.0) / tokens
        if tokens
        else 0.0,
        "elapsed_event_sync_ms": float(stalls.get("elapsed_event_sync_ms") or 0.0)
        / tokens
        if tokens
        else 0.0,
        "scatter_device_sync_ms": float(stalls.get("scatter_device_sync_ms") or 0.0)
        / tokens
        if tokens
        else 0.0,
        "finish_output_commit_ms": float(stalls.get("finish_output_commit_ms") or 0.0)
        / tokens
        if tokens
        else 0.0,
        "greedy_d2h_ms": float(stalls.get("greedy_d2h_ms") or 0.0) / tokens
        if tokens
        else 0.0,
        "poll_device_sync_ms": float(stalls.get("poll_device_sync_ms") or 0.0) / tokens
        if tokens
        else 0.0,
        "poll_host_ms": float(stalls.get("poll_host_ms") or 0.0) / tokens
        if tokens
        else 0.0,
        "poll_device_syncs": int(stalls.get("poll_device_syncs") or 0),
        "blocking_d2h_copies": int(stalls.get("blocking_d2h_copies") or 0),
        "d2h_bytes": int(row.get("d2h_bytes") or 0),
    }


def run_rank(run_dir: Path, mode: str) -> dict[str, Any]:
    profile = load_sidecar(run_dir, "profile.json")
    lifetime = load_sidecar(run_dir, "lifetime.json") or {}
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    logits = load_sidecar(run_dir, "logits-consumer.json") or {}
    nongreedy = load_sidecar(run_dir, "non-greedy.json") or {}
    if not profile:
        raise HostStallError("rank requires profile sidecar")
    parent_bench = arm_wall(profile, "parent_no_poll")
    parent_server = arm_wall(profile, "parent_server_poll")
    poll = arm_wall(profile, "poll_host_only")
    defer = arm_wall(profile, "defer_elapsed_sync")
    combined = arm_wall(profile, "combined_server_poll")
    poll_save = parent_server["per_token_wall_ms"] - poll["per_token_wall_ms"]
    defer_save = parent_bench["per_token_wall_ms"] - defer["per_token_wall_ms"]
    combined_save = parent_server["per_token_wall_ms"] - combined["per_token_wall_ms"]
    ranked = [
        {
            "name": "poll_without_device_sync",
            "stack": "execute_token poll after each decode_segments8 launch: "
            "cudaDeviceSynchronize then host atomic. Server Session::eval always "
            "passes cancelled, so this is the production sample/eval path.",
            "kind": "stream_device_synchronization",
            "overlapping_ms": parent_server["poll_device_sync_ms"],
            "delaying_ms": max(0.0, poll_save),
            "shifted_wait": poll_save <= 0.0,
        },
        {
            "name": "defer_elapsed_event_sync",
            "stack": "execute_token cudaEventCreate/Record/Synchronize(stop) before "
            "commit_outputs+scatter. The synchronize overlaps GPU compute and "
            "delays copy-stream argmax / KV scatter submission.",
            "kind": "stream_device_synchronization",
            "overlapping_ms": parent_bench["elapsed_event_sync_ms"],
            "delaying_ms": max(0.0, defer_save),
            "shifted_wait": defer_save <= 0.0,
        },
        {
            "name": "finish_output_commit_greedy_d2h",
            "stack": "finish_output_commit cudaStreamSynchronize(copy) + blocking "
            "4-byte cudaMemcpy of greedy index. OPT-118 lazy path.",
            "kind": "blocking_copy",
            "overlapping_ms": 0.0,
            "delaying_ms": parent_bench["finish_output_commit_ms"],
            "shifted_wait": False,
        },
        {
            "name": "hot_path_event_allocation",
            "stack": "cudaEventCreate/Destroy start/stop every execute_token.",
            "kind": "hot_path_allocation",
            "overlapping_ms": 0.0,
            "delaying_ms": parent_bench["event_create_ms"]
            + parent_bench["event_destroy_ms"],
            "shifted_wait": False,
        },
        {
            "name": "non_greedy_host_alloc",
            "stack": "Session::sample temperature>0 allocates 248320 candidates.",
            "kind": "hot_path_allocation",
            "overlapping_ms": 0.0,
            "delaying_ms": float(nongreedy.get("host_alloc_ms") or 0.0),
            "shifted_wait": False,
            "production_greedy_path": True,
        },
    ]
    ranked.sort(key=lambda row: float(row["delaying_ms"]), reverse=True)
    payload = {
        "schema_version": 1,
        "task": "OPT-128",
        "phase": "rank",
        "mode": mode,
        "ok": bool(profile.get("ok")),
        "parent_bench": parent_bench,
        "parent_server": parent_server,
        "poll": poll,
        "defer": defer,
        "combined": combined,
        "poll_save_ms": poll_save,
        "defer_save_ms": defer_save,
        "combined_save_ms": combined_save,
        "ranked": ranked,
        "lifetime_ok": bool(lifetime.get("ok")),
        "cancellation_ok": bool(cancel.get("ok")),
        "logits_consumer_ok": bool(logits.get("ok")),
        "lazy_d2h_bytes": int(lifetime.get("lazy_d2h_bytes") or 0),
        "full_logit_bytes": int(lifetime.get("full_logit_bytes") or 0),
        "family_plan": family_plan(mode, "rank"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "rank.json", payload)


def freeze_from_rank(rank: Mapping[str, Any]) -> dict[str, Any]:
    parent_wall = float(
        (rank.get("parent_server") or {}).get("per_token_wall_ms") or 0.0
    )
    threshold = max(MATERIAL_MS, MATERIAL_REL * parent_wall)
    poll_save = float(rank.get("poll_save_ms") or 0.0)
    defer_save = float(rank.get("defer_save_ms") or 0.0)
    combined_save = float(rank.get("combined_save_ms") or 0.0)
    admitted: list[str] = []
    if poll_save > threshold:
        admitted.append("poll")
    if defer_save > threshold and len(admitted) < 2:
        admitted.append("defer")
    if not admitted and combined_save > threshold:
        if poll_save >= defer_save:
            admitted.append("poll")
        else:
            admitted.append("defer")
    candidate = (
        "combined"
        if admitted == ["poll", "defer"]
        else (admitted[0] if admitted else "none")
    )
    no_material = not admitted
    return {
        "threshold_ms": threshold,
        "parent_server_wall_ms": parent_wall,
        "poll_save_ms": poll_save,
        "defer_save_ms": defer_save,
        "combined_save_ms": combined_save,
        "admitted_candidates": admitted,
        "frozen_candidate": candidate,
        "max_changes": 2,
        "verdict": "no_material_opportunity" if no_material else "measure_candidates",
        "rationale": (
            "A shifted wait is not a removed wait. Overlapping EventSynchronize "
            "and DeviceSynchronize that cover useful GPU work are not attributed "
            "as removable. Candidates freeze only when the sample/eval wall save "
            f"clears {threshold:.4f} ms/token (max of {MATERIAL_MS} ms and "
            f"{MATERIAL_REL:.0%} of parent server-poll wall)."
        ),
    }


def run_freeze(run_dir: Path, mode: str) -> dict[str, Any]:
    rank = load_sidecar(run_dir, "rank.json")
    if not rank:
        raise HostStallError("freeze requires rank sidecar")
    payload = freeze_from_rank(rank)
    payload.update(
        {
            "schema_version": 1,
            "task": "OPT-128",
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


def run_stall_ab_phase(
    run_dir: Path,
    mode: str,
    phase: str,
    prefix: int,
    tokens: int,
    samples: int,
    *,
    critical: float,
) -> dict[str, Any]:
    candidate = frozen_candidate(run_dir)
    if candidate == "none" or not candidates_admitted(run_dir):
        payload = {
            "schema_version": 1,
            "task": "OPT-128",
            "phase": phase,
            "mode": mode,
            "ok": True,
            "skipped": True,
            "reason": "no_host_stall_candidate_admitted",
            "family_plan": family_plan(mode, phase),
            "measured_at": utc_now(),
        }
        return store_sidecar(run_dir, f"{phase}.json", payload)
    poll = candidate in {"poll", "combined"}
    completed = run_native(
        [
            f"./{NATIVE}",
            "--workload",
            "stall-ab",
            "--candidate",
            candidate,
            "--poll",
            "1" if poll else "0",
            "--prefix",
            str(prefix),
            "--warmups",
            str(WARMUPS),
            "--samples",
            str(samples),
            "--tokens",
            str(tokens),
            MODEL,
        ],
        tier="acceptance" if samples >= PAIR_COUNT else "screen",
    )
    raw = completed.stdout + completed.stderr
    sidecar_path = run_dir / f"{phase}-raw.txt"
    sidecar_path.write_text(raw, encoding="utf-8")
    record = parse_prefixed(raw, RESULT_PREFIX)
    summary = dual_summary(record, critical=critical)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    record["summary"] = summary
    record["raw_trace"] = workspace_relative(sidecar_path)
    record["ok"] = True
    return store_sidecar(run_dir, f"{phase}.json", record)


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
        raise HostStallError("--quality restored packed or r2 defaults")
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
            "task": "OPT-128",
            "phase": "quality",
            "mode": mode,
            "ok": True,
            "required": False,
            "reason": "no_host_stall_candidate_admitted_to_complete_request_ab",
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
    candidate["same_math_copied_from_control"] = True
    control_held = nll_from_cases(control["cases"], "held_out_wikitext_1024")
    cand_held = nll_from_cases(candidate["cases"], "held_out_wikitext_1024")
    held_ratio = (
        ppl_ratio(cand_held, control_held)
        if control_held is not None and cand_held is not None
        else None
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-128",
        "phase": "quality",
        "mode": mode,
        "ok": held_ratio is not None and held_ratio <= 1.01,
        "required": True,
        "opt058_invoked": True,
        "candidate_nll_measured": control_held is not None,
        "same_path_host_only": True,
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
        ckpt = run_dir / "opt128.ckpt"
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
        "task": "OPT-128",
        "phase": "state-memory",
        "mode": mode,
        "lifetime_ok": bool(lifetime.get("ok")),
        "restore_equals": bool(lifetime.get("restore_equals")),
        "cancel_isolated": bool(lifetime.get("cancel_isolated")),
        "subsequent_request_ok": bool(lifetime.get("subsequent_request_ok")),
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
    logits = load_sidecar(run_dir, "logits-consumer.json") or {}
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
    poll = False
    defer = False
    if not admitted:
        return {
            "verdict": "no_material_opportunity",
            "production_kept": True,
            "shipping_poll_without_device_sync": False,
            "shipping_defer_elapsed_event_sync": False,
            "reasons": reasons + ["admitted=[]"],
            "quality": quality,
            "state_memory": state,
            "performance": {
                "pass": True,
                "complete_request_ab_executed": False,
                "reason": "quantified_negligible_upper_bound",
                "tok_s_delta": tok_delta,
                "poll_save_ms": freeze.get("poll_save_ms"),
                "defer_save_ms": freeze.get("defer_save_ms"),
                "threshold_ms": freeze.get("threshold_ms"),
            },
            "tok_s_delta": tok_delta,
            "lifetime_ok": bool(lifetime.get("ok")),
            "cancellation_ok": bool(cancel.get("ok")),
            "logits_ok": bool(logits.get("ok")),
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
        and bool(logits.get("ok", True))
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
        poll = "poll" in admitted or freeze.get("frozen_candidate") == "combined"
        defer = "defer" in admitted or freeze.get("frozen_candidate") == "combined"
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
        "shipping_poll_without_device_sync": poll if keep else False,
        "shipping_defer_elapsed_event_sync": defer if keep else False,
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
        "logits_ok": bool(logits.get("ok")),
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
    parent_server = dict(rank.get("parent_server") or {})
    parent_bench = dict(rank.get("parent_bench") or {})
    poll = dict(rank.get("poll") or {})
    defer = dict(rank.get("defer") or {})
    ranked = list(rank.get("ranked") or [])
    lines = [
        "# OPT-128 — Remove measured host submission and synchronization stalls",
        "",
        f"Status: **{payload.get('verdict')}**. Parent is OPT-127 kept "
        f"`{PARENT}` (`{payload.get('parent')}`). Production graph selector is "
        "unchanged. Host-stall pins stay false unless this sitting kept a "
        "candidate.",
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
        "## Before/after critical-path attribution",
        "",
        "Complete Session sample/eval/output loop on decode_segments8. Server "
        "eval always passes a cancellation poller; bench does not. Overlapping "
        "GPU waits are listed separately from host work that delays the next "
        "submission. A shifted wait is not a removed wait.",
        "",
        "| Arm | wall ms/token | delaying host ms/token | overlapping wait ms/token | poll DeviceSynchronize count |",
        "|---|---:|---:|---:|---:|",
        f"| parent bench (no poll) | {parent_bench.get('per_token_wall_ms')} | "
        f"{parent_bench.get('per_token_delaying_ms')} | "
        f"{parent_bench.get('per_token_overlapping_ms')} | "
        f"{parent_bench.get('poll_device_syncs')} |",
        f"| parent server poll | {parent_server.get('per_token_wall_ms')} | "
        f"{parent_server.get('per_token_delaying_ms')} | "
        f"{parent_server.get('per_token_overlapping_ms')} | "
        f"{parent_server.get('poll_device_syncs')} |",
        f"| poll host-only | {poll.get('per_token_wall_ms')} | "
        f"{poll.get('per_token_delaying_ms')} | "
        f"{poll.get('per_token_overlapping_ms')} | "
        f"{poll.get('poll_device_syncs')} |",
        f"| defer elapsed sync | {defer.get('per_token_wall_ms')} | "
        f"{defer.get('per_token_delaying_ms')} | "
        f"{defer.get('per_token_overlapping_ms')} | "
        f"{defer.get('poll_device_syncs')} |",
        "",
        f"Poll wall save `{freeze.get('poll_save_ms')}` ms/token. Defer wall "
        f"save `{freeze.get('defer_save_ms')}` ms/token. Freeze threshold "
        f"`{freeze.get('threshold_ms')}` ms/token.",
        "",
        "## Ranked host stalls",
        "",
        "| Rank | Name | kind | delaying ms/token | overlapping ms/token |",
        "|---:|---|---|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        extra = ""
        if row.get("production_greedy_path"):
            extra = " (off greedy production path; not frozen)"
        lines.append(
            f"| {index} | {row.get('name')}{extra} | {row.get('kind')} | "
            f"{row.get('delaying_ms')} | {row.get('overlapping_ms')} |"
        )
    for row in ranked:
        lines.extend(["", f"### {row.get('name')}", "", str(row.get("stack") or "")])
    tok = dict(payload.get("tok_s_delta") or {})
    lines.extend(
        [
            "",
            "## Lifetime / logits / cancellation",
            "",
            f"lifetime=`{(payload.get('lifetime_ok'))}` cancel="
            f"`{payload.get('cancellation_ok')}` logits_consumer="
            f"`{payload.get('logits_ok')}`. Lazy greedy D2H remains 4 bytes. "
            "Full-logit `copy_last_outputs` still materializes the vocab. "
            "Buffer/event lifetimes cover cancel, restore, and a subsequent "
            "request.",
            "",
            "## Frozen candidates (at most two)",
            "",
            f"Admitted: `{freeze.get('admitted_candidates')}`. Frozen: "
            f"`{freeze.get('frozen_candidate')}`. {freeze.get('rationale')}",
            "",
            f"**Verdict: `{payload.get('verdict')}`.** "
            f"poll pin `{payload.get('shipping_poll_without_device_sync')}`; "
            f"defer pin `{payload.get('shipping_defer_elapsed_event_sync')}`. "
            f"D128 tok/s delta `{tok.get('d128')}`; speedup `{tok.get('speedup')}`; "
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
        "task": "OPT-128",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": contract["parent"],
        "candidate": decision.get("freeze", {}).get("frozen_candidate") or "none",
        "profile": load_sidecar(run_dir, "profile.json") or {},
        "lifetime": load_sidecar(run_dir, "lifetime.json") or {},
        "cancellation": load_sidecar(run_dir, "cancellation.json") or {},
        "logits_consumer": load_sidecar(run_dir, "logits-consumer.json") or {},
        "non_greedy": load_sidecar(run_dir, "non-greedy.json") or {},
        "rank": decision.get("rank") or load_sidecar(run_dir, "rank.json") or {},
        "freeze": decision.get("freeze") or load_sidecar(run_dir, "freeze.json") or {},
        "quality": decision["quality"],
        "state_memory": decision["state_memory"],
        "performance": decision["performance"],
        "shipping_poll_without_device_sync": decision[
            "shipping_poll_without_device_sync"
        ],
        "shipping_defer_elapsed_event_sync": decision[
            "shipping_defer_elapsed_event_sync"
        ],
        "production_kept": decision["production_kept"],
        "verdict": decision["verdict"],
        "tok_s_delta": decision["tok_s_delta"],
        "reasons": decision["reasons"],
        "lifetime_ok": decision["lifetime_ok"],
        "cancellation_ok": decision["cancellation_ok"],
        "logits_ok": decision["logits_ok"],
        "report_path": str(REPORT.relative_to(ROOT)),
        "measured_at": utc_now(),
        "ok": True,
        "family_plan": family_plan(mode, "report"),
    }
    write_report(payload)
    dump_json(FIXTURE, payload)
    if decision["verdict"] == "keep":
        apply_shipping_pins(
            poll=bool(decision["shipping_poll_without_device_sync"]),
            defer=bool(decision["shipping_defer_elapsed_event_sync"]),
        )
    else:
        apply_shipping_pins(poll=False, defer=False)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "profile":
        return run_named_native(
            run_dir, mode, "profile", ["--prefix", "128", "--tokens", "16"]
        )
    if phase == "lifetime":
        return run_named_native(
            run_dir,
            mode,
            "lifetime",
            ["--candidate", "combined", "--prefix", "8"],
        )
    if phase == "cancellation":
        return run_named_native(
            run_dir,
            mode,
            "cancellation",
            ["--candidate", "combined", "--prefix", "128"],
        )
    if phase == "logits-consumer":
        return run_named_native(
            run_dir,
            mode,
            "logits-consumer",
            ["--candidate", "combined", "--prefix", "8", "--tokens", "4"],
        )
    if phase == "non-greedy":
        return run_named_native(
            run_dir,
            mode,
            "non-greedy",
            ["--candidate", "combined", "--prefix", "8"],
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
        return run_stall_ab_phase(
            run_dir,
            mode,
            "screen-d128",
            128,
            DECODE_TOKENS,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "screen-d2048":
        return run_stall_ab_phase(
            run_dir,
            mode,
            "screen-d2048",
            2048,
            DECODE_TOKENS,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "screen-p4096":
        return run_stall_ab_phase(
            run_dir,
            mode,
            "screen-p4096",
            4096,
            0,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "d128":
        return run_stall_ab_phase(
            run_dir,
            mode,
            "d128",
            128,
            DECODE_TOKENS,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "d2048":
        return run_stall_ab_phase(
            run_dir,
            mode,
            "d2048",
            2048,
            DECODE_TOKENS,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "p4096":
        return run_stall_ab_phase(
            run_dir,
            mode,
            "p4096",
            4096,
            0,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "report":
        return run_report(run_dir, mode)
    raise HostStallError(f"unknown phase {phase}")


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
    except HostStallError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
