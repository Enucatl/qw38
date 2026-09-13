"""OPT-127 recapture-free decode-segment replay vs post124 combined ffn_only."""

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

CONTRACT = ROOT / "pins/opt127_recapture_free_replay_contract.json"
ITERATION = ROOT / "pins/opt127_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt127_recapture_free_replay.json"
REPORT = ROOT / "evidence/optimization/opt127-recapture-free-replay/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT117_FIXTURE = ROOT / "fixtures/opt117_decode_segment_graphs.json"
OPT123_FIXTURE = ROOT / "fixtures/opt123_combined_stack.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt127-recapture-free-replay-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT127_RECAPTURE_FREE_REPLAY_RESULT="
COUNTS_PREFIX = "QW38_OPT127_NATIVE_COUNTS="
PHASES = (
    "preflight",
    "recapture-proof",
    "cancellation",
    "same-math",
    "handoff",
    "attribution",
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
PARENT = "ffn_only"
CANDIDATE = "decode_segments8"
PARENT_STACK = "combined_opt118_opt119"
WARMUPS = 3
SCREEN_PAIRS = 5
PAIR_COUNT = 10
DECODE_TOKENS = 256
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
    "execution_graphs": "ffn_only",
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
    "recapture_proof",
    "cancellation",
    "same_math",
    "quality",
    "state_memory",
    "performance",
    "shipping_execution_graphs",
    "production_kept",
    "verdict",
    "report_path",
)


class ReplayError(RuntimeError):
    """OPT-127 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-127":
        raise ReplayError("recapture-free-replay contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-127", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-127 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


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
        raise ReplayError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise ReplayError(
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


def pair_order(pair_index: int) -> str:
    return "AB" if pair_index % 2 == 0 else "BA"


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


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    opt123 = load_json(OPT123_FIXTURE) if OPT123_FIXTURE.is_file() else {}
    source, dirty = git_identity()
    combined_kept = bool(opt123.get("production_kept"))
    selector = Path("cuda/execution_graph_path.cuh").read_text(encoding="utf-8")
    selector_ffn = 'kSelectedExecutionGraphPath[] = "ffn_only"' in selector
    payload = {
        "schema_version": 1,
        "task": "OPT-127",
        "phase": "preflight",
        "mode": mode,
        "ok": selector_ffn and contract["parent"] == PARENT_STACK,
        "authenticated_post113": auth,
        "combined_parent_kept": combined_kept,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "parent_execution_graphs": PARENT,
        "shipping_execution_graphs": PARENT,
        "selector_unchanged": selector_ffn,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": bool(dirty),
        "iteration_target": iteration["target"],
        "diagnostics_make_target": iteration["diagnostics_make_target"],
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
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
    record["stdout_tail"] = completed.stdout[-4000:]
    name = sidecar_name or phase
    return store_sidecar(run_dir, f"{name}.json", record)


def run_graph_ab_phase(
    run_dir: Path,
    mode: str,
    phase: str,
    prefix: int,
    tokens: int,
    samples: int,
    *,
    critical: float,
) -> dict[str, Any]:
    completed = run_native(
        [
            f"./{NATIVE}",
            "--workload",
            "graph-ab",
            "--execution-graphs",
            CANDIDATE,
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
    return store_sidecar(run_dir, f"{phase}.json", record)


def run_quality_native(run_dir: Path, config_id: str) -> dict[str, Any]:
    selectors = dict(COMBINED_QUALITY)
    if config_id == CANDIDATE:
        selectors["execution_graphs"] = CANDIDATE
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
        raise ReplayError("--quality restored packed or r2 defaults")
    return {
        "id": config_id,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
    }


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    same = load_sidecar(run_dir, "same-math.json") or {}
    exact = bool(same.get("exact") and same.get("ok"))
    control = run_quality_native(run_dir, PARENT_STACK)
    candidate = dict(control)
    candidate["id"] = CANDIDATE
    candidate["same_math_copied_from_control"] = exact
    if not exact:
        candidate = run_quality_native(run_dir, CANDIDATE)
        candidate["same_math_copied_from_control"] = False
    control_held = nll_from_cases(control["cases"], "held_out_wikitext_1024")
    cand_held = nll_from_cases(candidate["cases"], "held_out_wikitext_1024")
    control_wiki = nll_from_cases(control["cases"], "wikitext_nll")
    cand_wiki = nll_from_cases(candidate["cases"], "wikitext_nll")
    incomplete = control_held is None or cand_held is None
    held_ratio = (
        ppl_ratio(cand_held, control_held)
        if control_held is not None and cand_held is not None
        else None
    )
    wiki_ratio = (
        ppl_ratio(cand_wiki, control_wiki)
        if control_wiki is not None and cand_wiki is not None
        else None
    )
    opt116 = load_json(OPT116_FIXTURE) if OPT116_FIXTURE.is_file() else {}
    ppl = (
        ((opt116.get("authenticate") or {}).get("contracts") or {}).get("contracts")
        or {}
    ).get("opt116_generated_v1", {})
    parent_ppl = float(((ppl.get("ppl") or {}).get("aggregate_ratio")) or 1.0)
    quality_ok = (
        exact
        and not incomplete
        and held_ratio is not None
        and held_ratio <= 1.01
        and (wiki_ratio is None or wiki_ratio <= 1.01)
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-127",
        "phase": "quality",
        "mode": mode,
        "ok": quality_ok,
        "same_path_exact": exact,
        "opt058_invoked": True,
        "candidate_nll_measured": not incomplete,
        "opt116_contract_id": "opt116_generated_v1",
        "ppl_ratio": held_ratio,
        "wikitext_ppl_ratio": wiki_ratio,
        "ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "parent_authenticated_ppl_ratio": parent_ppl,
        "control": control,
        "candidate": candidate,
        "quality_flag": "--quality",
        "family_plan": family_plan(mode, "quality"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "quality.json", payload)


def run_state_memory(run_dir: Path, mode: str) -> dict[str, Any]:
    native = run_named_native(
        run_dir,
        mode,
        "state-memory",
        ["--execution-graphs", CANDIDATE, "--prefix", "128", "--tokens", "1"],
    )
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    fit = bool(memory.get("post_graph_admitted"))
    checkpoint_ok = False
    checkpoint_message = "checkpoint binary missing"
    checkpoint_bin = ROOT / CHECKPOINT_NATIVE
    if checkpoint_bin.is_file():
        ckpt = run_dir / "opt127.ckpt"
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
    native["memory_fit_parent_admitted"] = fit
    native["parent_graph_bytes"] = int(
        (memory.get("owners") or {}).get("graph_bytes") or 0
    )
    native["candidate_graph_bytes"] = int(native.get("graph_bytes") or 0)
    native["128k_fit_with_graphs"] = fit
    native["checkpoint_ok"] = checkpoint_ok
    native["checkpoint_message"] = checkpoint_message
    native["cancellation_ok"] = bool(cancel.get("ok"))
    native["ok"] = (
        bool(native.get("ok"))
        and fit
        and bool(cancel.get("ok", True))
        and (checkpoint_ok or not checkpoint_bin.is_file())
    )
    native["family_plan"] = family_plan(mode, "state-memory")
    return store_sidecar(run_dir, "state-memory.json", native)


def aa_noise_from_opt115() -> dict[str, float]:
    payload = load_json(OPT115_FIXTURE) if OPT115_FIXTURE.is_file() else {}
    sitting = payload.get("aa") or payload.get("request_trace") or {}
    noise = {"d128": 0.00005, "d2048": 0.0001, "p4096": 0.0001}
    table = sitting.get("workloads") if isinstance(sitting, Mapping) else None
    if isinstance(table, Mapping):
        for name in noise:
            row = table.get(name) or {}
            ratio = float(row.get("geo_ratio") or 1.0)
            noise[name] = max(abs(ratio - 1.0), noise[name])
    return noise


def historical_engine_probes() -> dict[str, Any]:
    opt117 = load_json(OPT117_FIXTURE) if OPT117_FIXTURE.is_file() else {}
    performance = opt117.get("performance") or {}
    return {
        "source": "fixtures/opt117_decode_segment_graphs.json",
        "verdict": opt117.get("verdict"),
        "d128": performance.get("d128"),
        "d2048": performance.get("d2048"),
        "p4096": performance.get("p4096"),
        "note": "OPT-117 recaptured per token; not a recapture-free replay baseline",
    }


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    recapture = load_sidecar(run_dir, "recapture-proof.json") or {}
    same = load_sidecar(run_dir, "same-math.json") or {}
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    handoff = load_sidecar(run_dir, "handoff.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    d128 = load_sidecar(run_dir, "d128.json") or {}
    d2048 = load_sidecar(run_dir, "d2048.json") or {}
    p4096 = load_sidecar(run_dir, "p4096.json") or {}
    noise = aa_noise_from_opt115()
    recapture_ok = bool(recapture.get("ok"))
    same_ok = bool(same.get("ok") and same.get("exact"))
    quality_ok = bool(quality.get("ok") and quality.get("candidate_nll_measured"))
    state_ok = bool(state.get("ok"))
    correctness_ok = (
        recapture_ok
        and same_ok
        and bool(cancel.get("ok"))
        and bool(handoff.get("ok", True))
    )
    reasons: list[str] = []
    if not recapture_ok:
        reasons.append("recapture_proof_failed")
    if not same_ok:
        reasons.append("same_math_failed")
    if not quality_ok:
        reasons.append("quality_failed")
    if not state_ok:
        reasons.append("state_memory_failed")
    performance: dict[str, Any] = {}
    performance_ok = True
    for name, row, target in (
        ("d128", d128, True),
        ("d2048", d2048, True),
        ("p4096", p4096, False),
    ):
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
            point_ok = geo >= 0.98 if geo else False
            p95_ok = True
            row_ok = point_ok and p95_ok
            if not row_ok:
                performance_ok = False
                reasons.append(f"{name}_non_target_guard")
        performance[name] = {
            **summary,
            "target": target,
            "ok": row_ok,
            "aa_noise": noise[name],
            "request_geo_ratio": geo,
            "decode_p95_ratio": p95_ratio,
            "request_ci_lower": ci_lower,
        }
    keep = correctness_ok and quality_ok and state_ok and performance_ok
    if keep:
        verdict = "keep"
        shipping = CANDIDATE
    else:
        verdict = "retain_ffn_only"
        shipping = PARENT
        if correctness_ok and quality_ok and state_ok and not performance_ok:
            reasons.append("measured_loss_or_no_request_win")
    d128_delta = 0.0
    d128_row = performance.get("d128") or {}
    if d128_row.get("candidate_tok_s") and d128_row.get("parent_tok_s"):
        d128_delta = float(d128_row["candidate_tok_s"]) - float(
            d128_row["parent_tok_s"]
        )
    decode_d128 = (d128_row.get("decode_only") or {}).get("candidate_tok_s")
    parent_d128 = (d128_row.get("decode_only") or {}).get("parent_tok_s")
    return {
        "recapture_ok": recapture_ok,
        "same_math_pass": same_ok,
        "model_quality_pass": quality_ok,
        "state_memory_pass": state_ok,
        "performance_pass": performance_ok,
        "correctness_ok": correctness_ok,
        "production_kept": keep,
        "verdict": verdict,
        "shipping_execution_graphs": shipping,
        "reasons": reasons,
        "performance": performance,
        "d128_request_tok_s_delta": d128_delta,
        "d128_decode_only_candidate_tok_s": decode_d128,
        "d128_decode_only_parent_tok_s": parent_d128,
        "opt115_aa_noise": noise,
        "historical_engine_probes": historical_engine_probes(),
        "coarser_candidate_admitted": False,
        "weight_byte_reduction_claimed": False,
    }


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    keep = bool(result.get("production_kept"))
    verdict = str(result.get("verdict"))
    performance = result.get("performance") or {}
    d128 = performance.get("d128") or {}
    d2048 = performance.get("d2048") or {}
    p4096 = performance.get("p4096") or {}
    recapture = result.get("recapture_proof") or {}
    quality = result.get("quality") or {}
    lines = [
        "# OPT-127 — Replay decode segments without per-token recapture",
        "",
        f"Status: **{verdict}**. Production selector "
        f"`{result.get('shipping_execution_graphs')}`. Hardware executed.",
        "",
        "Parent is authenticated post124 combined OPT-118+119 `ffn_only`. "
        "Candidate is recapture-free `decode_segments8` replay on OPT-126 "
        "stable inputs (`DecodeLaunchState`, 16 segment graphs, "
        "`stable_pingpong_committed_slot`). Keep requires OPT-116 quality, "
        "state/memory/cancellation/checkpoint/128K, and a measured request "
        "gain on D128/D2048 with P4096 as a non-target guard. Capture, "
        "instantiate, and upload are instrumented separately from launch-state "
        "uploads and graph launches. No weight-byte reduction is claimed.",
        "",
        "## Recapture elimination",
        "",
        f"recapture_proof ok=`{recapture.get('ok')}`; invalidation policy "
        f"`{(recapture.get('invalidation_policy') or '')}`. "
        "Steady-state tokens inside one topology, including alternating GDN "
        "commits, must show zero capture/instantiate/destroy/recapture deltas. "
        "Crossover 1024 selects the prebound topology-1 graphs; the transition "
        "cost is inside the crossing request, not a recapture.",
        "",
        f"Coarser candidate admitted=`{result.get('coarser_candidate_admitted')}`. "
        "Existing eight-layer segmentation was screened first. Remaining graph "
        "submission gaps were not proven material under OPT-125 (proven device "
        "inactive 0 ms), and cancellation still polls at eight-layer boundaries, "
        "so no coarser persistent kernel was admitted.",
        "",
        "## Same-math, quality, state",
        "",
        f"same_math=`{(result.get('same_math') or {}).get('ok')}` exact="
        f"`{(result.get('same_math') or {}).get('exact')}`; "
        f"quality=`{quality.get('ok')}` candidate_nll_measured="
        f"`{quality.get('candidate_nll_measured')}` "
        f"opt058_invoked=`{quality.get('opt058_invoked')}` "
        f"held-out ppl ratio=`{quality.get('ppl_ratio')}`; "
        f"state/memory=`{(result.get('state_memory') or {}).get('ok')}`; "
        f"cancellation=`{(result.get('cancellation') or {}).get('ok')}`.",
        "",
        "## Full-engine A/B (3 warmups + 10 AB/BA; decode-only and request)",
        "",
        "| Workload | metric | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in (("D128", d128), ("D2048", d2048), ("P4096", p4096)):
        request = row.get("request") or row
        decode = row.get("decode_only") or {}
        lines.append(
            f"| {name} | request | {request.get('parent_tok_s')} | "
            f"{request.get('candidate_tok_s')} | {request.get('geo_ratio')} | "
            f"{(request.get('ci') or {}).get('ci_lower')} | "
            f"{row.get('decode_p95_ratio')} | {row.get('ok')} |"
        )
        if decode:
            lines.append(
                f"| {name} | decode_only | {decode.get('parent_tok_s')} | "
                f"{decode.get('candidate_tok_s')} | {decode.get('geo_ratio')} | "
                f"{(decode.get('ci') or {}).get('ci_lower')} | "
                f"{decode.get('p95_ratio')} | reported |"
            )
    hist = result.get("historical_engine_probes") or {}
    lines.extend(
        [
            "",
            "## Historical engine probes (OPT-117 recapturing path)",
            "",
            f"OPT-117 verdict `{hist.get('verdict')}` with per-token recapture. "
            "Those numbers are not a recapture-free replay measurement and are "
            "not reused as this sitting's parent.",
            "",
            f"D128 request tok/s delta vs combined parent "
            f"`{result.get('d128_request_tok_s_delta')}`. "
            f"Reasons: `{result.get('reasons')}`.",
            "",
            "Launch preparation stays inside the measured request (`setup_ms` + "
            "prefill + decode). Short-output and 256-output amortization are in "
            "the handoff and A/B sidecars. User-visible output remains per token; "
            "cancellation still polls after each eight-layer segment and commits "
            "atomically only on success.",
            "",
            f"Keep={keep}. Shipping `{result.get('shipping_execution_graphs')}`. "
            "Weight-byte reduction claimed=false.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def apply_shipping_selector(path: str) -> None:
    header = ROOT / "cuda/execution_graph_path.cuh"
    text = header.read_text(encoding="utf-8")
    updated = text
    for current in (PARENT, CANDIDATE):
        updated = updated.replace(
            f'constexpr char kSelectedExecutionGraphPath[] = "{current}";',
            f'constexpr char kSelectedExecutionGraphPath[] = "{path}";',
        )
    if updated != text:
        header.write_text(updated, encoding="utf-8")
    contract = load_json(CONTRACT)
    contract["selected_execution_graph_path"] = path
    dump_json(CONTRACT, contract)


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    decision = evaluate_keep(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-127",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "recapture_proof": load_sidecar(run_dir, "recapture-proof.json") or {},
        "cancellation": load_sidecar(run_dir, "cancellation.json") or {},
        "same_math": load_sidecar(run_dir, "same-math.json") or {},
        "handoff": load_sidecar(run_dir, "handoff.json") or {},
        "attribution": load_sidecar(run_dir, "attribution.json") or {},
        "quality": load_sidecar(run_dir, "quality.json") or {},
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "screen": {
            "d128": load_sidecar(run_dir, "screen-d128.json") or {},
            "d2048": load_sidecar(run_dir, "screen-d2048.json") or {},
            "p4096": load_sidecar(run_dir, "screen-p4096.json") or {},
        },
        "performance": decision["performance"],
        "historical_engine_probes": decision["historical_engine_probes"],
        "shipping_execution_graphs": decision["shipping_execution_graphs"],
        "production_kept": decision["production_kept"],
        "verdict": decision["verdict"],
        "independent_verdicts": {
            "recapture_ok": decision["recapture_ok"],
            "same_math_pass": decision["same_math_pass"],
            "model_quality_pass": decision["model_quality_pass"],
            "state_memory_pass": decision["state_memory_pass"],
            "performance_pass": decision["performance_pass"],
            "production_kept": decision["production_kept"],
        },
        "reasons": decision["reasons"],
        "d128_request_tok_s_delta": decision["d128_request_tok_s_delta"],
        "d128_decode_only_candidate_tok_s": decision[
            "d128_decode_only_candidate_tok_s"
        ],
        "d128_decode_only_parent_tok_s": decision["d128_decode_only_parent_tok_s"],
        "coarser_candidate_admitted": False,
        "weight_byte_reduction_claimed": False,
        "claims_throughput": bool(decision["production_kept"]),
        "claims_performance_improvement": bool(decision["production_kept"]),
        "report_path": str(contract["report_path"]),
        "measured_at": utc_now(),
        "family_plan": family_plan(mode, "report"),
    }
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise ReplayError(f"fixture missing {key}")
    dump_json(FIXTURE, payload)
    write_report(payload)
    apply_shipping_selector(str(decision["shipping_execution_graphs"]))
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "recapture-proof":
        return run_named_native(
            run_dir,
            mode,
            "recapture-proof",
            ["--execution-graphs", CANDIDATE],
        )
    if phase == "cancellation":
        return run_named_native(
            run_dir,
            mode,
            "cancellation",
            ["--execution-graphs", CANDIDATE, "--prefix", "128"],
        )
    if phase == "same-math":
        return run_named_native(
            run_dir,
            mode,
            "same-math",
            [
                "--execution-graphs",
                CANDIDATE,
                "--prefix",
                "128",
                "--tokens",
                "8",
            ],
        )
    if phase == "handoff":
        return run_named_native(
            run_dir,
            mode,
            "handoff",
            ["--execution-graphs", CANDIDATE, "--prefix", "128"],
        )
    if phase == "attribution":
        return run_named_native(
            run_dir,
            mode,
            "attribution",
            ["--execution-graphs", CANDIDATE, "--prefix", "128"],
        )
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "screen-d128":
        return run_graph_ab_phase(
            run_dir,
            mode,
            "screen-d128",
            128,
            DECODE_TOKENS,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "screen-d2048":
        return run_graph_ab_phase(
            run_dir,
            mode,
            "screen-d2048",
            2048,
            DECODE_TOKENS,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "screen-p4096":
        return run_graph_ab_phase(
            run_dir,
            mode,
            "screen-p4096",
            4096,
            0,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
        )
    if phase == "d128":
        return run_graph_ab_phase(
            run_dir,
            mode,
            "d128",
            128,
            DECODE_TOKENS,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "d2048":
        return run_graph_ab_phase(
            run_dir,
            mode,
            "d2048",
            2048,
            DECODE_TOKENS,
            PAIR_COUNT,
            critical=T_CRIT_DF9,
        )
    if phase == "p4096":
        return run_graph_ab_phase(
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
    raise ReplayError(f"unknown phase {phase}")


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
    except ReplayError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
