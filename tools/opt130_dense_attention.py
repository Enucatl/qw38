"""OPT-130 dense decode-attention occupancy/vec128_open vs post124+OPT-127 parent."""

from __future__ import annotations

import argparse
import json
import math
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

CONTRACT = ROOT / "pins/opt130_dense_attention_contract.json"
ITERATION = ROOT / "pins/opt130_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt130_dense_attention.json"
REPORT = ROOT / "evidence/optimization/opt130-dense-attention/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
NATIVE = "build/qw38-cuda-opt130-dense-attention-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT130_DENSE_ATTENTION_RESULT="
COUNTS_PREFIX = "QW38_OPT130_NATIVE_COUNTS="
PHASES = (
    "inspect",
    "identity",
    "primitive",
    "same-math",
    "cancellation",
    "handoff",
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
PARENT = "hybrid_crossover"
CANDIDATE = "occupancy_vec128_open"
PARENT_STACK = "post124_decode_segments8"
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
    "quality",
    "state_memory",
    "performance",
    "shipping_decode_attention",
    "production_kept",
    "verdict",
    "report_path",
)


class DenseAttentionError(RuntimeError):
    """OPT-130 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-130":
        raise DenseAttentionError("dense-attention contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-130", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-130 mode={mode} phase={family} "
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
        raise DenseAttentionError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise DenseAttentionError(
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


def pin_text() -> str:
    return PIN_PATH.read_text(encoding="utf-8")


def production_pins_parent() -> bool:
    text = pin_text()
    return (
        "kSelectedDecodeAttentionCrossoverThreshold = 1024" in text
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in text
        and "kSelectedVec128NParts = 16" in text
        and 'kSelectedDecodeAttentionGqaPath[] = "warp_query"' in text
    )


def run_inspect(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    source, dirty = git_identity()
    completed = run_native(
        [f"./{NATIVE}", "--workload", "inspect"],
        tier="correctness",
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = "inspect"
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, "inspect")
    record["authenticated_post113"] = auth
    record["parent"] = contract["parent"]
    record["candidate"] = contract["candidate"]
    record["second_candidate_launched"] = False
    record["gqa6_admitted"] = False
    record["production_pins_parent"] = production_pins_parent()
    record["source"] = source
    record["dirty"] = bool(dirty)
    record["iteration_target"] = iteration["target"]
    record["ok"] = bool(record.get("ok")) and production_pins_parent()
    record["measured_at"] = utc_now()
    return store_sidecar(run_dir, "inspect.json", record)


def run_named_native(
    run_dir: Path,
    mode: str,
    phase: str,
    extra: Sequence[str],
    *,
    check: bool = True,
    sidecar_name: str | None = None,
    needs_model: bool = True,
) -> dict[str, Any]:
    args = [f"./{NATIVE}", "--workload", phase, *extra]
    if needs_model:
        args.append(MODEL)
    completed = run_native(args, tier="correctness", check=check)
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
        selectors["decode_attention"] = CANDIDATE
        selectors["decode_attention_verified_max"] = 131072
        selectors["vec128_n_parts"] = 8
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
        raise DenseAttentionError("--quality restored packed or r2 defaults")
    if not cases:
        raise DenseAttentionError(f"OPT-058 produced no NLL cases for {config_id}")
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
    control = run_quality_native(run_dir, PARENT)
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
        not incomplete
        and held_ratio is not None
        and held_ratio <= 1.01
        and (wiki_ratio is None or wiki_ratio <= 1.01)
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-130",
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
        ["--prefix", "128", "--tokens", "1"],
    )
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    fit = bool(memory.get("post_graph_admitted"))
    checkpoint_ok = False
    checkpoint_message = "checkpoint binary missing"
    checkpoint_bin = ROOT / CHECKPOINT_NATIVE
    if checkpoint_bin.is_file():
        ckpt = run_dir / "opt130.ckpt"
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


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    identity = load_sidecar(run_dir, "identity.json") or {}
    primitive = load_sidecar(run_dir, "primitive.json") or {}
    same = load_sidecar(run_dir, "same-math.json") or {}
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    handoff = load_sidecar(run_dir, "handoff.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    d128 = load_sidecar(run_dir, "d128.json") or {}
    d2048 = load_sidecar(run_dir, "d2048.json") or {}
    p4096 = load_sidecar(run_dir, "p4096.json") or {}
    noise = aa_noise_from_opt115()
    same_ok = bool(same.get("ok") and same.get("exact"))
    quality_ok = bool(quality.get("ok") and quality.get("candidate_nll_measured"))
    state_ok = bool(state.get("ok"))
    identity_ok = bool(identity.get("ok"))
    correctness_ok = (
        identity_ok
        and same_ok
        and bool(cancel.get("ok"))
        and bool(handoff.get("ok", True))
    )
    reasons: list[str] = []
    if not identity_ok:
        reasons.append("identity_failed")
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
            row_ok = point_ok
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
        verdict = "reject"
        shipping = PARENT
        if correctness_ok and quality_ok and state_ok and not performance_ok:
            reasons.append("measured_loss_or_no_request_win")
    deltas = {}
    for name in ("d128", "d2048", "p4096"):
        row = performance.get(name) or {}
        decode = row.get("decode_only") or {}
        request = row.get("request") or row
        parent_d = float(decode.get("parent_tok_s") or 0.0)
        cand_d = float(decode.get("candidate_tok_s") or 0.0)
        parent_r = float(request.get("parent_tok_s") or row.get("parent_tok_s") or 0.0)
        cand_r = float(
            request.get("candidate_tok_s") or row.get("candidate_tok_s") or 0.0
        )
        deltas[name] = {
            "decode_only_parent_tok_s": parent_d,
            "decode_only_candidate_tok_s": cand_d,
            "decode_only_delta": cand_d - parent_d,
            "request_parent_tok_s": parent_r,
            "request_candidate_tok_s": cand_r,
            "request_delta": cand_r - parent_r,
        }
    return {
        "identity_ok": identity_ok,
        "primitive": primitive,
        "same_math_pass": same_ok,
        "model_quality_pass": quality_ok,
        "state_memory_pass": state_ok,
        "performance_pass": performance_ok,
        "correctness_ok": correctness_ok,
        "production_kept": keep,
        "verdict": verdict,
        "shipping_decode_attention": shipping,
        "reasons": reasons,
        "performance": performance,
        "tok_s_deltas": deltas,
        "opt115_aa_noise": noise,
        "gqa6_admitted": False,
        "mma_launched": False,
        "second_candidate": "llama_mma_decode_consumer",
    }


def maybe_flip_production_pins(keep: bool) -> None:
    if not keep:
        return
    text = pin_text()
    text = text.replace(
        "kSelectedDecodeAttentionVerifiedMax = 4096",
        "kSelectedDecodeAttentionVerifiedMax = 131072",
    )
    text = text.replace("kSelectedVec128NParts = 16", "kSelectedVec128NParts = 8")
    PIN_PATH.write_text(text, encoding="utf-8")


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    keep = bool(result.get("production_kept"))
    verdict = str(result.get("verdict"))
    performance = result.get("performance") or {}
    d128 = performance.get("d128") or {}
    d2048 = performance.get("d2048") or {}
    p4096 = performance.get("p4096") or {}
    quality = result.get("quality") or {}
    deltas = result.get("tok_s_deltas") or {}
    primitive = result.get("primitive") or {}
    lines = [
        "# OPT-130 — Implement the evidenced dense decode-attention improvement",
        "",
        f"Status: **{verdict}**. Shipping decode attention "
        f"`{result.get('shipping_decode_attention')}`. Hardware executed.",
        "",
        "Parent is authenticated post124 combined plus OPT-127 "
        "`decode_segments8`. Primary OPT-129 transfer "
        "`occupancy_partition_or_gqa_kv_reuse` is frozen as "
        "`hybrid_crossover@1024_vec128_open_nparts8`: occupancy-snapped "
        "legal `n_parts=8` (llama D2048 raw ~9) and `vec128_online` past "
        "verified_max 4096. GQA6 stays rejected. Rank-2 "
        "`llama_mma_decode_consumer` was not launched (unmatched BF16 MMA "
        "consumer; shadow F16 cache forbidden). Dense BF16 KV, causal "
        "visibility, partial RoPE, GQA mapping and output gates are unchanged.",
        "",
        "## Frozen dispatch",
        "",
        f"second_candidate_launched=`{result.get('mma_launched')}`. "
        f"gqa6_admitted=`{result.get('gqa6_admitted')}`. "
        "Graph topology remains two-valued (warp_query vs vec128_online); "
        "opening verified_max maps prefixes >4096 onto topology 1. "
        "`n_parts` is frozen per topology (not per-token occupancy) so "
        "OPT-126/127 captured grids stay representable. Eager fallback is "
        "preserved.",
        "",
        "## Primitive component",
        "",
        f"primitive ok=`{primitive.get('ok')}`.",
        "",
        "| prefix | parent ms | candidate ms | saving ms |",
        "|---:|---:|---:|---:|",
    ]
    for row in primitive.get("rows") or []:
        lines.append(
            f"| {row.get('prefix')} | {row.get('parent_ms')} | "
            f"{row.get('candidate_ms')} | {row.get('saving_ms')} |"
        )
    lines.extend(
        [
            "",
            "## Quality",
            "",
            (
                "OPT-058 invoked=`"
                + str(quality.get("opt058_invoked"))
                + "`; candidate NLL measured=`"
                + str(quality.get("candidate_nll_measured"))
                + "`; held-out PPL ratio=`"
                + str(quality.get("ppl_ratio"))
                + "`; wikitext PPL ratio=`"
                + str(quality.get("wikitext_ppl_ratio"))
                + "`."
            ),
            "",
            "## Engine tok/s versus parent (decode_segments8)",
            "",
            "| workload | parent decode-only | candidate decode-only | delta | "
            "parent request | candidate request | delta | geo | CI lower | p95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in (("d128", d128), ("d2048", d2048), ("p4096", p4096)):
        delta = deltas.get(name) or {}
        ci = (row.get("ci") or {}).get("ci_lower")
        lines.append(
            f"| {name} | {delta.get('decode_only_parent_tok_s')} | "
            f"{delta.get('decode_only_candidate_tok_s')} | "
            f"{delta.get('decode_only_delta')} | "
            f"{delta.get('request_parent_tok_s')} | "
            f"{delta.get('request_candidate_tok_s')} | "
            f"{delta.get('request_delta')} | {row.get('geo_ratio')} | "
            f"{ci} | {row.get('p95_ratio')} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"`{verdict}`. reasons=`{result.get('reasons')}`. "
            "On reject, production pins stay `hybrid_crossover@1024` / "
            "n_parts=16 / verified_max=4096 and tok/s speedup versus the "
            "sitting parent is 0.",
            "",
            f"claims_throughput: `{keep}`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixture(run_dir: Path, result: Mapping[str, Any], mode: str) -> None:
    quality = load_sidecar(run_dir, "quality.json") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-130",
        "mode": mode,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT_STACK,
        "candidate": CANDIDATE,
        "quality": quality,
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": result.get("performance"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "production_kept": bool(result.get("production_kept")),
        "verdict": result.get("verdict"),
        "report_path": str(REPORT.relative_to(ROOT)),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "reasons": result.get("reasons"),
        "identity": load_sidecar(run_dir, "identity.json") or {},
        "same_math": load_sidecar(run_dir, "same-math.json") or {},
        "inspect": load_sidecar(run_dir, "inspect.json") or {},
    }
    dump_json(FIXTURE, payload)


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    result = evaluate_keep(run_dir)
    result["quality"] = load_sidecar(run_dir, "quality.json") or {}
    result["primitive"] = load_sidecar(run_dir, "primitive.json") or {}
    maybe_flip_production_pins(bool(result.get("production_kept")))
    write_report(result)
    write_fixture(run_dir, result, mode)
    payload = {
        "schema_version": 1,
        "task": "OPT-130",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "verdict": result.get("verdict"),
        "production_kept": result.get("production_kept"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "report.json", payload)


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    return {
        "ok": not missing and payload.get("task") == "OPT-130",
        "task": "OPT-130",
        "missing": missing,
    }


def run_phase(run_dir: Path, mode: str, phase: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if phase == "inspect":
        return run_inspect(run_dir, mode)
    if phase == "identity":
        return run_named_native(run_dir, mode, "identity", [], needs_model=False)
    if phase == "primitive":
        return run_named_native(
            run_dir,
            mode,
            "primitive",
            ["--warmups", str(WARMUPS), "--samples", str(SCREEN_PAIRS)],
            needs_model=False,
        )
    if phase == "same-math":
        return run_named_native(
            run_dir, mode, "same-math", ["--prefix", "128", "--tokens", "8"]
        )
    if phase == "cancellation":
        return run_named_native(run_dir, mode, "cancellation", ["--prefix", "128"])
    if phase == "handoff":
        return run_named_native(run_dir, mode, "handoff", ["--prefix", "128"])
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
            run_dir, mode, "p4096", 4096, 0, PAIR_COUNT, critical=T_CRIT_DF9
        )
    if phase == "report":
        return run_report(run_dir, mode)
    raise DenseAttentionError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=("feedback", "acceptance", "release"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except DenseAttentionError as exc:
        print(
            json.dumps(
                {"task": "OPT-130", "phase": args.phase, "ok": False, "error": str(exc)}
            )
        )
        return 1
    print(
        json.dumps(
            {
                "task": "OPT-130",
                "phase": args.phase,
                "ok": bool(payload.get("ok", True)),
                "verdict": payload.get("verdict"),
            }
        )
    )
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
