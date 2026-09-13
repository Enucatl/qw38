"""OPT-121 keep/reject selective Q8/Q6→Q4_K weight requant vs post-OPT-119."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
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
from tools.opt116_generated_quality import cache_source_tokens  # noqa: E402
from tools.quality.cache_path import PREFIXES  # noqa: E402
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt121_weight_traffic_contract.json"
ITERATION = ROOT / "pins/opt121_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt121_weight_traffic.json"
REPORT = ROOT / "evidence/optimization/opt121-weight-traffic/REPORT.md"
EVIDENCE = REPORT.parent
HEADER = ROOT / "cuda/opt121_weight_requant.cuh"
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT116_GPU = ROOT / "build/optimization-runs/OPT-116"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
INVENTORY = ROOT / "pins/tensor_inventory.json"
GPU_LOCK = ROOT / "build/optimization-runs/qw38-gpu.lock"
NATIVE = "build/qw38-cuda-opt121-weight-traffic-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
LONG_CACHE_NATIVE = "build/qw38-cuda-opt116-generated-quality-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
REQUIRE_CANDIDATE_NLL = (
    True  # require_candidate_nll: always measure candidate OPT-058 NLL
)
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT121_WEIGHT_TRAFFIC_RESULT="
COUNTS_PREFIX = "QW38_OPT121_NATIVE_COUNTS="
PHASES = (
    "preflight",
    "reference",
    "inventory",
    "greedy-parity",
    "api",
    "format",
    "quality",
    "long-cache",
    "state-memory",
    "d128",
    "d2048",
    "p4096",
    "report",
)
PARENT = "post113_selected"
CANDIDATES = ("q8_to_q4k", "q8_q6_to_q4k", "fp4_study")
CONFIG_ENUM = {
    "none": 'kSelectedWeightRequantConfig[] = "none"',
    "q8_to_q4k": 'kSelectedWeightRequantConfig[] = "q8_to_q4k"',
    "q8_q6_to_q4k": 'kSelectedWeightRequantConfig[] = "q8_q6_to_q4k"',
    "fp4_study": 'kSelectedWeightRequantConfig[] = "fp4_study"',
}
WARMUPS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
POST113_QUALITY = {
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
    "primary_target",
    "inventory",
    "reference",
    "format",
    "quality",
    "long_cache",
    "state_memory",
    "performance",
    "selected_weight_requant",
    "production_kept",
    "verdict",
    "report_path",
)
KEEP_ROLES = ("gdn_alpha", "gdn_beta", "input_norm", "ffn_norm", "token_embedding")
CONVERT_Q8 = (
    "gdn_packed_qkv",
    "gdn_value_gate",
    "gdn_output",
    "attention_q_gate",
    "attention_k",
    "attention_v",
)
CONVERT_Q6 = ("attention_output", "output_projection")


class WeightTrafficError(RuntimeError):
    """OPT-121 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-121":
        raise WeightTrafficError("weight-traffic contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-121", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-121 mode={mode} phase={family} "
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
    records = [
        json.loads(line.removeprefix(prefix))
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if not records:
        raise WeightTrafficError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        os.environ.setdefault("QW38_CUDA_TEST_TIER", tier)
        return list(args)
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    if check and completed.returncode != 0:
        raise WeightTrafficError(
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


def summarize_pairs(
    record: Mapping[str, Any], *, critical: float = T_CRIT_DF9
) -> dict[str, Any]:
    pairs = list(record.get("pairs") or [])
    parent = [float(row["A"]["tok_s"]) for row in pairs]
    candidate = [float(row["B"]["tok_s"]) for row in pairs]
    parent_p95 = [float(row["A"]["p95_ms"]) for row in pairs]
    candidate_p95 = [float(row["B"]["p95_ms"]) for row in pairs]
    ratios = [
        cand / ctrl if ctrl > 0.0 else 0.0 for ctrl, cand in zip(parent, candidate)
    ]
    ci = log_speed_ratio_ci_lower(parent, candidate, threshold=1.0, critical=critical)
    parent_p95_mean = mean(parent_p95)
    candidate_p95_mean = mean(candidate_p95)
    if parent_p95_mean > 0.0:
        p95_ratio = candidate_p95_mean / parent_p95_mean
    else:
        p95_ratio = 0.0
    return {
        "n": len(pairs),
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


def candidate_for(run_dir: Path) -> str:
    selected = load_sidecar(run_dir, "selected-candidate.json") or {}
    ident = str(selected.get("candidate") or "q8_to_q4k")
    return ident if ident in CANDIDATES else "q8_to_q4k"


def rank_roles() -> list[dict[str, Any]]:
    inventory = load_json(INVENTORY)
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"bytes": 0, "count": 0, "dtype": ""}
    )
    for tensor in inventory.get("tensors") or []:
        role = str(tensor.get("role") or "other")
        totals[role]["bytes"] += int(tensor.get("storage_bytes") or 0)
        totals[role]["count"] += 1
        totals[role]["dtype"] = str(tensor.get("dtype") or totals[role]["dtype"])
    ranked = []
    for role, row in totals.items():
        lookup = role == "token_embedding"
        traffic = 0 if lookup else int(row["bytes"])
        ranked.append(
            {
                "role": role,
                "dtype": row["dtype"],
                "tensors": row["count"],
                "storage_bytes": int(row["bytes"]),
                "bytes_per_token": traffic,
                "embedding_lookup_excluded": lookup,
                "keep_precision": role in KEEP_ROLES,
                "q8_to_q4k": role in CONVERT_Q8,
                "q8_q6_to_q4k": role in CONVERT_Q8 or role in CONVERT_Q6,
            }
        )
    ranked.sort(key=lambda item: int(item["bytes_per_token"]), reverse=True)
    return ranked


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    source, dirty = git_identity()
    roles = rank_roles()
    payload = {
        "schema_version": 1,
        "task": "OPT-121",
        "phase": "preflight",
        "mode": mode,
        "ok": bool(auth.get("ok")),
        "authenticated_post113": auth,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "primary_target": "decode",
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": bool(dirty),
        "opt120_packed_kv_rejected": True,
        "parent_execution_graphs": "ffn_only",
        "max_configs": 3,
        "candidates": list(CANDIDATES),
        "calibration": "opt121_calib_v1_seed_20260913",
        "calibration_distinct_from_opt116": True,
        "role_ranking": roles,
        "iteration_target": iteration["target"],
        "diagnostics_make_target": iteration["diagnostics_make_target"],
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    store_sidecar(run_dir, "selected-candidate.json", {"candidate": "q8_to_q4k"})
    return store_sidecar(run_dir, "preflight.json", payload)


def run_named_native(
    run_dir: Path,
    mode: str,
    phase: str,
    extra: Sequence[str],
    *,
    check: bool = True,
) -> dict[str, Any]:
    ident = candidate_for(run_dir)
    completed = run_native(
        [
            f"./{NATIVE}",
            "--workload",
            phase,
            "--weight-requant",
            ident,
            *extra,
            MODEL,
        ],
        tier="correctness",
        check=check,
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    sidecar_path = run_dir / f"{phase}.txt"
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    record["sidecar"] = workspace_relative(sidecar_path)
    return store_sidecar(run_dir, f"{phase}.json", record)


def run_session_ab_phase(
    run_dir: Path,
    mode: str,
    phase: str,
    prefix: int,
    tokens: int,
    samples: int,
    warmups: int,
    capacity: int | None = None,
) -> dict[str, Any]:
    ident = candidate_for(run_dir)
    args = [
        f"./{NATIVE}",
        "--workload",
        "session-ab",
        "--weight-requant",
        ident,
        "--execution-graphs",
        "ffn_only",
        "--prefix",
        str(prefix),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--tokens",
        str(tokens),
    ]
    if capacity:
        args.extend(["--capacity", str(capacity)])
    args.append(MODEL)
    completed = run_native(args, tier="acceptance")
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    critical = T_CRIT_DF9 if samples >= 10 else T_CRIT_DF4
    summary = summarize_pairs(record, critical=critical)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    record["summary"] = summary
    record["candidate"] = ident
    sidecar_path = run_dir / f"{phase}.txt"
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return store_sidecar(run_dir, f"{phase}.json", record)


def quality_config(weight_requant: str | None) -> dict[str, Any]:
    cfg = build_quality_config(enabled=True, selectors=POST113_QUALITY)
    if weight_requant:
        cfg["weight_requant"] = weight_requant
    return cfg


def run_quality_native(
    run_dir: Path, config_id: str, weight_requant: str | None
) -> dict[str, Any]:
    cfg_path = run_dir / f"quality-config-{config_id}.json"
    dump_json(cfg_path, quality_config(weight_requant))
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
    if weight_requant:
        extra.extend(["--weight-requant", weight_requant])
    completed = run_native(
        [f"./{QUALITY_NATIVE}", MODEL, "--workload", "quality-baseline", *extra],
        tier="acceptance",
    )
    sidecar_path = run_dir / f"quality-{config_id}.txt"
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    if not restored:
        raise WeightTrafficError("--quality restored packed or r2 defaults")
    return {
        "id": config_id,
        "weight_requant": weight_requant,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
        "same_math_copied_from_control": False,
    }


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    ident = candidate_for(run_dir)
    control = run_quality_native(run_dir, "post113_selected", None)
    candidate = run_quality_native(run_dir, ident, ident)
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
    control_short = nll_from_cases(control["cases"], "recurrence_short")
    cand_short = nll_from_cases(candidate["cases"], "recurrence_short")
    control_long = nll_from_cases(control["cases"], "recurrence_long")
    cand_long = nll_from_cases(candidate["cases"], "recurrence_long")
    rec_short = (
        cand_short - control_short
        if control_short is not None and cand_short is not None
        else None
    )
    rec_long = (
        cand_long - control_long
        if control_long is not None and cand_long is not None
        else None
    )
    recurrence_ok = (
        rec_short is not None
        and rec_long is not None
        and rec_short <= 0.02
        and rec_long <= 0.02
    )
    opt116 = load_json(OPT116_FIXTURE) if OPT116_FIXTURE.is_file() else {}
    ppl = (
        ((opt116.get("authenticate") or {}).get("contracts") or {}).get("contracts")
        or {}
    ).get("opt116_generated_v1", {})
    parent_ppl = float(((ppl.get("ppl") or {}).get("aggregate_ratio")) or 1.0)
    quality_ok = (
        (not REQUIRE_CANDIDATE_NLL or not incomplete)
        and not incomplete
        and held_ratio is not None
        and held_ratio <= 1.01
        and (wiki_ratio is None or wiki_ratio <= 1.01)
        and recurrence_ok
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-121",
        "phase": "quality",
        "mode": mode,
        "ok": quality_ok,
        "same_path_exact": False,
        "opt058_invoked": True,
        "candidate_nll_measured": not incomplete,
        "opt116_contract_id": "opt116_generated_v1",
        "ppl_ratio": held_ratio,
        "wikitext_ppl_ratio": wiki_ratio,
        "ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "recurrence_short_delta": rec_short,
        "recurrence_long_delta": rec_long,
        "recurrence_ok": recurrence_ok,
        "parent_authenticated_ppl_ratio": parent_ppl,
        "control": control,
        "candidate": candidate,
        "weight_requant": ident,
        "quality_flag": "--quality",
        "family_plan": family_plan(mode, "quality"),
        "measured_at": utc_now(),
    }
    store_sidecar(
        run_dir,
        "selected-candidate.json",
        {"candidate": ident, "quality_ok": quality_ok},
    )
    return store_sidecar(run_dir, "quality.json", payload)


def prepare_long_cache_jobs(run_dir: Path) -> dict[str, Path]:
    cases_path = run_dir / "generate-cases.tsv"
    jobs_path = run_dir / "cache-jobs.tsv"
    tokens_path = run_dir / "cache-source-tokens.bin"
    src_cases = OPT116_GPU / "generate-cases.tsv"
    src_jobs = OPT116_GPU / "cache-jobs.tsv"
    src_tokens = OPT116_GPU / "cache-source-tokens.bin"
    if src_cases.is_file() and not cases_path.is_file():
        shutil.copy(src_cases, cases_path)
    if src_jobs.is_file() and not jobs_path.is_file():
        shutil.copy(src_jobs, jobs_path)
    if src_tokens.is_file() and not tokens_path.is_file():
        shutil.copy(src_tokens, tokens_path)
    if not jobs_path.is_file():
        lines = ["# id prefix scored capacity retrieval"]
        for row in PREFIXES:
            positions = ",".join(str(pos) for pos in row["retrieval_positions"])
            lines.append(
                f"{row['id']}\t{row['prefix_tokens']}\t{row['scored_or_generated']}\t"
                f"{row['capacity']}\t{positions}"
            )
        jobs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not tokens_path.is_file():
        tokens = cache_source_tokens()
        with tokens_path.open("wb") as handle:
            handle.write(struct.pack("<%dI" % len(tokens), *tokens))
    if not cases_path.is_file() or cases_path.stat().st_size < 64:
        from tools.opt116_generated_quality import write_gpu_jobs

        generated = write_gpu_jobs()
        shutil.copy(generated["cases"], cases_path)
        if not jobs_path.is_file():
            shutil.copy(generated["jobs"], jobs_path)
        if not tokens_path.is_file():
            shutil.copy(generated["tokens"], tokens_path)
    return {"cases": cases_path, "jobs": jobs_path, "tokens": tokens_path}


def parent_cache_nll() -> dict[str, float]:
    payload = load_json(OPT116_FIXTURE) if OPT116_FIXTURE.is_file() else {}
    cache = payload.get("cache_gdn") or payload.get("cache") or {}
    spans = list(cache.get("spans") or [])
    return {
        str(row.get("id")): float(
            row.get("post113_mean_nll") or row.get("candidate_mean_nll") or 0.0
        )
        for row in spans
        if row.get("id")
        and (row.get("post113_mean_nll") or row.get("candidate_mean_nll"))
    }


def run_long_cache(run_dir: Path, mode: str) -> dict[str, Any]:
    ident = candidate_for(run_dir)
    jobs = prepare_long_cache_jobs(run_dir)
    cfg_path = run_dir / f"quality-config-long-cache-{ident}.json"
    dump_json(cfg_path, quality_config(ident))
    out_path = run_dir / f"long-cache-{ident}.json"
    extra = [
        f"./{LONG_CACHE_NATIVE}",
        MODEL,
        "--workload",
        "gpu-baseline",
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
        "--weight-requant",
        ident,
        "--cases",
        workspace_relative(jobs["cases"]),
        "--cache-jobs",
        workspace_relative(jobs["jobs"]),
        "--cache-tokens",
        workspace_relative(jobs["tokens"]),
        "--out",
        workspace_relative(out_path),
    ]
    completed = run_native(extra, tier="acceptance")
    sidecar_path = run_dir / f"long-cache-{ident}.txt"
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    payload: dict[str, Any] = {}
    if out_path.is_file():
        payload = json.loads(out_path.read_text(encoding="utf-8", errors="replace"))
    spans = list(payload.get("cache_spans") or payload.get("spans") or [])
    parent_nll = parent_cache_nll()
    ratios: list[float] = []
    for row in spans:
        ident_span = str(row.get("id") or "")
        cand = row.get("candidate_mean_nll")
        parent = parent_nll.get(ident_span)
        if cand is None:
            continue
        if parent is None or parent <= 0.0:
            parent = row.get("post113_mean_nll")
        if parent is None or float(parent) <= 0.0:
            continue
        ratio = ppl_ratio(float(cand), float(parent))
        row["ppl_ratio_vs_opt116"] = ratio
        ratios.append(ratio)
    required_ids = {row["id"] for row in PREFIXES}
    have_ids = {str(row.get("id")) for row in spans}
    cache_ok = (
        bool(payload.get("long_cache_gpu_executed"))
        and required_ids.issubset(have_ids)
        and all(bool(row.get("finite", True)) for row in spans)
        and all(ratio <= 1.01 for ratio in ratios)
        and len(ratios) == len(required_ids)
    )
    record = {
        "schema_version": 1,
        "task": "OPT-121",
        "phase": "long-cache",
        "mode": mode,
        "ok": cache_ok,
        "weight_requant": ident,
        "opt116_contract_id": "opt116_generated_v1",
        "long_cache_gpu_executed": bool(payload.get("long_cache_gpu_executed")),
        "free_running_gpu_executed": bool(payload.get("free_running_gpu_executed")),
        "spans": spans,
        "ppl_ratios": ratios,
        "ppl_ratio_max": 1.01,
        "required_prefixes": [8192, 32768, 131040],
        "sidecar": workspace_relative(sidecar_path),
        "out": workspace_relative(out_path),
        "family_plan": family_plan(mode, "long-cache"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "long-cache.json", record)


def run_state_memory(run_dir: Path, mode: str) -> dict[str, Any]:
    native = run_named_native(
        run_dir,
        mode,
        "state-memory",
        ["--execution-graphs", "ffn_only", "--prefix", "128", "--tokens", "1"],
    )
    api = load_sidecar(run_dir, "api.json") or {}
    fmt = load_sidecar(run_dir, "format.json") or {}
    memory = load_json(MEMORY)
    fit = bool(memory.get("post_graph_admitted"))
    native["memory_fit_parent_admitted"] = fit
    native["128k_fit"] = fit
    native["api_ok"] = bool(api.get("ok"))
    native["format_ok"] = bool(fmt.get("ok"))
    native["ok"] = (
        bool(native.get("ok"))
        and fit
        and bool(api.get("ok", True))
        and bool(fmt.get("ok", True))
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
    reference = load_sidecar(run_dir, "reference.json") or {}
    inventory = load_sidecar(run_dir, "inventory.json") or {}
    greedy = load_sidecar(run_dir, "greedy-parity.json") or {}
    api = load_sidecar(run_dir, "api.json") or {}
    fmt = load_sidecar(run_dir, "format.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    long_cache = load_sidecar(run_dir, "long-cache.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    noise = aa_noise_from_opt115()
    kernel_ok = (
        bool(reference.get("ok"))
        and bool(inventory.get("ok"))
        and bool(greedy.get("ok"))
        and bool(api.get("ok"))
        and bool(fmt.get("ok"))
    )
    quality_ok = bool(quality.get("ok") and quality.get("candidate_nll_measured"))
    rec_short = quality.get("recurrence_short_delta")
    rec_long = quality.get("recurrence_long_delta")
    if rec_short is None or rec_long is None:
        control_cases = list((quality.get("control") or {}).get("cases") or [])
        cand_cases = list((quality.get("candidate") or {}).get("cases") or [])
        c_short = nll_from_cases(control_cases, "recurrence_short")
        b_short = nll_from_cases(cand_cases, "recurrence_short")
        c_long = nll_from_cases(control_cases, "recurrence_long")
        b_long = nll_from_cases(cand_cases, "recurrence_long")
        rec_short = (
            b_short - c_short if c_short is not None and b_short is not None else None
        )
        rec_long = (
            b_long - c_long if c_long is not None and b_long is not None else None
        )
    if rec_short is None or rec_long is None or rec_short > 0.02 or rec_long > 0.02:
        quality_ok = False
    long_ok = bool(long_cache.get("ok") and long_cache.get("long_cache_gpu_executed"))
    model_ok = quality_ok and long_ok
    state_ok = bool(state.get("ok"))
    reasons: list[str] = []
    if not kernel_ok:
        reasons.append("kernel_correctness_failed")
    if not quality_ok:
        reasons.append("quality_failed")
    if not long_ok:
        reasons.append("long_cache_failed")
    if not state_ok:
        reasons.append("state_memory_failed")
    performance: dict[str, Any] = {}
    performance_ok = True
    workloads = (
        ("d128", True, False),
        ("d2048", True, False),
        ("p4096", False, True),
    )
    for name, target, prefill in workloads:
        row = load_sidecar(run_dir, f"{name}.json") or {}
        summary = dict(row.get("summary") or {})
        ci = dict(summary.get("ci") or {})
        geo = float(summary.get("geo_ratio") or 0.0)
        p95_ratio = float(summary.get("p95_ratio") or 0.0)
        ci_lower = ci.get("ci_lower")
        n_pairs = int(summary.get("n") or 0)
        aa = float(noise.get(name, 0.0001))
        if target:
            if n_pairs >= 5 and ci_lower is not None:
                better = float(ci_lower) > 1.0 and abs(geo - 1.0) > aa
            else:
                better = geo > 1.0 and abs(geo - 1.0) > aa
            p95_ok = p95_ratio <= 1.05 if p95_ratio else True
            row_ok = better and p95_ok
            if not row_ok:
                performance_ok = False
                reasons.append(f"{name}_throughput_gate")
        else:
            point_ok = geo >= 0.98 if geo else False
            p95_ok = (
                True
                if prefill and p95_ratio == 0.0
                else (p95_ratio <= 1.05 if p95_ratio else True)
            )
            row_ok = point_ok and p95_ok
            if not row_ok:
                performance_ok = False
                reasons.append(f"{name}_non_target_guard")
        performance[name] = {
            **summary,
            "target": target,
            "ok": row_ok,
            "aa_noise": aa,
        }
    ident = candidate_for(run_dir)
    keep = kernel_ok and model_ok and state_ok and performance_ok
    if bool(reference.get("fp4_capability_absent")) and ident == "fp4_study":
        keep = False
        reasons.append("fp4_capability_absent")
    if not quality_ok:
        verdict = "quality_blocked"
    elif keep:
        verdict = "keep"
    else:
        verdict = "reject"
    measured: dict[str, float] = {}
    sitting: dict[str, float] = {}
    for name, _target, _prefill in workloads:
        row = performance.get(name) or {}
        parent = float(row.get("parent_tok_s") or 0.0)
        cand = float(row.get("candidate_tok_s") or 0.0)
        measured[name] = cand - parent
        sitting[name] = (cand - parent) if keep else 0.0
    return {
        "candidate": ident,
        "selected_weight_requant": ident if keep else "none",
        "production_kept": keep,
        "verdict": verdict,
        "kernel_correctness": kernel_ok,
        "model_quality_pass": model_ok,
        "state_memory_pass": state_ok,
        "performance_pass": performance_ok,
        "performance": performance,
        "reasons": reasons,
        "tok_s_delta_measured": measured,
        "tok_s_delta": sitting,
    }


def _row(label: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {float(summary.get('parent_tok_s') or 0.0):.3f} | "
        f"{float(summary.get('candidate_tok_s') or 0.0):.3f} | "
        f"{float(summary.get('geo_ratio') or 0.0):.4f} | "
        f"{float(((summary.get('ci') or {}).get('ci_lower')) or 0.0):.4f} | "
        f"{float(summary.get('p95_ratio') or 0.0):.4f} | "
        f"{'pass' if summary.get('ok') else 'fail'} |"
    )


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    inventory = result.get("inventory") or {}
    reference = result.get("reference") or {}
    quality = result.get("quality") or {}
    long_cache = result.get("long_cache") or {}
    state = result.get("state_memory") or {}
    performance = result.get("performance") or {}
    keep = bool(result.get("production_kept"))
    lines = [
        "# OPT-121 weight traffic requant",
        "",
        f"Verdict **{result.get('verdict')}**. Candidate "
        f"`{result.get('candidate')}` vs post-OPT-119 `ffn_only`. "
        f"Production pin `{result.get('selected_weight_requant')}`.",
        "",
        "## Role map and conversion",
        "",
        f"Converted tensors `{inventory.get('converted_tensors')}` src "
        f"`{inventory.get('src_bytes')}` dst `{inventory.get('dst_bytes')}` "
        f"saved `{inventory.get('bytes_saved')}` conversion_ms "
        f"`{inventory.get('conversion_ms')}`. Encoding "
        f"`q4_k_minmax_ref`. Irreversible. GGUF unmodified. "
        f"No dense shadow. FP4 study capability_absent="
        f"`{reference.get('fp4_capability_absent')}`.",
        "",
        "## Quality",
        "",
        f"OPT-058 `--quality` invoked; restored packed/r2=`false`; "
        f"candidate_nll_measured=`{quality.get('candidate_nll_measured')}`; "
        f"held-out PPL ratio `{quality.get('ppl_ratio')}`; "
        f"wikitext PPL ratio `{quality.get('wikitext_ppl_ratio')}`; "
        f"recurrence ΔNLL short `{quality.get('recurrence_short_delta')}` "
        f"long `{quality.get('recurrence_long_delta')}` "
        f"(max 0.02, ok=`{quality.get('recurrence_ok')}`); "
        "opt116=`opt116_generated_v1`.",
        f"long-cache GPU `{long_cache.get('long_cache_gpu_executed')}` "
        f"ok=`{long_cache.get('ok')}`.",
        f"state/memory=`{state.get('ok')}` 128k_fit=`{state.get('128k_fit')}`.",
        "",
        "## Full-engine A/B",
        "",
        "| Workload | parent tok/s | candidate tok/s | geo ratio | CI lower | "
        "p95 ratio | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
        _row("D128 (target)", performance.get("d128") or {}),
        _row("D2048 (target)", performance.get("d2048") or {}),
        _row("P4096 (guard)", performance.get("p4096") or {}),
        "",
        f"Measured tok/s deltas `{result.get('tok_s_delta_measured')}`. "
        f"Sitting deltas `{result.get('tok_s_delta')}`. Keep={keep}. Reasons: "
        f"{', '.join(result.get('reasons') or []) or 'none'}.",
        "",
        (
            "Production dispatch ships the selected requant map."
            if keep
            else "Rejected production dispatch retains baseline Q8/Q6 weights. "
            "Memory-only savings cannot admit a keep."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def pin_production_config(ident: str) -> None:
    text = HEADER.read_text(encoding="utf-8")
    for name in CONFIG_ENUM:
        text = text.replace(
            f'constexpr char kSelectedWeightRequantConfig[] = "{name}";',
            f'constexpr char kSelectedWeightRequantConfig[] = "{ident}";',
        )
    HEADER.write_text(text, encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    decision = evaluate_keep(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-121",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": contract["parent"],
        "candidate": decision["candidate"],
        "primary_target": "decode",
        "inventory": load_sidecar(run_dir, "inventory.json") or {},
        "reference": load_sidecar(run_dir, "reference.json") or {},
        "greedy_parity": load_sidecar(run_dir, "greedy-parity.json") or {},
        "api": load_sidecar(run_dir, "api.json") or {},
        "format": load_sidecar(run_dir, "format.json") or {},
        "quality": load_sidecar(run_dir, "quality.json") or {},
        "long_cache": load_sidecar(run_dir, "long-cache.json") or {},
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": decision["performance"],
        "selected_weight_requant": decision["selected_weight_requant"],
        "production_kept": decision["production_kept"],
        "verdict": decision["verdict"],
        "independent_verdicts": {
            "kernel_correctness": decision["kernel_correctness"],
            "model_quality": decision["model_quality_pass"],
            "state_memory": decision["state_memory_pass"],
            "performance": decision["performance_pass"],
            "production_kept": decision["production_kept"],
        },
        "reasons": decision["reasons"],
        "tok_s_delta": decision["tok_s_delta"],
        "tok_s_delta_measured": decision["tok_s_delta_measured"],
        "report_path": str(contract["report_path"]),
        "measured_at": utc_now(),
        "family_plan": family_plan(mode, "report"),
        "ok": True,
    }
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise WeightTrafficError(f"fixture missing {key}")
    dump_json(FIXTURE, payload)
    write_report(payload)
    pin_production_config(decision["selected_weight_requant"])
    contract["selected_weight_requant"] = decision["selected_weight_requant"]
    contract["candidate"] = decision["candidate"]
    dump_json(CONTRACT, contract)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(
    phase: str, run_dir: Path, mode: str, weight_requant: str
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    existing = load_sidecar(run_dir, "selected-candidate.json") or {}
    if weight_requant in CANDIDATES:
        store_sidecar(
            run_dir,
            "selected-candidate.json",
            {**existing, "candidate": weight_requant},
        )
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "reference":
        return run_named_native(run_dir, mode, "reference", [])
    if phase == "inventory":
        return run_named_native(
            run_dir, mode, "inventory", ["--execution-graphs", "ffn_only"]
        )
    if phase == "greedy-parity":
        return run_named_native(
            run_dir,
            mode,
            "greedy-parity",
            ["--execution-graphs", "ffn_only", "--prefix", "8"],
        )
    if phase == "api":
        return run_named_native(
            run_dir,
            mode,
            "api",
            ["--execution-graphs", "ffn_only", "--prefix", "8"],
        )
    if phase == "format":
        return run_named_native(
            run_dir,
            mode,
            "format",
            ["--execution-graphs", "ffn_only", "--prefix", "8"],
        )
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "long-cache":
        return run_long_cache(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "d128":
        return run_session_ab_phase(
            run_dir, mode, "d128", 128, DECODE_TOKENS, PAIR_COUNT, WARMUPS
        )
    if phase == "d2048":
        return run_session_ab_phase(
            run_dir, mode, "d2048", 2048, DECODE_TOKENS, PAIR_COUNT, WARMUPS
        )
    if phase == "p4096":
        return run_session_ab_phase(
            run_dir, mode, "p4096", 4096, 0, PAIR_COUNT, WARMUPS
        )
    if phase == "report":
        return run_report(run_dir, mode)
    raise WeightTrafficError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--weight-requant", default="q8_to_q4k", choices=CANDIDATES)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    result = run_phase(args.phase, Path(args.run_dir), args.mode, args.weight_requant)
    json.dump({"phase": args.phase, "ok": bool(result.get("ok", True))}, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("ok", True) or args.phase == "report" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WeightTrafficError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
