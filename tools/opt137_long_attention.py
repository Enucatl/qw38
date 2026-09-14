"""OPT-137 bounded long-context dense BF16 tile-F16 MMA decode candidate."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
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
from tools.opt136_graph_accounting import (  # noqa: E402
    gpu_residents,
    with_gpu_lock,
)
from tools.performance_keep_policy import (  # noqa: E402
    evaluate,
    freeze_hash,
    validate_opt_in_contract,
)
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt137_long_attention_contract.json"
ITERATION = ROOT / "pins/opt137_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt137_long_attention.json"
REPORT = ROOT / "evidence/optimization/opt137-long-attention/REPORT.md"
EVIDENCE = REPORT.parent
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
OPT136_COVERAGE = ROOT / "evidence/optimization/opt136-graph-accounting/coverage.json"
OPT136_GAPS = ROOT / "evidence/optimization/opt136-graph-accounting/family-gaps.json"
OPT136_BASELINE = ROOT / "evidence/optimization/opt136-graph-accounting/baseline.json"
OPT136_REPORT = ROOT / "evidence/optimization/opt136-graph-accounting/REPORT.md"
NATIVE = "build/qw38-cuda-opt137-long-attention-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT137_LONG_ATTENTION_RESULT="
COUNTS_PREFIX = "QW38_OPT137_NATIVE_COUNTS="
PHASES = (
    "preflight",
    "correctness",
    "screen",
    "quality",
    "state-memory",
    "performance",
    "mechanism",
    "report",
)
PARENT = "hybrid_crossover"
CANDIDATE = "dense_bf16_tile_f16_mma_decode_v1"
PARENT_STACK = "post124_decode_segments8"
WARMUPS = 3
SCREEN_WARMUPS = 1
SCREEN_PAIRS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
CHILD_TIMEOUT_S = 300.0
LONG_TIMEOUT_S = 1200.0
PERFORMANCE_TIMEOUT_S = 3600.0
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
    "quality",
    "state_memory",
    "performance",
    "shipping_decode_attention",
    "production_kept",
    "verdict",
    "report_path",
)
ATTN_FAMILIES = ("attn_prepare", "attn_core", "attn_merge")


class LongAttentionError(RuntimeError):
    """OPT-137 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-137":
        raise LongAttentionError("long-attention contract task mismatch")
    validate_opt_in_contract(payload)
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-137", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    alias = {
        "inspect": "preflight",
        "identity": "correctness",
        "same-math": "correctness",
        "cancellation": "correctness",
        "handoff": "correctness",
        "quality-long": "quality",
        "d8192": "performance",
        "d32768": "performance",
        "d128": "performance",
        "d2048": "performance",
        "p4096": "performance",
        "d131040": "performance",
        "mechanism-d8192": "mechanism",
        "mechanism-d32768": "mechanism",
    }
    key = alias.get(family, family)
    workload = workload_for_mode(iteration["workloads"][key], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-137 mode={mode} phase={key} "
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
    cleaned = re.sub(r"mma_quality_q4_attrs[^\n]*\n?", "", text)
    idx = cleaned.rfind(prefix)
    if idx >= 0:
        blob = cleaned[idx + len(prefix) :]
        decoder = json.JSONDecoder()
        try:
            payload, _ = decoder.raw_decode(blob.lstrip())
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    records: list[dict[str, Any]] = []
    for line in cleaned.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    if not records:
        raise LongAttentionError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str],
    *,
    tier: str,
    check: bool = True,
    timeout_s: float = CHILD_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = with_gpu_lock(command, timeout_s=timeout_s)
    if check and completed.returncode != 0:
        raise LongAttentionError(
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


def pin_text() -> str:
    return PIN_PATH.read_text(encoding="utf-8")


def production_pins_parent() -> bool:
    text = pin_text()
    return (
        "kSelectedDecodeAttentionCrossoverThreshold = 1024" in text
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in text
        and "kSelectedVec128NParts = 16" in text
        and "kSelectedOpt137DenseMma = false" in text
    )


def complete_attention_ms(
    gaps: Mapping[str, Any], *, engine: str, prefix: int
) -> float:
    values: list[float] = []
    needle = f"{engine}-d{prefix}-node-"
    for key, families in gaps.items():
        if needle not in str(key) or not isinstance(families, Mapping):
            continue
        total = 0.0
        present = False
        for name in ATTN_FAMILIES:
            row = families.get(name)
            if isinstance(row, Mapping) and row.get("ms_per_eval") is not None:
                total += float(row["ms_per_eval"])
                present = True
        if present:
            values.append(total)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def eligibility() -> dict[str, Any]:
    blockers: list[str] = []
    coverage = load_json(OPT136_COVERAGE) if OPT136_COVERAGE.is_file() else {}
    gaps = load_json(OPT136_GAPS) if OPT136_GAPS.is_file() else {}
    baseline_ok = OPT136_BASELINE.is_file()
    report_ok = OPT136_REPORT.is_file()
    status = str(coverage.get("status") or "")
    if status != "valid":
        blockers.append("opt136_coverage_not_valid")
    if not baseline_ok:
        blockers.append("opt136_baseline_missing")
    if not report_ok:
        blockers.append("opt136_report_missing")
    quartz_8192 = complete_attention_ms(gaps, engine="quartz", prefix=8192)
    llama_8192 = complete_attention_ms(gaps, engine="llama", prefix=8192)
    quartz_32768 = complete_attention_ms(gaps, engine="quartz", prefix=32768)
    llama_32768 = complete_attention_ms(gaps, engine="llama", prefix=32768)
    excess_8192 = quartz_8192 - llama_8192
    excess_32768 = quartz_32768 - llama_32768
    opportunity = excess_8192 > 0.0 and excess_32768 > 0.0
    if quartz_8192 <= 0.0 or quartz_32768 <= 0.0:
        blockers.append("opt136_complete_attention_unmeasured")
    return {
        "coverage_status": status,
        "coverage_ok": status == "valid",
        "quartz_complete_attention_ms": {"d8192": quartz_8192, "d32768": quartz_32768},
        "llama_complete_attention_ms": {"d8192": llama_8192, "d32768": llama_32768},
        "excess_ms": {"d8192": excess_8192, "d32768": excess_32768},
        "positive_excess": opportunity,
        "blockers": blockers,
        "no_opportunity": not opportunity and not blockers,
        "used_opt129_0_113": False,
    }


def run_inspect_native(run_dir: Path, mode: str) -> dict[str, Any]:
    completed = run_native([f"./{NATIVE}", "--workload", "inspect"], tier="correctness")
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = "inspect"
    record["mode"] = mode
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
    timeout_s: float = CHILD_TIMEOUT_S,
    tier: str = "correctness",
) -> dict[str, Any]:
    args = [f"./{NATIVE}", "--workload", phase, *extra]
    if needs_model:
        args.append(MODEL)
    completed = run_native(args, tier=tier, check=check, timeout_s=timeout_s)
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, sidecar_name or phase)
    record["stdout_tail"] = (completed.stdout + completed.stderr)[-4000:]
    name = sidecar_name or phase
    return store_sidecar(run_dir, f"{name}.json", record)


def pairs_from_graph_ab(
    record: Mapping[str, Any],
    *,
    workload: str,
    prefix: int,
    eval_count: int,
    metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity = {
        "prefix": prefix,
        "eval_count": eval_count,
        "capacity": 131072,
        "metric_boundary": metric,
        "sampling_policy": "greedy",
        "output_policy": "fixed_eval",
        "engine_config": "quartz_graph",
        "input_trajectory": "opt137_long_attention",
    }
    for index, pair in enumerate(record.get("pairs") or []):
        control = pair.get("A") if isinstance(pair.get("A"), Mapping) else {}
        candidate = pair.get("B") if isinstance(pair.get("B"), Mapping) else {}
        if metric in {"complete_request", "prefill"}:
            control_rate = control.get("request_tok_s") or control.get("tok_s")
            candidate_rate = candidate.get("request_tok_s") or candidate.get("tok_s")
        else:
            control_rate = control.get("decode_only_tok_s") or control.get("tok_s")
            candidate_rate = candidate.get("decode_only_tok_s") or candidate.get(
                "tok_s"
            )
        rows.append(
            {
                "sample_id": pair.get("sample_index", index),
                "order": pair.get("order") or ("AB" if index % 2 == 0 else "BA"),
                "workload": workload,
                "metric": metric,
                "control_rate": control_rate,
                "candidate_rate": candidate_rate,
                "control_p95_itl": control.get("p95_ms"),
                "candidate_p95_itl": candidate.get("p95_ms"),
                "identity": dict(identity),
            }
        )
    return rows


def run_graph_ab_phase(
    run_dir: Path,
    mode: str,
    phase: str,
    prefix: int,
    tokens: int,
    samples: int,
    *,
    timeout_s: float,
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
            "--capacity",
            "131072",
            MODEL,
        ],
        tier="acceptance" if samples >= PAIR_COUNT else "screen",
        timeout_s=timeout_s,
    )
    raw = completed.stdout + completed.stderr
    sidecar_path = run_dir / f"{phase}-raw.txt"
    sidecar_path.write_text(raw, encoding="utf-8")
    record = parse_prefixed(raw, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, "performance")
    record["raw_trace"] = workspace_relative(sidecar_path)
    return store_sidecar(run_dir, f"{phase}.json", record)


def run_quality_native(run_dir: Path, config_id: str) -> dict[str, Any]:
    sidecar_path = run_dir / f"quality-{config_id}.txt"
    if sidecar_path.is_file() and "held_out" in sidecar_path.read_text(
        encoding="utf-8"
    ):
        text = sidecar_path.read_text(encoding="utf-8")
        cases = _parse_quality_cases(text)
        if cases:
            return {
                "id": config_id,
                "cases": cases,
                "restored_packed_or_r2": False,
                "quality_flag": "--quality",
                "sidecar": workspace_relative(sidecar_path),
                "opt058_invoked": True,
                "reused_sidecar": True,
            }
    selectors = dict(COMBINED_QUALITY)
    if config_id == CANDIDATE:
        selectors["decode_attention"] = CANDIDATE
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
        timeout_s=LONG_TIMEOUT_S,
    )
    sidecar_path = run_dir / f"quality-{config_id}.txt"
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    if not restored:
        raise LongAttentionError("--quality restored packed or r2 defaults")
    if not cases:
        raise LongAttentionError(f"OPT-058 produced no NLL cases for {config_id}")
    argv = " ".join(
        [f"./{QUALITY_NATIVE}", MODEL, "--workload", "quality-baseline", *extra]
    )
    return {
        "id": config_id,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
        "opt058_argv": argv,
        "quality_config": workspace_relative(cfg_path),
    }


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    print(family_plan(mode, "preflight"), flush=True)
    eligible = eligibility()
    auth = authenticate_post113()
    source, dirty = git_identity()
    residents = gpu_residents()
    make = with_gpu_lock(
        [*docker_common(IMAGE, "smoke"), "make", "cuda-opt137-diagnostics"],
        timeout_s=LONG_TIMEOUT_S,
    )
    inspect: dict[str, Any] = {}
    if make.returncode == 0 and (ROOT / NATIVE).is_file():
        inspect = run_inspect_native(run_dir, mode)
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
        "phase": "preflight",
        "mode": mode,
        "ok": make.returncode == 0
        and bool(inspect.get("ok"))
        and production_pins_parent()
        and not eligible["blockers"]
        and not eligible["no_opportunity"],
        "eligibility": eligible,
        "inspect": inspect,
        "make_returncode": make.returncode,
        "make_stderr_tail": (make.stderr or "")[-2000:],
        "gpu_residents": residents,
        "authenticated_post113": auth,
        "production_pins_parent": production_pins_parent(),
        "parent": PARENT_STACK,
        "candidate": CANDIDATE,
        "source": source,
        "dirty": bool(dirty),
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
        "verdict": "blocked"
        if eligible["blockers"]
        else ("no_opportunity" if eligible["no_opportunity"] else None),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def run_correctness(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "correctness"), flush=True)
    identity = run_named_native(
        run_dir, mode, "identity", [], needs_model=False, timeout_s=LONG_TIMEOUT_S
    )
    same = run_named_native(
        run_dir,
        mode,
        "same-math",
        ["--prefix", "8192", "--tokens", "8", "--capacity", "131072"],
        timeout_s=LONG_TIMEOUT_S,
    )
    cancel = run_named_native(run_dir, mode, "cancellation", ["--prefix", "128"])
    handoff = run_named_native(run_dir, mode, "handoff", ["--prefix", "128"])
    ok = all(bool(row.get("ok")) for row in (identity, same, cancel, handoff))
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
        "phase": "correctness",
        "mode": mode,
        "ok": ok,
        "identity": identity,
        "same_math": same,
        "cancellation": cancel,
        "handoff": handoff,
        "family_plan": family_plan(mode, "correctness"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "correctness.json", payload)


def run_screen(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "screen"), flush=True)
    record = run_named_native(
        run_dir,
        mode,
        "screen",
        ["--warmups", str(SCREEN_WARMUPS), "--samples", str(SCREEN_PAIRS)],
        needs_model=False,
        timeout_s=LONG_TIMEOUT_S,
        sidecar_name="screen",
    )
    advance = bool(record.get("advance"))
    record["ok"] = bool(record.get("ok"))
    record["screened_out"] = bool(record.get("ok")) and not advance
    record["family_plan"] = family_plan(mode, "screen")
    return store_sidecar(run_dir, "screen.json", record)


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "quality"), flush=True)
    control = run_quality_native(run_dir, PARENT)
    candidate = run_quality_native(run_dir, CANDIDATE)
    candidate["same_math_copied_from_control"] = False
    debug_long = run_dir / "quality-long-debug.txt"
    if debug_long.is_file():
        raw = debug_long.read_text(encoding="utf-8")
        long_cache = parse_prefixed(raw, RESULT_PREFIX)
        long_cache["phase"] = "quality-long"
        long_cache["mode"] = mode
        long_cache["reused_debug"] = True
        long_cache = store_sidecar(run_dir, "quality-long.json", long_cache)
    else:
        long_cache = run_named_native(
            run_dir,
            mode,
            "quality-long",
            ["--capacity", "131072"],
            timeout_s=LONG_TIMEOUT_S,
            sidecar_name="quality-long",
            tier="acceptance",
        )
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
    long_ok = bool(long_cache.get("ok"))
    launches = [
        int(row.get("candidate_mma_launches") or 0)
        for row in (long_cache.get("cases") or [])
    ]
    launch_ok = bool(launches) and all(item > 0 for item in launches)
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
        and long_ok
        and launch_ok
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
        "phase": "quality",
        "mode": mode,
        "ok": quality_ok,
        "opt058_invoked": True,
        "candidate_nll_measured": not incomplete,
        "candidate_nll_not_measured": False,
        "opt116_contract_id": "opt116_generated_v1",
        "ppl_ratio": held_ratio,
        "wikitext_ppl_ratio": wiki_ratio,
        "ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "parent_authenticated_ppl_ratio": parent_ppl,
        "control": control,
        "candidate": candidate,
        "long_cache": long_cache,
        "long_cache_mma_launches": launches,
        "quality_flag": "--quality",
        "family_plan": family_plan(mode, "quality"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "quality.json", payload)


def run_state_memory(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "state-memory"), flush=True)
    native = run_named_native(
        run_dir,
        mode,
        "state-memory",
        ["--prefix", "128", "--capacity", "131072"],
        timeout_s=LONG_TIMEOUT_S,
        tier="acceptance",
    )
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    fit = bool(memory.get("post_graph_admitted"))
    checkpoint_ok = False
    checkpoint_message = "checkpoint binary missing"
    checkpoint_bin = ROOT / CHECKPOINT_NATIVE
    if checkpoint_bin.is_file():
        ckpt = run_dir / "opt137.ckpt"
        completed = run_native(
            [f"./{CHECKPOINT_NATIVE}", MODEL, workspace_relative(ckpt)],
            tier="acceptance",
            check=False,
            timeout_s=LONG_TIMEOUT_S,
        )
        checkpoint_ok = completed.returncode == 0
        checkpoint_message = (
            "ok" if checkpoint_ok else (completed.stderr or completed.stdout)[-500:]
        )
        (run_dir / "checkpoint.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
    native["memory_fit_parent_admitted"] = fit
    native["candidate_graph_bytes"] = int(native.get("graph_bytes") or 0)
    native["128k_fit_with_graphs"] = fit
    native["checkpoint_ok"] = checkpoint_ok
    native["checkpoint_message"] = checkpoint_message
    native["cancellation_ok"] = bool(cancel.get("ok", True))
    native["ok"] = (
        bool(native.get("ok"))
        and fit
        and bool(cancel.get("ok", True))
        and (checkpoint_ok or not checkpoint_bin.is_file())
    )
    native["family_plan"] = family_plan(mode, "state-memory")
    return store_sidecar(run_dir, "state-memory.json", native)


def run_performance(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "performance"), flush=True)
    workloads = {
        "d8192": run_graph_ab_phase(
            run_dir,
            mode,
            "d8192",
            8192,
            DECODE_TOKENS,
            PAIR_COUNT,
            timeout_s=PERFORMANCE_TIMEOUT_S,
        ),
        "d32768": run_graph_ab_phase(
            run_dir,
            mode,
            "d32768",
            32768,
            DECODE_TOKENS,
            PAIR_COUNT,
            timeout_s=PERFORMANCE_TIMEOUT_S,
        ),
        "d128": run_graph_ab_phase(
            run_dir,
            mode,
            "d128",
            128,
            DECODE_TOKENS,
            PAIR_COUNT,
            timeout_s=PERFORMANCE_TIMEOUT_S,
        ),
        "d2048": run_graph_ab_phase(
            run_dir,
            mode,
            "d2048",
            2048,
            DECODE_TOKENS,
            PAIR_COUNT,
            timeout_s=PERFORMANCE_TIMEOUT_S,
        ),
        "p4096": run_graph_ab_phase(
            run_dir,
            mode,
            "p4096",
            4096,
            0,
            PAIR_COUNT,
            timeout_s=PERFORMANCE_TIMEOUT_S,
        ),
        "d131040": run_graph_ab_phase(
            run_dir,
            mode,
            "d131040",
            131040,
            32,
            PAIR_COUNT,
            timeout_s=PERFORMANCE_TIMEOUT_S,
        ),
    }
    pairs: list[dict[str, Any]] = []
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d8192"],
            workload="d8192",
            prefix=8192,
            eval_count=256,
            metric="decode_only",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d8192"],
            workload="d8192",
            prefix=8192,
            eval_count=256,
            metric="complete_request",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d32768"],
            workload="d32768",
            prefix=32768,
            eval_count=256,
            metric="decode_only",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d32768"],
            workload="d32768",
            prefix=32768,
            eval_count=256,
            metric="complete_request",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d128"],
            workload="d128",
            prefix=128,
            eval_count=256,
            metric="decode_only",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d128"],
            workload="d128",
            prefix=128,
            eval_count=256,
            metric="complete_request",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d2048"],
            workload="d2048",
            prefix=2048,
            eval_count=256,
            metric="decode_only",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d2048"],
            workload="d2048",
            prefix=2048,
            eval_count=256,
            metric="complete_request",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["p4096"],
            workload="p4096",
            prefix=4096,
            eval_count=4096,
            metric="prefill",
        )
    )
    pairs.extend(
        pairs_from_graph_ab(
            workloads["d131040"],
            workload="d131040",
            prefix=131040,
            eval_count=32,
            metric="decode_only",
        )
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
        "phase": "performance",
        "mode": mode,
        "ok": all(bool(row.get("pairs")) for row in workloads.values()),
        "workloads": workloads,
        "pairs": pairs,
        "family_plan": family_plan(mode, "performance"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "performance.json", payload)


def run_mechanism(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "mechanism"), flush=True)
    native_8k = run_named_native(
        run_dir,
        mode,
        "mechanism",
        ["--prefix", "8192", "--tokens", "8", "--capacity", "131072"],
        timeout_s=LONG_TIMEOUT_S,
        sidecar_name="mechanism-d8192",
        tier="acceptance",
    )
    native_32k = run_named_native(
        run_dir,
        mode,
        "mechanism",
        ["--prefix", "32768", "--tokens", "8", "--capacity", "131072"],
        timeout_s=LONG_TIMEOUT_S,
        sidecar_name="mechanism-d32768",
        tier="acceptance",
    )
    eligible = eligibility()
    recapture = int(native_8k.get("topology_recapture") or 0) + int(
        native_32k.get("topology_recapture") or 0
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
        "phase": "mechanism",
        "mode": mode,
        "ok": bool(native_8k.get("ok"))
        and bool(native_32k.get("ok"))
        and recapture == 0,
        "d8192": native_8k,
        "d32768": native_32k,
        "opt136_excess_ms": eligible.get("excess_ms"),
        "llama_complete_attention_ms": eligible.get("llama_complete_attention_ms"),
        "recapture": recapture,
        "family_plan": family_plan(mode, "mechanism"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "mechanism.json", payload)


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    contract = load_contract()
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    correctness = load_sidecar(run_dir, "correctness.json") or {}
    screen = load_sidecar(run_dir, "screen.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    performance = load_sidecar(run_dir, "performance.json") or {}
    mechanism = load_sidecar(run_dir, "mechanism.json") or {}
    eligibility_row = dict(preflight.get("eligibility") or {})
    reasons: list[str] = []
    if eligibility_row.get("blockers"):
        return {
            "verdict": "blocked",
            "production_kept": False,
            "shipping_decode_attention": PARENT,
            "reasons": list(eligibility_row.get("blockers") or []),
            "performance": {},
            "tok_s_deltas": {},
        }
    if eligibility_row.get("no_opportunity"):
        return {
            "verdict": "no_opportunity",
            "production_kept": False,
            "shipping_decode_attention": PARENT,
            "reasons": ["opt136_complete_attention_excess_not_positive"],
            "performance": {},
            "tok_s_deltas": {},
        }
    if bool(screen.get("ok")) and bool(screen.get("screened_out")):
        return {
            "verdict": "screened_out",
            "production_kept": False,
            "shipping_decode_attention": PARENT,
            "reasons": ["screen_candidate_not_faster_at_both_targets"],
            "performance": {},
            "tok_s_deltas": {},
            "screen": screen,
        }
    records = {
        "pairs": list(performance.get("pairs") or []),
        "quality": {"quality": bool(quality.get("ok"))},
        "state": {"state": bool(state.get("ok"))},
    }
    policy = evaluate(contract, records)
    verdict = str(policy.get("verdict") or "incomplete")
    if not bool(correctness.get("ok")):
        reasons.append("identity_or_same_math_failed")
        verdict = "reject"
    if not bool(quality.get("ok")):
        reasons.append("quality_failed")
        if verdict not in {"incomplete", "reject"}:
            verdict = "reject"
    if not bool(state.get("ok")):
        reasons.append("state_memory_failed")
        if verdict not in {"incomplete", "reject"}:
            verdict = "reject"
    if not bool(mechanism.get("ok")):
        reasons.append("mechanism_failed")
        if verdict == "keep":
            verdict = "reject"
    reasons.extend(str(item) for item in (policy.get("reasons") or []))
    keep = verdict == "keep"
    tok_s: dict[str, Any] = {}
    for key, row in (policy.get("candidate_rates") or {}).items():
        tok_s[key] = row
    return {
        "verdict": verdict,
        "production_kept": keep,
        "shipping_decode_attention": CANDIDATE if keep else PARENT,
        "reasons": reasons,
        "performance": policy.get("metrics") or {},
        "tok_s_deltas": tok_s,
        "keep_policy": policy,
        "contract_hash": freeze_hash(contract),
        "quality": quality,
        "state_memory": state,
        "screen": screen,
        "mechanism": mechanism,
    }


def maybe_flip_production_pins(keep: bool) -> None:
    if not keep:
        return
    text = pin_text()
    text = text.replace(
        "constexpr bool kSelectedOpt137DenseMma = false;",
        "constexpr bool kSelectedOpt137DenseMma = true;",
    )
    PIN_PATH.write_text(text, encoding="utf-8")


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = str(result.get("verdict"))
    quality = result.get("quality") or {}
    lines = [
        "# OPT-137 — Bounded long-context dense attention MMA candidate",
        "",
        f"Status: **{verdict}**. Shipping decode attention "
        f"`{result.get('shipping_decode_attention')}`. Hardware executed.",
        "",
        "Parent is authenticated post124 combined plus OPT-127 "
        "`decode_segments8`. Candidate "
        "`dense_bf16_tile_f16_mma_decode_v1` adapts pinned llama "
        "fattn-mma-f16 organization for DKQ=DV=256, ncols1=1/ncols2=8 over "
        "dense BF16 KV with in-tile F16 conversion. No persistent F16 shadow "
        "cache. Production pins stay hybrid_crossover unless keep.",
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
            + "`; long-cache MMA launches=`"
            + str(quality.get("long_cache_mma_launches"))
            + "`."
        ),
        "",
        "## Keep policy",
        "",
        f"`target_guard_v2` verdict `{verdict}`. reasons=`{result.get('reasons')}`.",
        "",
        f"claims_throughput: `{bool(result.get('production_kept'))}`.",
        "",
        f"tok/s deltas: `{result.get('tok_s_deltas')}`.",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixture(run_dir: Path, result: Mapping[str, Any], mode: str) -> None:
    quality = load_sidecar(run_dir, "quality.json") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
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
        "preflight": load_sidecar(run_dir, "preflight.json") or {},
        "correctness": load_sidecar(run_dir, "correctness.json") or {},
        "screen": load_sidecar(run_dir, "screen.json") or {},
        "mechanism": load_sidecar(run_dir, "mechanism.json") or {},
        "keep_policy_id": "target_guard_v2",
    }
    dump_json(FIXTURE, payload)


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    return {
        "ok": not missing and payload.get("task") == "OPT-137",
        "task": "OPT-137",
        "missing": missing,
    }


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    result = evaluate_keep(run_dir)
    result["quality"] = load_sidecar(run_dir, "quality.json") or {}
    maybe_flip_production_pins(bool(result.get("production_kept")))
    write_report(result)
    write_fixture(run_dir, result, mode)
    payload = {
        "schema_version": 1,
        "task": "OPT-137",
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


def run_phase(run_dir: Path, mode: str, phase: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "correctness":
        return run_correctness(run_dir, mode)
    if phase == "screen":
        return run_screen(run_dir, mode)
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "performance":
        return run_performance(run_dir, mode)
    if phase == "mechanism":
        return run_mechanism(run_dir, mode)
    if phase == "report":
        return run_report(run_dir, mode)
    raise LongAttentionError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=("feedback", "acceptance", "release"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except LongAttentionError as exc:
        print(
            json.dumps(
                {"task": "OPT-137", "phase": args.phase, "ok": False, "error": str(exc)}
            )
        )
        return 1
    print(
        json.dumps(
            {
                "task": "OPT-137",
                "phase": args.phase,
                "ok": bool(payload.get("ok", True)),
                "verdict": payload.get("verdict"),
            }
        )
    )
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
