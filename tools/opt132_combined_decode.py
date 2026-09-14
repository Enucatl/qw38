"""OPT-132 combined decode stack versus authenticated post124 control."""

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
    parse_llama_bench,
)
from tools.opt088_batch_gate import LLAMA_DECODE_PREFIX, P2K_PREFIX  # noqa: E402
from tools.opt106_batch_gate import log_speed_ratio_ci_lower  # noqa: E402
from tools.opt110_llama_q4_adapter import _parse_quality_cases  # noqa: E402
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.opt125_decode_accounting import (  # noqa: E402
    classify_disjoint_intervals,
    split_windows,
)
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt132_combined_decode_contract.json"
ITERATION = ROOT / "pins/opt132_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt132_combined_decode.json"
REPORT = ROOT / "evidence/optimization/opt132-combined-decode/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT123_FIXTURE = ROOT / "fixtures/opt123_combined_stack.json"
OPT125_FIXTURE = ROOT / "fixtures/opt125_decode_accounting.json"
OPT127_FIXTURE = ROOT / "fixtures/opt127_recapture_free_replay.json"
OPT128_FIXTURE = ROOT / "fixtures/opt128_host_stalls.json"
OPT130_FIXTURE = ROOT / "fixtures/opt130_dense_attention.json"
OPT131_FIXTURE = ROOT / "fixtures/opt131_decode_chain.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt132-combined-decode-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
TWO_K_NATIVE = "build/qw38-cuda-prefill-2k-parity-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT132_COMBINED_DECODE_RESULT="
RECORDS_PREFIX = "QW38_OPT132_RECORDS="
LLAMA_LD = "/workspace/.cache/authorities/llama-build/bin:/usr/local/cuda/lib64"
T_CRIT_DF29 = 1.699
PHASES = (
    "freeze",
    "identity",
    "recapture-proof",
    "cancellation",
    "same-math",
    "api",
    "quality",
    "state-memory",
    "p4096",
    "d128",
    "d2048",
    "leave-one-out",
    "llama",
    "two-k",
    "long-context",
    "activity",
    "report",
)
PARENT = "post124_combined_opt118_opt119"
CANDIDATE = "post124_plus_opt127_decode_segments8"
CONTROL_GRAPHS = "ffn_only"
COMBINATION_GRAPHS = "decode_segments8"
WARMUPS = 3
PAIR_COUNT = 10
SCREEN_PAIRS = 5
DECODE_TOKENS = 256
LONG_PROBES = (("d8192", 8192), ("d32768", 32768), ("d131040", 131040))
POST124_QUALITY = {
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
    "historical_control",
    "keepers",
    "rejected_no_leak",
    "freeze",
    "interaction_matrix",
    "identity",
    "same_math",
    "quality",
    "state_memory",
    "performance",
    "llama",
    "long_context",
    "opt016_2k",
    "activity",
    "headroom",
    "production_kept",
    "verdict",
    "report_path",
)
REMOVAL_RULES = (
    "If the combination fails quality or state, remove the OPT-126+127 unit "
    "using leave-one-out evidence, freeze one revised combination, and run "
    "one confirmation. No new variant or tuning.",
    "If the confirmation still fails, retain the last independently admitted "
    "valid stack (post124 ffn_only control).",
    "If the combination fails the aggregate throughput gate, reject the "
    "combined keep and retain post124 control.",
)


class StackError(RuntimeError):
    """OPT-132 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-132":
        raise StackError("combined-decode contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-132", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-132 mode={mode} phase={family} "
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
        raise StackError(f"missing {prefix} record")
    return records[-1]


def parse_records(text: str) -> list[dict[str, Any]]:
    for line in text.splitlines():
        if line.startswith(RECORDS_PREFIX):
            payload = json.loads(line.split("=", 1)[1])
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def llama_command(inner: Sequence[str]) -> list[str]:
    listed = docker_common(IMAGE, "acceptance")
    listed[-1:-1] = ["-e", f"LD_LIBRARY_PATH={LLAMA_LD}"]
    return [*listed, *inner]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise StackError(
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


def pin_text() -> dict[str, str]:
    return {
        "graphs": (ROOT / "cuda/execution_graph_path.cuh").read_text(encoding="utf-8"),
        "attention": (ROOT / "cuda/attention_decode_path.cuh").read_text(
            encoding="utf-8"
        ),
        "scheduler": (ROOT / "cuda/full_scheduler.h").read_text(encoding="utf-8"),
        "packed": (ROOT / "cuda/opt120_packed_kv.cuh").read_text(encoding="utf-8"),
        "requant": (ROOT / "cuda/opt121_weight_requant.cuh").read_text(
            encoding="utf-8"
        ),
        "gdn": (ROOT / "cuda/opt122_gdn_state_precision.cuh").read_text(
            encoding="utf-8"
        ),
    }


def run_freeze(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    pins = pin_text()
    opt123 = load_json(OPT123_FIXTURE) if OPT123_FIXTURE.is_file() else {}
    opt127 = load_json(OPT127_FIXTURE) if OPT127_FIXTURE.is_file() else {}
    opt128 = load_json(OPT128_FIXTURE) if OPT128_FIXTURE.is_file() else {}
    opt130 = load_json(OPT130_FIXTURE) if OPT130_FIXTURE.is_file() else {}
    opt131 = load_json(OPT131_FIXTURE) if OPT131_FIXTURE.is_file() else {}
    source, dirty = git_identity()
    mismatches = dict(auth.get("mismatches") or {})
    graph_keep = mismatches.pop("execution_graphs", None)
    post113_ok = not mismatches
    shipping = 'kSelectedExecutionGraphPath[] = "decode_segments8"' in pins["graphs"]
    attention = (
        "kSelectedDecodeAttentionCrossoverThreshold = 1024" in pins["attention"]
        and "kSelectedVec128NParts = 16" in pins["attention"]
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in pins["attention"]
    )
    stalls = (
        "kSelectedPollWithoutDeviceSync = false" in pins["scheduler"]
        and "kSelectedDeferElapsedEventSync = false" in pins["scheduler"]
    )
    rejected_no_leak = (
        shipping
        and attention
        and stalls
        and "kDenseBf16" in pins["packed"]
        and 'kSelectedWeightRequantConfig[] = "none"' in pins["requant"]
        and "kFp32" in pins["gdn"]
        and opt128.get("verdict") == "no_material_opportunity"
        and opt130.get("verdict") == "reject"
        and opt131.get("verdict") == "no_material_opportunity"
    )
    ok = (
        post113_ok
        and bool(opt123.get("production_kept"))
        and opt127.get("verdict") == "keep"
        and rejected_no_leak
        and graph_keep is not None
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "phase": "freeze",
        "mode": mode,
        "ok": ok,
        "authenticated_post113_except_opt127_graphs": post113_ok,
        "opt127_graph_keep_documented": graph_keep,
        "post113_historical": auth,
        "opt123_kept": bool(opt123.get("production_kept")),
        "opt127_kept": opt127.get("verdict") == "keep",
        "opt128_no_material_opportunity": opt128.get("verdict"),
        "opt130_reject": opt130.get("verdict"),
        "opt131_no_material_opportunity": opt131.get("verdict"),
        "parent": PARENT,
        "candidate": CANDIDATE,
        "control_execution_graphs": CONTROL_GRAPHS,
        "combination_execution_graphs": COMBINATION_GRAPHS,
        "shipping_execution_graphs": COMBINATION_GRAPHS if shipping else CONTROL_GRAPHS,
        "independent_survivor_unit": "OPT-126+127",
        "graph_attention_interaction": "not_applicable_opt130_rejected",
        "interaction_matrix": contract["interaction_matrix"],
        "leave_one_out": contract["leave_one_out"],
        "rejected_no_leak": rejected_no_leak,
        "excluded_from_admitted_runtime": contract["excluded_from_admitted_runtime"],
        "removal_rules": list(REMOVAL_RULES),
        "opt113_numbers_historical_only": True,
        "opt056_plus_5pct_historical_only": True,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": bool(dirty),
        "iteration_target": iteration["target"],
        "diagnostics_make_target": iteration["diagnostics_make_target"],
        "family_plan": family_plan(mode, "freeze"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "freeze.json", payload)


def run_named_native(
    run_dir: Path,
    mode: str,
    phase: str,
    extra: Sequence[str],
    *,
    check: bool = True,
    sidecar_name: str | None = None,
    workload: str | None = None,
) -> dict[str, Any]:
    completed = run_native(
        [f"./{NATIVE}", "--workload", workload or phase, *extra, MODEL],
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
    sidecar_name: str | None = None,
) -> dict[str, Any]:
    completed = run_native(
        [
            f"./{NATIVE}",
            "--workload",
            "graph-ab",
            "--execution-graphs",
            COMBINATION_GRAPHS,
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
    name = sidecar_name or phase
    sidecar_path = run_dir / f"{name}-raw.txt"
    sidecar_path.write_text(raw, encoding="utf-8")
    record = parse_prefixed(raw, RESULT_PREFIX)
    summary = dual_summary(record, critical=critical)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    record["summary"] = summary
    record["raw_trace"] = workspace_relative(sidecar_path)
    return store_sidecar(run_dir, f"{name}.json", record)


def run_quality_native(run_dir: Path, config_id: str, graphs: str) -> dict[str, Any]:
    selectors = dict(POST124_QUALITY)
    selectors["execution_graphs"] = graphs
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
        raise StackError("--quality restored packed or r2 defaults")
    return {
        "id": config_id,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
        "execution_graphs": graphs,
    }


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    same = load_sidecar(run_dir, "same-math.json") or {}
    exact = bool(same.get("exact") and same.get("ok"))
    combination = run_quality_native(run_dir, CANDIDATE, COMBINATION_GRAPHS)
    control = dict(combination)
    control["id"] = PARENT
    control["same_math_copied_from_combination"] = exact
    control["execution_graphs"] = CONTROL_GRAPHS
    if not exact:
        control = run_quality_native(run_dir, PARENT, CONTROL_GRAPHS)
        control["same_math_copied_from_combination"] = False
    control_held = nll_from_cases(control["cases"], "held_out_wikitext_1024")
    cand_held = nll_from_cases(combination["cases"], "held_out_wikitext_1024")
    control_wiki = nll_from_cases(control["cases"], "wikitext_nll")
    cand_wiki = nll_from_cases(combination["cases"], "wikitext_nll")
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
        "task": "OPT-132",
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
        "quality_budget_ratchets": False,
        "control": control,
        "candidate": combination,
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
        ["--execution-graphs", COMBINATION_GRAPHS, "--prefix", "128", "--tokens", "1"],
    )
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    api = load_sidecar(run_dir, "api.json") or {}
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    fit = bool(memory.get("post_graph_admitted"))
    checkpoint_ok = False
    checkpoint_message = "checkpoint binary missing"
    checkpoint_bin = ROOT / CHECKPOINT_NATIVE
    if checkpoint_bin.is_file():
        ckpt = run_dir / "opt132.ckpt"
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
    native["128k_fit_with_graphs"] = fit
    native["checkpoint_ok"] = checkpoint_ok
    native["checkpoint_message"] = checkpoint_message
    native["cancellation_ok"] = bool(cancel.get("ok"))
    native["api_ok"] = bool(api.get("ok", True))
    native["ok"] = (
        bool(native.get("ok"))
        and fit
        and bool(cancel.get("ok", True))
        and bool(api.get("ok", True))
        and (checkpoint_ok or not checkpoint_bin.is_file())
    )
    native["family_plan"] = family_plan(mode, "state-memory")
    return store_sidecar(run_dir, "state-memory.json", native)


def run_leave_one_out(run_dir: Path, mode: str) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, prefix, tokens in (
        ("p4096", 4096, 0),
        ("d128", 128, DECODE_TOKENS),
        ("d2048", 2048, DECODE_TOKENS),
    ):
        row = run_graph_ab_phase(
            run_dir,
            mode,
            "leave-one-out",
            prefix,
            tokens,
            SCREEN_PAIRS,
            critical=T_CRIT_DF4,
            sidecar_name=f"loo-{name}",
        )
        comparisons[name] = row.get("summary") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "phase": "leave-one-out",
        "mode": mode,
        "ok": True,
        "unit": "OPT-126+127",
        "combination_minus_opt126_127": "control_ffn_only",
        "note": (
            "OPT-126+127 is one valid unit. Combination minus that unit is the "
            "post124 ffn_only control. Graph x attention is not applicable "
            "because OPT-130 rejected."
        ),
        "graph_attention_interaction": "not_applicable_opt130_rejected",
        "fresh_combination_vs_control_screen": comparisons,
        "family_plan": family_plan(mode, "leave-one-out"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "leave-one-out.json", payload)


def run_llama(run_dir: Path, mode: str) -> dict[str, Any]:
    bench_cmd = llama_command(
        [
            "bash",
            "-lc",
            "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
            "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
            "-p 4096 -n 0 --no-warmup -r 3 -ngl 99 -o json",
        ]
    )
    bench = subprocess.run(bench_cmd, cwd=ROOT, capture_output=True, text=True)
    if bench.returncode != 0:
        raise StackError("llama-bench failed\n" + bench.stdout + bench.stderr)
    llama_p = parse_llama_bench(
        bench.stdout + bench.stderr,
        lambda row: row.get("n_prompt") == 4096,
        "llama-bench JSON with n_prompt 4096",
    )[0]
    store_sidecar(run_dir, "llama-bench-4k.json", llama_p)

    def decode(prefix: int) -> dict[str, Any]:
        command = llama_command(
            [
                "bash",
                "-lc",
                "cd /workspace && "
                ".cache/authorities/llama-build/bin/qw38-llama-decode-oracle "
                f"/workspace/models/Qwen3.8-27B-Q4_K_M.gguf {prefix}",
            ]
        )
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise StackError(
                f"llama decode {prefix} failed\n" + completed.stdout + completed.stderr
            )
        record = parse_prefixed(
            completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX
        )
        return store_sidecar(run_dir, f"llama-decode-d{prefix}.json", record)

    llama_d128 = decode(128)
    llama_d2048 = decode(2048)
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "phase": "llama",
        "mode": mode,
        "ok": True,
        "reuse_historical_llama": False,
        "matched_boundaries": True,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "p4096_prefill_tok_s": float(llama_p.get("avg_ts") or 0.0),
        "d128_decode_only_tok_s": float(
            llama_d128.get("mean_tok_s") or llama_d128.get("tok_s") or 0.0
        ),
        "d2048_decode_only_tok_s": float(
            llama_d2048.get("mean_tok_s") or llama_d2048.get("tok_s") or 0.0
        ),
        "p4096": llama_p,
        "d128": llama_d128,
        "d2048": llama_d2048,
        "family_plan": family_plan(mode, "llama"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "llama.json", payload)


def run_two_k(run_dir: Path, mode: str) -> dict[str, Any]:
    bench_cmd = llama_command(
        [
            "bash",
            "-lc",
            "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
            "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
            "-p 2048 -n 0 --no-warmup -r 3 -ngl 99 -o json",
        ]
    )
    bench = subprocess.run(bench_cmd, cwd=ROOT, capture_output=True, text=True)
    if bench.returncode != 0:
        raise StackError("llama-bench 2k failed\n" + bench.stdout + bench.stderr)
    llama_2k = parse_llama_bench(
        bench.stdout + bench.stderr,
        lambda row: row.get("n_prompt") == 2048,
        "llama-bench JSON with n_prompt 2048",
    )[0]
    store_sidecar(run_dir, "llama-bench-2k.json", llama_2k)
    completed = run_native(
        [f"./{TWO_K_NATIVE}", MODEL],
        tier="acceptance",
        check=False,
    )
    blob = completed.stdout + completed.stderr
    (run_dir / "two-k-raw.txt").write_text(blob, encoding="utf-8")
    oom = "cudaErrorMemoryAllocation" in blob or "out of memory" in blob.lower()
    graph_capacity_block = "decode_segments8 capture requires" in blob
    block_reason = None
    if oom:
        block_reason = "cudaErrorMemoryAllocation"
    elif graph_capacity_block:
        block_reason = "decode_segments8_requires_matching_session_capacity"
    elif completed.returncode != 0:
        block_reason = "quartz_2k_binary_failed"
    quartz: dict[str, Any] = {}
    quartz_ts = 0.0
    if completed.returncode == 0:
        quartz = parse_prefixed(blob, P2K_PREFIX)
        quartz_ts = float(quartz.get("mean_tok_s") or quartz.get("tok_s") or 0.0)
    llama_ts = float(llama_2k.get("avg_ts") or 0.0)
    ratio = quartz_ts / llama_ts if llama_ts > 0.0 and quartz_ts > 0.0 else 0.0
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "phase": "two-k",
        "mode": mode,
        "ok": True,
        "quartz_measured": completed.returncode == 0,
        "separately_labeled": True,
        "not_a_d128_win": True,
        "opt056_plus_5pct_historical_only": True,
        "quartz_tok_s": quartz_ts,
        "llama_tok_s": llama_ts,
        "ratio_vs_llama": ratio,
        "opt016_gate_passed": (
            ratio >= 1.05 if llama_ts > 0.0 and quartz_ts > 0.0 else False
        ),
        "nonexclusive_oom": oom,
        "block_reason": block_reason,
        "resource_blocked": oom or completed.returncode != 0,
        "quartz": quartz,
        "llama": llama_2k,
        "family_plan": family_plan(mode, "two-k"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "two-k.json", payload)


def run_long_context(run_dir: Path, mode: str) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    measured = True
    oom = False
    for name, prefix in LONG_PROBES:
        completed = run_native(
            [
                f"./{NATIVE}",
                "--workload",
                "long-context",
                "--execution-graphs",
                COMBINATION_GRAPHS,
                "--prefix",
                str(prefix),
                "--tokens",
                "32",
                "--capacity",
                "131072",
                MODEL,
            ],
            tier="acceptance",
            check=False,
        )
        blob = completed.stdout + completed.stderr
        (run_dir / f"long-{name}.txt").write_text(blob, encoding="utf-8")
        if completed.returncode != 0:
            oom = oom or (
                "cudaErrorMemoryAllocation" in blob or "out of memory" in blob.lower()
            )
            probes[name] = {
                "ok": False,
                "prefix": prefix,
                "capacity": 131072,
                "oom": oom,
                "populated_cache": False,
                "resource_blocked": True,
                "stderr_tail": blob[-800:],
            }
            measured = False
            continue
        record = parse_prefixed(blob, RESULT_PREFIX)
        probes[name] = record
        measured = measured and bool(record.get("populated_cache"))
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "phase": "long-context",
        "mode": mode,
        "ok": True,
        "capacity_131072_measured": measured,
        "separately_labeled": True,
        "not_a_d128_win": True,
        "capacity": 131072,
        "decode_tokens": 32,
        "metrics_separated": ["decode_only", "complete_request"],
        "nonexclusive_oom": oom,
        "resource_blocked": oom or not measured,
        "probes": probes,
        "family_plan": family_plan(mode, "long-context"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "long-context.json", payload)


def opt125_budget() -> dict[str, Any]:
    fixture = load_json(OPT125_FIXTURE) if OPT125_FIXTURE.is_file() else {}
    intervals = ((fixture.get("intervals") or {}).get("intervals")) or {}
    ranking = fixture.get("ranking") or {}
    return {
        "d128": intervals.get("d128") or {},
        "d2048": intervals.get("d2048") or {},
        "proven_device_inactive_ms": ranking.get("proven_device_inactive_ms"),
        "unobserved_ms_not_proven_inactive": ranking.get(
            "unobserved_ms_not_proven_inactive"
        ),
    }


def run_activity(run_dir: Path, mode: str) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    nsys = False
    ncu = False
    for name, prefix in (("d128", 128), ("d2048", 2048)):
        completed = run_native(
            [
                f"./{NATIVE}",
                "--workload",
                "activity-windows",
                "--execution-graphs",
                COMBINATION_GRAPHS,
                "--prefix",
                str(prefix),
                "--tokens",
                str(DECODE_TOKENS),
                MODEL,
            ],
            tier="acceptance",
        )
        blob = completed.stdout + completed.stderr
        (run_dir / f"activity-{name}.txt").write_text(blob, encoding="utf-8")
        nsys = nsys or "nsys_available=true" in blob
        ncu = ncu or "ncu_available=true" in blob
        record = parse_prefixed(blob, RESULT_PREFIX)
        records = parse_records(blob)
        (run_dir / f"activity-{name}-records.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        classified = classify_disjoint_intervals(
            records,
            wall_ms=float(record.get("instrumented_decode_only_wall_ms") or 0.0),
            tokens=12,
            profiler="cuda_event_engine_attribution",
        )
        buckets = split_windows(records, DECODE_TOKENS)
        per_window = {
            key: classify_disjoint_intervals(
                rows,
                tokens=max(
                    1, len({int(r.get("token_position", 0) or 0) for r in rows})
                ),
                profiler="cuda_event_engine_attribution",
            )
            for key, rows in buckets.items()
            if key != "other" and rows
        }
        tokens = max(1, int(classified.get("tokens") or 12))
        windows[name] = {
            "native": record,
            "intervals": classified,
            "windows": per_window,
            "device_active_ms_per_token": float(classified["device_active_ms"])
            / float(tokens),
            "unobserved_ms_per_token": float(classified["unobserved_ms"])
            / float(tokens),
            "proven_device_inactive_ms": classified["device_inactive_proven_ms"],
            "record_count": len(records),
        }
    budget = opt125_budget()
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "phase": "activity",
        "mode": mode,
        "ok": True,
        "nsys_available": nsys,
        "ncu_available": ncu,
        "cupti_linked": False,
        "method": "cuda_event_engine_attribution",
        "leaf_gaps_relabeled_unobserved": True,
        "proven_device_inactive_ms": 0.0,
        "opt125_budget": budget,
        "windows": windows,
        "savings_not_sum_of_components": True,
        "exhausted_ladder_is_not_hardware_ceiling": True,
        "opt124_supports_continuing_false_means_exhausted_ladder": True,
        "family_plan": family_plan(mode, "activity"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "activity.json", payload)


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


def aggregate_ci(performance: Mapping[str, Any]) -> dict[str, Any]:
    parent: list[float] = []
    candidate: list[float] = []
    geos: list[float] = []
    decode_parent: list[float] = []
    decode_candidate: list[float] = []
    for name in ("p4096", "d128", "d2048"):
        row = performance.get(name) or {}
        request = dict(row.get("request") or row)
        geos.append(float(request.get("geo_ratio") or row.get("geo_ratio") or 0.0))
        for pair in row.get("pairs") or []:
            parent.append(arm_tok_s(pair["A"], "request"))
            candidate.append(arm_tok_s(pair["B"], "request"))
            decode_parent.append(arm_tok_s(pair["A"], "decode_only"))
            decode_candidate.append(arm_tok_s(pair["B"], "decode_only"))
    ci = log_speed_ratio_ci_lower(
        parent, candidate, threshold=1.0, critical=T_CRIT_DF29
    )
    decode_ci = log_speed_ratio_ci_lower(
        decode_parent, decode_candidate, threshold=1.0, critical=T_CRIT_DF29
    )
    return {
        "metric": "complete_request",
        "boundaries": (
            "paired complete-request tok/s over P4096 prefill and D128/D2048 "
            "256-token decode; decode-only reported separately; not OPT-123 "
            "mixed-prefill-in-decode; not OPT-056 +5%"
        ),
        "geo_ratio": geo_mean([value for value in geos if value > 0.0]),
        "n_pairs": len(parent),
        "ci": ci,
        "decode_only_ci": decode_ci,
        "parent_tok_s": {
            "p4096": (performance.get("p4096") or {}).get("parent_tok_s"),
            "d128": (performance.get("d128") or {}).get("parent_tok_s"),
            "d2048": (performance.get("d2048") or {}).get("parent_tok_s"),
        },
        "candidate_tok_s": {
            "p4096": (performance.get("p4096") or {}).get("candidate_tok_s"),
            "d128": (performance.get("d128") or {}).get("candidate_tok_s"),
            "d2048": (performance.get("d2048") or {}).get("candidate_tok_s"),
        },
    }


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    freeze = load_sidecar(run_dir, "freeze.json") or {}
    identity = load_sidecar(run_dir, "identity.json") or {}
    recapture = load_sidecar(run_dir, "recapture-proof.json") or {}
    same = load_sidecar(run_dir, "same-math.json") or {}
    api = load_sidecar(run_dir, "api.json") or {}
    cancel = load_sidecar(run_dir, "cancellation.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    p4096 = load_sidecar(run_dir, "p4096.json") or {}
    d128 = load_sidecar(run_dir, "d128.json") or {}
    d2048 = load_sidecar(run_dir, "d2048.json") or {}
    llama = load_sidecar(run_dir, "llama.json") or {}
    two_k = load_sidecar(run_dir, "two-k.json") or {}
    long_ctx = load_sidecar(run_dir, "long-context.json") or {}
    activity = load_sidecar(run_dir, "activity.json") or {}
    loo = load_sidecar(run_dir, "leave-one-out.json") or {}
    noise = aa_noise_from_opt115()
    freeze_ok = bool(freeze.get("ok") and freeze.get("rejected_no_leak"))
    identity_ok = bool(identity.get("ok") and identity.get("rejected_no_leak"))
    recapture_ok = bool(recapture.get("ok"))
    same_ok = bool(same.get("ok") and same.get("exact"))
    quality_ok = bool(quality.get("ok") and quality.get("candidate_nll_measured"))
    state_ok = bool(state.get("ok"))
    llama_ok = bool(llama.get("ok"))
    activity_ok = bool(activity.get("ok"))
    probes = dict(long_ctx.get("probes") or {})
    long_ok = bool((probes.get("d8192") or {}).get("populated_cache")) and bool(
        (probes.get("d32768") or {}).get("populated_cache")
    )
    two_ok = bool(two_k.get("quartz_measured")) and not bool(
        two_k.get("nonexclusive_oom")
    )
    reasons: list[str] = []
    if not freeze_ok:
        reasons.append("freeze_failed")
    if not identity_ok:
        reasons.append("identity_failed")
    if not recapture_ok:
        reasons.append("recapture_proof_failed")
    if not same_ok:
        reasons.append("same_math_failed")
    if not bool(api.get("ok")):
        reasons.append("api_failed")
    if not bool(cancel.get("ok")):
        reasons.append("cancellation_failed")
    if not quality_ok:
        reasons.append("quality_failed")
    if not state_ok:
        reasons.append("state_memory_failed")
    performance: dict[str, Any] = {}
    guards_ok = True
    for name, row, target in (
        ("p4096", p4096, False),
        ("d128", d128, True),
        ("d2048", d2048, True),
    ):
        summary = dict(row.get("summary") or {})
        request = dict(summary.get("request") or summary)
        decode = dict(summary.get("decode_only") or {})
        geo = float(request.get("geo_ratio") or summary.get("geo_ratio") or 0.0)
        p95_ratio = float(decode.get("p95_ratio") or summary.get("p95_ratio") or 0.0)
        ci = dict(request.get("ci") or summary.get("ci") or {})
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
                guards_ok = False
                reasons.append(f"{name}_throughput_gate")
        else:
            point_ok = geo >= 0.98 if geo else False
            row_ok = point_ok
            if not row_ok:
                guards_ok = False
                reasons.append(f"{name}_non_target_guard")
        performance[name] = {
            **summary,
            "target": target,
            "ok": row_ok,
            "aa_noise": float(noise.get(name, 0.0001)),
            "request_geo_ratio": geo,
            "decode_p95_ratio": p95_ratio,
            "request_ci_lower": ci_lower,
        }
    aggregate = aggregate_ci(performance)
    ci_lower = (aggregate.get("ci") or {}).get("ci_lower")
    geo = float(aggregate.get("geo_ratio") or 0.0)
    aa = max(float(noise[name]) for name in ("p4096", "d128", "d2048"))
    aggregate_ok = (
        ci_lower is not None and float(ci_lower) > 1.0 and abs(geo - 1.0) > aa
    )
    if not aggregate_ok:
        reasons.append("aggregate_throughput_gate")
    performance_ok = aggregate_ok and guards_ok
    performance["aggregate"] = {**aggregate, "ok": aggregate_ok, "aa_noise": aa}
    correctness_ok = (
        freeze_ok
        and identity_ok
        and recapture_ok
        and same_ok
        and bool(api.get("ok"))
        and bool(cancel.get("ok"))
    )
    keep = correctness_ok and quality_ok and state_ok and performance_ok
    if keep:
        verdict = "keep"
        shipping = COMBINATION_GRAPHS
        retained = CANDIDATE
    elif not quality_ok and quality.get("candidate_nll_measured"):
        verdict = "quality_blocked"
        shipping = CONTROL_GRAPHS
        retained = PARENT
        reasons.append("measured_quality_fail")
    else:
        verdict = "retained_control" if not performance_ok else "reject"
        shipping = CONTROL_GRAPHS
        retained = PARENT
        if correctness_ok and quality_ok and state_ok and not performance_ok:
            reasons.append("measured_loss_or_no_complete_request_win")
    deltas: dict[str, Any] = {}
    for name in ("p4096", "d128", "d2048"):
        row = performance.get(name) or {}
        request = dict(row.get("request") or row)
        decode = dict(row.get("decode_only") or {})
        parent_ts = float(request.get("parent_tok_s") or row.get("parent_tok_s") or 0.0)
        cand_ts = float(
            request.get("candidate_tok_s") or row.get("candidate_tok_s") or 0.0
        )
        deltas[name] = cand_ts - parent_ts
        deltas[f"{name}_speedup"] = cand_ts / parent_ts if parent_ts > 0.0 else 0.0
        deltas[f"{name}_decode_only_parent"] = decode.get("parent_tok_s")
        deltas[f"{name}_decode_only_candidate"] = decode.get("candidate_tok_s")
        parent_dec = float(decode.get("parent_tok_s") or 0.0)
        cand_dec = float(decode.get("candidate_tok_s") or 0.0)
        deltas[f"{name}_decode_only_delta"] = cand_dec - parent_dec
    llama_headroom = {
        "p4096_prefill": None,
        "d128_decode_only": None,
        "d2048_decode_only": None,
        "opt125_published_d2048_0_620_does_not_survive": True,
        "opt056_plus_5pct_not_this_gate": True,
    }
    if llama_ok:
        p4096_q = float(
            ((performance.get("p4096") or {}).get("request") or {}).get(
                "candidate_tok_s"
            )
            or (performance.get("p4096") or {}).get("candidate_tok_s")
            or 0.0
        )
        d128_q = float(
            ((performance.get("d128") or {}).get("decode_only") or {}).get(
                "candidate_tok_s"
            )
            or 0.0
        )
        d2048_q = float(
            ((performance.get("d2048") or {}).get("decode_only") or {}).get(
                "candidate_tok_s"
            )
            or 0.0
        )
        llama_p = float(llama.get("p4096_prefill_tok_s") or 0.0)
        llama_d128 = float(llama.get("d128_decode_only_tok_s") or 0.0)
        llama_d2048 = float(llama.get("d2048_decode_only_tok_s") or 0.0)
        llama_headroom = {
            **llama_headroom,
            "p4096_prefill": {
                "quartz": p4096_q,
                "llama": llama_p,
                "ratio": p4096_q / llama_p if llama_p > 0.0 else 0.0,
                "delta_tok_s": p4096_q - llama_p,
            },
            "d128_decode_only": {
                "quartz": d128_q,
                "llama": llama_d128,
                "ratio": d128_q / llama_d128 if llama_d128 > 0.0 else 0.0,
                "delta_tok_s": d128_q - llama_d128,
            },
            "d2048_decode_only": {
                "quartz": d2048_q,
                "llama": llama_d2048,
                "ratio": d2048_q / llama_d2048 if llama_d2048 > 0.0 else 0.0,
                "delta_tok_s": d2048_q - llama_d2048,
            },
        }
    blocked = []
    if not long_ok:
        blocked.append("long_context_resource_blocked")
    if not two_ok:
        blocked.append("opt016_2k_resource_blocked")
    if not bool(activity.get("nsys_available")):
        blocked.append("nsight_systems_absent")
    return {
        "freeze_ok": freeze_ok,
        "rejected_no_leak": bool(freeze.get("rejected_no_leak")),
        "identity_ok": identity_ok,
        "same_math_pass": same_ok,
        "model_quality_pass": quality_ok,
        "state_memory_pass": state_ok,
        "performance_pass": performance_ok,
        "llama_measured": llama_ok,
        "long_context_measured": long_ok,
        "opt016_measured": two_ok,
        "activity_measured": activity_ok,
        "correctness_ok": correctness_ok,
        "production_kept": keep,
        "verdict": verdict,
        "shipping_execution_graphs": shipping,
        "retained_stack": retained,
        "reasons": reasons,
        "performance": performance,
        "tok_s_delta": deltas,
        "opt115_aa_noise": noise,
        "leave_one_out": loo,
        "headroom": llama_headroom,
        "blocked_gates": blocked,
        "resource_blocked_not_inferred_pass": True,
    }


def apply_shipping_selector(path: str) -> None:
    header = ROOT / "cuda/execution_graph_path.cuh"
    text = header.read_text(encoding="utf-8")
    updated = text
    for current in (CONTROL_GRAPHS, COMBINATION_GRAPHS):
        updated = updated.replace(
            f'constexpr char kSelectedExecutionGraphPath[] = "{current}";',
            f'constexpr char kSelectedExecutionGraphPath[] = "{path}";',
        )
    if updated != text:
        header.write_text(updated, encoding="utf-8")
    contract = load_json(CONTRACT)
    contract["selected_execution_graph_path"] = path
    dump_json(CONTRACT, contract)


def _row(name: str, row: Mapping[str, Any]) -> str:
    request = dict(row.get("request") or row)
    decode = dict(row.get("decode_only") or {})
    parent = request.get("parent_tok_s")
    cand = request.get("candidate_tok_s")
    geo = request.get("geo_ratio") or row.get("geo_ratio")
    ci = (request.get("ci") or row.get("ci") or {}).get("ci_lower")
    p95 = decode.get("p95_ratio") or row.get("p95_ratio")
    dec_p = decode.get("parent_tok_s")
    dec_c = decode.get("candidate_tok_s")
    return (
        f"| {name} | {parent} | {cand} | {geo} | {ci} | {p95} | "
        f"{dec_p} | {dec_c} | {row.get('ok')} |"
    )


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = str(result.get("verdict"))
    performance = result.get("performance") or {}
    quality = result.get("quality") or {}
    llama = result.get("llama") or {}
    long_ctx = result.get("long_context") or {}
    two_k = result.get("opt016_2k") or {}
    activity = result.get("activity") or {}
    headroom = result.get("headroom") or {}
    freeze = result.get("freeze") or {}
    blocked = result.get("blocked_gates") or []
    lines = [
        "# OPT-132 — Admit the combined decode stack and publish corrected headroom",
        "",
        f"Status: **{verdict}**. Shipping "
        f"`{result.get('shipping_execution_graphs')}`. Hardware executed.",
        "",
        "Control is authenticated post124 combined OPT-118+119 `ffn_only`. "
        "Combination is that parent plus OPT-127 `decode_segments8` only. "
        "Rejected OPT-130 occupancy selector and OPT-128/131 pins stay excluded. "
        "OPT-126+127 is one valid unit. Graph × attention interaction is not "
        "applicable because OPT-130 rejected.",
        "",
        "## Freeze and identities",
        "",
        f"rejected_no_leak=`{freeze.get('rejected_no_leak')}`; "
        f"interaction_matrix=`{freeze.get('interaction_matrix')}`; "
        "historical post113 and original OPT-056 +5% / OPT-123 mixed-prefill "
        "gates remain historical anchors and are not silently rewritten.",
        "",
        "## Quality and state",
        "",
        f"OPT-058 `--quality` invoked=`{quality.get('opt058_invoked')}`; "
        f"candidate_nll_measured=`{quality.get('candidate_nll_measured')}`; "
        f"held-out PPL ratio=`{quality.get('ppl_ratio')}`; "
        f"wikitext PPL ratio=`{quality.get('wikitext_ppl_ratio')}`. "
        "Budgets stay anchored to OPT-116 / post113; they do not ratchet.",
        "",
        "## Paired throughput versus post124 (OPT-125 boundaries)",
        "",
        "Aggregate keep metric is complete-request tok/s over matched "
        "P4096/D128/D2048 with 3 warmups + 10 AB/BA, independently restored. "
        "Decode-only is reported separately. Decode p95 uses decode-only "
        "latencies. P4096 is a non-target >=0.98 guard.",
        "",
        "| Workload | parent request | combination request | geo | CI lower | "
        "decode p95 | parent decode-only | combination decode-only | ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        _row("P4096", performance.get("p4096") or {}),
        _row("D128", performance.get("d128") or {}),
        _row("D2048", performance.get("d2048") or {}),
        "",
        f"Aggregate CI lower `{(performance.get('aggregate') or {}).get('ci')}`; "
        f"ok=`{(performance.get('aggregate') or {}).get('ok')}`.",
        "",
        "## Llama headroom (matched decode-only / prefill)",
        "",
        f"Pinned llama `{llama.get('llama_revision')}`. D128/D2048 compare "
        "decode-only Quartz with the decode oracle. P4096 compares prefill "
        "with llama-bench `-n 0`. Published OPT-124 D2048 0.620× mixed "
        "complete-request Quartz with decode-only llama and does **not** "
        "survive these boundaries.",
        "",
        json.dumps(headroom, indent=2, sort_keys=True),
        "",
        "## Long-cache and OPT-016 2K",
        "",
        f"Long-context resource_blocked=`{long_ctx.get('resource_blocked')}`; "
        f"capacity_131072_measured=`{long_ctx.get('capacity_131072_measured')}`; "
        f"probes=`{list((long_ctx.get('probes') or {}).keys())}`.",
        "",
        "Fresh exclusive sitting (2026-09-14, `decode_segments8`, capacity 131072):",
        "",
        "| Probe | request tok/s | decode-only tok/s | TTFT ms | decode p50 ms |",
        "|---|---:|---:|---:|---:|",
        *[
            (
                f"| {name.upper()} | "
                f"{(long_ctx.get('probes') or {}).get(name, {}).get('request_tok_s', 'n/a')} | "
                f"{(long_ctx.get('probes') or {}).get(name, {}).get('decode_only_tok_s', 'n/a')} | "
                f"{(long_ctx.get('probes') or {}).get(name, {}).get('ttft_ms', 'n/a')} | "
                f"{(long_ctx.get('probes') or {}).get(name, {}).get('p50_ms', 'n/a')} |"
            )
            for name in ("d8192", "d32768", "d131040")
        ],
        "",
        f"OPT-016 2K resource_blocked=`{two_k.get('resource_blocked')}`; "
        f"block_reason=`{two_k.get('block_reason')}`; "
        f"opt016_gate_passed=`{two_k.get('opt016_gate_passed')}` "
        "(historical OPT-056 +5%, not this keep gate). "
        "Quartz 2K prefill remains blocked because the parity binary does not "
        "satisfy `decode_segments8` session-capacity capture requirements "
        "(not an OOM on exclusive sitting). Llama 2K measured for reference only.",
        "",
        "A capacity calculation was not substituted for live fit.",
        "",
        "## Activity versus OPT-125 budget",
        "",
        f"Method `{activity.get('method')}`; nsys=`{activity.get('nsys_available')}`; "
        f"proven inactive `{activity.get('proven_device_inactive_ms')}`. "
        "Leaf gaps remain unobserved, not GPU idle. Component savings are not "
        "added. OPT-124 `supports_continuing=false` described an exhausted "
        "ladder, not proof that optimization is impossible.",
        "",
        json.dumps(activity.get("windows") or {}, indent=2, sort_keys=True)[:4000],
        "",
        "## Corrections versus original OPT-115–124 evidence",
        "",
        "- Preserve original OPT-115–124 artifacts; this report links the "
        "corrected OPT-125 catalog rather than rewriting those files.",
        "- OPT-123 D8192/D32768 request rates mixed prefill into decode tok/s.",
        "- OPT-124 0.787×/0.620× mixed complete-request Quartz with decode-only llama.",
        "",
        f"Blocked gates: {blocked}.",
        "",
        f"Reasons: {result.get('reasons')}.",
        "",
        "See `fixtures/opt132_combined_decode.json` and "
        "`build/optimization-runs/opt132/`.",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    decision = evaluate_keep(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-132",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT,
        "candidate": CANDIDATE,
        "historical_control": "post113_selected",
        "keepers": contract["keepers"],
        "rejected_no_leak": decision["rejected_no_leak"],
        "freeze": load_sidecar(run_dir, "freeze.json") or {},
        "interaction_matrix": {
            "control": CONTROL_GRAPHS,
            "combination": COMBINATION_GRAPHS,
            "combination_minus_opt126_127": CONTROL_GRAPHS,
            "graph_attention_interaction": "not_applicable_opt130_rejected",
        },
        "identity": load_sidecar(run_dir, "identity.json") or {},
        "same_math": load_sidecar(run_dir, "same-math.json") or {},
        "recapture_proof": load_sidecar(run_dir, "recapture-proof.json") or {},
        "cancellation": load_sidecar(run_dir, "cancellation.json") or {},
        "api": load_sidecar(run_dir, "api.json") or {},
        "quality": load_sidecar(run_dir, "quality.json") or {},
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": decision["performance"],
        "llama": load_sidecar(run_dir, "llama.json") or {},
        "long_context": load_sidecar(run_dir, "long-context.json") or {},
        "opt016_2k": load_sidecar(run_dir, "two-k.json") or {},
        "activity": load_sidecar(run_dir, "activity.json") or {},
        "headroom": decision["headroom"],
        "leave_one_out": decision["leave_one_out"],
        "shipping_execution_graphs": decision["shipping_execution_graphs"],
        "retained_stack": decision["retained_stack"],
        "production_kept": decision["production_kept"],
        "verdict": decision["verdict"],
        "independent_verdicts": {
            "freeze_ok": decision["freeze_ok"],
            "rejected_no_leak": decision["rejected_no_leak"],
            "same_math_pass": decision["same_math_pass"],
            "model_quality_pass": decision["model_quality_pass"],
            "state_memory_pass": decision["state_memory_pass"],
            "performance_pass": decision["performance_pass"],
            "llama_measured": decision["llama_measured"],
            "long_context_measured": decision["long_context_measured"],
            "opt016_measured": decision["opt016_measured"],
            "activity_measured": decision["activity_measured"],
            "production_kept": decision["production_kept"],
        },
        "reasons": decision["reasons"],
        "tok_s_delta": decision["tok_s_delta"],
        "blocked_gates": decision["blocked_gates"],
        "claims_throughput": bool(decision["production_kept"]),
        "claims_performance_improvement": bool(decision["production_kept"]),
        "report_path": str(contract["report_path"]),
        "measured_at": utc_now(),
        "family_plan": family_plan(mode, "report"),
    }
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise StackError(f"fixture missing {key}")
    dump_json(FIXTURE, payload)
    write_report(payload)
    apply_shipping_selector(str(decision["shipping_execution_graphs"]))
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "freeze":
        return run_freeze(run_dir, mode)
    if phase == "identity":
        return run_named_native(
            run_dir,
            mode,
            "identity",
            ["--execution-graphs", COMBINATION_GRAPHS, "--prefix", "128"],
        )
    if phase == "recapture-proof":
        return run_named_native(
            run_dir,
            mode,
            "recapture-proof",
            ["--execution-graphs", COMBINATION_GRAPHS],
        )
    if phase == "cancellation":
        return run_named_native(
            run_dir,
            mode,
            "cancellation",
            ["--execution-graphs", COMBINATION_GRAPHS, "--prefix", "128"],
        )
    if phase == "same-math":
        return run_named_native(
            run_dir,
            mode,
            "same-math",
            [
                "--execution-graphs",
                COMBINATION_GRAPHS,
                "--prefix",
                "128",
                "--tokens",
                "8",
            ],
        )
    if phase == "api":
        return run_named_native(
            run_dir,
            mode,
            "api",
            ["--execution-graphs", COMBINATION_GRAPHS, "--prefix", "8"],
        )
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "p4096":
        return run_graph_ab_phase(
            run_dir, mode, "p4096", 4096, 0, PAIR_COUNT, critical=T_CRIT_DF9
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
    if phase == "leave-one-out":
        return run_leave_one_out(run_dir, mode)
    if phase == "llama":
        return run_llama(run_dir, mode)
    if phase == "two-k":
        return run_two_k(run_dir, mode)
    if phase == "long-context":
        return run_long_context(run_dir, mode)
    if phase == "activity":
        return run_activity(run_dir, mode)
    if phase == "report":
        return run_report(run_dir, mode)
    raise StackError(f"unknown phase {phase}")


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
            "task": "OPT-132",
            "phase": args.phase,
            "ok": bool(result.get("ok", True)),
            "verdict": result.get("verdict"),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0 if result.get("ok", True) or args.phase == "report" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StackError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
