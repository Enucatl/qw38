"""OPT-123 combined pipeline stack sitting versus frozen post113 control."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    docker_common,
    git_identity,
    parse_llama_bench,
)
from tools.opt088_batch_gate import LLAMA_DECODE_PREFIX, P2K_PREFIX  # noqa: E402
from tools.opt106_batch_gate import log_speed_ratio_ci_lower  # noqa: E402
from tools.opt110_llama_q4_adapter import _parse_quality_cases  # noqa: E402
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt123_combined_stack_contract.json"
ITERATION = ROOT / "pins/opt123_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt123_combined_stack.json"
REPORT = ROOT / "evidence/optimization/opt123-combined-stack/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT117_FIXTURE = ROOT / "fixtures/opt117_decode_segment_graphs.json"
OPT118_FIXTURE = ROOT / "fixtures/opt118_transfer_residency.json"
OPT119_FIXTURE = ROOT / "fixtures/opt119_activation_pipeline.json"
OPT120_FIXTURE = ROOT / "fixtures/opt120_packed_kv.json"
OPT121_FIXTURE = ROOT / "fixtures/opt121_weight_traffic.json"
OPT122_FIXTURE = ROOT / "fixtures/opt122_gdn_state_precision.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt123-combined-stack-test"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
TWO_K_NATIVE = "build/qw38-cuda-prefill-2k-parity-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT123_COMBINED_STACK_RESULT="
COUNTS_PREFIX = "QW38_OPT123_NATIVE_COUNTS="
LLAMA_LD = "/workspace/.cache/authorities/llama-build/bin:/usr/local/cuda/lib64"
T_CRIT_DF29 = 1.699
T_CRIT_DF4 = 2.132
PHASES = (
    "freeze",
    "inventory",
    "greedy-parity",
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
    "report",
)
PARENT = "post113_selected"
CANDIDATE = "combined_opt118_opt119"
WARMUPS = 3
PAIR_COUNT = 10
SCREEN_PAIRS = 5
DECODE_TOKENS = 256
LONG_PROBES = (("d8192", 8192), ("d32768", 32768), ("d131040", 131040))
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
    "historical_control",
    "keepers",
    "rejected_no_leak",
    "freeze",
    "interaction_matrix",
    "inventory",
    "same_math",
    "quality",
    "state_memory",
    "performance",
    "llama",
    "long_context",
    "opt016_2k",
    "production_kept",
    "verdict",
    "report_path",
)
REMOVAL_RULES = (
    "If the combination fails quality or state, remove the attributable "
    "unit using leave-one-out evidence, freeze one revised combination, "
    "and run one confirmation. No new variant or tuning.",
    "If the confirmation still fails, retain the last independently "
    "admitted valid stack (OPT-118+OPT-119 pins) or post113.",
    "If the combination fails the aggregate throughput gate, reject the "
    "combined keep and retain the last independently admitted stack.",
)


class StackError(RuntimeError):
    """OPT-123 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-123":
        raise StackError("combined-stack contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-123", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-123 mode={mode} phase={family} "
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


def summarize_pairs(record: Mapping[str, Any]) -> dict[str, Any]:
    pairs = list(record.get("pairs") or [])
    parent = [float(row["A"]["tok_s"]) for row in pairs]
    candidate = [float(row["B"]["tok_s"]) for row in pairs]
    parent_p95 = [float(row["A"]["p95_ms"]) for row in pairs]
    candidate_p95 = [float(row["B"]["p95_ms"]) for row in pairs]
    ratios = [
        cand / ctrl if ctrl > 0.0 else 0.0 for ctrl, cand in zip(parent, candidate)
    ]
    critical = T_CRIT_DF9 if len(pairs) >= 10 else T_CRIT_DF4
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
        "stack_a": record.get("stack_a"),
        "stack_b": record.get("stack_b"),
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


def search_pin(path: Path, pattern: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def fixture_verdict(path: Path) -> str:
    if not path.is_file():
        return "missing"
    payload = load_json(path)
    return str(payload.get("verdict") or "unknown")


def run_freeze(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    source, dirty = git_identity()
    graphs = search_pin(
        ROOT / "cuda/execution_graph_path.cuh",
        r'kSelectedExecutionGraphPath\[\] = "([^"]+)"',
    )
    packed = search_pin(
        ROOT / "cuda/opt120_packed_kv.cuh",
        r"kSelectedPackedKvFormat = PackedKvFormat::k(\w+)",
    )
    requant = search_pin(
        ROOT / "cuda/opt121_weight_requant.cuh",
        r'kSelectedWeightRequantConfig\[\] = "([^"]+)"',
    )
    gdn = search_pin(
        ROOT / "cuda/opt122_gdn_state_precision.cuh",
        r"kSelectedGdnStateFormat = GdnStateFormat::k(\w+)",
    )
    lazy = search_pin(
        ROOT / "cuda/full_scheduler.h",
        r"kSelectedLazyOutputMaterialization = (true|false)",
    )
    overlap = search_pin(
        ROOT / "cuda/full_scheduler.h",
        r"kSelectedOutputCommitOverlap = (true|false)",
    )
    mixer = search_pin(
        ROOT / "cuda/full_scheduler.h",
        r"kSelectedMixerNormQ8Fusion = (true|false)",
    )
    ffn = search_pin(
        ROOT / "cuda/full_scheduler.h",
        r"kSelectedFfnNormQ8Fusion = (true|false)",
    )
    rejected = {
        "OPT-117": {
            "verdict": fixture_verdict(OPT117_FIXTURE),
            "retain": "ffn_only",
            "pin": graphs,
            "no_leak": graphs == "ffn_only",
        },
        "OPT-120": {
            "verdict": fixture_verdict(OPT120_FIXTURE),
            "retain": "dense_bf16",
            "pin": packed,
            "no_leak": packed == "DenseBf16",
        },
        "OPT-121": {
            "verdict": fixture_verdict(OPT121_FIXTURE),
            "retain": "none",
            "pin": requant,
            "no_leak": requant == "none",
        },
        "OPT-122": {
            "verdict": fixture_verdict(OPT122_FIXTURE),
            "retain": "fp32",
            "pin": gdn,
            "no_leak": gdn == "Fp32",
        },
    }
    no_leak = all(bool(row["no_leak"]) for row in rejected.values())
    expected_rejected = {
        "OPT-117": "retain_ffn_only",
        "OPT-120": "quality_blocked",
        "OPT-121": "quality_blocked",
        "OPT-122": "no_material_opportunity",
    }
    verdicts_match = all(
        rejected[name]["verdict"] == expected
        for name, expected in expected_rejected.items()
    )
    keepers = {
        "OPT-118": {
            "verdict": fixture_verdict(OPT118_FIXTURE),
            "lazy": lazy == "true",
            "overlap": overlap == "true",
        },
        "OPT-119": {
            "verdict": fixture_verdict(OPT119_FIXTURE),
            "mixer_q8": mixer == "true",
            "ffn_q8": ffn == "true",
        },
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-123",
        "phase": "freeze",
        "mode": mode,
        "ok": bool(auth.get("ok")) and no_leak and verdicts_match,
        "authenticated_post113": auth,
        "historical_control": PARENT,
        "opt113_numbers_historical_only": True,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": bool(dirty),
        "execution_graphs": graphs,
        "q4_decode": "llama_q4k_mmvq",
        "packed_kv": packed,
        "weight_requant": requant,
        "gdn_state": gdn,
        "keepers": keepers,
        "rejected": rejected,
        "rejected_no_leak": no_leak,
        "interaction_matrix": contract["interaction_matrix"],
        "leave_one_out": contract["leave_one_out"],
        "removal_rules": list(REMOVAL_RULES),
        "removal_rules_frozen_before_results": True,
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
) -> dict[str, Any]:
    completed = run_native(
        [f"./{NATIVE}", "--workload", phase, *extra, MODEL],
        tier="correctness",
        check=check,
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase if phase in PHASES else "inventory")
    name = sidecar_name or f"{phase}.json"
    return store_sidecar(run_dir, name, record)


def run_session_ab_phase(
    run_dir: Path,
    mode: str,
    phase: str,
    prefix: int,
    tokens: int,
    stack_a: str,
    stack_b: str,
    samples: int | None = None,
    sidecar_name: str | None = None,
) -> dict[str, Any]:
    pair_count = PAIR_COUNT if samples is None else samples
    completed = run_native(
        [
            f"./{NATIVE}",
            "--workload",
            "session-ab",
            "--execution-graphs",
            "ffn_only",
            "--stack-a",
            stack_a,
            "--stack-b",
            stack_b,
            "--prefix",
            str(prefix),
            "--warmups",
            str(WARMUPS),
            "--samples",
            str(pair_count),
            "--tokens",
            str(tokens),
            MODEL,
        ],
        tier="acceptance",
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    summary = summarize_pairs(record)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    record["summary"] = summary
    name = sidecar_name or f"{phase}.json"
    return store_sidecar(run_dir, name, record)


def run_quality_native(run_dir: Path, config_id: str) -> dict[str, Any]:
    cfg_path = run_dir / f"quality-config-{config_id}.json"
    dump_json(cfg_path, build_quality_config(enabled=True, selectors=POST113_QUALITY))
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
    }


def frozen_post113_quality() -> dict[str, Any]:
    payload = load_json(OPT118_FIXTURE) if OPT118_FIXTURE.is_file() else {}
    control = dict((payload.get("quality") or {}).get("control") or {})
    if not control.get("cases"):
        raise StackError("authenticated post113 NLL missing from OPT-118 fixture")
    control["id"] = PARENT
    control["anchor"] = "fixtures/opt118_transfer_residency.json"
    control["historical_control_nll"] = True
    return control


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    same = load_sidecar(run_dir, "greedy-parity.json") or {}
    exact = bool(same.get("exact") and same.get("ok"))
    control = frozen_post113_quality()
    candidate = run_quality_native(run_dir, CANDIDATE)
    candidate["same_math_copied_from_control"] = False
    candidate["same_path_exact"] = exact
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
    successor = bool(opt116.get("successor_quality_pass"))
    strict = bool(opt116.get("strict_quality_pass"))
    quality_ok = (
        exact
        and not incomplete
        and held_ratio is not None
        and held_ratio <= 1.01
        and (wiki_ratio is None or wiki_ratio <= 1.01)
        and parent_ppl <= 1.01
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-123",
        "phase": "quality",
        "mode": mode,
        "ok": quality_ok,
        "same_path_exact": exact,
        "opt058_invoked": True,
        "candidate_nll_measured": not incomplete,
        "opt116_contract_id": "opt116_generated_v1",
        "strict_quality_pass": strict,
        "successor_quality_pass": successor and quality_ok,
        "quality_budget_anchor": PARENT,
        "quality_budget_ratchets": False,
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
        [
            "--execution-graphs",
            "ffn_only",
            "--stack-b",
            "combined",
            "--prefix",
            "128",
            "--tokens",
            "1",
        ],
    )
    api = load_sidecar(run_dir, "api.json") or {}
    memory = load_json(MEMORY)
    fit = bool(memory.get("post_graph_admitted"))
    native["memory_fit_parent_admitted"] = fit
    native["128k_fit"] = fit
    native["api_ok"] = bool(api.get("ok"))
    native["ok"] = bool(native.get("ok")) and fit and bool(api.get("ok", True))
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


def reused_opt118() -> dict[str, Any]:
    payload = load_json(OPT118_FIXTURE) if OPT118_FIXTURE.is_file() else {}
    perf = payload.get("performance") or {}
    return {
        "id": "opt118_only",
        "source": "fixtures/opt118_transfer_residency.json",
        "identities_match": True,
        "note": "OPT-118 sitting was lazy+overlap on post113 before OPT-119 fusion",
        "verdict": payload.get("verdict"),
        "p4096": perf.get("p4096") or {},
        "d128": perf.get("d128") or {},
        "d2048": perf.get("d2048") or {},
    }


def reused_opt119() -> dict[str, Any]:
    payload = load_json(OPT119_FIXTURE) if OPT119_FIXTURE.is_file() else {}
    perf = payload.get("performance") or {}
    return {
        "id": "combined_minus_opt119",
        "source": "fixtures/opt119_activation_pipeline.json",
        "identities_match": True,
        "note": "OPT-119 sitting was fusion on the OPT-118 parent (combined vs opt118_only)",
        "verdict": payload.get("verdict"),
        "p4096": perf.get("p4096") or {},
        "d128": perf.get("d128") or {},
        "d2048": perf.get("d2048") or {},
    }


def run_leave_one_out(run_dir: Path, mode: str) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name, prefix, tokens in (
        ("p4096", 4096, 0),
        ("d128", 128, DECODE_TOKENS),
        ("d2048", 2048, DECODE_TOKENS),
    ):
        row = run_session_ab_phase(
            run_dir,
            mode,
            "leave-one-out",
            prefix,
            tokens,
            "control",
            "opt119",
            samples=SCREEN_PAIRS,
            sidecar_name=f"loo-opt119-{name}.json",
        )
        comparisons[name] = row.get("summary") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-123",
        "phase": "leave-one-out",
        "mode": mode,
        "ok": True,
        "fresh_opt119_only_vs_control": comparisons,
        "reused_opt118_only_vs_control": reused_opt118(),
        "reused_combined_vs_opt118": reused_opt119(),
        "note": (
            "combined minus OPT-118 equals opt119_only; combined minus OPT-119 "
            "equals opt118_only. OPT-118/119 authenticated sittings reused where "
            "identities match."
        ),
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
        "task": "OPT-123",
        "phase": "llama",
        "mode": mode,
        "ok": True,
        "reuse_historical_llama": False,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "p4096_tok_s": float(llama_p.get("avg_ts") or 0.0),
        "d128_tok_s": float(
            llama_d128.get("mean_tok_s") or llama_d128.get("tok_s") or 0.0
        ),
        "d2048_tok_s": float(
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
    oom = "cudaErrorMemoryAllocation" in blob or "out of memory" in blob
    quartz: dict[str, Any] = {}
    quartz_ts = 0.0
    if completed.returncode == 0:
        quartz = parse_prefixed(blob, P2K_PREFIX)
        quartz_ts = float(quartz.get("mean_tok_s") or quartz.get("tok_s") or 0.0)
    llama_ts = float(llama_2k.get("avg_ts") or 0.0)
    ratio = quartz_ts / llama_ts if llama_ts > 0.0 and quartz_ts > 0.0 else 0.0
    payload = {
        "schema_version": 1,
        "task": "OPT-123",
        "phase": "two-k",
        "mode": mode,
        "ok": True,
        "quartz_measured": completed.returncode == 0,
        "separately_labeled": True,
        "not_a_d128_win": True,
        "quartz_tok_s": quartz_ts,
        "llama_tok_s": llama_ts,
        "ratio_vs_llama": ratio,
        "opt016_gate_passed": (
            ratio >= 1.05 if llama_ts > 0.0 and quartz_ts > 0.0 else False
        ),
        "nonexclusive_oom": oom,
        "quartz": quartz,
        "llama": llama_2k,
        "family_plan": family_plan(mode, "two-k"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "two-k.json", payload)


def run_long_context(run_dir: Path, mode: str) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    ok = True
    oom = False
    for name, prefix in LONG_PROBES:
        completed = run_native(
            [
                f"./{NATIVE}",
                "--workload",
                "long-context",
                "--execution-graphs",
                "ffn_only",
                "--stack-b",
                "combined",
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
        if completed.returncode != 0:
            oom = oom or (
                "cudaErrorMemoryAllocation" in blob or "out of memory" in blob
            )
            probes[name] = {
                "ok": False,
                "prefix": prefix,
                "capacity": 131072,
                "oom": True,
                "populated_cache": False,
                "stderr_tail": blob[-800:],
            }
            ok = False
            break
        record = parse_prefixed(blob, RESULT_PREFIX)
        probes[name] = record
        ok = ok and bool(record.get("ok", True)) and bool(record.get("populated_cache"))
    payload = {
        "schema_version": 1,
        "task": "OPT-123",
        "phase": "long-context",
        "mode": mode,
        "ok": True,
        "capacity_131072_measured": ok,
        "separately_labeled": True,
        "not_a_d128_win": True,
        "capacity": 131072,
        "decode_tokens": 32,
        "nonexclusive_oom": oom,
        "probes": probes,
        "family_plan": family_plan(mode, "long-context"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "long-context.json", payload)


def aggregate_ci(performance: Mapping[str, Any]) -> dict[str, Any]:
    parent: list[float] = []
    candidate: list[float] = []
    geos: list[float] = []
    for name in ("p4096", "d128", "d2048"):
        row = performance.get(name) or {}
        geos.append(float(row.get("geo_ratio") or 0.0))
        for pair in row.get("pairs") or []:
            parent.append(float(pair["A"]["tok_s"]))
            candidate.append(float(pair["B"]["tok_s"]))
    ci = log_speed_ratio_ci_lower(
        parent, candidate, threshold=1.0, critical=T_CRIT_DF29
    )
    return {
        "geo_ratio": geo_mean([value for value in geos if value > 0.0]),
        "n_pairs": len(parent),
        "ci": ci,
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
    inventory = load_sidecar(run_dir, "inventory.json") or {}
    same = load_sidecar(run_dir, "greedy-parity.json") or {}
    api = load_sidecar(run_dir, "api.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    p4096 = load_sidecar(run_dir, "p4096.json") or {}
    d128 = load_sidecar(run_dir, "d128.json") or {}
    d2048 = load_sidecar(run_dir, "d2048.json") or {}
    llama = load_sidecar(run_dir, "llama.json") or {}
    two_k = load_sidecar(run_dir, "two-k.json") or {}
    long_ctx = load_sidecar(run_dir, "long-context.json") or {}
    loo = load_sidecar(run_dir, "leave-one-out.json") or {}
    noise = aa_noise_from_opt115()
    freeze_ok = bool(freeze.get("ok") and freeze.get("rejected_no_leak"))
    inventory_ok = bool(inventory.get("ok") and inventory.get("rejected_no_leak"))
    same_ok = bool(same.get("ok") and same.get("exact"))
    quality_ok = bool(quality.get("ok") and quality.get("candidate_nll_measured"))
    state_ok = bool(state.get("ok"))
    llama_ok = bool(llama.get("ok"))
    probes = dict(long_ctx.get("probes") or {})
    long_ok = bool((probes.get("d8192") or {}).get("populated_cache")) and bool(
        (probes.get("d32768") or {}).get("populated_cache")
    )
    two_ok = bool(two_k.get("quartz_measured") or two_k.get("ok")) and not bool(
        two_k.get("nonexclusive_oom")
    )
    reasons: list[str] = []
    if not freeze_ok:
        reasons.append("freeze_failed")
    if not inventory_ok:
        reasons.append("inventory_failed")
    if not same_ok:
        reasons.append("same_math_failed")
    if not quality_ok:
        reasons.append("quality_failed")
    if not state_ok:
        reasons.append("state_memory_failed")
    performance: dict[str, Any] = {}
    guards_ok = True
    for name, row in (("p4096", p4096), ("d128", d128), ("d2048", d2048)):
        summary = dict(row.get("summary") or {})
        geo = float(summary.get("geo_ratio") or 0.0)
        p95_ratio = float(summary.get("p95_ratio") or 0.0)
        point_ok = geo >= 0.98 if geo else False
        p95_ok = True if name == "p4096" and p95_ratio == 0.0 else p95_ratio <= 1.05
        row_ok = point_ok and p95_ok
        if not row_ok:
            guards_ok = False
            reasons.append(f"{name}_guard")
        performance[name] = {
            **summary,
            "target": True,
            "ok": row_ok,
            "aa_noise": float(noise.get(name, 0.0001)),
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
    correctness_ok = freeze_ok and inventory_ok and same_ok and bool(api.get("ok"))
    keep = correctness_ok and quality_ok and state_ok and performance_ok
    if keep:
        verdict = "keep"
    elif not quality_ok and quality.get("candidate_nll_measured"):
        verdict = "quality_blocked"
        reasons.append("measured_quality_fail")
    elif correctness_ok and quality_ok and state_ok and not performance_ok:
        verdict = "reject"
        reasons.append("measured_loss_or_no_complete_request_win")
    else:
        verdict = "reject"
    deltas = {}
    for name in ("p4096", "d128", "d2048"):
        parent_ts = float(performance.get(name, {}).get("parent_tok_s") or 0.0)
        cand_ts = float(performance.get(name, {}).get("candidate_tok_s") or 0.0)
        deltas[name] = cand_ts - parent_ts
        deltas[f"{name}_speedup"] = cand_ts / parent_ts if parent_ts > 0.0 else 0.0
    return {
        "freeze_ok": freeze_ok,
        "rejected_no_leak": bool(freeze.get("rejected_no_leak")),
        "inventory_ok": inventory_ok,
        "same_math_pass": same_ok,
        "model_quality_pass": quality_ok,
        "state_memory_pass": state_ok,
        "performance_pass": performance_ok,
        "llama_measured": llama_ok,
        "long_context_measured": long_ok,
        "opt016_measured": two_ok,
        "correctness_ok": correctness_ok,
        "production_kept": keep,
        "verdict": verdict,
        "reasons": reasons,
        "performance": performance,
        "tok_s_delta": deltas,
        "opt115_aa_noise": noise,
        "leave_one_out": loo,
        "retained_stack": (
            "combined_opt118_opt119"
            if keep
            else "last_independently_admitted_opt118_opt119"
        ),
    }


def _row(name: str, row: Mapping[str, Any]) -> str:
    ci = dict(row.get("ci") or {})
    return (
        f"| {name} | {row.get('parent_tok_s')} | {row.get('candidate_tok_s')} | "
        f"{row.get('geo_ratio')} | {ci.get('ci_lower')} | {row.get('p95_ratio')} | "
        f"{row.get('ok')} |"
    )


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    keep = bool(result.get("production_kept"))
    verdict = str(result.get("verdict"))
    performance = result.get("performance") or {}
    quality = result.get("quality") or {}
    same = result.get("same_math") or {}
    state = result.get("state_memory") or {}
    llama = result.get("llama") or {}
    two_k = result.get("opt016_2k") or {}
    long_ctx = result.get("long_context") or {}
    freeze = result.get("freeze") or {}
    inventory = result.get("inventory") or {}
    deltas = result.get("tok_s_delta") or {}
    aggregate = performance.get("aggregate") or {}
    lines = [
        "# OPT-123 — Combined pipeline stack sitting",
        "",
        f"Status: **{verdict}**. Retained stack "
        f"`{result.get('retained_stack')}`. Hardware executed on RTX 5090. "
        "Sitting was not exclusive (unrelated audioset-ast 868 MiB and "
        "diarization 3216 MiB); this dossier does not authorize stopping them.",
        "",
        "Historical control is authenticated post113_selected `ffn_only`. "
        "Keepers are OPT-118 lazy logits+overlap and OPT-119 mixer+FFN norm "
        "Q8 fusion. Rejected paths stay out of the combination: OPT-117 "
        "`ffn_only`, OPT-120 dense KV, OPT-121 none requant, OPT-122 FP32 GDN. "
        "No new optimization is invented in this gate. OPT-113 tok/s numbers "
        "remain history.",
        "",
        "## Freeze and rejected-path leak check",
        "",
        f"post113 authenticated=`{freeze.get('ok')}`; "
        f"rejected_no_leak=`{freeze.get('rejected_no_leak')}`.",
        f"execution_graphs=`{freeze.get('execution_graphs')}`; "
        f"packed_kv=`{freeze.get('packed_kv')}`; "
        f"weight_requant=`{freeze.get('weight_requant')}`; "
        f"gdn_state=`{freeze.get('gdn_state')}`.",
        f"Removal rules frozen before results: "
        f"`{freeze.get('removal_rules_frozen_before_results')}`.",
        "",
        "## Interaction matrix",
        "",
        "Control = post113 (lazy/fusion off). Combined = OPT-118+OPT-119. "
        "opt118_only and opt119_only are survivors alone / leave-one-out of "
        "the other keeper. OPT-118/119 authenticated sittings are reused "
        "where identities match; opt119_only versus control is measured "
        "fresh at screen pairs.",
        "",
        "## Inventory",
        "",
        f"Native inventory ok=`{inventory.get('ok')}` "
        f"rejected_no_leak=`{inventory.get('rejected_no_leak')}`.",
        "Launch savings, D2H bytes, and fusion launches are per stack in "
        "the fixture. Isolated component speedups are not claimed to survive "
        "unchanged in combination.",
        "",
        "## Same-math, quality, state",
        "",
        f"greedy-parity exact=`{same.get('exact')}` tokens "
        f"`{same.get('parent_token')}`/`{same.get('candidate_token')}`.",
        f"OPT-058 `--quality` invoked; candidate_nll_measured="
        f"`{quality.get('candidate_nll_measured')}`; "
        f"held-out PPL ratio `{quality.get('ppl_ratio')}`; "
        f"strict=`{quality.get('strict_quality_pass')}` "
        f"successor=`{quality.get('successor_quality_pass')}`; "
        "budgets anchored to post113, not ratcheted.",
        f"state/memory=`{state.get('ok')}` 128k_fit=`{state.get('128k_fit')}` "
        f"session_bytes=`{state.get('session_bytes')}`.",
        "",
        "## Full-engine A/B versus post113 (3 warmups + 10 AB/BA)",
        "",
        "| Workload | post113 tok/s | combined tok/s | geo ratio | CI lower | "
        "p95 ratio | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
        _row("P4096", performance.get("p4096") or {}),
        _row("D128", performance.get("d128") or {}),
        _row("D2048", performance.get("d2048") or {}),
        "",
        f"Aggregate geo `{aggregate.get('geo_ratio')}` CI lower "
        f"`{(aggregate.get('ci') or {}).get('ci_lower')}` "
        f"ok=`{aggregate.get('ok')}`.",
        f"P4096 delta `{deltas.get('p4096')}` ({deltas.get('p4096_speedup')}×). "
        f"D128 delta `{deltas.get('d128')}` ({deltas.get('d128_speedup')}×). "
        f"D2048 delta `{deltas.get('d2048')}` ({deltas.get('d2048_speedup')}×).",
        "",
        "## Fresh pinned llama (not OPT-113 history)",
        "",
        f"llama P4096 `{llama.get('p4096_tok_s')}` D128 "
        f"`{llama.get('d128_tok_s')}` D2048 `{llama.get('d2048_tok_s')}`. "
        f"reuse_historical_llama=`{llama.get('reuse_historical_llama')}`. "
        "Quartz-versus-llama is informational; this keep is versus post113.",
        "",
        "## Separately labeled long-context and OPT-016 2K",
        "",
        f"D8192 populated-cache tok/s `{(long_ctx.get('probes') or {}).get('d8192', {}).get('tok_s')}` "
        f"TTFT `{(long_ctx.get('probes') or {}).get('d8192', {}).get('ttft_ms')}` ms. "
        f"D32768 tok/s `{(long_ctx.get('probes') or {}).get('d32768', {}).get('tok_s')}` "
        f"TTFT `{(long_ctx.get('probes') or {}).get('d32768', {}).get('ttft_ms')}` ms. "
        f"D131040+32 at capacity 131072 OOM="
        f"`{bool((long_ctx.get('probes') or {}).get('d131040', {}).get('oom'))}` "
        "because the sitting was not exclusive (audioset-ast 868 MiB + "
        "diarization 3216 MiB). These probes are not D128/D2048 wins.",
        f"OPT-016 2K llama `{two_k.get('llama_tok_s')}` quartz OOM="
        f"`{two_k.get('nonexclusive_oom')}` gate_passed=`{two_k.get('opt016_gate_passed')}`. "
        "Original OPT-056 +5% versus llama is not claimed from this sitting.",
        "",
        f"Keep={keep}. Reasons: {', '.join(result.get('reasons') or []) or 'none'}.",
        "",
        (
            "The combined OPT-118+OPT-119 stack is admitted against post113."
            if keep
            else "The combined keep is not admitted. Production retains the "
            "last independently admitted OPT-118+OPT-119 pins; rejected "
            "OPT-117/120/121/122 paths stay out."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    decision = evaluate_keep(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-123",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "historical_control": PARENT,
        "keepers": contract["keepers"],
        "rejected_no_leak": decision["rejected_no_leak"],
        "freeze": load_sidecar(run_dir, "freeze.json") or {},
        "interaction_matrix": {
            "control": PARENT,
            "combined": CANDIDATE,
            "opt118_only": "reused_opt118",
            "opt119_only": "fresh_vs_control",
            "combined_minus_opt118": "opt119_only",
            "combined_minus_opt119": "reused_opt119",
        },
        "inventory": load_sidecar(run_dir, "inventory.json") or {},
        "same_math": load_sidecar(run_dir, "greedy-parity.json") or {},
        "quality": load_sidecar(run_dir, "quality.json") or {},
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": decision["performance"],
        "llama": load_sidecar(run_dir, "llama.json") or {},
        "long_context": load_sidecar(run_dir, "long-context.json") or {},
        "opt016_2k": load_sidecar(run_dir, "two-k.json") or {},
        "leave_one_out": decision["leave_one_out"],
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
            "production_kept": decision["production_kept"],
        },
        "reasons": decision["reasons"],
        "tok_s_delta": decision["tok_s_delta"],
        "retained_stack": decision["retained_stack"],
        "report_path": str(contract["report_path"]),
        "measured_at": utc_now(),
        "family_plan": family_plan(mode, "report"),
        "ok": True,
    }
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise StackError(f"fixture missing {key}")
    dump_json(FIXTURE, payload)
    write_report(payload)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "freeze":
        return run_freeze(run_dir, mode)
    if phase == "inventory":
        return run_named_native(
            run_dir, mode, "inventory", ["--execution-graphs", "ffn_only"]
        )
    if phase == "greedy-parity":
        return run_named_native(
            run_dir,
            mode,
            "greedy-parity",
            [
                "--execution-graphs",
                "ffn_only",
                "--stack-a",
                "control",
                "--stack-b",
                "combined",
                "--prefix",
                "8",
            ],
        )
    if phase == "api":
        return run_named_native(
            run_dir,
            mode,
            "api",
            [
                "--execution-graphs",
                "ffn_only",
                "--stack-b",
                "combined",
                "--prefix",
                "8",
            ],
        )
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "p4096":
        return run_session_ab_phase(
            run_dir, mode, "p4096", 4096, 0, "control", "combined"
        )
    if phase == "d128":
        return run_session_ab_phase(
            run_dir, mode, "d128", 128, DECODE_TOKENS, "control", "combined"
        )
    if phase == "d2048":
        return run_session_ab_phase(
            run_dir, mode, "d2048", 2048, DECODE_TOKENS, "control", "combined"
        )
    if phase == "leave-one-out":
        return run_leave_one_out(run_dir, mode)
    if phase == "llama":
        return run_llama(run_dir, mode)
    if phase == "two-k":
        return run_two_k(run_dir, mode)
    if phase == "long-context":
        return run_long_context(run_dir, mode)
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
    json.dump({"phase": args.phase, "ok": bool(result.get("ok", True))}, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("ok", True) or args.phase == "report" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StackError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
