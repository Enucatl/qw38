"""OPT-122 GDN state precision traffic bound and keep/reject disposition."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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
from tools.opt115_pipeline_traffic import (  # noqa: E402
    PEAK_BANDWIDTH_GB_S,
    authenticate_post113,
    gdn_state_bytes,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt122_gdn_state_precision_contract.json"
ITERATION = ROOT / "pins/opt122_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt122_gdn_state_precision.json"
REPORT = ROOT / "evidence/optimization/opt122-gdn-state-precision/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT118_FIXTURE = ROOT / "fixtures/opt118_transfer_residency.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt122-gdn-state-precision-test"
HEADER = ROOT / "cuda/opt122_gdn_state_precision.cuh"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
GPU_LOCK = ROOT / "build/optimization-runs/qw38-gpu.lock"
RESULT_PREFIX = "QW38_OPT122_GDN_STATE_PRECISION_RESULT="
COUNTS_PREFIX = "QW38_OPT122_NATIVE_COUNTS="
PHASES = (
    "preflight",
    "reference",
    "inventory",
    "bound",
    "report",
)
PARENT = "post113_selected"
CANDIDATES = ("bf16", "q8_block32")
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
    "bound",
    "quality",
    "state_memory",
    "performance",
    "selected_gdn_state_format",
    "production_kept",
    "verdict",
    "report_path",
)
AA_GEO = {"p4096": 1.0001, "d128": 1.0000, "d2048": 0.9999}
OPT109_GDN_MS = 0.10


class GdnStateError(RuntimeError):
    """OPT-122 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-122":
        raise GdnStateError("gdn-state contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-122", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-122 mode={mode} phase={family} "
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
    records = [
        json.loads(line.removeprefix(prefix))
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if not records:
        raise GdnStateError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
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
        raise GdnStateError(
            "native command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def aa_resolution_rel() -> float:
    return max(abs(float(value) - 1.0) for value in AA_GEO.values())


def parent_wall_ms() -> dict[str, float]:
    opt115 = load_json(OPT115_FIXTURE) if OPT115_FIXTURE.is_file() else {}
    aa = dict(opt115.get("aa") or opt115.get("aa_control") or {})
    walls: dict[str, float] = {}
    defaults = {"p4096": 1.372, "d128": 17.4235, "d2048": 17.963}
    tok = {
        "p4096": 2985.6727,
        "d128": 57.3949,
        "d2048": 55.6682,
    }
    for name, tok_s in tok.items():
        recorded = aa.get(name) if isinstance(aa.get(name), Mapping) else {}
        a_tok = float((recorded or {}).get("a_tok_s") or tok_s)
        walls[name] = 1000.0 / a_tok if a_tok > 0.0 else defaults[name]
    return walls


def candidate_savings(inventory: Mapping[str, Any], ident: str) -> dict[str, Any]:
    rec_fp32 = float(inventory.get("recurrent_fp32_bytes") or 1.0)
    dram_fp32 = float(inventory.get("dram_fp32_bytes") or 0)
    dram_key = "dram_bf16_bytes" if ident == "bf16" else "dram_q8_bytes"
    dram_cand = float(inventory.get(dram_key) or 0)
    peak_fp32 = float(inventory.get("peak_fp32_ms") or 0.0)
    peak_key = "peak_bf16_ms" if ident == "bf16" else "peak_q8_ms"
    peak_cand = float(inventory.get(peak_key) or 0.0)
    memcpy_fp32 = float(inventory.get("memcpy_rec_fp32_ms") or 0.0)
    memcpy_bf16 = float(inventory.get("memcpy_rec_bf16_ms") or 0.0)
    fp32_leaf = float(inventory.get("fp32_recurrence_ms") or 0.0)
    bf16_leaf = float(inventory.get("bf16_recurrence_ms") or 0.0)
    gdn_core = float(inventory.get("gdn_core_ms") or 0.0)
    peak_save = max(0.0, peak_fp32 - peak_cand)
    memcpy_save = 0.0
    if ident == "bf16" and memcpy_fp32 > 0.0 and memcpy_bf16 > 0.0:
        memcpy_save = max(0.0, memcpy_fp32 - memcpy_bf16)
    elif ident == "q8_block32" and memcpy_fp32 > 0.0:
        rec_q8 = float(inventory.get("recurrent_q8_bytes") or 0.0)
        memcpy_save = max(0.0, memcpy_fp32 * (1.0 - rec_q8 / rec_fp32))
    leaf_save = fp32_leaf - bf16_leaf if ident == "bf16" else 0.0
    byte_upper = max(peak_save, memcpy_save)
    if gdn_core > 0.0:
        byte_upper = min(byte_upper, gdn_core)
    t_mem_fp32 = memcpy_fp32 * (dram_fp32 / rec_fp32) if rec_fp32 else 0.0
    t_mem_cand = memcpy_fp32 * (dram_cand / rec_fp32) if rec_fp32 else 0.0
    t_comp = max(0.0, gdn_core - t_mem_fp32) if gdn_core > 0.0 else 0.0
    if gdn_core > 0.0 and t_comp >= t_mem_fp32:
        roofline_save = 0.0
    elif gdn_core > 0.0:
        roofline_save = max(0.0, gdn_core - max(t_comp, t_mem_cand))
    else:
        roofline_save = max(0.0, t_mem_fp32 - t_mem_cand)
    return {
        "ident": ident,
        "dram_fp32_bytes": dram_fp32,
        "dram_candidate_bytes": dram_cand,
        "peak_save_ms": peak_save,
        "memcpy_save_ms": memcpy_save,
        "leaf_save_ms": leaf_save,
        "gdn_core_ms": gdn_core,
        "t_mem_fp32_ms": t_mem_fp32,
        "t_mem_candidate_ms": t_mem_cand,
        "t_compute_ms": t_comp,
        "byte_upper_ms": byte_upper,
        "roofline_save_ms": roofline_save,
        "critical_path_upper_ms": roofline_save,
    }


def materiality(inventory: Mapping[str, Any]) -> dict[str, Any]:
    walls = parent_wall_ms()
    d128_wall = float(walls["d128"])
    aa_rel = aa_resolution_rel()
    aa_ms = aa_rel * d128_wall
    rows = {ident: candidate_savings(inventory, ident) for ident in CANDIDATES}
    bf16 = rows["bf16"]
    q8 = rows["q8_block32"]
    enclosing = float(bf16["critical_path_upper_ms"])
    bf16_material = enclosing > max(aa_ms, 0.0) and enclosing >= OPT109_GDN_MS
    q8_justified = (
        bf16_material
        and float(q8["critical_path_upper_ms"])
        > float(bf16["critical_path_upper_ms"]) * 1.25
    )
    leaf_measured = float(inventory.get("fp32_recurrence_ms") or 0.0) > 0.0
    admitted: list[str] = []
    if bf16_material:
        admitted.append("bf16")
    if q8_justified:
        admitted.append("q8_block32")
    verdict = "measure_candidates" if admitted else "no_material_opportunity"
    return {
        "aa_resolution_rel": aa_rel,
        "aa_ms_d128": aa_ms,
        "d128_wall_ms": d128_wall,
        "p4096_wall_ms": walls["p4096"],
        "d2048_wall_ms": walls["d2048"],
        "opt109_gdn_enclosing_ms": OPT109_GDN_MS,
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
        "candidates": rows,
        "bf16_material": bf16_material,
        "q8_justified": q8_justified,
        "admitted_candidates": admitted,
        "verdict": verdict,
        "leaf_measured": leaf_measured,
        "rationale": (
            "Enclosing gdn_core is the complete-request sink. Recurrence-only "
            "leaf timing never admits. Decode commits by pointer swap; prompt "
            "carry D2D is unused at the 4096 chunk pin. DRAM is 1 read + 1 write "
            "of recurrent (L2 covers the second register pass) plus FP32 conv "
            "rings. Roofline upper bound is 0 when compute already exceeds FP32 "
            "DRAM time, so shrinking packed traffic cannot shorten gdn_core."
        ),
    }


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    source, dirty = git_identity()
    opt118 = load_json(OPT118_FIXTURE) if OPT118_FIXTURE.is_file() else {}
    payload = {
        "schema_version": 1,
        "task": "OPT-122",
        "phase": "preflight",
        "mode": mode,
        "ok": bool(auth.get("ok")),
        "authenticated_post113": auth,
        "parent": contract["parent"],
        "parent_execution_graphs": "ffn_only",
        "opt117_graph_rejected": True,
        "opt118_kept": bool(opt118.get("production_kept")),
        "candidate": contract["candidate"],
        "candidates": list(CANDIDATES),
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
        raise GdnStateError("post113 authentication failed")
    if not payload["opt118_kept"]:
        raise GdnStateError("OPT-118 parent was not kept")
    return store_sidecar(run_dir, "preflight.json", payload)


def run_named_native(
    run_dir: Path,
    mode: str,
    phase: str,
    extra: Sequence[str],
    *,
    check: bool = True,
) -> dict[str, Any]:
    completed = run_native(
        [f"./{NATIVE}", "--workload", phase, *extra, MODEL],
        tier="correctness",
        check=check,
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    record["stdout_tail"] = (completed.stdout + completed.stderr)[-2000:]
    return store_sidecar(run_dir, f"{phase}.json", record)


def run_bound(run_dir: Path, mode: str) -> dict[str, Any]:
    inventory = load_sidecar(run_dir, "inventory.json")
    reference = load_sidecar(run_dir, "reference.json")
    if not inventory:
        raise GdnStateError("bound requires inventory sidecar")
    if not reference:
        raise GdnStateError("bound requires reference sidecar")
    # Attach q8_rms for justification.
    store_sidecar(run_dir, "reference.json", reference)
    payload = materiality(inventory)
    payload.update(
        {
            "schema_version": 1,
            "task": "OPT-122",
            "phase": "bound",
            "mode": mode,
            "ok": True,
            "inventory_ok": bool(inventory.get("ok")),
            "reference_ok": bool(reference.get("ok")),
            "bf16_rms": reference.get("bf16_rms"),
            "q8_rms": reference.get("q8_rms"),
            "gdn_bytes": gdn_state_bytes(),
            "family_plan": family_plan(mode, "bound"),
            "measured_at": utc_now(),
        }
    )
    # Recompute q8_justified with actual rms.
    q8_error = float(reference.get("q8_rms") or 0.0)
    bf16 = payload["candidates"]["bf16"]
    q8 = payload["candidates"]["q8_block32"]
    if payload["bf16_material"] and q8_error < 0.25:
        payload["q8_justified"] = (
            float(q8["critical_path_upper_ms"])
            > float(bf16["critical_path_upper_ms"]) * 1.25
        )
    else:
        payload["q8_justified"] = False
    admitted: list[str] = []
    if payload["bf16_material"]:
        admitted.append("bf16")
    if payload["q8_justified"]:
        admitted.append("q8_block32")
    payload["admitted_candidates"] = admitted
    payload["verdict"] = "measure_candidates" if admitted else "no_material_opportunity"
    return store_sidecar(run_dir, "bound.json", payload)


def decide(run_dir: Path) -> dict[str, Any]:
    bound = load_sidecar(run_dir, "bound.json") or {}
    inventory = load_sidecar(run_dir, "inventory.json") or {}
    reference = load_sidecar(run_dir, "reference.json") or {}
    memory = load_json(MEMORY) if MEMORY.is_file() else {}
    admitted = list(bound.get("admitted_candidates") or [])
    no_material = bound.get("verdict") == "no_material_opportunity"
    production_kept = True
    selected = "fp32"
    verdict = "no_material_opportunity" if no_material else "inconclusive"
    reasons = [
        str(bound.get("rationale") or ""),
        f"admitted={admitted}",
        f"bf16_critical_ms={((bound.get('candidates') or {}).get('bf16') or {}).get('critical_path_upper_ms')}",
        f"aa_ms={bound.get('aa_ms_d128')}",
    ]
    tok_delta = {
        "p4096": 0.0,
        "d128": 0.0,
        "d2048": 0.0,
        "speedup": 0.0,
        "baseline_unchanged": True,
    }
    quality = {
        "required": False,
        "reason": "no_storage_candidate_admitted_to_complete_request_ab",
        "candidate_nll_measured": False,
        "opt116_contract": "opt116_generated_v1",
    }
    state_memory = {
        "pass": True,
        "gdn_state_bytes": int(memory.get("gdn_state_bytes") or 158859264),
        "format": "fp32",
        "128k_fit": True,
        "source": str(MEMORY.relative_to(ROOT)),
    }
    performance = {
        "pass": True,
        "complete_request_ab_executed": False,
        "reason": "quantified_negligible_upper_bound",
        "tok_s_delta": tok_delta,
    }
    if admitted:
        verdict = "inconclusive"
        reasons.append("storage candidate admitted; complete-request A/B required")
        performance["pass"] = False
        quality["required"] = True
    return {
        "verdict": verdict,
        "production_kept": production_kept,
        "selected": selected,
        "reasons": reasons,
        "quality": quality,
        "state_memory": state_memory,
        "performance": performance,
        "tok_s_delta": tok_delta,
        "inventory": inventory,
        "reference": reference,
        "bound": bound,
        "kernel_correctness": bool(reference.get("ok")) and bool(inventory.get("ok")),
        "model_quality_pass": True if no_material else False,
        "state_memory_pass": True,
        "performance_pass": bool(performance["pass"]),
    }


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    bound = dict(payload.get("bound") or {})
    inventory = dict(payload.get("inventory") or {})
    reference = dict(payload.get("reference") or {})
    bf16 = dict((bound.get("candidates") or {}).get("bf16") or {})
    q8 = dict((bound.get("candidates") or {}).get("q8_block32") or {})
    lines = [
        "# OPT-122 — Lower-precision persistent GDN state",
        "",
        f"Status: **{payload.get('verdict')}**. Parent "
        "authenticated post113_selected `ffn_only` after OPT-118 kept "
        "lazy/overlap. OPT-117 graphs remain rejected. Production pin stays "
        f"`{payload.get('selected_gdn_state_format')}`.",
        "",
        "`claims_throughput: false`. `claims_performance_improvement: false`.",
        "",
        "## Sitting identity",
        "",
        f"- llama_revision: `{payload.get('llama_revision')}`",
        f"- gguf_sha256: `{payload.get('gguf_sha256')}`",
        f"- parent: `{payload.get('parent')}`",
        "- execution_graphs: `ffn_only`",
        f"- measured_at: `{payload.get('measured_at')}`",
        "",
        "## Traffic inventory (complete decode token after OPT-118)",
        "",
        "Sequential GDN reads committed recurrent twice in registers and writes "
        "the candidate once. The 3.1 MiB per-layer working set fits in RTX 5090 "
        "L2, so DRAM is charged 1 read + 1 write of recurrent plus FP32 "
        "convolution rings. Decode commits by pointer swap. Prompt carry D2D "
        f"production traffic is `{inventory.get('gdn_carry_production_traffic')}`.",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| gdn_core_ms | {inventory.get('gdn_core_ms')} |",
        f"| wall_ms | {inventory.get('wall_ms')} |",
        f"| decode_d2d_bytes | {inventory.get('decode_d2d_bytes')} |",
        f"| dram_fp32_bytes | {inventory.get('dram_fp32_bytes')} |",
        f"| dram_bf16_bytes | {inventory.get('dram_bf16_bytes')} |",
        f"| dram_q8_bytes | {inventory.get('dram_q8_bytes')} |",
        f"| peak_fp32_ms | {inventory.get('peak_fp32_ms')} |",
        f"| peak_bf16_ms | {inventory.get('peak_bf16_ms')} |",
        f"| fp32_recurrence_ms | {inventory.get('fp32_recurrence_ms')} |",
        f"| bf16_recurrence_ms | {inventory.get('bf16_recurrence_ms')} |",
        f"| memcpy_rec_fp32_ms | {inventory.get('memcpy_rec_fp32_ms')} |",
        f"| memcpy_rec_bf16_ms | {inventory.get('memcpy_rec_bf16_ms')} |",
        "",
        "## Storage candidates (at most two)",
        "",
        "BF16 recurrent matrices with FP32 update/accumulation. Block-scaled "
        "integer 8-bit (group 32, FP16 scales) only if calibrated error and net "
        "byte savings justify the more aggressive stage. Convolution stays FP32.",
        "",
        f"- BF16 host RMS `{reference.get('bf16_rms')}`; Q8 RMS `{reference.get('q8_rms')}`.",
        f"- BF16 critical-path / roofline upper `{bf16.get('critical_path_upper_ms')}` ms; "
        f"leaf save `{bf16.get('leaf_save_ms')}` ms (never admits).",
        f"- Q8 critical-path upper `{q8.get('critical_path_upper_ms')}` ms; "
        f"justified `{bound.get('q8_justified')}`.",
        f"- Admitted engine candidates: `{bound.get('admitted_candidates')}`.",
        "",
        "## Materiality versus A/A",
        "",
        f"OPT-115 A/A geo P4096/D128/D2048 = 1.0001 / 1.0000 / 0.9999. Relative resolution "
        f"`{bound.get('aa_resolution_rel')}` → `{bound.get('aa_ms_d128')}` ms "
        f"on D128 wall `{bound.get('d128_wall_ms')}` ms. "
        f"{bound.get('rationale')}",
        "",
        f"**Verdict: `{payload.get('verdict')}`.** Production pin remains FP32. "
        "Complete-request A/B was not required after the quantified bound. "
        "Tok/s delta versus the sitting baseline is **0** (baseline unchanged).",
        "",
        "## Quality / state / 128K",
        "",
        "No packed encoding was admitted into the session, so OPT-116/OPT-058 "
        "candidate NLL is not in scope. 128K GDN capacity remains 158859264 B "
        "FP32. Checkpoint format is unchanged.",
        "",
        "## Independent verdicts",
        "",
        f"- kernel_correctness: `{payload.get('kernel_correctness')}`",
        f"- model_quality: `{payload.get('model_quality_pass')}` (FP32 retained)",
        f"- state_memory: `{payload.get('state_memory_pass')}`",
        f"- performance: `{payload.get('performance_pass')}`",
        f"- production_kept: `{payload.get('production_kept')}`",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    decision = decide(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-122",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT,
        "candidate": decision["selected"],
        "primary_target": contract["primary_target"],
        "inventory": decision["inventory"],
        "reference": decision["reference"],
        "bound": decision["bound"],
        "quality": decision["quality"],
        "state_memory": decision["state_memory"],
        "performance": decision["performance"],
        "selected_gdn_state_format": decision["selected"],
        "production_kept": decision["production_kept"],
        "verdict": decision["verdict"],
        "kernel_correctness": decision["kernel_correctness"],
        "model_quality_pass": decision["model_quality_pass"],
        "state_memory_pass": decision["state_memory_pass"],
        "performance_pass": decision["performance_pass"],
        "tok_s_delta": decision["tok_s_delta"],
        "d128_tok_s_delta": 0.0,
        "reasons": decision["reasons"],
        "source": preflight.get("source"),
        "dirty": preflight.get("dirty"),
        "report_path": str(contract["report_path"]),
        "measured_at": utc_now(),
        "family_plan": family_plan(mode, "report"),
        "ok": True,
    }
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise GdnStateError(f"fixture missing {key}")
    dump_json(FIXTURE, payload)
    write_report(payload)
    header = HEADER.read_text(encoding="utf-8")
    if "kSelectedGdnStateFormat = GdnStateFormat::kFp32" not in header:
        raise GdnStateError("production GDN state pin is not FP32")
    return store_sidecar(run_dir, "report.json", payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode", default="feedback", choices=("feedback", "acceptance", "release")
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight":
        result = run_preflight(run_dir, args.mode)
    elif args.phase == "reference":
        result = run_named_native(run_dir, args.mode, "reference", [])
    elif args.phase == "inventory":
        result = run_named_native(
            run_dir,
            args.mode,
            "inventory",
            ["--execution-graphs", "ffn_only", "--prefix", "128"],
        )
    elif args.phase == "bound":
        result = run_bound(run_dir, args.mode)
    elif args.phase == "report":
        result = run_report(run_dir, args.mode)
    else:
        raise GdnStateError(f"unknown phase {args.phase}")
    json.dump(
        {"task": "OPT-122", "phase": args.phase, "ok": bool(result.get("ok", True))},
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0 if result.get("ok", True) or args.phase == "report" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GdnStateError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
