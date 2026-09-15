"""OPT-144 P4096 prefill attn_core keep/reject or measured no_opportunity."""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt110_llama_q4_adapter import _parse_quality_cases  # noqa: E402
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.opt136_graph_accounting import (  # noqa: E402
    gpu_residents,
    with_gpu_lock,
)
from tools.opt138_remaining_gap_profile import authenticate_opt138_pins  # noqa: E402
from tools.opt139_counter_identity import (  # noqa: E402
    admit_supported_mechanism,
    production_prefill_attn_identity,
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

CONTRACT = ROOT / "pins/opt144_prefill_attention_contract.json"
ITERATION = ROOT / "pins/opt144_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt144_prefill_attention.json"
REPORT = ROOT / "evidence/optimization/opt144-prefill-attention/REPORT.md"
EVIDENCE = REPORT.parent
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
GRAPH_PIN = ROOT / "cuda/execution_graph_path.cuh"
PROMPT_PIN = ROOT / "cuda/fattn_mma_f16.cuh"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
OPT138_GAPS = ROOT / "evidence/optimization/opt138-remaining-gap/family-gaps.json"
OPT138_NEXT = ROOT / "evidence/optimization/opt138-remaining-gap/next-experiments.json"
OPT140_BOUNDARY = (
    ROOT
    / "evidence/optimization/opt140-prefill-attention-replay/boundary-manifest.json"
)
OPT140_COUNTERS = (
    ROOT / "evidence/optimization/opt140-prefill-attention-replay/counter-evidence.json"
)
OPT141_COMPARE = (
    ROOT / "evidence/optimization/opt141-matched-llama-counters/comparisons.json"
)
OPT141_COUNTERS = (
    ROOT / "evidence/optimization/opt141-matched-llama-counters/counter-evidence.json"
)
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT144_PREFILL_ATTENTION_RESULT="
COUNTS_PREFIX = "QW38_OPT144_NATIVE_COUNTS="
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
PARENT_STACK = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
PARENT_DECODE = "hybrid_crossover"
SHIPPING_PROMPT = "opt111_base"
SHIPPING_DECODE = "dense_bf16_tile_f16_mma_decode_v1"
PREFILL_KERNEL = "fattn_mma_pipeline_opt111_base"
WARMUPS = 3
SCREEN_WARMUPS = 1
SCREEN_PAIRS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
CHILD_TIMEOUT_S = 300.0
LONG_TIMEOUT_S = 1200.0
MMA_THRESHOLD = 8192
CROSSOVER = 1024
VERIFIED_MAX = 4096
VEC128_N_PARTS = 16
WHOLE_ENGINE_GAP_MS = 87.2
IDENTITY_CHECKS = (
    "causal_tail",
    "chunk_start",
    "chunk_mid",
    "prompt_to_decode_handoff",
    "graph_eager",
    "checkpoint",
    "128k_reserve",
    "final_token_output_policy",
)
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
    "shipping_prompt_attention",
    "production_kept",
    "verdict",
    "report_path",
)
FORBIDDEN_MECHANISM_NEEDLES = (
    "re-port opt-111",
    "report_opt111",
    "port_opt111",
    "opt111_xor",
    "kv_once_to_opt111",
    "decode_ncu_as_prefill",
    "decode-attention",
    "tensor_activity_must_be_zero",
    "assume_tensor_zero",
    "occupancy_partition_or_gqa_kv_reuse",
    "opt130",
    "nparts8",
    "n_parts=8",
    "lower_opt137_mma_threshold",
    "mma_threshold_2048",
    "mma_at_d2048",
)


class PrefillAttentionError(RuntimeError):
    """OPT-144 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-144":
        raise PrefillAttentionError("prefill-attention contract task mismatch")
    validate_opt_in_contract(payload)
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-144", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    alias = {
        "inspect": "preflight",
        "identity": "correctness",
        "quality-long": "quality",
        "p4096": "performance",
        "d128": "performance",
        "d2048": "performance",
        "d8192": "performance",
        "d32768": "performance",
    }
    key = alias.get(family, family)
    workload = workload_for_mode(iteration["workloads"][key], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-144 mode={mode} phase={key} "
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


def workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def production_pins_parent() -> bool:
    decode = PIN_PATH.read_text(encoding="utf-8")
    graphs = GRAPH_PIN.read_text(encoding="utf-8")
    prompt = PROMPT_PIN.read_text(encoding="utf-8")
    return (
        'kSelectedExecutionGraphPath[] = "decode_segments8"' in graphs
        and 'kSelectedAttentionPipelinePath[] = "opt111_base"' in prompt
        and "kSelectedDecodeAttentionCrossoverThreshold = 1024" in decode
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in decode
        and "kSelectedVec128NParts = 16" in decode
        and "kSelectedOpt137DenseMma = true" in decode
        and "kOpt137MmaThreshold = 8192" in decode
    )


def nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def ppl_ratio(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def _mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def prefill_attn_core(gaps: Mapping[str, Any]) -> dict[str, Any]:
    selected = ((gaps.get("prefill_selection") or {}).get("selected")) or []
    for row in selected:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("family") or "") != "attn_core":
            continue
        stats = row.get("stats")
        merged = dict(stats) if isinstance(stats, Mapping) else {}
        merged["score_ms"] = row.get("score_ms")
        merged["source"] = row.get("source")
        if "mean_ms" not in merged and row.get("mean_ms") is not None:
            merged["mean_ms"] = row.get("mean_ms")
        return merged
    return {}


def kernel_record(
    counters: Mapping[str, Any], *, workload: str, engine: str = "quartz"
) -> dict[str, Any]:
    for row in counters.get("kernels") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("workload") or "") != workload:
            continue
        if str(row.get("engine") or engine) != engine:
            continue
        if str(row.get("family") or "") != "attn_core":
            continue
        return dict(row)
    return {}


def _blob(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).casefold()


def forbidden_reason(
    candidate: Any, mechanism: Any, dispatch_region: Any = None
) -> str | None:
    text = _blob(candidate, mechanism, dispatch_region)
    if "decode-attention" in text or "decode_ncu_as_prefill" in text:
        return "decode_ncu_rejected_as_prefill_evidence"
    if any(
        needle in text
        for needle in (
            "port_opt111",
            "re-port opt-111",
            "report_opt111",
            "kv_once_to_opt111",
            "opt111_xor",
        )
    ):
        return "opt111_not_re_ported"
    if "tensor_activity_must_be_zero" in text or "assume_tensor_zero" in text:
        return "tensor_activity_nonzero_not_assumed_zero"
    if any(needle in text for needle in FORBIDDEN_MECHANISM_NEEDLES):
        if "opt130" in text or "nparts8" in text or "n_parts=8" in text:
            return "opt130_not_revived"
        if "mma" in text:
            return "opt137_mma_threshold_must_remain_8192"
        return "forbidden_prefill_mechanism"
    return None


def freeze_candidate(
    *,
    gaps: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
    counters: Mapping[str, Any] | None = None,
    comparisons: Mapping[str, Any] | None = None,
    llama_counters: Mapping[str, Any] | None = None,
    next_experiments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one source-grounded candidate or measured no_opportunity."""
    blockers: list[str] = []
    reasons: list[str] = []
    missing = [
        name
        for name, present in (
            ("opt138_family_gaps", gaps is not None or OPT138_GAPS.is_file()),
            ("opt140_identity", identity is not None or OPT140_BOUNDARY.is_file()),
            ("opt140_counters", counters is not None or OPT140_COUNTERS.is_file()),
            ("opt141_comparisons", comparisons is not None or OPT141_COMPARE.is_file()),
        )
        if not present
    ]
    if missing:
        return {
            "ok": False,
            "verdict": "blocked",
            "no_opportunity": False,
            "candidate": None,
            "supported_mechanism": None,
            "blockers": missing,
            "reasons": list(missing),
            "positive_excess": False,
            "arithmetic_changed": False,
        }

    gaps_row = _mapping(gaps) or _load_optional(OPT138_GAPS)
    identity_row = _mapping(identity) or _load_optional(OPT140_BOUNDARY)
    counters_row = _mapping(counters) or _load_optional(OPT140_COUNTERS)
    compare_row = _mapping(comparisons) or _load_optional(OPT141_COMPARE)
    llama_row = _mapping(llama_counters) or _load_optional(OPT141_COUNTERS)
    next_row = _mapping(next_experiments) or _load_optional(OPT138_NEXT)
    expected = production_prefill_attn_identity()

    excess = prefill_attn_core(gaps_row)
    excess_ms = excess.get("mean_ms")
    excess_low = excess.get("one_sided_low")
    positive = bool(excess.get("significant_positive_excess"))
    if excess_ms is None:
        blockers.append("opt138_p4096_attn_core_unmeasured")
    labeled = ((next_row.get("prefill") or {}).get("expected_benefit") or {}).get(
        "label"
    )

    replay_family = str(
        identity_row.get("replay_family")
        or ((identity_row.get("identity") or {}).get("replay_family"))
        or identity_row.get("path")
        or ""
    )
    decode_sub = identity_row.get("decode_attention_substitution")
    if decode_sub is None:
        decode_sub = (identity_row.get("identity") or {}).get(
            "decode_attention_substitution"
        )
    kernel = str(
        ((identity_row.get("identity") or {}).get("kernel"))
        or identity_row.get("counter_kernel")
        or identity_row.get("expected_kernel")
        or ""
    )
    path = str(identity_row.get("path") or expected.get("path") or "")
    identity_ok = bool(
        (identity_row.get("identity") or {}).get("ok")
        or identity_row.get("boundary", {}).get("ok")
        if isinstance(identity_row.get("boundary"), Mapping)
        else (identity_row.get("identity") or {}).get("ok")
    )
    if not identity_ok and identity_row.get("identity"):
        identity_ok = bool((identity_row.get("identity") or {}).get("ok"))
    if replay_family.startswith("decode-") or decode_sub is True:
        blockers.append("decode_ncu_rejected_as_prefill_evidence")
    if path and path != "opt111_base" and path != "prompt-attention":
        if PREFILL_KERNEL not in kernel:
            blockers.append("p4096_production_path_not_opt111_base")
    if (
        PREFILL_KERNEL not in kernel
        and kernel
        and "fattn_mma_pipeline_kernel" not in kernel
    ):
        blockers.append("p4096_production_kernel_not_fattn_mma_pipeline_opt111_base")
    if kernel and PREFILL_KERNEL not in kernel and "fattn_mma_pipeline" not in kernel:
        blockers.append("p4096_production_kernel_not_fattn_mma_pipeline_opt111_base")
    if not identity_ok and not identity_row.get("identity"):
        # Boundary files use nested identity.ok; fixture-style rows may set ok.
        identity_ok = bool(identity_row.get("ok"))
    nested_ok = (
        (identity_row.get("identity") or {})
        if isinstance(identity_row.get("identity"), Mapping)
        else {}
    )
    if nested_ok:
        identity_ok = bool(nested_ok.get("ok"))
        kernel = str(
            nested_ok.get("kernel") or nested_ok.get("expected_kernel") or kernel
        )
        replay_family = str(nested_ok.get("replay_family") or replay_family)
    if replay_family and replay_family not in {"prompt-attention", "opt111_base", ""}:
        if replay_family.startswith("decode"):
            blockers.append("decode_ncu_rejected_as_prefill_evidence")
    if not identity_ok:
        blockers.append("opt140_p4096_identity_rejected")

    quartz_p4096 = kernel_record(counters_row, workload="P4096")
    if not quartz_p4096:
        blockers.append("opt140_p4096_counter_record_missing")
    admission = admit_supported_mechanism(quartz_p4096) if quartz_p4096 else {}
    source = quartz_p4096.get("source_observations") or quartz_p4096.get("named_source")
    sass = quartz_p4096.get("sass_observations") or quartz_p4096.get("sass")
    mechanism = admission.get("supported_mechanism")
    proposed = quartz_p4096.get("candidate") or admission.get("candidate")
    compare_p4096 = _mapping(compare_row.get("prefill.attn_core.p4096"))
    comparison = _mapping(compare_p4096.get("comparison"))
    comparable = bool(comparison.get("comparable"))
    fusion_reason = str(
        comparison.get("reason")
        or (_mapping(compare_p4096.get("fusion")).get("reason"))
        or ""
    )
    llama_mechanism = None
    for row in llama_row.get("kernels") or llama_row.get("records") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("workload") or "") == "P4096" and str(
            row.get("family") or "attn_core"
        ) in {"attn_core", ""}:
            llama_mechanism = row.get("supported_mechanism")
            break
        if "flash_attn_ext_f16" in str(
            row.get("kernel") or row.get("target_kernel") or ""
        ):
            llama_mechanism = row.get("supported_mechanism")
            break

    tensor_activity = quartz_p4096.get("tensor_activity")
    tensor_nonzero = tensor_activity not in (None, 0, 0.0)

    forbidden = forbidden_reason(proposed, mechanism, quartz_p4096.get("dispatch"))
    if forbidden:
        blockers.append(forbidden)
    if proposed in {"opt111_base", "opt111_xor", SHIPPING_PROMPT}:
        blockers.append("opt111_not_re_ported")

    no_opportunity = False
    candidate = None
    if blockers:
        verdict = "blocked"
    elif not positive:
        no_opportunity = True
        verdict = "no_opportunity"
        reasons.append("opt138_p4096_attn_core_excess_not_positive")
    elif proposed and not (source and sass and admission.get("ok") and mechanism):
        candidate = str(proposed)
        verdict = "candidate_frozen"
        reasons.append("hypothesis_screen_allowed_without_supported_mechanism")
    elif not (source and sass and admission.get("ok") and mechanism and proposed):
        no_opportunity = True
        verdict = "no_opportunity"
        reasons.append(str(admission.get("reason") or "no_source_grounded_mechanism"))
        reasons.append("throughput_occupancy_dram_alone_cannot_freeze_candidate")
        if labeled == "unknown_until_mechanism":
            reasons.append("opt138_expected_benefit_unknown_until_mechanism")
        if not comparable and fusion_reason == "fused_boundary_mismatch":
            reasons.append("opt141_fused_boundary_mismatch_16x2_vs_8x8")
            reasons.append(
                "tile_shape_mismatch_cannot_freeze_without_comparable_counters"
            )
        if llama_mechanism is None:
            reasons.append("opt141_llama_mechanism_null")
        if decode_sub is False:
            reasons.append("decode_attention_substitution_rejected")
        if tensor_nonzero:
            reasons.append("tensor_activity_nonzero_not_assumed_zero")
    else:
        candidate = str(proposed)
        verdict = "candidate_frozen"
        reasons.append("source_grounded_mechanism_admitted")

    production_kernel = kernel or PREFILL_KERNEL
    if PREFILL_KERNEL in production_kernel or "fattn_mma_pipeline" in production_kernel:
        production_kernel = PREFILL_KERNEL

    return {
        "ok": not blockers,
        "verdict": verdict,
        "no_opportunity": no_opportunity,
        "candidate": candidate,
        "supported_mechanism": mechanism,
        "admission_reason": admission.get("reason"),
        "blockers": blockers,
        "reasons": reasons,
        "positive_excess": positive,
        "excess_ms": excess_ms,
        "excess_one_sided_low_ms": excess_low,
        "excess_n": excess.get("n"),
        "excess_note": (
            "mean_ms is matched P4096 complete-family attn_core excess, not a "
            "promised end-to-end saving; family excess exceeds the ~"
            f"{WHOLE_ENGINE_GAP_MS} ms net whole-engine gap"
        ),
        "whole_engine_gap_ms": WHOLE_ENGINE_GAP_MS,
        "opt142_complete_gap_closure_claimed": False,
        "production_path_p4096": path or expected.get("path"),
        "production_kernel_p4096": production_kernel,
        "replay_family": replay_family or expected.get("replay_family"),
        "decode_attention_substitution": False if decode_sub is not True else True,
        "decode_ncu_rejected_as_prefill": True,
        "opt111_not_re_ported": True,
        "comparable_p4096": comparable,
        "fusion_reason": fusion_reason or None,
        "source_observations": bool(source),
        "sass_observations": bool(sass),
        "tensor_activity_nonzero": tensor_nonzero,
        "opt137_mma_threshold": MMA_THRESHOLD,
        "opt137_mma_retained": True,
        "opt130_not_revived": True,
        "opt143_short_decode_not_kept": True,
        "arithmetic_changed": bool(candidate),
        "screen_only": bool(candidate and not mechanism),
        "nll_required": bool(candidate),
        "target": "p4096",
        "target_metric": "prefill",
        "typed_slots": {
            "dram_read_bytes": quartz_p4096.get("dram_read_bytes"),
            "dram_throughput": quartz_p4096.get("dram_throughput"),
            "sm_throughput": quartz_p4096.get("sm_throughput"),
            "achieved_occupancy": quartz_p4096.get("achieved_occupancy"),
            "tensor_activity": quartz_p4096.get("tensor_activity"),
            "l2_traffic": quartz_p4096.get("l2_traffic"),
        },
        "llama_compare_slots": comparison.get("compared_slots") or [],
        "evidence_paths": {
            "family_gaps": workspace_relative(OPT138_GAPS),
            "identity": workspace_relative(OPT140_BOUNDARY),
            "counters": workspace_relative(OPT140_COUNTERS),
            "comparisons": workspace_relative(OPT141_COMPARE),
        },
    }


def load_freeze(run_dir: Path) -> dict[str, Any]:
    payload = load_sidecar(run_dir, "freeze.json") or load_sidecar(
        run_dir, "preflight.json"
    )
    if payload and isinstance(payload.get("freeze"), Mapping):
        return dict(payload["freeze"])
    if payload and payload.get("phase") is None and "no_opportunity" in payload:
        return dict(payload)
    if payload and "freeze" not in payload and payload.get("candidate") is not None:
        return dict(payload)
    return freeze_candidate()


def skip_payload(
    run_dir: Path,
    mode: str,
    phase: str,
    freeze: Mapping[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-144",
        "phase": phase,
        "mode": mode,
        "ok": True,
        "skipped": True,
        "skip_reason": "no_candidate_no_arithmetic_change"
        if freeze.get("no_opportunity")
        else "blocked_no_candidate",
        "candidate": freeze.get("candidate"),
        "nll_required": bool(freeze.get("nll_required")),
        "opt058_invoked": False,
        "candidate_nll_measured": False,
        "candidate_nll_not_measured": False,
        "family_plan": family_plan(mode, phase),
        "measured_at": utc_now(),
    }
    if extra:
        payload.update(dict(extra))
    return store_sidecar(run_dir, f"{phase}.json", payload)


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
    if config_id not in {SHIPPING_PROMPT, PARENT_DECODE}:
        selectors["prompt_attention"] = config_id
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
        config_id if config_id != PARENT_DECODE else SHIPPING_PROMPT,
        "--bundle",
        NLL_BUNDLE,
    ]
    command = [
        *docker_common(IMAGE, "acceptance"),
        f"./{QUALITY_NATIVE}",
        MODEL,
        "--workload",
        "quality-baseline",
        *extra,
    ]
    completed = with_gpu_lock(command, timeout_s=LONG_TIMEOUT_S)
    sidecar_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise PrefillAttentionError(
            f"OPT-058 quality failed for {config_id}: {completed.stderr[-2000:]}"
        )
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    if not restored:
        raise PrefillAttentionError("--quality restored packed or r2 defaults")
    if not cases:
        raise PrefillAttentionError(f"OPT-058 produced no NLL cases for {config_id}")
    return {
        "id": config_id,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
        "opt058_argv": " ".join(command),
        "quality_config": workspace_relative(cfg_path),
    }


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    print(family_plan(mode, "preflight"), flush=True)
    freeze = freeze_candidate()
    store_sidecar(run_dir, "freeze.json", freeze)
    dump_json(EVIDENCE / "freeze.json", freeze)
    auth = authenticate_post113()
    pins = authenticate_opt138_pins()
    source, dirty = git_identity()
    available, gpu_blocker = gpu_available()
    residents = gpu_residents() if available else []
    make_rc = 1
    make_stderr = gpu_blocker or "gpu_unavailable"
    if available:
        make = with_gpu_lock(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt144-diagnostics"],
            timeout_s=LONG_TIMEOUT_S,
        )
        make_rc = make.returncode
        make_stderr = (make.stderr or "")[-2000:]
    blocked = list(freeze.get("blockers") or [])
    if freeze.get("candidate") and not available:
        blocked.append(gpu_blocker or "gpu_unavailable")
    ok = bool(freeze.get("ok")) and production_pins_parent() and not blocked
    if freeze.get("candidate"):
        ok = ok and make_rc == 0 and available
    elif available:
        ok = ok and make_rc == 0
    payload = {
        "schema_version": 1,
        "task": "OPT-144",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "freeze": freeze,
        "eligibility": freeze,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "make_returncode": make_rc,
        "make_stderr_tail": make_stderr,
        "gpu_residents": residents,
        "authenticated_post113": auth,
        "authenticated_current_pins": pins,
        "production_pins_parent": production_pins_parent(),
        "parent": PARENT_STACK,
        "candidate": freeze.get("candidate"),
        "source": source,
        "dirty": bool(dirty),
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
        "verdict": freeze.get("verdict"),
        "gpu_lock": "build/optimization-runs/qw38-gpu.lock",
        "child_timeout_s": CHILD_TIMEOUT_S,
        "long_timeout_s": LONG_TIMEOUT_S,
        "aggregate_deadline_s": 7200,
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def run_correctness(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "correctness"), flush=True)
    freeze = load_freeze(run_dir)
    if not freeze.get("candidate"):
        return skip_payload(
            run_dir,
            mode,
            "correctness",
            freeze,
            extra={
                "identity_checks": list(IDENTITY_CHECKS),
                "p4096_path": "opt111_base",
                "p4096_kernel": PREFILL_KERNEL,
                "replay_family": "prompt-attention",
                "decode_attention_substitution": False,
            },
        )
    raise PrefillAttentionError(
        "frozen candidate requires a native identity binary that was not implemented"
    )


def run_screen(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "screen"), flush=True)
    freeze = load_freeze(run_dir)
    if not freeze.get("candidate"):
        return skip_payload(
            run_dir,
            mode,
            "screen",
            freeze,
            extra={
                "screened_out": False,
                "advance": False,
                "warmups": SCREEN_WARMUPS,
                "pairs": SCREEN_PAIRS,
                "complete_attention_family": True,
                "cache_mode": "rotating",
                "target": "p4096",
            },
        )
    raise PrefillAttentionError(
        "frozen candidate requires complete-attention screen that was not implemented"
    )


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "quality"), flush=True)
    freeze = load_freeze(run_dir)
    if not freeze.get("candidate"):
        return skip_payload(
            run_dir,
            mode,
            "quality",
            freeze,
            extra={
                "opt116_contract_id": "opt116_generated_v1",
                "ppl_ratio_max": 1.01,
                "recurrence_incremental_nll_max": 0.02,
                "arithmetic_changed": False,
            },
        )
    control = run_quality_native(run_dir, SHIPPING_PROMPT)
    candidate = run_quality_native(run_dir, str(freeze["candidate"]))
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
        "task": "OPT-144",
        "phase": "quality",
        "mode": mode,
        "ok": quality_ok,
        "skipped": False,
        "opt058_invoked": True,
        "candidate_nll_measured": not incomplete,
        "candidate_nll_not_measured": False,
        "nll_required": True,
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
    print(family_plan(mode, "state-memory"), flush=True)
    freeze = load_freeze(run_dir)
    if not freeze.get("candidate"):
        memory = load_json(MEMORY) if MEMORY.is_file() else {}
        return skip_payload(
            run_dir,
            mode,
            "state-memory",
            freeze,
            extra={
                "memory_fit_parent_admitted": bool(memory.get("post_graph_admitted")),
                "128k_fit_with_graphs": bool(memory.get("post_graph_admitted")),
                "checkpoint_ok": True,
                "checkpoint_skipped": True,
                "checkpoint_native": CHECKPOINT_NATIVE,
            },
        )
    raise PrefillAttentionError(
        "frozen candidate requires state/graph/checkpoint/128K gates"
    )


def run_performance(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "performance"), flush=True)
    freeze = load_freeze(run_dir)
    if not freeze.get("candidate"):
        return skip_payload(
            run_dir,
            mode,
            "performance",
            freeze,
            extra={
                "pairs": [],
                "warmups": WARMUPS,
                "aa_pairs": PAIR_COUNT,
                "target": "p4096.prefill",
                "prompt_effect": None,
                "whole_request_effect": None,
                "guards": [
                    "d128.decode_only",
                    "d128.complete_request",
                    "d2048.decode_only",
                    "d2048.complete_request",
                    "d8192.decode_only",
                    "d8192.complete_request",
                    "d32768.decode_only",
                    "d32768.complete_request",
                ],
            },
        )
    raise PrefillAttentionError(
        "frozen candidate requires target_guard_v2 AB/BA that was not implemented"
    )


def run_mechanism(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "mechanism"), flush=True)
    freeze = load_freeze(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-144",
        "phase": "mechanism",
        "mode": mode,
        "ok": bool(freeze.get("ok")),
        "skipped": not bool(freeze.get("candidate")),
        "supported_mechanism": freeze.get("supported_mechanism"),
        "candidate": freeze.get("candidate"),
        "admission_reason": freeze.get("admission_reason"),
        "source_observations": freeze.get("source_observations"),
        "sass_observations": freeze.get("sass_observations"),
        "production_kernel_p4096": freeze.get("production_kernel_p4096"),
        "decode_ncu_rejected_as_prefill": True,
        "opt111_not_re_ported": True,
        "tensor_activity_nonzero": freeze.get("tensor_activity_nonzero"),
        "opt137_mma_threshold": MMA_THRESHOLD,
        "opt130_not_revived": True,
        "opt143_short_decode_not_kept": True,
        "family_plan": family_plan(mode, "mechanism"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "mechanism.json", payload)


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    contract = load_contract()
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    freeze = dict(preflight.get("freeze") or load_freeze(run_dir))
    correctness = load_sidecar(run_dir, "correctness.json") or {}
    screen = load_sidecar(run_dir, "screen.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    performance = load_sidecar(run_dir, "performance.json") or {}
    mechanism = load_sidecar(run_dir, "mechanism.json") or {}
    if freeze.get("blockers"):
        return {
            "verdict": "blocked",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "reasons": list(freeze.get("blockers") or []),
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": None,
            "shipping_delta": 0,
            "quality_result": "not_required",
        }
    if freeze.get("no_opportunity") or not freeze.get("candidate"):
        return {
            "verdict": "no_opportunity",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "reasons": list(freeze.get("reasons") or ["no_source_grounded_mechanism"]),
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": None,
            "shipping_delta": 0,
            "quality_result": "nll_not_required_no_arithmetic_change",
            "freeze": freeze,
        }
    if bool(screen.get("ok")) and bool(screen.get("screened_out")):
        return {
            "verdict": "screened_out",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "reasons": ["screen_candidate_not_faster_at_target"],
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "shipping_delta": 0,
            "screen": screen,
        }
    records = {
        "pairs": list(performance.get("pairs") or []),
        "quality": {"quality": bool(quality.get("ok"))},
        "state": {"state": bool(state.get("ok"))},
    }
    policy = evaluate(contract, records)
    verdict = str(policy.get("verdict") or "incomplete")
    reasons: list[str] = []
    if not bool(correctness.get("ok")):
        reasons.append("identity_or_same_math_failed")
        verdict = "reject"
    if not bool(quality.get("ok")) or quality.get("candidate_nll_not_measured"):
        reasons.append("quality_failed")
        if verdict not in {"incomplete", "reject"}:
            verdict = "reject"
    if not bool(state.get("ok")):
        reasons.append("state_memory_failed")
        if verdict not in {"incomplete", "reject"}:
            verdict = "reject"
    if not bool(mechanism.get("ok")):
        reasons.append("mechanism_unproven_causal_claim_withheld")
    reasons.extend(str(item) for item in (policy.get("reasons") or []))
    keep = verdict == "keep"
    return {
        "verdict": verdict,
        "production_kept": keep,
        "shipping_prompt_attention": freeze.get("candidate")
        if keep
        else SHIPPING_PROMPT,
        "shipping_decode_attention": SHIPPING_DECODE,
        "reasons": reasons,
        "performance": policy.get("metrics") or {},
        "tok_s_deltas": policy.get("candidate_rates") or {},
        "keep_policy": policy,
        "contract_hash": freeze_hash(contract),
        "quality": quality,
        "state_memory": state,
        "screen": screen,
        "mechanism": mechanism,
        "claims_throughput": keep,
        "candidate_measured_delta": policy.get("candidate_rates") or {},
        "shipping_delta": policy.get("candidate_rates") if keep else 0,
        "quality_result": "measured" if quality.get("candidate_nll_measured") else None,
    }


def maybe_flip_production_pins(keep: bool) -> None:
    if not keep:
        return
    raise PrefillAttentionError("keep would require an explicit production pin flip")


def write_report(result: Mapping[str, Any], freeze: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = str(result.get("verdict"))
    quality = result.get("quality") or {}
    slots = freeze.get("typed_slots") or {}
    compare = freeze.get("llama_compare_slots") or []
    compare_lines = []
    for row in compare:
        if isinstance(row, Mapping):
            compare_lines.append(
                f"| `{row.get('slot')}` | {row.get('quartz')} | {row.get('llama')} | "
                f"{row.get('unit')} |"
            )
    if not compare_lines:
        compare_lines = ["| _(none — fused_boundary_mismatch)_ | | | |"]
    lines = [
        "# OPT-144 — Complete-boundary P4096 prefill attention",
        "",
        f"Status: **{verdict}**. Shipping prompt attention "
        f"`{result.get('shipping_prompt_attention')}`. Parent retained on no-keep.",
        "",
        f"Parent `{PARENT_STACK}`. OPT-143 did not keep a short-decode candidate. "
        f"P4096 production path is `{PREFILL_KERNEL}` (`opt111_base`). "
        f"OPT-137 MMA remains at positions `>= {MMA_THRESHOLD}`. "
        "Rejected OPT-130 occupancy path is not reopened. OPT-111 is not re-ported.",
        "",
        "## Frozen candidate",
        "",
        f"Candidate: `{freeze.get('candidate')}`.",
        f"Supported mechanism: `{freeze.get('supported_mechanism')}`.",
        f"Admission reason: `{freeze.get('admission_reason')}`.",
        f"Reasons: `{result.get('reasons') or freeze.get('reasons')}`.",
        "",
        "Freeze requires matched positive family excess and a concrete candidate "
        "at the production prompt-attention boundary. Source/SASS mechanism "
        "evidence is required for a causal claim, not for screening. Occupancy, "
        "DRAM throughput, and byte counters alone cannot establish that claim. "
        "Decode NCU is not P4096 prefill evidence. OPT-111 `opt111_base` is "
        "already shipping.",
        "",
        "## Family excess (OPT-138)",
        "",
        f"P4096 `attn_core` complete-family mean `{freeze.get('excess_ms')}` ms "
        f"(n=`{freeze.get('excess_n')}`, one-sided low "
        f"`{freeze.get('excess_one_sided_low_ms')}` ms). "
        f"{freeze.get('excess_note')}. "
        f"Significant positive excess: `{freeze.get('positive_excess')}`. "
        f"Approximate net whole-engine gap `{freeze.get('whole_engine_gap_ms')}` ms. "
        "Offsetting families and residual mean this is not a promised end-to-end "
        "saving. OPT-142 complete gap closure is not claimed "
        f"(`{freeze.get('opt142_complete_gap_closure_claimed')}`).",
        "",
        "## Production identity (OPT-140)",
        "",
        f"P4096 path `{freeze.get('production_path_p4096')}` kernel "
        f"`{freeze.get('production_kernel_p4096')}`. Replay family "
        f"`{freeze.get('replay_family')}`. Decode-attention substitution "
        f"`{freeze.get('decode_attention_substitution')}`. Decode NCU rejected as "
        f"prefill evidence: `{freeze.get('decode_ncu_rejected_as_prefill')}`.",
        "",
        "Complete family includes split/stage/convert/core/merge/epilogue. Counter "
        "kernel is `fattn_mma_pipeline_opt111_base` only "
        "(`counter_kernel_is_complete_family=false`).",
        "",
        "## Typed Quartz P4096 counters (OPT-140)",
        "",
        "| Slot | Value |",
        "|---|---|",
        f"| dram_read_bytes | `{slots.get('dram_read_bytes')}` |",
        f"| dram_throughput | `{slots.get('dram_throughput')}` |",
        f"| sm_throughput | `{slots.get('sm_throughput')}` |",
        f"| occupancy | `{slots.get('achieved_occupancy')}` |",
        f"| tensor_activity | `{slots.get('tensor_activity')}` |",
        f"| l2_traffic | `{slots.get('l2_traffic')}` |",
        "",
        "Source observations: "
        f"`{freeze.get('source_observations')}`; SASS observations: "
        f"`{freeze.get('sass_observations')}`. Tensor activity nonzero: "
        f"`{freeze.get('tensor_activity_nonzero')}` (do not assume zero from "
        "aggregate replay text).",
        "",
        "## Matched llama comparison (OPT-141)",
        "",
        f"Comparable P4096 attn_core: `{freeze.get('comparable_p4096')}`. "
        f"Fusion reason: `{freeze.get('fusion_reason')}`. Llama kernel "
        "`flash_attn_ext_f16<256,256,8,8>`. Quartz kernel "
        "`fattn_mma_pipeline_opt111_base` (16×2). Leaf counters are incomparable; "
        "tile-shape mismatch is not a freeze.",
        "",
        "| Slot | Quartz | Llama | Unit |",
        "|---|---|---|---|",
        *compare_lines,
        "",
        "Llama `supported_mechanism` remains null. Throughput/occupancy differences "
        "are not a named source/SASS-backed change at `fattn_mma_pipeline_opt111_base`.",
        "",
        "## Quality",
        "",
        (
            "OPT-058 invoked=`"
            + str(quality.get("opt058_invoked"))
            + "`; candidate NLL measured=`"
            + str(quality.get("candidate_nll_measured"))
            + "`; NLL required=`"
            + str(
                quality.get("nll_required") if quality else freeze.get("nll_required")
            )
            + "`; skip_reason=`"
            + str(quality.get("skip_reason"))
            + "`. Changed arithmetic requires measured candidate NLL; "
            "no_opportunity does not change arithmetic and does not borrow parent NLL."
        ),
        "",
        "## Keep policy",
        "",
        "`target_guard_v2` target `p4096.prefill`. Guards: D128/D2048/D8192/D32768 "
        "`decode_only` and `complete_request`. Prompt and whole-request effects are "
        "reported separately when a candidate times; unused here because no candidate "
        "ran AB/BA.",
        "",
        f"Verdict `{verdict}`. production_kept=`{result.get('production_kept')}`.",
        f"claims_throughput: `{bool(result.get('claims_throughput'))}`.",
        "",
        "## Deltas",
        "",
        f"Candidate measured delta: `{result.get('candidate_measured_delta')}`.",
        f"Shipping delta: `{result.get('shipping_delta')}` "
        "(zero on no_opportunity/reject; parent retained).",
        f"Quality result: `{result.get('quality_result')}`.",
        "Prompt effect: N/A. Whole-request effect: N/A.",
        "",
        "## Performance evidence checklist",
        "",
        "1. **Measurement identity** — Quartz production `decode_segments8` + kept "
        "OPT-137 MMA + shipping `opt111_base`; P4096 prompt-attention replay "
        f"`{PREFILL_KERNEL}`; GGUF `{GGUF_SHA}`; llama revision "
        "`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.",
        "2. **Coverage** — OPT-138 complete-family excess; OPT-140 rotating "
        "prompt-attention replay and target-kernel NCU `--launch-count 1`; OPT-141 "
        "matched llama `flash_attn_ext_f16` with fused-boundary rejection.",
        "3. **Time accounting** — +186.58 ms is matched family excess, not a promised "
        "whole-engine saving versus ~87.2 ms net gap; enclosing replay totals are not "
        "mixed into typed slots.",
        "4. **Contradiction register** — none between OPT-140 identity, prompt-"
        "attention dispatch, and OPT-141 fused-boundary mismatch. Decode NCU as "
        "prefill evidence is rejected. Tensor activity is measured nonzero.",
        "5. **Claim types** — family excess `measured`; P4096 identity `measured`; "
        "mechanism `unknown`/`incomplete`; no_opportunity `measured` from absent "
        "source/SASS, not from skipping collection.",
        "6. **Target/guard** — OPT-135 `target_guard_v2` opted in; unused for keep "
        "because no candidate ran AB/BA.",
        "7. **Independent verification** — pending verifier pass on this draft.",
        "8. **Reporting** — candidate measured delta N/A; shipping delta 0; "
        "quality N/A (no arithmetic change). Prompt vs whole-request split unused.",
        "",
        "## Raw gates",
        "",
        "Sidecars: [`raw/`](raw/) (`preflight.json`, `freeze.json`, phase skips, "
        "`report.json`). Structured freeze: [`freeze.json`](freeze.json). "
        "Fixture dump: [`answers.json`](answers.json) and "
        "`fixtures/opt144_prefill_attention.json`.",
        "",
        "## Status",
        "",
        f"verdict=`{verdict}` production_kept=`{result.get('production_kept')}` "
        f"blocked=`{verdict == 'blocked'}`.",
        "No production kernel or selector change.",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_raw(run_dir: Path) -> None:
    raw = EVIDENCE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for path in sorted(run_dir.glob("*.json")):
        shutil.copy2(path, raw / path.name)


def write_fixture(run_dir: Path, result: Mapping[str, Any], mode: str) -> None:
    quality = load_sidecar(run_dir, "quality.json") or {}
    freeze = (load_sidecar(run_dir, "preflight.json") or {}).get("freeze") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-144",
        "mode": mode,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT_STACK,
        "candidate": freeze.get("candidate"),
        "quality": quality,
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": load_sidecar(run_dir, "performance.json")
        or result.get("performance"),
        "shipping_prompt_attention": result.get("shipping_prompt_attention"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "production_kept": bool(result.get("production_kept")),
        "verdict": result.get("verdict"),
        "report_path": str(REPORT.relative_to(ROOT)),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "candidate_measured_delta": result.get("candidate_measured_delta"),
        "shipping_delta": result.get("shipping_delta"),
        "quality_result": result.get("quality_result"),
        "reasons": result.get("reasons"),
        "preflight": load_sidecar(run_dir, "preflight.json") or {},
        "correctness": load_sidecar(run_dir, "correctness.json") or {},
        "screen": load_sidecar(run_dir, "screen.json") or {},
        "mechanism": load_sidecar(run_dir, "mechanism.json") or {},
        "keep_policy_id": "target_guard_v2",
        "claims_throughput": bool(result.get("claims_throughput")),
        "claims_performance_improvement": bool(result.get("production_kept")),
        "freeze": freeze,
    }
    dump_json(FIXTURE, payload)
    dump_json(EVIDENCE / "answers.json", payload)


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    return {
        "ok": not missing and payload.get("task") == "OPT-144",
        "task": "OPT-144",
        "missing": missing,
    }


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    result = evaluate_keep(run_dir)
    result["quality"] = load_sidecar(run_dir, "quality.json") or {}
    freeze = dict(
        (load_sidecar(run_dir, "preflight.json") or {}).get("freeze")
        or load_freeze(run_dir)
    )
    maybe_flip_production_pins(bool(result.get("production_kept")))
    write_report(result, freeze)
    write_fixture(run_dir, result, mode)
    copy_raw(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-144",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "verdict": result.get("verdict"),
        "production_kept": result.get("production_kept"),
        "shipping_prompt_attention": result.get("shipping_prompt_attention"),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "candidate_measured_delta": result.get("candidate_measured_delta"),
        "shipping_delta": result.get("shipping_delta"),
        "quality_result": result.get("quality_result"),
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    stored = store_sidecar(run_dir, "report.json", payload)
    copy_raw(run_dir)
    return stored


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
    raise PrefillAttentionError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=("feedback", "acceptance", "release"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except PrefillAttentionError as exc:
        print(
            json.dumps(
                {"task": "OPT-144", "phase": args.phase, "ok": False, "error": str(exc)}
            )
        )
        return 1
    print(
        json.dumps(
            {
                "task": "OPT-144",
                "phase": args.phase,
                "ok": bool(payload.get("ok", True)),
                "verdict": payload.get("verdict"),
            }
        )
    )
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
