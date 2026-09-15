"""OPT-145 remaining residual_norm_quant / prompt_mmq keep/reject."""

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
from tools.opt139_counter_identity import admit_supported_mechanism  # noqa: E402
from tools.opt141_matched_llama_counters import fused_boundary_ok  # noqa: E402
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

CONTRACT = ROOT / "pins/opt145_secondary_family_contract.json"
ITERATION = ROOT / "pins/opt145_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt145_secondary_family.json"
REPORT = ROOT / "evidence/optimization/opt145-secondary-family/REPORT.md"
EVIDENCE = REPORT.parent
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
GRAPH_PIN = ROOT / "cuda/execution_graph_path.cuh"
PROMPT_PIN = ROOT / "cuda/fattn_mma_f16.cuh"
RMS_PIN = ROOT / "cuda/rms_norm.cuh"
Q4_PIN = ROOT / "cuda/q4k_decode_path.cuh"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
OPT138_GAPS = ROOT / "evidence/optimization/opt138-remaining-gap/family-gaps.json"
OPT138_NEXT = ROOT / "evidence/optimization/opt138-remaining-gap/next-experiments.json"
OPT139_IDENTITY = ROOT / "evidence/optimization/opt139-counter-identity/identity.json"
OPT139_COUNTERS = (
    ROOT / "evidence/optimization/opt139-counter-identity/counter-evidence.json"
)
OPT141_COMPARE = (
    ROOT / "evidence/optimization/opt141-matched-llama-counters/comparisons.json"
)
OPT141_COUNTERS = (
    ROOT / "evidence/optimization/opt141-matched-llama-counters/counter-evidence.json"
)
OPT141_IDENTITY = (
    ROOT / "evidence/optimization/opt141-matched-llama-counters/identity.json"
)
OPT143_FIXTURE = ROOT / "fixtures/opt143_short_attention.json"
OPT144_FIXTURE = ROOT / "fixtures/opt144_prefill_attention.json"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT145_SECONDARY_FAMILY_RESULT="
COUNTS_PREFIX = "QW38_OPT145_NATIVE_COUNTS="
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
SHIPPING_PROMPT = "opt111_base"
SHIPPING_DECODE = "dense_bf16_tile_f16_mma_decode_v1"
SHIPPING_Q4 = "llama_q4k_mmvq"
SHIPPING_PROMPT_MMQ = "fma_async_x"
QUARTZ_RESIDUAL_KERNEL = "rms_norm_fp32_to_bf16_parallel"
QUARTZ_PROMPT_MMQ_KERNEL = "quant_mmq_mma_quality_kernel"
LLAMA_RESIDUAL_KERNEL = "quantize_q8_1"
LLAMA_PROMPT_MMQ_KERNEL = "mul_mat_q"
WARMUPS = 3
SCREEN_WARMUPS = 1
SCREEN_PAIRS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
CHILD_TIMEOUT_S = 300.0
LONG_TIMEOUT_S = 1200.0
MMA_THRESHOLD = 8192
CROSSOVER = 1024
DECODE_WORKLOADS = ("d128", "d2048", "d8192", "d32768")
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
    "selected_family",
    "unselected_family",
    "quality",
    "state_memory",
    "performance",
    "production_kept",
    "verdict",
    "report_path",
)
FORBIDDEN_MECHANISM_NEEDLES = (
    "opt110_report",
    "re-port opt-110",
    "port_opt110",
    "llama_q4k_mmvq_report",
    "opt112_norm_to_typed_q8",
    "residual_norm_to_typed_q8",
    "norm_to_typed_q8_rounded",
    "opt131_chain",
    "mixer_residual_add_to_ffn_norm",
    "opt122_packed_gdn",
    "opt130",
    "nparts8",
    "n_parts=8",
    "lower_opt137_mma_threshold",
    "mma_threshold_2048",
    "mma_at_d2048",
    "decode_mixer_stalls_as_rmsnorm",
    "entire_decode_mixer_replay",
)


class SecondaryFamilyError(RuntimeError):
    """OPT-145 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-145":
        raise SecondaryFamilyError("secondary-family contract task mismatch")
    validate_opt_in_contract(payload)
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-145", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    alias = {
        "inspect": "preflight",
        "identity": "correctness",
        "quality-long": "quality",
        "d128": "performance",
        "d2048": "performance",
        "d8192": "performance",
        "d32768": "performance",
        "p4096": "performance",
    }
    key = alias.get(family, family)
    workload = workload_for_mode(iteration["workloads"][key], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-145 mode={mode} phase={key} "
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
    rms = RMS_PIN.read_text(encoding="utf-8")
    q4 = Q4_PIN.read_text(encoding="utf-8")
    return (
        'kSelectedExecutionGraphPath[] = "decode_segments8"' in graphs
        and 'kSelectedAttentionPipelinePath[] = "opt111_base"' in prompt
        and "kSelectedDecodeAttentionCrossoverThreshold = 1024" in decode
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in decode
        and "kSelectedVec128NParts = 16" in decode
        and "kSelectedOpt137DenseMma = true" in decode
        and "kOpt137MmaThreshold = 8192" in decode
        and 'kSelectedRmsNormPath[] = "parallel_fma"' in rms
        and "llama_q4k_mmvq" in q4
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


def _blob(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).casefold()


def attention_not_kept() -> dict[str, Any]:
    opt143 = _load_optional(OPT143_FIXTURE)
    opt144 = _load_optional(OPT144_FIXTURE)
    return {
        "opt143_verdict": opt143.get("verdict"),
        "opt144_verdict": opt144.get("verdict"),
        "opt143_not_kept": opt143.get("verdict")
        in {
            "no_opportunity",
            "reject",
            "screened_out",
            None,
        }
        and not bool(opt143.get("production_kept")),
        "opt144_not_kept": opt144.get("verdict")
        in {
            "no_opportunity",
            "reject",
            "screened_out",
            None,
        }
        and not bool(opt144.get("production_kept")),
    }


def decode_family_row(gaps: Mapping[str, Any], family: str) -> dict[str, Any]:
    selected = ((gaps.get("decode_selection") or {}).get("selected")) or []
    ranked = ((gaps.get("decode_selection") or {}).get("ranked")) or []
    for row in list(selected) + list(ranked):
        if isinstance(row, Mapping) and str(row.get("family") or "") == family:
            return dict(row)
    return {}


def prefill_family_row(gaps: Mapping[str, Any], family: str) -> dict[str, Any]:
    selected = ((gaps.get("prefill_selection") or {}).get("selected")) or []
    ranked = ((gaps.get("prefill_selection") or {}).get("ranked")) or []
    for row in list(selected) + list(ranked):
        if isinstance(row, Mapping) and str(row.get("family") or "") == family:
            return dict(row)
    return {}


def regrouped_family(
    gaps: Mapping[str, Any], window: str, family: str
) -> dict[str, Any]:
    families = ((gaps.get(window) or {}).get("regrouped") or {}).get("families") or []
    for row in families:
        if isinstance(row, Mapping) and str(row.get("family") or "") == family:
            return dict(row)
    return {}


def window_ms(gaps: Mapping[str, Any], window: str) -> float | None:
    value = (gaps.get(window) or {}).get("quartz_window_ms")
    if value is None:
        recon = (gaps.get(window) or {}).get("reconciliation") or {}
        value = recon.get("quartz_wall_ms")
    return float(value) if value is not None else None


def excess_fraction(
    excess_ms: float | None, enclosing_ms: float | None
) -> float | None:
    if excess_ms is None or enclosing_ms in (None, 0, 0.0):
        return None
    return float(excess_ms) / float(enclosing_ms)


def residual_norm_quant_evidence(gaps: Mapping[str, Any]) -> dict[str, Any]:
    row = decode_family_row(gaps, "residual_norm_quant")
    d128 = _mapping(row.get("d128_middle"))
    d2048 = _mapping(row.get("d2048_middle"))
    d128_window = window_ms(gaps, "decode_d128_middle")
    d2048_window = window_ms(gaps, "decode_d2048_middle")
    d128_ms = d128.get("mean_ms")
    d2048_ms = d2048.get("mean_ms")
    d128_frac = excess_fraction(d128_ms, d128_window)
    d2048_frac = excess_fraction(d2048_ms, d2048_window)
    windows = [
        {
            "workload": "d128",
            "kind": "decode",
            "source": "middle_window",
            "excess_ms": d128_ms,
            "one_sided_low_ms": d128.get("one_sided_low"),
            "n": d128.get("n"),
            "window_ms": d128_window,
            "fraction": d128_frac,
            "significant_positive_excess": bool(
                d128.get("significant_positive_excess")
            ),
            "note": "twelve middle-window evals, not per eval",
        },
        {
            "workload": "d2048",
            "kind": "decode",
            "source": "middle_window",
            "excess_ms": d2048_ms,
            "one_sided_low_ms": d2048.get("one_sided_low"),
            "n": d2048.get("n"),
            "window_ms": d2048_window,
            "fraction": d2048_frac,
            "significant_positive_excess": bool(
                d2048.get("significant_positive_excess")
            ),
            "note": "twelve middle-window evals, not per eval",
        },
    ]
    supported = [
        item
        for item in windows
        if item["significant_positive_excess"] and item["fraction"] is not None
    ]
    supported.sort(key=lambda item: (-float(item["fraction"]), str(item["workload"])))
    best = supported[0] if supported else None
    regroup_d128 = regrouped_family(gaps, "decode_d128_middle", "residual_norm_quant")
    regroup_d2048 = regrouped_family(gaps, "decode_d2048_middle", "residual_norm_quant")
    return {
        "family": "residual_norm_quant",
        "phase": "decode",
        "windows": windows,
        "supported_fraction": None if best is None else float(best["fraction"]),
        "supported_excess_ms": None if best is None else best["excess_ms"],
        "supported_workload": None if best is None else best["workload"],
        "supported": bool(supported),
        "fused_members": [
            "residual_add",
            "rms_norm",
            "bf16_materialize",
            "typed_quant",
            "downstream_projections",
        ],
        "quartz_kernel": QUARTZ_RESIDUAL_KERNEL,
        "llama_enclosing": ["rms_norm_f32", "l2_norm_f32", "quantize_q8_1"],
        "replay_family": "decode-mixer",
        "decode_mixer_stalls_identify_rmsnorm": False,
        "regrouped_d128": regroup_d128,
        "regrouped_d2048": regroup_d2048,
        "coverage_status": regroup_d128.get("coverage_status")
        or regroup_d2048.get("coverage_status"),
    }


def prompt_mmq_evidence(gaps: Mapping[str, Any]) -> dict[str, Any]:
    row = prefill_family_row(gaps, "prompt_mmq")
    stats = _mapping(row.get("stats"))
    if not stats and row.get("mean_ms") is not None:
        stats = {
            "mean_ms": row.get("mean_ms"),
            "one_sided_low": row.get("one_sided_low"),
            "n": row.get("n"),
            "significant_positive_excess": row.get("significant_positive_excess"),
        }
    enclosing = window_ms(gaps, "prefill_p4096")
    excess_ms = stats.get("mean_ms") if stats else row.get("score_ms")
    fraction = excess_fraction(excess_ms, enclosing)
    positive = bool(stats.get("significant_positive_excess"))
    regroup = regrouped_family(gaps, "prefill_p4096", "prompt_mmq")
    return {
        "family": "prompt_mmq",
        "phase": "prefill",
        "windows": [
            {
                "workload": "p4096",
                "kind": "prefill",
                "source": row.get("source") or "p4096_complete",
                "excess_ms": excess_ms,
                "one_sided_low_ms": stats.get("one_sided_low"),
                "n": stats.get("n"),
                "window_ms": enclosing,
                "fraction": fraction,
                "significant_positive_excess": positive,
                "note": "one complete P4096 prompt, not ranked against twelve decode evals by raw ms",
            }
        ],
        "supported_fraction": fraction if positive else None,
        "supported_excess_ms": excess_ms if positive else None,
        "supported_workload": "p4096" if positive else None,
        "supported": bool(positive and fraction is not None),
        "fused_members": [
            "quantize_mmq_q8_1",
            QUARTZ_PROMPT_MMQ_KERNEL,
            "mul_mat_q_stream_k_fixup",
            "staging",
            "output",
        ],
        "quartz_kernel": QUARTZ_PROMPT_MMQ_KERNEL,
        "llama_enclosing": ["quantize_mmq_q8_1", "mul_mat_q_stream_k_fixup"],
        "replay_family": "prompt-ffn",
        "other_prompt_ffn_excluded": True,
        "regrouped": regroup,
        "coverage_status": regroup.get("coverage_status"),
    }


def select_secondary_family(
    residual: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> dict[str, Any]:
    """Largest supported excess fraction; tie-break by family name."""
    rows = [dict(residual), dict(prompt)]
    supported = [
        row
        for row in rows
        if row.get("supported") and row.get("supported_fraction") is not None
    ]
    supported.sort(
        key=lambda row: (-float(row["supported_fraction"]), str(row["family"]))
    )
    selected = supported[0] if supported else None
    unselected = None
    if selected is not None:
        other = residual if selected.get("family") == "prompt_mmq" else prompt
        unselected = dict(other)
        unselected["unselected_reason"] = (
            "smaller_supported_excess_fraction"
            if other.get("supported")
            else "not_supported_positive_fraction"
        )
    return {
        "selection_rule": "largest_supported_excess_fraction",
        "tie_break": "family_name",
        "ranked_raw_ms_rejected": True,
        "selected": selected,
        "unselected": unselected,
        "candidates": rows,
    }


def target_guard_roles(selected: Mapping[str, Any] | None) -> dict[str, Any]:
    if not selected:
        return {
            "target_workloads": [],
            "targets": [],
            "guards": [],
            "target_metric": None,
        }
    family = str(selected.get("family") or "")
    if family == "prompt_mmq":
        return {
            "target_workloads": ["p4096"],
            "targets": [
                {"workload": "p4096", "kind": "prefill", "primary_metric": "prefill"}
            ],
            "guards": [
                {"workload": name, "kind": "decode"} for name in DECODE_WORKLOADS
            ],
            "target_metric": "prefill",
            "complete_request_target_guard": False,
        }
    winning = str(selected.get("supported_workload") or "d128")
    guards = [name for name in DECODE_WORKLOADS if name != winning]
    return {
        "target_workloads": [winning],
        "targets": [
            {"workload": winning, "kind": "decode", "primary_metric": "decode_only"}
        ],
        "guards": [{"workload": name, "kind": "decode"} for name in guards]
        + [{"workload": "p4096", "kind": "prefill"}],
        "target_metric": "decode_only",
        "complete_request_target_guard": True,
    }


def forbidden_reason(
    candidate: Any, mechanism: Any, dispatch_region: Any = None
) -> str | None:
    text = _blob(candidate, mechanism, dispatch_region)
    if any(
        needle in text
        for needle in (
            "opt110",
            "port_opt110",
            "llama_q4k_mmvq_report",
            "re-port opt-110",
        )
    ):
        return "opt110_not_re_ported"
    if any(
        needle in text
        for needle in (
            "opt112",
            "norm_to_typed_q8",
            "residual_norm_to_typed_q8",
        )
    ):
        return "opt112_deferred_fusion_not_revived"
    if "opt131" in text or "mixer_residual_add_to_ffn_norm" in text:
        return "opt131_chain_fusion_not_revived"
    if "opt122" in text or "packed_gdn" in text:
        return "opt122_packed_gdn_out_of_family"
    if "decode_mixer_stalls" in text or "entire_decode_mixer_replay" in text:
        return "decode_mixer_stalls_cannot_identify_rmsnorm"
    if any(needle in text for needle in FORBIDDEN_MECHANISM_NEEDLES):
        if "opt130" in text or "nparts8" in text or "n_parts=8" in text:
            return "opt130_not_revived"
        if "mma" in text:
            return "opt137_mma_threshold_must_remain_8192"
        return "forbidden_secondary_family_mechanism"
    return None


def kernel_record(
    counters: Mapping[str, Any],
    *,
    family: str,
    workload: str,
    engine: str = "quartz",
) -> dict[str, Any]:
    for row in counters.get("kernels") or counters.get("records") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("family") or "") != family:
            continue
        if str(row.get("engine") or engine) != engine:
            continue
        if str(row.get("workload") or "") != workload:
            continue
        return dict(row)
    return {}


def resolving_measurement(
    selected: Mapping[str, Any],
    *,
    comparisons: Mapping[str, Any],
    identity: Mapping[str, Any],
    llama_counters: Mapping[str, Any],
) -> dict[str, Any]:
    """One bounded fused-boundary check; does not invent a kernel."""
    family = str(selected.get("family") or "")
    if family == "residual_norm_quant":
        key = "decode.residual_norm_quant.d128"
        compare = _mapping(comparisons.get(key)) or _mapping(
            comparisons.get("decode.residual_norm_quant.d2048")
        )
        fusion = fused_boundary_ok(
            family="residual_norm_quant",
            quartz_kernel=QUARTZ_RESIDUAL_KERNEL,
            llama_kernel=LLAMA_RESIDUAL_KERNEL,
            llama_enclosing=["rms_norm_f32", "quantize_q8_1"],
        )
        identity_row = _mapping((identity.get("rows") or {}).get(key)) or _mapping(
            identity.get(key)
        )
        return {
            "ok": True,
            "family": family,
            "kind": "fused_boundary_identity",
            "quartz_kernel": QUARTZ_RESIDUAL_KERNEL,
            "llama_kernel": LLAMA_RESIDUAL_KERNEL,
            "llama_enclosing": ["rms_norm_f32", "l2_norm_f32", "quantize_q8_1"],
            "fusion": fusion,
            "comparison": compare.get("comparison") or {},
            "identity_status": compare.get("identity_status")
            or identity_row.get("status"),
            "decode_mixer_stalls_identify_rmsnorm": False,
            "named_source_sass": False,
            "proof_limit": (
                "fused residual+rms+bf16+quant vs llama rms_norm_f32+quantize_q8_1; "
                "leaf counters incomparable; decode-mixer replay stalls are not an "
                "RMSNorm bottleneck"
            ),
        }
    key = "prefill.prompt_mmq.p4096"
    compare = _mapping(comparisons.get(key))
    fusion = fused_boundary_ok(
        family="prompt_mmq",
        quartz_kernel=QUARTZ_PROMPT_MMQ_KERNEL,
        llama_kernel=LLAMA_PROMPT_MMQ_KERNEL,
        llama_enclosing=["quantize_mmq_q8_1", "mul_mat_q_stream_k_fixup"],
    )
    llama = kernel_record(
        llama_counters, family="prompt_mmq", workload="P4096", engine="llama"
    )
    return {
        "ok": True,
        "family": family,
        "kind": "prompt_mmq_kernel_separation",
        "quartz_kernel": QUARTZ_PROMPT_MMQ_KERNEL,
        "llama_kernel": LLAMA_PROMPT_MMQ_KERNEL,
        "llama_enclosing": ["quantize_mmq_q8_1", "mul_mat_q_stream_k_fixup"],
        "fusion": fusion,
        "comparison": compare.get("comparison") or {},
        "other_prompt_ffn_excluded": True,
        "quant_staging_output_charged": True,
        "named_source_sass": False,
        "llama_kernel_record": {
            "kernel": llama.get("kernel") or llama.get("target_kernel"),
            "supported_mechanism": llama.get("supported_mechanism"),
        },
        "proof_limit": (
            "complete prompt_mmq family is quant_mmq_mma_quality_kernel plus "
            "quantization/staging/output; other prompt-ffn kernels are excluded"
        ),
    }


def freeze_candidate(
    *,
    gaps: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
    counters: Mapping[str, Any] | None = None,
    comparisons: Mapping[str, Any] | None = None,
    llama_counters: Mapping[str, Any] | None = None,
    next_experiments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one family by fraction, then freeze a candidate or no_opportunity."""
    blockers: list[str] = []
    reasons: list[str] = []
    missing = [
        name
        for name, present in (
            ("opt138_family_gaps", gaps is not None or OPT138_GAPS.is_file()),
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
            "selected_family": None,
            "unselected_family": None,
            "blockers": missing,
            "reasons": list(missing),
            "positive_excess": False,
            "arithmetic_changed": False,
        }

    gaps_row = _mapping(gaps) or _load_optional(OPT138_GAPS)
    identity_row = _mapping(identity) or _load_optional(OPT141_IDENTITY)
    if not identity_row:
        identity_row = _load_optional(OPT139_IDENTITY)
    counters_row = _mapping(counters) or _load_optional(OPT139_COUNTERS)
    compare_row = _mapping(comparisons) or _load_optional(OPT141_COMPARE)
    llama_row = _mapping(llama_counters) or _load_optional(OPT141_COUNTERS)
    next_row = _mapping(next_experiments) or _load_optional(OPT138_NEXT)
    attention = attention_not_kept()

    residual = residual_norm_quant_evidence(gaps_row)
    prompt = prompt_mmq_evidence(gaps_row)
    selection = select_secondary_family(residual, prompt)
    selected = selection.get("selected")
    unselected = selection.get("unselected")
    if selected is None:
        blockers.append("no_supported_positive_excess_fraction")

    roles = target_guard_roles(selected)
    family = str((selected or {}).get("family") or "")
    workload = str((selected or {}).get("supported_workload") or "")
    workload_key = {
        "d128": "D128",
        "d2048": "D2048",
        "p4096": "P4096",
    }.get(workload, workload.upper())

    quartz = kernel_record(
        counters_row, family=family, workload=workload_key, engine="quartz"
    )
    if family == "residual_norm_quant" and not quartz:
        quartz = kernel_record(
            llama_row, family=family, workload=workload_key, engine="quartz"
        )
    admission = (
        admit_supported_mechanism(quartz)
        if quartz
        else {
            "ok": False,
            "supported_mechanism": None,
            "candidate": None,
            "reason": "quartz_leaf_record_missing_or_unusable",
        }
    )
    source = quartz.get("source_observations") or quartz.get("named_source")
    sass = quartz.get("sass_observations") or quartz.get("sass")
    mechanism = admission.get("supported_mechanism")
    proposed = quartz.get("candidate") or admission.get("candidate")
    forbidden = forbidden_reason(proposed, mechanism, quartz.get("dispatch"))
    if forbidden:
        blockers.append(forbidden)

    compare_key = {
        "residual_norm_quant": f"decode.residual_norm_quant.{workload}",
        "prompt_mmq": "prefill.prompt_mmq.p4096",
    }.get(family)
    compare = _mapping(compare_row.get(compare_key or ""))
    if not compare and family == "residual_norm_quant":
        compare = _mapping(compare_row.get("decode.residual_norm_quant.d128"))
    comparison = _mapping(compare.get("comparison"))
    fusion = _mapping(compare.get("fusion"))
    comparable = bool(comparison.get("comparable"))
    fusion_reason = str(comparison.get("reason") or fusion.get("reason") or "")

    resolving = (
        resolving_measurement(
            selected,
            comparisons=compare_row,
            identity=identity_row,
            llama_counters=llama_row,
        )
        if selected
        else {}
    )

    labeled = None
    if family == "residual_norm_quant":
        labeled = ((next_row.get("decode") or {}).get("expected_benefit") or {}).get(
            "label"
        )
    elif family == "prompt_mmq":
        labeled = ((next_row.get("prefill") or {}).get("expected_benefit") or {}).get(
            "label"
        )

    no_opportunity = False
    candidate = None
    if blockers:
        verdict = "blocked"
    elif not selected or not selected.get("supported"):
        no_opportunity = True
        verdict = "no_opportunity"
        reasons.append("no_supported_positive_excess_fraction")
    elif not (source and sass and admission.get("ok") and mechanism and proposed):
        no_opportunity = True
        verdict = "no_opportunity"
        reasons.append(str(admission.get("reason") or "no_source_grounded_mechanism"))
        reasons.append("throughput_occupancy_dram_alone_cannot_freeze_candidate")
        if family == "residual_norm_quant":
            reasons.append("fused_boundary_mismatch_quartz_rms_plus_bf16_quant")
            reasons.append("decode_mixer_stalls_cannot_identify_rmsnorm")
            reasons.append("opt112_deferred_fusion_not_revived")
            reasons.append("opt131_chain_fusion_not_revived")
        if family == "prompt_mmq":
            reasons.append(
                "quant_mmq_mma_quality_kernel_separated_from_other_prompt_ffn"
            )
            reasons.append("opt110_decode_q4_adapter_not_a_prompt_mmq_candidate")
        if labeled == "unknown_until_mechanism":
            reasons.append("opt138_expected_benefit_unknown_until_mechanism")
        if not comparable:
            reasons.append(fusion_reason or "leaf_counters_incomparable")
        reasons.append("opt110_not_re_ported")
    else:
        candidate = str(proposed)
        verdict = "candidate_frozen"
        reasons.append("source_grounded_mechanism_admitted")

    selected_out = None if selected is None else dict(selected)
    unselected_out = None if unselected is None else dict(unselected)
    return {
        "ok": not blockers,
        "verdict": verdict,
        "no_opportunity": no_opportunity,
        "candidate": candidate,
        "supported_mechanism": mechanism,
        "admission_reason": admission.get("reason"),
        "blockers": blockers,
        "reasons": reasons,
        "selected_family": family or None,
        "unselected_family": None
        if unselected_out is None
        else unselected_out.get("family"),
        "selection": selection,
        "selected": selected_out,
        "unselected": unselected_out,
        "positive_excess": bool(selected and selected.get("supported")),
        "excess_ms": None if selected is None else selected.get("supported_excess_ms"),
        "excess_fraction": None
        if selected is None
        else selected.get("supported_fraction"),
        "excess_note": (
            "excess is a fraction of the matched whole-workload interval; "
            "twelve decode evals are not ranked against one P4096 prompt by raw ms"
        ),
        "roles": roles,
        "target": (roles.get("target_workloads") or [None])[0],
        "target_metric": roles.get("target_metric"),
        "resolving_measurement": resolving,
        "source_observations": bool(source),
        "sass_observations": bool(sass),
        "comparable": comparable,
        "fusion_reason": fusion_reason or None,
        "quartz_kernel": None if selected is None else selected.get("quartz_kernel"),
        "replay_family": None if selected is None else selected.get("replay_family"),
        "opt137_mma_threshold": MMA_THRESHOLD,
        "opt137_mma_retained": True,
        "opt130_not_revived": True,
        "opt110_not_re_ported": True,
        "opt112_deferred_fusion_not_revived": True,
        "opt131_chain_fusion_not_revived": True,
        "opt122_out_of_family": True,
        "opt143_short_decode_not_kept": bool(attention.get("opt143_not_kept")),
        "opt144_prefill_attention_not_kept": bool(attention.get("opt144_not_kept")),
        "attention": attention,
        "arithmetic_changed": bool(candidate),
        "nll_required": bool(candidate),
        "refresh_reason": "production_unchanged_after_opt143_opt144_no_opportunity",
        "evidence_paths": {
            "family_gaps": workspace_relative(OPT138_GAPS),
            "identity": workspace_relative(OPT141_IDENTITY),
            "counters": workspace_relative(OPT139_COUNTERS),
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
        "task": "OPT-145",
        "phase": phase,
        "mode": mode,
        "ok": True,
        "skipped": True,
        "skip_reason": "no_candidate_no_arithmetic_change"
        if freeze.get("no_opportunity")
        else "blocked_no_candidate",
        "candidate": freeze.get("candidate"),
        "selected_family": freeze.get("selected_family"),
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
        SHIPPING_PROMPT,
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
        raise SecondaryFamilyError(
            f"OPT-058 quality failed for {config_id}: {completed.stderr[-2000:]}"
        )
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    if not restored:
        raise SecondaryFamilyError("--quality restored packed or r2 defaults")
    if not cases:
        raise SecondaryFamilyError(f"OPT-058 produced no NLL cases for {config_id}")
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
    dump_json(EVIDENCE / "selection.json", freeze.get("selection") or {})
    auth = authenticate_post113()
    pins = authenticate_opt138_pins()
    source, dirty = git_identity()
    available, gpu_blocker = gpu_available()
    residents = gpu_residents() if available else []
    make_rc = 1
    make_stderr = gpu_blocker or "gpu_unavailable"
    if available:
        make = with_gpu_lock(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt145-diagnostics"],
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
        "task": "OPT-145",
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
        "selected_family": freeze.get("selected_family"),
        "unselected_family": freeze.get("unselected_family"),
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
    extra = {
        "identity_checks": list(load_contract().get("identity_checks") or []),
        "selected_family": freeze.get("selected_family"),
        "quartz_kernel": freeze.get("quartz_kernel"),
        "replay_family": freeze.get("replay_family"),
        "resolving_measurement": freeze.get("resolving_measurement"),
        "decode_mixer_stalls_identify_rmsnorm": False,
        "opt110_not_re_ported": True,
        "opt112_deferred_fusion_not_revived": True,
        "opt131_chain_fusion_not_revived": True,
    }
    if not freeze.get("candidate"):
        return skip_payload(run_dir, mode, "correctness", freeze, extra=extra)
    raise SecondaryFamilyError(
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
                "complete_family": True,
                "cache_mode": "rotating",
                "target": freeze.get("target"),
                "selected_family": freeze.get("selected_family"),
            },
        )
    raise SecondaryFamilyError(
        "frozen candidate requires complete-family screen that was not implemented"
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
    control = run_quality_native(run_dir, "parent")
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
        "task": "OPT-145",
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
    raise SecondaryFamilyError(
        "frozen candidate requires state/graph/checkpoint/128K gates"
    )


def run_performance(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "performance"), flush=True)
    freeze = load_freeze(run_dir)
    roles = freeze.get("roles") or {}
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
                "target": f"{freeze.get('target')}.{freeze.get('target_metric')}",
                "guards": [
                    f"{row.get('workload')}.decode_only"
                    if row.get("kind") == "decode"
                    else f"{row.get('workload')}.prefill"
                    for row in (roles.get("guards") or [])
                ],
            },
        )
    raise SecondaryFamilyError(
        "frozen candidate requires target_guard_v2 AB/BA that was not implemented"
    )


def run_mechanism(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "mechanism"), flush=True)
    freeze = load_freeze(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-145",
        "phase": "mechanism",
        "mode": mode,
        "ok": bool(freeze.get("ok")),
        "skipped": not bool(freeze.get("candidate")),
        "supported_mechanism": freeze.get("supported_mechanism"),
        "candidate": freeze.get("candidate"),
        "selected_family": freeze.get("selected_family"),
        "admission_reason": freeze.get("admission_reason"),
        "source_observations": freeze.get("source_observations"),
        "sass_observations": freeze.get("sass_observations"),
        "resolving_measurement": freeze.get("resolving_measurement"),
        "opt110_not_re_ported": True,
        "opt112_deferred_fusion_not_revived": True,
        "opt131_chain_fusion_not_revived": True,
        "opt137_mma_threshold": MMA_THRESHOLD,
        "opt130_not_revived": True,
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
            "shipping_q4_decode": SHIPPING_Q4,
            "shipping_prompt_mmq": SHIPPING_PROMPT_MMQ,
            "reasons": list(freeze.get("blockers") or []),
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": None,
            "shipping_delta": 0,
            "quality_result": "not_required",
            "selected_family": freeze.get("selected_family"),
            "unselected_family": freeze.get("unselected_family"),
        }
    if freeze.get("no_opportunity") or not freeze.get("candidate"):
        return {
            "verdict": "no_opportunity",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_q4_decode": SHIPPING_Q4,
            "shipping_prompt_mmq": SHIPPING_PROMPT_MMQ,
            "reasons": list(freeze.get("reasons") or ["no_source_grounded_mechanism"]),
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": None,
            "shipping_delta": 0,
            "quality_result": "nll_not_required_no_arithmetic_change",
            "freeze": freeze,
            "selected_family": freeze.get("selected_family"),
            "unselected_family": freeze.get("unselected_family"),
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
            "selected_family": freeze.get("selected_family"),
            "unselected_family": freeze.get("unselected_family"),
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
        reasons.append("mechanism_failed")
        if verdict == "keep":
            verdict = "reject"
    reasons.extend(str(item) for item in (policy.get("reasons") or []))
    keep = verdict == "keep"
    return {
        "verdict": verdict,
        "production_kept": keep,
        "shipping_prompt_attention": SHIPPING_PROMPT,
        "shipping_decode_attention": SHIPPING_DECODE,
        "shipping_q4_decode": SHIPPING_Q4,
        "shipping_prompt_mmq": freeze.get("candidate") if keep else SHIPPING_PROMPT_MMQ,
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
        "selected_family": freeze.get("selected_family"),
        "unselected_family": freeze.get("unselected_family"),
    }


def maybe_flip_production_pins(keep: bool) -> None:
    if not keep:
        return
    raise SecondaryFamilyError("keep would require an explicit production pin flip")


def write_report(result: Mapping[str, Any], freeze: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = str(result.get("verdict"))
    quality = result.get("quality") or {}
    selected = freeze.get("selected") or {}
    unselected = freeze.get("unselected") or {}
    resolving = freeze.get("resolving_measurement") or {}
    windows = selected.get("windows") or []
    unsel_windows = unselected.get("windows") or []
    window_lines = [
        f"| `{row.get('workload')}` | `{row.get('kind')}` | `{row.get('excess_ms')}` | "
        f"`{row.get('window_ms')}` | `{row.get('fraction')}` | "
        f"`{row.get('significant_positive_excess')}` |"
        for row in windows
        if isinstance(row, Mapping)
    ]
    unsel_lines = [
        f"| `{row.get('workload')}` | `{row.get('kind')}` | `{row.get('excess_ms')}` | "
        f"`{row.get('window_ms')}` | `{row.get('fraction')}` | "
        f"`{row.get('significant_positive_excess')}` |"
        for row in unsel_windows
        if isinstance(row, Mapping)
    ]
    lines = [
        "# OPT-145 — Remaining residual_norm_quant or prompt_mmq family",
        "",
        f"Status: **{verdict}**. Parent retained on no-keep. Shipping Q4 decode "
        f"`{result.get('shipping_q4_decode')}`; prompt MMQ "
        f"`{result.get('shipping_prompt_mmq')}`; prompt attention "
        f"`{result.get('shipping_prompt_attention')}`.",
        "",
        f"Parent `{PARENT_STACK}`. OPT-143/144 attention work did not keep a "
        "candidate, so residual_norm_quant decode and prompt_mmq prefill evidence "
        "was refreshed against the same production objects.",
        "",
        "## Family selection",
        "",
        "Rank by **excess / matched whole-workload interval**, not raw milliseconds. "
        "Twelve decode evals are not compared with one P4096 prompt by ms.",
        "",
        f"Selected family: `{freeze.get('selected_family')}`. "
        f"Supported fraction: `{freeze.get('excess_fraction')}`. "
        f"Excess ms: `{freeze.get('excess_ms')}`. "
        f"Winning workload: `{freeze.get('target')}` "
        f"(`{freeze.get('target_metric')}`).",
        "",
        "| Workload | Kind | Excess ms | Window ms | Fraction | Positive |",
        "|---|---|---|---|---|---|",
        *(window_lines or ["| _(none)_ | | | | | |"]),
        "",
        f"Unselected family: `{freeze.get('unselected_family')}`. "
        f"Unselected reason: `{unselected.get('unselected_reason')}`. "
        "Recorded for future prioritization; no automatic second experiment.",
        "",
        "| Workload | Kind | Excess ms | Window ms | Fraction | Positive |",
        "|---|---|---|---|---|---|",
        *(unsel_lines or ["| _(none)_ | | | | | |"]),
        "",
        "## Frozen candidate",
        "",
        f"Candidate: `{freeze.get('candidate')}`.",
        f"Supported mechanism: `{freeze.get('supported_mechanism')}`.",
        f"Admission reason: `{freeze.get('admission_reason')}`.",
        f"Reasons: `{result.get('reasons') or freeze.get('reasons')}`.",
        "",
        "Freeze requires matched positive family excess **and** a source-grounded "
        "mechanism at the fused comparison boundary. Occupancy, DRAM, and byte "
        "counters alone cannot freeze a kernel. Decode-mixer replay stalls cannot "
        "identify an RMSNorm bottleneck. OPT-110/112/131/122/130 are not revived.",
        "",
        "## Resolving measurement",
        "",
        f"Kind `{resolving.get('kind')}`. Quartz kernel `{resolving.get('quartz_kernel')}`. "
        f"Llama kernel `{resolving.get('llama_kernel')}`. Fusion ok "
        f"`{(resolving.get('fusion') or {}).get('ok')}` reason "
        f"`{(resolving.get('fusion') or {}).get('reason')}`. "
        f"Named source/SASS `{resolving.get('named_source_sass')}`.",
        "",
        f"{resolving.get('proof_limit')}",
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
        f"`target_guard_v2` target `{freeze.get('target')}.{freeze.get('target_metric')}`. "
        f"Roles `{freeze.get('roles')}`.",
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
        "",
        "## Performance evidence checklist",
        "",
        "1. **Measurement identity** — Quartz production `decode_segments8` + kept "
        f"OPT-137 MMA; shipping Q4 `{SHIPPING_Q4}`; RMS `{QUARTZ_RESIDUAL_KERNEL}` / "
        f"`parallel_fma`; prompt MMQ `{QUARTZ_PROMPT_MMQ_KERNEL}` / `{SHIPPING_PROMPT_MMQ}`; "
        f"GGUF `{GGUF_SHA}`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`. "
        "OPT-143/144 did not change production, so OPT-138 family windows remain the "
        "current sitting.",
        "2. **Coverage** — OPT-138 matched family excess on decode middle windows and "
        "P4096 complete; OPT-141 fused-boundary rejection for residual_norm_quant; "
        "prompt_mmq fusion ok with quartz leaf counters unusable.",
        "3. **Time accounting** — family excess is a fraction of the matched "
        "whole-workload interval (`quartz_window_ms`); overlapping graph parents are "
        "not summed with children.",
        "4. **Contradiction register** — raw ms would prefer prompt_mmq (+27.11 vs "
        "+7.21); fraction ranking prefers residual_norm_quant. Decode-mixer stalls "
        "are not treated as an RMSNorm diagnosis. OPT-112 deferred fusion is not a "
        "freeze.",
        "5. **Claim types** — family excess `measured`; fraction ranking `derived`; "
        "mechanism `unknown`/`incomplete`; no_opportunity `measured` from absent "
        "source/SASS after one fused-boundary resolving check.",
        "6. **Target/guard** — OPT-135 `target_guard_v2` opted in; unused for keep "
        "because no candidate ran AB/BA.",
        "7. **Independent verification** — pending verifier pass on this draft.",
        "8. **Reporting** — candidate measured delta N/A; shipping delta 0; "
        "quality N/A (no arithmetic change). Unselected family evidence retained.",
        "",
        "## Raw gates",
        "",
        "Sidecars: [`raw/`](raw/) (`preflight.json`, `freeze.json`, phase skips, "
        "`report.json`). Structured freeze: [`freeze.json`](freeze.json). "
        "Selection: [`selection.json`](selection.json). Fixture dump: "
        "[`answers.json`](answers.json) and `fixtures/opt145_secondary_family.json`.",
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
        "task": "OPT-145",
        "mode": mode,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT_STACK,
        "candidate": freeze.get("candidate"),
        "selected_family": freeze.get("selected_family")
        or result.get("selected_family"),
        "unselected_family": freeze.get("unselected_family")
        or result.get("unselected_family"),
        "quality": quality,
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": load_sidecar(run_dir, "performance.json")
        or result.get("performance"),
        "shipping_prompt_attention": result.get("shipping_prompt_attention"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "shipping_q4_decode": result.get("shipping_q4_decode"),
        "shipping_prompt_mmq": result.get("shipping_prompt_mmq"),
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
        "ok": not missing and payload.get("task") == "OPT-145",
        "task": "OPT-145",
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
        "task": "OPT-145",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "verdict": result.get("verdict"),
        "production_kept": result.get("production_kept"),
        "selected_family": result.get("selected_family"),
        "unselected_family": result.get("unselected_family"),
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
    raise SecondaryFamilyError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=("feedback", "acceptance", "release"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except SecondaryFamilyError as exc:
        print(
            json.dumps(
                {"task": "OPT-145", "phase": args.phase, "ok": False, "error": str(exc)}
            )
        )
        return 1
    print(
        json.dumps(
            {
                "task": "OPT-145",
                "phase": args.phase,
                "ok": bool(payload.get("ok", True)),
                "verdict": payload.get("verdict"),
                "selected_family": payload.get("selected_family"),
            }
        )
    )
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
