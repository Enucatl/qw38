"""OPT-124 practical ceiling analysis for the pinned model and sitting.

Analysis/diagnostics only. No production selector, kernel, or throughput
claim. Reuses OPT-123 samples only when identities match. Pinned llama
cc83d7b4824f73cfdda4dfbb47ee39804f71b328 remains the benchmark authority.
Does not assert universal optimality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    LLAMA_REV,
    docker_common,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt115_pipeline_traffic import (  # noqa: E402
    PEAK_BANDWIDTH_GB_S,
    authenticate_post113,
    bandwidth_bounds,
    bytes_to_peak_ms,
    classify_timeline,
    compulsory_decode_bytes,
    compulsory_prefill_bytes,
    inspect_repo_revision,
    inventory_bytes,
    kv_bytes,
    parse_prefixed,
    parse_records_blob,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt124_practical_ceiling_contract.json"
ITERATION = ROOT / "pins/opt124_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt124_practical_ceiling.json"
REPORT = ROOT / "evidence/optimization/opt124-practical-ceiling/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
OPT123_FIXTURE = ROOT / "fixtures/opt123_combined_stack.json"
NATIVE = "build/qw38-cuda-opt124-practical-ceiling-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT124_PRACTICAL_CEILING_RESULT="
COUNTS_PREFIX = "QW38_OPT124_NATIVE_COUNTS="
RECORDS_PREFIX = "QW38_OPT124_RECORDS="
PHASES = (
    "freeze",
    "bounds",
    "llama-compare",
    "mechanisms",
    "identity-timeline",
    "conclusions",
    "report",
)
WORKLOADS = (
    "p4096",
    "d128",
    "d2048",
    "d8192",
    "d32768",
    "d131040",
    "opt016_2k",
)
DECODE_WORKLOADS = ("d128", "d2048", "d8192", "d32768", "d131040")
PREFIX = {
    "p4096": 4096,
    "d128": 128,
    "d2048": 2048,
    "d8192": 8192,
    "d32768": 32768,
    "d131040": 131040,
    "opt016_2k": 2048,
}
PEAK_FP32_TFLOPS = 104.8
PEAK_FP16_TENSOR_TFLOPS = 209.5
N_PARAMS = 26_895_998_464
MATERIALITY = 0.02
COMBINED_DECODE_D2H = 4
OPT119_PREFILL_BF16_REREAD_REMOVED = 167_772_160
OPT121_WEIGHT_BYTES_SAVED = 3_355_443_200
OPT120_Q8Q8_KV_BYTES = 4_563_402_752
OPT120_DENSE_KV_BYTES = 8_589_934_592
OPT122_GDN_DRAM_FP32 = 317_718_528
OPT122_GDN_DRAM_BF16 = 166_723_584
CONCLUSION_LABELS = (
    "quartz_advantage",
    "measurable_headroom",
    "no_demonstrated_worthwhile_headroom",
    "insufficient_evidence",
)
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "measurement_utc",
    "identity",
    "opt123_identities_match",
    "bounds",
    "llama_comparison",
    "mechanism_disposition",
    "conclusions",
    "stop_reopen_criteria",
    "supports_continuing",
    "universal_optimality_claimed",
    "production_kept",
    "claims_throughput",
    "claims_performance_improvement",
    "report_path",
)
PROOF = (
    "analysis/diagnostics only: no production selector or kernel change; "
    "reuse OPT-123 samples only with matching identities; pinned llama "
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328 remains the benchmark authority; "
    "1792 GB/s and 104.8 TFLOPS are listed peaks, not measured application "
    "rates; a bound that excludes unpack/scale or serial launch idle is "
    "optimistic, not attainable; independent phase sums are not a bound; "
    "no universal-optimality assertion; speculation/MTP/batching/model "
    "replacement/sparse attention are outside this approved batch"
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class CeilingError(AssertionError):
    """Fail-closed OPT-124 ceiling-analysis error."""


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT)


def load_iteration() -> dict[str, Any]:
    return load_json(ITERATION)


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-124 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def tok_s_to_ms(tok_s: float | None, *, tokens: int = 1) -> float | None:
    if tok_s is None or float(tok_s) <= 0.0:
        return None
    return (float(tokens) / float(tok_s)) * 1000.0


def flops_to_ms(flops: float, *, tflops: float) -> float:
    if tflops <= 0.0:
        return float("inf")
    return (float(flops) / (tflops * 1.0e12)) * 1000.0


def dense_gemm_flops(*, tokens: int) -> float:
    """Logical dense MAC FLOPs (2*N*T). Excludes unpack/scale and extras."""
    return 2.0 * float(N_PARAMS) * float(tokens)


def attention_decode_flops(*, prefix: int) -> float:
    """QK/V attention extras beyond the weight GEMM, one decode token."""
    attention_layers = 16
    query_heads = 24
    head_width = 256
    return 2.0 * attention_layers * query_heads * float(prefix) * head_width


def combined_decode_bytes(*, prefix: int) -> dict[str, Any]:
    """OPT-123 combined-stack compulsory decode bytes (lazy logits)."""
    original = compulsory_decode_bytes(prefix=prefix)
    phases = dict(original["phases"])
    phases["transfers"] = COMBINED_DECODE_D2H
    phases["logits"] = COMBINED_DECODE_D2H
    total = sum(int(value) for value in phases.values())
    payload = dict(original)
    payload["representation"] = "combined_opt118_opt119"
    payload["phases"] = phases
    payload["bytes_per_token"] = total
    payload["bytes_per_request_one_token"] = total
    payload["logits_d2h_bytes"] = COMBINED_DECODE_D2H
    payload["hidden_d2h_bytes"] = 0
    payload["parent_transfers_bytes"] = original["phases"]["transfers"]
    payload["lazy_index_bytes"] = COMBINED_DECODE_D2H
    payload["measured_d2h_source"] = "fixtures/opt123_combined_stack.json inventory"
    payload["opt119_decode_note"] = (
        "mixer/FFN Q8 fusion removes standalone quantize rereads; weight "
        "sweeps unchanged. Architecture residual/FFN intermediates remain."
    )
    return payload


def combined_prefill_bytes(*, tokens: int = 4096) -> dict[str, Any]:
    original = compulsory_prefill_bytes(tokens=tokens)
    phases = dict(original["phases"])
    activations = max(
        0, int(phases.get("activations") or 0) - OPT119_PREFILL_BF16_REREAD_REMOVED
    )
    phases["activations"] = activations
    payload = dict(original)
    payload["representation"] = "combined_opt118_opt119"
    payload["phases"] = phases
    payload["bytes_per_request"] = sum(int(value) for value in phases.values())
    payload["bytes_per_token"] = payload["bytes_per_request"] / float(tokens)
    payload["opt119_bf16_reread_removed_bytes"] = OPT119_PREFILL_BF16_REREAD_REMOVED
    payload["parent_activation_bytes"] = original["phases"]["activations"]
    return payload


def lower_precision_alternatives(*, prefix: int) -> dict[str, Any]:
    """Byte models for encodings that were measured and not admitted."""
    original = combined_decode_bytes(prefix=prefix)
    weight_alt = max(0, int(original["phases"]["weights"]) - OPT121_WEIGHT_BYTES_SAVED)
    kv_scale = float(kv_bytes(prefix)) / float(OPT120_DENSE_KV_BYTES)
    kv_q8q8 = int(round(OPT120_Q8Q8_KV_BYTES * kv_scale))
    gdn_bf16 = OPT122_GDN_DRAM_BF16
    return {
        "admitted": False,
        "note": (
            "These encodings were quality-gated or bound-rejected. A smaller "
            "byte model is not an identical-arithmetic kernel win and is not "
            "the shipping representation."
        ),
        "opt121_q8_to_q4k": {
            "task": "OPT-121",
            "verdict": "quality_blocked",
            "weight_bytes": weight_alt,
            "peak_ms": bytes_to_peak_ms(weight_alt),
            "bytes_saved": OPT121_WEIGHT_BYTES_SAVED,
            "quality_matched": False,
        },
        "opt120_q8q8_kv": {
            "task": "OPT-120",
            "verdict": "quality_blocked",
            "kv_bytes": kv_q8q8,
            "peak_ms": bytes_to_peak_ms(kv_q8q8),
            "capacity_128k_bytes": OPT120_Q8Q8_KV_BYTES,
            "quality_matched": False,
        },
        "opt122_bf16_gdn": {
            "task": "OPT-122",
            "verdict": "no_material_opportunity",
            "gdn_dram_bytes": gdn_bf16,
            "peak_ms": bytes_to_peak_ms(gdn_bf16),
            "critical_path_save_ms": 0.0,
            "quality_matched": True,
            "note": "compute already exceeds FP32 DRAM time on gdn_core",
        },
    }


def compute_bound(*, tokens: int, prefix: int | None = None) -> dict[str, Any]:
    gemm = dense_gemm_flops(tokens=tokens)
    attn = attention_decode_flops(prefix=prefix) if prefix is not None else 0.0
    total = gemm + attn
    fp32_ms = flops_to_ms(total, tflops=PEAK_FP32_TFLOPS)
    fp16_ms = flops_to_ms(total, tflops=PEAK_FP16_TENSOR_TFLOPS)
    return {
        "logical_dense_mac_flops": gemm,
        "attention_extra_flops": attn,
        "total_flops": total,
        "excludes_unpack_scale": True,
        "unpack_scale_note": (
            "Q4_K/Q8_0/Q6_K dequant and scale application are required work "
            "not counted in 2*N*T. Excluding them makes the compute bound "
            "optimistic, not attainable."
        ),
        "peak_fp32_tflops": PEAK_FP32_TFLOPS,
        "peak_fp16_tensor_tflops_dense": PEAK_FP16_TENSOR_TFLOPS,
        "fp32_peak_ms": fp32_ms,
        "fp16_tensor_peak_ms": fp16_ms,
        "optimistic_compute_ms": fp16_ms,
        "listed_peaks_not_measured": True,
    }


def serial_dependency_bound() -> dict[str, Any]:
    return {
        "batch": 1,
        "sessions": 1,
        "autoregressive": True,
        "tokens_cannot_overlap": True,
        "request_time_ge_n_times_per_token_bound": True,
        "cross_token_weight_amortization": (
            "Prefill may reuse a weight tile across prompt rows inside one "
            "microbatch. Ordinary decode batch-one cannot; each token rereads "
            "streamed weights. That scope limit is not a license to implement "
            "speculation, MTP, or multi-session batching here."
        ),
        "excluded": [
            "speculative_decoding",
            "multi_token_prediction",
            "batching_concurrent_sessions",
            "model_replacement_retraining",
            "sparse_attention",
        ],
    }


def overlap_aware_bound(
    *,
    byte_critical_ms: float,
    compute_ms: float,
    serial_idle_ms: float | None,
) -> dict[str, Any]:
    compute_or_bytes = max(float(byte_critical_ms), float(compute_ms))
    idle = float(serial_idle_ms) if serial_idle_ms is not None else 0.0
    # Launch idle is serial with kernel busy; bytes and compute overlap.
    total = compute_or_bytes + idle
    return {
        "byte_critical_ms": float(byte_critical_ms),
        "compute_ms": float(compute_ms),
        "serial_idle_ms": serial_idle_ms,
        "busy_overlap_ms": compute_or_bytes,
        "overlap_aware_ms": total,
        "independent_sum_not_a_bound": True,
        "note": (
            "Weight traffic and GEMM share the busy interval (max). Launch/"
            "idle gaps are serial. Summing independent peak bounds is not a "
            "bound. A roofline bottleneck is not proof of an optimal algorithm."
        ),
    }


def opt123_identities(opt123: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(opt123 or load_json(OPT123_FIXTURE))
    freeze = payload.get("freeze") or {}
    llama = payload.get("llama") or {}
    quality = payload.get("quality") or {}
    return {
        "llama_revision": payload.get("llama_revision") or freeze.get("llama_revision"),
        "gguf_sha256": payload.get("gguf_sha256") or freeze.get("gguf_sha256"),
        "parent": payload.get("parent"),
        "candidate": payload.get("candidate"),
        "keepers": list(payload.get("keepers") or []),
        "execution_graphs": freeze.get("execution_graphs"),
        "q4_decode": freeze.get("q4_decode"),
        "packed_kv": freeze.get("packed_kv"),
        "weight_requant": freeze.get("weight_requant"),
        "gdn_state": freeze.get("gdn_state"),
        "rejected_no_leak": bool(payload.get("rejected_no_leak")),
        "reuse_historical_llama": bool(llama.get("reuse_historical_llama")),
        "llama_p4096_flash_attn": (llama.get("p4096") or {}).get("flash_attn"),
        "llama_n_gpu_layers": (llama.get("p4096") or {}).get("n_gpu_layers"),
        "llama_type_k": (llama.get("p4096") or {}).get("type_k"),
        "llama_type_v": (llama.get("p4096") or {}).get("type_v"),
        "quality_ppl_ratio": quality.get("ppl_ratio"),
        "opt116_contract_id": quality.get("opt116_contract_id"),
    }


def identities_match(opt123: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contract = load_contract()
    observed = opt123_identities(opt123)
    expected = {
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": contract["gguf_sha256"],
        "parent": contract["historical_control"],
        "candidate": contract["final_stack"],
        "keepers": ["OPT-118", "OPT-119"],
        "execution_graphs": "ffn_only",
        "q4_decode": "llama_q4k_mmvq",
        "packed_kv": "DenseBf16",
        "weight_requant": "none",
        "gdn_state": "Fp32",
        "rejected_no_leak": True,
        "reuse_historical_llama": False,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed.get(key)}
        for key in expected
        if observed.get(key) != expected[key]
    }
    return {
        "ok": not mismatches,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
        "reuse_opt123_samples": not mismatches,
    }


def sitting_identity() -> dict[str, Any]:
    source, dirty = git_identity()
    opt123 = opt123_identities()
    return {
        "device": "NVIDIA GeForce RTX 5090",
        "llama_revision": LLAMA_REV,
        "llama_inspection_head": load_contract()["llama_inspection_head"],
        "ds4_inspection_head": load_contract()["ds4_inspection_head"],
        "gguf_sha256": GGUF_SHA,
        "source_revision": source,
        "source_state": dirty,
        "nvccflags": "-O2 --fmad=false",
        "execution_graphs": "ffn_only",
        "final_stack": "combined_opt118_opt119",
        "historical_control": "post113_selected",
        "image": IMAGE,
        "model": MODEL,
        "opt123": opt123,
        "post113_authenticated": authenticate_post113(),
    }


def opt123_wall_ms() -> dict[str, Any]:
    opt123 = load_json(OPT123_FIXTURE)
    performance = opt123.get("performance") or {}
    llama = opt123.get("llama") or {}
    long_ctx = (opt123.get("long_context") or {}).get("probes") or {}
    two_k = opt123.get("opt016_2k") or {}

    def _pair(name: str) -> dict[str, Any]:
        block = performance.get(name) or {}
        post113 = block.get("parent_tok_s")
        combined = block.get("candidate_tok_s")
        llama_tok = llama.get(f"{name}_tok_s")
        tokens = 4096 if name == "p4096" else 1
        return {
            "post113_tok_s": post113,
            "combined_tok_s": combined,
            "llama_tok_s": llama_tok,
            "post113_ms": tok_s_to_ms(post113, tokens=tokens),
            "combined_ms": tok_s_to_ms(combined, tokens=tokens),
            "llama_ms": tok_s_to_ms(llama_tok, tokens=tokens),
            "geo_ratio_vs_post113": block.get("geo_ratio"),
            "ci_lower_vs_post113": (block.get("ci") or {}).get("ci_lower"),
            "source": "fixtures/opt123_combined_stack.json",
            "identity": "combined_opt118_opt119 vs post113_selected vs pinned llama",
        }

    rows = {name: _pair(name) for name in ("p4096", "d128", "d2048")}
    for name in ("d8192", "d32768", "d131040"):
        probe = long_ctx.get(name) or {}
        tok_s = probe.get("tok_s") if probe.get("ok") else None
        rows[name] = {
            "post113_tok_s": None,
            "combined_tok_s": tok_s,
            "llama_tok_s": None,
            "combined_ms": tok_s_to_ms(tok_s) if tok_s else None,
            "oom": bool(probe.get("oom")),
            "ok": bool(probe.get("ok")),
            "populated_cache": probe.get("populated_cache"),
            "separately_labeled": True,
            "source": "fixtures/opt123_combined_stack.json long_context",
        }
    rows["opt016_2k"] = {
        "quartz_tok_s": two_k.get("quartz_tok_s"),
        "llama_tok_s": two_k.get("llama_tok_s"),
        "quartz_measured": bool(two_k.get("quartz_measured")),
        "oom": not bool(two_k.get("quartz_measured")),
        "separately_labeled": True,
        "source": "fixtures/opt123_combined_stack.json opt016_2k",
    }
    rows["nonexclusive_sitting"] = bool(
        (opt123.get("long_context") or {}).get("nonexclusive_oom")
    )
    return rows


def opt115_timeline_overlay() -> dict[str, Any]:
    if not OPT115_FIXTURE.is_file():
        return {"available": False}
    payload = load_json(OPT115_FIXTURE)
    timeline = payload.get("timeline") or {}
    return {
        "available": True,
        "identity": "post113_selected",
        "matches_combined_stack": False,
        "reuse_as_combined": False,
        "note": (
            "OPT-115 timeline is post113, not combined_opt118_opt119. Idle/"
            "busy numbers are a labeled overlay, not a combined-stack measurement."
        ),
        "d128": {
            "gpu_busy_union_ms": (timeline.get("d128") or {}).get("gpu_busy_union_ms"),
            "gpu_idle_gap_ms": (timeline.get("d128") or {}).get("gpu_idle_gap_ms"),
            "unclassified_ms": (timeline.get("d128") or {}).get("unclassified_ms"),
            "interval_coverage": (timeline.get("d128") or {}).get("interval_coverage"),
            "wall_ms": (timeline.get("d128") or {}).get("wall_ms"),
        },
        "d2048": {
            "gpu_busy_union_ms": (timeline.get("d2048") or {}).get("gpu_busy_union_ms"),
            "gpu_idle_gap_ms": (timeline.get("d2048") or {}).get("gpu_idle_gap_ms"),
            "unclassified_ms": (timeline.get("d2048") or {}).get("unclassified_ms"),
            "interval_coverage": (timeline.get("d2048") or {}).get("interval_coverage"),
            "wall_ms": (timeline.get("d2048") or {}).get("wall_ms"),
        },
        "aa_repeatable": payload.get("aa_verdict") == "repeatable",
    }


def recompute_bounds(*, timeline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    walls = opt123_wall_ms()
    overlay = opt115_timeline_overlay()
    serial = serial_dependency_bound()
    workloads: dict[str, Any] = {}
    for name in ("p4096", *DECODE_WORKLOADS):
        prefix = PREFIX[name]
        if name == "p4096":
            traffic = combined_prefill_bytes(tokens=4096)
            phases = dict(traffic["phases"])
            tokens = 4096
            attn_prefix = None
            wall = (walls.get(name) or {}).get("combined_ms")
            idle = None
        else:
            traffic = combined_decode_bytes(prefix=prefix)
            phases = dict(traffic["phases"])
            tokens = 1
            attn_prefix = prefix
            wall = (walls.get(name) or {}).get("combined_ms")
            measured_tl = (timeline or {}).get(name) if timeline else None
            if measured_tl and measured_tl.get("valid"):
                idle = measured_tl.get("gpu_idle_gap_ms")
            elif name in ("d128", "d2048"):
                idle = (overlay.get(name) or {}).get("gpu_idle_gap_ms")
            else:
                idle = None
        byte_bound = bandwidth_bounds(phases, wall_ms=wall, prefix=prefix)
        compute = compute_bound(tokens=tokens, prefix=attn_prefix)
        overlap = overlap_aware_bound(
            byte_critical_ms=float(byte_bound["critical_path_peak_ms"]),
            compute_ms=float(compute["optimistic_compute_ms"]),
            serial_idle_ms=idle,
        )
        residual = None
        if wall is not None:
            residual = float(wall) - float(overlap["busy_overlap_ms"])
        llama_ms = (walls.get(name) or {}).get("llama_ms")
        workloads[name] = {
            "traffic": traffic,
            "byte_bound": byte_bound,
            "compute_bound": compute,
            "overlap_aware": overlap,
            "lower_precision": lower_precision_alternatives(prefix=prefix),
            "opt123_combined_ms": wall,
            "opt123_llama_ms": llama_ms,
            "residual_vs_busy_bound_ms": residual,
            "idle_source": (
                "combined_timeline"
                if timeline and (timeline.get(name) or {}).get("valid")
                else ("opt115_post113_overlay" if idle is not None else None)
            ),
            "bound_excludes_unpack_and_may_exclude_idle": True,
            "optimistic_not_attainable": True,
        }
    return {
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
        "peak_fp32_tflops": PEAK_FP32_TFLOPS,
        "peak_fp16_tensor_tflops_dense": PEAK_FP16_TENSOR_TFLOPS,
        "measured_sustainable_bandwidth_gb_s": None,
        "n_params": N_PARAMS,
        "serial_dependency": serial,
        "inventory": inventory_bytes(),
        "opt115_timeline_overlay": overlay,
        "workloads": workloads,
        "assumptions": [
            "RTX 5090 listed peak 1792 GB/s; listed FP32 104.8 TFLOPS; dense FP16 tensor 209.5 TFLOPS is 2x FP32 without sparsity",
            "peaks are optimistic listed values, not observed application rates",
            "2*N*T MAC FLOPs exclude Q4_K unpack/scale; excluding required work makes the compute bound optimistic",
            "combined decode D2H is the measured 4-byte greedy index (OPT-123 inventory), not 1,013,760 B logits",
            "OPT-119 removed 167,772,160 B prefill BF16 quantize rereads; weight sweeps unchanged",
            "GDN decode charges pointer-swap commit, not an 8 GiB KV copy",
            "embedding lookup charges one Q4_K row",
            "independent phase sums are not a bound",
            "lower-precision alternatives are not admitted and are not identical-arithmetic wins",
        ],
        "claims_throughput": False,
    }


def authenticate_pinned_llama() -> dict[str, Any]:
    opt123 = load_json(OPT123_FIXTURE)
    llama = opt123.get("llama") or {}
    p4096 = llama.get("p4096") or {}
    d128 = llama.get("d128") or {}
    flash = p4096.get("flash_attn")
    ngl = p4096.get("n_gpu_layers")
    graph_hint = (
        "llama-bench on the pinned authority used n_gpu_layers=99, CUDA "
        "backend, flash_attn auto (-1), type_k/v f16, n_ctx 4096 for decode "
        "oracles. That is the intended strong baseline for this pin: full GPU "
        "offload and the flash-attention auto path. Graph capture is llama's "
        "internal CUDA-graph evaluate/capture, not a Quartz decode_segments8 "
        "equivalent."
    )
    ok = (
        str(llama.get("llama_revision")) == LLAMA_REV
        and not bool(llama.get("reuse_historical_llama"))
        and int(ngl or 0) >= 99
        and str(p4096.get("backends") or "").upper() == "CUDA"
    )
    kv_policy = {
        "type_k": p4096.get("type_k"),
        "type_v": p4096.get("type_v"),
        "matches_quartz_dense_bf16": (
            str(p4096.get("type_k")) == "f16" and str(p4096.get("type_v")) == "f16"
        ),
        "asymmetry_note": (
            "Pinned llama KV is f16. Quartz ships dense BF16 KV. Packed 8/4-bit "
            "Quartz KV (OPT-120) was quality_blocked and slower; no quality-matched "
            "llama packed-KV baseline was introduced. Do not call a changed-"
            "quantization Quartz comparison an identical-arithmetic kernel win."
        ),
    }
    return {
        "ok": ok,
        "pinned_revision": LLAMA_REV,
        "authority_unchanged": True,
        "n_gpu_layers": ngl,
        "flash_attn": flash,
        "flash_attn_auto": flash == -1,
        "backends": p4096.get("backends"),
        "gpu_info": p4096.get("gpu_info"),
        "type_k": p4096.get("type_k"),
        "type_v": p4096.get("type_v"),
        "model_n_params": p4096.get("model_n_params"),
        "decode_n_ctx": d128.get("n_ctx"),
        "decode_tokens": d128.get("decode_tokens"),
        "kv_policy": kv_policy,
        "graph_and_offload": graph_hint,
        "intended_strong_baseline": ok,
        "source": "fixtures/opt123_combined_stack.json llama",
    }


def inspect_supplementary_llama() -> dict[str, Any]:
    contract = load_contract()
    llama_path = ROOT.parent / "llama.cpp"
    inspected = inspect_repo_revision(llama_path)
    pin = contract["llama_revision"]
    inspection_head = contract["llama_inspection_head"]
    head = inspected.get("revision")
    material = bool(head) and head not in {pin, pin[:12]} and head != inspection_head
    frozen: list[dict[str, Any]] = []
    if material:
        frozen = [
            {
                "id": "inspection_head_default",
                "revision": head,
                "flags": "llama-bench -p 4096 -n 0 -ngl 99 flash_attn auto",
                "frozen_before_timing": True,
                "measured": False,
                "reason": "no pre-built inspection-HEAD binary; unbounded rebuild not authorized",
            },
            {
                "id": "inspection_head_flash_on",
                "revision": head,
                "flags": "llama-bench -p 4096 -n 0 -ngl 99 -fa 1",
                "frozen_before_timing": True,
                "measured": False,
                "reason": "no pre-built inspection-HEAD binary; unbounded rebuild not authorized",
            },
        ]
    return {
        "path": inspected.get("path"),
        "present": bool(inspected.get("present")),
        "revision": head,
        "state": inspected.get("state"),
        "pinned_revision": pin,
        "recorded_inspection_head": inspection_head,
        "is_not_pin": head != pin,
        "materially_stronger_evidence": False,
        "supplementary_admitted": frozen,
        "max_configs": contract["max_supplementary_llama_configs"],
        "measured": False,
        "limits_this_conclusion": bool(frozen) or not inspected.get("present"),
        "does_not_invalidate_pinned_comparison": True,
        "note": (
            "Pinned authority is unchanged. Failure to build/run a "
            "supplementary baseline limits the 'is this the strongest local "
            "llama' clause but does not invalidate the pinned comparison."
        ),
    }


def matched_comparison() -> dict[str, Any]:
    walls = opt123_wall_ms()
    llama_auth = authenticate_pinned_llama()
    opt116 = load_json(OPT116_FIXTURE) if OPT116_FIXTURE.is_file() else {}
    opt123 = load_json(OPT123_FIXTURE)
    quality = opt123.get("quality") or {}
    state = opt123.get("state_memory") or {}
    rows = []
    for name in ("p4096", "d128", "d2048"):
        block = walls[name]
        combined_ms = block.get("combined_ms")
        llama_ms = block.get("llama_ms")
        ratio = None
        if combined_ms and llama_ms and llama_ms > 0:
            ratio = float(block["combined_tok_s"]) / float(block["llama_tok_s"])
        rows.append(
            {
                "workload": name,
                "post113_tok_s": block.get("post113_tok_s"),
                "quartz_combined_tok_s": block.get("combined_tok_s"),
                "llama_tok_s": block.get("llama_tok_s"),
                "post113_ms": block.get("post113_ms"),
                "quartz_combined_ms": combined_ms,
                "llama_ms": llama_ms,
                "quartz_over_llama": ratio,
                "matched_request_boundaries": True,
                "source": block.get("source"),
            }
        )
    return {
        "pinned_llama": llama_auth,
        "supplementary_llama": inspect_supplementary_llama(),
        "quality": {
            "opt116_contract": opt116.get("contract_id")
            or quality.get("opt116_contract_id")
            or "opt116_generated_v1",
            "strict_quality_pass": quality.get("strict_quality_pass"),
            "successor_quality_pass": quality.get("successor_quality_pass"),
            "ppl_ratio": quality.get("ppl_ratio"),
            "candidate_nll_measured": quality.get("candidate_nll_measured"),
            "anchor": "post113_selected",
        },
        "memory": {
            "session_bytes": state.get("session_bytes"),
            "workspace_bytes": state.get("workspace_bytes"),
            "128k_fit": state.get("128k_fit"),
        },
        "p4096_d128_d2048": rows,
        "long_context": {
            "d8192": walls["d8192"],
            "d32768": walls["d32768"],
            "d131040": walls["d131040"],
            "not_a_d128_win": True,
        },
        "opt016_2k": walls["opt016_2k"],
        "nonexclusive_sitting": walls.get("nonexclusive_sitting"),
        "opt113_numbers_historical_only": True,
        "smaller_context_or_skipped_sampling_is_not_an_improvement": True,
    }


def _fixture_verdict(path: Path) -> str:
    if not path.is_file():
        return "missing"
    payload = load_json(path)
    return str(payload.get("verdict") or payload.get("status") or "unknown")


def mechanism_disposition() -> dict[str, Any]:
    overlay = opt115_timeline_overlay()
    d128_idle = (overlay.get("d128") or {}).get("gpu_idle_gap_ms")
    rows = [
        {
            "task": "OPT-117",
            "mechanism": "decode-segment CUDA graphs",
            "bytes_or_time_upper": {
                "bytes": 0,
                "peak_ms": d128_idle,
                "kind": "launch_idle_gap",
            },
            "implemented_candidate": "decode_segments8",
            "quality_state_blockers": "none (same_math/quality/state passed)",
            "complete_request_outcome": "retain_ffn_only",
            "measured_loss": (
                "D128 geo CI lower 0.979; D2048 0.980; decode p95 > 1.05× parent. "
                "Per-token recapture (257 param updates / 256 tokens) exceeded "
                "the remaining launch/idle gap."
            ),
            "missing_evidence": None,
            "capability_failure": (
                "kernel-arg patching is not byte-exact; production replay recaptures "
                "when frontier/GDN/KV topology change"
            ),
            "reopen": False,
            "reopen_requires": (
                "a new causal capture/replay design that avoids per-token recapture; "
                "not another random launch/tile choice"
            ),
        },
        {
            "task": "OPT-118",
            "mechanism": "lazy logits/hidden + decode D2H overlap",
            "bytes_or_time_upper": {
                "bytes": 1_013_760,
                "peak_ms": bytes_to_peak_ms(1_013_760),
                "kind": "pcie_logits_hidden",
            },
            "implemented_candidate": "lazy+overlap",
            "quality_state_blockers": "none",
            "complete_request_outcome": "keep",
            "measured_loss": None,
            "measured_keep": "D128 +0.172 tok/s (1.003×); shipping pin true/true",
            "missing_evidence": None,
            "capability_failure": None,
            "reopen": False,
            "reopen_requires": "none; already in the OPT-123 combined stack",
        },
        {
            "task": "OPT-119",
            "mechanism": "mixer+FFN RMSNorm→MMQ Q8 fusion",
            "bytes_or_time_upper": {
                "bytes": OPT119_PREFILL_BF16_REREAD_REMOVED,
                "peak_ms": bytes_to_peak_ms(OPT119_PREFILL_BF16_REREAD_REMOVED),
                "kind": "prefill_activation_reread",
            },
            "implemented_candidate": "mixer_q8+ffn_q8",
            "quality_state_blockers": "none",
            "complete_request_outcome": "keep",
            "measured_loss": None,
            "measured_keep": "P4096 +17.66 tok/s (1.006×); D128/D2048 guards ~1.00",
            "missing_evidence": None,
            "capability_failure": None,
            "reopen": False,
            "reopen_requires": "none; already in the OPT-123 combined stack",
        },
        {
            "task": "OPT-120",
            "mechanism": "packed Q8/Q4 attention KV",
            "bytes_or_time_upper": {
                "bytes": OPT120_DENSE_KV_BYTES - OPT120_Q8Q8_KV_BYTES,
                "peak_ms": bytes_to_peak_ms(kv_bytes(131_040)),
                "kind": "long_context_kv",
            },
            "implemented_candidate": "q8q8",
            "quality_state_blockers": "long-cache generated quality fail; held-out PPL still <1.01",
            "complete_request_outcome": "quality_blocked",
            "measured_loss": (
                "D8192 0.624×, D32768 0.316×, D131040 0.215× vs dense parent; "
                "P4096 guard 0.789×. Byte savings did not become a request win."
            ),
            "missing_evidence": None,
            "capability_failure": None,
            "reopen": False,
            "reopen_requires": (
                "a quality-matched packed encoding that is actually faster on "
                "populated-cache requests; unpack-into-registers already tried"
            ),
        },
        {
            "task": "OPT-121",
            "mechanism": "selective Q8→Q4_K weight requant + native FP4 study",
            "bytes_or_time_upper": {
                "bytes": OPT121_WEIGHT_BYTES_SAVED,
                "peak_ms": bytes_to_peak_ms(OPT121_WEIGHT_BYTES_SAVED),
                "kind": "streamed_weights",
            },
            "implemented_candidate": "q8_to_q4k",
            "quality_state_blockers": (
                "recurrence ΔNLL 3.92/3.46 vs max 0.02; long-cache fail"
            ),
            "complete_request_outcome": "quality_blocked",
            "measured_loss": "D128 0.870×, D2048 0.909×; P4096 guard 1.014× but quality fail",
            "missing_evidence": None,
            "capability_failure": "native FP4 study capability_absent=true",
            "reopen": False,
            "reopen_requires": (
                "a quality-passing weight encoding with a different error "
                "profile, or an actual FP4 path; not another Q8→Q4_K launch"
            ),
        },
        {
            "task": "OPT-122",
            "mechanism": "lower-precision persistent GDN state",
            "bytes_or_time_upper": {
                "bytes": OPT122_GDN_DRAM_FP32 - OPT122_GDN_DRAM_BF16,
                "peak_ms": 0.0,
                "kind": "gdn_dram_vs_compute",
            },
            "implemented_candidate": "none admitted (BF16/Q8 bound-rejected)",
            "quality_state_blockers": "not in scope; FP32 retained",
            "complete_request_outcome": "no_material_opportunity",
            "measured_loss": None,
            "measured_keep": None,
            "missing_evidence": None,
            "capability_failure": None,
            "reopen": False,
            "reopen_requires": (
                "evidence that gdn_core became DRAM-bound after some other keep; "
                "current enclosing compute already exceeds FP32 DRAM time"
            ),
        },
    ]
    return {
        "parent_stack": "combined_opt118_opt119 on post113 ffn_only",
        "rejected_paths_do_not_leak": True,
        "opt115_idle_is_post113_overlay": True,
        "ranked": rows,
        "changed_under_final_stack": (
            "OPT-118 and OPT-119 are the keepers. OPT-117/120/121/122 outcomes "
            "do not reverse on the combined stack: graphs remain ffn_only, KV "
            "dense BF16, weights unrequantized, GDN FP32. No new causal reason "
            "to reopen a rejected idea appeared in OPT-123 leave-one-out."
        ),
    }


def materiality_ms(wall_ms: float | None) -> float | None:
    if wall_ms is None or wall_ms <= 0.0:
        return None
    return float(wall_ms) * MATERIALITY


def classify_workload(
    *,
    name: str,
    quartz_tok_s: float | None,
    llama_tok_s: float | None,
    quartz_ms: float | None,
    llama_ms: float | None,
    busy_bound_ms: float | None,
    residual_ms: float | None,
    measured: bool,
    oom: bool,
    remaining_mechanism: str | None,
    remaining_ms: float | None,
) -> dict[str, Any]:
    threshold = materiality_ms(quartz_ms)
    if oom or not measured:
        label = "insufficient_evidence"
        why = (
            "required sitting did not complete (OOM or unlabeled). Missing "
            "measurement, not a proof that llama is optimal."
        )
    elif (
        quartz_tok_s
        and llama_tok_s
        and quartz_tok_s > llama_tok_s * (1.0 + MATERIALITY)
    ):
        label = "quartz_advantage"
        why = "Quartz combined tok/s exceeds pinned llama by more than 2%."
    elif (
        remaining_ms is not None
        and threshold is not None
        and remaining_ms <= threshold
        and (residual_ms is None or residual_ms <= threshold)
        and (
            llama_ms is None
            or (quartz_ms is not None and abs(quartz_ms - llama_ms) <= threshold)
        )
    ):
        label = "no_demonstrated_worthwhile_headroom"
        why = (
            "complete traffic account is within measurement uncertainty and "
            "plausible remaining scoped savings are below the 2% materiality "
            "threshold. This is not a global optimality claim."
        )
    elif remaining_mechanism:
        label = "measurable_headroom"
        why = (
            f"remaining mechanism: {remaining_mechanism}. Distance to llama "
            "and/or the optimistic busy bound exceeds 2% of request time."
        )
    else:
        label = "insufficient_evidence"
        why = "comparison or bound residual is not fully accounted."
    quartz_over_llama = None
    if quartz_tok_s and llama_tok_s and llama_tok_s > 0:
        quartz_over_llama = float(quartz_tok_s) / float(llama_tok_s)
    return {
        "workload": name,
        "label": label,
        "why": why,
        "quartz_tok_s": quartz_tok_s,
        "llama_tok_s": llama_tok_s,
        "quartz_ms": quartz_ms,
        "llama_ms": llama_ms,
        "busy_bound_ms": busy_bound_ms,
        "residual_vs_busy_bound_ms": residual_ms,
        "materiality_ms": threshold,
        "remaining_mechanism": remaining_mechanism,
        "remaining_ms": remaining_ms,
        "quartz_over_llama": quartz_over_llama,
        "universal_optimality": False,
    }


def supported_conclusions(
    *,
    bounds: Mapping[str, Any],
    comparison: Mapping[str, Any],
    mechanisms: Mapping[str, Any],
    timeline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    walls = opt123_wall_ms()
    overlay = bounds.get("opt115_timeline_overlay") or opt115_timeline_overlay()
    d128_idle = None
    if timeline and (timeline.get("d128") or {}).get("valid"):
        d128_idle = (timeline.get("d128") or {}).get("gpu_idle_gap_ms")
    else:
        d128_idle = (overlay.get("d128") or {}).get("gpu_idle_gap_ms")
    d2048_idle = None
    if timeline and (timeline.get("d2048") or {}).get("valid"):
        d2048_idle = (timeline.get("d2048") or {}).get("gpu_idle_gap_ms")
    else:
        d2048_idle = (overlay.get("d2048") or {}).get("gpu_idle_gap_ms")
    work = bounds.get("workloads") or {}

    def _busy(name: str) -> float | None:
        overlap = (work.get(name) or {}).get("overlap_aware") or {}
        value = overlap.get("busy_overlap_ms")
        return float(value) if value is not None else None

    def _residual(name: str) -> float | None:
        value = (work.get(name) or {}).get("residual_vs_busy_bound_ms")
        return float(value) if value is not None else None

    rows = {
        "p4096": classify_workload(
            name="p4096",
            quartz_tok_s=(walls["p4096"] or {}).get("combined_tok_s"),
            llama_tok_s=(walls["p4096"] or {}).get("llama_tok_s"),
            quartz_ms=(walls["p4096"] or {}).get("combined_ms"),
            llama_ms=(walls["p4096"] or {}).get("llama_ms"),
            busy_bound_ms=_busy("p4096"),
            residual_ms=_residual("p4096"),
            measured=True,
            oom=False,
            remaining_mechanism=(
                "prefill MMQ/attention efficiency versus pinned llama "
                "(OPT-119 already kept activation fusion; weight requant "
                "quality_blocked)"
            ),
            remaining_ms=(
                None
                if walls["p4096"].get("combined_ms") is None
                or walls["p4096"].get("llama_ms") is None
                else float(walls["p4096"]["combined_ms"])
                - float(walls["p4096"]["llama_ms"])
            ),
        ),
        "d128": classify_workload(
            name="d128",
            quartz_tok_s=(walls["d128"] or {}).get("combined_tok_s"),
            llama_tok_s=(walls["d128"] or {}).get("llama_tok_s"),
            quartz_ms=(walls["d128"] or {}).get("combined_ms"),
            llama_ms=(walls["d128"] or {}).get("llama_ms"),
            busy_bound_ms=_busy("d128"),
            residual_ms=_residual("d128"),
            measured=True,
            oom=False,
            remaining_mechanism=(
                "launch/idle after OPT-117 recapture reject; optimistic weight "
                "byte bound still below both Quartz and llama walls"
            ),
            remaining_ms=d128_idle,
        ),
        "d2048": classify_workload(
            name="d2048",
            quartz_tok_s=(walls["d2048"] or {}).get("combined_tok_s"),
            llama_tok_s=(walls["d2048"] or {}).get("llama_tok_s"),
            quartz_ms=(walls["d2048"] or {}).get("combined_ms"),
            llama_ms=(walls["d2048"] or {}).get("llama_ms"),
            busy_bound_ms=_busy("d2048"),
            residual_ms=_residual("d2048"),
            measured=True,
            oom=False,
            remaining_mechanism=(
                "launch/idle plus decode-attention path versus llama flash-attn "
                "auto; hybrid_crossover@1024 is shipping and was not reopened"
            ),
            remaining_ms=d2048_idle,
        ),
        "d8192": classify_workload(
            name="d8192",
            quartz_tok_s=(walls["d8192"] or {}).get("combined_tok_s"),
            llama_tok_s=None,
            quartz_ms=(walls["d8192"] or {}).get("combined_ms"),
            llama_ms=None,
            busy_bound_ms=_busy("d8192"),
            residual_ms=_residual("d8192"),
            measured=bool((walls["d8192"] or {}).get("ok")),
            oom=bool((walls["d8192"] or {}).get("oom")),
            remaining_mechanism=(
                "populated-cache KV traffic exists in the byte model, but "
                "OPT-120 packed KV was slower and quality_blocked; no matched "
                "llama long-context number on this sitting"
            ),
            remaining_ms=_residual("d8192"),
        ),
        "d32768": classify_workload(
            name="d32768",
            quartz_tok_s=(walls["d32768"] or {}).get("combined_tok_s"),
            llama_tok_s=None,
            quartz_ms=(walls["d32768"] or {}).get("combined_ms"),
            llama_ms=None,
            busy_bound_ms=_busy("d32768"),
            residual_ms=_residual("d32768"),
            measured=bool((walls["d32768"] or {}).get("ok")),
            oom=bool((walls["d32768"] or {}).get("oom")),
            remaining_mechanism=(
                "same as D8192: KV-byte headroom in the bound, no admitted encoding"
            ),
            remaining_ms=_residual("d32768"),
        ),
        "d131040": classify_workload(
            name="d131040",
            quartz_tok_s=None,
            llama_tok_s=None,
            quartz_ms=None,
            llama_ms=None,
            busy_bound_ms=_busy("d131040"),
            residual_ms=None,
            measured=False,
            oom=True,
            remaining_mechanism=None,
            remaining_ms=None,
        ),
        "opt016_2k": classify_workload(
            name="opt016_2k",
            quartz_tok_s=None,
            llama_tok_s=(walls["opt016_2k"] or {}).get("llama_tok_s"),
            quartz_ms=None,
            llama_ms=tok_s_to_ms(
                (walls["opt016_2k"] or {}).get("llama_tok_s"), tokens=2048
            ),
            busy_bound_ms=None,
            residual_ms=None,
            measured=bool((walls["opt016_2k"] or {}).get("quartz_measured")),
            oom=True,
            remaining_mechanism=None,
            remaining_ms=None,
        ),
    }
    for row in rows.values():
        assert row["label"] in CONCLUSION_LABELS
        assert row["universal_optimality"] is False
    supports_continuing = False
    supports_llama_ceiling = False
    why_continue = (
        "The approved OPT-117–122 ladder is exhausted for quality-admitted "
        "encodings. Remaining decode gaps versus llama and versus the "
        "optimistic busy bound exceed 2%, so this is not evidence that further "
        "improvement is impractical. Continuing requires a new causal mechanism "
        "(capture/replay that avoids recapture, or a quality-passing weight "
        "encoding), not another random tile/launch. Speculation/MTP/batching "
        "are outside this batch and are not authorized here."
    )
    return {
        "per_workload": rows,
        "supports_continuing": supports_continuing,
        "supports_continuing_why": why_continue,
        "supports_claiming_llama_is_practical_ceiling": supports_llama_ceiling,
        "universal_optimality_claimed": False,
        "materiality_fraction": MATERIALITY,
        "mechanisms_reviewed": [row["task"] for row in mechanisms.get("ranked") or []],
        "quality_envelope": comparison.get("quality"),
    }


def stop_reopen_criteria() -> dict[str, Any]:
    return {
        "stop_this_batch_when": [
            "OPT-117–122 quality-admitted candidates are keep-or-reject complete",
            "no new causal mechanism is identified (random tile/launch is not a reason)",
            "speculation/MTP/batching/model-replacement/sparse attention remain out of scope",
        ],
        "do_not_stop_claiming": [
            "llama.cpp is globally optimal",
            "the kernel-tuning ladder proved a bandwidth bound",
            "a roofline bottleneck proved an optimal algorithm",
        ],
        "reopen_when": [
            "a new causal graph-capture/replay design avoids per-token recapture",
            "a quality-passing weight encoding with a different error profile exists",
            "gdn_core becomes DRAM-bound after some other keep",
            "a quality-matched packed-KV encoding is actually faster on populated cache",
            "exclusive sitting completes D131040 and OPT-016 2K for the combined stack",
        ],
        "practical_stop_requires": (
            "complete traffic account within measurement uncertainty AND "
            "plausible remaining scoped savings below 2% request time. Decode "
            "P4096/D128/D2048 do not currently meet that joint test versus llama."
        ),
        "missing_for_a_stronger_stop": [
            "combined-stack event-union timeline if identity-timeline GPU phase did not run",
            "measured sustainable bandwidth for weight-stream access",
            "exclusive-sitting D131040 and OPT-016 2K",
            "quality-matched llama packed-KV / lower-precision weight baseline",
            "built inspection-HEAD llama supplementary configs",
        ],
    }


def empty_payload(*, reason: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    match = identities_match()
    bounds = recompute_bounds()
    comparison = matched_comparison()
    mechanisms = mechanism_disposition()
    conclusions = supported_conclusions(
        bounds=bounds, comparison=comparison, mechanisms=mechanisms
    )
    return {
        "schema_version": 1,
        "task": "OPT-124",
        "status": "blocked_gpu_unavailable" if "gpu" in reason else "analysis",
        "measurement_utc": utc_now(),
        "identity": dict(identity),
        "hardware_executed": False,
        "gpu_blocker": reason,
        "opt123_identities_match": match,
        "bounds": bounds,
        "llama_comparison": comparison,
        "mechanism_disposition": mechanisms,
        "timeline": {},
        "conclusions": conclusions,
        "stop_reopen_criteria": stop_reopen_criteria(),
        "supports_continuing": conclusions["supports_continuing"],
        "universal_optimality_claimed": False,
        "production_kept": True,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "report_path": "evidence/optimization/opt124-practical-ceiling/REPORT.md",
        "proof_limit": PROOF,
    }


def persist(payload: Mapping[str, Any]) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump_json(FIXTURE, payload)
    write_report(payload)
    return dict(payload)


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        return load_json(FIXTURE)
    identity = sitting_identity()
    available, blocker = gpu_available()
    reason = blocker if not available else "scaffold_pending_gpu_phases"
    return empty_payload(reason=reason, identity=identity)


def merge_phase(
    existing: Mapping[str, Any], phase: str, block: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(existing)
    key = phase.replace("-", "_")
    payload[key] = dict(block)
    if phase == "freeze":
        payload["identity"] = block.get("identity") or payload.get("identity")
        payload["opt123_identities_match"] = block.get("opt123_identities_match")
    if phase == "bounds":
        payload["bounds"] = dict(block.get("bounds") or block)
    if phase == "llama-compare":
        payload["llama_comparison"] = dict(block.get("llama_comparison") or block)
    if phase == "mechanisms":
        payload["mechanism_disposition"] = dict(
            block.get("mechanism_disposition") or block
        )
    if phase == "identity-timeline":
        payload["timeline"] = dict(block.get("timeline") or block)
        payload["hardware_executed"] = bool(block.get("hardware_executed"))
        payload["gpu_blocker"] = block.get("gpu_blocker")
        if block.get("bounds"):
            payload["bounds"] = block["bounds"]
    if phase == "conclusions":
        payload["conclusions"] = dict(block.get("conclusions") or block)
        payload["supports_continuing"] = bool(
            (payload["conclusions"] or {}).get("supports_continuing")
        )
        payload["stop_reopen_criteria"] = dict(
            block.get("stop_reopen_criteria") or stop_reopen_criteria()
        )
    payload["measurement_utc"] = utc_now()
    payload["universal_optimality_claimed"] = False
    payload["production_kept"] = True
    payload["claims_throughput"] = False
    payload["claims_performance_improvement"] = False
    payload["status"] = "analysis"
    return payload


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    identity = payload.get("identity") or {}
    match = payload.get("opt123_identities_match") or {}
    bounds = payload.get("bounds") or {}
    comparison = payload.get("llama_comparison") or {}
    mechanisms = payload.get("mechanism_disposition") or {}
    conclusions = payload.get("conclusions") or {}
    timeline = payload.get("timeline") or {}
    stop = payload.get("stop_reopen_criteria") or stop_reopen_criteria()
    work = bounds.get("workloads") or {}
    per = conclusions.get("per_workload") or {}

    bound_rows = []
    for name in ("p4096", "d128", "d2048", "d8192", "d32768", "d131040"):
        block = work.get(name) or {}
        byte_b = block.get("byte_bound") or {}
        compute = block.get("compute_bound") or {}
        overlap = block.get("overlap_aware") or {}
        bound_rows.append(
            f"| {name} | {_fmt(byte_b.get('critical_path_peak_ms'))} | "
            f"{_fmt(compute.get('optimistic_compute_ms'))} | "
            f"{_fmt(overlap.get('serial_idle_ms'))} | "
            f"{_fmt(overlap.get('busy_overlap_ms'))} | "
            f"{_fmt(block.get('opt123_combined_ms'))} | "
            f"{_fmt(block.get('opt123_llama_ms'))} | "
            f"{_fmt(block.get('residual_vs_busy_bound_ms'))} | "
            f"{block.get('idle_source') or 'n/a'} |"
        )
    llama_rows = []
    for row in comparison.get("p4096_d128_d2048") or []:
        llama_rows.append(
            f"| {row.get('workload')} | {_fmt(row.get('post113_tok_s'))} | "
            f"{_fmt(row.get('quartz_combined_tok_s'))} | "
            f"{_fmt(row.get('llama_tok_s'))} | "
            f"{_fmt(row.get('quartz_combined_ms'))} | "
            f"{_fmt(row.get('llama_ms'))} | "
            f"{_fmt(row.get('quartz_over_llama'))} |"
        )
    mech_rows = []
    for row in mechanisms.get("ranked") or []:
        upper = row.get("bytes_or_time_upper") or {}
        mech_rows.append(
            f"| {row.get('task')} | {row.get('complete_request_outcome')} | "
            f"{upper.get('bytes')} | {_fmt(upper.get('peak_ms'))} | "
            f"{row.get('implemented_candidate')} | "
            f"{'yes' if row.get('reopen') else 'no'} |"
        )
    conclusion_rows = []
    for name in WORKLOADS:
        row = per.get(name) or {}
        conclusion_rows.append(
            f"| {name} | {row.get('label', 'n/a')} | "
            f"{_fmt(row.get('quartz_tok_s'))} | {_fmt(row.get('llama_tok_s'))} | "
            f"{_fmt(row.get('quartz_over_llama'))} | "
            f"{row.get('remaining_mechanism') or 'n/a'} |"
        )
    tl_rows = []
    for name in ("d128", "d2048"):
        block = timeline.get(name) or {}
        tl_rows.append(
            f"| {name} | {block.get('valid')} | "
            f"{_fmt(block.get('gpu_busy_union_ms'))} | "
            f"{_fmt(block.get('gpu_idle_gap_ms'))} | "
            f"{_fmt(block.get('unclassified_ms'))} | "
            f"{_fmt(block.get('interval_coverage'))} | "
            f"{block.get('identity') or block.get('reason') or 'n/a'} |"
        )
    llama_auth = comparison.get("pinned_llama") or {}
    supp = comparison.get("supplementary_llama") or {}
    quality = comparison.get("quality") or {}
    long_ctx = comparison.get("long_context") or {}
    two_k = comparison.get("opt016_2k") or {}
    text = f"""# OPT-124 — Practical performance ceiling under explicit constraints

Status: **{payload.get("status")}**. Hardware executed
`{payload.get("hardware_executed")}`.
universal_optimality_claimed=`{payload.get("universal_optimality_claimed")}`.
supports_continuing=`{payload.get("supports_continuing")}`.

{PROOF}

## Sitting identity

- device: {identity.get("device")}
- final Quartz stack: `{identity.get("final_stack")}`
- historical control: `{identity.get("historical_control")}`
- llama_revision (pinned authority): `{identity.get("llama_revision")}`
- llama inspection HEAD (not the pin): `{identity.get("llama_inspection_head")}`
- gguf_sha256: `{identity.get("gguf_sha256")}`
- source: `{identity.get("source_revision")}` ({identity.get("source_state")})
- nvccflags: `{identity.get("nvccflags")}`
- execution_graphs: `{identity.get("execution_graphs")}`
- OPT-123 identities match: `{match.get("ok")}` (reuse samples `{match.get("reuse_opt123_samples")}`)
- gpu_blocker: {payload.get("gpu_blocker")}

OPT-113 tok/s numbers remain historical. This comparison uses the OPT-123
combined sitting versus authenticated post113 and pinned llama on the same
request boundaries, generated token counts, capacities, and GGUF.

## Recomputed byte / compute / serial bounds

Listed peaks: 1792 GB/s DRAM, 104.8 TFLOPS FP32, 209.5 TFLOPS dense FP16
tensor (2× FP32, no sparsity). Measured sustainable bandwidth is `None`.
Independent phase sums are **not** a bound. Unpack/scale is excluded from
2×N×T, so the compute number is optimistic. Combined-stack decode D2H is
the measured 4-byte greedy index, not 1,013,760 B logits.

| Workload | byte critical ms | optimistic compute ms | serial idle ms | busy overlap ms | Quartz wall ms | llama wall ms | residual vs busy ms | idle source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(bound_rows)}

Serial dependency: ordinary autoregressive batch-one execution cannot overlap
tokens. Prefill may amortize a weight tile across prompt rows inside one
microbatch; decode cannot. Speculation, MTP, multi-session batching, model
replacement and sparse attention are outside this approved batch. Their
exclusion prevents a universal claim; it does not authorize implementation.

Lower-precision alternatives (OPT-120 q8q8 KV, OPT-121 Q8→Q4_K weights,
OPT-122 BF16 GDN) are **not admitted**. A smaller byte model is not an
identical-arithmetic kernel win.

## Matched llama / post113 / combined comparison

Pinned llama authentication: ok=`{llama_auth.get("ok")}`; n_gpu_layers=
`{llama_auth.get("n_gpu_layers")}`; flash_attn auto=`{llama_auth.get("flash_attn_auto")}`;
backends=`{llama_auth.get("backends")}`; type_k/v=`{llama_auth.get("type_k")}`/
`{llama_auth.get("type_v")}`. Authority unchanged. Supplementary inspection
HEAD present=`{supp.get("present")}` revision=`{supp.get("revision")}`
measured=`{supp.get("measured")}`. Failure to run a supplementary baseline
limits the strongest-local-llama clause and does not invalidate the pin.

Quality (OPT-116 successor, budgets anchored to post113): strict=
`{quality.get("strict_quality_pass")}` successor=`{quality.get("successor_quality_pass")}`
PPL ratio=`{quality.get("ppl_ratio")}` candidate NLL measured=
`{quality.get("candidate_nll_measured")}`.

| Workload | post113 tok/s | combined tok/s | llama tok/s | combined ms | llama ms | combined/llama |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(llama_rows)}

Long-context (separately labeled, not D128/D2048 wins): D8192 tok/s
`{_fmt((long_ctx.get("d8192") or {}).get("combined_tok_s"))}`; D32768
`{_fmt((long_ctx.get("d32768") or {}).get("combined_tok_s"))}`; D131040 OOM=
`{(long_ctx.get("d131040") or {}).get("oom")}`. OPT-016 2K llama tok/s
`{_fmt(two_k.get("llama_tok_s"))}` quartz measured=`{two_k.get("quartz_measured")}`.
Non-exclusive sitting (unrelated processes) is why 128K/2K OOM; this dossier
does not authorize stopping them. A smaller allocated context or skipped host
sampling is not an inference improvement under identical constraints.

## Combined-stack timeline (optional GPU)

Method: CUDA-event unions; overlapping intervals are not summed. OPT-115
post113 overlay is not silently treated as combined.

| Prefix | valid | GPU busy ms | GPU idle ms | unclassified ms | coverage | identity |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(tl_rows) if tl_rows else "| n/a | n/a | n/a | n/a | n/a | n/a | n/a |"}

## Mechanism disposition (OPT-117–122)

| Task | outcome | bytes upper | peak ms upper | candidate | reopen |
|---|---|---:|---:|---|---|
{chr(10).join(mech_rows)}

{mechanisms.get("changed_under_final_stack")}

## Supported conclusions per workload

Labels are only: quartz_advantage, measurable_headroom,
no_demonstrated_worthwhile_headroom, insufficient_evidence. None of these is a
universal optimality claim. Materiality default is 2% of request time.

| Workload | conclusion | Quartz tok/s | llama tok/s | ratio | remaining mechanism |
|---|---|---:|---:|---:|---|
{chr(10).join(conclusion_rows)}

supports_continuing=`{conclusions.get("supports_continuing")}`.
{conclusions.get("supports_continuing_why")}

## Stop / reopen criteria

Stop this batch when: {"; ".join(stop.get("stop_this_batch_when") or [])}.

Do not stop by claiming: {"; ".join(stop.get("do_not_stop_claiming") or [])}.

Reopen when: {"; ".join(stop.get("reopen_when") or [])}.

Practical-stop test: {stop.get("practical_stop_requires")}

Missing for a stronger stop: {"; ".join(stop.get("missing_for_a_stronger_stop") or [])}.

production_kept=True. claims_throughput=false.
claims_performance_improvement=false. No tok/s delta; this task does not
change the sitting baseline.
"""
    REPORT.write_text(text, encoding="utf-8")


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    if missing:
        raise CeilingError(f"fixture missing keys {missing}")
    if payload.get("claims_throughput") is not False:
        raise CeilingError("claims_throughput must be false")
    if payload.get("claims_performance_improvement") is not False:
        raise CeilingError("claims_performance_improvement must be false")
    if payload.get("universal_optimality_claimed") is not False:
        raise CeilingError("universal optimality must not be claimed")
    if payload.get("production_kept") is not True:
        raise CeilingError("production_kept must remain true")
    match = payload.get("opt123_identities_match") or {}
    if match.get("ok") is not True:
        raise CeilingError("OPT-123 identities must match before reuse")
    conclusions = payload.get("conclusions") or {}
    per = conclusions.get("per_workload") or {}
    for name in WORKLOADS:
        row = per.get(name) or {}
        if row.get("label") not in CONCLUSION_LABELS:
            raise CeilingError(f"{name} missing supported conclusion label")
        if row.get("universal_optimality"):
            raise CeilingError(f"{name} asserted universal optimality")
    bounds = payload.get("bounds") or {}
    d128 = (bounds.get("workloads") or {}).get("d128") or {}
    byte_b = d128.get("byte_bound") or {}
    if not byte_b.get("independent_sum_not_a_bound"):
        raise CeilingError("independent sums must not be treated as a bound")
    if (d128.get("traffic") or {}).get("logits_d2h_bytes") != COMBINED_DECODE_D2H:
        raise CeilingError("combined-stack decode D2H must be the 4-byte index")
    return {"ok": True, "task": "OPT-124"}


def parse_opt124_records(text: str) -> list[dict[str, Any]]:
    for line in text.splitlines():
        if line.startswith(RECORDS_PREFIX):
            payload = json.loads(line.split("=", 1)[1])
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [*docker_common(IMAGE, tier), *listed]
    completed = subprocess.run(listed, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CeilingError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def run_freeze(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = sitting_identity()
    match = identities_match()
    if not match["ok"]:
        raise CeilingError(f"OPT-123 identities mismatch: {match['mismatches']}")
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "freeze",
        "mode": mode,
        "ok": True,
        "identity": identity,
        "opt123_identities_match": match,
        "opt115_fixture": str(OPT115_FIXTURE.relative_to(ROOT)),
        "opt116_fixture": str(OPT116_FIXTURE.relative_to(ROOT)),
        "opt123_fixture": str(OPT123_FIXTURE.relative_to(ROOT)),
        "family_plan": family_plan(mode, "freeze"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "freeze.json", payload)
    fixture = merge_phase(current_fixture(), "freeze", payload)
    persist(fixture)
    return payload


def run_bounds(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    bounds = recompute_bounds(timeline=fixture.get("timeline"))
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "bounds",
        "mode": mode,
        "ok": True,
        "bounds": bounds,
        "family_plan": family_plan(mode, "bounds"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "bounds.json", payload)
    persist(merge_phase(fixture, "bounds", payload))
    return payload


def run_llama_compare(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison = matched_comparison()
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "llama-compare",
        "mode": mode,
        "ok": True,
        "llama_comparison": comparison,
        "family_plan": family_plan(mode, "llama-compare"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "llama-compare.json", payload)
    persist(merge_phase(current_fixture(), "llama-compare", payload))
    return payload


def run_mechanisms(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    table = mechanism_disposition()
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "mechanisms",
        "mode": mode,
        "ok": True,
        "mechanism_disposition": table,
        "family_plan": family_plan(mode, "mechanisms"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "mechanisms.json", payload)
    persist(merge_phase(current_fixture(), "mechanisms", payload))
    return payload


def run_identity_timeline(
    mode: str,
    run_dir: Path,
    runner: NativeRunner | None,
    *,
    skip_gpu: bool = False,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    available, blocker = gpu_available()
    if skip_gpu:
        available = False
        blocker = blocker or "skip_gpu"
    execute = runner
    if execute is None and available:
        execute = default_native_runner
    timeline: dict[str, Any] = {}
    native_identity: dict[str, Any] = {}
    hardware = False
    if execute is None:
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            timeline[name] = {
                "valid": False,
                "reason": blocker or "gpu_unavailable",
                "prefix": prefix,
                "identity": "unmeasured",
            }
    else:
        identity_cmd = execute(
            [f"./{NATIVE}", "--workload", "identity", MODEL],
            "screen",
        )
        text = identity_cmd.stdout + identity_cmd.stderr
        (run_dir / "identity.txt").write_text(text, encoding="utf-8")
        native_identity = parse_prefixed(identity_cmd.stdout, RESULT_PREFIX)
        hardware = True
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            completed = execute(
                [
                    f"./{NATIVE}",
                    "--workload",
                    "combined-timeline",
                    "--prefix",
                    str(prefix),
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--tokens",
                    "1",
                    MODEL,
                ],
                "screen",
            )
            raw = completed.stdout + completed.stderr
            (run_dir / f"timeline-d{prefix}.txt").write_text(raw, encoding="utf-8")
            summary = parse_prefixed(completed.stdout, RESULT_PREFIX)
            records = parse_opt124_records(completed.stdout)
            if not records:
                records = parse_records_blob(
                    completed.stdout.replace(
                        "QW38_OPT124_RECORDS=", "QW38_OPT115_RECORDS="
                    )
                )
            dump_json(run_dir / f"timeline-d{prefix}-records.json", records)
            split = classify_timeline(records, wall_ms=summary.get("wall_ms"), tokens=1)
            split["valid"] = bool(records) and not bool(summary.get("pool_overflow"))
            split["prefix"] = prefix
            split["identity"] = "combined_opt118_opt119"
            split["native"] = summary
            timeline[name] = split
    fixture = current_fixture()
    bounds = recompute_bounds(timeline=timeline)
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "identity-timeline",
        "mode": mode,
        "ok": True,
        "hardware_executed": hardware,
        "gpu_blocker": None if hardware else blocker,
        "native_identity": native_identity,
        "timeline": timeline,
        "bounds": bounds,
        "family_plan": family_plan(mode, "identity-timeline"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "identity-timeline.json", payload)
    persist(merge_phase(fixture, "identity-timeline", payload))
    persist(merge_phase(current_fixture(), "bounds", {"bounds": bounds}))
    return payload


def run_conclusions(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    bounds = fixture.get("bounds") or recompute_bounds(timeline=fixture.get("timeline"))
    comparison = fixture.get("llama_comparison") or matched_comparison()
    mechanisms = fixture.get("mechanism_disposition") or mechanism_disposition()
    conclusions = supported_conclusions(
        bounds=bounds,
        comparison=comparison,
        mechanisms=mechanisms,
        timeline=fixture.get("timeline"),
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "conclusions",
        "mode": mode,
        "ok": True,
        "conclusions": conclusions,
        "stop_reopen_criteria": stop_reopen_criteria(),
        "family_plan": family_plan(mode, "conclusions"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "conclusions.json", payload)
    persist(merge_phase(fixture, "conclusions", payload))
    return payload


def run_report(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    if not fixture.get("bounds"):
        fixture = merge_phase(fixture, "bounds", {"bounds": recompute_bounds()})
    if not fixture.get("llama_comparison"):
        fixture = merge_phase(
            fixture, "llama-compare", {"llama_comparison": matched_comparison()}
        )
    if not fixture.get("mechanism_disposition"):
        fixture = merge_phase(
            fixture, "mechanisms", {"mechanism_disposition": mechanism_disposition()}
        )
    if not fixture.get("conclusions"):
        conclusions = supported_conclusions(
            bounds=fixture["bounds"],
            comparison=fixture["llama_comparison"],
            mechanisms=fixture["mechanism_disposition"],
            timeline=fixture.get("timeline"),
        )
        fixture = merge_phase(
            fixture,
            "conclusions",
            {
                "conclusions": conclusions,
                "stop_reopen_criteria": stop_reopen_criteria(),
            },
        )
    if not fixture.get("opt123_identities_match"):
        fixture["opt123_identities_match"] = identities_match()
    if not fixture.get("identity"):
        fixture["identity"] = sitting_identity()
    fixture["status"] = "analysis"
    fixture["universal_optimality_claimed"] = False
    fixture["production_kept"] = True
    fixture["claims_throughput"] = False
    fixture["claims_performance_improvement"] = False
    fixture["report_path"] = "evidence/optimization/opt124-practical-ceiling/REPORT.md"
    fixture["proof_limit"] = PROOF
    fixture["measurement_utc"] = utc_now()
    validate_fixture(fixture)
    persist(fixture)
    payload = {
        "schema_version": 1,
        "task": "OPT-124",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "report_path": str(REPORT.relative_to(ROOT)),
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "report.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument(
        "--run-dir",
        default=str(ROOT / "build/optimization-runs/opt124"),
    )
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if args.phase == "freeze":
        run_freeze(args.mode, run_dir)
    elif args.phase == "bounds":
        run_bounds(args.mode, run_dir)
    elif args.phase == "llama-compare":
        run_llama_compare(args.mode, run_dir)
    elif args.phase == "mechanisms":
        run_mechanisms(args.mode, run_dir)
    elif args.phase == "identity-timeline":
        run_identity_timeline(args.mode, run_dir, None, skip_gpu=bool(args.skip_gpu))
    elif args.phase == "conclusions":
        run_conclusions(args.mode, run_dir)
    elif args.phase == "report":
        run_report(args.mode, run_dir)
    print(json.dumps({"task": "OPT-124", "phase": args.phase, "ok": True}))
    return 0


if __name__ == "__main__":
    try:
        validate_future_keep_policy("OPT-124", load_iteration())
        raise SystemExit(main())
    except CeilingError as exc:
        print(json.dumps({"task": "OPT-124", "ok": False, "error": str(exc)}))
        raise SystemExit(1) from exc
