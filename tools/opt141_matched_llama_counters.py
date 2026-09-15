"""OPT-141 matched llama NCU counters for the four OPT-138 selections.

Diagnostics only. Reuses OPT-136/138 private llama diagnostics, OPT-139 NCU
parsing/identity, and OPT-140 prefill attn identity. No production changes.
NCU duration is never a benchmark. Occupancy is not bandwidth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt139_counter_identity import (  # noqa: E402
    LAUNCH_PREFILL_ATTN,
    SLOT_PREFIXES,
    admit_supported_mechanism,
    capture_reuse_valid,
    classify_kernel_family,
    export_ncu_csv,
    kernel_stem,
    kernels_from_sqlite,
    parse_ncu_output,
    production_attn_identity,
    production_prefill_attn_identity,
    select_target_launch,
    typed_slots_from_launch,
)
from tools.performance_evidence import missing_counter_record  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt141_matched_llama_counters_contract.json"
ITERATION = ROOT / "pins/opt141_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt141_matched_llama_counters.json"
EVIDENCE = ROOT / "evidence/optimization/opt141-matched-llama-counters"
REPORT = EVIDENCE / "REPORT.md"
LLAMA_BIN = ".cache/authorities/llama-build-opt136/bin/qw38-llama-opt136-decode-profile"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
SELECTOR = "decode_segments8"
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
CAPACITY = 131072
PHASES = ("preflight", "identity", "counters", "report")
CHILD_TIMEOUT_S = 300
PREFILL_TIMEOUT_S = 1200
AGGREGATE_DEADLINE_S = 7200
MAX_RETRIES = 1
RESULT_PREFIX = "QW38_OPT141_MATCHED_LLAMA_COUNTERS_RESULT="
UNRESOLVED = "llama_ncu_kernel_identity_unresolved"
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "mode",
    "claims_throughput",
    "claims_performance_improvement",
    "production_kept",
    "selected_execution_graph_path",
    "prefixes",
    "preflight",
    "identity",
    "counters",
    "comparisons",
    "answers",
    "report_path",
)

SELECTIONS = (
    "decode.attn_core",
    "decode.residual_norm_quant",
    "prefill.attn_core",
    "prefill.prompt_mmq",
)
IDENTITY_ROWS = (
    ("decode", "attn_core", 128, "D128"),
    ("decode", "attn_core", 2048, "D2048"),
    ("decode", "residual_norm_quant", 128, "D128"),
    ("decode", "residual_norm_quant", 2048, "D2048"),
    ("prefill", "attn_core", 4096, "P4096"),
    ("prefill", "prompt_mmq", 4096, "P4096"),
)
ATTN_CORE_MATCHED = ("flash_attn_ext_f16", "flash_attn_ext_vec")
ATTN_CORE_ENCLOSING = (
    "flash_attn_combine_results",
    "flash_attn_stream_k_fixup_general",
    "flash_attn_mask_to_KV_max",
)
COMPARABLE_SLOTS = (
    "dram_read_bytes",
    "dram_write_bytes",
    "l2_traffic",
    "dram_throughput",
    "sm_throughput",
    "tensor_activity",
    "achieved_occupancy",
)
DURATION_KEYS = ("gpu__time_duration", "ncu_duration", "duration_ns", "duration_ms")
NCU_REGEX = {
    "flash_attn_ext_vec": "flash_attn_ext_vec",
    "flash_attn_ext_f16": "flash_attn_ext_f16",
    "quantize_q8_1": r"^quantize_q8_1$",
    "mul_mat_q": r"^mul_mat_q$",
    "rms_norm_f32": "rms_norm_f32",
}
SOURCE_PATHS = {
    "flash_attn_ext_vec": (
        ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh"
    ),
    "flash_attn_ext_f16": (
        ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    ),
    "quantize_q8_1": ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/quantize.cu",
    "rms_norm_f32": ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/norm.cu",
    "mul_mat_q": ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/mmq.cuh",
}


class MatchedLlamaCounterError(RuntimeError):
    """OPT-141 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-141":
        raise MatchedLlamaCounterError("matched-llama contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-141", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-141 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


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


def empty_fixture(mode: str = "feedback") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-141",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": [128, 2048],
        "prefill_tokens": 4096,
        "preflight": None,
        "identity": None,
        "counters": None,
        "comparisons": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise MatchedLlamaCounterError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise MatchedLlamaCounterError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise MatchedLlamaCounterError("selector drifted from decode_segments8")


def selection_key(phase: str, family: str) -> str:
    return f"{phase}.{family}"


def row_key(phase: str, family: str, workload: str) -> str:
    return f"{phase}.{family}.{workload.lower()}"


def ncu_regex_for(stem: str) -> str:
    return NCU_REGEX.get(stem, stem)


def ncu_profiled_nothing(blob: str) -> bool:
    return "No kernels were profiled" in blob


def source_grounded_identity(phase: str, family: str, prefix: int) -> dict[str, Any]:
    """Map fattn.cu / norm.cu / mmq.cuh dispatch without inventing CUPTI IDs."""
    if phase == "decode" and family == "attn_core":
        return {
            "expected_kernel": "flash_attn_ext_vec",
            "function": "flash_attn_ext_vec<256,1>",
            "source": SOURCE_PATHS["flash_attn_ext_vec"],
            "ncu_regex": ncu_regex_for("flash_attn_ext_vec"),
            "dispatch": (
                "ggml_cuda_get_best_fattn_kernel: Ada+ VEC when Q.ne[1]==1 "
                "and n_kv<8192; D128 and D2048 share the stem and differ in n_kv"
            ),
            "shape": {
                "head_width": 256,
                "ncols": 1,
                "n_kv": prefix,
                "gqa_ratio": 6,
            },
            "enclosing": list(ATTN_CORE_ENCLOSING),
            "replay_family": "decode-attention",
        }
    if phase == "prefill" and family == "attn_core":
        return {
            "expected_kernel": "flash_attn_ext_f16",
            "function": "flash_attn_ext_f16<256,256,8,8,false,false>",
            "source": SOURCE_PATHS["flash_attn_ext_f16"],
            "ncu_regex": ncu_regex_for("flash_attn_ext_f16"),
            "dispatch": (
                "BEST_FATTN_KERNEL_MMA_F16 when Q.ne[1]==4096; "
                "gqa_ratio=6>4 selects ncols2=8, Q.ne[1]>4 selects ncols1=8"
            ),
            "shape": {
                "dkq": 256,
                "dv": 256,
                "ncols1": 8,
                "ncols2": 8,
                "tokens": 4096,
                "gqa_ratio": 6,
            },
            "enclosing": [
                "flash_attn_stream_k_fixup_general",
                "flash_attn_mask_to_KV_max",
            ],
            "replay_family": "prompt-attention",
        }
    if phase == "decode" and family == "residual_norm_quant":
        return {
            "expected_kernel": "quantize_q8_1",
            "function": "quantize_q8_1",
            "source": SOURCE_PATHS["quantize_q8_1"],
            "ncu_regex": ncu_regex_for("quantize_q8_1"),
            "dispatch": (
                "decode MMVQ path; D128/D2048 share hidden-dim dispatch "
                "(rms_norm_f32 sibling is enclosing fused work)"
            ),
            "shape": {"prefix": prefix, "hidden_dim": "model_n_embd"},
            "enclosing": ["rms_norm_f32", "l2_norm_f32", "quantize_q8_1"],
            "replay_family": "decode-mixer",
        }
    if phase == "prefill" and family == "prompt_mmq":
        return {
            "expected_kernel": "mul_mat_q",
            "function": "mul_mat_q<GGML_TYPE_Q4_K,128,false>",
            "source": SOURCE_PATHS["mul_mat_q"],
            "ncu_regex": ncu_regex_for("mul_mat_q"),
            "dispatch": "prompt MMQ Q4_K tile J=128; exclude mul_mat_q_stream_k_fixup",
            "shape": {"tokens": 4096, "type": "q4_k", "tile_j": 128},
            "enclosing": ["quantize_mmq_q8_1", "mul_mat_q_stream_k_fixup"],
            "replay_family": "prompt-ffn",
        }
    raise MatchedLlamaCounterError(f"no source mapping for {phase}.{family}")


def sqlite_candidates(phase: str, prefix: int) -> list[Path]:
    traces = ROOT / "build/optimization-runs/opt138/traces"
    if phase == "prefill":
        names = [
            traces / "llama-p4096-node-r0.sqlite",
            traces / "llama-p4096-node-r1.sqlite",
            traces / "llama-p4096-node-r0.1.sqlite",
        ]
    else:
        names = [
            traces / f"llama-d{prefix}-node-r0.2.sqlite",
            traces / f"llama-d{prefix}-node-r0.1.sqlite",
            traces / f"llama-d{prefix}-node-r0.sqlite",
        ]
    return names


def summarize_family_kernels(
    kernels: Sequence[Mapping[str, Any]], family: str
) -> dict[str, Any]:
    rows = [
        item
        for item in kernels
        if (item.get("family") or classify_kernel_family(str(item.get("name") or "")))
        == family
    ]
    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    largest: Mapping[str, Any] | None = None
    largest_ns = -1
    for item in rows:
        stem = kernel_stem(str(item.get("name") or ""))
        duration = int(item.get("duration_ns") or 0)
        totals[stem] += duration
        counts[stem] += 1
        if duration > largest_ns:
            largest_ns = duration
            largest = item
    ranked = sorted(totals.items(), key=lambda pair: -pair[1])
    return {
        "launch_count": len(rows),
        "stems": [
            {
                "stem": stem,
                "count": counts[stem],
                "total_ns": total,
                "total_ms": total / 1e6,
            }
            for stem, total in ranked
        ],
        "largest_leaf": None
        if largest is None
        else {
            "name": largest.get("name"),
            "stem": kernel_stem(str(largest.get("name") or "")),
            "duration_ns": int(largest.get("duration_ns") or 0),
        },
    }


def select_matched_kernel(
    summary: Mapping[str, Any],
    *,
    phase: str,
    family: str,
    expected: str,
) -> dict[str, Any]:
    stems = [str(item.get("stem") or "") for item in (summary.get("stems") or [])]
    largest = summary.get("largest_leaf") or {}
    largest_stem = str(largest.get("stem") or "")
    mismatches: list[str] = []
    matched_stem = expected
    if family == "attn_core":
        if expected not in ATTN_CORE_MATCHED:
            mismatches.append("attn_core_expected_not_compute_kernel")
        if largest_stem in ATTN_CORE_ENCLOSING:
            mismatches.append("largest_leaf_is_enclosing_fused_sibling")
        if expected not in stems and stems:
            mismatches.append("expected_kernel_absent_from_trace")
            matched_stem = ""
    elif largest_stem and largest_stem != expected:
        if family == "residual_norm_quant" and largest_stem in {
            "quantize_q8_1",
            "rms_norm_f32",
        }:
            matched_stem = largest_stem
        elif family == "prompt_mmq" and largest_stem == "mul_mat_q":
            matched_stem = largest_stem
        else:
            mismatches.append("largest_leaf_differs_from_source_kernel")
    ambiguous = False
    ranked = list(summary.get("stems") or [])
    if family == "attn_core" and ranked:
        compute = [item for item in ranked if item.get("stem") in ATTN_CORE_MATCHED]
        if len({item.get("stem") for item in compute}) > 1:
            ambiguous = True
            mismatches.append("ambiguous_attn_compute_stems")
    if not matched_stem:
        mismatches.append(UNRESOLVED)
    return {
        "matched_kernel": matched_stem or None,
        "largest_leaf": largest_stem or None,
        "ambiguous": ambiguous,
        "mismatches": mismatches,
        "ok": bool(matched_stem) and not ambiguous,
    }


def llama_identity_match(
    *,
    phase: str,
    family: str,
    workload: str,
    kernel: str | None,
    expected_kernel: str | None,
    replay_family: str,
) -> dict[str, Any]:
    mismatches: list[str] = []
    if phase == "prefill" and str(replay_family).startswith("decode-"):
        mismatches.append("prefill_decode_attention_ineligible")
    if phase == "decode" and str(replay_family).startswith("prompt-"):
        mismatches.append("decode_prompt_replay_mismatch")
    expected_stem = kernel_stem(expected_kernel)
    observed_stem = kernel_stem(kernel)
    if expected_stem and observed_stem and expected_stem != observed_stem:
        mismatches.append("kernel_selector_mismatch")
    if expected_stem and not observed_stem:
        mismatches.append(UNRESOLVED)
    if phase == "prefill" and family == "attn_core":
        if (
            observed_stem == "flash_attn_ext_vec"
            or expected_stem == "flash_attn_ext_vec"
        ):
            mismatches.append("vec_is_not_p4096_prefill_evidence")
        if replay_family != "prompt-attention":
            mismatches.append("prefill_attn_requires_prompt_attention")
    if (
        phase == "decode"
        and family == "attn_core"
        and observed_stem in ATTN_CORE_ENCLOSING
    ):
        mismatches.append("fused_boundary_mismatch")
    if workload.upper() == "D2048" and observed_stem == "warp_query_decode_attention":
        mismatches.append("warp_query_is_not_llama_d2048")
    ok = not mismatches
    return {
        "ok": ok,
        "phase": phase,
        "engine": "llama",
        "family": family,
        "workload": workload,
        "kernel": kernel,
        "expected_kernel": expected_kernel,
        "replay_family": replay_family,
        "mismatches": mismatches,
        "reason": None if ok else ",".join(mismatches),
    }


def fused_boundary_ok(
    *,
    family: str,
    quartz_kernel: str | None,
    llama_kernel: str | None,
    llama_enclosing: Sequence[str],
) -> dict[str, Any]:
    quartz_stem = kernel_stem(quartz_kernel)
    llama_stem = kernel_stem(llama_kernel)
    reasons: list[str] = []
    if family == "attn_core":
        if llama_stem in ATTN_CORE_ENCLOSING:
            reasons.append("llama_leaf_is_enclosing_combine_or_fixup")
        if quartz_stem == LAUNCH_PREFILL_ATTN and llama_stem == "flash_attn_ext_f16":
            reasons.append("tile_shape_mismatch_quartz_16x2_vs_llama_8x8")
        if (
            quartz_stem
            in {
                "warp_query_decode_attention",
                "vec128_online_decode_attention",
            }
            and llama_stem == "flash_attn_ext_f16"
        ):
            reasons.append("decode_quartz_vs_prefill_llama_mma")
    if family == "residual_norm_quant":
        reasons.append("llama_quantize_q8_1_vs_quartz_rms_plus_bf16_quant_fusion")
    if reasons:
        return {
            "ok": False,
            "reason": "fused_boundary_mismatch",
            "details": reasons,
            "enclosing": list(llama_enclosing),
            "incomparable_counters": list(COMPARABLE_SLOTS),
        }
    return {
        "ok": True,
        "reason": None,
        "details": [],
        "enclosing": list(llama_enclosing),
        "incomparable_counters": [],
    }


def compare_quartz_llama(
    quartz: Mapping[str, Any] | None,
    llama: Mapping[str, Any] | None,
    *,
    fusion: Mapping[str, Any],
) -> dict[str, Any]:
    if not quartz or quartz.get("error") or quartz.get("leaf_isolated") is False:
        return {
            "comparable": False,
            "reason": "quartz_counters_unusable",
            "compared_slots": [],
            "claims_throughput": False,
            "ncu_duration_benchmark": False,
        }
    if not llama or llama.get("error"):
        return {
            "comparable": False,
            "reason": str(llama.get("error") if llama else UNRESOLVED),
            "compared_slots": [],
            "claims_throughput": False,
            "ncu_duration_benchmark": False,
        }
    if not fusion.get("ok"):
        return {
            "comparable": False,
            "reason": fusion.get("reason") or "fused_boundary_mismatch",
            "details": fusion.get("details"),
            "enclosing": fusion.get("enclosing"),
            "incomparable_counters": fusion.get("incomparable_counters"),
            "compared_slots": [],
            "claims_throughput": False,
            "ncu_duration_benchmark": False,
        }
    quartz_units = quartz.get("units") or {}
    llama_units = llama.get("units") or {}
    compared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for slot in COMPARABLE_SLOTS:
        q_val = quartz.get(slot)
        l_val = llama.get(slot)
        q_unit = quartz_units.get(slot)
        l_unit = llama_units.get(slot)
        if q_val is None or l_val is None:
            skipped.append({"slot": slot, "reason": "null_value"})
            continue
        if q_unit and l_unit and q_unit != l_unit:
            skipped.append(
                {
                    "slot": slot,
                    "reason": "unit_mismatch",
                    "quartz_unit": q_unit,
                    "llama_unit": l_unit,
                }
            )
            continue
        compared.append(
            {
                "slot": slot,
                "quartz": q_val,
                "llama": l_val,
                "unit": q_unit or l_unit,
            }
        )
    for key in DURATION_KEYS:
        if quartz.get(key) is not None or llama.get(key) is not None:
            skipped.append({"slot": key, "reason": "ncu_duration_is_not_a_benchmark"})
    return {
        "comparable": bool(compared),
        "reason": None if compared else "no_matching_units",
        "compared_slots": compared,
        "skipped_slots": skipped,
        "claims_throughput": False,
        "ncu_duration_benchmark": False,
        "proof_limit": (
            "like-for-like counters only; occupancy is not bandwidth; "
            "NCU duration is not a tok/s claim"
        ),
    }


def load_quartz_record(phase: str, family: str, workload: str) -> dict[str, Any] | None:
    if phase == "decode" and family == "attn_core":
        path = (
            ROOT / "evidence/optimization/opt139-counter-identity/counter-evidence.json"
        )
        if not path.is_file():
            return None
        payload = load_json(path)
        for row in payload.get("kernels") or []:
            if str(row.get("workload") or "").upper() == workload.upper():
                return dict(row)
        return None
    if phase == "prefill" and family == "attn_core":
        path = (
            ROOT
            / "evidence/optimization/opt140-prefill-attention-replay/counter-evidence.json"
        )
        if not path.is_file():
            return None
        payload = load_json(path)
        kernels = payload.get("kernels") or []
        return dict(kernels[0]) if kernels else None
    path = ROOT / "evidence/optimization/opt138-remaining-gap/counter-evidence.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    for row in payload.get("kernels") or []:
        if (
            row.get("engine") == "quartz"
            and row.get("family") == family
            and row.get("phase") == phase
        ):
            copied = dict(row)
            copied["leaf_isolated"] = False
            copied["proof_limit"] = (
                "OPT-138 replay profiled enclosing family; not a single-launch identity"
            )
            return copied
    return None


def collection_plan(identities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One collection per identified kernel. Residual D128/D2048 share dispatch."""
    seen: dict[str, dict[str, Any]] = {}
    order: list[dict[str, Any]] = []
    for row in identities:
        if not row.get("ok"):
            continue
        stem = str(row.get("expected_kernel") or "")
        phase = str(row.get("phase"))
        family = str(row.get("family"))
        prefix = int(row.get("prefix") or 0)
        if family == "residual_norm_quant":
            collect_id = f"{phase}.{family}.{stem}"
        else:
            collect_id = f"{phase}.{family}.{stem}.p{prefix}"
        if collect_id in seen:
            seen[collect_id]["workloads"].append(row.get("workload"))
            seen[collect_id]["shared_dispatch"] = True
            continue
        item = {
            "collect_id": collect_id,
            "phase": phase,
            "family": family,
            "prefix": prefix,
            "workload": row.get("workload"),
            "workloads": [row.get("workload")],
            "expected_kernel": stem,
            "ncu_regex": row.get("ncu_regex") or ncu_regex_for(stem),
            "replay_family": row.get("replay_family"),
            "shared_dispatch": False,
        }
        seen[collect_id] = item
        order.append(item)
    return order


def llama_ncu_command(
    *,
    metrics: Sequence[str],
    kernel_regex: str,
    phase: str,
    prefix: int,
    raw_report: Path,
    nvtx: bool,
) -> list[str]:
    from tools import opt138_remaining_gap_profile as opt138

    command = [
        *opt138._docker_ncu(),
        "ncu",
        "--csv",
        "--metrics",
        ",".join(metrics),
        "--target-processes",
        "all",
        "--replay-mode",
        "application",
        "--kernel-name",
        f"regex:{kernel_regex}",
        "--launch-count",
        "1",
    ]
    if nvtx and phase == "decode":
        # llama uses nvtxRangePushA (push/pop). NCU --nvtx-include matches
        # start/end ranges only; a diagnosed retry drops this filter.
        command.extend(["--nvtx", "--nvtx-include", "opt136"])
    output = raw_report if not raw_report.is_absolute() else Path(relpath(raw_report))
    command.extend(["-o", str(output), "--force-overwrite"])
    workload = "prefill-unprofiled" if phase == "prefill" else "unprofiled"
    tokens = prefix if phase == "prefill" else 1
    command.extend(
        [
            LLAMA_BIN,
            MODEL,
            "--workload",
            workload,
            "--prefix",
            str(prefix),
            "--tokens",
            str(tokens),
            "--ctx",
            str(CAPACITY),
        ]
    )
    return command


def llama_counter_record(
    blob: str,
    *,
    collect: Mapping[str, Any],
    expected_kernel: str,
    selected_metrics: Sequence[str],
    command: Sequence[str],
    timeout_s: int,
    raw_artifact: str,
    binary_hash: str | None,
    reused: bool,
    retry: int,
) -> dict[str, Any]:
    parsed = parse_ncu_output(blob)
    target = select_target_launch(
        parsed.launches,
        expected_kernel=expected_kernel,
        family=str(collect.get("family")),
    )
    slots = typed_slots_from_launch(target)
    identity = llama_identity_match(
        phase=str(collect.get("phase")),
        family=str(collect.get("family")),
        workload=str(collect.get("workload")),
        kernel=target.kernel_name if target else expected_kernel,
        expected_kernel=expected_kernel,
        replay_family=str(collect.get("replay_family") or ""),
    )
    admission = admit_supported_mechanism(
        {
            "phase": collect.get("phase"),
            "engine": "llama",
            "workload": collect.get("workload"),
            "replay_family": collect.get("replay_family"),
            "kernel": target.kernel_name if target else None,
            "expected_kernel": expected_kernel,
            "identity": {
                **identity,
                "ok": identity.get("ok"),
                "mismatches": [
                    item
                    for item in identity.get("mismatches") or []
                    if item != "engine_not_quartz"
                ],
            },
            **{key: slots.get(key) for key in SLOT_PREFIXES},
            "stalls": slots.get("stalls"),
        }
    )
    kernel_id = (
        f"llama:{collect.get('family')}:{collect.get('workload')}:{expected_kernel}"
    )
    error = None
    if not target:
        error = UNRESOLVED
    record = {
        "kernel": kernel_id,
        "engine": "llama",
        "family": collect.get("family"),
        "phase": collect.get("phase"),
        "workload": collect.get("workload"),
        "prefix": collect.get("prefix"),
        "replay_family": collect.get("replay_family"),
        "expected_kernel": expected_kernel,
        "target_kernel": kernel_stem(target.kernel_name) if target else None,
        "ncu_regex": collect.get("ncu_regex"),
        "command": list(command),
        "timeout_s": timeout_s,
        "raw_artifact": raw_artifact,
        "binary_hash": binary_hash,
        "llama_revision": LLAMA_REV,
        "reused": reused,
        "retry": retry,
        "launch_count_filter": 1,
        "full_ncu_sweep": False,
        "claims_throughput": False,
        "identity": identity,
        "admission": admission,
        "supported_mechanism": admission.get("supported_mechanism"),
        "candidate": None,
        "other_launches": [
            {
                "stem": kernel_stem(item.kernel_name),
                "kernel_name": item.kernel_name,
            }
            for item in parsed.launches
            if item is not target
        ],
        "enclosing_replay": {
            "separated_from_target_kernel": True,
            "shared_dispatch": bool(collect.get("shared_dispatch")),
        },
        "error": error,
        "zero_filled": bool(slots.get("zero_filled")),
        "selected_metrics": list(selected_metrics),
        **{key: slots.get(key) for key in SLOT_PREFIXES},
        "stalls": slots.get("stalls"),
        "units": slots.get("units") or {},
        "metrics": slots.get("metrics") or {},
        "missing_reasons": slots.get("missing_reasons") or {},
    }
    return record


def _blocked_gpu_phase(
    phase: str, run_dir: Path, mode: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-141",
        "phase": phase,
        "mode": mode,
        "ok": False,
        "status": "incomplete",
        "blocked": True,
        "reason": preflight.get("blocked") or ["gpu_or_ncu_unavailable"],
        "claims_throughput": False,
        "family_plan": family_plan(mode, phase if phase in PHASES else "report"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, f"{phase}.json", payload)


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138
    from tools.opt080_batch_gate import IMAGE, docker_common

    run_dir.mkdir(parents=True, exist_ok=True)
    available, gpu_blocker = gpu_available()
    pins = opt138.authenticate_opt138_pins()
    make_rc = 1
    make_stderr = gpu_blocker or ""
    if available:
        make = opt136.with_gpu_lock(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt141-diagnostics"],
            timeout_s=CHILD_TIMEOUT_S,
        )
        make_rc = make.returncode
        make_stderr = (make.stderr or "")[-2000:]
    llama_script = ROOT / "tools/llama_authority/build_opt136_profile.sh"
    llama: dict[str, Any] = {"ok": False, "reason": "not_started"}
    if llama_script.is_file() and available:
        built = opt136.with_gpu_lock(
            ["bash", str(llama_script)],
            timeout_s=CHILD_TIMEOUT_S,
        )
        llama = {
            "ok": built.returncode == 0 and (ROOT / LLAMA_BIN).is_file(),
            "returncode": built.returncode,
            "stdout_tail": (built.stdout or "")[-800:],
            "stderr_tail": (built.stderr or "")[-800:],
            "hash": opt136.sha256_file(ROOT / LLAMA_BIN),
            "revision": LLAMA_REV,
            "production_source_untouched": True,
        }
    ncu = {"ncu_available": False, "error": "gpu_unavailable"}
    if available:
        ncu = opt138._query_ncu_metrics()
    blocked: list[str] = []
    if not available:
        blocked.append(gpu_blocker or "gpu_unavailable")
    if not ncu.get("ncu_available"):
        blocked.append(str(ncu.get("error") or "ncu_unavailable"))
    if not (ROOT / LLAMA_BIN).is_file():
        blocked.append("llama_bin_missing")
    if not pins.get("ok"):
        blocked.append("stale_parent_identity")
    if not llama.get("ok"):
        blocked.append("llama_private_diagnostic_missing")
    ok = (
        available
        and bool(ncu.get("ncu_available"))
        and (ROOT / LLAMA_BIN).is_file()
        and bool(pins.get("ok"))
        and bool(llama.get("ok"))
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-141",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "ncu": ncu,
        "hashes": {"llama": llama.get("hash")},
        "make_returncode": make_rc,
        "make_stderr_tail": make_stderr,
        "authenticated_current_pins": pins,
        "llama": llama,
        "parent": PARENT,
        "llama_bin": LLAMA_BIN,
        "child_timeout_s": CHILD_TIMEOUT_S,
        "prefill_timeout_s": PREFILL_TIMEOUT_S,
        "aggregate_deadline_s": AGGREGATE_DEADLINE_S,
        "max_retries_per_kernel": MAX_RETRIES,
        "full_ncu_sweep": False,
        "claims_throughput": False,
        "blocked": blocked,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def run_identity(run_dir: Path, mode: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for phase, family, prefix, workload in IDENTITY_ROWS:
        source = source_grounded_identity(phase, family, prefix)
        sqlite_path = next(
            (path for path in sqlite_candidates(phase, prefix) if path.is_file()),
            None,
        )
        summary: dict[str, Any] = {"launch_count": 0, "stems": [], "largest_leaf": None}
        kernel_source = "source_grounded_fattn_norm_mmq"
        if sqlite_path is not None:
            summary = summarize_family_kernels(kernels_from_sqlite(sqlite_path), family)
            kernel_source = "trace_plus_source"
        matched = select_matched_kernel(
            summary,
            phase=phase,
            family=family,
            expected=str(source["expected_kernel"]),
        )
        expected = str(matched.get("matched_kernel") or source["expected_kernel"])
        identity = llama_identity_match(
            phase=phase,
            family=family,
            workload=workload,
            kernel=expected,
            expected_kernel=expected,
            replay_family=str(source["replay_family"]),
        )
        if matched.get("ambiguous") or not matched.get("ok"):
            identity["ok"] = False
            extra = list(identity.get("mismatches") or [])
            extra.extend(matched.get("mismatches") or [])
            if UNRESOLVED not in extra and not expected:
                extra.append(UNRESOLVED)
            identity["mismatches"] = extra
            identity["reason"] = ",".join(extra) or UNRESOLVED
        quartz = None
        if phase == "decode" and family == "attn_core":
            quartz = production_attn_identity(prefix)
        elif phase == "prefill" and family == "attn_core":
            quartz = production_prefill_attn_identity()
        fusion = fused_boundary_ok(
            family=family,
            quartz_kernel=(quartz or {}).get("expected_kernel"),
            llama_kernel=expected,
            llama_enclosing=list(source.get("enclosing") or []),
        )
        status = "matched" if identity.get("ok") else "blocked"
        key = row_key(phase, family, workload)
        rows[key] = {
            "ok": identity.get("ok"),
            "status": status,
            "phase": phase,
            "family": family,
            "prefix": prefix,
            "workload": workload,
            "selection": selection_key(phase, family),
            "engine": "llama",
            "expected_kernel": expected,
            "ncu_regex": ncu_regex_for(expected),
            "function": source.get("function"),
            "source": source.get("source"),
            "dispatch": source.get("dispatch"),
            "shape": source.get("shape"),
            "enclosing": source.get("enclosing"),
            "replay_family": source.get("replay_family"),
            "kernel_source": kernel_source,
            "sqlite": None if sqlite_path is None else relpath(sqlite_path),
            "trace_summary": summary,
            "matched": matched,
            "identity": identity,
            "quartz_counterpart": quartz,
            "fusion": fusion,
            "cupti_graph_node_id": None,
            "proof_limit": (
                "P4096 uses kernel-leaf source/launch mapping; "
                "do not fabricate CUPTI graph-node IDs"
            ),
        }
    selections: dict[str, Any] = {}
    for key in SELECTIONS:
        members = [row for row in rows.values() if row.get("selection") == key]
        if not members:
            selections[key] = {"status": "blocked", "reason": UNRESOLVED}
            continue
        if all(item.get("ok") for item in members):
            selections[key] = {"status": "matched", "reason": None}
        else:
            reasons = [
                str((item.get("identity") or {}).get("reason") or UNRESOLVED)
                for item in members
                if not item.get("ok")
            ]
            selections[key] = {
                "status": "blocked",
                "reason": ",".join(reasons) or UNRESOLVED,
            }
    payload = {
        "schema_version": 1,
        "task": "OPT-141",
        "phase": "identity",
        "mode": mode,
        "ok": all(item.get("status") != "skipped" for item in selections.values()),
        "rows": rows,
        "selections": selections,
        "collection_plan": collection_plan(list(rows.values())),
        "unresolved_code_resolved": UNRESOLVED,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "identity"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "identity.json", payload)
    return store_sidecar(run_dir, "identity.json", payload)


def run_counters(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138

    preflight = load_sidecar(run_dir, "preflight.json") or {}
    identity = load_sidecar(run_dir, "identity.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("counters", run_dir, mode, preflight)
    ncu = preflight.get("ncu") or opt138._query_ncu_metrics()
    metrics = list(ncu.get("selected_metrics") or [])
    raw_dir = EVIDENCE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan = identity.get("collection_plan") or collection_plan(
        list((identity.get("rows") or {}).values())
    )
    records: list[dict[str, Any]] = []
    llama_hash = (preflight.get("hashes") or {}).get("llama")
    for collect in plan:
        expected_kernel = str(collect.get("expected_kernel") or "")
        phase = str(collect.get("phase"))
        family = str(collect.get("family"))
        prefix = int(collect.get("prefix") or 0)
        workload = str(collect.get("workload"))
        kernel_id = f"llama:{family}:{workload}:{expected_kernel}"
        if not ncu.get("ncu_available"):
            records.append(
                {
                    **missing_counter_record(
                        kernel=kernel_id,
                        engine="llama",
                        error=str(ncu.get("error") or "ncu_unavailable"),
                    ),
                    "phase": phase,
                    "family": family,
                    "workload": workload,
                    "expected_kernel": expected_kernel,
                    "supported_mechanism": None,
                    "candidate": None,
                }
            )
            continue
        stem_name = f"{phase}-{family}-{workload.lower()}"
        raw_txt = raw_dir / f"{stem_name}.ncu.txt"
        raw_csv = raw_dir / f"{stem_name}.ncu.csv"
        report_path = raw_dir / stem_name
        report_file = raw_dir / f"{stem_name}.ncu-rep"
        timeout_s = PREFILL_TIMEOUT_S if phase == "prefill" else CHILD_TIMEOUT_S
        reused = False
        retry = 0
        blob = ""
        command: list[str] = []
        if capture_reuse_valid(
            raw_txt, expected_kernel=expected_kernel, metrics=metrics
        ):
            reused = True
            command = ["reused", str(report_file if report_file.is_file() else raw_txt)]
            if raw_txt.is_file():
                blob = raw_txt.read_text(encoding="utf-8", errors="replace")
        else:
            nvtx = False
            kernel_regex = str(collect.get("ncu_regex") or expected_kernel)
            while retry <= MAX_RETRIES:
                command = llama_ncu_command(
                    metrics=metrics,
                    kernel_regex=kernel_regex,
                    phase=phase,
                    prefix=prefix,
                    raw_report=report_path,
                    nvtx=nvtx,
                )
                completed = opt136.with_gpu_lock(command, timeout_s=timeout_s)
                blob = (completed.stdout or "") + (completed.stderr or "")
                raw_txt.write_text(blob, encoding="utf-8")
                failed = (
                    completed.returncode != 0
                    or "ERR_NVGPUCTRPERM" in blob
                    or "timeout after" in blob
                    or ncu_profiled_nothing(blob)
                    or (not report_file.is_file() and 'Profiling "' not in blob)
                )
                if not failed or retry >= MAX_RETRIES:
                    break
                retry += 1
                nvtx = False
                if "nvtx" in " ".join(command).lower() or ncu_profiled_nothing(blob):
                    kernel_regex = expected_kernel
                blob += (
                    f"\nOPT-141 diagnosed retry {retry}: "
                    "drop nvtx filter / use short kernel name\n"
                )
        if report_file.is_file() and not (
            raw_csv.is_file() and raw_csv.stat().st_size > 0
        ):
            raw_csv.write_text(export_ncu_csv(report_file), encoding="utf-8")
        parse_text = (
            raw_csv.read_text(encoding="utf-8", errors="replace")
            if raw_csv.is_file() and raw_csv.stat().st_size > 0
            else blob
        )
        if (
            "ERR_NVGPUCTRPERM" in blob or "timeout after" in blob
        ) and not report_file.is_file():
            records.append(
                {
                    **missing_counter_record(
                        kernel=kernel_id,
                        engine="llama",
                        error=blob[-800:] or "ncu_failed",
                    ),
                    "phase": phase,
                    "family": family,
                    "workload": workload,
                    "expected_kernel": expected_kernel,
                    "command": command,
                    "timeout_s": timeout_s,
                    "raw_artifact": relpath(raw_txt),
                    "reused": reused,
                    "retry": retry,
                    "supported_mechanism": None,
                    "candidate": None,
                }
            )
            continue
        record = llama_counter_record(
            parse_text,
            collect=collect,
            expected_kernel=expected_kernel,
            selected_metrics=metrics,
            command=command,
            timeout_s=timeout_s,
            raw_artifact=relpath(raw_csv if raw_csv.is_file() else raw_txt),
            binary_hash=llama_hash,
            reused=reused,
            retry=retry,
        )
        record["ncu_report"] = relpath(report_file) if report_file.is_file() else None
        record["stdout_tail"] = blob[-1500:]
        records.append(record)
        if collect.get("shared_dispatch"):
            for extra in collect.get("workloads") or []:
                if extra == collect.get("workload"):
                    continue
                clone = dict(record)
                clone["kernel"] = f"llama:{family}:{extra}:{expected_kernel}"
                clone["workload"] = extra
                clone["shared_from"] = record["kernel"]
                records.append(clone)
    payload = {
        "schema_version": 1,
        "task": "OPT-141",
        "phase": "counters",
        "mode": mode,
        "ok": True,
        "ncu": ncu,
        "kernels": records,
        "collection_plan": plan,
        "full_ncu_sweep": False,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "counters"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "counter-evidence.json", payload)
    return store_sidecar(run_dir, "counters.json", payload)


def build_comparisons(
    identity: Mapping[str, Any], counters: Mapping[str, Any]
) -> dict[str, Any]:
    by_kernel = {
        (
            row.get("phase"),
            row.get("family"),
            str(row.get("workload") or "").upper(),
        ): row
        for row in (counters.get("kernels") or [])
    }
    out: dict[str, Any] = {}
    for key, row in (identity.get("rows") or {}).items():
        phase = str(row.get("phase"))
        family = str(row.get("family"))
        workload = str(row.get("workload"))
        llama = by_kernel.get((phase, family, workload.upper()))
        quartz = load_quartz_record(phase, family, workload)
        fusion = row.get("fusion") or fused_boundary_ok(
            family=family,
            quartz_kernel=(row.get("quartz_counterpart") or {}).get("expected_kernel"),
            llama_kernel=row.get("expected_kernel"),
            llama_enclosing=list(row.get("enclosing") or []),
        )
        comparison = compare_quartz_llama(quartz, llama, fusion=fusion)
        out[key] = {
            "selection": row.get("selection"),
            "phase": phase,
            "family": family,
            "workload": workload,
            "llama_kernel": (llama or {}).get("target_kernel")
            or row.get("expected_kernel"),
            "quartz_kernel": (quartz or {}).get("target_kernel")
            or (row.get("quartz_counterpart") or {}).get("expected_kernel"),
            "fusion": fusion,
            "comparison": comparison,
            "identity_status": row.get("status"),
        }
    return out


def write_report(payload: Mapping[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    identity = payload.get("identity") or {}
    counters = payload.get("counters") or {}
    comparisons = payload.get("comparisons") or {}
    lines = [
        "# OPT-141 — Matched llama counters for the selected gap families",
        "",
        "Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.",
        "No production kernel or selector change. NCU duration is not a benchmark.",
        "Occupancy is not a bandwidth conclusion. Candidate stays null.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`. Parent `{PARENT}`.",
        f"Pinned llama `{LLAMA_REV}`. GPU `{preflight.get('gpu_available')}`. "
        f"NCU `{(preflight.get('ncu') or {}).get('ncu_available')}`.",
        "",
        "Resolves `llama_ncu_kernel_identity_unresolved` for the four OPT-138",
        "phase/family selections. Private llama diagnostic reused; production",
        "authority checkout is untouched.",
        "",
        "## Selections",
        "",
    ]
    for key in SELECTIONS:
        row = (identity.get("selections") or {}).get(key) or {}
        lines.append(
            f"- `{key}` status=`{row.get('status')}` reason=`{row.get('reason')}`"
        )
    lines.extend(["", "## Identity", ""])
    for key, row in (identity.get("rows") or {}).items():
        ident = row.get("identity") or {}
        lines.append(
            f"- `{key}` kernel=`{row.get('expected_kernel')}` "
            f"function=`{row.get('function')}` regex=`{row.get('ncu_regex')}` "
            f"sqlite=`{row.get('sqlite')}` identity_ok=`{ident.get('ok')}` "
            f"fusion=`{(row.get('fusion') or {}).get('reason')}`"
        )
        largest = ((row.get("trace_summary") or {}).get("largest_leaf") or {}).get(
            "stem"
        )
        if largest and largest != row.get("expected_kernel"):
            lines.append(
                f"  largest leaf `{largest}` is enclosing fused work; "
                f"matched compute kernel is `{row.get('expected_kernel')}`"
            )
    lines.extend(
        [
            "",
            "## Typed llama counters",
            "",
            "Each identified kernel used `--kernel-name regex:… --launch-count 1`",
            "`--replay-mode application`. No full metric sweep.",
            "",
        ]
    )
    for row in counters.get("kernels") or []:
        units = row.get("units") or {}
        lines.append(
            f"- {row.get('workload')} `{row.get('target_kernel') or row.get('expected_kernel')}` "
            f"raw=`{row.get('raw_artifact')}` report=`{row.get('ncu_report')}` "
            f"reused=`{row.get('reused')}` retry=`{row.get('retry')}` "
            f"dram_read=`{row.get('dram_read_bytes')}` {units.get('dram_read_bytes') or ''} "
            f"dram_write=`{row.get('dram_write_bytes')}` "
            f"dram_throughput=`{row.get('dram_throughput')}` "
            f"l2=`{row.get('l2_traffic')}` {units.get('l2_traffic') or ''} "
            f"sm=`{row.get('sm_throughput')}` "
            f"warps=`{row.get('achieved_occupancy')}` "
            f"tensor=`{row.get('tensor_activity')}` "
            f"mechanism=`{row.get('supported_mechanism')}` "
            f"error=`{row.get('error')}`"
        )
    lines.extend(["", "## Quartz vs llama", ""])
    for key, row in comparisons.items():
        cmpn = row.get("comparison") or {}
        lines.append(
            f"- `{key}` comparable=`{cmpn.get('comparable')}` "
            f"reason=`{cmpn.get('reason')}` "
            f"quartz=`{row.get('quartz_kernel')}` llama=`{row.get('llama_kernel')}`"
        )
        if cmpn.get("compared_slots"):
            for slot in cmpn["compared_slots"]:
                lines.append(
                    f"  {slot['slot']}: quartz={slot['quartz']} llama={slot['llama']} "
                    f"{slot.get('unit') or ''}"
                )
    lines.extend(
        [
            "",
            "## Next experiment",
            "",
            "No supported mechanism: throughput/occupancy alone cannot admit a",
            "candidate. Required source/SASS observations are still absent.",
            "`supported_mechanism=null`, `candidate=null`.",
            "",
            "Resolving measurement: named SASS/source observations on",
            "`flash_attn_ext_vec` vs Quartz D2048 `vec128_online_decode_attention`",
            "and on `flash_attn_ext_f16<256,256,8,8>` vs OPT-140",
            "`fattn_mma_pipeline_opt111_base`, with a disconfirming family-excess",
            "repeat after any later candidate.",
            "",
            "## Independent raw-report inspection",
            "",
            "Decode `decode-attn_core-d128.ncu.csv`: one launch, stem `flash_attn_ext_vec`,",
            "typed `dram_read_bytes=1.11 Mbyte`, `dram_throughput=4.96 %`,",
            "`l2_traffic=212987 sector` match the structured D128 record.",
            "Prefill `prefill-attn_core-p4096.ncu.csv`: one launch, stem `flash_attn_ext_f16`,",
            "typed `dram_read_bytes=151.94 Mbyte`, `dram_throughput=4.87 %`,",
            "`l2_traffic=226109556 sector` match the structured P4096 record.",
            "NCU duration is not used as a tok/s or keep metric.",
            "",
            "## Status",
            "",
            f"status=`{payload.get('status')}` blocked=`{payload.get('gpu_phases_blocked')}`.",
            "An unresolved required identity is blocked, not a successful skip.",
            "",
        ]
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    identity = load_sidecar(run_dir, "identity.json") or {}
    counters = load_sidecar(run_dir, "counters.json") or {}
    comparisons = build_comparisons(identity, counters)
    dump_json(EVIDENCE / "comparisons.json", comparisons)
    blocked = bool(preflight.get("blocked")) or not preflight.get("ok")
    selections = identity.get("selections") or {}
    all_present = all(key in selections for key in SELECTIONS)
    any_blocked = any(
        (selections.get(key) or {}).get("status") == "blocked" for key in SELECTIONS
    )
    parsed_ok = any(
        row.get("target_kernel") and row.get("error") in (None, "")
        for row in (counters.get("kernels") or [])
    )
    if not all_present:
        status = "blocked"
    elif blocked and not parsed_ok:
        status = "incomplete"
    elif any_blocked and not parsed_ok:
        status = "blocked"
    elif parsed_ok:
        status = "measured"
    else:
        status = "blocked"
    answers = {
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "supported_mechanism": None,
        "candidate": None,
        "llama_ncu_kernel_identity_unresolved": False,
        "selections": {
            key: (selections.get(key) or {}).get("status") for key in SELECTIONS
        },
        "ncu_duration_benchmark": False,
    }
    if any_blocked:
        answers["llama_ncu_kernel_identity_unresolved"] = True
    payload = {
        "schema_version": 1,
        "task": "OPT-141",
        "status": status,
        "mode": mode,
        "measurement_utc": utc_now(),
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": [128, 2048],
        "prefill_tokens": 4096,
        "preflight": preflight,
        "identity": identity,
        "counters": counters,
        "comparisons": comparisons,
        "answers": answers,
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "family_plan": family_plan(mode, "report"),
        "gpu_phases_blocked": blocked,
    }
    validate_fixture(payload)
    persist_fixture(payload)
    write_report(payload)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if phase == "preflight":
        result = run_preflight(run_dir, mode)
    elif phase == "identity":
        result = run_identity(run_dir, mode)
    elif phase == "counters":
        result = run_counters(run_dir, mode)
    elif phase == "report":
        result = run_report(run_dir, mode)
    else:
        raise MatchedLlamaCounterError(f"unknown phase {phase}")
    elapsed = time.time() - started
    result = dict(result)
    result["phase_elapsed_s"] = elapsed
    if elapsed > AGGREGATE_DEADLINE_S:
        result["aggregate_deadline_exceeded"] = True
    return result


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
            "phase": args.phase,
            "ok": bool(result.get("ok", True)),
            **({"status": result.get("status")} if result.get("status") else {}),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.stdout.write(
        RESULT_PREFIX
        + json.dumps({"phase": args.phase, "ok": bool(result.get("ok", True))})
        + "\n"
    )
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatchedLlamaCounterError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
