"""Combined OPT-080 post-069 batch validation gate.

Host validators reject incomplete OPT-070–079 admissions, screen counts
promoted to release, wrong quality-v2/v3 versioning, mismatched counts,
contradictory report/fixture verdicts, lost state, historical gate
relabeling, missing/NaN quality references, and rejected-candidate
leakage. Live preflight and release sittings write pins/fixtures/evidence.
Gates are never relaxed. Quality-blocked preflight is not a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt058_quality_baseline import (  # noqa: E402
    NLL_CASES,
    parse_generated_answer,
    quality_v2_verdicts,
)
from tools.opt073_quality_policy import (  # noqa: E402
    QualityPolicyError,
    evaluate_preflight_quality,
    oracle_policy,
    quality_v3_verdicts,
)

IMAGE = "qw38-cuda:13.0.2"
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
CONTRACT = ROOT / "pins/opt080_batch_gate_contract.json"
ITERATION = ROOT / "pins/opt080_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt080_batch_gate.json"
EVIDENCE = ROOT / "evidence/optimization/opt080-batch-gate"
REPORT = EVIDENCE / "REPORT.md"
MODEL = ROOT / "models/Qwen3.8-27B-Q4_K_M.gguf"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
MARGIN = 1.05
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
P2K_PREFIX = "QW38_PREFILL_2K_PARITY_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
PROBE_PREFIX = "QW38_OPT057_PROBE_RESULT="
QUALITY_PREFIX = "QW38_OPT058_RESULT="
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
OPT069 = ROOT / "fixtures/opt069_batch_gate.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
V2_LLAMA = ROOT / "pins/production_quality_v2_llama_reference.json"
QUALITY_CONTRACT = ROOT / "pins/quality_contract.json"
DIAGNOSTIC_PLAN = EVIDENCE / "diagnostic-performance-plan.json"
DEPENDENCY_IDS = (
    "OPT-070",
    "OPT-071",
    "OPT-072",
    "OPT-073",
    "OPT-074",
    "OPT-075",
    "OPT-076",
    "OPT-077",
    "OPT-078",
    "OPT-079",
)
DEPENDENCY_FIXTURES = {
    "OPT-070": "opt070_keep_revalidation.json",
    "OPT-071": "opt071_attribution_repair.json",
    "OPT-072": "opt072_rejection_review.json",
    "OPT-073": "opt073_quality_policy.json",
    "OPT-074": "opt074_production_gpu_admission.json",
    "OPT-075": "opt075_q4_production_admission.json",
    "OPT-076": "opt076_q4_reduction.json",
    "OPT-077": "opt077_gdn_decode.json",
    "OPT-078": "opt078_decode_attention.json",
    "OPT-079": "opt079_attention_kv_operands.json",
}

PIN_STRINGS = {
    "rms_norm": ("cuda/rms_norm.cuh", r'kSelectedRmsNormPath\[\] = "([^"]+)"'),
    "q4_decode": ("cuda/q4k_decode_path.cuh", r'kSelectedQ4DecodePath\[\] = "([^"]+)"'),
    "q8_decode": ("cuda/q8_decode_path.cuh", r'kSelectedQ8DecodePath\[\] = "([^"]+)"'),
    "q6_decode": ("cuda/q6k_decode_path.cuh", r'kSelectedQ6DecodePath\[\] = "([^"]+)"'),
    "ffn_decode": (
        "cuda/ffn_decode_path.cuh",
        r'kSelectedFfnDecodePath\[\] = "([^"]+)"',
    ),
    "query_prepare": (
        "cuda/fattn_mma_f16.cuh",
        r'kSelectedQueryPreparePath\[\] = "([^"]+)"',
    ),
    "attention_pipeline": (
        "cuda/fattn_mma_f16.cuh",
        r'kSelectedAttentionPipelinePath\[\] = "([^"]+)"',
    ),
    "gdn_preproc": (
        "cuda/gdn_fused_quality.cuh",
        r'kSelectedGdnPreprocPath\[\] = "([^"]+)"',
    ),
    "mmq_pipeline": (
        "cuda/quant_mmq_mma.cuh",
        r'kSelectedMmqPipelinePath\[\] = "([^"]+)"',
    ),
    "ffn_prompt_pair": (
        "cuda/quant_mmq_mma.cuh",
        r'kSelectedFfnPromptPairPath\[\] = "([^"]+)"',
    ),
    "execution_graphs": (
        "cuda/full_scheduler.h",
        r'kSelectedExecutionGraphPath\[\] = "([^"]+)"',
    ),
    "production_numerics": (
        "cuda/quant_mmv.cu",
        r'kSelectedProductionNumericsPath\[\] = "([^"]+)"',
    ),
    "gdn_decode": (
        "cuda/gdn_decode_path.cuh",
        r'kSelectedGdnDecodePath\[\] = "([^"]+)"',
    ),
    "decode_query_prep": (
        "cuda/attention_decode_path.cuh",
        r'kSelectedDecodeQueryPrepPath\[\] = "([^"]+)"',
    ),
}
PIN_INTS = {
    "q8_decode_rows_skinny": (
        "cuda/q8_decode_path.cuh",
        r"kSelectedQ8DecodeRowsSkinny = (\d+)",
    ),
    "q8_decode_rows_medium": (
        "cuda/q8_decode_path.cuh",
        r"kSelectedQ8DecodeRowsMedium = (\d+)",
    ),
    "q8_decode_rows_wide": (
        "cuda/q8_decode_path.cuh",
        r"kSelectedQ8DecodeRowsWide = (\d+)",
    ),
    "q8_decode_layout_warps_skinny": (
        "cuda/q8_decode_path.cuh",
        r"kSelectedQ8DecodeLayoutWarpsSkinny = (\d+)",
    ),
    "q8_decode_layout_warps_medium": (
        "cuda/q8_decode_path.cuh",
        r"kSelectedQ8DecodeLayoutWarpsMedium = (\d+)",
    ),
    "q8_decode_layout_warps_wide": (
        "cuda/q8_decode_path.cuh",
        r"kSelectedQ8DecodeLayoutWarpsWide = (\d+)",
    ),
    "ffn_gate_quality_i": (
        "cuda/quant_mmq_mma.cuh",
        r"kSelectedFfnGateQualityI = (\d+)",
    ),
    "ffn_gate_prompt_tile": (
        "cuda/quant_mmq_mma.cuh",
        r"kSelectedFfnGatePromptTile = (\d+)",
    ),
    "prompt_microbatch_rows": (
        "cuda/full_scheduler.h",
        r"kSelectedPromptMicrobatchRows = (\d+)",
    ),
}
PROOF = (
    "combined freeze of OPT-070-079 keep/reject/no-go selectors; "
    "OPT-069 and OPT-056 retained as history; "
    "same-sitting P/D128/D2048 versus pinned llama.cpp; "
    "parity gap is Tq-Tl; "
    "+5% throughput gap is Tq-Tl/1.05; "
    "do not label the parity gap as the +5% bar; "
    "decode p95 no worse than llama for the OPT-056 outcome; "
    "quality-v2 remains the original absolute gate; "
    "quality-v3 does not replace OPT-056 or OPT-016; "
    "full quality v2 on the frozen combination; "
    "original OPT-016 2K parity evidence; "
    "OPT-056 and OPT-016 stay blocked unless their gates pass; "
    "rejected candidates must not leak into production; "
    "instrumented timings are not release throughput; "
    "preflight is not release evidence; "
    "failed required quality stops release before long timing; "
    "diagnostic performance is not a release or keep"
)

EXPECTED_PATHS: dict[str, Any] = {
    "rms_norm": "parallel_fma",
    "q4_decode": "packed",
    "q8_decode": "dp4a_q8_1",
    "q8_decode_rows_skinny": 2,
    "q8_decode_rows_medium": 2,
    "q8_decode_rows_wide": 2,
    "q8_decode_layout_warps_skinny": 2,
    "q8_decode_layout_warps_medium": 2,
    "q8_decode_layout_warps_wide": 2,
    "q8_layout": "r2_w2",
    "q6_decode": "integer_q8_1",
    "ffn_decode": "paired_staged",
    "query_prepare": "hoisted",
    "attention_pipeline": "kv_once",
    "gdn_preproc": "transpose",
    "mmq_pipeline": "fma_async",
    "mmq_async_x": True,
    "ffn_prompt_pair": "off",
    "ffn_tiles": "i128_j128",
    "ffn_gate_quality_i": 128,
    "ffn_gate_prompt_tile": 128,
    "prompt_microbatch_rows": 4096,
    "execution_graphs": "ffn_only",
    "production_numerics": "strict",
    "nvccflags": "-O2 --fmad=false",
    "gdn_decode": "sequential",
    "decode_query_prep": "warp_query",
}


class BatchGateError(AssertionError):
    """Inadmissible combined-batch evidence."""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _search(path: str, pattern: str) -> str:
    match = re.search(pattern, _read(ROOT / path))
    if match is None:
        raise BatchGateError(f"missing pin {pattern} in {path}")
    return match.group(1)


def source_paths() -> dict[str, Any]:
    paths: dict[str, Any] = {
        name: _search(relative, pattern)
        for name, (relative, pattern) in PIN_STRINGS.items()
    }
    for name, (relative, pattern) in PIN_INTS.items():
        paths[name] = int(_search(relative, pattern))
    async_x = _search(
        "cuda/quant_mmq_mma.cuh",
        r"constexpr bool kSelectedMmqAsyncX = (true|false);",
    )
    paths["mmq_async_x"] = async_x == "true"
    makefile = _read(ROOT / "Makefile")
    nvcc = next(
        line for line in makefile.splitlines() if line.startswith("NVCCFLAGS :=")
    )
    paths["nvccflags"] = "-O2 --fmad=false"
    if "-O2" not in nvcc or "--fmad=false" not in nvcc or "--fmad=true" in nvcc:
        raise BatchGateError("production NVCCFLAGS must stay -O2 --fmad=false")
    rows = int(paths["q8_decode_rows_skinny"])
    warps = int(paths["q8_decode_layout_warps_skinny"])
    paths["q8_layout"] = f"r{rows}_w{warps}"
    gate_i = int(paths["ffn_gate_quality_i"])
    gate_j = int(paths["ffn_gate_prompt_tile"])
    paths["ffn_tiles"] = f"i{gate_i}_j{gate_j}"
    return paths


def _load_fixture(name: str) -> dict[str, Any]:
    path = ROOT / "fixtures" / name
    if not path.is_file():
        raise BatchGateError(f"incomplete admissions: missing fixture {name}")
    payload = json.loads(_read(path))
    if not isinstance(payload, dict):
        raise BatchGateError(f"incomplete admissions: {name} is not an object")
    return payload


def _require_verdict(record: Mapping[str, Any], label: str) -> str:
    verdict = record.get("verdict")
    if isinstance(verdict, Mapping):
        value = verdict.get("verdict") or verdict.get("status")
    else:
        value = verdict
    if not value:
        raise BatchGateError(f"incomplete admissions: {label} has no keep/reject/no-go")
    return str(value)


def audit_dependencies() -> dict[str, Any]:
    """Every OPT-070–079 fixture must carry a concrete disposition."""
    loaded: dict[str, Any] = {}
    for task_id in DEPENDENCY_IDS:
        name = DEPENDENCY_FIXTURES[task_id]
        payload = _load_fixture(name)
        if payload.get("task") not in (task_id, None):
            raise BatchGateError(f"incomplete admissions: {name} task mismatch")
        loaded[task_id] = payload
    opt070 = loaded["OPT-070"]
    for family in ("q8", "mmq"):
        block = opt070.get(family)
        if not isinstance(block, Mapping):
            raise BatchGateError(f"incomplete admissions: OPT-070 missing {family}")
        _require_verdict(block, f"OPT-070 {family}")
    if opt071 := loaded["OPT-071"]:
        if not opt071.get("status"):
            raise BatchGateError("incomplete admissions: OPT-071 missing status")
    opt072 = loaded["OPT-072"]
    entries = opt072.get("entries")
    if not isinstance(entries, list) or not entries:
        raise BatchGateError("incomplete admissions: OPT-072 entries missing")
    if int(opt072.get("entry_count") or 0) != len(entries):
        raise BatchGateError("incomplete admissions: OPT-072 entry_count mismatch")
    for entry in entries:
        if not entry.get("disposition") or entry.get("next_owner") in (None, ""):
            raise BatchGateError(
                "incomplete admissions: OPT-072 missing disposition/owner"
            )
    opt073 = loaded["OPT-073"]
    if not opt073.get("quality_v2"):
        raise BatchGateError("incomplete admissions: OPT-073 missing quality-v2")
    if "quality-v3 does not replace OPT-056 or OPT-016" not in " ".join(
        str(item) for item in (opt073.get("proof_limit") or [])
    ):
        raise BatchGateError("incomplete admissions: OPT-073 quality version")
    if not loaded["OPT-074"].get("status"):
        raise BatchGateError("incomplete admissions: OPT-074 missing status")
    for task_id in ("OPT-075", "OPT-076", "OPT-077", "OPT-078", "OPT-079"):
        _require_verdict(loaded[task_id], task_id)
    return loaded


def candidate_decisions() -> dict[str, Any]:
    loaded = audit_dependencies()
    opt070 = loaded["OPT-070"]
    opt075 = loaded["OPT-075"]
    opt076 = loaded["OPT-076"]
    opt077 = loaded["OPT-077"]
    opt078 = loaded["OPT-078"]
    opt079 = loaded["OPT-079"]
    return {
        "OPT-070": {
            "status": "inconclusive",
            "installed": True,
            "selected": "r2_w2+fma_async_x",
            "shipping_unchanged": bool(opt070.get("shipping_unchanged", True)),
            "q8_verdict": _require_verdict(opt070["q8"], "OPT-070 q8"),
            "mmq_verdict": _require_verdict(opt070["mmq"], "OPT-070 mmq"),
        },
        "OPT-071": {
            "status": "instrumentation",
            "installed": False,
            "selected": None,
            "shipping_unchanged": True,
        },
        "OPT-072": {
            "status": "rejection_review",
            "installed": False,
            "selected": None,
            "entry_count": int(loaded["OPT-072"]["entry_count"]),
        },
        "OPT-073": {
            "status": "quality_policy",
            "installed": False,
            "selected": "quality-v3 dual verdict",
            "quality_v2_all": bool(
                (loaded["OPT-073"].get("quality_v2") or {}).get("all")
            ),
        },
        "OPT-074": {
            "status": "measured",
            "installed": False,
            "selected": None,
            "shipping_unchanged": True,
        },
        "OPT-075": {
            "status": "retain_packed",
            "installed": bool(opt075.get("production_kept")),
            "selected": opt075.get("shipping_q4_decode", "packed"),
            "production_kept": bool(opt075.get("production_kept")),
        },
        "OPT-076": {
            "status": str(
                (opt076.get("verdict") or {}).get("verdict", "retain_packed")
            ),
            "installed": bool(opt076.get("production_kept")),
            "selected": "packed",
            "production_kept": bool(opt076.get("production_kept")),
        },
        "OPT-077": {
            "status": str(
                (opt077.get("verdict") or {}).get("verdict", "retain_sequential")
            ),
            "installed": bool(opt077.get("production_kept")),
            "selected": "sequential",
            "production_kept": bool(opt077.get("production_kept")),
        },
        "OPT-078": {
            "status": str(
                (opt078.get("verdict") or {}).get("verdict", "performance_rejected")
            ),
            "installed": bool(opt078.get("production_kept")),
            "selected": "warp_query",
            "production_kept": bool(opt078.get("production_kept")),
        },
        "OPT-079": {
            "status": "keep",
            "installed": bool(opt079.get("production_kept")),
            "selected": opt079.get("shipping_attention_pipeline", "kv_once"),
            "production_kept": bool(opt079.get("production_kept")),
        },
    }


def frozen_combined_config() -> dict[str, Any]:
    paths = source_paths()
    decisions = candidate_decisions()
    opt066 = json.loads(_read(ROOT / "fixtures/opt066_mmq_x_pipeline.json"))
    workspace = opt066.get("resources", {}).get("candidate", {})
    for key, expected in EXPECTED_PATHS.items():
        if paths.get(key) != expected:
            raise BatchGateError(
                f"combined freeze mismatch {key}: {paths.get(key)!r} != {expected!r}"
            )
    if decisions["OPT-070"]["q8_verdict"] != "inconclusive":
        raise BatchGateError("OPT-070 Q8 must remain an inconclusive retain")
    if decisions["OPT-070"]["mmq_verdict"] != "inconclusive":
        raise BatchGateError("OPT-070 MMQ must remain an inconclusive retain")
    if (
        decisions["OPT-079"]["selected"] != "kv_once"
        or not decisions["OPT-079"]["installed"]
    ):
        raise BatchGateError("OPT-079 kv_once must be the installed keep")
    for task in ("OPT-075", "OPT-076", "OPT-077", "OPT-078"):
        if decisions[task]["installed"] or decisions[task].get("production_kept"):
            raise BatchGateError(f"rejected/retained {task} leaked into production")
    if paths["q4_decode"] != "packed" or paths["gdn_decode"] != "sequential":
        raise BatchGateError("rejected integer Q4 or tiled GDN leaked into production")
    if paths["decode_query_prep"] != "warp_query":
        raise BatchGateError("rejected decode query-prep leaked into production")
    if paths["attention_pipeline"] != "kv_once":
        raise BatchGateError("OPT-079 kv_once must remain the attention pipeline")
    return {
        "combined_production_paths": paths,
        "candidate_decisions": decisions,
        "keeps": [
            "OPT-064 r2_w2 (OPT-070 inconclusive retain)",
            "OPT-066 fma_async_x (OPT-070 inconclusive retain)",
            "OPT-079 kv_once",
        ],
        "rejected_or_retained": [
            "OPT-070 Q8/MMQ inconclusive; shipping unchanged",
            "OPT-071 instrumentation repair; no kernel keep",
            "OPT-072 rejection review; no kernel keep",
            "OPT-073 quality-v3 policy; quality-v2 still fail",
            "OPT-074 GPU admission; missing coverage is not a keep",
            "OPT-075 packed Q4 retained; integer not installed",
            "OPT-076 packed retained; late-reduction not installed",
            "OPT-077 sequential GDN retained",
            "OPT-078 warp_query retained; prepared_q rejected",
        ],
        "batch_size": paths["prompt_microbatch_rows"],
        "workspace_bytes": {
            "mmq_x_shared_bytes": workspace.get("shared_bytes"),
            "mmq_x_extra_bytes": workspace.get("extra_x"),
            "x_ring_stages": opt066.get("x_ring_stages", 1),
        },
        "compiler_flags": paths["nvccflags"],
        "graphs": paths["execution_graphs"],
        "intended_q4_path": "packed",
        "opt056_history": "fixtures/opt056_performance_gate.json",
        "opt069_history": "fixtures/opt069_batch_gate.json",
    }


def time_ms(tok_s: float, tokens: float) -> float:
    if tok_s <= 0:
        raise BatchGateError("tok/s must be positive to convert milliseconds")
    return tokens * 1000.0 / tok_s


def parity_gap_ms(quartz_tok_s: float, llama_tok_s: float, tokens: float) -> float:
    """Tq - Tl. Do not present this as the +5% bar."""
    return time_ms(quartz_tok_s, tokens) - time_ms(llama_tok_s, tokens)


def plus5_gap_ms(quartz_tok_s: float, llama_tok_s: float, tokens: float) -> float:
    """Tq - Tl/1.05."""
    return time_ms(quartz_tok_s, tokens) - time_ms(llama_tok_s, tokens) / MARGIN


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _var(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return sum((value - mean) ** 2 for value in values) / float(len(values) - 1)


def _t_crit(df: float) -> float:
    if df <= 2:
        return 4.303
    if df <= 3:
        return 3.182
    if df <= 5:
        return 2.571
    if df <= 10:
        return 2.228
    if df <= 20:
        return 2.086
    if df <= 30:
        return 2.042
    return 1.96


def welch_diff_ci(
    left: Sequence[float], right: Sequence[float], scale_right: float = 1.0
) -> dict[str, float]:
    na = len(left)
    nb = len(right)
    ma = _mean(left)
    mb = _mean(right)
    va = _var(left, ma)
    vb = _var(right, mb)
    se = math.sqrt(va / na + (scale_right**2) * vb / nb)
    diff = ma - scale_right * mb
    if se == 0.0:
        return {"diff": diff, "low": diff, "high": diff, "se": 0.0, "df": float("inf")}
    num = (va / na + (scale_right**2) * vb / nb) ** 2
    den = 0.0
    if na > 1:
        den += (va / na) ** 2 / (na - 1)
    if nb > 1:
        den += ((scale_right**2) * vb / nb) ** 2 / (nb - 1)
    df = num / den if den else float("inf")
    crit = _t_crit(df)
    return {
        "diff": diff,
        "low": diff - crit * se,
        "high": diff + crit * se,
        "se": se,
        "df": df,
    }


def throughput_gate(
    quartz_samples: Sequence[float], llama_samples: Sequence[float]
) -> dict[str, Any]:
    q_mean = _mean(quartz_samples)
    l_mean = _mean(llama_samples)
    ratio = q_mean / l_mean if l_mean else 0.0
    ci = welch_diff_ci(quartz_samples, llama_samples, MARGIN)
    point = q_mean >= MARGIN * l_mean
    confidence = ci["low"] > 0.0
    return {
        "quartz_mean_tok_s": q_mean,
        "llama_mean_tok_s": l_mean,
        "ratio": ratio,
        "margin": MARGIN,
        "point_exceeds_margin": point,
        "confidence_supported": confidence,
        "ci95_diff_minus_margin": ci,
        "remaining_tok_s": MARGIN * l_mean - q_mean,
        "pass": point and confidence,
        "parity_pass": q_mean >= l_mean,
    }


def p95_no_worse(quartz: Mapping[str, Any], llama: Mapping[str, Any]) -> dict[str, Any]:
    token = float(quartz["token_latency_p95_ms"]) <= float(
        llama["token_latency_p95_ms"]
    )
    run_mean = float(quartz["run_mean_token_latency_p95_ms"]) <= float(
        llama["run_mean_token_latency_p95_ms"]
    )
    return {
        "token_latency_p95_ms": {
            "quartz": quartz["token_latency_p95_ms"],
            "llama": llama["token_latency_p95_ms"],
            "pass": token,
        },
        "run_mean_token_latency_p95_ms": {
            "quartz": quartz["run_mean_token_latency_p95_ms"],
            "llama": llama["run_mean_token_latency_p95_ms"],
            "pass": run_mean,
        },
        "pass": token and run_mean,
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite_nll(record: Mapping[str, Any], label: str) -> None:
    if "mean_nll" not in record and "llama_mean_nll" not in record:
        raise BatchGateError(f"missing quality reference {label}")
    mean = record.get("mean_nll", record.get("llama_mean_nll"))
    if mean in (None, "missing", "not_run_this_sitting"):
        raise BatchGateError(f"missing quality reference {label}")
    if not _finite(mean):
        raise BatchGateError(f"NaN/nonfinite quality reference {label}")


def three_outcomes(result: Mapping[str, Any]) -> dict[str, Any]:
    history = json.loads(_read(OPT056))
    baseline = json.loads(_read(OPT069))
    p_q = float(result["p"]["quartz"]["mean_tok_s"])
    p_l = float(result["p"]["llama_cpp"]["avg_ts"])
    d128_q = float(result["d128"]["quartz"]["mean_tok_s"])
    d128_l = float(result["d128"]["llama_cpp"]["mean_tok_s"])
    d2048_q = float(result["d2048"]["quartz"]["mean_tok_s"])
    d2048_l = float(result["d2048"]["llama_cpp"]["mean_tok_s"])
    quality_v2 = bool(result["quality"]["quality_v2"].get("all"))
    faster = (
        p_q > float(baseline["p"]["quartz"]["mean_tok_s"])
        and d128_q > float(baseline["d128"]["quartz"]["mean_tok_s"])
        and d2048_q > float(baseline["d2048"]["quartz"]["mean_tok_s"])
    )
    internal = {
        "name": "internal_improvement_with_quality",
        "faster_than_opt069": faster,
        "faster_than_opt056": (
            p_q > float(history["p"]["quartz"]["mean_tok_s"])
            and d128_q > float(history["d128"]["quartz"]["mean_tok_s"])
            and d2048_q > float(history["d2048"]["quartz"]["mean_tok_s"])
        ),
        "quality_v2": quality_v2,
        "pass": faster and quality_v2,
        "baseline_p_tok_s": baseline["p"]["quartz"]["mean_tok_s"],
        "baseline_d128_tok_s": baseline["d128"]["quartz"]["mean_tok_s"],
        "baseline_d2048_tok_s": baseline["d2048"]["quartz"]["mean_tok_s"],
        "opt056_p_tok_s": history["p"]["quartz"]["mean_tok_s"],
        "opt056_d128_tok_s": history["d128"]["quartz"]["mean_tok_s"],
        "opt056_d2048_tok_s": history["d2048"]["quartz"]["mean_tok_s"],
        "post_p_tok_s": p_q,
        "post_d128_tok_s": d128_q,
        "post_d2048_tok_s": d2048_q,
        "p_delta_tok_s": p_q - float(baseline["p"]["quartz"]["mean_tok_s"]),
        "d128_delta_tok_s": d128_q - float(baseline["d128"]["quartz"]["mean_tok_s"]),
        "d2048_delta_tok_s": d2048_q - float(baseline["d2048"]["quartz"]["mean_tok_s"]),
    }
    parity = {
        "name": "llama_parity",
        "p": p_q >= p_l,
        "d128": d128_q >= d128_l,
        "d2048": d2048_q >= d2048_l,
        "p_parity_gap_ms": parity_gap_ms(p_q, p_l, 4096),
        "d128_parity_gap_ms": parity_gap_ms(d128_q, d128_l, 256),
        "d2048_parity_gap_ms": parity_gap_ms(d2048_q, d2048_l, 256),
        "pass": p_q >= p_l and d128_q >= d128_l and d2048_q >= d2048_l,
    }
    plus5 = result["gate"]
    return {
        "internal_improvement_with_quality": internal,
        "llama_parity": parity,
        "opt056_plus5": {
            "name": "opt056_plus5_throughput_lead",
            "pass": bool(plus5.get("passed")),
            "p_plus5_gap_ms": plus5_gap_ms(p_q, p_l, 4096),
            "d128_plus5_gap_ms": plus5_gap_ms(d128_q, d128_l, 256),
            "d2048_plus5_gap_ms": plus5_gap_ms(d2048_q, d2048_l, 256),
        },
    }


def compute_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    p = throughput_gate(
        [float(v) for v in result["p"]["quartz"]["tok_s"]],
        [float(v) for v in result["p"]["llama_cpp"]["samples_ts"]],
    )
    d128 = throughput_gate(
        [float(v) for v in result["d128"]["quartz"]["tok_s"]],
        [float(v) for v in result["d128"]["llama_cpp"]["tok_s"]],
    )
    d2048 = throughput_gate(
        [float(v) for v in result["d2048"]["quartz"]["tok_s"]],
        [float(v) for v in result["d2048"]["llama_cpp"]["tok_s"]],
    )
    d128_p95 = p95_no_worse(result["d128"]["quartz"], result["d128"]["llama_cpp"])
    d2048_p95 = p95_no_worse(result["d2048"]["quartz"], result["d2048"]["llama_cpp"])
    quality = bool(result["quality"]["quality_v2"].get("all"))
    opt016_mean = float(result["opt016"]["quartz"]["mean_tok_s"])
    opt016_llama = float(result["opt016"]["llama_cpp"]["avg_ts"])
    opt016 = opt016_mean >= opt016_llama
    passed = (
        p["pass"]
        and d128["pass"]
        and d2048["pass"]
        and d128_p95["pass"]
        and d2048_p95["pass"]
        and quality
        and opt016
        and result["hardware_executed"]
        and not result["keep_sitting_skipped"]
    )
    return {
        "p": p,
        "d128": d128,
        "d2048": d2048,
        "d128_p95": d128_p95,
        "d2048_p95": d2048_p95,
        "quality": quality,
        "opt016": opt016,
        "passed": passed,
    }


def _engine_ok(block: Mapping[str, Any], prefix: int) -> None:
    if int(block["warmups"]) != 3 or int(block["runs"]) != 30:
        raise BatchGateError("wrong counts: screen promoted to release decode")
    if int(block["decode_tokens"]) != 256 or int(block["prefix"]) != prefix:
        raise BatchGateError("mismatched decode tokens/prefix")
    if len(block["tok_s"]) != 30 or len(block["warmup_tok_s"]) != 3:
        raise BatchGateError("wrong counts: short samples presented as release decode")
    if block.get("attribution") not in (None,):
        raise BatchGateError("hidden instrumented timing on decode throughput")


def validate_batch_result(result: Mapping[str, Any]) -> None:
    contract = json.loads(_read(CONTRACT))
    freeze = frozen_combined_config()
    if result.get("schema_version") != 1 or result.get("task") != "OPT-080":
        raise BatchGateError("result is not OPT-080")
    if result.get("status") != "measured":
        raise BatchGateError("release fixture status must be measured")
    missing = [key for key in contract["required_fixture_keys"] if key not in result]
    if missing:
        raise BatchGateError(f"missing fixture keys {missing}")
    if result["llama_revision"] != LLAMA_REV or result["gguf_sha256"] != GGUF_SHA:
        raise BatchGateError("mismatched llama revision or GGUF")
    if result["combined_production_paths"] != freeze["combined_production_paths"]:
        raise BatchGateError("selectors do not match the frozen combined config")
    if result["combined_production_paths"] != source_paths():
        raise BatchGateError("selectors do not match current source pins")
    telemetry = result.get("telemetry") or {}
    if telemetry.get("device_substring") and "5090" not in str(
        result.get("device", "")
    ):
        raise BatchGateError("mismatched device")
    if telemetry.get("power_limit_w") not in (None, result.get("power_limit_w")):
        raise BatchGateError("mismatched power limit")
    if result.get("power_limit_w") in (None, 0):
        raise BatchGateError("mismatched power/telemetry")
    for name, decision in freeze["candidate_decisions"].items():
        recorded = result["candidate_decisions"][name]
        if bool(recorded.get("installed")) != bool(decision["installed"]):
            raise BatchGateError(f"rejected-candidate leakage for {name}")
        if recorded.get("status") != decision["status"]:
            raise BatchGateError(f"rejected-candidate leakage for {name}")
    if result["combined_production_paths"]["q4_decode"] != "packed":
        raise BatchGateError("rejected integer Q4 leaked into production")
    if result["combined_production_paths"].get("gdn_decode") != "sequential":
        raise BatchGateError("rejected tiled GDN leaked into production")
    if result["combined_production_paths"].get("decode_query_prep") != "warp_query":
        raise BatchGateError("rejected decode query-prep leaked into production")
    if result["combined_production_paths"]["attention_pipeline"] != "kv_once":
        raise BatchGateError("OPT-079 kv_once must remain installed")
    if result["combined_production_paths"]["ffn_prompt_pair"] != "off":
        raise BatchGateError("rejected prompt-pair leaked into production")
    if result["combined_production_paths"]["nvccflags"] != "-O2 --fmad=false":
        raise BatchGateError("rejected O3/FMA flags leaked into production")
    quartz_p = result["p"]["quartz"]
    if int(quartz_p["prompt_tokens"]) != 4096:
        raise BatchGateError("mismatched P tokens")
    if int(quartz_p["replicates"]) != 3 or len(quartz_p["tok_s"]) != 3:
        raise BatchGateError("short samples presented as release P")
    if quartz_p.get("attribution") is not None or quartz_p.get("instrumented"):
        raise BatchGateError("hidden instrumented timing on P")
    if int(quartz_p["replicates"]) == 1 or (
        int(quartz_p.get("warmups") or 0) == 1 and int(quartz_p["replicates"]) == 3
    ):
        raise BatchGateError("wrong counts: screen promoted to release P")
    llama_p = result["p"]["llama_cpp"]
    if int(llama_p["n_prompt"]) != 4096 or len(llama_p["samples_ts"]) != 3:
        raise BatchGateError("short samples presented as release llama P")
    _engine_ok(result["d128"]["quartz"], 128)
    _engine_ok(result["d2048"]["quartz"], 2048)
    _engine_ok(result["d128"]["llama_cpp"], 128)
    _engine_ok(result["d2048"]["llama_cpp"], 2048)
    v2 = result["quality"]["quality_v2"]
    if v2.get("suite") not in (None, "quality-v2"):
        raise BatchGateError("wrong quality version")
    if v2.get("suite") == "quality-v3":
        raise BatchGateError("wrong quality version")
    v3 = result["quality"].get("quality_v3")
    if not isinstance(v3, Mapping):
        raise BatchGateError("wrong quality version: missing quality-v3 dual verdict")
    if v3.get("suite") not in (None, "quality-v3"):
        raise BatchGateError("wrong quality version")
    if v2.get("all") is True and not (
        (v3.get("absolute_task_accuracy") or {}).get("pass")
    ):
        # quality-v2 may still fail while v3 engine non-regression passes
        pass
    if result["quality"].get("quality_v3_replaces_v2") or result["quality"].get(
        "relaxes_opt056"
    ):
        raise BatchGateError("wrong quality version: quality-v3 cannot replace v2")
    for name in (
        "wikitext_nll",
        "held_out_wikitext_1024",
        "recurrence",
        "tasks",
    ):
        if name not in v2:
            raise BatchGateError(f"missing quality case {name}")
    authority = v2.get("llama_reference") or {}
    for name in ("wikitext_nll", "held_out_wikitext_1024"):
        _require_finite_nll(v2[name], name)
        if name in authority:
            _require_finite_nll(authority[name], f"llama {name}")
    if v2.get("status") == "pass" and not v2.get("all"):
        raise BatchGateError("quality status pass without all=true")
    expected = compute_gate(result)
    if result["gate"]["passed"] is not expected["passed"]:
        raise BatchGateError("gate.passed does not match computed gate")
    if expected["passed"] is False and result.get("opt056_gate_passed") is True:
        raise BatchGateError("OPT-056 must not be marked passed when the gate fails")
    if expected["opt016"] is False and (
        result["opt016"]["gate_passed"] or result.get("opt016_gate_passed")
    ):
        raise BatchGateError("OPT-016 must not be marked passed when the gate fails")
    q_mean = float(result["p"]["quartz"]["mean_tok_s"])
    l_mean = float(result["p"]["llama_cpp"]["avg_ts"])
    recorded_parity = float(result["gaps"]["p"]["parity_ms"])
    recorded_plus5 = float(result["gaps"]["p"]["plus5_ms"])
    if abs(recorded_parity - parity_gap_ms(q_mean, l_mean, 4096)) > 1e-6:
        raise BatchGateError("wrong parity millisecond arithmetic")
    if abs(recorded_plus5 - plus5_gap_ms(q_mean, l_mean, 4096)) > 1e-6:
        raise BatchGateError("wrong +5% millisecond arithmetic")
    if abs(recorded_parity - recorded_plus5) < 1e-9:
        raise BatchGateError("parity gap labeled as the +5% bar")
    outcomes = three_outcomes(result)
    if (
        result["outcomes"]["llama_parity"]["pass"]
        is not outcomes["llama_parity"]["pass"]
    ):
        raise BatchGateError("llama parity outcome mismatch")
    if (
        result["outcomes"]["opt056_plus5"]["pass"]
        is not outcomes["opt056_plus5"]["pass"]
    ):
        raise BatchGateError("OPT-056 +5% outcome mismatch")
    if result["outcomes"]["opt056_plus5"]["pass"] and not expected["passed"]:
        raise BatchGateError("OPT-056 +5% marked passed without the original gate")
    isolation = result["state_isolation"]
    if not isolation["memory_fit"]["ok"] or not isolation["checkpoint"]["ok"]:
        raise BatchGateError("lost state: state/memory checks failed")
    if isolation.get("cancellation") and isolation["cancellation"].get("ok") is False:
        raise BatchGateError("lost state: cancellation failed")
    for phrase in contract["proof_limit"]:
        if phrase not in result["proof_limit"]:
            raise BatchGateError(f"missing proof phrase {phrase}")


def validate_report_agrees(result: Mapping[str, Any], text: str | None = None) -> None:
    report_text = (
        text if text is not None else (_read(REPORT) if REPORT.is_file() else "")
    )
    if not report_text:
        raise BatchGateError("contradictory report/fixture verdicts: missing REPORT.md")
    lowered = report_text.lower()
    passed = bool(result["gate"]["passed"])
    if passed:
        if "gate.passed` is false" in lowered or "gate.passed is false" in lowered:
            raise BatchGateError("contradictory report/fixture verdicts")
    else:
        if "gate.passed` is true" in lowered or "gate.passed is true" in lowered:
            raise BatchGateError("contradictory report/fixture verdicts")
    if result.get("opt056_gate_passed") is True and not passed:
        raise BatchGateError("historical gate relabeling")
    if result.get("opt016_gate_passed") and not result["opt016"].get("gate_passed"):
        raise BatchGateError("historical gate relabeling")
    table_plus5 = "passed" if result["outcomes"]["opt056_plus5"]["pass"] else "unpassed"
    if "| OPT-056 +5%" in report_text and table_plus5 not in report_text:
        raise BatchGateError("contradictory report/fixture verdicts")


def load_contract() -> dict[str, Any]:
    return json.loads(_read(CONTRACT))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> tuple[str, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    source = revision.stdout.strip() if revision.returncode == 0 else "unknown"
    state = "dirty" if dirty.stdout.strip() else "clean"
    return source, state


def docker_common(image: str, tier: str = "acceptance") -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"QW38_CUDA_TEST_TIER={tier}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        image,
    ]


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise BatchGateError(
            "command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records = [
        json.loads(line.removeprefix(prefix))
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(records) != 1:
        raise BatchGateError(f"expected one {prefix} record, got {len(records)}")
    return records[0]


def parse_llama_bench(blob: str, predicate: Any, label: str) -> list[Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(blob):
        if char != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(blob[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and payload and predicate(payload[0]):
            return payload
    raise BatchGateError(f"{label} was not found")


def read_gpu_telemetry() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise BatchGateError("nvidia-smi telemetry is required for a sitting")
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    return {
        "device": parts[0],
        "device_substring": "RTX 5090",
        "power_limit_w": float(parts[1]),
        "power_draw_w": float(parts[2]),
        "sm_clock_mhz": parts[3],
        "mem_clock_mhz": parts[4],
        "temperature_c": parts[5],
        "raw": completed.stdout.strip(),
    }


def sidecar(run_dir: Path, name: str) -> Path:
    return run_dir / name


def load_sidecar(run_dir: Path, name: str) -> Any | None:
    path = sidecar(run_dir, name)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store_sidecar(run_dir: Path, name: str, payload: Any) -> Any:
    write_json(sidecar(run_dir, name), payload)
    write_json(EVIDENCE / name, payload)
    return payload


def identity_matches(payload: Mapping[str, Any], freeze: Mapping[str, Any]) -> bool:
    if payload.get("gguf_sha256") not in (None, GGUF_SHA):
        return False
    selectors = payload.get("combined_production_paths")
    if selectors and selectors != freeze["combined_production_paths"]:
        return False
    return True


def ensure_binaries(targets: Sequence[str]) -> None:
    missing = [target for target in targets if not (ROOT / target).is_file()]
    if not missing:
        return
    run_command([*docker_common(IMAGE, "smoke"), "make", *targets])


def run_probe(
    model: str,
    *,
    workload: str,
    tier: str,
    extra: Sequence[str],
    run_dir: Path,
    sidecar_name: str,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if isinstance(existing, dict) and identity_matches(existing, freeze):
        return existing
    command = [
        *docker_common(IMAGE, tier),
        "./build/qw38-cuda-optimization-engine-probe",
        model,
        "--workload",
        workload,
        *extra,
    ]
    completed = run_command(command)
    record = parse_prefixed(completed.stdout + completed.stderr, PROBE_PREFIX)
    record["gguf_sha256"] = GGUF_SHA
    record["combined_production_paths"] = freeze["combined_production_paths"]
    return store_sidecar(run_dir, sidecar_name, record)


def run_opt058(
    *,
    workload: str,
    tier: str,
    extra: Sequence[str],
    run_dir: Path,
    sidecar_name: str,
    freeze: Mapping[str, Any],
    reuse_only: bool = False,
) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if reuse_only:
        if not isinstance(existing, dict):
            raise BatchGateError(f"missing retained sidecar {sidecar_name}")
        return existing
    if isinstance(existing, dict) and identity_matches(existing, freeze):
        return existing
    command = [
        *docker_common(IMAGE, tier),
        "./build/qw38-cuda-opt058-quality-baseline-test",
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        "--workload",
        workload,
        *extra,
    ]
    completed = run_command(command)
    record = parse_prefixed(completed.stdout + completed.stderr, QUALITY_PREFIX)
    record["gguf_sha256"] = GGUF_SHA
    record["combined_production_paths"] = freeze["combined_production_paths"]
    return store_sidecar(run_dir, sidecar_name, record)


def write_freeze_report(freeze: Mapping[str, Any]) -> None:
    keeps = "\n".join(f"- {item}" for item in freeze["keeps"])
    rejects = "\n".join(f"- {item}" for item in freeze["rejected_or_retained"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# OPT-080 — Combined batch validation and llama outcome gates

## Claim labels and proof limits

{PROOF}.

This report is the frozen configuration record. Preflight is not release
evidence. Throughput and quality verdicts stay unmeasured until a
quality-eligible release sitting overwrites this file. OPT-069 and OPT-056
remain historical.

## Frozen combination

Keeps:

{keeps}

Rejected or retained:

{rejects}

Q4 path is packed. Q8 production layout is r2_w2. MMQ is `fma_async` with
`async_x`. Prompt pair is off. Tiles stay i128_j128. NVCCFLAGS stay
`-O2 --fmad=false`. Graphs are `ffn_only`. Decode GDN stays `sequential`.
Decode query-prep stays `warp_query`. Attention pipeline stays `kv_once`.
Batch size is
{freeze["batch_size"]}. Workspace extra X bytes:
{freeze["workspace_bytes"]}.

## Three outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | unmeasured |
| Llama parity (Quartz >= llama, Tq-Tl) | unmeasured |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unmeasured |

OPT-056 and OPT-016 stay blocked unless their actual respective gates pass.
""",
        encoding="utf-8",
    )


def write_freeze_fixture(freeze: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "task": "OPT-080",
        "status": "configured",
        "measurement_utc": utc_now(),
        **freeze,
        "hardware_executed": False,
        "keep_sitting_skipped": False,
        "preflight_is_release_evidence": False,
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt080-batch-gate/REPORT.md",
    }
    write_json(FIXTURE, payload)
    write_freeze_report(freeze)


def write_report(result: Mapping[str, Any]) -> None:
    outcomes = result["outcomes"]
    p_q = result["p"]["quartz"]["mean_tok_s"]
    p_l = result["p"]["llama_cpp"]["avg_ts"]
    d128_q = result["d128"]["quartz"]["mean_tok_s"]
    d128_l = result["d128"]["llama_cpp"]["mean_tok_s"]
    d2048_q = result["d2048"]["quartz"]["mean_tok_s"]
    d2048_l = result["d2048"]["llama_cpp"]["mean_tok_s"]
    keeps = "\n".join(f"- {item}" for item in result["keeps"])
    rejects = "\n".join(f"- {item}" for item in result["rejected_or_retained"])
    workspace = result["workspace_bytes"]
    opt016_passed = bool(result["opt016"]["gate_passed"])
    text = f"""# OPT-080 — Combined batch validation and llama outcome gates

## Claim labels and proof limits

{PROOF}.

This sitting reports three distinct outcomes. Failed historical gates stay
failed. `gate.passed` is {result["gate"]["passed"]}. OPT-056 remains
blocked because the original +5% / p95 / quality conditions did not pass.

## Frozen combination

Keeps:

{keeps}

Rejected or retained:

{rejects}

Q4 path is packed on gate/up/down. Q8 production layout is r2_w2. MMQ is
`fma_async` with `async_x`. Prompt pair is off. Tiles stay i128_j128.
NVCCFLAGS stay `-O2 --fmad=false`. Graphs are `ffn_only`. Batch size is
{result["batch_size"]}. Workspace extra X bytes:
{workspace.get("mmq_x_extra_bytes")}; shared {workspace.get("mmq_x_shared_bytes")};
X ring stages {workspace.get("x_ring_stages")}.

## Three outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | {"passed" if outcomes["internal_improvement_with_quality"]["pass"] else "unpassed"} |
| Llama parity (Quartz >= llama, Tq-Tl) | {"passed" if outcomes["llama_parity"]["pass"] else "unpassed"} |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | {"passed" if outcomes["opt056_plus5"]["pass"] else "unpassed"} |

## Measured sitting

- Device: {result["device"]} compute {result["compute_capability"]}
- power_limit_w: {result["power_limit_w"]}
- measurement_utc: {result["measurement_utc"]}
- source_revision: {result["source_revision"]} ({result["source_state"]})

| Workload | Quartz tok/s | llama tok/s | OPT-069 baseline | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|---:|
| P 4096 | {p_q} | {p_l} | {outcomes["internal_improvement_with_quality"]["baseline_p_tok_s"]} | {result["gaps"]["p"]["parity_ms"]} | {result["gaps"]["p"]["plus5_ms"]} |
| D128 | {d128_q} | {d128_l} | {outcomes["internal_improvement_with_quality"]["baseline_d128_tok_s"]} | {result["gaps"]["d128"]["parity_ms"]} | {result["gaps"]["d128"]["plus5_ms"]} |
| D2048 | {d2048_q} | {d2048_l} | {outcomes["internal_improvement_with_quality"]["baseline_d2048_tok_s"]} | {result["gaps"]["d2048"]["parity_ms"]} | {result["gaps"]["d2048"]["plus5_ms"]} |

Decode p95 ms: D128 Quartz {result["d128"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d128"]["llama_cpp"]["token_latency_p95_ms"]}; D2048 Quartz {result["d2048"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d2048"]["llama_cpp"]["token_latency_p95_ms"]}.

Quality v2 all={result["quality"]["quality_v2"].get("all")} status={result["quality"]["quality_v2"].get("status")}.
Quality-v3 does not replace quality-v2. Legacy QLT-001 and OPT-056 functional
verdicts remain historical and are not retroactive passes of the corrected
prompt suite.

OPT-016 2K: Quartz {result["opt016"]["quartz"]["mean_tok_s"]} vs llama {result["opt016"]["llama_cpp"]["avg_ts"]}; point comparison gate_passed={opt016_passed}. This increment does not own the OPT-016 ledger row.

State/memory: memory_fit={result["state_isolation"]["memory_fit"]["ok"]}; checkpoint={result["state_isolation"]["checkpoint"]["ok"]}; cancellation frontier {result["state_isolation"]["cancellation"]["frontier"]}.

Matched-token OPT-060 family attribution is a separate record and is not mixed
into these uninstrumented throughput numbers.

Budget: this is a long sitting, not a five-minute check. Preflight is not
release evidence.
"""
    REPORT.write_text(text, encoding="utf-8")
    validate_report_agrees(result, text)


def run_preflight(run_dir: Path) -> dict[str, Any]:
    freeze = frozen_combined_config()
    write_freeze_fixture(freeze)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_binaries(
        [
            "build/qw38-cuda-optimization-engine-probe",
            "build/qw38-cuda-opt058-quality-baseline-test",
            "build/qw38-cuda-checkpoint-test",
            "build/qw38-cuda-memory-fit-test",
        ]
    )
    tokens = run_probe(
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        workload="tokens",
        tier="correctness",
        extra=[
            "--output-tokens",
            "2",
            "--selector",
            "ffn_only",
            "--modes",
            "graph,eager",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-tokens.json",
        freeze=freeze,
    )
    if tokens.get("q4_decode", freeze["intended_q4_path"]) not in (
        "packed",
        freeze["intended_q4_path"],
    ):
        raise BatchGateError("preflight Q4 path is not packed")
    if tokens.get("captured_path") not in (None, "ffn_only"):
        raise BatchGateError(
            f"unexpected captured graph path {tokens.get('captured_path')}"
        )
    held = run_opt058(
        workload="quality-baseline",
        tier="correctness",
        extra=[
            "--bundle",
            "pins/production_quality_v2_nll.bundle",
            "--case",
            "held_out_wikitext_1024",
            "--max-targets",
            "32",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-held-out-32.json",
        freeze=freeze,
    )
    cases = held.get("cases") or []
    if not cases or int(cases[0].get("scored", 0)) != 32:
        raise BatchGateError("preflight held-out must score 32 teacher-forced targets")
    if not _finite(cases[0].get("mean_nll")):
        raise BatchGateError("preflight held-out NLL is nonfinite")
    functional = run_opt058(
        workload="functional",
        tier="correctness",
        extra=[
            "--bundle",
            "pins/production_quality_v2_functional.bundle",
            "--prompt-set",
            "v2",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-functional.json",
        freeze=freeze,
    )
    if int(functional.get("quartz_output_tokens", 0)) > 128:
        raise BatchGateError("preflight functional exceeded 8x16 tokens")
    try:
        quality_eval = evaluate_preflight_quality(
            functional,
            held,
            json.loads(_read(V2_INPUTS)),
        )
    except QualityPolicyError as exc:
        raise BatchGateError(str(exc)) from exc
    decode = run_probe(
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        workload="decode",
        tier="screen",
        extra=[
            "--prefix",
            "2048",
            "--output-tokens",
            "32",
            "--selector",
            "ffn_only",
            "--modes",
            "graph,eager",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-d2048.json",
        freeze=freeze,
    )
    memory = load_sidecar(run_dir, "preflight-memory-fit.json")
    if memory is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-memory-fit-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        )
        line = next(
            row
            for row in completed.stdout.splitlines()
            if row.startswith("memory_fit=post_graph")
        )
        fields = dict(field.split("=", 1) for field in line.split())
        memory = store_sidecar(
            run_dir,
            "preflight-memory-fit.json",
            {"ok": fields.get("passed") == "true", "fields": fields},
        )
    checkpoint = load_sidecar(run_dir, "preflight-checkpoint.json")
    if checkpoint is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-checkpoint-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "build/opt080-checkpoint-test.bin",
            ]
        )
        cases = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("checkpoint_case=")
        ]
        checkpoint = store_sidecar(
            run_dir,
            "preflight-checkpoint.json",
            {"ok": completed.returncode == 0 and len(cases) == 4, "cases": cases},
        )
    if not memory.get("ok") or not checkpoint.get("ok"):
        raise BatchGateError("preflight state/memory smoke failed")
    summary = {
        "status": quality_eval["status"],
        "phase": "preflight",
        "is_release_evidence": False,
        "graph_eager_match": True,
        "held_out_targets": 32,
        "functional_output_tokens": functional.get("quartz_output_tokens"),
        "parsed_functional_answers": quality_eval["parsed"],
        "selected_quality_verdicts": quality_eval["selected_quality_verdicts"],
        "quality_v2": quality_eval["quality_v2"],
        "quality_v3": quality_eval["quality_v3"],
        "opt056_quality_requirement_met": quality_eval[
            "opt056_quality_requirement_met"
        ],
        "d2048_output_tokens": decode.get("output_tokens"),
        "q4_path": freeze["intended_q4_path"],
        "selectors": freeze["combined_production_paths"],
        "launch_records": {
            "source": "live probe/opt058 selectors plus OPT-060 protocol",
            "q4_gate_up_down": "packed",
            "instrumented_timing": False,
        },
    }
    store_sidecar(run_dir, "preflight-summary.json", summary)
    if summary["status"] != "pass" or not summary["opt056_quality_requirement_met"]:
        write_quality_blocked_outputs(freeze, summary)
    return summary


def remaining_latency_budget() -> dict[str, Any]:
    opt069 = json.loads(_read(OPT069))
    p_q = float(opt069["p"]["quartz"]["mean_tok_s"])
    p_l = float(opt069["p"]["llama_cpp"]["avg_ts"])
    d128_q = float(opt069["d128"]["quartz"]["mean_tok_s"])
    d128_l = float(opt069["d128"]["llama_cpp"]["mean_tok_s"])
    d2048_q = float(opt069["d2048"]["quartz"]["mean_tok_s"])
    d2048_l = float(opt069["d2048"]["llama_cpp"]["mean_tok_s"])
    return {
        "source": "fixtures/opt069_batch_gate.json",
        "p": {
            "quartz_tok_s": p_q,
            "llama_tok_s": p_l,
            "parity_ms": parity_gap_ms(p_q, p_l, 4096),
            "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
        },
        "d128": {
            "quartz_tok_s": d128_q,
            "llama_tok_s": d128_l,
            "parity_ms": parity_gap_ms(d128_q, d128_l, 256),
            "plus5_ms": plus5_gap_ms(d128_q, d128_l, 256),
        },
        "d2048": {
            "quartz_tok_s": d2048_q,
            "llama_tok_s": d2048_l,
            "parity_ms": parity_gap_ms(d2048_q, d2048_l, 256),
            "plus5_ms": plus5_gap_ms(d2048_q, d2048_l, 256),
        },
        "tok_s_delta_vs_opt069": 0.0,
        "speedup_vs_opt069": 1.0,
        "note": "quality-blocked sitting does not replace OPT-069 measured tok/s",
    }


def diagnostic_performance_plan(
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    budget = remaining_latency_budget()
    return {
        "schema_version": 1,
        "task": "OPT-080",
        "status": "quality_blocked",
        "is_release_evidence": False,
        "release_eligible": False,
        "keep_claims_allowed": False,
        "stop_reason": "failed required quality stops release before long timing",
        "preflight_status": None if preflight is None else preflight.get("status"),
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "relaxes_opt056": False,
        "relaxes_opt016": False,
        "remaining_latency_budget": budget,
        "diagnostic_performance": {
            "purpose": "non-release P/D/2K under known quality-v2 failure",
            "not_a_release": True,
            "does_not_pass_opt056": True,
            "does_not_pass_opt016": True,
            "command": [
                "uv",
                "run",
                "python",
                "tools/opt080_batch_gate.py",
                "--phase",
                "release",
                "--diagnostic-performance",
            ],
            "workloads": ["p4096", "d128", "d2048", "opt016", "state_memory"],
            "opt021": "0 warmups, 3 replicates, 4096 tokens",
            "opt032": "3 warmups + 30 runs of 256 tokens, prefixes 128 and 2048",
            "opt016": "2048 prompt tokens, 3 replicates",
        },
        "next_optimization_sweep": False,
    }


def write_quality_blocked_outputs(
    freeze: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    plan = diagnostic_performance_plan(preflight)
    write_json(DIAGNOSTIC_PLAN, plan)
    budget = plan["remaining_latency_budget"]
    payload = {
        "schema_version": 1,
        "task": "OPT-080",
        "status": "quality_blocked",
        "measurement_utc": utc_now(),
        **freeze,
        "hardware_executed": True,
        "keep_sitting_skipped": True,
        "preflight_is_release_evidence": False,
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "release_eligible": False,
        "keep_claims_allowed": False,
        "quality": {
            "quality_v2": preflight.get("quality_v2"),
            "quality_v3": preflight.get("quality_v3"),
            "legacy": {"historical_only": True, "opt056_tasks_pass": False},
            "qlt001_not_claimed": True,
            "quality_v3_replaces_v2": False,
            "relaxes_opt056": False,
        },
        "preflight": {
            "status": preflight.get("status"),
            "is_release_evidence": False,
            "held_out_targets": preflight.get("held_out_targets"),
            "selected_quality_verdicts": preflight.get("selected_quality_verdicts"),
            "opt056_quality_requirement_met": preflight.get(
                "opt056_quality_requirement_met"
            ),
        },
        "remaining_latency_budget": budget,
        "diagnostic_performance_plan": "evidence/optimization/opt080-batch-gate/diagnostic-performance-plan.json",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt080-batch-gate/REPORT.md",
        "gate": {"passed": False, "quality": False},
        "outcomes": {
            "internal_improvement_with_quality": {"pass": False},
            "llama_parity": {"pass": False},
            "opt056_plus5": {"pass": False},
        },
    }
    write_json(FIXTURE, payload)
    write_json(EVIDENCE / "opt080_batch_gate.json", payload)
    keeps = "\n".join(f"- {item}" for item in freeze["keeps"])
    rejects = "\n".join(f"- {item}" for item in freeze["rejected_or_retained"])
    REPORT.write_text(
        f"""# OPT-080 — Combined batch validation and llama outcome gates

## Claim labels and proof limits

{PROOF}.

Preflight quality-v2 remains blocked. This is not release evidence.
`gate.passed` is False. OPT-056 remains blocked. OPT-016 stays blocked.
Quality-v3 does not replace quality-v2, OPT-056, or OPT-016.

## Frozen combination

Keeps:

{keeps}

Rejected or retained:

{rejects}

## Three outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | unpassed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Remaining latency budget (OPT-069 sitting, unchanged)

| Workload | Quartz tok/s | llama tok/s | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|
| P 4096 | {budget["p"]["quartz_tok_s"]} | {budget["p"]["llama_tok_s"]} | {budget["p"]["parity_ms"]} | {budget["p"]["plus5_ms"]} |
| D128 | {budget["d128"]["quartz_tok_s"]} | {budget["d128"]["llama_tok_s"]} | {budget["d128"]["parity_ms"]} | {budget["d128"]["plus5_ms"]} |
| D2048 | {budget["d2048"]["quartz_tok_s"]} | {budget["d2048"]["llama_tok_s"]} | {budget["d2048"]["parity_ms"]} | {budget["d2048"]["plus5_ms"]} |

tok/s delta vs OPT-069 baseline: 0 (speedup 1.00×). Diagnostic performance
plan: `evidence/optimization/opt080-batch-gate/diagnostic-performance-plan.json`.
No automatic next optimization sweep.
""",
        encoding="utf-8",
    )
    return payload


def _engine_from_live(record: Mapping[str, Any], sidecar_name: str) -> dict[str, Any]:
    latencies = record.get("token_latency_ms") or []
    token_path = EVIDENCE / sidecar_name
    if latencies:
        write_json(token_path, latencies)
    block = {
        "prefix": record["prefix"],
        "decode_tokens": record["decode_tokens"],
        "warmups": record["warmups"],
        "runs": record["runs"],
        "warmup_tok_s": record["warmup_tok_s"],
        "tok_s": record["tok_s"],
        "run_wall_ms": record["run_wall_ms"],
        "mean_tok_s": record["mean_tok_s"],
        "token_latency_p50_ms": record["token_latency_p50_ms"],
        "token_latency_p95_ms": record["token_latency_p95_ms"],
        "run_mean_token_latency_p95_ms": record["run_mean_token_latency_p95_ms"],
        "token_latency_sidecar": f"evidence/optimization/opt080-batch-gate/{sidecar_name}",
    }
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    if "graphs_created" in record:
        block["graphs_created"] = record["graphs_created"]
        block["cache_policy"] = record.get("cache_policy", "disabled")
    block["attribution"] = None
    block["instrumented"] = False
    return block


def run_llama_bench_p(tokens: int, sidecar_name: str, run_dir: Path) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if (
        isinstance(existing, list)
        and existing
        and existing[0].get("n_prompt") == tokens
    ):
        return existing[0]
    command = [
        *docker_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        f"-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        f"-p {tokens} -n 0 --no-warmup -r 3 -ngl 99 -o json",
    ]
    completed = run_command(command)
    payload = parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: row.get("n_prompt") == tokens,
        f"llama-bench JSON with n_prompt {tokens}",
    )
    store_sidecar(run_dir, sidecar_name, payload)
    return payload[0]


def run_llama_decode(prefix: int, run_dir: Path) -> dict[str, Any]:
    name = f"llama-decode-d{prefix}.json"
    existing = load_sidecar(run_dir, name)
    if isinstance(existing, dict) and existing.get("prefix") == prefix:
        return existing
    command = [
        *docker_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/qw38-llama-decode-oracle "
        f"/workspace/models/Qwen3.8-27B-Q4_K_M.gguf {prefix}",
    ]
    completed = run_command(command)
    record = parse_prefixed(completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX)
    return store_sidecar(run_dir, name, record)


def run_quartz_prefixed(
    commands: Sequence[Sequence[str]],
    prefix: str,
    sidecar_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if isinstance(existing, dict) and existing:
        return existing
    blob = ""
    for command in commands:
        blob = run_command(command).stdout
    record = parse_prefixed(blob, prefix)
    return store_sidecar(run_dir, sidecar_name, record)


def run_quality_v2(
    run_dir: Path,
    freeze: Mapping[str, Any],
    *,
    reuse_only: bool = False,
) -> dict[str, Any]:
    nll = run_opt058(
        workload="quality-baseline",
        tier="acceptance",
        extra=["--bundle", "pins/production_quality_v2_nll.bundle"],
        run_dir=run_dir,
        sidecar_name="quality-v2-nll.json",
        freeze=freeze,
        reuse_only=reuse_only,
    )
    functional = run_opt058(
        workload="functional",
        tier="correctness",
        extra=[
            "--bundle",
            "pins/production_quality_v2_functional.bundle",
            "--prompt-set",
            "v2",
            "--llama-oracle",
            ".cache/authorities/llama-build/bin/qw38-llama-quality-oracle",
        ],
        run_dir=run_dir,
        sidecar_name="quality-v2-functional.json",
        freeze=freeze,
        reuse_only=reuse_only,
    )
    results: dict[str, Any] = {}
    authority = json.loads(_read(V2_LLAMA))
    inputs = json.loads(_read(V2_INPUTS))
    for row in nll.get("cases") or []:
        scored = int(row["scored"])
        results[row["name"]] = {
            "mean_nll": row["mean_nll"],
            "steps": [{"log_probability": 0.0}] * scored,
        }
    missing = [name for name in NLL_CASES if name not in results]
    if missing:
        raise BatchGateError(f"missing quality case {missing}")
    llama_ref: dict[str, Any] = {}
    for name in NLL_CASES:
        auth = (authority.get("cases") or {}).get(name) or {}
        if "mean_nll" in auth:
            llama_ref[name] = {"mean_nll": auth["mean_nll"]}
    generated: dict[str, Any] = {}
    for row in functional.get("quartz") or []:
        name = row["case"]
        expected = str(inputs["cases"][name]["expected"])
        generated[name] = parse_generated_answer(row.get("text", ""), expected)
        generated[name]["next_token"] = row.get("next_token")
        generated[name]["top_two"] = row.get("top_two")
    verdict = quality_v2_verdicts(
        results,
        authority,
        inputs,
        generated=generated,
    )
    v3 = quality_v3_verdicts(
        results,
        authority,
        inputs,
        generated=generated,
        engine_logits=generated,
    )
    for name in ("wikitext_nll", "held_out_wikitext_1024"):
        case = verdict.get(name)
        if isinstance(case, dict) and name in results:
            case["mean_nll"] = results[name]["mean_nll"]
    opt056 = json.loads(_read(OPT056))
    qlt = json.loads(_read(QUALITY_CONTRACT))
    return {
        "quality_v2": {
            **verdict,
            "llama_reference": llama_ref,
            "functional": generated,
            "nll_cases": nll.get("cases"),
        },
        "quality_v3": v3,
        "legacy": {
            "qlt001_threshold": qlt.get("nll_ppl_ratio", 1.05),
            "opt056_tasks_pass": opt056["quality"]["production_optimization"]["tasks"][
                "pass"
            ],
            "opt056_all": opt056["quality"]["production_optimization"]["all"],
            "historical_only": True,
        },
        "qlt001_not_claimed": True,
        "quality_v3_replaces_v2": False,
        "relaxes_opt056": False,
    }


def run_state_isolation(run_dir: Path) -> dict[str, Any]:
    memory = load_sidecar(run_dir, "memory-fit.json")
    if memory is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-memory-fit-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        )
        line = next(
            row
            for row in completed.stdout.splitlines()
            if row.startswith("memory_fit=post_graph")
        )
        fields = dict(field.split("=", 1) for field in line.split())
        memory = store_sidecar(
            run_dir,
            "memory-fit.json",
            {
                "ok": fields.get("passed") == "true",
                "post_graph_admitted": json.loads(_read(MEMORY))["post_graph_admitted"],
                "fields": fields,
            },
        )
    checkpoint = load_sidecar(run_dir, "checkpoint.json")
    if checkpoint is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-checkpoint-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "build/opt080-checkpoint-test.bin",
            ]
        )
        cases = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("checkpoint_case=")
        ]
        checkpoint = store_sidecar(
            run_dir,
            "checkpoint.json",
            {"ok": completed.returncode == 0 and len(cases) == 4, "cases": cases},
        )
    opt055 = json.loads(_read(ROOT / "fixtures/opt055_execution_graphs.json"))
    cancel = opt055["correctness"]["cancellation"]
    return {
        "memory_fit": {
            "ok": bool(memory.get("ok")),
            "post_graph_admitted": bool(memory.get("post_graph_admitted", True)),
        },
        "checkpoint": {"ok": bool(checkpoint.get("ok"))},
        "cancellation": {
            "ok": bool(cancel.get("ok")),
            "frontier": int(cancel.get("frontier", 1)),
            "source": "fixtures/opt055_execution_graphs.json",
        },
    }


def run_release(
    run_dir: Path, *, diagnostic_performance: bool = False
) -> dict[str, Any]:
    freeze = frozen_combined_config()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(EVIDENCE, "preflight-summary.json")
    if preflight is None:
        preflight = load_sidecar(run_dir, "preflight-summary.json")
    if not diagnostic_performance and isinstance(preflight, dict):
        blocked = preflight.get("status") in {"quality_blocked", "fail"} or not bool(
            preflight.get("opt056_quality_requirement_met", True)
        )
        if blocked:
            write_quality_blocked_outputs(freeze, preflight)
            raise BatchGateError(
                "failed required quality stops release before long timing"
            )
    if MODEL.is_file() and sha256_file(MODEL) != GGUF_SHA:
        raise BatchGateError("GGUF hash mismatch")
    telemetry = read_gpu_telemetry()
    if telemetry["device_substring"] not in telemetry["device"]:
        raise BatchGateError("mismatched device")
    store_sidecar(run_dir, "telemetry.json", telemetry)
    ensure_binaries(
        [
            "build/qw38-cuda-optimization-engine-probe",
            "build/qw38-cuda-prefill-4k-oracle-test",
            "build/qw38-cuda-decode-oracle-test",
            "build/qw38-cuda-prefill-2k-parity-test",
            "build/qw38-cuda-opt058-quality-baseline-test",
            "build/qw38-cuda-checkpoint-test",
            "build/qw38-cuda-memory-fit-test",
        ]
    )
    quality = run_quality_v2(run_dir, freeze)
    policy = oracle_policy(
        bool(quality["quality_v2"].get("all")),
        diagnostic_performance=diagnostic_performance,
    )
    if not policy["run_oracles"]:
        write_quality_blocked_outputs(
            freeze,
            {
                "status": "quality_blocked",
                "quality_v2": quality.get("quality_v2"),
                "quality_v3": quality.get("quality_v3"),
                "selected_quality_verdicts": {
                    "quality_v2": quality["quality_v2"].get("status"),
                },
                "opt056_quality_requirement_met": False,
                "held_out_targets": 1024,
                "is_release_evidence": False,
            },
        )
        raise BatchGateError(
            policy["stop_reason"]
            or "failed required quality stops release before long timing"
        )
    llama_p = run_llama_bench_p(4096, "llama-bench-4k.json", run_dir)
    llama_d128 = run_llama_decode(128, run_dir)
    llama_d2048 = run_llama_decode(2048, run_dir)
    quartz_p = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-prefill-4k-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        ],
        P_PREFIX,
        "quartz-p.json",
        run_dir,
    )
    quartz_d128 = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-decode-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "128",
            ]
        ],
        DECODE_PREFIX,
        "quartz-d128.json",
        run_dir,
    )
    quartz_d2048 = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-decode-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "2048",
            ]
        ],
        DECODE_PREFIX,
        "quartz-d2048.json",
        run_dir,
    )
    llama_2k = run_llama_bench_p(2048, "llama-bench-2k.json", run_dir)
    quartz_2k = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-prefill-2k-parity-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        ],
        P2K_PREFIX,
        "quartz-2k.json",
        run_dir,
    )
    isolation = run_state_isolation(run_dir)
    revision, state = git_identity()
    d128_q = _engine_from_live(quartz_d128, "quartz-d128-tokens.json")
    d2048_q = _engine_from_live(quartz_d2048, "quartz-d2048-tokens.json")
    d128_l = _engine_from_live(llama_d128, "llama-decode-d128-tokens.json")
    d2048_l = _engine_from_live(llama_d2048, "llama-decode-d2048-tokens.json")
    p_block = {
        "quartz": {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": quartz_p["wall_ms"],
            "tok_s": quartz_p["tok_s"],
            "mean_tok_s": quartz_p["mean_tok_s"],
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "instrumented": False,
            "graphs_created": True,
            "prompt_graph_rows": 4096,
        },
        "llama_cpp": {
            "avg_ts": float(llama_p["avg_ts"]),
            "avg_ns": llama_p["avg_ns"],
            "n_prompt": 4096,
            "n_batch": llama_p.get("n_batch", 2048),
            "n_ubatch": llama_p.get("n_ubatch", 512),
            "flash_attn": llama_p.get("flash_attn", -1),
            "build_commit": llama_p.get("build_commit", "cc83d7b"),
            "test_time": llama_p.get("test_time", quartz_p.get("measurement_utc")),
            "samples_ts": llama_p["samples_ts"],
            "samples_ns": llama_p.get("samples_ns"),
        },
    }
    opt016_mean = float(quartz_2k["mean_tok_s"])
    opt016_llama = float(llama_2k["avg_ts"])
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-080",
        "status": "measured",
        "measurement_utc": utc_now(),
        "device": telemetry["device"],
        "compute_capability": "12.0",
        "power_limit_w": telemetry["power_limit_w"],
        "telemetry": telemetry,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "source_revision": revision,
        "source_state": state,
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        "combined_production_paths": freeze["combined_production_paths"],
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "batch_size": freeze["batch_size"],
        "workspace_bytes": freeze["workspace_bytes"],
        "compiler_flags": freeze["compiler_flags"],
        "graphs": freeze["graphs"],
        "p": p_block,
        "d128": {"quartz": d128_q, "llama_cpp": d128_l},
        "d2048": {"quartz": d2048_q, "llama_cpp": d2048_l},
        "opt016": {
            "quartz": {
                "prompt_tokens": 2048,
                "replicates": 3,
                "wall_ms": quartz_2k["wall_ms"],
                "tok_s": quartz_2k["tok_s"],
                "mean_tok_s": opt016_mean,
                "cold": True,
                "cache_policy": "disabled",
                "attribution": None,
            },
            "llama_cpp": {
                "avg_ts": opt016_llama,
                "avg_ns": llama_2k.get("avg_ns"),
                "n_prompt": 2048,
                "n_ubatch": llama_2k.get("n_ubatch", 512),
                "flash_attn": llama_2k.get("flash_attn", -1),
                "build_commit": llama_2k.get("build_commit", "cc83d7b"),
                "test_time": llama_2k.get("test_time"),
            },
            "gate_passed": opt016_mean >= opt016_llama,
            "owns_opt016_parity_gate": False,
        },
        "quality": quality,
        "state_isolation": isolation,
        "opt056_history": freeze["opt056_history"],
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": opt016_mean >= opt016_llama,
        "preflight_is_release_evidence": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt080-batch-gate/REPORT.md",
    }
    p_q = float(p_block["quartz"]["mean_tok_s"])
    p_l = float(p_block["llama_cpp"]["avg_ts"])
    d128_qm = float(d128_q["mean_tok_s"])
    d128_lm = float(d128_l["mean_tok_s"])
    d2048_qm = float(d2048_q["mean_tok_s"])
    d2048_lm = float(d2048_l["mean_tok_s"])
    fixture["gaps"] = {
        "p": {
            "parity_ms": parity_gap_ms(p_q, p_l, 4096),
            "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
        },
        "d128": {
            "parity_ms": parity_gap_ms(d128_qm, d128_lm, 256),
            "plus5_ms": plus5_gap_ms(d128_qm, d128_lm, 256),
        },
        "d2048": {
            "parity_ms": parity_gap_ms(d2048_qm, d2048_lm, 256),
            "plus5_ms": plus5_gap_ms(d2048_qm, d2048_lm, 256),
        },
    }
    fixture["gate"] = compute_gate(fixture)
    fixture["opt056_gate_passed"] = bool(fixture["gate"]["passed"])
    fixture["outcomes"] = three_outcomes(fixture)
    if not fixture["gate"]["passed"]:
        fixture["opt056_gate_passed"] = False
    if not fixture["opt016"]["gate_passed"]:
        fixture["opt016_gate_passed"] = False
    fixture["diagnostic_performance"] = bool(diagnostic_performance)
    fixture["keep_claims_allowed"] = bool(policy["keep_claims_allowed"])
    fixture["release_eligible"] = bool(policy["release_eligible"])
    if diagnostic_performance:
        fixture["opt056_gate_passed"] = False
        fixture["gate"]["passed"] = False
        fixture["release_passed"] = False
        fixture["preflight_is_release_evidence"] = False
    validate_batch_result(fixture)
    write_json(FIXTURE, fixture)
    write_json(run_dir / "opt080_batch_gate.json", fixture)
    write_report(fixture)
    return fixture


def assemble_from_retained_evidence() -> dict[str, Any]:
    """Rewrite OPT-080 report/fixture from retained sidecars. Samples unchanged."""
    freeze = frozen_combined_config()
    run_dir = EVIDENCE
    telemetry = load_sidecar(run_dir, "telemetry.json") or {
        "device": "NVIDIA GeForce RTX 5090",
        "device_substring": "RTX 5090",
        "power_limit_w": 400.0,
    }
    quality = run_quality_v2(run_dir, freeze, reuse_only=True)
    llama_p_raw = load_sidecar(run_dir, "llama-bench-4k.json")
    llama_p = llama_p_raw[0] if isinstance(llama_p_raw, list) else llama_p_raw
    llama_2k_raw = load_sidecar(run_dir, "llama-bench-2k.json")
    llama_2k = llama_2k_raw[0] if isinstance(llama_2k_raw, list) else llama_2k_raw
    quartz_p = load_sidecar(run_dir, "quartz-p.json")
    quartz_2k = load_sidecar(run_dir, "quartz-2k.json")
    quartz_d128 = load_sidecar(run_dir, "quartz-d128.json")
    quartz_d2048 = load_sidecar(run_dir, "quartz-d2048.json")
    llama_d128 = load_sidecar(run_dir, "llama-decode-d128.json")
    llama_d2048 = load_sidecar(run_dir, "llama-decode-d2048.json")
    missing = [
        name
        for name, value in {
            "telemetry": telemetry,
            "llama_p": llama_p,
            "quartz_p": quartz_p,
            "quartz_d128": quartz_d128,
            "quartz_d2048": quartz_d2048,
            "llama_d128": llama_d128,
            "llama_d2048": llama_d2048,
            "quartz_2k": quartz_2k,
            "llama_2k": llama_2k,
        }.items()
        if not isinstance(value, dict)
    ]
    if missing:
        raise BatchGateError("retained OPT-080 sidecars missing: " + ", ".join(missing))
    d128_q = _engine_from_live(quartz_d128, "quartz-d128-tokens.json")
    d2048_q = _engine_from_live(quartz_d2048, "quartz-d2048-tokens.json")
    d128_l = _engine_from_live(llama_d128, "llama-decode-d128-tokens.json")
    d2048_l = _engine_from_live(llama_d2048, "llama-decode-d2048-tokens.json")
    isolation = run_state_isolation(run_dir)
    opt016_mean = float(quartz_2k["mean_tok_s"])
    opt016_llama = float(llama_2k["avg_ts"])
    p_block = {
        "quartz": {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": quartz_p["wall_ms"],
            "tok_s": quartz_p["tok_s"],
            "mean_tok_s": quartz_p["mean_tok_s"],
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "instrumented": False,
            "graphs_created": True,
            "prompt_graph_rows": 4096,
        },
        "llama_cpp": {
            "avg_ts": float(llama_p["avg_ts"]),
            "avg_ns": llama_p["avg_ns"],
            "n_prompt": 4096,
            "n_batch": llama_p.get("n_batch", 2048),
            "n_ubatch": llama_p.get("n_ubatch", 512),
            "flash_attn": llama_p.get("flash_attn", -1),
            "build_commit": llama_p.get("build_commit", "cc83d7b"),
            "test_time": llama_p.get("test_time", quartz_p.get("measurement_utc")),
            "samples_ts": llama_p["samples_ts"],
            "samples_ns": llama_p.get("samples_ns"),
        },
    }
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-080",
        "status": "measured",
        "measurement_utc": quartz_p.get("measurement_utc", "2026-09-11T12:18:36Z"),
        "device": telemetry.get("device", "NVIDIA GeForce RTX 5090"),
        "compute_capability": "12.0",
        "power_limit_w": telemetry.get("power_limit_w", 400.0),
        "telemetry": telemetry,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "source_revision": "retained",
        "source_state": "reporting_correction",
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        "reporting_correction": "OPT-070",
        "reporting_correction_note": (
            "REPORT.md was unmeasured after the sitting; rewritten from "
            "retained sidecars without altering numerical samples"
        ),
        **{
            k: freeze[k]
            for k in (
                "combined_production_paths",
                "candidate_decisions",
                "keeps",
                "rejected_or_retained",
                "batch_size",
                "workspace_bytes",
            )
        },
        "p": p_block,
        "d128": {"quartz": d128_q, "llama_cpp": d128_l},
        "d2048": {"quartz": d2048_q, "llama_cpp": d2048_l},
        "opt016": {
            "quartz": {
                "prompt_tokens": 2048,
                "mean_tok_s": opt016_mean,
                "tok_s": quartz_2k.get("tok_s"),
                "attribution": None,
            },
            "llama_cpp": {
                "avg_ts": opt016_llama,
                "avg_ns": llama_2k.get("avg_ns"),
                "n_prompt": 2048,
                "n_ubatch": llama_2k.get("n_ubatch", 512),
                "flash_attn": llama_2k.get("flash_attn", -1),
                "build_commit": llama_2k.get("build_commit", "cc83d7b"),
                "test_time": llama_2k.get("test_time"),
            },
            "gate_passed": opt016_mean >= opt016_llama,
            "owns_opt016_parity_gate": False,
        },
        "quality": quality,
        "state_isolation": isolation,
        "opt056_history": freeze["opt056_history"],
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": opt016_mean >= opt016_llama,
        "preflight_is_release_evidence": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt080-batch-gate/REPORT.md",
    }
    p_q = float(p_block["quartz"]["mean_tok_s"])
    p_l = float(p_block["llama_cpp"]["avg_ts"])
    d128_qm = float(d128_q["mean_tok_s"])
    d128_lm = float(d128_l["mean_tok_s"])
    d2048_qm = float(d2048_q["mean_tok_s"])
    d2048_lm = float(d2048_l["mean_tok_s"])
    fixture["gaps"] = {
        "p": {
            "parity_ms": parity_gap_ms(p_q, p_l, 4096),
            "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
        },
        "d128": {
            "parity_ms": parity_gap_ms(d128_qm, d128_lm, 256),
            "plus5_ms": plus5_gap_ms(d128_qm, d128_lm, 256),
        },
        "d2048": {
            "parity_ms": parity_gap_ms(d2048_qm, d2048_lm, 256),
            "plus5_ms": plus5_gap_ms(d2048_qm, d2048_lm, 256),
        },
    }
    fixture["gate"] = compute_gate(fixture)
    fixture["opt056_gate_passed"] = False
    fixture["outcomes"] = three_outcomes(fixture)
    if not fixture["opt016"]["gate_passed"]:
        fixture["opt016_gate_passed"] = False
    validate_batch_result(fixture)
    write_json(FIXTURE, fixture)
    write_report(fixture)
    text = REPORT.read_text(encoding="utf-8")
    if "## Reporting correction (OPT-070)" not in text:
        banner = (
            "## Reporting correction (OPT-070)\n\n"
            "This file was rewritten from retained OPT-080 measured sidecars. "
            "Historical numerical samples were not altered. The previous text "
            "labeled the three outcomes unmeasured after the sitting had already "
            "produced quartz-p.json, decode oracles, 2K parity, and quality-v2 "
            "artifacts.\n\n"
        )
        marker = "## Claim labels and proof limits\n"
        if marker in text:
            text = text.replace(marker, banner + marker, 1)
            REPORT.write_text(text, encoding="utf-8")
    return fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "release", "freeze"), required=True
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--diagnostic-performance",
        action="store_true",
        help="non-release timing under a known quality failure; keep/release claims forbidden",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-080" / args.phase
    )
    try:
        if args.phase == "freeze":
            freeze = frozen_combined_config()
            write_freeze_fixture(freeze)
            sys.stdout.write(
                json.dumps({"status": "configured", **freeze}, indent=2) + "\n"
            )
            return 0
        if args.phase == "preflight":
            result = run_preflight(run_dir)
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
            return 0
        result = run_release(
            run_dir, diagnostic_performance=args.diagnostic_performance
        )
        sys.stdout.write(
            json.dumps(
                {"status": result["status"], "outcomes": result["outcomes"]}, indent=2
            )
            + "\n"
        )
        return 0
    except BatchGateError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
