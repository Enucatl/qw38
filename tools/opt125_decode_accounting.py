"""OPT-125 decode timing, GPU activity, and traffic-evidence reconciliation.

Diagnostics/evidence only. No production selector, kernel, or throughput
claim. Combined OPT-118+119 is the authenticated parent. Pinned llama
cc83d7b4824f73cfdda4dfbb47ee39804f71b328 remains the benchmark authority.
Historical tok/s fields are preserved; corrections are dated.
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

from tools.opt060_engine_attribution import (  # noqa: E402
    chargeable_records,
    interval,
    union_ms,
)
from tools.opt075_q4_production_admission import (  # noqa: E402
    dump_json,
    load_json,
    mean,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_REV,
    docker_common,
    git_identity,
    parse_llama_bench,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt088_batch_gate import LLAMA_DECODE_PREFIX, P2K_PREFIX  # noqa: E402
from tools.opt115_pipeline_traffic import (  # noqa: E402
    GRAPH_ROLES,
    HOST_ROLES,
    PEAK_BANDWIDTH_GB_S,
    SYNC_ROLES,
    aa_verdict_for_workload,
    authenticate_post113,
    bytes_to_peak_ms,
    classify_timeline,
    combine_aa_verdicts,
    compulsory_decode_bytes,
    embedding_row_bytes,
    inspect_repo_revision,
    inventory_bytes,
    is_kernel_leaf,
)
from tools.opt124_practical_ceiling import (  # noqa: E402
    COMBINED_DECODE_D2H,
    authenticate_pinned_llama,
    combined_decode_bytes,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt125_decode_accounting_contract.json"
ITERATION = ROOT / "pins/opt125_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt125_decode_accounting.json"
REPORT = ROOT / "evidence/optimization/opt125-decode-accounting/REPORT.md"
EVIDENCE = REPORT.parent
OPT115_FIXTURE = ROOT / "fixtures/opt115_pipeline_traffic.json"
OPT123_FIXTURE = ROOT / "fixtures/opt123_combined_stack.json"
OPT124_FIXTURE = ROOT / "fixtures/opt124_practical_ceiling.json"
NATIVE = "build/qw38-cuda-opt125-decode-accounting-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT125_DECODE_ACCOUNTING_RESULT="
COUNTS_PREFIX = "QW38_OPT125_NATIVE_COUNTS="
RECORDS_PREFIX = "QW38_OPT125_RECORDS="
LLAMA_LD = "/workspace/.cache/authorities/llama-build/bin:/usr/local/cuda/lib64"
CORRECTION_DATE = "2026-09-13"
DECODE_TOKENS = 256
LONG_DECODE_TOKENS = 32
AA_WARMUPS = 3
AA_PAIRS = 5
ACCEPTANCE_PAIRS = 10
MATERIALITY = 0.02
AA_WORKLOADS = ("p4096", "d128", "d2048")
LONG_PROBES = ("d8192", "d32768", "d131040")
RANKED_TASKS = (
    "OPT-126",
    "OPT-127",
    "OPT-128",
    "OPT-129",
    "OPT-130",
    "OPT-131",
)
DISPOSITIONS = (
    "proceed",
    "no_material_opportunity",
    "insufficient_evidence",
)
PHASES = (
    "metrics",
    "freeze",
    "matched-baseline",
    "activity-trace",
    "intervals",
    "weight-reads",
    "long-context",
    "ranking",
    "report",
)
PREFIX = {
    "p4096": 4096,
    "d128": 128,
    "d2048": 2048,
    "d8192": 8192,
    "d32768": 32768,
    "d131040": 131040,
}
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "measurement_utc",
    "identity",
    "metric_catalog",
    "historical_corrections",
    "matched_baseline",
    "activity_trace",
    "disjoint_intervals",
    "weight_reads",
    "long_context",
    "ranking",
    "shared_keep_protocol",
    "answers",
    "production_kept",
    "claims_throughput",
    "claims_performance_improvement",
    "report_path",
)
PROOF = (
    "diagnostics only: no production selector or kernel change; do not "
    "silently rewrite historical tok/s; OPT-123 request tok/s includes "
    "prefill; pinned llama decode oracle is decode-only; leaf gaps are "
    "unobserved not GPU idle; busy below listed-peak compulsory bound "
    "labels the bound unsupported; profiler absence is an attribution "
    "limitation not zero activity"
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class AccountingError(AssertionError):
    """Fail-closed OPT-125 decode-accounting error."""


def load_contract() -> dict[str, Any]:
    return load_json(CONTRACT)


def load_iteration() -> dict[str, Any]:
    return load_json(ITERATION)


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-125 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def tok_s_from_ms(wall_ms: float | None, *, tokens: int) -> float | None:
    if wall_ms is None or float(wall_ms) <= 0.0 or tokens <= 0:
        return None
    return float(tokens) * 1000.0 / float(wall_ms)


def ms_from_tok_s(tok_s: float | None, *, tokens: int = 1) -> float | None:
    if tok_s is None or float(tok_s) <= 0.0:
        return None
    return (float(tokens) / float(tok_s)) * 1000.0


def metric_catalog() -> dict[str, Any]:
    """Frozen names and timestamp-to-report boundaries."""
    return {
        "schema_version": 1,
        "correction_date": CORRECTION_DATE,
        "first_sampled_token_needs_eval": False,
        "first_sampled_token_reason": (
            "Prefill (or the previous decode eval) already produced logits "
            "for the last prompt token. Sampling that token does not require "
            "a new eval. The first decode eval consumes the first emitted "
            "token to produce logits for the next sample. Planned-token "
            "engine probes skip the sampler and count eval_calls = emitted "
            "decode tokens."
        ),
        "metrics": {
            "setup": {
                "name": "setup",
                "clock": "steady_clock around session/workspace/graph create",
                "excluded_from_request_and_decode": True,
                "denominator": None,
            },
            "prefill": {
                "name": "prefill",
                "clock": "steady_clock from request start through sync_tokens",
                "denominator": "prompt token count",
                "matches": "llama-bench -n 0; OPT-115 run_prefill",
            },
            "time_to_first_emitted_token": {
                "name": "time_to_first_emitted_token",
                "clock": "prefill plus first decode eval (planned-token probe)",
                "note": (
                    "A free-running sample/eval/output loop can emit the first "
                    "token as soon as prefill logits are sampled, before the "
                    "first decode eval. These two TTFTs are distinct."
                ),
            },
            "decode_only": {
                "name": "decode_only",
                "clock": "steady_clock after prefill through last decode eval",
                "denominator": "actual emitted decode tokens / eval_calls",
                "matches": (
                    "OPT-115 run_decode; pinned llama qw38-llama-decode-oracle; "
                    "Quartz decode_oracle_test.cu"
                ),
            },
            "sample_eval_output": {
                "name": "sample_eval_output",
                "clock": "host sample plus eval plus output transfer inside decode",
                "counted_explicitly": ["emitted_tokens", "eval_calls"],
            },
            "complete_request": {
                "name": "complete_request",
                "clock": "steady_clock from before prefill through last decode eval",
                "denominator": "actual emitted decode tokens (not prompt tokens)",
                "matches": "OPT-123 run_session_arm tok_s when outputs>0",
                "does_not_match": "OPT-115 decode-only tok_s",
            },
            "fixed_token_engine_probe": {
                "name": "fixed_token_engine_probe",
                "clock": "planned tokens, no sampler; independently restored sessions",
                "note": "Separate from free-running request comparisons.",
            },
        },
        "opt123_bug": {
            "site": "cuda/opt123_combined_stack_test.cu::run_session_arm",
            "issue": (
                "Clock starts before prefill; decode tok_s uses outputs / "
                "entire wall. Long-context uses the same function."
            ),
            "opt115_request_probe": "subtracts prefill before decode tok/s",
            "do_not_silently_rewrite": True,
        },
        "llama_boundaries": {
            "decode_oracle": (
                "tools/llama_authority/decode_oracle.cpp starts llama_time_us "
                "after prefix llama_decode; 256 outputs; 3+30; n_gpu_layers=99; "
                "n_ctx=4096. This is decode_only, not complete_request."
            ),
            "llama_bench_prefill": "llama-bench -p 4096 -n 0 is prefill only",
            "whole_engine_ratio_is_not_attention_ratio": True,
        },
    }


def derive_metrics(
    *,
    prefill_ms: float | None,
    decode_only_ms: float | None,
    outputs: int,
    eval_calls: int | None = None,
    prompt_tokens: int = 0,
    setup_ms: float | None = None,
    missing: bool = False,
) -> dict[str, Any]:
    """Derive named metrics from timestamps. Missing measurements stay None."""
    if missing:
        return {
            "ok": False,
            "status": "missing_or_failed",
            "setup_ms": None,
            "prefill_ms": None,
            "decode_only_ms": None,
            "complete_request_ms": None,
            "ttft_ms": None,
            "decode_only_tok_s": None,
            "request_tok_s": None,
            "prefill_tok_s": None,
            "emitted_tokens": outputs,
            "eval_calls": eval_calls,
            "first_sampled_token_needs_eval": False,
        }
    request_ms = None
    if prefill_ms is not None and decode_only_ms is not None:
        request_ms = float(prefill_ms) + float(decode_only_ms)
    ttft = None
    if prefill_ms is not None:
        ttft = float(prefill_ms)
        if outputs > 0 and decode_only_ms is not None and outputs:
            ttft = float(prefill_ms) + float(decode_only_ms) / float(outputs)
    return {
        "ok": True,
        "status": "derived",
        "setup_ms": setup_ms,
        "prefill_ms": prefill_ms,
        "decode_only_ms": decode_only_ms,
        "complete_request_ms": request_ms,
        "ttft_ms": ttft,
        "decode_only_tok_s": tok_s_from_ms(decode_only_ms, tokens=outputs)
        if outputs
        else tok_s_from_ms(prefill_ms, tokens=prompt_tokens),
        "request_tok_s": tok_s_from_ms(request_ms, tokens=outputs)
        if outputs
        else tok_s_from_ms(prefill_ms, tokens=prompt_tokens),
        "prefill_tok_s": tok_s_from_ms(prefill_ms, tokens=prompt_tokens),
        "emitted_tokens": outputs,
        "eval_calls": eval_calls if eval_calls is not None else outputs,
        "first_sampled_token_needs_eval": False,
    }


def synthetic_metric_cases() -> list[dict[str, Any]]:
    """Known-duration cases including a missing/failed measurement."""
    prefill_ok = derive_metrics(
        prefill_ms=100.0,
        decode_only_ms=0.0,
        outputs=0,
        prompt_tokens=4096,
        setup_ms=12.0,
    )
    decode_ok = derive_metrics(
        prefill_ms=80.0,
        decode_only_ms=2560.0,
        outputs=256,
        eval_calls=256,
        prompt_tokens=128,
        setup_ms=15.0,
    )
    failed = derive_metrics(
        prefill_ms=None,
        decode_only_ms=None,
        outputs=32,
        missing=True,
    )
    cases = [
        {
            "id": "known_prefill",
            **prefill_ok,
            "expect_prefill_tok_s": 40960.0,
            "expect_request_tok_s": 40960.0,
        },
        {
            "id": "known_decode",
            **decode_ok,
            "expect_decode_only_tok_s": 100.0,
            "expect_request_tok_s": 256000.0 / 2640.0,
        },
        {
            "id": "missing_failed",
            **failed,
        },
    ]
    if abs(float(prefill_ok["prefill_tok_s"] or 0.0) - 40960.0) > 1e-9:
        raise AccountingError("synthetic prefill tok/s mismatch")
    if abs(float(decode_ok["decode_only_tok_s"] or 0.0) - 100.0) > 1e-9:
        raise AccountingError("synthetic decode-only tok/s mismatch")
    expected_request = 256000.0 / 2640.0
    if abs(float(decode_ok["request_tok_s"] or 0.0) - expected_request) > 1e-9:
        raise AccountingError("synthetic request tok/s mismatch")
    if failed["ok"] is not False or failed["status"] != "missing_or_failed":
        raise AccountingError("synthetic missing case must stay failed")
    if decode_ok["first_sampled_token_needs_eval"] is not False:
        raise AccountingError("first sampled token must not require eval")
    if decode_ok["eval_calls"] != 256:
        raise AccountingError("eval_calls must equal emitted decode tokens")
    return cases


def correct_session_arm(
    arm: Mapping[str, Any],
    *,
    outputs: int,
    historical_tok_s_field: str = "tok_s",
) -> dict[str, Any]:
    """Recompute decode-only from stored wall/ttft without rewriting history."""
    wall = arm.get("wall_ms")
    ttft = arm.get("ttft_ms")
    historical = arm.get(historical_tok_s_field)
    if wall is None or ttft is None:
        return {
            "comparable": False,
            "reason": "stored_fields_missing_wall_or_ttft",
            "historical_tok_s": historical,
            "historical_field": historical_tok_s_field,
            "correction_date": CORRECTION_DATE,
        }
    decode_ms = float(wall) - float(ttft)
    derived = derive_metrics(
        prefill_ms=float(ttft),
        decode_only_ms=decode_ms,
        outputs=outputs,
        eval_calls=outputs,
    )
    return {
        "comparable": True,
        "correction_date": CORRECTION_DATE,
        "do_not_silently_rewrite": True,
        "historical_field": historical_tok_s_field,
        "historical_tok_s": historical,
        "historical_metric": "complete_request" if outputs else "prefill",
        "request_wall_ms": float(wall),
        "prefill_wall_ms": float(ttft),
        "decode_only_wall_ms": decode_ms,
        "decode_only_tok_s": derived["decode_only_tok_s"],
        "request_tok_s": derived["request_tok_s"],
        "emitted_tokens": outputs,
        "eval_calls": outputs,
        "source": "opt123_run_session_arm_stored_wall_ttft",
    }


def _pair_means(
    pairs: Sequence[Mapping[str, Any]], stack_key: str, outputs: int
) -> dict[str, Any]:
    request: list[float] = []
    decode: list[float] = []
    walls: list[float] = []
    ttfts: list[float] = []
    for row in pairs:
        arm = row.get(stack_key) or {}
        corrected = correct_session_arm(arm, outputs=outputs)
        if not corrected.get("comparable"):
            continue
        if corrected.get("request_tok_s"):
            request.append(float(corrected["request_tok_s"]))
        if corrected.get("decode_only_tok_s"):
            decode.append(float(corrected["decode_only_tok_s"]))
        walls.append(float(corrected["request_wall_ms"]))
        ttfts.append(float(corrected["prefill_wall_ms"]))
    if not request:
        return {"comparable": False, "reason": "no_pair_arms"}
    return {
        "comparable": True,
        "n": len(request),
        "mean_request_tok_s": mean(request),
        "mean_decode_only_tok_s": mean(decode) if decode else None,
        "mean_request_wall_ms": mean(walls),
        "mean_prefill_ms": mean(ttfts),
        "mean_decode_only_ms": mean([w - t for w, t in zip(walls, ttfts)]),
        "historical_mean_request_tok_s": mean(request),
        "correction_date": CORRECTION_DATE,
        "raw_pair_count": len(pairs),
    }


def historical_corrections() -> dict[str, Any]:
    opt123 = load_json(OPT123_FIXTURE) if OPT123_FIXTURE.is_file() else {}
    performance = opt123.get("performance") or {}
    llama = opt123.get("llama") or {}
    long_ctx = (opt123.get("long_context") or {}).get("probes") or {}
    two_k = opt123.get("opt016_2k") or {}
    workloads: dict[str, Any] = {}
    for name, outputs in (
        ("p4096", 0),
        ("d128", DECODE_TOKENS),
        ("d2048", DECODE_TOKENS),
    ):
        block = performance.get(name) or {}
        pairs = block.get("pairs") or []
        parent = _pair_means(pairs, "A", outputs)
        combined = _pair_means(pairs, "B", outputs)
        llama_tok = llama.get(f"{name}_tok_s")
        llama_metric = "prefill" if name == "p4096" else "decode_only"
        combined_decode = combined.get("mean_decode_only_tok_s")
        ratio_request = None
        ratio_decode = None
        if (
            combined.get("mean_request_tok_s")
            and llama_tok
            and llama_metric == "decode_only"
        ):
            ratio_request = float(combined["mean_request_tok_s"]) / float(llama_tok)
        if combined_decode and llama_tok and llama_metric == "decode_only":
            ratio_decode = float(combined_decode) / float(llama_tok)
        workloads[name] = {
            "historical_parent_tok_s": block.get("parent_tok_s"),
            "historical_combined_tok_s": block.get("candidate_tok_s"),
            "historical_metric": "complete_request" if outputs else "prefill",
            "parent_corrected": parent,
            "combined_corrected": combined,
            "llama_tok_s": llama_tok,
            "llama_metric": llama_metric,
            "published_combined_over_llama_request_over_decode_only": ratio_request,
            "matched_decode_only_combined_over_llama": ratio_decode,
            "source": "fixtures/opt123_combined_stack.json",
            "correction_date": CORRECTION_DATE,
            "do_not_silently_rewrite": True,
        }
    probes: dict[str, Any] = {}
    for name in LONG_PROBES:
        probe = long_ctx.get(name) or {}
        outputs = int(probe.get("tokens") or LONG_DECODE_TOKENS)
        if not probe.get("ok"):
            probes[name] = {
                "comparable": False,
                "ok": False,
                "oom": bool(probe.get("oom")),
                "reason": "opt123_probe_failed",
                "stderr_identifies_cudaErrorMemoryAllocation": "cudaErrorMemoryAllocation"
                in str(probe.get("stderr_tail") or ""),
                "generic_failed_process_is_not_oom": True,
                "historical_tok_s": probe.get("tok_s"),
                "source": "fixtures/opt123_combined_stack.json long_context",
                "correction_date": CORRECTION_DATE,
            }
            continue
        corrected = correct_session_arm(probe, outputs=outputs)
        probes[name] = {
            **corrected,
            "historical_tok_s": probe.get("tok_s"),
            "p50_ms": probe.get("p50_ms"),
            "populated_cache": probe.get("populated_cache"),
            "prefix": probe.get("prefix"),
        }
    opt115 = load_json(OPT115_FIXTURE) if OPT115_FIXTURE.is_file() else {}
    opt115_long = opt115.get("long_context") or {}
    return {
        "correction_date": CORRECTION_DATE,
        "do_not_silently_rewrite": True,
        "opt123_run_session_arm_includes_prefill": True,
        "opt115_request_probe_subtracts_prefill": True,
        "llama_decode_oracle_is_decode_only": True,
        "p4096_d128_d2048": workloads,
        "long_context": probes,
        "opt115_decode_only_long_context": {
            name: (opt115_long.get(name) or {})
            for name in LONG_PROBES
            if isinstance(opt115_long.get(name), Mapping)
        }
        if isinstance(opt115_long, Mapping)
        else {},
        "opt016_2k": {
            "quartz_measured": bool(two_k.get("quartz_measured")),
            "quartz_tok_s": two_k.get("quartz_tok_s"),
            "llama_tok_s": two_k.get("llama_tok_s"),
            "nonexclusive_oom": bool(two_k.get("nonexclusive_oom")),
            "reason": "opt123 sitting was not exclusive; allocation failure recorded",
            "source": "fixtures/opt123_combined_stack.json opt016_2k",
        },
        "published_llama_ratios_are_mixed_boundaries": True,
    }


def recorded_span_ms(records: Sequence[Mapping[str, Any]]) -> float | None:
    if not records:
        return None
    starts = []
    ends = []
    for row in records:
        start, end = interval(row)
        starts.append(start)
        ends.append(end)
    return max(ends) - min(starts) if starts else None


def classify_disjoint_intervals(
    records: Sequence[Mapping[str, Any]],
    *,
    wall_ms: float | None = None,
    tokens: int = 1,
    profiler: str = "cuda_event_engine_attribution",
) -> dict[str, Any]:
    """Disjoint device-active / inactive / unobserved. Leaf gaps ≠ idle."""
    charged = chargeable_records(records)
    leaves = [row for row in charged if is_kernel_leaf(row)]
    copies = [
        row
        for row in charged
        if str(row.get("role", "")) in SYNC_ROLES
        and str(row.get("attribution_role", "member")) != "enclosing"
    ]
    host = [row for row in charged if str(row.get("role", "")) in HOST_ROLES]
    graph = [row for row in charged if str(row.get("role", "")) in GRAPH_ROLES]
    device_active = union_ms(leaves + copies + graph)
    host_union = union_ms(host)
    classified = union_ms(leaves + copies + host + graph)
    wall = float(wall_ms) if wall_ms is not None else classified
    unobserved = max(0.0, wall - classified)
    overlap_host_device = max(
        0.0, host_union + device_active - union_ms(leaves + copies + graph + host)
    )
    device_inactive_proven = 0.0
    inactive_method = (
        "not_proven_from_leaf_gaps; CUPTI/Nsight Systems required for "
        "hardware inactivity independent of event leaves"
    )
    if profiler in {"nsight_systems", "cupti"}:
        device_inactive_proven = unobserved
        inactive_method = "profiler_covered_inactive"
    leaf_gaps = classify_timeline(records, wall_ms=wall, tokens=tokens)
    return {
        "tokens": tokens,
        "record_count": len(records),
        "charged_count": len(charged),
        "leaf_kernel_count": len(leaves),
        "device_active_ms": device_active,
        "device_inactive_proven_ms": device_inactive_proven,
        "unobserved_ms": unobserved,
        "host_union_ms": host_union,
        "host_device_overlap_ms": overlap_host_device,
        "never_count_cpu_wait_overlapping_gpu_twice": True,
        "leaf_gap_is_not_gpu_idle": True,
        "leaf_gap_ms_opt115_label": leaf_gaps.get("gpu_idle_gap_ms"),
        "leaf_gap_relabeled": "unobserved_or_unrecorded_device_work",
        "wall_ms": wall,
        "interval_coverage": (classified / wall) if wall > 0.0 else 0.0,
        "profiler": profiler,
        "inactive_method": inactive_method,
        "overlap_not_summed": True,
        "opt115_classify_timeline_not_independent_inactivity_proof": True,
    }


def split_windows(
    records: Sequence[Mapping[str, Any]], outputs: int
) -> dict[str, list[dict[str, Any]]]:
    """Bucket windowed records by distinct token_position, not 0-based step.

    Native instrumentation records only early/middle/late windows. Attribution
    token_position is the session frontier (prefix+step), so 0-based cuts miss
    every bucket.
    """
    del outputs
    positions = sorted(
        {
            int(row.get("token_position", -1) or -1)
            for row in records
            if int(row.get("token_position", -1) or -1) >= 0
        }
    )
    buckets: dict[str, list[dict[str, Any]]] = {
        "early": [],
        "middle": [],
        "late": [],
        "other": [],
    }
    if not positions:
        third = max(1, len(records) // 3)
        return {
            "early": [dict(row) for row in records[:third]],
            "middle": [dict(row) for row in records[third : 2 * third]],
            "late": [dict(row) for row in records[2 * third :]],
            "other": [],
        }
    early_pos = set(positions[: min(4, len(positions))])
    late_pos = set(positions[-min(4, len(positions)) :])
    remaining = [pos for pos in positions if pos not in early_pos | late_pos]
    mid_pos = set(remaining)
    for row in records:
        pos = int(row.get("token_position", -1) or -1)
        if pos in early_pos:
            buckets["early"].append(dict(row))
        elif pos in late_pos:
            buckets["late"].append(dict(row))
        elif pos in mid_pos:
            buckets["middle"].append(dict(row))
        else:
            buckets["other"].append(dict(row))
    return buckets


def weight_read_account(*, prefix: int = 128) -> dict[str, Any]:
    inv = inventory_bytes()
    original = compulsory_decode_bytes(prefix=prefix)
    combined = combined_decode_bytes(prefix=prefix)
    embed_full = int((inv.get("by_role_bytes") or {}).get("token_embedding") or 0)
    embed_row = embedding_row_bytes()
    output_full = int((inv.get("by_role_bytes") or {}).get("output_projection") or 0)
    stored_weights = (
        int(inv.get("q4_k_bytes") or 0)
        + int(inv.get("q8_0_bytes") or 0)
        + int(inv.get("q6_k_bytes") or 0)
        + int(inv.get("f32_bytes") or 0)
    )
    logical = int(combined["phases"]["weights"])
    peak_ms = bytes_to_peak_ms(logical)
    return {
        "prefix": prefix,
        "units": "bytes",
        "stored_bytes": stored_weights,
        "logical_reads_one_decode_token": logical,
        "physical_dram_traffic": None,
        "physical_dram_note": (
            "No DRAM/L2 counter sample on this sitting. Listed 1792 GB/s peak "
            "is not a measured application rate. L2 reuse can make physical "
            "traffic smaller than logical reads."
        ),
        "embedding": {
            "full_table_bytes": embed_full,
            "row_bytes": embed_row,
            "full_table_excluded": True,
            "consumer": "one token embedding lookup per decode token",
        },
        "output_head": {
            "full_projection_bytes": output_full,
            "combined_d2h_bytes": COMBINED_DECODE_D2H,
            "lazy_index_not_full_logits": True,
            "consumer": "OPT-118 lazy greedy index",
        },
        "graph_coverage": {
            "execution_graphs": "ffn_only",
            "ffn_weights_in_graph": True,
            "attention_and_gdn_eager": True,
        },
        "original_compulsory_weights": original["phases"]["weights"],
        "combined_compulsory_weights": combined["phases"]["weights"],
        "listed_peak_gb_s": PEAK_BANDWIDTH_GB_S,
        "listed_peak_ms": peak_ms,
        "measured": False,
        "source": "tensor_inventory_plus_architecture_constants",
    }


def reconcile_busy_vs_compulsory(
    *,
    busy_ms: float | None,
    compulsory_peak_ms: float | None,
    request_ms: float | None,
    timeline_ms: float | None,
    workload: str,
) -> dict[str, Any]:
    contradiction = False
    bound_supported = True
    reasons: list[str] = []
    if busy_ms is not None and compulsory_peak_ms is not None:
        if float(busy_ms) + 1e-6 < float(compulsory_peak_ms):
            contradiction = True
            bound_supported = False
            reasons.append(
                "event busy union is below the listed-peak compulsory weight "
                "minimum; the bound assumes 1792 GB/s with no L2 reuse and "
                "complete leaf coverage. Either physical DRAM < logical reads, "
                "unobserved kernels occupy the gap, or the bound is optimistic."
            )
    timeline_mismatch = False
    if request_ms is not None and timeline_ms is not None:
        if abs(float(request_ms) - float(timeline_ms)) > 0.5:
            timeline_mismatch = True
            reasons.append(
                "timeline wall and request average are different quantities "
                "(single-token instrumented decode vs multi-token request "
                "including prefill). Not a demonstrated savings."
            )
    return {
        "workload": workload,
        "busy_ms": busy_ms,
        "compulsory_peak_ms": compulsory_peak_ms,
        "request_ms": request_ms,
        "timeline_ms": timeline_ms,
        "busy_below_compulsory": contradiction,
        "bound_supported": bound_supported,
        "timeline_vs_request_mismatch": timeline_mismatch,
        "observed_idle_is_not_unavoidable_lower_bound": True,
        "reasons": reasons,
        "do_not_add_observed_idle_as_unavoidable_bound": True,
    }


def weight_reads_and_reconcile() -> dict[str, Any]:
    opt124 = load_json(OPT124_FIXTURE) if OPT124_FIXTURE.is_file() else {}
    timeline = opt124.get("timeline") or {}
    bounds = (opt124.get("bounds") or {}).get("workloads") or {}
    account = {
        name: weight_read_account(prefix=PREFIX[name]) for name in ("d128", "d2048")
    }
    recon: dict[str, Any] = {}
    d128_busy = (timeline.get("d128") or {}).get("gpu_busy_union_ms")
    d128_wall = (timeline.get("d128") or {}).get("wall_ms")
    d128_comp = ((bounds.get("d128") or {}).get("byte_bound") or {}).get(
        "byte_critical_ms"
    ) or 10.182878178571428
    d2048_busy = (timeline.get("d2048") or {}).get("gpu_busy_union_ms")
    d2048_wall = (timeline.get("d2048") or {}).get("wall_ms")
    d2048_req = ((bounds.get("d2048") or {}).get("byte_bound") or {}).get("wall_ms")
    recon["d128"] = reconcile_busy_vs_compulsory(
        busy_ms=d128_busy,
        compulsory_peak_ms=d128_comp,
        request_ms=ms_from_tok_s(
            (
                (load_json(OPT123_FIXTURE).get("performance") or {}).get("d128") or {}
            ).get("candidate_tok_s")
        )
        if OPT123_FIXTURE.is_file()
        else None,
        timeline_ms=d128_wall,
        workload="d128",
    )
    recon["d2048"] = reconcile_busy_vs_compulsory(
        busy_ms=d2048_busy,
        compulsory_peak_ms=10.182878178571428,
        request_ms=d2048_req or 24.019346388313107,
        timeline_ms=d2048_wall,
        workload="d2048",
    )
    return {
        "account": account,
        "reconcile": recon,
        "counter_sampled_dram_l2": False,
        "matching_access_pattern_probe": False,
        "opt124_d128_busy_vs_compulsory": recon["d128"],
        "opt124_d2048_timeline_vs_request": recon["d2048"],
    }


def shared_keep_protocol() -> dict[str, Any]:
    return {
        "frozen_for": [
            "OPT-126",
            "OPT-127",
            "OPT-128",
            "OPT-129",
            "OPT-130",
            "OPT-131",
            "OPT-132",
        ],
        "parent": "combined_opt118_opt119",
        "parent_alias": "post124_combined",
        "historical_control": "post113_selected",
        "quality_against": "post113",
        "quality_contract": "opt116_generated_v1",
        "quality_budget_ratchets": False,
        "screen_warmups": 3,
        "screen_pairs": 5,
        "acceptance_warmups": 3,
        "acceptance_pairs": 10,
        "decode_output_tokens": DECODE_TOKENS,
        "workloads": ["p4096", "d128", "d2048"],
        "metrics_required": ["decode_only", "complete_request"],
        "throughput_ratio_lower_bound": 1.0,
        "non_target_point_ratio_min": 0.98,
        "decode_p95_ratio_max": 1.05,
        "request_time_materiality": MATERIALITY,
        "report_2pct_materiality_separately_from_smaller_keep": True,
        "no_extra_sampling": True,
        "no_speculation_mtp_batching_sparse_attention": True,
        "preserve_gguf_tokenizer_dense_bf16_kv_fp32_gdn": True,
        "preserve_lazy_logits_and_norm_q8_fusion": True,
        "live_131072_capacity": True,
        "reserve_bytes": 1_610_612_736,
        "serialize_gpu": True,
        "max_candidates_per_implementation": 2,
        "source_dossier": "tasks/OPT-125.md",
        "rejected_prerequisite_supplies_retained_path": True,
    }


def rank_opt126_131(
    *,
    intervals: Mapping[str, Any] | None = None,
    corrections: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    intervals = intervals or {}
    corrections = corrections or historical_corrections()
    d128 = (corrections.get("p4096_d128_d2048") or {}).get("d128") or {}
    d2048 = (corrections.get("p4096_d128_d2048") or {}).get("d2048") or {}
    unobserved = 0.0
    proven_inactive = 0.0
    for name in ("d128", "d2048"):
        block = intervals.get(name) or {}
        unobserved = max(unobserved, float(block.get("unobserved_ms") or 0.0))
        proven_inactive = max(
            proven_inactive, float(block.get("device_inactive_proven_ms") or 0.0)
        )
    decode_gap_d128 = None
    decode_gap_d2048 = None
    llama_d128 = d128.get("llama_tok_s")
    llama_d2048 = d2048.get("llama_tok_s")
    comb_d128 = (d128.get("combined_corrected") or {}).get("mean_decode_only_tok_s")
    comb_d2048 = (d2048.get("combined_corrected") or {}).get("mean_decode_only_tok_s")
    if comb_d128 and llama_d128:
        decode_gap_d128 = ms_from_tok_s(comb_d128) - ms_from_tok_s(llama_d128)
    if comb_d2048 and llama_d2048:
        decode_gap_d2048 = ms_from_tok_s(comb_d2048) - ms_from_tok_s(llama_d2048)
    rows = [
        {
            "task": "OPT-126",
            "disposition": "proceed",
            "confidence": "source_grounded",
            "causal_removable_ms": None,
            "overlap_with": ["OPT-127"],
            "reason": (
                "OPT-117 recapture on frontier/GDN pointer changes is a "
                "source-grounded graph-input instability. Missing profiling "
                "does not block this correctness prerequisite."
            ),
        },
        {
            "task": "OPT-127",
            "disposition": "proceed",
            "confidence": "conditional_on_opt126",
            "causal_removable_ms": None,
            "overlap_with": ["OPT-126", "OPT-128"],
            "reason": (
                "Recapture-free replay is the mechanism that can convert "
                "OPT-126 into request time. Quantify after stable inputs; "
                "do not treat ~10 ms leaf gaps as the recapture savings."
            ),
        },
        {
            "task": "OPT-128",
            "disposition": "insufficient_evidence",
            "confidence": "low_until_api_correlated_inactive",
            "causal_removable_ms": proven_inactive if proven_inactive > 0 else None,
            "overlap_with": ["OPT-127"],
            "reason": (
                "Host stalls overlapping GPU execution must not be counted "
                "twice. Leaf-gap ~10 ms/token is unobserved, not proven "
                "CPU-submission starvation. Proceed only after OPT-125 "
                "activity traces attribute true inactive spans."
            ),
        },
        {
            "task": "OPT-129",
            "disposition": "proceed",
            "confidence": "source_grounded_component_unknown",
            "causal_removable_ms": decode_gap_d2048,
            "overlap_with": ["OPT-130"],
            "reason": (
                "Matched decode-only D2048 still trails pinned llama after "
                "boundary correction, but 0.620× was a mixed-boundary "
                "whole-engine ratio and is not an attention component ratio. "
                "Component replay is required; shipping hybrid_crossover@1024 "
                "stays."
            ),
        },
        {
            "task": "OPT-130",
            "disposition": "insufficient_evidence",
            "confidence": "blocked_on_opt129",
            "causal_removable_ms": None,
            "overlap_with": ["OPT-129", "OPT-131"],
            "reason": (
                "No matched attention-component candidate is frozen. OPT-129 "
                "must name at most two causal transfers before implementation."
            ),
        },
        {
            "task": "OPT-131",
            "disposition": "insufficient_evidence",
            "confidence": "blocked_on_graph_host_attention",
            "causal_removable_ms": None,
            "overlap_with": ["OPT-128", "OPT-130"],
            "reason": (
                "Residual launch-chain fusion is only justified after graph, "
                "host, and attention verdicts. OPT-119 already kept norm→Q8."
            ),
        },
    ]
    return {
        "ranked": rows,
        "unobserved_ms_not_proven_inactive": unobserved,
        "proven_device_inactive_ms": proven_inactive,
        "matched_decode_only_gap_ms": {
            "d128": decode_gap_d128,
            "d2048": decode_gap_d2048,
        },
        "missing_profiling_does_not_block_opt126_or_opt129": True,
        "labels": list(DISPOSITIONS),
    }


def answers(
    *,
    intervals: Mapping[str, Any] | None = None,
    corrections: Mapping[str, Any] | None = None,
    ranking: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    intervals = intervals or {}
    corrections = corrections or historical_corrections()
    ranking = ranking or rank_opt126_131(intervals=intervals, corrections=corrections)
    d128 = intervals.get("d128") or {}
    leaf_gap = d128.get("leaf_gap_ms_opt115_label")
    proven = float(d128.get("device_inactive_proven_ms") or 0.0)
    d128_ratio = ((corrections.get("p4096_d128_d2048") or {}).get("d128") or {}).get(
        "matched_decode_only_combined_over_llama"
    )
    d2048_ratio = ((corrections.get("p4096_d128_d2048") or {}).get("d2048") or {}).get(
        "matched_decode_only_combined_over_llama"
    )
    published_d128 = (
        (corrections.get("p4096_d128_d2048") or {}).get("d128") or {}
    ).get("published_combined_over_llama_request_over_decode_only")
    published_d2048 = (
        (corrections.get("p4096_d128_d2048") or {}).get("d2048") or {}
    ).get("published_combined_over_llama_request_over_decode_only")
    d8192 = (corrections.get("long_context") or {}).get("d8192") or {}
    d32768 = (corrections.get("long_context") or {}).get("d32768") or {}
    ten_ms_true_inactive = False
    ten_ms_reason = (
        "The ~10 ms/token figure is the OPT-115/124 leaf-gap remainder after "
        "event-union busy time, not an independent CUPTI/Nsight proof of "
        "hardware inactivity. Relabeled unobserved. Proven inactive "
        f"ms/token={proven}."
    )
    removable = ranking.get("proven_device_inactive_ms") or 0.0
    llama_survives = False
    if d128_ratio and d2048_ratio and published_d128 and published_d2048:
        llama_survives = (
            abs(float(d128_ratio) - float(published_d128)) < 0.02
            and abs(float(d2048_ratio) - float(published_d2048)) < 0.02
        )
    long_survives = False
    hist_d8192 = d8192.get("historical_tok_s")
    corr_d8192 = d8192.get("decode_only_tok_s")
    if hist_d8192 and corr_d8192:
        long_survives = abs(float(corr_d8192) / float(hist_d8192) - 1.0) < 0.05
    return {
        "is_10ms_per_token_true_inactive": ten_ms_true_inactive,
        "is_10ms_per_token_true_inactive_reason": ten_ms_reason,
        "leaf_gap_ms_historical": leaf_gap,
        "causally_removable_ms_per_token": removable if removable else None,
        "causally_removable_reason": (
            "Only API-correlated proven device-inactive spans are removable "
            "host/submission time. Source-grounded graph (OPT-126/127) and "
            "attention (OPT-129) work may still proceed without that number."
        ),
        "llama_ratios_survive_matched_boundaries": llama_survives,
        "llama_published_d128": published_d128,
        "llama_matched_decode_only_d128": d128_ratio,
        "llama_published_d2048": published_d2048,
        "llama_matched_decode_only_d2048": d2048_ratio,
        "long_context_ratios_survive_matched_boundaries": long_survives,
        "d8192_historical_request_tok_s": hist_d8192,
        "d8192_decode_only_tok_s": corr_d8192,
        "d32768_historical_request_tok_s": d32768.get("historical_tok_s"),
        "d32768_decode_only_tok_s": d32768.get("decode_only_tok_s"),
        "unknowns_are_valid_findings": True,
    }


def sitting_identity() -> dict[str, Any]:
    source, dirty = git_identity()
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
        "parent": "combined_opt118_opt119",
        "image": IMAGE,
        "model": MODEL,
        "post113_authenticated": authenticate_post113(),
        "pinned_llama": authenticate_pinned_llama(),
        "llama_cpp": inspect_repo_revision(ROOT.parent / "llama.cpp"),
        "ds4": inspect_repo_revision(ROOT.parent / "ds4"),
    }


def empty_payload(*, reason: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    catalog = metric_catalog()
    corrections = historical_corrections()
    weights = weight_reads_and_reconcile()
    protocol = shared_keep_protocol()
    ranking = rank_opt126_131(corrections=corrections)
    ans = answers(corrections=corrections, ranking=ranking)
    return {
        "schema_version": 1,
        "task": "OPT-125",
        "status": "analysis",
        "measurement_utc": utc_now(),
        "identity": dict(identity),
        "metric_catalog": catalog,
        "historical_corrections": corrections,
        "matched_baseline": {
            "hardware_executed": False,
            "reason": reason,
            "aa": {},
            "combined": {},
            "post113": {},
            "llama": {},
        },
        "activity_trace": {
            "hardware_executed": False,
            "reason": reason,
            "nsys_available": None,
            "ncu_available": None,
            "cupti_linked": False,
            "profiler_absence_is_limitation_not_zero_activity": True,
        },
        "disjoint_intervals": {
            "d128": classify_disjoint_intervals([], wall_ms=None),
            "d2048": classify_disjoint_intervals([], wall_ms=None),
        },
        "weight_reads": weights,
        "long_context": {
            "hardware_executed": False,
            "reason": reason,
            "probes": corrections.get("long_context"),
            "opt016_2k": corrections.get("opt016_2k"),
        },
        "ranking": ranking,
        "shared_keep_protocol": protocol,
        "answers": ans,
        "production_kept": True,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "gpu_blocker": reason,
        "hardware_executed": False,
        "report_path": "evidence/optimization/opt125-decode-accounting/REPORT.md",
        "proof_limit": PROOF,
        "synthetic_metric_cases": synthetic_metric_cases(),
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
    if phase == "metrics":
        payload["metric_catalog"] = block.get("metric_catalog")
        payload["synthetic_metric_cases"] = block.get("synthetic_metric_cases")
        payload["historical_corrections"] = block.get("historical_corrections")
    if phase == "freeze":
        payload["identity"] = block.get("identity") or payload.get("identity")
    if phase == "matched-baseline":
        payload["matched_baseline"] = dict(block)
        payload["hardware_executed"] = bool(payload.get("hardware_executed")) or bool(
            block.get("hardware_executed")
        )
        if block.get("hardware_executed"):
            payload["gpu_blocker"] = None
    if phase == "activity-trace":
        payload["activity_trace"] = dict(block)
        payload["hardware_executed"] = bool(payload.get("hardware_executed")) or bool(
            block.get("hardware_executed")
        )
    if phase == "intervals":
        payload["disjoint_intervals"] = dict(block.get("intervals") or block)
    if phase == "weight-reads":
        payload["weight_reads"] = dict(block.get("weight_reads") or block)
    if phase == "long-context":
        payload["long_context"] = dict(block)
        payload["hardware_executed"] = bool(payload.get("hardware_executed")) or bool(
            block.get("hardware_executed")
        )
        if block.get("hardware_executed"):
            corrections = dict(
                payload.get("historical_corrections") or historical_corrections()
            )
            merged = dict(corrections.get("long_context") or {})
            for name, probe in dict(block.get("probes") or {}).items():
                if not probe.get("ok"):
                    continue
                prior = dict(merged.get(name) or {})
                prior.update(
                    {
                        "fresh_measured": True,
                        "fresh_exclusive_sitting": True,
                        "fresh_measured_at": block.get("measured_at"),
                        "historical_tok_s": prior.get("historical_tok_s")
                        or probe.get("request_tok_s"),
                        "decode_only_tok_s": probe.get("decode_only_tok_s"),
                        "request_tok_s": probe.get("request_tok_s"),
                        "prefill_wall_ms": probe.get("prefill_wall_ms"),
                        "decode_only_wall_ms": probe.get("decode_only_wall_ms"),
                        "request_wall_ms": probe.get("request_wall_ms"),
                        "populated_cache": probe.get("populated_cache"),
                        "prefix": probe.get("prefix"),
                        "oom": False,
                        "source": (
                            "build/optimization-runs/opt125-exclusive long-context "
                            "after stopping zanzara-archive GPU services"
                        ),
                    }
                )
                merged[name] = prior
            corrections["long_context"] = merged
            payload["historical_corrections"] = corrections
            payload["exclusive_long_context_sitting"] = {
                "measured_at": block.get("measured_at"),
                "note": (
                    "Exclusive GPU sitting; zanzara-archive audioset_ast and "
                    "diarization containers stopped before measurement."
                ),
                "probes": block.get("probes"),
            }
    if phase == "ranking":
        payload["ranking"] = dict(block.get("ranking") or block)
        payload["answers"] = dict(block.get("answers") or payload.get("answers") or {})
        payload["shared_keep_protocol"] = dict(
            block.get("shared_keep_protocol")
            or payload.get("shared_keep_protocol")
            or {}
        )
    payload["measurement_utc"] = utc_now()
    payload["production_kept"] = True
    payload["claims_throughput"] = False
    payload["claims_performance_improvement"] = False
    payload["status"] = "analysis"
    payload["report_path"] = "evidence/optimization/opt125-decode-accounting/REPORT.md"
    payload["proof_limit"] = PROOF
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
    catalog = payload.get("metric_catalog") or metric_catalog()
    corr = payload.get("historical_corrections") or {}
    baseline = payload.get("matched_baseline") or {}
    activity = payload.get("activity_trace") or {}
    intervals = payload.get("disjoint_intervals") or {}
    weights = payload.get("weight_reads") or {}
    long_ctx = payload.get("long_context") or {}
    ranking = payload.get("ranking") or {}
    ans = payload.get("answers") or {}
    protocol = payload.get("shared_keep_protocol") or {}
    d128c = (corr.get("p4096_d128_d2048") or {}).get("d128") or {}
    d2048c = (corr.get("p4096_d128_d2048") or {}).get("d2048") or {}
    d8192 = (corr.get("long_context") or {}).get("d8192") or {}
    d32768 = (corr.get("long_context") or {}).get("d32768") or {}
    d131040 = (corr.get("long_context") or {}).get("d131040") or {}
    exclusive = payload.get("exclusive_long_context_sitting") or {}
    fresh_probes = dict(long_ctx.get("probes") or exclusive.get("probes") or {})
    fresh_long_hardware = bool(
        long_ctx.get("hardware_executed") or exclusive.get("probes")
    )
    opt016 = long_ctx.get("opt016_2k") or {}
    long_context_note = (
        "Fresh exclusive long-cache probes measured at capacity 131072 after "
        "stopping zanzara-archive GPU services. Historical OPT-123 request "
        "tok/s values are retained; fresh decode-only and request rates confirm "
        "the boundary correction (request includes prefill)."
        if fresh_long_hardware
        and not any((fresh_probes.get(n) or {}).get("oom") for n in LONG_PROBES)
        else (
            "Fresh long-cache probes blocked on this sitting. "
            "Historical stored-field corrections remain authoritative."
        )
    )
    opt016_note = (
        f"OPT-016 2K Quartz blocked: `{opt016.get('reason')}` "
        f"(not OOM on exclusive sitting)."
        if opt016.get("reason") == "decode_segments8_requires_matching_session_capacity"
        else (
            "OPT-016 2K remains blocked; see sidecar reason."
            if not opt016.get("measured")
            else "OPT-016 2K measured."
        )
    )
    recon = (
        weights.get("reconcile") or weights.get("opt124_d128_busy_vs_compulsory") or {}
    )
    if "d128" not in recon and isinstance(
        weights.get("opt124_d128_busy_vs_compulsory"), Mapping
    ):
        recon = {
            "d128": weights.get("opt124_d128_busy_vs_compulsory"),
            "d2048": weights.get("opt124_d2048_timeline_vs_request"),
        }
    rank_rows = ranking.get("ranked") or []
    rank_md = "\n".join(
        f"| {row.get('task')} | {row.get('disposition')} | "
        f"{_fmt(row.get('causal_removable_ms'))} | {row.get('confidence')} | "
        f"{row.get('reason')} |"
        for row in rank_rows
    )
    aa = baseline.get("aa") or {}
    text = f"""# OPT-125 — Decode timing, GPU activity, and traffic evidence

Status: `{payload.get("status")}`. measurement_utc=`{payload.get("measurement_utc")}`.
Diagnostics only. claims_throughput=`false`. claims_performance_improvement=`false`.
production_kept=`true`.

## Identity

device=`{identity.get("device")}` source=`{identity.get("source_revision")}`
state=`{identity.get("source_state")}` parent=`{identity.get("parent")}`
graphs=`{identity.get("execution_graphs")}` llama pin=`{identity.get("llama_revision")}`.
post113 ok=`{(identity.get("post113_authenticated") or {}).get("ok")}`.
pinned llama ok=`{(identity.get("pinned_llama") or {}).get("ok")}`.
hardware_executed=`{payload.get("hardware_executed")}`.
gpu_blocker=`{payload.get("gpu_blocker")}`.

## Frozen metric names

Correction date `{catalog.get("correction_date")}`. First sampled token needs
eval: `{catalog.get("first_sampled_token_needs_eval")}`.
{catalog.get("first_sampled_token_reason")}

| Name | Clock / boundary |
|---|---|
| setup | session/workspace/graph create; excluded from request and decode |
| prefill | request start through `sync_tokens` |
| time_to_first_emitted_token | prefill plus first decode eval on planned-token probes |
| decode_only | after prefill through last decode eval; denominator = emitted tokens |
| sample_eval_output | host sample + eval + output inside decode; count evals explicitly |
| complete_request | before prefill through last decode eval; OPT-123 `run_session_arm` tok/s |
| fixed_token_engine_probe | planned tokens, no sampler |

OPT-123 `run_session_arm` starts the clock before prefill and divides outputs
by that entire wall. OPT-115's request probe subtracts prefill. Pinned llama
decode oracle is decode-only. Historical fields are retained; corrections are
dated `{CORRECTION_DATE}`.

## Historical corrections (stored fields)

| Workload | historical combined tok/s | metric then | decode-only tok/s | request tok/s | llama tok/s | llama metric | published ratio | matched decode-only ratio |
|---|---:|---|---:|---:|---:|---|---:|---:|
| D128 | {_fmt(d128c.get("historical_combined_tok_s"))} | complete_request | {_fmt((d128c.get("combined_corrected") or {}).get("mean_decode_only_tok_s"))} | {_fmt((d128c.get("combined_corrected") or {}).get("mean_request_tok_s"))} | {_fmt(d128c.get("llama_tok_s"))} | {d128c.get("llama_metric")} | {_fmt(d128c.get("published_combined_over_llama_request_over_decode_only"))} | {_fmt(d128c.get("matched_decode_only_combined_over_llama"))} |
| D2048 | {_fmt(d2048c.get("historical_combined_tok_s"))} | complete_request | {_fmt((d2048c.get("combined_corrected") or {}).get("mean_decode_only_tok_s"))} | {_fmt((d2048c.get("combined_corrected") or {}).get("mean_request_tok_s"))} | {_fmt(d2048c.get("llama_tok_s"))} | {d2048c.get("llama_metric")} | {_fmt(d2048c.get("published_combined_over_llama_request_over_decode_only"))} | {_fmt(d2048c.get("matched_decode_only_combined_over_llama"))} |

Long-context OPT-123 request tok/s versus decode-only from wall−ttft:

| Probe | historical request tok/s | decode-only tok/s | prefill ms | decode ms |
|---|---:|---:|---:|---:|
| D8192 | {_fmt(d8192.get("historical_tok_s"))} | {_fmt(d8192.get("decode_only_tok_s"))} | {_fmt(d8192.get("prefill_wall_ms"))} | {_fmt(d8192.get("decode_only_wall_ms"))} |
| D32768 | {_fmt(d32768.get("historical_tok_s"))} | {_fmt(d32768.get("decode_only_tok_s"))} | {_fmt(d32768.get("prefill_wall_ms"))} | {_fmt(d32768.get("decode_only_wall_ms"))} |
| D131040 | n/a (OPT-123 OOM) | {_fmt(d131040.get("decode_only_tok_s"))} | {_fmt(d131040.get("prefill_wall_ms"))} | {_fmt(d131040.get("decode_only_wall_ms"))} |

Fresh exclusive sitting (2026-09-14, post124 `ffn_only`):

| Probe | request tok/s | decode-only tok/s | TTFT ms | decode p50 ms |
|---|---:|---:|---:|---:|
| D8192 | {_fmt((fresh_probes.get("d8192") or {}).get("request_tok_s"))} | {_fmt((fresh_probes.get("d8192") or {}).get("decode_only_tok_s"))} | {_fmt((fresh_probes.get("d8192") or {}).get("ttft_ms"))} | {_fmt((fresh_probes.get("d8192") or {}).get("p50_ms"))} |
| D32768 | {_fmt((fresh_probes.get("d32768") or {}).get("request_tok_s"))} | {_fmt((fresh_probes.get("d32768") or {}).get("decode_only_tok_s"))} | {_fmt((fresh_probes.get("d32768") or {}).get("ttft_ms"))} | {_fmt((fresh_probes.get("d32768") or {}).get("p50_ms"))} |
| D131040 | {_fmt((fresh_probes.get("d131040") or {}).get("request_tok_s"))} | {_fmt((fresh_probes.get("d131040") or {}).get("decode_only_tok_s"))} | {_fmt((fresh_probes.get("d131040") or {}).get("ttft_ms"))} | {_fmt((fresh_probes.get("d131040") or {}).get("p50_ms"))} |

41.420→8.066 (D8192) and 22.118→1.352 (D32768) mix OPT-115 decode-only with
OPT-123 complete-request. They are not evidence of a decode regression until
both sides use the same boundary.

## Fresh matched baseline

hardware=`{baseline.get("hardware_executed")}` reason=`{baseline.get("reason")}`.
A/A `{aa}`.

Combined / post113 / llama raw records are retained under
`build/optimization-runs/opt125/` when GPU phases run. Fixed-token engine
probes are labeled separately from free-running sample/eval/output loops.

## Activity trace

method=`{(activity.get("d128") or activity).get("method") if isinstance(activity.get("d128"), Mapping) else activity.get("method")}`.
nsys=`{activity.get("nsys_available")}` ncu=`{activity.get("ncu_available")}`
cupti_linked=`{activity.get("cupti_linked")}`.
{"Nsight Systems is installed in the pinned CUDA image; this phase still used CUDA-event attribution only (no nsys capture run)." if activity.get("nsys_available") else "Profiler absence is an attribution limitation, not zero activity."}
Windows: early / middle / late on the same 256-output trajectory.

## Disjoint intervals

Leaf gaps from `classify_timeline` are **unobserved**, not GPU idle.
CPU waits that overlap device-active intervals are not counted twice.

| Workload | device_active_ms | proven_inactive_ms | unobserved_ms | leaf_gap_ms (opt115 label) |
|---|---:|---:|---:|---:|
| D128 | {_fmt((intervals.get("d128") or {}).get("device_active_ms"))} | {_fmt((intervals.get("d128") or {}).get("device_inactive_proven_ms"))} | {_fmt((intervals.get("d128") or {}).get("unobserved_ms"))} | {_fmt((intervals.get("d128") or {}).get("leaf_gap_ms_opt115_label"))} |
| D2048 | {_fmt((intervals.get("d2048") or {}).get("device_active_ms"))} | {_fmt((intervals.get("d2048") or {}).get("device_inactive_proven_ms"))} | {_fmt((intervals.get("d2048") or {}).get("unobserved_ms"))} | {_fmt((intervals.get("d2048") or {}).get("leaf_gap_ms_opt115_label"))} |

## Weight reads and busy-time contradiction

D128 busy vs compulsory: bound_supported=
`{(recon.get("d128") or {}).get("bound_supported")}` busy_below_compulsory=
`{(recon.get("d128") or {}).get("busy_below_compulsory")}`.
D2048 timeline vs request mismatch=
`{(recon.get("d2048") or {}).get("timeline_vs_request_mismatch")}`.
Observed idle is not an unavoidable lower bound. DRAM/L2 was not
counter-sampled; 1792 GB/s remains a listed peak.

Embedding consumers read one row, not the full table. Combined-stack decode
D2H is the 4-byte lazy index, not full logits.

## Long-context / OPT-016 2K

Fresh probes: hardware=`{long_ctx.get("hardware_executed")}`
reason=`{long_ctx.get("reason")}`. {long_context_note}
{opt016_note}
Historical OPT-123 D131040 OOM on a non-exclusive sitting is retained in
fixtures; the exclusive sitting measured D131040+32 successfully.

## Ranking for OPT-126–131

| Task | disposition | causal ms | confidence | reason |
|---|---|---:|---|---|
{rank_md}

Parent for later tasks: `{protocol.get("parent")}`. Keep protocol: 3+5 screen,
3+10 acceptance, both decode-only and complete-request metrics, 2% request
materiality reported separately.

## Answers

1. **Is 10 ms/token true inactive?** `{ans.get("is_10ms_per_token_true_inactive")}`.
   {ans.get("is_10ms_per_token_true_inactive_reason")}
2. **How much is causally removable?** `{_fmt(ans.get("causally_removable_ms_per_token"))}` ms/token proven.
   {ans.get("causally_removable_reason")}
3. **Do llama and long-context ratios survive matched boundaries?**
   llama `{ans.get("llama_ratios_survive_matched_boundaries")}`
   (D128 published `{_fmt(ans.get("llama_published_d128"))}` vs matched
   `{_fmt(ans.get("llama_matched_decode_only_d128"))}`; D2048 published
   `{_fmt(ans.get("llama_published_d2048"))}` vs matched
   `{_fmt(ans.get("llama_matched_decode_only_d2048"))}`).
   Fresh sitting decode-only combined/llama: D128
   `{_fmt(ans.get("fresh_matched_decode_only_d128"))}`
   (`{_fmt(ans.get("fresh_decode_only_combined_d128"))}` /
   `{_fmt(ans.get("fresh_llama_decode_only_d128"))}`), D2048
   `{_fmt(ans.get("fresh_matched_decode_only_d2048"))}`
   (`{_fmt(ans.get("fresh_decode_only_combined_d2048"))}` /
   `{_fmt(ans.get("fresh_llama_decode_only_d2048"))}`).
   long-context `{ans.get("long_context_ratios_survive_matched_boundaries")}`
   (D8192 request `{_fmt(ans.get("d8192_historical_request_tok_s"))}` vs
   decode-only `{_fmt(ans.get("d8192_decode_only_tok_s"))}`).
   {"Fresh exclusive long-cache probes measured at 131072 capacity (see table above)." if fresh_long_hardware else "Fresh long-cache probes remain blocked on this sitting."}
   {"OPT-016 2K Quartz remains blocked by decode_segments8 session-capacity requirements (not OOM)." if opt016.get("reason") == "decode_segments8_requires_matching_session_capacity" else ""}

Unknowns are valid findings. Throughput improvement is not a requirement.
"""
    REPORT.write_text(text, encoding="utf-8")


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    if missing:
        raise AccountingError(f"fixture missing keys {missing}")
    if payload.get("claims_throughput") is not False:
        raise AccountingError("claims_throughput must be false")
    if payload.get("claims_performance_improvement") is not False:
        raise AccountingError("claims_performance_improvement must be false")
    if payload.get("production_kept") is not True:
        raise AccountingError("production_kept must remain true")
    catalog = payload.get("metric_catalog") or {}
    names = set((catalog.get("metrics") or {}).keys())
    required = {
        "setup",
        "prefill",
        "time_to_first_emitted_token",
        "decode_only",
        "sample_eval_output",
        "complete_request",
        "fixed_token_engine_probe",
    }
    if not required.issubset(names):
        raise AccountingError(f"metric catalog missing {required - names}")
    if catalog.get("first_sampled_token_needs_eval") is not False:
        raise AccountingError("first sampled token must not require eval")
    ranked = (payload.get("ranking") or {}).get("ranked") or []
    tasks = [row.get("task") for row in ranked]
    if tasks != list(RANKED_TASKS):
        raise AccountingError(f"ranking tasks {tasks} != {list(RANKED_TASKS)}")
    for row in ranked:
        if row.get("disposition") not in DISPOSITIONS:
            raise AccountingError(f"{row.get('task')} missing disposition")
    answers_block = payload.get("answers") or {}
    if "is_10ms_per_token_true_inactive" not in answers_block:
        raise AccountingError("answers must address 10ms/token inactivity")
    if (payload.get("historical_corrections") or {}).get(
        "do_not_silently_rewrite"
    ) is not True:
        raise AccountingError("historical corrections must not silently rewrite")
    intervals = payload.get("disjoint_intervals") or {}
    for name in ("d128", "d2048"):
        block = intervals.get(name) or {}
        if block and block.get("leaf_gap_is_not_gpu_idle") is False:
            raise AccountingError("leaf gaps must not be treated as GPU idle")
    return {"ok": True, "task": "OPT-125"}


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records = [
        json.loads(line[len(prefix) :])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if not records:
        return {}
    return records[-1]


def parse_opt125_records(text: str) -> list[dict[str, Any]]:
    for line in text.splitlines():
        if line.startswith(RECORDS_PREFIX):
            payload = json.loads(line.split("=", 1)[1])
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def mean_arm_metric(summary: Mapping[str, Any], key: str) -> float | None:
    values: list[float] = []
    for row in summary.get("pairs") or []:
        for arm_name in ("A", "B"):
            value = (row.get(arm_name) or {}).get(key)
            if value is not None:
                values.append(float(value))
    if not values:
        return None
    return mean(values)


def aa_from_payload(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    pairs = payload.get("pairs") or []
    arm_a = []
    arm_b = []
    for row in pairs:
        a = (row.get("A") or {}).get(key)
        b = (row.get("B") or {}).get(key)
        if a is not None:
            arm_a.append(float(a))
        if b is not None:
            arm_b.append(float(b))
    return aa_verdict_for_workload(arm_a, arm_b)


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [*docker_common(IMAGE, tier), *listed]
    completed = subprocess.run(listed, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise AccountingError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def llama_command(inner: Sequence[str]) -> list[str]:
    listed = docker_common(IMAGE, "acceptance")
    listed[-1:-1] = ["-e", f"LD_LIBRARY_PATH={LLAMA_LD}"]
    return [*listed, *inner]


def classify_cuda_failure(text: str) -> dict[str, Any]:
    oom = "cudaErrorMemoryAllocation" in text or "out of memory" in text.lower()
    return {
        "oom": oom,
        "generic_failed_process_is_not_oom": not oom,
        "cudaErrorMemoryAllocation": "cudaErrorMemoryAllocation" in text,
        "tail": text[-1500:],
    }


def run_metrics(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "metrics",
        "mode": mode,
        "ok": True,
        "metric_catalog": metric_catalog(),
        "synthetic_metric_cases": synthetic_metric_cases(),
        "historical_corrections": historical_corrections(),
        "family_plan": family_plan(mode, "metrics"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "metrics.json", payload)
    persist(merge_phase(current_fixture(), "metrics", payload))
    return payload


def run_freeze(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = sitting_identity()
    post113 = identity.get("post113_authenticated") or {}
    llama = identity.get("pinned_llama") or {}
    if post113.get("ok") is not True:
        raise AccountingError(f"post113 authentication failed: {post113}")
    if llama.get("ok") is not True:
        raise AccountingError(f"pinned llama authentication failed: {llama}")
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "freeze",
        "mode": mode,
        "ok": True,
        "identity": identity,
        "family_plan": family_plan(mode, "freeze"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "freeze.json", payload)
    persist(merge_phase(current_fixture(), "freeze", payload))
    return payload


def run_matched_baseline(
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
    hardware = False
    aa: dict[str, Any] = {}
    combined: dict[str, Any] = {}
    post113: dict[str, Any] = {}
    llama_block: dict[str, Any] = {"measured": False}
    identity_native: dict[str, Any] = {}
    if execute is None:
        reason = blocker or "gpu_unavailable"
    else:
        reason = None
        ident = execute([f"./{NATIVE}", "--workload", "identity", MODEL], "screen")
        (run_dir / "identity.txt").write_text(
            ident.stdout + ident.stderr, encoding="utf-8"
        )
        identity_native = parse_prefixed(ident.stdout, RESULT_PREFIX)
        hardware = True
        for stack, sink in (("combined", combined), ("post113", post113)):
            for name, prefix in (("p4096", 4096), ("d128", 128), ("d2048", 2048)):
                tokens = 0 if name == "p4096" else DECODE_TOKENS
                completed = execute(
                    [
                        f"./{NATIVE}",
                        "--workload",
                        "matched-baseline",
                        "--stack",
                        stack,
                        "--prefix",
                        str(prefix),
                        "--warmups",
                        str(AA_WARMUPS),
                        "--samples",
                        str(AA_PAIRS),
                        "--tokens",
                        str(tokens),
                        MODEL,
                    ],
                    "screen",
                )
                raw = completed.stdout + completed.stderr
                (run_dir / f"{stack}-{name}.txt").write_text(raw, encoding="utf-8")
                summary = parse_prefixed(completed.stdout, RESULT_PREFIX)
                dump_json(run_dir / f"{stack}-{name}.json", summary)
                sink[name] = summary
                if stack == "combined":
                    aa[name] = {
                        "decode_only": aa_from_payload(summary, "decode_only_tok_s"),
                        "request": aa_from_payload(summary, "request_tok_s"),
                    }
        try:
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
                raise AccountingError(
                    "llama-bench failed\n" + bench.stdout + bench.stderr
                )
            llama_p = parse_llama_bench(
                bench.stdout + bench.stderr,
                lambda row: row.get("n_prompt") == 4096,
                "llama-bench JSON with n_prompt 4096",
            )[0]
            dump_json(run_dir / "llama-bench-4k.json", llama_p)

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
                completed = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True
                )
                if completed.returncode != 0:
                    raise AccountingError(
                        f"llama decode {prefix} failed\n"
                        + completed.stdout
                        + completed.stderr
                    )
                record = parse_prefixed(
                    completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX
                )
                dump_json(run_dir / f"llama-decode-d{prefix}.json", record)
                return record

            llama_d128 = decode(128)
            llama_d2048 = decode(2048)
            llama_block = {
                "measured": True,
                "metric": {
                    "p4096": "prefill",
                    "d128": "decode_only",
                    "d2048": "decode_only",
                },
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
                "llama_revision": LLAMA_REV,
            }
        except AccountingError as exc:
            llama_block = {"measured": False, "error": str(exc)}
    aa_status = {}
    for name, block in aa.items():
        decode_status = (block.get("decode_only") or {}).get("status")
        request_status = (block.get("request") or {}).get("status")
        aa_status[name] = {
            "decode_only": decode_status,
            "request": request_status,
        }
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "matched-baseline",
        "mode": mode,
        "ok": True,
        "hardware_executed": hardware,
        "reason": reason,
        "identity_native": identity_native,
        "aa": aa,
        "aa_status": aa_status,
        "aa_combined_decode_only": combine_aa_verdicts(
            {name: (block.get("decode_only") or {}) for name, block in aa.items()}
        )
        if aa
        else "unmeasured",
        "combined": combined,
        "post113": post113,
        "llama": llama_block,
        "fixed_token_not_free_running": True,
        "family_plan": family_plan(mode, "matched-baseline"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "matched-baseline.json", payload)
    persist(merge_phase(current_fixture(), "matched-baseline", payload))
    return payload


def run_activity_trace(
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
    windows: dict[str, Any] = {}
    hardware = False
    nsys = None
    ncu = None
    if execute is None:
        reason = blocker or "gpu_unavailable"
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            windows[name] = {
                "valid": False,
                "reason": reason,
                "prefix": prefix,
                "profiler_absence_is_limitation_not_zero_activity": True,
            }
    else:
        reason = None
        for name, prefix in (("d128", 128), ("d2048", 2048)):
            completed = execute(
                [
                    f"./{NATIVE}",
                    "--workload",
                    "activity-windows",
                    "--prefix",
                    str(prefix),
                    "--warmups",
                    "0",
                    "--samples",
                    "1",
                    "--tokens",
                    str(DECODE_TOKENS),
                    MODEL,
                ],
                "screen",
            )
            raw = completed.stdout + completed.stderr
            (run_dir / f"activity-{name}.txt").write_text(raw, encoding="utf-8")
            if "nsys_available=true" in raw:
                nsys = True
            elif "nsys_available=false" in raw:
                nsys = False
            if "ncu_available=true" in raw:
                ncu = True
            elif "ncu_available=false" in raw:
                ncu = False
            summary = parse_prefixed(completed.stdout, RESULT_PREFIX)
            records = parse_opt125_records(completed.stdout)
            dump_json(run_dir / f"activity-{name}-records.json", records)
            span = recorded_span_ms(records)
            split = classify_disjoint_intervals(
                records,
                wall_ms=span,
                tokens=12,
                profiler="cuda_event_engine_attribution",
            )
            split["instrumented_windows_only"] = True
            split["full_decode_wall_ms"] = summary.get(
                "instrumented_decode_only_wall_ms"
            )
            split["note"] = (
                "Events cover early/middle/late windows only (12 tokens). "
                "Do not divide 256-token wall by 12-token event union."
            )
            buckets = split_windows(records, DECODE_TOKENS)
            window_stats = {
                label: classify_disjoint_intervals(
                    rows,
                    wall_ms=recorded_span_ms(rows),
                    tokens=max(
                        1, len({int(r.get("token_position", -1) or -1) for r in rows})
                    ),
                    profiler="cuda_event_engine_attribution",
                )
                for label, rows in buckets.items()
                if label != "other"
            }
            windows[name] = {
                "valid": bool(records) and not bool(summary.get("pool_overflow")),
                "prefix": prefix,
                "native": summary,
                "intervals": split,
                "windows": window_stats,
                "record_count": len(records),
                "uninstrumented_decode_only_wall_ms": summary.get(
                    "uninstrumented_decode_only_wall_ms"
                ),
                "instrumented_decode_only_wall_ms": summary.get(
                    "instrumented_decode_only_wall_ms"
                ),
                "profiler_perturbation_ms": summary.get("profiler_perturbation_ms"),
                "method": "cuda_event_engine_attribution",
                "profiler_absence_is_limitation_not_zero_activity": nsys is not True,
            }
            hardware = True
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "activity-trace",
        "mode": mode,
        "ok": True,
        "hardware_executed": hardware,
        "reason": reason,
        "nsys_available": nsys,
        "ncu_available": ncu,
        "cupti_linked": False,
        "d128": windows.get("d128"),
        "d2048": windows.get("d2048"),
        "family_plan": family_plan(mode, "activity-trace"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "activity-trace.json", payload)
    persist(merge_phase(current_fixture(), "activity-trace", payload))
    return payload


def recompute_activity_intervals(run_dir: Path, name: str) -> dict[str, Any] | None:
    rec_path = run_dir / f"activity-{name}-records.json"
    if not rec_path.is_file():
        return None
    records = load_json(rec_path)
    if not isinstance(records, list):
        return None
    buckets = split_windows(records, DECODE_TOKENS)
    window_stats = {}
    window_wall = 0.0
    for label, rows in buckets.items():
        if label == "other" or not rows:
            continue
        span = recorded_span_ms(rows)
        window_stats[label] = classify_disjoint_intervals(
            rows,
            wall_ms=span,
            tokens=max(
                1,
                len({int(row.get("token_position", -1) or -1) for row in rows}),
            ),
            profiler="cuda_event_engine_attribution",
        )
        if span:
            window_wall += float(span)
    split = classify_disjoint_intervals(
        records,
        wall_ms=window_wall if window_wall > 0 else recorded_span_ms(records),
        tokens=12,
        profiler="cuda_event_engine_attribution",
    )
    split["instrumented_windows_only"] = True
    split["source"] = "opt125_activity_windows_records"
    split["windows"] = window_stats
    split["window_wall_ms"] = window_wall
    return split


def run_intervals(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    opt124 = load_json(OPT124_FIXTURE) if OPT124_FIXTURE.is_file() else {}
    timeline = opt124.get("timeline") or {}
    intervals: dict[str, Any] = {}
    for name in ("d128", "d2048"):
        recomputed = recompute_activity_intervals(run_dir, name)
        if recomputed is not None:
            intervals[name] = recomputed
            continue
        hist = timeline.get(name) or {}
        intervals[name] = classify_disjoint_intervals(
            [],
            wall_ms=hist.get("wall_ms"),
            tokens=1,
            profiler="cuda_event_engine_attribution",
        )
        intervals[name]["source"] = "opt124_timeline_overlay_without_raw_records"
        intervals[name]["historical_gpu_busy_union_ms"] = hist.get("gpu_busy_union_ms")
        intervals[name]["historical_gpu_idle_gap_ms"] = hist.get("gpu_idle_gap_ms")
        intervals[name]["leaf_gap_ms_opt115_label"] = hist.get("gpu_idle_gap_ms")
        if hist.get("gpu_busy_union_ms") is not None and hist.get("wall_ms"):
            intervals[name]["device_active_ms"] = hist.get("gpu_busy_union_ms")
            intervals[name]["unobserved_ms"] = max(
                0.0,
                float(hist.get("wall_ms") or 0.0)
                - float(hist.get("gpu_busy_union_ms") or 0.0),
            )
            intervals[name]["device_inactive_proven_ms"] = 0.0
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "intervals",
        "mode": mode,
        "ok": True,
        "intervals": intervals,
        "family_plan": family_plan(mode, "intervals"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "intervals.json", payload)
    persist(merge_phase(fixture, "intervals", payload))
    return payload


def run_weight_reads(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = weight_reads_and_reconcile()
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "weight-reads",
        "mode": mode,
        "ok": True,
        "weight_reads": weights,
        "family_plan": family_plan(mode, "weight-reads"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "weight-reads.json", payload)
    persist(merge_phase(current_fixture(), "weight-reads", payload))
    return payload


def run_long_context(
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
    probes: dict[str, Any] = {}
    two_k: dict[str, Any] = {"measured": False}
    hardware = False
    if execute is None:
        reason = blocker or "gpu_unavailable"
        probes = dict((historical_corrections().get("long_context") or {}))
        two_k = dict(historical_corrections().get("opt016_2k") or {})
        two_k["measured"] = False
        two_k["reason"] = reason
    else:
        reason = None
        for name, prefix in (("d8192", 8192), ("d32768", 32768), ("d131040", 131040)):
            try:
                completed = execute(
                    [
                        f"./{NATIVE}",
                        "--workload",
                        "long-context",
                        "--prefix",
                        str(prefix),
                        "--capacity",
                        "131072",
                        "--warmups",
                        "0" if name == "d131040" else "1",
                        "--samples",
                        "1",
                        "--tokens",
                        str(LONG_DECODE_TOKENS),
                        MODEL,
                    ],
                    "screen",
                )
                raw = completed.stdout + completed.stderr
                (run_dir / f"long-{name}.txt").write_text(raw, encoding="utf-8")
                summary = parse_prefixed(completed.stdout, RESULT_PREFIX)
                dump_json(run_dir / f"long-{name}.json", summary)
                probes[name] = {**summary, "ok": True, "oom": False}
                hardware = True
            except AccountingError as exc:
                classified = classify_cuda_failure(str(exc))
                (run_dir / f"long-{name}-error.txt").write_text(
                    str(exc), encoding="utf-8"
                )
                probes[name] = {
                    "ok": False,
                    "prefix": prefix,
                    **classified,
                    "historical_fallback": (
                        historical_corrections().get("long_context") or {}
                    ).get(name),
                }
        try:
            two_path = ROOT / "build/qw38-cuda-prefill-2k-parity-test"
            if two_path.is_file() or available:
                completed = subprocess.run(
                    llama_command(
                        [
                            "bash",
                            "-lc",
                            "cd /workspace && test -x build/qw38-cuda-prefill-2k-parity-test "
                            "&& ./build/qw38-cuda-prefill-2k-parity-test "
                            "models/Qwen3.8-27B-Q4_K_M.gguf || true",
                        ]
                    ),
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                raw = completed.stdout + completed.stderr
                (run_dir / "opt016-2k.txt").write_text(raw, encoding="utf-8")
                record = parse_prefixed(raw, P2K_PREFIX)
                if record:
                    two_k = {
                        "measured": True,
                        "ok": True,
                        "native": record,
                        "separately_labeled": True,
                    }
                elif "decode_segments8 capture requires" in raw:
                    two_k = {
                        "measured": False,
                        "ok": False,
                        "reason": "decode_segments8_requires_matching_session_capacity",
                        "block_reason": (
                            "prefill-2k-parity binary does not match "
                            "decode_segments8 session capacity requirements"
                        ),
                        "nonexclusive_oom": False,
                        "resource_blocked": True,
                        "separately_labeled": True,
                    }
                else:
                    llama_hist = historical_corrections().get("opt016_2k") or {}
                    two_k = {
                        "measured": False,
                        "reason": "opt016_binary_produced_no_prefixed_record",
                        "historical": llama_hist,
                        "separately_labeled": True,
                    }
        except Exception as exc:  # noqa: BLE001 — record allocation/run failure
            two_k = {
                "measured": False,
                "reason": str(exc),
                **classify_cuda_failure(str(exc)),
                "historical": historical_corrections().get("opt016_2k"),
            }
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "long-context",
        "mode": mode,
        "ok": True,
        "hardware_executed": hardware,
        "reason": reason,
        "capacity": 131072,
        "decode_tokens": LONG_DECODE_TOKENS,
        "probes": probes,
        "opt016_2k": two_k,
        "exclusivity": "not_authorized_to_stop_unrelated_services",
        "family_plan": family_plan(mode, "long-context"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "long-context.json", payload)
    persist(merge_phase(current_fixture(), "long-context", payload))
    return payload


def run_ranking(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    intervals = fixture.get("disjoint_intervals") or {}
    corrections = fixture.get("historical_corrections") or historical_corrections()
    ranking = rank_opt126_131(intervals=intervals, corrections=corrections)
    protocol = shared_keep_protocol()
    ans = answers(intervals=intervals, corrections=corrections, ranking=ranking)
    baseline = fixture.get("matched_baseline") or {}
    llama = baseline.get("llama") or {}
    combined = baseline.get("combined") or {}
    if llama.get("measured") and combined:
        fresh_d128 = mean_arm_metric(combined.get("d128") or {}, "decode_only_tok_s")
        fresh_d2048 = mean_arm_metric(combined.get("d2048") or {}, "decode_only_tok_s")
        llama_d128 = llama.get("d128_tok_s")
        llama_d2048 = llama.get("d2048_tok_s")
        if fresh_d128 and llama_d128:
            ans["fresh_decode_only_combined_d128"] = fresh_d128
            ans["fresh_llama_decode_only_d128"] = llama_d128
            ans["fresh_matched_decode_only_d128"] = float(fresh_d128) / float(
                llama_d128
            )
        if fresh_d2048 and llama_d2048:
            ans["fresh_decode_only_combined_d2048"] = fresh_d2048
            ans["fresh_llama_decode_only_d2048"] = llama_d2048
            ans["fresh_matched_decode_only_d2048"] = float(fresh_d2048) / float(
                llama_d2048
            )
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
        "phase": "ranking",
        "mode": mode,
        "ok": True,
        "ranking": ranking,
        "shared_keep_protocol": protocol,
        "answers": ans,
        "family_plan": family_plan(mode, "ranking"),
        "measured_at": utc_now(),
    }
    dump_json(run_dir / "ranking.json", payload)
    persist(merge_phase(fixture, "ranking", payload))
    return payload


def run_report(mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture = current_fixture()
    if "metric_catalog" not in fixture:
        fixture = empty_payload(
            reason=str(fixture.get("gpu_blocker") or "scaffold"),
            identity=sitting_identity(),
        )
    fixture["claims_throughput"] = False
    fixture["claims_performance_improvement"] = False
    fixture["production_kept"] = True
    if fixture.get("hardware_executed"):
        fixture["gpu_blocker"] = None
    fixture["report_path"] = "evidence/optimization/opt125-decode-accounting/REPORT.md"
    fixture["proof_limit"] = PROOF
    fixture["measurement_utc"] = utc_now()
    validate_fixture(fixture)
    persist(fixture)
    payload = {
        "schema_version": 1,
        "task": "OPT-125",
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
        default=str(ROOT / "build/optimization-runs/opt125"),
    )
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    skip = bool(args.skip_gpu)
    if args.phase == "metrics":
        run_metrics(args.mode, run_dir)
    elif args.phase == "freeze":
        run_freeze(args.mode, run_dir)
    elif args.phase == "matched-baseline":
        run_matched_baseline(args.mode, run_dir, None, skip_gpu=skip)
    elif args.phase == "activity-trace":
        run_activity_trace(args.mode, run_dir, None, skip_gpu=skip)
    elif args.phase == "intervals":
        run_intervals(args.mode, run_dir)
    elif args.phase == "weight-reads":
        run_weight_reads(args.mode, run_dir)
    elif args.phase == "long-context":
        run_long_context(args.mode, run_dir, None, skip_gpu=skip)
    elif args.phase == "ranking":
        run_ranking(args.mode, run_dir)
    elif args.phase == "report":
        run_report(args.mode, run_dir)
    print(json.dumps({"task": "OPT-125", "phase": args.phase, "ok": True}))
    return 0


if __name__ == "__main__":
    try:
        validate_future_keep_policy("OPT-125", load_iteration())
        raise SystemExit(main())
    except AccountingError as exc:
        print(json.dumps({"task": "OPT-125", "ok": False, "error": str(exc)}))
        raise SystemExit(1) from exc
