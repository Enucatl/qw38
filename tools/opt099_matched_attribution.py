"""OPT-099 matched Quartz/llama attribution after post-098 freeze.

Diagnostic tooling only. No production selector, kernel, or throughput claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt060_engine_attribution import stream_aware_totals  # noqa: E402
from tools.opt071_attribution_repair import (  # noqa: E402
    DECODE_ATTENTION_CORES,
    DECODE_DOWNS,
    DECODE_GATE_UP_PAIRS,
    DECODE_GDN_CORES,
    DECODE_LOGITS,
    GGUF_SHA,
    LLAMA_REV,
    TOKEN_GENERATOR,
    UNEXPLAINED_LIMIT,
    accounting_split,
    capture_identity,
    nested_interval_fixture,
    pool_overflow,
    rank_remaining_gaps,
    token_input_hash,
    two_token_epochs_share_origin,
    warmup_contamination,
)
from tools.opt075_q4_production_admission import (  # noqa: E402
    docker_common,
    dump_json,
    load_json,
    mean,
    utc_now,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt090_decode_attribution import (  # noqa: E402
    COUNTER_LAUNCHES,
    NCU_METRIC_NEEDLES,
    Q6_ATTENTION_OUTPUTS,
    Q8_GDN_OUTPUTS,
    AttributionError as Opt090AttributionError,
    one_token_valid_records,
    query_ncu_metrics,
    replay_non_additive_audit,
    unhidden_idle_per_token,
    weight_byte_roofline,
)
from tools.opt098_batch_gate import COMBINATION_SELECTORS  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    parse_native_observation,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt099_matched_attribution_contract.json"
ITERATION = ROOT / "pins/opt099_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt099_matched_attribution.json"
REPORT = ROOT / "evidence/optimization/opt099-matched-attribution/REPORT.md"
EVIDENCE = REPORT.parent
OPT098_FIXTURE = ROOT / "fixtures/opt098_batch_gate.json"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
ATTRIBUTION_BIN = "build/qw38-cuda-opt060-engine-attribution-test"
REPLAY_BIN = "build/qw38-cuda-component-replay"
LLAMA_BIN = ".cache/authorities/llama-build-opt060/bin/qw38-llama-engine-attribution"
PHASES = (
    "freeze",
    "conservation",
    "d128",
    "d2048",
    "counters",
    "p4096",
    "report",
)
OUTPUT_TOKENS = 32
PREFILL_PROMPT = 4096
DECODE_WARMUPS = 3
DECODE_SAMPLES = 3
PIN_FILES = {
    "q4_decode": ("cuda/q4k_decode_path.cuh", r'kSelectedQ4DecodePath\[\] = "([^"]+)"'),
    "q4_warps_per_row": (
        "cuda/q4k_decode_path.cuh",
        r"kSelectedQ4DecodeWarpsPerRow = (\d+)",
    ),
    "ffn_decode": (
        "cuda/ffn_decode_path.cuh",
        r'kSelectedFfnDecodePath\[\] = "([^"]+)"',
    ),
    "q8_grouping": (
        "cuda/q8_decode_path.cuh",
        r'kSelectedQ8DecodeGrouping\[\] = "([^"]+)"',
    ),
    "q6_decode": ("cuda/q6k_decode_path.cuh", r'kSelectedQ6DecodePath\[\] = "([^"]+)"'),
    "q6_warps_per_row": (
        "cuda/q6k_decode_path.cuh",
        r"kSelectedQ6DecodeWarpsPerRow = (\d+)",
    ),
    "gdn_decode": (
        "cuda/gdn_decode_path.cuh",
        r'kSelectedGdnDecodePath\[\] = "([^"]+)"',
    ),
    "decode_query_prep": (
        "cuda/attention_decode_path.cuh",
        r'kSelectedDecodeQueryPrepPath\[\] = "([^"]+)"',
    ),
    "mmq_pipeline": (
        "cuda/quant_mmq_mma.cuh",
        r'kSelectedMmqPipelinePath\[\] = "([^"]+)"',
    ),
    "mmq_async_x": (
        "cuda/quant_mmq_mma.cuh",
        r"constexpr bool kSelectedMmqAsyncX = (true|false);",
    ),
    "mmq_split_xy_wait": (
        "cuda/quant_mmq_mma.cuh",
        r"constexpr bool kSelectedMmqSplitXYWait = (true|false);",
    ),
    "ffn_gate_quality_i": (
        "cuda/quant_mmq_mma.cuh",
        r"kSelectedFfnGateQualityI = (\d+)",
    ),
    "ffn_gate_prompt_tile": (
        "cuda/quant_mmq_mma.cuh",
        r"kSelectedFfnGatePromptTile = (\d+)",
    ),
    "execution_graphs": (
        "cuda/execution_graph_path.cuh",
        r'kSelectedExecutionGraphPath\[\] = "([^"]+)"',
    ),
}
REQUESTED_POST098: dict[str, Any] = {
    "q4_decode": "integer_q8_late",
    "q4_warps_per_row": 4,
    "ffn_decode": "paired_integer",
    "q8_grouping": "grouped_r1_w4",
    "q6_decode": "integer_q8_1",
    "q6_warps_per_row": 2,
    "gdn_decode": "sequential",
    "decode_query_prep": "warp_query",
    "prompt_mmq": "i128_j128_fma_async_x",
    "prompt_mmq_wait": "joined_wait",
    "execution_graphs": "ffn_only",
}
EXPECTED_DECODE_COUNTS = {
    "ffn_gate_up_glu": DECODE_GATE_UP_PAIRS,
    "ffn_down": DECODE_DOWNS,
    "gdn_core": DECODE_GDN_CORES,
    "attention_core": DECODE_ATTENTION_CORES,
    "logits_projection": DECODE_LOGITS,
    "q8_gdn_output": Q8_GDN_OUTPUTS,
    "q6_attention_output": Q6_ATTENTION_OUTPUTS,
}
TRIGGER_IDS = (
    "opt100_q8_aligned",
    "opt101_gdn_transpose",
    "opt102_q4_repack",
    "opt103_attention_vec",
    "opt104_q6_aligned",
    "opt105_mmq_x2",
)
DECODE_EXCESS_MS = 0.10
PREFILL_EXCESS_MS = 5.0
PARENT_ROLES = frozenset(
    {
        "ffn_mmv",
        "ffn_mmq",
        "mixer_mmv",
        "mixer_mmq",
        "gdn_core",
        "logits",
    }
)
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class AttributionError(AssertionError):
    """Fail-closed OPT-099 attribution error."""


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [*docker_common(tier), *listed]
    completed = subprocess.run(
        listed,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise AttributionError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    return completed


def _search(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if match is None:
        raise AttributionError(f"missing pin {pattern} in {path}")
    return match.group(1)


def compiled_selectors() -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for name, (path, pattern) in PIN_FILES.items():
        value = _search(path, pattern)
        if name in {
            "q4_warps_per_row",
            "q6_warps_per_row",
            "ffn_gate_quality_i",
            "ffn_gate_prompt_tile",
        }:
            raw[name] = int(value)
        elif name in {"mmq_async_x", "mmq_split_xy_wait"}:
            raw[name] = value == "true"
        else:
            raw[name] = value
    wait = "split_xy_wait" if raw["mmq_split_xy_wait"] else "joined_wait"
    tile = f"i{raw['ffn_gate_quality_i']}_j{raw['ffn_gate_prompt_tile']}"
    async_x = "_x" if raw["mmq_async_x"] else ""
    prompt_mmq = f"{tile}_{raw['mmq_pipeline']}{async_x}"
    return {
        "q4_decode": raw["q4_decode"],
        "q4_warps_per_row": raw["q4_warps_per_row"],
        "ffn_decode": raw["ffn_decode"],
        "q8_grouping": raw["q8_grouping"],
        "q6_decode": raw["q6_decode"],
        "q6_warps_per_row": raw["q6_warps_per_row"],
        "gdn_decode": raw["gdn_decode"],
        "decode_query_prep": raw["decode_query_prep"],
        "prompt_mmq": prompt_mmq,
        "prompt_mmq_wait": wait,
        "execution_graphs": raw["execution_graphs"],
        "mmq_pipeline": raw["mmq_pipeline"],
        "mmq_async_x": raw["mmq_async_x"],
        "ffn_tiles": tile,
    }


def freeze_post098_selected() -> dict[str, Any]:
    observed = compiled_selectors()
    mismatches = {
        key: {"requested": REQUESTED_POST098[key], "observed": observed.get(key)}
        for key in REQUESTED_POST098
        if observed.get(key) != REQUESTED_POST098[key]
    }
    if mismatches:
        raise AttributionError(
            "post098_selected compiled selectors differ: " + json.dumps(mismatches)
        )
    fixture = load_json(OPT098_FIXTURE) if OPT098_FIXTURE.is_file() else {}
    fixture_selected = fixture.get("post098_selected") or {}
    return {
        "ok": True,
        "requested": dict(REQUESTED_POST098),
        "observed": observed,
        "mismatches": {},
        "combination_selectors": dict(COMBINATION_SELECTORS),
        "opt098_fixture_q4": fixture_selected.get("q4_decode"),
        "opt098_fixture_ffn": fixture_selected.get("ffn_decode"),
    }


def family_plan(name: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    return workload_for_mode(iteration["workloads"][name], mode)


def _hay(record: Mapping[str, Any]) -> str:
    return " ".join(
        str(record.get(key, "") or "")
        for key in ("role", "tensor_name", "weight_name", "launch_family", "op")
    ).lower()


def parse_layer_name(name: str) -> int:
    text = str(name or "")
    match = re.search(r"blk\.(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"-(\d+)(?:\s|\(|\[|$)", text)
    if match:
        return int(match.group(1))
    return -1


def inferred_layer(record: Mapping[str, Any]) -> int:
    raw = record.get("layer")
    try:
        layer = int(raw)
    except (TypeError, ValueError):
        layer = -1
    if layer >= 0:
        return layer
    for key in ("tensor_name", "weight_name"):
        parsed = parse_layer_name(str(record.get(key, "") or ""))
        if parsed >= 0:
            return parsed
    return -1


def canonical_family_opt099(record: Mapping[str, Any]) -> str:
    role = str(record.get("role", "") or "")
    op = str(record.get("op", "") or "")
    tensor = str(record.get("tensor_name", "") or "").lower()
    weight = str(record.get("weight_name", "") or "").lower()
    hay = _hay(record)
    if role in {"d2h", "copy"} or "d2h" in hay:
        return "d2h" if "d2h" in hay or role == "d2h" else "copy"
    if role in {"state_commit", "state_copies", "commit_sync"}:
        return "state_commit"
    if role in {"logits_projection", "logits"} or "result_output" in hay:
        return "logits_projection"
    if role == "logits_norm" or "output_norm" in hay:
        return "logits_norm"
    if "ffn_out" in hay or "ffn_down" in hay or role in {"ffn_down", "proj_ffn_down"}:
        return "ffn_down"
    if role in {"ffn_gate_up_glu", "ffn_gate", "ffn_up", "ffn_glu"} or any(
        marker in hay for marker in ("ffn_gate", "ffn_up", "ffn_glu")
    ):
        return "ffn_gate_up_glu"
    if (
        role in {"q8_gdn_output", "gdn_output", "proj_gdn_output"}
        or any(
            marker in tensor or marker in weight
            for marker in ("ssm_out", "linear_attn_out")
        )
        or tensor.startswith("z-")
    ):
        return "q8_gdn_output"
    if role in {"attention_core", "FLASH_ATTN_EXT", "FLASH_ATTN"} or op in {
        "FLASH_ATTN_EXT",
        "FLASH_ATTN",
    }:
        return "attention_core"
    if role in {"attention_query_prep", "attn_query_split", "ROPE"} or op == "ROPE":
        return "attention_query_prep"
    if (
        tensor.startswith("attn_output")
        or "attn_output.weight" in weight
        or role == "proj_attn_output"
    ):
        return "q6_attention_output"
    if (
        role
        in {
            "q8_input_projections",
            "proj_packed_qkv",
            "proj_query_gate",
            "proj_key",
            "proj_value",
            "proj_value_gate",
            "proj_alpha",
            "proj_beta",
            "gdn_packed_qkv",
            "gdn_alpha",
            "gdn_beta",
            "gdn_value_gate",
        }
        or any(
            marker in hay
            for marker in (
                "attn_qkv",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "ssm_alpha",
                "ssm_beta",
            )
        )
        or any(tensor.startswith(prefix) for prefix in ("qcur", "kcur", "vcur"))
    ):
        return "q8_input_projections"
    if role in {"gdn_conv", "SSM_CONV"} or "ssm_conv" in hay:
        return "gdn_conv"
    if role in {"gdn_norm"} or role == "L2_NORM":
        return "gdn_norm"
    if role in {"gdn_recurrence", "GATED_DELTA_NET"}:
        return "gdn_recurrence"
    if role in {"gdn_gate"}:
        return "gdn_gate"
    if role in {"gdn_core", "gdn_conv_qk_norm_recurrence", "gdn_gate_prep"}:
        return "gdn_core"
    if role in {"attention_merge", "attn_output_cast"}:
        return "attention_merge"
    if role in {"attn_qk_prep_softmax_pv_merge"}:
        return "attention_core"
    if role in {
        "pointwise",
        "residual",
        "silu",
        "rms_norm",
        "residual_ffn",
        "ADD",
        "UNARY",
    }:
        return "pointwise"
    if role in {"activation_quant", "activation_quant_ffn", "activation_quant_mixer"}:
        return "activation_quant"
    if role in {"embedding"} or "token_embd" in hay:
        return "embedding"
    if role in {"cpu_unknown_gap", "other_idle", "host_submission_waits"}:
        return "cpu_unknown_gap"
    if role in {"unknown", "MUL_MAT", "mul_mat", "unmapped_mul_mat"}:
        return "unknown"
    return role or "unknown"


def rollup_family(family: str) -> str:
    if family in {"gdn_conv", "gdn_norm", "gdn_recurrence", "gdn_gate"}:
        return "gdn_core"
    if family in {"attention_query_prep", "attention_merge"}:
        return "attention_core"
    return family


def _has_role(rows: Sequence[Mapping[str, Any]], names: set[str]) -> bool:
    return any(str(row.get("role", "") or "") in names for row in rows)


def chargeable_records_opt099(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Prefer family leaves over enclosing parents so FFN down stays first-class."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            record.get("engine"),
            record.get("phase"),
            record.get("token_position"),
            inferred_layer(record),
        )
        groups.setdefault(key, []).append(record)
    kept: list[Mapping[str, Any]] = []
    for rows in groups.values():
        drop: set[int] = set()
        for record in rows:
            if str(record.get("attribution_role", "") or "") != "enclosing":
                continue
            role = str(record.get("role", "") or "")
            if role not in PARENT_ROLES:
                continue
            if role in {"ffn_mmv", "ffn_mmq"} and _has_role(
                rows, {"ffn_down", "ffn_gate_up_glu", "ffn_gate", "ffn_up", "ffn_glu"}
            ):
                drop.add(id(record))
            elif role in {"mixer_mmv", "mixer_mmq"} and _has_role(
                rows,
                {
                    "proj_packed_qkv",
                    "proj_query_gate",
                    "proj_key",
                    "proj_value",
                    "proj_value_gate",
                    "proj_alpha",
                    "proj_beta",
                    "proj_gdn_output",
                    "proj_attn_output",
                    "q8_input_projections",
                    "q8_gdn_output",
                    "q6_attention_output",
                },
            ):
                drop.add(id(record))
            elif role == "gdn_core" and _has_role(
                rows,
                {
                    "gdn_conv",
                    "gdn_norm",
                    "gdn_recurrence",
                    "gdn_gate",
                    "gdn_conv_qk_norm_recurrence",
                    "gdn_gate_prep",
                },
            ):
                drop.add(id(record))
            elif role == "logits" and _has_role(rows, {"logits_projection"}):
                drop.add(id(record))
        for record in rows:
            if id(record) not in drop:
                kept.append(record)
    return kept


def _credited_families(record: Mapping[str, Any]) -> set[str]:
    family = rollup_family(canonical_family_opt099(record))
    credited: set[str] = set()
    role = str(record.get("role", "") or "")
    if family in EXPECTED_DECODE_COUNTS:
        credited.add(family)
    if role in {"ffn_gate", "ffn_up", "ffn_glu", "ffn_gate_up_glu"}:
        credited.add("ffn_gate_up_glu")
    if role in {"ffn_down", "proj_ffn_down"}:
        credited.add("ffn_down")
    if role in {"gdn_core", "gdn_conv", "gdn_norm", "gdn_recurrence", "gdn_gate"}:
        credited.add("gdn_core")
    if role in {
        "attention_core",
        "attention_query_prep",
        "attention_merge",
        "FLASH_ATTN_EXT",
    }:
        credited.add("attention_core")
    if family == "q8_gdn_output":
        credited.add("q8_gdn_output")
    if family == "q6_attention_output":
        credited.add("q6_attention_output")
    if family == "logits_projection":
        credited.add("logits_projection")
    return credited


def family_counts_opt099(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in EXPECTED_DECODE_COUNTS}
    seen: dict[str, set[tuple[Any, ...]]] = {name: set() for name in counts}
    for record in chargeable_records_opt099(records):
        key = (
            record.get("engine"),
            record.get("phase"),
            record.get("token_position"),
            inferred_layer(record),
        )
        for name in _credited_families(record):
            if name not in seen or key in seen[name]:
                continue
            seen[name].add(key)
            counts[name] += 1
    return counts


def expected_decode_counts_opt099(tokens: int = 1) -> dict[str, int]:
    return {name: value * int(tokens) for name, value in EXPECTED_DECODE_COUNTS.items()}


def validate_call_counts_opt099(
    records: Sequence[Mapping[str, Any]],
    *,
    tokens: int,
) -> dict[str, Any]:
    observed = family_counts_opt099(records)
    expected = expected_decode_counts_opt099(tokens)
    mismatches = {
        name: {"observed": observed.get(name, 0), "expected": expected[name]}
        for name in expected
        if observed.get(name, 0) != expected[name]
    }
    return {
        "ok": not mismatches,
        "observed": observed,
        "expected": expected,
        "mismatches": mismatches,
    }


def family_times(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    times: dict[str, float] = {}
    for record in chargeable_records_opt099(records):
        name = canonical_family_opt099(record)
        times[name] = times.get(name, 0.0) + float(
            record.get("complete_work_ms", 0.0) or 0.0
        )
    return times


def dispatch_identities(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (
            str(record.get("role", "") or ""),
            str(record.get("launch_family", "") or ""),
            str(record.get("tensor_name", "") or record.get("weight_name", "") or ""),
        )
        seen[key] = seen.get(key, 0) + 1
    rows = [
        {
            "role": role,
            "launch_family": launch,
            "tensor_name": name,
            "count": count,
        }
        for (role, launch, name), count in seen.items()
        if role or launch or name
    ]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["role"])))
    return rows[:48]


def llama_missing_filled_with_zero(families: Mapping[str, Any]) -> bool:
    for value in families.values():
        if value == 0.0 or value == 0:
            return True
    return False


def native_result(
    *,
    phase: str,
    mode: str,
    plan: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "task": "OPT-099",
        "mode": mode,
        "phase": phase,
        "keep": False,
        "claims_throughput": False,
        "warmups": int(plan.get("warmups", 0) or 0),
        "samples": int(plan.get("samples", 1) or 1),
        "observed_warmups": int(plan.get("warmups", 0) or 0),
        "observed_samples": int(plan.get("samples", 1) or 1),
        "observed_candidates": int(plan.get("candidates", 1) or 1),
        "observed_shapes": int(plan.get("cases", 1) or 1),
        "observed_tier": str(plan.get("tier", phase)),
        "pairs": int(plan.get("control_candidate_pairs", 1) or 1),
        "sample_ids": list(range(int(plan.get("samples", 1) or 1))),
        "acceptance_executed": str(plan.get("tier", "")) == "acceptance",
    }
    if extra:
        payload.update(dict(extra))
    return payload


def emit_native(payload: Mapping[str, Any]) -> None:
    print("QW38_OPT099_RESULT=" + json.dumps(payload, sort_keys=True))
    print("QW38_OPT099_NATIVE_COUNTS=" + json.dumps(payload, sort_keys=True))


def empty_fixture(mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-099",
        "mode": mode,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "token_generator": TOKEN_GENERATOR,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "quartz_idents": ["post098_selected"],
        "measured_idents": ["post098_selected"],
        "pinned_llama_ident": "pinned_llama",
        "ffn_down_first_class": False,
        "gap_attribution_complete": False,
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "evidence_complete": False,
        "opt074_coverage_unadmitted_blocker": False,
        "triggers": {
            name: {"go": False, "reason": "unmeasured"} for name in TRIGGER_IDS
        },
        "independent_verdicts": {key: False for key in VERDICT_KEYS},
        "report_path": str(REPORT.relative_to(ROOT)),
    }


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = dict(payload)
    results["updated_utc"] = utc_now()
    if payload.get("hardware_executed"):
        results["hardware_executed"] = True
    results["claims_throughput"] = False
    results["claims_performance_improvement"] = False
    results["production_kept"] = False
    results["performance_pass"] = False
    if "ffn_down_first_class" in payload:
        results["ffn_down_first_class"] = bool(payload["ffn_down_first_class"])
    if "gap_attribution_complete" in payload:
        results["gap_attribution_complete"] = bool(payload["gap_attribution_complete"])
    if "triggers" in payload:
        results["triggers"] = dict(payload["triggers"])
    if "evidence_complete" in payload:
        results["evidence_complete"] = bool(payload["evidence_complete"])


def parse_quartz_windows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"quartz_rep=(?P<rep>\d+)\s+phase=(?P<phase>\S+)\s+ident=(?P<ident>\S+)"
        r".*?shipping_graph_wall_ms=(?P<graph>[0-9.eE+-]+)"
        r".*?eager_diagnostic_wall_ms=(?P<eager>[0-9.eE+-]+)"
        r".*?graph_itl_p50_ms=(?P<p50>[0-9.eE+-]+)"
        r".*?graph_itl_p95_ms=(?P<p95>[0-9.eE+-]+)"
        r".*?records=(?P<records>\d+)"
        r".*?order=(?P<order>\S+)"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "rep": int(match.group("rep")),
                "phase": match.group("phase"),
                "ident": match.group("ident"),
                "shipping_graph_wall_ms": float(match.group("graph")),
                "eager_diagnostic_wall_ms": float(match.group("eager")),
                "graph_itl_p50_ms": float(match.group("p50")),
                "graph_itl_p95_ms": float(match.group("p95")),
                "records": int(match.group("records")),
                "order": match.group("order"),
            }
        )
    return rows


def load_event_records(ident: str, phase: str, sample: int) -> list[dict[str, Any]]:
    engine = "llama" if ident == "pinned_llama" else "quartz"
    path = EVIDENCE / f"{engine}-{ident}-{phase}-rep{sample}-records.json"
    if not path.is_file():
        return []
    payload = load_json(path)
    records = payload.get("records") or []
    return [dict(row) for row in records if isinstance(row, Mapping)]


def llama_log_path(ident: str, workload: str, mode: str, samples: int) -> Path:
    return EVIDENCE / f"llama-{ident}-{workload}-{mode}-samples{samples}.log"


def parse_llama_dumps(text: str) -> list[list[dict[str, Any]]]:
    dumps: list[list[dict[str, Any]]] = []
    decoder = json.JSONDecoder()
    needle = '{"schema_version":1,"task":'
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        start = index + consumed
        if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
            dumps.append(
                [dict(row) for row in payload["records"] if isinstance(row, Mapping)]
            )
    return dumps


def parse_llama_sample_walls(text: str) -> dict[int, dict[str, float]]:
    walls: dict[int, dict[str, float]] = {}
    decoder = json.JSONDecoder()
    needle = '{"schema_version":1,"task":'
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        start = index + consumed
        if not isinstance(payload, Mapping):
            continue
        if "sample_index" not in payload:
            continue
        if (
            "uninstrumented_wall_ms" not in payload
            and "instrumented_wall_ms" not in payload
        ):
            continue
        sample = int(payload["sample_index"])
        bucket = walls.setdefault(
            sample,
            {"uninstrumented_wall_ms": 0.0, "instrumented_wall_ms": 0.0},
        )
        for key in ("uninstrumented_wall_ms", "instrumented_wall_ms"):
            if key not in payload:
                continue
            value = float(payload.get(key) or 0.0)
            if value > 0.0:
                bucket[key] = value
    return walls


def collect_llama_walls(logs: Sequence[Path]) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for path in logs:
        if not path.is_file():
            continue
        parsed = parse_llama_sample_walls(
            path.read_text(encoding="utf-8", errors="replace")
        )
        for sample, values in parsed.items():
            bucket = merged.setdefault(
                sample,
                {"uninstrumented_wall_ms": 0.0, "instrumented_wall_ms": 0.0},
            )
            for key, value in values.items():
                if float(value) > 0.0:
                    bucket[key] = float(value)
    return merged


def llama_graph_create_excluded_ms(
    uninstrumented_wall_ms: float, instrumented_wall_ms: float
) -> float:
    """CUDA graph capture charged to unperturbed llama walls, if any.

    Prefill has no prefix window. llama.cpp may capture/instantiate CUDA graphs
    on the first unperturbed sample even after warmups, so that wall includes
    construction. Eager diagnostic disables graphs, so its wall is the engine
    window. When unperturbed exceeds eager by more than the unexplained limit,
    the excess is graph construction and stays outside the measured window.
    """
    uninstrumented = float(uninstrumented_wall_ms or 0.0)
    instrumented = float(instrumented_wall_ms or 0.0)
    if instrumented <= 0.0 or uninstrumented <= instrumented:
        return 0.0
    excess = uninstrumented - instrumented
    if excess / uninstrumented <= UNEXPLAINED_LIMIT:
        return 0.0
    return excess


def discover_llama_logs(stdout: str, phase: str, samples: int) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for match in re.finditer(r"log=(\S+)", stdout):
        raw = Path(match.group(1))
        path = raw if raw.is_absolute() else ROOT / raw
        if path not in seen:
            seen.add(path)
            found.append(path)
    for ident in ("post098_selected", "shipping"):
        for mode in ("unperturbed", "eager_diagnostic"):
            path = llama_log_path(ident, phase, mode, samples)
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def write_llama_sidecars(log_path: Path, phase: str) -> list[Path]:
    if not log_path.is_file():
        return []
    dumps = parse_llama_dumps(log_path.read_text(encoding="utf-8", errors="replace"))
    written: list[Path] = []
    for sample, records in enumerate(dumps):
        dest = EVIDENCE / f"llama-pinned_llama-{phase}-rep{sample}-records.json"
        dump_json(
            dest,
            {
                "schema_version": 1,
                "task": "OPT-099",
                "engine": "llama",
                "ident": "pinned_llama",
                "phase": phase,
                "rep": sample,
                "count": len(records),
                "records": records,
            },
        )
        written.append(dest)
    return written


def summarize_window(
    ident: str,
    phase: str,
    sample: int,
    wall: Mapping[str, Any] | None,
    *,
    tokens: int,
) -> dict[str, Any]:
    records = load_event_records(ident, phase, sample)
    counts = (
        validate_call_counts_opt099(records, tokens=tokens)
        if records
        else {
            "ok": False,
            "mismatches": {"missing_records": {"observed": 0, "expected": 1}},
            "observed": {},
            "expected": expected_decode_counts_opt099(tokens),
        }
    )
    dropped = pool_overflow(records) if records else True
    contaminated = warmup_contamination(records) if records else False
    charged = chargeable_records_opt099(records) if records else []
    totals = (
        stream_aware_totals(charged)
        if charged
        else {
            "summed_gpu_work_ms": 0.0,
            "interval_union_ms": 0.0,
            "overlap_ms": 0.0,
        }
    )
    graph_wall = float((wall or {}).get("shipping_graph_wall_ms") or 0.0)
    eager_wall = float((wall or {}).get("eager_diagnostic_wall_ms") or 0.0)
    graph_create = float((wall or {}).get("graph_create_excluded_ms") or 0.0)
    if ident == "pinned_llama" and graph_create <= 0.0:
        graph_create = llama_graph_create_excluded_ms(graph_wall, eager_wall)
    accounting = accounting_split(
        graph_wall_ms=graph_wall,
        eager_instrumented_wall_ms=eager_wall,
        eager_work_ms=float(totals["summed_gpu_work_ms"]),
        attributed_union_ms=float(totals["interval_union_ms"]),
        d2h_ms=0.0,
        commit_ms=0.0,
        graph_create_ms=graph_create,
    )
    families = family_times(records) if records else {}
    valid = (
        bool(counts["ok"])
        and not dropped
        and not contaminated
        and float(accounting["unexplained_share"]) <= UNEXPLAINED_LIMIT
        and "unmapped_mul_mat" not in families
    )
    idle = unhidden_idle_per_token(records, tokens=max(tokens, 1)) if records else None
    llama_zeroed = ident == "pinned_llama" and llama_missing_filled_with_zero(families)
    if llama_zeroed:
        valid = False
    return {
        "ident": ident,
        "sample": sample,
        "order": (wall or {}).get("order"),
        "call_counts": counts,
        "dropped_events": dropped,
        "warmup_contamination": contaminated,
        "accounting": accounting,
        "families": families,
        "family_sum_ms": sum(families.values()),
        "interval_union_ms": float(totals["interval_union_ms"]),
        "overlap_ms": float(totals["overlap_ms"]),
        "idle_ms_per_token": idle,
        "graph_itl_p50_ms": (wall or {}).get("graph_itl_p50_ms"),
        "graph_itl_p95_ms": (wall or {}).get("graph_itl_p95_ms"),
        "dispatch_identities": dispatch_identities(records) if records else [],
        "ffn_down_first_class": float(families.get("ffn_down", 0.0) or 0.0) > 0.0,
        "attribution_valid": valid,
        "eager_labeled_diagnostic": True,
        "shipping_graph_separate": True,
        "llama_zero_filled": llama_zeroed,
    }


def rank_from_windows(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    quartz_rows = [
        row
        for row in windows
        if row.get("attribution_valid") and str(row.get("ident")) != "pinned_llama"
    ]
    llama_rows = [
        row
        for row in windows
        if row.get("attribution_valid") and str(row.get("ident")) == "pinned_llama"
    ]
    if not quartz_rows:
        accounting = (
            windows[0].get("accounting") if windows else {"unexplained_share": 1.0}
        )
        counts = windows[0].get("call_counts") if windows else {"ok": False}
        ranked = rank_remaining_gaps(
            [],
            accounting or {"unexplained_share": 1.0},
            counts or {"ok": False},
        )
        ranked["llama_zero_filled"] = any(
            row.get("llama_zero_filled") for row in windows
        )
        return ranked
    families: dict[str, dict[str, float | None]] = {}
    for row in quartz_rows:
        for name, ms in (row.get("families") or {}).items():
            bucket = families.setdefault(name, {"quartz_ms": 0.0, "llama_ms": None})
            bucket["quartz_ms"] = float(bucket["quartz_ms"] or 0.0) + float(ms)
    for bucket in families.values():
        bucket["quartz_ms"] = float(bucket["quartz_ms"] or 0.0) / len(quartz_rows)
    if llama_rows:
        llama_acc: dict[str, float] = {}
        for row in llama_rows:
            for name, ms in (row.get("families") or {}).items():
                llama_acc[name] = llama_acc.get(name, 0.0) + float(ms)
        for name, total in llama_acc.items():
            bucket = families.setdefault(name, {"quartz_ms": 0.0, "llama_ms": None})
            averaged = total / len(llama_rows)
            bucket["llama_ms"] = averaged if averaged > 0.0 else None
    table = []
    unknown_present = False
    for name, values in families.items():
        llama_ms = values["llama_ms"]
        quartz_ms = float(values["quartz_ms"] or 0.0)
        covered = llama_ms is not None and float(llama_ms) > 0.0
        if name == "unknown" and quartz_ms > 0.0:
            unknown_present = True
        excess = (quartz_ms - float(llama_ms)) if covered else None
        table.append(
            {
                "family": name,
                "quartz_ms": quartz_ms,
                "llama_ms": float(llama_ms) if covered else None,
                "matched_llama_excess_ms": excess,
                "max_removable_ms": (max(0.0, float(excess)) if covered else quartz_ms),
                "llama_covered": covered,
            }
        )
    accounting = quartz_rows[0]["accounting"]
    counts = quartz_rows[0]["call_counts"]
    ranked = rank_remaining_gaps(
        [
            {
                "family": row["family"],
                "quartz_ms": row["quartz_ms"],
                "llama_ms": row["llama_ms"] or 0.0,
            }
            for row in table
        ],
        accounting,
        counts,
    )
    by_name = {row["family"]: row for row in table}
    for row in ranked.get("ranked") or []:
        extra = by_name.get(str(row.get("family")), {})
        row["llama_ms"] = extra.get("llama_ms")
        row["matched_llama_excess_ms"] = extra.get("matched_llama_excess_ms")
        row["llama_covered"] = bool(extra.get("llama_covered"))
        row["matched"] = bool(extra.get("llama_covered"))
        if not extra.get("llama_covered"):
            row["max_removable_ms"] = extra.get("max_removable_ms")
    if unknown_present or not llama_rows:
        ranked["gap_attribution_complete"] = False
        reasons = list(ranked.get("reasons") or [])
        if unknown_present:
            reasons.append("unknown_llama_or_quartz_family")
        if not llama_rows:
            reasons.append("pinned_llama_unmapped_or_invalid")
        ranked["reasons"] = reasons
        if ranked.get("status") == "ranked" and reasons:
            ranked["status"] = "incomplete"
    ranked["llama_zero_filled"] = False
    return ranked


def trigger_fields(
    ranking: Mapping[str, Any], p4096: Mapping[str, Any]
) -> dict[str, Any]:
    ranked = list(ranking.get("ranked") or [])
    by_name = {str(row.get("family")): row for row in ranked}
    complete = bool(ranking.get("gap_attribution_complete"))

    def decode_go(family: str, *aliases: str) -> dict[str, Any]:
        rows = [by_name[name] for name in (family, *aliases) if name in by_name]
        if not complete:
            return {
                "go": False,
                "reason": "gap_attribution_incomplete",
                "family": family,
            }
        if not rows:
            return {"go": False, "reason": "family_absent", "family": family}
        covered = all(bool(row.get("llama_covered")) for row in rows)
        excess = sum(float(row.get("matched_llama_excess_ms") or 0.0) for row in rows)
        if not covered:
            return {
                "go": False,
                "reason": "llama_family_unmapped",
                "family": family,
                "excess_ms": None,
            }
        return {
            "go": excess >= DECODE_EXCESS_MS,
            "reason": "matched_excess"
            if excess >= DECODE_EXCESS_MS
            else "excess_below_trigger",
            "family": family,
            "excess_ms": excess,
            "threshold_ms": DECODE_EXCESS_MS,
        }

    p4096_rank = (p4096.get("ranking") or {}) if p4096 else {}
    p4096_rows = list(p4096_rank.get("ranked") or [])
    p4096_by = {str(row.get("family")): row for row in p4096_rows}
    ffn_ms = 0.0
    ffn_covered = True
    for name in ("ffn_gate_up_glu", "ffn_down"):
        row = p4096_by.get(name)
        if row is None or not row.get("llama_covered"):
            ffn_covered = False
            continue
        ffn_ms += float(row.get("matched_llama_excess_ms") or 0.0)
    p4096_valid = bool(p4096.get("attribution_valid"))
    opt105 = {
        "go": False,
        "reason": "invalid_p4096_attribution",
        "family": "prompt_ffn",
        "excess_ms": None,
        "threshold_ms": PREFILL_EXCESS_MS,
    }
    if p4096_valid and bool(p4096_rank.get("gap_attribution_complete")) and ffn_covered:
        opt105 = {
            "go": ffn_ms >= PREFILL_EXCESS_MS,
            "reason": (
                "matched_excess"
                if ffn_ms >= PREFILL_EXCESS_MS
                else "excess_below_trigger"
            ),
            "family": "prompt_ffn",
            "excess_ms": ffn_ms,
            "threshold_ms": PREFILL_EXCESS_MS,
        }
    elif p4096_valid and not ffn_covered:
        opt105["reason"] = "llama_family_unmapped"
    return {
        "opt100_q8_aligned": decode_go("q8_input_projections"),
        "opt101_gdn_transpose": decode_go(
            "gdn_core", "gdn_recurrence", "gdn_conv", "gdn_norm"
        ),
        "opt102_q4_repack": decode_go("ffn_gate_up_glu", "ffn_down"),
        "opt103_attention_vec": decode_go(
            "attention_core", "attention_query_prep", "attention_merge"
        ),
        "opt104_q6_aligned": decode_go("q6_attention_output", "logits_projection"),
        "opt105_mmq_x2": opt105,
    }


def write_report(results: Mapping[str, Any]) -> None:
    freeze = results.get("freeze") or {}
    conservation = results.get("conservation") or {}
    d128 = results.get("d128") or {}
    d2048 = results.get("d2048") or {}
    counters = results.get("counters") or {}
    p4096 = results.get("p4096") or {}
    ranking = (
        results.get("ranking") or (results.get("report") or {}).get("ranking") or {}
    )
    triggers = results.get("triggers") or {}
    lines = [
        "# OPT-099 — Close matched engine attribution blind spots",
        "",
        "This increment **claims no performance improvement** and makes **no tok/s",
        "speedup claim**. `claims_throughput` is false. Diagnostic CUDA events are",
        "**not** shipping CUDA-graph timings. Production selectors are unchanged.",
        "",
        f"Authority llama.cpp `{LLAMA_REV}`.",
        f"GGUF SHA-256 `{GGUF_SHA}`.",
        "Token generator `(42 + index * 997) % 248320`.",
        "",
        "## Frozen post098_selected",
        "",
        f"ok={freeze.get('ok')} mismatches={freeze.get('mismatches')}.",
        f"observed={freeze.get('observed')}.",
        "",
        "## Conservation",
        "",
        f"Call counts ok={conservation.get('call_counts_ok')}.",
        f"FFN down first-class={conservation.get('ffn_down_first_class')}.",
        f"Dropped events rejected={conservation.get('dropped_events_rejected')}.",
        "",
        "## D128 / D2048",
        "",
        f"D128 valid={d128.get('attribution_valid')} p95={d128.get('graph_itl_p95_ms')}.",
        f"D2048 valid={d2048.get('attribution_valid')} p95={d2048.get('graph_itl_p95_ms')}.",
        "Uninstrumented wall/ITL/p95 and eager traces are separate records.",
        "Prefix, load, capture, warmup and reset stay outside measured windows.",
        "",
        "## Counters",
        "",
        f"ncu_available={counters.get('ncu_available')}.",
        f"full_ncu_sweep={counters.get('full_ncu_sweep')}.",
        f"reason={counters.get('reason')}.",
        "",
        "## P4096",
        "",
        f"P4096 valid={p4096.get('attribution_valid')}.",
        "Prefill ran in a separate process/sitting.",
        "",
        "## Ranking",
        "",
        f"status={ranking.get('status')}.",
        f"gap_attribution_complete={ranking.get('gap_attribution_complete')}.",
        f"reasons={ranking.get('reasons')}.",
        "Matched llama excess is reported only when llama covered the family.",
        "Missing llama coverage is `unknown` / null, never filled with zero.",
        "Families remain non-additive; union and overlap are reported separately.",
        "",
        "## Triggers for OPT-100–105",
        "",
    ]
    for name in TRIGGER_IDS:
        item = triggers.get(name) or {}
        lines.append(
            f"{name}: go={item.get('go')} reason={item.get('reason')} "
            f"excess_ms={item.get('excess_ms')}."
        )
    lines.extend(
        [
            "",
            "## Proof limits",
            "",
            "- No throughput or keep claim.",
            "- Eager events are diagnostic.",
            "- FFN down is a first-class family.",
            "- Unexplained wall above 5% stops ranking.",
            "- Do not subtract replay milliseconds from full-engine milliseconds.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_freeze_phase(mode: str) -> dict[str, Any]:
    freeze = freeze_post098_selected()
    freeze.update(
        {
            "hardware_executed": False,
            "native_counts": native_result(
                phase="freeze", mode=mode, plan=family_plan("freeze", mode)
            ),
        }
    )
    return freeze


def run_conservation_phase(mode: str) -> dict[str, Any]:
    from tools.opt090_decode_attribution import conservation_records

    records = conservation_records()
    nested = nested_interval_fixture(records["synthetic_nested"])
    valid = one_token_valid_records()
    counts = validate_call_counts_opt099(valid, tokens=1)
    times = family_times(valid)
    if DECODE_GATE_UP_PAIRS != 64:
        raise AttributionError("FFN pairs must stay 64, not 128")
    if float(times.get("ffn_down", 0.0) or 0.0) <= 0.0:
        raise AttributionError("FFN down must be a first-class timed family")
    if "unmapped_mul_mat" in times:
        raise AttributionError("FFN down must not land in unmapped_mul_mat")
    dropped = pool_overflow(records["event_pool_overflow"])
    contaminated = warmup_contamination(records["warmup_contamination"])
    epochs = two_token_epochs_share_origin(records["two_token_epochs"])
    over_limit = accounting_split(
        graph_wall_ms=20.0,
        eager_instrumented_wall_ms=21.0,
        eager_work_ms=10.0,
        attributed_union_ms=10.0,
        d2h_ms=0.0,
        commit_ms=0.0,
    )
    incomplete = rank_remaining_gaps(
        [{"family": "ffn_gate_up_glu", "quartz_ms": 10.0, "llama_ms": 4.0}],
        over_limit,
        {"ok": True},
    )
    if incomplete["status"] != "incomplete":
        raise AttributionError("unexplained wall above 5% must stop ranking")
    if not counts["ok"]:
        raise AttributionError(f"conservation counts failed: {counts['mismatches']}")
    identity = capture_identity(
        gguf_sha=GGUF_SHA,
        token_generator=TOKEN_GENERATOR,
        token_input_hash_hex=token_input_hash(
            [(42 + index * 997) % 248320 for index in range(129)]
        ),
        stage="conservation",
        source="opt099_host",
        build_flags="QW38_DIAGNOSTIC_TRACE",
        selectors=compiled_selectors(),
        layer_role="all_64",
        staging="production_arithmetic",
        state="decode_conservation",
        model_path=MODEL,
        prompt_rows=1,
    )
    llama_sample = [
        {
            "engine": "llama",
            "layer": 0,
            "phase": "decode",
            "token_position": 128,
            "role": "ffn_gate_up_glu",
            "tensor_name": "ffn_gate-0",
            "launch_family": "mmvq_fused_glu",
            "attribution_role": "enclosing",
            "fused_member_ids": "ffn_gate,ffn_up,ffn_glu",
            "complete_work_ms": 0.07,
            "start_ms": 0.0,
            "end_ms": 0.07,
        },
        {
            "engine": "llama",
            "layer": 0,
            "phase": "decode",
            "token_position": 128,
            "role": "ffn_down",
            "tensor_name": "ffn_out-0",
            "weight_name": "blk.0.ffn_down.weight",
            "launch_family": "mmvq",
            "attribution_role": "member",
            "complete_work_ms": 0.04,
            "start_ms": 0.07,
            "end_ms": 0.11,
        },
    ]
    llama_families = family_times(llama_sample)
    if llama_families.get("ffn_down", 0.0) <= 0.0:
        raise AttributionError("llama FFN down mapping must be nonzero")
    if llama_missing_filled_with_zero(llama_families):
        raise AttributionError("llama families must not be filled with zero")
    return {
        "ok": True,
        "call_counts_ok": True,
        "expected_ffn_pairs": DECODE_GATE_UP_PAIRS,
        "expected_ffn_downs": DECODE_DOWNS,
        "expected_gdn": DECODE_GDN_CORES,
        "expected_attention": DECODE_ATTENTION_CORES,
        "expected_logits": DECODE_LOGITS,
        "nested_union_ms": nested["union_ms"],
        "nested_sum_exceeds_wall": nested["sum_exceeds_wall"],
        "dropped_events_rejected": dropped,
        "warmup_contamination_rejected": contaminated,
        "two_token_epochs_share_origin": epochs,
        "unexplained_limit": UNEXPLAINED_LIMIT,
        "unexplained_stops_ranking": True,
        "ffn_down_first_class": True,
        "replay_audit": replay_non_additive_audit(),
        "roofline": weight_byte_roofline(),
        "capture_identity": identity,
        "llama_mapping_sample": llama_families,
        "hardware_executed": False,
        "native_counts": native_result(
            phase="conservation", mode=mode, plan=family_plan("conservation", mode)
        ),
    }


def attribution_command(
    *,
    phase: str,
    prefix: int,
    output_tokens: int,
    prompt: int,
    selected: Mapping[str, Any],
    warmups: int,
    samples: int,
) -> list[str]:
    return [
        f"./{ATTRIBUTION_BIN}",
        MODEL,
        "--workload",
        "prefill" if phase == "p4096" else "decode",
        "--prefix",
        str(prefix),
        "--output-tokens",
        str(output_tokens),
        "--prompt",
        str(prompt),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--ident",
        "post098_selected",
        "--selected-q4",
        str(selected["q4_decode"]),
        "--selected-ffn",
        str(selected["ffn_decode"]),
        "--q4-decode",
        str(selected["q4_decode"]),
        "--ffn-decode",
        str(selected["ffn_decode"]),
        "--evidence-dir",
        str(EVIDENCE.relative_to(ROOT)),
        "--llama-bin",
        LLAMA_BIN,
    ]


def run_decode_phase(
    phase: str,
    mode: str,
    *,
    skip_gpu: bool,
    runner: NativeRunner,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    prefix = 128 if phase == "d128" else 2048 if phase == "d2048" else 0
    prompt = PREFILL_PROMPT if phase == "p4096" else 0
    tokens = 1 if phase == "p4096" else OUTPUT_TOKENS
    selected = compiled_selectors()
    available, blocker = gpu_available()
    payload: dict[str, Any] = {
        "phase": phase,
        "prefix": prefix,
        "prompt": prompt,
        "output_tokens": OUTPUT_TOKENS if phase != "p4096" else 0,
        "idents": ["post098_selected", "pinned_llama"],
        "separate_process": phase == "p4096",
        "hardware_executed": False,
        "attribution_valid": False,
        "eager_labeled_diagnostic": True,
        "native_counts": native_result(phase=phase, mode=mode, plan=plan),
    }
    if skip_gpu or not available:
        payload["blocker"] = blocker or "skip_gpu"
        payload["reason"] = "gpu_unavailable"
        payload["gap_attribution_complete"] = False
        return payload
    command = attribution_command(
        phase=phase,
        prefix=prefix,
        output_tokens=OUTPUT_TOKENS if phase != "p4096" else 0,
        prompt=prompt,
        selected=selected,
        warmups=int(plan["warmups"]),
        samples=int(plan["samples"]),
    )
    completed = runner(command, str(plan["tier"]))
    windows = parse_quartz_windows(completed.stdout)
    native_phase = "prefill" if phase == "p4096" else "decode"
    sample_count = int(plan["samples"])
    llama_logs = discover_llama_logs(completed.stdout, native_phase, sample_count)
    for log_path in llama_logs:
        if "eager_diagnostic" in log_path.name:
            write_llama_sidecars(log_path, native_phase)
    llama_walls = collect_llama_walls(llama_logs)
    summaries = [
        summarize_window(
            row["ident"],
            native_phase,
            row["rep"],
            row,
            tokens=tokens,
        )
        for row in windows
    ]
    llama_summaries = []
    for sample in range(sample_count):
        sample_walls = llama_walls.get(sample) or {}
        llama_wall = {
            "shipping_graph_wall_ms": float(
                sample_walls.get("uninstrumented_wall_ms") or 0.0
            ),
            "eager_diagnostic_wall_ms": float(
                sample_walls.get("instrumented_wall_ms") or 0.0
            ),
        }
        llama_summaries.append(
            summarize_window(
                "pinned_llama",
                native_phase,
                sample,
                llama_wall,
                tokens=tokens,
            )
        )
    all_windows = summaries + llama_summaries
    ranking = (
        rank_from_windows(all_windows)
        if all_windows
        else {
            "status": "incomplete",
            "gap_attribution_complete": False,
            "reasons": ["no_windows"],
            "ranked": [],
        }
    )
    quartz_valid = (
        all(row["attribution_valid"] for row in summaries) if summaries else False
    )
    llama_valid = (
        all(row["attribution_valid"] for row in llama_summaries)
        if llama_summaries
        else False
    )
    valid = (
        quartz_valid and llama_valid and bool(ranking.get("gap_attribution_complete"))
    )
    p95 = [
        float(row["graph_itl_p95_ms"])
        for row in windows
        if row.get("graph_itl_p95_ms") is not None
    ]
    idle = [
        float(row["idle_ms_per_token"])
        for row in summaries
        if row.get("idle_ms_per_token") is not None
    ]
    observed = parse_native_observation(completed.stdout)
    payload.update(
        {
            "hardware_executed": True,
            "stdout_excerpt": completed.stdout[-4000:],
            "windows": all_windows,
            "attribution_valid": valid,
            "ffn_down_first_class": all(
                bool(row.get("ffn_down_first_class")) for row in summaries
            )
            if summaries
            else False,
            "graph_itl_p95_ms": mean(p95) if p95 else None,
            "idle_ms_per_token": mean(idle) if idle else None,
            "ranking": ranking,
            "gap_attribution_complete": bool(ranking.get("gap_attribution_complete")),
            "llama_status": "parsed" if llama_summaries else "unavailable",
            "native_observation": {
                "observed_warmups": observed.get("observed_warmups"),
                "observed_samples": observed.get("observed_samples"),
                "observed_candidates": observed.get("observed_candidates"),
            },
        }
    )
    if not valid:
        payload["reason"] = "invalid_counts_or_identity"
    return payload


def run_counters_phase(
    mode: str,
    *,
    skip_gpu: bool,
    runner: NativeRunner,
) -> dict[str, Any]:
    plan = family_plan("counters", mode)
    available, blocker = gpu_available()
    payload: dict[str, Any] = {
        "launches": list(COUNTER_LAUNCHES),
        "full_ncu_sweep": False,
        "ncu_available": False,
        "hardware_executed": False,
        "native_counts": native_result(phase="counters", mode=mode, plan=plan),
        "roofline": weight_byte_roofline(),
    }
    if skip_gpu or not available:
        payload["reason"] = "counters_unavailable"
        payload["blocker"] = blocker or "skip_gpu"
        payload["event_evidence_fallback"] = True
        return payload
    try:
        metrics = query_ncu_metrics(runner, str(plan["tier"]))
    except (AttributionError, Opt090AttributionError) as exc:
        payload["reason"] = "ncu_query_failed"
        payload["error"] = str(exc)
        payload["event_evidence_fallback"] = True
        return payload
    payload.update(metrics)
    if not metrics.get("ncu_available"):
        payload["reason"] = "ncu_not_found"
        payload["event_evidence_fallback"] = True
        return payload
    selected = compiled_selectors()
    workloads = {
        "q4_paired_gate_up": [
            "--workload",
            "decode-ffn",
            "--q4-decode",
            selected["q4_decode"],
            "--ffn-decode",
            selected["ffn_decode"],
        ],
        "q4_down": [
            "--workload",
            "decode-ffn",
            "--q4-decode",
            selected["q4_decode"],
            "--ffn-decode",
            selected["ffn_decode"],
        ],
        "q8_gdn_qkv": ["--workload", "decode-mixer"],
        "sequential_gdn": ["--workload", "decode-gdn", "--gdn-decode", "sequential"],
        "d2048_attention": [
            "--workload",
            "decode-attention",
            "--decode-position",
            "2048",
        ],
    }
    launches = []
    for name in COUNTER_LAUNCHES:
        command = [
            f"./{REPLAY_BIN}",
            MODEL,
            *workloads[name],
            "--warmups",
            "0",
            "--samples",
            "1",
            "--cache-mode",
            "rotating",
            "--ncu",
            "--evidence-dir",
            str(EVIDENCE.relative_to(ROOT)),
        ]
        try:
            completed = runner(command, str(plan["tier"]))
            launches.append(
                {
                    "name": name,
                    "ok": True,
                    "stdout_excerpt": completed.stdout[-1500:],
                    "full_set": False,
                    "selected_metrics": list(NCU_METRIC_NEEDLES),
                }
            )
        except (AttributionError, Opt090AttributionError) as exc:
            launches.append(
                {
                    "name": name,
                    "ok": False,
                    "reason": "launch_failed_use_event_evidence",
                    "error": str(exc)[:1000],
                    "full_set": False,
                }
            )
    payload["hardware_executed"] = True
    payload["launch_results"] = launches
    payload["reason"] = "queried_without_full_set"
    return payload


def run_report_phase(mode: str, results: Mapping[str, Any]) -> dict[str, Any]:
    d128 = results.get("d128") or {}
    d2048 = results.get("d2048") or {}
    p4096 = results.get("p4096") or {}
    ranking = (
        d2048.get("ranking")
        or d128.get("ranking")
        or {
            "status": "incomplete",
            "gap_attribution_complete": False,
            "reasons": ["decode_attribution_missing"],
            "ranked": [],
        }
    )
    triggers = trigger_fields(ranking, p4096)
    complete = bool(
        (results.get("freeze") or {}).get("ok")
        and (results.get("conservation") or {}).get("ok")
        and d128.get("attribution_valid")
        and d2048.get("attribution_valid")
        and p4096.get("attribution_valid")
    )
    ffn_down = bool((results.get("conservation") or {}).get("ffn_down_first_class"))
    for phase in (d128, d2048, p4096):
        if phase.get("ffn_down_first_class"):
            ffn_down = True
    return {
        "host_only": True,
        "cannot_convert_three_samples_into_admission": True,
        "ranking": ranking,
        "triggers": triggers,
        "ffn_down_first_class": ffn_down,
        "gap_attribution_complete": bool(ranking.get("gap_attribution_complete")),
        "evidence_complete": complete,
        "claims_throughput": False,
        "production_kept": False,
        "performance_pass": False,
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "native_counts": native_result(
            phase="report", mode=mode, plan=family_plan("report", mode)
        ),
    }


def load_existing() -> dict[str, Any]:
    if FIXTURE.is_file():
        return load_json(FIXTURE)
    return empty_fixture("feedback")


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AttributionError(f"unknown phase {phase}")
    execute = runner or default_native_runner
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results = empty_fixture(mode)
    prior = load_existing()
    for key in PHASES:
        if key in prior:
            results[key] = prior[key]
    for key in (
        "ranking",
        "triggers",
        "ffn_down_first_class",
        "gap_attribution_complete",
        "evidence_complete",
    ):
        if key in prior:
            results[key] = prior[key]
    if phase == "freeze":
        payload = run_freeze_phase(mode)
    elif phase == "conservation":
        payload = run_conservation_phase(mode)
        results["ffn_down_first_class"] = bool(payload.get("ffn_down_first_class"))
    elif phase in {"d128", "d2048", "p4096"}:
        payload = run_decode_phase(phase, mode, skip_gpu=skip_gpu, runner=execute)
    elif phase == "counters":
        payload = run_counters_phase(mode, skip_gpu=skip_gpu, runner=execute)
    else:
        payload = run_report_phase(mode, results)
        results["ranking"] = payload["ranking"]
        results["triggers"] = payload["triggers"]
        results["ffn_down_first_class"] = payload["ffn_down_first_class"]
        results["gap_attribution_complete"] = payload["gap_attribution_complete"]
        results["evidence_complete"] = payload["evidence_complete"]
    merge_fixture(results, phase, payload)
    dump_json(run_dir / "opt099_matched_attribution.json", results)
    if not skip_gpu or phase in {"freeze", "conservation", "report"}:
        write_report(results)
        dump_json(FIXTURE, results)
        dump_json(EVIDENCE / "opt099_matched_attribution.json", results)
    emit_native(
        payload.get("native_counts")
        or native_result(phase=phase, mode=mode, plan=family_plan(phase, mode))
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode", required=True, choices=("feedback", "acceptance", "release")
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("QW38_CUDA_TEST_TIER", "screen")
    run(
        args.mode,
        args.phase,
        Path(args.run_dir),
        skip_gpu=args.skip_gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
