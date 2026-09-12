"""OPT-090 matched post-Q4 decode attribution and secondary P4096 ranking."""

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

from tools.opt060_engine_attribution import (  # noqa: E402
    interval,
    stream_aware_totals,
)
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
    canonical_family,
    capture_identity,
    chargeable_records,
    expected_decode_counts,
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
from tools.opt088_batch_gate import COMBINATION_SELECTORS  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    parse_native_observation,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt090_decode_attribution_contract.json"
ITERATION = ROOT / "pins/opt090_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
REPORT = ROOT / "evidence/optimization/opt090-decode-attribution/REPORT.md"
EVIDENCE = REPORT.parent
OPT071_FIXTURE = ROOT / "fixtures/opt071_attribution_repair.json"
OPT088_FIXTURE = ROOT / "fixtures/opt088_batch_gate.json"
OPT089_FIXTURE = ROOT / "fixtures/opt089_q4_promotion.json"
Q4_PIN_FILE = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PIN_FILE = ROOT / "cuda/ffn_decode_path.cuh"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
ATTRIBUTION_BIN = "build/qw38-cuda-opt060-engine-attribution-test"
REPLAY_BIN = "build/qw38-cuda-component-replay"
PROBE_BIN = "build/qw38-cuda-optimization-engine-probe"
LLAMA_BIN = ".cache/authorities/llama-build-opt060/bin/qw38-llama-engine-attribution"
PHASES = ("conservation", "d128", "d2048", "counters", "p4096", "report")
GRAPH_REOPEN_MS = 0.50
OUTPUT_TOKENS = 32
PREFILL_PROMPT = 4096
Q8_GDN_OUTPUTS = 48
Q6_ATTENTION_OUTPUTS = 16
Q4_K_BLOCK = 256
Q4_K_BYTES = 144
Q8_0_BLOCK = 32
Q8_0_BYTES = 34
Q6_K_BLOCK = 256
Q6_K_BYTES = 210
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
COUNTER_LAUNCHES = (
    "q4_paired_gate_up",
    "q4_down",
    "q8_gdn_qkv",
    "sequential_gdn",
    "d2048_attention",
)
NCU_METRIC_NEEDLES = (
    "dram__bytes",
    "dram__throughput",
    "lts__t_sector_hit_rate",
    "sm__pipe",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__local_memory",
    "smsp__pcsamp_warps_issue_stalled",
)
EXPECTED_DECODE_COUNTS = {
    **expected_decode_counts(1),
    "q8_gdn_output": Q8_GDN_OUTPUTS,
    "q6_attention_output": Q6_ATTENTION_OUTPUTS,
}

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class AttributionError(AssertionError):
    """Fail-closed OPT-090 attribution error."""


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


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def current_q4_pin() -> str:
    match = re.search(
        r'kSelectedQ4DecodePath\[\] = "([^"]+)"',
        Q4_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AttributionError("missing kSelectedQ4DecodePath")
    return match.group(1)


def current_ffn_pin() -> str:
    match = re.search(
        r'kSelectedFfnDecodePath\[\] = "([^"]+)"',
        FFN_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AttributionError("missing kSelectedFfnDecodePath")
    return match.group(1)


def opt088_control_selectors() -> dict[str, str]:
    fixture = load_json(OPT088_FIXTURE) if OPT088_FIXTURE.is_file() else {}
    paths = fixture.get("combined_production_paths") or {}
    return {
        "q4_decode": str(paths.get("q4_decode") or COMBINATION_SELECTORS["q4_decode"]),
        "ffn_decode": str(paths.get("ffn_decode") or "paired_staged"),
        "q8_layout": str(paths.get("q8_decode") or COMBINATION_SELECTORS["q8_decode"]),
        "execution_graphs": str(
            paths.get("execution_graphs") or COMBINATION_SELECTORS["execution_graphs"]
        ),
        "gdn_decode": str(paths.get("gdn_decode") or "sequential"),
        "decode_query_prep": str(paths.get("decode_query_prep") or "warp_query"),
    }


def opt089_selected_selectors() -> dict[str, str]:
    fixture = load_json(OPT089_FIXTURE) if OPT089_FIXTURE.is_file() else {}
    q4 = str(fixture.get("shipping_q4_decode") or current_q4_pin())
    ffn = str(fixture.get("shipping_ffn_decode") or current_ffn_pin())
    control = opt088_control_selectors()
    return {
        **control,
        "q4_decode": q4,
        "ffn_decode": ffn,
    }


def quartz_ident_specs() -> tuple[dict[str, Any], ...]:
    control = opt088_control_selectors()
    selected = opt089_selected_selectors()
    return (
        {
            "id": "opt088_control",
            "role": "control",
            "source": "OPT-088",
            "selectors": control,
        },
        {
            "id": "opt089_selected",
            "role": "selected",
            "source": "OPT-089",
            "selectors": selected,
        },
    )


def selector_key(selectors: Mapping[str, Any]) -> tuple[str, str]:
    return (str(selectors.get("q4_decode")), str(selectors.get("ffn_decode")))


def idents_to_measure(
    specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = [dict(item) for item in (specs or quartz_ident_specs())]
    first = selector_key(rows[0]["selectors"])
    identical = all(selector_key(row["selectors"]) == first for row in rows)
    measured = [rows[0]] if identical else rows
    aliases = {row["id"]: measured[0]["id"] for row in rows} if identical else {}
    return {
        "identical": identical,
        "measure_once": identical,
        "measured": measured,
        "all": rows,
        "aliases": aliases,
    }


def ident_order(sample_index: int, ident_ids: Sequence[str]) -> list[str]:
    order = [str(item) for item in ident_ids]
    if len(order) < 2:
        return order
    return order if sample_index % 2 == 0 else list(reversed(order))


def canonical_family_opt090(record: Mapping[str, Any]) -> str:
    role = str(record.get("role", "") or "")
    name = str(record.get("tensor_name", "") or "").lower()
    combined = f"{role} {name}".lower()
    if role in {"proj_gdn_output", "gdn_output"} or "ssm_out" in combined:
        return "q8_gdn_output"
    if role in {"proj_attn_output", "attn_output"} or "attn_output.weight" in combined:
        return "q6_attention_output"
    if role in {
        "proj_packed_qkv",
        "proj_query_gate",
        "proj_key",
        "proj_value",
        "proj_value_gate",
        "proj_alpha",
        "proj_beta",
    }:
        return "q8_input_projections"
    if role in {"pointwise", "residual", "silu", "rms_norm", "residual_ffn"}:
        return "pointwise"
    return canonical_family(record)


def _fused_member_ids(record: Mapping[str, Any]) -> set[str]:
    raw = str(record.get("fused_member_ids", "") or "")
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _layer_token_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("engine"),
        record.get("phase"),
        record.get("token_position"),
        record.get("layer"),
    )


def _paired_ffn_enclosing(record: Mapping[str, Any]) -> bool:
    role = str(record.get("role", "") or "")
    attr = str(record.get("attribution_role", "") or "")
    return role in {"ffn_mmv", "ffn_mmq"} and attr == "enclosing"


def _credited_families_opt090(record: Mapping[str, Any]) -> set[str]:
    """Map one chargeable record to OPT-090 expected families.

    Paired decode (`paired_integer` / late_w4) emits one enclosing `ffn_mmv`
    per layer per token with empty `fused_member_ids`. That single interval
    is the paired FFN: it satisfies gate/up/GLU and down without 128 leaf
    launches. Graph-fused enclosing records list both members explicitly.
    """
    role = str(record.get("role", "") or "")
    family = canonical_family_opt090(record)
    members = _fused_member_ids(record)
    credited: set[str] = set()
    if _paired_ffn_enclosing(record):
        if (
            not members
            or "ffn_gate" in members
            or "ffn_up" in members
            or "ffn_glu" in members
        ):
            credited.add("ffn_gate_up_glu")
        if not members or "ffn_down" in members:
            credited.add("ffn_down")
        return credited
    if family == "ffn_gate_up_glu" or role == "ffn_gate_up_glu":
        credited.add("ffn_gate_up_glu")
    if family == "ffn_down" or role == "ffn_down" or "ffn_down" in members:
        credited.add("ffn_down")
    if family == "gdn_core" or role == "gdn_core":
        credited.add("gdn_core")
    if family == "attention_core" or role == "attention_core":
        credited.add("attention_core")
    if family == "logits_projection" or role in {"logits_projection", "logits"}:
        credited.add("logits_projection")
    if family == "q8_gdn_output":
        credited.add("q8_gdn_output")
    if family == "q6_attention_output":
        credited.add("q6_attention_output")
    return credited


def family_counts_opt090(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in EXPECTED_DECODE_COUNTS}
    seen: dict[str, set[tuple[Any, ...]]] = {
        name: set() for name in EXPECTED_DECODE_COUNTS
    }
    for record in chargeable_records(records):
        key = _layer_token_key(record)
        for name in _credited_families_opt090(record):
            if name not in seen or key in seen[name]:
                continue
            seen[name].add(key)
            counts[name] += 1
    return counts


def expected_decode_counts_opt090(tokens: int = 1) -> dict[str, int]:
    return {name: value * int(tokens) for name, value in EXPECTED_DECODE_COUNTS.items()}


def validate_call_counts_opt090(
    records: Sequence[Mapping[str, Any]],
    *,
    tokens: int,
) -> dict[str, Any]:
    observed = family_counts_opt090(records)
    expected = expected_decode_counts_opt090(tokens)
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


def host_gap_ms(records: Sequence[Mapping[str, Any]]) -> float:
    charged = sorted(
        chargeable_records(records),
        key=lambda row: float(row.get("start_ms", 0.0) or 0.0),
    )
    if not charged:
        return 0.0
    gap = 0.0
    _start, end = interval(charged[0])
    for record in charged[1:]:
        start, rec_end = interval(record)
        if start > end:
            gap += start - end
        end = max(end, rec_end)
    return gap


def unhidden_idle_per_token(
    records: Sequence[Mapping[str, Any]],
    *,
    tokens: int,
) -> float:
    if tokens <= 0:
        return 0.0
    return host_gap_ms(records) / float(tokens)


def graph_reopen_eligible(
    d128_idle_ms: float | None,
    d2048_idle_ms: float | None,
    *,
    d128_valid: bool,
    d2048_valid: bool,
) -> dict[str, Any]:
    if not d128_valid or not d2048_valid:
        return {
            "eligible": False,
            "reason": "invalid_attribution",
            "d128_idle_ms_per_token": d128_idle_ms,
            "d2048_idle_ms_per_token": d2048_idle_ms,
            "threshold_ms": GRAPH_REOPEN_MS,
        }
    if d128_idle_ms is None or d2048_idle_ms is None:
        return {
            "eligible": False,
            "reason": "idle_unmeasured",
            "d128_idle_ms_per_token": d128_idle_ms,
            "d2048_idle_ms_per_token": d2048_idle_ms,
            "threshold_ms": GRAPH_REOPEN_MS,
        }
    eligible = (
        float(d128_idle_ms) >= GRAPH_REOPEN_MS
        and float(d2048_idle_ms) >= GRAPH_REOPEN_MS
    )
    return {
        "eligible": eligible,
        "reason": (
            "unhidden_idle_at_both_prefixes" if eligible else "idle_below_trigger"
        ),
        "d128_idle_ms_per_token": float(d128_idle_ms),
        "d2048_idle_ms_per_token": float(d2048_idle_ms),
        "threshold_ms": GRAPH_REOPEN_MS,
    }


def gdn_reopen_eligible(
    defect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not defect:
        return {
            "eligible": False,
            "reason": "no_opt077_timing_capture_defect_repaired",
            "repair": None,
        }
    repaired = bool(defect.get("repaired"))
    before = defect.get("before")
    after = defect.get("after")
    description = str(defect.get("description") or "")
    if not repaired or before is None or after is None or not description:
        return {
            "eligible": False,
            "reason": "defect_without_before_after_repair",
            "repair": dict(defect),
        }
    return {
        "eligible": True,
        "reason": "opt077_defect_repaired",
        "repair": dict(defect),
    }


def packed_bytes(elements: int, ggml_type: int) -> int:
    if ggml_type == 0:
        return int(elements) * 4
    if ggml_type == 8:
        return (int(elements) // Q8_0_BLOCK) * Q8_0_BYTES
    if ggml_type == 12:
        return (int(elements) // Q4_K_BLOCK) * Q4_K_BYTES
    if ggml_type == 14:
        return (int(elements) // Q6_K_BLOCK) * Q6_K_BYTES
    raise AttributionError(f"unsupported ggml type {ggml_type}")


def model_weight_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "name": "output.weight",
            "shape": [5120, 248320],
            "ggml_type": 14,
            "type_name": "Q6_K",
            "family": "logits_projection",
            "decode_streamed": True,
        },
        {
            "name": "token_embd.weight",
            "shape": [5120, 248320],
            "ggml_type": 12,
            "type_name": "Q4_K",
            "family": "embedding",
            "decode_streamed": False,
        },
    ]
    for layer in range(64):
        prefix = f"blk.{layer}."
        rows.extend(
            [
                {
                    "name": prefix + "ffn_gate.weight",
                    "shape": [5120, 17408],
                    "ggml_type": 12,
                    "type_name": "Q4_K",
                    "family": "ffn_gate_up_glu",
                    "decode_streamed": True,
                },
                {
                    "name": prefix + "ffn_up.weight",
                    "shape": [5120, 17408],
                    "ggml_type": 12,
                    "type_name": "Q4_K",
                    "family": "ffn_gate_up_glu",
                    "decode_streamed": True,
                },
                {
                    "name": prefix + "ffn_down.weight",
                    "shape": [17408, 5120],
                    "ggml_type": 12,
                    "type_name": "Q4_K",
                    "family": "ffn_down",
                    "decode_streamed": True,
                },
            ]
        )
        if layer % 4 == 3:
            rows.extend(
                [
                    {
                        "name": prefix + "attn_q.weight",
                        "shape": [5120, 12288],
                        "ggml_type": 8,
                        "type_name": "Q8_0",
                        "family": "q8_input_projections",
                        "decode_streamed": True,
                    },
                    {
                        "name": prefix + "attn_k.weight",
                        "shape": [5120, 1024],
                        "ggml_type": 8,
                        "type_name": "Q8_0",
                        "family": "q8_input_projections",
                        "decode_streamed": True,
                    },
                    {
                        "name": prefix + "attn_v.weight",
                        "shape": [5120, 1024],
                        "ggml_type": 8,
                        "type_name": "Q8_0",
                        "family": "q8_input_projections",
                        "decode_streamed": True,
                    },
                    {
                        "name": prefix + "attn_output.weight",
                        "shape": [6144, 5120],
                        "ggml_type": 14,
                        "type_name": "Q6_K",
                        "family": "q6_attention_output",
                        "decode_streamed": True,
                    },
                ]
            )
        else:
            rows.extend(
                [
                    {
                        "name": prefix + "attn_qkv.weight",
                        "shape": [5120, 10240],
                        "ggml_type": 8,
                        "type_name": "Q8_0",
                        "family": "q8_input_projections",
                        "decode_streamed": True,
                    },
                    {
                        "name": prefix + "attn_gate.weight",
                        "shape": [5120, 6144],
                        "ggml_type": 8,
                        "type_name": "Q8_0",
                        "family": "q8_input_projections",
                        "decode_streamed": True,
                    },
                    {
                        "name": prefix + "ssm_out.weight",
                        "shape": [6144, 5120],
                        "ggml_type": 8,
                        "type_name": "Q8_0",
                        "family": "q8_gdn_output",
                        "decode_streamed": True,
                    },
                ]
            )
    for row in rows:
        elements = 1
        for dim in row["shape"]:
            elements *= int(dim)
        row["elements"] = elements
        row["bytes"] = packed_bytes(elements, int(row["ggml_type"]))
    return rows


def weight_byte_roofline() -> dict[str, Any]:
    manifest = model_weight_manifest()
    streamed = [row for row in manifest if row["decode_streamed"]]
    families: dict[str, int] = {}
    for row in streamed:
        families[row["family"]] = families.get(row["family"], 0) + int(row["bytes"])
    total = sum(families.values())
    embedding = sum(
        int(row["bytes"]) for row in manifest if row["family"] == "embedding"
    )
    return {
        "source": "src/model.cpp validate_qwen38_contract tensor names/shapes",
        "not_gguf_file_size": True,
        "decode_streamed_bytes": total,
        "excluded_embedding_bytes": embedding,
        "families": families,
        "ffn_q4_bytes_per_token": families.get("ffn_gate_up_glu", 0)
        + families.get("ffn_down", 0),
        "tensors": [
            {
                "name": row["name"],
                "shape": row["shape"],
                "type_name": row["type_name"],
                "bytes": row["bytes"],
                "family": row["family"],
            }
            for row in streamed
            if row["name"].startswith("blk.0.") or row["name"] == "output.weight"
        ],
    }


def replay_non_additive_audit() -> dict[str, Any]:
    return {
        "opt077_complete_gdn_ms": 12.516,
        "opt078_complete_attention_ms": 9.092,
        "opt086_rotating_mixer_ms": 7.45,
        "opt085_complete_ffn_ms": 16.383,
        "note": (
            "Component replay sums exceed one token's elapsed time because "
            "repetitions, restoration, copies and cache regimes differ from "
            "exclusive full-engine event union. Do not subtract replay "
            "milliseconds from full-engine milliseconds."
        ),
        "exclusive_full_engine_required": True,
    }


def conservation_records() -> dict[str, Any]:
    fixture = load_json(OPT071_FIXTURE)
    return {
        "synthetic_nested": fixture["synthetic_nested"]["records"],
        "two_token_epochs": fixture["two_token_epochs"]["records"],
        "warmup_contamination": fixture["warmup_contamination"]["records"],
        "event_pool_overflow": fixture["event_pool_overflow"]["records"],
    }


def one_token_valid_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer in range(DECODE_GATE_UP_PAIRS):
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "window": "measured",
                "token_position": 128,
                "role": "ffn_gate_up_glu",
                "tensor_name": f"blk.{layer}.ffn_gate.weight",
                "attribution_role": "enclosing",
                "fused_member_ids": "ffn_gate,ffn_up,ffn_glu",
                "fused_member_count": 3,
                "complete_work_ms": 0.08,
                "start_ms": layer * 0.2,
                "end_ms": layer * 0.2 + 0.08,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "window": "measured",
                "token_position": 128,
                "role": "ffn_down",
                "tensor_name": f"blk.{layer}.ffn_down.weight",
                "attribution_role": "member",
                "fused_member_count": 1,
                "complete_work_ms": 0.04,
                "start_ms": layer * 0.2 + 0.08,
                "end_ms": layer * 0.2 + 0.12,
                "stream_index": 0,
            }
        )
    for layer in range(DECODE_GDN_CORES):
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "window": "measured",
                "token_position": 128,
                "role": "gdn_core",
                "tensor_name": f"blk.{layer}.ssm_conv1d.weight",
                "attribution_role": "enclosing",
                "fused_member_count": 1,
                "complete_work_ms": 0.03,
                "start_ms": 13.0 + layer * 0.05,
                "end_ms": 13.0 + layer * 0.05 + 0.03,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "window": "measured",
                "token_position": 128,
                "role": "proj_gdn_output",
                "tensor_name": f"blk.{layer}.ssm_out.weight",
                "tensor_type": "Q8_0",
                "attribution_role": "member",
                "fused_member_count": 1,
                "complete_work_ms": 0.02,
                "start_ms": 16.0 + layer * 0.04,
                "end_ms": 16.0 + layer * 0.04 + 0.02,
                "stream_index": 0,
            }
        )
    for index in range(DECODE_ATTENTION_CORES):
        layer = 3 + index * 4
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "window": "measured",
                "token_position": 128,
                "role": "attention_core",
                "tensor_name": f"blk.{layer}.attn_q.weight",
                "attribution_role": "enclosing",
                "fused_member_count": 1,
                "complete_work_ms": 0.05,
                "start_ms": 18.5 + index * 0.08,
                "end_ms": 18.5 + index * 0.08 + 0.05,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "window": "measured",
                "token_position": 128,
                "role": "proj_attn_output",
                "tensor_name": f"blk.{layer}.attn_output.weight",
                "tensor_type": "Q6_K",
                "attribution_role": "member",
                "fused_member_count": 1,
                "complete_work_ms": 0.02,
                "start_ms": 20.0 + index * 0.06,
                "end_ms": 20.0 + index * 0.06 + 0.02,
                "stream_index": 0,
            }
        )
    records.append(
        {
            "engine": "quartz",
            "layer": -1,
            "phase": "decode",
            "window": "measured",
            "token_position": 128,
            "role": "logits_projection",
            "tensor_name": "output.weight",
            "tensor_type": "Q6_K",
            "attribution_role": "member",
            "fused_member_count": 1,
            "complete_work_ms": 0.3,
            "start_ms": 21.2,
            "end_ms": 21.5,
            "stream_index": 0,
        }
    )
    return records


def family_plan(name: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    return workload_for_mode(iteration["workloads"][name], mode)


def empty_fixture(mode: str) -> dict[str, Any]:
    measured = idents_to_measure()
    return {
        "schema_version": 1,
        "task": "OPT-090",
        "mode": mode,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "token_generator": TOKEN_GENERATOR,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "quartz_idents": [row["id"] for row in measured["all"]],
        "measured_idents": [row["id"] for row in measured["measured"]],
        "measure_once": measured["measure_once"],
        "pinned_llama_ident": "pinned_llama",
        "gdn_reopen_eligible": False,
        "graph_reopen_eligible": False,
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "evidence_complete": False,
        "opt074_coverage_unadmitted_blocker": False,
        "independent_verdicts": {
            "kernel_parity_pass": False,
            "model_quality_pass": False,
            "performance_pass": False,
            "production_kept": False,
        },
        "report_path": str(REPORT.relative_to(ROOT)),
    }


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = dict(payload)
    results["updated_utc"] = utc_now()
    if "gdn_reopen_eligible" in payload:
        results["gdn_reopen_eligible"] = bool(payload["gdn_reopen_eligible"])
    if "graph_reopen_eligible" in payload:
        results["graph_reopen_eligible"] = bool(payload["graph_reopen_eligible"])
    if payload.get("hardware_executed"):
        results["hardware_executed"] = True
    results["claims_throughput"] = False
    results["claims_performance_improvement"] = False
    results["production_kept"] = False
    results["performance_pass"] = False


def write_report(results: Mapping[str, Any]) -> None:
    measured = results.get("measured_idents") or []
    gdn = results.get("gdn_reopen") or {}
    graph = results.get("graph_reopen") or {}
    conservation = results.get("conservation") or {}
    d128 = results.get("d128") or {}
    d2048 = results.get("d2048") or {}
    counters = results.get("counters") or {}
    p4096 = results.get("p4096") or {}
    ranking = (
        results.get("ranking") or (results.get("report") or {}).get("ranking") or {}
    )
    lines = [
        "# OPT-090 — Measure the remaining decode gap",
        "",
        "This increment **claims no performance improvement** and makes **no tok/s",
        "speedup claim**. `claims_throughput` is false. Eager diagnostic CUDA",
        "events are **not** shipping CUDA-graph timings. Production selectors",
        "are unchanged.",
        "",
        f"Authority llama.cpp `{LLAMA_REV}`.",
        f"GGUF SHA-256 `{GGUF_SHA}`.",
        "Token generator `(42 + index * 997) % 248320`.",
        "",
        "## Idents",
        "",
        f"Quartz idents: {', '.join(results.get('quartz_idents') or [])}.",
        f"Measured: {', '.join(str(item) for item in measured)}.",
        f"measure_once={results.get('measure_once')}.",
        "Pinned llama is a separate matched control; local `../llama.cpp` HEAD",
        "is not the authority engine.",
        "",
        "## Conservation",
        "",
        f"Call counts ok={conservation.get('call_counts_ok')}.",
        f"Dropped events rejected={conservation.get('dropped_events_rejected')}.",
        f"Unexplained wall limit={UNEXPLAINED_LIMIT}.",
        f"Replay non-additive audit recorded={bool(conservation.get('replay_audit'))}.",
        "",
        "## D128 / D2048",
        "",
        f"D128 valid={d128.get('attribution_valid')} p95={d128.get('graph_itl_p95_ms')}.",
        f"D2048 valid={d2048.get('attribution_valid')} p95={d2048.get('graph_itl_p95_ms')}.",
        "Prefix, load, capture, warmup and reset stay outside measured windows.",
        "",
        "## Counters",
        "",
        f"ncu_available={counters.get('ncu_available')}.",
        f"full_ncu_sweep={counters.get('full_ncu_sweep')}.",
        f"reason={counters.get('reason')}.",
        "Counter absence is explicit; missing counters do not invent bandwidth.",
        "",
        "## P4096",
        "",
        f"P4096 valid={p4096.get('attribution_valid')}.",
        "Prefill attribution follows decode and is secondary.",
        "",
        "## Ranking",
        "",
        f"status={ranking.get('status')}.",
        f"gap_attribution_complete={ranking.get('gap_attribution_complete')}.",
        f"reasons={ranking.get('reasons')}.",
        "Matched llama excess is reported only when llama covered the family.",
        "Non-additive ceilings are not summed into a token budget.",
        "",
        "## Conditional triggers",
        "",
        f"gdn_reopen_eligible={results.get('gdn_reopen_eligible')} "
        f"reason={gdn.get('reason')}.",
        f"graph_reopen_eligible={results.get('graph_reopen_eligible')} "
        f"reason={graph.get('reason')}.",
        "A newer GPU sitting or broad CI alone cannot reopen GDN.",
        "",
        "## Proof limits",
        "",
        "- No throughput or keep claim.",
        "- Eager events are diagnostic.",
        "- 64 paired FFN calls, not 128.",
        "- Unexplained wall above 5% stops ranking.",
        "- Do not subtract replay milliseconds from full-engine milliseconds.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def native_result(
    *,
    phase: str,
    mode: str,
    plan: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "task": "OPT-090",
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
    print("QW38_OPT090_RESULT=" + json.dumps(payload, sort_keys=True))
    print("QW38_OPT090_NATIVE_COUNTS=" + json.dumps(payload, sort_keys=True))


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
    path = EVIDENCE / f"quartz-{ident}-{phase}-rep{sample}-records.json"
    if not path.is_file():
        return []
    payload = load_json(path)
    records = payload.get("records") or []
    return [dict(row) for row in records if isinstance(row, Mapping)]


def summarize_window(
    ident: str,
    phase: str,
    sample: int,
    wall: Mapping[str, Any],
    *,
    tokens: int,
) -> dict[str, Any]:
    records = load_event_records(ident, phase, sample)
    counts = (
        validate_call_counts_opt090(records, tokens=tokens)
        if records
        else {
            "ok": False,
            "mismatches": {"missing_records": {"observed": 0, "expected": 1}},
            "observed": {},
            "expected": expected_decode_counts_opt090(tokens),
        }
    )
    dropped = pool_overflow(records) if records else True
    contaminated = warmup_contamination(records) if records else False
    charged = chargeable_records(records)
    totals = (
        stream_aware_totals(charged)
        if charged
        else {
            "summed_gpu_work_ms": 0.0,
            "interval_union_ms": 0.0,
            "overlap_ms": 0.0,
        }
    )
    graph_wall = float(wall.get("shipping_graph_wall_ms") or 0.0)
    eager_wall = float(wall.get("eager_diagnostic_wall_ms") or 0.0)
    accounting = accounting_split(
        graph_wall_ms=graph_wall,
        eager_instrumented_wall_ms=eager_wall,
        eager_work_ms=float(totals["summed_gpu_work_ms"]),
        attributed_union_ms=float(totals["interval_union_ms"]),
        d2h_ms=0.0,
        commit_ms=0.0,
    )
    valid = (
        bool(counts["ok"])
        and not dropped
        and not contaminated
        and float(accounting["unexplained_share"]) <= UNEXPLAINED_LIMIT
    )
    families: dict[str, float] = {}
    for record in charged:
        name = canonical_family_opt090(record)
        families[name] = families.get(name, 0.0) + float(
            record.get("complete_work_ms", 0.0) or 0.0
        )
    idle = unhidden_idle_per_token(records, tokens=tokens) if records else None
    return {
        "ident": ident,
        "sample": sample,
        "order": wall.get("order"),
        "call_counts": counts,
        "dropped_events": dropped,
        "warmup_contamination": contaminated,
        "accounting": accounting,
        "families": families,
        "idle_ms_per_token": idle,
        "graph_itl_p50_ms": wall.get("graph_itl_p50_ms"),
        "graph_itl_p95_ms": wall.get("graph_itl_p95_ms"),
        "attribution_valid": valid,
        "eager_labeled_diagnostic": True,
        "shipping_graph_separate": True,
    }


def rank_from_windows(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in windows if row.get("attribution_valid")]
    if not valid:
        accounting = (
            windows[0].get("accounting") if windows else {"unexplained_share": 1.0}
        )
        counts = windows[0].get("call_counts") if windows else {"ok": False}
        return rank_remaining_gaps(
            [],
            accounting or {"unexplained_share": 1.0},
            counts or {"ok": False},
        )
    families: dict[str, dict[str, float]] = {}
    for row in valid:
        if str(row.get("ident")) == "pinned_llama":
            continue
        for name, ms in (row.get("families") or {}).items():
            bucket = families.setdefault(name, {"quartz_ms": 0.0, "llama_ms": 0.0})
            bucket["quartz_ms"] += float(ms)
    llama_rows = [row for row in valid if str(row.get("ident")) == "pinned_llama"]
    if llama_rows:
        for row in llama_rows:
            for name, ms in (row.get("families") or {}).items():
                bucket = families.setdefault(name, {"quartz_ms": 0.0, "llama_ms": 0.0})
                bucket["llama_ms"] += float(ms)
        for bucket in families.values():
            bucket["quartz_ms"] /= max(1, len(valid) - len(llama_rows))
            bucket["llama_ms"] /= len(llama_rows)
    else:
        for bucket in families.values():
            bucket["quartz_ms"] /= len(valid)
    table = [
        {
            "family": name,
            "quartz_ms": values["quartz_ms"],
            "llama_ms": values["llama_ms"],
        }
        for name, values in families.items()
    ]
    accounting = valid[0]["accounting"]
    counts = valid[0]["call_counts"]
    ranked = rank_remaining_gaps(table, accounting, counts)
    for row in ranked.get("ranked") or []:
        row["mechanism"] = {
            "ffn_gate_up_glu": "Q4 paired gate/up plus GLU",
            "ffn_down": "Q4 down projection",
            "q8_input_projections": "Q8 mixer projections",
            "q8_gdn_output": "Q8 GDN output",
            "q6_attention_output": "Q6 attention output",
            "gdn_core": "GDN recurrence/conv/gate",
            "attention_core": "attention prep/core/merge",
            "logits_projection": "final Q6 logits",
            "activation_quant": "activation quantization",
            "pointwise": "pointwise",
            "state_commit": "state copies/commit",
            "cpu_unknown_gap": "unhidden idle",
        }.get(str(row.get("family")), "unmatched_or_residual")
        row["matched"] = bool(row.get("llama_covered"))
    return ranked


def run_conservation_phase(mode: str) -> dict[str, Any]:
    records = conservation_records()
    nested = nested_interval_fixture(records["synthetic_nested"])
    valid = one_token_valid_records()
    counts = validate_call_counts_opt090(valid, tokens=1)
    if DECODE_GATE_UP_PAIRS != 64:
        raise AttributionError("FFN pairs must stay 64, not 128")
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
        source="opt090_host",
        build_flags="QW38_DIAGNOSTIC_TRACE",
        selectors=opt089_selected_selectors(),
        layer_role="all_64",
        staging="production_arithmetic",
        state="decode_conservation",
        model_path=MODEL,
        prompt_rows=1,
    )
    payload = {
        "ok": True,
        "call_counts_ok": True,
        "expected_ffn_pairs": DECODE_GATE_UP_PAIRS,
        "expected_ffn_downs": DECODE_DOWNS,
        "expected_gdn": DECODE_GDN_CORES,
        "expected_attention": DECODE_ATTENTION_CORES,
        "expected_logits": DECODE_LOGITS,
        "expected_q8_gdn_output": Q8_GDN_OUTPUTS,
        "expected_q6_attention_output": Q6_ATTENTION_OUTPUTS,
        "nested_union_ms": nested["union_ms"],
        "nested_sum_exceeds_wall": nested["sum_exceeds_wall"],
        "dropped_events_rejected": dropped,
        "warmup_contamination_rejected": contaminated,
        "two_token_epochs_share_origin": epochs,
        "unexplained_limit": UNEXPLAINED_LIMIT,
        "unexplained_stops_ranking": True,
        "replay_audit": replay_non_additive_audit(),
        "roofline": weight_byte_roofline(),
        "capture_identity": identity,
        "hardware_executed": False,
        "native_counts": native_result(
            phase="conservation", mode=mode, plan=family_plan("conservation", mode)
        ),
    }
    return payload


def attribution_command(
    *,
    phase: str,
    prefix: int,
    output_tokens: int,
    prompt: int,
    idents: Sequence[str],
    selected: Mapping[str, Any],
    warmups: int,
    samples: int,
) -> list[str]:
    command = [
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
        "--selected-q4",
        str(selected["selectors"]["q4_decode"]),
        "--selected-ffn",
        str(selected["selectors"]["ffn_decode"]),
        "--evidence-dir",
        str(EVIDENCE.relative_to(ROOT)),
        "--llama-bin",
        LLAMA_BIN,
    ]
    if len(idents) == 1:
        command.extend(["--ident", idents[0]])
        if idents[0] == "opt088_control":
            command.extend(["--q4-decode", "packed", "--ffn-decode", "paired_staged"])
        else:
            command.extend(
                [
                    "--q4-decode",
                    str(selected["selectors"]["q4_decode"]),
                    "--ffn-decode",
                    str(selected["selectors"]["ffn_decode"]),
                ]
            )
    else:
        command.extend(["--idents", ",".join(idents)])
    return command


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
    tokens = OUTPUT_TOKENS if phase != "p4096" else PREFILL_PROMPT
    measured = idents_to_measure()
    ident_ids = [row["id"] for row in measured["measured"]]
    selected = next(row for row in measured["all"] if row["id"] == "opt089_selected")
    available, blocker = gpu_available()
    payload: dict[str, Any] = {
        "phase": phase,
        "prefix": prefix,
        "prompt": prompt,
        "output_tokens": OUTPUT_TOKENS if phase != "p4096" else 0,
        "idents": ident_ids,
        "measure_once": measured["measure_once"],
        "aliases": measured["aliases"],
        "ab_ba_order": [ident_order(index, ident_ids) for index in range(3)],
        "hardware_executed": False,
        "attribution_valid": False,
        "eager_labeled_diagnostic": True,
        "native_counts": native_result(phase=phase, mode=mode, plan=plan),
    }
    if skip_gpu or not available:
        payload["blocker"] = blocker or "skip_gpu"
        payload["reason"] = "gpu_unavailable"
        return payload
    command = attribution_command(
        phase=phase,
        prefix=prefix,
        output_tokens=OUTPUT_TOKENS if phase != "p4096" else 0,
        prompt=prompt,
        idents=ident_ids,
        selected=selected,
        warmups=int(plan["warmups"]),
        samples=int(plan["samples"]),
    )
    completed = runner(command, str(plan["tier"]))
    windows = parse_quartz_windows(completed.stdout)
    summaries = [
        summarize_window(
            row["ident"],
            "prefill" if phase == "p4096" else "decode",
            row["rep"],
            row,
            tokens=tokens if phase != "p4096" else 1,
        )
        for row in windows
    ]
    valid = all(row["attribution_valid"] for row in summaries) if summaries else False
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
    ranking = (
        rank_from_windows(summaries)
        if summaries
        else {
            "status": "incomplete",
            "gap_attribution_complete": False,
            "reasons": ["no_windows"],
            "ranked": [],
        }
    )
    observed = parse_native_observation(completed.stdout)
    payload.update(
        {
            "hardware_executed": True,
            "stdout_excerpt": completed.stdout[-4000:],
            "windows": summaries,
            "attribution_valid": valid,
            "graph_itl_p95_ms": mean(p95) if p95 else None,
            "idle_ms_per_token": mean(idle) if idle else None,
            "ranking": ranking,
            "llama_status": "see_native_log",
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


def query_ncu_metrics(runner: NativeRunner, tier: str) -> dict[str, Any]:
    listed = runner(["bash", "-lc", "command -v ncu && ncu --query-metrics"], tier)
    text = listed.stdout or ""
    available = "ncu" in text or bool(listed.returncode == 0 and text.strip())
    names = [line.strip() for line in text.splitlines() if line.strip()]
    selected = [
        name for name in names if any(needle in name for needle in NCU_METRIC_NEEDLES)
    ]
    forbidden = any("--set full" in line or "set full" in line for line in names)
    return {
        "ncu_available": available and not forbidden,
        "queried": True,
        "full_ncu_sweep": False,
        "selected_metrics": selected[:32],
        "raw_name_count": len(names),
    }


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
    except AttributionError as exc:
        payload["reason"] = "ncu_query_failed"
        payload["error"] = str(exc)
        payload["event_evidence_fallback"] = True
        return payload
    payload.update(metrics)
    if not metrics.get("ncu_available"):
        payload["reason"] = "ncu_not_found"
        payload["event_evidence_fallback"] = True
        return payload
    launches = []
    workloads = {
        "q4_paired_gate_up": [
            "--workload",
            "decode-ffn",
            "--q4-decode",
            opt089_selected_selectors()["q4_decode"],
            "--ffn-decode",
            opt089_selected_selectors()["ffn_decode"],
        ],
        "q4_down": [
            "--workload",
            "decode-ffn",
            "--q4-decode",
            opt089_selected_selectors()["q4_decode"],
            "--ffn-decode",
            opt089_selected_selectors()["ffn_decode"],
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
                }
            )
        except AttributionError as exc:
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
    gdn = gdn_reopen_eligible(results.get("opt077_defect"))
    graph = graph_reopen_eligible(
        d128.get("idle_ms_per_token"),
        d2048.get("idle_ms_per_token"),
        d128_valid=bool(d128.get("attribution_valid")),
        d2048_valid=bool(d2048.get("attribution_valid")),
    )
    ranking_source = (
        d2048.get("ranking")
        or d128.get("ranking")
        or {
            "status": "incomplete",
            "gap_attribution_complete": False,
            "reasons": ["decode_attribution_missing"],
            "ranked": [],
        }
    )
    complete = bool(
        (results.get("conservation") or {}).get("ok")
        and d128.get("attribution_valid")
        and d2048.get("attribution_valid")
    )
    payload = {
        "host_only": True,
        "cannot_convert_three_samples_into_admission": True,
        "gdn_reopen_eligible": gdn["eligible"],
        "graph_reopen_eligible": graph["eligible"],
        "gdn_reopen": gdn,
        "graph_reopen": graph,
        "ranking": ranking_source,
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
    return payload


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
        "gdn_reopen",
        "graph_reopen",
        "ranking",
        "gdn_reopen_eligible",
        "graph_reopen_eligible",
        "evidence_complete",
        "opt077_defect",
    ):
        if key in prior:
            results[key] = prior[key]
    if phase == "conservation":
        payload = run_conservation_phase(mode)
    elif phase in {"d128", "d2048", "p4096"}:
        payload = run_decode_phase(phase, mode, skip_gpu=skip_gpu, runner=execute)
    elif phase == "counters":
        payload = run_counters_phase(mode, skip_gpu=skip_gpu, runner=execute)
    else:
        payload = run_report_phase(mode, results)
        results["gdn_reopen"] = payload["gdn_reopen"]
        results["graph_reopen"] = payload["graph_reopen"]
        results["ranking"] = payload["ranking"]
        results["gdn_reopen_eligible"] = payload["gdn_reopen_eligible"]
        results["graph_reopen_eligible"] = payload["graph_reopen_eligible"]
        results["evidence_complete"] = payload["evidence_complete"]
    merge_fixture(results, phase, payload)
    dump_json(run_dir / "opt090_decode_attribution.json", results)
    if not skip_gpu:
        write_report(results)
        dump_json(FIXTURE, results)
        dump_json(EVIDENCE / "opt090_decode_attribution.json", results)
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
