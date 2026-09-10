from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from test_opt038_post_ladder_gap import (
    DECODE_CATEGORIES,
    DECODE_EVENTS,
    DECODE_PREFIX,
    IMAGE,
    LLAMA_DECODE,
    LLAMA_DECODE_PREFIX,
    LLAMA_IMAGE,
    PREFILL_CATEGORIES,
    PREFILL_EVENTS,
    P_PREFIX,
    _common,
    _nvcc,
    _parse_llama_bench,
    _parse_prefixed,
    _run,
    percentile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt043_component_gap_contract.json"
FIXTURE = ROOT / "fixtures/opt043_component_gap.json"
OPT038 = ROOT / "fixtures/opt038_post_ladder_gap.json"
OPT041 = ROOT / "fixtures/opt041_fattn_warp_qk.json"
EVIDENCE = ROOT / "evidence/optimization/opt043-component-gap"
REPORT = EVIDENCE / "REPORT.md"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
LLAMA_BENCH = ROOT / ".cache/authorities/llama-build/bin/llama-bench"
LLAMA_COMPONENT = ROOT / ".cache/authorities/llama-build/bin/qw38-llama-component-gap"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
PREFILL_LEAF_PREFIX = "QW38_OPT043_PREFILL_ATTRIBUTION_RESULT="
DECODE_LEAF_PREFIX = "QW38_OPT043_DECODE_ATTRIBUTION_RESULT="
CAPTURE_PREFIX = "QW38_OPT043_CAPTURE_RESULT="
LLAMA_COMPONENT_PREFIX = "QW38_OPT043_LLAMA_COMPONENT_RESULT="
KEEP_KEYS = (
    "quartz_p_mean_tok_s",
    "quartz_d128_mean_tok_s",
    "quartz_d2048_mean_tok_s",
    "quartz_d128_token_latency_p95_ms",
    "quartz_d128_run_mean_token_latency_p95_ms",
    "quartz_d2048_token_latency_p95_ms",
    "quartz_d2048_run_mean_token_latency_p95_ms",
)
CAPTURE_LAYERS = (0, 3, 31, 32, 62, 63)
ATTENTION_LAYERS = {layer for layer in CAPTURE_LAYERS if layer % 4 == 3}
SUCCESSORS = (
    "OPT-044",
    "OPT-045",
    "OPT-046",
    "OPT-047",
    "OPT-048",
    "OPT-049",
    "OPT-050",
    "OPT-051",
    "OPT-052",
    "OPT-053",
    "OPT-054",
    "OPT-055",
    "OPT-056",
)
NSIGHT_COUNTERS = (
    "dram_bytes",
    "l2_hit_rate",
    "achieved_bandwidth",
    "instructions",
    "tensor_activity",
    "registers",
    "spills",
    "barrier_stalls",
    "occupancy",
)
PROOF = (
    "claims no performance improvement; D128 and D2048 oracles; "
    "exclusive leaf intervals; matched pinned llama.cpp; "
    "activation captures; Nsight counters marked missing or recorded; "
    "successor decisions; accepted keep denominators remain historical; "
    "does not substitute for the 2K llama.cpp parity gate; "
    "llama-bench random decode is informational"
)
OVERLAPPING_DECODE_LEAVES = ("d2h", "state_copies")


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _opt041_keep() -> dict[str, Any]:
    packed = json.loads(OPT041.read_text())
    return {
        "quartz_p_mean_tok_s": packed["p"]["quartz"]["mean_tok_s"],
        "quartz_d128_mean_tok_s": packed["d128"]["quartz"]["mean_tok_s"],
        "quartz_d2048_mean_tok_s": packed["d2048"]["quartz"]["mean_tok_s"],
        "quartz_d128_token_latency_p95_ms": packed["d128"]["quartz"][
            "token_latency_p95_ms"
        ],
        "quartz_d128_run_mean_token_latency_p95_ms": packed["d128"]["quartz"][
            "run_mean_token_latency_p95_ms"
        ],
        "quartz_d2048_token_latency_p95_ms": packed["d2048"]["quartz"][
            "token_latency_p95_ms"
        ],
        "quartz_d2048_run_mean_token_latency_p95_ms": packed["d2048"]["quartz"][
            "run_mean_token_latency_p95_ms"
        ],
        "source_fixture": "fixtures/opt041_fattn_warp_qk.json",
    }


def _ms(timing: dict[str, Any] | None) -> float:
    if not isinstance(timing, dict) or not timing.get("measured"):
        return 0.0
    return float(timing["ms"])


def _leaf_sum(leaves: dict[str, Any], names: tuple[str, ...]) -> float:
    return sum(_ms(leaves[name]) for name in names)


def _close(left: float, right: float, abs_tol: float, rel_tol: float) -> bool:
    scale = max(abs(left), abs(right))
    return abs(left - right) <= max(abs_tol, rel_tol * scale)


def _tensor_inventory() -> dict[str, Any]:
    inventory = json.loads((ROOT / "pins/tensor_inventory.json").read_text())
    roles: dict[str, dict[str, int]] = {}
    for tensor in inventory["tensors"]:
        role = str(tensor["role"])
        bucket = roles.setdefault(role, {"count": 0, "storage_bytes": 0})
        bucket["count"] += 1
        bucket["storage_bytes"] += int(tensor["storage_bytes"])
    embedding = int(roles.get("token_embedding", {}).get("storage_bytes", 0))
    output = int(roles.get("output_projection", {}).get("storage_bytes", 0))
    layer_bytes = 0
    for role, bucket in roles.items():
        if role in {"token_embedding", "output_projection", "final_norm"}:
            continue
        layer_bytes += int(bucket["storage_bytes"])
    unique = sum(int(bucket["storage_bytes"]) for bucket in roles.values())
    return {
        "source": "pins/tensor_inventory.json",
        "tensor_count": int(inventory["tensor_count"]),
        "unique_weight_bytes": unique,
        "embedding_storage_bytes": embedding,
        "output_projection_storage_bytes": output,
        "layer_weight_bytes": layer_bytes,
        "per_decode_token_weight_bytes": layer_bytes + output,
        "embedding_lookup_separated_from_output": True,
        "roles": roles,
    }


def _llama_family(components: list[dict[str, Any]], family: str) -> dict[str, Any]:
    for component in components:
        if component.get("family") == family:
            return component
    raise AssertionError(f"missing llama family {family}")


def _bound(gap_ms: float, fraction: float, remaining_ms: float) -> float:
    return max(0.0, min(max(0.0, gap_ms) * fraction, max(0.0, remaining_ms)))


def _successor_decisions(result: dict[str, Any]) -> dict[str, Any]:
    eager = result["decode_leaves_d2048_eager_ffn"]
    graph = result["decode_leaves_d2048"]
    prefill = result["prefill_leaves"]
    llama = result["llama_components"]["components"]
    eager_leaves = eager["leaves"]
    graph_leaves = graph["leaves"]
    prefill_leaves = prefill["leaves"]
    p_q = float(result["p"]["quartz"]["mean_tok_s"])
    p_l = float(result["p"]["llama_cpp"]["avg_ts"])
    d_q = float(result["d2048"]["quartz"]["mean_tok_s"])
    d_l = float(result["d2048"]["llama_cpp"]["mean_tok_s"])
    p_remaining = 4096.0 * 1000.0 / p_q - 4096.0 * 1000.0 / p_l
    d_remaining = 1000.0 / d_q - 1000.0 / d_l

    def llama_ms(family: str) -> float:
        component = _llama_family(llama, family)
        return float(component["ms"]) if component.get("ok") else 0.0

    quartz_norm = _leaf_sum(
        eager_leaves,
        ("input_norm", "ffn_norm", "logits_norm", "residual_mixer", "residual_ffn"),
    )
    llama_norm = llama_ms("rms_norm") * 129.0
    quartz_ffn = _leaf_sum(
        eager_leaves, ("proj_ffn_gate", "proj_ffn_up", "proj_ffn_down")
    )
    llama_ffn = (
        llama_ms("q4k_ffn_gate") + llama_ms("q4k_ffn_up") + llama_ms("q4k_ffn_down")
    ) * 64.0
    quartz_mixer = _leaf_sum(
        eager_leaves,
        (
            "proj_packed_qkv",
            "proj_value_gate",
            "proj_alpha",
            "proj_beta",
            "proj_gdn_output",
            "proj_query_gate",
            "proj_key",
            "proj_value",
            "proj_attn_output",
        ),
    )
    llama_mixer = llama_ms("q8_mixer_qkv") * 48.0
    quartz_logits = _ms(eager_leaves["logits_projection"])
    llama_logits = llama_ms("q6k_output")
    quartz_swiglu = _ms(eager_leaves["swiglu"])
    prefill_attn = float(prefill["categories_ms"]["attention_core"])
    prefill_gdn = float(prefill["categories_ms"]["gdn_core"])
    prefill_ffn = float(prefill["categories_ms"]["ffn_mmq"])
    graph_host = float(prefill["categories_ms"]["graph"]) + float(
        graph["categories_ms"]["graph"]
    )
    idle = float(prefill["categories_ms"]["other_idle"]) + float(
        graph["categories_ms"]["other_idle"]
    )
    fused_attn = bool(prefill["fused"]["attention_prep_fused_with_core"])
    fused_gdn = bool(prefill["fused"]["gdn_conv_fused_with_recurrence"])
    fused_ffn = bool(prefill["fused"]["ffn_graph_fused"])
    decisions = {
        "OPT-044": {
            "measured_gap_ms": 0.0,
            "affected_fraction": 1.0,
            "mechanism": (
                "Admit documented production arithmetic and quality budgets "
                "before changing kernels"
            ),
            "recoverable_ms_bound": 0.0,
            "workload": "policy",
            "paired_component": "none_policy",
        },
        "OPT-045": {
            "measured_gap_ms": quartz_norm - llama_norm,
            "affected_fraction": 1.0,
            "mechanism": "Cooperative residual/head RMSNorm and admitted FMA",
            "recoverable_ms_bound": _bound(quartz_norm - llama_norm, 1.0, d_remaining),
            "workload": "d2048",
            "paired_component": "rms_norm",
        },
        "OPT-046": {
            "measured_gap_ms": quartz_ffn - llama_ffn,
            "affected_fraction": 1.0,
            "mechanism": "Cooperative packed Q4_K decode dots on gate/up/down",
            "recoverable_ms_bound": _bound(quartz_ffn - llama_ffn, 1.0, d_remaining),
            "workload": "d2048",
            "paired_component": "q4k_ffn_gate+up+down",
        },
        "OPT-047": {
            "measured_gap_ms": quartz_mixer - llama_mixer,
            "affected_fraction": 48.0 / 64.0,
            "mechanism": "Shared Q8_0 mixer staging and packed integer dots",
            "recoverable_ms_bound": _bound(
                quartz_mixer - llama_mixer, 48.0 / 64.0, d_remaining
            ),
            "workload": "d2048",
            "paired_component": "q8_mixer_qkv",
        },
        "OPT-048": {
            "measured_gap_ms": quartz_logits - llama_logits,
            "affected_fraction": 1.0,
            "mechanism": "Full 248320-row Q6_K packed dots; no vocabulary pruning",
            "recoverable_ms_bound": _bound(
                quartz_logits - llama_logits, 1.0, d_remaining
            ),
            "workload": "d2048",
            "paired_component": "q6k_output",
        },
        "OPT-049": {
            "measured_gap_ms": quartz_swiglu,
            "affected_fraction": 1.0,
            "mechanism": "Share decode FFN staging and fuse gate/up/SwiGLU",
            "recoverable_ms_bound": _bound(quartz_swiglu, 1.0, d_remaining),
            "workload": "d2048",
            "paired_component": "swiglu_enclosing",
        },
        "OPT-050": {
            "measured_gap_ms": prefill_attn,
            "affected_fraction": 0.35 if fused_attn else 1.0,
            "mechanism": (
                "Prepare prompt Q norm/RoPE once; fused leaf "
                "attn_qk_prep_softmax_pv_merge"
            ),
            "recoverable_ms_bound": _bound(
                prefill_attn, 0.35 if fused_attn else 1.0, p_remaining
            ),
            "workload": "prefill_4k",
            "paired_component": (
                "attention_flash_enclosing" if fused_attn else "attention_flash"
            ),
        },
        "OPT-051": {
            "measured_gap_ms": prefill_attn,
            "affected_fraction": 0.65 if fused_attn else 0.0,
            "mechanism": (
                "Pipeline prompt attention with register softmax; same fused "
                "enclosing leaf as OPT-050, not double-counted"
            ),
            "recoverable_ms_bound": _bound(
                prefill_attn, 0.65 if fused_attn else 0.0, p_remaining
            ),
            "workload": "prefill_4k",
            "paired_component": "attention_flash_enclosing",
        },
        "OPT-052": {
            "measured_gap_ms": prefill_gdn,
            "affected_fraction": 1.0,
            "mechanism": (
                "Hoist GDN scaled Q/K and decay; fused leaf gdn_conv_qk_norm_recurrence"
                if fused_gdn
                else "Hoist GDN scaled Q/K and decay"
            ),
            "recoverable_ms_bound": _bound(prefill_gdn, 1.0, p_remaining),
            "workload": "prefill_4k",
            "paired_component": "gdn_fused",
        },
        "OPT-053": {
            "measured_gap_ms": prefill_ffn,
            "affected_fraction": 1.0,
            "mechanism": (
                "MMQ scaling/tile staging inside existing quality MMA; "
                "ffn_graph_fused so enclosing ffn_mmq is the complete component"
                if fused_ffn
                else "MMQ scaling/tile staging"
            ),
            "recoverable_ms_bound": _bound(prefill_ffn, 1.0, p_remaining),
            "workload": "prefill_4k",
            "paired_component": "ffn_mmq_enclosing",
        },
        "OPT-054": {
            "measured_gap_ms": p_remaining,
            "affected_fraction": 1.0,
            "mechanism": "Physical 512/1024/2048/4096 batches inside atomic 4K",
            "recoverable_ms_bound": _bound(p_remaining, 0.25, p_remaining),
            "workload": "prefill_4k",
            "paired_component": "full_prefill_wall",
        },
        "OPT-055": {
            "measured_gap_ms": graph_host + idle,
            "affected_fraction": 1.0,
            "mechanism": "Broader stable graphs for remaining host/GPU launch gaps",
            "recoverable_ms_bound": _bound(
                graph_host + idle, 1.0, d_remaining + p_remaining
            ),
            "workload": "prefill_and_decode",
            "paired_component": "host_graph_submit",
        },
        "OPT-056": {
            "measured_gap_ms": max(p_remaining, 0.0) + max(d_remaining, 0.0),
            "affected_fraction": 1.0,
            "mechanism": "End-to-end P/D128/D2048 outcome gate, not a kernel rewrite",
            "recoverable_ms_bound": 0.0,
            "workload": "full_engine",
            "paired_component": "throughput_oracles",
        },
    }
    _ = graph_leaves
    _ = prefill_leaves
    return decisions


def _ranked_gap(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_id, decision in decisions.items():
        if task_id in {"OPT-044", "OPT-056"}:
            continue
        rows.append({"task": task_id, **decision})
    rows.sort(key=lambda row: float(row["recoverable_ms_bound"]), reverse=True)
    return rows


def _reconstructs(
    categories: dict[str, Any], wall: float, names: tuple[str, ...]
) -> None:
    contract = _contract()
    attributed = sum(float(categories[name]) for name in names)
    assert math.isclose(
        attributed,
        float(wall),
        rel_tol=contract["rel_tol"],
        abs_tol=contract["abs_tol_ms"],
    )


def _raw_wall(
    record: dict[str, Any], event_names: tuple[str, ...], names: tuple[str, ...]
) -> None:
    contract = _contract()
    cats = record["categories_ms"]
    gpu = sum(float(cats[name]) for name in event_names)
    graph = float(cats["graph"])
    attributed = sum(float(cats[name]) for name in names)
    assert math.isclose(
        float(record["gpu_event_sum_ms"]),
        gpu,
        rel_tol=contract["rel_tol"],
        abs_tol=contract["abs_tol_ms"],
    )
    assert math.isclose(
        float(record["graph_host_interval_ms"]),
        graph,
        rel_tol=contract["rel_tol"],
        abs_tol=contract["abs_tol_ms"],
    )
    raw = float(record["raw_host_wall_ms"])
    adjusted = float(record["adjusted_reconstruction_ms"])
    exclusive = gpu + graph
    exclusive_exceeds = exclusive > raw and not math.isclose(
        exclusive, raw, rel_tol=contract["rel_tol"], abs_tol=contract["abs_tol_ms"]
    )
    if exclusive_exceeds:
        assert record["wall_raised"] is True
    else:
        leftover = max(0.0, adjusted - exclusive)
        assert math.isclose(
            float(cats["other_idle"]),
            leftover,
            rel_tol=contract["rel_tol"],
            abs_tol=contract["abs_tol_ms"],
        )
    _reconstructs(cats, adjusted, names)
    assert attributed == pytest.approx(
        float(record["attributed_sum_ms"]), rel=1e-9, abs=1e-9
    )


def _reject_double_counted_leaves(record: dict[str, Any], kind: str) -> None:
    fused = record["fused"]
    leaves = record["leaves"]
    contract = _contract()
    host = _ms(leaves["host_graph_submit"])
    assert host == pytest.approx(
        float(record["categories_ms"]["graph"]),
        rel=contract["rel_tol"],
        abs=contract["abs_tol_ms"],
    )
    exclusive_names = [
        name
        for name, timing in leaves.items()
        if isinstance(timing, dict) and name != "host_graph_submit"
    ]
    if kind == "prefill" and fused.get("d2h_overlaps_state_copies"):
        exclusive_names = [
            name for name in exclusive_names if name not in OVERLAPPING_DECODE_LEAVES
        ]
        assert float(record["categories_ms"]["commit_sync"]) >= 0.0
    gpu_leaves = sum(_ms(leaves[name]) for name in exclusive_names)
    gpu_enclosing = float(record["gpu_event_sum_ms"])
    assert gpu_leaves <= gpu_enclosing + contract["leaf_recon_abs_tol_ms"] or _close(
        gpu_leaves,
        gpu_enclosing,
        contract["leaf_recon_abs_tol_ms"],
        contract["leaf_recon_rel_tol"],
    )
    if fused.get("ffn_graph_fused"):
        for name in ("proj_ffn_gate", "proj_ffn_up", "swiglu", "proj_ffn_down"):
            timing = leaves.get(name)
            if isinstance(timing, dict) and timing.get("measured"):
                assert float(timing["ms"]) == 0.0


def _engine_block(block: dict[str, Any], prefix: int) -> None:
    assert block["warmups"] == 3
    assert block["runs"] == 30
    assert block["decode_tokens"] == 256
    assert block["prefix"] == prefix
    assert len(block["tok_s"]) == 30
    mean = sum(float(v) for v in block["tok_s"]) / 30.0
    assert block["mean_tok_s"] == pytest.approx(mean, rel=1e-6, abs=1e-6)
    sidecar = ROOT / block["token_latency_sidecar"]
    assert sidecar.is_file()
    latencies = json.loads(sidecar.read_text())
    assert len(latencies) == 30 * 256
    assert block["token_latency_p95_ms"] == pytest.approx(
        percentile(latencies, 0.95), rel=1e-5, abs=1e-5
    )


def _validate_capture(capture: dict[str, Any], stage: str) -> None:
    assert capture["stage"] == stage
    assert capture["unfused"] is True
    assert capture["graphs_created"] is False
    assert len(capture["slots"]) == 6
    final_sha = capture["final_norm_sha256"]
    assert len(final_sha) == 64
    assert capture["output_count"] == 248320
    assert len(capture["output_sha256"]) == 64
    seen: set[int] = set()
    for slot in capture["slots"]:
        layer = int(slot["layer"])
        seen.add(layer)
        assert slot["mixer_captured"] is True
        assert slot["ffn_captured"] is True
        assert slot["mixer_dtype"] == "BF16"
        assert slot["ffn_dtype"] == "BF16"
        assert slot["mixer_sha256"] != final_sha
        assert slot["ffn_sha256"] != final_sha
        assert slot["mixer_sha256"] != slot["ffn_sha256"]
        expected_kind = "attention" if layer in ATTENTION_LAYERS else "gdn"
        assert slot["layer_kind"] == expected_kind
        assert "ffn" not in slot["layer_kind"]
        assert slot["mixer_shape"] == [1, 5120]
        assert slot["ffn_shape"] == [1, 5120]
    assert seen == set(CAPTURE_LAYERS)


def _validate_llama(components: dict[str, Any]) -> None:
    assert components["q8_layout"] == "llama_block_q8_1_not_quartz_Q8Block"
    assert "Quartz BF16" in components["numeric_label"]
    families = {row["family"] for row in components["components"]}
    required = {
        "q4k_ffn_gate",
        "q4k_ffn_up",
        "q4k_ffn_down",
        "q8_mixer_qkv",
        "q6k_output",
        "rms_norm",
        "gdn_fused",
        "attention_flash",
    }
    assert required <= families
    for row in components["components"]:
        if row["ok"]:
            assert float(row["ms"]) > 0.0
        if row["family"] in {
            "q4k_ffn_gate",
            "q4k_ffn_up",
            "q4k_ffn_down",
            "q8_mixer_qkv",
            "q6k_output",
            "rms_norm",
        }:
            assert row["ok"] is True
            assert row["identical_input"] is True
        if not row["ok"]:
            assert row["error"]
            assert row["family"] in {"gdn_fused", "attention_flash"}


def validate_result(result: Any) -> None:
    contract = _contract()
    keep = _opt041_keep()
    assert isinstance(result, dict)
    assert set(result) == set(contract["required_fixture_keys"])
    assert result["schema_version"] == 1 and result["task"] == "OPT-043"
    assert result["status"] == "measured"
    assert contract["claims_performance_improvement"] is False
    assert result["claims_performance_improvement"] is False
    assert result["publishes_successor_oracle"] is False
    assert result["llama_revision"] == LLAMA_REV == contract["llama_revision"]
    assert result["gguf_sha256"] == GGUF_SHA == contract["gguf_sha256"]
    assert contract["device_substring"] in result["device"]
    assert result["compute_capability"] == "12.0"
    assert result["owns_opt016_parity_gate"] is False
    assert result["substitutes_for_opt016"] is False
    assert result["historical_opt038_fixture"] == "fixtures/opt038_post_ladder_gap.json"
    accepted = result["accepted_keep_denominators"]
    for key in KEEP_KEYS:
        assert accepted[key] == keep[key]
        assert accepted[key] == contract["accepted_keep_denominators"][key]
    assert accepted["source_fixture"] == "fixtures/opt041_fattn_warp_qk.json"

    quartz_p = result["p"]["quartz"]
    assert quartz_p["prompt_tokens"] == 4096
    assert quartz_p["replicates"] == 3
    assert quartz_p["attribution"] is None
    assert quartz_p["graphs_created"] is True
    assert quartz_p["cold"] is True
    mean_p = sum(float(v) for v in quartz_p["tok_s"]) / 3.0
    assert quartz_p["mean_tok_s"] == pytest.approx(mean_p, rel=1e-6, abs=1e-6)

    for prefix, key in ((128, "d128"), (2048, "d2048")):
        block = result[key]
        assert block["quartz"]["graphs_created"] is True
        assert block["quartz"]["attribution"] is None
        _engine_block(block["quartz"], prefix)
        _engine_block(block["llama_cpp"], prefix)

    prefill = result["prefill_leaves"]
    assert prefill["record_leaves"] is True
    assert set(prefill["categories_ms"]) == set(PREFILL_CATEGORIES)
    _raw_wall(prefill, PREFILL_EVENTS, PREFILL_CATEGORIES)
    _reject_double_counted_leaves(prefill, "prefill")
    assert prefill["fused"]["ffn_graph_fused"] is True
    assert prefill["fused"]["gdn_conv_fused_with_recurrence"] is True
    assert prefill["fused"]["attention_prep_fused_with_core"] is True

    for key, prefix, eager in (
        ("decode_leaves_d128", 128, False),
        ("decode_leaves_d2048", 2048, False),
        ("decode_leaves_d2048_eager_ffn", 2048, True),
    ):
        decoded = result[key]
        assert decoded["prefix"] == prefix
        assert decoded["eager_ffn"] is eager
        assert set(decoded["categories_ms"]) == set(DECODE_CATEGORIES)
        _raw_wall(decoded, DECODE_EVENTS, DECODE_CATEGORIES)
        _reject_double_counted_leaves(decoded, "decode")
        if eager:
            assert decoded["fused"]["ffn_graph_fused"] is False
            assert _ms(decoded["leaves"]["proj_ffn_gate"]) > 0.0
            assert _ms(decoded["leaves"]["swiglu"]) > 0.0
        else:
            assert decoded["graphs_created"] is True

    for stage in contract["capture_stages"]:
        _validate_capture(result["captures"][stage], stage)
    _validate_llama(result["llama_components"])

    inventory = result["tensor_inventory"]
    expected_inventory = _tensor_inventory()
    assert inventory["unique_weight_bytes"] == expected_inventory["unique_weight_bytes"]
    assert inventory["embedding_lookup_separated_from_output"] is True
    assert (
        inventory["output_projection_storage_bytes"]
        != inventory["embedding_storage_bytes"]
    )

    p_gap = float(result["p"]["llama_cpp"]["avg_ts"]) / float(
        result["p"]["quartz"]["mean_tok_s"]
    )
    d_gap = float(result["d2048"]["llama_cpp"]["mean_tok_s"]) / float(
        result["d2048"]["quartz"]["mean_tok_s"]
    )
    assert result["p_gap"] == pytest.approx(p_gap, rel=1e-9, abs=1e-9)
    assert result["d2048_gap"] == pytest.approx(d_gap, rel=1e-9, abs=1e-9)
    assert result["decode_deficit_larger"] is (d_gap > p_gap)

    decisions = result["successor_decisions"]
    assert set(decisions) == set(SUCCESSORS)
    expected = _successor_decisions(result)
    for task_id in SUCCESSORS:
        for field in (
            "measured_gap_ms",
            "affected_fraction",
            "mechanism",
            "recoverable_ms_bound",
        ):
            left = decisions[task_id][field]
            right = expected[task_id][field]
            if field == "mechanism":
                assert left == right
            else:
                assert left == pytest.approx(right, rel=1e-6, abs=1e-6)
    ranked = result["ranked_gap"]
    assert [row["task"] for row in ranked] == [
        row["task"] for row in _ranked_gap(expected)
    ]

    nsight_s = result["nsight_systems"]
    nsight_c = result["nsight_compute"]
    for blob in (nsight_s, nsight_c):
        assert blob.get("bandwidth_or_compute_claim") is False
        counters = blob["counters"]
        for name in NSIGHT_COUNTERS:
            assert name in counters
            value = counters[name]
            assert value == "unavailable" or isinstance(value, (int, float, str))
        if blob.get("available") is not True:
            assert blob.get("error")

    assert result["proof_limit"] == PROOF
    assert result["report_path"] == contract["report_path"]
    report = (ROOT / result["report_path"]).read_text()
    for phrase in contract["proof_limit"]:
        assert phrase in report
    assert "in progress" in report.lower() or "measurement-only" in report.lower()


def test_opt043_contract_and_source_pins() -> None:
    contract = _contract()
    assert contract["task"] == "OPT-043"
    assert contract["claims_performance_improvement"] is False
    assert contract["attribution_on_throughput"] == "null"
    assert contract["q8_layout"] == "llama_block_q8_1_not_quartz_Q8Block"
    cmake = (ROOT / "tools/llama_authority/CMakeLists.txt").read_text()
    assert "qw38-llama-component-gap" in cmake
    assert "qw38-llama-decode-oracle" in cmake
    assert "component_gap.cpp" in cmake
    scheduler = (ROOT / "cuda/full_scheduler.h").read_text()
    assert "struct LeafTimings" in scheduler
    assert "record_leaves" in scheduler
    opt038 = json.loads(OPT038.read_text())
    assert opt038["task"] == "OPT-038"
    assert opt038["accepted_keep_denominators"]["source_fixture"] == (
        "fixtures/opt034_packed_mmv.json"
    )
    makefile = (ROOT / "Makefile").read_text()
    assert "opt043" not in makefile


def test_opt043_opt038_history_untouched() -> None:
    opt038 = json.loads(OPT038.read_text())
    assert opt038["task"] == "OPT-038"
    assert opt038["claims_performance_improvement"] is False
    assert "leaves" not in opt038["prefill_attribution"]


def test_opt043_validator_rejects_inadmissible_evidence() -> None:
    result = json.loads(FIXTURE.read_text())
    if result.get("status") != "measured":
        pytest.skip("native sitting has not written the measured fixture")
    mutations: list[dict[str, Any]] = []

    def add(mutate: Any) -> None:
        changed = copy.deepcopy(result)
        mutate(changed)
        mutations.append(changed)

    add(lambda x: x.__setitem__("claims_performance_improvement", True))
    add(lambda x: x["p"]["quartz"].__setitem__("attribution", "leaves"))
    add(lambda x: x["d2048"]["quartz"].__setitem__("attribution", "callback"))
    add(lambda x: x["captures"]["d128"]["slots"][0].__setitem__("layer_kind", "ffn"))
    add(
        lambda x: x["captures"]["d128"]["slots"][0].__setitem__(
            "mixer_sha256", x["captures"]["d128"]["final_norm_sha256"]
        )
    )
    add(lambda x: x["llama_components"]["components"].pop(0))
    add(lambda x: x["successor_decisions"].pop("OPT-046"))
    add(lambda x: x.__setitem__("report_path", "/tmp/opt043/REPORT.md"))
    add(
        lambda x: x["prefill_leaves"]["leaves"].__setitem__(
            "host_graph_submit",
            {
                "ms": float(x["prefill_leaves"]["categories_ms"]["graph"]) + 25.0,
                "measured": True,
            },
        )
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_opt043_fixture_connected() -> None:
    result = json.loads(FIXTURE.read_text())
    if result.get("status") != "measured":
        pytest.skip("native sitting has not written the measured fixture")
    validate_result(result)
    report = REPORT.read_text()
    assert "claims no performance improvement" in report
    assert "successor decisions" in report


def _ensure_quartz_objects() -> None:
    _run([*_common(IMAGE), "make", "build/qw38-cuda-timing-test"])


def _ensure_llama_tools() -> None:
    if not LLAMA_BENCH.is_file():
        configure = [
            *_common(LLAMA_IMAGE),
            "cmake",
            "-S",
            "/workspace/.cache/authorities/llama.cpp",
            "-B",
            "/workspace/.cache/authorities/llama-build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=120",
            "-DGGML_CUDA=ON",
            "-DGGML_NATIVE=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
        ]
        build = [
            *_common(LLAMA_IMAGE),
            "cmake",
            "--build",
            "/workspace/.cache/authorities/llama-build",
            "--target",
            "llama-bench",
            "-j",
            "6",
        ]
        _run(configure)
        _run(build)
    configure_adapter = [
        *_common(LLAMA_IMAGE),
        "cmake",
        "-S",
        "/workspace/tools/llama_authority",
        "-B",
        "/workspace/.cache/authorities/llama-adapter-build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_SOURCE=/workspace/.cache/authorities/llama.cpp",
        "-DLLAMA_BUILD=/workspace/.cache/authorities/llama-build",
    ]
    targets = ["qw38-llama-component-gap"]
    if not LLAMA_DECODE.is_file():
        targets.append("qw38-llama-decode-oracle")
    _run(configure_adapter)
    for target in targets:
        _run(
            [
                *_common(LLAMA_IMAGE),
                "cmake",
                "--build",
                "/workspace/.cache/authorities/llama-adapter-build",
                "--target",
                target,
                "-j",
                "6",
            ]
        )
    assert LLAMA_DECODE.is_file()
    assert LLAMA_COMPONENT.is_file()


def _write_json(name: str, record: dict[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(record, indent=2) + "\n")


def _run_prefixed(
    commands: list[list[str]], prefix: str, sidecar: str
) -> dict[str, Any]:
    outputs: list[str] = []
    for command in commands:
        outputs.append(_run(command).stdout)
    record = _parse_prefixed(outputs[-1], prefix)
    assert "status=passed" in outputs[-1]
    _write_json(sidecar, record)
    return record


def _run_quartz_p() -> dict[str, Any]:
    return _run_prefixed(
        [
            _nvcc(
                "cuda/prefill_4k_oracle_test.cu",
                "build/qw38-cuda-prefill-4k-oracle-test",
            ),
            [
                *_common(IMAGE),
                "./build/qw38-cuda-prefill-4k-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ],
        ],
        P_PREFIX,
        "quartz-p.json",
    )


def _run_quartz_decode(prefix: int) -> dict[str, Any]:
    binary = "build/qw38-cuda-decode-oracle-test"
    return _run_prefixed(
        [
            _nvcc("cuda/decode_oracle_test.cu", binary),
            [
                *_common(IMAGE),
                f"./{binary}",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                str(prefix),
            ],
        ],
        DECODE_PREFIX,
        f"quartz-d{prefix}.json",
    )


def _run_llama_bench_p() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        "-p 4096 -n 0 --no-warmup -r 3 -ngl 99 -o json",
    ]
    completed = _run(command)
    payload = _parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: row.get("n_prompt") == 4096,
        "llama-bench JSON with n_prompt 4096",
    )
    _write_json("llama-bench-4k.json", payload)
    return payload[0]


def _run_llama_bench_decode(depth: int) -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        "-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        f"-p 0 -n 256 -d {depth} --no-warmup -r 30 -ngl 99 -o json",
    ]
    completed = _run(command)
    payload = _parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: True,
        f"llama-bench JSON for depth {depth}",
    )
    _write_json(f"llama-bench-d{depth}.json", payload)
    return payload[0]


def _run_llama_decode(prefix: int) -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/qw38-llama-decode-oracle "
        f"/workspace/models/Qwen3.8-27B-Q4_K_M.gguf {prefix}",
    ]
    completed = _run(command)
    record = _parse_prefixed(completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX)
    _write_json(f"llama-decode-d{prefix}.json", record)
    return record


def _run_prefill_leaves() -> dict[str, Any]:
    binary = "build/qw38-cuda-opt043-prefill-attribution-test"
    return _run_prefixed(
        [
            _nvcc("cuda/opt043_prefill_attribution_test.cu", binary),
            [*_common(IMAGE), f"./{binary}", "models/Qwen3.8-27B-Q4_K_M.gguf"],
        ],
        PREFILL_LEAF_PREFIX,
        "quartz-prefill-leaves-4k.json",
    )


def _run_decode_leaves(prefix: int, eager: bool) -> dict[str, Any]:
    binary = "build/qw38-cuda-opt043-decode-attribution-test"
    command = [
        *_common(IMAGE),
        f"./{binary}",
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        str(prefix),
    ]
    if eager:
        command.append("--eager-ffn")
    sidecar = (
        f"quartz-decode-leaves-d{prefix}-eager.json"
        if eager
        else f"quartz-decode-leaves-d{prefix}.json"
    )
    return _run_prefixed(
        [_nvcc("cuda/opt043_decode_attribution_test.cu", binary), command],
        DECODE_LEAF_PREFIX,
        sidecar,
    )


def _write_real_text_tokens() -> str:
    quality = json.loads((ROOT / "fixtures/quality_inputs.json").read_text())
    tokens = quality["cases"]["continuation_2048"]["context"]
    relative = "evidence/optimization/opt043-component-gap/real-text-tokens.txt"
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(token) for token in tokens) + "\n")
    return relative


def _run_capture(stage: str, token_file: str | None = None) -> dict[str, Any]:
    binary = "build/qw38-cuda-opt043-activation-capture-test"
    command = [
        *_common(IMAGE),
        f"./{binary}",
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        stage,
    ]
    if token_file is not None:
        command.append(token_file)
    return _run_prefixed(
        [_nvcc("cuda/opt043_activation_capture_test.cu", binary), command],
        CAPTURE_PREFIX,
        f"capture-{stage}.json",
    )


def _run_llama_components() -> dict[str, Any]:
    command = [
        *_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/qw38-llama-component-gap "
        "/workspace/models/Qwen3.8-27B-Q4_K_M.gguf",
    ]
    completed = _run(command)
    record = _parse_prefixed(
        completed.stdout + completed.stderr, LLAMA_COMPONENT_PREFIX
    )
    _write_json("llama-components.json", record)
    return record


def _strip_cuda_banner(text: str) -> str:
    ignored = (
        "CUDA Version",
        "NVIDIA CORPORATION",
        "Deep Learning Container License",
        "NGC-DL-CONTAINER-LICENSE",
        "developer.nvidia.com",
        "By pulling and using",
        "A copy of this license",
        "Container image Copyright",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=="):
            continue
        if any(token in line for token in ignored):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _probe_tool(image: str, tool: str) -> dict[str, Any]:
    counters = {name: "unavailable" for name in NSIGHT_COUNTERS}
    command = [*_common(image), "bash", "-lc", f"command -v {tool} && {tool} --version"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    cleaned = _strip_cuda_banner(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        error = cleaned or f"{tool} not found in {image}"
        _write_json(
            f"{tool}-error.json", {"error": error, "returncode": completed.returncode}
        )
        return {
            "available": False,
            "error": error,
            "counters": counters,
            "bandwidth_or_compute_claim": False,
        }
    version = cleaned.splitlines()[0] if cleaned else f"{tool} present"
    return {
        "available": True,
        "error": "",
        "version": version,
        "counters": counters,
        "bandwidth_or_compute_claim": False,
        "note": "version present; targeted kernel counters were not collected in this sitting",
    }


def _write_token_sidecar(name: str, latencies: list[Any]) -> str:
    path = EVIDENCE / name
    path.write_text(json.dumps(latencies) + "\n")
    return f"evidence/optimization/opt043-component-gap/{name}"


def _engine_from_live(record: dict[str, Any], sidecar_name: str) -> dict[str, Any]:
    sidecar = _write_token_sidecar(sidecar_name, record["token_latency_ms"])
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
        "token_latency_sidecar": sidecar,
    }
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    if "graphs_created" in record:
        block["graphs_created"] = record["graphs_created"]
        block["attribution"] = record.get("attribution")
        block["cache_policy"] = record.get("cache_policy", "disabled")
    return block


def _leaf_from_live(record: dict[str, Any], task_label: str) -> dict[str, Any]:
    cats = record["categories_ms"]
    if "mixer_mmq" in cats:
        events, names = PREFILL_EVENTS, PREFILL_CATEGORIES
    else:
        events, names = DECODE_EVENTS, DECODE_CATEGORIES
    gpu = sum(float(cats[name]) for name in events)
    graph = float(cats["graph"])
    attributed = sum(float(cats[name]) for name in names)
    payload = {
        "task": task_label,
        "categories_ms": cats,
        "leaves": record["leaves"],
        "fused": record["fused"],
        "record_leaves": True,
        "attributed_sum_ms": attributed,
        "wall_ms": record["wall_ms"],
        "raw_host_wall_ms": record["raw_host_wall_ms"],
        "gpu_event_sum_ms": gpu,
        "graph_host_interval_ms": graph,
        "adjusted_reconstruction_ms": record["adjusted_reconstruction_ms"],
        "wall_raised": record["wall_raised"],
        "eager_ffn": record.get("eager_ffn", False),
        "graphs_created": record.get("graphs_created", True),
    }
    if "prompt_tokens" in record:
        payload["prompt_tokens"] = record["prompt_tokens"]
        payload["prompt_graph_launches"] = record["prompt_graph_launches"]
    if "prefix" in record:
        payload["prefix"] = record["prefix"]
    return payload


def _write_report(fixture: dict[str, Any]) -> None:
    decisions = fixture["successor_decisions"]
    rows = "\n".join(
        f"| {task} | {decisions[task]['measured_gap_ms']:.6g} | "
        f"{decisions[task]['affected_fraction']:.6g} | "
        f"{decisions[task]['mechanism']} | "
        f"{decisions[task]['recoverable_ms_bound']:.6g} |"
        for task in SUCCESSORS
    )
    ranked = ", ".join(row["task"] for row in fixture["ranked_gap"])
    nsight_s = fixture["nsight_systems"]
    nsight_c = fixture["nsight_compute"]
    text = f"""# OPT-043 — Measure the actual post-042 component gap

## Claim labels and proof limits

This increment **claims no performance improvement**. It retains **D128 and D2048 oracles**
with **exclusive leaf intervals**, **matched pinned llama.cpp** component timings,
**activation captures**, **Nsight counters marked missing or recorded**, and
**successor decisions**. **accepted keep denominators remain historical**.
This protocol **does not substitute for the 2K llama.cpp parity gate**.
**llama-bench random decode is informational**.

Measurement-only / in progress diagnostic. Production dispatch is unchanged.
Live numbers in `fixtures/opt043_component_gap.json` are the Measured same-sitting
exclusive RTX 5090 record. Accepted keep denominators remain the OPT-041 copies.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `{GGUF_SHA}` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `{LLAMA_REV}` |
| Throughput attribution | null; graphs created |
| Capture layers | 0, 3, 31, 32, 62, 63 |
| Q8 layout | llama `block_q8_1`, never Quartz `Q8Block` bytes |
| Numeric label | Quartz BF16 vs llama F32/Q8_1 activations; Quartz BF16 KV vs llama F16 KV |

## Measured sitting

- Device: {fixture["device"]} compute {fixture["compute_capability"]}
- measurement_utc: {fixture["measurement_utc"]}
- P Quartz mean tok/s: {fixture["p"]["quartz"]["mean_tok_s"]} ; llama.cpp avg_ts: {fixture["p"]["llama_cpp"]["avg_ts"]}
- D128 Quartz mean tok/s: {fixture["d128"]["quartz"]["mean_tok_s"]} ; llama.cpp mean tok/s: {fixture["d128"]["llama_cpp"]["mean_tok_s"]}
- D2048 Quartz mean tok/s: {fixture["d2048"]["quartz"]["mean_tok_s"]} ; llama.cpp mean tok/s: {fixture["d2048"]["llama_cpp"]["mean_tok_s"]}
- p_gap: {fixture["p_gap"]} ; d2048_gap: {fixture["d2048_gap"]} ; decode_deficit_larger: {json.dumps(fixture["decode_deficit_larger"])}
- ranked_gap: {ranked}
- claims_performance_improvement: false
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

## Exclusive leaf reconstruction

Prefill 4K: raw_host_wall_ms {fixture["prefill_leaves"]["raw_host_wall_ms"]}; gpu_event_sum_ms {fixture["prefill_leaves"]["gpu_event_sum_ms"]}; fused {json.dumps(fixture["prefill_leaves"]["fused"])}

Decode D2048 graph: raw_host_wall_ms {fixture["decode_leaves_d2048"]["raw_host_wall_ms"]}; eager FFN isolates gate/up/SwiGLU/down.

Overlapping D2H/state copies are labeled and excluded from exclusive GPU reconstruction.

## Nsight

Systems available={nsight_s.get("available")} error={nsight_s.get("error", "")}
Compute available={nsight_c.get("available")} error={nsight_c.get("error", "")}
Missing counters remain `unavailable`. This report does not claim bandwidth-bound or compute-bound from occupancy.

## Successor decisions

| Task | measured gap ms | affected fraction | mechanism | recoverable ms bound |
|---|---:|---:|---|---:|
{rows}
"""
    REPORT.write_text(text)


def _build_fixture(
    llama_p: dict[str, Any],
    quartz_p: dict[str, Any],
    llama_d128: dict[str, Any],
    quartz_d128: dict[str, Any],
    llama_d2048: dict[str, Any],
    quartz_d2048: dict[str, Any],
    bench_d128: dict[str, Any],
    bench_d2048: dict[str, Any],
    prefill_leaves: dict[str, Any],
    decode_d128: dict[str, Any],
    decode_d2048: dict[str, Any],
    decode_eager: dict[str, Any],
    captures: dict[str, Any],
    llama_components: dict[str, Any],
    nsight_systems: dict[str, Any],
    nsight_compute: dict[str, Any],
) -> dict[str, Any]:
    p_gap = float(llama_p["avg_ts"]) / float(quartz_p["mean_tok_s"])
    d_gap = float(llama_d2048["mean_tok_s"]) / float(quartz_d2048["mean_tok_s"])
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-043",
        "status": "measured",
        "measurement_utc": quartz_p["measurement_utc"],
        "device": quartz_p["device"],
        "compute_capability": quartz_p["compute_capability"],
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "claims_performance_improvement": False,
        "publishes_successor_oracle": False,
        "accepted_keep_denominators": _opt041_keep(),
        "historical_opt038_fixture": "fixtures/opt038_post_ladder_gap.json",
        "p": {
            "quartz": {
                "prompt_tokens": 4096,
                "replicates": 3,
                "wall_ms": quartz_p["wall_ms"],
                "tok_s": quartz_p["tok_s"],
                "mean_tok_s": float(quartz_p["mean_tok_s"]),
                "cold": True,
                "cache_policy": "disabled",
                "attribution": None,
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
                "test_time": llama_p.get("test_time", quartz_p["measurement_utc"]),
            },
        },
        "d128": {
            "quartz": _engine_from_live(quartz_d128, "quartz-d128-tokens.json"),
            "llama_cpp": _engine_from_live(llama_d128, "llama-decode-d128-tokens.json"),
        },
        "d2048": {
            "quartz": _engine_from_live(quartz_d2048, "quartz-d2048-tokens.json"),
            "llama_cpp": _engine_from_live(
                llama_d2048, "llama-decode-d2048-tokens.json"
            ),
        },
        "prefill_leaves": _leaf_from_live(prefill_leaves, "OPT-043"),
        "decode_leaves_d128": _leaf_from_live(decode_d128, "OPT-043"),
        "decode_leaves_d2048": _leaf_from_live(decode_d2048, "OPT-043"),
        "decode_leaves_d2048_eager_ffn": _leaf_from_live(decode_eager, "OPT-043"),
        "captures": captures,
        "llama_components": llama_components,
        "tensor_inventory": _tensor_inventory(),
        "p_gap": p_gap,
        "d2048_gap": d_gap,
        "decode_deficit_larger": d_gap > p_gap,
        "owns_opt016_parity_gate": False,
        "substitutes_for_opt016": False,
        "llama_bench_informational": {"d128": bench_d128, "d2048": bench_d2048},
        "nsight_systems": nsight_systems,
        "nsight_compute": nsight_compute,
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt043-component-gap/REPORT.md",
    }
    fixture["successor_decisions"] = _successor_decisions(fixture)
    fixture["ranked_gap"] = _ranked_gap(fixture["successor_decisions"])
    return fixture


def test_opt043_native_component_gap() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _ensure_llama_tools()
    _ensure_quartz_objects()

    llama_p = _run_llama_bench_p()
    quartz_p = _run_quartz_p()
    llama_d128 = _run_llama_decode(128)
    quartz_d128 = _run_quartz_decode(128)
    llama_d2048 = _run_llama_decode(2048)
    quartz_d2048 = _run_quartz_decode(2048)
    bench_d128 = _run_llama_bench_decode(128)
    bench_d2048 = _run_llama_bench_decode(2048)
    prefill_leaves = _run_prefill_leaves()
    decode_d128 = _run_decode_leaves(128, eager=False)
    decode_d2048 = _run_decode_leaves(2048, eager=False)
    decode_eager = _run_decode_leaves(2048, eager=True)
    real_text = _write_real_text_tokens()
    captures = {
        "d128": _run_capture("d128"),
        "d2048": _run_capture("d2048"),
        "real_text": _run_capture("real_text", real_text),
    }
    llama_components = _run_llama_components()
    nsight_systems = _probe_tool(IMAGE, "nsys")
    nsight_compute = _probe_tool(IMAGE, "ncu")

    fixture = _build_fixture(
        llama_p,
        quartz_p,
        llama_d128,
        quartz_d128,
        llama_d2048,
        quartz_d2048,
        bench_d128,
        bench_d2048,
        prefill_leaves,
        decode_d128,
        decode_d2048,
        decode_eager,
        captures,
        llama_components,
        nsight_systems,
        nsight_compute,
    )
    _write_report(fixture)
    validate_result(fixture)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
