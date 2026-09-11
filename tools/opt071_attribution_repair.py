"""Host helpers for OPT-071 attribution window and capture-replay repair."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.opt060_engine_attribution import (
    stream_aware_totals,
)

ROOT = Path(__file__).resolve().parents[1]
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt071_attribution_repair_contract.json"
ITERATION = ROOT / "pins/opt071_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt071_attribution_repair.json"
REPORT = ROOT / "evidence/optimization/opt071-attribution-repair/REPORT.md"
OVERLAY = ROOT / "tools/llama_authority/patches/qw38_opt060_attribution.cuh"
ADAPTER = ROOT / "tools/llama_authority/engine_attribution.cpp"
NATIVE_OPT060 = ROOT / "cuda/opt060_engine_attribution_test.cu"
NATIVE_REPLAY = ROOT / "cuda/optimization_component_replay.cu"
HEADER = ROOT / "cuda/engine_attribution.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
CAPTURE_DIR = ROOT / "evidence/optimization/opt071-attribution-repair/captures"

DECODE_GATE_UP_PAIRS = 64
DECODE_DOWNS = 64
DECODE_GDN_CORES = 48
DECODE_ATTENTION_CORES = 16
DECODE_LOGITS = 1
UNEXPLAINED_LIMIT = 0.05

CANONICAL_FAMILIES = (
    "embedding",
    "ffn_gate_up_glu",
    "ffn_down",
    "gdn_core",
    "attention_core",
    "logits_projection",
    "activation_quant",
    "copy",
    "d2h",
    "state_commit",
    "cuda_graph_launch",
    "cpu_unknown_gap",
)

FFN_LEAVES = frozenset(
    {
        "ffn_norm",
        "activation_quant_ffn",
        "ffn_gate",
        "ffn_up",
        "ffn_glu",
        "ffn_down",
        "residual_ffn",
        "ffn_gate_up_glu",
        "proj_ffn_gate",
        "proj_ffn_up",
        "proj_ffn_down",
    }
)
FFN_ENCLOSING = frozenset({"ffn_mmv", "ffn_mmq", "ffn_gate_up_glu"})
GDN_LEAVES = frozenset(
    {
        "gdn_gate_prep",
        "gdn_conv_qk_norm_recurrence",
        "gdn_output_norm",
        "proj_packed_qkv",
        "proj_value_gate",
        "proj_alpha",
        "proj_beta",
        "proj_gdn_output",
        "gdn_conv",
        "gdn_qk_norm",
        "gdn_recurrence",
    }
)
ATTENTION_LEAVES = frozenset(
    {
        "attn_query_split",
        "attn_qk_prep_softmax_pv_merge",
        "attn_output_cast",
        "proj_query_gate",
        "proj_key",
        "proj_value",
        "proj_attn_output",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_out",
    }
)
MIXER_WEIGHT_MARKERS = (
    "attn_qkv",
    "attn_gate",
    "ssm_alpha",
    "ssm_beta",
    "ssm_out",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "attn_output",
    "packed_qkv",
    "value_gate",
)
FFN_WEIGHT_MARKERS = ("ffn_gate", "ffn_up", "ffn_down")

EXPECTED_DECODE_COUNTS = {
    "ffn_gate_up_glu": DECODE_GATE_UP_PAIRS,
    "ffn_down": DECODE_DOWNS,
    "gdn_core": DECODE_GDN_CORES,
    "attention_core": DECODE_ATTENTION_CORES,
    "logits_projection": DECODE_LOGITS,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def token_input_hash(tokens: Sequence[int]) -> str:
    packed = b"".join(
        int(token).to_bytes(4, "little", signed=False) for token in tokens
    )
    return sha256_bytes(packed)


def capture_identity(
    *,
    gguf_sha: str,
    token_generator: str,
    token_input_hash_hex: str,
    stage: str,
    source: str,
    build_flags: str,
    selectors: Mapping[str, Any],
    layer_role: str,
    staging: str,
    state: str,
    model_path: str,
    prompt_rows: int,
) -> str:
    payload = {
        "build_flags": build_flags,
        "gguf_sha": gguf_sha,
        "layer_role": layer_role,
        "model_path": model_path,
        "prompt_rows": int(prompt_rows),
        "selectors": dict(selectors),
        "source": source,
        "staging": staging,
        "stage": stage,
        "state": state,
        "token_generator": token_generator,
        "token_input_hash": token_input_hash_hex,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)


def _member_names(record: Mapping[str, Any]) -> set[str]:
    raw = str(record.get("fused_member_ids", "") or "")
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def canonical_family(record: Mapping[str, Any]) -> str:
    role = str(record.get("role", "") or "")
    name = str(record.get("tensor_name", "") or "")
    launch = str(record.get("launch_family", "") or "")
    combined = f"{role} {name} {launch}".lower()
    if role in {"d2h", "copy"} or "d2h" in combined:
        return "d2h" if "d2h" in combined or role == "d2h" else "copy"
    if role in {"state_commit", "state_copies", "commit_sync"}:
        return "state_commit"
    if role in {"logits_projection", "logits"}:
        return "logits_projection"
    if role == "logits_norm":
        return "logits_norm"
    if name in {"output", "output.weight"} or "output.weight" in combined:
        return "logits_projection"
    if role in FFN_ENCLOSING:
        members = _member_names(record)
        if role == "ffn_gate_up_glu" or "ffn_gate" in members:
            return "ffn_gate_up_glu"
        if "ffn_down" in members and "ffn_gate" not in members:
            return "ffn_down"
        return "ffn_gate_up_glu"
    if role in {"ffn_down", "proj_ffn_down"}:
        return "ffn_down"
    if (
        role in {"gdn_core"}
        or role in GDN_LEAVES
        and role == "gdn_conv_qk_norm_recurrence"
    ):
        return "gdn_core"
    if role == "gdn_core" or "gdn_core" in combined:
        return "gdn_core"
    if role in {"attention_core"} or role == "attn_qk_prep_softmax_pv_merge":
        return "attention_core"
    if role == "embedding" or "token_embd" in combined:
        return "embedding"
    if role in {"activation_quant", "activation_quant_ffn", "activation_quant_mixer"}:
        return "activation_quant"
    if role in {"cuda_graph_launch", "graph", "host_graph_submit"}:
        return "cuda_graph_launch"
    if role in {"cpu_unknown_gap", "other_idle", "host_submission_waits"}:
        return "cpu_unknown_gap"
    if any(marker in combined for marker in FFN_WEIGHT_MARKERS):
        if "ffn_down" in combined:
            return "ffn_down"
        if "ffn_gate" in combined or "ffn_up" in combined:
            return "ffn_gate_up_glu"
    if "ssm_" in combined or "attn_qkv" in combined:
        return (
            "gdn_core"
            if "ssm_" in combined or role in GDN_LEAVES
            else "mixer_projection"
        )
    if any(marker in combined for marker in ("attn_q", "attn_k", "attn_v", "attn_out")):
        return (
            "attention_core"
            if role in ATTENTION_LEAVES or "attn_out" in combined
            else "mixer_projection"
        )
    if role in {"MUL_MAT", "mul_mat"} or launch in {"mmq", "mmvq", "mmv"}:
        if any(marker in combined for marker in MIXER_WEIGHT_MARKERS):
            return "mixer_projection"
        if any(marker in combined for marker in FFN_WEIGHT_MARKERS):
            return "ffn_down" if "ffn_down" in combined else "ffn_gate_up_glu"
        return "unmapped_mul_mat"
    if role in {"mixer_mmv", "mixer_mmq"}:
        return "mixer_projection"
    return role or "unknown"


def _layer_key(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (record.get("engine"), record.get("layer"), record.get("phase"))


def chargeable_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Charge each interval once. Enclosing FFN is not added to its leaves."""
    enclosing = [
        record
        for record in records
        if str(record.get("attribution_role", "")) == "enclosing"
        and int(record.get("fused_member_count", 1) or 1) >= 1
    ]
    covered: dict[tuple[Any, Any, Any], set[str]] = {}
    wide_ffn: set[tuple[Any, Any, Any]] = set()
    for record in enclosing:
        key = _layer_key(record)
        members = _member_names(record)
        members.add(str(record.get("role", "")))
        covered.setdefault(key, set()).update(members)
        family = canonical_family(record)
        if str(record.get("role", "")) in {"ffn_mmv", "ffn_mmq"} or (
            family in {"ffn_mmv", "ffn_mmq"} and len(members) > 3
        ):
            wide_ffn.add(key)
            covered[key].update(FFN_LEAVES)
            covered[key].update(FFN_ENCLOSING)
    kept: list[Mapping[str, Any]] = []
    for record in records:
        role = str(record.get("role", ""))
        key = _layer_key(record)
        blocked = covered.get(key, set())
        attr = str(record.get("attribution_role", "member"))
        if attr == "member" and (
            role in blocked or bool(_member_names(record) & blocked)
        ):
            continue
        if (
            key in wide_ffn
            and role in FFN_LEAVES | FFN_ENCLOSING
            and role
            not in {
                "ffn_mmv",
                "ffn_mmq",
            }
        ):
            continue
        kept.append(record)
    return kept


def family_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in EXPECTED_DECODE_COUNTS}
    for record in chargeable_records(records):
        members = _member_names(record)
        role = str(record.get("role", ""))
        family = canonical_family(record)
        if family == "ffn_gate_up_glu" or role in {
            "ffn_gate_up_glu",
            "ffn_mmv",
            "ffn_mmq",
        }:
            counts["ffn_gate_up_glu"] += 1
        if family == "ffn_down" or role == "ffn_down" or "ffn_down" in members:
            counts["ffn_down"] += 1
        if family == "gdn_core" or role == "gdn_core":
            counts["gdn_core"] += 1
        if family == "attention_core" or role == "attention_core":
            counts["attention_core"] += 1
        if family == "logits_projection" or role in {"logits_projection", "logits"}:
            counts["logits_projection"] += 1
    return counts


def expected_decode_counts(tokens: int = 1) -> dict[str, int]:
    return {name: value * int(tokens) for name, value in EXPECTED_DECODE_COUNTS.items()}


def validate_call_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    tokens: int,
) -> dict[str, Any]:
    observed = family_counts(records)
    expected = expected_decode_counts(tokens)
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


def two_token_epochs_share_origin(records: Sequence[Mapping[str, Any]]) -> bool:
    by_token: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        token = int(record.get("token_position", -1))
        by_token.setdefault(token, []).append(record)
    if len(by_token) < 2:
        return False
    tokens = sorted(by_token)
    first_start = min(
        float(row.get("start_ms", 0.0) or 0.0) for row in by_token[tokens[0]]
    )
    second_start = min(
        float(row.get("start_ms", 0.0) or 0.0) for row in by_token[tokens[1]]
    )
    return second_start > first_start + 1e-9


def warmup_contamination(records: Sequence[Mapping[str, Any]]) -> bool:
    for record in records:
        phase = str(record.get("phase", ""))
        window = str(record.get("window", "measured"))
        if phase in {"warmup", "setup", "prefix"} or window in {
            "warmup",
            "setup",
            "prefix",
        }:
            return True
        if record.get("warmup") is True:
            return True
    return False


def pool_overflow(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(bool(record.get("pool_overflow")) for record in records)


def accounting_split(
    *,
    graph_wall_ms: float,
    eager_instrumented_wall_ms: float,
    eager_work_ms: float,
    attributed_union_ms: float,
    d2h_ms: float,
    commit_ms: float,
    prefix_ms: float = 0.0,
    graph_create_ms: float = 0.0,
) -> dict[str, Any]:
    decode_graph_wall = float(graph_wall_ms) - float(prefix_ms) - float(graph_create_ms)
    overhead = float(eager_instrumented_wall_ms) - decode_graph_wall
    uncovered = decode_graph_wall - float(attributed_union_ms)
    if uncovered < 0.0:
        uncovered = 0.0
    unexplained_share = (
        uncovered / decode_graph_wall if decode_graph_wall > 1e-12 else 1.0
    )
    return {
        "shipping_graph_wall_ms": decode_graph_wall,
        "eager_diagnostic_work_ms": float(eager_work_ms),
        "instrumentation_overhead_ms": overhead,
        "attributed_union_ms": float(attributed_union_ms),
        "uncovered_wall_ms": uncovered,
        "d2h_ms": float(d2h_ms),
        "commit_ms": float(commit_ms),
        "prefix_excluded_ms": float(prefix_ms),
        "graph_create_excluded_ms": float(graph_create_ms),
        "unexplained_share": unexplained_share,
        "invented_overlap": False,
    }


def rank_remaining_gaps(
    families: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    call_counts: Mapping[str, Any],
) -> dict[str, Any]:
    incomplete_reasons: list[str] = []
    if not call_counts.get("ok", False):
        incomplete_reasons.append("invalid_call_counts")
    if float(accounting.get("unexplained_share", 1.0)) > UNEXPLAINED_LIMIT:
        incomplete_reasons.append("unexplained_measured_window")
    if accounting.get("invented_overlap"):
        incomplete_reasons.append("invented_overlap")
    if incomplete_reasons:
        return {
            "status": "incomplete",
            "gap_attribution_complete": False,
            "reasons": incomplete_reasons,
            "ranked": [],
            "unknown_ms": float(accounting.get("uncovered_wall_ms", 0.0)),
        }
    ranked = []
    for row in families:
        quartz_ms = float(row.get("quartz_ms", 0.0) or 0.0)
        llama_ms = float(row.get("llama_ms", 0.0) or 0.0)
        excess = quartz_ms - llama_ms if llama_ms > 0.0 else quartz_ms
        ranked.append(
            {
                "family": row.get("family") or row.get("role"),
                "quartz_ms": quartz_ms,
                "llama_ms": llama_ms,
                "matched_llama_excess_ms": excess if llama_ms > 0.0 else None,
                "max_removable_ms": max(0.0, excess) if llama_ms > 0.0 else quartz_ms,
                "llama_covered": llama_ms > 0.0,
            }
        )
    ranked.sort(key=lambda item: float(item["max_removable_ms"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return {
        "status": "ranked",
        "gap_attribution_complete": True,
        "reasons": [],
        "ranked": ranked,
        "unknown_ms": float(accounting.get("uncovered_wall_ms", 0.0)),
    }


def validate_capture_key(
    provided: str | None,
    computed: str,
    bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if provided is None or provided == "":
        if bundle is not None and str(bundle.get("capture_key", "")) == computed:
            return {"ok": True, "action": "reuse", "recapture": False}
        return {"ok": True, "action": "capture", "recapture": bundle is None}
    if len(provided) != 64:
        raise ValueError("capture key must be sha256 hex")
    if bundle is None:
        raise ValueError("capture key was provided but its bundle was not loaded")
    bundle_key = str(bundle.get("capture_key", ""))
    if bundle_key != provided:
        raise ValueError("loaded bundle does not match --capture-key")
    if provided != computed:
        raise ValueError("stale capture key does not match current identity")
    return {"ok": True, "action": "reuse", "recapture": False}


def reject_wrong_role_layer_bundle(bundle: Mapping[str, Any]) -> None:
    layers = bundle.get("layers") or bundle.get("layer_inputs") or []
    if not layers:
        raise ValueError("bundle missing layer activations")
    seen: set[tuple[int, str]] = set()
    for slot in layers:
        layer = int(slot["layer"])
        role = str(slot["role"])
        if layer < 0:
            raise ValueError(f"invalid layer {layer}")
        if not role:
            raise ValueError("missing role in bundle slot")
        if slot.get("required_role") and slot["required_role"] != role:
            raise ValueError(
                f"wrong role for layer {layer}: {role} != {slot['required_role']}"
            )
        key = (layer, role)
        if key in seen:
            raise ValueError(f"duplicate layer/role {key}")
        seen.add(key)
        values = slot.get("values")
        digest = str(slot.get("hash", ""))
        if values is not None:
            packed = b"".join(struct.pack("<f", float(value)) for value in values)
            computed = sha256_bytes(packed)
            if digest and computed != digest:
                raise ValueError("bundle hash does not match values")


def nested_interval_fixture(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    charged = chargeable_records(records)
    totals = stream_aware_totals(charged)
    return {
        "charged": charged,
        "summed_work_ms": totals["summed_gpu_work_ms"],
        "union_ms": totals["interval_union_ms"],
        "overlap_ms": totals["overlap_ms"],
        "sum_exceeds_wall": totals["sum_exceeds_wall"],
    }


def compare_engine_families(
    quartz: Sequence[Mapping[str, Any]],
    llama: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for record in chargeable_records(rows):
            name = canonical_family(record)
            out[name] = out.get(name, 0.0) + float(
                record.get("complete_work_ms", 0.0) or 0.0
            )
        return out

    left = bucket(quartz)
    right = bucket(llama)
    names = sorted(set(left) | set(right) | set(EXPECTED_DECODE_COUNTS))
    return [
        {
            "family": name,
            "quartz_ms": left.get(name, 0.0),
            "llama_ms": right.get(name, 0.0),
        }
        for name in names
    ]


def cache_path(capture_key: str) -> Path:
    return CAPTURE_DIR / capture_key / "bundle.json"
