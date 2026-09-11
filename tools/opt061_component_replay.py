"""Host helpers for OPT-061 real-input streaming component replay."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
CALIBRATION_LAYERS = (0, 3, 31, 32)
HELD_OUT_LAYERS = (62, 63)
CAPTURE_LAYERS = CALIBRATION_LAYERS + HELD_OUT_LAYERS
FAMILIES = ("decode-ffn", "decode-mixer", "prompt-ffn")
CACHE_MODES = ("hot", "rotating")
GATE_UP_DOWN_CALLS_PER_TOKEN = 64
PROMPT_SAMPLED_ROWS = (0, 1024, 2048, 4095)
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
INVENTORY = ROOT / "pins/tensor_inventory.json"
CONTRACT = ROOT / "pins/opt061_component_replay_contract.json"
ITERATION = ROOT / "pins/opt061_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt061_component_replay.json"
NATIVE = ROOT / "cuda/optimization_component_replay.cu"
HEADER = ROOT / "cuda/optimization_component_replay.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
REPORT = ROOT / "evidence/optimization/opt061-component-replay/REPORT.md"
OPT059_FIXTURE = ROOT / "fixtures/opt059_gpu_numerics.json"
OPT060_FIXTURE = ROOT / "fixtures/opt060_engine_attribution.json"


def sha256_floats(values: Sequence[float]) -> str:
    packed = b"".join(struct.pack("<f", float(value)) for value in values)
    return hashlib.sha256(packed).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture_identity(
    *,
    gguf_sha: str,
    token_generator: str,
    stage: str,
    layers: Sequence[int],
    prompt_rows: int,
    source: str,
) -> str:
    payload = json.dumps(
        {
            "gguf_sha": gguf_sha,
            "token_generator": token_generator,
            "stage": stage,
            "layers": list(layers),
            "prompt_rows": prompt_rows,
            "source": source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("utf-8"))


def inventory_tensor(
    name: str, inventory: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    data = inventory or json.loads(INVENTORY.read_text(encoding="utf-8"))
    for tensor in data["tensors"]:
        if tensor["name"] == name:
            return dict(tensor)
    raise ValueError(f"unknown tensor {name}")


def tensor_name(layer: int, role: str) -> str:
    mapping = {
        "ffn_gate": f"blk.{layer}.ffn_gate.weight",
        "ffn_up": f"blk.{layer}.ffn_up.weight",
        "ffn_down": f"blk.{layer}.ffn_down.weight",
        "ffn_norm": f"blk.{layer}.post_attention_norm.weight",
        "input_norm": f"blk.{layer}.attn_norm.weight",
        "gdn_packed_qkv": f"blk.{layer}.attn_qkv.weight",
        "gdn_value_gate": f"blk.{layer}.attn_gate.weight",
        "gdn_alpha": f"blk.{layer}.ssm_alpha.weight",
        "gdn_beta": f"blk.{layer}.ssm_beta.weight",
        "gdn_output": f"blk.{layer}.ssm_out.weight",
        "attention_q_gate": f"blk.{layer}.attn_q.weight",
        "attention_key": f"blk.{layer}.attn_k.weight",
        "attention_value": f"blk.{layer}.attn_v.weight",
        "attention_output": f"blk.{layer}.attn_out.weight",
        "output_norm": "output_norm.weight",
        "output_projection": "output.weight",
    }
    if role not in mapping:
        raise ValueError(f"unknown role {role}")
    return mapping[role]


def validate_capture_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "layer",
        "role",
        "token",
        "shape",
        "dtype",
        "tensor_name",
        "gguf_identity",
        "selector_set",
        "hash",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"capture missing fields {missing}")
    layer = int(record["layer"])
    if layer not in CAPTURE_LAYERS:
        raise ValueError(f"capture layer {layer} is not calibration/held-out")
    role = str(record["role"])
    name = str(record["tensor_name"])
    expected = (
        tensor_name(layer, role)
        if role not in {"residual", "ffn_down_input", "output_norm_input"}
        else name
    )
    if (
        role not in {"residual", "ffn_down_input", "output_norm_input"}
        and name != expected
    ):
        raise ValueError(
            f"role {role} must resolve through inventory name {expected}, got {name}"
        )
    shape = list(record["shape"])
    if len(shape) != 2 or int(shape[0]) < 1 or int(shape[1]) < 1:
        raise ValueError(f"invalid shape {shape} for role {role}")
    digest = str(record["hash"])
    if len(digest) != 64:
        raise ValueError("capture hash must be sha256 hex")
    values = record.get("values")
    if values is not None:
        computed = sha256_floats(list(values))
        if computed != digest:
            raise ValueError("capture hash does not match values")
    return {"ok": True, "tensor_name": name, "layer": layer, "role": role}


def reject_wrong_role_shape(record: Mapping[str, Any]) -> None:
    role = str(record["role"])
    shape = [int(part) for part in record["shape"]]
    dtype = str(record["dtype"])
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"wrong role/shape for {role}: {shape} {dtype}")
    if role in {"ffn_gate", "ffn_up", "ffn_down"} and dtype != "Q4_K":
        raise ValueError(f"{role} cannot be {dtype}")
    if role.startswith("gdn_") and dtype != "Q8_0":
        raise ValueError("mixer GDN projections are Q8_0, not Q4_K")
    if role == "residual" and dtype != "FP32":
        raise ValueError("residual must be FP32")
    if role in {"ffn_preprojection", "mixer_preprojection"} and dtype != "BF16":
        raise ValueError("projection inputs must be BF16")
    if not record.get("synthetic"):
        expected = {
            "ffn_gate": [5120, 17408],
            "ffn_up": [5120, 17408],
            "ffn_down": [17408, 5120],
        }
        if role in expected and shape != expected[role]:
            raise ValueError(f"wrong role/shape for {role}: {shape} {dtype}")


def paired_ffn_ms(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    enclosing = 0.0
    members = 0.0
    enclosing_count = 0
    member_count = 0
    for record in records:
        role = str(record.get("role", ""))
        attr = str(record.get("attribution_role", "member"))
        ms = float(record.get("complete_work_ms", 0.0) or 0.0)
        if role == "ffn_gate_up_glu" or attr == "enclosing":
            enclosing += ms
            enclosing_count += 1
        elif role in {"ffn_gate", "ffn_up", "ffn_glu"}:
            members += ms
            member_count += 1
    charged = enclosing if enclosing_count else members
    double_counted = (
        enclosing_count > 0
        and member_count > 0
        and abs(enclosing + members - charged) > 1e-9
    )
    if double_counted:
        raise ValueError(
            "paired FFN double-counted: enclosing already includes gate/up/GLU"
        )
    calls = GATE_UP_DOWN_CALLS_PER_TOKEN
    if enclosing_count:
        # Combined gate/up pair is 64 calls, not 128 pairs.
        if enclosing_count != calls and enclosing_count not in {1, calls}:
            raise ValueError(
                f"paired gate/up call count {enclosing_count} is not 64 per token"
            )
    return {
        "charged_ms": charged,
        "gate_up_calls": calls if enclosing_count or member_count else 0,
        "paired_not_128": True,
        "double_counted": False,
    }


def independent_rounds(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for sample in samples:
        if sample.get("warmup"):
            continue
        unit = str(sample.get("observation_unit", ""))
        if unit != "independent_round":
            raise ValueError("timing sample is not an independent round")
        key = (
            sample.get("family"),
            sample.get("cache_mode"),
            sample.get("sample_index"),
            sample.get("round_id"),
        )
        if key in seen:
            raise ValueError("timing rounds are not independent")
        seen.add(key)
        rounds.append(dict(sample))
    return rounds


def refuse_hot_as_production(result: Mapping[str, Any]) -> None:
    if result.get("accept_hot_as_production"):
        raise ValueError("hot cache_mode cannot be sole production acceptance")
    modes = result.get("accepted_cache_modes") or result.get("cache_modes")
    if modes == ["hot"] or modes == "hot":
        raise ValueError("hot cache_mode cannot be sole production acceptance")
    status = str(result.get("status", ""))
    if (
        status in {"component_accepted", "production_kept"}
        and result.get("cache_mode") == "hot"
    ):
        raise ValueError("hot cache_mode cannot be sole production acceptance")


def guards_intact(payload: bytes, guard: bytes, prefix: int, suffix: int) -> bool:
    if len(payload) < prefix + suffix:
        return False
    return payload[:prefix] == guard and payload[-suffix:] == guard


def useful_byte_rate_gbps(useful_bytes: float, elapsed_ms: float) -> dict[str, Any]:
    if elapsed_ms <= 0:
        raise ValueError("elapsed_ms must be positive")
    gbps = (useful_bytes / 1.0e9) / (elapsed_ms / 1.0e3)
    return {
        "useful_bytes": useful_bytes,
        "elapsed_ms": elapsed_ms,
        "useful_byte_rate_gbps": gbps,
        "kind": "estimate",
        "dram_counter_claim": False,
    }


def opt059_reference_identity() -> dict[str, Any]:
    if not OPT059_FIXTURE.is_file():
        return {
            "available": False,
            "identity": "independent_fp64_control_dots",
            "immutable": False,
        }
    payload = json.loads(OPT059_FIXTURE.read_text(encoding="utf-8"))
    return {
        "available": True,
        "identity": "fixtures/opt059_gpu_numerics.json",
        "task": payload.get("task"),
        "immutable": True,
        "status": payload.get("status"),
    }


def replay_command(
    family: str,
    cache_mode: str,
    *,
    capture_key: str,
    rows: Sequence[int],
) -> str:
    row_list = ",".join(str(row) for row in rows)
    return (
        "./build/qw38-cuda-component-replay "
        f"--workload {family} --cache-mode {cache_mode} "
        f"--capture-key {capture_key} --rows {row_list}"
    )


def rank_next_families(
    attribution: Mapping[str, Any],
    rotating: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, ms in rotating.items():
        attr_ms = float(attribution.get(name, 0.0) or 0.0)
        rows.append(
            {
                "family": name,
                "rotating_replay_ms": float(ms),
                "full_engine_attribution_ms": attr_ms,
                "rank_key": attr_ms if attr_ms > 0 else float(ms),
            }
        )
    rows.sort(key=lambda row: row["rank_key"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows
