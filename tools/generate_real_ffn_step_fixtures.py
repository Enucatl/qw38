from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_quant_fixtures import f32  # noqa: E402
from tools.generate_tensor_row_fixtures import (  # noqa: E402
    MODEL,
    MODEL_SHA256,
    blocked_dot,
    decode_blocks,
)

INVENTORY = ROOT / "pins" / "tensor_inventory.json"
OUTPUT = ROOT / "fixtures" / "real_ffn_step.json"
TAPS = (0, 1, 8703, 8704, 17407)
LIBM = ctypes.CDLL("libm.so.6")
for function_name in ("expf", "sqrtf"):
    function = getattr(LIBM, function_name)
    function.restype = ctypes.c_float
    function.argtypes = [ctypes.c_float]


def unary(name: str, value: float) -> float:
    return float(getattr(LIBM, name)(ctypes.c_float(value)))


def add(left: float, right: float) -> float:
    return f32(left + right)


def multiply(left: float, right: float) -> float:
    return f32(left * right)


def vector_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    residual = [f32(((index * 37) % 101 - 50) / 32.0) for index in range(5120)]
    row_hashes: dict[str, list[str]] = {}
    with MODEL.open("rb") as model:
        norm_info = tensors["blk.0.post_attention_norm.weight"]
        model.seek(norm_info["absolute_offset"])
        norm = list(struct.unpack("<5120f", model.read(norm_info["storage_bytes"])))
        total = 0.0
        for value in residual:
            total = add(total, multiply(value, value))
        mean = f32(total / len(residual))
        inverse = f32(1.0 / unary("sqrtf", add(mean, f32(1e-6))))
        normalized = [
            multiply(multiply(value, inverse), scale)
            for value, scale in zip(residual, norm, strict=True)
        ]

        def selected_projection(name: str) -> list[float]:
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            values: list[float] = []
            hashes: list[str] = []
            for row in TAPS:
                model.seek(tensor["absolute_offset"] + row * row_bytes)
                payload = model.read(row_bytes)
                weights, block_values = decode_blocks(tensor["dtype"], payload)
                values.append(blocked_dot(weights, normalized, block_values))
                hashes.append(hashlib.sha256(payload).hexdigest())
            row_hashes[name] = hashes
            return values

        gate = selected_projection("blk.0.ffn_gate.weight")
        up = selected_projection("blk.0.ffn_up.weight")
        activated = [
            multiply(f32(gate_value / add(1.0, unary("expf", -gate_value))), up_value)
            for gate_value, up_value in zip(gate, up, strict=True)
        ]
    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "authority": "pinned Transformers SwiGLU equation plus independently decoded GGUF rows",
        "tolerances": {"absolute": 2e-6, "relative": 2e-5, "rms": 1e-6},
        "intermediate_taps": list(TAPS),
        "normalized_f32_le_hex": vector_hex(
            [normalized[index] for index in (0, 1, 2559, 5119)]
        ),
        "gate_f32_le_hex": vector_hex(gate),
        "up_f32_le_hex": vector_hex(up),
        "activated_f32_le_hex": vector_hex(activated),
        "row_sha256": row_hashes,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
