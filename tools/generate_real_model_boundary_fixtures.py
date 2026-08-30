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
OUTPUT = ROOT / "fixtures" / "real_model_boundaries.json"
EMBEDDING_ROWS = (0, 42, 248319)
LOGIT_ROWS = (0, 1, 42, 1000, 248319)
HIDDEN_TAPS = (0, 1, 2559, 5119)
LIBM = ctypes.CDLL("libm.so.6")
LIBM.sqrtf.restype = ctypes.c_float
LIBM.sqrtf.argtypes = [ctypes.c_float]


def add(left: float, right: float) -> float:
    return f32(left + right)


def multiply(left: float, right: float) -> float:
    return f32(left * right)


def vector_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    hashes: dict[str, dict[str, str]] = {}
    with MODEL.open("rb") as model:

        def row(name: str, index: int) -> list[float]:
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            model.seek(tensor["absolute_offset"] + index * row_bytes)
            payload = model.read(row_bytes)
            hashes.setdefault(name, {})[str(index)] = hashlib.sha256(
                payload
            ).hexdigest()
            values, _ = decode_blocks(tensor["dtype"], payload)
            return values

        embeddings = {
            index: row("token_embd.weight", index) for index in EMBEDDING_ROWS
        }
        norm_info = tensors["output_norm.weight"]
        model.seek(norm_info["absolute_offset"])
        scales = list(struct.unpack("<5120f", model.read(norm_info["storage_bytes"])))
        hidden = embeddings[42]
        total = 0.0
        for value in hidden:
            total = add(total, multiply(value, value))
        mean = f32(total / len(hidden))
        inverse = f32(1.0 / float(LIBM.sqrtf(ctypes.c_float(add(mean, f32(1e-6))))))
        normalized = [
            multiply(multiply(value, inverse), scale)
            for value, scale in zip(hidden, scales, strict=True)
        ]
        logits: list[float] = []
        for index in LOGIT_ROWS:
            tensor = tensors["output.weight"]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            model.seek(tensor["absolute_offset"] + index * row_bytes)
            payload = model.read(row_bytes)
            hashes.setdefault("output.weight", {})[str(index)] = hashlib.sha256(
                payload
            ).hexdigest()
            weights, block_values = decode_blocks(tensor["dtype"], payload)
            logits.append(blocked_dot(weights, normalized, block_values))

    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "authority": "independently decoded pinned GGUF embedding and output rows with direct-scale FP32 final RMSNorm",
        "tolerances": {"absolute": 3e-6, "relative": 3e-5, "rms": 1e-6},
        "embedding_rows": list(EMBEDDING_ROWS),
        "logit_rows": list(LOGIT_ROWS),
        "embedding_0_f32_le_hex": vector_hex(
            [embeddings[0][index] for index in HIDDEN_TAPS]
        ),
        "embedding_42_f32_le_hex": vector_hex(
            [embeddings[42][index] for index in HIDDEN_TAPS]
        ),
        "embedding_last_f32_le_hex": vector_hex(
            [embeddings[248319][index] for index in HIDDEN_TAPS]
        ),
        "final_normalized_f32_le_hex": vector_hex(
            [normalized[index] for index in HIDDEN_TAPS]
        ),
        "logits_f32_le_hex": vector_hex(logits),
        "row_sha256": hashes,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
