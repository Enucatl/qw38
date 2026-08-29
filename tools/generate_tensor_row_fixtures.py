from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_quant_fixtures import (  # noqa: E402
    decode_q4,
    decode_q6,
    decode_q8,
    f32,
)

MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"
OUTPUT = ROOT / "fixtures" / "tensor_rows.json"
MODEL_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
SELECTED_ROWS = {
    "token_embd.weight": 42,
    "output.weight": 17,
    "blk.3.attn_q.weight": 9,
    "blk.0.ssm_conv1d.weight": 5,
}


def decode_blocks(kind: str, payload: bytes) -> tuple[list[float], int]:
    if kind == "Q4_K":
        size, values, decoder = 144, 256, decode_q4
    elif kind == "Q6_K":
        size, values, decoder = 210, 256, decode_q6
    elif kind == "Q8_0":
        size, values, decoder = 34, 32, decode_q8
    else:
        return list(struct.unpack(f"<{len(payload) // 4}f", payload)), 1
    return (
        [
            value
            for offset in range(0, len(payload), size)
            for value in decoder(payload[offset : offset + size])
        ],
        values,
    )


def blocked_dot(
    weights: list[float], activation: list[float], block_values: int
) -> float:
    total = 0.0
    for offset in range(0, len(weights), block_values):
        partial = 0.0
        for weight, value in zip(
            weights[offset : offset + block_values],
            activation[offset : offset + block_values],
        ):
            partial = f32(partial + f32(weight * value))
        total = f32(total + partial)
    return total


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    cases: list[dict[str, object]] = []
    with MODEL.open("rb") as model:
        for name, row in SELECTED_ROWS.items():
            tensor = tensors[name]
            columns, rows = tensor["shape"]
            row_bytes = tensor["storage_bytes"] // rows
            model.seek(tensor["absolute_offset"] + row * row_bytes)
            payload = model.read(row_bytes)
            weights, block_values = decode_blocks(tensor["dtype"], payload)
            activation = [
                f32(((index * 37) % 101 - 50) / 32.0) for index in range(columns)
            ]
            dot = blocked_dot(weights, activation, block_values)
            cases.append(
                {
                    "name": name,
                    "row": row,
                    "dtype": tensor["dtype"],
                    "columns": columns,
                    "rows": rows,
                    "row_bytes": row_bytes,
                    "row_sha256": hashlib.sha256(payload).hexdigest(),
                    "activation": "a[i] = (((i * 37) % 101) - 50) / 32, rounded to FP32",
                    "dot_f32_le_hex": struct.pack("<f", dot).hex(),
                }
            )
    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "dimension_order": "GGUF dimension 0 is the contiguous row width; dimension 1 is the output row count",
        "accumulation": "left-to-right FP32 within each format block, then left-to-right FP32 block totals",
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
