from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_tensor_row_fixtures import (  # noqa: E402
    MODEL,
    MODEL_SHA256,
    blocked_dot,
    decode_blocks,
)
from tools.generate_quant_fixtures import f32  # noqa: E402

INVENTORY = ROOT / "pins" / "tensor_inventory.json"
OUTPUT = ROOT / "fixtures" / "mixer_projections.json"
SELECTED = {
    "blk.0.attn_qkv.weight": [0, 2047, 2048, 4095, 4096, 10239],
    "blk.0.attn_gate.weight": [0, 6143],
    "blk.0.ssm_alpha.weight": [0, 47],
    "blk.0.ssm_beta.weight": [0, 47],
    "blk.3.attn_q.weight": [0, 255, 256, 511, 512, 767, 12287],
    "blk.3.attn_k.weight": [0, 1023],
    "blk.3.attn_v.weight": [0, 1023],
}


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    activation = [f32(((index * 37) % 101 - 50) / 32.0) for index in range(5120)]
    cases: list[dict[str, object]] = []
    with MODEL.open("rb") as model:
        for name, rows in SELECTED.items():
            tensor = tensors[name]
            columns, row_count = tensor["shape"]
            row_bytes = tensor["storage_bytes"] // row_count
            values: list[float] = []
            row_hashes: list[str] = []
            for row in rows:
                model.seek(tensor["absolute_offset"] + row * row_bytes)
                payload = model.read(row_bytes)
                weights, block_values = decode_blocks(tensor["dtype"], payload)
                values.append(blocked_dot(weights, activation, block_values))
                row_hashes.append(hashlib.sha256(payload).hexdigest())
            cases.append(
                {
                    "name": name,
                    "rows": rows,
                    "row_sha256": row_hashes,
                    "output_f32_le_hex": b"".join(
                        struct.pack("<f", value) for value in values
                    ).hex(),
                }
            )
    OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": MODEL_SHA256,
                "activation": "a[i] = (((i * 37) % 101) - 50) / 32, rounded to FP32",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
