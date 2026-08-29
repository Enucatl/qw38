from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"
OUTPUT = ROOT / "fixtures" / "gguf_conversion.json"
MODEL_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
SELECTED = (
    "blk.0.ssm_a",
    "blk.0.ssm_dt.bias",
    "blk.0.attn_norm.weight",
    "blk.0.ssm_norm.weight",
    "blk.0.ssm_conv1d.weight",
)


def tensor_fixture(model: Any, tensor: dict[str, Any]) -> dict[str, Any]:
    model.seek(tensor["absolute_offset"])
    payload = model.read(tensor["storage_bytes"])
    if len(payload) != tensor["storage_bytes"]:
        raise ValueError(f"short read for {tensor['name']}")
    values = struct.unpack(f"<{len(payload) // 4}f", payload)
    return {
        "name": tensor["name"],
        "shape": tensor["shape"],
        "dtype": tensor["dtype"],
        "storage_bytes": tensor["storage_bytes"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first_8_f32_le_hex": payload[:32].hex(),
        "last_8_f32_le_hex": payload[-32:].hex(),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    with MODEL.open("rb") as model:
        cases = [tensor_fixture(model, tensors[name]) for name in SELECTED]
    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "authority": "pins/gguf_conversion_contract.json",
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
