#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["gguf", "numpy==2.5.3"]
# ///
"""Check whether the three GGUF arms share identical F32 control payloads."""

import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np


def f32_payloads(path: Path) -> dict[str, np.ndarray]:
    """Return memory-mapped F32 tensor payloads keyed by tensor name."""
    reader = gguf.GGUFReader(str(path))
    return {
        tensor.name: tensor.data
        for tensor in reader.tensors
        if tensor.tensor_type == gguf.GGMLQuantizationType.F32
    }


def main() -> None:
    """Write a count, digest, and exact-difference audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_gguf", type=Path)
    parser.add_argument("proxy_gguf", type=Path)
    parser.add_argument("comparator_gguf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    arms = {
        "bf16": f32_payloads(args.bf16_gguf),
        "proxy": f32_payloads(args.proxy_gguf),
        "comparator": f32_payloads(args.comparator_gguf),
    }
    names = set(arms["bf16"])
    if any(set(tensors) != names for tensors in arms.values()):
        raise ValueError("F32 tensor name sets differ")
    digest = hashlib.sha256()
    mismatches = []
    values = 0
    nonzero_low_bits = 0
    for name in sorted(names):
        reference = arms["bf16"][name]
        payload = reference.tobytes()
        values += reference.size
        nonzero_low_bits += int(np.count_nonzero(reference.view(np.uint32) & 0xFFFF))
        digest.update(name.encode() + b"\0")
        digest.update(payload)
        for arm in ("proxy", "comparator"):
            if arms[arm][name].tobytes() != payload:
                mismatches.append({"arm": arm, "tensor": name})
    report = {
        "schema": "task022-f32-control-audit-v1",
        "f32_tensors": len(names),
        "f32_values": values,
        "nonzero_low_16_bits": nonzero_low_bits,
        "canonical_payload_sha256": digest.hexdigest(),
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
