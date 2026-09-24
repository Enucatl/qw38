#!/usr/bin/env python3
"""Extract BF16 projection weights and layer-zero normalized embeddings."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

CHECKPOINT = Path(".cache/authorities/qwen3.8-27b-transformers")
EMBED = "model.language_model.embed_tokens.weight"
NORM = "model.language_model.layers.0.input_layernorm.weight"


def bf16_to_float(code: int) -> float:
    """Decode one BF16 value without an array dependency."""
    return struct.unpack("<f", struct.pack("<I", code << 16))[0]


def float_to_bf16(value: float) -> int:
    """Round one finite float to BF16 with nearest-even ties."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def tensor_spec(name: str, index: dict[str, object]) -> tuple[Path, list[int], int]:
    """Resolve a BF16 tensor to its shard, shape, and byte offset."""
    shard = CHECKPOINT / index["weight_map"][name]
    with shard.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensor header: {shard}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    entry = header[name]
    if entry["dtype"] != "BF16":
        raise ValueError(f"expected BF16: {name}")
    return shard, entry["shape"], 8 + header_length + entry["data_offsets"][0]


def read_row(spec: tuple[Path, list[int], int], row: int) -> bytes:
    """Read one contiguous BF16 tensor row from the source shard."""
    shard, shape, offset = spec
    width = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    if row < 0 or row >= (shape[0] if len(shape) > 1 else 1):
        raise ValueError("row exceeds tensor geometry")
    with shard.open("rb") as handle:
        handle.seek(offset + row * width * 2)
        data = handle.read(width * 2)
    if len(data) != width * 2:
        raise ValueError(f"truncated tensor row in {shard}")
    return data


def main() -> None:
    """Write benchmark input slices and metadata without payload digests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0 or (
        not args.weights_only and args.k != 5120
    ):
        raise ValueError("activation extraction supports positive K=5120 only")
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
    weight = tensor_spec(args.tensor, index)
    if len(weight[1]) != 2 or args.n > weight[1][0] or args.k > weight[1][1]:
        raise ValueError("requested weight slice exceeds a 2D projection")
    args.out.mkdir(parents=True, exist_ok=True)
    a_path = args.out / "a.bf16le"
    b_path = args.out / "b.bf16le"
    token_ids: list[int] = []
    embed = norm = None
    if not args.weights_only:
        embed = tensor_spec(EMBED, index)
        norm = tensor_spec(NORM, index)
        if embed[1][1] != args.k or norm[1] != [args.k]:
            raise ValueError("embedding/norm geometry differs from projection K")
        gamma_codes = struct.unpack(f"<{args.k}H", read_row(norm, 0))
        gamma = [bf16_to_float(code) for code in gamma_codes]
        with a_path.open("wb") as output:
            for row in range(args.m):
                token_id = (17 + 7919 * row) % embed[1][0]
                token_ids.append(token_id)
                codes = struct.unpack(f"<{args.k}H", read_row(embed, token_id))
                values = [bf16_to_float(code) for code in codes]
                inverse_rms = 1.0 / math.sqrt(
                    sum(value * value for value in values) / args.k + 1.0e-6
                )
                packed = [float_to_bf16(value * inverse_rms * (1.0 + gain))
                          for value, gain in zip(values, gamma, strict=True)]
                output.write(struct.pack(f"<{args.k}H", *packed))
    with b_path.open("wb") as output:
        if args.k == weight[1][1]:
            remaining = args.n * args.k * 2
            with weight[0].open("rb") as source:
                source.seek(weight[2])
                while remaining:
                    chunk = source.read(min(8 << 20, remaining))
                    if not chunk:
                        raise ValueError("truncated projection tensor payload")
                    output.write(chunk)
                    remaining -= len(chunk)
        else:
            for row in range(args.n):
                output.write(read_row(weight, row)[: args.k * 2])
    metadata = {
        "checkpoint_index": str(CHECKPOINT / "model.safetensors.index.json"),
        "weight_tensor": args.tensor,
        "weight_shard": str(weight[0]),
        "weight_rows": [0, args.n],
        "embedding_tensor": EMBED,
        "embedding_shard": str(embed[0]) if embed else None,
        "normalizer_tensor": NORM,
        "normalizer_shard": str(norm[0]) if norm else None,
        "activation": "BF16 round((embedding * rsqrt(mean(square(embedding)) + 1e-6)) * (1 + layer0_input_norm))" if embed else None,
        "token_id_formula": "(17 + 7919 * row) % 248320",
        "token_ids": token_ids,
        "m": args.m, "n": args.n, "k": args.k,
        "a_file": str(a_path) if embed else None, "b_file": str(b_path),
        "a_bytes": a_path.stat().st_size if embed else 0,
        "b_bytes": b_path.stat().st_size,
    }
    (args.out / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in
                      ("weight_tensor", "m", "n", "k", "a_bytes", "b_bytes")}))


if __name__ == "__main__":
    main()
