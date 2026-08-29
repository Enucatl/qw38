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
OUTPUT = ROOT / "fixtures" / "real_attention_step.json"
HEAD_WIDTH = 256
QUERY_HEADS = (0, 5, 6, 23)
KV_HEADS = (0, 1, 3)
HEAD_TAPS = ((0, 0), (0, 63), (0, 64), (0, 255), (5, 0), (6, 0), (23, 255))
CACHE_TAPS = (
    (0, 0, 0),
    (0, 0, 31),
    (0, 0, 32),
    (0, 0, 63),
    (0, 0, 64),
    (0, 0, 255),
    (0, 1, 0),
    (0, 1, 255),
    (1, 0, 0),
    (1, 0, 31),
    (1, 0, 32),
    (1, 0, 63),
    (1, 0, 64),
    (1, 0, 255),
    (1, 1, 0),
    (1, 1, 255),
)
LIBM = ctypes.CDLL("libm.so.6")
for function_name in ("cosf", "expf", "sinf", "sqrtf", "powf"):
    function = getattr(LIBM, function_name)
    function.restype = ctypes.c_float
    function.argtypes = (
        [ctypes.c_float, ctypes.c_float]
        if function_name == "powf"
        else [ctypes.c_float]
    )


def unary(name: str, value: float) -> float:
    return float(getattr(LIBM, name)(ctypes.c_float(value)))


def add(left: float, right: float) -> float:
    return f32(left + right)


def multiply(left: float, right: float) -> float:
    return f32(left * right)


def vector_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def normalize(values: list[float], scales: list[float]) -> list[float]:
    total = 0.0
    for value in values:
        total = add(total, multiply(value, value))
    mean = f32(total / len(values))
    inverse = f32(1.0 / unary("sqrtf", add(mean, f32(1e-6))))
    return [
        multiply(multiply(value, inverse), scale)
        for value, scale in zip(values, scales, strict=True)
    ]


def rotate(values: list[float], position: int) -> list[float]:
    output = values.copy()
    for lane in range(32):
        exponent = f32((lane * 2) / 64)
        denominator = float(
            LIBM.powf(ctypes.c_float(10_000_000.0), ctypes.c_float(exponent))
        )
        angle = f32(position / denominator)
        cosine = unary("cosf", angle)
        sine = unary("sinf", angle)
        output[lane] = add(
            multiply(values[lane], cosine), -multiply(values[32 + lane], sine)
        )
        output[32 + lane] = add(
            multiply(values[32 + lane], cosine), multiply(values[lane], sine)
        )
    return output


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = unary("expf", -value)
        return f32(1.0 / add(1.0, exponential))
    exponential = unary("expf", value)
    return f32(exponential / add(1.0, exponential))


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    row_hashes: dict[str, dict[str, str]] = {}
    with MODEL.open("rb") as model:

        def vector(name: str) -> list[float]:
            tensor = tensors[name]
            model.seek(tensor["absolute_offset"])
            return list(
                struct.unpack(
                    f"<{tensor['storage_bytes'] // 4}f",
                    model.read(tensor["storage_bytes"]),
                )
            )

        def project(name: str, row: int, activation: list[float]) -> float:
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            model.seek(tensor["absolute_offset"] + row * row_bytes)
            payload = model.read(row_bytes)
            weights, block_values = decode_blocks(tensor["dtype"], payload)
            return blocked_dot(weights, activation, block_values)

        def hash_row(name: str, row: int) -> None:
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            model.seek(tensor["absolute_offset"] + row * row_bytes)
            digest = hashlib.sha256(model.read(row_bytes)).hexdigest()
            row_hashes.setdefault(name, {})[str(row)] = digest

        input_scale = vector("blk.3.attn_norm.weight")
        query_scale = vector("blk.3.attn_q_norm.weight")
        key_scale = vector("blk.3.attn_k_norm.weight")
        residuals = [
            [
                f32((((index * 37 + token * 17) % 101) - 50) / 32.0)
                for index in range(5120)
            ]
            for token in range(2)
        ]
        normalized = [normalize(residual, input_scale) for residual in residuals]

        keys: dict[tuple[int, int], list[float]] = {}
        values: dict[tuple[int, int, int], float] = {}
        needed_value_lanes = {0: (0, 31, 32, 63, 64, 255), 1: (0, 255), 3: (255,)}
        for token in range(2):
            for kv_head in KV_HEADS:
                raw_key = [
                    project(
                        "blk.3.attn_k.weight",
                        kv_head * HEAD_WIDTH + lane,
                        normalized[token],
                    )
                    for lane in range(HEAD_WIDTH)
                ]
                keys[token, kv_head] = rotate(normalize(raw_key, key_scale), token)
                for lane in needed_value_lanes[kv_head]:
                    values[token, kv_head, lane] = project(
                        "blk.3.attn_v.weight",
                        kv_head * HEAD_WIDTH + lane,
                        normalized[token],
                    )

        queries: dict[int, list[float]] = {}
        gates: dict[tuple[int, int], float] = {}
        for query_head in QUERY_HEADS:
            raw_query = [
                project(
                    "blk.3.attn_q.weight",
                    query_head * 512 + lane,
                    normalized[1],
                )
                for lane in range(HEAD_WIDTH)
            ]
            queries[query_head] = rotate(normalize(raw_query, query_scale), 1)
        for query_head, lane in HEAD_TAPS:
            gates[query_head, lane] = project(
                "blk.3.attn_q.weight",
                query_head * 512 + HEAD_WIDTH + lane,
                normalized[1],
            )

        attention: list[float] = []
        for query_head, lane in HEAD_TAPS:
            kv_head = query_head // 6
            scores: list[float] = []
            for token in range(2):
                total = 0.0
                for query_value, key_value in zip(
                    queries[query_head], keys[token, kv_head], strict=True
                ):
                    total = add(total, multiply(query_value, key_value))
                scores.append(f32(total / unary("sqrtf", 256.0)))
            maximum = max(scores)
            exponentials = [unary("expf", add(score, -maximum)) for score in scores]
            denominator = add(exponentials[0], exponentials[1])
            mixed = 0.0
            for token, exponential in enumerate(exponentials):
                probability = f32(exponential / denominator)
                mixed = add(
                    mixed,
                    multiply(probability, values[token, kv_head, lane]),
                )
            attention.append(multiply(mixed, sigmoid(gates[query_head, lane])))

        query_taps = [
            project(
                "blk.3.attn_q.weight",
                head * 512 + lane,
                normalized[1],
            )
            for head, lane in HEAD_TAPS
        ]
        gate_taps = [gates[item] for item in HEAD_TAPS]
        for head, lane in HEAD_TAPS:
            hash_row("blk.3.attn_q.weight", head * 512 + lane)
            hash_row("blk.3.attn_q.weight", head * 512 + HEAD_WIDTH + lane)
        for _, kv_head, lane in CACHE_TAPS:
            hash_row("blk.3.attn_k.weight", kv_head * HEAD_WIDTH + lane)
            hash_row("blk.3.attn_v.weight", kv_head * HEAD_WIDTH + lane)

    document = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "authority": "pinned Transformers grouped eager attention plus independently decoded GGUF rows",
        "tolerances": {"absolute": 3e-6, "relative": 3e-5, "rms": 1e-6},
        "head_taps": [list(item) for item in HEAD_TAPS],
        "cache_taps": [list(item) for item in CACHE_TAPS],
        "normalized_f32_le_hex": vector_hex(
            [normalized[1][index] for index in (0, 1, 2559, 5119)]
        ),
        "query_f32_le_hex": vector_hex(query_taps),
        "gate_f32_le_hex": vector_hex(gate_taps),
        "key_cache_f32_le_hex": vector_hex(
            [keys[token, head][lane] for token, head, lane in CACHE_TAPS]
        ),
        "value_cache_f32_le_hex": vector_hex(
            [values[token, head, lane] for token, head, lane in CACHE_TAPS]
        ),
        "attention_output_f32_le_hex": vector_hex(attention),
        "row_sha256": row_hashes,
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
