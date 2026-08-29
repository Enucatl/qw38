from __future__ import annotations

import ctypes
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "attention_ffn_authority.json"
REVISION = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
LIBM = ctypes.CDLL("libm.so.6")
for function_name in ("cosf", "expf", "sinf", "sqrtf", "powf"):
    function = getattr(LIBM, function_name)
    function.restype = ctypes.c_float
    function.argtypes = (
        [ctypes.c_float, ctypes.c_float]
        if function_name == "powf"
        else [ctypes.c_float]
    )


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left: float, right: float) -> float:
    return f32(left + right)


def multiply(left: float, right: float) -> float:
    return f32(left * right)


def vector_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def unary_f32(name: str, value: float) -> float:
    return float(getattr(LIBM, name)(ctypes.c_float(value)))


def pow_f32(base: float, exponent: float) -> float:
    return float(LIBM.powf(ctypes.c_float(base), ctypes.c_float(exponent)))


def fixture_value(layer: int, token: int, head: int, lane: int, salt: int) -> float:
    value = (layer * 3 + token * 11 + head * 7 + lane * 5 + salt) % 31 - 15
    return f32(value / 8.0)


def rms_norm(values: list[float], weights: list[float]) -> list[float]:
    total = 0.0
    for value in values:
        total = add(total, multiply(value, value))
    mean = f32(total / len(values))
    inverse = f32(1.0 / unary_f32("sqrtf", add(mean, f32(1e-6))))
    return [
        multiply(multiply(value, inverse), add(1.0, weight))
        for value, weight in zip(values, weights)
    ]


def rotate(values: list[float], width: int, position: int) -> list[float]:
    output = values.copy()
    half = width // 2
    for lane in range(half):
        angle = f32(position / pow_f32(10_000_000.0, f32((lane * 2) / width)))
        cosine = unary_f32("cosf", angle)
        sine = unary_f32("sinf", angle)
        output[lane] = add(
            multiply(values[lane], cosine), -multiply(values[half + lane], sine)
        )
        output[half + lane] = add(
            multiply(values[half + lane], cosine), multiply(values[lane], sine)
        )
    return output


def sigmoid(value: float) -> float:
    return f32(1.0 / (1.0 + unary_f32("expf", -value)))


def attention_fixture(layer: int) -> dict[str, object]:
    query_heads, kv_heads, width, rotary, tokens = 6, 2, 8, 4, 4
    group = query_heads // kv_heads
    q_weights = [f32((lane - 3) / 64.0) for lane in range(width)]
    k_weights = [f32((3 - lane) / 64.0) for lane in range(width)]
    key_cache = [f32(1000 + index) for index in range(tokens * kv_heads * width)]
    value_cache = [f32(-1000 - index) for index in range(tokens * kv_heads * width)]
    outputs: list[float] = []
    for token in range(tokens):
        queries = []
        gates = []
        for head in range(query_heads):
            raw = [fixture_value(layer, token, head, lane, 1) for lane in range(width)]
            queries.append(rotate(rms_norm(raw, q_weights), rotary, token))
            gates.append(
                [fixture_value(layer, token, head, lane, 9) for lane in range(width)]
            )
        keys = []
        values = []
        for head in range(kv_heads):
            raw_key = [
                fixture_value(layer, token, head, lane, 4) for lane in range(width)
            ]
            keys.append(rotate(rms_norm(raw_key, k_weights), rotary, token))
            values.append(
                [fixture_value(layer, token, head, lane, 13) for lane in range(width)]
            )
            base = (token * kv_heads + head) * width
            key_cache[base : base + width] = keys[-1]
            value_cache[base : base + width] = values[-1]
        for query_head in range(query_heads):
            kv_head = query_head // group
            scores = []
            for context in range(token + 1):
                base = (context * kv_heads + kv_head) * width
                score = 0.0
                for lane in range(width):
                    score = add(
                        score,
                        multiply(queries[query_head][lane], key_cache[base + lane]),
                    )
                scores.append(f32(score / unary_f32("sqrtf", width)))
            maximum = max(scores)
            exponentials = [unary_f32("expf", add(score, -maximum)) for score in scores]
            denominator = 0.0
            for exponential in exponentials:
                denominator = add(denominator, exponential)
            for lane in range(width):
                result = 0.0
                for context, exponential in enumerate(exponentials):
                    base = (context * kv_heads + kv_head) * width
                    probability = f32(exponential / denominator)
                    result = add(
                        result, multiply(probability, value_cache[base + lane])
                    )
                outputs.append(multiply(result, sigmoid(gates[query_head][lane])))
    return {
        "layer": layer,
        "shape": {
            "tokens": tokens,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_width": width,
            "rotary_width": rotary,
        },
        "output_f32_le_hex": vector_hex(outputs),
        "key_cache_f32_le_hex": vector_hex(key_cache),
        "value_cache_f32_le_hex": vector_hex(value_cache),
    }


def matrix_vector(
    weights: list[float], rows: int, columns: int, values: list[float]
) -> list[float]:
    output = []
    for row in range(rows):
        total = 0.0
        for column in range(columns):
            total = add(
                total, multiply(weights[row * columns + column], values[column])
            )
        output.append(total)
    return output


def ffn_fixture(layer: int) -> dict[str, object]:
    hidden, intermediate = 4, 6
    values = [fixture_value(layer, 0, 0, lane, 2) for lane in range(hidden)]
    gate_weights = [
        fixture_value(layer, row, 0, column, 3) / 4
        for row in range(intermediate)
        for column in range(hidden)
    ]
    up_weights = [
        fixture_value(layer, row, 0, column, 7) / 4
        for row in range(intermediate)
        for column in range(hidden)
    ]
    down_weights = [
        fixture_value(layer, row, 0, column, 11) / 4
        for row in range(hidden)
        for column in range(intermediate)
    ]
    gate = matrix_vector(gate_weights, intermediate, hidden, values)
    up = matrix_vector(up_weights, intermediate, hidden, values)
    activated = [
        multiply(f32(g / (1.0 + unary_f32("expf", -g))), u) for g, u in zip(gate, up)
    ]
    output = matrix_vector(down_weights, hidden, intermediate, activated)
    return {
        "layer": layer,
        "gate_f32_le_hex": vector_hex(gate),
        "up_f32_le_hex": vector_hex(up),
        "activated_f32_le_hex": vector_hex(activated),
        "output_f32_le_hex": vector_hex(output),
    }


def main() -> None:
    document = {
        "schema_version": 1,
        "authority": {
            "implementation": "Transformers Qwen3_5 eager attention/RoPE/RMSNorm/MLP, transcribed as explicit FP32 scalar operations",
            "revision": REVISION,
            "contract": "pins/attention_ffn_contract.json",
        },
        "tolerances": {
            "absolute": 3e-6,
            "relative": 3e-5,
            "rms": 1e-6,
            "cosine_minimum": 0.999999,
        },
        "attention_cases": [attention_fixture(layer) for layer in (3, 7, 63)],
        "ffn_cases": [ffn_fixture(layer) for layer in (3, 7, 63)],
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
