from __future__ import annotations

import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "gdn_authority.json"
TRANSFORMERS_REVISION = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left: float, right: float) -> float:
    return f32(left + right)


def multiply(left: float, right: float) -> float:
    return f32(left * right)


def vector_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def softplus(value: float) -> float:
    if value > 20.0:
        return f32(value)
    if value < -20.0:
        return f32(math.exp(value))
    return f32(math.log1p(math.exp(value)))


def gates(token: int, value_heads: int) -> tuple[list[float], list[float]]:
    log_decay: list[float] = []
    beta: list[float] = []
    for head in range(value_heads):
        a = f32(((token * 5 + head * 3) % 13 - 6) / 8.0)
        b = f32(((token * 7 + head * 2) % 11 - 5) / 8.0)
        a_log = f32(math.log(0.25 * (head + 1)))
        dt_bias = f32((head - 2) / 8.0)
        beta.append(f32(1.0 / (1.0 + math.exp(-b))))
        log_decay.append(
            f32(-multiply(f32(math.exp(a_log)), softplus(add(a, dt_bias))))
        )
    return log_decay, beta


def raw_query(token: int, head: int, lane: int) -> float:
    return f32(((token * 11 + head * 7 + lane * 3) % 19 - 9) / 8.0)


def raw_key(token: int, head: int, lane: int) -> float:
    return f32(((token * 13 + head * 5 + lane * 7) % 23 - 11) / 8.0)


def raw_value(token: int, head: int, lane: int) -> float:
    return f32(((token * 17 + head * 3 + lane * 5) % 29 - 14) / 8.0)


def normalize(vector: list[float], query_scale: bool) -> list[float]:
    total = 0.0
    for value in vector:
        total = add(total, multiply(value, value))
    inverse = f32(1.0 / math.sqrt(add(total, f32(1e-6))))
    if query_scale:
        inverse = f32(inverse / math.sqrt(len(vector)))
    return [multiply(value, inverse) for value in vector]


def recurrent_fixture() -> tuple[list[float], list[float], list[float], list[float]]:
    key_heads, value_heads, key_width, value_width, tokens = 2, 6, 3, 2, 5
    state = [
        f32(((head * 17 + key_lane * 5 + value_lane * 3) % 13 - 6) / 32.0)
        for head in range(value_heads)
        for key_lane in range(key_width)
        for value_lane in range(value_width)
    ]
    all_outputs: list[float] = []
    all_log_decay: list[float] = []
    all_beta: list[float] = []
    for token in range(tokens):
        queries = [
            normalize([raw_query(token, head, lane) for lane in range(key_width)], True)
            for head in range(key_heads)
        ]
        keys = [
            normalize([raw_key(token, head, lane) for lane in range(key_width)], False)
            for head in range(key_heads)
        ]
        log_decay, beta = gates(token, value_heads)
        all_log_decay.extend(log_decay)
        all_beta.extend(beta)
        for head in range(value_heads):
            key_head = head // (value_heads // key_heads)
            state_base = head * key_width * value_width
            decay = f32(math.exp(log_decay[head]))
            for index in range(key_width * value_width):
                state[state_base + index] = multiply(state[state_base + index], decay)
            prediction = [0.0] * value_width
            for key_lane in range(key_width):
                for value_lane in range(value_width):
                    index = state_base + key_lane * value_width + value_lane
                    prediction[value_lane] = add(
                        prediction[value_lane],
                        multiply(keys[key_head][key_lane], state[index]),
                    )
            delta = [
                multiply(
                    add(raw_value(token, head, lane), -prediction[lane]), beta[head]
                )
                for lane in range(value_width)
            ]
            for key_lane in range(key_width):
                for value_lane in range(value_width):
                    index = state_base + key_lane * value_width + value_lane
                    state[index] = add(
                        state[index],
                        multiply(keys[key_head][key_lane], delta[value_lane]),
                    )
            for value_lane in range(value_width):
                output = 0.0
                for key_lane in range(key_width):
                    index = state_base + key_lane * value_width + value_lane
                    output = add(
                        output,
                        multiply(queries[key_head][key_lane], state[index]),
                    )
                all_outputs.append(output)
    return all_outputs, state, all_log_decay, all_beta


def convolution_fixture() -> tuple[list[float], list[float]]:
    channels, width, tokens = 3, 4, 6
    state = [0.0] * (channels * width)
    output: list[float] = []
    for token in range(tokens):
        for channel in range(channels):
            base = channel * width
            for index in range(width - 1):
                state[base + index] = state[base + index + 1]
            state[base + width - 1] = f32(((token * 7 + channel * 5) % 17 - 8) / 8.0)
            total = 0.0
            for index in range(width):
                weight = f32(((channel * 11 + index * 3) % 13 - 6) / 8.0)
                total = add(total, multiply(state[base + index], weight))
            output.append(f32(total / (1.0 + math.exp(-total))))
    return output, state


def main() -> None:
    recurrent_output, recurrent_state, log_decay, beta = recurrent_fixture()
    convolution_output, convolution_state = convolution_fixture()
    document = {
        "schema_version": 1,
        "authority": {
            "implementation": "Transformers Qwen3_5 recurrent fallback, transcribed as explicit FP32 scalar operations",
            "revision": TRANSFORMERS_REVISION,
            "contract": "pins/gdn_contract.json",
        },
        "tolerances": {
            "absolute": 2e-6,
            "relative": 2e-5,
            "rms": 1e-6,
            "cosine_minimum": 0.999999,
        },
        "recurrent": {
            "shape": {
                "tokens": 5,
                "key_heads": 2,
                "value_heads": 6,
                "key_width": 3,
                "value_width": 2,
                "key_head_reuse": 3,
            },
            "chunkings": [[5], [2, 1, 2], [1, 1, 1, 1, 1]],
            "output_f32_le_hex": vector_hex(recurrent_output),
            "final_state_f32_le_hex": vector_hex(recurrent_state),
            "log_decay_f32_le_hex": vector_hex(log_decay),
            "beta_f32_le_hex": vector_hex(beta),
        },
        "convolution": {
            "shape": {"tokens": 6, "channels": 3, "width": 4},
            "chunkings": [[6], [1, 2, 3], [1, 1, 1, 1, 1, 1]],
            "output_f32_le_hex": vector_hex(convolution_output),
            "final_state_f32_le_hex": vector_hex(convolution_state),
        },
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
