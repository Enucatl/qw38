from __future__ import annotations

import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "quant_authority.json"
LLAMA_CPP_REVISION = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> str:
    return struct.pack("<f", f32(value)).hex()


def half(value: float) -> bytes:
    return struct.pack("<e", value)


def q4_scale_min(packed: bytes, index: int) -> tuple[int, int]:
    if index < 4:
        return packed[index] & 63, packed[index + 4] & 63
    scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4)
    minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)
    return scale, minimum


def decode_q4(block: bytes) -> list[float]:
    d, dmin = struct.unpack_from("<ee", block)
    packed = block[4:16]
    quants = block[16:]
    values: list[float] = []
    scale_index = 0
    for quant_offset in range(0, 128, 32):
        scale_low, min_low = q4_scale_min(packed, scale_index)
        scale_high, min_high = q4_scale_min(packed, scale_index + 1)
        for quant in quants[quant_offset : quant_offset + 32]:
            values.append(f32(f32(d * scale_low) * (quant & 15) - dmin * min_low))
        for quant in quants[quant_offset : quant_offset + 32]:
            values.append(f32(f32(d * scale_high) * (quant >> 4) - dmin * min_high))
        scale_index += 2
    return values


def decode_q6(block: bytes) -> list[float]:
    low = block[:128]
    high = block[128:192]
    scales = struct.unpack_from("<16b", block, 192)
    (d,) = struct.unpack_from("<e", block, 208)
    values = [0.0] * 256
    for half_index in range(2):
        low_offset = half_index * 64
        high_offset = half_index * 32
        scale_offset = half_index * 8
        output_offset = half_index * 128
        for lane in range(32):
            scale_pair = lane // 16
            q1 = (
                (low[low_offset + lane] & 15)
                | (((high[high_offset + lane] >> 0) & 3) << 4)
            ) - 32
            q2 = (
                (low[low_offset + lane + 32] & 15)
                | (((high[high_offset + lane] >> 2) & 3) << 4)
            ) - 32
            q3 = (
                (low[low_offset + lane] >> 4)
                | (((high[high_offset + lane] >> 4) & 3) << 4)
            ) - 32
            q4 = (
                (low[low_offset + lane + 32] >> 4)
                | (((high[high_offset + lane] >> 6) & 3) << 4)
            ) - 32
            for group, quant in enumerate((q1, q2, q3, q4)):
                scale = scales[scale_offset + scale_pair + group * 2]
                values[output_offset + lane + group * 32] = f32(d * scale * quant)
    return values


def decode_q8(block: bytes) -> list[float]:
    (scale,) = struct.unpack_from("<e", block)
    quants = struct.unpack_from("<32b", block, 2)
    return [f32(scale * quant) for quant in quants]


def activation(index: int) -> float:
    return f32(((index * 37) % 101 - 50) / 32.0)


def scalar_dot(values: list[float]) -> float:
    total = 0.0
    for index, value in enumerate(values):
        total = f32(total + f32(value * activation(index)))
    return total


def fixture(name: str, kind: str, block: bytes) -> dict[str, object]:
    if kind == "q4_k":
        values = decode_q4(block)
    elif kind == "q6_k":
        values = decode_q6(block)
    else:
        values = decode_q8(block)
    return {
        "name": name,
        "kind": kind,
        "block_hex": block.hex(),
        "decoded_f32_le_hex": "".join(f32_bits(value) for value in values),
        "dot_f32_le_hex": f32_bits(scalar_dot(values)),
    }


def main() -> None:
    q4_pattern = (
        half(0.5)
        + half(0.25)
        + bytes(
            [0x00, 0x3F, 0xC0, 0xFF, 0x15, 0x2A, 0x95, 0xEA, 0xF0, 0x0F, 0xA5, 0x5A]
        )
        + bytes((index * 29 + 7) & 255 for index in range(128))
    )
    q4_zero = bytes(144)
    q6_pattern = (
        bytes((index * 17 + 3) & 255 for index in range(128))
        + bytes((index * 43 + 11) & 255 for index in range(64))
        + struct.pack(
            "<16b", -128, -64, -17, -1, 0, 1, 7, 16, 31, 63, 95, 127, -95, -31, 2, 48
        )
        + half(0.25)
    )
    q6_zero = bytes(208) + half(1.0)
    q8_pattern = half(0.25) + struct.pack(
        "<32b",
        -128,
        -127,
        -96,
        -64,
        -33,
        -32,
        -17,
        -1,
        0,
        1,
        2,
        7,
        15,
        16,
        31,
        32,
        47,
        63,
        64,
        79,
        95,
        96,
        111,
        126,
        127,
        -2,
        -7,
        -15,
        -48,
        -80,
        -112,
        48,
    )
    q8_zero = bytes(34)
    document = {
        "schema_version": 1,
        "authority": {
            "implementation": "llama.cpp GGML Q4_K/Q6_K block definitions and dequantizers",
            "revision": LLAMA_CPP_REVISION,
            "contract": "pins/quant_contract.json",
        },
        "activation": "a[i] = (((i * 37) % 101) - 50) / 32, rounded to FP32",
        "accumulation": "left-to-right FP32 multiply then FP32 add",
        "cases": [
            fixture("q4_packed_boundaries", "q4_k", q4_pattern),
            fixture("q4_all_zero", "q4_k", q4_zero),
            fixture("q6_signed_extremes", "q6_k", q6_pattern),
            fixture("q6_zero_scale", "q6_k", q6_zero),
            fixture("q8_signed_extremes", "q8_0", q8_pattern),
            fixture("q8_all_zero", "q8_0", q8_zero),
        ],
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
