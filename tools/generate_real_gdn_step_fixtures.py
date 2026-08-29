from __future__ import annotations

import ctypes
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
OUTPUT = ROOT / "fixtures" / "real_gdn_step.json"
LIBM = ctypes.CDLL("libm.so.6")
for function_name in ("expf", "log1pf", "sqrtf"):
    function = getattr(LIBM, function_name)
    function.restype = ctypes.c_float
    function.argtypes = [ctypes.c_float]


def unary(name: str, value: float) -> float:
    return float(getattr(LIBM, name)(ctypes.c_float(value)))


def add(left: float, right: float) -> float:
    return f32(left + right)


def multiply(left: float, right: float) -> float:
    return f32(left * right)


def vector_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def rms_scale(values: list[float], scale: list[float]) -> list[float]:
    total = 0.0
    for value in values:
        total = add(total, multiply(value, value))
    mean = f32(total / len(values))
    inverse = f32(1.0 / unary("sqrtf", add(mean, f32(1e-6))))
    return [
        multiply(multiply(value, inverse), weight)
        for value, weight in zip(values, scale)
    ]


def l2(values: list[float], query: bool) -> list[float]:
    total = 0.0
    for value in values:
        total = add(total, multiply(value, value))
    inverse = f32(1.0 / unary("sqrtf", add(total, f32(1e-6))))
    if query:
        inverse = f32(inverse / unary("sqrtf", 128.0))
    return [multiply(value, inverse) for value in values]


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = unary("expf", -value)
        return f32(1.0 / add(1.0, exponential))
    exponential = unary("expf", value)
    return f32(exponential / add(1.0, exponential))


def softplus(value: float) -> float:
    if value > 20.0:
        return f32(value)
    if value < -20.0:
        return unary("expf", value)
    return unary("log1pf", unary("expf", value))


def silu(value: float) -> float:
    return f32(value / add(1.0, unary("expf", -value)))


def main() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    with MODEL.open("rb") as model:

        def vector(name: str) -> list[float]:
            tensor = tensors[name]
            model.seek(tensor["absolute_offset"])
            payload = model.read(tensor["storage_bytes"])
            return list(struct.unpack(f"<{len(payload) // 4}f", payload))

        def projection(name: str, row: int, activation: list[float]) -> float:
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            model.seek(tensor["absolute_offset"] + row * row_bytes)
            weights, block_values = decode_blocks(
                tensor["dtype"], model.read(row_bytes)
            )
            return blocked_dot(weights, activation, block_values)

        def convolved(channel: int, activation: list[float]) -> tuple[float, float]:
            projected = projection("blk.0.attn_qkv.weight", channel, activation)
            tensor = tensors["blk.0.ssm_conv1d.weight"]
            model.seek(tensor["absolute_offset"] + channel * 16)
            weights = struct.unpack("<4f", model.read(16))
            convolution = multiply(projected, weights[3])
            return silu(convolution), projected

        residual = [f32(((index * 37) % 101 - 50) / 32.0) for index in range(5120)]
        normalized = rms_scale(residual, vector("blk.0.attn_norm.weight"))
        selected_channels = [0, 127, 2048, 2175, 4096, 4223, 6144, 6271, 4224, 4351]
        convolved_taps = [
            convolved(channel, normalized)[0] for channel in selected_channels
        ]

        folded_tiled = vector("blk.0.ssm_a")
        dt_tiled = vector("blk.0.ssm_dt.bias")
        norm_weight = vector("blk.0.ssm_norm.weight")
        grouped_heads = [0, 1, 3]
        recurrent_outputs: dict[int, list[float]] = {}
        gated_outputs: dict[int, list[float]] = {}
        states: dict[int, list[float]] = {}
        gate_controls: dict[int, float] = {}
        log_decays: dict[int, float] = {}
        update_gates: dict[int, float] = {}
        values: dict[int, list[float]] = {}

        for grouped_head in grouped_heads:
            key_head = grouped_head // 3
            replica = grouped_head % 3
            tiled_head = replica * 16 + key_head
            query = l2(
                [
                    convolved(key_head * 128 + lane, normalized)[0]
                    for lane in range(128)
                ],
                True,
            )
            key = l2(
                [
                    convolved(2048 + key_head * 128 + lane, normalized)[0]
                    for lane in range(128)
                ],
                False,
            )
            value = [
                convolved(4096 + tiled_head * 128 + lane, normalized)[0]
                for lane in range(128)
            ]
            gate = [
                projection(
                    "blk.0.attn_gate.weight", tiled_head * 128 + lane, normalized
                )
                for lane in range(128)
            ]
            alpha = projection("blk.0.ssm_alpha.weight", tiled_head, normalized)
            beta_raw = projection("blk.0.ssm_beta.weight", tiled_head, normalized)
            update = sigmoid(beta_raw)
            decay = multiply(
                folded_tiled[tiled_head], softplus(add(alpha, dt_tiled[tiled_head]))
            )
            output_values: list[float] = []
            state_values: list[float] = []
            for key_lane in range(128):
                for value_lane in range(128):
                    delta = multiply(value[value_lane], update)
                    state_values.append(multiply(key[key_lane], delta))
            for value_lane in range(128):
                result = 0.0
                for key_lane in range(128):
                    state_value = state_values[key_lane * 128 + value_lane]
                    result = add(result, multiply(query[key_lane], state_value))
                output_values.append(result)
            gated = rms_scale(output_values, norm_weight)
            gated = [
                multiply(value, silu(gate_value))
                for value, gate_value in zip(gated, gate)
            ]
            values[grouped_head] = value
            gate_controls[grouped_head] = alpha
            log_decays[grouped_head] = decay
            update_gates[grouped_head] = update
            recurrent_outputs[grouped_head] = output_values
            gated_outputs[grouped_head] = gated
            states[grouped_head] = state_values

        fixture = {
            "schema_version": 1,
            "model_sha256": MODEL_SHA256,
            "authority": "pinned Transformers scalar equations plus independently decoded GGUF rows",
            "tolerances": {"absolute": 3e-6, "relative": 3e-5, "rms": 1e-6},
            "normalized_f32_le_hex": vector_hex(
                [normalized[0], normalized[1], normalized[5118], normalized[5119]]
            ),
            "convolved_f32_le_hex": vector_hex(convolved_taps),
            "value_grouped_f32_le_hex": vector_hex(
                [
                    values[0][0],
                    values[0][127],
                    values[1][0],
                    values[1][127],
                    values[3][0],
                    values[3][127],
                ]
            ),
            "gate_controls_f32_le_hex": vector_hex(
                [
                    gate_controls[0],
                    gate_controls[1],
                    gate_controls[3],
                    projection("blk.0.ssm_alpha.weight", 47, normalized),
                ]
            ),
            "log_decay_f32_le_hex": vector_hex(
                [
                    log_decays[0],
                    log_decays[1],
                    log_decays[3],
                    multiply(
                        folded_tiled[47],
                        softplus(
                            add(
                                projection("blk.0.ssm_alpha.weight", 47, normalized),
                                dt_tiled[47],
                            )
                        ),
                    ),
                ]
            ),
            "update_gate_f32_le_hex": vector_hex(
                [
                    update_gates[0],
                    update_gates[1],
                    update_gates[3],
                    sigmoid(projection("blk.0.ssm_beta.weight", 47, normalized)),
                ]
            ),
            "recurrent_output_f32_le_hex": vector_hex(
                [
                    recurrent_outputs[0][0],
                    recurrent_outputs[0][127],
                    recurrent_outputs[1][0],
                    recurrent_outputs[1][127],
                    recurrent_outputs[3][0],
                    recurrent_outputs[3][127],
                ]
            ),
            "gated_grouped_f32_le_hex": vector_hex(
                [
                    gated_outputs[0][0],
                    gated_outputs[0][127],
                    gated_outputs[1][0],
                    gated_outputs[1][127],
                    gated_outputs[3][0],
                    gated_outputs[3][127],
                ]
            ),
            "convolution_state_f32_le_hex": vector_hex(
                [
                    0.0,
                    0.0,
                    0.0,
                    convolved(0, normalized)[1],
                    0.0,
                    0.0,
                    0.0,
                    convolved(4096, normalized)[1],
                ]
            ),
            "recurrent_state_f32_le_hex": vector_hex(
                [
                    states[0][0],
                    states[0][127],
                    states[0][16383],
                    states[1][0],
                    states[1][16383],
                    states[3][0],
                    states[3][16383],
                ]
            ),
        }
    OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
