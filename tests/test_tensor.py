from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_quant_fixtures import (  # noqa: E402
    decode_q4,
    decode_q6,
    decode_q8,
    f32,
)

EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
QUANT_FIXTURES = ROOT / "fixtures" / "quant_authority.json"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"
ROW_FIXTURES = ROOT / "fixtures" / "tensor_rows.json"


def floats_hex(values: list[float]) -> str:
    return b"".join(struct.pack("<f", value) for value in values).hex()


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def scalar_dot(weights: list[float], activation: list[float]) -> float:
    total = 0.0
    for weight, value in zip(weights, activation):
        total = f32(total + f32(weight * value))
    return total


def blocked_dot(
    weights: list[float], activation: list[float], block_values: int
) -> float:
    total = 0.0
    for offset in range(0, len(weights), block_values):
        partial = scalar_dot(
            weights[offset : offset + block_values],
            activation[offset : offset + block_values],
        )
        total = f32(total + partial)
    return total


def run_matvec(
    kind: str, columns: int, rows: int, payload: bytes, activation: list[float]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(EVAL),
            "--check-matvec",
            kind,
            str(columns),
            str(rows),
            payload.hex(),
            floats_hex(activation),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_synthetic_f32_matrix_uses_first_dimension_as_columns() -> None:
    rows = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [-1.0, 0.5, 2.0, -0.25, 8.0],
        [0.0, -2.0, 1.0, 3.0, -4.0],
    ]
    activation = [0.5, -1.0, 2.0, 0.25, -0.5]
    payload = b"".join(struct.pack("<f", value) for row in rows for value in row)
    result = run_matvec("f32", 5, 3, payload, activation)
    assert result.returncode == 0, result.stderr
    actual = parse_output(result.stdout)
    assert actual["columns"] == "5"
    assert actual["rows"] == "3"
    assert actual["row_bytes"] == "20"
    assert actual["row0_f32_le_hex"] == floats_hex(rows[0])
    assert actual["output_f32_le_hex"] == floats_hex(
        [scalar_dot(row, activation) for row in rows]
    )


def test_synthetic_quantized_multiblock_rows_decode_and_dot_exactly() -> None:
    fixtures = json.loads(QUANT_FIXTURES.read_text())["cases"]
    cases = {
        "q4_k": ("q4_packed_boundaries", "q4_all_zero", 256, decode_q4),
        "q6_k": ("q6_signed_extremes", "q6_zero_scale", 256, decode_q6),
        "q8_0": ("q8_signed_extremes", "q8_all_zero", 32, decode_q8),
    }
    by_name = {case["name"]: case for case in fixtures}
    for kind, (pattern_name, zero_name, block_values, decoder) in cases.items():
        pattern = bytes.fromhex(by_name[pattern_name]["block_hex"])
        zero = bytes.fromhex(by_name[zero_name]["block_hex"])
        pattern_values = decoder(pattern)
        zero_values = decoder(zero)
        rows = [pattern_values + zero_values, zero_values + pattern_values]
        activation = [
            f32(((index * 17) % 43 - 21) / 16.0) for index in range(block_values * 2)
        ]
        result = run_matvec(
            kind, block_values * 2, 2, pattern + zero + zero + pattern, activation
        )
        assert result.returncode == 0, f"{kind}: {result.stderr}"
        actual = parse_output(result.stdout)
        assert actual["row0_f32_le_hex"] == floats_hex(rows[0]), kind
        assert actual["output_f32_le_hex"] == floats_hex(
            [blocked_dot(row, activation, block_values) for row in rows]
        ), kind


def test_tensor_view_rejects_partial_rows_and_unaligned_blocks() -> None:
    activation = [0.0] * 256
    partial = run_matvec("q4_k", 256, 1, bytes(143), activation)
    assert partial.returncode == 1
    assert "exactly the declared rows" in partial.stderr

    unaligned = run_matvec("q8_0", 31, 1, bytes(34), [0.0] * 31)
    assert unaligned.returncode == 1
    assert "block-aligned row width" in unaligned.stderr

    wrong_activation = run_matvec("f32", 2, 1, struct.pack("<2f", 1.0, 2.0), [1.0])
    assert wrong_activation.returncode == 1
    assert "activation count is out of range" in wrong_activation.stderr


def decode_payload(kind: str, payload: bytes) -> tuple[list[float], int]:
    if kind == "Q4_K":
        return (
            [
                value
                for offset in range(0, len(payload), 144)
                for value in decode_q4(payload[offset : offset + 144])
            ],
            256,
        )
    if kind == "Q6_K":
        return (
            [
                value
                for offset in range(0, len(payload), 210)
                for value in decode_q6(payload[offset : offset + 210])
            ],
            256,
        )
    if kind == "Q8_0":
        return (
            [
                value
                for offset in range(0, len(payload), 34)
                for value in decode_q8(payload[offset : offset + 34])
            ],
            32,
        )
    return list(struct.unpack(f"<{len(payload) // 4}f", payload)), 1


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_admitted_artifact_rows_match_independent_mapped_bytes() -> None:
    inventory = json.loads(INVENTORY.read_text())
    by_name = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    fixtures = json.loads(ROW_FIXTURES.read_text())
    with MODEL.open("rb") as model:
        for case in fixtures["cases"]:
            name = case["name"]
            row = case["row"]
            tensor = by_name[name]
            columns, rows = tensor["shape"]
            row_bytes = tensor["storage_bytes"] // rows
            model.seek(tensor["absolute_offset"] + row * row_bytes)
            payload = model.read(row_bytes)
            assert hashlib.sha256(payload).hexdigest() == case["row_sha256"]
            weights, block_values = decode_payload(tensor["dtype"], payload)
            activation = [
                f32(((index * 37) % 101 - 50) / 32.0) for index in range(columns)
            ]
            expected = blocked_dot(weights, activation, block_values)
            assert floats_hex([expected]) == case["dot_f32_le_hex"]
            result = subprocess.run(
                [str(EVAL), "--check-tensor-row", str(MODEL), name, str(row)],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"{name}: {result.stderr}"
            actual = parse_output(result.stdout)
            assert actual["dtype"] == tensor["dtype"]
            assert actual["columns"] == str(columns)
            assert actual["rows"] == str(rows)
            assert actual["row_bytes"] == str(row_bytes)
            assert actual["dot_f32_le_hex"] == case["dot_f32_le_hex"], name


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_admitted_binding_rejects_unknown_rank_and_row() -> None:
    unknown = subprocess.run(
        [str(EVAL), "--check-tensor-row", str(MODEL), "missing.weight", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unknown.returncode == 1
    assert "tensor name is not admitted" in unknown.stderr

    rank = subprocess.run(
        [str(EVAL), "--check-tensor-row", str(MODEL), "output_norm.weight", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rank.returncode == 1
    assert "two-dimensional host matrix" in rank.stderr

    row = subprocess.run(
        [
            str(EVAL),
            "--check-tensor-row",
            str(MODEL),
            "blk.0.ssm_conv1d.weight",
            "10240",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert row.returncode == 1
    assert "row or activation count is out of range" in row.stderr
