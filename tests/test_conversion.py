from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"
FIXTURES = ROOT / "fixtures" / "gguf_conversion.json"


def run_conversion(component: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-conversion", component],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def floats_from_hex(value: str) -> tuple[float, ...]:
    raw = bytes.fromhex(value)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def test_grouped_tiled_permutation_and_roundtrip_are_exact() -> None:
    result = run_conversion("permutation")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    grouped = (
        0.0,
        1.0,
        10.0,
        11.0,
        20.0,
        21.0,
        100.0,
        101.0,
        110.0,
        111.0,
        120.0,
        121.0,
    )
    tiled = (0.0, 1.0, 100.0, 101.0, 10.0, 11.0, 110.0, 111.0, 20.0, 21.0, 120.0, 121.0)
    assert floats_from_hex(output["grouped_f32_le_hex"]) == grouped
    assert floats_from_hex(output["tiled_f32_le_hex"]) == tiled
    assert output["roundtrip_f32_le_hex"] == output["grouped_f32_le_hex"]


def test_source_and_folded_gguf_gate_math_are_bit_equal() -> None:
    result = run_conversion("gates")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    assert output["source_decay_f32_le_hex"] == output["gguf_decay_f32_le_hex"]
    assert output["source_beta_f32_le_hex"] == output["gguf_beta_f32_le_hex"]


def test_source_offset_and_gguf_scale_norms_are_bit_equal() -> None:
    result = run_conversion("norm")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    assert output["source_output_f32_le_hex"] == output["gguf_output_f32_le_hex"]


def test_nonnegative_folded_a_fails_closed() -> None:
    result = run_conversion("invalid_folded_a")
    assert result.returncode == 1
    assert "must be finite and negative" in result.stderr


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_admitted_converted_parameter_bytes_are_frozen() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    fixtures = json.loads(FIXTURES.read_text())
    with MODEL.open("rb") as model:
        for case in fixtures["cases"]:
            tensor = tensors[case["name"]]
            model.seek(tensor["absolute_offset"])
            payload = model.read(tensor["storage_bytes"])
            values = struct.unpack(f"<{len(payload) // 4}f", payload)
            assert hashlib.sha256(payload).hexdigest() == case["sha256"]
            assert payload[:32].hex() == case["first_8_f32_le_hex"]
            assert payload[-32:].hex() == case["last_8_f32_le_hex"]
            assert math.isclose(min(values), case["minimum"], rel_tol=0, abs_tol=0)
            assert math.isclose(max(values), case["maximum"], rel_tol=0, abs_tol=0)
    decay = next(case for case in fixtures["cases"] if case["name"] == "blk.0.ssm_a")
    convolution = next(
        case for case in fixtures["cases"] if case["name"] == "blk.0.ssm_conv1d.weight"
    )
    assert decay["maximum"] < 0.0
    assert convolution["shape"] == [4, 10240]
