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
FIXTURE = ROOT / "fixtures" / "real_attention_step.json"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"


def run_step(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-real-attention-step", str(MODEL), mode],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def floats_from_hex(value: str) -> tuple[float, ...]:
    raw = bytes.fromhex(value)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def assert_metrics(
    actual_hex: str, expected_hex: str, fixture: dict[str, object]
) -> None:
    actual = floats_from_hex(actual_hex)
    expected = floats_from_hex(expected_hex)
    assert len(actual) == len(expected)
    assert all(math.isfinite(value) for value in actual)
    errors = [abs(left - right) for left, right in zip(actual, expected, strict=True)]
    relative = [
        error / max(abs(reference), 1e-6)
        for error, reference in zip(errors, expected, strict=True)
    ]
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    tolerances = fixture["tolerances"]
    assert isinstance(tolerances, dict)
    assert max(errors) <= tolerances["absolute"]
    assert max(relative) <= tolerances["relative"]
    assert rms <= tolerances["rms"]


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_layer_three_attention_and_kv_taps_match_authority() -> None:
    fixture = json.loads(FIXTURE.read_text())
    result = run_step("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    for name in (
        "normalized_f32_le_hex",
        "query_f32_le_hex",
        "gate_f32_le_hex",
        "key_cache_f32_le_hex",
        "value_cache_f32_le_hex",
        "attention_output_f32_le_hex",
    ):
        assert_metrics(output[name], fixture[name], fixture)
    assert output["kv_values"] == "4096"


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_attention_fixture_names_the_exact_physical_rows() -> None:
    fixture = json.loads(FIXTURE.read_text())
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    with MODEL.open("rb") as model:
        for name, rows in fixture["row_sha256"].items():
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            for row_text, expected_hash in rows.items():
                row = int(row_text)
                model.seek(tensor["absolute_offset"] + row * row_bytes)
                assert (
                    hashlib.sha256(model.read(row_bytes)).hexdigest() == expected_hash
                )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_attention_output_adds_the_projection_to_token_one() -> None:
    result = run_step("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    correction = floats_from_hex(output["mixer_output_f32_le_hex"])
    residual_output = floats_from_hex(output["residual_output_f32_le_hex"])
    indices = (0, 1, 2559, 5119)
    residual = [f32(((((index * 37 + 17) % 101) - 50) / 32.0)) for index in indices]
    assert residual_output == tuple(
        f32(left + right) for left, right in zip(residual, correction, strict=True)
    )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize("mode", ["invalid_workspace", "capacity"])
def test_real_attention_rejects_before_mutating_kv(mode: str) -> None:
    result = run_step(mode)
    assert result.returncode == 1
    assert "state, or workspace are invalid" in result.stderr
    output = parse_output(result.stdout)
    assert output["state_unchanged"] == "1"
    assert output["output_untouched"] == "1"
