from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
FIXTURE = ROOT / "fixtures" / "real_gdn_step.json"


def run_step(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-real-gdn-step", str(MODEL), mode],
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
    assert not any(not math.isfinite(value) for value in actual)
    errors = [abs(left - right) for left, right in zip(actual, expected, strict=True)]
    relative = [
        error / max(abs(reference), 1e-6)
        for error, reference in zip(errors, expected, strict=True)
    ]
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    tolerances = fixture["tolerances"]
    assert max(errors) <= tolerances["absolute"]
    assert max(relative) <= tolerances["relative"]
    assert rms <= tolerances["rms"]


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_layer_zero_gdn_state_and_taps_match_scalar_authority() -> None:
    fixture = json.loads(FIXTURE.read_text())
    result = run_step("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    for name in (
        "normalized_f32_le_hex",
        "convolved_f32_le_hex",
        "value_grouped_f32_le_hex",
        "gate_controls_f32_le_hex",
        "log_decay_f32_le_hex",
        "update_gate_f32_le_hex",
        "recurrent_output_f32_le_hex",
        "gated_grouped_f32_le_hex",
        "convolution_state_f32_le_hex",
        "recurrent_state_f32_le_hex",
    ):
        assert_metrics(output[name], fixture[name], fixture)


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_gdn_output_adds_the_mixer_correction_to_the_residual() -> None:
    result = run_step("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    mixer = floats_from_hex(output["mixer_output_f32_le_hex"])
    residual_output = floats_from_hex(output["residual_output_f32_le_hex"])
    indices = (0, 1, 2559, 5119)
    residual = [f32((((index * 37) % 101) - 50) / 32.0) for index in indices]
    assert residual_output == tuple(
        f32(left + right) for left, right in zip(residual, mixer, strict=True)
    )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_invalid_gdn_workspace_fails_before_mutating_state() -> None:
    result = run_step("invalid_workspace")
    assert result.returncode == 1
    assert "workspace are invalid" in result.stderr
    assert parse_output(result.stdout)["state_unchanged"] == "1"
