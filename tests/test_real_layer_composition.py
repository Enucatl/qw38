from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
FIXTURE = ROOT / "fixtures" / "real_layer_composition.json"
COMMANDS = {
    "gdn_layer_0_position_0": "--check-real-gdn-step",
    "attention_layer_3_position_1": "--check-real-attention-step",
}


def run_step(command: str, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), command, str(MODEL), mode],
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


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize("case_name", list(COMMANDS))
def test_complete_layer_composition_matches_frozen_regression(case_name: str) -> None:
    fixture = json.loads(FIXTURE.read_text())
    result = run_step(COMMANDS[case_name], "layer")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    for name, expected in fixture["cases"][case_name].items():
        assert output[name] == expected


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize("case_name", list(COMMANDS))
def test_complete_layer_feeds_the_mixer_residual_to_its_ffn(case_name: str) -> None:
    command = COMMANDS[case_name]
    mixer = run_step(command, "valid")
    layer = run_step(command, "layer")
    assert mixer.returncode == 0, mixer.stderr
    assert layer.returncode == 0, layer.stderr
    assert (
        parse_output(layer.stdout)["post_mixer_f32_le_hex"]
        == parse_output(mixer.stdout)["residual_output_f32_le_hex"]
    )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize("case_name", list(COMMANDS))
def test_complete_layer_adds_ffn_correction_to_post_mixer(case_name: str) -> None:
    result = run_step(COMMANDS[case_name], "layer")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    post_mixer = floats_from_hex(output["post_mixer_f32_le_hex"])
    correction = floats_from_hex(output["ffn_correction_f32_le_hex"])
    layer_output = floats_from_hex(output["residual_output_f32_le_hex"])
    assert layer_output == tuple(
        f32(left + right) for left, right in zip(post_mixer, correction, strict=True)
    )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize("command", list(COMMANDS.values()))
def test_invalid_ffn_fails_before_mixer_state_mutation(command: str) -> None:
    result = run_step(command, "layer_invalid_ffn")
    assert result.returncode == 1
    assert "FFN parameters, activation, or workspace are invalid" in result.stderr
    output = parse_output(result.stdout)
    assert output["state_unchanged"] == "1"
    assert output["output_untouched"] == "1"
