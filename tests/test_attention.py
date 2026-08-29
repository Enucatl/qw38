from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
FIXTURES = ROOT / "fixtures" / "attention_ffn_authority.json"


def run_check(kind: str, layer: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), f"--check-{kind}", str(layer)],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def floats_from_hex(value: str) -> tuple[float, ...]:
    raw = bytes.fromhex(value)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def assert_metrics(
    actual_hex: str, expected_hex: str, tolerances: dict[str, float]
) -> None:
    actual = floats_from_hex(actual_hex)
    expected = floats_from_hex(expected_hex)
    errors = [abs(left - right) for left, right in zip(actual, expected)]
    relative = [
        error / max(abs(reference), 1e-6) for error, reference in zip(errors, expected)
    ]
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    numerator = sum(left * right for left, right in zip(actual, expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    assert len(actual) == len(expected)
    assert not any(math.isnan(value) or math.isinf(value) for value in actual)
    assert max(errors) <= tolerances["absolute"]
    assert max(relative) <= tolerances["relative"]
    assert rms <= tolerances["rms"]
    assert numerator / (actual_norm * expected_norm) >= tolerances["cosine_minimum"]


def test_attention_layers_3_7_63_meet_frozen_metrics() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    for case in fixtures["attention_cases"]:
        result = run_check("attention", case["layer"])
        assert result.returncode == 0, result.stderr
        actual = parse_output(result.stdout)
        for name in (
            "output_f32_le_hex",
            "key_cache_f32_le_hex",
            "value_cache_f32_le_hex",
        ):
            assert_metrics(actual[name], case[name], fixtures["tolerances"])


def test_attention_is_causal_despite_future_cache_sentinels() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    case = fixtures["attention_cases"][0]
    result = run_check("attention", case["layer"])
    first_token_values = floats_from_hex(
        parse_output(result.stdout)["output_f32_le_hex"]
    )[: 6 * 8]
    assert max(abs(value) for value in first_token_values) < 2.0


def test_ffn_gate_up_activation_and_down_taps_meet_frozen_metrics() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    for case in fixtures["ffn_cases"]:
        result = run_check("ffn", case["layer"])
        assert result.returncode == 0, result.stderr
        actual = parse_output(result.stdout)
        for name in (
            "gate_f32_le_hex",
            "up_f32_le_hex",
            "activated_f32_le_hex",
            "output_f32_le_hex",
        ):
            assert_metrics(actual[name], case[name], fixtures["tolerances"])


def test_attention_and_ffn_reject_non_attention_layer() -> None:
    attention = run_check("attention", 4)
    assert attention.returncode == 1
    assert "layer must be 3, 7, or 63" in attention.stderr

    ffn = run_check("ffn", 4)
    assert ffn.returncode == 1
    assert "layer must be 3, 7, or 63" in ffn.stderr
