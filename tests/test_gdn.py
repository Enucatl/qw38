from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
FIXTURES = ROOT / "fixtures" / "gdn_authority.json"


def run_gdn(component: str, chunking: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-gdn", component, chunking],
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
    assert len(actual) == len(expected)
    assert not any(math.isnan(value) or math.isinf(value) for value in actual)
    errors = [abs(left - right) for left, right in zip(actual, expected)]
    relative = [
        error / max(abs(reference), 1e-6) for error, reference in zip(errors, expected)
    ]
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    numerator = sum(left * right for left, right in zip(actual, expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    cosine = numerator / (actual_norm * expected_norm)
    assert max(errors) <= tolerances["absolute"]
    assert max(relative) <= tolerances["relative"]
    assert rms <= tolerances["rms"]
    assert cosine >= tolerances["cosine_minimum"]


def test_recurrent_oracle_meets_frozen_metrics() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    expected = fixtures["recurrent"]
    result = run_gdn("recurrent", "whole")
    assert result.returncode == 0, result.stderr
    actual = parse_output(result.stdout)
    for name in (
        "output_f32_le_hex",
        "final_state_f32_le_hex",
        "log_decay_f32_le_hex",
        "beta_f32_le_hex",
    ):
        assert_metrics(actual[name], expected[name], fixtures["tolerances"])


def test_convolution_warmup_and_state_meet_frozen_metrics() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    expected = fixtures["convolution"]
    result = run_gdn("convolution", "whole")
    assert result.returncode == 0, result.stderr
    actual = parse_output(result.stdout)
    assert_metrics(
        actual["output_f32_le_hex"],
        expected["output_f32_le_hex"],
        fixtures["tolerances"],
    )
    assert actual["final_state_f32_le_hex"] == expected["final_state_f32_le_hex"]


def test_arbitrary_chunks_equal_tokenwise_state_mutation_exactly() -> None:
    for component in ("recurrent", "convolution"):
        whole = run_gdn(component, "whole")
        mixed = run_gdn(component, "mixed")
        token = run_gdn(component, "token")
        assert whole.returncode == mixed.returncode == token.returncode == 0
        assert whole.stdout == mixed.stdout == token.stdout


def test_gdn_diagnostic_rejects_unknown_component_and_chunking() -> None:
    component = run_gdn("attention", "whole")
    assert component.returncode == 1
    assert "component must be recurrent or convolution" in component.stderr

    chunking = run_gdn("recurrent", "random")
    assert chunking.returncode == 1
    assert "unknown GDN chunking" in chunking.stderr


def test_gdn_core_rejects_invalid_head_mapping_and_buffer_counts() -> None:
    shape = run_gdn("invalid_shape", "unused")
    assert shape.returncode == 1
    assert "value heads must be an exact multiple of key heads" in shape.stderr

    buffers = run_gdn("invalid_buffers", "unused")
    assert buffers.returncode == 1
    assert "buffer counts do not match the declared shape" in buffers.stderr
