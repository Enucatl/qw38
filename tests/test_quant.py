from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
FIXTURES = ROOT / "fixtures" / "quant_authority.json"


def run_quant(kind: str, block_hex: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-quant", kind, block_hex],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


def floats_from_hex(value: str) -> tuple[float, ...]:
    raw = bytes.fromhex(value)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def test_native_quant_matches_every_frozen_value_and_dot_bit() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    for case in fixtures["cases"]:
        result = run_quant(case["kind"], case["block_hex"])
        assert result.returncode == 0, f"{case['name']}: {result.stderr}"
        actual = parse_output(result.stdout)
        assert actual["decoded_f32_le_hex"] == case["decoded_f32_le_hex"], case["name"]
        assert actual["dot_f32_le_hex"] == case["dot_f32_le_hex"], case["name"]


def test_exact_fixtures_imply_frozen_zero_error_metrics() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    for case in fixtures["cases"]:
        result = run_quant(case["kind"], case["block_hex"])
        actual = floats_from_hex(parse_output(result.stdout)["decoded_f32_le_hex"])
        expected = floats_from_hex(case["decoded_f32_le_hex"])
        differences = [abs(left - right) for left, right in zip(actual, expected)]
        assert max(differences) == 0.0
        assert (
            math.sqrt(sum(value * value for value in differences) / len(differences))
            == 0.0
        )
        assert not any(math.isnan(value) or math.isinf(value) for value in actual)
        if any(expected):
            numerator = sum(left * right for left, right in zip(actual, expected))
            left_norm = math.sqrt(sum(value * value for value in actual))
            right_norm = math.sqrt(sum(value * value for value in expected))
            assert math.isclose(
                numerator / (left_norm * right_norm),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )


def test_quant_diagnostic_rejects_wrong_sizes_kinds_and_hex() -> None:
    wrong_size = run_quant("q4_k", "00" * 143)
    assert wrong_size.returncode == 1
    assert "quant block has the wrong byte size" in wrong_size.stderr

    wrong_kind = run_quant("q5_k", "00" * 144)
    assert wrong_kind.returncode == 1
    assert "quant kind must be q4_k, q6_k, or q8_0" in wrong_kind.stderr

    invalid_hex = run_quant("q4_k", "not-hex")
    assert invalid_hex.returncode == 1
    assert "not even-length hexadecimal" in invalid_hex.stderr
