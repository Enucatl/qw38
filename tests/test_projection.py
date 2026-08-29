from __future__ import annotations

import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"


def run_layout(component: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-projection-layout", component],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for line in stdout.splitlines():
        name, value = line.split("=", 1)
        raw = bytes.fromhex(value)
        result[name] = struct.unpack(f"<{len(raw) // 4}f", raw)
    return result


def test_gdn_projection_uses_three_global_contiguous_ranges() -> None:
    result = run_layout("gdn")
    assert result.returncode == 0, result.stderr
    assert parse_output(result.stdout) == {
        "query_f32_le_hex": (0.0, 1.0, 2.0, 3.0),
        "key_f32_le_hex": (10.0, 11.0, 12.0, 13.0),
        "value_f32_le_hex": (20.0, 21.0, 22.0, 23.0),
    }


def test_attention_projection_splits_query_and_gate_inside_each_head() -> None:
    result = run_layout("attention")
    assert result.returncode == 0, result.stderr
    assert parse_output(result.stdout) == {
        "query_f32_le_hex": (0.0, 1.0, 100.0, 101.0, 200.0, 201.0),
        "gate_f32_le_hex": (10.0, 11.0, 110.0, 111.0, 210.0, 211.0),
    }


def test_projection_split_rejects_aliases_and_wrong_counts() -> None:
    alias = run_layout("invalid_alias")
    assert alias.returncode == 1
    assert "must not overlap" in alias.stderr

    count = run_layout("invalid_count")
    assert count.returncode == 1
    assert "dimensions or counts are invalid" in count.stderr
