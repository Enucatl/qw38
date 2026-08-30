from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
EVAL = ROOT / "build" / "qw38-eval"
FIXTURE = ROOT / "fixtures" / "real_scalar_chunk.json"


def run_chunk(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-real-scalar-chunk", str(MODEL), mode],
        check=False,
        capture_output=True,
        text=True,
    )


def fields(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_two_token_chunk_exactly_matches_repeated_token_execution() -> None:
    fixture = json.loads(FIXTURE.read_text())
    result = run_chunk("valid")

    assert result.returncode == 0, result.stderr
    assert fields(result.stdout) == {
        key: str(value) for key, value in fixture["expected"].items()
    }


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize(
    "mode", ["invalid_token", "insufficient_capacity", "invalid_logits"]
)
def test_chunk_preflight_failure_does_not_mutate_state_or_logits(mode: str) -> None:
    result = run_chunk(mode)

    assert result.returncode == 1
    assert fields(result.stdout) == {
        "state_unchanged": "1",
        "logits_untouched": "1",
        "frontier": "0",
        "layers_completed": "0",
    }
    assert "invalid_argument:" in result.stderr
