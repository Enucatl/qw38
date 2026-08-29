from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"


def run_binding(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-weight-binding", str(MODEL), mode],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_all_admitted_weights_bind_to_the_complete_typed_schema() -> None:
    result = run_binding("valid")
    assert result.returncode == 0, result.stderr
    assert parse_output(result.stdout) == {
        "bound_tensors": "851",
        "gdn_layers": "48",
        "attention_layers": "16",
        "embedding_columns": "5120",
        "embedding_rows": "248320",
        "final_norm_values": "5120",
        "logit_rows": "248320",
        "final_norm_endpoints_f32_le_hex": "0000fb3f0000f13f",
    }


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing", "typed weight role"),
        ("role", "typed weight role"),
        ("shape", "typed weight role"),
        ("range", "exceeds the mapped artifact"),
    ],
)
def test_missing_wrong_role_shape_and_range_fail_closed(
    mode: str, message: str
) -> None:
    result = run_binding(mode)
    assert result.returncode == 1
    assert message in result.stderr
