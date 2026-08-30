from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
FIXTURE = ROOT / "fixtures" / "real_scalar_token.json"


def run_token(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-real-scalar-token", str(MODEL), mode],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_token_crosses_all_layers_and_matches_structural_regression() -> None:
    fixture = json.loads(FIXTURE.read_text())
    result = run_token("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    assert output == fixture["expected"]
    assert output["layers_completed"] == "64"
    assert output["gdn_slots_mutated"] == "48"
    assert output["attention_slots_mutated"] == "16"
    assert output["frontier"] == "1"
    assert output["logit_count"] == "248320"
    assert output["prepared_values"] == "2645504"
    assert output["state_values"] == "39747584"
    assert output["workspace_values"] == "204161"


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_invalid_global_workspace_fails_before_any_layer_or_state_change() -> None:
    result = run_token("invalid_workspace")
    assert result.returncode == 1
    assert "state, workspace, or output are invalid" in result.stderr
    output = parse_output(result.stdout)
    assert output == {
        "state_unchanged": "1",
        "logits_untouched": "1",
        "frontier": "0",
        "layers_completed": "0",
    }
