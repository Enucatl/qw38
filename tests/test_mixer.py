from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"
FIXTURES = ROOT / "fixtures" / "mixer_projections.json"


def run_mixer(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-mixer-projections", str(MODEL), mode],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines())


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_complete_real_mixer_projections_match_independent_rows() -> None:
    fixture = json.loads(FIXTURES.read_text())
    result = run_mixer("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    for case in fixture["cases"]:
        assert output[case["name"]] == case["output_f32_le_hex"]
    assert output["attention_query_split_f32_le_hex"] == (
        output["blk.3.attn_q.weight"][0:16] + output["blk.3.attn_q.weight"][32:48]
    )
    assert output["attention_gate_split_f32_le_hex"] == (
        output["blk.3.attn_q.weight"][16:32] + output["blk.3.attn_q.weight"][-8:]
    )
    assert output["computed_values"] == "43104"


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_fixture_rows_still_identify_the_admitted_payloads() -> None:
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    fixture = json.loads(FIXTURES.read_text())
    with MODEL.open("rb") as model:
        for case in fixture["cases"]:
            tensor = tensors[case["name"]]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            for row, expected_hash in zip(
                case["rows"], case["row_sha256"], strict=True
            ):
                model.seek(tensor["absolute_offset"] + row * row_bytes)
                assert (
                    hashlib.sha256(model.read(row_bytes)).hexdigest() == expected_hash
                )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_mixer_projection_rejects_short_workspace_before_matvec() -> None:
    result = run_mixer("invalid_workspace")
    assert result.returncode == 1
    assert "workspace is invalid" in result.stderr
