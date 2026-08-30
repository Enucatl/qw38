from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
FIXTURE = ROOT / "fixtures" / "real_model_boundaries.json"
INVENTORY = ROOT / "pins" / "tensor_inventory.json"


def run_boundary(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EVAL), "--check-real-model-boundaries", str(MODEL), mode],
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
    actual_hex: str, expected_hex: str, fixture: dict[str, object]
) -> None:
    actual = floats_from_hex(actual_hex)
    expected = floats_from_hex(expected_hex)
    assert len(actual) == len(expected)
    assert all(math.isfinite(value) for value in actual)
    errors = [abs(left - right) for left, right in zip(actual, expected, strict=True)]
    relative = [
        error / max(abs(reference), 1e-6)
        for error, reference in zip(errors, expected, strict=True)
    ]
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    tolerances = fixture["tolerances"]
    assert isinstance(tolerances, dict)
    assert max(errors) <= tolerances["absolute"]
    assert max(relative) <= tolerances["relative"]
    assert rms <= tolerances["rms"]


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_real_embeddings_final_norm_and_selected_logits_match_authority() -> None:
    fixture = json.loads(FIXTURE.read_text())
    result = run_boundary("valid")
    assert result.returncode == 0, result.stderr
    output = parse_output(result.stdout)
    for name in (
        "embedding_0_f32_le_hex",
        "embedding_42_f32_le_hex",
        "embedding_last_f32_le_hex",
        "final_normalized_f32_le_hex",
        "logits_f32_le_hex",
    ):
        assert_metrics(output[name], fixture[name], fixture)
    assert output["logit_count"] == "248320"
    assert 0 <= int(output["greedy_token"]) < 248320


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_model_boundary_fixture_names_exact_embedding_and_output_rows() -> None:
    fixture = json.loads(FIXTURE.read_text())
    inventory = json.loads(INVENTORY.read_text())
    tensors = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    with MODEL.open("rb") as model:
        for name, rows in fixture["row_sha256"].items():
            tensor = tensors[name]
            row_bytes = tensor["storage_bytes"] // tensor["shape"][1]
            for row_text, expected_hash in rows.items():
                row = int(row_text)
                model.seek(tensor["absolute_offset"] + row * row_bytes)
                assert (
                    hashlib.sha256(model.read(row_bytes)).hexdigest() == expected_hash
                )


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_out_of_range_token_fails_before_embedding_write() -> None:
    result = run_boundary("invalid_token")
    assert result.returncode == 1
    assert "token ID or embedding output is invalid" in result.stderr
    assert parse_output(result.stdout)["embedding_untouched"] == "1"


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_invalid_output_workspace_fails_before_norm_or_logits_write() -> None:
    result = run_boundary("invalid_workspace")
    assert result.returncode == 1
    assert "output workspace, or logits are invalid" in result.stderr
    output = parse_output(result.stdout)
    assert output["normalized_untouched"] == "1"
    assert output["logits_untouched"] == "1"
