from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tools.capture_scalar_trace import capture_scalar_trace
from tools.qw38_trace import TraceError, read_trace_bundle

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
DIAGNOSTIC_EVAL = ROOT / "build" / "qw38-eval-diagnostic"
FIXTURE = ROOT / "fixtures" / "real_scalar_token.json"


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_filtered_real_final_norm_round_trips_through_trace_v1(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "trace"
    capture_scalar_trace(
        MODEL,
        DIAGNOSTIC_EVAL,
        destination,
        token=42,
        layer=None,
        tap="final_norm",
    )
    loaded = read_trace_bundle(destination)
    fixture = json.loads(FIXTURE.read_text())
    expected_raw = bytes.fromhex(fixture["expected"]["final_normalized_f32_le_hex"])
    expected = struct.unpack(f"<{len(expected_raw) // 4}f", expected_raw)
    actual = loaded.tensors["final_norm"]

    assert tuple(actual[index] for index in (0, 1, 2559, 5119)) == expected
    assert loaded.manifest["model"]["sha256"] == fixture["model_sha256"]
    assert loaded.manifest["prompt"]["token_ids"] == [42]
    assert loaded.manifest["prompt"]["positions"] == [0]
    assert loaded.manifest["session"]["before"]["frontier"] == 0
    assert loaded.manifest["session"]["after"]["frontier"] == 1
    assert loaded.manifest["tensors"][0]["shape"] == [5120]
    assert loaded.manifest["tensors"][0]["layer"] is None


def test_capture_rejects_unpinned_model_before_execution(tmp_path: Path) -> None:
    wrong_model = tmp_path / "wrong.gguf"
    wrong_model.write_bytes(b"not the pinned model")

    with pytest.raises(TraceError, match="pinned trace identity"):
        capture_scalar_trace(
            wrong_model,
            DIAGNOSTIC_EVAL,
            tmp_path / "trace",
            token=42,
            layer=None,
            tap="final_norm",
        )
