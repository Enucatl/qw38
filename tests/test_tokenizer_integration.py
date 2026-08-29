from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
EVAL = ROOT / "build" / "qw38-eval"


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_native_tokenizer_matches_every_authority_fixture() -> None:
    fixtures = json.loads((ROOT / "fixtures" / "tokenizer_authority.json").read_text())
    for case in fixtures["cases"]:
        result = subprocess.run(
            [str(EVAL), "--tokenize-hex", str(MODEL), case["utf8_hex"]],
            check=False,
            capture_output=True,
            text=True,
        )
        actual = (
            []
            if not result.stdout.strip()
            else [int(token_id) for token_id in result.stdout.strip().split(",")]
        )
        assert result.returncode == 0, f"{case['name']}: {result.stderr}"
        assert actual == case["ids"], case["name"]


@pytest.mark.skipif(not MODEL.exists(), reason="pinned runtime GGUF is not installed")
def test_native_tokenizer_rejects_invalid_utf8() -> None:
    result = subprocess.run(
        [str(EVAL), "--tokenize-hex", str(MODEL), "ff"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "invalid_argument: invalid UTF-8" in result.stderr
