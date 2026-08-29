from __future__ import annotations

import json
import hashlib
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BRAND = "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast."


def run_binary(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BUILD / name), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_all_product_binaries_exist_and_fail_closed() -> None:
    for name in ("qw38", "qw38-server", "qw38-bench"):
        result = run_binary(name)
        assert result.returncode == 2
        assert BRAND in result.stdout
        assert "has not passed its delivery gate" in result.stderr


def test_eval_reports_pinned_build_identity() -> None:
    pins = json.loads((ROOT / "pins" / "artifacts.lock.json").read_text())
    result = run_binary("qw38-eval", "--build-info")
    assert result.returncode == 0
    assert f"brand={BRAND}" in result.stdout
    assert "cxx=17" in result.stdout
    assert "cuda_target=sm_120" in result.stdout
    assert f"model_revision={pins['model']['revision']}" in result.stdout
    assert f"model_sha256={pins['model']['sha256']}" in result.stdout


def test_cuda_container_is_immutable_and_sm120_is_declared() -> None:
    dockerfile = (ROOT / "docker" / "cuda.Dockerfile").read_text()
    makefile = (ROOT / "Makefile").read_text()
    pins = json.loads((ROOT / "pins" / "artifacts.lock.json").read_text())
    digest = pins["containers"]["cuda_devel"]["linux_amd64_digest"]
    assert f"@{digest}" in dockerfile
    assert "cuda_target=sm_120" in (ROOT / "src" / "eval.cpp").read_text()
    assert "-fno-exceptions -fno-rtti" in makefile
    assert "-arch=sm_120" in makefile


def test_model_contract_has_exact_hybrid_schedule() -> None:
    contract = json.loads((ROOT / "pins" / "model_contract.json").read_text())
    text = contract["text"]
    assert text["layers"] == 64
    assert len(text["layer_schedule"]) == 64
    assert text["layer_schedule"].count("gdn") == 48
    assert text["layer_schedule"].count("attention") == 16
    assert [
        index
        for index, layer in enumerate(text["layer_schedule"])
        if layer == "attention"
    ] == list(range(3, 64, 4))
    assert text["residual_width"] == 5120
    assert text["recurrence_dtype"] == "float32"
    assert text["partial_rotary_factor"] == 0.25


def test_native_gguf_inspector_rejects_malformed_input(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.gguf"
    malformed.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
    result = run_binary("qw38-eval", "--inspect-gguf", str(malformed))
    assert result.returncode == 1
    assert "incompatible_artifact: malformed GGUF" in result.stderr


def test_native_sha256_matches_standard_library(tmp_path: Path) -> None:
    fixture = tmp_path / "sha.fixture"
    payload = bytes(range(256)) * 1000 + b"quartz-watch-38"
    fixture.write_bytes(payload)
    result = run_binary("qw38-eval", "--sha256", str(fixture))
    assert result.returncode == 0
    assert result.stdout.strip() == hashlib.sha256(payload).hexdigest()
