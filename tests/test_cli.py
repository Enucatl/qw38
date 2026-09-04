from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
BRAND = "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast."


def test_cli_contract_and_beginner_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cli_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cli_smoke.json").read_text())
    assert contract["context_capacity"] == 131_072
    assert contract["production_binary"] == "build/cuda/qw38"
    assert contract["host_binary_scope"] == "validation_only"
    assert contract["generation_order"] == ["sample", "eval", "publish"]
    assert fixture["environment"]["model_sha256"] == contract["model_sha256"]
    assert fixture["result"]["passed"] is True
    assert fixture["result"]["visible_answer"] == "hello"
    assert fixture["result"]["restore_in_same_process"] is True

    handbook = (ROOT / "docs" / "56-interactive-text-cli.md").read_text().casefold()
    for term in [
        "terminal",
        "chat template",
        "token id",
        "encode",
        "decode",
        "common prefix",
        "sample",
        "eval",
        "atomic",
        "stop token",
        "checkpoint",
        "cuda",
        "codex",
        "proof boundary",
    ]:
        assert term in handbook
    readme = (ROOT / "README.md").read_text()
    assert "## Chat with Quartz now" in readme
    assert "./build/cuda/qw38 models/Qwen3.8-27B-Q4_K_M.gguf" in readme
    assert "--load checkpoints/chat.qw38 --save checkpoints/chat.qw38" in readme


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned GGUF is required")
def test_cuda_cli_generates_saves_and_restores() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    checkpoint = ROOT / "build" / "cli-smoke.qw38"
    checkpoint.unlink(missing_ok=True)
    common = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        IMAGE,
    ]
    build = subprocess.run(
        [*common, "make", "build/cuda/qw38"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    try:
        run = subprocess.run(
            [
                *common,
                "./build/cuda/qw38",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "--reasoning",
                "off",
                "--max-tokens",
                "8",
                "--temperature",
                "0",
                "--save",
                "build/cli-smoke.qw38",
            ],
            input="Reply with exactly: hello\n/load build/cli-smoke.qw38\n/quit\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert BRAND in run.stdout
        assert "assistant> hello" in run.stdout
        assert "restored> build/cli-smoke.qw38" in run.stdout
        assert run.stderr == ""
        assert checkpoint.read_bytes()[:8] == b"QW38CKP1"
        assert checkpoint.stat().st_size == 161_118_596
    finally:
        checkpoint.unlink(missing_ok=True)
