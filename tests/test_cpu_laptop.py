from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
MODEL = ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf"
CONTRACT = ROOT / "pins" / "cpu_laptop_contract.json"
INVENTORY = ROOT / "pins" / "cpu_tensor_inventory.json"


def test_cpu_laptop_contract_matches_geometry_and_handbook() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["not_v1_cuda"] is True
    assert contract["model_id"] == "qwen3.5-2b-q4_k_m"
    geometry = contract["geometry"]
    assert geometry["layers"] == 24
    assert geometry["gdn_layers"] == 18
    assert geometry["attention_layers"] == 6
    assert geometry["residual_width"] == 2048
    assert geometry["ffn_width"] == 6144
    assert geometry["tied_embeddings"] is True
    assert contract["host"]["default_context"] == 4096
    assert contract["host"]["maximum_context"] == 8192
    assert contract["host"]["required_isa"] == "AVX2"
    handbook = (ROOT / "docs" / "60-cpu-laptop-2b.md").read_text()
    for term in (
        "tied",
        "4096",
        "AVX2",
        "Qwen3.5-2B",
        "does not replace v1",
        "measured",
    ):
        assert term in handbook
    readme = (ROOT / "README.md").read_text()
    assert "Chat on a CPU MacBook" in readme
    assert "Qwen3.5-2B-Q4_K_M.gguf" in readme


def test_cpu_pin_is_recorded_without_committing_weights() -> None:
    pins = json.loads((ROOT / "pins" / "artifacts.lock.json").read_text())
    cpu = pins["cpu_model"]
    assert cpu["filename"] == "Qwen3.5-2B-Q4_K_M.gguf"
    assert cpu["quantization"] == "Q4_K_M"
    assert cpu["architecture"] == "qwen35"
    assert len(cpu["sha256"]) == 64
    assert cpu["bytes"] > 1_000_000_000
    gitignore = (ROOT / ".gitignore").read_text()
    assert "models/" in gitignore
    if MODEL.exists():
        assert MODEL.stat().st_size == cpu["bytes"]


@pytest.mark.skipif(not INVENTORY.exists(), reason="2B inventory is generated from the pinned GGUF")
def test_cpu_tensor_inventory_matches_tied_2b_contract() -> None:
    inventory = json.loads(INVENTORY.read_text())
    contract = json.loads(CONTRACT.read_text())
    pins = json.loads((ROOT / "pins" / "artifacts.lock.json").read_text())
    assert inventory["tensor_count"] == contract["geometry"]["expected_tensor_count_tied"]
    assert inventory["model_sha256"] == pins["cpu_model"]["sha256"]
    names = {tensor["name"] for tensor in inventory["tensors"]}
    assert "token_embd.weight" in names
    assert "output.weight" not in names
    assert all(tensor["role"] for tensor in inventory["tensors"])


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned 2B GGUF is required")
def test_host_engine_opens_2b_and_rejects_wrong_context() -> None:
    inspect = subprocess.run(
        [str(BUILD / "qw38-eval"), "--check-contract", str(MODEL)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert inspect.returncode == 0, inspect.stderr
    assert "contract=qwen3.5-2b-q4_k_m" in inspect.stdout
    help_text = subprocess.run(
        [str(BUILD / "qw38"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_text.returncode == 0
    assert "--ctx N" in help_text.stdout


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned 2B GGUF is required")
def test_host_cli_one_shot_is_optional_smoke() -> None:
    if os.environ.get("QW38_RUN_CPU_SMOKE") != "1":
        pytest.skip("set QW38_RUN_CPU_SMOKE=1 to run the slow 2B CLI smoke")
    result = subprocess.run(
        [
            str(BUILD / "qw38"),
            str(MODEL),
            "--reasoning",
            "off",
            "--temperature",
            "0",
            "--max-tokens",
            "8",
            "--ctx",
            "512",
            "--prompt",
            "Reply with exactly: hello",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "assistant>" in result.stdout
