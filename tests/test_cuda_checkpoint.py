from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"


def test_checkpoint_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_checkpoint_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_checkpoint.json").read_text())
    assert contract["format"]["magic"] == "QW38CKP1"
    assert contract["format"]["header_bytes"] == 248
    assert fixture["checkpoint_bytes"] == sum(fixture["section_bytes"].values())
    assert fixture["checkpoint_bytes"] == 160_004_416
    assert all(case["passed"] for case in fixture["cases"])
    chapter = (ROOT / "docs" / "49-cuda-checkpoints.md").read_text().casefold()
    for term in [
        "checkpoint",
        "little-endian",
        "compatibility hash",
        "payload digest",
        "temporary file",
        "fsync",
        "rename",
        "sampler",
        "committed kv rows",
        "corrupt",
        "exact continuation",
        "proof boundary",
    ]:
        assert term in chapter


def test_checkpoint_round_trip_and_exact_continuation() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    if not MODEL.exists():
        pytest.skip("the pinned GGUF is required")
    common = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        IMAGE,
    ]
    build = subprocess.run(
        [*common, "make", "build/qw38-cuda-checkpoint-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-checkpoint-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            "build/checkpoint-test.bin",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    cases = [line for line in lines if line.startswith("checkpoint_case=")]
    assert len(cases) == 4
    assert all("passed=true" in line for line in cases)
    assert (
        "checkpoint_case=round_trip bytes=160004416 frontier=2 published=true state_equal=true passed=true"
        in lines
    )
    assert "checkpoint_run=complete frontier=3 passed=true" in lines
    assert "status=passed" in lines
