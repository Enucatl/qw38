from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"


def test_atomic_eval_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "cuda_atomic_eval_contract.json").read_text()
    )
    fixture = json.loads((ROOT / "fixtures" / "cuda_atomic_eval.json").read_text())
    assert contract["transaction"]["unit"] == "one token"
    assert contract["transaction"]["publication_order"][-1] == "frontier last"
    assert contract["sampling"]["mutation"] == "none"
    assert fixture["workspace_bytes_at_capacity_3"] == 186_300_192
    assert all(case["passed"] for case in fixture["cases"])
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    chapter = (ROOT / "docs" / "48-atomic-eval-and-sampling.md").read_text().casefold()
    for term in [
        "transaction",
        "committed state",
        "candidate state",
        "pointer swap",
        "frontier last",
        "cancellation",
        "injected error",
        "sampling",
        "read-only",
        "186.30 mb",
        "proof boundary",
    ]:
        assert term in chapter


def test_atomic_eval_cancel_error_commit_and_sample() -> None:
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
        [*common, "make", "build/qw38-cuda-atomic-eval-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-atomic-eval-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    cases = [line for line in lines if line.startswith("atomic_case=")]
    assert len(cases) == 5
    assert all("passed=true" in line for line in cases)
    assert (
        "atomic_case=separate_sampling token=1277 frontier=2 state_equal=true passed=true"
        in lines
    )
    assert (
        "atomic_run=complete workspace_bytes=186300192 frontier=2 passed=true" in lines
    )
    assert "status=passed" in lines
