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


def test_prefix_sync_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "cuda_prefix_sync_contract.json").read_text()
    )
    fixture = json.loads((ROOT / "fixtures" / "cuda_prefix_sync.json").read_text())
    assert contract["history"]["maximum_tokens"] == 131_072
    assert contract["session_owned_outputs"]["logits"]["values"] == 248_320
    assert contract["session_owned_outputs"]["hidden"]["values"] == 5_120
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert [case["name"] for case in fixture["cases"]] == [
        "initial",
        "append",
        "no_op",
        "divergent",
        "shorter",
        "empty_reset",
    ]
    assert all(case["passed"] for case in fixture["cases"])
    assert all(case["passed"] for case in fixture["exact_comparisons"])
    assert fixture["invalid_input"]["passed"]
    chapter = (ROOT / "docs" / "47-cuda-prefix-sync.md").read_text().casefold()
    for term in [
        "token history",
        "common prefix",
        "no-op",
        "append",
        "diverge",
        "shorter",
        "159 mb",
        "byte-exact",
        "preflight",
        "fixture equality",
        "proof boundary",
    ]:
        assert term in chapter


def test_prefix_sync_matches_fresh_execution_exactly() -> None:
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
        [*common, "make", "build/qw38-cuda-prefix-sync-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-prefix-sync-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    cases = [line for line in lines if line.startswith("prefix_case=")]
    exact = [line for line in lines if line.startswith("prefix_exact=")]
    assert len(cases) == 6
    assert len(exact) == 4
    assert all("passed=true" in line for line in cases + exact)
    assert (
        "prefix_invalid=rejected_before_mutation rejected=true "
        "state_equal=true passed=true" in lines
    )
    assert "prefix_run=complete frontier=0 token_count=0 passed=true" in lines
    assert "status=passed" in lines
