from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"


def test_cuda_gdn_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_gdn_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_gdn_step.json").read_text())
    assert contract["target"] == "sm_120"
    assert contract["production_shape"]["recurrent_state_values"] == 786_432
    assert contract["admission"]["maximum_absolute_error"] == 5.0e-8
    assert contract["admission"]["maximum_rms_error"] == 5.0e-9
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert [case["name"] for case in fixture["cases"]] == ["small", "production"]
    assert all(
        case["prepare_atomic"] and case["commit_exact"] and case["nonfinite"] == 0
        for case in fixture["cases"]
    )
    chapter = (ROOT / "docs" / "41-cuda-gdn-step.md").read_text().casefold()
    for term in [
        "candidate state",
        "committed state",
        "frontier",
        "causal convolution",
        "delta rule",
        "cancellation",
    ]:
        assert term in chapter


def test_cuda_gdn_matches_scalar_and_commits_atomically() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
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
        [*common, "make", "build/qw38-cuda-gdn-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*common, "./build/qw38-cuda-gdn-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = [line for line in run.stdout.splitlines() if line.startswith("gdn_case=")]
    assert [line.split()[0] for line in lines] == [
        "gdn_case=small",
        "gdn_case=production",
    ]
    for line in lines:
        fields = dict(field.split("=", 1) for field in line.split())
        assert fields["prepare_atomic"] == "true"
        assert fields["commit_exact"] == "true"
        assert fields["nonfinite"] == "0"
        assert float(fields["max_abs"]) <= 5.0e-8
        assert float(fields["rms"]) <= 5.0e-9
        assert float(fields["mean_ms"]) > 0.0
    assert "status=passed" in run.stdout
