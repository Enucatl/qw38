from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"


def test_cuda_attention_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_attention_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_attention_decode.json").read_text())
    shape = contract["production_shape"]
    assert contract["target"] == "sm_120"
    assert (shape["query_heads"], shape["kv_heads"], shape["kv_group_size"]) == (
        24,
        4,
        6,
    )
    assert shape["rotary_width"] == 64
    assert shape["kv_bytes_per_element"] == 2
    assert contract["admission"]["frozen_scalar_maximum_absolute_error"] == 0.051
    assert [case["name"] for case in fixture["cases"]] == [
        "layer_3",
        "layer_7",
        "layer_63",
        "production_layer_3",
    ]
    assert all(
        case["candidate_exact"]
        and case["prepare_atomic"]
        and case["commit_exact"]
        and case["nonfinite"] == 0
        for case in fixture["cases"]
    )
    chapter = (ROOT / "docs" / "43-cuda-attention-decode.md").read_text().casefold()
    for term in [
        "grouped-query attention",
        "partial rope",
        "bf16",
        "candidate row",
        "causal",
        "stable softmax",
        "frontier",
        "cancellation",
    ]:
        assert term in chapter


def test_cuda_attention_matches_oracles_and_commits_atomically() -> None:
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
        [*common, "make", "build/qw38-cuda-attention-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*common, "./build/qw38-cuda-attention-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = [
        line for line in run.stdout.splitlines() if line.startswith("attention_case=")
    ]
    assert [line.split()[0] for line in lines] == [
        "attention_case=layer_3",
        "attention_case=layer_7",
        "attention_case=layer_63",
        "attention_case=production_layer_3",
    ]
    for line in lines:
        fields = dict(field.split("=", 1) for field in line.split())
        assert fields["candidate_exact"] == "true"
        assert fields["prepare_atomic"] == "true"
        assert fields["commit_exact"] == "true"
        assert fields["nonfinite"] == "0"
        assert float(fields["max_abs"]) <= 5.0e-5
        assert float(fields["rms"]) <= 5.0e-6
        assert float(fields["oracle_max_abs"]) <= 0.051
        assert float(fields["oracle_rms"]) <= 0.0016
        assert float(fields["oracle_cosine"]) >= 0.999424
        assert float(fields["mean_ms"]) > 0.0
    assert "status=passed" in run.stdout
