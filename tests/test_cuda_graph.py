from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"


def test_graph_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_graph_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_graph.json").read_text())
    run = fixture["admitted_run"]
    assert contract["scope"]["graphs"] == fixture["graphs"] == 64
    assert fixture["graph_device_bytes"] > 0
    assert run["warmups"] >= contract["admission"]["minimum_warmups"]
    assert len(run["graph_ms"]) == len(run["ordinary_ms"]) == 30
    assert len(run["graph_ms"]) >= contract["admission"]["minimum_paired_samples"]
    assert math.isclose(sum(run["graph_ms"]) / 30, run["graph_mean_ms"], rel_tol=5e-7)
    assert math.isclose(
        sum(run["ordinary_ms"]) / 30, run["ordinary_mean_ms"], rel_tol=5e-7
    )
    assert run["graph_mean_ms"] < run["ordinary_mean_ms"]
    assert all(run["exact"].values())
    assert all(speedup > 1.0 for speedup in fixture["replicate_speedups"])
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    chapter = (
        (ROOT / "docs" / "53-stable-address-cuda-graphs.md").read_text().casefold()
    )
    for term in [
        "capture",
        "instantiate",
        "upload",
        "replay",
        "stable address",
        "64",
        "attention",
        "dynamic",
        "unavailable",
        "byte-exact",
        "6 mib",
        "proof boundary",
    ]:
        assert term in chapter


def test_real_graph_and_ordinary_schedulers_are_identical() -> None:
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
        [*common, "make", "build/qw38-cuda-graph-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-graph-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    assert len([line for line in lines if line.startswith("graph_sample=")]) == 30
    compare = next(line for line in lines if line.startswith("graph_compare="))
    fields = dict(field.split("=", 1) for field in compare.split())
    assert fields["frontier"] == "33"
    for name in [
        "exact_logits",
        "exact_hidden",
        "exact_taps",
        "exact_state",
        "greedy_equal",
    ]:
        assert fields[name] == "true"
    summary = next(line for line in lines if line.startswith("graph_run="))
    summary_fields = dict(field.split("=", 1) for field in summary.split())
    assert summary_fields["graphs"] == "64"
    assert int(summary_fields["graph_bytes"]) > 0
    assert float(summary_fields["graph_launch_cpu_ms"]) > 0.0
    assert summary_fields["warmups"] == "3"
    assert summary_fields["samples"] == "30"
    assert "status=passed" in lines
