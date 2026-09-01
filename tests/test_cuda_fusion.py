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


def test_fusion_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_fusion_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_fusion.json").read_text())
    run = fixture["admitted_run"]
    assert contract["fusion"]["removed_launches_per_token"] == 63
    assert contract["fusion"]["retained_reference"] == "unfused"
    assert run["warmups"] >= contract["admission"]["minimum_warmups"]
    assert len(run["fused_ms"]) == len(run["unfused_ms"]) == 30
    assert len(run["fused_ms"]) >= contract["admission"]["minimum_paired_samples"]
    assert math.isclose(sum(run["fused_ms"]) / 30, run["fused_mean_ms"], rel_tol=5e-7)
    assert math.isclose(
        sum(run["unfused_ms"]) / 30, run["unfused_mean_ms"], rel_tol=5e-7
    )
    assert run["fused_mean_ms"] < run["unfused_mean_ms"]
    assert all(run["exact"].values())
    assert all(speedup > 1.0 for speedup in fixture["replicate_speedups"])
    assert fixture["rejected_serial_variant"]["speedup"] < 1.0
    assert fixture["profiler"]["q8_mmv_rejected_candidate"][
        "classification"
    ].startswith("balanced")
    report = contract["profiling_report"]
    assert (
        hashlib.sha256((ROOT / report["path"]).read_bytes()).hexdigest()
        == report["sha256"]
    )
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    chapter = (ROOT / "docs" / "52-profiler-led-fusion.md").read_text().casefold()
    for term in [
        "kernel launch",
        "nsight compute",
        "sys_admin",
        "fused",
        "unfused",
        "63",
        "bit-exact",
        "alternat",
        "rejected",
        "raw samples",
        "proof boundary",
    ]:
        assert term in chapter


def test_real_fused_and_unfused_schedulers_are_identical() -> None:
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
        [*common, "make", "build/qw38-cuda-fusion-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-fusion-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    samples = [line for line in lines if line.startswith("fusion_sample=")]
    assert len(samples) == 30
    compare = next(line for line in lines if line.startswith("fusion_compare="))
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
    timing = next(line for line in lines if line.startswith("fusion_timing="))
    timing_fields = dict(field.split("=", 1) for field in timing.split())
    assert timing_fields["warmups"] == "3"
    assert timing_fields["samples"] == "30"
    assert float(timing_fields["fused_mean_ms"]) > 0.0
    assert float(timing_fields["unfused_mean_ms"]) > 0.0
    assert "status=passed" in lines
