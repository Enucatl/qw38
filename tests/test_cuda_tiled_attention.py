from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"


def test_tiled_attention_contract_and_fixture_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins/cuda_tiled_attention_contract.json").read_text()
    )
    fixture = json.loads((ROOT / "fixtures/cuda_tiled_attention.json").read_text())
    assert contract["task"] == "OPT-005"
    assert contract["launches"]["production_kernels_per_chunk"] == 2
    assert contract["memory_boundary"]["score_workspace_touched"] is False
    assert fixture["task"] == "OPT-005"
    assert fixture["status"] == "measured"
    assert fixture["semantic"]["production_kernel_nodes"] == {"3": 2, "9": 2, "64": 2}
    assert fixture["semantic"]["reference_kernel_nodes"] == {"3": 9, "9": 27, "64": 192}
    assert fixture["semantic"]["scratch_unchanged"]
    assert [case["prefix"] for case in fixture["scaling"]] == [2048, 8192, 32768]
    for case in fixture["scaling"]:
        assert len(case["tiled_samples"]) == 30
        assert len(case["reference_samples"]) == 3
        assert case["mean_ms_tiled"] > 0 < case["mean_ms_reference"]
        assert case["speedup"] == pytest.approx(
            case["mean_ms_reference"] / case["mean_ms_tiled"], rel=1e-6
        )
        assert case["mean_ms_reference"] > case["mean_ms_tiled"]
    assert fixture["compute_capability"] == "12.0"
    assert fixture["toolkit"] == f"CUDA {IMAGE.split(':', 1)[1]}"
    for key in (
        "finite",
        "candidate_bf16_exact",
        "scratch_unchanged",
        "prepare_isolation",
        "commit_exact",
        "frontier_exact",
        "future_row_excluded",
        "overflow_rejected",
        "alias_rejected",
    ):
        assert fixture["semantic"][key] is True


def test_tiled_attention_native_smoke() -> None:
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
    object_build = subprocess.run(
        [*common, "make", "build/attention_decode.cuda.o"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert object_build.returncode == 0, object_build.stdout + object_build.stderr
    build = subprocess.run(
        [
            *common,
            "nvcc",
            "-std=c++17",
            "-O2",
            "-arch=sm_120",
            "--expt-relaxed-constexpr",
            "--fmad=false",
            "-Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread",
            "-Iinclude",
            "-Isrc",
            "-Ithird_party/utf8proc",
            "-Icuda",
            "cuda/tiled_attention_test.cu",
            "build/attention_decode.cuda.o",
            "-o",
            "build/qw38-cuda-tiled-attention-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*common, "./build/qw38-cuda-tiled-attention-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    records = []
    for line in run.stdout.splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
    assert records and all(record.get("status") == "passed" for record in records)
    semantic = next(record for record in records if record.get("type") == "semantic")
    assert semantic["production_nodes"] == {"1": 2, "3": 2, "9": 2, "64": 2}
    assert semantic["reference_nodes"] == {"1": 3, "3": 9, "9": 27, "64": 192}
    assert semantic["finite"] and semantic["scratch_unchanged"]
    for record in records:
        if record["type"] == "scale":
            assert len(record["tiled_samples"]) == 30
            assert len(record["reference_samples"]) == 3
            assert record["mean_ms_reference"] > record["mean_ms_tiled"] > 0
            assert record["speedup"] == pytest.approx(
                record["mean_ms_reference"] / record["mean_ms_tiled"], rel=1e-6
            )
