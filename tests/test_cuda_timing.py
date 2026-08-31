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


def test_timing_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_timing_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_timing.json").read_text())
    required = {
        "loading",
        "embedding",
        "gdn",
        "attention",
        "ffn",
        "logits",
        "sampling",
        "graph_launch",
        "queueing",
        "persistence",
        "idle_gaps",
        "state_commit",
        "token_total",
    }
    assert set(contract["categories"]) == required
    assert set(fixture["categories_ms"]) == required
    assert fixture["categories_ms"]["graph_launch"] is None
    assert fixture["categories_ms"]["queueing"] is None
    assert fixture["categories_ms"]["ffn"] > fixture["categories_ms"]["gdn"]
    assert math.isclose(
        fixture["attributed_sum_ms"],
        fixture["categories_ms"]["token_total"],
        abs_tol=1e-6,
    )
    assert fixture["profiler_environment"]["nsight_compute_result"] == (
        "ERR_NVGPUCTRPERM"
    )
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    chapter = (ROOT / "docs" / "51-runtime-timing-and-nvtx.md").read_text().casefold()
    for term in [
        "asynchronous",
        "cuda event",
        "synchronization",
        "nvtx",
        "cpu clock",
        "unavailable",
        "idle gaps",
        "perturb",
        "err_nvgpuctrperm",
        "proof boundary",
    ]:
        assert term in chapter


def test_real_scheduler_exposes_complete_timing_attribution(tmp_path: Path) -> None:
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
        [*common, "make", "build/qw38-cuda-timing-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    checkpoint = f"build/timing-{tmp_path.name}.qw38"
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-timing-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            checkpoint,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    categories: dict[str, dict[str, str]] = {}
    for line in lines:
        if line.startswith("timing_category="):
            fields = dict(field.split("=", 1) for field in line.split())
            categories[fields["timing_category"]] = fields
    assert set(categories) == set(
        json.loads((ROOT / "pins" / "cuda_timing_contract.json").read_text())[
            "categories"
        ]
    )
    for name in set(categories) - {"graph_launch", "queueing"}:
        assert categories[name]["availability"] == "measured"
        assert float(categories[name]["milliseconds"]) >= 0.0
    for name in ["graph_launch", "queueing"]:
        assert categories[name]["availability"] == "unavailable"
        assert categories[name]["milliseconds"] == "null"
    summary = next(line for line in lines if line.startswith("timing_run="))
    fields = dict(field.split("=", 1) for field in summary.split())
    assert fields["sampled"] == "1277"
    assert fields["frontier"] == "2"
    assert math.isclose(
        float(fields["attributed_ms"]),
        float(fields["total_ms"]),
        rel_tol=1e-6,
        abs_tol=1e-5,
    )
    assert "status=passed" in lines
