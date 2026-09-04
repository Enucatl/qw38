from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"


def test_dispatch_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "cuda_dispatch_tuning_contract.json").read_text()
    )
    fixture = json.loads((ROOT / "fixtures" / "cuda_dispatch_tuning.json").read_text())
    assert fixture["warmups_per_candidate"] == 3
    assert fixture["samples_per_candidate"] == 30
    assert fixture["replicates"] == 3
    assert len(fixture["candidates"]) == 156
    means: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for candidate in fixture["candidates"]:
        key = (
            candidate["operation"],
            candidate["rows"],
            candidate["prompt_rows"],
            candidate["variant"],
        )
        means[key].append(candidate["mean_ms"])
    winners: dict[tuple[str, int, int], int] = {}
    for key, values in means.items():
        group = key[:3]
        mean = sum(values) / len(values)
        if group not in winners:
            winners[group] = key[3]
            continue
        admitted = means[(*group, winners[group])]
        if mean < sum(admitted) / len(admitted):
            winners[group] = key[3]
    for rows, warps in contract["selection"]["mmv_rows_to_warps"].items():
        assert winners[("mmv", int(rows), 0)] == warps
    for rows, tile in contract["selection"]["mmq_prompt_rows_to_tile"].items():
        assert winners[("mmq", 17_408, int(rows))] == tile
    raw = ROOT / fixture["raw_run"]
    assert (
        sum(line.startswith("tune=") for line in raw.read_text().splitlines()) == 1560
    )
    chapter = (ROOT / "docs" / "55-offline-dispatch-tuning.md").read_text().casefold()
    for term in [
        "dispatch table",
        "launch shape",
        "row bucket",
        "warp",
        "prompt tile",
        "warm-up",
        "30",
        "three replicates",
        "losers",
        "zero-filled",
        "proof boundary",
    ]:
        assert term in chapter


def test_real_rtx5090_dispatch_sweep_completes() -> None:
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
        [*common, "make", "build/qw38-cuda-dispatch-tuning-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*common, "./build/qw38-cuda-dispatch-tuning-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    assert len([line for line in lines if line.startswith("tune=")]) == 1560
    assert "status=passed" in lines
