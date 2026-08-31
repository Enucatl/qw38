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


def test_pre_graph_memory_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_memory_fit_contract.json").read_text())
    fixture = json.loads(
        (ROOT / "fixtures" / "cuda_memory_fit_pre_graph.json").read_text()
    )
    assert contract["capacity"] == 131_072
    assert fixture["owners"]["attention_kv_bytes"] == 8_589_934_592
    assert fixture["owners"]["explicit_quartz_bytes"] == (
        fixture["owners"]["resident_model_bytes"]
        + fixture["owners"]["session_total_bytes"]
        + fixture["owners"]["workspace_bytes"]
    )
    assert fixture["free_after_allocation_bytes"] >= fixture["required_reserve_bytes"]
    assert fixture["pre_graph_passed"]
    assert not fixture["post_graph_admitted"]
    assert fixture["owners"]["graph_bytes"] is None
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    chapter = (ROOT / "docs" / "50-pre-graph-128k-memory.md").read_text().casefold()
    for term in [
        "131,072",
        "gib versus gb",
        "8 gib",
        "resident model",
        "gdn state",
        "workspace",
        "runtime context",
        "allocator delta",
        "5,241,044,992",
        "1.5 gib",
        "graphs",
        "in progress",
        "proof boundary",
    ]:
        assert term in chapter


def test_real_128k_pre_graph_allocation_preserves_reserve() -> None:
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
        [*common, "make", "build/qw38-cuda-memory-fit-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [
            *common,
            "./build/qw38-cuda-memory-fit-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    fit = next(line for line in lines if line.startswith("memory_fit=pre_graph"))
    fields = dict(field.split("=", 1) for field in fit.split())
    assert fields["capacity"] == "131072"
    assert fields["explicit_bytes"] == "27895627616"
    assert int(fields["free_bytes"]) >= int(fields["reserve_required"])
    assert fields["arithmetic"] == "true"
    assert fields["passed"] == "true"
    assert "memory_owner=graphs bytes=unavailable status=pending_OPT-003" in lines
    assert "memory_admission=post_graph passed=false reason=OPT-003_pending" in lines
    assert "status=passed" in lines
