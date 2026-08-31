from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"


def test_attention_prefill_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "cuda_attention_prefill_contract.json").read_text()
    )
    fixture = json.loads(
        (ROOT / "fixtures" / "cuda_attention_prefill.json").read_text()
    )
    boundary = contract["capacity_boundary"]
    assert boundary["tokens"] == 131_072
    assert boundary["last_valid_position"] == 131_071
    assert boundary["one_layer_kv_bytes"] == 512 * 1024 * 1024
    assert boundary["all_16_attention_layer_kv_bytes"] == 8 * 1024**3
    assert boundary["maximum_score_bytes"] == 12 * 1024 * 1024
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert [case["name"] for case in fixture["chunk_cases"]] == [
        "small_3",
        "small_9",
        "production_9",
    ]
    assert all(
        case["prepare_atomic"]
        and case["tokenwise_equal"]
        and case["commit_exact"]
        and case["nonfinite"] == 0
        for case in fixture["chunk_cases"]
    )
    capacity = fixture["capacity_case"]
    assert capacity["position"] == 131_071
    assert capacity["prepare_atomic"]
    assert capacity["commit_exact"]
    assert capacity["overflow_rejected"]
    chapter = (ROOT / "docs" / "44-cuda-attention-prefill.md").read_text().casefold()
    for term in [
        "memory-bounded",
        "candidate rows",
        "token-major",
        "quadratic",
        "131,072",
        "512 mib",
        "12 mib",
        "cancellation",
    ]:
        assert term in chapter


def test_attention_prefill_matches_tokens_and_executes_128k_boundary() -> None:
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
        [*common, "make", "build/qw38-cuda-attention-chunk-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*common, "./build/qw38-cuda-attention-chunk-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    chunks = [
        line for line in run.stdout.splitlines() if line.startswith("attention_chunk=")
    ]
    assert [line.split()[0] for line in chunks] == [
        "attention_chunk=small_3",
        "attention_chunk=small_9",
        "attention_chunk=production_9",
    ]
    for line in chunks:
        fields = dict(field.split("=", 1) for field in line.split())
        assert fields["prepare_atomic"] == "true"
        assert fields["tokenwise_equal"] == "true"
        assert fields["commit_exact"] == "true"
        assert fields["nonfinite"] == "0"
        assert int(fields["score_values"]) < int(fields["quadratic_values"])
        assert float(fields["mean_ms"]) > 0.0
    capacity_line = next(
        line
        for line in run.stdout.splitlines()
        if line.startswith("attention_capacity=")
    )
    capacity = dict(field.split("=", 1) for field in capacity_line.split())
    assert capacity["capacity"] == "131072"
    assert capacity["position"] == "131071"
    assert capacity["cache_bytes"] == "536870912"
    assert capacity["score_bytes"] == "12582912"
    assert capacity["prepare_atomic"] == "true"
    assert capacity["commit_exact"] == "true"
    assert capacity["overflow_rejected"] == "true"
    assert capacity["nonfinite"] == "0"
    assert float(capacity["elapsed_ms"]) > 0.0
    assert "status=passed" in run.stdout
