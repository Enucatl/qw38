from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "build" / "qw38-bench"
MODEL = ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
IMAGE = "qw38-cuda:13.0.2"


def test_benchmark_statistics_cli_and_failed_run_retention(tmp_path: Path) -> None:
    self_test = subprocess.run(
        [str(BENCH), "--self-test"], check=False, capture_output=True, text=True
    )
    assert self_test.returncode == 0
    assert json.loads(self_test.stdout) == {
        "p50": 3,
        "p95": 4.8,
        "schema": "qw38.benchmark-result.v1",
        "status": "passed",
    }

    help_result = subprocess.run(
        [str(BENCH), "--help"], check=False, capture_output=True, text=True
    )
    assert help_result.returncode == 0
    for term in [
        "prefill|decode",
        "expected-prompt-tokens",
        "source-revision",
        "agent-reuse",
        "warmups",
        "samples",
        "smoke",
    ]:
        assert term in help_result.stdout

    rejected = subprocess.run(
        [
            str(BENCH),
            "missing.gguf",
            "--workload",
            "prefill",
            "--prompt",
            "x",
            "--output",
            str(tmp_path / "not-created.json"),
            "--warmups",
            "2",
            "--samples",
            "29",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1
    assert "release runs require" in rejected.stderr
    assert not (tmp_path / "not-created.json").exists()

    failed_path = tmp_path / "failed.json"
    failed = subprocess.run(
        [
            str(BENCH),
            "missing.gguf",
            "--workload",
            "prefill",
            "--prompt",
            "x",
            "--output",
            str(failed_path),
            "--warmups",
            "0",
            "--samples",
            "1",
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    retained = json.loads(failed_path.read_text())
    assert retained["status"] == "failed"
    assert retained["error"]["code"] == "io_error"
    assert retained["warmups"] == []
    assert retained["samples"] == []


def test_benchmark_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "benchmark_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "benchmark_harness.json").read_text())
    assert contract["schema"] == "qw38.benchmark-contract.v1"
    assert contract["minimum_warmups"] == 3
    assert contract["minimum_samples"] == 30
    assert contract["primary_cache_policy"] == "disabled"
    assert contract["workloads"]["prefill"] == [128, 2048, 8192]
    assert contract["workloads"]["decode_output_tokens"] == 256
    assert fixture["smoke"]["passed"] is True
    assert fixture["smoke"]["admission_eligible"] is False
    assert (
        fixture["negative"]["historical_full_scheduler_prompt_path_was_token_wise"]
        is True
    )
    assert fixture["negative"]["resolved_by"] == "SCH-002"
    for relative, expected in fixture["smoke"]["raw_result_sha256"].items():
        raw_path = ROOT / relative
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == expected
        raw = json.loads(raw_path.read_text())
        assert raw["status"] == "success"
        assert raw["admission_eligible"] is False
        assert raw["environment"]["source_state"] == "dirty"
    reuse = json.loads(
        (ROOT / "evidence/benchmark/ben001-reuse-smoke.json").read_text()
    )
    assert reuse["cache_policy"] == "agent-reuse"
    assert reuse["samples"][1]["reused_prefix_tokens"] > 0
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    chapter = (ROOT / "docs" / "61-benchmark-harness.md").read_text().casefold()
    for term in [
        "warm-up",
        "sample",
        "prefill",
        "time to first token",
        "inter-token latency",
        "p50",
        "p95",
        "throughput",
        "telemetry",
        "raw",
        "agent reuse",
        "null",
        "atomic",
        "proof boundary",
        "sch-002",
    ]:
        assert term in chapter


@pytest.mark.skipif(not MODEL.exists(), reason="the pinned GGUF is required")
def test_cuda_benchmark_retains_raw_decode_reuse_and_components() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    build = subprocess.run(
        [
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
            "make",
            "build/cuda/qw38-bench",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    relative = Path("build") / f"benchmark-smoke-{os.getpid()}.json"
    result = subprocess.run(
        [
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
            "./build/cuda/qw38-bench",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            "--workload",
            "decode",
            "--prompt",
            "Quartz benchmark prompt.",
            "--context-label",
            "smoke",
            "--output-tokens",
            "2",
            "--warmups",
            "1",
            "--samples",
            "2",
            "--cache-policy",
            "agent-reuse",
            "--source-revision",
            "cuda-smoke",
            "--source-state",
            "dirty",
            "--smoke",
            "--output",
            str(relative),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        output = json.loads((ROOT / relative).read_text())
        assert output["status"] == "success"
        assert output["admission_eligible"] is False
        assert len(output["warmups"]) == 1
        assert len(output["samples"]) == 2
        assert output["samples"][0]["reused_prefix_tokens"] == 0
        assert output["samples"][1]["reused_prefix_tokens"] > 0
        assert len(output["samples"][0]["per_token"]) == 2
        assert output["summary"]["itl_ms_p50"] > 0
        assert output["summary"]["queue_time_ms"] is None
        assert output["component_probe"]["perturbs_execution"] is True
        assert output["component_probe"]["used_for_throughput_summary"] is False
        assert output["component_probe"]["gdn_ms"] > 0
        assert output["environment"]["telemetry"]["gpu_name"] == (
            "NVIDIA GeForce RTX 5090"
        )
        assert output["environment"]["cuda_runtime_version"] == 13000
    finally:
        (ROOT / relative).unlink(missing_ok=True)
