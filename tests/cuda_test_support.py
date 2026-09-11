from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
_TIERS = frozenset({"smoke", "correctness", "screen", "acceptance"})
_SHARED_SUITE_TIERS = frozenset({"smoke", "correctness", "acceptance"})


def cuda_test_tier() -> str:
    """Return the explicitly selected CUDA test tier."""
    configured = os.environ.get("QW38_CUDA_TEST_TIER")
    if configured is None or not configured.strip():
        raise ValueError(
            "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
            "screen, or acceptance"
        )
    tier = configured.strip().lower()
    if tier not in _TIERS:
        raise ValueError(
            f"QW38_CUDA_TEST_TIER must be one of {sorted(_TIERS)}, got {tier!r}"
        )
    return tier


def _common_command(tier: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"QW38_CUDA_TEST_TIER={tier}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        IMAGE,
    ]


@lru_cache(maxsize=3)
def run_cuda_suite(tier: str) -> dict[str, str]:
    """Build and run the shared CUDA binaries once per pytest process and tier."""
    if tier not in _SHARED_SUITE_TIERS:
        raise ValueError(
            f"shared CUDA suites do not implement tier {tier!r}; "
            "use smoke, correctness, or acceptance"
        )
    common = _common_command(tier)
    build = subprocess.run(
        [
            *common,
            "make",
            "build/qw38-cuda-quant-test",
            "build/qw38-cuda-scheduler-primitives-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise AssertionError(build.stdout + build.stderr)

    outputs: dict[str, str] = {}
    for binary in ("quant", "scheduler"):
        executable = (
            "./build/qw38-cuda-quant-test"
            if binary == "quant"
            else "./build/qw38-cuda-scheduler-primitives-test"
        )
        run = subprocess.run(
            [*common, executable],
            check=False,
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            raise AssertionError(run.stdout + run.stderr)
        outputs[binary] = run.stdout
    return outputs
