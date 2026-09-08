from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
PREFIX = "QW38_GDN_SCAN_RESULT="
CONTRACT = ROOT / "pins/cuda_gdn_scan_contract.json"
FIXTURE = ROOT / "fixtures/cuda_gdn_scan.json"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _envelope_ok(case: dict[str, Any], contract: dict[str, Any]) -> None:
    admission = contract["admission"]
    assert case["prepare_atomic"] is True
    assert case["nonfinite"] == admission["nonfinite_count"]
    assert case["max_abs"] <= admission["maximum_absolute_error"]
    assert case["rms"] <= admission["maximum_rms_error"]
    assert case["passed"] is True


def validate_result(result: Any) -> None:
    contract = _contract()
    assert isinstance(result, dict) and set(result) == {
        "schema_version",
        "task",
        "status",
        "device",
        "compute_capability",
        "driver",
        "runtime",
        "toolkit",
        "pinned_image",
        "measurement_utc",
        "semantic",
        "sequential_regression",
        "parallel_vs_sequential",
        "chunk_boundaries",
        "tiled",
        "launch",
        "fail_closed",
        "overlay",
        "speedup",
        "launch_counts",
        "proof_limit",
    }
    assert result["schema_version"] == 1 and result["task"] == "OPT-013"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert "component-only" in result["proof_limit"]
    assert "GDN-002" in result["proof_limit"]
    assert "byte-exact" in result["proof_limit"]
    assert "Nsight" in result["proof_limit"]
    assert "end-to-end" in result["proof_limit"]
    assert "128K quality" in result["proof_limit"]
    assert set(result["semantic"]) == set(contract["semantic_predicates"])
    assert all(type(value) is bool and value for value in result["semantic"].values())

    sequential = result["sequential_regression"]
    assert [(case["name"], case["tokens"]) for case in sequential] == [
        ("small_3", 3),
        ("small_64", 64),
        ("small_65", 65),
        ("small_129", 129),
        ("production_65", 65),
    ]
    for case in sequential:
        _envelope_ok(case, contract)
        assert case["tokenwise_equal"] is True

    parallel = result["parallel_vs_sequential"]
    assert [(case["name"], case["tokens"]) for case in parallel] == [
        ("small_3", 3),
        ("small_64", 64),
        ("small_65", 65),
        ("small_129", 129),
        ("production_64", 64),
        ("production_65", 65),
        ("production_129", 129),
        ("production_256", 256),
        ("production_4096", 4096),
    ]
    for case in parallel:
        _envelope_ok(case, contract)
        if case["tokens"] <= 64:
            assert case["conv_memcmp"] is True
        if case["tokens"] <= 129:
            assert case["vs_scalar"] is True

    boundaries = result["chunk_boundaries"]
    assert boundaries["production_65"] is True
    assert boundaries["production_129"] is True
    split = boundaries["production_4096_vs_64_windows"]
    _envelope_ok(
        {
            "prepare_atomic": split["prepare_atomic"],
            "nonfinite": split["nonfinite"],
            "max_abs": split["max_abs"],
            "rms": split["rms"],
            "passed": split["passed"],
        },
        contract,
    )

    tiled = result["tiled"]
    assert tiled["name"] == "production_65_tiled" and tiled["tokens"] == 65
    _envelope_ok(tiled, contract)

    launch = result["launch"]
    assert launch["conv_64"]["kernel_nodes"] == 1
    assert launch["conv_64"]["grid"] == [40, 64, 1]
    assert launch["conv_64"]["block"] == [256, 1, 1]
    assert launch["conv_4096"]["kernel_nodes"] == 1
    assert launch["conv_4096"]["grid"] == [40, 4096, 1]
    assert launch["conv_4096"]["block"] == [256, 1, 1]
    assert launch["intra_4096"]["kernel_nodes"] == 1
    assert launch["intra_4096"]["grid"] == contract["launch"]["intra_grid"]
    assert launch["intra_4096"]["block"] == contract["launch"]["intra_block"]
    assert launch["prefix_4096"]["kernel_nodes"] == 1
    assert launch["prefix_4096"]["grid"] == contract["launch"]["prefix_grid"]
    assert launch["prefix_4096"]["block"] == contract["launch"]["intra_block"]
    assert launch["from_state_4096"]["kernel_nodes"] == 1
    assert launch["from_state_4096"]["grid"] == contract["launch"]["from_state_grid"]
    assert launch["from_state_4096"]["block"] == contract["launch"]["intra_block"]
    assert (
        launch["sequential_4096_recurrence_nodes"]
        == contract["launch"]["sequential_4096_recurrence_nodes"]
    )

    fail_closed = result["fail_closed"]
    assert fail_closed == {
        "null_scratch": True,
        "zero_scratch_floats": True,
        "aliased": True,
        "token_count_0": True,
        "invalid_path": True,
    }
    assert result["overlay"] == contract["overlay"]

    speedup = result["speedup"]
    assert speedup["warmups"] == 3
    assert speedup["samples"] == contract["speedup_samples"]
    assert speedup["parallel_below_sequential"] is True
    assert len(speedup["sequential_ms"]) == contract["speedup_samples"]
    assert len(speedup["parallel_ms"]) == contract["speedup_samples"]
    assert all(sample > 0.0 for sample in speedup["sequential_ms"])
    assert all(sample > 0.0 for sample in speedup["parallel_ms"])
    sequential_mean = sum(speedup["sequential_ms"]) / len(speedup["sequential_ms"])
    parallel_mean = sum(speedup["parallel_ms"]) / len(speedup["parallel_ms"])
    assert speedup["sequential_mean_ms"] == pytest.approx(sequential_mean, rel=1e-6)
    assert speedup["parallel_mean_ms"] == pytest.approx(parallel_mean, rel=1e-6)
    assert parallel_mean < sequential_mean

    counts = result["launch_counts"]
    assert counts["parallel_convolution"] == 1
    assert counts["parallel_intra"] == 1
    assert counts["parallel_prefix"] == 1
    assert counts["parallel_from_state"] == 1
    assert counts["sequential_convolution"] == 64
    assert counts["sequential_recurrence"] == 64


def _common() -> list[str]:
    return [
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


def _build_and_run() -> dict[str, Any]:
    nvcc = [
        *_common(),
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
        "cuda/gdn_scan_test.cu",
        "build/gdn_step.cuda.o",
        "build/gdn.o",
        "build/status.o",
        "-o",
        "build/qw38-cuda-gdn-scan-test",
    ]
    commands = [
        [*_common(), "make", "build/qw38-cuda-gdn-chunk-test"],
        nvcc,
        [*_common(), "./build/qw38-cuda-gdn-scan-test"],
    ]
    outputs: list[str] = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs.append(completed.stdout)
    records = [
        json.loads(line.removeprefix(PREFIX))
        for line in outputs[-1].splitlines()
        if line.startswith(PREFIX)
    ]
    assert len(records) == 1
    validate_result(records[0])
    return records[0]


def test_cuda_gdn_scan_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))


def test_cuda_gdn_scan_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"].pop("speedup"),
        lambda x: x["sequential_regression"][0].__setitem__("tokenwise_equal", False),
        lambda x: x["parallel_vs_sequential"][0].__setitem__("conv_memcmp", False),
        lambda x: x["parallel_vs_sequential"][-1].__setitem__("max_abs", 1.0),
        lambda x: x["chunk_boundaries"]["production_4096_vs_64_windows"].__setitem__(
            "prepare_atomic", False
        ),
        lambda x: x["launch"]["intra_4096"].__setitem__("grid", [48, 45, 1]),
        lambda x: x["launch"]["prefix_4096"].__setitem__("grid", [48, 64, 1]),
        lambda x: x["fail_closed"].__setitem__("invalid_path", False),
        lambda x: x["overlay"].__setitem__("w_fit_65", 1),
        lambda x: x["speedup"].__setitem__("parallel_below_sequential", False),
        lambda x: x["launch_counts"].__setitem__("parallel_from_state", 2),
        lambda x: x.__setitem__("driver", "placeholder"),
        lambda x: x.__setitem__("proof_limit", "end-to-end speedup from Nsight"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_cuda_gdn_scan_native_smoke() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    _build_and_run()


def _regenerate() -> None:
    result = _build_and_run()
    fd, temporary = tempfile.mkstemp(
        dir=FIXTURE.parent, prefix=f".{FIXTURE.name}.", text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, FIXTURE)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate-fixture"]:
        raise SystemExit("usage: test_cuda_gdn_scan.py --regenerate-fixture")
    _regenerate()
