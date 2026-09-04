from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"
RESULT_PREFIX = "QW38_TILED_ATTENTION_RESULT="
CONTRACT_PATH = ROOT / "pins/cuda_tiled_attention_contract.json"
FIXTURE_PATH = ROOT / "fixtures/cuda_tiled_attention.json"


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


def validate_result(result: Any, contract: dict[str, Any] | None = None) -> None:
    c = contract or _load_contract()
    required_top = {
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
        "warmups",
        "proof_limits",
        "semantic",
        "scaling",
        "claim",
    }
    assert isinstance(result, dict) and set(result) == required_top
    assert result["schema_version"] == c["schema_version"]
    assert result["task"] == c["task"] and result["status"] == "measured"
    for key in c["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert (
            "unknown" not in result[key].lower()
            and "placeholder" not in result[key].lower()
        )
    assert result["compute_capability"] == "12.0"
    assert (
        result["toolkit"] == c["toolkit"]
        and result["pinned_image"] == c["pinned_image"]
    )
    assert (
        result["warmups"] == c["warmups"]
        and result["proof_limits"] == c["proof_limits"]
    )
    semantic = result["semantic"]
    assert set(semantic) == {
        "predicates",
        "metrics",
        "production_kernel_nodes",
        "reference_kernel_nodes",
    }
    predicates = semantic["predicates"]
    assert set(predicates) == set(c["semantic_predicates"])
    assert all(type(value) is bool and value for value in predicates.values())
    expected_prod = {
        str(k): v for k, v in c["graph_kernel_nodes"]["production"].items()
    }
    expected_ref = {str(k): v for k, v in c["graph_kernel_nodes"]["reference"].items()}
    assert semantic["production_kernel_nodes"] == expected_prod
    assert semantic["reference_kernel_nodes"] == expected_ref
    assert set(semantic["metrics"]) == {"3", "9"}
    limits = c["proof_limits"]
    for metrics in semantic["metrics"].values():
        assert set(metrics) == {"max_abs", "rms", "cosine"}
        assert all(
            type(v) in (int, float) and math.isfinite(v) for v in metrics.values()
        )
        assert metrics["max_abs"] <= limits["bf16_max_abs"]
        assert metrics["rms"] <= limits["bf16_rms"]
        assert metrics["cosine"] >= limits["oracle_cosine"]
    assert len(result["scaling"]) == len(c["scaling_prefixes"])
    for case, prefix in zip(result["scaling"], c["scaling_prefixes"], strict=True):
        assert set(case) == {
            "prefix",
            "rows",
            "tiled_samples",
            "reference_samples",
            "mean_ms_tiled",
            "mean_ms_reference",
            "speedup",
        }
        assert case["prefix"] == prefix and case["rows"] == 64
        tiled, reference = case["tiled_samples"], case["reference_samples"]
        assert (
            len(tiled) == c["samples"]["tiled"]
            and len(reference) == c["samples"]["reference"]
        )
        assert len(set(tiled)) > 1 and len(set(reference)) > 1
        assert all(
            type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in tiled + reference
        )
        tm, rm = sum(tiled) / len(tiled), sum(reference) / len(reference)
        assert case["mean_ms_tiled"] == pytest.approx(tm, rel=1e-7, abs=1e-7)
        assert case["mean_ms_reference"] == pytest.approx(rm, rel=1e-7, abs=1e-7)
        assert case["speedup"] == pytest.approx(rm / tm, rel=1e-7, abs=1e-7)
        assert rm > tm > 0


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
    obj = subprocess.run(
        [*_common(), "make", "build/attention_decode.cuda.o"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert obj.returncode == 0, obj.stdout + obj.stderr
    build = subprocess.run(
        [
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
            "cuda/tiled_attention_test.cu",
            "build/attention_decode.cuda.o",
            "-o",
            "build/qw38-cuda-tiled-attention-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*_common(), "./build/qw38-cuda-tiled-attention-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    records = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in run.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(records) == 1, run.stdout + run.stderr
    validate_result(records[0])
    return records[0]


def test_tiled_attention_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE_PATH.read_text()))


def test_tiled_attention_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    mutations = []
    missing = json.loads(json.dumps(fixture))
    missing["semantic"]["predicates"].pop(next(iter(missing["semantic"]["predicates"])))
    mutations.append(missing)
    synthetic = json.loads(json.dumps(fixture))
    synthetic["scaling"][0]["tiled_samples"] = [
        synthetic["scaling"][0]["mean_ms_tiled"]
    ] * 30
    mutations.append(synthetic)
    summary = json.loads(json.dumps(fixture))
    summary["scaling"][0]["speedup"] += 1
    mutations.append(summary)
    nodes = json.loads(json.dumps(fixture))
    nodes["semantic"]["production_kernel_nodes"]["64"] = 3
    mutations.append(nodes)
    metadata = json.loads(json.dumps(fixture))
    metadata["driver"] = "unknown"
    mutations.append(metadata)
    status = json.loads(json.dumps(fixture))
    status["status"] = "passed"
    mutations.append(status)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_tiled_attention_native_smoke() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    _build_and_run()


def _regenerate_fixture() -> None:
    result = _build_and_run()
    fd, temporary = tempfile.mkstemp(
        dir=FIXTURE_PATH.parent, prefix=f".{FIXTURE_PATH.name}.", text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, FIXTURE_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate-fixture"]:
        raise SystemExit("usage: test_cuda_tiled_attention.py --regenerate-fixture")
    _regenerate_fixture()
