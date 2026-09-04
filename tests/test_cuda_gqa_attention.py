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
PREFIX = "QW38_GQA_ATTENTION_RESULT="
CONTRACT = ROOT / "pins/cuda_gqa_attention_contract.json"
FIXTURE = ROOT / "fixtures/cuda_gqa_attention.json"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def validate_result(result: Any) -> None:
    c = _contract()
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
        "production_shape",
        "semantic",
        "traffic",
        "proof_limit",
    }
    assert result["schema_version"] == 1 and result["task"] == "OPT-006"
    assert result["status"] == "measured"
    for key in c["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == c["toolkit"]
    assert result["pinned_image"] == c["pinned_image"]
    assert result["production_shape"] == c["production_shape"]
    assert "physical DRAM" in result["proof_limit"]
    assert "end-to-end" in result["proof_limit"]
    semantic = result["semantic"]
    assert set(semantic) == {"predicates", "untiled_metrics"}
    assert set(semantic["predicates"]) == set(c["semantic_predicates"])
    assert all(type(v) is bool and v for v in semantic["predicates"].values())
    assert set(semantic["untiled_metrics"]) == {"1", "3", "9", "64"}
    for metric in semantic["untiled_metrics"].values():
        assert set(metric) == {"max_abs", "rms", "cosine"}
        assert all(
            type(v) in (int, float) and math.isfinite(v) for v in metric.values()
        )
        assert metric["max_abs"] <= c["proof_limits"]["oracle_max_abs"]
        assert metric["rms"] <= c["proof_limits"]["oracle_rms"]
        assert metric["cosine"] >= c["proof_limits"]["oracle_cosine"]
    assert len(result["traffic"]) == 3
    for case, start in zip(result["traffic"], c["traffic_prefixes"], strict=True):
        assert set(case) == {
            "start_position",
            "token_count",
            "contexts",
            "per_query_values",
            "grouped_values",
            "per_query_bytes",
            "grouped_bytes",
            "ratio",
        }
        assert case["start_position"] == start and case["token_count"] == 64
        contexts = sum(start + token + 1 for token in range(64))
        per_query = 24 * contexts * 256 * 2
        grouped = 4 * contexts * 256 * 2
        assert case["contexts"] == contexts
        assert case["per_query_values"] == per_query
        assert case["grouped_values"] == grouped
        assert case["per_query_bytes"] == per_query * 2
        assert case["grouped_bytes"] == grouped * 2
        assert case["ratio"] == 6


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
    commands = [
        [*_common(), "make", "build/attention_decode.cuda.o"],
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
            "cuda/gqa_attention_test.cu",
            "build/attention_decode.cuda.o",
            "-o",
            "build/qw38-cuda-gqa-attention-test",
        ],
        [*_common(), "./build/qw38-cuda-gqa-attention-test"],
    ]
    outputs = []
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


def test_gqa_attention_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))


def test_gqa_attention_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"]["predicates"].pop("finite_output"),
        lambda x: x["traffic"][0].__setitem__(
            "grouped_values", x["traffic"][0]["grouped_values"] + 1
        ),
        lambda x: x["traffic"][0].__setitem__("ratio", 5),
        lambda x: x["traffic"][0].__setitem__(
            "grouped_bytes", x["traffic"][0]["grouped_bytes"] + 2
        ),
        lambda x: x.__setitem__("driver", "placeholder"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_gqa_attention_native_smoke() -> None:
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
        raise SystemExit("usage: test_cuda_gqa_attention.py --regenerate-fixture")
    _regenerate()
