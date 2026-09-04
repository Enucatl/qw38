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
PREFIX = "QW38_QUERY_ROW_ATTENTION_RESULT="
CONTRACT = ROOT / "pins/cuda_query_row_attention_contract.json"
FIXTURE = ROOT / "fixtures/cuda_query_row_attention.json"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


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
        "production_shape",
        "semantic",
        "launches",
        "kernel_attributes",
        "proof_limit",
    }
    assert result["schema_version"] == 1 and result["task"] == "OPT-007"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert result["production_shape"] == contract["production_shape"]
    assert "component-only" in result["proof_limit"]
    assert "no end-to-end" in result["proof_limit"]
    assert set(result["semantic"]) == set(contract["semantic_predicates"])
    assert all(type(value) is bool and value for value in result["semantic"].values())
    assert len(result["launches"]) == len(contract["row_cases"])
    for launch, rows in zip(result["launches"], contract["row_cases"], strict=True):
        assert set(launch) == {
            "rows",
            "kernel_nodes",
            "staging_grid",
            "staging_block",
            "attention_grid",
            "attention_block",
            "dynamic_shared_bytes",
        }
        assert launch["rows"] == rows and launch["kernel_nodes"] == 2
        assert launch["staging_grid"] == [4, rows, 1]
        assert launch["staging_block"] == [256, 1, 1]
        assert launch["attention_grid"] == [4, (rows + 1) // 2, 1]
        assert launch["attention_block"] == [256, 1, 1]
        assert launch["dynamic_shared_bytes"] == contract["dynamic_shared_bytes"]
    attributes = result["kernel_attributes"]
    assert set(attributes) == {
        "registers",
        "static_shared_bytes",
        "local_bytes_per_thread",
        "maximum_dynamic_shared_bytes",
        "active_blocks_per_sm",
        "sm_count",
        "launch_blocks",
    }
    assert attributes["registers"] > 0
    assert attributes["static_shared_bytes"] >= 0
    assert (
        attributes["local_bytes_per_thread"]
        <= contract["maximum_local_bytes_per_thread"]
    )
    assert (
        attributes["maximum_dynamic_shared_bytes"] >= contract["dynamic_shared_bytes"]
    )
    assert attributes["active_blocks_per_sm"] >= 1
    assert attributes["sm_count"] > 0 and attributes["launch_blocks"] > 0


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
            "cuda/query_row_attention_test.cu",
            "build/attention_decode.cuda.o",
            "-o",
            "build/qw38-cuda-query-row-attention-test",
        ],
        [*_common(), "./build/qw38-cuda-query-row-attention-test"],
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


def test_query_row_attention_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))


def test_query_row_attention_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"].pop("finite_output"),
        lambda x: x["launches"][0]["staging_grid"].__setitem__(0, 3),
        lambda x: x["launches"][0]["staging_block"].__setitem__(2, 2),
        lambda x: x["launches"][2]["attention_grid"].__setitem__(1, 3),
        lambda x: x["launches"][2]["attention_block"].__setitem__(1, 2),
        lambda x: x["launches"][0].__setitem__("dynamic_shared_bytes", 33791),
        lambda x: x["kernel_attributes"].__setitem__("active_blocks_per_sm", 0),
        lambda x: x["kernel_attributes"].__setitem__("local_bytes_per_thread", 513),
        lambda x: x.__setitem__("driver", "placeholder"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_query_row_attention_native_smoke() -> None:
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
        raise SystemExit("usage: test_cuda_query_row_attention.py --regenerate-fixture")
    _regenerate()
