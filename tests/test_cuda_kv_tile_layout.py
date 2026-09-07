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
PREFIX = "QW38_KV_TILE_LAYOUT_RESULT="
CONTRACT = ROOT / "pins/cuda_kv_tile_layout_contract.json"
FIXTURE = ROOT / "fixtures/cuda_kv_tile_layout.json"


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
        "capacity",
        "round_trip_prefixes",
        "tile_span_prefixes",
        "exact_row_counts",
        "exact_prefixes",
        "launch",
        "proof_limit",
    }
    assert result["schema_version"] == 1 and result["task"] == "OPT-010"
    assert result["status"] == "measured"
    for key in contract["metadata_required"]:
        assert isinstance(result[key], str) and result[key].strip()
        assert "unknown" not in result[key].lower()
        assert "placeholder" not in result[key].lower()
    assert result["compute_capability"] == "12.0"
    assert result["toolkit"] == contract["toolkit"]
    assert result["pinned_image"] == contract["pinned_image"]
    assert result["production_shape"] == contract["production_shape"]
    assert "pointer-span" in result["proof_limit"]
    assert "Nsight" in result["proof_limit"]
    assert "end-to-end" in result["proof_limit"]
    assert set(result["semantic"]) == set(contract["semantic_predicates"])
    assert all(type(value) is bool and value for value in result["semantic"].values())
    assert (
        result["capacity"]["attention_cache_values"]
        == contract["attention_cache_values"]
    )
    assert (
        result["capacity"]["sixteen_layer_kv_bytes"]
        == contract["sixteen_layer_kv_bytes"]
    )
    assert result["round_trip_prefixes"] == contract["round_trip_prefixes"]
    assert result["tile_span_prefixes"] == contract["tile_span_prefixes"]
    assert result["exact_row_counts"] == contract["exact_row_counts"]
    assert result["exact_prefixes"] == contract["exact_prefixes"]
    launch = result["launch"]
    assert set(launch) == {
        "kernel_nodes",
        "staging_grid",
        "attention_grid",
        "staging_block",
        "attention_block",
        "dynamic_shared_bytes",
    }
    assert launch["kernel_nodes"] == 2
    assert launch["staging_grid"] == [4, 64, 1]
    assert launch["attention_grid"] == [4, 32, 1]
    assert launch["staging_block"] == [256, 1, 1]
    assert launch["attention_block"] == [256, 1, 1]
    assert launch["dynamic_shared_bytes"] == contract["dynamic_shared_bytes"]


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
            "cuda/kv_tile_layout_test.cu",
            "build/attention_decode.cuda.o",
            "-o",
            "build/qw38-cuda-kv-tile-layout-test",
        ],
        [*_common(), "./build/qw38-cuda-kv-tile-layout-test"],
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


def test_kv_tile_layout_contract_and_fixture_are_connected() -> None:
    validate_result(json.loads(FIXTURE.read_text()))


def test_kv_tile_layout_validator_rejects_inadmissible_evidence() -> None:
    fixture = json.loads(FIXTURE.read_text())
    mutations = []
    for mutate in (
        lambda x: x["semantic"].pop("finite_output"),
        lambda x: x["launch"].__setitem__("kernel_nodes", 3),
        lambda x: x["launch"]["staging_grid"].__setitem__(0, 3),
        lambda x: x["launch"]["attention_grid"].__setitem__(1, 31),
        lambda x: x["launch"].__setitem__("dynamic_shared_bytes", 33791),
        lambda x: x["capacity"].__setitem__("attention_cache_values", 1),
        lambda x: x.__setitem__("driver", "placeholder"),
        lambda x: x.__setitem__("proof_limit", "speedup claimed from Nsight"),
    ):
        changed = json.loads(json.dumps(fixture))
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)


def test_kv_tile_layout_native_smoke() -> None:
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
        raise SystemExit("usage: test_cuda_kv_tile_layout.py --regenerate-fixture")
    _regenerate()
