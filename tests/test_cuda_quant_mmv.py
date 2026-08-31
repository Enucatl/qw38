from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "qw38-cuda:13.0.2"


def test_cuda_quant_contract_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_quant_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_quant_mmv.json").read_text())
    assert contract["target"] == "sm_120"
    assert contract["admission"]["maximum_absolute_error"] == 3.0e-4
    assert contract["admission"]["maximum_rms_error"] == 2.0e-4
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert len(fixture["cases"]) == 4
    assert all(case["q8_equal"] for case in fixture["cases"])
    chapter = (ROOT / "docs" / "39-cuda-quant-mmv.md").read_text().casefold()
    for term in [
        "matrix-vector multiplication",
        "transient",
        "warp",
        "relative error",
        "boundary",
    ]:
        assert term in chapter


def test_cuda_mmq_contract_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_mmq_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_quant_mmq.json").read_text())
    assert contract["target"] == "sm_120"
    assert contract["tile"] == {
        "prompt_rows": 4,
        "output_rows": 8,
        "threads": 256,
        "ownership": "one warp owns one output row and up to four prompt rows",
    }
    assert contract["admission"]["maximum_absolute_error"] == 5.0e-4
    assert contract["admission"]["maximum_rms_error"] == 2.5e-4
    for relative, expected in contract["local_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert [case["prompt_rows"] for case in fixture["cases"]] == [3, 5, 1, 9]
    assert all(case["q8_equal"] and case["nonfinite"] == 0 for case in fixture["cases"])
    chapter = (ROOT / "docs" / "40-cuda-prompt-mmq.md").read_text().casefold()
    for term in ["prompt row", "tile", "weight reuse", "token-major", "tail"]:
        assert term in chapter


def test_cuda_quant_mmv_matches_scalar_reference() -> None:
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
        [*common, "make", "build/qw38-cuda-quant-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [*common, "./build/qw38-cuda-quant-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    lines = [line for line in run.stdout.splitlines() if line.startswith("case=")]
    assert [line.split()[0] for line in lines] == [
        "case=q4_k_17x256",
        "case=q4_k_257x512",
        "case=q6_k_17x256",
        "case=q6_k_257x512",
    ]
    for line in lines:
        fields = dict(field.split("=", 1) for field in line.split())
        assert fields["q8_equal"] == "true"
        assert fields["nonfinite"] == "0"
        assert float(fields["max_abs"]) <= 3.0e-4
        assert float(fields["rms"]) <= 2.0e-4
        assert float(fields["mean_ms"]) > 0.0
    mmq_lines = [
        line for line in run.stdout.splitlines() if line.startswith("mmq_case=")
    ]
    assert [line.split()[0] for line in mmq_lines] == [
        "mmq_case=q4_k_3x17x256",
        "mmq_case=q4_k_5x257x512",
        "mmq_case=q6_k_1x17x256",
        "mmq_case=q6_k_9x257x512",
    ]
    for line in mmq_lines:
        fields = dict(field.split("=", 1) for field in line.split())
        assert fields["q8_equal"] == "true"
        assert fields["nonfinite"] == "0"
        assert float(fields["max_abs"]) <= 5.0e-4
        assert float(fields["rms"]) <= 2.5e-4
        assert float(fields["mean_ms"]) > 0.0
    assert "status=passed" in run.stdout
