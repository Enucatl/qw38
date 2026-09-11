from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cuda_test_support import cuda_test_tier, run_cuda_suite

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_quant_contract_and_handbook_are_connected() -> None:
    contract = json.loads((ROOT / "pins" / "cuda_quant_contract.json").read_text())
    fixture = json.loads((ROOT / "fixtures" / "cuda_quant_mmv.json").read_text())
    assert contract["target"] == "sm_120"
    assert contract["admission"]["maximum_absolute_error"] == 3.0e-4
    assert contract["admission"]["maximum_rms_error"] == 2.0e-4
    assert len(fixture["cases"]) == 4
    assert all(case["q8_equal"] for case in fixture["cases"])
    mmv = (ROOT / "cuda" / "quant_mmv.cu").read_text()
    assert 'kSelectedProductionNumericsPath[] = "strict"' in mmv
    assert "kProductionAdmission[]" in mmv
    assert (
        "production_numerics_optimized_admitted() noexcept { return false; }" not in mmv
    )
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
    assert contract["admission"]["q4k_q6k_association_gate"] == "ds4_q4k_parity"
    assert contract["admission"]["q4k_q6k_abs_scale"] == 0.2
    assert contract["admission"]["q4k_q6k_rel_tol"] == 0.05
    assert contract["admission"]["legacy_fixed_abs_rms_retired_for_q4k_q6k_mmq"] is True
    assert contract["admission"]["maximum_absolute_error"] == 5.0e-4
    assert contract["admission"]["maximum_rms_error"] == 2.5e-4
    assert contract["admission"]["maximum_absolute_error_applies_to"].startswith(
        "q8_0_mmq"
    )
    assert [case["prompt_rows"] for case in fixture["cases"]] == [3, 5, 1, 9]
    assert all(case["q8_equal"] and case["nonfinite"] == 0 for case in fixture["cases"])
    chapter = (ROOT / "docs" / "40-cuda-prompt-mmq.md").read_text().casefold()
    for term in ["prompt row", "tile", "weight reuse", "token-major", "tail"]:
        assert term in chapter


def test_cuda_quant_mmv_matches_scalar_reference() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")

    tier = cuda_test_tier()
    run = run_cuda_suite(tier)["quant"]
    lines = [
        line
        for line in run.splitlines()
        if line.startswith(("case=q4_k_", "case=q6_k_"))
    ]
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
        line
        for line in run.splitlines()
        if line.startswith(("mmq_case=q4_k_", "mmq_case=q6_k_"))
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
        assert fields["association_bad"] == "0"
        assert fields["gate"] == "ds4_q4k_parity"
        assert fields["ref"] == "cpu_dequant_gemm"
        assert float(fields["mean_ms"]) > 0.0
    assert "status=passed" in run
    assert f"test_tier={tier}" in run
    assert "production_numerics_path=mixed" in run
    assert "optimized_admitted=true" in run
    assert "unrepresented=strict" in run
    assert "strict_reference=retained" in run
    summary = next(
        line for line in run.splitlines() if line.startswith("test_run_end_epoch_ms=")
    )
    summary_fields = dict(field.split("=", 1) for field in summary.split())
    assert int(summary_fields["reference_sampled_calls"]) > 0
    assert int(summary_fields["reference_sampled_points"]) > 0
    assert "reference_policy=sampled_large_cases" in summary
    assert any(line.startswith("test_phase=") for line in run.splitlines())
