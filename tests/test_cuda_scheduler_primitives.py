from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cuda_test_support import cuda_test_tier, run_cuda_suite

ROOT = Path(__file__).resolve().parents[1]


def test_scheduler_primitives_contract_fixture_and_handbook_are_connected() -> None:
    contract = json.loads(
        (ROOT / "pins" / "cuda_scheduler_primitives_contract.json").read_text()
    )
    fixture = json.loads(
        (ROOT / "fixtures" / "cuda_scheduler_primitives.json").read_text()
    )
    shape = contract["production_shape"]
    assert shape["residual_width"] == 5_120
    assert shape["ffn_width"] == 17_408
    assert contract["resident_weight_formats"] == ["Q4_K", "Q6_K", "Q8_0"]
    assert all(
        case["transient_q8_exact"] and case["nonfinite"] == 0
        for case in fixture["q8_0_mmv"]
    )
    assert fixture["embedding"]["bf16_exact"]
    assert fixture["layouts"]["attention_split_exact"]
    assert fixture["gdn"]["exact"]
    primitives = (ROOT / "cuda" / "scheduler_primitives_test.cu").read_text()
    assert "production_numerics_path" in primitives
    chapter = (ROOT / "docs" / "45-cuda-scheduler-primitives.md").read_text().casefold()
    for term in [
        "resident weights",
        "transient activation",
        "embedding row",
        "rmsnorm",
        "residual",
        "swiglu",
        "tiled",
        "grouped",
        "proof boundary",
    ]:
        assert term in chapter


def test_scheduler_primitives_match_device_references() -> None:
    if os.environ.get("QW38_RUN_CUDA_TESTS") != "1":
        pytest.skip("set QW38_RUN_CUDA_TESTS=1 for the exclusive RTX 5090 gate")
    tier = cuda_test_tier()
    outputs = run_cuda_suite(tier)
    quant = outputs["quant"]
    q8_mmv = [line for line in quant.splitlines() if line.startswith("case=q8_0_")]
    assert [line.split()[0] for line in q8_mmv] == [
        "case=q8_0_17x256",
        "case=q8_0_257x512",
    ]
    for line in q8_mmv:
        fields = dict(field.split("=", 1) for field in line.split())
        assert fields["q8_equal"] == "true"
        assert fields["nonfinite"] == "0"
        assert float(fields["max_abs"]) <= 3.0e-4
        assert float(fields["rms"]) <= 2.0e-4
        assert float(fields["mean_ms"]) > 0.0
    q8_mmq = next(
        line for line in quant.splitlines() if line.startswith("mmq_case=q8_0_")
    )
    q8_mmq_fields = dict(field.split("=", 1) for field in q8_mmq.split())
    assert q8_mmq_fields["q8_equal"] == "true"
    assert q8_mmq_fields["nonfinite"] == "0"
    assert float(q8_mmq_fields["max_abs"]) <= 5.0e-4
    assert float(q8_mmq_fields["rms"]) <= 2.5e-4

    run = outputs["scheduler"]
    lines = run.splitlines()
    pointwise = dict(
        field.split("=", 1)
        for field in next(
            line for line in lines if line.startswith("scheduler_pointwise=")
        ).split()
    )
    assert pointwise["nonfinite"] == "0"
    assert float(pointwise["max_abs"]) <= 0.0078125
    assert float(pointwise["rms"]) <= 0.0005
    assert float(pointwise["mean_ms"]) > 0.0
    layout = dict(
        field.split("=", 1)
        for field in next(
            line for line in lines if line.startswith("scheduler_layout=")
        ).split()
    )
    assert layout["split_exact"] == "true"
    assert layout["nonfinite"] == "0"
    assert float(layout["max_abs"]) <= 5.0e-7
    assert float(layout["rms"]) <= 1.0e-7
    assert "scheduler_embedding=q4_k bf16_exact=true" in lines
    assert "scheduler_gdn=tiled_to_grouped exact=true" in lines
    assert any("production_numerics_path=mixed" in line for line in lines)
    assert any("optimized_admitted=true" in line for line in lines)
    assert any("unrepresented=strict" in line for line in lines)
    assert any("strict_reference=retained" in line for line in lines)
    assert f"test_tier={tier}" in lines
    assert "status=passed" in lines
