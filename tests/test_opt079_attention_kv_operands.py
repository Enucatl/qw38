"""Host tests for OPT-079 prompt attention KV operand conversion reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt075_q4_production_admission import AdmissionError
from tools.opt079_attention_kv_operands import (
    CANDIDATE_ID,
    CONFIGS,
    CONTROL_ID,
    CONTRACT,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    REPORT,
    TINY_CASES,
    decide_verdict,
    dispatch_matches,
    family_plan,
    parse_attn_dispatch,
    pick_survivor,
    tiny_cases,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_performance_admission,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
PIPELINE = ROOT / "cuda/fattn_mma_f16_pipeline.cuh"
FATTN = ROOT / "cuda/fattn_mma_f16.cuh"
NATIVE = ROOT / "cuda/opt079_attention_kv_operands_test.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
DECODE_H = ROOT / "cuda/attention_decode.h"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-079")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-079"
    assert contract["claims_throughput"] is False
    assert contract["min_saving_ms_per_prompt"] == MIN_SAVING_MS
    assert len(contract["configurations"]) == 2
    assert iteration["task"] == "OPT-079"
    assert iteration["diagnostics_make_target"] == "cuda-opt079-diagnostics"
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert fixture["shipping_attention_pipeline"] == "kv_once"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt079-diagnostics" in makefile
    assert "qw38-cuda-opt079-attention-kv-operands-test" in makefile
    pipe = PIPELINE.read_text(encoding="utf-8")
    fattn = FATTN.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    header = DECODE_H.read_text(encoding="utf-8")
    assert 'kSelectedAttentionPipelinePath[] = "kv_once"' in fattn
    assert 'kLegalAttentionPipelineKvOnce[] = "kv_once"' in pipe
    assert "fattn_pipeline_convert_kv_stage" in pipe
    assert "ConvertKvOnce" in pipe
    assert "KeysAreF16" in fattn
    assert "fattn_load_value_f16_pair" in pipe
    assert "apply_attention_pipeline_ident" in header
    assert "attn-ab" in probe
    assert "--attention-pipeline" in probe
    assert "kv_once" in native
    assert "stage_lifetime" in native


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-079")
    attention = iteration["workloads"]["attention"]
    assert loop_product(workload_for_mode(attention, "feedback")) == 8
    assert loop_product(workload_for_mode(attention, "acceptance")) == 26
    plan = family_plan("attention", "feedback")
    assert plan["candidates"] == 2
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    accept = family_plan("attention", "acceptance")
    assert accept["candidates"] == 2
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-079", "feedback", iteration, "attention")
    assert "phase=attention" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "in-place f16" in proof
    assert "no gqa6" in proof
    assert ">=5 ms/prompt" in proof
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300


def test_tiny_cases_cover_rows_and_ring() -> None:
    cases = tiny_cases()
    shapes = {(row["tokens"], row["start"]) for row in cases}
    assert shapes == {(1, 96), (17, 96), (33, 96)}
    native = NATIVE.read_text(encoding="utf-8")
    assert "kTinyTokens[] = {1, 17, 33}" in native
    assert "ring_wraps=2" in native
    assert "stage_lifetime" in native
    for spec in TINY_CASES:
        assert spec["tokens"] in {1, 17, 33}


def test_dispatch_uses_kv_once_launch() -> None:
    control = CONFIGS[0]
    candidate = CONFIGS[1]
    observed_c = {
        "path": "f16_async",
        "launch": "fattn_mma_pipeline_f16_async",
        "convert_once": 0,
        "occupancy": 1,
    }
    observed_k = {
        "path": "kv_once",
        "launch": "fattn_mma_pipeline_kv_once",
        "convert_once": 1,
        "occupancy": 1,
    }
    assert dispatch_matches(observed_c, control) is True
    assert dispatch_matches(observed_k, candidate) is True
    wrong = dict(observed_k)
    wrong["convert_once"] = 0
    assert dispatch_matches(wrong, candidate) is False
    parsed = parse_attn_dispatch(
        "round family=prompt-attn cache_mode=rotating warmup=false "
        "sample_index=0 observation_unit=independent_round enclosing_ms=1.0 "
        "kernel_only_ms=1.0 attention_layers=16 path=kv_once "
        "launch=fattn_mma_pipeline_kv_once convert_once=1 occupancy=1",
        "kv_once",
    )
    assert parsed["path"] == "kv_once"
    assert parsed["convert_once"] == 1


def test_negative_saving_and_below_threshold_reject() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    numeric = {"nonfinite": 0}
    slower = decide_verdict(
        component={
            "positive": False,
            "mean_diff_ms": -1.0,
            "ci95_high": -0.2,
            "ci95_low": -1.5,
        },
        engine={
            "control_ms": [1400.0] * 5,
            "control_mean_ms": 1400.0,
            "regression_upper_ms": 0.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="kv_once",
    )
    assert slower["verdict"] == "performance_rejected"
    small = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 1.0,
            "ci95_high": 1.5,
            "ci95_low": 0.5,
        },
        engine={
            "control_ms": [1400.0] * 5,
            "control_mean_ms": 1400.0,
            "regression_upper_ms": 0.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="kv_once",
    )
    assert small["verdict"] == "performance_rejected"
    assert "saving_below_5_ms_prompt" in small["reasons"]
    keep = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 8.0,
            "ci95_high": 10.0,
            "ci95_low": 6.0,
        },
        engine={
            "control_ms": [1400.0] * 5,
            "control_mean_ms": 1400.0,
            "regression_upper_ms": 1.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="kv_once",
    )
    assert keep["verdict"] == "keep"
    assert keep["production_kept"] is True


def test_pick_survivor_is_kv_once() -> None:
    survivor = pick_survivor(
        {
            CONTROL_ID: {"mean_ms": 260.0},
            CANDIDATE_ID: {"mean_ms": 240.0},
        }
    )
    assert survivor == CANDIDATE_ID
    assert survivor != CONTROL_ID


def test_spill_occupancy_and_dispatch_failure_are_hard_stops() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    numeric = {"nonfinite": 0}
    spill = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 8.0,
            "ci95_high": 10.0,
            "ci95_low": 6.0,
        },
        engine={
            "control_ms": [1400.0] * 5,
            "control_mean_ms": 1400.0,
            "regression_upper_ms": 1.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="kv_once",
        local_bytes=256,
    )
    assert spill["verdict"] == "performance_rejected"
    occ = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 8.0,
            "ci95_high": 10.0,
            "ci95_low": 6.0,
        },
        engine={
            "control_ms": [1400.0] * 5,
            "control_mean_ms": 1400.0,
            "regression_upper_ms": 1.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="kv_once",
        occupancy_loss=True,
    )
    assert occ["verdict"] == "performance_rejected"
    with pytest.raises(AdmissionError, match="nonfinite"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 8.0},
            engine=None,
            quality=quality,
            numeric={"nonfinite": 1},
            dispatch_ok=True,
            mode="acceptance",
            survivor="kv_once",
        )
    with pytest.raises(AdmissionError, match="dispatch"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 8.0},
            engine=None,
            quality=quality,
            numeric={"nonfinite": 0},
            dispatch_ok=False,
            mode="acceptance",
            survivor="kv_once",
        )


def test_screen_keep_is_rejected_and_counts_must_match() -> None:
    iteration = load_contract("OPT-079")
    attention = workload_for_mode(iteration["workloads"]["attention"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 2,
            "observed_shapes": 1,
            "observed_tier": "screen",
            "pairs": 1,
            "sample_ids": [0, 1, 2],
            "acceptance_executed": False,
            "keep": True,
        }
    )
    screen_keep = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="attention",
        workload=attention,
        stdout=stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    good = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 2,
            "observed_shapes": 1,
            "observed_tier": "screen",
            "pairs": 1,
            "sample_ids": [0, 1, 2],
            "acceptance_executed": False,
            "keep": False,
        }
    )
    admitted = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="attention",
        workload=attention,
        stdout=good,
        success=True,
    )
    assert admitted["ok"] is True


def test_no_gqa6_retry_or_global_converted_kv() -> None:
    pipe = PIPELINE.read_text(encoding="utf-8")
    fattn = FATTN.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "fattn_pipeline_convert_kv_stage" in pipe
    assert 'kSelectedAttentionPipelinePath[] = "kv_once"' in fattn
    assert "ConvertKvOnce" in pipe
    assert "kv_once" in pipe
    convert = pipe.split("fattn_pipeline_convert_kv_stage")[1][:500]
    assert "cudaMalloc" not in convert
    assert "stage_lifetime" in native
    assert "no_clamping=" in native
