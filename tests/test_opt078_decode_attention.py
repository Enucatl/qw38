"""Host tests for OPT-078 decode query-prep and vector KV admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt078_decode_attention import (
    CONFIGS,
    CONTROL_ID,
    CONTRACT,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    REPORT,
    TINY_POSITIONS,
    decide_verdict,
    dispatch_matches,
    family_plan,
    parse_attn_dispatch,
    pick_survivor,
    tiny_positions,
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
ATTN = ROOT / "cuda/attention_decode.cu"
ATTN_H = ROOT / "cuda/attention_decode.h"
ATTN_PATH = ROOT / "cuda/attention_decode_path.cuh"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
REPLAY_H = ROOT / "cuda/optimization_component_replay.h"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
NATIVE = ROOT / "cuda/opt078_decode_attention_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
SCHEDULER_H = ROOT / "cuda/full_scheduler.h"
PLAN = ROOT / "plan.md"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-078")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert ATTN_PATH.is_file()
    assert contract["task"] == "OPT-078"
    assert contract["claims_throughput"] is False
    assert contract["n_parts"] == 16
    assert contract["prompt_path_unchanged"] is True
    assert contract["partition_count_fixed"] is True
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert len(contract["configurations"]) == 3
    assert iteration["task"] == "OPT-078"
    assert iteration["diagnostics_make_target"] == "cuda-opt078-diagnostics"
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["modes"]["release"]["skip_compile"] is True
    assert fixture["shipping_decode_query_prep"] == "warp_query"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt078-diagnostics" in makefile
    assert "qw38-cuda-opt078-decode-attention-test" in makefile
    path = ATTN_PATH.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    header = ATTN_H.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    replay_h = REPLAY_H.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    scheduler_h = SCHEDULER_H.read_text(encoding="utf-8")
    assert 'kSelectedDecodeQueryPrepPath[] = "warp_query"' in path
    assert "prepared_q" in path and "prepared_q_veckv" in path
    assert "DecodeQueryPrepPathScope" in path
    assert "prepare_decode_query" in attn
    assert "query_heads, kWarpThreads" in attn or "block = 32" in attn.lower()
    assert "normalized_query" in attn
    assert "uint4" in attn or "16-byte" in attn or "load_bf16x8" in attn
    assert "alignment" in attn.casefold() or "unaligned" in attn.casefold()
    assert "Never skip because the destination pointer" in attn
    assert "launch_prepare_decode_query" in header
    assert "kDecodeAttention" in replay_h
    assert "decode-attention" in replay
    assert "--decode-query-prep" in replay
    assert "attn-ab" in probe
    assert "--decode-query-prep" in probe
    assert "maybe_capture_decode_attention" in scheduler
    assert "capture_decode_attention" in scheduler_h
    assert "tiny_p0" in native or "position=0" in native or "tiny_p0" in native
    assert "prepared_q_veckv" in native


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-078")
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    attention = iteration["workloads"]["attention"]
    quality = iteration["workloads"]["quality-state"]
    assert loop_product(workload_for_mode(d128, "feedback")) == 12
    assert loop_product(workload_for_mode(d2048, "feedback")) == 12
    assert loop_product(workload_for_mode(attention, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 1
    plan = family_plan("d128", "feedback")
    assert plan["candidates"] == 3
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    assert plan["cases"] == 1
    assert plan["engine_pairs"] == 1
    assert plan["decode_position"] == 128
    d2 = family_plan("d2048", "feedback")
    assert d2["decode_position"] == 2048
    accept = family_plan("attention", "acceptance")
    assert accept["candidates"] == 2
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-078", "feedback", iteration, "d128")
    assert "phase=d128" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "n_parts fixed at 16" in proof
    assert "one warp per query head" in proof
    assert "no prefill/prompt path change" in proof
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300
    assert iteration["modes"]["feedback"]["aggregate_deadline_s"] == 300


def test_tiny_positions_and_source_proofs() -> None:
    assert set(tiny_positions()) == {0, 1, 31, 32}
    native = NATIVE.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    for position in TINY_POSITIONS:
        assert str(position) in native
    for position in (127, 128, 2047, 2048):
        assert str(position) in native
    assert "grid = query_heads" in attn or "config.query_heads, kWarpThreads" in attn
    assert "kWarpThreads" in attn
    assert "load_bf16x8" in attn
    assert "atomicOr" in attn or "unaligned" in attn
    assert "stage_and_validate_chunk" in attn
    assert "Never skip because the destination pointer is unchanged" in attn


def test_dispatch_uses_prep_and_vec_variants() -> None:
    control = CONFIGS[0]
    prepared = CONFIGS[1]
    veckv = CONFIGS[2]
    observed_ctrl = {
        "path": "warp_query",
        "launch": "warp_query_decode_attention",
        "prep_grid": 0,
        "n_parts": 16,
        "prep_launches": 0,
        "prepared_q": False,
        "vec_kv": False,
    }
    observed_prep = {
        "path": "prepared_q",
        "launch": "prepare_decode_query+warp_query_prepared_q",
        "prep_grid": 24,
        "n_parts": 16,
        "prep_launches": 1,
        "prepared_q": True,
        "vec_kv": False,
    }
    observed_vec = {
        "path": "prepared_q_veckv",
        "launch": "prepare_decode_query+warp_query_prepared_q_veckv",
        "prep_grid": 24,
        "n_parts": 16,
        "prep_launches": 1,
        "prepared_q": True,
        "vec_kv": True,
    }
    assert dispatch_matches(observed_ctrl, control) is True
    assert dispatch_matches(observed_prep, prepared) is True
    assert dispatch_matches(observed_vec, veckv) is True
    wrong = dict(observed_prep)
    wrong["n_parts"] = 8
    assert dispatch_matches(wrong, prepared) is False
    parsed = parse_attn_dispatch(
        "decode_attention_dispatch path=prepared_q "
        "launch=prepare_decode_query+warp_query_prepared_q prep_grid=24 "
        "prep_block=32 n_parts=16 prep_launches=1 prepared_q=true vec_kv=false "
        "attention_layers=16 capture_positions=2"
    )
    assert parsed["path"] == "prepared_q"
    assert parsed["prep_grid"] == 24
    assert parsed["n_parts"] == 16


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
            "control_ms": [50.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="prepared_q",
    )
    assert slower["verdict"] == "performance_rejected"
    small = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 0.05,
            "ci95_high": 0.08,
            "ci95_low": 0.02,
        },
        engine={
            "control_ms": [50.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.0,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="prepared_q_veckv",
    )
    assert small["verdict"] == "performance_rejected"
    assert "saving_below_0_10_ms" in small["reasons"]
    keep = decide_verdict(
        component={
            "positive": True,
            "mean_diff_ms": 1.0,
            "ci95_high": 1.2,
            "ci95_low": 0.8,
        },
        engine={
            "control_ms": [50.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.01,
        },
        quality=quality,
        numeric=numeric,
        dispatch_ok=True,
        mode="acceptance",
        survivor="prepared_q",
    )
    assert keep["verdict"] == "keep"
    assert keep["production_kept"] is True


def test_pick_survivor_is_the_faster_prepared_path() -> None:
    survivor = pick_survivor(
        {
            CONTROL_ID: {"mean_ms": 2.4},
            "prepared_q": {"mean_ms": 1.8},
            "prepared_q_veckv": {"mean_ms": 1.5},
        }
    )
    assert survivor == "prepared_q_veckv"
    assert survivor != CONTROL_ID


def test_dispatch_failure_is_a_hard_stop() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    with pytest.raises(AssertionError, match="nonfinite"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 1.0},
            engine=None,
            quality=quality,
            numeric={"nonfinite": 1},
            dispatch_ok=True,
            mode="acceptance",
            survivor="prepared_q",
        )
    with pytest.raises(AssertionError, match="dispatch"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 1.0},
            engine=None,
            quality=quality,
            numeric={"nonfinite": 0},
            dispatch_ok=False,
            mode="acceptance",
            survivor="prepared_q",
        )


def test_screen_keep_is_rejected_and_counts_must_match() -> None:
    iteration = load_contract("OPT-078")
    d128 = workload_for_mode(iteration["workloads"]["d128"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 3,
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
        workload_name="d128",
        workload=d128,
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
            "observed_candidates": 3,
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
        workload_name="d128",
        workload=d128,
        stdout=good,
        success=True,
    )
    assert admitted["ok"] is True


def test_forbidden_prefill_partition_sweep_and_plan_untouched() -> None:
    attn = ATTN.read_text(encoding="utf-8")
    path = ATTN_PATH.read_text(encoding="utf-8")
    iteration = load_contract("OPT-078")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "no partition-count or gqa-layout sweep" in proof
    assert "no prefill/prompt path change" in proof
    assert k_selected_parts_fixed(attn)
    assert 'kSelectedDecodeQueryPrepPath[] = "warp_query"' in path
    assert "prepare_prompt_query_kernel" in attn
    decode = attn.split("prepare_decode_query")[1].split(
        "template <bool kUsePreparedQ, bool kVecKv>"
    )[0]
    assert "kWarpThreads" in decode
    assert PLAN.is_file()


def k_selected_parts_fixed(attn: str) -> bool:
    return (
        "kSelectedDecodeKvPartsLow = 16" in attn
        and "kSelectedDecodeKvPartsHigh = 16" in attn
    )
