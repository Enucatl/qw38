"""Host tests for OPT-077 parallel decode GDN recurrence admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt077_gdn_decode import (
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
    parse_gdn_dispatch,
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
GDN_STEP = ROOT / "cuda/gdn_step.cu"
GDN_PATH = ROOT / "cuda/gdn_decode_path.cuh"
GDN_REC = ROOT / "cuda/gdn_decode_recurrence.cuh"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
REPLAY_H = ROOT / "cuda/optimization_component_replay.h"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
NATIVE = ROOT / "cuda/opt077_gdn_decode_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-077")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-077"
    assert contract["claims_throughput"] is False
    assert contract["transposed_state_forbidden"] is True
    assert contract["prompt_path_unchanged"] is True
    assert contract["convolution_separate"] is True
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert contract["recurrence_nll_ceiling"] == 0.02
    assert len(contract["configurations"]) == 3
    assert iteration["task"] == "OPT-077"
    assert iteration["diagnostics_make_target"] == "cuda-opt077-diagnostics"
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert fixture["shipping_gdn_decode"] == "sequential"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt077-diagnostics" in makefile
    assert "qw38-cuda-opt077-gdn-decode-test" in makefile
    path = GDN_PATH.read_text(encoding="utf-8")
    rec = GDN_REC.read_text(encoding="utf-8")
    step = GDN_STEP.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    replay_h = REPLAY_H.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert 'kSelectedGdnDecodePath[] = "sequential"' in path
    assert "tile16" in path and "tile32" in path
    assert "prepare_recurrence_decode_tiled" in rec
    assert "kGdnDecodeValueTile16" in path
    assert "kGdnDecodeValueTile32" in path
    assert "kGdnDecodeWarpsPerCta" in path
    assert "__shared__ float query_inverse" in rec
    assert "launch_gdn_decode_tiled_recurrence" in rec
    assert "gdn_decode_uses_tiled" in step
    assert "prepare_recurrence_window" in step
    assert "kSequentialWindows" in step
    assert "kDecodeGdn" in replay_h
    assert "decode-gdn" in replay
    assert "--gdn-decode" in replay
    assert "gdn-ab" in probe
    assert "--gdn-decode" in probe
    assert "maybe_capture_gdn_decode" in scheduler
    assert "W8_U1" in native or "tiny8" in native
    assert "tile16" in native and "tile32" in native


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-077")
    gdn = iteration["workloads"]["gdn"]
    quality = iteration["workloads"]["quality-state"]
    assert loop_product(workload_for_mode(gdn, "feedback")) == 24
    assert loop_product(workload_for_mode(gdn, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 1
    plan = family_plan("gdn", "feedback")
    assert plan["candidates"] == 3
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    assert plan["cases"] == 2
    accept = family_plan("gdn", "acceptance")
    assert accept["candidates"] == 2
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-077", "feedback", iteration, "gdn")
    assert "phase=gdn" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "value tiles 16 and 32" in proof
    assert "four warps" in proof
    assert "no prompt-path change" in proof
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300


def test_tiny_cases_cover_widths_and_updates() -> None:
    cases = tiny_cases()
    shapes = {(row["key_width"], row["value_width"], row["updates"]) for row in cases}
    assert shapes == {(8, 8, 1), (8, 8, 4), (32, 32, 1), (32, 32, 4)}
    native = NATIVE.read_text(encoding="utf-8")
    rec = GDN_REC.read_text(encoding="utf-8")
    assert "updates=1" in native or "u1" in native.lower() or "U1" in native
    assert "key_width=8" in native or "tiny8" in native
    assert "prepare_recurrence_decode_tiled<kGdnDecodeValueTile16>" in rec
    assert "prepare_recurrence_decode_tiled<kGdnDecodeValueTile32>" in rec
    for spec in TINY_CASES:
        assert spec["key_width"] in {8, 32}
        assert spec["updates"] in {1, 4}


def test_dispatch_uses_tiled_launch_variants() -> None:
    sequential = CONFIGS[0]
    tile16 = CONFIGS[1]
    tile32 = CONFIGS[2]
    observed16 = {
        "path": "tile16",
        "launch": "prepare_recurrence_decode_tiled16",
        "value_tile": 16,
        "warps_per_cta": 4,
        "grid_x": 48,
        "grid_y": 8,
    }
    observed32 = {
        "path": "tile32",
        "launch": "prepare_recurrence_decode_tiled32",
        "value_tile": 32,
        "warps_per_cta": 4,
        "grid_x": 48,
        "grid_y": 4,
    }
    observed_seq = {
        "path": "sequential",
        "launch": "prepare_recurrence_window",
        "value_tile": 0,
        "warps_per_cta": 4,
        "grid_x": 48,
        "grid_y": 1,
    }
    assert dispatch_matches(observed16, tile16) is True
    assert dispatch_matches(observed32, tile32) is True
    assert dispatch_matches(observed_seq, sequential) is True
    wrong = dict(observed16)
    wrong["value_tile"] = 32
    assert dispatch_matches(wrong, tile16) is False
    parsed = parse_gdn_dispatch(
        "gdn_dispatch path=tile16 launch=prepare_recurrence_decode_tiled16 "
        "value_tile=16 grid_x=48 grid_y=8 warps_per_cta=4 sequential_windows=false"
    )
    assert parsed["path"] == "tile16"
    assert parsed["grid_y"] == 8


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
        survivor="tile16",
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
        survivor="tile32",
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
        survivor="tile16",
    )
    assert keep["verdict"] == "keep"
    assert keep["production_kept"] is True


def test_pick_survivor_is_the_faster_tile_against_sequential() -> None:
    survivor = pick_survivor(
        {
            CONTROL_ID: {"mean_ms": 2.4},
            "tile16": {"mean_ms": 1.8},
            "tile32": {"mean_ms": 1.5},
        }
    )
    assert survivor == "tile32"
    assert survivor != CONTROL_ID


def test_spill_and_dispatch_failure_are_hard_stops() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    numeric = {"nonfinite": 0}
    spill = decide_verdict(
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
        survivor="tile16",
        local_bytes=256,
    )
    assert spill["verdict"] == "performance_rejected"
    assert spill["retain_reason"] == "resource_spill"
    with pytest.raises(AssertionError, match="nonfinite"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 1.0},
            engine=None,
            quality=quality,
            numeric={"nonfinite": 1},
            dispatch_ok=True,
            mode="acceptance",
            survivor="tile16",
        )
    with pytest.raises(AssertionError, match="dispatch"):
        decide_verdict(
            component={"positive": True, "mean_diff_ms": 1.0},
            engine=None,
            quality=quality,
            numeric={"nonfinite": 0},
            dispatch_ok=False,
            mode="acceptance",
            survivor="tile16",
        )


def test_screen_keep_is_rejected_and_counts_must_match() -> None:
    iteration = load_contract("OPT-077")
    gdn = workload_for_mode(iteration["workloads"]["gdn"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 3,
            "observed_shapes": 2,
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
        workload_name="gdn",
        workload=gdn,
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
            "observed_shapes": 2,
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
        workload_name="gdn",
        workload=gdn,
        stdout=good,
        success=True,
    )
    assert admitted["ok"] is True


def test_no_prompt_fusion_or_transpose() -> None:
    rec = GDN_REC.read_text(encoding="utf-8")
    step = GDN_STEP.read_text(encoding="utf-8")
    fused = ROOT.joinpath("cuda/gdn_fused_quality.cuh").read_text(encoding="utf-8")
    assert "transposed S is not a drop-in" in rec or "transposed S" in rec
    assert "Convolution stays a separate launch" in rec
    assert 'kSelectedGdnFusePath[] = "off"' in fused
    assert "launch_gdn_prepare_chunk_tiled" in step
    assert "gdn_decode_uses_tiled" in step
    sequential = step.split("launch_sequential_windows")[1].split(
        "launch_convolution_then_recurrence"
    )[0]
    assert "launch_gdn_decode_tiled_recurrence" in sequential
    prompt = step.split("cudaError_t launch_fused_token_loop(")[1].split(
        "cudaError_t launch_gdn_prepare_chunk_layout("
    )[0]
    assert "launch_gdn_decode_tiled_recurrence" not in prompt
