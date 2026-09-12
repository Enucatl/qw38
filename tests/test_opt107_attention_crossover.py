"""Host tests for OPT-107 prefix-aware decode attention selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt107_attention_crossover import (
    CANDIDATE_ID,
    CONTRACT,
    CONTROL_ID,
    FALLBACK_128K,
    FIXTURE,
    HYBRID_ID,
    ITERATION,
    MEASURE_POSITIONS,
    REPORT,
    SCREEN_THRESHOLDS,
    VERIFIED_MAX,
    choose_threshold,
    decide_verdict,
    expected_hybrid_config,
    family_plan,
    path_for_position,
    selected_threshold,
)
from tools.opt075_q4_production_admission import load_json
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
ATTN_PATH = ROOT / "cuda/attention_decode_path.cuh"
ATTN = ROOT / "cuda/attention_decode.cu"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
NATIVE = ROOT / "cuda/opt107_attention_crossover_test.cu"
OPT103_TEST = ROOT / "cuda/opt103_vector_attention_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-107")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    path = ATTN_PATH.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-107"
    assert contract["uses_opt098_arrays_for_keep_reject"] is False
    assert contract["opt103_kernels_unchanged"] is True
    assert contract["screen_thresholds"] == [512, 1024, 1536, 2048]
    assert contract["measure_positions"] == [128, 512, 1024, 1536, 2048, 4096]
    assert contract["fallback_position"] == 131072
    assert contract["candidate"] == HYBRID_ID
    assert iteration["task"] == "OPT-107"
    assert iteration["diagnostics_make_target"] == "cuda-opt107-diagnostics"
    assert "cuda-opt107-diagnostics" in makefile
    assert "qw38-cuda-opt107-attention-crossover-test" in makefile
    assert 'kSelectedDecodeAttentionVec128Path[] = "warp_query"' in path
    assert "kSelectedDecodeAttentionCrossoverThreshold = " in path
    assert "decode_attention_vec128_path_for_position" in path
    assert "decode_attention_vec128_uses_online_at" in attn
    assert "--decode-attention-crossover-threshold" in replay
    assert "--decode-position 128|512|1024|1536|2048|4096" in replay
    assert "--decode-attention-crossover-threshold" in probe
    assert "path_for_position" in native
    assert "bound_tm1" in native
    assert "131072" in native
    fixture = load_json(FIXTURE)
    assert fixture["uses_opt098_arrays_for_keep_reject"] is False
    assert fixture["shipping_decode_attention_vec128"] == CONTROL_ID


def test_opt103_kernels_not_rewritten() -> None:
    attn = ATTN.read_text(encoding="utf-8")
    native_opt103 = OPT103_TEST.read_text(encoding="utf-8")
    assert "vec128_online_decode_attention" in attn
    assert "__launch_bounds__(kVec128Threads, 1)" in attn
    assert "QW38_OPT103_VECTOR_ATTENTION_RESULT=" in native_opt103
    path = ATTN_PATH.read_text(encoding="utf-8")
    assert 'kSelectedDecodeAttentionVec128Path[] = "warp_query"' in path
    assert "kSelectedVec128NParts = 16" in path


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-107")
    eligibility = iteration["workloads"]["eligibility"]
    screen = iteration["workloads"]["screen"]
    select = iteration["workloads"]["select"]
    parity = iteration["workloads"]["parity"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    fallback = iteration["workloads"]["fallback-128k"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(screen, "feedback")) == 48
    assert loop_product(workload_for_mode(select, "feedback")) == 1
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 832
    assert loop_product(workload_for_mode(fallback, "acceptance")) == 1
    described = describe_plan("OPT-107", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt-098" in proof
    assert "warp_query" in proof
    assert "4096" in proof


def test_path_for_position_keeps_warp_query_outside_range() -> None:
    assert path_for_position(128, threshold=2048) == CONTROL_ID
    assert path_for_position(2047, threshold=2048) == CONTROL_ID
    assert path_for_position(2048, threshold=2048) == CANDIDATE_ID
    assert path_for_position(4096, threshold=2048) == CANDIDATE_ID
    assert path_for_position(4097, threshold=2048) == CONTROL_ID
    assert path_for_position(FALLBACK_128K, threshold=2048) == CONTROL_ID
    assert path_for_position(4096, threshold=0) == CONTROL_ID
    assert path_for_position(128, threshold=512) == CONTROL_ID
    assert path_for_position(512, threshold=512) == CANDIDATE_ID
    expected = expected_hybrid_config(128, 2048)
    assert expected["decode_attention_vec128"] == CONTROL_ID
    assert expected["expected_launch"] == "warp_query_decode_attention"
    assert expected["expected_prep_launches"] == 0
    long_cfg = expected_hybrid_config(2048, 2048)
    assert long_cfg["decode_attention_vec128"] == CANDIDATE_ID
    assert long_cfg["expected_block_y"] == 4


def test_choose_lowest_positive_threshold() -> None:
    by_position = {
        128: {"positive": False, "mean_diff_ms": -0.10},
        512: {"positive": False, "mean_diff_ms": -0.01},
        1024: {"positive": True, "mean_diff_ms": 0.05},
        1536: {"positive": True, "mean_diff_ms": 0.08},
        2048: {"positive": True, "mean_diff_ms": 0.39},
        4096: {"positive": True, "mean_diff_ms": 0.50},
    }
    chosen = choose_threshold(by_position)
    assert chosen["admitted"] is True
    assert chosen["threshold"] == 1024
    missing_4096 = dict(by_position)
    missing_4096[4096] = {"positive": False, "mean_diff_ms": -0.02}
    rejected = choose_threshold(missing_4096)
    assert rejected["admitted"] is False
    assert rejected["threshold"] == 0
    point_only = {
        pos: {"positive": False, "mean_diff_ms": 0.20} for pos in MEASURE_POSITIONS
    }
    assert choose_threshold(point_only)["admitted"] is False


def test_d128_below_every_screen_threshold() -> None:
    for threshold in SCREEN_THRESHOLDS:
        assert 128 < threshold
        assert path_for_position(128, threshold=threshold) == CONTROL_ID
    assert VERIFIED_MAX == 4096
    plan = family_plan("d128", "acceptance")
    assert plan["decode_position"] == 128
    assert plan["engine_pairs"] == 5


def test_selected_threshold_recovers_from_screen_select() -> None:
    assert selected_threshold({"selected_threshold": 0}) == 0
    assert (
        selected_threshold(
            {
                "selected_threshold": 0,
                "select": {"threshold": 1024, "admitted": True},
                "screen": {"selected_threshold": 1024},
            }
        )
        == 1024
    )
    assert selected_threshold({"selected_threshold": 2048}) == 2048
    assert selected_threshold({"selected_threshold": 7}) == 0


def test_decide_verdict_incomplete_does_not_reject() -> None:
    quality = {"quality_v3_engine_non_regression": "pass"}
    incomplete = decide_verdict(
        threshold=1024,
        d128={"component": {"selected_path": "warp_query"}},
        d2048=None,
        quality=quality,
        mode="acceptance",
    )
    assert incomplete["verdict"] == "inconclusive"
    assert incomplete["production_kept"] is False
    no_threshold = decide_verdict(
        threshold=0,
        d128=None,
        d2048=None,
        quality=quality,
        mode="acceptance",
    )
    assert no_threshold["verdict"] == "no_threshold"


def test_future_keep_policy_and_no_opt098_keep() -> None:
    iteration = load_contract("OPT-107")
    validate_future_keep_policy("OPT-107", iteration)
    assert iteration["modes"]["release"]["historical_oracles"] is False
    contract = _json(CONTRACT)
    assert contract["uses_opt098_arrays_for_keep_reject"] is False
