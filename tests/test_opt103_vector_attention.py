"""Host tests for OPT-103 128-thread vector attention admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt103_vector_attention import (
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    OPT090_FIXTURE,
    PART_CANDIDATES,
    REPORT,
    attention_vec_eligible,
    configs,
    decide_no_go_verdicts,
    dispatch_matches,
    family_plan,
    parse_attn_dispatch,
    replay_command,
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
ATTN_H = ROOT / "cuda/attention_decode.h"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
NATIVE = ROOT / "cuda/opt103_vector_attention_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-103")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    path = ATTN_PATH.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    header = ATTN_H.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-103"
    assert contract["claims_throughput"] is True
    assert contract["candidate"] == "vec128_online"
    assert contract["partition_candidates"] == [4, 8, 16]
    assert contract["warp_query_gqa6_rejected"] is True
    assert contract["threads"] == 128
    assert contract["head_width"] == 256
    assert len(contract["configurations"]) == 2
    assert iteration["task"] == "OPT-103"
    assert iteration["diagnostics_make_target"] == "cuda-opt103-diagnostics"
    assert "cuda-opt103-diagnostics" in makefile
    assert "qw38-cuda-opt103-vector-attention-test" in makefile
    shipping = str(
        (load_json(FIXTURE) if FIXTURE.is_file() else {}).get(
            "shipping_decode_attention_vec128"
        )
        or CONTROL_ID
    )
    n_parts = int(
        (load_json(FIXTURE) if FIXTURE.is_file() else {}).get("selected_n_parts") or 16
    )
    assert f'kSelectedDecodeAttentionVec128Path[] = "{shipping}"' in path
    assert 'kLegalDecodeAttentionVec128Online[] = "vec128_online"' in path
    assert f"kSelectedVec128NParts = {n_parts}" in path
    assert 'kSelectedDecodeAttentionGqaPath[] = "warp_query"' in path
    assert "vec128_online_decode_attention" in attn
    assert "__launch_bounds__(kVec128Threads, 1)" in attn
    assert "decode_kv_vec128_online_occupancy" in header
    assert "--decode-attention-vec128" in replay
    assert "--vec128-n-parts" in replay
    assert "--decode-attention-vec128" in probe
    assert "vec128_online" in native
    assert "flash_attn_ext_vec" in attn or "fattn-vec" in attn


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-103")
    eligibility = iteration["workloads"]["eligibility"]
    parts = iteration["workloads"]["parts"]
    parity = iteration["workloads"]["parity"]
    attention128 = iteration["workloads"]["attention128"]
    attention2048 = iteration["workloads"]["attention2048"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(parts, "feedback")) == 24
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(attention128, "feedback")) == 8
    assert loop_product(workload_for_mode(attention2048, "feedback")) == 8
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 832
    described = describe_plan("OPT-103", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "128 threads" in proof
    assert "{4,8,16}" in proof or "4,8,16" in proof
    assert "warp_query_gqa6 remains rejected" in proof


def test_two_configs_and_partitions() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["decode_attention_vec128"] == "warp_query"
    assert CONFIGS[1]["decode_attention_vec128"] == "vec128_online"
    assert PART_CANDIDATES == (4, 8, 16)
    assert configs(8)[1]["n_parts"] == 8
    assert all(int(row["expected_prep_launches"]) == 0 for row in CONFIGS)


def test_tiny_and_production_positions() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    for position in (0, 1, 31, 32, 127, 128, 2047, 2048):
        assert str(position) in native
    assert "89" in native


def test_dispatch_matches_vec128() -> None:
    control = CONFIGS[0]
    candidate = configs(8)[1]
    observed_ctrl = {
        "path": "warp_query",
        "launch": "warp_query_decode_attention",
        "prep_grid": 24,
        "prep_block": 32,
        "n_parts": 16,
        "prep_launches": 0,
        "prepared_q": False,
        "vec_kv": False,
        "gqa_block_y": 1,
    }
    observed_vec = {
        "path": "vec128_online",
        "launch": "vec128_online_decode_attention",
        "prep_grid": 24,
        "prep_block": 32,
        "n_parts": 8,
        "prep_launches": 0,
        "prepared_q": False,
        "vec_kv": False,
        "gqa_block_y": 4,
    }
    assert dispatch_matches(observed_ctrl, control) is True
    assert dispatch_matches(observed_vec, candidate) is True
    parsed = parse_attn_dispatch(
        "decode_attention_dispatch path=vec128_online "
        "launch=vec128_online_decode_attention prep_grid=24 prep_block=32 "
        "n_parts=8 prep_launches=0 prepared_q=false vec_kv=false gqa_block_y=4 "
        "attention_layers=16"
    )
    assert parsed["gqa_block_y"] == 4
    assert parsed["n_parts"] == 8


def test_eligibility_from_opt090() -> None:
    assert OPT090_FIXTURE.is_file()
    result = attention_vec_eligible(load_json(OPT090_FIXTURE))
    assert result["eligible"] is True
    assert result["verdict"] == "proceed"
    assert float(result["attention_ms_per_token"]) >= MIN_SAVING_MS
    assert result["llama_dispatch"] == "flash_attn_ext_vec<256,1>"


def test_no_go_verdicts_keep_warp_query() -> None:
    decided = decide_no_go_verdicts()
    assert decided["status"] == "no_go_no_measured_sink"
    assert decided["production_kept"] is True
    assert decided["selected_path"] == CONTROL_ID
    assert decided["claims_throughput"] is False


def test_replay_command_includes_vec128_flag() -> None:
    plan = family_plan("attention128", "feedback")
    command = replay_command(plan, configs(4)[1], None)
    assert "--decode-attention-vec128" in command
    assert "vec128_online" in command
    assert "--vec128-n-parts" in command
    assert "4" in command
    assert "--decode-position" in command
    assert "128" in command


def test_future_keep_policy() -> None:
    load_contract("OPT-103")
    validate_future_keep_policy("OPT-103", load_json(ITERATION))
