"""Host tests for OPT-095 grouped-KV decode attention admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt095_attention_gqa import (
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    OPT090_FIXTURE,
    REPORT,
    TINY_POSITIONS,
    attention_gqa_eligible,
    decide_no_go_verdicts,
    dispatch_matches,
    family_plan,
    parse_attn_dispatch,
    replay_command,
    tiny_positions,
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
NATIVE = ROOT / "cuda/opt095_attention_gqa_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-095")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    path = ATTN_PATH.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    header = ATTN_H.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-095"
    assert contract["claims_throughput"] is True
    assert contract["n_parts"] == 16
    assert contract["gqa_group"] == 6
    assert len(contract["configurations"]) == 2
    assert iteration["task"] == "OPT-095"
    assert iteration["diagnostics_make_target"] == "cuda-opt095-diagnostics"
    assert "cuda-opt095-diagnostics" in makefile
    assert "qw38-cuda-opt095-attention-gqa-test" in makefile
    assert 'kSelectedDecodeAttentionGqaPath[] = "warp_query"' in path
    assert "warp_query_gqa6" in path
    assert "DecodeAttentionGqaPathScope" in path
    assert "warp_query_gqa6_decode_attention" in attn
    assert "blockDim.y" in attn or "(32,6)" in native or "kWarpThreads, group" in attn
    assert "shared_key" in attn or "shared_k" in attn.lower()
    assert "load_bf16x8" in attn
    assert "decode_kv_warp_query_gqa6_occupancy" in header
    assert "--decode-attention-gqa" in replay
    assert "warp_query_gqa6" in native
    assert "launch_attention_prepare_partitioned" in scheduler


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-095")
    eligibility = iteration["workloads"]["eligibility"]
    parity = iteration["workloads"]["parity"]
    attention128 = iteration["workloads"]["attention128"]
    attention2048 = iteration["workloads"]["attention2048"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(attention128, "feedback")) == 8
    assert loop_product(workload_for_mode(attention2048, "feedback")) == 8
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 832
    described = describe_plan("OPT-095", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "n_parts fixed at 16" in proof
    assert "no prep launch" in proof
    assert "opt-090 attention evidence gate" in proof


def test_two_configs_only() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["decode_attention_gqa"] == "warp_query"
    assert CONFIGS[1]["decode_attention_gqa"] == "warp_query_gqa6"
    assert all(int(row["expected_prep_launches"]) == 0 for row in CONFIGS)


def test_tiny_positions_and_parity_positions() -> None:
    assert set(tiny_positions()) == {0, 1, 31, 32}
    native = NATIVE.read_text(encoding="utf-8")
    for position in TINY_POSITIONS:
        assert str(position) in native
    for position in (127, 128, 2047, 2048):
        assert str(position) in native
    assert "seed, 89" in native or "89," in native


def test_dispatch_matches_gqa6() -> None:
    control = CONFIGS[0]
    candidate = CONFIGS[1]
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
    observed_gqa = {
        "path": "warp_query_gqa6",
        "launch": "warp_query_gqa6_decode_attention",
        "prep_grid": 4,
        "prep_block": 32,
        "n_parts": 16,
        "prep_launches": 0,
        "prepared_q": False,
        "vec_kv": False,
        "gqa_block_y": 6,
    }
    assert dispatch_matches(observed_ctrl, control) is True
    assert dispatch_matches(observed_gqa, candidate) is True
    parsed = parse_attn_dispatch(
        "decode_attention_dispatch path=warp_query_gqa6 "
        "launch=warp_query_gqa6_decode_attention prep_grid=4 prep_block=32 "
        "n_parts=16 prep_launches=0 prepared_q=false vec_kv=false gqa_block_y=6 "
        "attention_layers=16"
    )
    assert parsed["gqa_block_y"] == 6


def test_eligibility_from_opt090() -> None:
    assert OPT090_FIXTURE.is_file()
    result = attention_gqa_eligible(load_json(OPT090_FIXTURE))
    assert result["eligible"] is True
    assert result["verdict"] == "proceed"
    assert float(result["attention_ms_per_token"]) >= MIN_SAVING_MS


def test_no_go_verdicts_keep_warp_query() -> None:
    decided = decide_no_go_verdicts()
    assert decided["status"] == "no_go_no_measured_sink"
    assert decided["production_kept"] is True
    assert decided["selected_path"] == CONTROL_ID
    assert decided["claims_throughput"] is False


def test_replay_command_includes_gqa_flag() -> None:
    plan = family_plan("attention128", "feedback")
    command = replay_command(plan, CONFIGS[1], None)
    assert "--decode-attention-gqa" in command
    assert "warp_query_gqa6" in command
    assert "--decode-position" in command
    assert "128" in command


def test_future_keep_policy() -> None:
    load_contract("OPT-095")
    validate_future_keep_policy("OPT-095", load_json(ITERATION))
