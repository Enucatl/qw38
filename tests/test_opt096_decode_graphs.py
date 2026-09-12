"""Host tests for OPT-096 conditional eight-layer decode graph admission."""

from __future__ import annotations

import json
from pathlib import Path

from tools.opt096_decode_graphs import (
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    GRAPH_REOPEN_MS,
    ITERATION,
    MIN_SAVING_MS,
    OPT090_FIXTURE,
    REPORT,
    evaluate_eligibility,
    decide_no_reopen_verdicts,
    dispatch_matches,
    family_plan,
    parse_graph_dispatch,
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
SCHEDULER_H = ROOT / "cuda/full_scheduler.h"
EXEC_PATH = ROOT / "cuda/execution_graph_path.cuh"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
NATIVE = ROOT / "cuda/opt096_decode_graphs_test.cu"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-096")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    exec_path = EXEC_PATH.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-096"
    assert contract["segment_count"] == 8
    assert contract["segment_layer_count"] == 8
    assert iteration["diagnostics_make_target"] == "cuda-opt096-diagnostics"
    assert "cuda-opt096-diagnostics" in makefile
    assert "qw38-cuda-opt096-decode-graphs-test" in makefile
    assert "decode_segments8" in exec_path
    assert "ffn_only" in exec_path
    assert "ExecutionGraphPathScope" in exec_path
    assert "decode_segment_graph_count" in scheduler
    assert "execution_graph_uses_decode_segments8" in scheduler
    assert "--execution-graphs" in native
    assert "decode_segments8" in native


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-096")
    eligibility = iteration["workloads"]["eligibility"]
    same_math = iteration["workloads"]["same-math"]
    decode = iteration["workloads"]["decode"]
    body128 = iteration["workloads"]["body128"]
    d128 = iteration["workloads"]["d128"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(same_math, "feedback")) == 2
    assert loop_product(workload_for_mode(decode, "feedback")) == 256
    assert loop_product(workload_for_mode(body128, "acceptance")) == 26
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    described = describe_plan("OPT-096", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "exactly two selectors" in proof
    assert "poll at every eight-layer boundary" in proof


def test_two_configs_only() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["expected_decode_graph_count"] == 64
    assert CONFIGS[1]["expected_decode_segment_graph_count"] == 8


def test_dispatch_matches_paths() -> None:
    control = CONFIGS[0]
    candidate = CONFIGS[1]
    assert dispatch_matches(
        {
            "path": "ffn_only",
            "decode_graph_count": 64,
            "decode_segment_graph_count": 0,
        },
        control,
    )
    assert dispatch_matches(
        {
            "path": "decode_segments8",
            "decode_graph_count": 0,
            "decode_segment_graph_count": 8,
        },
        candidate,
    )
    parsed = parse_graph_dispatch(
        "decode_graph_dispatch path=decode_segments8 decode_graph_count=0 "
        "decode_segment_graph_count=8 launch_param_updates=3"
    )
    assert parsed["launch_param_updates"] == 3


def test_eligibility_from_opt090_graph_reopen() -> None:
    assert OPT090_FIXTURE.is_file()
    opt090 = load_json(OPT090_FIXTURE)
    result = evaluate_eligibility(opt090)
    assert opt090["graph_reopen_eligible"] is True
    assert result["graph_reopen_eligible"] is True
    assert float(result["overhead"]["d128_idle_ms_per_token"]) >= GRAPH_REOPEN_MS
    assert float(result["overhead"]["d2048_idle_ms_per_token"]) >= GRAPH_REOPEN_MS
    assert result["eligible"] is True
    assert result["verdict"] == "proceed"


def test_no_reopen_when_overhead_below_trigger() -> None:
    result = evaluate_eligibility(
        {"graph_reopen_eligible": True},
        {
            "d128_idle_ms_per_token": 0.08555,
            "d2048_idle_ms_per_token": 0.10514,
            "d128_valid": True,
            "d2048_valid": True,
            "source": "synthetic",
        },
    )
    assert result["eligible"] is False
    assert result["verdict"] == "no_reopen_overhead_below_trigger"


def test_no_reopen_verdicts_keep_ffn_only() -> None:
    decided = decide_no_reopen_verdicts("no_reopen_overhead_below_trigger")
    assert decided["status"] == "no_reopen_overhead_below_trigger"
    assert decided["production_kept"] is True
    assert decided["selected_path"] == CONTROL_ID
    assert decided["claims_throughput"] is False


def test_family_plan_decode_screen() -> None:
    plan = family_plan("decode", "feedback")
    assert plan["prefix"] == 2048
    assert plan["tokens"] == 32
    assert plan["samples"] == 3
    assert plan["warmups"] == 1


def test_future_keep_policy() -> None:
    load_contract("OPT-096")
    validate_future_keep_policy("OPT-096", load_json(ITERATION))


def test_min_saving_threshold() -> None:
    assert MIN_SAVING_MS == 0.10
