"""Host tests for OPT-114 sitting calibration and launch-overhead measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt106_batch_gate import EXPECTED_PATHS
from tools.opt114_sitting_launch import (
    AA_WORKLOADS,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    GRAPH_REOPEN_MS,
    ITERATION,
    MEASUREMENT_GUARD,
    PAIR_COUNT,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    WARMUPS,
    aa_verdict_for_workload,
    combine_aa_verdicts,
    evaluate_graph_opportunity,
    family_plan,
    historical_opt098_calibration,
    pair_order,
    reconcile_memory,
    split_timeline,
    validate_fixture,
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
NATIVE = ROOT / "cuda/opt114_sitting_launch_test.cu"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
OPT106 = ROOT / "fixtures/opt106_batch_gate.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-114")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-114"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["uses_opt098_arrays_for_keep_reject"] is False
    assert contract["other_idle_is_legacy_only"] is True
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["graph_reopen_ms"] == GRAPH_REOPEN_MS
    assert contract["measurement_guard"] == MEASUREMENT_GUARD
    assert iteration["diagnostics_make_target"] == "cuda-opt114-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt114-diagnostics" in makefile
    assert "qw38-cuda-opt114-sitting-launch-test" in makefile
    assert "--workload aa-control|launch-overhead|graph-ab|graph-bench" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "legacy_other_idle_ms" in native
    assert "decode_segments8" in native
    assert "QW38_OPT114_SITTING_LAUNCH_RESULT=" in native
    q4 = (ROOT / "cuda/q4k_decode_path.cuh").read_text(encoding="utf-8")
    graphs = (ROOT / "cuda/execution_graph_path.cuh").read_text(encoding="utf-8")
    assert 'kSelectedQ4DecodePath[] = "integer_q8_late"' in q4
    assert 'kSelectedExecutionGraphPath[] = "ffn_only"' in graphs


def test_iteration_loop_products_and_plan() -> None:
    iteration = load_contract("OPT-114")
    preflight = iteration["workloads"]["preflight"]
    aa_control = iteration["workloads"]["aa-control"]
    launch = iteration["workloads"]["launch-overhead"]
    graph = iteration["workloads"]["graph-eligibility"]
    memory = iteration["workloads"]["memory-reconcile"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(aa_control, "feedback")) == 78
    assert loop_product(workload_for_mode(launch, "feedback")) == 4
    assert loop_product(workload_for_mode(graph, "feedback")) == 26
    assert loop_product(workload_for_mode(memory, "feedback")) == 1
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-114", "feedback", iteration, "preflight")
    assert "phase=preflight" in described
    plan = family_plan("aa-control", "feedback")
    assert plan["warmups"] == 3
    assert plan["samples"] == 10
    assert plan["candidates"] == 2
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt-098" in proof
    assert "other_idle" in proof
    validate_future_keep_policy("OPT-114", iteration)


def test_aa_repeatable_within_two_percent() -> None:
    arm_a = [50.0 + (index % 3) * 0.05 for index in range(10)]
    arm_b = [value * 1.004 for value in arm_a]
    result = aa_verdict_for_workload(arm_a, arm_b)
    assert result["status"] == "repeatable"
    assert result["equality_excluded"] is False
    assert abs(float(result["geometric_ratio"]) - 1.0) <= MEASUREMENT_GUARD
    assert combine_aa_verdicts({name: result for name in AA_WORKLOADS}) == "repeatable"


def test_aa_unstable_when_ci_excludes_equality() -> None:
    arm_a = [50.0] * 10
    arm_b = [60.0] * 10
    result = aa_verdict_for_workload(arm_a, arm_b)
    assert result["status"] == "measurement_unstable"
    assert result["equality_excluded"] is True or result["point_outside_guard"] is True


def test_pair_order_interleaves_ab_ba() -> None:
    assert [pair_order(index) for index in range(4)] == ["AB", "BA", "AB", "BA"]


def test_launch_budget_is_leaf_gaps_not_other_idle() -> None:
    records = [
        {
            "role": "ffn_mmv",
            "attribution_role": "enclosing",
            "start_ms": 0.0,
            "end_ms": 10.0,
            "complete_work_ms": 10.0,
        },
        {
            "role": "ffn_down",
            "attribution_role": "member",
            "start_ms": 0.0,
            "end_ms": 4.0,
            "complete_work_ms": 4.0,
        },
        {
            "role": "ffn_gate",
            "attribution_role": "member",
            "start_ms": 4.8,
            "end_ms": 8.8,
            "complete_work_ms": 4.0,
        },
        {
            "role": "cpu_unknown_gap",
            "attribution_role": "member",
            "start_ms": 0.0,
            "end_ms": 0.05,
            "complete_work_ms": 0.05,
        },
    ]
    split = split_timeline(records, tokens=1, legacy_other_idle_ms=0.05)
    assert split["legacy_other_idle_is_launch_budget"] is False
    assert split["legacy_other_idle_ms"] == 0.05
    assert abs(split["gpu_idle_gap_ms"] - 0.8) < 1e-9
    assert split["removable_launch_ms_per_token"] >= 0.8
    assert split["kernel_interval_ms"] == 8.0
    gate = evaluate_graph_opportunity(
        {
            "d128": {**split, "valid": True},
            "d2048": {
                **split,
                "valid": True,
                "removable_launch_ms_per_token": 0.85,
            },
        }
    )
    assert gate["eligible"] is True
    assert gate["legacy_other_idle"]["used_as_launch_budget"] is False


def test_no_graph_opportunity_below_trigger() -> None:
    low = {
        "valid": True,
        "removable_launch_ms_per_token": 0.12,
        "legacy_other_idle_ms": 0.07,
    }
    gate = evaluate_graph_opportunity({"d128": low, "d2048": low})
    assert gate["eligible"] is False
    assert gate["verdict"] == "no_graph_opportunity"
    assert float(low["removable_launch_ms_per_token"]) < GRAPH_REOPEN_MS


def test_memory_reconcile_explains_opt106_arithmetic() -> None:
    inventory = _json(MEMORY)
    opt106 = _json(OPT106)
    result = reconcile_memory(inventory, opt106)
    assert result["reserve_cap_increased"] is False
    assert result["opt106_memory_fit_ok"] is False
    assert result["opt106_reserve_ok"] is True
    assert result["explicit_delta_bytes"] != 0
    assert "1.5 GiB" in result["explanation"]
    assert result["required_reserve_bytes"] == 1_610_612_736


def test_historical_opt098_is_calibration_only() -> None:
    historical = historical_opt098_calibration()
    assert historical["uses_for_keep_reject"] is False
    assert historical["opt098_d128_tok_s"] is not None
    assert historical["opt106_d128_tok_s"] is not None
    assert abs(float(historical["opt106_d128_tok_s"]) - 53.50) < 0.02
    fixture = _json(FIXTURE)
    assert fixture["uses_opt098_arrays_for_keep_reject"] is False
    for name in AA_WORKLOADS:
        block = fixture["aa_control"][name]
        assert block.get("uses_opt098_arrays") is False
        assert historical["opt098_d128_tok_s"] not in (block.get("a_tok_s") or [])
    validate_fixture(fixture)


def test_frozen_selectors_match_opt106() -> None:
    from tools.opt114_sitting_launch import source_paths

    paths = source_paths()
    for key, value in EXPECTED_PATHS.items():
        assert paths[key] == value
    contract = load_json(CONTRACT)
    assert contract["execution_graphs"] == "ffn_only"
    assert REQUIRED_FIXTURE_KEYS[-1] == "report_path"
