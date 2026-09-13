"""Host tests for OPT-115 pipeline traffic audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt115_pipeline_traffic import (
    AA_WORKLOADS,
    CAPACITY_128K,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    LONG_PROBES,
    MEASUREMENT_GUARD,
    PAIR_COUNT,
    PEAK_BANDWIDTH_GB_S,
    PHASES,
    RANKED_TASKS,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    TRAFFIC_PHASES,
    WARMUPS,
    aa_verdict_for_workload,
    architecture_traffic_map,
    authenticate_post113,
    bandwidth_bounds,
    bytes_to_peak_ms,
    classify_timeline,
    combine_aa_verdicts,
    compulsory_decode_bytes,
    embedding_row_bytes,
    family_plan,
    gdn_state_bytes,
    kv_bytes,
    pair_order,
    rank_opportunities,
    shared_keep_protocol,
    source_comparison_matrix,
    validate_fixture,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt115_pipeline_traffic_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-115")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-115"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["peak_bandwidth_gb_s"] == PEAK_BANDWIDTH_GB_S
    assert contract["llama_revision"] == "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    assert iteration["diagnostics_make_target"] == "cuda-opt115-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt115-diagnostics" in makefile
    assert "qw38-cuda-opt115-pipeline-traffic-test" in makefile
    assert (
        "--workload request-trace|session-loop|long-context|traffic-timeline" in native
    )
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "populated_cache=true" in native
    assert "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=" in native
    assert "QW38_OPT115_NATIVE_COUNTS=" in native
    assert "QW38_OPT115_PIPELINE_TRAFFIC_RESULT=" in runner
    assert "QW38_OPT115_NATIVE_COUNTS=" in runner
    assert PHASES == (
        "preflight",
        "request-trace",
        "long-context",
        "traffic-timeline",
        "ranking",
        "report",
    )


def test_iteration_loop_products_and_plan() -> None:
    iteration = load_contract("OPT-115")
    preflight = iteration["workloads"]["preflight"]
    trace = iteration["workloads"]["request-trace"]
    long_ctx = iteration["workloads"]["long-context"]
    timeline = iteration["workloads"]["traffic-timeline"]
    ranking = iteration["workloads"]["ranking"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(trace, "feedback")) == 78
    assert loop_product(workload_for_mode(long_ctx, "feedback")) == 6
    assert loop_product(workload_for_mode(timeline, "feedback")) == 4
    assert loop_product(workload_for_mode(ranking, "feedback")) == 1
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-115", "feedback", iteration, "preflight")
    assert "phase=preflight" in described
    plan = family_plan("request-trace", "feedback")
    assert plan["warmups"] == 3
    assert plan["samples"] == 10
    assert plan["candidates"] == 2
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "post113" in proof
    assert "1792" in " ".join(iteration["proof_limit"]) or "peak" in proof
    validate_future_keep_policy("OPT-115", iteration)


def test_post113_authentication_and_keep_protocol() -> None:
    auth = authenticate_post113()
    assert auth["ok"] is True
    assert auth["llama_q4k_mmvq"] is True
    assert auth["opt111_base"] is True
    assert auth["hybrid_crossover"] is True
    assert auth["sequential_gdn"] is True
    assert auth["ffn_only_graphs"] is True
    assert auth["opt113_numbers_historical_only"] is True
    protocol = shared_keep_protocol()
    assert protocol["parent"] == "post113_selected"
    assert protocol["quality_against"] == "post113"
    assert protocol["screen_pairs"] == 5
    assert protocol["acceptance_pairs"] == 10
    assert protocol["decode_output_tokens"] == 256
    assert protocol["throughput_ratio_lower_bound"] == 1.0
    assert protocol["memory_only_is_not_throughput_keep"] is True
    for task in RANKED_TASKS + ("OPT-123",):
        assert task in protocol["frozen_for"]


def test_aa_repeatable_and_pair_order() -> None:
    arm_a = [50.0 + (index % 3) * 0.05 for index in range(10)]
    arm_b = [value * 1.004 for value in arm_a]
    result = aa_verdict_for_workload(arm_a, arm_b)
    assert result["status"] == "repeatable"
    assert abs(float(result["geometric_ratio"]) - 1.0) <= MEASUREMENT_GUARD
    assert combine_aa_verdicts({name: result for name in AA_WORKLOADS}) == "repeatable"
    assert [pair_order(index) for index in range(4)] == ["AB", "BA", "AB", "BA"]


def test_compulsory_bytes_counts_units_and_kv_scaling() -> None:
    d128 = compulsory_decode_bytes(prefix=128)
    d2048 = compulsory_decode_bytes(prefix=2048)
    d131040 = compulsory_decode_bytes(prefix=131040)
    assert d128["units"] == "bytes"
    assert set(d128["phases"]) == set(TRAFFIC_PHASES)
    assert d128["embedding_full_table_excluded"] is True
    assert d128["embedding_row_bytes"] == embedding_row_bytes()
    assert d128["logits_d2h_bytes"] == 248320 * 4
    gdn = gdn_state_bytes()
    assert gdn["total_bytes"] == 158_859_264
    assert kv_bytes(CAPACITY_128K) == 8_589_934_592
    assert d128["kv_read_bytes"] == kv_bytes(128)
    assert d2048["kv_read_bytes"] == kv_bytes(2048)
    assert d131040["kv_read_bytes"] > d2048["kv_read_bytes"]
    assert d128["phases"]["weights"] > d128["phases"]["kv"]
    assert d128["phases"]["weights"] > d128["phases"]["logits"]
    memory = _json(MEMORY)
    assert memory["owners"]["attention_kv_bytes"] == kv_bytes(CAPACITY_128K)
    assert memory["owners"]["gdn_state_bytes"] == gdn["total_bytes"]
    traffic = architecture_traffic_map()
    assert traffic["units"] == "bytes"
    for name in (*AA_WORKLOADS, *LONG_PROBES):
        assert name in traffic
        assert (
            traffic[name].get("phases") or traffic[name].get("bytes_per_request")
        ) is not None


def test_bandwidth_bounds_use_peak_not_sum() -> None:
    phases = {"weights": 17_500_000_000, "kv": 8_388_608, "logits": 993_280}
    bounds = bandwidth_bounds(phases, prefix=128)
    assert bounds["peak_bandwidth_gb_s"] == 1792.0
    assert bounds["independent_sum_not_a_bound"] is True
    assert bounds["claims_throughput"] is False
    assert bounds["critical_path_peak_ms"] == bounds["per_phase"]["weights"]["peak_ms"]
    assert bounds["independent_sum_peak_ms"] > bounds["critical_path_peak_ms"]
    assert abs(bytes_to_peak_ms(1_792_000_000) - 1.0) < 1e-9
    assert "1792" in " ".join(bounds["assumptions"])
    assert bounds["measured_sustainable_bandwidth_gb_s"] is None


def test_interval_coverage_uses_unions_not_overlap_sums() -> None:
    records = [
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
            "start_ms": 2.0,
            "end_ms": 6.0,
            "complete_work_ms": 4.0,
        },
        {
            "role": "d2h",
            "attribution_role": "member",
            "start_ms": 6.0,
            "end_ms": 6.5,
            "complete_work_ms": 0.5,
        },
        {
            "role": "cpu_unknown_gap",
            "attribution_role": "member",
            "start_ms": 0.0,
            "end_ms": 0.2,
            "complete_work_ms": 0.2,
        },
    ]
    split = classify_timeline(records, wall_ms=8.0, tokens=1)
    assert split["overlap_not_summed"] is True
    assert split["gpu_busy_union_ms"] == 6.0
    assert split["interval_coverage"] <= 1.0
    assert split["unclassified_ms"] >= 0.0
    assert split["sum_exceeds_wall"] is True


def test_source_comparison_and_ranking() -> None:
    matrix = source_comparison_matrix()
    assert matrix["pinned_llama_revision"] == "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    assert matrix["llama_inspection_is_not_pin"] is True
    assert "OPT-120_packed_kv" in matrix["mechanisms"]
    assert "OPT-121_weight_bytes" in matrix["mechanisms"]
    kv_mech = matrix["mechanisms"]["OPT-120_packed_kv"]
    assert kv_mech["ds4"]["present"] is True
    assert "fp8_kv_quantize_row" in kv_mech["ds4"]["found"]
    traffic = architecture_traffic_map()
    ranked = rank_opportunities(traffic=traffic, timeline={}, bounds={})
    names = [row["task"] for row in ranked["ranked"]]
    assert names == list(RANKED_TASKS) or set(names) == set(RANKED_TASKS)
    by_task = {row["task"]: row for row in ranked["ranked"]}
    assert by_task["OPT-121"]["go"] is True
    assert by_task["OPT-120"]["peak_ms_upper"] > by_task["OPT-118"]["peak_ms_upper"]
    assert by_task["OPT-121"]["peak_ms_upper"] >= by_task["OPT-122"]["peak_ms_upper"]
    assert ranked["claims_throughput"] is False


def test_reproducibility_hooks_and_required_keys() -> None:
    contract = _json(CONTRACT)
    assert contract["token_generator"] == "(42 + index * 997) % 248320"
    assert contract["gguf_sha256"].startswith("31629f53")
    assert REQUIRED_FIXTURE_KEYS[-1] == "report_path"
    assert contract["long_context_capacity"] == CAPACITY_128K
    assert "d131040" in contract["long_context_probes"]
    q4 = (ROOT / "cuda/q4k_decode_path.cuh").read_text(encoding="utf-8")
    graphs = (ROOT / "cuda/execution_graph_path.cuh").read_text(encoding="utf-8")
    assert 'kSelectedQ4DecodePath[] = "llama_q4k_mmvq"' in q4
    assert 'kSelectedExecutionGraphPath[] = "ffn_only"' in graphs


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file() or not REPORT.is_file():
        return
    fixture = _json(FIXTURE)
    if "schema_version" not in fixture:
        return
    validate_fixture(fixture)
    assert fixture["claims_throughput"] is False
    assert fixture["claims_performance_improvement"] is False
