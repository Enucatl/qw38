"""Host tests for OPT-139 NCU counter identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt138_remaining_gap_profile import _experiment_entry
from tools.opt139_counter_identity import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    LAUNCH_VEC128_ONLINE,
    LAUNCH_WARP_QUERY,
    OPT140_INELIGIBLE,
    PHASES,
    PREFIXES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTOR,
    admit_supported_mechanism,
    counter_record_from_ncu_blob,
    decode_attention_vec128_path_for_position,
    family_plan,
    identity_match,
    kernel_stem,
    parse_ncu_output,
    parse_numeric,
    production_attn_identity,
    select_largest_duration_kernel,
    select_target_launch,
    typed_slots_from_launch,
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
PATH_SRC = ROOT / "cuda/attention_decode_path.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"
OPT138_TOOL = ROOT / "tools/opt138_remaining_gap_profile.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-139")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    path_src = PATH_SRC.read_text(encoding="utf-8")
    opt138 = OPT138_TOOL.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt139_counter_identity.py").read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-139"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["prefixes"] == list(PREFIXES)
    assert contract["prefill_decode_attention_eligible"] is False
    assert contract["d128_expected_kernel"] == LAUNCH_WARP_QUERY
    assert contract["d2048_expected_kernel"] == LAUNCH_VEC128_ONLINE
    assert iteration["diagnostics_make_target"] == "cuda-opt139-diagnostics"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt139-diagnostics" in makefile
    assert "QW38_OPT139_COUNTER_IDENTITY_RESULT=" in runner
    assert "decode_attention_vec128_path_for_position" in path_src
    assert "opt139_counter_identity" in opt138
    assert "--csv" in opt138
    assert "throughput_alone_cannot_establish_mechanism" in tool
    assert PHASES == ("preflight", "identity", "counters", "report")
    validate_future_keep_policy("OPT-139", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-139")
    preflight = iteration["workloads"]["preflight"]
    identity = iteration["workloads"]["identity"]
    counters = iteration["workloads"]["counters"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(identity, "acceptance")) == 2
    assert loop_product(workload_for_mode(counters, "acceptance")) == 2
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-139", "acceptance", iteration, "counters")
    assert "phase=counters" in described
    assert "loop_product=1" in family_plan("acceptance", "preflight")


def _quoted_csv() -> str:
    return (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","void warp_query_decode_attention<false, false>(float *, int)",'
        '"dram__bytes_op_read.sum","byte","1,234"\n'
        '"0","void warp_query_decode_attention<false, false>(float *, int)",'
        '"dram__bytes_op_write.sum","byte","56"\n'
        '"0","void warp_query_decode_attention<false, false>(float *, int)",'
        '"dram__throughput.avg.pct_of_peak_sustained_elapsed","%","12.5"\n'
        '"0","void warp_query_decode_attention<false, false>(float *, int)",'
        '"sm__pipe_tensor_cycles_active.sum","cycle","0"\n'
        '"0","void warp_query_decode_attention<false, false>(float *, int)",'
        '"sm__warps_active.avg.pct_of_peak_sustained_active","%","37.0"\n'
        '"0","void warp_query_decode_attention<false, false>(float *, int)",'
        '"smsp__warps_issue_stalled_long_scoreboard.avg","%","41.2"\n'
        '"1","merge_decode_kv_parts","dram__bytes_op_read.sum","byte","9"\n'
        '"1","merge_decode_kv_parts","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","90.0"\n'
    )


def test_parse_quoted_csv_multiple_launches_and_units() -> None:
    parsed = parse_ncu_output(_quoted_csv())
    assert parsed.format == "csv"
    assert len(parsed.launches) == 2
    target = select_target_launch(
        parsed.launches,
        expected_kernel=LAUNCH_WARP_QUERY,
        family="attn_core",
    )
    assert target is not None
    assert kernel_stem(target.kernel_name) == LAUNCH_WARP_QUERY
    slots = typed_slots_from_launch(target)
    assert slots["dram_read_bytes"] == 1234.0
    assert slots["units"]["dram_read_bytes"] == "byte"
    assert slots["dram_throughput"] == 12.5
    assert slots["units"]["dram_throughput"] == "%"
    assert slots["tensor_activity"] == 0.0
    assert "registers" in slots["missing_reasons"]
    other = [item for item in parsed.launches if item is not target]
    assert len(other) == 1
    assert kernel_stem(other[0].kernel_name) == "merge_decode_kv_parts"
    assert parsed.enclosing_replay["separated_from_target_kernel"] is True


def test_missing_metrics_stay_null_real_zeros_stay_zero() -> None:
    csv = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","warp_query_decode_attention","sm__pipe_tensor_cycles_active.sum","cycle","0"\n'
        '"0","warp_query_decode_attention","smsp__warps_issue_stalled_barrier.avg","%","0"\n'
    )
    parsed = parse_ncu_output(csv)
    slots = typed_slots_from_launch(parsed.launches[0])
    assert slots["tensor_activity"] == 0.0
    assert slots["dram_read_bytes"] is None
    assert slots["missing_reasons"]["dram_read_bytes"] == "metric_absent_from_capture"
    assert slots["zero_filled"] is False
    assert slots["stalls"]["barrier"]["value"] == 0.0
    assert parse_numeric("0") == 0.0
    assert parse_numeric("") is None


def test_table_format_populates_typed_fields() -> None:
    table = (
        '==PROF== Profiling "warp_query_decode_attention" - 0: Application replay pass 1\n'
        "    dram__bytes_op_read.sum                                 byte         4096\n"
        "    dram__bytes_op_write.sum                                byte            0\n"
        "    lts__t_sectors.sum                                      sector        2937\n"
        "    sm__pipe_tensor_cycles_active.sum                       cycle            0\n"
        '==PROF== Profiling "merge_decode_kv_parts" - 1: Application replay pass 1\n'
        "    dram__bytes_op_read.sum                                 byte           12\n"
    )
    record = counter_record_from_ncu_blob(
        table,
        kernel_id="quartz:attn_core",
        engine="quartz",
        family="attn_core",
        phase="decode",
        replay_family="decode-attention",
        expected_kernel=LAUNCH_WARP_QUERY,
        workload="D128",
        prefix=128,
    )
    assert record["dram_read_bytes"] == 4096.0
    assert record["dram_write_bytes"] == 0.0
    assert record["l2_traffic"] == 2937.0
    assert record["tensor_activity"] == 0.0
    assert record["target_kernel"] == LAUNCH_WARP_QUERY
    assert record["enclosing_replay"]["separated_from_target_kernel"] is True
    assert len(record["other_launches"]) == 1
    assert record["other_launches"][0]["stem"] == "merge_decode_kv_parts"
    assert record["metrics"]["dram__bytes_op_read.sum"]["value"] == 4096.0


def test_phase_cross_contamination_is_rejected() -> None:
    admission = admit_supported_mechanism(
        {
            "phase": "prefill",
            "engine": "quartz",
            "workload": "P4096",
            "replay_family": "decode-attention",
            "kernel": LAUNCH_WARP_QUERY,
            "expected_kernel": LAUNCH_WARP_QUERY,
            "dram_throughput": 80.0,
            "dram_read_bytes": 1000,
        }
    )
    assert admission["ok"] is False
    assert admission["supported_mechanism"] is None
    assert admission["candidate"] is None
    assert admission["reason"] == OPT140_INELIGIBLE


def test_selector_mismatch_rejects_warp_query_as_d2048() -> None:
    expected = production_attn_identity(2048)
    assert expected["path"] == "vec128_online"
    assert expected["expected_kernel"] == LAUNCH_VEC128_ONLINE
    mismatch = identity_match(
        phase="decode",
        engine="quartz",
        workload="D2048",
        kernel=LAUNCH_WARP_QUERY,
        expected_kernel=expected["expected_kernel"],
        replay_family="decode-attention",
        replay_boundary=None,
    )
    assert mismatch["ok"] is False
    assert "kernel_selector_mismatch" in mismatch["mismatches"]
    matched = identity_match(
        phase="decode",
        engine="quartz",
        workload="D128",
        kernel=LAUNCH_WARP_QUERY,
        expected_kernel=production_attn_identity(128)["expected_kernel"],
        replay_family="decode-attention",
        replay_boundary=None,
    )
    assert matched["ok"] is True
    assert decode_attention_vec128_path_for_position(128) == "warp_query"
    assert decode_attention_vec128_path_for_position(2048) == "vec128_online"
    mangled = (
        "void qw38::cuda::<unnamed>::warp_query_decode_attention<(bool)0, "
        "(bool)0>(qw38::cuda::AttentionConfig, unsigned long)"
    )
    assert kernel_stem(mangled) == LAUNCH_WARP_QUERY


def test_throughput_only_cannot_establish_mechanism() -> None:
    admission = admit_supported_mechanism(
        {
            "phase": "decode",
            "engine": "quartz",
            "workload": "D128",
            "replay_family": "decode-attention",
            "kernel": LAUNCH_WARP_QUERY,
            "expected_kernel": LAUNCH_WARP_QUERY,
            "dram_throughput": 55.0,
            "sm_throughput": 40.0,
        }
    )
    assert admission["ok"] is False
    assert admission["supported_mechanism"] is None
    assert admission["candidate"] is None
    assert admission["reason"] == "throughput_alone_cannot_establish_mechanism"


def test_opt138_experiment_entry_rejects_throughput_only() -> None:
    entry = _experiment_entry(
        "decode",
        {
            "selected": [
                {
                    "family": "attn_core",
                    "score_ms": 47.5,
                    "stats": {"mean_ms": 47.5, "one_sided_low": 47.3, "n": 3},
                }
            ]
        },
        {
            "kernels": [
                {
                    "family": "attn_core",
                    "engine": "quartz",
                    "phase": "decode",
                    "replay_family": "decode-attention",
                    "kernel": LAUNCH_WARP_QUERY,
                    "expected_kernel": LAUNCH_WARP_QUERY,
                    "dram_throughput": 62.0,
                    "sm_throughput": 18.0,
                    "error": None,
                }
            ]
        },
        {"rounds": [{"family": "attn_core", "phase": "decode", "ok": True}]},
    )
    assert entry["supported_mechanism"] is None
    assert entry["candidate"] is None


def test_largest_duration_kernel_stays_inside_family() -> None:
    kernels = [
        {"name": "stage_chunk_rows", "duration_ns": 9_000_000},
        {"name": "warp_query_decode_attention", "duration_ns": 4_000_000},
        {"name": "vec128_online_decode_attention", "duration_ns": 8_000_000},
        {"name": "merge_decode_kv_parts", "duration_ns": 500_000},
    ]
    largest = select_largest_duration_kernel(kernels, "attn_core")
    assert largest is not None
    assert largest["stem"] == LAUNCH_VEC128_ONLINE
    assert largest["duration_ns"] == 8_000_000


def test_enclosing_replay_stays_separate_from_kernel_counters() -> None:
    text = (
        "round family=decode-attention cache_mode=rotating warmup=false "
        "sample_index=0 observation_unit=independent_round enclosing_ms=4.35 "
        "kernel_only_ms=2.93 gate_up_calls=0\n"
        "decode_attention_dispatch path=vec128_online "
        "launch=vec128_online_decode_attention prep_grid=24 prep_block=32 "
        "n_parts=16 prep_launches=0 prepared_q=false vec_kv=false "
        "gqa_block_y=4 attention_layers=16 capture_positions=2 "
        "eager_or_captured=true decode_position=2048 crossover_threshold=1024 "
        "path_for_position=vec128_online extra_kernel=false\n"
    )
    from tools.opt139_counter_identity import (
        parse_enclosing_replay,
        parse_replay_dispatch,
    )

    enclosing = parse_enclosing_replay(text)
    assert enclosing["separated_from_target_kernel"] is True
    assert enclosing["measured_enclosing_ms"] == [4.35]
    dispatch = parse_replay_dispatch(text)
    assert dispatch is not None
    assert dispatch["launch"] == LAUNCH_VEC128_ONLINE
    assert dispatch["decode_position"] == 2048


def test_fixture_keys() -> None:
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert REPORT.parent.as_posix().endswith("opt139-counter-identity")
