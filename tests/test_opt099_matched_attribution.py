"""Host tests for OPT-099 matched Quartz/llama attribution. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt071_attribution_repair import (
    DECODE_ATTENTION_CORES,
    DECODE_DOWNS,
    DECODE_GATE_UP_PAIRS,
    DECODE_GDN_CORES,
    UNEXPLAINED_LIMIT,
    accounting_split,
    rank_remaining_gaps,
)
from tools.opt090_decode_attribution import one_token_valid_records
from tools.opt099_matched_attribution import (
    CONTRACT,
    COUNTER_LAUNCHES,
    DECODE_SAMPLES,
    DECODE_WARMUPS,
    EXPECTED_DECODE_COUNTS,
    FIXTURE,
    ITERATION,
    REPORT,
    REQUESTED_POST098,
    TRIGGER_IDS,
    VERDICT_KEYS,
    canonical_family_opt099,
    chargeable_records_opt099,
    compiled_selectors,
    expected_decode_counts_opt099,
    family_plan,
    family_times,
    freeze_post098_selected,
    llama_graph_create_excluded_ms,
    llama_log_path,
    llama_missing_filled_with_zero,
    parse_layer_name,
    parse_llama_dumps,
    parse_llama_sample_walls,
    run,
    trigger_fields,
    validate_call_counts_opt099,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PATH = ROOT / "cuda/ffn_decode_path.cuh"
Q8_PATH = ROOT / "cuda/q8_decode_path.cuh"
Q6_PATH = ROOT / "cuda/q6k_decode_path.cuh"
NATIVE = ROOT / "cuda/opt060_engine_attribution_test.cu"
OVERLAY = ROOT / "tools/llama_authority/patches/qw38_opt060_attribution.cuh"
ADAPTER = ROOT / "tools/llama_authority/engine_attribution.cpp"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-099")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    overlay = OVERLAY.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-099"
    assert contract["claims_throughput"] is False
    assert contract["decode_warmups"] == 3
    assert contract["decode_samples"] == 3
    assert contract["requested_post098_selected"]["q4_decode"] == "integer_q8_late"
    assert contract["requested_post098_selected"]["q4_warps_per_row"] == 4
    assert contract["requested_post098_selected"]["prompt_mmq"] == (
        "i128_j128_fma_async_x"
    )
    assert contract["requested_post098_selected"]["prompt_mmq_wait"] == "joined_wait"
    assert iteration["task"] == "OPT-099"
    assert iteration["claims_throughput"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt099-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["case_ids"] == [
        "freeze",
        "conservation",
        "d128",
        "d2048",
        "counters",
        "p4096",
        "report",
    ]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt099-diagnostics" in makefile
    assert "QW38_OPT099_ATTRIBUTION_RESULT=" in native
    assert "post098_selected" in native
    assert "--evidence-dir" in adapter
    assert "weight_name" in overlay
    assert "ffn_out" in overlay
    assert "q8_input_projections" in overlay
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    q8 = Q8_PATH.read_text(encoding="utf-8")
    q6 = Q6_PATH.read_text(encoding="utf-8")
    assert 'kSelectedQ4DecodePath[] = "integer_q8_late"' in q4
    assert "kSelectedQ4DecodeWarpsPerRow = 4" in q4
    assert 'kSelectedFfnDecodePath[] = "paired_integer"' in ffn
    assert 'kSelectedQ8DecodeGrouping[] = "grouped_r1_w4"' in q8
    assert 'kSelectedQ6DecodePath[] = "integer_q8_1"' in q6
    assert "kSelectedQ6DecodeWarpsPerRow = 2" in q6


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-099")
    freeze = iteration["workloads"]["freeze"]
    conservation = iteration["workloads"]["conservation"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    counters = iteration["workloads"]["counters"]
    p4096 = iteration["workloads"]["p4096"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(freeze, "feedback")) == 1
    assert loop_product(workload_for_mode(conservation, "feedback")) == 1
    assert loop_product(workload_for_mode(d128, "feedback")) == 24
    assert loop_product(workload_for_mode(d2048, "feedback")) == 24
    assert loop_product(workload_for_mode(counters, "feedback")) == 5
    assert loop_product(workload_for_mode(p4096, "feedback")) == 24
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    assert d128["warmups"] == DECODE_WARMUPS == 3
    assert d128["samples"] == DECODE_SAMPLES == 3
    assert d128["output_tokens"] == 32
    described = describe_plan("OPT-099", "feedback", iteration, "freeze")
    assert "phase=freeze" in described
    assert "historical_oracles=none" in described
    assert family_plan("d128", "feedback")["execution_modes"] == 2


def test_future_keep_policy_and_instrumentation_only() -> None:
    iteration = load_contract("OPT-099")
    validate_future_keep_policy("OPT-099", iteration)
    assert list(VERDICT_KEYS) == [
        "kernel_parity_pass",
        "model_quality_pass",
        "performance_pass",
        "production_kept",
    ]
    assert list(TRIGGER_IDS) == [
        "opt100_q8_aligned",
        "opt101_gdn_transpose",
        "opt102_q4_repack",
        "opt103_attention_vec",
        "opt104_q6_aligned",
        "opt105_mmq_x2",
    ]


def test_freeze_matches_compiled_selectors() -> None:
    freeze = freeze_post098_selected()
    observed = compiled_selectors()
    assert freeze["ok"] is True
    assert freeze["mismatches"] == {}
    for key, value in REQUESTED_POST098.items():
        assert observed[key] == value
    assert observed["q4_warps_per_row"] == 4
    assert observed["q6_warps_per_row"] == 2
    assert observed["prompt_mmq_wait"] == "joined_wait"


def test_ffn_down_is_first_class_not_unmapped() -> None:
    records = one_token_valid_records()
    enclosing = {
        "engine": "quartz",
        "layer": 0,
        "phase": "decode",
        "token_position": 128,
        "role": "ffn_mmv",
        "tensor_name": "blk.0.ffn_mmv",
        "launch_family": "mmv",
        "attribution_role": "enclosing",
        "fused_member_ids": "",
        "complete_work_ms": 0.16,
        "start_ms": 0.0,
        "end_ms": 0.16,
        "stream_index": 0,
    }
    combined = [enclosing, *records]
    charged = chargeable_records_opt099(combined)
    roles = {str(row.get("role")) for row in charged}
    assert "ffn_down" in roles
    assert "ffn_mmv" not in roles
    times = family_times(combined)
    assert times["ffn_down"] > 0.0
    assert "unmapped_mul_mat" not in times
    checked = validate_call_counts_opt099(combined, tokens=1)
    assert checked["ok"] is True
    assert checked["observed"]["ffn_down"] == DECODE_DOWNS == 64
    assert expected_decode_counts_opt099(1)["ffn_gate_up_glu"] == DECODE_GATE_UP_PAIRS
    assert DECODE_GATE_UP_PAIRS == 64
    assert EXPECTED_DECODE_COUNTS["ffn_gate_up_glu"] != 128


def test_mixer_parent_is_not_unmapped_mul_mat() -> None:
    records = [
        {
            "engine": "quartz",
            "layer": 0,
            "phase": "decode",
            "token_position": 128,
            "role": "mixer_mmv",
            "tensor_name": "blk.0.mixer_mmv",
            "launch_family": "mmv",
            "attribution_role": "enclosing",
            "complete_work_ms": 0.08,
            "start_ms": 0.0,
            "end_ms": 0.08,
        },
        {
            "engine": "quartz",
            "layer": 0,
            "phase": "decode",
            "token_position": 128,
            "role": "proj_packed_qkv",
            "tensor_name": "blk.0.attn_qkv.weight",
            "launch_family": "mmv",
            "attribution_role": "member",
            "complete_work_ms": 0.05,
            "start_ms": 0.01,
            "end_ms": 0.06,
        },
    ]
    times = family_times(records)
    assert times["q8_input_projections"] == 0.05
    assert "unmapped_mul_mat" not in times
    assert canonical_family_opt099(records[1]) == "q8_input_projections"


def test_llama_mapping_distinguishes_ffn_down_and_layers() -> None:
    records = [
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "ffn_gate_up_glu",
            "tensor_name": "ffn_gate-0",
            "launch_family": "mmvq_fused_glu",
            "attribution_role": "enclosing",
            "fused_member_ids": "ffn_gate,ffn_up,ffn_glu",
            "complete_work_ms": 0.07,
            "start_ms": 0.0,
            "end_ms": 0.07,
        },
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "ffn_down",
            "tensor_name": "ffn_out-0",
            "weight_name": "blk.0.ffn_down.weight",
            "launch_family": "mmvq",
            "attribution_role": "member",
            "complete_work_ms": 0.04,
            "start_ms": 0.07,
            "end_ms": 0.11,
        },
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "unknown",
            "tensor_name": "Qcur_full-3",
            "launch_family": "mmvq",
            "attribution_role": "member",
            "complete_work_ms": 0.04,
            "start_ms": 0.2,
            "end_ms": 0.24,
        },
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "q6_attention_output",
            "tensor_name": "attn_output-3",
            "launch_family": "kernel",
            "attribution_role": "member",
            "complete_work_ms": 0.02,
            "start_ms": 0.3,
            "end_ms": 0.32,
        },
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "gdn_conv",
            "tensor_name": "conv_output_raw-0",
            "launch_family": "kernel",
            "attribution_role": "member",
            "complete_work_ms": 0.01,
            "start_ms": 0.02,
            "end_ms": 0.03,
        },
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "attention_core",
            "op": "FLASH_ATTN_EXT",
            "tensor_name": "node_216",
            "weight_name": "Qcur-3 (view) (permuted)",
            "launch_family": "kernel",
            "attribution_role": "member",
            "complete_work_ms": 0.03,
            "start_ms": 0.25,
            "end_ms": 0.28,
        },
        {
            "engine": "llama",
            "layer": -1,
            "phase": "decode",
            "token_position": 2048,
            "role": "pointwise",
            "op": "RMS_NORM",
            "tensor_name": "norm-0",
            "weight_name": "attn_output-0",
            "launch_family": "kernel",
            "attribution_role": "member",
            "complete_work_ms": 0.005,
            "start_ms": 0.04,
            "end_ms": 0.045,
        },
    ]
    assert parse_layer_name("ffn_out-0") == 0
    assert parse_layer_name("attn_output-3") == 3
    times = family_times(records)
    assert times["ffn_down"] == 0.04
    assert times["ffn_gate_up_glu"] == 0.07
    assert times["q8_input_projections"] == 0.04
    assert times["q6_attention_output"] == 0.02
    assert times["gdn_conv"] == 0.01
    assert times["attention_core"] == 0.03
    assert times["pointwise"] == 0.005
    assert "q6_attention_output" in times
    assert times["q6_attention_output"] == 0.02
    assert llama_missing_filled_with_zero(times) is False


def test_missing_llama_is_null_not_zero() -> None:
    ranking = {
        "status": "ranked",
        "gap_attribution_complete": True,
        "ranked": [
            {
                "family": "q8_input_projections",
                "quartz_ms": 4.7,
                "llama_ms": None,
                "matched_llama_excess_ms": None,
                "llama_covered": False,
            }
        ],
    }
    triggers = trigger_fields(ranking, {})
    assert triggers["opt100_q8_aligned"]["go"] is False
    assert triggers["opt100_q8_aligned"]["reason"] == "llama_family_unmapped"
    assert triggers["opt100_q8_aligned"]["excess_ms"] is None


def test_wrong_counts_and_unexplained_wall_stop_ranking() -> None:
    records = one_token_valid_records()[:-1]
    checked = validate_call_counts_opt099(records, tokens=1)
    assert checked["ok"] is False
    incomplete = rank_remaining_gaps(
        [{"family": "ffn_gate_up_glu", "quartz_ms": 10.0, "llama_ms": 4.0}],
        {"unexplained_share": 0.06, "uncovered_wall_ms": 1.2},
        {"ok": True},
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["ranked"] == []
    assert UNEXPLAINED_LIMIT == 0.05
    assert DECODE_GDN_CORES == 48
    assert DECODE_ATTENTION_CORES == 16


def test_conservation_freeze_and_report_are_host_only(tmp_path: Path) -> None:
    freeze = run("feedback", "freeze", tmp_path / "freeze")
    assert freeze["freeze"]["ok"] is True
    results = run("feedback", "conservation", tmp_path / "conservation")
    assert results["conservation"]["ok"] is True
    assert results["conservation"]["ffn_down_first_class"] is True
    assert results["claims_throughput"] is False
    report = run("acceptance", "report", tmp_path / "report")
    assert report["report"]["host_only"] is True
    assert report["production_kept"] is False
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "no throughput" in text or "no tok/s" in text
    fixture = _json(FIXTURE)
    for key in _json(CONTRACT)["required_fixture_keys"]:
        assert key in fixture


def test_skip_gpu_decode_does_not_invent_llama_zeros(tmp_path: Path) -> None:
    d128 = run("feedback", "d128", tmp_path / "d128", skip_gpu=True)
    assert d128["d128"]["hardware_executed"] is False
    assert d128["d128"]["attribution_valid"] is False
    assert d128["d128"]["gap_attribution_complete"] is False
    counters = run("feedback", "counters", tmp_path / "counters", skip_gpu=True)
    assert counters["counters"]["ncu_available"] is False
    assert counters["counters"]["full_ncu_sweep"] is False
    assert COUNTER_LAUNCHES[0] == "q4_paired_gate_up"


def test_llama_prefill_graph_capture_is_excluded_from_measured_wall() -> None:
    contaminated = llama_graph_create_excluded_ms(2462.64, 1253.25)
    assert contaminated > 1000.0
    assert llama_graph_create_excluded_ms(1244.62, 1239.30) == 0.0
    assert llama_graph_create_excluded_ms(1246.60, 1249.44) == 0.0
    split = accounting_split(
        graph_wall_ms=2462.64,
        eager_instrumented_wall_ms=1253.25,
        eager_work_ms=1223.80,
        attributed_union_ms=1223.80,
        d2h_ms=0.0,
        commit_ms=0.0,
        graph_create_ms=contaminated,
    )
    assert split["graph_create_excluded_ms"] == contaminated
    assert split["unexplained_share"] <= UNEXPLAINED_LIMIT
    adapter = ADAPTER.read_text(encoding="utf-8")
    assert "Graph construction stays outside measured windows" in adapter


def test_llama_logs_and_walls_are_parsed_not_zero_filled() -> None:
    log = llama_log_path("post098_selected", "decode", "eager_diagnostic", 3)
    assert log.name == "llama-post098_selected-decode-eager_diagnostic-samples3.log"
    text = (
        'QW38_OPT060_LLAMA_ATTRIBUTION_RESULT={"schema_version":1,"task":"OPT-099",'
        '"engine":"llama","sample_index":0,"uninstrumented_wall_ms":12.5,'
        '"instrumented_wall_ms":0}\n'
        '{"schema_version":1,"task":"OPT-099","engine":"llama","graph_mode":'
        '"eager_diagnostic","count":1,"records":[{"role":"ffn_out-0",'
        '"tensor_name":"ffn_out-0","launch_family":"mmvq","complete_work_ms":0.04,'
        '"start_ms":0.0,"end_ms":0.04}]}\n'
    )
    walls = parse_llama_sample_walls(text)
    dumps = parse_llama_dumps(text)
    assert walls[0]["uninstrumented_wall_ms"] == 12.5
    assert dumps[0][0]["tensor_name"] == "ffn_out-0"
    ranking = {
        "status": "ranked",
        "gap_attribution_complete": True,
        "ranked": [
            {
                "family": "ffn_down",
                "quartz_ms": 4.0,
                "llama_ms": 0.04,
                "matched_llama_excess_ms": 3.96,
                "llama_covered": True,
            }
        ],
    }
    triggers = trigger_fields(ranking, {})
    assert triggers["opt102_q4_repack"]["go"] is True
    assert triggers["opt102_q4_repack"]["excess_ms"] == 3.96


def test_opt099_result_prefix_is_parsed() -> None:
    stdout = (
        'QW38_OPT099_RESULT={"task":"OPT-099","keep":false,'
        '"observed_warmups":3,"observed_samples":3,"observed_candidates":2,'
        '"observed_shapes":1,"observed_tier":"screen","pairs":1,'
        '"sample_ids":[0,1,2]}\n'
    )
    observed = parse_native_observation(stdout)
    assert observed["keep"] is False
    assert observed["observed_warmups"] == 3
    assert observed["observed_samples"] == 3
    assert observed["observed_candidates"] == 2
    assert observed["sample_ids"] == [0, 1, 2]
