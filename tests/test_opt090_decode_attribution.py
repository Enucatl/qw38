"""Host tests for OPT-090 matched decode attribution. GPU-free."""

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
    rank_remaining_gaps,
)
from tools.opt090_decode_attribution import (
    CONTRACT,
    COUNTER_LAUNCHES,
    EXPECTED_DECODE_COUNTS,
    FIXTURE,
    GRAPH_REOPEN_MS,
    ITERATION,
    Q6_ATTENTION_OUTPUTS,
    Q8_GDN_OUTPUTS,
    REPORT,
    VERDICT_KEYS,
    canonical_family_opt090,
    expected_decode_counts_opt090,
    family_plan,
    gdn_reopen_eligible,
    graph_reopen_eligible,
    ident_order,
    idents_to_measure,
    model_weight_manifest,
    one_token_valid_records,
    opt088_control_selectors,
    opt089_selected_selectors,
    packed_bytes,
    quartz_ident_specs,
    run,
    selector_key,
    validate_call_counts_opt090,
    weight_byte_roofline,
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
NATIVE = ROOT / "cuda/opt060_engine_attribution_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-090")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-090"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["decode_output_tokens"] == 32
    assert contract["decode_prefixes"] == [128, 2048]
    assert contract["expected_decode_counts_per_token"]["ffn_gate_up_glu"] == 64
    assert iteration["task"] == "OPT-090"
    assert iteration["claims_throughput"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt090-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["case_ids"] == [
        "conservation",
        "d128",
        "d2048",
        "counters",
        "p4096",
        "report",
    ]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt090-diagnostics" in makefile
    assert "qw38-cuda-opt060-engine-attribution-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "--prefix" in native
    assert "--output-tokens" in native
    assert "--idents" in native
    assert "independently_restored" in native
    assert "QW38_OPT090_ATTRIBUTION_RESULT=" in native
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    assert 'kSelectedQ4DecodePath[] = "integer_q8_late"' in q4
    assert 'kSelectedFfnDecodePath[] = "paired_integer"' in ffn


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-090")
    conservation = iteration["workloads"]["conservation"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    counters = iteration["workloads"]["counters"]
    p4096 = iteration["workloads"]["p4096"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(conservation, "feedback")) == 1
    assert loop_product(workload_for_mode(d128, "feedback")) == 16
    assert loop_product(workload_for_mode(d2048, "feedback")) == 16
    assert loop_product(workload_for_mode(counters, "feedback")) == 5
    assert loop_product(workload_for_mode(p4096, "feedback")) == 16
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    assert d128["output_tokens"] == 32
    assert d128["prefix"] == 128
    assert d2048["prefix"] == 2048
    assert d128["warmups"] == 1
    assert d128["samples"] == 3
    assert counters["cases"] == 5
    described = describe_plan("OPT-090", "feedback", iteration, "conservation")
    assert "phase=conservation" in described
    assert "historical_oracles=none" in described
    assert family_plan("d128", "feedback")["execution_modes"] == 2
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "eager diagnostic is not production graph timing" in proof


def test_future_keep_policy_and_instrumentation_only() -> None:
    iteration = load_contract("OPT-090")
    validate_future_keep_policy("OPT-090", iteration)
    fixture = _json(FIXTURE)
    assert fixture["claims_throughput"] is False
    assert fixture["production_kept"] is False
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert list(VERDICT_KEYS) == [
        "kernel_parity_pass",
        "model_quality_pass",
        "performance_pass",
        "production_kept",
    ]


def test_two_quartz_idents_measure_once_if_identical() -> None:
    specs = quartz_ident_specs()
    assert [row["id"] for row in specs] == ["opt088_control", "opt089_selected"]
    control = opt088_control_selectors()
    selected = opt089_selected_selectors()
    current = idents_to_measure()
    if selector_key(control) == selector_key(selected):
        assert current["measure_once"] is True
        assert len(current["measured"]) == 1
        assert current["aliases"]["opt089_selected"] == current["measured"][0]["id"]
    else:
        assert current["measure_once"] is False
        assert [row["id"] for row in current["measured"]] == [
            "opt088_control",
            "opt089_selected",
        ]
    identical = (
        {**specs[0], "selectors": dict(control)},
        {**specs[1], "selectors": dict(control)},
    )
    once = idents_to_measure(identical)
    assert once["measure_once"] is True
    assert len(once["measured"]) == 1


def test_ab_ba_order_is_fixed() -> None:
    idents = ["opt088_control", "opt089_selected"]
    assert ident_order(0, idents) == ["opt088_control", "opt089_selected"]
    assert ident_order(1, idents) == ["opt089_selected", "opt088_control"]
    assert ident_order(2, idents) == ["opt088_control", "opt089_selected"]
    assert ident_order(0, ["opt089_selected"]) == ["opt089_selected"]


def test_expected_decode_counts_are_paired_not_128() -> None:
    expected = expected_decode_counts_opt090(1)
    assert expected["ffn_gate_up_glu"] == DECODE_GATE_UP_PAIRS == 64
    assert expected["ffn_down"] == DECODE_DOWNS == 64
    assert expected["gdn_core"] == DECODE_GDN_CORES == 48
    assert expected["attention_core"] == DECODE_ATTENTION_CORES == 16
    assert expected["q8_gdn_output"] == Q8_GDN_OUTPUTS == 48
    assert expected["q6_attention_output"] == Q6_ATTENTION_OUTPUTS == 16
    assert EXPECTED_DECODE_COUNTS["ffn_gate_up_glu"] != 128
    records = one_token_valid_records()
    checked = validate_call_counts_opt090(records, tokens=1)
    assert checked["ok"] is True
    families = {canonical_family_opt090(row) for row in records}
    assert "q8_gdn_output" in families
    assert "q6_attention_output" in families


def _paired_integer_token_records() -> list[dict[str, Any]]:
    """Native paired_integer decode shape: enclosing ffn_mmv plus leaf events."""
    records: list[dict[str, Any]] = []
    for layer in range(DECODE_GATE_UP_PAIRS):
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "ffn_mmv",
                "tensor_name": f"blk.{layer}.ffn_mmv",
                "attribution_role": "enclosing",
                "fused_member_ids": "",
                "fused_member_count": 1,
                "complete_work_ms": 0.16,
                "start_ms": layer * 0.4,
                "end_ms": layer * 0.4 + 0.16,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "ffn_gate_up_glu",
                "tensor_name": f"blk.{layer}.ffn_gate",
                "attribution_role": "enclosing",
                "fused_member_ids": "ffn_gate,ffn_up,ffn_glu",
                "fused_member_count": 3,
                "complete_work_ms": 0.08,
                "start_ms": layer * 0.4 + 0.02,
                "end_ms": layer * 0.4 + 0.10,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "ffn_down",
                "tensor_name": f"blk.{layer}.ffn_down",
                "attribution_role": "member",
                "fused_member_ids": "",
                "fused_member_count": 1,
                "complete_work_ms": 0.04,
                "start_ms": layer * 0.4 + 0.10,
                "end_ms": layer * 0.4 + 0.14,
                "stream_index": 0,
            }
        )
    for layer in range(DECODE_GDN_CORES):
        if layer == 0:
            records.append(
                {
                    "engine": "quartz",
                    "layer": 0,
                    "phase": "decode",
                    "token_position": 128,
                    "role": "gdn_core",
                    "tensor_name": "blk.0.gdn_core",
                    "attribution_role": "enclosing",
                    "fused_member_count": 1,
                    "complete_work_ms": 0.02,
                    "start_ms": 0.0,
                    "end_ms": 0.02,
                    "stream_index": 0,
                }
            )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "gdn_core",
                "tensor_name": f"blk.{layer}.gdn_core",
                "attribution_role": "enclosing",
                "fused_member_count": 1,
                "complete_work_ms": 0.03,
                "start_ms": 26.0 + layer * 0.05,
                "end_ms": 26.0 + layer * 0.05 + 0.03,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "gdn_conv_qk_norm_recurrence",
                "tensor_name": f"blk.{layer}.gdn_conv_qk_norm_recurrence",
                "attribution_role": "enclosing",
                "fused_member_ids": "gdn_conv,gdn_qk_norm,gdn_recurrence",
                "fused_member_count": 3,
                "complete_work_ms": 0.02,
                "start_ms": 26.0 + layer * 0.05 + 0.01,
                "end_ms": 26.0 + layer * 0.05 + 0.03,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "proj_gdn_output",
                "tensor_name": f"blk.{layer}.ssm_out.weight",
                "tensor_type": "Q8_0",
                "attribution_role": "member",
                "fused_member_count": 1,
                "complete_work_ms": 0.02,
                "start_ms": 30.0 + layer * 0.04,
                "end_ms": 30.0 + layer * 0.04 + 0.02,
                "stream_index": 0,
            }
        )
    for index in range(DECODE_ATTENTION_CORES):
        layer = 3 + index * 4
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "attention_core",
                "tensor_name": f"blk.{layer}.attention_core",
                "attribution_role": "enclosing",
                "fused_member_count": 1,
                "complete_work_ms": 0.05,
                "start_ms": 32.0 + index * 0.08,
                "end_ms": 32.0 + index * 0.08 + 0.05,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "attn_qk_prep_softmax_pv_merge",
                "tensor_name": f"blk.{layer}.attn_qk_prep_softmax_pv_merge",
                "attribution_role": "member",
                "fused_member_count": 1,
                "complete_work_ms": 0.03,
                "start_ms": 32.0 + index * 0.08 + 0.01,
                "end_ms": 32.0 + index * 0.08 + 0.04,
                "stream_index": 0,
            }
        )
        records.append(
            {
                "engine": "quartz",
                "layer": layer,
                "phase": "decode",
                "token_position": 128,
                "role": "proj_attn_output",
                "tensor_name": f"blk.{layer}.attn_output.weight",
                "tensor_type": "Q6_K",
                "attribution_role": "member",
                "fused_member_count": 1,
                "complete_work_ms": 0.02,
                "start_ms": 34.0 + index * 0.06,
                "end_ms": 34.0 + index * 0.06 + 0.02,
                "stream_index": 0,
            }
        )
    records.append(
        {
            "engine": "quartz",
            "layer": 63,
            "phase": "decode",
            "token_position": 128,
            "role": "logits",
            "tensor_name": "blk.63.logits",
            "attribution_role": "enclosing",
            "fused_member_count": 1,
            "complete_work_ms": 0.3,
            "start_ms": 36.0,
            "end_ms": 36.3,
            "stream_index": 0,
        }
    )
    records.append(
        {
            "engine": "quartz",
            "layer": 63,
            "phase": "decode",
            "token_position": 128,
            "role": "logits_projection",
            "tensor_name": "output.weight",
            "tensor_type": "Q6_K",
            "attribution_role": "member",
            "fused_member_count": 1,
            "complete_work_ms": 0.3,
            "start_ms": 36.0,
            "end_ms": 36.3,
            "stream_index": 0,
        }
    )
    return records


def test_enclosing_ffn_mmv_empty_members_is_one_paired_ffn() -> None:
    records = _paired_integer_token_records()
    checked = validate_call_counts_opt090(records, tokens=1)
    assert checked["ok"] is True
    assert checked["observed"]["ffn_gate_up_glu"] == 64
    assert checked["observed"]["ffn_down"] == 64
    assert checked["observed"]["gdn_core"] == 48
    assert checked["observed"]["attention_core"] == 16
    assert checked["observed"]["logits_projection"] == 1
    assert checked["observed"]["q8_gdn_output"] == 48
    assert checked["observed"]["q6_attention_output"] == 16


def test_wrong_counts_and_unexplained_wall_stop_ranking() -> None:
    records = one_token_valid_records()[:-1]
    checked = validate_call_counts_opt090(records, tokens=1)
    assert checked["ok"] is False
    incomplete = rank_remaining_gaps(
        [{"family": "ffn_gate_up_glu", "quartz_ms": 10.0, "llama_ms": 4.0}],
        {"unexplained_share": 0.06, "uncovered_wall_ms": 1.2},
        {"ok": True},
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["ranked"] == []
    assert UNEXPLAINED_LIMIT == 0.05
    invalid_counts = rank_remaining_gaps(
        [{"family": "ffn_gate_up_glu", "quartz_ms": 10.0, "llama_ms": 4.0}],
        {"unexplained_share": 0.01, "uncovered_wall_ms": 0.1},
        checked,
    )
    assert invalid_counts["gap_attribution_complete"] is False


def test_gdn_reopen_requires_repaired_defect() -> None:
    absent = gdn_reopen_eligible(None)
    assert absent["eligible"] is False
    incomplete = gdn_reopen_eligible({"description": "wide CI", "repaired": False})
    assert incomplete["eligible"] is False
    repaired = gdn_reopen_eligible(
        {
            "description": "restore boundary counted warmup",
            "repaired": True,
            "before": {"calls": 96, "identity": "stale"},
            "after": {"calls": 48, "identity": "matched"},
        }
    )
    assert repaired["eligible"] is True


def test_graph_reopen_requires_both_prefixes() -> None:
    missing = graph_reopen_eligible(0.6, 0.6, d128_valid=False, d2048_valid=True)
    assert missing["eligible"] is False
    below = graph_reopen_eligible(0.08, 0.10, d128_valid=True, d2048_valid=True)
    assert below["eligible"] is False
    assert GRAPH_REOPEN_MS == 0.50
    both = graph_reopen_eligible(0.55, 0.61, d128_valid=True, d2048_valid=True)
    assert both["eligible"] is True
    one = graph_reopen_eligible(0.55, 0.10, d128_valid=True, d2048_valid=True)
    assert one["eligible"] is False


def test_weight_roofline_uses_tensor_names_not_file_size() -> None:
    roofline = weight_byte_roofline()
    assert roofline["not_gguf_file_size"] is True
    ffn = packed_bytes(64 * 3 * 5120 * 17408, 12)
    assert roofline["ffn_q4_bytes_per_token"] == ffn
    names = {row["name"] for row in model_weight_manifest()}
    assert "blk.0.ffn_gate.weight" in names
    assert "blk.0.ssm_out.weight" in names
    assert "output.weight" in names
    assert COUNTER_LAUNCHES == (
        "q4_paired_gate_up",
        "q4_down",
        "q8_gdn_qkv",
        "sequential_gdn",
        "d2048_attention",
    )


def test_conservation_and_report_are_host_only(tmp_path: Path) -> None:
    results = run("feedback", "conservation", tmp_path / "conservation")
    assert results["conservation"]["ok"] is True
    assert results["conservation"]["call_counts_ok"] is True
    assert results["conservation"]["expected_ffn_pairs"] == 64
    assert results["conservation"]["dropped_events_rejected"] is True
    assert results["conservation"]["unexplained_stops_ranking"] is True
    assert results["claims_throughput"] is False
    prior = _json(FIXTURE)
    expected_graph = graph_reopen_eligible(
        prior.get("d128", {}).get("idle_ms_per_token"),
        prior.get("d2048", {}).get("idle_ms_per_token"),
        d128_valid=bool(prior.get("d128", {}).get("attribution_valid")),
        d2048_valid=bool(prior.get("d2048", {}).get("attribution_valid")),
    )
    report = run("acceptance", "report", tmp_path / "report")
    assert report["report"]["host_only"] is True
    assert report["report"]["cannot_convert_three_samples_into_admission"] is True
    assert report["gdn_reopen_eligible"] is False
    assert report["graph_reopen_eligible"] is expected_graph["eligible"]
    assert report["production_kept"] is False
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "no throughput" in text or "no tok/s" in text
    assert "eager" in text


def test_skip_gpu_decode_does_not_invent_bandwidth(tmp_path: Path) -> None:
    d128 = run("feedback", "d128", tmp_path / "d128", skip_gpu=True)
    assert d128["d128"]["hardware_executed"] is False
    assert d128["d128"]["attribution_valid"] is False
    counters = run("feedback", "counters", tmp_path / "counters", skip_gpu=True)
    assert counters["counters"]["ncu_available"] is False
    assert counters["counters"]["full_ncu_sweep"] is False
    assert counters["counters"]["event_evidence_fallback"] is True
    assert "unavailable" in str(counters["counters"].get("reason"))


def test_opt090_result_prefix_is_parsed() -> None:
    stdout = (
        'QW38_OPT090_RESULT={"task":"OPT-090","keep":false,'
        '"observed_warmups":1,"observed_samples":3,"observed_candidates":2,'
        '"observed_shapes":1,"observed_tier":"screen","pairs":1,'
        '"sample_ids":[0,1,2]}\n'
    )
    observed = parse_native_observation(stdout)
    assert observed["keep"] is False
    assert observed["observed_warmups"] == 1
    assert observed["observed_samples"] == 3
    assert observed["observed_candidates"] == 2
    assert observed["sample_ids"] == [0, 1, 2]
