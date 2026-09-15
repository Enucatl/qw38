"""Host tests for the OPT-138 remaining-gap profiler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt138_remaining_gap_profile import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    PHASES,
    PREFIXES,
    PREFILL_TOKENS,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTOR,
    _experiment_entry,
    _parse_ncu_metric_names,
    _select_ncu_metrics,
    counter_timeout_s,
    family_plan,
    replay_family_for,
    validate_fixture,
)
from tools.performance_evidence import (
    classify_kernel_family,
    missing_counter_record,
    parent_identity_ok,
    prefill_output_policy_ok,
    reconcile_whole_gap,
    regroup_fused_families,
    replay_production_boundary_ok,
    select_top_two_families,
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
NATIVE_SRC = ROOT / "cuda/opt136_matched_decode_profile_test.cu"
REPLAY_SRC = ROOT / "cuda/optimization_component_replay.cu"
REPLAY_HDR = ROOT / "cuda/optimization_component_replay.h"
LLAMA_SRC = ROOT / "tools/llama_authority/opt136_decode_profile.cpp"
ATTRIB = ROOT / "cuda/engine_attribution.h"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-138")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    replay = REPLAY_SRC.read_text(encoding="utf-8")
    header = REPLAY_HDR.read_text(encoding="utf-8")
    llama = LLAMA_SRC.read_text(encoding="utf-8")
    attrib = ATTRIB.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt138_remaining_gap_profile.py").read_text(encoding="utf-8")
    opt136 = (ROOT / "tools/opt136_graph_accounting.py").read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-138"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["opt137_dense_mma"] is True
    assert contract["prefixes"] == list(PREFIXES)
    assert contract["prefill_tokens"] == PREFILL_TOKENS
    assert iteration["diagnostics_make_target"] == "cuda-opt138-diagnostics"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt138-diagnostics" in makefile
    assert "prefill-unprofiled|prefill-graph-capture|prefill-node-capture" in native
    assert "opt138.prefill" in native or "kOpt138PrefillNvtx" in native
    assert "final_token_logits_only" in native
    assert "--opt138-protocol" in replay
    assert "kOpt138ReplayProtocol" in header
    assert "prefill-unprofiled" in llama
    assert "n_ubatch" in llama
    assert "kOpt138DiagnosticIdentityMarkers" in attrib
    assert "opt136_graph_accounting" in tool
    assert "PREFILL_TOKENS" in opt136
    assert PHASES == (
        "preflight",
        "baseline",
        "capture",
        "families",
        "counters",
        "replay",
        "report",
    )
    validate_future_keep_policy("OPT-138", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-138")
    preflight = iteration["workloads"]["preflight"]
    baseline = iteration["workloads"]["baseline"]
    capture = iteration["workloads"]["capture"]
    families = iteration["workloads"]["families"]
    counters = iteration["workloads"]["counters"]
    replay = iteration["workloads"]["replay"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(baseline, "acceptance")) == 78
    assert loop_product(workload_for_mode(capture, "acceptance")) == 54
    assert loop_product(workload_for_mode(families, "acceptance")) == 3
    assert loop_product(workload_for_mode(counters, "acceptance")) == 8
    assert loop_product(workload_for_mode(replay, "acceptance")) == 32
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-138", "acceptance", iteration, "capture")
    assert "phase=capture" in described
    assert "loop_product=1" in family_plan("acceptance", "preflight")


def test_fused_family_regrouping() -> None:
    quartz = {
        "q4_gate": 4.0,
        "q4_up": 2.0,
        "q4_swiglu": 1.0,
        "q4_down": 5.0,
        "unclassified": 0.5,
    }
    llama = {"q4_gate": 2.0, "q4_up": 1.0, "q4_swiglu": 0.5, "q4_down": 3.0}
    grouped = regroup_fused_families(quartz, llama)
    families = {row["family"]: row for row in grouped["families"]}
    assert "q4_ffn_fused" in families
    assert families["q4_ffn_fused"]["Quartz_ms"] == 7.0
    assert families["q4_ffn_fused"]["llama_ms"] == 3.5
    assert families["q4_ffn_fused"]["delta_ms"] == 3.5
    assert grouped["unclassified_ms"] == 0.5
    assert grouped["unmatched_work_gap_ms"] == 0.5


def test_stale_parent_identity() -> None:
    ok = parent_identity_ok(
        {
            "execution_graphs": "decode_segments8",
            "q4_decode": "llama_q4k_mmvq",
            "opt137_dense_mma": True,
        }
    )
    assert ok["ok"] is True
    stale = parent_identity_ok(
        {
            "execution_graphs": "decode_segments8",
            "q4_decode": "llama_q4k_mmvq",
            "opt137_dense_mma": False,
        }
    )
    assert stale["ok"] is False
    assert stale["reason"] == "stale parent identity"
    assert "stale_opt137_mma_parent" in stale["mismatches"]


def test_parse_ncu_metric_names_from_csv_and_table() -> None:
    csv = (
        '"Metric name","Metric type"\n'
        '"dram__bytes","Counter"\n'
        '"dram__bytes_op_read","Counter"\n'
    )
    assert _parse_ncu_metric_names(csv) == [
        "dram__bytes",
        "dram__bytes_op_read",
    ]
    table = (
        "Device NVIDIA GeForce RTX 5090 (GB202)\n"
        "Metric Name                                                                 Metric Type\n"
        "--------------------------------------------------------------------------- ---------------\n"
        "dram__bytes                                                                 Counter\n"
    )
    assert _parse_ncu_metric_names(table) == ["dram__bytes"]


def test_select_ncu_metrics_uses_aliases_not_table_lines() -> None:
    names = [
        "dram__bytes",
        "dram__bytes_op_read",
        "dram__bytes_op_write",
        "dram__throughput",
        "lts__t_sector_hit_rate",
        "lts__t_sectors",
        "sm__pipe_tensor_cycles_active",
        "sm__throughput",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_long_scoreboard",
        "smsp__warps_issue_stalled_barrier",
        "launch__registers_per_thread",
    ]
    selected = _select_ncu_metrics(names)
    assert selected == [
        "dram__bytes_op_read",
        "dram__bytes_op_write",
        "dram__bytes",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors",
        "lts__t_sector_hit_rate",
        "sm__pipe_tensor_cycles_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_long_scoreboard",
        "smsp__warps_issue_stalled_barrier",
        "launch__registers_per_thread",
    ]


def test_counter_timeout_is_extended_for_prompt_ffn_replay() -> None:
    assert counter_timeout_s("decode-attention") == 300
    assert counter_timeout_s("prompt-ffn") == 1200
    assert counter_timeout_s("prompt-attention") == 1200
    assert replay_family_for("prefill", "attn_core") == "prompt-attention"
    assert replay_family_for("decode", "attn_core") == "decode-attention"


def test_missing_counters_are_null_never_zero() -> None:
    row = missing_counter_record(
        kernel="quartz:attn_core",
        engine="quartz",
        error="ERR_NVGPUCTRPERM",
    )
    assert row["dram_read_bytes"] is None
    assert row["dram_throughput"] is None
    assert row["achieved_occupancy"] is None
    assert row["zero_filled"] is False
    assert row["full_ncu_sweep"] is False
    assert row["error"] == "ERR_NVGPUCTRPERM"


def test_replay_production_boundary_mismatch() -> None:
    ok = replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": "decode",
            "replay_family": "decode-ffn",
        }
    )
    assert ok["ok"] is True
    mismatch = replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": "prefill",
            "replay_family": "decode-attention",
        }
    )
    assert mismatch["ok"] is False
    assert mismatch["reason"] == "replay/production boundary mismatch"
    prefill_ok = replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": "prefill",
            "replay_family": "prompt-attention",
        }
    )
    assert prefill_ok["ok"] is True
    hot = replay_production_boundary_ok(
        {
            "cache_mode": "hot",
            "production_weights": True,
            "phase": "decode",
            "replay_family": "decode-ffn",
        }
    )
    assert hot["ok"] is False


def test_prefill_output_policy_mismatch() -> None:
    ok = prefill_output_policy_ok(
        {
            "final_token_logits_only": True,
            "logits_rows": 1,
            "full_vocabulary_logits_every_row": False,
        }
    )
    assert ok["ok"] is True
    bad = prefill_output_policy_ok(
        {
            "final_token_logits_only": False,
            "logits_rows": 4096,
            "full_vocabulary_logits_every_row": True,
        }
    )
    assert bad["ok"] is False
    assert bad["reason"] == "prefill output-policy mismatch"


def test_residual_above_five_percent_stops_ranking() -> None:
    recon = reconcile_whole_gap(
        quartz_wall_ms=100.0,
        llama_wall_ms=60.0,
        matched_disjoint_excess_ms=10.0,
        unmatched_work_gap_ms=5.0,
        scheduling_host_residual_ms=5.0,
    )
    assert recon["observed_gap_ms"] == 40.0
    assert recon["ranking_complete"] is False
    assert recon["absolute_residual_ms"] == 20.0
    tight = reconcile_whole_gap(
        quartz_wall_ms=100.0,
        llama_wall_ms=60.0,
        matched_disjoint_excess_ms=30.0,
        unmatched_work_gap_ms=8.0,
        scheduling_host_residual_ms=2.0,
    )
    assert tight["ranking_complete"] is True


def test_deterministic_top_two_selection() -> None:
    rows = [
        {
            "family": "z_family",
            "d128_middle_deltas_ms": [3.0, 3.0, 3.0],
            "d2048_middle_deltas_ms": [1.0, 1.0, 1.0],
        },
        {
            "family": "a_family",
            "d128_middle_deltas_ms": [3.0, 3.0, 3.0],
            "d2048_middle_deltas_ms": [1.0, 1.0, 1.0],
        },
        {
            "family": "edge_only",
            "d128_middle_deltas_ms": [-0.1, -0.1, -0.1],
            "d2048_middle_deltas_ms": [-0.2, -0.2, -0.2],
            "d128_early_deltas_ms": [9.0, 9.0, 9.0],
            "d128_late_deltas_ms": [8.0, 8.0, 8.0],
        },
        {
            "family": "bigger",
            "d128_middle_deltas_ms": [1.0, 1.0, 1.0],
            "d2048_middle_deltas_ms": [5.0, 5.0, 5.0],
        },
    ]
    selected = select_top_two_families(rows, phase="decode")
    names = [row["family"] for row in selected["selected"]]
    assert names[0] == "bigger"
    assert names[1] == "a_family"
    assert "edge_only" not in names
    assert "edge_only" in selected["unstable"]
    prefill = select_top_two_families(
        [
            {"family": "prompt_mmq", "p4096_deltas_ms": [4.0, 4.0, 4.0]},
            {"family": "attn_fused", "p4096_deltas_ms": [6.0, 6.0, 6.0]},
        ],
        phase="prefill",
    )
    assert [row["family"] for row in prefill["selected"]] == [
        "attn_fused",
        "prompt_mmq",
    ]


def test_prompt_mmq_vs_decode_mmv_classification() -> None:
    assert classify_kernel_family("mul_mat_q_q4_K") == "prompt_mmq"
    assert classify_kernel_family("mul_mat_vec_q") == "decode_mmv"
    assert classify_kernel_family("q4k_mmvq") == "decode_mmv"


def test_fixture_keys() -> None:
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert REPORT.parent.as_posix().endswith("opt138-remaining-gap")


def test_throughput_only_does_not_set_supported_mechanism() -> None:
    entry = _experiment_entry(
        "decode",
        {
            "selected": [
                {
                    "family": "attn_core",
                    "score_ms": 4.2,
                    "stats": {"mean_ms": 4.2, "one_sided_low": 4.1, "n": 3},
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
                    "dram_throughput": 70.0,
                    "sm_throughput": 20.0,
                    "error": None,
                }
            ]
        },
        {"rounds": []},
    )
    assert entry["supported_mechanism"] is None
    assert entry["candidate"] is None


def test_collector_requests_csv_and_imports_opt139() -> None:
    tool = (ROOT / "tools/opt138_remaining_gap_profile.py").read_text(encoding="utf-8")
    assert "from tools import opt139_counter_identity as opt139" in tool
    assert '"--csv"' in tool
    assert "counter_record_from_ncu_blob" in tool
    assert "OPT140_INELIGIBLE" in tool
