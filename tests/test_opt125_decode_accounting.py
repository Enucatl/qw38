"""Host tests for OPT-125 decode timing and traffic-evidence accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt115_pipeline_traffic import bytes_to_peak_ms
from tools.opt125_decode_accounting import (
    AA_PAIRS,
    AA_WARMUPS,
    CONTRACT,
    CORRECTION_DATE,
    DISPOSITIONS,
    FIXTURE,
    ITERATION,
    NATIVE,
    PHASES,
    RANKED_TASKS,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    answers,
    classify_disjoint_intervals,
    correct_session_arm,
    derive_metrics,
    historical_corrections,
    metric_catalog,
    rank_opt126_131,
    reconcile_busy_vs_compulsory,
    shared_keep_protocol,
    synthetic_metric_cases,
    validate_fixture,
    weight_read_account,
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
NATIVE_SRC = ROOT / "cuda/opt125_decode_accounting_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-125")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-125"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["pinned_llama_authority_unchanged"] is True
    assert contract["llama_revision"] == "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    assert contract["final_stack"] == "combined_opt118_opt119"
    assert contract["aa_warmups"] == AA_WARMUPS
    assert contract["aa_pairs"] == AA_PAIRS
    assert contract["correction_date"] == CORRECTION_DATE
    assert iteration["diagnostics_make_target"] == "cuda-opt125-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt125-diagnostics" in makefile
    assert "qw38-cuda-opt125-decode-accounting-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "identity|matched-baseline" in native
    assert "activity-windows|long-context" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "decode_only_tok_s" in native
    assert "request_tok_s" in native
    assert "first_sampled_token_needs_eval" in native
    assert "QW38_OPT125_DECODE_ACCOUNTING_RESULT=" in native
    assert "QW38_OPT125_NATIVE_COUNTS=" in native
    assert "QW38_OPT125_DECODE_ACCOUNTING_RESULT=" in runner
    assert "QW38_OPT125_NATIVE_COUNTS=" in runner
    assert PHASES == (
        "metrics",
        "freeze",
        "matched-baseline",
        "activity-trace",
        "intervals",
        "weight-reads",
        "long-context",
        "ranking",
        "report",
    )
    validate_future_keep_policy("OPT-125", iteration)


def test_iteration_loop_products_and_plan() -> None:
    iteration = load_contract("OPT-125")
    metrics = iteration["workloads"]["metrics"]
    freeze = iteration["workloads"]["freeze"]
    matched = iteration["workloads"]["matched-baseline"]
    activity = iteration["workloads"]["activity-trace"]
    intervals = iteration["workloads"]["intervals"]
    weights = iteration["workloads"]["weight-reads"]
    long_ctx = iteration["workloads"]["long-context"]
    ranking = iteration["workloads"]["ranking"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(metrics, "feedback")) == 1
    assert loop_product(workload_for_mode(freeze, "feedback")) == 1
    assert loop_product(workload_for_mode(matched, "feedback")) == 48
    assert loop_product(workload_for_mode(activity, "feedback")) == 4
    assert loop_product(workload_for_mode(intervals, "feedback")) == 1
    assert loop_product(workload_for_mode(weights, "feedback")) == 1
    assert loop_product(workload_for_mode(long_ctx, "feedback")) == 6
    assert loop_product(workload_for_mode(ranking, "feedback")) == 1
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-125", "feedback", iteration, "metrics")
    assert "phase=metrics" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "leaf" in proof or "unobserved" in proof
    assert "1792" in " ".join(iteration["proof_limit"]) or "peak" in proof


def test_synthetic_metrics_known_durations_and_missing_case() -> None:
    catalog = metric_catalog()
    assert catalog["first_sampled_token_needs_eval"] is False
    assert set(catalog["metrics"]) == {
        "setup",
        "prefill",
        "time_to_first_emitted_token",
        "decode_only",
        "sample_eval_output",
        "complete_request",
        "fixed_token_engine_probe",
    }
    cases = {row["id"]: row for row in synthetic_metric_cases()}
    assert cases["known_prefill"]["prefill_tok_s"] == 40960.0
    assert cases["known_decode"]["decode_only_tok_s"] == 100.0
    assert (
        abs(float(cases["known_decode"]["request_tok_s"]) - (256000.0 / 2640.0)) < 1e-9
    )
    assert cases["known_decode"]["eval_calls"] == 256
    assert cases["missing_failed"]["ok"] is False
    assert cases["missing_failed"]["status"] == "missing_or_failed"
    failed = derive_metrics(
        prefill_ms=None, decode_only_ms=None, outputs=32, missing=True
    )
    assert failed["decode_only_tok_s"] is None
    assert failed["request_tok_s"] is None


def test_opt123_session_arm_correction_does_not_rewrite_history() -> None:
    arm = {"wall_ms": 3967.39429, "ttft_ms": 3010.93018, "tok_s": 8.06574726}
    corrected = correct_session_arm(arm, outputs=32)
    assert corrected["comparable"] is True
    assert corrected["do_not_silently_rewrite"] is True
    assert corrected["historical_tok_s"] == 8.06574726
    assert corrected["historical_metric"] == "complete_request"
    assert abs(float(corrected["request_tok_s"]) - 8.06574726) < 1e-4
    decode = float(corrected["decode_only_tok_s"])
    assert decode > 30.0
    assert decode < 40.0
    missing = correct_session_arm({"tok_s": 1.0}, outputs=32)
    assert missing["comparable"] is False
    hist = historical_corrections()
    assert hist["do_not_silently_rewrite"] is True
    assert hist["opt123_run_session_arm_includes_prefill"] is True
    d8192 = (hist.get("long_context") or {}).get("d8192") or {}
    assert d8192.get("comparable") is True
    assert float(d8192["decode_only_tok_s"]) > float(d8192["historical_tok_s"])
    d128 = (hist.get("p4096_d128_d2048") or {}).get("d128") or {}
    published = d128.get("published_combined_over_llama_request_over_decode_only")
    matched = d128.get("matched_decode_only_combined_over_llama")
    assert published is not None and matched is not None
    d2048 = (hist.get("p4096_d128_d2048") or {}).get("d2048") or {}
    assert d2048.get("matched_decode_only_combined_over_llama") > d2048.get(
        "published_combined_over_llama_request_over_decode_only"
    )


def test_leaf_gaps_are_unobserved_not_gpu_idle() -> None:
    records = [
        {
            "role": "ffn_mmq",
            "attribution_role": "member",
            "start_ms": 0.0,
            "end_ms": 8.0,
            "complete_work_ms": 8.0,
        },
        {
            "role": "attention_core",
            "attribution_role": "member",
            "start_ms": 18.0,
            "end_ms": 20.0,
            "complete_work_ms": 2.0,
        },
    ]
    split = classify_disjoint_intervals(records, wall_ms=20.0, tokens=1)
    assert split["leaf_gap_is_not_gpu_idle"] is True
    assert split["device_inactive_proven_ms"] == 0.0
    assert split["unobserved_ms"] > 0.0
    assert split["opt115_classify_timeline_not_independent_inactivity_proof"] is True
    assert split["never_count_cpu_wait_overlapping_gpu_twice"] is True


def test_busy_below_compulsory_labels_bound_unsupported() -> None:
    recon = reconcile_busy_vs_compulsory(
        busy_ms=8.2922,
        compulsory_peak_ms=10.1829,
        request_ms=18.51,
        timeline_ms=19.10,
        workload="d128",
    )
    assert recon["busy_below_compulsory"] is True
    assert recon["bound_supported"] is False
    assert recon["observed_idle_is_not_unavoidable_lower_bound"] is True
    mismatch = reconcile_busy_vs_compulsory(
        busy_ms=10.54,
        compulsory_peak_ms=10.18,
        request_ms=24.0193,
        timeline_ms=20.9921,
        workload="d2048",
    )
    assert mismatch["timeline_vs_request_mismatch"] is True
    account = weight_read_account(prefix=128)
    assert account["embedding"]["full_table_excluded"] is True
    assert account["output_head"]["combined_d2h_bytes"] == 4
    assert account["physical_dram_traffic"] is None
    assert abs(bytes_to_peak_ms(1_792_000_000) - 1.0) < 1e-9


def test_ranking_and_protocol_cover_126_131() -> None:
    ranking = rank_opt126_131()
    tasks = [row["task"] for row in ranking["ranked"]]
    assert tasks == list(RANKED_TASKS)
    by_task = {row["task"]: row for row in ranking["ranked"]}
    assert by_task["OPT-126"]["disposition"] == "proceed"
    assert by_task["OPT-127"]["disposition"] == "proceed"
    assert by_task["OPT-128"]["disposition"] == "insufficient_evidence"
    assert by_task["OPT-129"]["disposition"] == "proceed"
    assert by_task["OPT-130"]["disposition"] == "insufficient_evidence"
    assert by_task["OPT-131"]["disposition"] == "insufficient_evidence"
    assert ranking["missing_profiling_does_not_block_opt126_or_opt129"] is True
    for row in ranking["ranked"]:
        assert row["disposition"] in DISPOSITIONS
    protocol = shared_keep_protocol()
    assert protocol["parent"] == "combined_opt118_opt119"
    assert protocol["screen_pairs"] == 5
    assert protocol["acceptance_pairs"] == 10
    assert "decode_only" in protocol["metrics_required"]
    assert "complete_request" in protocol["metrics_required"]
    ans = answers()
    assert ans["is_10ms_per_token_true_inactive"] is False
    assert ans["llama_ratios_survive_matched_boundaries"] is False
    assert ans["long_context_ratios_survive_matched_boundaries"] is False
    assert ans["unknowns_are_valid_findings"] is True


def test_split_windows_uses_frontier_token_positions() -> None:
    from tools.opt125_decode_accounting import split_windows

    records = []
    for pos in (128, 129, 130, 131, 254, 255, 256, 257, 380, 381, 382, 383):
        records.append(
            {
                "token_position": pos,
                "role": "ffn_mmq",
                "start_ms": float(pos),
                "end_ms": float(pos) + 1.0,
                "complete_work_ms": 1.0,
            }
        )
    buckets = split_windows(records, 256)
    assert {row["token_position"] for row in buckets["early"]} == {128, 129, 130, 131}
    assert {row["token_position"] for row in buckets["middle"]} == {254, 255, 256, 257}
    assert {row["token_position"] for row in buckets["late"]} == {380, 381, 382, 383}


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    result = validate_fixture(payload)
    assert result["ok"] is True
    assert payload["claims_throughput"] is False
    assert payload["claims_performance_improvement"] is False
    assert REPORT.is_file()
    report = REPORT.read_text(encoding="utf-8")
    assert "Is 10 ms/token true inactive" in report
    assert "causally removable" in report.casefold() or "Causally removable" in report
    assert "matched boundaries" in report
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
