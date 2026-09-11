"""Host tests for OPT-070 keep revalidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt070_keep_revalidation import (
    decide_verdict,
    opt073_quality,
    opt074_coverage,
    paired_student_t,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt070_keep_revalidation_contract.json"
ITERATION = ROOT / "pins/opt070_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt070_keep_revalidation.json"
REPORT = ROOT / "evidence/optimization/opt070-keep-revalidation/REPORT.md"
OPT069_REPORT = ROOT / "evidence/optimization/opt069-batch-gate/REPORT.md"
OPT069_FIXTURE = ROOT / "fixtures/opt069_batch_gate.json"
MAKEFILE = ROOT / "Makefile"
PATH_H = ROOT / "cuda/q8_decode_path.cuh"
DOTS = ROOT / "cuda/q8_decode_dots.cu"
REPLAY = ROOT / "cuda/optimization_component_replay.cu"
PROBE = ROOT / "cuda/optimization_engine_probe.cu"
MMV_H = ROOT / "cuda/quant_mmv.h"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(
    *,
    warmups: int,
    samples: int,
    candidates: int = 2,
    shapes: int = 1,
    tier: str,
    keep: bool,
    pairs: int = 1,
    sample_ids: list[int] | None = None,
    acceptance_executed: bool | None = None,
) -> str:
    payload = {
        "warmups": warmups,
        "samples": samples,
        "observed_warmups": warmups,
        "observed_samples": samples,
        "observed_candidates": candidates,
        "observed_shapes": shapes,
        "observed_tier": tier,
        "pairs": pairs,
        "sample_ids": sample_ids if sample_ids is not None else list(range(samples)),
        "acceptance_executed": (
            acceptance_executed
            if acceptance_executed is not None
            else tier == "acceptance"
        ),
        "keep": keep,
    }
    return "QW38_OPT070_NATIVE_COUNTS=" + json.dumps(payload)


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-070")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-070"
    assert contract["claims_throughput"] is False
    assert contract["installed_q8_layout"] == "r2_w2"
    assert contract["installed_mmq"] == "fma_async_x"
    assert iteration["task"] == "OPT-070"
    assert iteration["diagnostics_make_target"] == "cuda-opt070-diagnostics"
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["claims_throughput"] is False
    assert fixture["installed_q8_layout"] == "r2_w2"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt070-diagnostics" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    path = PATH_H.read_text(encoding="utf-8")
    dots = DOTS.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    header = MMV_H.read_text(encoding="utf-8")
    assert "apply_q8_layout_ident" in path
    assert "Q8DecodeLayoutScope" in path
    assert "record_q8_decode_dispatch" in dots
    assert "--q8-layout" in replay
    assert "--mmq-async-x" in replay
    assert "layer.gdn.output" in replay
    assert "layer.attention.output" in replay
    assert "invalidate_q8_decode_staging" in replay
    assert "q8-ab" in probe
    assert "mmq-ab" in probe
    assert "override_before_capture_applied" in probe
    assert 'set_mmq_pipeline_path_override("off")' in probe
    assert "set_mmq_async_x_override" in header
    assert "set_q8_decode_layout_override" in path


def test_iteration_loop_products_and_mode_overrides() -> None:
    iteration = load_contract("OPT-070")
    q8 = iteration["workloads"]["q8"]
    mmq = iteration["workloads"]["mmq"]
    assert loop_product(workload_for_mode(q8, "feedback")) == 8
    assert loop_product(workload_for_mode(mmq, "feedback")) == 8
    assert loop_product(workload_for_mode(q8, "acceptance")) == 26
    assert loop_product(workload_for_mode(mmq, "acceptance")) == 26
    assert workload_for_mode(q8, "acceptance")["tier"] == "acceptance"
    assert workload_for_mode(q8, "feedback")["tier"] == "screen"
    plan = describe_plan("OPT-070", "feedback", iteration, "q8")
    assert "phase=q8" in plan
    assert "historical_oracles=none" in plan
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "screen-only keep=true" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["modes"]["acceptance"]["warm_repetitions"] == 0
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300


def test_paired_t_and_unadmitted_coverage_is_inconclusive() -> None:
    stats = paired_student_t(
        [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 10.0],
        [9.0, 9.1, 8.9, 9.2, 8.8, 9.0, 9.3, 8.7, 9.1, 9.0],
        critical=2.262,
    )
    assert stats["positive"] is True
    assert stats["mean_diff_ms"] == pytest.approx(1.0, abs=0.05)
    q8 = opt074_coverage(("q8_mixer_k5120", "q8_mixer_k6144"))
    mmq = opt074_coverage(("q4_gate_up_k5120", "q4_down_k17408"))
    assert q8["all_admitted"] is False
    assert mmq["all_admitted"] is False
    quality = opt073_quality()
    assert quality["quality_v3_engine_non_regression"] == "pass"
    verdict = decide_verdict(
        family="q8",
        component=stats,
        engine={
            "control_ms": [100.0] * 5,
            "candidate_ms": [99.0] * 5,
            "control_mean_ms": 100.0,
            "regression_upper_ms": 0.1,
        },
        coverage=q8,
        quality=quality,
        mode="acceptance",
    )
    assert verdict["verdict"] == "inconclusive"
    assert "opt074_coverage_unadmitted" in verdict["reasons"]
    assert verdict["shipping_unchanged"] is True


def test_retain_requires_admitted_coverage_and_positive_ci() -> None:
    stats = paired_student_t(
        [10.0] * 10,
        [9.0] * 10,
        critical=2.262,
    )
    verdict = decide_verdict(
        family="q8",
        component=stats,
        engine={
            "control_ms": [50.0] * 5,
            "candidate_ms": [49.0] * 5,
            "control_mean_ms": 50.0,
            "regression_upper_ms": 0.01,
        },
        coverage={"all_admitted": True},
        quality={"quality_v3_engine_non_regression": "pass"},
        mode="acceptance",
    )
    assert verdict["verdict"] == "retain"


def test_malformed_evidence_cannot_install_a_keep() -> None:
    iteration = load_contract("OPT-070")
    q8 = workload_for_mode(iteration["workloads"]["q8"], "feedback")
    screen_keep = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q8",
        workload=q8,
        stdout=_counts(warmups=1, samples=3, tier="screen", keep=True, pairs=3),
        success=True,
    )
    assert screen_keep["ok"] is False
    assert screen_keep["result_class"] == "screen_only_keep"
    wrong = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q8",
        workload=q8,
        stdout=_counts(warmups=0, samples=3, tier="screen", keep=False),
        success=True,
    )
    assert wrong["ok"] is False
    assert wrong["result_class"] == "native_count_mismatch"
    reused = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q8",
        workload=q8,
        stdout=_counts(
            warmups=1,
            samples=3,
            tier="screen",
            keep=False,
            sample_ids=[0, 0, 1],
        ),
        success=True,
    )
    assert reused["ok"] is False
    assert reused["result_class"] == "reused_sample_ids"
    instrumented = validate_performance_admission(
        {"performance_admission": {"instrumentation_only": True}},
        mode="acceptance",
        workload_name="q8",
        workload=q8,
        stdout=_counts(
            warmups=3,
            samples=10,
            tier="acceptance",
            keep=True,
            pairs=5,
        ),
        success=True,
    )
    assert instrumented["ok"] is False
    assert instrumented["result_class"] == "instrumentation_keep_forbidden"
    missing = validate_performance_admission(
        iteration,
        mode="acceptance",
        workload_name="q8",
        workload=workload_for_mode(iteration["workloads"]["q8"], "acceptance"),
        stdout=_counts(
            warmups=3,
            samples=10,
            tier="acceptance",
            keep=False,
            pairs=0,
        ),
        success=True,
    )
    assert missing["ok"] is False
    assert missing["result_class"] == "missing_pairs"
    unexecuted = validate_performance_admission(
        iteration,
        mode="acceptance",
        workload_name="q8",
        workload=workload_for_mode(iteration["workloads"]["q8"], "acceptance"),
        stdout=_counts(
            warmups=3,
            samples=10,
            tier="acceptance",
            keep=False,
            pairs=5,
            acceptance_executed=False,
        ),
        success=True,
    )
    assert unexecuted["ok"] is False
    assert unexecuted["result_class"] == "unexecuted_acceptance"


def test_parse_native_observation_reads_prefixed_counts() -> None:
    observed = parse_native_observation(
        _counts(warmups=1, samples=3, tier="screen", keep=False)
    )
    assert observed["observed_warmups"] == 1
    assert observed["observed_samples"] == 3
    assert observed["keep"] is False


def test_opt069_report_is_a_reporting_correction() -> None:
    from tools.opt069_batch_gate import assemble_from_retained_evidence

    assemble_from_retained_evidence()
    text = OPT069_REPORT.read_text(encoding="utf-8")
    fixture = _json(OPT069_FIXTURE)
    assert "Reporting correction (OPT-070)" in text
    assert fixture["status"] == "measured"
    assert "| Internal improvement with quality | unmeasured |" not in text
    assert "unmeasured until the release sitting" not in text
    assert fixture["p"]["quartz"]["mean_tok_s"] == pytest.approx(2914.65698)
    assert fixture["p"]["llama_cpp"]["avg_ts"] == pytest.approx(3142.517034)
    assert fixture["d2048"]["quartz"]["mean_tok_s"] == pytest.approx(35.4498482)
    assert fixture["opt056_gate_passed"] is False
    assert fixture["reporting_correction"] == "OPT-070"


def test_report_records_this_sitting_and_unresolved_keeps() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    assert "measured this sitting" in text
    assert "inconclusive" in text
    assert "r1_w4" in text
    assert "r2_w2" in text
    assert "fma_async_x" in text
    assert "opt-074" in text
    assert "override_before_capture" in text
    assert "48 gdn" in text
    fixture = _json(FIXTURE)
    assert fixture["status"] == "measured"
    assert fixture["q8"]["verdict"]["verdict"] == "inconclusive"
    assert fixture["mmq"]["verdict"]["verdict"] == "inconclusive"
    assert fixture["shipping_unchanged"] is True
