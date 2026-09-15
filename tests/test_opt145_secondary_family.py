"""Host tests for OPT-145 secondary-family freeze and keep/reject wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt139_counter_identity import admit_supported_mechanism
from tools.opt141_matched_llama_counters import fused_boundary_ok
from tools.opt145_secondary_family import (
    CONTRACT,
    CROSSOVER,
    FIXTURE,
    ITERATION,
    MMA_THRESHOLD,
    PARENT_STACK,
    PHASES,
    QUALITY_NATIVE,
    QUARTZ_PROMPT_MMQ_KERNEL,
    QUARTZ_RESIDUAL_KERNEL,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    SHIPPING_Q4,
    PAIR_COUNT,
    WARMUPS,
    excess_fraction,
    family_plan,
    forbidden_reason,
    freeze_candidate,
    load_contract,
    production_pins_parent,
    prompt_mmq_evidence,
    residual_norm_quant_evidence,
    select_secondary_family,
    skip_payload,
    target_guard_roles,
    validate_fixture,
)
from tools.performance_keep_policy import freeze_hash, validate_opt_in_contract
from tools.run_optimization_task import (
    describe_plan,
    load_contract as load_iteration,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
RUNNER = ROOT / "tools/opt145_secondary_family.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt145_secondary_family_provenance.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gaps(
    *, residual_positive: bool = True, prompt_positive: bool = True
) -> dict[str, Any]:
    return {
        "decode_selection": {
            "selected": [
                {
                    "family": "residual_norm_quant",
                    "score_ms": 7.21,
                    "d128_middle": {
                        "n": 3,
                        "mean_ms": 7.21,
                        "one_sided_low": 7.17,
                        "significant_positive_excess": residual_positive,
                    },
                    "d2048_middle": {
                        "n": 3,
                        "mean_ms": 6.66,
                        "one_sided_low": 6.64,
                        "significant_positive_excess": residual_positive,
                    },
                    "source": "middle_window",
                }
            ]
        },
        "prefill_selection": {
            "selected": [
                {
                    "family": "prompt_mmq",
                    "score_ms": 27.11,
                    "stats": {
                        "n": 3,
                        "mean_ms": 27.11,
                        "one_sided_low": 25.27,
                        "significant_positive_excess": prompt_positive,
                    },
                    "source": "p4096_complete",
                }
            ]
        },
        "decode_d128_middle": {"quartz_window_ms": 201.6},
        "decode_d2048_middle": {"quartz_window_ms": 246.2},
        "prefill_p4096": {"quartz_window_ms": 1347.8},
    }


def _counter_record(*, family: str, mechanism: bool) -> dict[str, Any]:
    workload = "D128" if family == "residual_norm_quant" else "P4096"
    kernel = (
        QUARTZ_RESIDUAL_KERNEL
        if family == "residual_norm_quant"
        else QUARTZ_PROMPT_MMQ_KERNEL
    )
    record: dict[str, Any] = {
        "engine": "quartz",
        "family": family,
        "phase": "decode" if family == "residual_norm_quant" else "prefill",
        "workload": workload,
        "expected_kernel": kernel,
        "target_kernel": kernel,
        "replay_family": "decode-mixer"
        if family == "residual_norm_quant"
        else "prompt-ffn",
        "dram_read_bytes": 8.0,
        "dram_throughput": 2.0,
        "sm_throughput": 5.0,
        "achieved_occupancy": 10.0,
        "identity": {
            "ok": True,
            "phase": "decode" if family == "residual_norm_quant" else "prefill",
            "engine": "quartz",
            "workload": workload,
            "kernel": kernel,
            "expected_kernel": kernel,
            "replay_family": "decode-mixer"
            if family == "residual_norm_quant"
            else "prompt-ffn",
            "replay_boundary": {
                "ok": True,
                "replay_family": "decode-mixer"
                if family == "residual_norm_quant"
                else "prompt-ffn",
            },
            "mismatches": [],
        },
    }
    if mechanism:
        record["source_observations"] = [f"{kernel} named SASS bank conflict"]
        record["sass_observations"] = ["LDSM.16 vs cp.async reduction"]
        record["proposed_mechanism"] = "named_sass_load_change"
        record["candidate"] = f"{kernel}_v2"
    return record


def _comparisons() -> dict[str, Any]:
    return {
        "decode.residual_norm_quant.d128": {
            "fusion": {
                "ok": False,
                "reason": "fused_boundary_mismatch",
                "details": ["llama_quantize_q8_1_vs_quartz_rms_plus_bf16_quant_fusion"],
            },
            "comparison": {
                "comparable": False,
                "reason": "quartz_counters_unusable",
            },
        },
        "prefill.prompt_mmq.p4096": {
            "fusion": {"ok": True, "reason": None, "details": []},
            "comparison": {
                "comparable": False,
                "reason": "quartz_counters_unusable",
            },
        },
    }


def test_contracts_makefile_and_registration() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-145")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert contract["task"] == "OPT-145"
    assert contract["parent"] == PARENT_STACK
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["crossover_threshold"] == CROSSOVER
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["selection_rule"] == "largest_supported_excess_fraction"
    assert contract["tie_break"] == "family_name"
    assert iteration["diagnostics_make_target"] == "cuda-opt145-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "cuda-opt145-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "QW38_OPT145_SECONDARY_FAMILY_RESULT=" in task_runner
    assert "QW38_OPT145_NATIVE_COUNTS=" in task_runner
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "--q4-decode" in runner
    assert QUALITY_NATIVE in runner
    assert "candidate_nll_not_measured" in runner
    assert "no_source_grounded_mechanism" in runner
    assert "opt110_not_re_ported" in runner
    assert "opt112_deferred_fusion_not_revived" in runner
    assert "decode_mixer_stalls_cannot_identify_rmsnorm" in runner
    assert QUARTZ_PROMPT_MMQ_KERNEL in runner
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["copied_organization"] == []
    validate_future_keep_policy("OPT-145", iteration)
    validate_opt_in_contract(contract)


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-145")
    preflight = iteration["workloads"]["preflight"]
    correctness = iteration["workloads"]["correctness"]
    screen = iteration["workloads"]["screen"]
    quality = iteration["workloads"]["quality"]
    performance = iteration["workloads"]["performance"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(correctness, "feedback")) == 16
    assert loop_product(workload_for_mode(screen, "feedback")) == 8
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(performance, "acceptance")) == 130
    described = describe_plan("OPT-145", "feedback", iteration, "preflight")
    assert "phase=preflight" in described
    assert "loop_product=1" in family_plan("feedback", "preflight")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "opt-058" in proof
    assert "excess fraction" in proof


def test_fraction_ranking_prefers_residual_over_raw_ms() -> None:
    gaps = _gaps()
    residual = residual_norm_quant_evidence(gaps)
    prompt = prompt_mmq_evidence(gaps)
    assert residual["supported_fraction"] == excess_fraction(7.21, 201.6)
    assert prompt["supported_fraction"] == excess_fraction(27.11, 1347.8)
    assert residual["supported_fraction"] > prompt["supported_fraction"]
    assert prompt["supported_excess_ms"] > residual["supported_excess_ms"]
    selection = select_secondary_family(residual, prompt)
    assert selection["selected"]["family"] == "residual_norm_quant"
    assert selection["unselected"]["family"] == "prompt_mmq"
    assert selection["ranked_raw_ms_rejected"] is True
    roles = target_guard_roles(selection["selected"])
    assert roles["target_workloads"] == ["d128"]
    assert roles["target_metric"] == "decode_only"
    assert roles["complete_request_target_guard"] is True
    assert "p4096" in {row["workload"] for row in roles["guards"]}


def test_tie_break_by_family_name() -> None:
    residual = {
        "family": "residual_norm_quant",
        "supported": True,
        "supported_fraction": 0.02,
        "supported_workload": "d128",
    }
    prompt = {
        "family": "prompt_mmq",
        "supported": True,
        "supported_fraction": 0.02,
        "supported_workload": "p4096",
    }
    selection = select_secondary_family(residual, prompt)
    assert selection["selected"]["family"] == "prompt_mmq"


def test_live_freeze_is_measured_no_opportunity() -> None:
    freeze = freeze_candidate()
    assert freeze["ok"] is True
    assert freeze["blockers"] == []
    assert freeze["selected_family"] == "residual_norm_quant"
    assert freeze["unselected_family"] == "prompt_mmq"
    assert freeze["positive_excess"] is True
    assert freeze["excess_fraction"] is not None
    assert float(freeze["excess_fraction"]) > float(
        (freeze["unselected"] or {})["supported_fraction"]
    )
    assert freeze["target"] == "d128"
    assert freeze["target_metric"] == "decode_only"
    assert freeze["quartz_kernel"] == QUARTZ_RESIDUAL_KERNEL
    assert freeze["source_observations"] is False
    assert freeze["sass_observations"] is False
    assert freeze["candidate"] is None
    assert freeze["no_opportunity"] is True
    assert freeze["verdict"] == "no_opportunity"
    assert freeze["nll_required"] is False
    assert freeze["arithmetic_changed"] is False
    assert freeze["opt110_not_re_ported"] is True
    assert freeze["opt112_deferred_fusion_not_revived"] is True
    assert freeze["opt131_chain_fusion_not_revived"] is True
    assert freeze["opt143_short_decode_not_kept"] is True
    assert freeze["opt144_prefill_attention_not_kept"] is True
    resolving = freeze["resolving_measurement"]
    assert resolving["kind"] == "fused_boundary_identity"
    assert resolving["decode_mixer_stalls_identify_rmsnorm"] is False
    fusion = fused_boundary_ok(
        family="residual_norm_quant",
        quartz_kernel=QUARTZ_RESIDUAL_KERNEL,
        llama_kernel="quantize_q8_1",
        llama_enclosing=["rms_norm_f32", "quantize_q8_1"],
    )
    assert fusion["ok"] is False
    reasons = " ".join(str(item) for item in freeze["reasons"])
    assert "throughput" in reasons or "no_source_grounded_mechanism" in reasons
    assert "decode_mixer_stalls" in reasons


def test_freeze_blocks_unmeasured_excess() -> None:
    freeze = freeze_candidate(
        gaps={
            "decode_selection": {"selected": []},
            "prefill_selection": {"selected": []},
        },
        identity={},
        counters={"kernels": []},
        comparisons=_comparisons(),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "blocked"
    assert "no_supported_positive_excess_fraction" in freeze["blockers"]


def test_freeze_forbids_shipped_and_rejected_repeats() -> None:
    assert forbidden_reason("opt110_report", "port_opt110") == "opt110_not_re_ported"
    assert (
        forbidden_reason("residual_norm_to_typed_q8", "opt112_norm_to_typed_q8")
        == "opt112_deferred_fusion_not_revived"
    )
    assert (
        forbidden_reason("mixer_residual_add_to_ffn_norm", "opt131_chain")
        == "opt131_chain_fusion_not_revived"
    )
    record = _counter_record(family="residual_norm_quant", mechanism=True)
    record["candidate"] = "residual_norm_to_typed_q8_rounded"
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity={},
        counters={"kernels": [record]},
        comparisons=_comparisons(),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "blocked"
    assert "opt112_deferred_fusion_not_revived" in freeze["blockers"]


def test_freeze_admits_source_grounded_candidate() -> None:
    record = _counter_record(family="residual_norm_quant", mechanism=True)
    admitted = admit_supported_mechanism(record)
    assert admitted["ok"] is True
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity={},
        counters={"kernels": [record]},
        comparisons=_comparisons(),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "candidate_frozen"
    assert freeze["candidate"] == f"{QUARTZ_RESIDUAL_KERNEL}_v2"
    assert freeze["no_opportunity"] is False
    assert freeze["nll_required"] is True
    assert freeze["arithmetic_changed"] is True
    assert freeze["selected_family"] == "residual_norm_quant"


def test_quality_skip_does_not_set_nll_not_measured(tmp_path: Path) -> None:
    freeze = freeze_candidate()
    payload = skip_payload(tmp_path, "acceptance", "quality", freeze)
    assert payload["ok"] is True
    assert payload["skipped"] is True
    assert payload["opt058_invoked"] is False
    assert payload["candidate_nll_measured"] is False
    assert payload["candidate_nll_not_measured"] is False
    assert payload["nll_required"] is False
    assert payload["skip_reason"] == "no_candidate_no_arithmetic_change"


def test_production_pins_retained() -> None:
    assert production_pins_parent() is True
    if FIXTURE.is_file():
        payload = _json(FIXTURE)
        if payload.get("verdict") in {"no_opportunity", "reject", "screened_out"}:
            assert payload["production_kept"] is False
            assert payload["shipping_q4_decode"] == SHIPPING_Q4


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-145"
    assert payload["parent"] == PARENT_STACK
    quality = payload.get("quality") or {}
    if payload.get("verdict") == "no_opportunity":
        assert payload["candidate"] is None
        assert payload["shipping_delta"] == 0
        assert quality.get("candidate_nll_not_measured") is not True
        assert quality.get("opt058_invoked") is False
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert quality.get("opt058_invoked") is True
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("candidate_nll_not_measured") is not True
    checked = validate_fixture(payload)
    assert checked["ok"] is True
    freeze_hash(load_contract())
    assert REPORT.is_file() or payload.get("mode") != "acceptance"
