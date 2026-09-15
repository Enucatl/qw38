"""Host tests for OPT-144 prefill attention freeze and keep/reject wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt139_counter_identity import admit_supported_mechanism
from tools.opt144_prefill_attention import (
    CONTRACT,
    CROSSOVER,
    FIXTURE,
    ITERATION,
    MMA_THRESHOLD,
    PARENT_STACK,
    PHASES,
    PREFILL_KERNEL,
    QUALITY_NATIVE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    SHIPPING_PROMPT,
    PAIR_COUNT,
    WARMUPS,
    family_plan,
    forbidden_reason,
    freeze_candidate,
    load_contract,
    production_pins_parent,
    skip_payload,
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
DECODE_PIN = ROOT / "cuda/attention_decode_path.cuh"
PROMPT_PIN = ROOT / "cuda/fattn_mma_f16.cuh"
RUNNER = ROOT / "tools/opt144_prefill_attention.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt144_prefill_attention_provenance.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gaps(*, positive: bool = True) -> dict[str, Any]:
    return {
        "prefill_selection": {
            "selected": [
                {
                    "family": "attn_core",
                    "score_ms": 186.58,
                    "stats": {
                        "n": 3,
                        "mean_ms": 186.58,
                        "one_sided_low": 186.10,
                        "significant_positive_excess": positive,
                    },
                    "source": "p4096_complete",
                }
            ]
        }
    }


def _identity(
    *, replay: str = "prompt-attention", substitution: bool = False
) -> dict[str, Any]:
    return {
        "replay_family": replay,
        "counter_kernel": PREFILL_KERNEL,
        "expected_kernel": PREFILL_KERNEL,
        "path": "opt111_base",
        "decode_attention_substitution": substitution,
        "identity": {
            "ok": True,
            "kernel": PREFILL_KERNEL,
            "expected_kernel": PREFILL_KERNEL,
            "replay_family": replay,
        },
        "boundary": {"ok": True, "replay_family": replay},
    }


def _counter_record(
    *, mechanism: bool, replay: str = "prompt-attention"
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "engine": "quartz",
        "family": "attn_core",
        "phase": "prefill",
        "workload": "P4096",
        "expected_kernel": PREFILL_KERNEL,
        "target_kernel": PREFILL_KERNEL,
        "replay_family": replay,
        "dram_read_bytes": 252.12,
        "dram_write_bytes": 168.42,
        "dram_throughput": 1.22,
        "sm_throughput": 22.4,
        "achieved_occupancy": 8.33,
        "tensor_activity": 810024960.0,
        "l2_traffic": 216065551.0,
        "identity": {
            "ok": True,
            "phase": "prefill",
            "engine": "quartz",
            "workload": "P4096",
            "kernel": PREFILL_KERNEL,
            "expected_kernel": PREFILL_KERNEL,
            "replay_family": replay,
            "replay_boundary": {"ok": True, "replay_family": replay},
            "mismatches": [],
        },
    }
    if mechanism:
        record["source_observations"] = [
            "fattn_mma_pipeline_opt111_base named SASS bank conflict"
        ]
        record["sass_observations"] = ["LDSM.16 vs cp.async reduction"]
        record["proposed_mechanism"] = "named_sass_kv_load_change"
        record["candidate"] = "fattn_mma_pipeline_kv_load_v2"
    return record


def _comparisons(*, comparable: bool = False) -> dict[str, Any]:
    return {
        "prefill.attn_core.p4096": {
            "fusion": {
                "ok": comparable,
                "reason": None if comparable else "fused_boundary_mismatch",
                "details": []
                if comparable
                else ["tile_shape_mismatch_quartz_16x2_vs_llama_8x8"],
            },
            "comparison": {
                "comparable": comparable,
                "reason": None if comparable else "fused_boundary_mismatch",
                "compared_slots": [],
            },
        }
    }


def test_contracts_makefile_and_registration() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-144")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert contract["task"] == "OPT-144"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["parent_prompt_attention"] == "opt111_base"
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["crossover_threshold"] == CROSSOVER
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["target_workloads"] == ["p4096"]
    assert "d128" in contract["guard_workloads"]
    assert "d2048" in contract["guard_workloads"]
    assert "d8192" in contract["guard_workloads"]
    assert "d32768" in contract["guard_workloads"]
    assert iteration["diagnostics_make_target"] == "cuda-opt144-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "cuda-opt144-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "QW38_OPT144_PREFILL_ATTENTION_RESULT=" in task_runner
    assert "QW38_OPT144_NATIVE_COUNTS=" in task_runner
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "--q4-decode" in runner
    assert QUALITY_NATIVE in runner
    assert "candidate_nll_not_measured" in runner
    assert "no_source_grounded_mechanism" in runner
    assert "opt111_not_re_ported" in runner
    assert "decode_ncu_rejected_as_prefill" in runner
    assert "kOpt137MmaThreshold = 8192" in decode_pin
    assert 'kSelectedAttentionPipelinePath[] = "opt111_base"' in prompt_pin
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["copied_organization"] == []
    validate_future_keep_policy("OPT-144", iteration)
    validate_opt_in_contract(contract)


def test_target_guard_v2_roles() -> None:
    contract = load_contract()
    roles = {row["workload"]: "target" for row in contract["targets"]}
    roles.update({row["workload"]: "guard" for row in contract["guards"]})
    assert roles["p4096"] == "target"
    assert roles["d128"] == "guard"
    assert roles["d2048"] == "guard"
    assert roles["d8192"] == "guard"
    assert roles["d32768"] == "guard"
    assert "p4096" not in {row["workload"] for row in contract["guards"]}
    freeze_hash(contract)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt144-prefill-attention")


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-144")
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
    described = describe_plan("OPT-144", "feedback", iteration, "preflight")
    assert "phase=preflight" in described
    assert "loop_product=1" in family_plan("feedback", "preflight")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "opt-058" in proof
    assert "target_guard_v2" in proof or "target l>1.00" in proof


def test_live_freeze_is_measured_no_opportunity() -> None:
    freeze = freeze_candidate()
    assert freeze["ok"] is True
    assert freeze["blockers"] == []
    assert freeze["positive_excess"] is True
    assert freeze["excess_ms"] is not None
    assert float(freeze["excess_ms"]) > 180.0
    assert freeze["production_kernel_p4096"] == PREFILL_KERNEL
    assert freeze["replay_family"] == "prompt-attention"
    assert freeze["decode_attention_substitution"] is False
    assert freeze["decode_ncu_rejected_as_prefill"] is True
    assert freeze["opt111_not_re_ported"] is True
    assert freeze["source_observations"] is False
    assert freeze["sass_observations"] is False
    assert freeze["supported_mechanism"] is None
    assert freeze["candidate"] is None
    assert freeze["no_opportunity"] is True
    assert freeze["verdict"] == "no_opportunity"
    assert freeze["nll_required"] is False
    assert freeze["arithmetic_changed"] is False
    assert freeze["opt137_mma_threshold"] == 8192
    assert freeze["opt143_short_decode_not_kept"] is True
    assert freeze["tensor_activity_nonzero"] is True
    assert freeze["comparable_p4096"] is False
    assert freeze["fusion_reason"] == "fused_boundary_mismatch"
    reasons = " ".join(str(item) for item in freeze["reasons"])
    assert "throughput" in reasons or "no_source_grounded_mechanism" in reasons
    assert "16x2" in reasons or "fused_boundary" in reasons
    typed = freeze["typed_slots"]
    assert typed["tensor_activity"] not in (None, 0, 0.0)


def test_freeze_blocks_unmeasured_excess() -> None:
    freeze = freeze_candidate(
        gaps={"prefill_selection": {"selected": []}},
        identity=_identity(),
        counters={"kernels": [_counter_record(mechanism=False)]},
        comparisons=_comparisons(),
        llama_counters={"kernels": []},
        next_experiments={"prefill": {"expected_benefit": {"label": "unknown"}}},
    )
    assert freeze["verdict"] == "blocked"
    assert "opt138_p4096_attn_core_unmeasured" in freeze["blockers"]


def test_freeze_rejects_decode_ncu_as_prefill() -> None:
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=_identity(replay="decode-attention", substitution=True),
        counters={
            "kernels": [_counter_record(mechanism=False, replay="decode-attention")]
        },
        comparisons=_comparisons(),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "blocked"
    assert "decode_ncu_rejected_as_prefill_evidence" in freeze["blockers"]


def test_freeze_forbids_opt111_report_and_mma_lowering() -> None:
    assert forbidden_reason("opt111_xor", "port_opt111") == "opt111_not_re_ported"
    assert (
        forbidden_reason("decode_ncu_as_prefill", "decode-attention")
        == "decode_ncu_rejected_as_prefill_evidence"
    )
    record = _counter_record(mechanism=True)
    record["candidate"] = "opt111_base"
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=_identity(),
        counters={"kernels": [record]},
        comparisons=_comparisons(comparable=True),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "blocked"
    assert "opt111_not_re_ported" in freeze["blockers"]


def test_freeze_admits_source_grounded_candidate() -> None:
    record = _counter_record(mechanism=True)
    admitted = admit_supported_mechanism(record)
    assert admitted["ok"] is True
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=_identity(),
        counters={"kernels": [record]},
        comparisons=_comparisons(comparable=True),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "candidate_frozen"
    assert freeze["candidate"] == "fattn_mma_pipeline_kv_load_v2"
    assert freeze["no_opportunity"] is False
    assert freeze["nll_required"] is True
    assert freeze["arithmetic_changed"] is True


def test_freeze_admits_hypothesis_without_mechanism_for_screen() -> None:
    record = _counter_record(mechanism=False)
    record["candidate"] = "fattn_mma_pipeline_8x8_screen_v1"
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=_identity(),
        counters={"kernels": [record]},
        comparisons=_comparisons(),
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "candidate_frozen"
    assert freeze["candidate"] == "fattn_mma_pipeline_8x8_screen_v1"
    assert freeze["screen_only"] is True
    assert freeze["supported_mechanism"] is None


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
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    assert "kSelectedOpt137DenseMma = true" in decode_pin
    assert "kOpt137MmaThreshold = 8192" in decode_pin
    assert 'kSelectedAttentionPipelinePath[] = "opt111_base"' in prompt_pin
    if FIXTURE.is_file():
        payload = _json(FIXTURE)
        if payload.get("verdict") in {"no_opportunity", "reject", "screened_out"}:
            assert payload["production_kept"] is False
            assert payload["shipping_prompt_attention"] == SHIPPING_PROMPT


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-144"
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
    assert REPORT.is_file() or payload.get("mode") != "acceptance"
