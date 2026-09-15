"""Host tests for OPT-143 short-decode attention freeze and keep/reject wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt139_counter_identity import admit_supported_mechanism
from tools.opt143_short_attention import (
    CONTRACT,
    CROSSOVER,
    D2048_KERNEL,
    D128_KERNEL,
    FIXTURE,
    ITERATION,
    MMA_THRESHOLD,
    PARENT_STACK,
    PHASES,
    QUALITY_NATIVE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    SHIPPING_PARENT,
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
PIN = ROOT / "cuda/attention_decode_path.cuh"
RUNNER = ROOT / "tools/opt143_short_attention.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt143_short_attention_provenance.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gaps(*, positive: bool = True) -> dict[str, Any]:
    return {
        "decode_selection": {
            "selected": [
                {
                    "family": "attn_core",
                    "d128_middle": {
                        "n": 3,
                        "mean_ms": 4.3,
                        "one_sided_low": 4.2,
                        "significant_positive_excess": True,
                    },
                    "d2048_middle": {
                        "n": 3,
                        "mean_ms": 47.51,
                        "one_sided_low": 47.38,
                        "significant_positive_excess": positive,
                    },
                }
            ]
        }
    }


def _identity() -> dict[str, Any]:
    return {
        "decode_shapes": {
            "d128": {
                "path": "warp_query",
                "expected_kernel": D128_KERNEL,
                "identity": {"ok": True, "kernel": D128_KERNEL},
            },
            "d2048": {
                "path": "vec128_online",
                "expected_kernel": D2048_KERNEL,
                "identity": {"ok": True, "kernel": D2048_KERNEL},
                "warp_query_is_not_d2048_evidence": {
                    "reason": "kernel_selector_mismatch"
                },
            },
        }
    }


def _counter_record(*, mechanism: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "engine": "quartz",
        "family": "attn_core",
        "phase": "decode",
        "workload": "D2048",
        "expected_kernel": D2048_KERNEL,
        "target_kernel": D2048_KERNEL,
        "replay_family": "decode-attention",
        "dram_read_bytes": 8.85,
        "dram_write_bytes": 0.0,
        "dram_throughput": 2.33,
        "sm_throughput": 5.43,
        "achieved_occupancy": 15.62,
        "l2_traffic": 1932797.0,
        "identity": {
            "ok": True,
            "phase": "decode",
            "engine": "quartz",
            "workload": "D2048",
            "kernel": D2048_KERNEL,
            "expected_kernel": D2048_KERNEL,
            "replay_family": "decode-attention",
            "replay_boundary": {"ok": True, "replay_family": "decode-attention"},
            "mismatches": [],
        },
    }
    if mechanism:
        record["source_observations"] = [
            "vec128_online_decode_attention fused GQA KV reuse in source"
        ]
        record["sass_observations"] = ["LDG.128 reduction vs replay"]
        record["proposed_mechanism"] = "gqa_kv_reuse_in_vec128_online"
        record["candidate"] = "vec128_online_gqa_kv_reuse_v1"
    return record


def test_contracts_makefile_and_registration() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-143")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    pin = PIN.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert contract["task"] == "OPT-143"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["crossover_threshold"] == CROSSOVER
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["target_workloads"] == ["d2048"]
    assert "d8192" in contract["guard_workloads"]
    assert "d32768" in contract["guard_workloads"]
    assert "p4096" in contract["guard_workloads"]
    assert iteration["diagnostics_make_target"] == "cuda-opt143-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "cuda-opt143-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "QW38_OPT143_SHORT_ATTENTION_RESULT=" in task_runner
    assert "QW38_OPT143_NATIVE_COUNTS=" in task_runner
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "--q4-decode" in runner
    assert QUALITY_NATIVE in runner
    assert "candidate_nll_not_measured" in runner
    assert "no_source_grounded_mechanism" in runner
    assert "opt130_not_revived" in runner
    assert "kOpt137MmaThreshold = 8192" in pin
    assert "kSelectedOpt137DenseMma = true" in pin
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["copied_organization"] == []
    validate_future_keep_policy("OPT-143", iteration)
    validate_opt_in_contract(contract)


def test_target_guard_v2_roles() -> None:
    contract = load_contract()
    roles = {row["workload"]: "target" for row in contract["targets"]}
    roles.update({row["workload"]: "guard" for row in contract["guards"]})
    assert roles["d2048"] == "target"
    assert roles["d128"] == "guard"
    assert roles["d8192"] == "guard"
    assert roles["d32768"] == "guard"
    assert roles["p4096"] == "guard"
    assert "d2048" not in {row["workload"] for row in contract["guards"]}
    freeze_hash(contract)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt143-short-attention")


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-143")
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
    described = describe_plan("OPT-143", "feedback", iteration, "preflight")
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
    assert float(freeze["excess_ms"]) > 40.0
    assert freeze["production_path_d2048"] == "vec128_online"
    assert D2048_KERNEL in str(freeze["production_kernel_d2048"])
    assert freeze["warp_query_rejected_as_d2048"] is True
    assert freeze["source_observations"] is False
    assert freeze["sass_observations"] is False
    assert freeze["supported_mechanism"] is None
    assert freeze["candidate"] is None
    assert freeze["no_opportunity"] is True
    assert freeze["verdict"] == "no_opportunity"
    assert freeze["nll_required"] is False
    assert freeze["arithmetic_changed"] is False
    assert freeze["opt137_mma_threshold"] == 8192
    assert freeze["opt130_not_revived"] is True
    reasons = " ".join(str(item) for item in freeze["reasons"])
    assert "throughput" in reasons or "no_source_grounded_mechanism" in reasons


def test_freeze_blocks_unmeasured_excess() -> None:
    freeze = freeze_candidate(
        gaps={"decode_selection": {"selected": []}},
        identity=_identity(),
        counters={"kernels": [_counter_record(mechanism=False)]},
        comparisons={"decode.attn_core.d2048": {"comparison": {"comparable": True}}},
        llama_counters={"kernels": []},
        next_experiments={"decode": {"expected_benefit": {"label": "unknown"}}},
    )
    assert freeze["verdict"] == "blocked"
    assert "opt138_d2048_attn_core_unmeasured" in freeze["blockers"]


def test_freeze_rejects_warp_query_as_d2048() -> None:
    identity = _identity()
    identity["decode_shapes"]["d2048"]["path"] = "warp_query"
    identity["decode_shapes"]["d2048"]["identity"]["kernel"] = D128_KERNEL
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=identity,
        counters={"kernels": [_counter_record(mechanism=False)]},
        comparisons={"decode.attn_core.d2048": {"comparison": {"comparable": True}}},
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "blocked"
    assert "d2048_production_path_not_vec128_online" in freeze["blockers"]


def test_freeze_forbids_opt130_and_mma_lowering() -> None:
    assert forbidden_reason("opt130_nparts8", "occupancy") == "opt130_not_revived"
    assert (
        forbidden_reason("mma_at_d2048", "lower_opt137_mma_threshold")
        == "opt137_mma_threshold_must_remain_8192"
    )
    record = _counter_record(mechanism=True)
    record["candidate"] = "hybrid_crossover@1024_vec128_open_nparts8"
    record["proposed_mechanism"] = "occupancy_partition_or_gqa_kv_reuse"
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=_identity(),
        counters={"kernels": [record]},
        comparisons={"decode.attn_core.d2048": {"comparison": {"comparable": True}}},
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "blocked"
    assert "opt130_not_revived" in freeze["blockers"]


def test_freeze_admits_source_grounded_candidate() -> None:
    record = _counter_record(mechanism=True)
    admitted = admit_supported_mechanism(record)
    assert admitted["ok"] is True
    freeze = freeze_candidate(
        gaps=_gaps(),
        identity=_identity(),
        counters={"kernels": [record]},
        comparisons={"decode.attn_core.d2048": {"comparison": {"comparable": True}}},
        llama_counters={"kernels": []},
        next_experiments={},
    )
    assert freeze["verdict"] == "candidate_frozen"
    assert freeze["candidate"] == "vec128_online_gqa_kv_reuse_v1"
    assert freeze["no_opportunity"] is False
    assert freeze["nll_required"] is True
    assert freeze["arithmetic_changed"] is True


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
    pin = PIN.read_text(encoding="utf-8")
    assert "kSelectedOpt137DenseMma = true" in pin
    assert "kOpt137MmaThreshold = 8192" in pin
    assert "kSelectedDecodeAttentionCrossoverThreshold = 1024" in pin
    assert "kSelectedVec128NParts = 16" in pin
    if FIXTURE.is_file():
        payload = _json(FIXTURE)
        if payload.get("verdict") in {"no_opportunity", "reject", "screened_out"}:
            assert payload["production_kept"] is False
            assert payload["shipping_decode_attention"] == SHIPPING_PARENT


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-143"
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
