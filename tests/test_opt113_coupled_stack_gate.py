"""Host tests for the OPT-113 coupled-stack sitting gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt080_batch_gate import (
    IMAGE,
    LLAMA_IMAGE,
    MARGIN,
    plus5_gap_ms,
    parity_gap_ms,
)
from tools.opt113_coupled_stack_gate import (
    CONTRACT,
    FAMILIES,
    ITERATION,
    OPT106_CONTRACT,
    OPT106_FIXTURE,
    OPT106_REPORT,
    audit_dependencies,
    family_kernel_parity,
    frozen_combined_config,
    llama_docker_command,
    load_contract,
    opt106_artifacts_unmodified,
    post106_control_paths,
    source_paths,
    three_outcomes,
    validate_batch_result,
    validate_report_agrees,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract as load_iteration,
    loop_product,
    validate_future_keep_policy,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
RUNNER = ROOT / "tools/opt113_coupled_stack_gate.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_fields() -> dict[str, Any]:
    opt106 = _json(OPT106_FIXTURE)
    vs_base = copy.deepcopy(opt106["quartz_vs_baseline_quality_delta"])
    vs_llama = copy.deepcopy(opt106["quartz_vs_llama_quality_delta"])
    recurrence = copy.deepcopy(opt106["recurrence_state_status"])
    return {
        "model_quality_pass": True,
        "absolute_quality_status": "fail",
        "quartz_baseline_regression_status": "pass",
        "quartz_vs_baseline_quality_delta": vs_base,
        "quartz_vs_llama_quality_delta": vs_llama,
        "recurrence_state_status": recurrence,
        "single_boolean": None,
        "candidate_nll_measured": True,
    }


def _synthetic_measured() -> dict[str, Any]:
    freeze = frozen_combined_config()
    opt106 = _json(OPT106_FIXTURE)
    quality = _quality_fields()
    from tools.opt080_batch_gate import compute_gate

    p = copy.deepcopy(opt106["p"])
    d128 = copy.deepcopy(opt106["d128"])
    d2048 = copy.deepcopy(opt106["d2048"])
    p["quartz_control"] = copy.deepcopy(p["quartz"])
    d128["quartz_control"] = copy.deepcopy(d128["quartz"])
    d2048["quartz_control"] = copy.deepcopy(d2048["quartz"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-113",
        "status": "measured",
        "measurement_utc": "2026-09-13T06:00:00Z",
        "device": opt106["device"],
        "compute_capability": "12.0",
        "power_limit_w": 400.0,
        "telemetry": copy.deepcopy(opt106["telemetry"]),
        "llama_revision": opt106["llama_revision"],
        "gguf_sha256": opt106["gguf_sha256"],
        "source_revision": "synthetic",
        "source_state": "dirty",
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        "post106_control": freeze["post106_control"],
        "post113_selected": freeze["post113_selected"],
        "combined_production_paths": freeze["combined_production_paths"],
        "selected_equals_control": False,
        "opt112_omitted": True,
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "deferred": freeze["deferred"],
        "kernel_parity_pass": freeze["kernel_parity_pass"],
        "matched_family_attribution": freeze["matched_family_attribution"],
        **quality,
        "performance_pass": False,
        "production_kept": True,
        "release_eligible": True,
        "p": p,
        "d128": d128,
        "d2048": d2048,
        "opt016": copy.deepcopy(opt106["opt016"]),
        "quality": copy.deepcopy(opt106["quality"]),
        "gaps": copy.deepcopy(opt106["gaps"]),
        "state_isolation": copy.deepcopy(opt106["state_isolation"]),
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "preflight_is_release_evidence": False,
        "opt074_coverage_unadmitted_blocker": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": " ".join(load_contract()["proof_limit"]),
        "report_path": "evidence/optimization/opt113-coupled-stack-gate/REPORT.md",
    }
    mem = _json(ROOT / "fixtures/cuda_memory_fit_post_graph.json")
    result["state_isolation"]["memory_fit"] = {
        "ok": True,
        "post_graph_admitted": True,
        "reserve_ok": True,
        "returncode": 0,
        "fields": {
            "memory_fit": "post_graph",
            "capacity": "131072",
            "explicit_bytes": str(mem["owners"]["explicit_quartz_bytes"]),
            "measured_delta": str(mem["measured_quartz_delta_bytes"]),
            "free_bytes": str(mem["free_after_graph_creation_bytes"]),
            "reserve_required": str(mem["required_reserve_bytes"]),
            "total_bytes": str(mem["total_device_bytes"]),
            "arithmetic": "true",
            "passed": "true",
        },
        "reconciled_against_allocation_inventory": True,
    }
    result["quality"]["candidate_nll_measured"] = True
    result["recurrence_state_status"]["memory_fit"] = True
    result["quality"]["single_boolean"] = None
    result["p"]["quartz"]["attribution"] = None
    result["p"]["quartz"]["instrumented"] = False
    result["d128"]["quartz"]["attribution"] = None
    result["d2048"]["quartz"]["attribution"] = None
    result["opt016"]["gate_passed"] = False
    result["gate"] = compute_gate(result)
    result["outcomes"] = three_outcomes(result)
    result["performance_pass"] = False
    return result


def test_contracts_iteration_plan_and_freeze() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-113")
    freeze = frozen_combined_config()
    paths = source_paths()
    control = post106_control_paths()
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-113"
    assert contract["strict_ppl_ratio_max"] == 1.01
    assert contract["require_candidate_nll"] is True
    assert contract["selected_equals_control"] is False
    assert contract["reuse_historical_llama"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt113-diagnostics"
    assert "quality" in iteration["modes"]["release"]["workloads"]
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt113-diagnostics" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "qw38-cuda-opt113-coupled-stack-gate-test" in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    runner = RUNNER.read_text(encoding="utf-8")
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "candidate_nll_measured" in runner
    assert paths["q4_decode"] == "llama_q4k_mmvq"
    assert paths["attention_pipeline"] == "opt111_base"
    assert paths["decode_attention"] == "hybrid_crossover"
    assert paths["decode_attention_crossover_threshold"] == 1024
    assert paths["decode_attention_vec128"] == "warp_query"
    assert paths["gdn_decode"] == "sequential"
    assert control["q4_decode"] == "integer_q8_late"
    assert control["attention_pipeline"] == "kv_once"
    assert control["decode_attention"] == "warp_query"
    assert freeze["selected_equals_control"] is False
    assert freeze["opt112_omitted"] is True
    assert (
        freeze["post113_selected"]["q4_decode"]
        != freeze["post106_control"]["q4_decode"]
    )
    plan = describe_plan("OPT-113", "acceptance", iteration, "quality")
    assert "phase=quality" in plan
    assert loop_product(iteration["workloads"]["quality"]) >= 12


def test_opt106_artifacts_remain_unmodified() -> None:
    hashes = opt106_artifacts_unmodified()
    contract = load_contract()
    assert hashes["report"] == contract["opt106_historical_hashes"]["report"]
    assert hashlib.sha256(OPT106_REPORT.read_bytes()).hexdigest() == hashes["report"]
    assert hashlib.sha256(OPT106_FIXTURE.read_bytes()).hexdigest() == hashes["fixture"]
    assert (
        hashlib.sha256(OPT106_CONTRACT.read_bytes()).hexdigest() == hashes["contract"]
    )
    opt106 = _json(OPT106_FIXTURE)
    assert opt106["task"] == "OPT-106"


def test_dependency_audit_keep_reject_defer() -> None:
    loaded = audit_dependencies()
    assert set(loaded) == {
        "OPT-107",
        "OPT-108",
        "OPT-109",
        "OPT-110",
        "OPT-111",
        "OPT-112",
    }
    assert loaded["OPT-107"]["production_kept"] is True
    assert loaded["OPT-108"]["production_kept"] is False
    assert loaded["OPT-109"]["production_kept"] is False
    assert loaded["OPT-110"]["production_kept"] is True
    assert loaded["OPT-111"]["production_kept"] is True
    assert loaded["OPT-112"]["screen"]["verdict"] == "deferred_below_trigger"
    parity = family_kernel_parity(loaded)
    for name in FAMILIES:
        assert parity[name] is True
    assert parity["all"] is True


def test_rejects_quality_reduced_to_one_boolean() -> None:
    fake = _synthetic_measured()
    fake["quality"] = True
    with pytest.raises(AssertionError, match="quality reduced to one boolean"):
        validate_batch_result(fake)


def test_rejects_opt074_unadmitted_as_blocker() -> None:
    iteration = load_iteration("OPT-113")
    validate_future_keep_policy("OPT-113", iteration)
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-113",
            {
                "task": "OPT-113",
                "opt074_family_admission_required": True,
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_parity_and_plus5_millisecond_arithmetic() -> None:
    quartz = 3046.23218
    llama = 3170.927662
    parity = parity_gap_ms(quartz, llama, 4096)
    plus5 = plus5_gap_ms(quartz, llama, 4096)
    tq = 4096 * 1000.0 / quartz
    tl = 4096 * 1000.0 / llama
    assert parity == pytest.approx(tq - tl)
    assert plus5 == pytest.approx(tq - tl / MARGIN)
    assert parity != pytest.approx(plus5)


def test_does_not_relabel_opt056_or_opt016() -> None:
    fake = _synthetic_measured()
    assert fake["gate"]["passed"] is False
    assert fake["outcomes"]["opt056_plus5"]["pass"] is False
    fake["opt056_gate_passed"] = True
    with pytest.raises(AssertionError, match="OPT-056|historical|gate.passed"):
        validate_batch_result(fake)


def test_rejects_selector_leakage() -> None:
    leaked = _synthetic_measured()
    leaked["combined_production_paths"] = dict(leaked["combined_production_paths"])
    leaked["combined_production_paths"]["decode_attention_vec128"] = "vec128_online"
    with pytest.raises(AssertionError, match="selector|freeze|combined|vec128|108"):
        validate_batch_result(leaked)
    gdn = _synthetic_measured()
    gdn["combined_production_paths"] = dict(gdn["combined_production_paths"])
    gdn["combined_production_paths"]["gdn_decode"] = "persistent_transposed"
    with pytest.raises(AssertionError, match="selector|freeze|combined|109|sequential"):
        validate_batch_result(gdn)


def test_three_outcomes_reads_quality_blob_when_top_level_missing() -> None:
    honest = _synthetic_measured()
    honest.pop("model_quality_pass")
    outcomes = three_outcomes(honest)
    assert outcomes["internal_improvement_with_quality"]["quality_pass"] is True


def test_three_outcomes_are_independent() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    outcomes = honest["outcomes"]
    assert "internal_improvement_with_quality" in outcomes
    assert "llama_parity" in outcomes
    assert "opt056_plus5" in outcomes
    assert outcomes["internal_improvement_with_quality"]["pass"] is False
    assert outcomes["llama_parity"]["pass"] is False
    assert outcomes["opt056_plus5"]["pass"] is False
    assert honest["model_quality_pass"] is True
    assert honest["performance_pass"] is False
    assert honest["selected_equals_control"] is False
    assert outcomes["internal_improvement_with_quality"]["p_delta_vs_llama_tok_s"] != 0


def test_measured_fixture_internal_improvement_passes_after_mem_reconcile() -> None:
    fixture = _json(ROOT / "fixtures/opt113_coupled_stack_gate.json")
    validate_batch_result(fixture)
    assert fixture["state_isolation"]["memory_fit"]["ok"] is True
    assert fixture["state_isolation"]["memory_fit"]["fields"]["arithmetic"] == "true"
    assert fixture["outcomes"]["internal_improvement_with_quality"]["pass"] is True
    assert fixture["outcomes"]["llama_parity"]["pass"] is False
    assert fixture["outcomes"]["opt056_plus5"]["pass"] is False


def test_memory_fit_failure_is_honest_not_lost_state() -> None:
    honest = _synthetic_measured()
    honest["state_isolation"]["memory_fit"]["ok"] = False
    honest["outcomes"] = three_outcomes(honest)
    assert honest["outcomes"]["internal_improvement_with_quality"]["pass"] is False
    validate_batch_result(honest)


def test_synthetic_measured_validates() -> None:
    honest = _synthetic_measured()
    validate_batch_result(honest)
    validate_report_agrees(
        honest,
        "OPT-106 remains historical\n`gate.passed` is False\n",
    )


def test_native_and_oracles_carry_selector_flags() -> None:
    native = (ROOT / "cuda/opt113_coupled_stack_gate_test.cu").read_text(
        encoding="utf-8"
    )
    helper = (ROOT / "cuda/opt113_sitting_selectors.cuh").read_text(encoding="utf-8")
    decode = (ROOT / "cuda/decode_oracle_test.cu").read_text(encoding="utf-8")
    prefill = (ROOT / "cuda/prefill_4k_oracle_test.cu").read_text(encoding="utf-8")
    two_k = (ROOT / "cuda/prefill_2k_parity_test.cu").read_text(encoding="utf-8")
    assert "opt113_selected_pins_ok" in helper
    assert "kOpt113SelectedQ4" in helper
    assert "QW38_OPT113_COUPLED_STACK_GATE_RESULT=" in native
    assert "opt113_sitting_selectors.cuh" in decode
    assert "opt113_sitting_selectors.cuh" in prefill
    assert "opt113_sitting_selectors.cuh" in two_k
    assert "--q4-decode" in helper
    assert "--attention-pipeline" in helper
    assert "--decode-attention-crossover-threshold" in helper
    assert "apply_sitting_selector_flags" in decode
    assert "apply_sitting_selector_flags" in prefill
    assert "apply_sitting_selector_flags" in two_k


def test_llama_authority_image_falls_back_to_cuda_runtime() -> None:
    rewritten = llama_docker_command(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            str(LLAMA_IMAGE),
            "bash",
            "-lc",
            "llama-bench",
        ]
    )
    assert str(IMAGE) in rewritten
    assert str(LLAMA_IMAGE) not in rewritten
    assert any(str(part).startswith("LD_LIBRARY_PATH=") for part in rewritten)
