"""Host tests for OPT-089 late_w4 strict admission. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt082_kernel_parity import catalog
from tools.opt089_q4_promotion import (
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    OPT061_EVIDENCE,
    OPT082_FIXTURE,
    OPT082_REPORT,
    OPT084_FIXTURE,
    OPT084_REPORT,
    PARITY_ONLY_IDENTS,
    Q4_PERFORMANCE_IDENTS,
    REPORT,
    VERDICT_KEYS,
    AdmissionError,
    assert_opt089_write_path,
    authenticate_anchor,
    authenticate_opt084_freeze,
    authenticate_opt088_control,
    candidate_selectors,
    config_by_id,
    decide_independent_verdicts,
    dispatch_matches,
    empty_verdict_row,
    evaluate_parity,
    evaluate_quality,
    expanded_catalog,
    expected_dispatch,
    extra_catalog,
    family_plan,
    historical_reports_intact,
    ident_parity_from_source,
    load_json,
    pair_stats,
    parity_record,
    parse_prefill_wall_ms,
    replay_command,
    _parse_quality_cases,
)
from tools.quality.quality_mode import (
    OPT088_CONTROL_SELECTORS,
    SHIPPING_SELECTORS,
    apply_quality_mode,
    build_quality_config,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    validate_performance_admission,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CUDA_PARITY = ROOT / "cuda/opt082_kernel_parity_test.cu"
CUDA_QUALITY = ROOT / "cuda/opt058_quality_baseline_test.cu"
CUDA_REPLAY = ROOT / "cuda/optimization_component_replay.cu"
CUDA_PROBE = ROOT / "cuda/optimization_engine_probe.cu"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PATH = ROOT / "cuda/ffn_decode_path.cuh"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_pass_gpu(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "candidate": row["candidate"],
            "pass": True,
            "fallback": False,
            "gpu_pass": True,
        }
        for row in cases
    ]


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-089")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-089"
    assert contract["claims_throughput"] is True
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["opt074_family_admission_required"] is False
    assert contract["warps_per_row"] == 4
    assert contract["half_scale_forbidden"] is True
    assert contract["at_most_one_q4_path"] is True
    assert contract["integer_q8_paired_parity_only"] is True
    assert contract["q8_layout_fixed"] == "r1_w4"
    assert contract["quality_contract_id"] == "opt089_strict"
    assert contract["ppl_ratio_max"] == 1.01
    assert contract["recurrence_incremental_nll_max"] == 0.02
    assert contract["require_candidate_nll"] is True
    assert contract["allow_incomplete_quality"] is False
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        "packed_paired_staged",
        "late_w4",
    ]
    assert "integer_q8_paired" not in [row["id"] for row in CONFIGS]
    assert iteration["task"] == "OPT-089"
    assert iteration["diagnostics_make_target"] == "cuda-opt089-diagnostics"
    assert iteration["opt074_family_admission_required"] is False
    assert iteration["performance_admission"]["enabled"] is True
    assert iteration["case_ids"] == [
        "parity",
        "q4",
        "quality-alarm",
        "quality",
        "d128",
        "d2048",
        "prefill-guard",
    ]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["modes"]["release"]["release_host_command"] == [
        "uv",
        "run",
        "python",
        "tools/opt089_q4_promotion.py",
        "--phase",
        "quality",
        "--mode",
        "release",
        "--run-dir",
        "{run_dir}",
    ]
    assert "cuda-opt089-diagnostics" in makefile
    assert "qw38-cuda-opt082-kernel-parity-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    shipping = load_json(FIXTURE)
    shipping_q4 = str(shipping.get("shipping_q4_decode") or "packed")
    shipping_ffn = str(shipping.get("shipping_ffn_decode") or "paired_staged")
    assert "integer_q8_late" in q4
    assert f'kSelectedQ4DecodePath[] = "{shipping_q4}"' in q4
    assert f'kSelectedFfnDecodePath[] = "{shipping_ffn}"' in ffn


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-089")
    parity = iteration["workloads"]["parity"]
    q4 = iteration["workloads"]["q4"]
    alarm = iteration["workloads"]["quality-alarm"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 198
    assert loop_product(workload_for_mode(q4, "feedback")) == 8
    assert loop_product(workload_for_mode(q4, "acceptance")) == 26
    assert loop_product(workload_for_mode(alarm, "feedback")) == 64
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert family_plan("parity", "feedback")["cases"] == 198
    plan = family_plan("q4", "feedback")
    assert plan["candidates"] == 2
    assert plan["warmups"] == 1
    assert plan["samples"] == 3
    assert plan["engine_pairs"] == 1
    accept = family_plan("q4", "acceptance")
    assert accept["candidates"] == 2
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-089", "feedback", iteration, "parity")
    assert "phase=parity" in described
    assert "historical_oracles=none" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt074_coverage_unadmitted is not a blocker" in proof
    assert "integer_q8_paired is parity-only" in proof
    assert "r1_w4" in proof


def test_two_performance_configs_integer_q8_not_swept() -> None:
    assert [row["id"] for row in CONFIGS] == ["packed_paired_staged", "late_w4"]
    assert Q4_PERFORMANCE_IDENTS == ("packed", "integer_q8_late")
    assert PARITY_ONLY_IDENTS == ("integer_q8",)
    assert all(row["q4_decode"] != "integer_q8" for row in CONFIGS)


def test_expanded_catalog_retains_opt082_and_adds_extras() -> None:
    retained = catalog("parity")
    extras = extra_catalog()
    expanded = expanded_catalog()
    assert len(retained) == 105
    assert len(extras) == 93
    assert len(expanded) == 198
    retained_ids = {row["id"] for row in retained}
    extra_ids = {row["id"] for row in extras}
    assert retained_ids <= {row["id"] for row in expanded}
    assert "Q4_K_mmv_packed_M1_N1_K5120_random_assoc" in retained_ids
    assert "Q4_K_mmv_packed_M1_N1_K5120_random_assoc" not in extra_ids
    assert "Q4_K_mmv_integer_q8_late_M17_N1_K17408_random_assoc" in extra_ids
    assert "Q8_0_mmv_r1_w4_M3_N1_K5120_seed85_staged_assoc" in extra_ids
    assert "Q8_0_mmv_r2_w2_M3_N1_K256_zero_staged_assoc" in extra_ids
    assert (
        "Q4_K_fused_vs_independent_integer_q8_late_M17_N1_K256_gen0_same" in extra_ids
    )
    assert "Q4_K_eager_vs_graph_packed_M17_N1_K5120_random_same" in extra_ids
    assert extra_ids.isdisjoint(retained_ids)


def test_dispatch_uses_launch_variants_not_selector_strings() -> None:
    packed = expected_dispatch(CONFIGS[0])
    late = expected_dispatch(CONFIGS[1])
    assert packed["gate_variant"] == "q4k_gate_up_swiglu_prequant"
    assert late["gate_variant"] == "q4k_coop_gate_up_swiglu_late_prequant_q8"
    assert late["down_variant"] == "q4k_coop_mmv_late_q8"
    assert late["warps_per_row"] == 4
    observed = {
        "gate_variant": late["gate_variant"],
        "up_variant": late["up_variant"],
        "down_variant": late["down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": 4,
    }
    assert dispatch_matches(observed, late) is True
    wrong = dict(observed)
    wrong["gate_variant"] = "packed"
    assert dispatch_matches(wrong, late) is False


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-089")
    validate_future_keep_policy("OPT-089", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-089",
            {
                "task": "OPT-089",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_opt074_unadmitted_is_not_a_keep_blocker() -> None:
    parity = {
        "by_ident": {
            "packed": {"kernel_parity_pass": True},
            "integer_q8_late": {"kernel_parity_pass": True},
        }
    }
    quality = {
        CONTROL_ID: {"model_quality_pass": True},
        "late_w4": {"model_quality_pass": True},
    }
    performance = {
        CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
        "late_w4": {"performance_pass": True, "mean_diff_ms": 6.6},
    }
    decided = decide_independent_verdicts(
        parity=parity,
        quality_by_id=quality,
        performance_by_id=performance,
        mode="acceptance",
    )
    text = json.dumps(decided)
    assert "opt074_coverage_unadmitted_missing_evidence" not in text
    assert decided["opt074_coverage_unadmitted_blocker"] is False
    assert decided["selected_path"] == "late_w4"
    assert decided["independent_verdicts"]["late_w4"]["production_kept"] is True
    assert decided["independent_verdicts"][CONTROL_ID]["production_kept"] is False


def test_independent_verdicts_and_retain_packed_without_perf() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                "packed": {"kernel_parity_pass": True},
                "integer_q8_late": {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            "late_w4": {"model_quality_pass": False, "incomplete": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            "late_w4": {"performance_pass": True, "mean_diff_ms": 6.6},
        },
        mode="acceptance",
    )
    packed = decided["independent_verdicts"][CONTROL_ID]
    late = decided["independent_verdicts"]["late_w4"]
    assert packed["kernel_parity_pass"] is True
    assert packed["model_quality_pass"] is True
    assert packed["performance_pass"] is False
    assert packed["production_kept"] is True
    assert late["performance_pass"] is True
    assert late["model_quality_pass"] is False
    assert late["production_kept"] is False
    assert decided["selected_path"] == CONTROL_ID
    assert decided["shipping_unchanged"] is True
    assert decided["production_kept"] is False


def test_feedback_cannot_keep() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                "packed": {"kernel_parity_pass": True},
                "integer_q8_late": {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            "late_w4": {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            "late_w4": {"performance_pass": True, "mean_diff_ms": 6.6},
        },
        mode="feedback",
    )
    assert decided["production_kept"] is False
    assert all(
        not row["production_kept"] for row in decided["independent_verdicts"].values()
    )


def test_parity_fail_rejects_candidate_before_quality() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                "packed": {"kernel_parity_pass": True},
                "integer_q8_late": {"kernel_parity_pass": False},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            "late_w4": {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            "late_w4": {"performance_pass": True, "mean_diff_ms": 9.0},
        },
        mode="acceptance",
    )
    assert decided["independent_verdicts"]["late_w4"]["production_kept"] is False
    assert decided["selected_path"] == CONTROL_ID


def test_same_math_fail_rejects_and_packed_hard_stop() -> None:
    cases = expanded_catalog()
    gpu_cases = _all_pass_gpu(cases)
    for row in gpu_cases:
        if "fused_vs_independent_packed" in row["id"]:
            row["pass"] = False
            row["gpu_pass"] = False
    with pytest.raises(AdmissionError, match="packed Q4_K kernel parity failed"):
        evaluate_parity(skip_gpu=True, synthetic={"gpu_cases": gpu_cases})
    gpu_ok = _all_pass_gpu(cases)
    for row in gpu_ok:
        if row["candidate"] == "integer_q8":
            row["fallback"] = True
    parity = evaluate_parity(
        skip_gpu=True, synthetic={"gpu_cases": gpu_ok, "source": "synthetic"}
    )
    assert parity["by_ident"]["packed"]["kernel_parity_pass"] is True
    assert parity["by_ident"]["integer_q8"]["kernel_parity_pass"] is False
    assert parity["by_ident"]["integer_q8"]["fallback_measured_as_candidate"] is True
    assert parity["by_ident"]["integer_q8"]["parity_only"] is True
    assert parity["opt074_coverage_unadmitted_blocker"] is False


def test_quality_does_not_restore_packed_or_r2() -> None:
    late = candidate_selectors(config_by_id("late_w4"))
    assert late["q4_decode"] == "integer_q8_late"
    assert late["q8_decode"] == "r1_w4"
    applied = apply_quality_mode(enabled=True, selectors=late)
    assert applied["selectors"]["q4_decode"] == "integer_q8_late"
    assert applied["selectors"]["q8_decode"] == "r1_w4"
    assert applied["does_not_restore_packed_or_r2"] is True
    cfg = build_quality_config(enabled=True, selectors=late)
    assert cfg["does_not_restore_packed_or_r2"] is True
    assert "--quality" in cfg.get("argv", [])
    assert SHIPPING_SELECTORS["q8_decode"] == "r2_w2"
    assert OPT088_CONTROL_SELECTORS["q8_decode"] == "r1_w4"
    default = apply_quality_mode(enabled=True)
    assert default["does_not_restore_packed_or_r2"] is True


def test_authenticate_opt088_and_opt084_separately() -> None:
    opt088 = authenticate_opt088_control()
    opt084 = authenticate_opt084_freeze()
    assert opt088["authenticated"] is True
    assert opt084["authenticated"] is True
    assert opt088["expected"]["q8_layout"] == "r1_w4"
    assert opt084["expected"]["q8_layout"] == "r2_w2"
    assert opt088["selectors"]["q8_layout"] == "r1_w4"
    assert opt084["selectors"]["q8_decode"] == "r2_w2"
    packed = evaluate_quality(config_by_id(CONTROL_ID))
    assert packed["opt088_control"]["authenticated"] is True
    assert packed["opt084_freeze"]["authenticated"] is True
    late = evaluate_quality(config_by_id("late_w4"))
    assert late["model_quality_pass"] is False
    assert late["incomplete"] is True
    assert late["reason"] == "candidate_specific_quality_not_measured"
    injected = evaluate_quality(
        config_by_id("late_w4"),
        synthetic={
            "quality_by_id": {
                "late_w4": {"model_quality_pass": True, "regression": False}
            }
        },
    )
    assert injected["model_quality_pass"] is False
    assert injected.get("synthetic_injection_ignored") is True
    alarm = evaluate_quality(config_by_id("late_w4"), alarm_only=True)
    assert alarm["model_quality_pass"] is False
    assert alarm["cannot_pass_full_q"] is True


def test_incomplete_cache_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "no-proof.json"
    cache.write_text(json.dumps({"task": "OPT-088", "quality": {"nll_cases": []}}))
    proof = authenticate_anchor(
        label="missing",
        path=cache,
        expected_q4="packed",
        expected_q8="r1_w4",
    )
    assert proof["authenticated"] is False
    assert proof["rescore_required"] is True


def test_screen_keep_is_rejected() -> None:
    iteration = load_contract("OPT-089")
    q4 = workload_for_mode(iteration["workloads"]["q4"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 2,
            "observed_shapes": 1,
            "observed_tier": "screen",
            "pairs": 1,
            "sample_ids": [0, 1, 2],
            "acceptance_executed": False,
            "keep": True,
        }
    )
    screen_keep = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q4",
        workload=q4,
        stdout="QW38_OPT089_RESULT=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    assert screen_keep["result_class"] == "screen_only_keep"
    observed = parse_native_observation("QW38_OPT089_RESULT=" + stdout)
    assert observed["keep"] is True


def test_parse_quality_cases_from_corrupted_opt058_json() -> None:
    messy = (
        'QW38_OPT058_RESULT={"effective_q4_decode":"xxx","cases":'
        '[{"name":"held_out_wikitext_1024","scored":32,"mean_nll":3.32}]}\n'
    )
    cases = _parse_quality_cases(messy)
    assert cases[0]["name"] == "held_out_wikitext_1024"
    assert cases[0]["mean_nll"] == pytest.approx(3.32)


def test_historical_opt082_085_086_reports_unmodified() -> None:
    intact = historical_reports_intact()
    assert intact["unmodified"] is True
    assert intact["opt082_task"] == "OPT-082"
    assert intact["opt082_catalog_count"] == 105
    assert intact["opt082_case_count"] == 105
    assert intact["opt082_catalog_intact"] is True
    assert intact["opt082_gpu_verdict_intact"] is True
    assert intact["opt084_task"] == "OPT-084"
    assert intact["opt084_unchanged_task"] is True
    assert intact["opt089_writes_into_opt082_or_opt084"] is False
    assert intact["opt085_task"] == "OPT-085"
    assert intact["opt085_production_kept"] is False
    assert OPT082_FIXTURE.is_file()
    assert OPT084_FIXTURE.is_file()
    assert OPT082_REPORT.is_file()
    assert OPT084_REPORT.is_file()
    report082 = OPT082_REPORT.read_text(encoding="utf-8")
    assert "OPT-082" in report082
    assert "available=True ran=True success=True" in report082
    assert "catalog_count=105" in report082
    assert "OPT-089" not in report082
    assert "OPT-089" not in OPT084_REPORT.read_text(encoding="utf-8")
    opt082 = load_json(OPT082_FIXTURE)
    assert opt082["task"] == "OPT-082"
    assert int(opt082["catalog_count"]) == 105
    assert len(opt082["cases"]) == 105
    assert opt082["gpu"]["ran"] is True
    opt084 = load_json(OPT084_FIXTURE)
    assert opt084["task"] == "OPT-084"
    assert opt084["gpu"].get("blocker") != "skip_gpu"


def test_historical_guard_detects_opt082_catalog_and_gpu_erasure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt089_q4_promotion as mod

    fake_fixture = tmp_path / "opt082_kernel_parity.json"
    fake_report = tmp_path / "opt082-REPORT.md"
    fake_fixture.write_text(
        json.dumps({"task": "OPT-089", "catalog_count": 52, "cases": [{}] * 52})
    )
    fake_report.write_text("# OPT-082\navailable=False ran=False blocker=skip_gpu\n")
    monkeypatch.setattr(mod, "OPT082_FIXTURE", fake_fixture)
    monkeypatch.setattr(mod, "OPT082_REPORT", fake_report)
    intact = mod.historical_reports_intact()
    assert intact["opt082_catalog_intact"] is False
    assert intact["opt082_gpu_verdict_intact"] is False
    assert intact["opt089_writes_into_opt082_or_opt084"] is True
    assert intact["unmodified"] is False


def test_pair_stats_compares_late_w4_even_if_survivor_is_control() -> None:
    by_config = {
        CONTROL_ID: {"ms": [16.18, 16.17, 16.28], "mean_ms": 16.21},
        "late_w4": {"ms": [9.76, 9.79, 9.83], "mean_ms": 9.79},
    }
    stats = pair_stats(by_config, CONTROL_ID, {"samples": 3})
    assert stats["compared"] == "late_w4"
    assert stats["candidate_ms"] == [9.76, 9.79, 9.83]
    assert stats["mean_diff_ms"] == pytest.approx(6.4166666667)
    assert stats["positive"] is True


def test_nested_parity_record_unwraps_by_ident() -> None:
    nested = {
        "parity": {
            "parity": {
                "by_ident": {
                    "packed": {"kernel_parity_pass": True},
                    "integer_q8_late": {"kernel_parity_pass": True},
                }
            }
        },
        "independent_verdicts": {
            CONTROL_ID: {"kernel_parity_pass": True},
            "late_w4": {"kernel_parity_pass": True},
        },
    }
    record = parity_record(nested)
    assert record["by_ident"]["integer_q8_late"]["kernel_parity_pass"] is True
    by_ident = ident_parity_from_source(nested)
    assert by_ident["integer_q8_late"]["kernel_parity_pass"] is True


def test_replay_command_isolates_opt061_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "component-replay"
    command = replay_command(
        {"warmups": 1, "samples": 3},
        config_by_id("late_w4"),
        None,
        evidence_dir=evidence,
    )
    assert "--evidence-dir" in command
    directed = command[command.index("--evidence-dir") + 1]
    assert "opt061-component-replay" not in directed
    assert "component-replay" in directed
    replay_cu = CUDA_REPLAY.read_text(encoding="utf-8")
    assert "--evidence-dir" in replay_cu
    with pytest.raises(AdmissionError, match="OPT-061"):
        assert_opt089_write_path(OPT061_EVIDENCE / "rounds.jsonl")
    with pytest.raises(AdmissionError, match="must not write"):
        assert_opt089_write_path(OPT082_FIXTURE)
    with pytest.raises(AdmissionError, match="must not write"):
        assert_opt089_write_path(OPT084_FIXTURE)


def test_cuda_repaired_comparators_and_quality_config() -> None:
    parity = CUDA_PARITY.read_text(encoding="utf-8")
    quality = CUDA_QUALITY.read_text(encoding="utf-8")
    assert "silu_mul_bf16_rne" in parity
    assert "host_stage_q8_1_independent" in parity
    assert "original_bf16_is_approximation_only" in parity
    assert "--expand-opt089" in parity
    assert "cpu_dequant_q8_0_times_independent_q8_1" in parity
    assert "separate_gpu_prequant_plus_cuda_silu_bf16_rne" in quality or (
        "separate_gpu_prequant_plus_cuda_silu_bf16_rne" in parity
    )
    assert "--quality-config" in quality
    assert "applied_before_graph=true" in quality
    assert "restored_packed_or_r2=false" in quality
    assert "apply_q4_decode_ident" in quality
    assert "apply_ffn_decode_ident" in quality
    assert "apply_q8_layout_ident" in quality
    replay = CUDA_REPLAY.read_text(encoding="utf-8")
    assert "--evidence-dir" in replay
    assert "set_evidence_dir" in replay
    probe = CUDA_PROBE.read_text(encoding="utf-8")
    assert "grouping-ab" in probe
    assert "attn_ab || grouping_ab" in probe.replace("\n", " ")
    assert "const bool prefill_ok = prefill && options.prompt == kScreenPrompt" in probe
    assert "effective_q4_decode_path()" in probe
    assert "effective_ffn_decode_path()" in probe


def test_parse_prefill_wall_ms_prefers_graph_json() -> None:
    text = (
        '{"schema_version":1,"workload":"prefill","graph_wall_ms":123.5,'
        '"eager_wall_ms":0}\nstatus=passed\n'
    )
    assert parse_prefill_wall_ms(text) == 123.5
    assert (
        parse_prefill_wall_ms("keep_ab_launch wall_ms=88.25 captured_path=x\n") == 88.25
    )
    with pytest.raises(AdmissionError, match="missing prefill wall_ms"):
        parse_prefill_wall_ms("status=passed\n")


def test_apply_production_pins_rewrites_only_on_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    q4 = tmp_path / "q4k_decode_path.cuh"
    ffn = tmp_path / "ffn_decode_path.cuh"
    q4.write_text(
        'constexpr char kSelectedQ4DecodePath[] = "packed";\n',
        encoding="utf-8",
    )
    ffn.write_text(
        'constexpr char kSelectedFfnDecodePath[] = "paired_staged";\n',
        encoding="utf-8",
    )
    import tools.opt089_q4_promotion as mod

    monkeypatch.setattr(mod, "Q4_PIN_FILE", q4)
    monkeypatch.setattr(mod, "FFN_PIN_FILE", ffn)
    retained = mod.apply_production_pins(
        {
            "production_kept": False,
            "shipping_unchanged": True,
            "shipping_q4_decode": "packed",
            "shipping_ffn_decode": "paired_staged",
        }
    )
    assert retained["q4"] is False
    assert retained["ffn"] is False
    assert 'kSelectedQ4DecodePath[] = "packed"' in q4.read_text(encoding="utf-8")
    kept = mod.apply_production_pins(
        {
            "production_kept": True,
            "shipping_unchanged": False,
            "shipping_q4_decode": "integer_q8_late",
            "shipping_ffn_decode": "paired_integer",
        }
    )
    assert kept["q4"] is True
    assert kept["ffn"] is True
    assert kept["q4_pin"] == "integer_q8_late"
    assert kept["ffn_pin"] == "paired_integer"
    assert 'kSelectedQ4DecodePath[] = "integer_q8_late"' in q4.read_text(
        encoding="utf-8"
    )
    assert 'kSelectedFfnDecodePath[] = "paired_integer"' in ffn.read_text(
        encoding="utf-8"
    )


def test_host_phases_write_fixture_and_retain_packed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt089_q4_promotion as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt089_q4_promotion.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    tmp = tmp_path / "run"
    parity = mod.run("feedback", "parity", tmp, skip_gpu=True)
    assert parity["task"] == "OPT-089"
    assert (tmp_path / "opt089_q4_promotion.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    assert parity["opt074_coverage_unadmitted_blocker"] is False
    dumped = json.dumps(parity.get("independent_verdicts"))
    assert "opt074_coverage_unadmitted_missing_evidence" not in dumped
    quality = mod.run("acceptance", "quality", tmp, skip_gpu=True)
    assert quality["independent_verdicts"]["late_w4"]["model_quality_pass"] is False
    alarm = mod.run("feedback", "quality-alarm", tmp, skip_gpu=True)
    assert alarm["independent_verdicts"]["late_w4"]["model_quality_pass"] is False
    q4 = mod.run("feedback", "q4", tmp, skip_gpu=True)
    assert q4["production_kept"] is False
    assert q4["claims_throughput"] is False
    fixture = load_json(tmp_path / "opt089_q4_promotion.json")
    assert fixture["shipping_q4_decode"] == "packed"
    assert fixture["shipping_ffn_decode"] == "paired_staged"
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "kernel_parity_pass" in report
    lowered = report.casefold()
    assert "not" in lowered and "blocker" in lowered
    assert empty_verdict_row()["opt074_coverage_unadmitted_blocker"] is False
    assert fixture["historical_reports"]["unmodified"] is True
    assert quality["mode"] == "acceptance"
    committed = load_json(FIXTURE)
    assert committed["task"] == "OPT-089"
    assert "integer_q8_paired" not in committed["configurations"]


def test_quality_cli_path_invokes_native_when_runner_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import tools.opt089_q4_promotion as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt089_q4_promotion.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    calls: list[list[str]] = []

    def fake_runner(command: list[str], tier: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        stdout = (
            "restored_packed_or_r2=false\n"
            "applied_before_graph=true\n"
            "effective_q4=integer_q8_late\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(mod, "default_native_runner", fake_runner)
    result = mod.run(
        "feedback",
        "quality-alarm",
        tmp_path / "run",
        skip_gpu=False,
    )
    assert calls, "quality-alarm must invoke the native when skip_gpu is false"
    assert any("--quality" in cmd for cmd in calls)
    measured = (result.get("quality-alarm") or {}).get("measured") or {}
    assert "late_w4" in measured
    assert result["independent_verdicts"]["late_w4"]["model_quality_pass"] is False


def test_committed_fixture_has_independent_verdicts() -> None:
    fixture = load_json(FIXTURE)
    report = REPORT.read_text(encoding="utf-8")
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert fixture["quality_contract_id"] == "opt089_strict"
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    assert "late_w4" in fixture["independent_verdicts"]
    shipping_q4 = str(fixture.get("shipping_q4_decode") or "packed")
    shipping_ffn = str(fixture.get("shipping_ffn_decode") or "paired_staged")
    assert f'kSelectedQ4DecodePath[] = "{shipping_q4}"' in q4
    assert f'kSelectedFfnDecodePath[] = "{shipping_ffn}"' in ffn
    if fixture.get("production_kept"):
        assert fixture["claims_throughput"] is True
        assert fixture["evidence_complete"] is True
        assert fixture["selected_path"] == CANDIDATE_ID
        assert shipping_q4 == "integer_q8_late"
        assert shipping_ffn == "paired_integer"
        assert kept == [CANDIDATE_ID]
        assert fixture.get("reason") == "late_w4_kept"
        assert fixture.get("packed_retained") is False
    else:
        assert fixture["claims_throughput"] is False
        assert shipping_q4 == "packed"
        assert kept in ([], [CONTROL_ID])
        assert fixture["independent_verdicts"]["late_w4"]["production_kept"] is False
        assert fixture["evidence_complete"] is False
    assert "opt074_coverage_unadmitted" in report.casefold()
    assert "not" in report.casefold() and "blocker" in report.casefold()
