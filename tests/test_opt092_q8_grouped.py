"""Host tests for OPT-092 grouped Q8 r1_w4 admission. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt092_q8_grouped import (
    CANDIDATE_ID,
    CANDIDATE_LAUNCHES,
    CONFIGS,
    CONTROL_ID,
    CONTROL_LAUNCHES,
    CONTRACT,
    FIXTURE,
    ITERATION,
    LAUNCH_REDUCTION,
    MIN_MIXER_SAVING_MS,
    REPORT,
    VERDICT_KEYS,
    decide_independent_verdicts,
    dispatch_ok,
    empty_verdict_row,
    evaluate_quality,
    family_plan,
    load_json,
    parity_catalog,
    planned_observation,
    replay_command,
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
CUDA_PARITY = ROOT / "cuda/opt092_q8_grouped_test.cu"
Q8_PATH = ROOT / "cuda/q8_decode_path.cuh"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-092")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-092"
    assert contract["claims_throughput"] is True
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["q8_layout_fixed"] == "r1_w4"
    assert contract["min_mixer_saving_ms"] == MIN_MIXER_SAVING_MS
    assert contract["control_input_projection_launches"] == CONTROL_LAUNCHES
    assert contract["candidate_input_projection_launches"] == CANDIDATE_LAUNCHES
    assert contract["input_projection_launch_reduction"] == LAUNCH_REDUCTION
    assert contract["quality_contract_id"] == "opt089_strict"
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        "separate",
        "grouped_r1_w4",
    ]
    assert iteration["task"] == "OPT-092"
    assert iteration["diagnostics_make_target"] == "cuda-opt092-diagnostics"
    assert iteration["target"] == "tools/opt092_q8_grouped.py"
    assert iteration["case_ids"] == [
        "parity",
        "mixer",
        "quality",
        "d128",
        "d2048",
        "prefill-guard",
    ]
    assert "cuda-opt092-diagnostics" in makefile
    assert "qw38-cuda-opt092-q8-grouped-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    q8 = Q8_PATH.read_text(encoding="utf-8")
    fixture = load_json(FIXTURE)
    shipping = str(fixture.get("shipping_grouping") or "separate")
    assert 'kSelectedQ8DecodeGrouping[] = "' in q8
    assert f'kSelectedQ8DecodeGrouping[] = "{shipping}"' in q8
    assert "kQ8GroupedSeparateInputLaunches = 240" in q8
    assert "kQ8GroupedCandidateInputLaunches = 64" in q8
    assert "kQ8GroupedLaunchReduction = 176" in q8
    assert "QW38_OPT092_RESULT=" in CUDA_PARITY.read_text(encoding="utf-8")
    assert "QW38_OPT092_CASE=" in CUDA_PARITY.read_text(encoding="utf-8")


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-092")
    parity = iteration["workloads"]["parity"]
    mixer = iteration["workloads"]["mixer"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["prefill-guard"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 11
    assert loop_product(workload_for_mode(mixer, "feedback")) == 8
    assert loop_product(workload_for_mode(mixer, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "release")) == 12
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 2
    assert family_plan("parity", "feedback")["cases"] == 11
    screen = family_plan("mixer", "feedback")
    assert screen["candidates"] == 2
    assert screen["warmups"] == 1
    assert screen["samples"] == 3
    assert screen["engine_pairs"] == 1
    accept = family_plan("mixer", "acceptance")
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-092", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt074_coverage_unadmitted is not a blocker" in proof
    assert "240 control 64 candidate" in proof


def test_parity_catalog_shapes() -> None:
    cases = parity_catalog()
    assert len(cases) == 11
    ids = {row["id"] for row in cases}
    assert "Q8_grouped_M0_N1_K5120_staged_exact" in ids
    assert "Q8_gdn_input_group_prod" in ids
    assert "Q8_attn_input_group_prod" in ids
    assert "Q8_gdn_output_separate_prod" in ids


def test_two_grouping_configs_r1_w4_fixed() -> None:
    assert [row["id"] for row in CONFIGS] == ["separate", "grouped_r1_w4"]
    assert all(row["q8_layout"] == "r1_w4" for row in CONFIGS)
    assert CONFIGS[0]["q8_grouping"] == "separate"
    assert CONFIGS[1]["q8_grouping"] == "grouped_r1_w4"


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-092")
    validate_future_keep_policy("OPT-092", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-092",
            {
                "task": "OPT-092",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_independent_verdicts_keep_grouped_when_all_pass() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            CANDIDATE_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            CANDIDATE_ID: {"performance_pass": True, "mean_diff_ms": 0.12},
        },
        mode="acceptance",
    )
    assert decided["selected_path"] == CANDIDATE_ID
    assert decided["production_kept"] is True
    assert decided["shipping_grouping"] == CANDIDATE_ID
    assert decided["independent_verdicts"][CANDIDATE_ID]["production_kept"] is True


def test_independent_verdicts_retain_separate_without_perf() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            CANDIDATE_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            CANDIDATE_ID: {"performance_pass": False, "mean_diff_ms": 0.02},
        },
        mode="acceptance",
    )
    assert decided["selected_path"] == CONTROL_ID
    assert decided["production_kept"] is False
    assert decided["shipping_grouping"] == CONTROL_ID


def test_feedback_cannot_keep() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            CANDIDATE_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            CANDIDATE_ID: {"performance_pass": True, "mean_diff_ms": 0.12},
        },
        mode="feedback",
    )
    assert decided["production_kept"] is False
    assert all(
        not row["production_kept"] for row in decided["independent_verdicts"].values()
    )


def test_dispatch_ok_checks_launch_counts() -> None:
    separate_stdout = (
        "gdn_input_output_groups=48 attention_input_output_groups=16 "
        "q8_input_projection_launches=240 q8_grouping=separate\n"
    )
    grouped_stdout = (
        "gdn_input_output_groups=48 attention_input_output_groups=16 "
        "q8_input_projection_launches=64 q8_grouping=grouped_r1_w4\n"
    )
    assert dispatch_ok(CONFIGS[0], separate_stdout) is True
    assert dispatch_ok(CONFIGS[1], grouped_stdout) is True
    assert dispatch_ok(CONFIGS[1], separate_stdout) is False


def test_replay_command_uses_q8_grouping() -> None:
    command = replay_command(
        {"warmups": 1, "samples": 3},
        CONFIGS[1],
        None,
    )
    assert "--workload" in command
    assert command[command.index("--workload") + 1] == "decode-mixer"
    assert "--q8-grouping" in command
    assert command[command.index("--q8-grouping") + 1] == "grouped_r1_w4"
    assert "--q8-layout" in command
    assert command[command.index("--q8-layout") + 1] == "r1_w4"


def test_quality_reuses_opt089_authenticated_scores() -> None:
    control = evaluate_quality(CONFIGS[0])
    candidate = evaluate_quality(CONFIGS[1])
    assert control["reuse_opt089_authenticated_scores"] is False
    assert candidate["reuse_opt089_authenticated_scores"] is True
    assert "opt089_admission" in candidate
    assert candidate["quality_contract_id"] == "opt089_strict"


def test_screen_keep_is_rejected() -> None:
    iteration = load_contract("OPT-092")
    mixer = workload_for_mode(iteration["workloads"]["mixer"], "feedback")
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
        workload_name="mixer",
        workload=mixer,
        stdout="QW38_OPT092_RESULT=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    observed = parse_native_observation("QW38_OPT092_RESULT=" + stdout)
    assert observed["keep"] is True


def test_parse_native_observation_opt092_prefix() -> None:
    payload = (
        'QW38_OPT092_RESULT={"task":"OPT-092","phase":"parity","success":true}\n'
        'QW38_OPT092_NATIVE_COUNTS={"observed_shapes":11,"observed_tier":"correctness"}\n'
    )
    observed = parse_native_observation(payload)
    assert observed["task"] == "OPT-092"
    assert observed["phase"] == "parity"
    assert observed["observed_shapes"] == 11


def test_host_phases_write_fixture_and_retain_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt092_q8_grouped as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt092_q8_grouped.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    tmp = tmp_path / "run"
    parity = mod.run("feedback", "parity", tmp, skip_gpu=True)
    assert parity["task"] == "OPT-092"
    assert (tmp_path / "opt092_q8_grouped.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    quality = mod.run("release", "quality", tmp, skip_gpu=True)
    assert (quality.get("quality") or {}).get("phase") == "quality"
    mixer = mod.run("feedback", "mixer", tmp, skip_gpu=True)
    assert mixer["production_kept"] is False
    assert mixer["claims_throughput"] is False
    fixture = load_json(tmp_path / "opt092_q8_grouped.json")
    assert fixture["control_grouping"] == "separate"
    assert fixture["shipping_grouping"] == "separate"
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    assert empty_verdict_row()["opt074_coverage_unadmitted_blocker"] is False


def test_committed_fixture_skeleton() -> None:
    fixture = load_json(FIXTURE)
    report = REPORT.read_text(encoding="utf-8")
    q8 = Q8_PATH.read_text(encoding="utf-8")
    assert fixture["task"] == "OPT-092"
    assert fixture["control_grouping"] == "separate"
    assert fixture["shipping_grouping"] in {"separate", "grouped_r1_w4"}
    assert fixture["production_kept"] == (
        fixture["shipping_grouping"] == "grouped_r1_w4"
    )
    assert fixture["claims_throughput"] == (
        fixture["production_kept"] and fixture.get("evidence_complete")
    )
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert "separate" in fixture["independent_verdicts"]
    assert "grouped_r1_w4" in fixture["independent_verdicts"]
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    if fixture["production_kept"]:
        assert kept == ["grouped_r1_w4"]
    assert f'kSelectedQ8DecodeGrouping[] = "{fixture["shipping_grouping"]}"' in q8
    assert "kernel_parity_pass" in report
    assert "opt074_coverage_unadmitted" in report.casefold()


def test_planned_observation_counts() -> None:
    plan = family_plan("mixer", "feedback")
    observed = planned_observation(plan)
    assert observed["task"] == "OPT-092"
    assert observed["observed_candidates"] == 2
    assert observed["observed_samples"] == 3
