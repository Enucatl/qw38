"""Host tests for OPT-094 conditional GDN tile32 replication admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt094_gdn_replication import (
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    OPT090_FIXTURE,
    REPORT,
    T_CRIT_DF29,
    VERDICT_KEYS,
    decide_independent_verdicts,
    decide_no_reopen_verdicts,
    evaluate_eligibility,
    family_plan,
    load_json,
    replay_command,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
GDN_PATH = ROOT / "cuda/gdn_decode_path.cuh"
NATIVE = ROOT / "cuda/opt077_gdn_decode_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-094")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    gdn = GDN_PATH.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-094"
    assert contract["claims_throughput"] is False
    assert contract["control"] == CONTROL_ID
    assert contract["candidate"] == CANDIDATE_ID
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert contract["component_t_critical_df29"] == T_CRIT_DF29
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        CONTROL_ID,
        CANDIDATE_ID,
    ]
    assert iteration["task"] == "OPT-094"
    assert iteration["diagnostics_make_target"] == "cuda-opt094-diagnostics"
    assert iteration["target"] == "tools/opt094_gdn_replication.py"
    assert iteration["case_ids"] == [
        "eligibility",
        "parity",
        "gdn",
        "quality",
        "gdn128",
        "gdn2048",
        "d128",
        "d2048",
        "state-prefill-guard",
    ]
    assert "cuda-opt094-diagnostics" in makefile
    assert "qw38-cuda-opt077-gdn-decode-test" in makefile
    assert "tile32" in gdn
    assert NATIVE.is_file()


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-094")
    eligibility = iteration["workloads"]["eligibility"]
    parity = iteration["workloads"]["parity"]
    gdn = iteration["workloads"]["gdn"]
    gdn128 = iteration["workloads"]["gdn128"]
    gdn2048 = iteration["workloads"]["gdn2048"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["state-prefill-guard"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(gdn, "feedback")) == 8
    assert loop_product(workload_for_mode(gdn128, "acceptance")) == 66
    assert loop_product(workload_for_mode(gdn2048, "acceptance")) == 66
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 1
    described = describe_plan("OPT-094", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "gdn_reopen_eligible" in proof
    assert "tile32" in proof


def test_two_gdn_configs_only() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["gdn_decode"] == "sequential"
    assert CONFIGS[1]["gdn_decode"] == "tile32"
    assert all(int(row.get("value_tile", 0) or 0) in {0, 32} for row in CONFIGS)


def test_eligibility_from_opt090_no_reopen() -> None:
    assert OPT090_FIXTURE.is_file()
    result = evaluate_eligibility(load_json(OPT090_FIXTURE))
    assert result["eligible"] is False
    assert result["verdict"] == "no_reopen"
    assert result["reason"] == "no_opt077_timing_capture_defect_repaired"


def test_no_reopen_verdicts_keep_sequential() -> None:
    decided = decide_no_reopen_verdicts()
    assert decided["status"] == "no_reopen"
    assert decided["production_kept"] is True
    assert decided["selected_path"] == CONTROL_ID
    assert decided["claims_throughput"] is False
    assert decided["independent_verdicts"][CONTROL_ID]["production_kept"] is True
    assert decided["independent_verdicts"][CANDIDATE_ID]["not_applicable_reason"]


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-094")
    validate_future_keep_policy("OPT-094", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-094",
            {
                "task": "OPT-094",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_independent_verdicts_promote_tile32_when_all_pass() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality={"model_quality_pass": True},
        component_by_prefix={
            "2048": {
                "component": {
                    "n": 30,
                    "positive": True,
                    "mean_diff_ms": 0.12,
                    "ci95_low": 0.05,
                }
            },
            "128": {
                "component": {
                    "n": 30,
                    "positive": True,
                    "mean_diff_ms": 0.11,
                    "ci95_low": 0.04,
                }
            },
        },
        engine_by_prefix={
            "d128": {"control_mean_ms": 100.0, "regression_upper_ms": 1.0},
            "d2048": {"control_mean_ms": 900.0, "regression_upper_ms": 5.0},
        },
        mode="acceptance",
    )
    assert decided["selected_path"] == CANDIDATE_ID
    assert decided["production_kept"] is True
    assert decided["shipping_gdn_decode"] == "tile32"


def test_parse_native_observation_opt094_prefix() -> None:
    observed = parse_native_observation(
        'QW38_OPT094_RESULT={"task":"OPT-094","phase":"eligibility","success":true}\n'
        'QW38_OPT094_NATIVE_COUNTS={"observed_shapes":1}\n'
    )
    assert observed["task"] == "OPT-094"
    assert observed["phase"] == "eligibility"
    assert observed["observed_shapes"] == 1


def test_replay_command_includes_gdn_selector_and_prefix() -> None:
    plan = family_plan("gdn2048", "acceptance")
    cmd = replay_command(plan, CONFIGS[1], None)
    joined = " ".join(cmd)
    assert "--gdn-decode tile32" in joined
    assert "--prefix 2048" in joined


def test_fixture_required_keys_present() -> None:
    fixture = load_json(FIXTURE)
    contract = _json(CONTRACT)
    for key in contract["required_fixture_keys"]:
        assert key in fixture
    assert fixture["task"] == "OPT-094"
    assert REPORT.is_file()
