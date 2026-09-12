"""Host tests for OPT-093 factored Q4 pair_w4 admission. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt093_q4_factored import (
    CANDIDATE_ID,
    CONFIGS,
    CONTROL_ID,
    CONTRACT,
    FIXTURE,
    ITERATION,
    MIN_FFN_SAVING_MS,
    REPORT,
    VERDICT_KEYS,
    decide_independent_verdicts,
    dispatch_ok,
    evaluate_eligibility,
    family_plan,
    load_json,
    parity_catalog,
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
CUDA_PARITY = ROOT / "cuda/opt093_q4_factored_test.cu"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-093")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    q4 = Q4_PATH.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-093"
    assert contract["claims_throughput"] is True
    assert contract["control"] == CONTROL_ID
    assert contract["candidate"] == CANDIDATE_ID
    assert contract["min_ffn_saving_ms"] == MIN_FFN_SAVING_MS
    assert contract["quality_contract_id"] == "opt089_strict"
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        CONTROL_ID,
        CANDIDATE_ID,
    ]
    assert iteration["task"] == "OPT-093"
    assert iteration["diagnostics_make_target"] == "cuda-opt093-diagnostics"
    assert iteration["target"] == "tools/opt093_q4_factored.py"
    assert iteration["case_ids"] == [
        "eligibility",
        "parity",
        "q4",
        "quality",
        "d128",
        "d2048",
        "prefill-guard",
    ]
    assert "cuda-opt093-diagnostics" in makefile
    assert "qw38-cuda-opt093-q4-factored-test" in makefile
    assert "integer_q8_factored" in q4
    assert "QW38_OPT093_RESULT=" in CUDA_PARITY.read_text(encoding="utf-8")
    assert "QW38_OPT093_CASE=" in CUDA_PARITY.read_text(encoding="utf-8")


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-093")
    eligibility = iteration["workloads"]["eligibility"]
    parity = iteration["workloads"]["parity"]
    q4 = iteration["workloads"]["q4"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["prefill-guard"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(parity, "feedback")) == 13
    assert loop_product(workload_for_mode(q4, "feedback")) == 8
    assert loop_product(workload_for_mode(q4, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "release")) == 12
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 2
    described = describe_plan("OPT-093", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt090 eligibility gate" in proof
    assert "integer_q8_factored" in proof


def test_parity_catalog_shapes() -> None:
    cases = parity_catalog()
    assert len(cases) == 13
    ids = {row["id"] for row in cases}
    assert "Q4_factored_M3_N1_K256_zero" in ids
    assert "Q4_factored_M17_N1_K17408_sampled" in ids
    assert "Q4_factored_M17_N1_K256_misaligned" in ids


def test_two_q4_configs_fixed() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, CANDIDATE_ID]
    assert CONFIGS[0]["q4_decode"] == "integer_q8_late"
    assert CONFIGS[1]["q4_decode"] == "integer_q8_factored"
    assert all(row["ffn_decode"] == "paired_integer" for row in CONFIGS)
    assert all(int(row["warps_per_row"]) == 4 for row in CONFIGS)


def test_eligibility_from_opt090_material_q4() -> None:
    assert OPT090_FIXTURE.is_file()
    result = evaluate_eligibility(load_json(OPT090_FIXTURE))
    assert result["eligible"] is True
    assert result["verdict"] == "proceed"
    assert result["no_go_bandwidth_bound"] is False


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-093")
    validate_future_keep_policy("OPT-093", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-093",
            {
                "task": "OPT-093",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_independent_verdicts_keep_factored_when_all_pass() -> None:
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
    assert decided["shipping_q4_decode"] == "integer_q8_factored"


def test_parse_native_observation_opt093_prefix() -> None:
    observed = parse_native_observation(
        'QW38_OPT093_RESULT={"task":"OPT-093","phase":"parity","success":true}\n'
        'QW38_OPT093_NATIVE_COUNTS={"observed_shapes":13}\n'
    )
    assert observed["task"] == "OPT-093"
    assert observed["phase"] == "parity"
    assert observed["observed_shapes"] == 13


def test_replay_command_includes_q4_selectors() -> None:
    plan = family_plan("q4", "feedback")
    cmd = replay_command(plan, CONFIGS[1], None)
    joined = " ".join(cmd)
    assert "--q4-decode integer_q8_factored" in joined
    assert "--ffn-decode paired_integer" in joined
    assert "--q4-warps 4" in joined


def test_dispatch_ok_matches_expected_variants() -> None:
    expected = CONFIGS[1]
    observed = {
        "gate_variant": expected["expected_gate_variant"],
        "up_variant": expected["expected_up_variant"],
        "down_variant": expected["expected_down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": 4,
    }
    assert dispatch_ok(observed, expected) is True


def test_fixture_required_keys_present() -> None:
    fixture = load_json(FIXTURE)
    contract = _json(CONTRACT)
    for key in contract["required_fixture_keys"]:
        assert key in fixture
    assert fixture["task"] == "OPT-093"
    assert fixture["report_path"] == str(REPORT.relative_to(ROOT))
