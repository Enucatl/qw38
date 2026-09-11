"""Host tests for OPT-081 kernel_parity_v1 admission policy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.kernel_parity import (
    ASSOCIATION,
    CONTRACT,
    HEADER,
    SAME_MATH,
    STAGING_SUM_Q,
    STAGING_SUM_X,
    KernelParityError,
    abs_tolerance,
    check_close,
    cpp_check_close,
    diagnostics_agree,
    family_envelope,
    header_constants,
)
from tools.opt081_kernel_parity_policy import (
    FIXTURE,
    ITERATION,
    REPORT,
    load_json,
    run_phase,
    synthetic_cases,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
DOCS_NUMERICS = ROOT / "docs/04-numerics.md"
DOCS_OPT = ROOT / "docs/06-system-optimization.md"
PLAN = ROOT / "plan.md"
HISTORICAL = (
    ROOT / "fixtures/opt059_gpu_numerics.json",
    ROOT / "fixtures/opt074_production_gpu_admission.json",
    ROOT / "fixtures/opt070_keep_revalidation.json",
    ROOT / "fixtures/opt075_q4_production_admission.json",
    ROOT / "fixtures/opt076_q4_reduction.json",
    ROOT / "evidence/optimization/opt059-gpu-numerics/REPORT.md",
    ROOT / "evidence/optimization/opt074-production-gpu-admission/REPORT.md",
    ROOT / "evidence/optimization/opt070-keep-revalidation/REPORT.md",
    ROOT / "evidence/optimization/opt075-q4-production-admission/REPORT.md",
    ROOT / "evidence/optimization/opt076-q4-reduction/REPORT.md",
)


def test_contracts_iteration_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-081")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert contract["id"] == "kernel_parity_v1"
    assert contract["task"] == "OPT-081"
    assert contract["claims_throughput"] is False
    assert contract["opt074_family_admission_required"] is False
    assert contract["historical_diagnostic_only"]["cud001_abs"] == 0.0003
    assert contract["families"]["Q8_0"]["abs_scale"] == 0.05
    assert contract["families"]["Q4_K"]["abs_scale"] == 0.20
    assert contract["families"]["Q6_K"]["abs_scale"] == 0.20
    assert contract["families"]["Q2_K"]["status"] == "not_applicable"
    assert contract["families"]["IQ2"]["status"] == "not_applicable"
    assert iteration["host_only"] is True
    assert iteration["skip_compile"] is True
    assert iteration["case_ids"] == ["policy"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt081-diagnostics" in makefile
    assert HEADER.is_file()
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    parsed = header_constants()
    assert parsed["q8_0_abs_scale"] == 0.05
    assert parsed["q4_k_abs_scale"] == parsed["q6_k_abs_scale"] == 0.20
    assert parsed["opt074_family_admission_required"] is False


def test_association_or_rule_abs_or_rel_suffices() -> None:
    columns = 100
    abs_pass = check_close(
        [0.4, 0.4, 0.4, 0.4],
        [0.1, 0.1, 0.1, 0.1],
        family="Q8_0",
        columns=columns,
        class_name=ASSOCIATION,
    )
    assert abs_pass["abs_tol"] == pytest.approx(0.5)
    assert abs_pass["max_abs"] == pytest.approx(0.3)
    assert abs_pass["max_rel"] == pytest.approx(3.0)
    assert abs_pass["pass"] is True
    rel_pass = check_close(
        [104.0, 104.0, 104.0, 104.0],
        [100.0, 100.0, 100.0, 100.0],
        family="Q8_0",
        columns=columns,
        class_name=ASSOCIATION,
    )
    assert rel_pass["max_abs"] == pytest.approx(4.0)
    assert rel_pass["max_rel"] == pytest.approx(0.04)
    assert rel_pass["pass"] is True


def test_both_fail_and_nonfinite_reject() -> None:
    both = check_close(
        [10.6, 10.6, 10.6, 10.6],
        [10.0, 10.0, 10.0, 10.0],
        family="Q8_0",
        columns=100,
        class_name=ASSOCIATION,
    )
    assert both["pass"] is False
    assert both["failing_count"] == 4
    nan = check_close(
        [1.0, math.nan, 1.0],
        [1.0, 1.0, 1.0],
        family="Q8_0",
        columns=100,
        class_name=ASSOCIATION,
    )
    inf = check_close(
        [1.0, math.inf, 1.0],
        [1.0, 1.0, 1.0],
        family="Q8_0",
        columns=100,
        class_name=ASSOCIATION,
    )
    assert nan["pass"] is False
    assert nan["nonfinite_count"] == 1
    assert inf["pass"] is False
    assert inf["nonfinite_count"] == 1


def test_q4_q8_q6_scales_and_q2_iq2_inapplicable() -> None:
    got = [11.0, 11.0, 11.0, 11.0]
    ref = [10.0, 10.0, 10.0, 10.0]
    q8 = check_close(got, ref, family="Q8_0", columns=100, class_name=ASSOCIATION)
    q4 = check_close(got, ref, family="Q4_K", columns=100, class_name=ASSOCIATION)
    q6 = check_close(got, ref, family="Q6_K", columns=100, class_name=ASSOCIATION)
    assert abs_tolerance(0.05, 100) == pytest.approx(0.5)
    assert abs_tolerance(0.20, 100) == pytest.approx(2.0)
    assert q8["abs_tol"] == pytest.approx(0.5)
    assert q4["abs_tol"] == q6["abs_tol"] == pytest.approx(2.0)
    assert q8["pass"] is False
    assert q4["pass"] is True
    assert q6["pass"] is True
    q2 = check_close([1.0], [1.0], family="Q2_K", columns=256, class_name=ASSOCIATION)
    iq2 = check_close([1.0], [1.0], family="IQ2", columns=256, class_name=ASSOCIATION)
    assert q2["applicable"] is False
    assert q2["reason"] == "not_applicable"
    assert q2["pass"] is False
    assert iq2["applicable"] is False
    assert family_envelope("Q2_K")["applies"] is False
    assert family_envelope("IQ2")["applies"] is False


def test_same_math_bit_identity() -> None:
    identical = check_close(
        [1.25, -2.0, 0.0, 8.0],
        [1.25, -2.0, 0.0, 8.0],
        family="Q8_0",
        columns=100,
        class_name=SAME_MATH,
    )
    assert identical["abs_tol"] == 0.0
    assert identical["rel_tol"] == 0.0
    assert identical["pass"] is True
    drifted = check_close(
        [1.0, 1.0, 1.0, 1.25],
        [1.0, 1.0, 1.0, 1.0],
        family="Q8_0",
        columns=100,
        class_name=SAME_MATH,
    )
    assert drifted["pass"] is False
    assert drifted["failing_count"] == 1


def test_sum_q_versus_sum_x_cannot_be_equated() -> None:
    numeric = check_close(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        family="Q8_0",
        columns=100,
        class_name=ASSOCIATION,
    )
    assert numeric["pass"] is True
    with pytest.raises(KernelParityError, match="sum_q"):
        check_close(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            family="Q8_0",
            columns=100,
            class_name=ASSOCIATION,
            candidate_field=STAGING_SUM_Q,
            reference_field=STAGING_SUM_X,
        )


def test_cpp_and_python_agree_on_synthetic_vectors() -> None:
    for case in synthetic_cases():
        python = check_close(
            list(case["got"]),
            list(case["ref"]),
            family=str(case["family"]),
            columns=int(case["columns"]),
            class_name=str(case["class_name"]),
        )
        cpp = cpp_check_close(
            list(case["got"]),
            list(case["ref"]),
            family=str(case["family"]),
            columns=int(case["columns"]),
            class_name=str(case["class_name"]),
        )
        diagnostics_agree(python, cpp)
        assert python["pass"] is bool(case["expect_pass"])


def test_opt074_unadmitted_is_not_active_keep_blocker() -> None:
    payload = {
        "schema_version": 1,
        "task": "OPT-999",
        "host_only": True,
        "skip_compile": True,
        "target": "tools/opt081_kernel_parity_policy.py",
        "promotion_blockers": ["opt074_coverage_unadmitted"],
        "modes": {
            "feedback": {
                "target": "tools/opt081_kernel_parity_policy.py",
                "tier_sequence": ["policy"],
                "workloads": ["policy"],
                "skip_compile": True,
            }
        },
        "workloads": {
            "policy": {
                "cases": 1,
                "candidates": 1,
                "warmups": 0,
                "samples": 1,
                "tokens": 1,
                "execution_modes": 1,
                "control_candidate_pairs": 1,
            }
        },
    }
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy("OPT-999", payload)
    validate_future_keep_policy(
        "OPT-070",
        {"task": "OPT-070", "promotion_blockers": ["opt074_coverage_unadmitted"]},
    )
    validate_future_keep_policy(
        "OPT-075",
        {"task": "OPT-075", "promotion_blockers": ["opt074_coverage_unadmitted"]},
    )
    validate_future_keep_policy(
        "OPT-076",
        {"task": "OPT-076", "promotion_blockers": ["opt074_coverage_unadmitted"]},
    )
    load_contract("OPT-070")
    load_contract("OPT-075")
    load_contract("OPT-076")


def test_historical_opt059_074_070_075_076_verdicts_preserved() -> None:
    opt070 = load_json(ROOT / "fixtures/opt070_keep_revalidation.json")
    opt075 = load_json(ROOT / "fixtures/opt075_q4_production_admission.json")
    opt076 = load_json(ROOT / "fixtures/opt076_q4_reduction.json")
    assert "opt074_coverage_unadmitted" in json.dumps(opt070)
    assert "opt074_coverage_unadmitted_missing_evidence" in json.dumps(opt075)
    assert "opt074_coverage_unadmitted_missing_evidence" in json.dumps(opt076)
    for path in HISTORICAL:
        assert path.is_file(), path


def test_iteration_loop_products_and_dry_run_phases() -> None:
    iteration = load_contract("OPT-081")
    assert loop_product(iteration["workloads"]["policy"]) == 11
    plan = describe_plan("OPT-081", "feedback", iteration, None)
    assert "host_only=true" in plan
    assert "skip_compile=true" in plan
    assert "historical_oracles=none" in plan
    policy = describe_plan("OPT-081", "feedback", iteration, "policy")
    assert "phase=policy" in policy
    assert "product=11" in policy
    acceptance = describe_plan("OPT-081", "acceptance", iteration, "policy")
    assert "phase=policy" in acceptance


def test_host_policy_writes_fixture_and_proposed_docs() -> None:
    result = run_phase("policy")
    assert result["success"] is True
    assert result["claims_throughput"] is False
    assert result["cpp_agrees"] is True
    assert result["q8_1_typed_reject"] is True
    assert result["future_opt074_blocker_fail_closed"] is True
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    fixture = load_json(FIXTURE)
    assert fixture["opt074_family_admission_required"] is False
    text = REPORT.read_text(encoding="utf-8")
    numerics = DOCS_NUMERICS.read_text(encoding="utf-8")
    system = DOCS_OPT.read_text(encoding="utf-8")
    for item in load_json(CONTRACT)["proof_limit"]:
        assert item in text
    assert "Kernel admission proves implementation of the intended quantized" in text
    assert "**Proposed**" in numerics
    assert "structural" in numerics and "kernel parity" in numerics
    assert "historical" in numerics.lower()
    assert "**Proposed**" in system
    assert "structural" in system and "kernel parity" in system
    plan = PLAN.read_text(encoding="utf-8")
    assert "kernel_parity_v1" not in plan
