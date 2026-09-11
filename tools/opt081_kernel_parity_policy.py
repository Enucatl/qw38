"""OPT-081 kernel_parity_v1 admission policy.

Host-only. Freezes the active kernel-admission authority, records synthetic
tolerance cases, and writes the owned fixture/report. No kernel change and no
throughput claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kernel_parity import (  # noqa: E402
    ASSOCIATION,
    CONTRACT,
    CUD001_HISTORICAL_ABS,
    FAMILIES,
    HEADER,
    OPT074_FAMILY_ADMISSION_REQUIRED,
    POLICY_STATEMENT,
    SAME_MATH,
    STAGING_SUM_Q,
    STAGING_SUM_X,
    KernelParityError,
    abs_tolerance,
    assert_cpp_matches_python,
    check_close,
    cpp_check_close,
    diagnostics_agree,
    family_envelope,
)
from tools.run_optimization_task import (  # noqa: E402
    HISTORICAL_OPT074_KEEP_TASKS,
    KernelParityPolicyError,
    validate_future_keep_policy,
)

ITERATION = ROOT / "pins/opt081_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt081_kernel_parity_policy.json"
REPORT = ROOT / "evidence/optimization/opt081-kernel-parity-policy/REPORT.md"
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
COLUMNS = 100
HISTORICAL_FIXTURES = (
    ROOT / "fixtures/opt059_gpu_numerics.json",
    ROOT / "fixtures/opt074_production_gpu_admission.json",
    ROOT / "fixtures/opt070_keep_revalidation.json",
    ROOT / "fixtures/opt075_q4_production_admission.json",
    ROOT / "fixtures/opt076_q4_reduction.json",
)
HISTORICAL_REPORTS = (
    ROOT / "evidence/optimization/opt059-gpu-numerics/REPORT.md",
    ROOT / "evidence/optimization/opt074-production-gpu-admission/REPORT.md",
    ROOT / "evidence/optimization/opt070-keep-revalidation/REPORT.md",
    ROOT / "evidence/optimization/opt075-q4-production-admission/REPORT.md",
    ROOT / "evidence/optimization/opt076-q4-reduction/REPORT.md",
)
PROOF = (
    "no throughput claim",
    "no arithmetic kernel change",
    "no production selector change",
    "kernel admission is not model quality",
    "OPT-059/074 are historical_diagnostic_only",
    "opt074_family_admission_required is false",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _vector(*values: float) -> list[float]:
    return [float(value) for value in values]


def synthetic_cases() -> list[dict[str, Any]]:
    """Named host cases that freeze the association OR-rule and family table."""
    return [
        {
            "id": "or_rule_abs_pass",
            "family": "Q8_0",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(0.4, 0.4, 0.4, 0.4),
            "ref": _vector(0.1, 0.1, 0.1, 0.1),
            "expect_pass": True,
            "note": "abs 0.3 <= 0.05*sqrt(100)=0.5; rel 3.0 fails; OR passes",
        },
        {
            "id": "or_rule_rel_pass",
            "family": "Q8_0",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(104.0, 104.0, 104.0, 104.0),
            "ref": _vector(100.0, 100.0, 100.0, 100.0),
            "expect_pass": True,
            "note": "abs 4.0 > 0.5; rel 0.04 <= 0.05; OR passes",
        },
        {
            "id": "both_fail_rejects",
            "family": "Q8_0",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(10.6, 10.6, 10.6, 10.6),
            "ref": _vector(10.0, 10.0, 10.0, 10.0),
            "expect_pass": False,
            "note": "abs 0.6 > 0.5 and rel 0.06 > 0.05",
        },
        {
            "id": "nonfinite_rejects",
            "family": "Q8_0",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(1.0, math.nan, 1.0, 1.0),
            "ref": _vector(1.0, 1.0, 1.0, 1.0),
            "expect_pass": False,
            "expect_reason": "nonfinite",
        },
        {
            "id": "q8_scale_rejects_q4_envelope_gap",
            "family": "Q8_0",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(11.0, 11.0, 11.0, 11.0),
            "ref": _vector(10.0, 10.0, 10.0, 10.0),
            "expect_pass": False,
            "note": "abs 1.0 > Q8 0.5 and rel 0.10 > 0.05",
        },
        {
            "id": "q4_scale_accepts_same_gap",
            "family": "Q4_K",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(11.0, 11.0, 11.0, 11.0),
            "ref": _vector(10.0, 10.0, 10.0, 10.0),
            "expect_pass": True,
            "note": "abs 1.0 <= Q4 2.0; OR passes on abs",
        },
        {
            "id": "q6_reuses_q4_envelope",
            "family": "Q6_K",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(11.0, 11.0, 11.0, 11.0),
            "ref": _vector(10.0, 10.0, 10.0, 10.0),
            "expect_pass": True,
            "note": "Q6_K reuses the frozen Q4_K K-quant envelope",
        },
        {
            "id": "q2_not_applicable",
            "family": "Q2_K",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(1.0, 1.0),
            "ref": _vector(1.0, 1.0),
            "expect_pass": False,
            "expect_reason": "not_applicable",
        },
        {
            "id": "iq2_not_applicable",
            "family": "IQ2",
            "class_name": ASSOCIATION,
            "columns": COLUMNS,
            "got": _vector(1.0, 1.0),
            "ref": _vector(1.0, 1.0),
            "expect_pass": False,
            "expect_reason": "not_applicable",
        },
        {
            "id": "same_math_bit_identity",
            "family": "Q8_0",
            "class_name": SAME_MATH,
            "columns": COLUMNS,
            "got": _vector(1.25, -2.0, 0.0, 8.0),
            "ref": _vector(1.25, -2.0, 0.0, 8.0),
            "expect_pass": True,
        },
        {
            "id": "same_math_rejects_ulp",
            "family": "Q8_0",
            "class_name": SAME_MATH,
            "columns": COLUMNS,
            "got": _vector(1.0, 1.0, 1.0, 1.25),
            "ref": _vector(1.0, 1.0, 1.0, 1.0),
            "expect_pass": False,
        },
    ]


def evaluate_case(case: Mapping[str, Any], *, with_cpp: bool = False) -> dict[str, Any]:
    python = check_close(
        list(case["got"]),
        list(case["ref"]),
        family=str(case["family"]),
        columns=int(case["columns"]),
        class_name=str(case["class_name"]),
    )
    record = {
        "id": case["id"],
        "family": case["family"],
        "class": case["class_name"],
        "columns": case["columns"],
        "expect_pass": case["expect_pass"],
        "python": python,
        "matched_expectation": python["pass"] is bool(case["expect_pass"]),
    }
    if case.get("expect_reason"):
        record["matched_expectation"] = (
            record["matched_expectation"] and python["reason"] == case["expect_reason"]
        )
        record["expect_reason"] = case["expect_reason"]
    if with_cpp:
        cpp = cpp_check_close(
            list(case["got"]),
            list(case["ref"]),
            family=str(case["family"]),
            columns=int(case["columns"]),
            class_name=str(case["class_name"]),
        )
        diagnostics_agree(python, cpp)
        record["cpp"] = {
            "pass": cpp["pass"],
            "applicable": cpp["applicable"],
            "max_abs": cpp["max_abs"],
            "max_rel": cpp["max_rel"],
            "rms": cpp["rms"],
            "failing_count": cpp["failing_count"],
            "nonfinite_count": cpp["nonfinite_count"],
            "cosine": cpp.get("cosine"),
        }
        record["cpp_agrees"] = True
    return record


def q8_1_cannot_equate() -> dict[str, Any]:
    numeric = check_close(
        _vector(1.0, 2.0, 3.0),
        _vector(1.0, 2.0, 3.0),
        family="Q8_0",
        columns=COLUMNS,
        class_name=ASSOCIATION,
    )
    rejected = False
    message = ""
    try:
        check_close(
            _vector(1.0, 2.0, 3.0),
            _vector(1.0, 2.0, 3.0),
            family="Q8_0",
            columns=COLUMNS,
            class_name=ASSOCIATION,
            candidate_field=STAGING_SUM_Q,
            reference_field=STAGING_SUM_X,
        )
    except KernelParityError as exc:
        rejected = True
        message = str(exc)
    return {
        "numeric_identity_would_pass": numeric["pass"] is True,
        "typed_comparison_rejected": rejected,
        "message": message,
        "pass": numeric["pass"] is True and rejected,
    }


def future_keep_opt074_unadmitted_is_policy_error() -> dict[str, Any]:
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
    raised = False
    message = ""
    try:
        validate_future_keep_policy("OPT-999", payload)
    except KernelParityPolicyError as exc:
        raised = True
        message = str(exc)
    historical_ok = True
    for task in sorted(HISTORICAL_OPT074_KEEP_TASKS):
        try:
            validate_future_keep_policy(
                task,
                {
                    "task": task,
                    "promotion_blockers": ["opt074_coverage_unadmitted"],
                },
            )
        except KernelParityPolicyError:
            historical_ok = False
    return {
        "future_task_failed_closed": raised,
        "message": message,
        "historical_opt070_075_076_still_admitted": historical_ok,
        "pass": raised and historical_ok and "opt074_coverage_unadmitted" in message,
    }


def historical_paths_present() -> dict[str, Any]:
    missing = [
        str(path.relative_to(ROOT))
        for path in (*HISTORICAL_FIXTURES, *HISTORICAL_REPORTS)
        if not path.is_file()
    ]
    opt070 = load_json(ROOT / "fixtures/opt070_keep_revalidation.json")
    opt075 = load_json(ROOT / "fixtures/opt075_q4_production_admission.json")
    opt076 = load_json(ROOT / "fixtures/opt076_q4_reduction.json")
    return {
        "missing": missing,
        "opt070_reasons_include_unadmitted": "opt074_coverage_unadmitted"
        in json.dumps(opt070),
        "opt075_reasons_include_unadmitted": "opt074_coverage_unadmitted_missing_evidence"
        in json.dumps(opt075),
        "opt076_reasons_include_unadmitted": "opt074_coverage_unadmitted_missing_evidence"
        in json.dumps(opt076),
        "successor_note": (
            "OPT-074 unadmitted rows are not active keep blockers under "
            "kernel_parity_v1"
        ),
        "pass": not missing,
    }


def evaluate_policy(*, with_cpp: bool = True) -> dict[str, Any]:
    contract = load_json(CONTRACT)
    cases = [evaluate_case(case, with_cpp=with_cpp) for case in synthetic_cases()]
    q81 = q8_1_cannot_equate()
    keep = future_keep_opt074_unadmitted_is_policy_error()
    historical = historical_paths_present()
    constants = assert_cpp_matches_python()
    all_matched = all(row["matched_expectation"] for row in cases)
    success = (
        all_matched
        and q81["pass"]
        and keep["pass"]
        and historical["pass"]
        and contract["opt074_family_admission_required"] is False
        and contract["claims_throughput"] is False
        and contract["historical_diagnostic_only"]["cud001_abs"]
        == CUD001_HISTORICAL_ABS
        and OPT074_FAMILY_ADMISSION_REQUIRED is False
        and family_envelope("Q4_K")["abs_scale"] == family_envelope("Q6_K")["abs_scale"]
        and abs_tolerance(0.05, COLUMNS) == 0.5
        and abs_tolerance(0.20, COLUMNS) == 2.0
    )
    return {
        "success": success,
        "cases": cases,
        "q8_1": q81,
        "opt074_keep_policy": keep,
        "historical": historical,
        "cpp_constants": constants,
        "header": str(HEADER.relative_to(ROOT)),
        "contract": str(CONTRACT.relative_to(ROOT)),
        "policy_statement": POLICY_STATEMENT,
        "opt074_family_admission_required": False,
        "claims_throughput": False,
        "families": FAMILIES,
    }


def build_fixture(evaluated: Mapping[str, Any]) -> dict[str, Any]:
    contract = load_json(CONTRACT)
    return {
        "schema_version": 1,
        "task": "OPT-081",
        "status": "kernel_parity_policy",
        "id": "kernel_parity_v1",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "opt074_family_admission_required": False,
        "policy_statement": POLICY_STATEMENT,
        "active_hierarchy": contract["active_hierarchy"],
        "classes": contract["classes"],
        "families": contract["families"],
        "association": contract["association"],
        "same_math_equivalence": contract["same_math_equivalence"],
        "hard_invariants": contract["hard_invariants"],
        "historical_diagnostic_only": contract["historical_diagnostic_only"],
        "successor_note": contract["successor_note"],
        "cases": [
            {
                "id": row["id"],
                "family": row["family"],
                "class": row["class"],
                "pass": row["python"]["pass"],
                "applicable": row["python"]["applicable"],
                "max_abs": row["python"]["max_abs"],
                "max_rel": row["python"]["max_rel"],
                "rms": row["python"]["rms"],
                "failing_count": row["python"]["failing_count"],
                "nonfinite_count": row["python"]["nonfinite_count"],
                "cosine": row["python"]["cosine"],
                "reason": row["python"]["reason"],
                "matched_expectation": row["matched_expectation"],
                "cpp_agrees": row.get("cpp_agrees", False),
            }
            for row in evaluated["cases"]
        ],
        "q8_1_sum_q_versus_sum_x": evaluated["q8_1"],
        "opt074_keep_policy": evaluated["opt074_keep_policy"],
        "historical_opt059_074_070_075_076_preserved": evaluated["historical"],
        "cpp_constants": evaluated["cpp_constants"],
        "proof_limit": list(PROOF),
        "report_path": "evidence/optimization/opt081-kernel-parity-policy/REPORT.md",
    }


def write_report(fixture: Mapping[str, Any]) -> None:
    case_lines = [
        f"| {row['id']} | {row['family']} | {row['class']} | "
        f"{'pass' if row['pass'] else 'fail'} | {row['reason']} |"
        for row in fixture["cases"]
    ]
    text = f"""# OPT-081 — Kernel parity policy reset

Status: **kernel_parity_v1 frozen**. `claims_throughput: false`. No arithmetic
kernel change, no production selector change, and no speedup are claimed.
Authority remains llama.cpp `{LLAMA_REV}` and GGUF SHA-256 `{GGUF_SHA}`.

## Policy

{fixture["policy_statement"]}

Active hierarchy (**Proposed**): structural → kernel parity → same-math →
model quality → performance → release. OPT-059 v2 and OPT-074 production GPU
admission are `historical_diagnostic_only`. CUD-001 `3e-4` is not an active
kernel-parity ceiling.

`opt074_family_admission_required` is **false**. OPT-074 unadmitted rows
(`held_out_exceeds_frozen_calibration_ceiling` /
`opt074_coverage_unadmitted`) are not active keep blockers. Historical
OPT-059/074/070/075/076 fixtures and reports remain as written.

## Reference and classes

Reference = independent CPU/dequant of the **same** quantized weights and
staged/input activations. Classes:
`quantized_operation_association` (element fails only when
`abs > abs_scale*sqrt(K)` **and** `rel > rel_tol`) and
`same_math_equivalence` (bit identity unless a named tiny epsilon is
justified). Nonfinite count must be 0. Hard invariants are never waived by a
tolerance. Q8_1 `sum_q` cannot be equated to `sum_x`.

| Family | Applies | abs_scale | rel_tol |
|---|---|---:|---:|
| Q8_0 | yes | 0.05 | 0.05 |
| Q4_K | yes | 0.20 | 0.05 |
| Q6_K | yes (Q4_K envelope) | 0.20 | 0.05 |
| Q2_K / IQ2 | not_applicable | n/a | n/a |

Canonical checkers: `cuda/kernel_parity.cuh` and `tools/kernel_parity.py`.

## Synthetic cases

| Case | Family | Class | Verdict | Reason |
|---|---|---|---|---|
{chr(10).join(case_lines)}

Q8_1 typed comparison rejected:
{fixture["q8_1_sum_q_versus_sum_x"]["typed_comparison_rejected"]}.
Future keep requiring `opt074_coverage_unadmitted` failed closed:
{fixture["opt074_keep_policy"]["future_task_failed_closed"]}.

## Proof limit

- no throughput claim
- no arithmetic kernel change
- no production selector change
- kernel admission is not model quality
- OPT-059/074 are historical_diagnostic_only
- opt074_family_admission_required is false
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def run_phase(phase: str, run_dir: Path | None = None) -> dict[str, Any]:
    if phase != "policy":
        raise ValueError(f"unknown OPT-081 phase {phase}")
    evaluated = evaluate_policy(with_cpp=True)
    fixture = build_fixture(evaluated)
    write_json(FIXTURE, fixture)
    write_report(fixture)
    payload: dict[str, Any] = {
        "success": bool(evaluated["success"]),
        "result_class": "ok" if evaluated["success"] else "policy_failure",
        "gpu_work": False,
        "task": "OPT-081",
        "phase": phase,
        "claims_throughput": False,
        "opt074_family_admission_required": False,
        "case_count": len(evaluated["cases"]),
        "cases_matched": all(row["matched_expectation"] for row in evaluated["cases"]),
        "cpp_agrees": all(row.get("cpp_agrees") for row in evaluated["cases"]),
        "q8_1_typed_reject": evaluated["q8_1"]["pass"],
        "future_opt074_blocker_fail_closed": evaluated["opt074_keep_policy"]["pass"],
        "policy_statement": POLICY_STATEMENT,
        "fixture": "fixtures/opt081_kernel_parity_policy.json",
        "report": "evidence/optimization/opt081-kernel-parity-policy/REPORT.md",
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt081-result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("policy",))
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase(args.phase, args.run_dir)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
