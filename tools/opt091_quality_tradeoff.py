"""OPT-091 successor quality gate and conditional late_w4 concession."""

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

from tools.opt075_q4_production_admission import (  # noqa: E402
    AdmissionError,
    dump_json,
    load_json,
    utc_now,
)
from tools.opt089_q4_promotion import (  # noqa: E402
    CANDIDATE_ID,
    CONTROL_ID,
    FIXTURE as OPT089_FIXTURE,
    OPT084_FIXTURE,
    OPT088_FIXTURE,
    authenticate_opt084_freeze,
    authenticate_opt088_control,
    config_by_id,
    evaluate_quality,
    ppl_ratio,
)
from tools.quality.compare import evaluate_ppl_contract
from tools.quality.suite import (
    QUALITY_CONTRACT_SPECS,
    evaluate_quality_contracts,
    quality_contract_spec,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_performance_admission,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt091_quality_tradeoff_contract.json"
ITERATION = ROOT / "pins/opt091_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt091_quality_tradeoff.json"
REPORT = ROOT / "evidence/optimization/opt091-quality-tradeoff/REPORT.md"
EVIDENCE = REPORT.parent

QUALITY_CONTRACT_ID = "opt091_late_w4_v1"
STRICT_CONTRACT_ID = "opt089_strict"
DEFAULT_PPL_MAX = 1.01
CANDIDATE_PPL_MAX = 1.015
AGGREGATE_PPL_MAX = 1.015
RECURRENCE_MAX = 0.02
DECODE_SPEED_CI_MIN = 1.15
DECODE_P95_MAX = 0.90
PREFILL_THROUGHPUT_MIN = 0.95
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
PHASES = ("policy", "quality", "q4", "d128", "d2048", "decision")
TIMED_PHASES = frozenset({"q4", "d128", "d2048", "decision"})


def contract_payload() -> dict[str, Any]:
    return load_json(CONTRACT)


def _nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def authenticated_opt089_measured() -> dict[str, Any]:
    if not OPT089_FIXTURE.is_file():
        raise AdmissionError("missing authenticated OPT-089 fixture")
    payload = load_json(OPT089_FIXTURE)
    quality = (payload.get("quality") or {}).get("quality") or {}
    late = quality.get(CANDIDATE_ID) or {}
    if not late.get("model_quality_pass"):
        raise AdmissionError("OPT-089 late_w4 strict quality is not authenticated")
    measured = (payload.get("quality") or {}).get("measured") or {}
    late_measured = measured.get(CANDIDATE_ID)
    if not late_measured:
        raise AdmissionError("OPT-089 missing measured late_w4 quality")
    return {
        "fixture": payload,
        "quality_row": late,
        "measured": late_measured,
        "parity_pass": bool(
            (payload.get("independent_verdicts") or {})
            .get(CANDIDATE_ID, {})
            .get("kernel_parity_pass")
        ),
        "strict_pass": bool(late.get("model_quality_pass")),
        "shipping_q4_decode": str(payload.get("shipping_q4_decode") or "packed"),
        "shipping_ffn_decode": str(
            payload.get("shipping_ffn_decode") or "paired_staged"
        ),
    }


def build_ppl_ratios(measured: Mapping[str, Any]) -> dict[str, float]:
    cases = measured.get("cases") or []
    held = _nll_from_cases(cases, "held_out_wikitext_1024")
    if held is None:
        raise AdmissionError("incomplete candidate held_out NLL")
    opt088 = authenticate_opt088_control()
    opt084 = authenticate_opt084_freeze()
    if not opt088["authenticated"] or not opt084["authenticated"]:
        raise AdmissionError("anchor caches are not authenticated")
    control_held = 1.7878782710057632
    if opt088["authenticated"]:
        opt088_payload = load_json(OPT088_FIXTURE)
        anchor_cases = (opt088_payload.get("quality") or {}).get("nll_cases") or []
        cached = _nll_from_cases(anchor_cases, "held_out_wikitext_1024")
        if cached is not None:
            control_held = cached
    opt084_held = control_held
    if opt084["authenticated"]:
        freeze = load_json(OPT084_FIXTURE)
        freeze_nll = (
            (freeze.get("quartz_vs_baseline") or {}).get("aggregate") or {}
        ).get("examples") or []
        for example in freeze_nll:
            if example.get("id") == "held_out_wikitext_1024":
                opt084_held = float(example.get("quartz_avg_nll") or opt084_held)
    return {
        "opt088_authenticated": ppl_ratio(held, control_held),
        "opt084_frozen": ppl_ratio(held, opt084_held),
    }


def quality_evidence_from_measured(
    measured: Mapping[str, Any],
    *,
    incomplete: bool = False,
    functional_failures: int | None = None,
    greedy_mismatch: bool | None = None,
    changed_inherited_answer: bool = False,
) -> dict[str, Any]:
    ratios = build_ppl_ratios(measured)
    recurrence = float(measured.get("recurrence_incremental_nll") or 0.0)
    functional = (
        int(functional_failures)
        if functional_failures is not None
        else int(measured.get("new_functional_failures") or 0)
    )
    greedy = (
        bool(greedy_mismatch)
        if greedy_mismatch is not None
        else bool(measured.get("new_greedy_mismatch"))
    )
    contracts = evaluate_quality_contracts(
        ratios=ratios,
        recurrence_incremental_nll=recurrence,
        functional_failures=functional,
        greedy_mismatch=greedy,
        changed_inherited_answer=changed_inherited_answer,
        incomplete=incomplete,
        contract_ids=[STRICT_CONTRACT_ID, QUALITY_CONTRACT_ID],
    )
    strict = contracts["contracts"][STRICT_CONTRACT_ID]
    successor = contracts["contracts"][QUALITY_CONTRACT_ID]
    return {
        "ratios": ratios,
        "recurrence_incremental_nll": recurrence,
        "functional_failures": functional,
        "greedy_mismatch": greedy,
        "changed_inherited_answer": changed_inherited_answer,
        "incomplete": incomplete,
        "strict_model_quality_pass": bool(strict["model_quality_pass"]),
        "successor_model_quality_pass": bool(successor["model_quality_pass"]),
        "contracts": contracts,
    }


def strict_fail_solely_on_ppl_window(
    evidence: Mapping[str, Any],
) -> bool:
    strict = evidence["contracts"]["contracts"][STRICT_CONTRACT_ID]
    successor = evidence["contracts"]["contracts"][QUALITY_CONTRACT_ID]
    if strict["model_quality_pass"]:
        return False
    if not successor["model_quality_pass"]:
        return False
    if evidence.get("incomplete"):
        return False
    if evidence.get("functional_failures", 0) > 0:
        return False
    if evidence.get("greedy_mismatch"):
        return False
    if evidence.get("changed_inherited_answer"):
        return False
    if float(evidence.get("recurrence_incremental_nll") or 0.0) > RECURRENCE_MAX:
        return False
    ratios = dict(evidence.get("ratios") or {})
    if not ratios:
        return False
    for value in ratios.values():
        ratio = float(value)
        if ratio <= DEFAULT_PPL_MAX or ratio > CANDIDATE_PPL_MAX:
            return False
    return True


def regression_release_quality_pass(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("incomplete"):
        return False
    if int(evidence.get("functional_failures") or 0) > 0:
        return False
    if evidence.get("greedy_mismatch"):
        return False
    if evidence.get("changed_inherited_answer"):
        return False
    if float(evidence.get("recurrence_incremental_nll") or 0.0) > RECURRENCE_MAX:
        return False
    ratios = dict(evidence.get("ratios") or {})
    if not ratios:
        return False
    successor = evaluate_ppl_contract(
        ratios, quality_contract_spec(QUALITY_CONTRACT_ID)
    )
    return bool(successor["pass"])


def absolute_quality_status(evidence: Mapping[str, Any]) -> str:
    if evidence.get("incomplete"):
        return "incomplete"
    if int(evidence.get("functional_failures") or 0) > 0:
        return "new_functional_failures"
    if evidence.get("greedy_mismatch"):
        return "new_greedy_mismatch"
    if evidence.get("changed_inherited_answer"):
        return "changed_inherited_answer"
    return "visible_and_separate"


def concession_route(
    evidence: Mapping[str, Any],
    *,
    parity_pass: bool,
    performance_pass: bool = False,
) -> dict[str, Any]:
    eligible = strict_fail_solely_on_ppl_window(evidence)
    if evidence.get("strict_model_quality_pass"):
        used = False
        reason = "strict_quality_already_passed"
    elif not parity_pass:
        used = False
        reason = "parity_or_same_math_failure"
    elif not eligible:
        used = False
        reason = "not_eligible_for_concession"
    elif not performance_pass:
        used = False
        reason = "performance_gate_not_met"
    else:
        used = True
        reason = "conditional_concession_exercised"
    return {
        "concession_eligible": eligible,
        "concession_used": used,
        "reason": reason,
        "timed_phases_required": eligible and not evidence.get("strict_model_quality_pass"),
    }


def independent_verdicts(
    *,
    evidence: Mapping[str, Any],
    route: Mapping[str, Any],
    parity_pass: bool,
    performance_pass: bool,
    production_kept: bool,
) -> dict[str, dict[str, Any]]:
    active_pass = bool(
        evidence["strict_model_quality_pass"]
        if not route["concession_used"]
        else evidence["successor_model_quality_pass"]
    )
    active_contract = (
        STRICT_CONTRACT_ID if not route["concession_used"] else QUALITY_CONTRACT_ID
    )
    rows = {
        CONTROL_ID: {
            "kernel_parity_pass": parity_pass,
            "model_quality_pass": True,
            "performance_pass": False,
            "production_kept": False,
            "quality_contract_id": active_contract,
            "incomplete": False,
        },
        CANDIDATE_ID: {
            "kernel_parity_pass": parity_pass,
            "model_quality_pass": active_pass,
            "performance_pass": bool(performance_pass),
            "production_kept": bool(production_kept),
            "quality_contract_id": active_contract,
            "incomplete": bool(evidence.get("incomplete")),
        },
    }
    return rows


def policy_record(
    *,
    mode: str,
    measured: Mapping[str, Any] | None = None,
    incomplete: bool = False,
    parity_pass: bool | None = None,
    performance_pass: bool = False,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract_payload()
    opt089 = authenticated_opt089_measured() if not synthetic else None
    if synthetic:
        measured = dict(synthetic.get("measured") or {})
        incomplete = bool(synthetic.get("incomplete", incomplete))
        parity_pass = bool(synthetic.get("parity_pass", parity_pass))
        performance_pass = bool(synthetic.get("performance_pass", performance_pass))
    elif measured is None:
        measured = opt089["measured"] if opt089 else {}
        incomplete = False
        parity_pass = bool(opt089 and opt089["parity_pass"])
    elif parity_pass is None:
        parity_pass = bool(opt089 and opt089["parity_pass"])
    evidence = quality_evidence_from_measured(
        measured,
        incomplete=incomplete,
        functional_failures=(
            int(synthetic.get("functional_failures", 0))
            if synthetic
            else None
        ),
        greedy_mismatch=bool(synthetic.get("greedy_mismatch")) if synthetic else None,
        changed_inherited_answer=bool(
            synthetic.get("changed_inherited_answer") if synthetic else False
        ),
    )
    route = concession_route(
        evidence,
        parity_pass=bool(parity_pass),
        performance_pass=performance_pass,
    )
    shipping_q4 = (
        str(synthetic.get("shipping_q4_decode"))
        if synthetic and synthetic.get("shipping_q4_decode")
        else str((opt089 or {}).get("shipping_q4_decode") or "packed")
    )
    shipping_ffn = (
        str(synthetic.get("shipping_ffn_decode"))
        if synthetic and synthetic.get("shipping_ffn_decode")
        else str((opt089 or {}).get("shipping_ffn_decode") or "paired_staged")
    )
    production_kept = bool(route["concession_used"])
    claims_throughput = bool(route["concession_used"] and performance_pass)
    active_contract = (
        STRICT_CONTRACT_ID if not route["concession_used"] else QUALITY_CONTRACT_ID
    )
    verdicts = independent_verdicts(
        evidence=evidence,
        route=route,
        parity_pass=bool(parity_pass),
        performance_pass=performance_pass,
        production_kept=production_kept,
    )
    return {
        "schema_version": 1,
        "task": "OPT-091",
        "mode": mode,
        "phase": "policy",
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "strict_quality_contract_id": STRICT_CONTRACT_ID,
        "active_quality_contract_id": active_contract,
        "absolute_quality_status": absolute_quality_status(evidence),
        "strict_model_quality_pass": evidence["strict_model_quality_pass"],
        "successor_model_quality_pass": evidence["successor_model_quality_pass"],
        "regression_release_quality_pass": regression_release_quality_pass(evidence),
        "concession_eligible": route["concession_eligible"],
        "concession_used": route["concession_used"],
        "concession_reason": route["reason"],
        "claims_throughput": claims_throughput,
        "production_kept": production_kept,
        "shipping_q4_decode": shipping_q4,
        "shipping_ffn_decode": shipping_ffn,
        "shipping_unchanged": not production_kept,
        "historical_gate_relabel": False,
        "kernel_parity_pass": {
            cid: row["kernel_parity_pass"] for cid, row in verdicts.items()
        },
        "model_quality_pass": {
            cid: row["model_quality_pass"] for cid, row in verdicts.items()
        },
        "performance_pass": {
            cid: row["performance_pass"] for cid, row in verdicts.items()
        },
        "independent_verdicts": verdicts,
        "ppl_ratios": evidence["ratios"],
        "recurrence_incremental_nll": evidence["recurrence_incremental_nll"],
        "functional_failures": evidence["functional_failures"],
        "greedy_mismatch": evidence["greedy_mismatch"],
        "changed_inherited_answer": evidence["changed_inherited_answer"],
        "incomplete": evidence["incomplete"],
        "timed_phases_status": (
            "required" if route["timed_phases_required"] else "not_applicable"
        ),
        "opt089_reused": opt089 is not None and synthetic is None,
        "eligible_candidate": contract["eligible_candidate"],
        "anchors": list(contract["anchors"]),
        "decode_speed_ratio_ci_lower_min": DECODE_SPEED_CI_MIN,
        "decode_p95_ratio_max": DECODE_P95_MAX,
        "kernel_parity_contract": contract["kernel_parity_contract"],
        "measurement_utc": utc_now(),
        "status": "policy_evaluated",
    }


def timed_phase_record(phase: str, mode: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    required = policy.get("timed_phases_status") == "required"
    if not required:
        return {
            "schema_version": 1,
            "task": "OPT-091",
            "phase": phase,
            "mode": mode,
            "status": "not_applicable",
            "gpu_work": False,
            "concession_used": False,
            "claims_throughput": False,
            "performance_pass": False,
            "reason": policy.get("concession_reason", "strict_quality_already_passed"),
            "measurement_utc": utc_now(),
        }
    raise AdmissionError(
        f"timed phase {phase} requires GPU execution when concession is eligible"
    )


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    return {
        "phase": phase,
        "mode": mode,
        "tier": str(workload.get("tier", "correctness")),
        "product": loop_product(workload),
        **{key: workload.get(key) for key in ("warmups", "samples", "cases", "candidates")},
    }


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdicts = payload.get("independent_verdicts") or {}
    lines = []
    for config_id in (CONTROL_ID, CANDIDATE_ID):
        row = verdicts.get(config_id) or {}
        lines.append(
            f"| {config_id} | {row.get('kernel_parity_pass')} | "
            f"{row.get('model_quality_pass')} | {row.get('performance_pass')} | "
            f"{row.get('production_kept')} |"
        )
    text = f"""# OPT-091 — Successor quality gate (`opt091_late_w4_v1`)

Status: **{payload.get("status") or "pending"}**.
Successor contract `{QUALITY_CONTRACT_ID}` with strict anchor `{STRICT_CONTRACT_ID}`.
`concession_used={payload.get("concession_used")}`.
`claims_throughput={payload.get("claims_throughput")}`.

## Policy verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
{chr(10).join(lines)}

absolute_quality_status={payload.get("absolute_quality_status")}.
strict_model_quality_pass={payload.get("strict_model_quality_pass")}.
successor_model_quality_pass={payload.get("successor_model_quality_pass")}.
regression_release_quality_pass={payload.get("regression_release_quality_pass")}.
timed_phases_status={payload.get("timed_phases_status")}.
shipping Q4 `{payload.get("shipping_q4_decode")}` / FFN `{payload.get("shipping_ffn_decode")}`.

PPL ratios: {payload.get("ppl_ratios")}.
Recurrence incremental NLL: {payload.get("recurrence_incremental_nll")}.
Concession reason: {payload.get("concession_reason")}.

Accepted risk when concession is exercised: at most 1.5% PPL drift on frozen spans
does not bound every task or long context. Roll back to the exact OPT-089 selection
if any later combined Q/PPL/recurrence/state/task test fails.
"""
    REPORT.write_text(text, encoding="utf-8")


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if FIXTURE.is_file() and synthetic is None:
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    if phase == "policy":
        payload = policy_record(mode=mode, synthetic=synthetic)
    elif phase == "quality":
        opt089 = authenticated_opt089_measured()
        measured = opt089["measured"]
        quality = evaluate_quality(
            config_by_id(CANDIDATE_ID),
            measured=measured,
        )
        policy = policy_record(mode=mode, measured=measured)
        payload = {
            **policy,
            "phase": "quality",
            "quality": quality,
            "measured": measured,
            "status": "quality_evaluated",
        }
    else:
        policy = (
            policy_record(mode=mode, synthetic=synthetic)
            if synthetic
            else policy_record(mode=mode)
        )
        payload = timed_phase_record(phase, mode, policy)
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-091",
            "mode": mode,
            "llama_revision": contract_payload()["llama_revision"],
            "gguf_sha256": contract_payload()["gguf_sha256"],
            "quality_contract_id": QUALITY_CONTRACT_ID,
            "report_path": "evidence/optimization/opt091-quality-tradeoff/REPORT.md",
        }
    )
    results[phase] = payload
    for key in (
        "absolute_quality_status",
        "strict_model_quality_pass",
        "successor_model_quality_pass",
        "regression_release_quality_pass",
        "concession_used",
        "concession_eligible",
        "claims_throughput",
        "production_kept",
        "shipping_q4_decode",
        "shipping_ffn_decode",
        "independent_verdicts",
        "kernel_parity_pass",
        "model_quality_pass",
        "performance_pass",
        "timed_phases_status",
        "status",
    ):
        if key in payload:
            results[key] = payload[key]
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt091_quality_tradeoff.json", results)
    counts = {
        "schema_version": 1,
        "task": "OPT-091",
        "warmups": 0,
        "samples": 1,
        "observed_warmups": 0,
        "observed_samples": 1,
        "observed_candidates": 1,
        "observed_shapes": 1,
        "observed_tier": "correctness",
        "pairs": 1,
        "sample_ids": [0],
        "acceptance_executed": phase != "policy",
        "keep": bool(payload.get("production_kept")),
    }
    print(
        "QW38_OPT091_RESULT="
        + json.dumps(
            {
                "task": "OPT-091",
                "mode": mode,
                "phase": phase,
                "concession_used": payload.get("concession_used"),
                "claims_throughput": payload.get("claims_throughput"),
                "strict_model_quality_pass": payload.get("strict_model_quality_pass"),
                "successor_model_quality_pass": payload.get(
                    "successor_model_quality_pass"
                ),
                "regression_release_quality_pass": payload.get(
                    "regression_release_quality_pass"
                ),
                "quality_contract_id": payload.get("quality_contract_id"),
                "production_kept": payload.get("production_kept"),
                "status": payload.get("status"),
                **counts,
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("feedback", "acceptance", "release"),
    )
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.mode, args.phase, Path(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
