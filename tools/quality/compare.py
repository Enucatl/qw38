"""Per-example and aggregate comparator, ported from ds4 compare_scores.py."""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Mapping, Sequence

from tools.quality.errors import QualityFrameworkError

HISTORICAL_QUALITY_CONTRACT_ID = "opt084_frozen"
HISTORICAL_PPL_RATIO_MAX = 1.01
HISTORICAL_RECURRENCE_MAX = 0.02


def refuse_single_boolean(record: Mapping[str, Any] | None = None) -> bool:
    """Quality evidence is multi-field. A single boolean is not a verdict."""
    del record
    raise QualityFrameworkError(
        "quality results cannot be reduced to one boolean; inspect per-example "
        "NLL, aggregate NLL, deltas, first-token matches, greedy LCP, finite "
        "vocab, dual verdicts, and identities separately"
    )


def resolve_quality_contract(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return PPL/recurrence limits. Omitting contract keeps the 1.01 freeze."""
    if contract is None:
        return {
            "quality_contract_id": HISTORICAL_QUALITY_CONTRACT_ID,
            "ppl_ratio_max": HISTORICAL_PPL_RATIO_MAX,
            "recurrence_incremental_nll_max": HISTORICAL_RECURRENCE_MAX,
            "explicit": False,
        }
    ppl = contract.get("ppl_ratio_max")
    if ppl is None and str(contract.get("quality_contract_id") or "") == (
        "opt091_late_w4_v1"
    ):
        ppl = contract.get("candidate_ppl_ratio_max")
    if ppl is None:
        ppl = HISTORICAL_PPL_RATIO_MAX
    rec = contract.get("recurrence_incremental_nll_max")
    if rec is None:
        rec = HISTORICAL_RECURRENCE_MAX
    cid = str(contract.get("quality_contract_id") or HISTORICAL_QUALITY_CONTRACT_ID)
    return {
        "quality_contract_id": cid,
        "ppl_ratio_max": float(ppl),
        "recurrence_incremental_nll_max": float(rec),
        "explicit": True,
    }


def ppl_ratio_from_nll(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def ratios_within_contract(
    ratios: Sequence[float],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    limits = resolve_quality_contract(contract)
    finite = [float(value) for value in ratios]
    if not finite or any(not math.isfinite(value) for value in finite):
        return {
            "pass": False,
            "incomplete": True,
            "max_ratio": None,
            "ppl_ratio_max": limits["ppl_ratio_max"],
            "quality_contract_id": limits["quality_contract_id"],
            "explicit": limits["explicit"],
        }
    return {
        "pass": all(value <= limits["ppl_ratio_max"] for value in finite),
        "incomplete": False,
        "max_ratio": max(finite),
        "ratios": finite,
        "ppl_ratio_max": limits["ppl_ratio_max"],
        "quality_contract_id": limits["quality_contract_id"],
        "explicit": limits["explicit"],
    }


def compare_engine_records(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    left_name: str = "quartz",
    right_name: str = "llama",
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise QualityFrameworkError("no common cases")
    left_nll = right_nll = 0.0
    left_first = right_first = 0
    left_lcp = right_lcp = 0
    tokens = 0
    left_wins = right_wins = ties = 0
    examples: list[dict[str, Any]] = []
    for case_id in ids:
        old = left[case_id]
        new = right[case_id]
        if int(old["target_tokens"]) != int(new["target_tokens"]):
            raise QualityFrameworkError(f"token-count mismatch for {case_id}")
        count = int(old["target_tokens"])
        tokens += count
        left_nll += float(old["nll"])
        right_nll += float(new["nll"])
        left_first += int(old["first_match"])
        right_first += int(new["first_match"])
        left_lcp += int(old["greedy_lcp"])
        right_lcp += int(new["greedy_lcp"])
        delta = float(new["nll"]) - float(old["nll"])
        if delta < -1e-9:
            right_wins += 1
        elif delta > 1e-9:
            left_wins += 1
        else:
            ties += 1
        examples.append(
            {
                "id": case_id,
                "target_tokens": count,
                f"{left_name}_nll": float(old["nll"]),
                f"{right_name}_nll": float(new["nll"]),
                f"{left_name}_avg_nll": float(old["avg_nll"]),
                f"{right_name}_avg_nll": float(new["avg_nll"]),
                "delta_right_minus_left": delta,
                f"{left_name}_first_match": int(old["first_match"]),
                f"{right_name}_first_match": int(new["first_match"]),
                f"{left_name}_greedy_lcp": int(old["greedy_lcp"]),
                f"{right_name}_greedy_lcp": int(new["greedy_lcp"]),
            }
        )
    avg_left = left_nll / tokens
    avg_right = right_nll / tokens
    limits = resolve_quality_contract(contract)
    ppl_ratio = ppl_ratio_from_nll(avg_right, avg_left) if tokens else float("nan")
    gated = ratios_within_contract([ppl_ratio], contract)
    return {
        "cases": len(ids),
        "tokens": tokens,
        f"{left_name}_avg_nll": avg_left,
        f"{right_name}_avg_nll": avg_right,
        "delta_right_minus_left": avg_right - avg_left,
        "relative_nll_change": (avg_right / avg_left - 1.0) if avg_left else 0.0,
        "ppl_ratio": ppl_ratio,
        "ppl_ratio_max": limits["ppl_ratio_max"],
        "quality_contract_id": limits["quality_contract_id"],
        "ppl_ratio_pass": gated["pass"],
        "case_wins_right_left_ties": [right_wins, left_wins, ties],
        "first_token_matches_left_right": [left_first, right_first],
        "avg_greedy_lcp_left_right": [left_lcp / len(ids), right_lcp / len(ids)],
        "examples": examples,
        "single_boolean": None,
        "inspectable": True,
        "authoritative_local_comparison": f"{left_name}+{right_name}",
    }


def evaluate_ppl_contract(
    ratios: Mapping[str, float],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate named PPL ratios against explicit contract thresholds."""
    if not ratios:
        return {
            "pass": False,
            "incomplete": True,
            "reason": "missing_ppl_ratios",
            "per_ratio": {},
            "aggregate_ratio": None,
            "aggregate_pass": False,
            "contract_id": contract.get("quality_contract_id")
            or contract.get("contract_id"),
        }
    ppl_max = float(
        contract.get("ppl_ratio_max")
        or contract.get("candidate_ppl_ratio_max")
        or contract.get("default_ppl_ratio_max")
        or 1.01
    )
    aggregate_max = float(contract.get("aggregate_ppl_ratio_max", ppl_max) or ppl_max)
    per_ratio = {name: float(value) <= ppl_max for name, value in ratios.items()}
    aggregate = max(float(value) for value in ratios.values())
    aggregate_pass = aggregate <= aggregate_max
    passed = all(per_ratio.values()) and aggregate_pass
    return {
        "pass": passed,
        "incomplete": False,
        "ppl_ratio_max": ppl_max,
        "aggregate_ppl_ratio_max": aggregate_max,
        "per_ratio": per_ratio,
        "aggregate_ratio": aggregate,
        "aggregate_pass": aggregate_pass,
        "contract_id": contract.get("quality_contract_id")
        or contract.get("contract_id"),
    }


def records_to_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fields = [
        "id",
        "engine",
        "target_tokens",
        "nll",
        "avg_nll",
        "first_match",
        "greedy_lcp",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=fields, extrasaction="ignore", delimiter="\t"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})
    return buf.getvalue()
