#!/usr/bin/env python3
"""Compare paired model scores on frozen development continuations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = (
    "id",
    "prompt_sha256",
    "target_sha256",
    "prompt_tokens",
    "target_tokens",
    "nll",
    "first_match",
    "greedy_lcp",
)


def read_scores(path: Path) -> dict[str, dict[str, str]]:
    """Read a complete score table, rejecting malformed or repeated cases."""
    cases: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"score columns differ: {path}")
        for row in reader:
            case_id = row["id"]
            if not case_id or case_id in cases or None in row:
                raise ValueError(f"empty or repeated score case: {case_id}")
            for identity in ("prompt_sha256", "target_sha256"):
                value = row[identity]
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"invalid {identity}: {case_id}")
            prompt = int(row["prompt_tokens"])
            targets = int(row["target_tokens"])
            nll = float(row["nll"])
            first = int(row["first_match"])
            lcp = int(row["greedy_lcp"])
            if (
                prompt < 1
                or targets < 1
                or not math.isfinite(nll)
                or nll < 0
                or first not in (0, 1)
                or not 0 <= lcp <= targets
                or first != int(lcp > 0)
            ):
                raise ValueError(f"invalid score values: {case_id}")
            cases[case_id] = row
    if not cases:
        raise ValueError(f"empty score table: {path}")
    return cases


def compare(
    reference: dict[str, dict[str, str]], candidate: dict[str, dict[str, str]]
) -> dict[str, object]:
    """Calculate DS4-style paired NLL and greedy diagnostics."""
    if reference.keys() != candidate.keys():
        raise ValueError("reference and candidate case sets differ")
    cases = []
    reference_nll = candidate_nll = 0.0
    reference_first = candidate_first = 0
    reference_lcp = candidate_lcp = 0
    target_total = 0
    candidate_wins = reference_wins = ties = 0
    for case_id in sorted(reference):
        old, new = reference[case_id], candidate[case_id]
        for key in ("prompt_sha256", "target_sha256", "prompt_tokens", "target_tokens"):
            if old[key] != new[key]:
                raise ValueError(f"paired {key} differs: {case_id}")
        tokens = int(old["target_tokens"])
        old_nll, new_nll = float(old["nll"]), float(new["nll"])
        delta = new_nll - old_nll
        target_total += tokens
        reference_nll += old_nll
        candidate_nll += new_nll
        reference_first += int(old["first_match"])
        candidate_first += int(new["first_match"])
        reference_lcp += int(old["greedy_lcp"])
        candidate_lcp += int(new["greedy_lcp"])
        if delta < -1e-9:
            candidate_wins += 1
        elif delta > 1e-9:
            reference_wins += 1
        else:
            ties += 1
        cases.append(
            {
                "id": case_id,
                "target_tokens": tokens,
                "reference_nll": old_nll,
                "candidate_nll": new_nll,
                "delta_nll": delta,
            }
        )
    return {
        "schema": "qw38-task020-development-comparison-v1",
        "status": "DIAGNOSTIC_ONLY",
        "cases": len(cases),
        "target_tokens": target_total,
        "reference_avg_nll": reference_nll / target_total,
        "candidate_avg_nll": candidate_nll / target_total,
        "delta_candidate_minus_reference_nats_per_token": (
            candidate_nll - reference_nll
        )
        / target_total,
        "case_wins": {
            "candidate": candidate_wins,
            "reference": reference_wins,
            "ties": ties,
        },
        "first_token_matches": {
            "candidate": candidate_first,
            "reference": reference_first,
        },
        "mean_greedy_lcp": {
            "candidate": candidate_lcp / len(cases),
            "reference": reference_lcp / len(cases),
        },
        "per_case": cases,
    }


def main() -> None:
    """Compare two exact score tables and write diagnostic JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(read_scores(args.reference), read_scores(args.candidate))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "cases",
                    "target_tokens",
                    "delta_candidate_minus_reference_nats_per_token",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
