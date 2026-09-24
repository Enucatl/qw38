"""Pairing and aggregation checks for development continuation scores."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.task020_compare_continuations import FIELDS, compare, read_scores


def write_scores(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a score table with the public tabular contract."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def row(case: str, tokens: int, nll: float, lcp: int) -> dict[str, str]:
    """Create one fixed-identity score case."""
    return {
        "id": case,
        "prompt_sha256": "a" * 64,
        "target_sha256": "b" * 64,
        "prompt_tokens": "384",
        "target_tokens": str(tokens),
        "nll": str(nll),
        "first_match": str(int(lcp > 0)),
        "greedy_lcp": str(lcp),
    }


def test_token_weighted_nll_and_pairing(tmp_path: Path) -> None:
    """Weight NLL by target count while counting case wins separately."""
    old_file, new_file = tmp_path / "old.tsv", tmp_path / "new.tsv"
    write_scores(old_file, [row("one", 1, 2.0, 0), row("two", 9, 9.0, 4)])
    write_scores(new_file, [row("one", 1, 1.0, 1), row("two", 9, 11.0, 2)])
    result = compare(read_scores(old_file), read_scores(new_file))
    assert result["delta_candidate_minus_reference_nats_per_token"] == 0.1
    assert result["case_wins"] == {"candidate": 1, "reference": 1, "ties": 0}
    assert result["first_token_matches"] == {"candidate": 2, "reference": 1}


def test_different_target_identity_is_rejected(tmp_path: Path) -> None:
    """Reject a paired score when token counts match but target bytes differ."""
    old_file, new_file = tmp_path / "old.tsv", tmp_path / "new.tsv"
    write_scores(old_file, [row("one", 3, 1.0, 0)])
    changed = row("one", 3, 1.0, 0)
    changed["target_sha256"] = "c" * 64
    write_scores(new_file, [changed])
    with pytest.raises(ValueError, match="paired target_sha256 differs"):
        compare(read_scores(old_file), read_scores(new_file))
