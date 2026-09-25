"""Check the frozen routine-core selection and source identity."""

from collections import Counter
from pathlib import Path

import pytest

from scripts.task018_core_selection import CORE_SPEC, select_cases


def test_core_54_is_a_subset_of_the_frozen_full_inventory() -> None:
    """Keep the selected IDs, family counts and source ordering fixed."""
    root = Path(".cache/evaluation/qw38-language-v2")
    if not (root / "prompts.jsonl").is_file():
        pytest.skip("frozen evaluation fixtures are not installed")
    routine, spec = select_cases(root)
    full, _ = select_cases(root, full=True)
    assert spec is not None and CORE_SPEC.is_file()
    assert len(routine) == 54 and len(full) == 216
    assert Counter(row["family"] for row in routine) == {
        "P100": 15,
        "C92": 15,
        "L12": 12,
        "R": 12,
    }
    assert [row["id"] for row in routine] == spec["case_ids"]
    assert set(spec["case_ids"]) <= {row["id"] for row in full}
    assert Counter(
        row["details"]["source"] for row in routine if row["family"] == "C92"
    ) == {"COMPSEC": 4, "SuperGPQA": 4, "GPQA Diamond": 4, "AIME2025": 3}


def test_changed_source_inventory_is_rejected(tmp_path: Path) -> None:
    """Refuse to silently reuse sampled IDs with different source prompts."""
    root = Path(".cache/evaluation/qw38-language-v2")
    if not (root / "prompts.jsonl").is_file():
        pytest.skip("frozen evaluation fixtures are not installed")
    altered = tmp_path / "prompts.jsonl"
    altered.write_bytes((root / "prompts.jsonl").read_bytes() + b"\n")
    with pytest.raises(ValueError):
        select_cases(tmp_path)
