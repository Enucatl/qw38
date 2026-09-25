"""Select the frozen 54-case routine core from the 216-case inventory."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

CORE_SPEC = (
    Path(__file__).resolve().parents[1] / "docs/implementation/eval-core-54.json"
)


def select_cases(
    root: Path, full: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Validate the source inventory and return its full or routine core."""
    prompts_path = root / "prompts.jsonl"
    rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
    ]
    core = [
        row
        for row in rows
        if row["family"] in {"P100", "C92", "L12"}
        or row["family"] == "R"
        and row["details"]["horizon"] in {512, 4096}
    ]
    if (
        len(core) != 216
        or Counter(row["family"] for row in core)
        != Counter({"P100": 100, "C92": 92, "L12": 12, "R": 12})
        or len({row["id"] for row in core}) != 216
    ):
        raise ValueError("frozen 216-case source inventory differs")
    if full:
        return core, None
    spec = json.loads(CORE_SPEC.read_text(encoding="utf-8"))
    if (
        hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        != spec["source_prompts_sha256"]
    ):
        raise ValueError("54-case sample source prompt hash differs")
    rng = random.Random(spec["sampling"]["seed"])
    selected_ids = set()
    for family in ("P100", "C92"):
        selected_ids.update(
            row["id"]
            for row in rng.sample([row for row in core if row["family"] == family], 15)
        )
    selected_ids.update(row["id"] for row in core if row["family"] in {"L12", "R"})
    selected = [row for row in core if row["id"] in selected_ids]
    if [row["id"] for row in selected] != spec["case_ids"] or Counter(
        row["family"] for row in selected
    ) != Counter({"P100": 15, "C92": 15, "L12": 12, "R": 12}):
        raise ValueError("frozen 54-case sample differs from its selection rule")
    return selected, spec
