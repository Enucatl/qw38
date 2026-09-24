# /// script
# requires-python = "==3.12.*"
# ///
"""Prepare and check a bounded full-model TASK-018 development smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASE_IDS = ("case_000", "case_080")
GENERATION_CAP = 8


def prepare(core: Path, output: Path) -> None:
    """Write two frozen prompt rows with short, generation-only caps."""
    rows = {}
    for line in core.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError("malformed frozen core row")
        if fields[0] in CASE_IDS:
            if fields[0] in rows:
                raise ValueError(f"duplicate frozen case: {fields[0]}")
            rows[fields[0]] = fields
    if set(rows) != set(CASE_IDS):
        raise ValueError("development smoke prompts are missing from frozen core")
    selected = []
    for case_id in CASE_IDS:
        prompt = Path(rows[case_id][1])
        if (
            not prompt.is_file()
            or prompt.stat().st_size % 4
            or not prompt.stat().st_size
        ):
            raise ValueError(f"invalid frozen prompt: {case_id}")
        selected.append(f"{case_id}\t{prompt.resolve()}\t-\t-\t{GENERATION_CAP}")
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")


def check(output: Path) -> None:
    """Require two complete generated cases and finite-logit execution."""
    rows = [
        json.loads(line)
        for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [row["id"] for row in rows] != list(CASE_IDS):
        raise ValueError("development smoke result inventory differs")
    for row in rows:
        if row["status"] != "complete" or row["target_tokens"] != 0:
            raise ValueError(f"incomplete development smoke case: {row['id']}")
        count = row["generation_tokens"]
        if not 1 <= count <= GENERATION_CAP:
            raise ValueError(f"invalid generation count: {row['id']}")
        if row["generation_stop"] not in {"eos", "cap"}:
            raise ValueError(f"invalid generation stop: {row['id']}")
        if row["generation_stop"] == "cap" and count != GENERATION_CAP:
            raise ValueError(f"early generation cap: {row['id']}")
        tokens = output / f"{row['id']}.generated.u32le"
        if tokens.stat().st_size != 4 * count:
            raise ValueError(f"generated token file differs: {row['id']}")
    record = {
        "status": "PASS_DIAGNOSTIC",
        "coverage": "partial",
        "purpose": "bounded full-model execution smoke; not TASK-018 quality acceptance",
        "case_ids": list(CASE_IDS),
        "generation_cap": GENERATION_CAP,
        "cases": rows,
    }
    (output / "dev-smoke.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, sort_keys=True))


def main() -> None:
    """Prepare a short case list or validate its evaluator output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.core is None:
            parser.error("prepare requires --core")
        prepare(args.core, args.output)
    else:
        check(args.output)


if __name__ == "__main__":
    main()
