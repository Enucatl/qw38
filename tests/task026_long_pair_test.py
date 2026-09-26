# /// script
# requires-python = "==3.12.*"
# dependencies = ["transformers==5.17.0"]
# ///
"""Negative lineage checks for the fixed long-context case scorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task026_long_pair import score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--llama-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    args = parser.parse_args()
    assert score(args)["status"] == "PASS"
    case_id = "R-32768-s0-d0.1"
    targets = {
        "inventory": args.fixtures / "prompts.jsonl",
        "same_length_prompt": args.fixtures / "tokens" / f"{case_id}.prompt.u32le",
        "candidate_tsv": args.candidate_run / "long.tsv",
        "generated_output": args.candidate_run / f"{case_id}.generated.u32le",
    }
    original = Path.read_bytes
    for label, target in targets.items():
        def changed(path: Path) -> bytes:
            raw = original(path)
            return bytes([raw[0] ^ 1]) + raw[1:] if path == target else raw
        with patch.object(Path, "read_bytes", changed):
            try:
                score(args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"altered {label} was accepted")
    print("long scorer PASS; four altered-input cases INVALID")


if __name__ == "__main__":
    main()
