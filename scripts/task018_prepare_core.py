# /// script
# requires-python = "==3.12.*"
# dependencies = ["transformers==5.17.0"]
# ///
"""Build routine or full evaluation TSVs from frozen fixture IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import transformers

try:
    from .task018_core_selection import CORE_SPEC, select_cases
except ImportError:
    from task018_core_selection import CORE_SPEC, select_cases


def sha256_file(path: Path) -> str:
    """Hash a small fixture-input manifest or case-list file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(fixtures: Path, output: Path, full: bool = False) -> dict[str, Any]:
    """Write the routine core or the manually requested full core."""
    root = fixtures.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["suite"] != "qw38-language-v2":
        raise ValueError("unexpected fixture suite")
    if manifest["reference_capture"]["status"] != "COMPLETE":
        raise ValueError("complete teacher references are required")
    if manifest.get("teacher_probability_capture", {}).get("status") != "COMPLETE":
        raise ValueError("complete top-20 teacher probability replay is required")
    refs = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (
                root / manifest["reference_capture"]["source_refs_jsonl"]["path"]
            )
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    core, _ = select_cases(root, full=full)
    counts = Counter(case["family"] for case in core)
    if len({case["id"] for case in core}) != len(core):
        raise ValueError("duplicate core case ID")

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        manifest["tokenizer"]["checkpoint"],
        trust_remote_code=True,
        local_files_only=True,
    )
    tsv: list[str] = []
    fixed_answer_targets: list[dict[str, Any]] = []
    for case in core:
        family = case["family"]
        prompt_path = (root / case["prompt_token_file"]).resolve()
        target_path: Path | None = None
        mask_path: Path | None = None
        if family in {"P100", "C92"}:
            ref = refs.get(case["id"])
            if ref is None:
                raise ValueError(f"missing frozen teacher target: {case['id']}")
            target_path = (root / ref["target_file"]).resolve()
            mask_path = (root / ref["loss_mask_file"]).resolve()
        elif family == "R":
            details = case["details"]
            target_path = (root / details["target_token_file"]).resolve()
            mask_path = (root / details["loss_mask_file"]).resolve()
        elif family == "L12":
            target_path = output.parent / "targets" / f"{case['id']}.target.u32le"
            mask_path = output.parent / "targets" / f"{case['id']}.loss-mask.u8"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_ids = tokenizer.encode(
                case["details"]["expected"], add_special_tokens=False
            )
            target_path.write_bytes(
                b"".join(int(token).to_bytes(4, "little") for token in target_ids)
            )
            mask_path.write_bytes(bytes([1]) * len(target_ids))
            fixed_answer_targets.append(
                {
                    "id": case["id"],
                    "expected": case["details"]["expected"],
                    "target_sha256": sha256_file(target_path),
                    "mask_sha256": sha256_file(mask_path),
                    "target_tokens": len(target_ids),
                }
            )
        cap = int(case["details"]["cap"])
        tsv.append(
            "\t".join(
                [
                    case["id"],
                    str(prompt_path),
                    str(target_path) if target_path else "-",
                    str(mask_path) if mask_path else "-",
                    str(cap),
                ]
            )
        )
    output.write_text("\n".join(tsv) + "\n", encoding="utf-8", newline="\n")
    record = {
        "suite": manifest["suite"],
        "fixture_manifest_sha256": sha256_file(root / "manifest.json"),
        "prompts_sha256": manifest["files"]["prompts.jsonl"]["sha256"],
        "teacher_refs_sha256": manifest["reference_capture"]["source_refs_jsonl"][
            "sha256"
        ],
        "teacher_probabilities_sha256": manifest["teacher_probability_capture"][
            "sha256"
        ],
        "core_cases": len(core),
        "coverage": "full" if full else "core-54",
        "sample_spec_sha256": None if full else sha256_file(CORE_SPEC),
        "families": dict(sorted(counts.items())),
        "retrieval_horizons": {"512": 6, "4096": 6},
        "l12_fixed_answer_targets": fixed_answer_targets,
        "case_tsv": str(output),
        "case_tsv_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> int:
    """Materialize the routine core or manually requested full input list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="prepare all 216 cases")
    args = parser.parse_args()
    print(json.dumps(prepare(args.fixtures, args.output, args.full), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
