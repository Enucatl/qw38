#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.5.3"]
# ///
"""Score the RoPE-corrected artifact on the unchanged eight-window screen."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VOCAB = 248320
ROWS = 128


def digest(path: Path) -> str:
    """Return a small frozen input file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(logits_path: Path, targets: np.ndarray) -> float:
    """Compute FP64 NLL from complete FP32 logits and fixed target IDs."""
    if logits_path.stat().st_size != ROWS * VOCAB * 4:
        raise ValueError(f"unexpected logit size: {logits_path}")
    logits = np.memmap(logits_path, dtype="<f4", mode="r", shape=(ROWS, VOCAB))
    total = 0.0
    for row, target in zip(logits, targets, strict=True):
        values = np.asarray(row, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite logits: {logits_path}")
        peak = values.max()
        total += float(peak + np.log(np.exp(values - peak).sum()) - values[target])
    return total


def main() -> None:
    """Validate case identity and write a paired development result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("frozen_dir", type=Path)
    parser.add_argument("corrected_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text())
    cases = baseline["cases"]
    if len(cases) != 8 or baseline["target_tokens"] != 1024:
        raise ValueError("incomplete frozen development report")
    if digest(args.frozen_dir / "cases.tsv") != baseline["cases_tsv_sha256"]:
        raise ValueError("development cases changed")
    records = [
        json.loads(line)
        for line in (args.corrected_dir / "cases.jsonl").read_text().splitlines()
    ]
    if [record["id"] for record in records] != [case["id"] for case in cases]:
        raise ValueError("corrected run has incomplete or reordered cases")
    identity = json.loads((args.corrected_dir / "candidate_identity.json").read_text())
    old_identity = baseline["artifact_identities"]["q4k"]
    if (
        identity["compiler"] != {**old_identity["compiler"], "patch": 2}
        or identity["manifest_digest"] == old_identity["manifest_digest"]
        or any(
            identity[field] != old_identity[field]
            for field in (
                "source_hash",
                "config_hash",
                "tokenizer_hash",
                "precision_policy_id",
                "tensor_count",
                "storage_counts",
                "logical_quantizer_counts",
                "decode_dispatch",
            )
        )
    ):
        raise ValueError("corrected artifact identity differs beyond RoPE revision")
    per_case = []
    for case, record, old in zip(cases, records, baseline["per_case"], strict=True):
        if (
            record["status"] != "complete"
            or record["prompt_tokens"] != 384
            or record["target_tokens"] != ROWS
            or old["id"] != case["id"]
        ):
            raise ValueError(f"incomplete corrected case: {case['id']}")
        target_path = args.frozen_dir / f"{case['id']}.target"
        prompt_path = args.frozen_dir / f"{case['id']}.prompt"
        if (
            digest(target_path) != case["target_sha256"]
            or digest(prompt_path) != case["prompt_sha256"]
        ):
            raise ValueError(f"frozen token identity changed: {case['id']}")
        targets = np.fromfile(target_path, dtype="<u4")
        if targets.size != ROWS or np.any(targets >= VOCAB):
            raise ValueError(f"invalid targets: {case['id']}")
        nll = score(
            args.corrected_dir / f"{case['id']}.candidate-logits.f32le", targets
        )
        per_case.append(
            {
                "id": case["id"],
                "corrected_nll_sum": nll,
                "old_q4k_nll_sum": old["q4k_nll_sum"],
                "comparator_nll_sum": old["comparator_nll_sum"],
            }
        )
    corrected = sum(row["corrected_nll_sum"] for row in per_case) / 1024
    old = baseline["q4k_nll"]
    comparator = baseline["comparator_nll"]
    report = {
        "schema": "task022-rope-fixed-development-v1",
        "purpose": "diagnostic eight-window screen; not TASK-022 acceptance",
        "cases_tsv_sha256": baseline["cases_tsv_sha256"],
        "target_tokens": 1024,
        "corrected_artifact_identity": identity,
        "corrected_nll": corrected,
        "old_q4k_nll": old,
        "comparator_nll": comparator,
        "corrected_minus_old": corrected - old,
        "corrected_minus_comparator": corrected - comparator,
        "per_case": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "corrected_nll",
                    "old_q4k_nll",
                    "comparator_nll",
                    "corrected_minus_old",
                    "corrected_minus_comparator",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
