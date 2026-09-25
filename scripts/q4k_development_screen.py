# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy"]
# ///
"""Freeze and score the Q4_K candidate's bounded Quartz development screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(".cache/task020")
OUT = Path(".cache/q4k-candidate/development")
COUNT = 8
VOCAB = 248320


def prepare() -> None:
    """Reuse eight frozen development windows and declare promotion criteria."""
    manifest_data = (BASE / "tokens/validation.manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    if manifest["role"] != "development-screening":
        raise ValueError("expected frozen development windows")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = []
    with (BASE / "development-cases.tsv").open() as stream:
        frozen = list(csv.reader(stream, delimiter="\t"))[:COUNT]
    windows = manifest["windows"][:COUNT]
    if len(frozen) != COUNT or len(windows) != COUNT:
        raise ValueError("incomplete frozen development inventory")
    for (case_id, filename, prompt_hash, target_hash), window in zip(
        frozen, windows, strict=True
    ):
        data = Path(filename).read_bytes()
        if (
            case_id != Path(window["tokens_file"]).stem
            or hashlib.sha256(data).hexdigest() != window["tokens_sha256"]
        ):
            raise ValueError("development case differs from frozen manifest")
        prompt, target = data[: 384 * 4], data[384 * 4 :]
        if len(data) != 512 * 4 or (
            hashlib.sha256(prompt).hexdigest() != prompt_hash
            or hashlib.sha256(target).hexdigest() != target_hash
        ):
            raise ValueError(f"frozen token identity mismatch: {case_id}")
        for suffix, payload in (
            ("prompt", prompt),
            ("target", target),
            ("mask", bytes([1]) * 128),
        ):
            (OUT / f"{case_id}.{suffix}").write_bytes(payload)
        paths = [
            str((OUT / f"{case_id}.{suffix}").resolve())
            for suffix in ("prompt", "target", "mask")
        ]
        rows.append("\t".join([case_id, *paths, "1"]))
        cases.append(
            {"id": case_id, "prompt_sha256": prompt_hash, "target_sha256": target_hash}
        )
    (OUT / "cases.tsv").write_text("\n".join(rows) + "\n")
    (OUT / "screen.json").write_text(
        json.dumps(
            {
                "purpose": "development only; not TASK-022 acceptance",
                "cases": cases,
                "target_tokens": COUNT * 128,
                "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
                "cases_tsv_sha256": hashlib.sha256(
                    (OUT / "cases.tsv").read_bytes()
                ).hexdigest(),
                "comparator_scores_sha256": hashlib.sha256(
                    (BASE / "gpu-existing-q4-k-m-scores.tsv").read_bytes()
                ).hexdigest(),
                "promotion": "Q4_K NLL below Q4G64 and comparator delta <= 0.03 nats/token",
                "runtime": "Quartz BF16 operands, FP32 accumulation, unchanged schedule",
            },
            indent=2,
        )
        + "\n"
    )


def validate_identities(identities: dict[str, dict]) -> None:
    """Reject arms that change source, Q8 weights, or runtime precision."""
    baseline, candidate = identities["q4g64"], identities["q4k"]
    if (
        baseline["precision_policy_id"] != 0x0402
        or candidate["precision_policy_id"] != 0x0403
    ):
        raise ValueError("screen arms have incorrect precision policy IDs")
    if (
        not baseline["compiler"]["ident"].startswith("qw38-candidate-v1-policy-")
        or candidate["compiler"]["ident"]
        != "qw38-candidate-v2-q4k-llama-e6ab7c1a4-no-imatrix-bf16-operands"
    ):
        raise ValueError("screen arms have incorrect compiler identities")
    if baseline["manifest_digest"] == candidate["manifest_digest"]:
        raise ValueError("screen arms use the same artifact")
    for field in (
        "source_hash",
        "config_hash",
        "tokenizer_hash",
        "tensor_count",
        "storage_counts",
        "decode_dispatch",
    ):
        if baseline[field] != candidate[field]:
            raise ValueError(f"screen arms differ in {field}")
    for field in ("source_hash", "config_hash", "tokenizer_hash", "manifest_digest"):
        for arm in (baseline, candidate):
            value = arm[field]
            if (
                len(value) != 64
                or set(value) == {"0"}
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"invalid artifact {field}")
    dispatch = {
        "activation_policy": "bf16",
        "quantized_kernel": "grouped_gemv",
        "fallback": None,
    }
    if candidate["decode_dispatch"] != dispatch:
        raise ValueError("screen requires the existing Quartz BF16 runtime")
    old = dict(baseline["logical_quantizer_counts"])
    new = dict(candidate["logical_quantizer_counts"])
    if (
        old.pop("q4_g64_candidate_v1", 0) != 192
        or old.pop("q4_k_candidate_v2", 0) != 0
        or new.pop("q4_k_candidate_v2", 0) != 192
        or new.pop("q4_g64_candidate_v1", 0) != 0
    ):
        raise ValueError("screen requires exactly 192 changed MLP matrices")
    if old != new or old.get("q8_g32_candidate_v1", 0) <= 0:
        raise ValueError("screen changed non-MLP quantizer inventory")


def validate_screen(screen: dict) -> None:
    """Require the complete frozen case inventory and unchanged input files."""
    cases = screen["cases"]
    if (
        len(cases) != COUNT
        or len({case["id"] for case in cases}) != COUNT
        or screen["target_tokens"] != COUNT * 128
    ):
        raise ValueError("incomplete screen inventory")
    for key, path in (
        ("manifest_sha256", BASE / "tokens/validation.manifest.json"),
        ("cases_tsv_sha256", OUT / "cases.tsv"),
        ("comparator_scores_sha256", BASE / "gpu-existing-q4-k-m-scores.tsv"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != screen[key]:
            raise ValueError(f"frozen screen identity changed: {key}")
    for case in cases:
        for suffix, length in (("prompt", 384 * 4), ("target", 128 * 4)):
            data = (OUT / f"{case['id']}.{suffix}").read_bytes()
            if (
                len(data) != length
                or hashlib.sha256(data).hexdigest() != case[f"{suffix}_sha256"]
            ):
                raise ValueError(f"{suffix} identity changed")
            if np.any(np.frombuffer(data, dtype="<u4") >= VOCAB):
                raise ValueError(f"{suffix} contains an invalid token")
        if (OUT / f"{case['id']}.mask").read_bytes() != bytes([1]) * 128:
            raise ValueError("screen requires every target token to be scored")


def score_arm(arm: str, cases: list[dict[str, str]]) -> list[float]:
    """Compute stable FP64 target NLL from Quartz's actual FP32 logits."""
    records = [
        json.loads(line)
        for line in (OUT / arm / "cases.jsonl").read_text().splitlines()
    ]
    if [row["id"] for row in records] != [case["id"] for case in cases]:
        raise ValueError(f"incomplete case inventory: {arm}")
    scores = []
    for case, record in zip(cases, records, strict=True):
        if (
            record["status"] != "complete"
            or record["target_tokens"] != 128
            or record["prompt_tokens"] != 384
        ):
            raise ValueError(f"incomplete case: {record}")
        target_data = (OUT / f"{case['id']}.target").read_bytes()
        if hashlib.sha256(target_data).hexdigest() != case["target_sha256"]:
            raise ValueError("target identity changed")
        targets = np.frombuffer(target_data, dtype="<u4")
        path = OUT / arm / f"{case['id']}.candidate-logits.f32le"
        if path.stat().st_size != 128 * VOCAB * 4:
            raise ValueError(f"logit length differs: {path}")
        logits = np.memmap(path, dtype="<f4", mode="r", shape=(128, VOCAB))
        total = 0.0
        for row, target in zip(logits, targets, strict=True):
            values = np.asarray(row, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite runtime logits")
            peak = values.max()
            total += float(peak + np.log(np.exp(values - peak).sum()) - values[target])
        scores.append(total)
    return scores


def report() -> None:
    """Compare complete paired runtime arms with the frozen llama scores."""
    screen = json.loads((OUT / "screen.json").read_text())
    validate_screen(screen)
    identities = {
        arm: json.loads((OUT / arm / "candidate_identity.json").read_text())
        for arm in ("q4g64", "q4k")
    }
    validate_identities(identities)
    cases = screen["cases"]
    with (BASE / "gpu-existing-q4-k-m-scores.tsv").open() as stream:
        reference = {row["id"]: row for row in csv.DictReader(stream, delimiter="\t")}
    teacher = []
    for case in cases:
        row = reference[case["id"]]
        if (
            any(row[key] != case[key] for key in ("prompt_sha256", "target_sha256"))
            or int(row["target_tokens"]) != 128
            or int(row["prompt_tokens"]) != 384
        ):
            raise ValueError("comparator target identity differs")
        nll = float(row["nll"])
        if not np.isfinite(nll) or nll < 0:
            raise ValueError("comparator NLL must be finite and nonnegative")
        teacher.append(nll)
    q4g64 = score_arm("q4g64", cases)
    q4k = score_arm("q4k", cases)
    count = len(cases) * 128
    baseline, candidate, comparator = (
        sum(values) / count for values in (q4g64, q4k, teacher)
    )
    result = {
        **screen,
        "q4g64_nll": baseline,
        "q4k_nll": candidate,
        "comparator_nll": comparator,
        "q4k_minus_q4g64": candidate - baseline,
        "q4k_minus_comparator": candidate - comparator,
        "proceed_full_gate": candidate < baseline and candidate - comparator <= 0.03,
        "per_case": [
            {
                "id": c["id"],
                "q4g64_nll_sum": a,
                "q4k_nll_sum": b,
                "comparator_nll_sum": t,
            }
            for c, a, b, t in zip(cases, q4g64, q4k, teacher, strict=True)
        ],
        "artifact_identities": identities,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("per_case", "artifact_identities", "cases")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "report"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else report()
