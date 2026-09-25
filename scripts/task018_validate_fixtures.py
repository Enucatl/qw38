"""Validate immutable input identities and complete EVAL-01 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash a stored fixture or metadata file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_u32le(path: Path) -> list[int]:
    """Read little-endian uint32 elements and reject truncated arrays."""
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid uint32 array length: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows and reject malformed or empty files."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def validate_source_refs(
    root: Path, manifest: dict[str, Any], prompts: dict[str, Any]
) -> int:
    """Validate all frozen source targets when the capture is complete."""
    capture = manifest["reference_capture"]
    if capture["status"] != "COMPLETE":
        return 0
    record = capture["source_refs_jsonl"]
    refs_path = root / record["path"]
    if (
        refs_path.stat().st_size != record["bytes"]
        or sha256_file(refs_path) != record["sha256"]
    ):
        raise ValueError("source-reference JSONL identity mismatch")
    rows = load_jsonl(refs_path)
    expected = {
        case_id
        for case_id, case in prompts.items()
        if case["family"] in {"P100", "C92"}
    }
    if len(rows) != 192 or {row["id"] for row in rows} != expected:
        raise ValueError("source-reference inventory must be exactly P100+C92")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source-reference case ID")
    eos_ids = set(manifest["source_teacher"]["stop_token_ids"])
    for row in rows:
        case = prompts[row["id"]]
        if row["prompt_token_sha256"] != case["prompt_token_sha256"]:
            raise ValueError(f"source prompt binding mismatch: {row['id']}")
        target_path = root / row["target_file"]
        mask_path = root / row["loss_mask_file"]
        target_raw, mask_raw = target_path.read_bytes(), mask_path.read_bytes()
        if (
            len(target_raw) != row["target_bytes"]
            or sha256_file(target_path) != row["target_sha256"]
            or len(mask_raw) != row["loss_mask_bytes"]
            or sha256_file(mask_path) != row["loss_mask_sha256"]
            or len(target_raw) != row["target_count"] * 4
            or len(mask_raw) != row["target_count"]
            or mask_raw != bytes([1]) * row["target_count"]
            or not 1 <= row["target_count"] <= 24
        ):
            raise ValueError(f"source target/mask invalid: {row['id']}")
        target_ids = read_u32le(target_path)
        if row["stop_reason"] == "eos" and target_ids[-1] not in eos_ids:
            raise ValueError(f"EOS stop omits its EOS target: {row['id']}")
        if row["stop_reason"] == "limit" and row["target_count"] != 24:
            raise ValueError(f"generation limit differs from 24-token cap: {row['id']}")
        if row["stop_reason"] not in {"eos", "limit"}:
            raise ValueError(f"unknown source stop reason: {row['id']}")
    return len(rows)


def validate_teacher_probabilities(
    root: Path, manifest: dict[str, Any], prompts: dict[str, Any]
) -> int:
    """Validate top-20 probabilities against immutable teacher target IDs."""
    capture = manifest.get("teacher_probability_capture")
    if not capture or capture.get("status") != "COMPLETE":
        return 0
    probability_path = root / capture["path"]
    if (
        probability_path.stat().st_size != capture["bytes"]
        or sha256_file(probability_path) != capture["sha256"]
        or capture["n_probs"] != 20
        or capture["full_vocabulary_logits_available"] is not False
    ):
        raise ValueError("teacher top-20 probability manifest/hash mismatch")
    refs_path = root / manifest["reference_capture"]["source_refs_jsonl"]["path"]
    refs = {
        row["id"]: row
        for row in load_jsonl(refs_path)
    }
    rows = load_jsonl(probability_path)
    if len(rows) != 192 or {row["id"] for row in rows} != set(refs):
        raise ValueError("teacher probability inventory must be exactly 192 refs")
    seen: set[str] = set()
    total_targets = total_hits = 0
    identity = manifest["reference_capture"]["teacher_identity_sha256"]
    for row in rows:
        case_id = row["id"]
        ref = refs[case_id]
        case = prompts[case_id]
        target_path = root / ref["target_file"]
        mask_path = root / ref["loss_mask_file"]
        target_ids = read_u32le(target_path)
        mask = mask_path.read_bytes()
        if (
            row["teacher_identity_sha256"] != identity
            or row["target_file_sha256"] != ref["target_sha256"]
            or row["loss_mask_sha256"] != ref["loss_mask_sha256"]
            or row["target_count"] != ref["target_count"]
            or len(row["positions"]) != len(target_ids)
            or len(mask) != len(target_ids)
            or mask != bytes([1]) * len(target_ids)
            or row["stop_reason"] != ref["stop_reason"]
            or case["family"] not in {"P100", "C92"}
        ):
            raise ValueError(f"teacher probability target binding mismatch: {case_id}")
        for token_id, position in zip(target_ids, row["positions"], strict=True):
            top_ids, top_logprobs = position["top20_ids"], position["top20_logprobs"]
            if (
                position["target_id"] != token_id
                or not isinstance(position["target_logprob"], (int, float))
                or not math.isfinite(position["target_logprob"])
                or position["target_logprob"] > 0.0
                or len(top_ids) != 20
                or len(top_logprobs) != 20
                or len(set(top_ids)) != 20
                or token_id not in top_ids
                or any(not isinstance(value, int) or not 0 <= value < 248320 for value in top_ids)
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value > 0.0
                    for value in top_logprobs
                )
            ):
                raise ValueError(f"invalid teacher top-20 row: {case_id}")
        if row["top20_target_hits"] != len(target_ids):
            raise ValueError(f"top-20 target coverage count mismatch: {case_id}")
        total_targets += len(target_ids)
        total_hits += row["top20_target_hits"]
        seen.add(case_id)
    if (
        len(seen) != 192
        or total_targets != capture["target_tokens"]
        or total_hits != capture["top20_target_hits"]
        or total_targets != total_hits
    ):
        raise ValueError("teacher top-20 target coverage totals mismatch")
    return len(rows)


def validate(
    root: Path, require_source_refs: bool, require_teacher_probabilities: bool
) -> dict[str, Any]:
    """Validate manifest, hashes, target masks and all selected case IDs."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = (root / "manifest.sha256").read_text().split()[0]
    if sha256_file(manifest_path) != sidecar:
        raise ValueError("manifest SHA-256 sidecar mismatch")
    if manifest["suite"] != "qw38-language-v2":
        raise ValueError("unexpected suite identity")
    repo_root = Path(__file__).resolve().parents[1]
    if manifest["policy_sha256"] != sha256_file(repo_root / manifest["policy_path"]):
        raise ValueError("evaluation policy identity mismatch")
    for family, vendor_name in (("p100", "p100.jsonl"), ("c92", "ds4_eval.c")):
        vendor = root / "source" / vendor_name
        if sha256_file(vendor) != manifest["sources"][family]["sha256"]:
            raise ValueError(f"vendored {family} source identity mismatch")
    license_path = root / "source" / "ds4.LICENSE"
    if sha256_file(license_path) != manifest["sources"]["p100"]["license_file_sha256"]:
        raise ValueError("vendored DS4 license identity mismatch")
    if (
        sha256_file(root / "source" / "p100.README.md")
        != manifest["sources"]["p100"]["readme_sha256"]
    ):
        raise ValueError("vendored P100 attribution README identity mismatch")
    prompts_path = root / "prompts.jsonl"
    prompts_file = manifest["files"]["prompts.jsonl"]
    if (
        prompts_path.stat().st_size != prompts_file["bytes"]
        or sha256_file(prompts_path) != prompts_file["sha256"]
    ):
        raise ValueError("prompt JSONL identity mismatch")
    scorer = repo_root / manifest["scoring_implementation"]["path"]
    if sha256_file(scorer) != manifest["scoring_implementation"]["sha256"]:
        raise ValueError("grader implementation identity mismatch")
    metrics = repo_root / manifest["metrics_implementation"]["path"]
    if sha256_file(metrics) != manifest["metrics_implementation"]["sha256"]:
        raise ValueError("metric implementation identity mismatch")
    materializer = repo_root / manifest["materializer_implementation"]["path"]
    if sha256_file(materializer) != manifest["materializer_implementation"]["sha256"]:
        raise ValueError("fixture materializer identity mismatch")
    teacher = manifest["source_teacher"]
    if (
        teacher["quantization"] != "Q4_K_M"
        or teacher["gguf_architecture"] != "qwen35"
        or teacher["gguf_file_type"] != 15
        or teacher["tokenizer_model"] != "gpt2"
        or teacher["tokenizer_template_sha256"]
        != manifest["tokenizer"]["files"]["chat_template.jinja"]["sha256"]
        or teacher["eos_token_ids"] != [248046]
        or teacher["stop_token_ids"] != [248044, 248046, 248063, 248064, 248065]
        or teacher["generation"]["stop_token_ids"] != teacher["stop_token_ids"]
        or teacher["generation"]["samplers"] != ["temperature"]
        or teacher["generation"]["max_tokens"] != 24
    ):
        raise ValueError(
            "Q4_K_M teacher, tokenizer, template or generation contract mismatch"
        )
    capture = manifest["reference_capture"]
    if capture["status"] == "COMPLETE":
        identity = {
            key: value
            for key, value in teacher.items()
            if key not in {"capture_attempt"}
        }
        identity_sha = hashlib.sha256(
            json.dumps(
                identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if capture["teacher_identity_sha256"] != identity_sha:
            raise ValueError("Q4_K_M teacher identity hash mismatch")
    prompts_rows = load_jsonl(prompts_path)
    prompts = {case["id"]: case for case in prompts_rows}
    if len(prompts) != 222 or len(prompts) != len(prompts_rows):
        raise ValueError("fixture case IDs are missing or duplicated")
    families = {key: [] for key in ("P100", "C92", "L12", "R")}
    for case in prompts_rows:
        if case["family"] not in families:
            raise ValueError(f"unknown fixture family: {case['family']}")
        families[case["family"]].append(case)
        prompt_path = root / case["prompt_token_file"]
        raw = prompt_path.read_bytes()
        tokens = read_u32le(prompt_path)
        if (
            len(tokens) != case["prompt_tokens"]
            or len(raw) != case["prompt_token_file_bytes"]
            or sha256_file(prompt_path) != case["prompt_token_sha256"]
        ):
            raise ValueError(f"prompt token array mismatch: {case['id']}")
        if case["family"] == "R":
            details = case["details"]
            target = root / details["target_token_file"]
            mask = root / details["loss_mask_file"]
            target_ids = read_u32le(target)
            target_mask = mask.read_bytes()
            if (
                len(target_ids) != details["target_token_count"]
                or sha256_file(target) != details["target_token_sha256"]
                or len(target_mask) != details["loss_mask_bytes"]
                or sha256_file(mask) != details["loss_mask_sha256"]
                or target_mask != bytes([1]) * len(target_ids)
            ):
                raise ValueError(f"retrieval target/mask mismatch: {case['id']}")
            if abs(case["prompt_tokens"] - details["horizon"]) > 64:
                raise ValueError(f"retrieval prompt misses horizon: {case['id']}")
    expected = {"P100": 100, "C92": 92, "L12": 12, "R": 18}
    if {name: len(cases) for name, cases in families.items()} != expected:
        raise ValueError("suite family counts differ from EVAL-01")
    c92_counts = {}
    for case in families["C92"]:
        source = case["details"]["source"]
        key = "COMPSEC" if source == "COMPSEC" else source
        c92_counts[key] = c92_counts.get(key, 0) + 1
    if c92_counts != {
        "GPQA Diamond": 25,
        "SuperGPQA": 25,
        "AIME2025": 25,
        "COMPSEC": 17,
    }:
        raise ValueError(f"C92 slice counts invalid: {c92_counts}")
    modified = [
        case
        for case in families["C92"]
        if case["details"].get("modified_question", False)
    ]
    if (
        len(modified) != 1
        or modified[0]["details"]["source"] != "GPQA Diamond"
        or modified[0]["details"]["source_label_in_ds4"] != "GPQA Diamond (modified)"
    ):
        raise ValueError("modified GPQA provenance is missing")
    horizons = {
        horizon: sum(case["details"]["horizon"] == horizon for case in families["R"])
        for horizon in (512, 4096, 32768)
    }
    if horizons != {512: 6, 4096: 6, 32768: 6}:
        raise ValueError(f"retrieval horizon inventory invalid: {horizons}")
    refs = validate_source_refs(root, manifest, prompts)
    if require_source_refs and refs != 192:
        raise ValueError(f"complete source references missing; validated {refs}/192")
    teacher_probabilities = validate_teacher_probabilities(root, manifest, prompts)
    if require_teacher_probabilities and teacher_probabilities != 192:
        raise ValueError(
            "complete teacher top-20 probabilities missing; "
            f"validated {teacher_probabilities}/192"
        )
    return {
        "suite": manifest["suite"],
        "manifest_sha256": sha256_file(manifest_path),
        "cases": len(prompts_rows),
        "families": expected,
        "retrieval_horizons": horizons,
        "validated_source_references": refs,
        "validated_teacher_probability_cases": teacher_probabilities,
        "source_capture_status": manifest["reference_capture"]["status"],
        "fixtures_valid": True,
    }


def main() -> int:
    """Validate fixture identities and report their explicit coverage state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--require-source-refs", action="store_true")
    parser.add_argument("--require-teacher-probabilities", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(
        args.fixtures,
        args.require_source_refs,
        args.require_teacher_probabilities,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
