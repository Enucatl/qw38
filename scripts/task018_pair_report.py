# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.5.3", "transformers==5.17.0"]
# ///
"""Build identity-bound paired metrics and review records for TASK-018 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import transformers

try:
    from .task018_core_selection import CORE_SPEC, select_cases
    from .task018_metrics import CaseMetric, paired_case_bootstrap
    from .task018_scoring import Grade, grade_c92, grade_l12, grade_retrieval
except ImportError:
    from task018_core_selection import CORE_SPEC, select_cases
    from task018_metrics import CaseMetric, paired_case_bootstrap
    from task018_scoring import Grade, grade_c92, grade_l12, grade_retrieval

EOG_IDS = {248044, 248046, 248063, 248064, 248065}
VOCAB = 248_320


class IncompleteCoverageError(Exception):
    """Identify missing arm coverage before paired scoring starts."""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"misaligned token file: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def fixed_answer_target_path(
    v0_run: Path, fixtures: Path, case: dict[str, Any]
) -> Path:
    """Locate the frozen fixed-answer target used for candidate scoring."""
    if case["family"] == "L12":
        return v0_run / "targets" / f"{case['id']}.target.u32le"
    return fixtures / case["details"]["target_token_file"]


def unique_rows(path: Path, expected: int, label: str) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [row["id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise ValueError(f"{label} must contain {expected} unique case IDs")
    return dict(zip(ids, rows, strict=True))


def require_hash(path: Path, expected: str, label: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"{label} hash differs from frozen identity")


def validate_policy_rebind(
    root: Path, manifest: dict[str, Any], rebind_path: Path | None
) -> tuple[str, str | None]:
    """Bind preserved fixture bytes to the reconciled policy identity."""
    current = sha(Path(manifest["policy_path"]))
    previous = manifest["policy_sha256"]
    if current == previous:
        if rebind_path is not None:
            raise ValueError("policy rebind is unnecessary for matching identity")
        return current, None
    if rebind_path is None:
        raise ValueError("policy hash differs from frozen identity")
    rebind = json.loads(rebind_path.read_text(encoding="utf-8"))
    required = {
        "schema": "qw38-eval-policy-rebind-v1",
        "fixture_manifest_sha256": sha(root / "manifest.json"),
        "previous_policy_sha256": previous,
        "effective_policy_sha256": current,
        "prompts_sha256": manifest["files"]["prompts.jsonl"]["sha256"],
        "teacher_refs_sha256": manifest["reference_capture"]["source_refs_jsonl"][
            "sha256"
        ],
        "teacher_probabilities_sha256": manifest["teacher_probability_capture"][
            "sha256"
        ],
        "scoring_sha256": manifest["scoring_implementation"]["sha256"],
        "metrics_sha256": manifest["metrics_implementation"]["sha256"],
    }
    if any(rebind.get(key) != value for key, value in required.items()):
        raise ValueError(
            "policy rebind does not match preserved fixture and code identities"
        )
    return current, sha(rebind_path)


def require_mask(mask: bytes, targets: int, case_id: str) -> None:
    if len(mask) != targets or mask != bytes([1]) * targets:
        raise ValueError(f"loss mask count differs: {case_id}")


def require_target_alignment(
    reference: list[int], probability_rows: list[dict[str, Any]], case_id: str
) -> None:
    if (
        len(probability_rows) != len(reference)
        or [int(row["target_id"]) for row in probability_rows] != reference
    ):
        raise ValueError(
            f"teacher probability targets differ from frozen IDs: {case_id}"
        )


def require_replay_match(expected: list[int], replay: list[int], case_id: str) -> None:
    if replay != expected:
        raise ValueError(f"fresh llama request replay changed generated IDs: {case_id}")


def require_fresh_llama_generation(run_manifest: dict[str, Any]) -> None:
    """Reject unauthenticated reuse of generated files as a complete arm."""
    if run_manifest.get("reused_generation_from") is not None:
        raise ValueError(
            "llama generated outputs were reused without source-run provenance"
        )


def write_unscored_result(output: Path, status: str, reason: str) -> None:
    """Atomically record why paired scoring could not run."""
    output.mkdir(parents=True, exist_ok=True)
    invalid = {
        "status": status,
        "mode": "language-only",
        "mtp_enabled": False,
        "scoring_performed": False,
        "reason": reason,
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    temp = output / "result.json.tmp"
    temp.write_text(
        json.dumps(invalid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(output / "result.json")


def validate_lineage(
    root: Path,
    llama: Path,
    v0: Path,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    lm: dict[str, Any],
    vm: dict[str, Any],
    lrows: dict[str, dict[str, Any]],
    vrows: dict[str, dict[str, Any]],
    refs: dict[str, dict[str, Any]],
    probs: dict[str, dict[str, Any]],
    effective_policy_sha256: str,
) -> dict[str, Any]:
    """Reject stale or mixed run outputs before any paired score is computed."""
    fixture_hash = sha(root / "manifest.json")
    require_fresh_llama_generation(lm)
    if manifest["suite"] != "qw38-language-v2":
        raise ValueError("unexpected suite")
    require_hash(Path(manifest["policy_path"]), effective_policy_sha256, "policy")
    for key in ("scoring_implementation", "metrics_implementation"):
        require_hash(Path(manifest[key]["path"]), manifest[key]["sha256"], key)
    require_hash(
        root / "prompts.jsonl", manifest["files"]["prompts.jsonl"]["sha256"], "prompts"
    )
    if (
        lm.get("suite") != manifest["suite"]
        or lm.get("fixture_manifest_sha256") != fixture_hash
        or lm.get("policy_sha256") != manifest["policy_sha256"]
        or lm.get("prompts_sha256") != manifest["files"]["prompts.jsonl"]["sha256"]
        or lm.get("teacher_references_sha256")
        != manifest["reference_capture"]["source_refs_jsonl"]["sha256"]
        or lm.get("teacher_probability_sidecar_sha256")
        != manifest["teacher_probability_capture"]["sha256"]
        or lm.get("teacher_identity_sha256")
        != manifest["reference_capture"]["teacher_identity_sha256"]
        or lm.get("image_digest") != manifest["source_teacher"]["image_digest"]
        or lm.get("server_binary_sha256")
        != manifest["source_teacher"]["llama_server_binary_sha256"]
        or lm.get("model_size_bytes") != manifest["source_teacher"]["file_size_bytes"]
        or lm.get("cases_validated")
        != (216 if lm.get("evaluation_scope", "full-216") == "full-216" else len(cases))
    ):
        raise ValueError("llama run is not bound to current frozen fixture/teacher")
    replay_rows = lm.get("fresh_request_replay", [])
    if (
        {row.get("id") for row in replay_rows}
        != {"case_000", "recNu3MXkvWUzHZr9", "L01", "R-512-s0-d0.1"}
        or len(replay_rows) != 4
        or any(row.get("identical_ids") is not True for row in replay_rows)
    ):
        raise ValueError("llama fresh-request replay is incomplete")
    require_hash(llama / "cases.jsonl", lm["cases_jsonl_sha256"], "llama cases")
    core_meta_path = v0 / "core.manifest.json"
    require_hash(core_meta_path, vm["case_tsv_manifest_sha256"], "V0 core manifest")
    core_meta = json.loads(core_meta_path.read_text(encoding="utf-8"))
    core_path = v0 / "core.tsv"
    require_hash(core_path, vm["case_tsv_sha256"], "V0 core TSV")
    if (
        core_meta.get("suite") != manifest["suite"]
        or core_meta.get("fixture_manifest_sha256") != fixture_hash
        or core_meta.get("prompts_sha256")
        != manifest["files"]["prompts.jsonl"]["sha256"]
        or core_meta.get("teacher_refs_sha256")
        != manifest["reference_capture"]["source_refs_jsonl"]["sha256"]
        or core_meta.get("teacher_probabilities_sha256")
        != manifest["teacher_probability_capture"]["sha256"]
        or core_meta.get("case_tsv_sha256") != vm["case_tsv_sha256"]
        or core_meta.get("core_cases") != len(cases)
        or core_meta.get("sample_spec_sha256")
        != (sha(CORE_SPEC) if len(cases) == 54 else None)
        or (
            lm.get("cases_validated") == len(cases)
            and lm.get("sample_spec_sha256") != core_meta.get("sample_spec_sha256")
        )
    ):
        raise ValueError("V0 core is not bound to current frozen fixture/teacher")
    require_hash(v0 / "cases.jsonl", vm["cases_jsonl_sha256"], "V0 cases")
    require_hash(Path(vm["binary"]), vm["binary_sha256"], "V0 evaluator binary")
    if Path(vm["artifact"]).stat().st_size != vm["artifact_size_bytes"]:
        raise ValueError("V0 artifact size changed")
    if (
        json.loads((v0 / "candidate_identity.json").read_text())
        != vm["artifact_identity"]
    ):
        raise ValueError("V0 embedded artifact identity changed")
    fixed = {row["id"]: row for row in core_meta["l12_fixed_answer_targets"]}
    if len(fixed) != 12:
        raise ValueError("L12 fixed target inventory differs")
    tsv_lines = core_path.read_text(encoding="utf-8").splitlines()
    if len(tsv_lines) != len(cases):
        raise ValueError("V0 core TSV is incomplete")
    expected_ids = {case["id"] for case in cases}
    if set(lrows) != expected_ids or set(vrows) != expected_ids:
        raise ValueError("paired arm IDs differ from frozen core")
    for case, line in zip(cases, tsv_lines, strict=True):
        cid = case["id"]
        fields = line.split("\t")
        if (
            len(fields) != 5
            or fields[0] != cid
            or int(fields[4]) != case["details"]["cap"]
        ):
            raise ValueError(f"V0 core row differs from frozen case: {cid}")
        prompt = (root / case["prompt_token_file"]).resolve()
        require_hash(prompt, case["prompt_token_sha256"], f"prompt {cid}")
        if Path(fields[1]) != prompt:
            raise ValueError(f"V0 prompt path differs from frozen case: {cid}")
        if case["family"] in {"P100", "C92"}:
            target = (root / refs[cid]["target_file"]).resolve()
            mask = (root / refs[cid]["loss_mask_file"]).resolve()
            target_hash, mask_hash = (
                refs[cid]["target_sha256"],
                refs[cid]["loss_mask_sha256"],
            )
            if (
                refs[cid]["prompt_token_sha256"] != case["prompt_token_sha256"]
                or probs[cid]["target_file_sha256"] != target_hash
                or probs[cid]["loss_mask_sha256"] != mask_hash
            ):
                raise ValueError(f"teacher target lineage differs: {cid}")
        elif case["family"] == "R":
            details = case["details"]
            target = (root / details["target_token_file"]).resolve()
            mask = (root / details["loss_mask_file"]).resolve()
            target_hash, mask_hash = (
                details["target_token_sha256"],
                details["loss_mask_sha256"],
            )
        else:
            target = (v0 / "targets" / f"{cid}.target.u32le").resolve()
            mask = (v0 / "targets" / f"{cid}.loss-mask.u8").resolve()
            target_hash, mask_hash = (
                fixed[cid]["target_sha256"],
                fixed[cid]["mask_sha256"],
            )
            if fixed[cid]["expected"] != case["details"]["expected"]:
                raise ValueError(f"L12 answer key differs: {cid}")
        if Path(fields[2]).resolve() != target or Path(fields[3]).resolve() != mask:
            raise ValueError(f"V0 target/mask path differs from frozen case: {cid}")
        require_hash(target, target_hash, f"target {cid}")
        require_hash(mask, mask_hash, f"loss mask {cid}")
        require_mask(mask.read_bytes(), len(read_ids(target)), cid)
        lr, vr = lrows[cid], vrows[cid]
        if (
            lr.get("family") != case["family"]
            or lr.get("prompt_tokens") != case["prompt_tokens"]
            or lr.get("cap") != case["details"]["cap"]
            or vr.get("status") != "complete"
            or vr.get("prompt_tokens") != case["prompt_tokens"]
            or vr.get("target_tokens") != len(read_ids(target))
            or lr.get("generation_tokens") != len(lr.get("output_token_ids", []))
            or lr.get("generation_stop") not in {"eos", "limit"}
            or vr.get("generation_stop") not in {"eos", "cap"}
        ):
            raise ValueError(f"paired case metadata differs from frozen case: {cid}")
        ltoken = llama / lr["token_file"]
        require_hash(ltoken, lr["token_sha256"], f"llama tokens {cid}")
        if read_ids(ltoken) != lr["output_token_ids"]:
            raise ValueError(f"llama output IDs differ from run record: {cid}")
        l_ids = lr["output_token_ids"]
        cap = case["details"]["cap"]
        if (
            not 1 <= len(l_ids) <= cap
            or (lr["generation_stop"] == "limit" and len(l_ids) != cap)
            or (lr["generation_stop"] == "eos" and l_ids[-1] not in EOG_IDS)
        ):
            raise ValueError(f"llama stop/count differs from frozen cap: {cid}")
        vtoken = v0 / f"{cid}.generated.u32le"
        require_hash(
            vtoken, vm["generated_token_sha256"][vtoken.name], f"V0 tokens {cid}"
        )
        v_ids = read_ids(vtoken)
        if (
            len(v_ids) != vr["generation_tokens"]
            or not 1 <= len(v_ids) <= cap
            or (vr["generation_stop"] == "cap" and len(v_ids) != cap)
            or (vr["generation_stop"] == "eos" and v_ids[-1] not in EOG_IDS)
        ):
            raise ValueError(f"V0 generation count differs: {cid}")
        logits = v0 / f"{cid}.candidate-logits.f32le"
        require_hash(
            logits, vm["candidate_logits_sha256"][logits.name], f"V0 logits {cid}"
        )
    replay_path = v0 / "state-replay.json"
    require_hash(
        Path(vm["binary"]).with_name("qw38-state-replay"),
        vm["state_replay_binary_sha256"],
        "V0 replay binary",
    )
    for cid, key in (
        ("case_000", "state_replay_prompt_sha256"),
        ("recNu3MXkvWUzHZr9", "state_replay_interleave_sha256"),
    ):
        require_hash(
            root / "tokens" / f"{cid}.prompt.u32le",
            vm[key],
            f"state replay input {cid}",
        )
    if vm.get("state_replay_result_sha256") is None:
        return {"status": "INCONCLUSIVE", "reason": "state replay produced no result"}
    require_hash(replay_path, vm["state_replay_result_sha256"], "V0 state replay")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if (
        replay.get("artifact_manifest_digest")
        != vm["artifact_identity"]["manifest_digest"]
    ):
        raise ValueError("state replay used a different artifact")
    checkpoints = replay.get("checkpoints", [])
    required_lengths = [1, 3, 4, 63, 64, 65, 255, 256, 257]
    observed_lengths = [row.get("length") for row in checkpoints]
    if observed_lengths != required_lengths[: len(observed_lengths)] or any(
        row.get("suffix_tokens") != 8
        or not isinstance(row.get("reset_replay"), bool)
        or not isinstance(row.get("snapshot_restore"), bool)
        for row in checkpoints
    ):
        raise ValueError("state replay omitted a declared boundary or interleave")
    state = replay.get("status")
    if state not in {"PASS", "FAIL", "INCOMPLETE"}:
        raise ValueError("state replay has unknown status")
    if state == "PASS" and (
        observed_lengths != required_lengths
        or replay.get("interleave", {}).get("schedule") != "A,reset,B,reset,A"
    ):
        raise ValueError("state replay PASS omits a declared check")
    if (vm.get("state_replay_exit") == 0) != (state == "PASS"):
        raise ValueError("state replay status and process exit disagree")
    if state == "PASS" and (
        not all(row["reset_replay"] and row["snapshot_restore"] for row in checkpoints)
        or replay["interleave"].get("identical") is not True
    ):
        raise ValueError("state replay PASS conflicts with its checks")
    if state == "INCOMPLETE":
        replay["status"] = "INCONCLUSIVE"
        replay["reason"] = "state replay did not finish all declared checks"
    return replay


def decode(tokenizer: Any, ids: list[int]) -> str:
    if ids and ids[-1] in EOG_IDS:
        ids = ids[:-1]
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def grade(case: dict[str, Any], text: str, stop: str) -> Grade | None:
    family, details = case["family"], case["details"]
    if family == "C92":
        result = grade_c92(
            text, details["source"], details["answer"], len(details.get("choices", []))
        )
    elif family == "L12":
        result = grade_l12(text, details["expected"])
    elif family == "R":
        result = grade_retrieval(text, details["answer"])
    else:
        return None
    if stop in {"cap", "limit"}:
        return Grade(result.valid, result.extracted, False, "output_cap_exhausted")
    return result


def summarize_fixed_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require every fixed-key answer from both arms to be correct."""
    arms = {
        arm: {
            "correct": sum(bool(row[f"{arm}_answer"]["correct"]) for row in rows),
            "incorrect_ids": sorted(
                row["id"] for row in rows if not row[f"{arm}_answer"]["correct"]
            ),
            "cap_exhausted_ids": sorted(
                row["id"] for row in rows if row[f"{arm}_stop"] == "cap"
            ),
        }
        for arm in ("llama", "v0")
    }
    return {
        "cases": len(rows),
        "arms": arms,
        "status": "PASS"
        if all(arm["correct"] == len(rows) for arm in arms.values())
        else "FAIL",
    }


def review_gate(
    rows: list[dict[str, Any]], submitted: Path | None
) -> tuple[str, list[dict[str, Any]]]:
    """Require complete, output-bound human adjudication before closing P100."""
    if submitted is None:
        return "PENDING", rows
    ratings = unique_rows(submitted, len(rows), "P100 human reviews")
    if set(ratings) != {row["id"] for row in rows}:
        raise ValueError("P100 human review IDs differ from frozen outputs")
    statuses: list[str] = []
    resolved: list[dict[str, Any]] = []
    for row in rows:
        rating = ratings[row["id"]]
        for key in ("llama_text_sha256", "v0_text_sha256"):
            if rating.get(key) != row[key]:
                raise ValueError(f"P100 review output hash differs: {row['id']}/{key}")
        out = dict(row)
        for arm in ("llama", "v0"):
            review = rating.get(f"{arm}_review")
            if not isinstance(review, dict):
                raise TypeError(f"P100 {arm} review missing: {row['id']}")
            if (
                not isinstance(review.get("reviewer"), str)
                or not review["reviewer"].strip()
            ):
                raise ValueError(f"P100 reviewer identity missing: {row['id']}/{arm}")
            if review.get("decision") not in {
                "no_material_failure",
                "confirmed_failure",
                "unresolved",
            }:
                raise ValueError(f"P100 decision invalid: {row['id']}/{arm}")
            if (
                not isinstance(review.get("supporting_text"), str)
                or not review["supporting_text"].strip()
            ):
                raise ValueError(f"P100 supporting text missing: {row['id']}/{arm}")
            fields = (
                "requested_language_followed",
                "requested_format_followed",
                "question_addressed",
                "degeneration",
            )
            if any(
                review.get(field) not in {"yes", "no", "not_applicable"}
                for field in fields
            ):
                raise ValueError(f"P100 assessment value invalid: {row['id']}/{arm}")
            error = review.get("concrete_error")
            if not isinstance(error, dict) or error.get("present") not in {
                "yes",
                "no",
                "not_applicable",
            }:
                raise ValueError(
                    f"P100 concrete error value invalid: {row['id']}/{arm}"
                )
            has_negative = (
                any(review[field] == "no" for field in fields[:3])
                or review["degeneration"] == "yes"
                or error["present"] == "yes"
            )
            if has_negative and (
                not isinstance(review.get("negative_assessment_notes"), str)
                or not review["negative_assessment_notes"].strip()
            ):
                raise ValueError(
                    f"P100 negative assessment lacks note: {row['id']}/{arm}"
                )
            if error["present"] == "yes" and (
                not isinstance(error.get("explanation"), str)
                or not error["explanation"].strip()
            ):
                raise ValueError(
                    f"P100 concrete error lacks explanation: {row['id']}/{arm}"
                )
            if has_negative and review["decision"] == "no_material_failure":
                raise ValueError(
                    f"P100 negative assessment conflicts with decision: {row['id']}/{arm}"
                )
            out[f"{arm}_review"] = review
        llama_decision = out["llama_review"]["decision"]
        v0_decision = out["v0_review"]["decision"]
        if "unresolved" in (llama_decision, v0_decision):
            statuses.append("INCONCLUSIVE")
        elif (
            v0_decision == "confirmed_failure"
            and llama_decision == "no_material_failure"
        ):
            statuses.append("FAIL")
        else:
            statuses.append("PASS")
        resolved.append(out)
    return (
        "FAIL"
        if "FAIL" in statuses
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in statuses
        else "PASS"
    ), resolved


def nll(logits: np.ndarray, target: int) -> float:
    values = np.asarray(logits, dtype=np.float32)
    if (
        values.shape != (VOCAB,)
        or not np.isfinite(values).all()
        or not 0 <= target < VOCAB
    ):
        raise ValueError("invalid full-vocabulary logits or target")
    maximum = np.max(values)
    shifted = np.subtract(values, maximum, dtype=np.float32)
    normalizer = np.log(
        np.sum(np.exp(shifted, dtype=np.float32), dtype=np.float32), dtype=np.float32
    )
    return float(np.float64(maximum + normalizer - values[target]))


def top20(values: np.ndarray) -> list[int]:
    ids = np.arange(values.size, dtype=np.int64)
    return np.lexsort((ids, -values))[:20].tolist()


def bootstrap_nll(rows: list[CaseMetric], group: str) -> dict[str, float]:
    raw = paired_case_bootstrap(rows, metric_group=group)
    return {
        key: value
        for key, value in raw.items()
        if key.startswith("nll_") or key == "replicates"
    }


def bootstrap_accuracy(rows: list[CaseMetric], group: str) -> dict[str, float]:
    raw = paired_case_bootstrap(rows, metric_group=group)
    return {
        key: value
        for key, value in raw.items()
        if key.startswith("accuracy_") or key == "replicates"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--llama-run", type=Path, required=True)
    parser.add_argument("--v0-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-rebind", type=Path)
    parser.add_argument(
        "--reviews", type=Path, help="completed human P100 review JSONL"
    )
    args = parser.parse_args()
    root, llama, v0, output = (
        p.resolve() for p in (args.fixtures, args.llama_run, args.v0_run, args.output)
    )
    lm, vm = (
        json.loads((llama / "run_manifest.json").read_text()),
        json.loads((v0 / "result.json").read_text()),
    )
    if (
        lm.get("status") != "COMPLETE"
        or vm.get("evaluation_exit_code", vm.get("exit_code")) != 0
    ):
        raise IncompleteCoverageError(
            "paired report requires complete llama and V0 runs"
        )
    scope = vm.get("evaluation_scope", "full-216")
    llama_scope = lm.get("evaluation_scope", "full-216")
    if scope not in {"core-54", "full-216"} or llama_scope not in {scope, "full-216"}:
        raise ValueError("paired arms have incompatible evaluation scopes")
    expected_count = 54 if scope == "core-54" else 216
    vrows = unique_rows(v0 / "cases.jsonl", expected_count, "V0 run")
    all_cases = [
        json.loads(line) for line in (root / "prompts.jsonl").read_text().splitlines()
    ]
    cases, _ = select_cases(root, full=scope == "full-216")
    frozen_32768 = sum(
        c["family"] == "R" and c["details"]["horizon"] == 32768 for c in all_cases
    )
    if frozen_32768 != 6:
        raise ValueError("R32768 deferred inventory is not exactly six frozen cases")
    lrows = unique_rows(
        llama / "cases.jsonl",
        216 if llama_scope == "full-216" else expected_count,
        "llama run",
    )
    expected_llama_ids = (
        {c["id"] for c in select_cases(root, full=True)[0]}
        if llama_scope == "full-216"
        else {c["id"] for c in cases}
    )
    if set(lrows) != expected_llama_ids or set(vrows) != {c["id"] for c in cases}:
        raise ValueError("paired arm IDs differ from frozen core")
    lrows = {case["id"]: lrows[case["id"]] for case in cases}
    manifest = json.loads((root / "manifest.json").read_text())
    effective_policy_sha256, rebind_sha256 = validate_policy_rebind(
        root, manifest, args.policy_rebind
    )
    if (
        manifest["teacher_probability_capture"].get("n_probs") != 20
        or manifest["teacher_probability_capture"].get(
            "full_vocabulary_logits_available"
        )
        is not False
    ):
        raise ValueError(
            "teacher probability sidecar is not the declared truncated top-20 API output"
        )
    prob_path = root / manifest["teacher_probability_capture"]["path"]
    if sha(prob_path) != manifest["teacher_probability_capture"]["sha256"]:
        raise ValueError("teacher top-20 sidecar hash differs from fixture manifest")
    probs = unique_rows(prob_path, 192, "teacher probabilities")
    refs_meta = manifest["reference_capture"]["source_refs_jsonl"]["path"]
    if (
        sha(root / refs_meta)
        != manifest["reference_capture"]["source_refs_jsonl"]["sha256"]
    ):
        raise ValueError("teacher reference JSONL hash differs from fixture manifest")
    refs = unique_rows(root / refs_meta, 192, "teacher references")
    teacher_ids = {
        case["id"] for case in all_cases if case["family"] in {"P100", "C92"}
    }
    if len(teacher_ids) != 192 or set(refs) != teacher_ids or set(probs) != teacher_ids:
        raise ValueError(
            "teacher references/probabilities are not exactly the frozen 192 P100/C92 cases"
        )
    replay = validate_lineage(
        root,
        llama,
        v0,
        manifest,
        cases,
        lm,
        vm,
        lrows,
        vrows,
        refs,
        probs,
        effective_policy_sha256,
    )
    tokenizer_dir = Path(manifest["tokenizer"]["checkpoint"])
    for name, metadata in manifest["tokenizer"]["files"].items():
        if sha(tokenizer_dir / name) != metadata["sha256"]:
            raise ValueError(
                f"tokenizer file hash differs from fixture manifest: {name}"
            )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True, local_files_only=True
    )
    output.mkdir(parents=True, exist_ok=True)
    per_case: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    nll_cases: list[CaseMetric] = []
    nll_slice_cases: dict[str, list[CaseMetric]] = defaultdict(list)
    fixed_answer_nll: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"nll_sum": 0.0, "targets": 0, "cases": 0}
    )
    nll_by_prompt_length: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {"llama_sum": 0.0, "v0_sum": 0.0, "targets": 0, "cases": 0}
    )
    c92_by_source: dict[str, list[CaseMetric]] = defaultdict(list)
    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coverage = Counter()
    teacher_top20_hits = 0
    teacher_top20_overlap = 0
    teacher_top20_positions = 0
    all_greedy_equal = 0
    for case in cases:
        cid, family = case["id"], case["family"]
        lr, vr = lrows[cid], vrows[cid]
        l_ids = read_ids(llama / lr["token_file"])
        v_ids = read_ids(v0 / f"{cid}.generated.u32le")
        l_text, v_text = decode(tokenizer, l_ids), decode(tokenizer, v_ids)
        (output / f"{cid}.llama.txt").write_text(l_text, encoding="utf-8")
        (output / f"{cid}.v0.txt").write_text(v_text, encoding="utf-8")
        l_stop = "eos" if lr["generation_stop"] == "eos" else "cap"
        v_stop = "eos" if vr["generation_stop"] == "eos" else "cap"
        lg, vg = grade(case, l_text, l_stop), grade(case, v_text, v_stop)
        common = 0
        for a, b in zip(l_ids, v_ids):
            if a != b:
                break
            common += 1
        same = l_ids == v_ids
        all_greedy_equal += int(same)
        greedy_divergence = common if not same else None
        row: dict[str, Any] = {
            "id": cid,
            "family": family,
            "slices": case["details"].get("slices", []),
            "prompt_tokens": case["prompt_tokens"],
            "llama_stop": l_stop,
            "v0_stop": v_stop,
            "llama_tokens": len(l_ids),
            "v0_tokens": len(v_ids),
            "llama_text_file": f"{cid}.llama.txt",
            "v0_text_file": f"{cid}.v0.txt",
            "llama_text_sha256": sha(output / f"{cid}.llama.txt"),
            "v0_text_sha256": sha(output / f"{cid}.v0.txt"),
            "llama_token_file": str(llama / lr["token_file"]),
            "v0_token_file": str(v0 / f"{cid}.generated.u32le"),
            "llama_token_sha256": sha(llama / lr["token_file"]),
            "v0_token_sha256": sha(v0 / f"{cid}.generated.u32le"),
            "greedy_exact_match": same,
            "first_token_match": bool(l_ids and v_ids and l_ids[0] == v_ids[0]),
            "divergence_index": greedy_divergence,
            "common_prefix_tokens": common,
            "llama_answer": None
            if lg is None
            else {
                "valid": lg.valid,
                "extracted": lg.extracted,
                "correct": lg.correct,
                "reason": lg.reason,
            },
            "v0_answer": None
            if vg is None
            else {
                "valid": vg.valid,
                "extracted": vg.extracted,
                "correct": vg.correct,
                "reason": vg.reason,
            },
        }
        if family in {"C92", "L12", "R"}:
            row["expected_answer"] = case["details"].get(
                "answer", case["details"].get("expected")
            )
        if family in {"L12", "R"}:
            answer_groups[
                "L12" if family == "L12" else f"R-{case['details']['horizon']}"
            ].append(row)
        if family == "P100":
            ref = read_ids(root / refs[cid]["target_file"])
            p_common = 0
            for a, b in zip(ref, l_ids):
                if a != b:
                    break
                p_common += 1
            row["teacher_first_token_match"] = bool(
                ref and l_ids and ref[0] == l_ids[0]
            )
            row["teacher_divergence_index"] = (
                p_common if p_common < max(len(ref), len(l_ids)) else None
            )
            row["teacher_common_prefix_tokens"] = p_common
            pending_review = {
                "reviewer": None,
                "decision": None,
                "supporting_text": None,
                "requested_language_followed": None,
                "requested_format_followed": None,
                "question_addressed": None,
                "concrete_error": {"present": None, "explanation": None},
                "degeneration": None,
                "negative_assessment_notes": None,
            }
            review.append(
                {
                    "id": cid,
                    "prompt": case["user_prompt"],
                    "teacher_short": decode(tokenizer, ref),
                    "llama_full": l_text,
                    "v0_full": v_text,
                    "llama_text_sha256": row["llama_text_sha256"],
                    "v0_text_sha256": row["v0_text_sha256"],
                    "llama_review": pending_review.copy(),
                    "v0_review": pending_review.copy(),
                }
            )
        if family in {"P100", "C92"}:
            ref = read_ids(root / refs[cid]["target_file"])
            logits_path = v0 / f"{cid}.candidate-logits.f32le"
            expected_bytes = len(ref) * VOCAB * 4
            if logits_path.stat().st_size != expected_bytes:
                raise ValueError(f"candidate logits size mismatch for {cid}")
            mat = np.memmap(logits_path, mode="r", dtype="<f4", shape=(len(ref), VOCAB))
            prob = probs[cid]["positions"]
            effective_top_k = probs[cid]["effective_generation_settings"].get("n_probs")
            if effective_top_k != 20:
                raise ValueError(f"teacher effective n_probs is not 20: {cid}")
            require_target_alignment(ref, prob, cid)
            if any(
                len(p["top20_ids"]) != 20 or len(p["top20_logprobs"]) != 20
                for p in prob
            ):
                raise ValueError(
                    f"teacher top-k rows are not exactly 20 entries: {cid}"
                )
            sums = [0.0, 0.0]
            overlap_sum = target_hits = 0
            for i, target in enumerate(ref):
                candidate = np.asarray(mat[i], dtype=np.float32)
                sums[0] += -float(prob[i]["target_logprob"])
                sums[1] += nll(candidate, target)
                vtop = top20(candidate)
                overlap_sum += len(set(vtop) & set(prob[i]["top20_ids"]))
                target_hits += int(target in prob[i]["top20_ids"])
            del mat
            row["teacher_forced_nll"] = {
                "llama_sum": sums[0],
                "v0_sum": sums[1],
                "targets": len(ref),
                "delta_nats_per_token": (sums[1] - sums[0]) / len(ref),
            }
            row["top20"] = {
                "requested_n_probs": 20,
                "effective_n_probs": effective_top_k,
                "selected_target_logprobs": [float(p["target_logprob"]) for p in prob],
                "target_coverage": target_hits,
                "positions": len(ref),
                "mean_overlap_fraction": overlap_sum / (20 * len(ref)),
                "scope": "teacher-generated target positions; top-20 only",
                "full_vocabulary_kl": None,
            }
            row["candidate_logits_file"] = str(logits_path)
            row["candidate_logits_sha256"] = sha(logits_path)
            coverage[family] += len(ref)
            teacher_top20_hits += target_hits
            teacher_top20_overlap += overlap_sum
            teacher_top20_positions += len(ref)
            length_bin = int(case["prompt_tokens"])
            bin_row = nll_by_prompt_length[length_bin]
            bin_row["llama_sum"] = float(bin_row["llama_sum"]) + sums[0]
            bin_row["v0_sum"] = float(bin_row["v0_sum"]) + sums[1]
            bin_row["targets"] = int(bin_row["targets"]) + len(ref)
            bin_row["cases"] = int(bin_row["cases"]) + 1
            stratum = (
                case["details"].get("slices", [case["details"].get("source", family)])[
                    0
                ]
                if family == "P100"
                else case["details"]["source"]
            )
            metric_family = family if family == "P100" else f"C92/{stratum}"
            cm = CaseMetric(
                cid,
                metric_family,
                sums[0],
                sums[1],
                len(ref),
                int(bool(lg and lg.correct)),
                int(bool(vg and vg.correct)),
            )
            nll_cases.append(cm)
            if family == "C92":
                c92_by_source[stratum].append(cm)
                nll_slice_cases[f"C92 {stratum}"].append(cm)
            else:
                for slice_name in case["details"].get("slices", []):
                    nll_slice_cases[slice_name].append(cm)
        else:
            row["teacher_forced_nll"] = None
            row["teacher_forced_nll_reason"] = (
                "llama REST exposes selected-token log probabilities only for generated teacher continuations"
            )
            row["top20"] = None
            if family in {"L12", "R"}:
                target_path = fixed_answer_target_path(v0, root, case)
                fixed_targets = read_ids(target_path)
                logits_path = v0 / f"{cid}.candidate-logits.f32le"
                if logits_path.stat().st_size != len(fixed_targets) * VOCAB * 4:
                    raise ValueError(
                        f"fixed-answer candidate logits size mismatch for {cid}"
                    )
                fixed_logits = np.memmap(
                    logits_path,
                    mode="r",
                    dtype="<f4",
                    shape=(len(fixed_targets), VOCAB),
                )
                fixed_sum = sum(
                    nll(np.asarray(fixed_logits[i], dtype=np.float32), target)
                    for i, target in enumerate(fixed_targets)
                )
                del fixed_logits
                horizon_group = (
                    "L12" if family == "L12" else f"R-{case['details']['horizon']}"
                )
                agg = fixed_answer_nll[horizon_group]
                agg["nll_sum"] = float(agg["nll_sum"]) + fixed_sum
                agg["targets"] = int(agg["targets"]) + len(fixed_targets)
                agg["cases"] = int(agg["cases"]) + 1
                row["v0_fixed_answer_nll"] = {
                    "sum": fixed_sum,
                    "targets": len(fixed_targets),
                    "mean_nats_per_token": fixed_sum / len(fixed_targets),
                    "teacher_nll": None,
                }
        if lg is not None:
            coverage[f"{family}_llama_graded"] += 1
        if vg is not None:
            coverage[f"{family}_v0_graded"] += 1
        per_case.append(row)

    review_input_sha256 = sha(args.reviews.resolve()) if args.reviews else None
    qualitative_status, review = review_gate(
        review, args.reviews.resolve() if args.reviews else None
    )
    nll_total = (
        sum(x.nll_reference_sum for x in nll_cases),
        sum(x.nll_candidate_sum for x in nll_cases),
        sum(x.target_count for x in nll_cases),
    )
    nll_point = (nll_total[1] - nll_total[0]) / nll_total[2]
    aggregate_ci = bootstrap_nll(nll_cases, "TASK018-NLL/P100+C92")
    nll_slice = {}
    for group, rows in sorted(nll_slice_cases.items()):
        delta = (
            sum(x.nll_candidate_sum for x in rows)
            - sum(x.nll_reference_sum for x in rows)
        ) / sum(x.target_count for x in rows)
        nll_slice[group] = {
            "cases": len(rows),
            "targets": sum(x.target_count for x in rows),
            "delta_nats_per_token": delta,
            "bootstrap_95": bootstrap_nll(rows, f"TASK018-NLL/{group}"),
        }
    c92_summary = {}
    for source, rows in sorted(c92_by_source.items()):
        llama_correct = sum(x.correct_reference for x in rows)
        v0_correct = sum(x.correct_candidate for x in rows)
        accuracy_loss = (llama_correct - v0_correct) / len(rows)
        ci = bootstrap_accuracy(rows, f"TASK018-C92/{source}")
        source_cases = [
            r
            for r in per_case
            if r["family"] == "C92"
            and next(c for c in cases if c["id"] == r["id"])["details"]["source"]
            == source
        ]
        llama_pass_v0_fail = sorted(
            r["id"]
            for r in source_cases
            if r["llama_answer"]["correct"] and not r["v0_answer"]["correct"]
        )
        llama_fail_v0_pass = sorted(
            r["id"]
            for r in source_cases
            if not r["llama_answer"]["correct"] and r["v0_answer"]["correct"]
        )
        c92_summary[source] = {
            "cases": len(rows),
            "llama_correct": llama_correct,
            "v0_correct": v0_correct,
            "loss_llama_minus_v0": accuracy_loss,
            "paired_95": ci,
            "llama_pass_v0_fail": {
                "count": len(llama_pass_v0_fail),
                "ids": llama_pass_v0_fail,
            },
            "llama_fail_v0_pass": {
                "count": len(llama_fail_v0_pass),
                "ids": llama_fail_v0_pass,
            },
            "status": "PASS"
            if accuracy_loss <= 0.02
            else "FAIL"
            if ci["accuracy_loss_low_95"] > 0.02
            else "INCONCLUSIVE",
        }
    c92_rows = [r for r in per_case if r["family"] == "C92"]
    c92_llama = sum(r["llama_answer"]["correct"] for r in c92_rows)
    c92_v0 = sum(r["v0_answer"]["correct"] for r in c92_rows)
    c92_ci = bootstrap_accuracy(
        [x for rows in c92_by_source.values() for x in rows],
        "TASK018-C92/all",
    )
    c92_loss = (c92_llama - c92_v0) / len(c92_rows)
    all_llama_pass_v0_fail = sorted(
        r["id"]
        for r in c92_rows
        if r["llama_answer"]["correct"] and not r["v0_answer"]["correct"]
    )
    all_llama_fail_v0_pass = sorted(
        r["id"]
        for r in c92_rows
        if not r["llama_answer"]["correct"] and r["v0_answer"]["correct"]
    )
    c92_aggregate = {
        "cases": len(c92_rows),
        "llama_correct": c92_llama,
        "v0_correct": c92_v0,
        "loss_llama_minus_v0": c92_loss,
        "llama_accuracy_micro": c92_llama / len(c92_rows),
        "v0_accuracy_micro": c92_v0 / len(c92_rows),
        "llama_pass_v0_fail": {
            "count": len(all_llama_pass_v0_fail),
            "ids": all_llama_pass_v0_fail,
        },
        "llama_fail_v0_pass": {
            "count": len(all_llama_fail_v0_pass),
            "ids": all_llama_fail_v0_pass,
        },
        "paired_95": c92_ci,
        "status": "PASS"
        if c92_loss <= 0.02
        else "FAIL"
        if c92_ci["accuracy_loss_low_95"] > 0.02
        else "INCONCLUSIVE",
    }
    answer_summary = {}
    if {group: len(rows) for group, rows in answer_groups.items()} != {
        "L12": 12,
        "R-512": 6,
        "R-4096": 6,
    }:
        raise ValueError("fixed-answer core grade inventory is incomplete")
    for group, rows in sorted(answer_groups.items()):
        answer_summary[group] = summarize_fixed_answers(rows)
    state_status = replay["status"]
    nll_aggregate_status = "PASS" if nll_point <= 0.03 else "FAIL"
    nll_slice_status = {
        name: "PASS" if value["delta_nats_per_token"] <= 0.06 else "FAIL"
        for name, value in nll_slice.items()
    }
    mandatory_statuses = [
        nll_aggregate_status,
        *nll_slice_status.values(),
        c92_aggregate["status"],
        *(x["status"] for x in c92_summary.values()),
        *(x["status"] for x in answer_summary.values()),
        state_status,
        qualitative_status,
    ]
    if "FAIL" in mandatory_statuses:
        overall_status = "FAIL"
    elif any(value != "PASS" for value in mandatory_statuses):
        overall_status = "INCONCLUSIVE"
    else:
        overall_status = "PASS"
    candidate_policy = vm["artifact_identity"]["precision_policy_id"]
    candidate_names = {
        0x0401: "QW38 V0",
        0x0402: "QW38 CandidateV1",
        0x0403: "QW38 CandidateV2",
    }
    candidate_name = candidate_names[candidate_policy]
    task_name = "TASK-018" if candidate_policy == 0x0401 else "TASK-022"
    text_outputs = [
        {"id": row["id"], "llama": row["llama_text_file"], "v0": row["v0_text_file"]}
        for row in per_case
    ]
    summary = {
        "pair_report_driver_sha256": sha(Path(__file__).resolve()),
        "mode": "language-only",
        "mtp_enabled": False,
        "roles": {"comparator": "Q4_K_M llama.cpp", "candidate": candidate_name},
        "schedule": {
            "core_cases": f"{len(cases)} frozen {scope} P100/C92/L12/R512/R4096 cases once; reset at document boundaries",
            "target_alignment": "same frozen Q4_K_M teacher IDs for P100/C92; frozen fixed keys for L12/R",
            "retrieval_32768": "frozen now; execution assigned to TASK-026",
        },
        "precision": {
            "teacher_probability": "llama.cpp REST top-20 selected target logprobs only",
            "candidate_logits": "FP32 full vocabulary",
            "full_vocabulary_kl": "unavailable",
        },
        "raw_generated_text": text_outputs,
        "suite": manifest["suite"],
        "evaluation_scope": scope,
        "llama_evaluation_scope": llama_scope,
        "sample_spec_sha256": sha(CORE_SPEC) if scope == "core-54" else None,
        "policy_sha256": effective_policy_sha256,
        "capture_policy_sha256": manifest["policy_sha256"],
        "policy_rebind_sha256": rebind_sha256,
        "fixture_manifest_sha256": sha(root / "manifest.json"),
        "llama_run_manifest_sha256": sha(llama / "run_manifest.json"),
        "v0_result_sha256": sha(v0 / "result.json"),
        "teacher_identity_sha256": lm["teacher_identity_sha256"],
        "candidate_binary_sha256": vm["binary_sha256"],
        "core_tsv_sha256": vm["case_tsv_sha256"],
        "core_tsv_manifest_sha256": vm["case_tsv_manifest_sha256"],
        "candidate_schema_identity": vm["artifact_identity"],
        "core_cases": len(per_case),
        "family_counts": dict(sorted(Counter(x["family"] for x in per_case).items())),
        "teacher_target_nll": {
            "cases": len(nll_cases),
            "targets": nll_total[2],
            "delta_v0_minus_llama_nats_per_token": nll_point,
            "bootstrap_95": aggregate_ci,
            "slices": nll_slice,
            "full_vocabulary_kl": None,
            "full_vocabulary_kl_reason": "llama REST returned top-20 only",
        },
        "v0_fixed_answer_nll_diagnostic": {
            group: {
                "cases": value["cases"],
                "targets": value["targets"],
                "sum": value["nll_sum"],
                "mean_nats_per_token": float(value["nll_sum"]) / int(value["targets"]),
                "paired_teacher_nll": None,
                "paired_teacher_nll_reason": "llama REST provides target logprobs only for generated teacher continuations",
            }
            for group, value in sorted(fixed_answer_nll.items())
        },
        "teacher_target_nll_by_actual_prompt_tokens": {
            "bin_definition": "exact prompt token count; cases with the same frozen prompt length are grouped",
            "bins": {
                str(length): {
                    "cases": value["cases"],
                    "targets": value["targets"],
                    "delta_v0_minus_llama_nats_per_token": (
                        float(value["v0_sum"]) - float(value["llama_sum"])
                    )
                    / int(value["targets"]),
                }
                for length, value in sorted(nll_by_prompt_length.items())
            },
        },
        "top20": {
            "requested_n_probs": 20,
            "effective_n_probs": 20,
            "scope": "teacher-generated target positions only; truncated top-20 REST values",
            "selected_target_logprobs": {
                "positions": teacher_top20_positions,
                "source": manifest["teacher_probability_capture"]["path"],
            },
            "teacher_targets": sum(coverage[f] for f in ("P100", "C92")),
            "target_ids_returned": teacher_top20_hits,
            "mean_position_weighted_overlap_fraction": teacher_top20_overlap
            / (20 * teacher_top20_positions),
            "full_vocabulary_kl": None,
            "full_vocabulary_kl_reason": "teacher REST returned top-20 only",
        },
        "c92": {"aggregate": c92_aggregate, "slices": c92_summary},
        "fixed_answer_grades": answer_summary,
        "retrieval_coverage": {
            "512": {"status": "COVERED", "cases": len(answer_groups["R-512"])},
            "4096": {"status": "COVERED", "cases": len(answer_groups["R-4096"])},
            "32768": {
                "status": "DEFERRED",
                "cases_frozen": frozen_32768,
                "reason": "execution assigned to TASK-026 after production prefill",
            },
        },
        "greedy": {
            "cases_exactly_equal": all_greedy_equal,
            "cases": len(per_case),
            "full_vocabulary_kl": None,
        },
        "cap_truncated_ids": {
            "llama": sorted(r["id"] for r in per_case if r["llama_stop"] == "cap"),
            "v0": sorted(r["id"] for r in per_case if r["v0_stop"] == "cap"),
        },
        "grade_coverage": {
            k: v for k, v in sorted(coverage.items()) if k.endswith("_graded")
        },
        "fresh_llama_request_replay": lm["fresh_request_replay"],
        "v0_state_replay": replay,
        "quality_gates": {
            "nll_aggregate": {
                "threshold_nats_per_token": 0.03,
                "status": nll_aggregate_status,
            },
            "nll_slices": {
                name: {
                    "threshold_nats_per_token": 0.06,
                    "status": nll_slice_status[name],
                }
                for name in nll_slice
            },
            "c92_aggregate": c92_aggregate["status"],
            "c92_slices": {
                name: value["status"] for name, value in c92_summary.items()
            },
            "fixed_answer_groups": {
                name: value["status"] for name, value in answer_summary.items()
            },
            "qualitative_review": qualitative_status,
            "v0_state_replay": state_status,
        },
        "qualitative_review": qualitative_status,
        "overall_status": overall_status,
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    (output / "cases.jsonl").write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in per_case
        ),
        encoding="utf-8",
    )
    (output / "p100-review.jsonl").write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in review
        ),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        f"# {task_name} paired run\n\n"
        f"Status: **{overall_status}**.\n\n"
        f"Mode: `language-only`; MTP enabled: `false`. Comparator: Q4_K_M llama.cpp. Candidate: {candidate_name}. "
        f"Schedule: all {len(cases)} {scope} cases once, using frozen teacher IDs for P100/C92 and fixed keys for L12/R; "
        "R32768 execution is assigned to TASK-026. Precision: teacher top-20 selected target logprobs; candidate FP32 full-vocabulary logits.\n\n"
        f"Core coverage: {len(per_case)}/{len(cases)}. Teacher-forced NLL: {nll_total[2]} aligned P100/C92 tokens; "
        f"Candidate minus llama = {nll_point:.6f} nats/token. Full-vocabulary KL is unavailable.\n\n"
        f"C92: llama {c92_llama}/{len(c92_rows)}, candidate {c92_v0}/{len(c92_rows)}; R/L fixed-key grades and per-case details are in `cases.jsonl`.\n\n"
        f"P100 output review: {qualitative_status}; rows are in `p100-review.jsonl`.\n\n"
        "## Generated text\n\n"
        + "\n".join(
            f"- [{row['id']} llama]({row['llama_text_file']}) · [{row['id']} candidate]({row['v0_text_file']})"
            for row in per_case
        )
        + "\n",
        encoding="utf-8",
    )
    report_manifest = {
        "pair_report_driver_sha256": sha(Path(__file__).resolve()),
        "mode": "language-only",
        "mtp_enabled": False,
        "roles": {"comparator": "Q4_K_M llama.cpp", "candidate": candidate_name},
        "schedule": summary["schedule"],
        "precision": summary["precision"],
        "raw_generated_text": text_outputs,
        "suite": manifest["suite"],
        "evaluation_scope": scope,
        "llama_evaluation_scope": llama_scope,
        "sample_spec_sha256": sha(CORE_SPEC) if scope == "core-54" else None,
        "fixture_manifest_sha256": sha(root / "manifest.json"),
        "policy_sha256": effective_policy_sha256,
        "capture_policy_sha256": manifest["policy_sha256"],
        "policy_rebind_sha256": rebind_sha256,
        "llama_run_manifest_sha256": sha(llama / "run_manifest.json"),
        "v0_result_sha256": sha(v0 / "result.json"),
        "paired_summary_sha256": sha(output / "summary.json"),
        "teacher_identity_sha256": lm["teacher_identity_sha256"],
        "llama_image_digest": lm["image_digest"],
        "candidate_binary_sha256": vm["binary_sha256"],
        "candidate_schema_identity": vm["artifact_identity"],
        "candidate_case_tsv_sha256": vm["case_tsv_sha256"],
        "candidate_case_tsv_manifest_sha256": vm["case_tsv_manifest_sha256"],
        "tokenizer_file_sha256": {
            name: row["sha256"] for name, row in manifest["tokenizer"]["files"].items()
        },
        "paired_cases_sha256": sha(output / "cases.jsonl"),
        "p100_review_sha256": sha(output / "p100-review.jsonl"),
        "p100_review_input_sha256": review_input_sha256,
        "report_sha256": sha(output / "report.md"),
        "overall_status": overall_status,
    }
    (output / "manifest.json").write_text(
        json.dumps(report_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["overall_status"],
                "cases": len(per_case),
                "nll_targets": nll_total[2],
                "summary_sha256": sha(output / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        import sys

        output_arg = None
        try:
            output_arg = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
        except (ValueError, IndexError):
            pass
        if output_arg is None:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        write_unscored_result(
            output_arg,
            "INCONCLUSIVE" if isinstance(exc, IncompleteCoverageError) else "INVALID",
            reason,
        )
        invalid = json.loads((output_arg / "result.json").read_text(encoding="utf-8"))
        print(json.dumps(invalid, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
