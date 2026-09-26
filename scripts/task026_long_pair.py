# /// script
# requires-python = "==3.12.*"
# dependencies = ["transformers==5.17.0"]
# ///
"""Score the fixed R-32768 case in both comparison arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import transformers

from task018_core_selection import select_long_cases
from task018_metrics import CaseMetric, paired_case_bootstrap
from task018_scoring import grade_retrieval

EOG_IDS = {248044, 248046, 248063, 248064, 248065}
REPO = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid token file: {path}")
    return [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]


def rows(path: Path, expected: list[str]) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [row["id"] for row in records] != expected:
        raise ValueError(f"case order or coverage differs: {path}")
    return {row["id"]: row for row in records}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--llama-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def score(args: argparse.Namespace) -> dict:
    fixture_path = args.fixtures / "manifest.json"
    fixture = json.loads(fixture_path.read_text())
    llama_path = args.llama_run / "run_manifest.json"
    llama = json.loads(llama_path.read_text())
    candidate_path = args.candidate_run / "result.json"
    candidate = json.loads(candidate_path.read_text())
    tsv_path = args.candidate_run / "long.tsv"
    tsv_manifest_path = args.candidate_run / "long.manifest.json"
    tsv_manifest = json.loads(tsv_manifest_path.read_text())
    rebind_path = REPO / "docs/implementation/task026-eval-policy-rebind.json"
    rebind = json.loads(rebind_path.read_text())
    fixture_sha = sha(fixture_path)
    inventory_sha = sha(args.fixtures / "prompts.jsonl")
    require(fixture["suite"] == "qw38-language-v2" and
            inventory_sha == fixture["files"]["prompts.jsonl"]["sha256"] and
            rebind["fixture_manifest_sha256"] == fixture_sha and
            rebind["prompts_sha256"] == inventory_sha and
            rebind["previous_policy_sha256"] == fixture["policy_sha256"] and
            rebind["effective_policy_sha256"] ==
            sha(REPO / "docs/architecture/evaluation-policy-v0.md") and
            sha(REPO / fixture["scoring_implementation"]["path"]) ==
            fixture["scoring_implementation"]["sha256"] and
            sha(REPO / fixture["metrics_implementation"]["path"]) ==
            fixture["metrics_implementation"]["sha256"],
            "fixture, policy or grader identity differs")
    cases = select_long_cases(args.fixtures)
    expected = [case["id"] for case in cases]
    require(expected == ["R-32768-s0-d0.1"],
            "frozen long case IDs differ")
    require(llama["status"] == "COMPLETE" and
            llama["evaluation_scope"] == "long-32768" and
            llama["fixture_manifest_sha256"] == fixture_sha and
            llama["prompts_sha256"] == inventory_sha and
            llama["policy_sha256"] == fixture["policy_sha256"] and
            llama["teacher_identity_sha256"] ==
            fixture["reference_capture"]["teacher_identity_sha256"] and
            llama["cases_expected"] == llama["cases_validated"] == 1 and
            llama["cases_jsonl_sha256"] == sha(args.llama_run / "cases.jsonl") and
            llama["cases_jsonl_bytes"] ==
            (args.llama_run / "cases.jsonl").stat().st_size and
            all(row["identical_ids"] for row in llama["fresh_request_replay"]),
            "llama run lineage or replay differs")
    require(candidate["status"] == "COMPLETE" and
            candidate["evaluation_scope"] == "long-32768" and
            candidate["case_ids"] == expected and
            candidate["cases_expected"] == candidate["cases_completed"] == 1 and
            candidate["case_tsv_sha256"] == sha(tsv_path) and
            candidate["case_tsv_manifest_sha256"] == sha(tsv_manifest_path) and
            candidate["cases_jsonl_sha256"] == sha(args.candidate_run / "cases.jsonl") and
            candidate["candidate_identity_sha256"] ==
            sha(args.candidate_run / "candidate_identity.json") and
            candidate["candidate_identity"] == json.loads(
                (args.candidate_run / "candidate_identity.json").read_text()) and
            tsv_manifest["coverage"] == "long-32768" and
            tsv_manifest["core_cases"] == 1 and
            tsv_manifest["fixture_manifest_sha256"] == fixture_sha and
            tsv_manifest["prompts_sha256"] == inventory_sha and
            tsv_manifest["case_tsv_sha256"] == sha(tsv_path),
            "candidate run or TSV lineage differs")
    tsv = [line.split("\t") for line in tsv_path.read_text().splitlines()]
    require(len(tsv) == 1 and all(len(fields) == 5 for fields in tsv),
            "candidate TSV coverage differs")
    lrows = rows(args.llama_run / "cases.jsonl", expected)
    crows = rows(args.candidate_run / "cases.jsonl", expected)
    checkpoint = REPO / fixture["tokenizer"]["checkpoint"]
    for name, record in fixture["tokenizer"]["files"].items():
        path = checkpoint / name
        require(path.stat().st_size == record["bytes"] and
                sha(path) == record["sha256"], f"tokenizer identity differs: {name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True,
        local_files_only=True)
    scored = []
    for case, fields in zip(cases, tsv, strict=True):
        case_id = case["id"]
        details = case["details"]
        source = lrows[case_id]
        target = crows[case_id]
        prompt_path = args.fixtures / case["prompt_token_file"]
        target_path = args.fixtures / details["target_token_file"]
        mask_path = args.fixtures / details["loss_mask_file"]
        require(fields == [case_id, str(prompt_path.resolve()),
                           str(target_path.resolve()), str(mask_path.resolve()),
                           str(details["cap"])],
                f"candidate TSV row differs: {case_id}")
        require(prompt_path.stat().st_size == case["prompt_token_file_bytes"] and
                sha(prompt_path) == case["prompt_token_sha256"] and
                sha(target_path) == details["target_token_sha256"] and
                sha(mask_path) == details["loss_mask_sha256"],
                f"frozen input differs: {case_id}")
        prompt_tokens = len(ids(prompt_path))
        require(prompt_tokens == case["prompt_tokens"] ==
                source["prompt_tokens"] == target["prompt_tokens"] and
                target["status"] == "complete" and
                target["target_tokens"] == details["target_token_count"],
                f"prompt or execution differs: {case_id}")
        llama_output = args.llama_run / source["token_file"]
        candidate_output = args.candidate_run / f"{case_id}.generated.u32le"
        candidate_logits = args.candidate_run / f"{case_id}.candidate-logits.f32le"
        require(sha(llama_output) == source["token_sha256"] and
                sha(args.llama_run / source["text_file"]) == source["text_sha256"] and
                sha(candidate_output) == candidate["output_sha256"][case_id]["generated"] and
                sha(candidate_logits) ==
                candidate["output_sha256"][case_id]["candidate_logits"],
                f"generated output hash differs: {case_id}")
        llama_tokens, candidate_tokens = ids(llama_output), ids(candidate_output)
        require(llama_tokens == source["output_token_ids"] and
                len(llama_tokens) == source["generation_tokens"] and
                len(candidate_tokens) == target["generation_tokens"],
                f"generated token record differs: {case_id}")
        candidate_text = tokenizer.decode(
            candidate_tokens[:-1] if candidate_tokens[-1] in EOG_IDS else candidate_tokens,
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expected_answer = details["answer"]
        llama_text = (args.llama_run / source["text_file"]).read_text()
        arms = {}
        for arm, text, stop, output_path in (
            ("llama", llama_text, source["generation_stop"], llama_output),
            ("candidate", candidate_text, target["generation_stop"], candidate_output),
        ):
            grade = grade_retrieval(text, expected_answer)
            arms[arm] = {
                "text": text, "extracted": grade.extracted,
                "correct": grade.correct and stop == "eos",
                "stop": stop, "generated_ids_sha256": sha(output_path),
            }
        scored.append({"id": case_id, "prompt_tokens": prompt_tokens,
                       "expected": expected_answer, "arms": arms})
    status = "PASS" if all(all(arm["correct"] for arm in row["arms"].values())
                           for row in scored) else "FAIL"
    paired = paired_case_bootstrap([
        CaseMetric(row["id"], "R-32768", 0.0, 0.0, 1,
                   int(row["arms"]["llama"]["correct"]),
                   int(row["arms"]["candidate"]["correct"]))
        for row in scored
    ], metric_group="TASK026-R32768")
    report = {
        "status": status, "evaluation_scope": "long-32768", "cases": 1,
        "correct": {arm: sum(row["arms"][arm]["correct"] for row in scored)
                    for arm in ("llama", "candidate")},
        "paired_accuracy_loss_95": {
            "low": paired["accuracy_loss_low_95"],
            "high": paired["accuracy_loss_high_95"],
            "replicates": int(paired["replicates"]),
        },
        "teacher_target_nll": None,
        "teacher_target_nll_reason": "paired teacher probabilities are unavailable for fixed-answer retrieval targets",
        "fixture_manifest_sha256": fixture_sha,
        "prompt_inventory_sha256": inventory_sha,
        "effective_policy_sha256": rebind["effective_policy_sha256"],
        "policy_rebind_sha256": sha(rebind_path),
        "grader_sha256": fixture["scoring_implementation"]["sha256"],
        "metrics_sha256": fixture["metrics_implementation"]["sha256"],
        "scorer_sha256": sha(Path(__file__)),
        "tokenizer_files": fixture["tokenizer"]["files"],
        "llama_run_manifest_sha256": sha(llama_path),
        "candidate_result_sha256": sha(candidate_path),
        "candidate_artifact_manifest_digest": candidate["candidate_identity"]["manifest_digest"],
        "rows": scored,
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        report = score(args)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report = {"status": "INVALID", "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "correct": report.get("correct"),
                      "reason": report.get("reason")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
