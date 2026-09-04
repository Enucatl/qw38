"""Policy-only QLT-001 orchestrator; native scoring remains in qw38-eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from tools.qw38_eval import ScoreRequest, ScoreResult, run_native

CASES = (
    "wikitext_nll",
    "continuation_2048",
    "continuation_4096",
    "continuation_6144",
    "continuation_8192",
    "recurrence_short",
    "recurrence_long",
    "retrieval_128k",
    "task_arithmetic",
    "task_python_len",
    "task_inference",
    "task_minutes",
    "task_sort",
    "task_json",
    "task_reading",
    "task_sequence",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _greedy(result: ScoreResult) -> list[int]:
    return [step.greedy_token for step in result.steps]


def verdicts(
    results: dict[str, ScoreResult],
    authority: dict[str, Any],
    inputs: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    thresholds = contract["thresholds"]
    verdict: dict[str, Any] = {}
    nll = results["wikitext_nll"]
    auth_nll = float(authority["cases"]["wikitext_nll"]["mean_nll"])
    ratio = math.exp(float(nll.record["mean_nll"]) - auth_nll)
    verdict["wikitext_nll"] = {
        "pass": len(nll.steps) == 1024
        and _finite(ratio)
        and ratio <= thresholds["nll_ppl_ratio"],
        "ppl_ratio": ratio,
    }
    continuation = [f"continuation_{offset}" for offset in (2048, 4096, 6144, 8192)]
    expected = [
        token
        for name in continuation
        for token in authority["cases"][name].get("greedy_tokens", [])
    ]
    actual = [token for name in continuation for token in _greedy(results[name])]
    verdict["continuation"] = {"pass": actual == expected, "positions": len(actual)}
    short, long = results["recurrence_short"], results["recurrence_long"]
    short_a, long_a = (
        authority["cases"]["recurrence_short"],
        authority["cases"]["recurrence_long"],
    )
    drift = (float(long.record["mean_nll"]) - float(long_a["mean_nll"])) - (
        float(short.record["mean_nll"]) - float(short_a["mean_nll"])
    )
    verdict["recurrence"] = {
        "pass": math.exp(float(short.record["mean_nll"]) - float(short_a["mean_nll"]))
        <= thresholds["recurrence_ppl_ratio"]
        and math.exp(float(long.record["mean_nll"]) - float(long_a["mean_nll"]))
        <= thresholds["recurrence_ppl_ratio"]
        and drift <= thresholds["recurrence_incremental_nll"],
        "incremental_nll": drift,
    }
    retrieval = results["retrieval_128k"]
    verdict["retrieval_128k"] = {
        "pass": retrieval.record["frontiers"]["final"]
        == thresholds["retrieval_frontier"]
        and _greedy(retrieval) == inputs["cases"]["retrieval_128k"]["continuation"]
    }
    task_pass = all(
        _greedy(results[name]) == inputs["cases"][name]["continuation"]
        for name in CASES
        if name.startswith("task_")
    )
    verdict["tasks"] = {"pass": task_pass, "count": 8}
    verdict["all"] = all(item["pass"] for key, item in verdict.items() if key != "all")
    return verdict


def run_suite(
    model: Path,
    binary: Path,
    contract_path: Path,
    inputs_path: Path,
    authority_path: Path,
    evidence_dir: Path,
    revision: str,
    source_state: str,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if (
        inputs.get("dataset_sha256") != contract["dataset"]["sha256"]
        or inputs.get("tokenizer_sha256") != contract["tokenizer"]["sha256"]
    ):
        raise ValueError("quality input identity differs from contract")
    if set(inputs.get("cases", ())) != set(CASES):
        raise ValueError("quality input case set differs from contract")
    if authority.get("quality_contract_sha256") != sha256(
        contract_path
    ) or authority.get("quality_inputs_sha256") != sha256(inputs_path):
        raise ValueError("authority identity does not match frozen inputs")
    if source_state not in {"clean", "dirty"} or not revision:
        raise ValueError("explicit source revision and state are required")
    native = evidence_dir / "native"
    native.mkdir(parents=True, exist_ok=True)
    results: dict[str, ScoreResult] = {}
    try:
        for name in CASES:
            case = inputs["cases"][name]
            output = native / name
            request = ScoreRequest(
                model,
                tuple(case["context"]),
                output,
                revision,
                source_state,
                tuple(case["continuation"]),
            )
            result = run_native(request, binary=binary)
            results[name] = result
        verdict = verdicts(results, authority, inputs, contract)
        report = {
            "schema": "qw38.quality-report",
            "version": 1,
            "status": "pass" if verdict["all"] else "fail",
            "verdicts": verdict,
            "cases": {name: str(native / name / "result.json") for name in CASES},
            "inputs_sha256": sha256(inputs_path),
            "authority_sha256": sha256(authority_path),
            "contract_sha256": sha256(contract_path),
            "source_revision": revision,
            "source_state": source_state,
        }
        temporary = evidence_dir / f"report.json.tmp.{os.getpid()}"
        temporary.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(evidence_dir / "report.json")
        if not verdict["all"]:
            raise RuntimeError("quality thresholds failed")
        return report
    except Exception as error:
        failures = evidence_dir / "failures.json"
        existing = json.loads(failures.read_text()) if failures.exists() else []
        existing.append(
            {
                "error": str(error),
                "source_revision": revision,
                "source_state": source_state,
            }
        )
        failures.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-state", required=True)
    args = parser.parse_args()
    run_suite(
        args.model,
        args.binary,
        args.contract,
        args.inputs,
        args.authority,
        args.evidence,
        args.source_revision,
        args.source_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
