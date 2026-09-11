"""Run the pinned llama.cpp quality adapter and freeze its seven-case output."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from tools.opt058_quality_baseline import bundle_cases

CASES = (
    "wikitext_nll",
    "continuation_2048",
    "continuation_4096",
    "continuation_6144",
    "continuation_8192",
    "recurrence_short",
    "recurrence_long",
)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bundle(inputs: dict, path: Path, cases: Sequence[str] = CASES) -> None:
    bundle_cases(inputs, path, cases)


def selected_cases(args: argparse.Namespace, inputs: dict) -> tuple[str, ...]:
    if args.cases:
        names = tuple(item.strip() for item in args.cases.split(",") if item.strip())
        if not names:
            raise ValueError("case list is empty")
        missing = [name for name in names if name not in inputs["cases"]]
        if missing:
            raise ValueError("missing input cases: " + ",".join(missing))
        return names
    if not set(CASES).issubset(inputs["cases"]):
        raise ValueError("reference case set differs")
    return CASES


def parse_oracle_stdout(stdout: str, case_names: Sequence[str], generate: int) -> dict:
    cases = {
        name: {"steps": [], "greedy_tokens": [], "generated_tokens": []}
        for name in case_names
    }
    for line in stdout.splitlines():
        fields = line.split("\t")
        if not fields:
            continue
        if fields[0] == "gen":
            _, name, pos, token, logit, runner, runner_logit = fields
            cases[name]["generated_tokens"].append(
                {
                    "position": int(pos),
                    "token": int(token),
                    "logit": float(logit),
                    "runner_up_token": int(runner),
                    "runner_up_logit": float(runner_logit),
                }
            )
            continue
        if fields[0] != "step":
            continue
        _, name, pos, target, lp, greedy, gl, runner, rl, margin = fields
        row = {
            "position": int(pos),
            "target_token": int(target),
            "log_probability": float(lp),
            "greedy_token": int(greedy),
            "greedy_logit": float(gl),
            "runner_up_token": int(runner),
            "runner_up_logit": float(rl),
            "margin": float(margin),
        }
        cases[name]["steps"].append(row)
        cases[name]["greedy_tokens"].append(row["greedy_token"])
    for name in case_names:
        rows = cases[name]["steps"]
        if not rows and not generate:
            raise ValueError(f"missing case {name}")
        if rows:
            mean = sum(-r["log_probability"] for r in rows) / len(rows)
            cases[name]["mean_nll"] = mean
            cases[name]["perplexity"] = __import__("math").exp(mean)
    return cases


def run(args: argparse.Namespace) -> dict:
    inputs = json.loads(args.inputs.read_text())
    llama_contract = json.loads(args.llama_contract.read_text())
    if digest(args.model) != llama_contract["model"]["sha256"]:
        raise ValueError("model hash differs")
    if llama_contract["source"]["revision"] != args.llama_source_revision:
        raise ValueError("llama revision differs")
    case_names = selected_cases(args, inputs)
    if args.from_log:
        stdout = args.from_log.read_text()
    else:
        with tempfile.TemporaryDirectory() as td:
            req = Path(td) / "quality.bundle"
            bundle(inputs, req, case_names)
            command = [str(args.llama_binary), str(args.model), str(req)]
            if args.generate:
                command.extend(["--generate", str(args.generate)])
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "LD_LIBRARY_PATH": str(args.llama_binary.parent)
                    + ":"
                    + os.environ.get("LD_LIBRARY_PATH", ""),
                },
            )
            if proc.returncode:
                raise RuntimeError(
                    f"llama quality oracle failed ({proc.returncode}): "
                    f"stderr={proc.stderr[-2000:]} stdout={proc.stdout[-4000:]}"
                )
            stdout = proc.stdout
    cases = parse_oracle_stdout(stdout, case_names, args.generate)
    return {
        "schema": "qw38.quality-llama-reference",
        "version": 1,
        "case_list": list(case_names),
        "cases": cases,
        "model_sha256": digest(args.model),
        "quality_contract_sha256": digest(args.quality_contract),
        "quality_inputs_sha256": digest(args.inputs),
        "llama_source_revision": args.llama_source_revision,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--llama-source", type=Path, required=True)
    p.add_argument("--llama-binary", type=Path, required=True)
    p.add_argument("--llama-contract", type=Path, required=True)
    p.add_argument("--quality-contract", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cases",
        default="",
        help="optional comma-separated case list; default is the frozen seven-case set",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=0,
        help="optional free-running token count; 0 keeps teacher-forced scoring",
    )
    p.add_argument(
        "--from-log",
        type=Path,
        default=None,
        help="parse an existing oracle stdout log instead of spawning the binary",
    )
    p.add_argument(
        "--llama-source-revision", default="cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    )
    a = p.parse_args()
    out = run(a)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    tmp.replace(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
