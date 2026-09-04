"""Run the pinned llama.cpp quality adapter and freeze its seven-case output."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

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


def bundle(inputs: dict, path: Path) -> None:
    with path.open("wb") as f:
        f.write(b"QW38Q\1\0\0")
        f.write(struct.pack("<I", len(CASES)))
        for name in CASES:
            c = inputs["cases"][name]
            encoded = name.encode("ascii")
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<II", len(c["context"]), len(c["continuation"])))
            f.write(struct.pack("<%dI" % len(c["context"]), *c["context"]))
            f.write(struct.pack("<%dI" % len(c["continuation"]), *c["continuation"]))


def run(args: argparse.Namespace) -> dict:
    inputs = json.loads(args.inputs.read_text())
    llama_contract = json.loads(args.llama_contract.read_text())
    if digest(args.model) != llama_contract["model"]["sha256"]:
        raise ValueError("model hash differs")
    if llama_contract["source"]["revision"] != args.llama_source_revision:
        raise ValueError("llama revision differs")
    if not set(CASES).issubset(inputs["cases"]):
        raise ValueError("reference case set differs")
    with tempfile.TemporaryDirectory() as td:
        req = Path(td) / "quality.bundle"
        bundle(inputs, req)
        proc = subprocess.run(
            [str(args.llama_binary), str(args.model), str(req)],
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
    cases = {name: {"steps": [], "greedy_tokens": []} for name in CASES}
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
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
    for name in CASES:
        rows = cases[name]["steps"]
        if not rows:
            raise ValueError(f"missing case {name}")
        mean = sum(-r["log_probability"] for r in rows) / len(rows)
        cases[name]["mean_nll"] = mean
        cases[name]["perplexity"] = __import__("math").exp(mean)
    return {
        "schema": "qw38.quality-llama-reference",
        "version": 1,
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
