"""Emit release-time repository and artifact provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def artifact(value: str) -> dict[str, str]:
    path = ROOT / value
    result = {"path": value}
    if path.is_file():
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--build", action="append", default=[])
    parser.add_argument("--container", action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    state = "clean" if not git("status", "--porcelain") else "dirty"
    if args.require_clean and state != "clean":
        parser.error("release provenance requires a clean working tree")
    record = {
        "schema_version": 1,
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "status": state,
        },
        "builds": [artifact(v) for v in args.build],
        "containers": args.container,
        "evidence": [artifact(v) for v in args.evidence],
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
