from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_provenance_is_machine_readable_and_deterministic() -> None:
    command = [
        sys.executable,
        "tools/release_provenance.py",
        "--build",
        "Makefile",
        "--evidence",
        "plan.md",
    ]
    first = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    second = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert first.stdout == second.stdout
    record = json.loads(first.stdout)
    assert set(record["git"]) == {"commit", "tree", "status"}
    assert record["builds"][0]["sha256"]
    assert record["evidence"][0]["sha256"]


def test_release_provenance_rejects_dirty_state_when_requested() -> None:
    result = subprocess.run(
        [sys.executable, "tools/release_provenance.py", "--require-clean"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
