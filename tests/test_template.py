from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"


def render(name: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(EVAL), "--render-template-case", name],
        check=False,
        capture_output=True,
    )


def test_native_template_matches_official_rendered_bytes() -> None:
    fixtures = json.loads((ROOT / "fixtures" / "template_authority.json").read_text())
    for case in fixtures["successes"] + fixtures["policy_successes"]:
        result = render(case["name"])
        assert result.returncode == 0, case["name"]
        assert result.stdout == bytes.fromhex(case["rendered_utf8_hex"]), case["name"]


def test_template_errors_have_explicit_owners_and_messages() -> None:
    fixtures = json.loads((ROOT / "fixtures" / "template_authority.json").read_text())
    for case in fixtures["errors"] + fixtures["policy_errors"]:
        result = render(case["name"])
        assert result.returncode == 1, case["name"]
        assert case["error"].encode() in result.stderr, case["name"]
