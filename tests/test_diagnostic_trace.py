from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_EVAL = ROOT / "build" / "qw38-eval"
DIAGNOSTIC_EVAL = ROOT / "build" / "qw38-eval-diagnostic"


def run_filter(layer: str, tap: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(DIAGNOSTIC_EVAL), "--check-trace-filter", layer, tap],
        check=False,
        capture_output=True,
        text=True,
    )


def test_diagnostic_target_builds_separately() -> None:
    result = subprocess.run(
        ["make", "diagnostic"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert DIAGNOSTIC_EVAL.exists()


def test_exact_layer_and_tap_filters_select_typed_views() -> None:
    all_taps = run_filter("all", "*")
    exact = run_filter("3", "attention.rope_query")
    absent_at_layer = run_filter("0", "attention.rope_query")

    assert all_taps.returncode == 0
    assert all_taps.stdout.count("tap=") == 3
    assert "matched=3" in all_taps.stdout
    assert exact.returncode == 0
    assert exact.stdout == "tap=attention.rope_query,layer=3,count=4\nmatched=1\n"
    assert absent_at_layer.returncode == 0
    assert absent_at_layer.stdout == "matched=0\n"


def test_invalid_filters_fail_closed() -> None:
    bad_layer = run_filter("65", "embedding")
    malformed_layer = run_filter("three", "embedding")
    unknown_tap = run_filter("all", "attention.queryish")

    assert bad_layer.returncode == 1
    assert "layer or tap filter is invalid" in bad_layer.stderr
    assert malformed_layer.returncode == 1
    assert "malformed diagnostic trace layer" in malformed_layer.stderr
    assert unknown_tap.returncode == 1
    assert "layer or tap filter is invalid" in unknown_tap.stderr


def test_release_binary_has_no_trace_command_or_stable_tap_names() -> None:
    release = subprocess.run(
        [str(RELEASE_EVAL), "--check-trace-filter", "all", "*"],
        check=False,
        capture_output=True,
        text=True,
    )
    image = RELEASE_EVAL.read_bytes()

    assert release.returncode == 2
    assert b"--check-trace-filter" not in image
    assert b"attention.rope_query" not in image
    assert b"gdn.recurrent_state" not in image
