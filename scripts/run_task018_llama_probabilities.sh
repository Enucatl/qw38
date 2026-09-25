#!/usr/bin/env bash
set -u

fixture_dir=".cache/evaluation/qw38-language-v2"
job_dir="$fixture_dir/source_capture_job"
attempt_id="probabilities-$(date -u +%Y%m%dT%H%M%SZ)-$$"
attempt_dir="$job_dir/attempts/$attempt_id"
mkdir -p "$attempt_dir"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
uv run --script scripts/task018_capture_llama_probabilities.py \
  --fixtures "$fixture_dir" \
  --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --attempt-dir "$attempt_dir" \
  --port 18110 >"$attempt_dir/capture.log" 2>&1
exit_code=$?
ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds=$(($(date +%s) - started_epoch))
python3 - "$attempt_dir/result.json.tmp" "$attempt_dir/result.json" \
  "$attempt_id" "$started_utc" "$ended_utc" "$elapsed_seconds" "$exit_code" <<'PY'
import json
import os
import sys
from pathlib import Path

tmp, result, attempt, started, ended, elapsed, code = sys.argv[1:]
record = {
    "attempt_id": attempt,
    "command": [
        "uv", "run", "--script", "scripts/task018_capture_llama_probabilities.py",
        "--fixtures", ".cache/evaluation/qw38-language-v2",
        "--model", "models/Qwen3.8-27B-Q4_K_M.gguf",
        "--attempt-dir", str(Path(result).parent), "--port", "18110",
    ],
    "started_utc": started,
    "ended_utc": ended,
    "elapsed_seconds": int(elapsed),
    "exit_code": int(code),
    "log": str(Path(result).parent / "capture.log"),
}
Path(tmp).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(tmp, result)
PY
exit "$exit_code"
