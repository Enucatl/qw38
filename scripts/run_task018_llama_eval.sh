#!/usr/bin/env bash
set -u

fixture_dir=".cache/evaluation/qw38-language-v2"
run_dir="$fixture_dir/runs/llama-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
uv run --script scripts/task018_run_llama.py \
  --fixtures "$fixture_dir" \
  --output "$run_dir" \
  --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --port 18111 >"$run_dir/run.log" 2>&1
exit_code=$?
ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds=$(($(date +%s) - started_epoch))
python3 - "$run_dir/result.json.tmp" "$run_dir/result.json" \
  "$started_utc" "$ended_utc" "$elapsed_seconds" "$exit_code" <<'PY'
import json
import os
import sys
from pathlib import Path

tmp, result, started, ended, elapsed, code = sys.argv[1:]
record = {
    "command": [
        "uv", "run", "--script", "scripts/task018_run_llama.py",
        "--fixtures", ".cache/evaluation/qw38-language-v2",
        "--output", str(Path(result).parent),
        "--model", "models/Qwen3.8-27B-Q4_K_M.gguf", "--port", "18111",
    ],
    "started_utc": started,
    "ended_utc": ended,
    "elapsed_seconds": int(elapsed),
    "exit_code": int(code),
    "log": str(Path(result).parent / "run.log"),
}
Path(tmp).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(tmp, result)
PY
exit "$exit_code"
