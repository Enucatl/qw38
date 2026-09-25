#!/usr/bin/env bash
set -u

eval_args=()
if [ "${1:-}" = "--manual-full" ] && [ "$#" -eq 1 ]; then
  if [ ! -t 0 ]; then
    echo "full 216-case evaluation requires an interactive manual launch" >&2
    exit 2
  fi
  read -r -p "Type RUN FULL 216 to launch the full evaluation: " confirmation
  if [ "$confirmation" != "RUN FULL 216" ]; then exit 2; fi
  eval_args=(--manual-full)
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--manual-full]" >&2
  exit 2
fi

fixture_dir=".cache/evaluation/qw38-language-v2"
run_dir="$fixture_dir/runs/llama-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
uv run --script scripts/task018_run_llama.py \
  --fixtures "$fixture_dir" \
  --output "$run_dir" \
  --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --port 18111 "${eval_args[@]}" >"$run_dir/run.log" 2>&1
exit_code=$?
ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds=$(($(date +%s) - started_epoch))
python3 - "$run_dir/result.json.tmp" "$run_dir/result.json" \
  "$started_utc" "$ended_utc" "$elapsed_seconds" "$exit_code" "${eval_args[*]}" <<'PY'
import json
import os
import sys
from pathlib import Path

tmp, result, started, ended, elapsed, code, mode = sys.argv[1:]
record = {
    "command": [
        "uv", "run", "--script", "scripts/task018_run_llama.py",
        "--fixtures", ".cache/evaluation/qw38-language-v2",
        "--output", str(Path(result).parent),
        "--model", "models/Qwen3.8-27B-Q4_K_M.gguf", "--port", "18111",
    ] + (["--manual-full"] if mode else []),
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
