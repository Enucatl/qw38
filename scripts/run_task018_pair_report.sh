#!/usr/bin/env bash
set -u
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: run_task018_pair_report.sh LLAMA_RUN V0_RUN [P100_REVIEWS_JSONL]" >&2
  exit 2
fi
fixture_dir=".cache/evaluation/qw38-language-v2"
llama_run="$1"
v0_run="$2"
review_args=()
if [ "$#" -eq 3 ]; then review_args=(--reviews "$3"); fi
rebind_args=()
if [ -n "${QW38_EVAL_POLICY_REBIND:-}" ]; then
  rebind_args=(--policy-rebind "$QW38_EVAL_POLICY_REBIND")
fi
report_dir="$fixture_dir/paired/$(basename "$llama_run")-$(basename "$v0_run")"
mkdir -p "$report_dir"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
uv run --script scripts/task018_pair_report.py \
  --fixtures "$fixture_dir" --llama-run "$llama_run" \
  --v0-run "$v0_run" --output "$report_dir" "${rebind_args[@]}" "${review_args[@]}" >"$report_dir/run.log" 2>&1
exit_code=$?
ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds=$(($(date +%s) - started_epoch))
python3 - "$report_dir/execution.json.tmp" "$report_dir/execution.json" \
  "$started_utc" "$ended_utc" "$elapsed_seconds" "$exit_code" \
  "$llama_run" "$v0_run" "${3:-}" "${QW38_EVAL_POLICY_REBIND:-}" <<'PY'
import json
import os
import sys
from pathlib import Path

tmp, result, started, ended, elapsed, code, llama, v0, reviews, rebind = sys.argv[1:]
command = ["uv", "run", "--script", "scripts/task018_pair_report.py",
           "--fixtures", ".cache/evaluation/qw38-language-v2",
           "--llama-run", llama, "--v0-run", v0,
           "--output", str(Path(result).parent)]
if rebind:
    command.extend(["--policy-rebind", rebind])
if reviews:
    command.extend(["--reviews", reviews])
record = {
    "command": command,
    "started_utc": started, "ended_utc": ended,
    "elapsed_seconds": int(elapsed), "exit_code": int(code),
    "llama_run": llama, "v0_run": v0,
    "log": str(Path(result).parent / "run.log"),
}
Path(tmp).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(tmp, result)
PY
exit "$exit_code"
