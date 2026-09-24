#!/usr/bin/env bash
set -euo pipefail

fixture_dir=".cache/evaluation/qw38-language-v1"
binary=".cache/task018-build/src/qw38-evaluate"
artifact="build/pinned-debug/qwen-v0.qw38"
run_dir="$(mktemp -d .cache/task018-dev-smoke.XXXXXXXX)"
mkdir -p "$fixture_dir/source_capture_job"
exec 9>"$fixture_dir/source_capture_job/v0-evaluation.lock"
if ! flock -n 9; then
  echo "another TASK-018 V0 evaluation holds the lock" >&2
  exit 2
fi

uv run --script scripts/task018_dev_smoke.py prepare \
  --core "$fixture_dir/core.tsv" --output "$run_dir/cases.tsv"
if ! docker run --rm --gpus all \
  -v "$PWD:/repo" -v "$PWD:$PWD" -w /repo \
  qw38-dev:cuda13.4.1-pinned "$binary" \
  --artifact "$artifact" --cases "$run_dir/cases.tsv" --output "$run_dir" \
  --eos-ids 248044,248046,248063,248064,248065 >"$run_dir/run.log" 2>&1; then
  cat "$run_dir/run.log" >&2
  echo "development smoke failed; output: $run_dir" >&2
  exit 1
fi
uv run --script scripts/task018_dev_smoke.py check --output "$run_dir"
echo "development smoke output: $run_dir"
