#!/usr/bin/env bash
set -u

fixture_dir=".cache/evaluation/qw38-language-v2"
run_prefix="${QW38_EVAL_RUN_PREFIX:-v0}"
run_dir="$fixture_dir/runs/$run_prefix-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
mkdir -p "$fixture_dir/source_capture_job"
exec 9>"$fixture_dir/source_capture_job/v0-evaluation.lock"
if ! flock -n 9; then
  echo "another TASK-018 V0 evaluation holds the lock" >&2
  exit 2
fi
active="$(docker ps --format '{{.Names}}' | grep -E '^qw38-task018-(eval|probs)-' || true)"
if [ -n "$active" ]; then
  echo "llama.cpp capture/evaluation is still active: $active" >&2
  exit 2
fi
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
binary="${QW38_EVAL_BINARY:-.cache/task018-build/src/qw38-evaluate}"
artifact="${QW38_EVAL_ARTIFACT:-build/pinned-debug/qwen-v0.qw38}"
state_binary="${QW38_STATE_REPLAY_BINARY:-.cache/task018-build/src/qw38-state-replay}"
uv run --script scripts/task018_prepare_core.py \
  --fixtures "$fixture_dir" --output "$run_dir/core.tsv" >"$run_dir/prepare.log" 2>&1
prepare_code=$?
if [ "$prepare_code" -eq 0 ]; then
  docker run --rm --gpus all -v /home/user/qw38:/repo -v /home/user/qw38:/home/user/qw38 -w /repo \
    qw38-dev:cuda13.4.1-pinned "$binary" \
    --artifact "$artifact" --cases "$run_dir/core.tsv" --output "$run_dir" \
    --eos-ids 248044,248046,248063,248064,248065 >"$run_dir/run.log" 2>&1
  exit_code=$?
  eval_code=$exit_code
  state_code=NOT_RUN
  if [ "$exit_code" -eq 0 ]; then
    docker run --rm --gpus all -v /home/user/qw38:/repo -w /repo \
      qw38-dev:cuda13.4.1-pinned "$state_binary" \
      --artifact "$artifact" \
      --prompt "$fixture_dir/tokens/case_000.prompt.u32le" \
      --interleave-prompt "$fixture_dir/tokens/recNu3MXkvWUzHZr9.prompt.u32le" \
      --output "$run_dir/state-replay.json" >"$run_dir/state-replay.log" 2>&1
    state_code=$?
    if [ "$state_code" -ne 0 ]; then exit_code=$state_code; fi
  fi
else
  exit_code=$prepare_code
  eval_code=$prepare_code
  state_code=NOT_RUN
  cp "$run_dir/prepare.log" "$run_dir/run.log"
fi
ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds=$(($(date +%s) - started_epoch))
python3 - "$run_dir/result.json.tmp" "$run_dir/result.json" \
  "$started_utc" "$ended_utc" "$elapsed_seconds" "$exit_code" "$binary" "$artifact" "$state_binary" "$state_code" "$eval_code" <<'PY'
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

tmp, result, started, ended, elapsed, code, binary, artifact, state_binary, state_code, eval_code = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
run_dir = Path(result).parent
completed_cases = 0
for line in (run_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.fullmatch(r"evaluated_cases=(\d+) last_id=.+", line)
    if match:
        completed_cases = max(completed_cases, int(match.group(1)))
durable_case_rows = malformed_case_rows = 0
cases_path = run_dir / "cases.jsonl"
if cases_path.is_file():
    for line in cases_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            json.loads(line)
            durable_case_rows += 1
        except json.JSONDecodeError:
            malformed_case_rows += 1
generated = {path.name: sha(path) for path in sorted(run_dir.glob("*.generated.u32le"))}
logits = {path.name: sha(path) for path in sorted(run_dir.glob("*.candidate-logits.f32le"))}
core_case_rows = len((run_dir / "core.tsv").read_text().splitlines()) if (run_dir / "core.tsv").is_file() else 0
status = ("COMPLETE" if int(code) == 0 and core_case_rows == 216 and completed_cases == 216
          and durable_case_rows == 216 and len(generated) == 216 and len(logits) == 216
          else "PARTIAL" if completed_cases or generated or logits else "INCOMPLETE")
record = {
    "status": status,
    "coverage": {"expected_core_cases": 216, "prepared_core_rows": core_case_rows,
                 "completed_cases_from_log": completed_cases,
                 "durable_case_jsonl_rows": durable_case_rows,
                 "malformed_case_jsonl_rows": malformed_case_rows,
                 "durable_generated_files": len(generated),
                 "durable_candidate_logit_files": len(logits),
                 "quality_scoring_performed": False},
    "command": ["docker", "run", "--rm", "--gpus", "all", "qw38-dev:cuda13.4.1-pinned", binary,
                "--artifact", artifact, "--cases", str(Path(result).parent / "core.tsv"),
                "--output", str(Path(result).parent), "--eos-ids", "248044,248046,248063,248064,248065"],
    "started_utc": started,
    "ended_utc": ended,
    "elapsed_seconds": int(elapsed),
    "exit_code": int(code),
    "evaluation_exit_code": int(eval_code),
    "binary": binary,
    "binary_sha256": sha(binary),
    "artifact": artifact,
    "artifact_size_bytes": Path(artifact).stat().st_size,
    "artifact_identity": json.loads((Path(result).parent / "candidate_identity.json").read_text()) if (Path(result).parent / "candidate_identity.json").is_file() else None,
    "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_dirty_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
    "dev_image_id": subprocess.check_output(["docker", "image", "inspect", "qw38-dev:cuda13.4.1-pinned", "--format", "{{.Id}}"], text=True).strip(),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
    "case_tsv": str(Path(result).parent / "core.tsv"),
    "case_tsv_sha256": sha(Path(result).parent / "core.tsv"),
    "case_tsv_manifest": str(Path(result).parent / "core.manifest.json"),
    "case_tsv_manifest_sha256": sha(Path(result).parent / "core.manifest.json"),
    "cases_jsonl_sha256": sha(Path(result).parent / "cases.jsonl") if (Path(result).parent / "cases.jsonl").is_file() else None,
    "generated_token_sha256": generated,
    "candidate_logits_sha256": logits,
    "log": str(Path(result).parent / "run.log"),
    "state_replay_exit": int(state_code) if state_code != "NOT_RUN" else "NOT_RUN",
    "state_replay_log": str(Path(result).parent / "state-replay.log"),
    "state_replay_binary_sha256": sha(state_binary) if state_code != "NOT_RUN" else None,
    "state_replay_result_sha256": sha(Path(result).parent / "state-replay.json") if (Path(result).parent / "state-replay.json").is_file() else None,
    "state_replay_prompt_sha256": sha(".cache/evaluation/qw38-language-v2/tokens/case_000.prompt.u32le") if state_code != "NOT_RUN" else None,
    "state_replay_interleave_sha256": sha(".cache/evaluation/qw38-language-v2/tokens/recNu3MXkvWUzHZr9.prompt.u32le") if state_code != "NOT_RUN" else None,
}
Path(tmp).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(tmp, result)
PY
exit "$exit_code"
