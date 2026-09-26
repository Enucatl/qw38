#!/usr/bin/env bash
set -euo pipefail

fixture_dir=.cache/evaluation/qw38-language-v2
run_dir="${2:-$fixture_dir/runs/task026-prefill-r32768-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
artifact=.cache/candidates/candidate-v2-q4k-rope-fixed.qw38
binary=build/pinned-release/src/qw38-evaluate
if [[ "${1:-}" == "--finalize" ]]; then
  [[ -n "${2:-}" && -d "$run_dir" ]] || exit 2
else
  [[ $# -eq 0 ]] || exit 2
  mkdir -p "$run_dir"
  uv run --script scripts/task018_prepare_core.py --fixtures "$fixture_dir" \
    --output "$run_dir/long.tsv" --long-only >"$run_dir/prepare.log" 2>&1
  docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD:$PWD" -w "$PWD" qw38-dev:cuda13.4.1-pinned \
    "$binary" --artifact "$artifact" --cases "$run_dir/long.tsv" \
    --output "$run_dir" --eos-ids 248044,248046,248063,248064,248065 \
    >"$run_dir/run.log" 2>&1
fi
python3 - "$run_dir" "$artifact" "$binary" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

run, artifact, binary = map(Path, sys.argv[1:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
rows = [json.loads(line) for line in (run / 'cases.jsonl').read_text().splitlines()]
ids = [row['id'] for row in rows]
expected = ['R-32768-s0-d0.1']
status = ('COMPLETE' if ids == expected and
          all(row['status'] == 'complete' for row in rows) and
          all((run / f'{case}.generated.u32le').is_file() and
              (run / f'{case}.candidate-logits.f32le').is_file() for case in ids)
          else 'INCOMPLETE')
record = {
    'status': status, 'evaluation_scope': 'long-32768',
    'cases_expected': 1, 'cases_completed': len(rows), 'case_ids': ids,
    'case_tsv_sha256': sha(run / 'long.tsv'),
    'case_tsv_manifest_sha256': sha(run / 'long.manifest.json'),
    'cases_jsonl_sha256': sha(run / 'cases.jsonl'),
    'candidate_identity_sha256': sha(run / 'candidate_identity.json'),
    'output_sha256': {
        case: {
            'generated': sha(run / f'{case}.generated.u32le'),
            'candidate_logits': sha(run / f'{case}.candidate-logits.f32le'),
        }
        for case in ids
    },
    'candidate_identity': json.loads((run / 'candidate_identity.json').read_text()),
    'candidate_binary_sha256': sha(binary),
    'artifact_size_bytes': artifact.stat().st_size,
    'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_dirty_status': subprocess.check_output(['git', 'status', '--short'], text=True).splitlines(),
    'image_id': subprocess.check_output(['docker', 'image', 'inspect',
        'qw38-dev:cuda13.4.1-pinned', '--format', '{{.Id}}'], text=True).strip(),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
        '--format=csv,noheader'], text=True).strip(),
}
(run / 'result.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps({'run': str(run), 'status': status, 'cases': len(rows)}))
if status != 'COMPLETE': sys.exit(1)
PY
