#!/usr/bin/env bash
# Re-score the fixed development cases with every model tensor assigned to CUDA0.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
runner=(docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo
        --entrypoint /repo/.cache/task020/task020-score-llama "$image")
variants=(existing-q4-k-m q4-k-m nvfp4-projections nvfp4-down-control
          nvfp4-mlp-gate-up nvfp4-attn-gdn mxfp4-projections
          nvfp4-head q8-head q4-controls sensitive-q8)
for variant in "${variants[@]}"; do
    if [[ "$variant" == existing-q4-k-m ]]; then
        model=/repo/models/Qwen3.8-27B-Q4_K_M.gguf
    else
        model="/repo/.cache/task020/qwen-$variant.gguf"
    fi
    "${runner[@]}" "$model" /repo/.cache/task020/development-cases.tsv \
        "/repo/.cache/task020/gpu-$variant-scores.tsv" \
        > ".cache/task020/gpu-$variant-score.log" 2>&1
    if grep -q 'CPU_Mapped model buffer size' ".cache/task020/gpu-$variant-score.log"; then
        echo "CPU model buffer remains for $variant" >&2
        exit 1
    fi
    grep -m 1 'CUDA0 model buffer size' ".cache/task020/gpu-$variant-score.log"
    grep -m 1 'scored_cases=32' ".cache/task020/gpu-$variant-score.log"
done
python3 - <<'PY'
import json
from pathlib import Path
from scripts.task020_compare_continuations import compare, read_scores

base = Path('.cache/task020')
external = read_scores(base / 'gpu-existing-q4-k-m-scores.tsv')
controlled = read_scores(base / 'gpu-q4-k-m-scores.tsv')
variants = ('q4-k-m', 'nvfp4-projections', 'nvfp4-down-control',
            'nvfp4-mlp-gate-up', 'nvfp4-attn-gdn', 'mxfp4-projections',
            'nvfp4-head', 'q8-head', 'q4-controls', 'sensitive-q8')
results = {}
for name in variants:
    gpu = read_scores(base / f'gpu-{name}-scores.tsv')
    original = read_scores(base / f'{name}-scores.tsv')
    if gpu != original:
        raise ValueError(f'placement changed score rows: {name}')
    results[name] = {
        'versus_existing': compare(external, gpu),
        'versus_controlled_q4': compare(controlled, gpu),
    }
if external != read_scores(base / 'existing-q4-k-m-scores.tsv'):
    raise ValueError('placement changed existing GGUF score rows')
(base / 'full-gpu-comparisons.json').write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
for name, result in results.items():
    print(name, result['versus_existing']['delta_candidate_minus_reference_nats_per_token'],
          result['versus_controlled_q4']['delta_candidate_minus_reference_nats_per_token'])
PY
