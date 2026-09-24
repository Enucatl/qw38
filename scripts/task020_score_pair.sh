#!/usr/bin/env bash
# Score paired GPU models on the same frozen development windows.
set -euo pipefail

python3 scripts/task020_prepare_cases.py \
    .cache/task020/tokens/validation.manifest.json \
    .cache/task020/development-cases.tsv --limit 32

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
runner=(docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo"
        -w /repo --entrypoint /repo/.cache/task020/task020-score-llama "$image")
for variant in q4-k-m nvfp4-projections; do
    "${runner[@]}" "/repo/.cache/task020/qwen-$variant.gguf" \
        /repo/.cache/task020/development-cases.tsv \
        "/repo/.cache/task020/$variant-scores.tsv" \
        > ".cache/task020/$variant-score.log" 2>&1
    tail -n 5 ".cache/task020/$variant-score.log"
done

python3 scripts/task020_compare_continuations.py \
    .cache/task020/q4-k-m-scores.tsv \
    .cache/task020/nvfp4-projections-scores.tsv \
    .cache/task020/paired-comparison.json
