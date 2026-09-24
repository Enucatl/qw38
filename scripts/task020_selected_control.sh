#!/usr/bin/env bash
# Screen the provisional Q4 family controls against the existing llama.cpp Q4_K_M reference.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /app/llama-quantize "$image" --max-buffer-size 512 \
    --token-embedding-type bf16 --output-tensor-type q8_0 \
    --tensor-type 'ssm_(alpha|beta)\.weight=bf16' \
    /repo/.cache/task020/qwen-bf16.gguf \
    /repo/.cache/task020/qwen-q4-controls.gguf Q4_K_M 16 \
    > .cache/task020/q4-controls-quantize.log 2>&1
docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /repo/.cache/task020/task020-score-llama "$image" \
    /repo/.cache/task020/qwen-q4-controls.gguf \
    /repo/.cache/task020/development-cases.tsv \
    /repo/.cache/task020/q4-controls-scores.tsv \
    > .cache/task020/q4-controls-score.log 2>&1
python3 scripts/task020_compare_continuations.py \
    .cache/task020/q4-k-m-scores.tsv \
    .cache/task020/q4-controls-scores.tsv \
    .cache/task020/q4-controls-comparison.json
tail -n 5 .cache/task020/q4-controls-quantize.log
tail -n 5 .cache/task020/q4-controls-score.log
