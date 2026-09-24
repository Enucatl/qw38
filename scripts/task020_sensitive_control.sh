#!/usr/bin/env bash
# Screen selective Q8 attention/GDN and Q4 MLP from the trusted BF16 source.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /app/llama-quantize "$image" --max-buffer-size 512 \
    --token-embedding-type bf16 --output-tensor-type q8_0 \
    --tensor-type '(attn_(qkv|gate|q|k|v|output)|ssm_out)\.weight=q8_0' \
    --tensor-type 'ssm_(alpha|beta)\.weight=bf16' \
    --tensor-type 'ffn_(gate|up|down)\.weight=q4_k' \
    /repo/.cache/task020/qwen-bf16.gguf \
    /repo/.cache/task020/qwen-sensitive-q8.gguf Q4_K_M 16 \
    > .cache/task020/sensitive-q8-quantize.log 2>&1
docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /repo/.cache/task020/task020-score-llama "$image" \
    /repo/.cache/task020/qwen-sensitive-q8.gguf \
    /repo/.cache/task020/development-cases.tsv \
    /repo/.cache/task020/sensitive-q8-scores.tsv \
    > .cache/task020/sensitive-q8-score.log 2>&1
python3 scripts/task020_compare_continuations.py \
    .cache/task020/existing-q4-k-m-scores.tsv \
    .cache/task020/sensitive-q8-scores.tsv \
    .cache/task020/sensitive-q8-vs-existing.json
tail -n 5 .cache/task020/sensitive-q8-quantize.log
tail -n 5 .cache/task020/sensitive-q8-score.log
