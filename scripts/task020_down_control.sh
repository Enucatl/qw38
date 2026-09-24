#!/usr/bin/env bash
# Keep llama.cpp's Q4_K_M down-projection choices while changing other main projections to NVFP4.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /app/llama-quantize "$image" --max-buffer-size 512 \
    --tensor-type '(ffn_(gate|up)|attn_(qkv|gate|q|k|v|output)|ssm_out)\.weight=nvfp4' \
    /repo/.cache/task020/qwen-bf16.gguf \
    /repo/.cache/task020/qwen-nvfp4-down-control.gguf Q4_K_M 16 \
    > .cache/task020/nvfp4-down-control-quantize.log 2>&1

docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /repo/.cache/task020/task020-score-llama "$image" \
    /repo/.cache/task020/qwen-nvfp4-down-control.gguf \
    /repo/.cache/task020/development-cases.tsv \
    /repo/.cache/task020/nvfp4-down-control-scores.tsv \
    > .cache/task020/nvfp4-down-control-score.log 2>&1

python3 scripts/task020_compare_continuations.py \
    .cache/task020/q4-k-m-scores.tsv \
    .cache/task020/nvfp4-down-control-scores.tsv \
    .cache/task020/down-control-comparison.json
python3 scripts/task020_compare_continuations.py \
    .cache/task020/nvfp4-down-control-scores.tsv \
    .cache/task020/nvfp4-projections-scores.tsv \
    .cache/task020/down-attribution-comparison.json
tail -n 5 .cache/task020/nvfp4-down-control-quantize.log
tail -n 5 .cache/task020/nvfp4-down-control-score.log
