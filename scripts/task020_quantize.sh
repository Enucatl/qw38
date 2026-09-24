#!/usr/bin/env bash
# Build paired development models directly from the BF16 source GGUF.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
source_gguf=/repo/.cache/task020/qwen-bf16.gguf
output_dir=/repo/.cache/task020
common=(docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo
        --entrypoint /app/llama-quantize "$image" --max-buffer-size 512)

"${common[@]}" "$source_gguf" "$output_dir/qwen-q4-k-m.gguf" Q4_K_M 16 \
    > .cache/task020/q4-quantize.log 2>&1
"${common[@]}" --tensor-type \
    '(ffn_(gate|up|down)|attn_(qkv|gate|q|k|v|output)|ssm_out)\.weight=nvfp4' \
    "$source_gguf" "$output_dir/qwen-nvfp4-projections.gguf" Q4_K_M 16 \
    > .cache/task020/nvfp4-quantize.log 2>&1

ls -lh .cache/task020/qwen-{q4-k-m,nvfp4-projections}.gguf
tail -n 6 .cache/task020/q4-quantize.log
tail -n 6 .cache/task020/nvfp4-quantize.log
