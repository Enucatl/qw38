#!/usr/bin/env bash
# Isolate projection families and screen MXFP4 using the fixed development cases.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
quantizer=(docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo
           --entrypoint /app/llama-quantize "$image" --max-buffer-size 512)
scorer=(docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo
        --entrypoint /repo/.cache/task020/task020-score-llama "$image")

for variant in nvfp4-mlp-gate-up nvfp4-attn-gdn mxfp4-projections; do
    case "$variant" in
        nvfp4-mlp-gate-up)
            override='ffn_(gate|up)\.weight=nvfp4' ;;
        nvfp4-attn-gdn)
            override='(attn_(qkv|gate|q|k|v|output)|ssm_out)\.weight=nvfp4' ;;
        mxfp4-projections)
            override='(ffn_(gate|up|down)|attn_(qkv|gate|q|k|v|output)|ssm_out)\.weight=mxfp4' ;;
    esac
    "${quantizer[@]}" --tensor-type "$override" \
        /repo/.cache/task020/qwen-bf16.gguf \
        "/repo/.cache/task020/qwen-$variant.gguf" Q4_K_M 16 \
        > ".cache/task020/$variant-quantize.log" 2>&1
    "${scorer[@]}" "/repo/.cache/task020/qwen-$variant.gguf" \
        /repo/.cache/task020/development-cases.tsv \
        "/repo/.cache/task020/$variant-scores.tsv" \
        > ".cache/task020/$variant-score.log" 2>&1
    python3 scripts/task020_compare_continuations.py \
        .cache/task020/q4-k-m-scores.tsv \
        ".cache/task020/$variant-scores.tsv" \
        ".cache/task020/$variant-comparison.json"
    tail -n 5 ".cache/task020/$variant-quantize.log"
    tail -n 5 ".cache/task020/$variant-score.log"
done
