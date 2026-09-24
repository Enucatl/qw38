#!/usr/bin/env bash
# Compare Q8 and NVFP4 vocabulary heads with otherwise identical Q4_K_M tensors.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
for variant in q8-head nvfp4-head; do
    case "$variant" in
        q8-head) output_type=q8_0 ;;
        nvfp4-head) output_type=nvfp4 ;;
    esac
    docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
        --entrypoint /app/llama-quantize "$image" --max-buffer-size 512 \
        --output-tensor-type "$output_type" \
        /repo/.cache/task020/qwen-bf16.gguf \
        "/repo/.cache/task020/qwen-$variant.gguf" Q4_K_M 16 \
        > ".cache/task020/$variant-quantize.log" 2>&1
    docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
        --entrypoint /repo/.cache/task020/task020-score-llama "$image" \
        "/repo/.cache/task020/qwen-$variant.gguf" \
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
