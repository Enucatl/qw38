#!/usr/bin/env bash
# Collect quantized-model activation importance from the disjoint train split.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
inputs=()
while IFS=$'\t' read -r path chunks; do
    segment=${path##*.segment-}
    segment=${segment%.txt}
    output=".cache/task020/calibration-imatrix-segment-$segment.gguf"
    docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
        --entrypoint /app/llama-imatrix "$image" \
        -m /repo/.cache/task020/qwen-q4-k-m.gguf \
        -f "/repo/$path" -o "/repo/$output" \
        --no-ppl --chunks "$chunks" -c 512 -ngl 99 \
        > "${output%.gguf}.log" 2>&1
    inputs+=("/repo/$output")
done < <(python3 -c 'import json; d=json.load(open(".cache/task020/calibration-exact.json")); [print(s["path"],s["chunks"],sep="\t") for s in d["segments"]]')
input_files=$(IFS=,; echo "${inputs[*]}")
docker run --rm -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /app/llama-imatrix "$image" \
    -m /repo/.cache/task020/qwen-q4-k-m.gguf \
    --in-file "$input_files" -o /repo/.cache/task020/calibration-imatrix.gguf -c 512 \
    > .cache/task020/calibration-imatrix.log 2>&1
tail -n 12 .cache/task020/calibration-imatrix.log
ls -lh .cache/task020/calibration-imatrix.gguf
