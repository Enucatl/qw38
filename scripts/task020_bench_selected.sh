#!/usr/bin/env bash
# Whole-model llama.cpp GPU throughput diagnostic for the selected GGUF proxy.
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
for variant in q4-k-m sensitive-q8; do
    docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
        --entrypoint /app/llama-bench "$image" \
        -m "/repo/.cache/task020/qwen-$variant.gguf" \
        -p 384 -n 128 -r 3 -ngl 99 -ot token_embd.weight=CUDA0 \
        --no-host 1 -o json \
        > ".cache/task020/bench-$variant.json" \
        2> ".cache/task020/bench-$variant.log"
    python3 - ".cache/task020/bench-$variant.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1]))
assert len(rows) == 2
for row in rows:
    assert row["backends"] == "CUDA"
    assert row["n_gpu_layers"] == 99
    assert row["tensor_buft_overrides"] == "token_embd.weight=CUDA0"
    assert row["no_host"] is True
    print(row["model_filename"], row["n_prompt"], row["n_gen"], row["avg_ts"])
PY
done
