#!/usr/bin/env bash
set -euo pipefail

image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
mkdir -p .cache/task022/layer3-proxy .cache/task022/layer3-quartz

docker run --rm -v "$PWD:/repo" -v /home/user/llama.cpp:/llama-src -w /repo \
    --entrypoint /usr/bin/g++ "$image" -std=c++17 -O2 \
    -I/llama-src/include -I/llama-src/ggml/include \
    scripts/task022_trace_llama.cpp -Wl,-rpath,/app \
    /app/libllama.so /app/libggml.so /app/libggml-base.so \
    -o .cache/task022/task022-trace-llama

docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned \
    bash -lc 'cmake --build build/pinned-release --target qw38_task022_trace -j4'

docker run --rm --gpus all -e LD_LIBRARY_PATH=/app -v "$PWD:/repo" -w /repo \
    --entrypoint /repo/.cache/task022/task022-trace-llama "$image" \
    /repo/.cache/task020/qwen-sensitive-q8.gguf \
    /repo/.cache/task022/proxy-trace/one-case.tsv \
    /repo/.cache/task022/layer3-proxy/scores.tsv \
    /repo/.cache/task022/layer3-proxy

docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned \
    build/pinned-release/src/qw38-task022-trace \
    .cache/q4k-candidate/candidate.qw38 \
    .cache/task020/tokens/validation.window-000.u32le \
    .cache/task022/layer3-quartz
