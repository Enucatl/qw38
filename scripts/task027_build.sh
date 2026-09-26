#!/usr/bin/env bash
# Pinned public API headers and shared libraries; production runtime is unchanged.
set -euo pipefail
out=.cache/task027
image=ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6
revision=e6ab7c1a41054a888ada952eab4c886444c2f5ad
mkdir -p "$out/llama-headers" "$out/llama-lib"
for header in include/llama.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/include/ggml-alloc.h ggml/include/ggml-cpu.h ggml/include/ggml-opt.h ggml/include/gguf.h; do
    curl --fail --location --retry 2 "https://raw.githubusercontent.com/ggml-org/llama.cpp/$revision/$header" \
        -o "$out/llama-headers/${header##*/}"
done
container=$(docker create "$image")
trap 'docker rm "$container" >/dev/null' EXIT
for library in libllama.so.0 libggml.so.0 libggml-base.so.0; do
    docker cp -L "$container:/app/$library" "$out/llama-lib/$library"
done
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/workspace" -w /workspace \
    qw38-dev:cuda13.4.1-pinned bash -c '
        cmake --build build/pinned-release --target qw38_bench_request -j4 &&
        g++-14 -std=c++23 -O3 -Wall -Wextra -Wpedantic -static-libstdc++ -static-libgcc \
          -DQW38_LLAMA_BENCH -Isrc -I.cache/task027/llama-headers -I/usr/local/cuda/include \
          benchmarks/request_bench.cpp .cache/task027/llama-lib/libllama.so.0 \
          .cache/task027/llama-lib/libggml.so.0 .cache/task027/llama-lib/libggml-base.so.0 \
          -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/app \
          -o .cache/task027/llama-request-bench'
uv run --script scripts/task027_prepare.py prepare
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:$PWD" -w "$PWD" \
    -e LD_LIBRARY_PATH=/app --entrypoint "$PWD/$out/llama-request-bench" "$image" \
    --vocabulary models/Qwen3.8-27B-Q4_K_M.gguf "$out/prompt.txt" "$out/llama-vocab" \
    > "$out/vocabulary.log" 2>&1
uv run --script scripts/task027_prepare.py validate
uv run --with ruff ruff format scripts/task027_prepare.py scripts/task027_benchmark.py tests/test_task027_benchmark.py
uv run --with pytest pytest -q tests/test_task027_benchmark.py

# Use the same profiler binaries in both engine containers. Inference libraries
# remain those from each original image.
if [[ ! -d "$out/nsight-host" ]]; then
    profiler_container=$(docker create qw38-dev:cuda13.4.1-pinned)
    docker cp "$profiler_container:/opt/nvidia/nsight-compute/2026.3.0/host" "$out/nsight-host"
    docker rm "$profiler_container" >/dev/null
fi
