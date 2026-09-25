#!/usr/bin/env bash
set -euo pipefail

artifact=${1:-.cache/candidates/candidate-v2-q4k-rope-fixed.qw38}
mkdir -p .cache/task023
for tokens in 128 256 512 1024; do
  prefix=".cache/task023/nsys-${tokens}"
  QW38_NSYS_CAPTURE=1 nsys profile \
    --trace=cuda,cublas,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$prefix" \
    build/pinned-release/benchmarks/qw38_bench_prefill \
    "$artifact" 512 all "$tokens" 1 > "${prefix}-run.csv"
  nsys stats --force-export=true --report nvtx_kern_sum --format csv \
    "${prefix}.nsys-rep" > "${prefix}-kernels.csv"
  nsys stats --report cuda_api_sum --format csv \
    "${prefix}.nsys-rep" > "${prefix}-apis.csv"
done
