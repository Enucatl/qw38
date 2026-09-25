#!/usr/bin/env bash
set -euo pipefail

artifact=${1:-.cache/candidates/candidate-v2-q4k-rope-fixed.qw38}
mkdir -p .cache/task023
for tile in 128 256 512; do
  build/pinned-release/benchmarks/qw38_bench_prefill "$artifact" "$tile" all 0 20 \
    > ".cache/task023/sweep-${tile}.csv"
done
