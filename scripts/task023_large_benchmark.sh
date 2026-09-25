#!/usr/bin/env bash
set -euo pipefail

artifact=${1:-.cache/candidates/candidate-v2-q4k-rope-fixed.qw38}
mkdir -p .cache/task023
for tokens in 256 512 1024; do
  build/pinned-release/benchmarks/qw38_bench_prefill "$artifact" 512 all "$tokens" 20 \
    > ".cache/task023/large-${tokens}.csv"
done
