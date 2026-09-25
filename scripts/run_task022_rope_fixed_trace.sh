#!/usr/bin/env bash
set -euo pipefail

mkdir -p .cache/task022/layer3-corrected
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned \
    build/pinned-release/src/qw38-task022-trace \
    .cache/task022/rope-fixed-candidate.qw38 \
    .cache/task022/frozen-window-000.u32le \
    .cache/task022/layer3-corrected
uv run --script scripts/task022_compare_layer3_traces.py \
    .cache/task022/layer3-proxy \
    .cache/task022/layer3-corrected \
    .cache/task022/layer3-boundaries-corrected.json
