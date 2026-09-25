#!/usr/bin/env bash
set -euo pipefail

root=.cache/task022
runner=(docker run --rm --gpus all --entrypoint "" -u "$(id -u):$(id -g)"
        -v "$PWD:/workspace" -v "$PWD:$PWD" -w /workspace
        qw38-dev:cuda13.4.1-pinned)

sha256sum build/pinned-release/src/qw38-compile \
    build/pinned-release/src/qw38-evaluate > "$root/rope-fixed-binaries.sha256"
"${runner[@]}" build/pinned-release/src/qw38-compile \
    --checkpoint .cache/authorities/qwen3.8-27b-transformers \
    --output "$root/rope-fixed-candidate.qw38" --format candidate-q4k \
    --verify-reconstruction > "$root/rope-fixed-compile.log" 2>&1
cat "$root/rope-fixed-compile.log"
"${runner[@]}" build/pinned-release/src/qw38-evaluate \
    --artifact "$root/rope-fixed-candidate.qw38" \
    --cases .cache/q4k-candidate/development/cases.tsv \
    --output "$root/rope-fixed-development" \
    --eos-ids 248044,248046,248063,248064,248065 \
    > "$root/rope-fixed-development.log" 2>&1
cat "$root/rope-fixed-development.log"
uv run --script scripts/task022_report_rope_fix.py \
    .cache/q4k-candidate/development/report.json \
    .cache/q4k-candidate/development \
    "$root/rope-fixed-development" \
    "$root/rope-fixed-development-report.json"
