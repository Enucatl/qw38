#!/usr/bin/env bash
set -euo pipefail

root=.cache/q4k-candidate
image=qw38-dev:cuda13.4.1-pinned
mkdir -p "$root"
runner=(docker run --rm --gpus all --entrypoint "" -u "$(id -u):$(id -g)"
        -v "$PWD:/workspace" -v "$PWD:$PWD" -w /workspace "$image")

uv run --script scripts/q4k_development_screen.py prepare
sha256sum build/pinned-release/src/qw38-compile \
    build/pinned-release/src/qw38-evaluate \
    build/pinned-release/tests/qw38_q4k_artifact_audit > "$root/binaries.sha256"
"${runner[@]}" build/pinned-release/src/qw38-compile \
    --checkpoint .cache/authorities/qwen3.8-27b-transformers \
    --output "$root/candidate.qw38" --format candidate-q4k \
    --verify-reconstruction > "$root/compile.log" 2>&1
cat "$root/compile.log"
"${runner[@]}" build/pinned-release/tests/qw38_q4k_artifact_audit \
    .cache/task021/candidate.qw38 "$root/candidate.qw38" > "$root/artifact-audit.json"
cat "$root/artifact-audit.json"

for arm in q4g64 q4k; do
    artifact=.cache/task021/candidate.qw38
    if [[ "$arm" == q4k ]]; then artifact="$root/candidate.qw38"; fi
    "${runner[@]}" build/pinned-release/src/qw38-evaluate \
        --artifact "$artifact" --cases "$root/development/cases.tsv" \
        --output "$root/development/$arm" \
        --eos-ids 248044,248046,248063,248064,248065 \
        > "$root/development/$arm.log" 2>&1
    cat "$root/development/$arm.log"
done
uv run --script scripts/q4k_development_screen.py report
