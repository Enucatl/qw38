#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned \
    bash -lc 'cmake --build build/pinned-release --target qw38_compiler_identity_test qw38_compiler_transform_test qw38_compiler_integration_test qw38_compile qw38_evaluate -j4 && ctest --test-dir build/pinned-release --output-on-failure -R "^(compiler_identity|compiler_transform|compiler_integration)$"'
