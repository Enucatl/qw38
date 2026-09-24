#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="qw38-dev:cuda13.4.1-pinned"
expected="sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49"
test "$(docker image inspect --format '{{.Id}}' "${image}")" = "${expected}"
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 A.bf16le B.bf16le M N K [samples]" >&2
  exit 2
fi

docker run --rm --gpus all -v "${repo_root}:/workspace" -w /workspace \
  "${image}" bash -lc '
    set -euo pipefail
    binary=.cache/task019/task019_controls_sm120
    nvcc -std=c++17 -O2 -arch=sm_120a scripts/task019_controls.cu \
      -lcublas -o "${binary}"
    sha256sum "${binary}"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.current.sm,power.draw,memory.free --format=csv,noheader
    echo "Command: ${binary} $*"
    "${binary}" "$@"
  ' task019_controls "$@"
