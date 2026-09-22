#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <cmake-target> <binary-path>" >&2
  exit 2
fi

target=$1
binary=$2
image=${QW38_BENCHMARK_IMAGE:-qw38-dev:cuda13.4.1}

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ -n $(git status --porcelain) ]]; then
  echo "benchmark evidence requires a clean source tree" >&2
  exit 1
fi

source_revision=$(git rev-parse --verify HEAD)
image_identity=$(docker image inspect "$image" --format '{{.Id}}')
if [[ ! $image_identity =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "could not resolve immutable container image ID for $image" >&2
  exit 1
fi

docker run --rm -u "$(id -u):$(id -g)" \
  -v "$repo_root":/workspace -w /workspace "$image_identity" \
  cmake --build build/release --target "$target" -j2

if [[ ! -x $binary ]]; then
  echo "benchmark binary is not executable: $binary" >&2
  exit 1
fi

executable_sha256=$(sha256sum "$binary" | awk '{print $1}')
build_id=$(readelf -n "$binary" |
  awk '/Build ID:/ { print $3; found=1; exit } END { if (!found) exit 1 }')

printf '%s\n' \
  "source_revision=$source_revision" \
  "executable_sha256=$executable_sha256" \
  "executable_build_id=$build_id" \
  "container_image_digest=$image_identity"

docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$repo_root":/workspace -w /workspace "$image_identity" \
  "$binary"
