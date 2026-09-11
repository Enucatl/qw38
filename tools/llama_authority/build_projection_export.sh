#!/usr/bin/env bash
set -euo pipefail

# Rebuild only the pinned llama GPU projection exporter. Does not mutate
# production llama.cpp sources.

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly IMAGE="qw38-llama-authority:cuda-13.0.2"
readonly SOURCE="${ROOT}/.cache/authorities/llama.cpp"
readonly BUILD="${ROOT}/.cache/authorities/llama-build"
readonly ADAPTER="${ROOT}/.cache/authorities/llama-adapter-build"
readonly REVISION="cc83d7b4824f73cfdda4dfbb47ee39804f71b328"

if [[ ! -d "${SOURCE}/.git" ]]; then
  echo "missing llama.cpp authority checkout" >&2
  exit 1
fi
head="$(git -C "${SOURCE}" rev-parse HEAD)"
if [[ "${head}" != "${REVISION}" ]]; then
  echo "llama.cpp HEAD ${head} != ${REVISION}" >&2
  exit 1
fi

mkdir -p "${ADAPTER}"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake -S /workspace/tools/llama_authority \
        -B /workspace/.cache/authorities/llama-adapter-build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_SOURCE=/workspace/.cache/authorities/llama.cpp \
        -DLLAMA_BUILD=/workspace/.cache/authorities/llama-build
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake --build /workspace/.cache/authorities/llama-adapter-build \
        --target qw38-llama-projection-export -j 6
