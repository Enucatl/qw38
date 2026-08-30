#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REVISION="cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
readonly SOURCE="${ROOT}/.cache/authorities/llama.cpp"
readonly IMAGE="qw38-llama-authority:cuda-13.0.2"

mkdir -p "${SOURCE}" "${ROOT}/.cache/authorities/llama-build"
if [[ ! -d "${SOURCE}/.git" ]]; then
  git -C "${SOURCE}" init
  git -C "${SOURCE}" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [[ -n "$(git -C "${SOURCE}" status --porcelain)" ]]; then
  echo "llama.cpp authority checkout is dirty" >&2
  exit 1
fi
git -C "${SOURCE}" fetch --depth 1 origin "${REVISION}"
git -C "${SOURCE}" checkout --detach "${REVISION}"
if [[ "$(git -C "${SOURCE}" rev-parse HEAD)" != "${REVISION}" ]]; then
  echo "llama.cpp authority revision mismatch" >&2
  exit 1
fi

docker build \
  -f "${ROOT}/docker/llama-authority.Dockerfile" \
  -t "${IMAGE}" \
  "${ROOT}"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake -S /workspace/.cache/authorities/llama.cpp \
        -B /workspace/.cache/authorities/llama-build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake --build /workspace/.cache/authorities/llama-build \
        --target llama-eval-callback -j 6
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
  cmake --build /workspace/.cache/authorities/llama-adapter-build -j 6

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  bash -c '/workspace/.cache/authorities/llama-build/bin/llama-eval-callback --help >/dev/null || exit 1; /workspace/.cache/authorities/llama-build/bin/qw38-llama-token-oracle 2>/dev/null; status=$?; test "$status" -eq 2'
