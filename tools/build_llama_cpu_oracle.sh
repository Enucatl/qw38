#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/bin:${PATH}"
readonly REVISION="cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
readonly SOURCE="${ROOT}/.cache/authorities/llama.cpp"
readonly BUILD="${ROOT}/.cache/authorities/llama-cpu-build"
readonly ADAPTER="${ROOT}/.cache/authorities/llama-cpu-adapter-build"

mkdir -p "${SOURCE}" "${BUILD}" "${ADAPTER}"
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

cmake -S "${SOURCE}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "${BUILD}" --target llama-eval-callback -j "$(sysctl -n hw.ncpu)"

cmake -S "${ROOT}/tools/llama_authority" -B "${ADAPTER}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SOURCE="${SOURCE}" \
  -DLLAMA_BUILD="${BUILD}"
cmake --build "${ADAPTER}" -j "$(sysctl -n hw.ncpu)"

"${BUILD}/bin/qw38-llama-token-oracle" >/dev/null 2>&1 || status=$?
if [[ "${status:-0}" -ne 2 ]]; then
  echo "CPU llama token oracle did not fail closed on missing arguments" >&2
  exit 1
fi
echo "cpu_llama_oracle=${BUILD}/bin/qw38-llama-token-oracle"
