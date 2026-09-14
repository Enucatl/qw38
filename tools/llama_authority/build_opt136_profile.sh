#!/usr/bin/env bash
set -euo pipefail

# Private OPT-136 llama decode-profile build. Does not mutate the production
# llama.cpp checkout, llama-build tree, or qw38-llama-authority image.

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REVISION="cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
readonly PROD_SOURCE="${ROOT}/.cache/authorities/llama.cpp"
readonly PROD_BUILD="${ROOT}/.cache/authorities/llama-build"
readonly ADAPTER_BUILD="${ROOT}/.cache/authorities/llama-adapter-build-opt136"
readonly OUTPUT="${ROOT}/.cache/authorities/llama-build-opt136/bin"
readonly AUTHORITY_IMAGE="qw38-llama-authority:cuda-13.0.2"
readonly CUDA_IMAGE="qw38-cuda:13.0.2"
readonly PROVENANCE="${ROOT}/.cache/authorities/opt136-profile-provenance.json"
readonly BINARY="${OUTPUT}/qw38-llama-opt136-decode-profile"
readonly SRC="${ROOT}/tools/llama_authority/opt136_decode_profile.cpp"

mkdir -p "${OUTPUT}" "${ADAPTER_BUILD}" "$(dirname "${PROVENANCE}")"

write_provenance() {
  local status="$1"
  local image="${2:-none}"
  cat > "${PROVENANCE}" <<EOF
{
  "schema_version": 1,
  "task": "OPT-136",
  "kind": "private_profiling_build",
  "status": "${status}",
  "source_revision": "${REVISION}",
  "production_source": ".cache/authorities/llama.cpp",
  "production_build": ".cache/authorities/llama-build",
  "adapter_build_dir": ".cache/authorities/llama-adapter-build-opt136",
  "output": ".cache/authorities/llama-build-opt136/bin/qw38-llama-opt136-decode-profile",
  "production_source_untouched": true,
  "production_image_untouched": true,
  "image": "${image}"
}
EOF
}

stage_libs() {
  shopt -s nullglob
  for lib in "${PROD_BUILD}/bin"/libllama.so* "${PROD_BUILD}/bin"/libggml*.so*; do
    cp -f "${lib}" "${OUTPUT}/"
  done
}

if [[ -x "${BINARY}" && "${SRC}" -ot "${BINARY}" ]]; then
  write_provenance "ok" "cached"
  echo "OPT-136 llama profile already present: ${BINARY}"
  exit 0
fi

if [[ ! -f "${PROD_SOURCE}/include/llama.h" ]]; then
  write_provenance "source_missing"
  echo "pinned llama.cpp source missing at ${PROD_SOURCE}" >&2
  exit 2
fi
if [[ -d "${PROD_SOURCE}/.git" ]]; then
  prod_head="$(git -C "${PROD_SOURCE}" rev-parse HEAD)"
  if [[ "${prod_head}" != "${REVISION}" ]]; then
    echo "production llama.cpp HEAD ${prod_head} != ${REVISION}" >&2
    write_provenance "revision_mismatch"
    exit 1
  fi
fi
if [[ ! -d "${PROD_BUILD}/bin" ]]; then
  write_provenance "build_missing"
  echo "pinned llama build missing at ${PROD_BUILD}" >&2
  exit 2
fi

stage_libs

have_authority=0
have_cuda=0
if command -v docker >/dev/null 2>&1; then
  if docker image inspect "${AUTHORITY_IMAGE}" >/dev/null 2>&1; then
    have_authority=1
  fi
  if docker image inspect "${CUDA_IMAGE}" >/dev/null 2>&1; then
    have_cuda=1
  fi
fi

if [[ "${have_authority}" -eq 1 ]]; then
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "${ROOT}:/workspace" \
    "${AUTHORITY_IMAGE}" \
    cmake -S /workspace/tools/llama_authority \
          -B /workspace/.cache/authorities/llama-adapter-build-opt136 \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_SOURCE=/workspace/.cache/authorities/llama.cpp \
          -DLLAMA_BUILD=/workspace/.cache/authorities/llama-build \
          -DQW38_PROFILE_OUTPUT_DIR=/workspace/.cache/authorities/llama-build-opt136/bin
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "${ROOT}:/workspace" \
    "${AUTHORITY_IMAGE}" \
    cmake --build /workspace/.cache/authorities/llama-adapter-build-opt136 \
          --target qw38-llama-opt136-decode-profile -j 6
  if [[ -f "${ADAPTER_BUILD}/bin/qw38-llama-opt136-decode-profile" && ! -f "${BINARY}" ]]; then
    cp -f "${ADAPTER_BUILD}/bin/qw38-llama-opt136-decode-profile" "${BINARY}"
  fi
  write_provenance "ok" "${AUTHORITY_IMAGE}"
elif [[ "${have_cuda}" -eq 1 ]]; then
  # Compile only the private probe against the existing pinned llama libs.
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "${ROOT}:/workspace" \
    "${CUDA_IMAGE}" \
    g++ -std=c++17 -O2 -Wall -Wextra \
      -I /workspace/.cache/authorities/llama.cpp/include \
      -I /workspace/.cache/authorities/llama.cpp/ggml/include \
      -I /usr/local/cuda/include \
      -L /workspace/.cache/authorities/llama-build/bin \
      -L /usr/local/cuda/lib64 \
      -Wl,-rpath,'$ORIGIN' \
      /workspace/tools/llama_authority/opt136_decode_profile.cpp \
      -o /workspace/.cache/authorities/llama-build-opt136/bin/qw38-llama-opt136-decode-profile \
      -lllama -lggml -lggml-cuda -lggml-base -lcudart -pthread
  write_provenance "ok" "${CUDA_IMAGE}"
else
  echo "neither ${AUTHORITY_IMAGE} nor ${CUDA_IMAGE} is available" >&2
  write_provenance "image_missing"
  exit 2
fi

if [[ ! -x "${BINARY}" ]]; then
  echo "qw38-llama-opt136-decode-profile was not built" >&2
  write_provenance "adapter_missing"
  exit 1
fi

echo "OPT-136 llama profile ready: ${BINARY}"
