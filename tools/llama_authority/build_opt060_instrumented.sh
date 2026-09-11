#!/usr/bin/env bash
set -euo pipefail

# Private OPT-060 instrumentation build. Does not mutate the production
# llama.cpp checkout, llama-build tree, or qw38-llama-authority image.

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REVISION="cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
readonly PROD_SOURCE="${ROOT}/.cache/authorities/llama.cpp"
readonly SOURCE="${ROOT}/.cache/authorities/llama.cpp-opt060"
readonly BUILD="${ROOT}/.cache/authorities/llama-build-opt060"
readonly ADAPTER_BUILD="${ROOT}/.cache/authorities/llama-adapter-build-opt060"
readonly IMAGE="qw38-llama-authority:cuda-13.0.2"
readonly PATCH="${ROOT}/tools/llama_authority/patches/opt060-engine-attribution.patch"
readonly OVERLAY="${ROOT}/tools/llama_authority/patches/qw38_opt060_attribution.cuh"
readonly PROVENANCE="${ROOT}/.cache/authorities/opt060-instrumented-provenance.json"
readonly CMAKE_FLAGS_TEXT="CMAKE_BUILD_TYPE=Release CMAKE_CUDA_ARCHITECTURES=120 GGML_CUDA=ON GGML_NATIVE=OFF LLAMA_CURL=OFF LLAMA_BUILD_TESTS=OFF LLAMA_BUILD_EXAMPLES=ON QW38_OPT060_ATTRIBUTION=ON"

if [[ ! -f "${PATCH}" ]]; then
  echo "missing ${PATCH}" >&2
  exit 1
fi
if [[ ! -f "${OVERLAY}" ]]; then
  echo "missing ${OVERLAY}" >&2
  exit 1
fi

if [[ -d "${PROD_SOURCE}/.git" ]]; then
  prod_head="$(git -C "${PROD_SOURCE}" rev-parse HEAD)"
  if [[ "${prod_head}" != "${REVISION}" ]]; then
    echo "production llama.cpp HEAD ${prod_head} != ${REVISION}" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${PROD_SOURCE}" status --porcelain -- ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmq.cuh ggml/src/ggml-cuda/mmq.cu)" ]]; then
    echo "production llama.cpp CUDA sources are dirty; refusing to continue" >&2
    exit 1
  fi
fi

mkdir -p "${SOURCE}" "${BUILD}" "${ADAPTER_BUILD}" "$(dirname "${PROVENANCE}")"
if [[ ! -d "${SOURCE}/.git" ]]; then
  git clone --no-checkout "${PROD_SOURCE}" "${SOURCE}" 2>/dev/null || true
  if [[ ! -d "${SOURCE}/.git" ]]; then
    git -C "${SOURCE}" init
    git -C "${SOURCE}" remote add origin https://github.com/ggml-org/llama.cpp.git
  fi
fi
if [[ ! -f "${SOURCE}/ggml/src/ggml-cuda/ggml-cuda.cu" ]]; then
  git -C "${SOURCE}" fetch --depth 1 origin "${REVISION}" 2>/dev/null || \
    git -C "${SOURCE}" fetch --depth 1 "${PROD_SOURCE}" "${REVISION}" || true
  git -C "${SOURCE}" checkout --force --detach "${REVISION}"
fi
if git -C "${SOURCE}" rev-parse HEAD >/dev/null 2>&1; then
  if [[ "$(git -C "${SOURCE}" rev-parse HEAD)" != "${REVISION}" ]]; then
    echo "OPT-060 llama.cpp revision mismatch" >&2
    exit 1
  fi
fi

if [[ ! -f "${SOURCE}/ggml/src/ggml-cuda/qw38_opt060_attribution.cuh" ]]; then
  (cd "${SOURCE}" && patch -p1 --forward < "${PATCH}")
fi
/bin/cp -f "${OVERLAY}" "${SOURCE}/ggml/src/ggml-cuda/qw38_opt060_attribution.cuh"
if ! grep -q "QW38_OPT060_ATTRIBUTION" "${SOURCE}/ggml/src/ggml-cuda/ggml-cuda.cu"; then
  echo "patch did not instrument ggml-cuda.cu" >&2
  exit 1
fi

PATCH_SHA="$(sha256sum "${PATCH}" | awk '{print $1}')"
OVERLAY_SHA="$(sha256sum "${OVERLAY}" | awk '{print $1}')"
IMAGE_ID="unavailable"
if command -v docker >/dev/null 2>&1; then
  IMAGE_ID="$(docker image inspect -f '{{.Id}}' "${IMAGE}" 2>/dev/null || echo missing)"
fi

write_provenance() {
  local status="$1"
  cat > "${PROVENANCE}" <<EOF
{
  "schema_version": 1,
  "task": "OPT-060",
  "kind": "private_instrumentation_build",
  "status": "${status}",
  "source_revision": "${REVISION}",
  "worktree": ".cache/authorities/llama.cpp-opt060",
  "build_dir": ".cache/authorities/llama-build-opt060",
  "adapter_build_dir": ".cache/authorities/llama-adapter-build-opt060",
  "production_source_untouched": true,
  "production_image_untouched": true,
  "patch": "tools/llama_authority/patches/opt060-engine-attribution.patch",
  "patch_sha256": "${PATCH_SHA}",
  "overlay": "tools/llama_authority/patches/qw38_opt060_attribution.cuh",
  "overlay_sha256": "${OVERLAY_SHA}",
  "cmake_flags": "${CMAKE_FLAGS_TEXT}",
  "image": "${IMAGE}",
  "image_id": "${IMAGE_ID}",
  "binary": ".cache/authorities/llama-build-opt060/bin/qw38-llama-engine-attribution"
}
EOF
}

if [[ -x "${BUILD}/bin/qw38-llama-engine-attribution" && -f "${SOURCE}/ggml/src/ggml-cuda/qw38_opt060_attribution.cuh" ]]; then
  adapter_src="${ROOT}/tools/llama_authority/engine_attribution.cpp"
  adapter_newer=false
  cuda_newer=false
  if [[ "${adapter_src}" -nt "${BUILD}/bin/qw38-llama-engine-attribution" ]]; then
    adapter_newer=true
  fi
  if [[ "${OVERLAY}" -nt "${BUILD}/bin/qw38-llama-engine-attribution" ]]; then
    cuda_newer=true
  fi
  if [[ "${PATCH}" -nt "${BUILD}/bin/qw38-llama-engine-attribution" ]]; then
    cuda_newer=true
  fi
  if [[ "${adapter_newer}" == false && "${cuda_newer}" == false ]]; then
    write_provenance "ok"
    echo "OPT-060 instrumented llama already present: ${BUILD}/bin/qw38-llama-engine-attribution"
    exit 0
  fi
  if [[ "${adapter_newer}" == true && "${cuda_newer}" == false ]]; then
    echo "OPT-060 adapter source is newer than the binary; rebuilding adapter only"
    write_provenance "configured"
    if [[ "${IMAGE_ID}" == "missing" || "${IMAGE_ID}" == "unavailable" ]]; then
      echo "docker image ${IMAGE} is missing" >&2
      write_provenance "image_missing"
      exit 2
    fi
    docker run --rm --gpus all \
      --user "$(id -u):$(id -g)" \
      -v "${ROOT}:/workspace" \
      "${IMAGE}" \
      cmake -S /workspace/tools/llama_authority \
            -B /workspace/.cache/authorities/llama-adapter-build-opt060 \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_SOURCE=/workspace/.cache/authorities/llama.cpp-opt060 \
            -DLLAMA_BUILD=/workspace/.cache/authorities/llama-build-opt060 \
            -DQW38_OPT060_ATTRIBUTION=ON
    docker run --rm --gpus all \
      --user "$(id -u):$(id -g)" \
      -v "${ROOT}:/workspace" \
      "${IMAGE}" \
      cmake --build /workspace/.cache/authorities/llama-adapter-build-opt060 \
            --target qw38-llama-engine-attribution -j 6
    if [[ ! -x "${BUILD}/bin/qw38-llama-engine-attribution" ]]; then
      echo "qw38-llama-engine-attribution was not built" >&2
      write_provenance "adapter_missing"
      exit 1
    fi
    write_provenance "ok"
    echo "OPT-060 instrumented llama ready: ${BUILD}/bin/qw38-llama-engine-attribution"
    exit 0
  fi
  echo "OPT-060 CUDA instrumentation sources are newer than the binary; rebuilding"
fi

write_provenance "configured"

if [[ "${IMAGE_ID}" == "missing" || "${IMAGE_ID}" == "unavailable" ]]; then
  echo "docker image ${IMAGE} is missing; build with: docker build -f docker/llama-authority.Dockerfile -t ${IMAGE} ." >&2
  write_provenance "image_missing"
  exit 2
fi

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake -S /workspace/.cache/authorities/llama.cpp-opt060 \
        -B /workspace/.cache/authorities/llama-build-opt060 \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DCMAKE_CUDA_FLAGS="-DQW38_OPT060_ATTRIBUTION" \
        -DCMAKE_CXX_FLAGS="-DQW38_OPT060_ATTRIBUTION"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake --build /workspace/.cache/authorities/llama-build-opt060 \
        --target ggml ggml-cuda llama -j 6

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake -S /workspace/tools/llama_authority \
        -B /workspace/.cache/authorities/llama-adapter-build-opt060 \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_SOURCE=/workspace/.cache/authorities/llama.cpp-opt060 \
        -DLLAMA_BUILD=/workspace/.cache/authorities/llama-build-opt060 \
        -DQW38_OPT060_ATTRIBUTION=ON

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  cmake --build /workspace/.cache/authorities/llama-adapter-build-opt060 \
        --target qw38-llama-engine-attribution -j 6

# Stage the adapter next to libllama for RPATH $ORIGIN.
mkdir -p "${BUILD}/bin"
if [[ -f "${ADAPTER_BUILD}/bin/qw38-llama-engine-attribution" ]]; then
  cp -f "${ADAPTER_BUILD}/bin/qw38-llama-engine-attribution" \
        "${BUILD}/bin/qw38-llama-engine-attribution"
fi
if [[ ! -x "${BUILD}/bin/qw38-llama-engine-attribution" ]]; then
  echo "qw38-llama-engine-attribution was not built" >&2
  write_provenance "adapter_missing"
  exit 1
fi

write_provenance "ok"
echo "OPT-060 instrumented llama ready: ${BUILD}/bin/qw38-llama-engine-attribution"
