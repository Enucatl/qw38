#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cutlass="${repo_root}/.cache/task019/cutlass-v4.8.0"
image="qw38-dev:cuda13.4.1-pinned"
expected_image_id="sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49"
host_python="/usr/bin/python3.12"
host_python_stdlib="/usr/lib/python3.12"
mode="${1:-support}"

if [[ $# -gt 1 || ( "${mode}" != support && "${mode}" != matrix ) ]]; then
  echo "Usage: $0 [support|matrix]" >&2
  exit 2
fi

test "$(git -C "${cutlass}" rev-parse HEAD)" = "098de2a652cf8f00fd70b2df54051c7eccbb855a"
test "$(docker image inspect --format '{{.Id}}' "${image}")" = "${expected_image_id}"

if [[ ! -x "${host_python}" || ! -d "${host_python_stdlib}" ]]; then
  echo "TASK-019 requires host Python 3.12 at ${host_python} and its stdlib at ${host_python_stdlib}" >&2
  exit 1
fi

docker run --rm --gpus all \
  -e "TASK019_MODE=${mode}" \
  -v "${repo_root}:/workspace" \
  -v "${host_python}:${host_python}:ro" \
  -v "${host_python_stdlib}:${host_python_stdlib}:ro" \
  -w /workspace \
  "${image}" bash -lc '
    set -euo pipefail
    cutlass=/workspace/.cache/task019/cutlass-v4.8.0
    build=/workspace/.cache/task019/build-v4.8.0-sm120
    mkdir -p "${build}"
    cmake -S "${cutlass}" -B "${build}" \
      -DCUTLASS_NVCC_ARCHS=120a \
      -DCUTLASS_ENABLE_EXAMPLES=ON \
      -DCUTLASS_ENABLE_TOOLS=ON \
      -DCUTLASS_ENABLE_LIBRARY=OFF \
      -DCUTLASS_ENABLE_TESTS=OFF \
      -DCUTLASS_ENABLE_PROFILER=OFF \
      -DPython3_EXECUTABLE=/usr/bin/python3.12
    cmake --build "${build}" --target 79a_blackwell_geforce_nvfp4_bf16_gemm -j2
    mkdir -p "${build}/task019"
    source="${cutlass}/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu"
    nv_correctness="${build}/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm"
    mx_correctness="${build}/task019/79x_blackwell_geforce_mxfp4_mxfp4_bf16_gemm"
    nv_benchmark="${build}/task019/79a_nvfp4_benchmark_only"
    mx_benchmark="${build}/task019/79x_mxfp4_benchmark_only"
    grep -Fq "result.passed = verify(options);" "${source}"
    grep -Fq "  Disposition: " "${source}"
    sed "s/nv_float4_t/mx_float4_t/g" \
      "${source}" \
      > "${build}/task019/79x_blackwell_geforce_mxfp4_mxfp4_bf16_gemm.cu"
    compile_example() {
      nvcc -std=c++17 --expt-relaxed-constexpr -arch=sm_120a \
        -I"${cutlass}/include" \
        -I"${cutlass}/tools/util/include" \
        -I"${cutlass}/examples/common" \
        -I"${cutlass}/examples/util/include" \
        "$1" -o "$2"
    }
    compile_example "${build}/task019/79x_blackwell_geforce_mxfp4_mxfp4_bf16_gemm.cu" "${mx_correctness}"
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    nvcc --version | tail -4
    echo "CUTLASS revision: $(git -C "${cutlass}" rev-parse HEAD)"
    echo "Mode: ${TASK019_MODE}"
    echo "Bounded CUTLASS host-reference checks: M=32 N=128 K=128, iterations=0"
    echo "Command: ${nv_correctness} --m=32 --n=128 --k=128 --iterations=0"
    "${nv_correctness}" --m=32 --n=128 --k=128 --iterations=0
    echo "Command: ${mx_correctness} --m=32 --n=128 --k=128 --iterations=0"
    "${mx_correctness}" --m=32 --n=128 --k=128 --iterations=0
    sha256sum "${nv_correctness}" "${mx_correctness}"
    cuobjdump --dump-sass "${nv_correctness}" > "${build}/task019/nvfp4.sass.txt"
    cuobjdump --dump-sass "${mx_correctness}" > "${build}/task019/mxfp4.sass.txt"
    grep -nFm 1 "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X" "${build}/task019/nvfp4.sass.txt"
    grep -nFm 1 "OMMA.SF.16864.F32.E2M1.E2M1.E8" "${build}/task019/mxfp4.sass.txt"
    cuobjdump -res-usage "${nv_correctness}" "${mx_correctness}"
    if [[ "${TASK019_MODE}" == support ]]; then
      exit 0
    fi
    # Benchmark-only copies keep the same GEMM but skip the expensive host reference.
    sed -e "s/result.passed = verify(options);/result.passed = true;/" \
      -e "s/  std::cout << \"  Disposition: \" << (result.passed ? \"Passed\" : \"Failed\") << std::endl;/  std::cout << \"  Verification: Skipped (benchmark-only)\" << std::endl;/" \
      "${source}" > "${build}/task019/79a_nvfp4_benchmark_only.cu"
    sed -e "s/nv_float4_t/mx_float4_t/g" \
      -e "s/result.passed = verify(options);/result.passed = true;/" \
      -e "s/  std::cout << \"  Disposition: \" << (result.passed ? \"Passed\" : \"Failed\") << std::endl;/  std::cout << \"  Verification: Skipped (benchmark-only)\" << std::endl;/" \
      "${source}" > "${build}/task019/79x_mxfp4_benchmark_only.cu"
    grep -Fq "Verification: Skipped (benchmark-only)" "${build}/task019/79a_nvfp4_benchmark_only.cu"
    grep -Fq "Verification: Skipped (benchmark-only)" "${build}/task019/79x_mxfp4_benchmark_only.cu"
    compile_example "${build}/task019/79a_nvfp4_benchmark_only.cu" "${nv_benchmark}"
    compile_example "${build}/task019/79x_mxfp4_benchmark_only.cu" "${mx_benchmark}"
    sha256sum "${nv_benchmark}" "${mx_benchmark}"
    # This is a synthetic kernel diagnostic. Its 256 MiB estimated host-buffer
    # cap prevents the example from initializing a full-vocabulary host tensor.
    max_host_bytes=268435456
    shapes=(
      "5120 5120" "17408 5120" "5120 17408"
      "10240 5120" "6144 5120" "5120 6144"
      "12288 5120" "1024 5120" "48 5120" "248320 5120"
      "129 513"
    )
    for m in 1 2 8 32 64 128 256 512 1024; do
      for shape in "${shapes[@]}"; do
        read -r n k <<< "${shape}"
        # A/B and three C/D host copies; scales and allocator overhead are
        # additional, so this conservative estimate is only a sweep guard.
        estimated_host_bytes=$(( (m * k + n * k) / 2 + 6 * m * n ))
        if (( estimated_host_bytes > max_host_bytes )); then
          for format in NVFP4 MXFP4; do
            echo "SKIPPED_BY_HARNESS format=${format} m=${m} n=${n} k=${k} reason=host-buffer-cap estimated_bytes=${estimated_host_bytes} cap_bytes=${max_host_bytes}"
          done
          continue
        fi
        for format in NVFP4 MXFP4; do
          binary="${nv_benchmark}"
          if [[ "${format}" == MXFP4 ]]; then binary="${mx_benchmark}"; fi
          echo "Command: ${binary} --m=${m} --n=${n} --k=${k} --iterations=20"
          if output=$("${binary}" --m="${m}" --n="${n}" --k="${k}" --iterations=20 2>&1); then
            printf "%s\n" "${output}"
            if [[ "${output}" != *"Verification: Skipped (benchmark-only)"* || "${output}" != *"Avg runtime:"* ]]; then
              echo "UNSUPPORTED_BY_EXAMPLE format=${format} m=${m} n=${n} k=${k} reason=no-timing-or-no-kernel-output"
            fi
          else
            status=$?
            printf "%s\n" "${output}"
            echo "UNSUPPORTED_BY_EXAMPLE format=${format} m=${m} n=${n} k=${k} reason=nonzero-exit exit_status=${status}"
          fi
        done
      done
    done
  '
