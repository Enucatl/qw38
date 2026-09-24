#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cutlass="${repo_root}/.cache/task019/cutlass-v4.8.0"
build="${repo_root}/.cache/task019/build-v4.8.0-sm120"
image="qw38-dev:cuda13.4.1-pinned"
host_python="/usr/bin/python3.12"
host_python_stdlib="/usr/lib/python3.12"

test "$(git -C "${cutlass}" rev-parse HEAD)" = "098de2a652cf8f00fd70b2df54051c7eccbb855a"

if [[ ! -x "${host_python}" || ! -d "${host_python_stdlib}" ]]; then
  echo "TASK-019 requires host Python 3.12 at ${host_python} and its stdlib at ${host_python_stdlib}" >&2
  exit 1
fi

docker run --rm --gpus all \
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
    # These generated binaries skip the exhaustive CUTLASS host reference.
    # They report that verification was skipped and are only used for timing.
    sed -e "s/result.passed = verify(options);/result.passed = true;/" \
      -e "s/  std::cout << \"  Disposition: \" << (result.passed ? \"Passed\" : \"Failed\") << std::endl;/  std::cout << \"  Verification: Skipped (benchmark-only)\" << std::endl;/" \
      "${source}" > "${build}/task019/79a_nvfp4_benchmark_only.cu"
    sed -e "s/nv_float4_t/mx_float4_t/g" \
      -e "s/result.passed = verify(options);/result.passed = true;/" \
      -e "s/  std::cout << \"  Disposition: \" << (result.passed ? \"Passed\" : \"Failed\") << std::endl;/  std::cout << \"  Verification: Skipped (benchmark-only)\" << std::endl;/" \
      "${source}" > "${build}/task019/79x_mxfp4_benchmark_only.cu"
    compile_example "${build}/task019/79a_nvfp4_benchmark_only.cu" "${nv_benchmark}"
    compile_example "${build}/task019/79x_mxfp4_benchmark_only.cu" "${mx_benchmark}"
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    nvcc --version | tail -4
    echo "Bounded CUTLASS host-reference checks: M=32 N=128 K=128, iterations=0"
    "${nv_correctness}" --m=32 --n=128 --k=128 --iterations=0
    "${mx_correctness}" --m=32 --n=128 --k=128 --iterations=0
    cuobjdump --dump-sass "${nv_benchmark}" > "${build}/task019/nvfp4.sass.txt"
    cuobjdump --dump-sass "${mx_benchmark}" > "${build}/task019/mxfp4.sass.txt"
    for sass in "${build}/task019/nvfp4.sass.txt" "${build}/task019/mxfp4.sass.txt"; do
      grep -nEm 20 "MMA|BLOCKSCALED|HMMA" "${sass}"
    done
    for m in 1 2 8 32 64 128 256 512 1024; do
      for shape in "5120 5120" "17408 5120" "248320 5120" "12288 5120"; do
        read -r n k <<< "${shape}"
        echo "Benchmark-only NVFP4 m=${m} n=${n} k=${k}"
        "${nv_benchmark}" \
          --m="${m}" --n="${n}" --k="${k}" --iterations=20
        echo "Benchmark-only MXFP4 m=${m} n=${n} k=${k}"
        "${mx_benchmark}" \
          --m="${m}" --n="${n}" --k="${k}" --iterations=20
      done
    done
  '
