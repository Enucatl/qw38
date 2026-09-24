#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-smoke}"
if [[ $# -gt 1 || ( "${mode}" != smoke && "${mode}" != core &&
                    "${mode}" != head && "${mode}" != full ) ]]; then
  echo "usage: $0 [smoke|core|head|full]" >&2
  exit 2
fi
image=qw38-dev:cuda13.4.1-pinned
expected=sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49
test "$(docker image inspect --format '{{.Id}}' "${image}")" = "${expected}"
test "$(git -C "${repo_root}/.cache/task019/cutlass-v4.8.0" rev-parse HEAD)" = \
  098de2a652cf8f00fd70b2df54051c7eccbb855a

docker run --rm --gpus all -e "TASK019_MATRIX_MODE=${mode}" \
  -v "${repo_root}:/workspace" -w /workspace "${image}" bash -lc '
    set -euo pipefail
    base=.cache/task019/inputs/real-matrix
    out=.cache/task019/real-matrix/${TASK019_MATRIX_MODE}
    mkdir -p "${out}"
    nv=.cache/task019/build-v4.8.0-sm120/task019/task019_nvfp4_independent_probe
    mx=.cache/task019/build-v4.8.0-sm120/task019/task019_mxfp4_independent_probe
    control=.cache/task019/task019_controls_sm120
    sha256sum "${nv}" "${mx}" "${control}" scripts/task019_independent_probe.cu scripts/task019_controls.cu
    nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.current.sm,power.draw,memory.free --format=csv,noheader
    nvcc --version | tail -4
    failed=0
    if [[ "${TASK019_MATRIX_MODE}" == smoke ]]; then
      cases=("mlp_gate 17408 5120 1" "gdn_out 5120 6144 8" \
             "mlp_down 5120 17408 8")
      samples=5
    else
      cases=(
        "mlp_gate 17408 5120" "mlp_up 17408 5120"
        "mlp_down 5120 17408" "gdn_qkv 10240 5120"
        "gdn_z 6144 5120" "gdn_out 5120 6144"
        "gdn_a 48 5120" "gdn_b 48 5120"
        "attn_q 12288 5120" "attn_k 1024 5120"
        "attn_v 1024 5120" "attn_out 5120 6144"
        "lm_head 248320 5120"
      )
      if [[ "${TASK019_MATRIX_MODE}" == core ]]; then
        cases=("${cases[@]:0:12}")
      elif [[ "${TASK019_MATRIX_MODE}" == head ]]; then
        cases=("${cases[12]}")
      fi
      samples=20
    fi
    for descriptor in "${cases[@]}"; do
      read -r label n k smoke_m <<< "${descriptor}"
      a="${base}/${label}/a.bf16le"
      b="${base}/${label}/b.bf16le"
      if [[ "${TASK019_MATRIX_MODE}" == smoke ]]; then
        m_values=("${smoke_m}")
      else
        m_values=(1 2 8 32 64 128 256 512 1024)
      fi
      for m in "${m_values[@]}"; do
        log="${out}/${label}-m${m}.log"
        {
          echo "Descriptor: family=${label} m=${m} n=${n} k=${k} logical_flops=$((2*m*n*k)) output=BF16 epilogue=identity"
          echo "Input metadata: ${base}/${label}/inputs.json"
          if [[ "${k}" != 5120 ]]; then echo "Activation: derived BF16 proxy; see prepare-real-inputs.log"; fi
          for format in nvfp4 mxfp4; do
            binary="${nv}"
            if [[ "${format}" == mxfp4 ]]; then binary="${mx}"; fi
            echo "Command: ${binary} --m=${m} --n=${n} --k=${k} --a-file=${a} --b-file=${b} --samples=${samples}"
            if "${binary}" --m="${m}" --n="${n}" --k="${k}" \
              --a-file="${a}" --b-file="${b}" --samples="${samples}"; then
              echo "Result: ${format} PASS"
            else
              status=$?
              echo "Result: ${format} FAILED exit=${status}"
              failed=1
            fi
          done
          echo "Command: ${control} ${a} ${b} ${m} ${n} ${k} ${samples}"
          if "${control}" "${a}" "${b}" "${m}" "${n}" "${k}" "${samples}"; then
            echo "Result: bf16_q4_controls PASS"
          else
            status=$?
            echo "Result: bf16_q4_controls FAILED exit=${status}"
            failed=1
          fi
        } > "${log}" 2>&1
        echo "Finished ${label} m=${m} log=${log}"
      done
    done
    exit "${failed}"
  '
