#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
base=.cache/task019/inputs/real-matrix
mkdir -p "${base}"

extract() {
  local label="$1" tensor="$2" n="$3" k="$4"
  local extra=()
  if [[ "${k}" != 5120 ]]; then extra+=(--weights-only); fi
  echo "Input: ${label} tensor=${tensor} n=${n} k=${k}"
  python3 scripts/task019_real_inputs.py --tensor "${tensor}" --m 1024 \
    --n "${n}" --k "${k}" --out "${base}/${label}" "${extra[@]}"
}

extract mlp_gate model.language_model.layers.0.mlp.gate_proj.weight 17408 5120
extract mlp_up model.language_model.layers.0.mlp.up_proj.weight 17408 5120
extract mlp_down model.language_model.layers.0.mlp.down_proj.weight 5120 17408
extract gdn_qkv model.language_model.layers.0.linear_attn.in_proj_qkv.weight 10240 5120
extract gdn_z model.language_model.layers.0.linear_attn.in_proj_z.weight 6144 5120
extract gdn_out model.language_model.layers.0.linear_attn.out_proj.weight 5120 6144
extract gdn_a model.language_model.layers.0.linear_attn.in_proj_a.weight 48 5120
extract gdn_b model.language_model.layers.0.linear_attn.in_proj_b.weight 48 5120
extract attn_q model.language_model.layers.3.self_attn.q_proj.weight 12288 5120
extract attn_k model.language_model.layers.3.self_attn.k_proj.weight 1024 5120
extract attn_v model.language_model.layers.3.self_attn.v_proj.weight 1024 5120
extract attn_out model.language_model.layers.3.self_attn.o_proj.weight 5120 6144
extract lm_head lm_head.weight 248320 5120

image=qw38-dev:cuda13.4.1-pinned
expected=sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49
test "$(docker image inspect --format '{{.Id}}' "${image}")" = "${expected}"
docker run --rm --gpus all -v "${repo_root}:/workspace" -w /workspace \
  "${image}" bash -lc '
    set -euo pipefail
    binary=.cache/task019/task019_derive_activations_sm120
    nvcc -std=c++17 -O2 -arch=sm_120a scripts/task019_derive_activations.cu \
      -lcublas -o "${binary}"
    sha256sum "${binary}"
    base=.cache/task019/inputs/real-matrix
    echo "Activation proxy: silu(GDN z projection of normalized embedding), not complete GDN output"
    "${binary}" silu "${base}/gdn_z/a.bf16le" \
      "${base}/gdn_z/b.bf16le" - 1024 6144 5120 \
      "${base}/gdn_out/a.bf16le"
    cp "${base}/gdn_out/a.bf16le" "${base}/attn_out/a.bf16le"
    echo "Activation proxy: SwiGLU of layer-zero gate/up projections of normalized embedding"
    "${binary}" swiglu "${base}/mlp_gate/a.bf16le" \
      "${base}/mlp_gate/b.bf16le" "${base}/mlp_up/b.bf16le" \
      1024 17408 5120 "${base}/mlp_down/a.bf16le"
  '
