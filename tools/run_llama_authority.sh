#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL="${ROOT}/models/Qwen3.8-27B-Q4_K_M.gguf"
readonly EVIDENCE="${ROOT}/.cache/authorities/llama-evidence"
readonly IMAGE="qw38-llama-authority:cuda-13.0.2"

mkdir -p "${EVIDENCE}"
"${ROOT}/build/qw38-eval" --render-template-case user_no_thinking \
  >"${EVIDENCE}/template.bytes"
readonly TEMPLATE_HEX="$(od -An -vtx1 "${EVIDENCE}/template.bytes" | tr -d ' \n')"
"${ROOT}/build/qw38-eval" --tokenize-hex "${MODEL}" "${TEMPLATE_HEX}" \
  >"${EVIDENCE}/native-template.ids" \
  2>"${EVIDENCE}/native-template.stderr"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  /workspace/.cache/authorities/llama-build/bin/qw38-llama-token-oracle \
    --tokenize-file /workspace/models/Qwen3.8-27B-Q4_K_M.gguf \
    /workspace/.cache/authorities/llama-evidence/template.bytes \
  >"${EVIDENCE}/llama-template.stdout" \
  2>"${EVIDENCE}/llama-template.stderr"
"${ROOT}/build/qw38-eval" --verify-model "${MODEL}" \
  >"${EVIDENCE}/model.stdout" 2>"${EVIDENCE}/model.stderr"
"${ROOT}/build/qw38-eval" --dump-real-scalar-logits \
  "${MODEL}" "${EVIDENCE}/native.f32le.bin" \
  >"${EVIDENCE}/native.stdout" 2>"${EVIDENCE}/native.stderr"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${ROOT}:/workspace" \
  "${IMAGE}" \
  /workspace/.cache/authorities/llama-build/bin/qw38-llama-token-oracle \
    /workspace/models/Qwen3.8-27B-Q4_K_M.gguf \
    /workspace/.cache/authorities/llama-evidence/llama.f32le.bin 42 3649 \
  >"${EVIDENCE}/llama.stdout" 2>"${EVIDENCE}/llama.stderr"
(
  cd "${ROOT}"
  uv run python -m tools.compare_llama_authority \
    --native-logits "${EVIDENCE}/native.f32le.bin" \
    --llama-logits "${EVIDENCE}/llama.f32le.bin" \
    --native-stdout "${EVIDENCE}/native.stdout" \
    --llama-stdout "${EVIDENCE}/llama.stdout" \
    --contract "${ROOT}/pins/llama_authority_contract.json" \
    --template-bytes "${EVIDENCE}/template.bytes" \
    --native-template-ids "${EVIDENCE}/native-template.ids" \
    --llama-template-stdout "${EVIDENCE}/llama-template.stdout" \
    --output "${ROOT}/fixtures/llama_scalar_authority.json"
)
