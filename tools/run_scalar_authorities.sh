#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
evidence_root="${repository_root}/.cache/authorities/scalar-alignment"
mkdir -p "${evidence_root}"
run_directory="$(mktemp -d "${evidence_root}/run.XXXXXX")"
model="${repository_root}/models/Qwen3.8-27B-Q4_K_M.gguf"
transformers_source="${repository_root}/.cache/authorities/transformers"
transformers_environment="${repository_root}/.cache/authorities/transformers-venv"
checkpoint="${repository_root}/.cache/authorities/qwen3.8-27b-transformers"

make -C "${repository_root}" diagnostic
"${repository_root}/tools/build_llama_authority.sh"

"${repository_root}/build/qw38-eval-diagnostic" \
  --capture-real-scalar-bundle \
  "${model}" "${run_directory}/quartz.f32le.bin" 42 3649 \
  >"${run_directory}/quartz.fields"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${repository_root}:/workspace" \
  qw38-llama-authority:cuda-13.0.2 \
  /workspace/.cache/authorities/llama-build/bin/qw38-llama-token-oracle \
  --trace /workspace/models/Qwen3.8-27B-Q4_K_M.gguf \
  "/workspace/${run_directory#"${repository_root}/"}/llama.f32le.bin" 42 3649 \
  >"${run_directory}/llama.fields" 2>"${run_directory}/llama.stderr"

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
PYTHONPATH="${repository_root}:${transformers_source}/src" \
  "${transformers_environment}/bin/python" \
  -m tools.run_transformers_authority \
  --checkpoint "${checkpoint}" \
  --source "${transformers_source}" \
  --contract "${repository_root}/pins/transformers_authority_contract.json" \
  --offload "${run_directory}/offload" \
  --logits "${run_directory}/official-logits.f32le.bin" \
  --taps "${run_directory}/official-taps.f32le.bin" \
  --output "${run_directory}/official-run.json"

(
  cd "${repository_root}"
  uv run python -m tools.compare_scalar_authorities \
    --official-run "${run_directory}/official-run.json" \
    --official-blob "${run_directory}/official-taps.f32le.bin" \
    --quartz-fields "${run_directory}/quartz.fields" \
    --quartz-blob "${run_directory}/quartz.f32le.bin" \
    --llama-fields "${run_directory}/llama.fields" \
    --llama-blob "${run_directory}/llama.f32le.bin" \
    --output "${run_directory}/comparison.json"
  uv run python -m tools.freeze_scalar_tolerances \
    --comparison "${run_directory}/comparison.json" \
    --official-logits "${run_directory}/official-logits.f32le.bin" \
    --quartz-logits "${repository_root}/.cache/authorities/llama-evidence/native.f32le.bin" \
    --llama-logits "${repository_root}/.cache/authorities/llama-evidence/llama.f32le.bin" \
    --fixture "${repository_root}/fixtures/scalar_authority_alignment.json" \
    --tolerances "${repository_root}/pins/scalar_oracle_tolerances.json"
)

printf 'evidence_directory=%s\n' "${run_directory}"
