#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
authority_root="${repository_root}/.cache/authorities"
checkpoint="${authority_root}/qwen3.8-27b-transformers"
source_checkout="${authority_root}/transformers"
environment="${authority_root}/transformers-venv"
checkpoint_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
source_revision="42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"

mkdir -p "${authority_root}"

if [[ ! -d "${source_checkout}/.git" ]]; then
  git init "${source_checkout}"
  git -C "${source_checkout}" remote add origin https://github.com/huggingface/transformers.git
fi
git -C "${source_checkout}" fetch --depth 1 origin "${source_revision}"
git -C "${source_checkout}" checkout --detach "${source_revision}"
if [[ -n "$(git -C "${source_checkout}" status --porcelain)" ]]; then
  echo "Transformers authority checkout is dirty" >&2
  exit 1
fi

uv venv --python 3.12 "${environment}"
uv pip sync \
  --python "${environment}/bin/python" \
  --require-hashes \
  "${repository_root}/pins/transformers_authority_requirements.lock.txt"

"${environment}/bin/hf" download \
  Qwen/Qwen3.8-27B \
  --revision "${checkpoint_revision}" \
  --local-dir "${checkpoint}" \
  config.json generation_config.json model.safetensors.index.json \
  tokenizer.json tokenizer_config.json chat_template.jinja
"${environment}/bin/hf" download \
  Qwen/Qwen3.8-27B \
  --revision "${checkpoint_revision}" \
  --local-dir "${checkpoint}" \
  --include '*.safetensors'

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
PYTHONPATH="${repository_root}:${source_checkout}/src" \
  "${environment}/bin/python" -m tools.verify_transformers_authority \
  --contract "${repository_root}/pins/transformers_authority_contract.json" \
  --checkpoint "${checkpoint}" \
  --source "${source_checkout}"
