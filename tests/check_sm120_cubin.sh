#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cuda-smoke-binary>" >&2
  exit 1
fi

bin=$1
if [[ ! -f "$bin" ]]; then
  echo "binary not found: $bin" >&2
  exit 1
fi

if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "cuobjdump not found on PATH" >&2
  exit 1
fi

out=$(cuobjdump -lelf "$bin")
printf '%s\n' "$out"

mapfile -t archs < <(printf '%s\n' "$out" | grep -Eo 'sm_[0-9]+' | sort -u)
if [[ ${#archs[@]} -eq 0 ]]; then
  echo "no sm_* cubin entries found in $bin" >&2
  exit 1
fi
if [[ ${#archs[@]} -ne 1 || ${archs[0]} != sm_120 ]]; then
  echo "expected only sm_120 cubin, found: ${archs[*]}" >&2
  exit 1
fi

echo "native sm_120 cubin confirmed"
