#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cuda-binary>" >&2
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

sass=$(cuobjdump -sass "$bin")
check_packed_loads() {
  local kind=$1
  local required=$2
  local blocks
  blocks=$(printf '%s\n' "$sass" | awk -v kind="$kind" '
    BEGIN { RS = "\t\tFunction : " }
    /decode_mmv_(ranges_)?kernel/ && index($0, "WeightKindE" kind "E") {
      print
    }
  ')
  local count
  count=$(printf '%s\n' "$blocks" |
    grep -c 'decode_mmv_\(ranges_\)\?kernel' || true)
  if [[ $count -ne 3 ]]; then
    echo "expected three generated decode kernels for weight kind $kind, found $count" >&2
    exit 1
  fi
  if printf '%s\n' "$blocks" | grep -q 'LDG\.E\.U8'; then
    echo "decode weight kind $kind regressed to byte loads" >&2
    exit 1
  fi
  local load_count
  load_count=$(printf '%s\n' "$blocks" | grep -c "$required" || true)
  if [[ $load_count -lt $count ]]; then
    echo "decode weight kind $kind is missing required packed-word loads" >&2
    exit 1
  fi
}

# WeightKind E0 is Q4 (one 32-bit word/lane); E1 is Q8 (one 64-bit word/lane).
check_packed_loads 0 'LDG\.E '
check_packed_loads 1 'LDG\.E\.64 '
echo "Q4/Q8 packed-word SASS loads confirmed"
