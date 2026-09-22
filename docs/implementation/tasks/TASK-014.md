# TASK-014 — Attention preparation and KV cache

## Status
DONE
## Milestone
M5 — Attention execution
## Purpose
Implement full-attention projection semantics, per-head preparation, and persistent cache append independently of the attention scan.
## Depends on
- TASK-009
## Normative references
- `docs/architecture/architecture-v0.md` — Full-attention sequence; convolution/KV layout
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01, P-01–P-02, G-01, M-01 | Projection, precision, attention semantics and stores | LOCKED |
| T-03 | 128-thread preparation start | TUNING |
## Starting point
RMS, Q4 projections, QK RMS, partial RoPE, and session KV storage exist.
## Scope
Implement attention steps 1–3: input RMS; one Q4 launch for q/g, k, v ranges preserving per-query-head q/g association; preparation applying zero-centered QK norms in FP32 and partial RoPE; write BF16 Q and append one BF16 rotated K and BF16 V to cache; advance populated length transactionally after successful work. Add CPU reference and reset/restore tests.
## Out of scope
Attention score/softmax/value scan, output gating/projection, prefill cache batch append, repeated GQA cache copies.
## Required interfaces
Attention-prep plan binds norm/projections/QK norms/RoPE/cache/scratch; decode call takes absolute position and verifies it equals append/populated contract.
## Required semantics
24 query heads, four KV heads, width 256; q/g split within each head; QK norm before RoPE; only first 64 coordinates rotate; K stored after norm/RoPE, V stored once; six query heads map to each KV head.
## Data representation
Prepared Q BF16 `[24,256]`; g scratch aligned per query head; K/V BF16 `[4,capacity,256]`, coordinate-contiguous; no sixfold copies.
## Implementation constraints
Never expose advanced populated length on failure; capacity and position errors are typed; no full prepared K/V duplicate.
## Tuning defaults
One preparation block/head, 128 threads.
## Expected files/modules
Attention plan/preparation kernel/cache append tests.
## Tests required
### Unit tests
q/g ordering, head mapping, capacity/full cache, position mismatch, reset, unaffected RoPE suffix.
### Reference/numerical tests
Projected/prepared Q/K/V/g versus reference at positions 0,1,large; BF16 store tolerance.
### Integration tests
Multiple appends, snapshot/restore, two sessions, cache byte/index verification.
## Benchmark required
No.
## Acceptance criteria
- [x] Per-head q/g, QK norm, and partial RoPE semantics pass.
- [x] Cache contains one BF16 K/V copy per KV head/token.
- [x] Append length changes only after success.
- [x] Continuation/reset tests pass.
## Architecture blocker rule
On locked conflict stop with complete blocker fields; do not change cache layout or RoPE scope.
## Completion report
### Result
DONE
### Changes made
- AttentionWorkspace aliases (M-01): projected q/g `[24,512]`, k/v `[4,256]`, prepared Q and per-query-head g `[24,256]`; trailing 24768 bytes reserved for TASK-015 segment partials. Total 78016 bytes. K/V append directly to cache; no prepared K/V duplicate.
- One ranged decode MMV launch (`launch_decode_mmv_ranges`) computes q/g, k, and v into distinct slices, sharing K/input/layout (Q4 or BF16-control).
- CUDA `launch_attention_prepare`: 28 blocks × 128 threads (T-03). Fused zero-centered QK RMS in FP32, partial RoPE on the first 64 of 256, one BF16 store for Q and rotated K. g copied per query head; V bit-copied once. Writes K/V at `[layer, component, kv_head, token, 256]`.
- CPU `attn_split_qg`, `attn_qk_norm_rope`, `attn_cache_append`, `attn_prep_reference`.
- Runtime `AttentionPrepPlan` / `bind_attention_prep_plan` (views and Model/Session) plus `execute_decode_attention_prep`. Steps: hidden RMS; one ranged projection; fused prep+append. Position must equal populated and be `< capacity`. Stream-syncs, then sets populated to `position+1` only after success. Typed `InvalidCapacity` / `InvalidPopulatedLength`.
- Tests: `attention_unit` (bind, GQA mapping, per-head q/g, isolated prep, RoPE suffix, one K/V per head, capacity/mismatch, session reset); `attention_reference` (Q4 and BF16-control at positions 0, 1, 123456); `attention_integration` (artifact layer 3, two sessions, three appends, snapshot/restore continuation, `kv_byte_offset`, reset, no malloc on execute).
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 37/37 tests passed (`attention_unit` 1.81s, `attention_reference` 9.92s, `attention_integration` 3.94s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 37/37 tests passed (`attention_unit` 0.60s, `attention_reference` 3.21s, `attention_integration` 1.00s).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- AttentionWorkspace bytes 53248–78016 are reserved for TASK-015 FP32 segment partials (`24 × (2 + 256)`); unused by this task.
- `execute_decode_attention_prep` synchronizes the stream before advancing host populated length so a failed kernel does not commit the append.
- Restored missing `gdn_unit`/`gdn_reference`/`gdn_integration` CMake labels (`cuda;gpu;gdn`) dropped when adding attention tests.
