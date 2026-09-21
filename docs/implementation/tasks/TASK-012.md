# TASK-012 — GDN recurrence and continuation

## Status
DONE
## Milestone
M4 — GDN execution
## Purpose
Validate the architecture-critical recurrent update independently of the complete layer.
## Depends on
- TASK-011
## Normative references
- `docs/architecture/architecture-v0.md` — Chosen GDN ownership and recurrence equation
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| S-01 | Persistent GDN S is FP32 | LOCKED |
| S-02 | Physical `[value_head,value,key]`, warp/value ownership | LOCKED |
| P-01 | Prediction/update/readout and reductions FP32 | LOCKED |
## Starting point
Prepared q/k/alpha/beta/v arrays and session S storage exist.
## Scope
Implement a simple CPU reference and decode CUDA recurrence: one warp owns one 128-key row for one value coordinate; four warps/block own adjacent values; load old S once, compute decay/prediction/error/update/readout, store new S once, emit FP32 o. Add reset, multi-token continuation, and exact snapshot/restore tests.
## Out of scope
Alternate `[head,key,value]`, BF16 state, prefill 64-token loop, gated RMS, projection, fusion, tuning experiments.
## Required interfaces
Typed recurrence launcher over prepared arrays, FP32 state and output, layer/state offsets, ordered stream; no allocation.
## Required semantics
For every row: `D_k=alpha*S_old_k`; `p=sum D_k*khat_k`; `e=beta*(v_j-p)`; `S_new_k=D_k+khat_k*e`; `o_j=sum S_new_k*qhat_k/sqrt(128)`. All operations/reductions FP32 and update precedes readout.
## Data representation
Per layer S FP32 `[48,128,128]` in `[value_head,value,key]`; lane `l` owns keys `l,l+32,l+64,l+96`; o FP32 `[48,128]`.
## Implementation constraints
No atomics/cross-block reductions; no state transpose; q/k shared within block and indexed rather than duplicated.
## Tuning defaults
128-thread blocks/four value rows; ownership axis is locked, block size may later tune consistently.
## Expected files/modules
Reference recurrence, CUDA kernel/wrapper, continuation fixtures.
## Tests required
### Unit tests
Physical indexing, zero state, reset, invalid views, independent layers/heads.
### Reference/numerical tests
One step and 1/2/17/128 consecutive steps against CPU reference; adversarial alpha/beta; stage-level S and o tolerances.
### Integration tests
Run N steps, snapshot, restore in new session, continue, and compare with uninterrupted reference/execute path.
## Benchmark required
Diagnostic one-step timing/resource report only.
## Acceptance criteria
- [x] One-step and multi-step S/o match reference tolerance.
- [x] Reset and restored continuation are correct.
- [x] Compiled resource report and diagnostic timing are recorded.
- [x] Physical layout and FP32 persistence exactly match S-01/S-02.
## Architecture blocker rule
If S-01/S-02 prevents correctness, stop with the complete required blocker report; never transpose/narrow silently.
## Completion report
### Result
DONE
### Changes made
- CPU reference `qw38::reference::gdn_recurrence_step`: in-place FP32 `[value_head,value,key]` update, BF16 v, 16-head q/k indexed by `value_head/3`. Order is D=αS, p=Σ D k̂, e=β(v−p), S=D+k̂e, then o=Σ S q̂/√128 (update before readout).
- CUDA decode kernel in `cuda/gdn.{hpp,cu}`: 1536 blocks/layer, 128 threads; one warp owns one value row; lane `l` holds keys `l,l+32,l+64,l+96`; q/k staged once in 1 KiB shared memory. `launch_gdn_recurrence` takes prepared arrays, FP32 S with `s_layer` offset, FP32 o, and an ordered stream. No allocation, atomics, or state transpose.
- Runtime `GdnRecurrencePlan` binds prepared workspace views (including reserved FP32 o at `kGdnOffO`) onto session S. `execute_gdn_recurrence` is a single launch. `bind_gdn_workspace` now overlays o.
- Tests: `reference_math_unit` (α=0/β=0, shape reject); `gdn_unit` (bind/invalid views, HVK vs transposed slot, zero state, independent layers, session reset on the runtime stream); `gdn_reference` (front→recurrence, 1/2/17/128 steps, adversarial α/β); `gdn_integration` (5-step front+recurrence, snapshot at step 3, restore in a new session, continue).
- Diagnostic bench `qw38_bench_gdn_recurrence` (`EXCLUDE_FROM_ALL`, not ctest).
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 34/34 tests passed (`reference_math_unit` 0.02s, `gdn_unit` 0.24s, `gdn_reference` 2.13s, `gdn_integration` 4.62s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 34/34 tests passed (`gdn_unit` 0.23s, `gdn_reference` 0.66s, `gdn_integration` 1.12s).

### Benchmark results
Identity: `qw38_bench_gdn_recurrence` Release, container `qw38-dev:cuda13.4.1`, device NVIDIA GeForce RTX 5090 `sm_120`. Geometry: four warps/block, 128 threads, 1536 blocks/layer, warp-per-value `[value_head,value,key]`. Command:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake --build build/release --target qw38_bench_gdn_recurrence && ./build/release/benchmarks/qw38_bench_gdn_recurrence'
```

| Case | registers | shared_bytes | local_bytes | occupancy_blocks/SM | s_bytes | launches | ms |
|---| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| one-step decode recurrence | 31 | 1024 | 0 | 12 | 3145728 | 8 | 0.003176 |

Diagnostic only; does not authorize layout, precision, or block-size change.
### Architecture blocker
None.
### Verification
VERIFICATION: PASS

Commands run:
- Debug: `cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure`
- Release: `cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure`

Results: 34/34 tests passed in Debug and Release (`reference_math_unit`, `gdn_unit`, `gdn_reference`, `gdn_integration`). Benchmark `qw38_bench_gdn_recurrence` recorded one-step diagnostic timing and resource report.

Unmet acceptance criteria: none.

### Follow-up observations
- Declared tolerances: one-step S/o 2e-5 abs / 1e-5 rel; multi-step (17/128) 5e-4 abs / 1e-4 rel (warp-tree vs sequential 128-key reduction feeding S).
- Session reset and recurrence launches must share the session's ordered stream; a second test stream can race the persistent S zero.
- 128-step fixtures use L2-normalized q/k, matching prepared decode arrays. Unnormalized keys overflow in FP32 within a long continuation; that is a fixture concern, not an S-01/S-02 issue.


