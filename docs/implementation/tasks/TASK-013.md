# TASK-013 — Complete decode GDN mixer

## Status
DONE
## Milestone
M4 — GDN execution
## Purpose
Integrate all eight selected GDN regions into a continuation-correct mixer residual transition.
## Depends on
- TASK-010
- TASK-012
## Normative references
- `docs/architecture/architecture-v0.md` — GDN initial kernel sequence; precision/materialization
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01, Q-01, Q-03, P-01–P-02, S-01–S-02, M-01 | Complete GDN semantics, precision, state, and cuts | LOCKED |
| T-01, T-03 | Initial kernel geometries | TUNING |
## Starting point
GDN steps 1–6, packed contractions, activation kernels, session state, and MLP exist.
## Scope
Implement gated output transform (per-value-head RMS with multiplicative gamma and FP32 SiLU(z), BF16 u), Q4 output projection with direct FP32 residual add, and a concrete eight-region GDN plan/executor. Add full CPU/BF16-control comparison and repeated-token continuation. Preserve each V0 diagnostic boundary.
## Out of scope
Prefill, region fusion, alternate layout/state precision, whole layer scheduler, attention.
## Required interfaces
GDN plan binds all named weights/parameters/state/scratch; decode mixer accepts FP32 residual and position/session and returns the other FP32 residual buffer without allocation.
## Required semantics
Steps exactly: input RMS; qkv/z; a/b; conv+SiLU; prep; recurrence; gated RMS/SiLU(z); out projection+residual. GDN gamma is multiplicative, not `1+gamma`. Input residual remains live until step 8.
## Data representation
Use normative shapes/dtypes and TASK-007 arena aliases; o FP32 `[48,128]`, u BF16 `[48,128]`.
## Implementation constraints
No fusion beyond grouped projections already selected; eager one-stream ordering; state commits once/token.
## Tuning defaults
Inherited TASK-009/011/012 geometries.
## Expected files/modules
Runtime GDN plan/executor, output transform kernel, complete GDN tests.
## Tests required
### Unit tests
Bindings/lifetimes, gamma role, z gate, residual preservation, state isolation.
### Reference/numerical tests
Every materialized boundary plus final residual and state against reference for one/many tokens.
### Integration tests
Complete GDN mixer followed by TASK-010 MLP for two sessions and restored continuation.
## Benchmark required
Report region timings diagnostically; no fusion authorization.
## Acceptance criteria
- [x] Eight regions and declared stores remain separately attributable.
- [x] Final residual/state pass reference and continuation tests.
- [x] Gated norm/gate precision semantics are correct.
- [x] Steady state performs no allocations.
## Architecture blocker rule
On locked conflict stop with full required blocker report; do not fuse or alter boundaries as a workaround.
## Completion report
### Result
DONE
### Changes made
- CPU `qw38::reference::gdn_mixer_reference`: front + recurrence + per-head multiplicative gated RMS (`y=(γ⊙o/RMS(o))⊙SiLU(z)`) + BF16 `u` + GEMV `out_proj` + FP32 residual add. Input residual is not mutated; S/history/cursor are in/out.
- CUDA `launch_gdn_output_transform` wraps TASK-008 `launch_gdn_gated_rms` with 48 value heads. Workspace overlay `u` BF16 `[48,128]` at `kGdnOffU`.
- Runtime `GdnPlan` / `bind_gdn_plan` (views and Model/Session) plus `execute_decode_gdn` / `_timed`. Eight named regions: rms, qkvz (two MMVs), ab, conv, prep, recur, gated, out-residual. Region 8 copies `residual_h` → `residual_h_mid` then Q4/BF16-control `ResidualAddFp32`. Returns `h_mid`. No fusion, one stream, no allocation on execute.
- Tests: `gdn_unit` (bind/lifetimes, γ=0 Mix=0, SiLU(z)=0 Mix=0, residual live, S isolation, no malloc); `gdn_reference` (Q4 3 tokens + BF16-control 2 tokens, every materialized boundary); `gdn_integration` (mixer then TASK-010 MLP, two sessions, snapshot/restore at step 3).
- Diagnostic bench `qw38_bench_gdn_mixer` (`EXCLUDE_FROM_ALL`, not ctest).
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 34/34 tests passed (`gdn_unit` 2.70s, `gdn_reference` 7.83s, `gdn_integration` 18.68s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 34/34 tests passed (`gdn_unit` 1.09s, `gdn_reference` 1.87s, `gdn_integration` 4.13s).

### Benchmark results
Historical diagnostic run: source revision `7661c46`; Release target
`qw38_bench_gdn_mixer`, mutable container tag `qw38-dev:cuda13.4.1`, device
NVIDIA GeForce RTX 5090 `sm_120`. The executable SHA-256/build ID and immutable
container image digest were not preserved. Consequently, the timing below is
retained only as a historical log and is **withdrawn as reproducible measurement
evidence**. Future reported measurements must use the identity-bound wrapper in
`docs/implementation/dev-environment.md`. Q4 decode mixer, eight regions, 8
timed mixer iterations after 2 warmups; the source report recorded 9 kernel
launches and 1 device copy. Those counts are superseded by the current
eight-launch, zero-copy path and are withdrawn with the timing. Historical command:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake --build build/release --target qw38_bench_gdn_mixer && ./build/release/benchmarks/qw38_bench_gdn_mixer'
```

| Case | weight_bytes | iterations | kernel launches | device copies | ms |
|---| ---:| ---:| ---:| ---:| ---:|
| decode GDN mixer (8 regions) | 62351808 | 8 | 72 | 8 | 0.128784 |

Region mean ms: rms=0.009988, qkvz=0.081184, ab=0.01182, conv=0.002232, prep=0.004096, recur=0.004068, gated=0.004124, out-residual=0.032768.

Diagnostic only; does not authorize fusion, layout, or precision change.
### Architecture blocker
None.
### Follow-up observations
- Historical: Region 2 stays two MMV launches (qkv then z) in one named region; no fusion.
- Superseded: Region 8 was `copy_d2d` then in-place `ResidualAddFp32`. The current caller writes projection plus original residual directly to the distinct output buffer.
- Superseded: the original TASK-010 in-place `h_mid` description does not define the current ping-pong output contract.
- Delivered with commit and push after verification PASS.
