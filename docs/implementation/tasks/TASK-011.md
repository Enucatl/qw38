# TASK-011 — GDN convolution and preparation

## Status
DONE
## Milestone
M4 — GDN execution
## Purpose
Implement and isolate the stateful GDN front half before recurrence.
## Depends on
- TASK-009
## Normative references
- `docs/architecture/architecture-v0.md` — GDN sequence; convolution/state layout
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-03 | Convolution, a/b, A_log, dt_bias stay BF16 | LOCKED |
| P-01–P-02 | FP32 convolution/gates/qk prep and BF16 stored qkv/v/z | LOCKED |
| M-01 | Selected GDN scratch boundaries | LOCKED |
| T-03 | Initial preparation block geometry | TUNING |
## Starting point
Large Q4 and grouped small BF16 projections plus RMS are available.
## Scope
Implement independent reference math and decode CUDA for four-tap causal convolution + SiLU, circular raw-qkv history update, q/k head RMS normalization, mapping `value_head/3` to 16 key heads, and FP32 alpha/beta from BF16 a/b, A_log, dt_bias according to normative equations. Integrate GDN steps 1–5 while preserving stage outputs for diagnosis.
## Out of scope
Recurrence, output transform/projection, prefill parallel FIR, fusion across listed regions, no-RoPE substitution questions.
## Required interfaces
Concrete GDN-front plan binds qkv/z Q4, a/b BF16, tap-major convolution, time parameters, history/cursor, and typed scratch.
## Required semantics
History contains last three raw pre-convolution qkv values oldest→newest; current raw qkv enters history, never SiLU output. Four taps accumulate FP32 then SiLU FP32/store BF16. GDN has no RoPE. q/k and alpha/beta staging are FP32; v aliases convolved BF16 slice.
## Data representation
qkv BF16 `[10240]`; z BF16 `[48,128]`; history BF16 `[3,10240]`; taps BF16 `[4,10240]`; normalized q/k FP32 `[16,128]`; gates FP32 `[48]`.
## Implementation constraints
Convolution/update and preparation remain separate launches; no physical triplication of q/k.
## Tuning defaults
Channel-parallel convolution; one block/head preparation.
## Expected files/modules
Reference GDN front math, CUDA convolution/preparation kernels, runtime plan/tests.
## Tests required
### Unit tests
History wrap/reset, tap order, zero/nonzero a/b/time values, `value_head/3` boundaries, q/k zero norms.
### Reference/numerical tests
One and many consecutive steps versus reference; stage outputs compared separately.
### Integration tests
RMS→large/small projections→conv/history→preparation on deterministic fixture; snapshot/restore history continuation.
## Benchmark required
No.
## Acceptance criteria
- [x] Convolution tap/history semantics pass multi-step tests.
- [x] Gate equations and q/k normalization match reference in FP32 tolerance.
- [x] Required scratch/precision boundaries are visible.
- [x] No q/k persistent duplication or fusion drift.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not alter convolution or preparation semantics.
## Completion report
### Result
DONE
### Changes made
- Named GdnWorkspace overlays in `src/runtime/sizes.hpp`: qkv/z/convolved BF16, q/k FP32 `[16,128]`, a/b/alpha/beta FP32 `[48]`, reserved o/u for TASK-012/013; total 107264 bytes. `v` aliases `convolved[4096:]`. `kGdnRepeat=3` so `value_head/3` indexes 16 key heads with no physical triplication.
- CPU reference in `src/reference/math.{hpp,cpp}`: four-tap causal FIR + SiLU with circular raw-qkv history (Eq. 14), per-head L2 q/k (Eq. 16, ε inside the sqrt), α/β from FP32 a/b and BF16 `A_log`/`dt_bias` (Eq. 15), composed `gdn_front_reference` for steps 1–5.
- CUDA `cuda/gdn.{hpp,cu}`: channel-parallel FIR+SiLU (256 threads) overwrites the oldest history slot with current raw qkv; T-03 prepare is one 128-thread block per key head for L2 q/k plus three gates. Separate launches; no fusion.
- Runtime `src/runtime/gdn.{hpp,cpp}`: `GdnFrontPlan` binds Q4 or BF16-control qkv/z, grouped BF16 a/b → FP32, tap-major conv `[4,10240]`, history/cursor, and typed scratch. `execute_gdn_front` runs RMS, qkv MMV, z MMV, grouped a/b, conv+history, prepare. Host cursor advances after a successful conv. `Session::conv_cursor_slot` exposes the per-layer host cursor.
- Tests: `gdn_unit` (bind/shape/layout, tap order, wrap, zero-norm q/k, α=0.5/β=0.5, `value_head/3`, scratch aliases, missing identities), `gdn_reference` (7 consecutive kernel steps and full Q4 front vs CPU stages), `gdn_integration` (`.qw38` fixture, 5-step continuation, snapshot/restore at step 3). Host `reference_math_unit` covers FIR/history and gates. CMake/ctest Debug+Release GPU.
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 34/34 tests passed (`reference_math_unit` 0.01s, `gdn_unit` 0.20s, `gdn_reference` 1.56s, `gdn_integration` 4.32s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 34/34 tests passed (`gdn_unit` 0.20s, `gdn_reference` 0.55s, `gdn_integration` 1.10s).

### Verification
VERIFICATION: PASS
Commands run: Debug and Release Docker ctest (34/34 each).
Results: All acceptance criteria met; no unmet criteria.
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- Architecture V0 step 2 asks for one launch with separate output-tile ranges for Q4 qkv and z. TASK-009 `decode_mmv` has no ranged-N kernel (N is 10240 vs 6144). This task uses two sequential `launch_decode_mmv` calls so each contraction keeps its own reduction. Grouped a/b remains one BF16 launch.
- Task wording says “q/k head RMS”; locked Eq. 16 is L2 (`x/sqrt(||x||²+ε)`, ε=1e-6). Implementation follows Eq. 16.
- Declared tolerances: convolved BF16 8e-3; q/k FP32 2e-5 abs / 1e-5 rel; gates FP32 2e-5 abs / 1e-5 rel.

