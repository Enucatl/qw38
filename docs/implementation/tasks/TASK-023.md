# TASK-023 — Production prefill projections and workspace

## Status

DONE

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Deliver conversion-inclusive, bounded prefill projections for the accepted candidate, with reusable activation operands and correct output precision.

## Depends on

- [TASK-022](TASK-022.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — unchanged
  quality criteria and measurement definitions, with task ownership remapped by the ledger.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| Q-01/Q-02, projection part of P-02, A-01/L-01 | Use the accepted candidate representation and precision policy | TASK-020–022 selection |
| M-01, T-02, G-02 | Bounded workspace and a distinct measured prefill schedule | OVERALL-01 |
| P-01, S-01/S-02 | FP32 residual/accumulation and unchanged state controls | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-022 provides full candidate decode and a passing core quality gate. TASK-019 has real-shape GEMM and conversion measurements; TASK-021 defines the consumer layout.

## Scope

Integrate the selected native block-scaled GEMMs, or the measured fallback,
into bounded token-major prefill. Include activation scale/pack generation,
reuse across compatible projections, paired gate/up organization, proper
SwiGLU and residual epilogues, and requested-row vocabulary handling. Preserve
the precision contract of each consumer; BF16-output example kernels alone
do not satisfy FP32 residual/logit obligations.

Select chunk sizes and GEMM tiles from TASK-019 results rather than freezing
the old 256-token/32×64×64 defaults. Validate dynamic tails, padding masks,
workspace bounds, epilogue equivalence and per-layer projection error.
**Exit:** complete projection paths with end-to-end conversion-inclusive
timings, reusable workspace and correct generation/evaluation modes. Repeated
decode and full-model BF16 expansion are not production prefill paths.

## Out of scope

Full-model prefill/handoff, GDN recurrence and attention cores, whole-model BF16 expansion, unmeasured duplicate weights and repeated decode as production prefill.

## Required interfaces and data representation

Typed projection plans describe valid token count, absolute positions, family/shape, scales/layout, dispatch, output precision and epilogue. A preallocated token-major arena accounts for live activation/scale buffers, GEMM workspace and tails. Head interfaces distinguish final-position generation logits from requested evaluation rows in bounded tiles.

## Required semantics and constraints

Include scale reductions and activation packing in execution and reuse compatible quantized inputs. Preserve FP32 accumulation/residual/logit obligations and the declared rounding of ordinary staging. Paired gate/up performs the correct SiLU/product; padded rows must not influence outputs, scales or future state. Candidate BF16-output examples require appropriate epilogues before satisfying runtime contracts.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Choose chunks, tiles, buffering and dispatch from TASK-019 measurements within the frozen memory reserve. The former 256-token/32×64×64 geometry is not a requirement. Freeze comparison settings when measuring a candidate.

## Expected files/modules

Prefill projection CUDA consumers/wrappers, activation quantization/reuse, chunk/workspace plans, MLP prefill integration and projection/epilogue checks.

## Tests required

### Unit and contract checks

Valid token counts and tails around actual tile/chunk sizes, padding/scales, requested-row sets, buffer lifetime/overlap and precision-specific epilogues.

### Reference and numerical checks

Projection and complete MLP outputs against independent reconstructed-operand references and the accepted candidate decode control with declared tolerances; separate activation-policy effects from arithmetic errors.

### Integration checks

Complete GDN/attention projection sets and MLP prefill without their recurrence/attention cores. Exercise generation/evaluation head modes, bounded workspace and stable allocation behavior.

## Benchmark required

Required for each selected family/shape/chunk: complete quantize/pack/GEMM/epilogue time, activation reuse, resources and peak workspace, plus kernel-only attribution. Confirm native Tensor Core execution or explicitly identify the measured fallback.

## Acceptance criteria

- [x] All selected projection families, MLP and head modes consume the candidate representation correctly.
- [x] Activation scaling/packing/reuse and required output precision are integrated and included in timings.
- [x] Tails, masks, epilogues and requested rows satisfy independent numerical checks.
- [x] Workspace is bounded, preallocated and accounts for scale buffers and library requirements.
- [x] Chunk/tile/dispatch choices have conversion-inclusive measurements and fit the frozen memory budget.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

Completed production prefill projections for accepted TASK-022 candidate
`candidate-v2-q4k-rope-fixed.qw38`. Native block-scaled MMA was not selected;
the measured fallback uses bounded unpack-to-BF16 followed by cuBLAS Tensor
Core GEMM. Activation operands remain BF16, with no activation scale or pack
buffers. Accumulation, residuals, and logits remain FP32. Generation reads the
final head row; evaluation supports up to eight arbitrary requested rows.

Accepted artifact manifest digest:
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`.
Its policy is Q4_K MLP weights and Q8 other projections/head. Source base was
`472ccd6c369430531af3f67b13b22551b17c98a5`; the independently reviewed
candidate bundle SHA-256 was
`6ee3301d530507b7dc3c57a495793e4ab213266249044ebcb674de2f037a2c9b`.
The synthetic projection test binary SHA-256 was
`b4e59ca523f07026cbd8fd0167a6725fc81925b422ef81739398ec6092bce6ad`; the
artifact-integration test binary SHA-256 was
`50f3d95d00439fec10c2412d75a33c7dab7ec3547fd8f1bd2f2da1a09367b7e5`; and the
benchmark binary SHA-256 was
`4b70379c14b9ef4b5eea4501a0eb64c394df94248792e73cab7f9573f0584660`.

The pinned image was `qw38-dev:cuda13.4.1-pinned`, digest
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`; it
used CUDA 13.4.59, GCC 14.2, C++23, Release, and `sm_120`. Hardware was an RTX
5090 with 32607 MiB and driver 590.48.01.

### Changes made

Integrated bounded token-major projection plans, workspace, reusable
activation operands, paired MLP gate/up, SwiGLU/residual handling, and head
generation/evaluation modes. The measured keep decision is BF16 activations
with bounded conversion-inclusive fallback GEMMs for Q4_K MLP and Q8 other
projections/head. The projection integration covers GDN/attention projection
sets but does not implement their recurrence or attention cores. No full-model
prefill/handoff is claimed.

### Tests run

In the pinned image and mounted workspace, the acceptance command was:

```sh
cmake --build build/pinned-release --target qw38_prefill_projection_test qw38_prefill_artifact_integration_test qw38_bench_prefill -j4 && QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 ctest --test-dir build/pinned-release --output-on-failure -R ^prefill_ && QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 build/pinned-release/tests/qw38_prefill_artifact_integration_test
```

It passed: CTest reported 2/2 tests, and the direct artifact integration test
passed with 64 bound layers, two mixer sets, three MLP sample rows, and two
head sample rows. The log is `.codex-wake-run/01e09c733d74.log`. MLP decode
maximum absolute error was 0.000106812; independent MLP error was 0.000228882
against `0.05 + 0.001 relative`; all-layer projection-prefix maximum was
0.000139125 against `0.02 + 0.002 relative`; head-prefix maximum was
2.42144e-08 against that same tolerance. Workspace was 28,311,552 bytes.

Coverage included synthetic Q4/Q8/BF16 operands, RNE BF16 stores, FP32
residuals, tails of 3/17/128/1023/1024, padding guards, bad pointers and
overlap, stream move/close, the 64-layer binder, independent first-projection
samples, complete GDN/attention projection sets at layers 0 and 3, CPU and
decode MLP references, ordered eight-row head evaluation, and overlap rejection
without mutation. The separate final test-only regression run passed; log:
`.codex-wake-run/1f7cf8a01680.log`. Exact command:

```sh
cmake --build build/pinned-release --target qw38_prefill_projection_test -j4 && ctest --test-dir build/pinned-release --output-on-failure -R ^prefill_projection$ && sha256sum build/pinned-release/tests/qw38_prefill_projection_test
```

Independent Astra review returned `REVIEW: PASS` on pass 2 with no code,
acceptance, or evidence findings/requests. The reviewed bundle SHA-256 was
`6ee3301d530507b7dc3c57a495793e4ab213266249044ebcb674de2f037a2c9b`;
the main thread confirmed the same bundle and `git diff --check` after review.

### Benchmark results

Benchmark/profiler command passed in the same pinned container; log:
`.codex-wake-run/4ed5fdf58578.log`. Exact inner command:

```sh
bash scripts/task023_benchmark.sh && bash scripts/task023_large_benchmark.sh && bash scripts/task023_profile.sh && sha256sum build/pinned-release/benchmarks/qw38_bench_prefill build/pinned-release/tests/qw38_prefill_projection_test build/pinned-release/tests/qw38_prefill_artifact_integration_test && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

Raw 20-repetition CSV sweeps, temporarily retained under `.cache/task023/`
for review and slated for removal during task delivery, were
`.cache/task023/sweep-{128,256,512}.csv`
(105 family/chunk points each) and `.cache/task023/large-{256,512,1024}.csv`
(15 points each). Their SHA-256 values: sweep-128
`d126e89d5a5f06f55a37c485505ce89fe7bb6a4106256f02c7be5b611323f834`,
sweep-256 `726ebda383f9d485f602fa61f0ad6503b8a850a3e7f472f4be9f997c1d182224`,
sweep-512 `2d85daf36d8514baa6c366f3f11c756123122f6562be414b85515414db4e19b6`,
large-256 `5929ada021d10d46e390ed619a8f6d67263488cf3ed700a88ae6279f2ef44a00`,
large-512 `abf8486475ab4e4db0d526fbc770a17743ea105d7e3ed10b7c44b7e5b7c07087`,
and large-1024
`5fcdf9fcbdfecdb937ba9d0ad1ad631f3202a6e3d837001b706c9d006db8d3bd`.

Timing covers full GPU operation and host dispatch-plus-synchronization
boundaries, after five warmups; source used eight populated decode rows, with
some family layer selections not matching for every family. At tile 512,
M=128 host milliseconds for GDN qkv / attention qg / full MLP / head generation
/ head evaluation (two rows) were 0.3158 / 0.3793 / 1.8422 / 6.7774 / 13.534.
Tile 256 measured 0.4247 / 0.5106 / 2.2853 / 12.9226 / 25.8331; tile 128
measured 0.7048 / 0.8382 / 3.3337 / 16.803 / 33.582. At tile 512, full MLP
times for M=256/512/1024 were 2.2657 / 3.5244 / 5.175 ms; GDN qkv was
0.4219 / 0.6505 / 1.0162 ms and attention qg was 0.5048 / 0.7794 / 1.2565
ms. Maximum workspace was 72,351,744 bytes, including an explicit 4 MiB
cuBLAS workspace and no activation scale buffers. Free GPU memory after setup
was 10,343,153,664 bytes against the frozen 2,147,483,648-byte reserve.

Per-family Nsight Systems evidence, temporarily retained under
`.cache/task023/` for review and slated for removal during task delivery, is in
`.cache/task023/nsys-{128,256,512,1024}-{kernels,apis}.csv` and `.nsys-rep`:
15/15 family ranges per M; captured CUDA API summaries had
no `cudaMalloc` or other allocation APIs. At M=128 full MLP GPU kernel time
was 56.5% unpack, 33.5% SM120 MMA, and 6.8% split-K; at M=1024 it was 19.8%
unpack and 77.7% GEMM. Head generation M=128 was 62.6% unpack, 24.2% GEMM,
8.6% split-K, and 4.6% epilogue. Profiles include five warmups plus one
sample, so kernel attribution is not direct host latency. Selected production
defaults are 1024 maximum chunk capacity and a 512 output-row tile; downstream
GDN/attention scheduling may use smaller chunks. No architecture blocker was
found.

### Architecture blocker

None.

### Follow-up observations

TASK-024 and TASK-025 own the GDN recurrence and causal attention cores.
TASK-026 owns full-model prefill/handoff and complete quality coverage;
TASK-027 owns matched whole-request performance. TASK-023's family benchmark
uses selected synthetic/populated decode rows and does not claim those later
integration gates or matched full-prefill performance.
