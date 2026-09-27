# TASK-045 — Efficient FP8 and Q4 decode projections

## Status

DONE

## User-approved development acceptance — 2026-09-27

After reviewing the preserved results and their limitations, the repository
owner instructed: "accept it then, finish this task". This explicitly accepts
R045-1 (complete MLP +0.539%, +3.466 microseconds) and R045-2 (4096-token
prefill +0.480%, +7.450101 ms) for this candidate and TASK-045 only. The
eight-step decode observation improves 12.675% (-19.142426 ms), and the
complete 4096+8 inference request improves 0.687% (-11.692294 ms).

This amendment permits retaining both changed kernel families despite those
two failed performance gates. It does not relabel either measurement as a
pass, establish their causes, prove measurement noise, or claim statistically
repeatable performance. The integrated improvement does not independently
attribute a complete-request benefit to Q4. Candidate identity is fixed by
`candidate-sources.sha256` under the evidence directory below; its source,
test and benchmark hashes were rechecked successfully on continuation.

Resume the preserved uncommitted work in place under this explicit approval.
All other numerical, state, evidence and independent-review requirements
remain applicable. TASK-043 remains production; TASK-048's quality, capacity
and nine-row performance promotion gates are unchanged. TASK-046–048 remain
TODO. Independent Astra verification passed; documentation and cache cleanup
are complete. Luna (gpt-6-luna, medium) prepared final documentation and
delivery; commit and push outcome will be reported by the delivery run.

## Milestone and dependency

M24 — Reduce the dominant recurring decode costs. Depends on TASK-044.
[FAST-04](../task_ledger.md#optimization-amendment--fast-04-2026-09-27)
governs this batch. Start from the TASK-043 approved policy-1030 runtime plus
TASK-044's reporting changes. Development completion does not promote a new
production candidate.

## Evidence and scope

TASK-043's 128-token decode at context 4096 takes 2520.961 ms: Q4 MMVQ
914.006 ms and FP8 GEMV 905.600 ms, together 72.2% of elapsed time.
The relevant source is `cuda/fp8.cu` (`gemv_kernel`) and `cuda/q4k_q8.cu`
(`mmvq`); callers include `cuda/decode_mmv.cu`, the Q8 MLP path and runtime
GDN/attention projection plans. Optimize these existing M=1 consumers.
Keep the artifact, quantizers, prefill consumers, head and state ABI unchanged.

## Implementation guide

1. Reuse TASK-043/044 timelines and record the actual matrix shapes, call
   counts, launch resources and complete operation costs. If those do not
   discriminate the proposed load/instruction change, obtain a targeted
   Nsight Compute diagnostic for one representative shape per changed kernel
   family. Record memory transactions, instruction/stall evidence and resource
   limits. Counter collection is diagnostic, never acceptance timing; document
   profiler replay passes and do not run a full-model counter capture.
2. FP8 first candidate: vectorize contiguous weight/input loads and conversion,
   explicitly reuse each K128 weight scale, and retain BF16 input with FP32
   accumulation. Start with the existing row-major bytes and output epilogues.
   Change row/warp ownership only where the diagnostic supports it. Reusing
   a loaded scale is different from moving multiplication outside a sum;
   start by preserving the current scaled-weight/FMA order.
3. Q4 first candidate: replace the eight small loads per lane/K32 group with
   aligned wider loads and register unpacking. Inspect the strided lane access
   pattern before changing ownership. Keep exact per-K32 integer dots and
   the FP32 affine correction. A new lane assignment may change FP32 reduction
   order and must be checked accordingly. Do not copy upstream half metadata
   or change the activation recipe to simplify the kernel.
4. Select fixed launch choices for the real shapes; retain existing fallbacks
   only where a changed shape fails to improve. No generic autotuner, tile
   sweep, extra weight view, new dependency or compiler/layout conversion.
   Adapt useful patterns from pinned llama.cpp
   `e6ab7c1a41054a888ada952eab4c886444c2f5ad`, with existing attribution.
   A failed measured hypothesis permits a specific diagnosed repair, not a
   search through unrelated schedules.

Gate/up activation packing is already shared, SwiGLU supplies a fresh down
pack, and down residual addition is already fused. Preserve those lifetimes.
Do not bundle speculative gate/up fusion or graph preparation into this task.

## Numerical and integration contract

Retain policy 1030: FP8 stored weights/scales and BF16 decode input;
Q4_K weights with signed Q8 K32 activation codes, FP32 scales and exact sums.
The MLP contribution remains
`a * (d*s*sum(q*z) - dmin*m*sum(z))`, with dot bound 60,960 and sum bound
4,064. No per-weight BF16 rounding in this MLP path; no FP8 activation
quantization in decode. Keep FP32 accumulation, residuals and nonlinearities,
existing BF16 stores, nonfinite failure behavior and scratch ownership.
Scheduling/reduction changes within this contract are tuning; a different
operand/rounding policy is outside scope.

Exercise standalone and production ranged FP8 callers, all three MLP
contractions, residual/in-place output, real session stream, graph capture
and replay. No per-token allocation, host-state commit before completion,
or weakened validation/poison/reset/restore behavior.

## Verification and completion

- Build affected Release targets. Extend existing `fp8_test.cpp`,
  `q4k_q8_test.cpp`, decode/MLP/binding tests only where needed for wider loads,
  tails, changed ownership and production selection. Retain independent
  references, existing tolerances, asymmetric scales/minima, zero/subnormal
  groups, nonfinite inputs and guarded extents. Run affected full-model
  integration cases for graph/eager equality and failure recovery.
- Once per parent/candidate, measure complete M=1 MLP using
  `benchmarks/mlp_bench.cpp`, and complete real-shape M=1 FP8 projection groups
  using `benchmarks/fp8_group_bench.cpp` with minimal extensions if necessary.
  Cover each distinct production shape/epilogue touched, with identical
  nonzero operands. Include preparation, all sibling consumers, epilogues,
  readout/completion and workspace. Report per-shape costs and aggregate
  family cost weighted by production call counts; isolated contraction time
  alone is insufficient. Enforce one execution and zero warmups rather than
  inheriting benchmark repetition defaults.
- Run the matched 4096+8 development check defined in FAST-04 once per arm.
  Decode must improve and prefill must not regress. Report first decode and
  remaining seven separately; a first-use anomaly remains visible.
- Run the frozen TASK-035 eight-window NLL development screen if any operand
  value or FP32 reduction order changes; aggregate delta <= +0.03 against
  its frozen reference. Otherwise record exact unchanged outputs from the
  affected checks as the basis for reusing that evidence. No core-54 rerun.

Except for the two candidate-specific performance exceptions approved above,
DONE requires passing affected correctness/state checks, a retained changed
consumer in each family, lower complete FP8 family and MLP costs, and the
integrated result above. Any retained shape must not regress in its matched
complete-operation observation; rejected alternatives keep their evidence.
Failure to qualify a required family is BLOCKED pending a concrete repair or
explicit replan, not completion via the unchanged fallback. Record exact
commands, identities, resource reports and capacity impact for TASK-048.

## Historical blocked implementation report — 2026-09-27

This section preserves the original blocked decision and evidence. The
user-approved development acceptance above supersedes its stop instruction
and pending-amendment status; the measurements and unresolved causes remain.
Review and completion are recorded below.

Main-thread Codex (GPT-6) implemented the first measured candidate after the
user authorized committing the FAST-04 plan. The planning commit is
`4ef8d43`; the candidate remains uncommitted and is not promoted. No downstream
task was activated. At the time of this report, independent Astra review had
not started (zero passes), and delivery had not run because mandatory timing
gates failed.

### Blocking results

- **R045-1:** complete first-use M=1 MLP did not improve: host
  **0.643063 → 0.646529 ms (+0.539%)**; GPU event interval including readout
  **0.641535997391 → 0.644608020782 ms**. Identical output bytes and faster
  isolated contractions do not satisfy the complete-operation criterion.
- **R045-2:** matched 4096+8 prefill regressed:
  **1551.589881 → 1559.039982 ms (+0.480%)**. Eight-step decode improved
  **151.028688 → 131.886262 ms (-12.675%)**, but does not waive prefill.
  Neither regression has an established cause. These are single-run
  observations; no statistical claim, acceptance retry, or tolerance waiver
  was substituted for the specified gates.

The follow-up diagnostic supports the wider-load hypothesis at kernel scope,
but does not justify another specific repair or establish why the unprofiled
complete-operation result regressed. Stop pending a concrete repair or an
explicit acceptance amendment/replan. Preserve the candidate and evidence.

### Candidate and scope

`cuda/fp8.cu` loads four contiguous FP8 weights and BF16 inputs per lane,
uses packed FP8 conversion, and shares each K128 scale across four scaled
weight/FMAs. FP32 lane accumulation order changes; scale multiplication stays
before each FMA. Two-byte-aligned BF16 inputs retain a scalar-load path.
`cuda/q4k_q8.cu` loads Q4 codes and Q8 inputs in 16-byte vectors when aligned,
with a four-byte-aligned fallback. K32 integer dots, affine arithmetic and
FP32 reduction order remain unchanged. No quantizer, persistent state, prefill
consumer, runtime caller or allocation policy was changed.

The existing FP8 group benchmark now additionally accepts
`ARTIFACT gdn-in|gdn-out|attention-in|attention-out OUTPUT_BIN`, consuming
actual layer-0/layer-3 artifact weights through production launchers. MLP's
artifact mode optionally saves output. Both complete-operation modes include
readout/completion inside timing and use one execution, zero warmups.
Setup/upload remain separate. Original benchmark modes remain available.

All changed FP8 production shapes are covered by these groups:

| Group | N×K and epilogue | Groups per token | Parent host ms | Candidate host ms |
| --- | --- | ---: | ---: | ---: |
| GDN input | 10240×5120 and 6144×5120 BF16; includes RMS and two 48×5120 BF16-weight/FP32-output siblings | 48 | 1.226781 | 1.180134 |
| GDN output | 5120×6144, residual FP32 | 48 | 0.428164 | 0.364734 |
| Attention input | 12288×5120 and two 1024×5120, BF16; includes RMS | 16 | 0.847720 | 0.780744 |
| Attention output | 5120×6144, residual FP32 | 16 | 0.357900 | 0.347851 |

Count-weighted complete FP8 group cost is **98.727280 → 92.211184 ms
(-6.600%)**. This weights independent first-use observations, and is **not**
an estimate of one recurring full-model token. No finer per-contraction FP8
timing is claimed by these group observations. Q4 MLP includes gate/up
17408×5120 and down 5120×17408, each occurring 64 times per model token.

Matched request first decode is **21.770530 → 19.372122 ms**, remaining seven
**129.257286 → 112.513258 ms**, and complete inference request
**1702.618609 → 1690.926315 ms**. Cold setup is
**5494.644668 → 5428.382872 ms**, plus CUDA initialization
**117.385967 → 118.497860 ms**. All nine observed output IDs agree; final
position is 4104 with finite logits. FP8 outputs change as expected from
reduction order; final-logit maximum absolute difference is 0.118024259806.
This diagnostic difference is not a replacement numerical acceptance metric.

### Correctness, quality and capacity

- Release `q4k_q8`, `q8_mlp_integration`, and `language_model_integration`
  pass on the final kernel candidate. The last covers graph/eager bitwise
  equality, reset/restore, interleave, movement, early/late failure and
  uncommitted poisoned state. It also completes first decode and maximum
  graph upload at capacity 32896, with **8,331,984,896 free bytes** sampled.
- Final Release `fp8` passes after repairing two new-fixture mistakes:
  production K must be divisible by 256, and its FP32 output case must
  explicitly select the FP32 epilogue. No kernel repair was needed.
  Final tests cover an independent reference for every output row, minimum
  input alignment, guarded FP32/BF16/in-place residual epilogues and nonfinite
  decode. Q4 tests cover independent affine/tail cases plus separately shifted
  code/input pointers and guarded residual outputs. Complete MLP outputs are
  byte-identical to the parent.
- Frozen TASK-035 eight-window screen passes: 1024 target tokens,
  candidate NLL **1.7742927192144804**, reference **1.773838532533603**,
  delta **+0.00045418668087737757**, below +0.03. The existing scorer
  authenticates all eight prompt/target identities and masks. No core-54 run.
- FP8 remains at 40 registers/thread; Q4 changes 38 → 39 for aligned loads
  and retains 38 for the fallback. Both report zero stack/local bytes and no
  added application shared memory (the resource dump includes 1024 driver
  bytes). Model bytes **21,013,686,400**, maximum-chunk arena bytes
  **513,802,240**, Q8 workspace **82,444,292**, and steady-state allocation
  count are unchanged; complete component measurements report zero hot
  allocations. The free-memory sample is not a continuous peak claim.

### Exact commands and preserved evidence

All relative evidence paths below are under
`.cache/evaluation/qw38-language-v2/task045-support/`. Raw commands are in the
scripts, including the immutable Docker image and all CLI arguments.

1. `bash .cache/evaluation/qw38-language-v2/task045-support/parent.sh` — exit 0;
   Release parent drivers, four FP8 groups, complete M1 MLP, authenticated
   TASK-027 4096+128 token file with first eight fixed continuation inputs,
   and targeted parent FP8/Q4 diagnostics. Each NCU capture used 14 replay
   passes on one kernel; counter timing is diagnostic only.
2. Original `candidate.sh` (preserved as `candidate-attempt1.sh`) — exit 8;
   affected Release build succeeded; three tests passed, new FP8 fixture
   failed admission. `candidate-tests-initial.log` and
   `candidate-build-initial.log` preserve these results. A second fixture
   attempt failed output-type admission; see `candidate-tests-fixture2.log`.
3. Final `bash .cache/evaluation/qw38-language-v2/task045-support/candidate.sh`
   — exit 0; rebuild/rerun only `qw38_fp8_test` / `ctest --test-dir
   build/pinned-release --output-on-failure -V -R "^fp8$"`, then the four
   complete groups, M1 MLP, matched request, eight-window evaluation and
   `uv run --script .../score.py`. The three unchanged passing tests are
   reused, not rerun. The initial test expression was
   `"^(fp8|q4k_q8|q8_mlp_integration|language_model_integration)$"`, with
   `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38`.
4. `bash .cache/evaluation/qw38-language-v2/task045-support/diagnose-mlp.sh`
   — exit 0; separate Nsight Systems diagnostic per parent/candidate and one
   14-pass candidate Q4 NCU capture. Profiled Q4 gate/up/down times are
   **39.296/39.200/36.768 → 33.120/37.312/36.032 microseconds**. Total kernel
   time is **132.928 → 124.096 microseconds**, versus approximately 0.82 ms
   profiled complete operations. Library loading and launch overhead remain
   inside the operation. This does not explain the original regression or
   supersede unprofiled acceptance. The filtered SASS extraction yielded no
   kernel body; no generated-instruction claim is based on those files.
5. `bash .cache/evaluation/qw38-language-v2/task045-support/analyze.sh` —
   exit 0; writes `comparison.json`, status **BLOCKED**, with all comparisons,
   output differences, diagnostic kernel rows and the quality report.
   `sha256sum --check .../candidate-sources.sha256` and `git diff --check`
   pass. Candidate source/test/benchmark identities, parent binary identities,
   resources, raw counter exports and complete logs are preserved alongside.

The environment is RTX 5090, driver 590.48.01, 400 W configured power limit,
CUDA 13.4.1 / nvcc V13.4.59, image
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`.
`toolchain.txt` records compiler/CMake versions. The unchanged policy-1030
artifact is `.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38`, manifest
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
No artifact or checkpoint payload digest was computed. Frozen token-file
SHA-256 is `fa28a473299ca58495ee860d08f8c532987f3df7de95563d0f575275817ba923`.
At the historical blocked point, executable copies under `.cache/task045/`
and support evidence were retained. TASK-046–048 remained TODO.

## Completion Report — 2026-09-27

TASK-045 is complete for development under the owner-approved, candidate-only
exceptions above. The main-thread implementation used Codex/GPT-6. Independent
formal review by Astra (gpt-6-astra, high) passed on the first pass with no
findings, acceptance gaps or evidence requests. The complete review is
preserved at
`.cache/evaluation/qw38-language-v2/task045-support/astra-review-pass1.md`
(SHA-256 `d08d5781fb0a032a1d57904a8593d55d337a66c34464a1df21aa6ece53f8b641`).
It reviewed the full eight-file candidate diff against planning commit
`4ef8d4389bb6baa2320770e3077737a946f01e82`, production callers, contract,
normative documents, tests and preserved evidence.

The acceptance evidence and exact command lines remain in the preceding
historical report and in the referenced support scripts and logs. Acceptance
commands/results were:

- `bash .cache/evaluation/qw38-language-v2/task045-support/parent.sh` — exit
  0; parent baselines and diagnostics completed.
- `bash .cache/evaluation/qw38-language-v2/task045-support/candidate-attempt1.sh`
  — exit 8 from the documented initial FP8 fixture admission error. The final
  candidate was subsequently tested after repairing the fixture only.
- `bash .cache/evaluation/qw38-language-v2/task045-support/candidate.sh` —
  exit 0; the final Release FP8 test passed, final candidate benchmarks and
  the matched request completed, and the frozen eight-window NLL screen
  passed. The unchanged final Release tests `q4k_q8`,
  `q8_mlp_integration`, and `language_model_integration` were reused from
  their passing run. All four required Release tests therefore passed.
- `bash .cache/evaluation/qw38-language-v2/task045-support/diagnose-mlp.sh`
  and `bash .cache/evaluation/qw38-language-v2/task045-support/analyze.sh` —
  exit 0; diagnostic analysis completed. The analysis script's original
  BLOCKED classification records the unamended gates and is preserved as
  historical output; the explicit development acceptance supersedes that
  status for this candidate.
- `sha256sum --check
  .cache/evaluation/qw38-language-v2/task045-support/candidate-sources.sha256`
  and `git diff --check` — passed at implementation/review handoff.

On continuation, the main thread reconfirmed unchanged reviewed source,
test, benchmark, parent/candidate benchmark-binary and evaluator-binary
hashes; the source identity file remains the candidate identity record.
The preserved `comparison.json` SHA-256 is
`d4fea6bbe06b7502c27c4505d78a00ae79a4d9ca174be148541d116ed2d4d4b8`.
No tests, benchmarks or evidence-producing commands were rerun during Luna's
documentation phase. The final evidence reports NLL delta
`+0.00045418668087737757`, decode improvement `12.675%`, FP8-family reduction
`6.600%`, and complete inference reduction `0.687%`. It retains the MLP
`+0.539%` and prefill `+0.480%` measurements as failed original gates accepted
only for this development candidate; their causes remain unresolved.
TASK-043 remains production. TASK-048 gates are unchanged, and TASK-046–048
remain TODO. Integrated Q4 attribution remains limited as stated above.

The main thread removed the only eligible task-specific cache directory,
`.cache/task045/` (22 MB, six copied benchmark executables), after confirming
that no live process used it. `test ! -e .cache/task045` passed after cleanup.
The support evidence directory is retained. To reproduce, create
`.cache/task045/` with `mkdir -p` and rebuild the appropriate parent and
candidate revisions using the recorded driver diff and commands; the preserved
scripts do not create the directory, and running `parent.sh` against the final
candidate does not recreate the parent binaries. Luna (gpt-6-luna, medium)
prepared final documentation and delivery; commit and push outcome will be
reported by the delivery run. No production promotion is claimed.
