# TASK-024 — GDN prefill algorithm and layer integration

## Status

DONE

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Select a numerically validated GDN prefill algorithm using complete-layer costs and integrate its convolution, state and output paths.

## Depends on

- [TASK-023](TASK-023.md)

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
| G-02, T-02, M-01 | Compare ordered and chunkwise recurrence before final schedule selection | OVERALL-01 |
| S-01/S-02, P-01 | FP32 state in the current ABI and unchanged recurrent equations/arithmetic | Retained control |
| Q-01/P-02/L-01 | Keep accepted projection representation fixed during algorithm comparison | TASK-022/023 selection |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-023 supplies complete projection and MLP prefill paths. Decode convolution/recurrence and continuation references exist.

## Scope

Implement parallel convolution/preparation with correct incoming raw history
and race-free history commit. Compare a state-resident ordered recurrence with
a mathematically equivalent chunkwise/WY-style formulation early enough to
affect the prefill design. Keep FP32 state and the same equations/weights while
isolating algorithmic differences; specify arithmetic precision and workspace
for transformed intermediates. Do not quantize recurrent state to FP4.

Check independent recurrence equations, arbitrary incoming state, short tails,
chunk partitions and continuation into decode. Measure whole GDN+MLP layers,
state traffic and workspace alongside recurrence timing. **Exit:** a justified
prefill algorithm and complete GDN layer with bounded memory and numerical
evidence; the serial 64-token schedule is a control, not a mandatory final
choice. Any deferred faster candidate has a concrete measured reason.

## Out of scope

State precision/layout changes, FP4 recurrent state, new recurrence semantics, and selecting an algorithm from isolated recurrence speed while ignoring whole-layer costs.

## Required interfaces and data representation

A GDN chunk plan declares valid rows, incoming convolution history, positions, FP32 recurrent state, recurrence algorithm/interval, output staging and workspace lifetimes. Both compared algorithms consume the same projection policy and produce one output per valid token.

## Required semantics and constraints

Convolution reads incoming raw qkv history and valid chunk projections; history commit occurs after readers finish and preserves needed old entries for short chunks. Recurrence must realize the same ordered GDN map for arbitrary incoming state. Specify transformed intermediates and precision for chunkwise/WY evaluation, independently verify equations and retain FP32 recurrent arithmetic. Padding never updates history or state.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use the selected projection chunk plan; isolate recurrence-algorithm comparisons at matching chunks/precision/weights before broader tuning. The old 64-token serial interval is a control; record any reason for deferring a faster algorithm.

## Expected files/modules

GDN prefill FIR/preparation/history commit, ordered and chunkwise recurrence candidate, output transform, complete GDN+MLP runtime integration and reference evidence.

## Tests required

### Unit and contract checks

History lengths below/at/above the convolution boundary, valid-token masking, arbitrary initial state, recurrence tails and workspace limits.

### Reference and numerical checks

Independent convolution and recurrence equations; ordered versus chunkwise output/state with adversarial gates, short and long streams and documented finite-precision tolerances.

### Integration checks

Complete GDN+MLP layers across chunk partitions and prefill-to-decode continuation; confirm positions, history and same-schedule reset/snapshot replay. Full model behavior is TASK-026.

## Benchmark required

Required: recurrence and complete GDN+MLP timing, state read/write traffic, temporary workspace, resource usage and algorithm comparison at matching settings.

## Acceptance criteria

- [x] Convolution/preparation and history commit are correct for incoming prefixes, tails and valid rows; focused unit, reference and integration checks passed, including continuation, metadata, replay and failure recovery.
- [x] The ordered recurrence passed correctness checks. The chunkwise affine recurrence equation was independently checked in FP64; GPU WY execution is deferred for the measured dense-workspace and missing bounded low-rank GPU support reasons recorded below.
- [x] The selected ordered FP32 recurrence passes output/state and repaired artifact continuation checks while retaining the GDN equations.
- [x] Matched complete-layer timings, state traffic, workspace and resource usage support the 64-token interval; increasing the interval to 128 or 256 produced at most a 0.10% layer difference at M=256.
- [x] Complete GDN+MLP prefill runs within measured bounded scratch; the deferred GPU WY candidate has the concrete workspace/support limitation recorded below.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Review gate block — 2026-09-25

The independent Astra second review returned `CHANGES_REQUIRED` for R24-04.
`tests/gdn_prefill_artifact_test.cpp` loads five BF16 embedding rows but
allocates and converts only four FP32 input rows. Its fifth-row host copies
read beyond `input`, so the apparently passing continuation comparison cannot
establish the required numerical handoff. The tested candidate patch is
`.cache/task024/candidate.patch` (SHA-256
`08876c25248948b10a0aa6b228f1bbbbee840fb42211296386e1310b88765245`).
The second review resolved R24-01, R24-02, R24-03, R24-05 and R24-06, and
reported no other implementation or benchmark findings. The task skill's
second-review stop rule requires human-directed continuation before further
repair or review. On continuation, allocate and convert five FP32 rows, rerun
the artifact test with the accepted artifact, and refresh its binary/source
identity and continuation evidence. No commit or push has been made.

### Result

TASK-024 completed on 2026-09-25. R24-04 was repaired on user-directed
continuation: the FP32 input buffer covers all five loaded BF16 rows, and the
artifact test reports fifth-token intermediate and output differences. The
repaired artifact test passed with the accepted CandidateV2 artifact. The
independent Astra final review returned PASS with no code findings, acceptance
gaps or evidence requests. This was the third review pass overall, following
two historical review passes and the user-directed continuation. Production
code and benchmark evidence remained unchanged after review.

### R24-04 repair evidence — 2026-09-25

Source base and prior candidate patch are recorded above. Repaired test source
SHA-256: `45bcff34c55cf5c1925f2d45ec2efe22e161f9b00ebdc2b7083e6f6447f2711b`.
Rebuilt artifact-test binary SHA-256:
`96c505302220cf3df42023409ff279262b9e80c85de17155c05c601686d2af3e`.
Accepted artifact: `.cache/candidates/candidate-v2-q4k-rope-fixed.qw38`,
manifest-only SHA-256
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`.
Pinned image ID:
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.

Exact command from the repository root:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc 'cmake --build build/pinned-release --target qw38_gdn_prefill_artifact_test -j 8 && sha256sum tests/gdn_prefill_artifact_test.cpp build/pinned-release/tests/qw38_gdn_prefill_artifact_test && ctest --test-dir build/pinned-release --output-on-failure -V -R "^gdn_prefill_artifact$"'
```

Exit code 0; 1/1 test passed in 3.64 seconds. Full command/build/test log:
`.codex-wake-run/7b510ce342f9.log`. Maximum absolute differences:

| Comparison | Observed maximum |
| --- | ---: |
| Partition intermediate / output / state | 0.000151873 / 0.00314561 / 8.66968e-06 |
| Four-token prefix intermediate / output vs decode | 0.000368915 / 0.00537595 |
| Fifth-token continuation intermediate / output | 3.11006e-05 / 0.00368273 |
| Continuation FP32 state / BF16 history | 1.16595e-05 / 0.00195312 |

Continuation tolerances remain unchanged: intermediate 0.2 absolute + 0.005
relative, output 0.35 + 0.005, state 8e-4 + 2e-4, history 0.02 + 0.006.
Metadata equality, output guards, chunk partitions, snapshot replay and
injected stream-failure recovery also passed. The injected failure messages
in the log are expected. This supersedes the earlier invalid continuation
evidence; production code and benchmark measurements are unchanged.

### Changes made

Selected the ordered, state-resident FP32 recurrence at 64-token intervals for
the complete GDN+MLP prefill path. The production path integrates projection,
parallel convolution/preparation, valid-row masking, post-reader history
commit, recurrence, output transform and MLP. All algorithm comparisons used
the accepted CandidateV2 artifact (manifest-only SHA-256
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`): Q4_K
MLP, Q8 other projection/head weights, BF16 activations/history and FP32
recurrent state and arithmetic.

The independent affine map writes each step as
`A_t = alpha_t (I - beta_t k_t k_t^T)`, `b_t = beta_t v_t k_t`, then composes
`P_t = A_t P_(t-1)` and `c_t = A_t c_(t-1) + b_t`, yielding
`S_t = P_t S_0 + c_t`. Its seven-token FP64 diagnostic uses 264,192 bytes for
one value row and agrees with the independent FP32 step reference within
`8e-4` absolute. This validates the transformed equation, not a GPU WY kernel
or finite-precision GPU equivalence. GPU WY was deferred because the selected
projection dependency does not provide the rank-one affine composition and
causal outputs; a direct dense FP32 representation for 48 value heads and a
64-token interval alone needs 192 MiB before composition/output workspace.
An efficient low-rank GPU triangular solve would require unsupported custom
work with no validated bounded implementation in this candidate.

### Tests run

The focused six-test run passed (6/6); exact command and initial full log are
in [benchmark evidence](../task-024-benchmark-evidence.md) and
`.codex-wake-run/e1cb155179b1.log`. It covered `gdn_unit`, `gdn_reference`,
`gdn_integration`, `gdn_prefill`, `gdn_prefill_artifact` and
`prefill_projection`. Low-level 193-token stress exercised near-one, tiny and
zero gates, cancellation-sensitive alternating signs, arbitrary nonzero
initial state and three chunk calls crossing interval boundaries, with FP32
state/output tolerance `1e-3` absolute plus `3e-4` relative. The seven-token
FP64 affine diagnostic used `8e-4` absolute.

The initial artifact continuation result was invalidated by the R24-04 source
review finding and is not used for acceptance. The repaired artifact test was
rebuilt and passed 1/1 in 3.64 seconds; its exact command, source/binary hashes,
accepted artifact identity, reported numerical differences, unchanged
tolerances, and metadata/replay/failure-recovery results are recorded in the
[R24-04 repair evidence](#r24-04-repair-evidence-2026-09-25) and
`.codex-wake-run/7b510ce342f9.log`. The repaired source SHA-256 is
`45bcff34c55cf5c1925f2d45ec2efe22e161f9b00ebdc2b7083e6f6447f2711b`; binary
SHA-256 is `96c505302220cf3df42023409ff279262b9e80c85de17155c05c601686d2af3e`.

No full-model behavior or quality-gate completion is claimed; full-model
prefill behavior belongs to TASK-026.

### Benchmark results

The pinned environment was RTX 5090 (32,607 MiB), SM120, CUDA 13.4.1,
driver 590.48.01, 400 W limit, image
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
The benchmark binary SHA-256 is
`71e5f789a05e43f3abe52c2c37227c2cdf2a7de699ad851c156c40929e0dda6f`. It uses
16 real candidate artifact embedding rows repeated to each token count, resets
session state for each sample, discards the first of five repetitions and
reports medians of the four remaining samples. These are development layer
measurements, not whole-model or quality acceptance.

At M=64/128/256, the selected interval-64 complete-layer medians were
2.253745/2.639230/3.518990 ms and recurrence medians were
0.068880/0.132160/0.254992 ms. Interval-1 complete-layer controls were
2.347695/2.827425/3.892605 ms. At M=256, intervals 128 and 256 measured
3.517505 and 3.515505 ms, at most 0.10% from interval 64, so the larger
interval did not justify changing the default. Removing recurrence entirely
could save at most 3.06%, 5.01%, and 7.25% at M=64/128/256 before WY work.
The selected interval reads and writes 6 MiB of FP32 state per interval and
has modeled state traffic of 24 MiB at M=256. The 64-token prefill scratch plus
projection engine uses 64,684,032 bytes at 256 tokens. Recurrence resource
usage: 40 registers/thread, 1,024 shared bytes/block, no local bytes and 12
resident blocks/SM. State traffic is modeled at launch boundaries, not
profiler-measured DRAM transactions. Raw samples, timing boundaries and full
context are preserved in
[benchmark evidence](../task-024-benchmark-evidence.md); the duplicate cache
CSV is removed during task delivery.

### Architecture blocker

None. The GPU WY candidate was deferred under the contract's fallback rule; it
does not block the selected ordered implementation.

### Follow-up observations

Full-model prefill, handoff and quality gate are owned by TASK-026. The measured
layer results remain development measurements; whole-request performance and
promotion are owned by TASK-027 and TASK-032. The initial invalid R24-04
continuation evidence is superseded by the repaired evidence above. The
historical review-gate block is retained as history. Final reviewed candidate
identity: source base `e0be8fc4617e3ac9e67d650b9cbea016767ad34f` plus
`.cache/task024/candidate-final.patch`, SHA-256
`8d779b064066b3d20e8a653b4087f5c6fb3828f09660502ef180ff1ef2cfef9b`. The
patch was used for review and is removed with the task cache on delivery.
