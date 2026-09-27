# TASK-042 — Shared-KV attention with bounded 512-token prefill

## Status

DONE

## Milestone and dependency

M21 — Prefill head reuse with enough parallel work. Depends on TASK-041.
[FAST-03](../task_ledger.md#prefill-amendment--fast-03-2026-09-27) reopens
the former fixed-256 chunk decision for this bounded change. Keep the
TASK-041 compact MLP consumer, policy 1030 and decode graph/MMVQ arithmetic.

## Delivered behavior and first implementation

Use full 512-token chunks for long prefill and a grouped attention consumer
sharing a staged K64/V64 tile between **two sibling query heads × Q32**.
There are three head pairs per KV head. Use eight warps with independent
per-head causal softmax, Q fragments and FP32 PV accumulators, retaining the
existing K/V staging scheme. At M512 this supplies 12 pairs × 16 query tiles
= 192 CTAs on the 170-SM GPU. At M256 the same grouping supplies only 96;
therefore retain the current per-head prefill kernel for M<512, including
short prompts and final tails. Do not copy the decode head mapping blindly.

Use four warps per head, adapt staging strides for 256 threads and provide
distinct per-head output scratch before direct FP8 packing. Derive compiled
register/shared-memory requirements rather than multiplying an optimistic
occupancy estimate.
Current prefill uses 171 registers/thread and 67,584 shared bytes/block;
the proposed grouping should share KV storage, not duplicate it per head.
Reuse pinned comparator fragment-layout patterns if necessary, without
importing its lower probability precision or a generic scheduler.

Extend the full bounded prefill capacity chain to 512: arena/residual
allocations, Q8 MLP codes/scales/sums and gate/up slabs, Q8 geometry guards,
FP8 operands/accumulators, attention/GDN workspaces, token input storage and
requested-row handling. `kArenaTokenCapacity`, `kQ8MlpMaxTokens`, and the
literal M<=256 validators must agree. The standalone FP8 engine already
supports larger M; that does not establish full-runtime support. Preserve
public behavior for valid smaller requests and reject invalid capacities
before mutation. Requests of 257–511 tokens may now form one chunk: retaining
the smaller-chunk kernel does not imply the old partitioning or bitwise
cross-schedule equality. Leave the GDN state-resident recurrence interval at 64.

This is a coupled scheduling change: larger chunks alone do not reduce the
number of fixed-Q32 scans or fixed-J weight stages. The larger chunk makes
the grouped attention mapping viable, while TASK-041 supplies wider MLP
reuse. Do not add a chunk sweep, 1024-token mode, prefill graphs, extra
persistent KV/weight views or new GDN algorithm.

Likely code: `cuda/attention.{cu,hpp}`, `cuda/q4k_q8.{cu,hpp}`,
`src/runtime/{sizes,session,language_model,prefill,prefill_attention,prefill_gdn}.*`,
and affected existing tests/benchmarks. Extend the request benchmark's
existing development mode only enough to accept the frozen 4096-token
prompt plus eight fixed continuation inputs; no new benchmark framework.

## Numerical, state and memory contract

Keep QK normalization/RoPE, absolute-position causality, BF16 KV/history,
FP32 online softmax/statistics/PV, gating and existing local BF16 rounding.
Retain both `P_hi=BF16_RNE(P)` and `P_lo=BF16_RNE(P-float(P_hi))` PV products.
Preserve direct FP8 output packing, per-row K128 scales, zero padded rows and
per-head output offsets. Each head's result must be independent of sibling
head contents; a shared KV tile must not share softmax state.

Prefill state commits only after completed chunks; invalid pre-enqueue input
leaves state usable, and post-mutation failure poisons it with the failed
chunk uncommitted. Preserve requested-row sink completion/failure behavior,
nonempty continuation, reset/restore, movement/interleaving and decode graph
address/bucket validity after larger workspace allocation. No warmup token or
execution retry after partial mutation. Persistent session/state ABI is
unchanged; scratch capacity is an execution choice.

Compute the new peak workspace before allocation, then observe it in checks.
Count all new slabs/pack padding, library and graph allocations, including
first use. Keep the 2 GiB free reserve at 32,768+128 slots; TASK-043 measures
final whole-request capacity. Do not infer fit solely from TASK-040's 8.741 GB
free sample. Keep one session-owned reusable workspace rather than accumulating
one allocation per chunk.

## Checks and completion

Use existing attention/reference/artifact, Q8 MLP and full-model session tests.
Exercise all three head pairs and all six members of a KV group with distinct
queries/gates; causality; nonzero prefix; K64/Q32 tails; one-token and short
prefill; dispatch and capacity edges 255/256/257 and 511/512/513. Include
prefill→decode, requested-row readback, same-schedule replay, snapshot/restore,
interleaving and failure recovery. Same-schedule replay stays bitwise;
cross-schedule full-model differences are diagnostics under existing policy,
not permission to apply isolated attention tolerances to the whole model.

For the attention operation, compare two M256 calls versus one M512 call on
the same 512 input tokens and starting state, at an empty prefix and a prefix
of 32,256 tokens (ending at 32K). Include prepare/cache append, attention,
gating and output projection. Use one execution per schedule/case, zero
warmups/repetitions, identical arithmetic and inputs. Include a smaller
unchanged-path check; do not require a sweep over every prefix. Validate
numerics with the independent reference and existing tolerances, and record
actual dispatch/resources. Retain the same-input state/reference controls.

Compare one matched 4096-token development prefill plus eight teacher-forced
continuation inputs against TASK-041's 256-chunk runtime, using frozen
TASK-027 inputs. Report both phases, all first use and final logits. This
exercises grouped prefill; the old 256-only development check cannot do so.
Record target-NLL/logit diagnostics against the authenticated saved reference
continuation where available; no answer selection or new calibration. Final
core-54 and fixed 32K quality remain TASK-043 obligations.

Completion requires correct selected 512/grouped execution, improved complete
long-prefix attention and integrated prefill, no observed matched decode
regression, valid capacity accounting and passing affected state checks.
If the first schedule regresses, a single M512 comparison with the existing
per-head attention kernel may identify whether grouping or outer chunking is
responsible. This is a diagnostic, not an automatic fallback completion or
permission for a tile/partition search. Retain the parent until repaired;
record a concrete blocker/replan if the intended combined milestone fails.
No full quality/performance suite before TASK-043. Long main-thread commands
use `wake-run`.

## Historical blocked implementation report — 2026-09-27

This records the original pre-amendment blocked state. The acceptance and
verification findings below were later resolved for TASK-042 delivery as
described in the completion report; the original measurements and review
remain part of the history.

The M512 paired-prefill candidate is implemented but is **not accepted**.
The required matched eight-input decode phase regressed from 150.054052 to
156.324259 ms (+4.1786%). Keep TASK-041 as the accepted development parent;
production remains TASK-026/027. TASK-043 is not activated. Candidate changes
remain uncommitted for a specific repair or explicit acceptance amendment.

The implementation gives each of two sibling query heads four warps, separate
Q/softmax/PV registers and separate output scratch, while sharing K64/V64.
M512 launches 12 head pairs × 16 query tiles = 192 CTAs, 256 threads each.
Smaller chunks retain the per-head path. All capacity-dependent residual,
arena, Q8, FP8, attention/GDN, token-input and row-readback storage now supports
512. Q8 validators agree; runtime attention/GDN workspace constructors reject
513 before allocation. GDN recurrence remains 64. Policy 1030, persistent
state ABI, decode MMVQ/graph arithmetic and the single weight/KV view remain.

Main-thread implementation: GPT-6 Codex (specific model variant unavailable).
Parent: `56cd53bda200bfa76ab5c279e58616e27b112945`.
Artifact: `.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38`, policy 1030,
manifest `6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
No tensor payload hashes were computed. Input identity matches TASK-027's
`inputs.json`: `tokens-4096.u32le` SHA-256
`fa28a473299ca58495ee860d08f8c532987f3df7de95563d0f575275817ba923`.

Evidence is retained under
`.cache/evaluation/qw38-language-v2/task042-support/`; task-specific executables
remain under `.cache/task042/`. `candidate.diff`, `candidate-sources.sha256`
and `candidate-binaries.sha256` freeze the reviewed candidate. Hardware is
RTX 5090, driver 590.48.01, CUDA 13.4.1 / nvcc V13.4.59, pinned image
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`.

Exact commands (all long commands launched via wake-run; each script preserves
its full Docker/build/test/benchmark arguments):

- `bash .cache/evaluation/qw38-language-v2/task042-support/baseline.sh` — PASS
  after fixing an initial benchmark fixture alias rejection, before measurement.
- `bash .cache/evaluation/qw38-language-v2/task042-support/check.sh` — build
  PASS; six candidate tests PASS. The legacy GDN artifact fixture failed because
  it was invoked with an unsupported FP8 artifact, not its Q4 control.
- `bash .cache/evaluation/qw38-language-v2/task042-support/finish-checks.sh` —
  PASS after restoring the original ungrouped warp expression. Final affected
  `attention_prefill`, `attention_prefill_artifact`, `language_model_integration`
  passed with policy 1030. `runtime_plan`, `q4k_q8`, `q8_mlp_integration` remain
  valid from the preceding run. `gdn_prefill_artifact` passed with its intended
  `candidate-v2-q4k-rope-fixed.qw38` control. The candidate FP8 GDN path is covered
  by the full-model M512 tests. The additional command
  `build/pinned-release/tests/qw38_attention_prefill_test 32k` passed; maximum
  BF16 error 0.00634766 under the existing 0.016 tolerance.
- `bash .cache/evaluation/qw38-language-v2/task042-support/measure.sh` — all
  executions completed; required matched decode performance gate FAIL.
- `bash .cache/evaluation/qw38-language-v2/task042-support/diagnose-first-use.sh`
  — one diagnostic trace per arm, completed; not replacement acceptance results.
- `git diff --check` — PASS. Source/executable identity rechecks — PASS.

Checks cover all head pairs, independent references, sibling isolation, direct
FP8 pack codes/scales and zero padding, causality and nonzero prefixes,
255/256/257 and 511/512/513 edges, real Q8 MLP shapes, same-schedule replay,
requested rows and sink failures, snapshot/restore, graph handoff/buckets,
movement/interleaving and poisoning/recovery. A failed second chunk leaves only
the first 512 committed. No tolerance or quality gate was weakened.

| Complete operation | Parent | Candidate | Result |
| --- | ---: | ---: | --- |
| Attention, empty prefix, 512 inputs, GPU | 0.673024 ms | 3.927488 ms | First-use regression recorded |
| Attention, prefix 32256, 512 inputs, GPU | 8.793728 ms | 7.315488 ms | Required long-prefix improvement |
| Attention, prefix 65, M32, GPU | 0.709440 ms | 0.885408 ms | First-use regression recorded |
| Matched 4096-token prefill | 1916.225098 ms | 1553.510184 ms | 18.9286% lower |
| Matched eight-input decode | 150.054052 ms | 156.324259 ms | **4.1786% higher; required gate FAIL** |

These are single observations per arm/case, zero warmups/repetitions, including
first use. Complete attention includes preparation/cache append, scan/gating,
direct FP8 packing, real output projection/residual add and completion. Setup
is recorded separately in raw logs; candidate empty-prefix setup was 13.403 s
versus parent 3.454 s, and full request setup 20.233 s versus 5.530 s. No setup or
first-use difference is hidden. All three complete-attention output arrays match
bitwise. Integrated output IDs match and all final logits are finite at position
4104. Maximum final-logit delta is 1.479426384, a cross-schedule diagnostic only.
NLL for the authenticated next fixed input417 is 0.00008884623796 versus
0.00015495475525; this parent comparison is not a core-quality acceptance claim.

The first decode observation was 21.262324 → 29.279882 ms; the remaining seven
sum to 128.790766 → 127.043607 ms. Diagnostic traces show the same 883 first-token
kernels. Parent/candidate GPU kernel sums are 18.33096/18.518773 ms and initial
host-to-first-kernel intervals 6.390148/6.650210 ms, including graph instantiation
3.657109/3.886116 ms. The original ~8 ms gap was not reproduced under tracing;
its cause remains unproven. Diagnostic decode totals also slightly regressed,
158.266690 → 158.720723 ms. No evidence-supported in-scope repair was identified;
these runs do not waive or replace the original failed criterion.

Actual paired launch resources are 179 registers/thread, 67,584 dynamic shared
bytes, zero local bytes and one resident CTA/SM. Per-head resources remain
171 registers/thread and the same shared bytes. The independent trace confirms
12×16/256 versus parent 24×8/128 launch geometry.

`workspace-plan.txt` was calculated before candidate allocation. Observed
capacity 32,896 with first M512 use and maximum-capacity graph capture matches:
prefill 254,248,960 bytes; arena 513,802,240; two residuals 20,971,520; Q8 scratch
82,444,292; persistent state 2,309,816,320; model 21,013,686,400. Free memory is
8,331,984,896 bytes, above the 2 GiB reserve. This is provisioning/first-use
accounting; final populated 32K capacity, core-54, retrieval and PERF-01 remain
TASK-043 obligations.

Required follow-up: a specific authorized repair/replan of the decode regression,
or an explicit acceptance amendment retaining the regression as unresolved.
Final Luna delivery was not started. No commit or push was made.

Independent Astra review: **BLOCKED**, pass 1, recorded verbatim in
`astra-review-1.md` in the evidence directory. Source and executable identity
checks passed immediately before recording the outcome. No runtime changes
were made after measurement or during review.

- **R042-1 (blocking):** matched decode performance fails the unchanged gate.
  The first-use cause remains unresolved; diagnostic runs do not replace the
  acceptance result. A specific repair/replan or explicit acceptance amendment
  is needed before delivery.
- **R042-2 (medium verification defect):** the maximum-capacity test stops after
  M512 prefill and graph instantiation. It does not execute first decode or
  upload the maximum-capacity executable graph before the memory sample.
  Therefore, **8,331,984,896 free bytes is a post-prefill/post-instantiation
  observation, not complete graph-first-use capacity acceptance**. The stdout
  label refers to library first use and graph capture only. On resumption,
  extend the capacity fixture with valid first decode plus completed maximum
  graph upload, keep all relevant allocations live, and rerun that affected
  check once. Do not move setup out of acceptance timers. Final populated 32K
  capacity testing remains TASK-043.

Astra found no additional material runtime correctness defects. Both findings
remain open. The candidate and task-specific caches are preserved, TASK-042 is
BLOCKED, and the sequence stops without activating TASK-043.


## User-directed candidate acceptance — 2026-09-27

The user explicitly approves this TASK-042 candidate after reviewing the
measured regressions and their uncertainty. This acceptance supersedes the
candidate's no-observed-matched-decode-regression requirement (R042-1).
Preserve the original 150.054052 → 156.324259 ms result (+4.1786%), the slower
first decode, the empty-prefix/short attention observations and cold-load
variability. Do not relabel any measured regression as a performance pass or
claim its cause is established. Other numerical, state, capacity and review
requirements remain; R042-2 was still pending at this decision and is resolved
in the completion report below.
The user authorizes resuming the preserved candidate toward delivery.

FOLLOW_UP_REQUIRED: honor the user's direction to use distinct paths below
and above 4096 tokens, using `../ds4` as a design reference. This is follow-up
scheduling work, not a condition requiring a new dispatcher in this accepted
candidate. Current dispatch remains M512 paired attention, smaller chunks
per-head attention, and existing decode. DS4's phase/shape selection patterns
are relevant (`cuda/mmq/ds4_mmq.h`, `cuda/mmq/ds4_mmq.cu`); no claim is made that
its literal K<=4096 matrix-dimension guard is a prompt-length threshold.
Preserve this direction for subsequent planning without adding roadmap tasks
or changing TASK-043's final gate in this delivery.

## Completion Report — 2026-09-27

TASK-042 is **DONE** under the user's explicit acceptance amendment. R042-1
was resolved by accepting the candidate with its measured regressions; it was
not repaired or waived silently. The original +4.1786% matched decode result,
first-token delay, isolated attention regressions and cold-load variability,
with causes unresolved, remain documented above. The user-directed distinct
paths below and above 4096 tokens, with `../ds4` as a reference, are
`FOLLOW_UP_REQUIRED`. This delivery adds no prompt-length dispatcher or downstream
specification change. Production remains TASK-026/027, and TASK-043 remains
`TODO` with its final quality, populated-32K capacity and performance gates.

R042-2 was repaired in the capacity test and `cuda/graph.hpp` friend access
only. The test completed a valid first decode, required successful maximum
graph upload and synchronization, then sampled capacity while allocations
remained live. At capacity 32,896, with 512-token chunks, it observed
8,331,984,896 free bytes, above the 2 GiB reserve. State remained committed at
513. No production execution or benchmark code changed in this repair.

Implementation model: GPT-6 Codex (exact variant unavailable). Candidate
source and binaries match review pass 2 identities; the main thread also
verified the reviewed source/executable hashes and candidate diff remained
unchanged. The full evidence, scripts, outputs and identity manifests remain
under `.cache/evaluation/qw38-language-v2/task042-support/`. Preserved parent
executables with verified executable hashes are in
`.cache/evaluation/qw38-language-v2/task042-support/parent-binaries/`; the
original command paths under `.cache/task042` are historical. Hardware and
artifact identity are recorded in the blocked implementation report above.

Exact acceptance commands and results:

- `bash .cache/evaluation/qw38-language-v2/task042-support/baseline.sh` —
  PASS after correcting the benchmark fixture alias.
- `bash .cache/evaluation/qw38-language-v2/task042-support/check.sh` — build
  and six candidate tests PASS; the legacy GDN artifact fixture used the
  unsupported FP8 artifact and failed, then passed with its Q4 control in the
  final check.
- `bash .cache/evaluation/qw38-language-v2/task042-support/finish-checks.sh`
  — affected checks, legacy GDN control and independent 32K reference PASS.
- `bash .cache/evaluation/qw38-language-v2/task042-support/measure.sh` —
  executions completed; matched eight-input decode measured
  150.054052 → 156.324259 ms (+4.1786%), accepted by user with regression
  retained as unresolved. Matched prefill measured 1916.225098 → 1553.510184
  ms (18.9286% lower).
- `bash .cache/evaluation/qw38-language-v2/task042-support/diagnose-first-use.sh`
  — diagnostic traces completed; they do not replace acceptance measurements
  or establish the regression cause.
- `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 build/pinned-release/tests/qw38_language_model_integration_test --capacity-only`
  — PASS; valid first decode and maximum-capacity executable graph upload
  completed before the memory sample.
- `bash .cache/evaluation/qw38-language-v2/task042-support/review-repair.sh`
  — PASS; capacity-only first-decode and maximum-graph-upload verification.
- `git diff --check` — PASS.

Independent Astra review: pass 1 BLOCKED on R042-1 and R042-2; pass 2 PASS
with no findings or evidence requests after the acceptance amendment and
capacity repair. Review files `astra-review-1.md` and `astra-review-2.md` are
preserved in the evidence directory. Commit and push results are reported at
delivery.

Main-thread documentation and identity checks passed. Removed `.cache/task042`
(7.7 MiB); no earlier DONE-task caches remained. Parent executables and shared
evaluation evidence were preserved.
