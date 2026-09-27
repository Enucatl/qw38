# TASK-042 — Shared-KV attention with bounded 512-token prefill

## Status

TODO

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
