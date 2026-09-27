# TASK-039 — Token-boundary submission and decode graph replay

## Status

TODO

## Milestone and dependency

M18 — Bounded submission overhead with correct state commits. Depends on
TASK-038. [FAST-02](../task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
authorizes the scheduling change and supersedes the historical graph deferral.
This task does not change numerical policy, model equations or persistent
state layout. Host wait time is not additive to GPU time; no twofold graph-only
speedup is presumed from the TASK-036 trace.

## Enqueue and completion contract

Separate private enqueue operations from public synchronous layer APIs.
Validate immutable bindings once; validate session health, stream, token IDs,
position, populated lengths, cursors and capacity before enqueueing a token
or chunk. Carry explicit pending position/populated/cursor values through
internal calls. Attention must include the newly appended KV without relying
on prematurely updated host metadata. Remove per-layer stream waits and
host commits from the full-model route; standalone APIs retain their current
documented completion guarantees.

For decode, enqueue embedding, all layers, final norm/head and full FP32 logit
readback on the session stream. Synchronize once at the complete-token
boundary. After successful completion and finite-logit/greedy validation,
commit all layer cursors/positions/populated counts and the global token
position together. This host commit must not partially fail: prevalidate all
commit conditions and use a no-fail internal update after completion. Preserve
greedy ties and existing evaluator logits; GPU argmax is out of scope.

For prefill, retain 256-token chunks and defer layer/global commits to each
completed chunk. Requested-row readbacks/sink calls may require their existing
completion boundaries; do not introduce a new streaming output API. A failed
sink poisons the session as today; no partially executed chunk is reported
as committed. Earlier fully completed chunks may remain committed. This is
not a rollback transaction over already-mutated device bytes.

Invalid input rejected before enqueue leaves state unchanged and usable.
Any failure after work may have mutated device state poisons the session,
leaves the failed unit's host counters uncommitted, and forbids continuation
or snapshot until successful reset/restore. Drain queued work before releasing
borrowed buffers; propagate the original typed error. Do not retry partially
executed tokens through an eager fallback. Reset/restore must repair both
device bytes and host metadata. Preserve one-stream and move/lifetime rules.

## Static decode graph

Use CUDA graph RAII with one session/plan-owned executable graph, stable
addresses and a small device control record containing token ID, absolute
position, populated length and convolution cursor. Update controls on the
session stream before replay; never bake the first token's values into
subsequent launches. Keep readback outside the captured computation using
stable owned host staging, followed by the single completion boundary.

Use power-of-two active-KV launch buckets starting at 256 and capped by the
session capacity; kernels mask by actual length and retain TASK-038's
deterministic active partition rule. Reuse the graph within a bucket. On a
bucket change, replace the one executable graph after prior work completes;
do not cache a graph per position or capture every token. Reset/restore must
refresh controls and check the bucket. Invalidate on changed addresses,
capacity, model/policy, device/stream or relevant plan geometry. Moves must
not leave graph references to moved host objects. Session interleaving must
not share mutable graph controls or scratch.

Capture/instantiation must not execute a token, mutate state, commit counters
or read logits. Preallocate graph buffers/descriptors and initialize any
capture-incompatible library setup before capture. No warmup inference to
prepare capture. Report instantiation/initialization in actual first-use
timings, including bucket transitions. External user stream capture remains
unsupported at public synchronous APIs; the private route is the authorized
capture path. Use a concrete graph, not a runtime graph optimizer.

A graph construction failure before mutation may fall back to the new
enqueue-only eager route with the reason recorded. Replay/execution failure
requires poisoning, not retry. Graph replay must be implemented and tested
on the pinned platform; an unimplemented graph is not this fallback.

Likely code: `src/runtime/{language_model,language_layer,session,mlp,gdn,attention,prefill_gdn,prefill_attention}.*`,
CUDA launch wrappers and existing CUDA ownership/error helpers. Hoist only
static setup encountered in this execution chain. No general scheduler,
new concurrency feature, prefill graph, new chunk size or unrelated fusion.

## Checks and completion

1. Compare enqueue eager and graph execution bitwise for identical arithmetic
   and attention partition schedules: logits, persistent bytes, cursors and
   all positions. Use different consecutive tokens, nonzero positions,
   convolution ring wrap, attention partition changes and graph bucket
   boundaries (255/256/257 and 511/512/513 populated lengths).
2. Exercise prefill→decode, requested-row sink failure, reset, snapshot/restore,
   two interleaved sessions, move/invalidation and external-capture rejection.
   Verify capture alone changes neither host metadata nor persistent bytes.
   Inject a pre-enqueue validation failure and failures after early and late
   device mutation; prove unchanged/usable versus poisoned behavior, forbidden
   snapshots and exact recovery through production callers.
3. Use the existing trace/allocation checks to establish one ordinary decode
   stream completion per token, no per-layer waits/commits, one graph replay
   per eligible token after first setup, no within-bucket recapture and no
   steady-state allocation. Count bounded graph setup allocations separately.
4. Once per arm, time 128 fixed decode inputs at populated length 4096 for
   TASK-038's synchronized route, new enqueue eager, and graph replay. This
   is the one justified longer development comparison: eight tokens cannot
   characterize the first-use graph cost and subsequent reuse together.
   Include first capture/instantiation and readout in each applicable timing.
   Reuse authenticated identical existing rows where genuinely comparable;
   no llama.cpp rerun or other context sweep. Prefill setup is excluded from
   decode timing. Existing state checks cover short request integration.

Complete when token/chunk completion semantics and the reduced-wait eager
path pass, graph replay passes on the pinned platform, and selected decode
has lower observed total latency than the TASK-038 route. Select graph replay
by default only if its complete measured cost beats enqueue eager; otherwise
retain enqueue eager by default with graph available through explicit
diagnostic selection and record the remaining overhead. No other numerical
or kernel redesign is authorized to obtain that win. If neither submission
path improves total latency or state correctness cannot be preserved, record
the blocker rather than counting removed waits as achieved speedup.
