# TASK-039 — Token-boundary submission and decode graph replay

## Status

DONE

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

## Completion report

**Status:** DONE. Implemented by GPT-6 Codex; independently reviewed by GPT-6
Astra (high), pass 1: PASS with no code findings, acceptance gaps, or evidence
requests. The reviewed source/documentation hashes in
`.cache/evaluation/qw38-language-v2/task039-support/candidate-files.json`
were confirmed unchanged before delivery documentation.

**Acceptance commands and results.** Exact command scripts are preserved in
`.cache/evaluation/qw38-language-v2/task039-support/`:

- `bash .cache/task039/baseline.sh` — PASS; built the TASK-038 parent and
  recorded the synchronized decode baseline.
- `bash .cache/task039/check.sh` — PASS; full policy-1030 model check and six
  focused public-API CTests passed; final source rechecked by `final-check.sh`
  below.
- `bash .cache/task039/validate.sh` — PASS; real-artifact Q8 MLP checks,
  Nsight trace/allocation check, and eager/graph populated-decode timings
  passed.
- `bash .cache/task039/final-check.sh` — PASS; final graph default and private
  prefill path passed the full model check; affected benchmark clients built,
  and the GDN mixer, MLP, and attention clients smoke-executed.
- `git diff --check` — PASS.

The literal CTest commands, run in the pinned image, were:

```sh
QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R '^language_model_integration$'
ctest --test-dir build/pinned-release --output-on-failure -V -R '^(gdn_integration|attention_integration|language_layer_integration|gdn_prefill|attention_prefill|runtime_session_integration)$'
QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R '^q8_mlp_integration$'
```

All passed. Focused outcomes and acceptance mapping are in `evidence-notes.md`;
preserved logs include `model-check.log`, `focused-checks.log`,
`q8-standalone.log`, `submission-check.log`, `trace-token-api.json`, and
`selected-model-check.log`. The native trace showed one ordinary decode
completion per token, one graph launch on graph tokens, no within-bucket
instantiation, and no steady-state allocation. Eager and graph outputs,
persistent bytes, cursors, and positions matched at tested graph boundaries;
failure poisoning/recovery, prefill chunk commits, capture rejection, moves,
restore, and interleaved sessions passed.

**Matched timing acceptance.** One zero-warmup run per arm used the same
authenticated 128 fixed continuation IDs at populated length 4096, the same
policy-1030 artifact, and included first graph capture/instantiation and full
readout. Platform: RTX 5090, driver 590.48.01, CUDA 13.4.1 pinned image.

| Arm | Total decode time | First token |
| --- | ---: | ---: |
| TASK-038 synchronized | 2536.146837 ms | 19.898666 ms |
| Enqueue eager | 2392.705637 ms | 18.722648 ms |
| Graph | 2380.103560 ms | 21.691609 ms |

All output IDs matched. Graph was selected by default: its observed total was
6.1528% below TASK-038 and 0.5267% below enqueue eager. This is a single-run
observation, not a statistical claim. The explicit diagnostic selection is
`LanguageModelPlan::set_decode_submission(DecodeSubmission::Eager)`
(benchmark environment:
`QW38_DECODE_SUBMISSION=eager|graph`). `comparison.json` retains the arithmetic
and `hardware.txt` records platform details.

The pinned host staging allocation is 993,500 bytes; each plan owns a 4-byte
status flag and 216 bytes of graph controls. First graph setup showed a
2,097,152-byte decrease in free device memory; tested bucket replacements
showed no additional decrease. Graph setup allocation is counted separately
from the zero steady-state allocation result.

Earlier validation failures and their repairs are retained in the support
logs: hidden per-layer Q8 MLP waits were deferred with a sticky unit error
flag; a test lambda capture/build error and a missing standalone-test artifact
environment were corrected. No unresolved new blocker remains.

Post-timing edits selected the default and hoisted redundant private prefill
validation without changing either timed decode path; the final model check
covers the final source. No timing rerun was needed. Earlier TASK-037 M=1 and
TASK-038 prefill regressions remain recorded; TASK-040 retains final quality,
replay, capacity, and performance obligations. No promotion decision is
claimed. Durable scripts, logs, identities, comparison, trace, and review are
indexed in
`.cache/evaluation/qw38-language-v2/task039-support/evidence-notes.md`.
The main thread confirmed the reviewed implementation stayed unchanged,
verified preserved evidence hashes against `delivery-evidence-identities.json`,
and removed the matching `.cache/task039` directory (7,834,064 bytes) after
preserving the evidence; no open handles remained. Completion documentation
was prepared by GPT-6 Luna.
