# TASK-038 — Pipelined attention with shared KV reuse

## Status

DONE

## Milestone and dependency

M17 — Efficient prefill and decode attention. Depends on TASK-037.
[FAST-02](../task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
governs scope and validation. Use TASK-037's selected development policy.

## Delivered behavior and first implementation

Replace the current fixed-D256 attention mainloop with an explicit MMA
fragment layout and pipelined KV staging. Reuse installed CUTLASS/CuTe
primitives and the established layout/scheduling patterns in pinned llama.cpp
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`
(`fattn-mma-f16.cuh`, `fattn-common.cuh`, `fattn.cu`), with attribution for
adapted source. This authorizes using those patterns, not copying their
lower-precision accumulation policy. No new attention framework or dependency.

Prefill starts with Q32/K64 and head dimension 256, retaining the running
FP32 PV numerator in the MMA accumulator layout across key tiles. Convert
score fragments directly into the probability operand layout, eliminating
the repeated 32 KiB shared PV-product round trip. Use asynchronous, separately
staged K/V loads with a bounded two-stage pipeline; stage smaller load subtiles
where necessary rather than requiring two full K64/V64 buffer pairs to fit.
Preserve online
rescaling when the running maximum changes. A resource-driven tile adjustment
is tuning, not a new architectural decision; justify it with the affected
resource report rather than a tile sweep. Integrate the existing gated output
and direct FP8 pack for the output projection.

Decode forms one small query tile from the six query heads sharing each KV
head, with two masked padding slots if an eight-slot MMA layout is used.
Load K/V once per such tile and KV partition, retaining independent scores,
maxima, denominators, gates and output coordinates per query head. Start with
the pinned grouped-head layout and split scheduling pattern. Split visible
KV into a bounded number of contiguous ranges to provide enough blocks for
the device, instead of generating one partial for every fixed 256 keys.
Merge partials cooperatively in a deterministic partition order; final
division, sigmoid gating and stores are distributed across threads.

Choose one deterministic partition rule from populated length, query tile
count and device SM count, capped by a checked workspace bound. Freeze it in
the selected implementation and report it. Kernels must distinguish launch
capacity from active length, so TASK-039 can use a fixed launch bucket with
changing positions. Changing the partition rule can change reduction order;
same-schedule replay must remain bitwise, not necessarily bitwise versus the
old segmentation.

## Numerical and memory contract

Keep BF16 Q/K/V and the existing KV ABI; FP32 scores, maxima, exponentials,
denominators, rescaling and output accumulation. Prefill retains TASK-033's
two-component probability recipe, `P_hi = BF16_RNE(P)` and
`P_lo = BF16_RNE(P - float(P_hi))`, contributing both products to FP32 PV.
For grouped MMA decode, use this same two-component recipe as an explicitly
authorized new decode rounding boundary; validate it against the independent
FP32 attention reference. Do not silently substitute a single BF16 P or FP16
accumulator. Gating/nonlinearities remain FP32, followed by existing producer
rounding/packing. No KV requantization, global score/probability matrix,
duplicate persistent KV or changes to model equations/QK norm/RoPE.

Query heads h=6g..6g+5 read KV head g; padding heads never alter statistics or
write outputs. Causal limits use absolute positions even for populated-prefix
prefill. Empty/tail partitions are neutral in merge and never read uninitialized
partial data. Keep split statistics/output scratch session-owned and bounded;
size it once from capacity/dispatch limits and reuse it on the session stream.
Preserve complete-token/chunk commits and poison/reset/restore semantics.

Likely code: `cuda/attention.{cu,hpp}`, attention runtime/prefill binders and
workspace, existing attention references/tests and attention benchmarks.
Q4/FP8 projection policy, GDN recurrence and token chunk capacity remain fixed.

## Checks and completion

- Run affected independent attention tests once, covering asymmetric heads,
  all six GQA siblings, padding heads, query/key tails, strongly separated
  logits, online rescaling, empty/tail splits, causal masking, populated prefix,
  segment/partition transitions and nonempty prefill/decode handoff. Retain
  existing bounds: synthetic FP32-reference outputs within 0.016 absolute;
  artifact mixer output within `0.02 + 0.002*abs(reference)`. Keep any stricter
  applicable existing tests. Do not apply isolated-component tolerances to
  whole-model states/logits. Run the frozen small precision screen because
  the selected decode MMA path changes probability rounding.
- Check same-schedule replay and relevant snapshot/reset/poison behavior,
  guarded scratch tails and no hot allocations. The production binder and
  FP8 output-projection caller must be exercised, not just a standalone kernel.
- Once per arm, compare complete attention preparation/core/merge/gating at
  a 32768-token populated prefix: M=32 prefill and M=1 decode, using valid
  capacity for appended queries and identical nonzero inputs/KV. Measure the
  attention region separately from unchanged projection work. Existing 4K
  correctness fixtures cover short lengths; no full-model 32K run here.
- Report selected registers/shared/local bytes, resident blocks and bounded
  split workspace from the actual compiled kernel. These diagnose resource
  use, not a mandatory instruction audit or proof of peak performance.
- Run one FAST-02 short integrated check and calculate final capacity impact.

Completion requires the new prefill and grouped decode paths selected in real
callers, all affected checks passing, lower observed complete attention cost
in both specified operations and no observed matched integrated phase
regression. Preserve working precision if a schedule fails; fix the concrete
cause within scope. Retaining only the previous schedule or parallelizing
the final store alone does not complete this task. A required path that
cannot qualify is a blocker, not a completed performance improvement.
TASK-040 owns final long-context quality and whole-request measurements.

### User-directed acceptance amendment — 2026-09-27

The user explicitly accepts the implemented candidate and directs that
TASK-038 be marked done after reviewing the attention gains and unresolved
short-prefill timing regression. This supersedes the no-observed-matched-
integrated-phase-regression completion requirement for this candidate.
The measured 247.119085 → 270.714379 ms prefill result remains a regression;
its cause is unresolved, and the diagnostic timing does not replace it.
All numerical, state, resource and development-quality requirements remain,
as do TASK-040's final quality, replay, capacity and performance gates.

## Completion report

TASK-038 is DONE under the explicit user acceptance amendment above. The
selected pipelined prefill and grouped decode implementation is retained,
with the passed checks and the 9.55% matched short-prefill regression recorded
below. The cause of that regression remains unresolved. The original reviewer
was unavailable, so a fresh GPT-6 Astra high-reasoning reviewer completed
review pass 2 and returned `PASS`; pass 1's `CHANGES_REQUIRED` finding was
solely the now-waived matched-prefill criterion, with no additional material
code defect or evidence request. See `astra-review-2.md` in the preserved
support archive.

The main-thread implementation used GPT-6 Codex. The reviewed implementation
and evidence are unchanged. Only task documentation and ledger bookkeeping
changed after review; no implementation changes or benchmark reruns were
needed. Documentation validation: `git diff --check`. TASK-039 is eligible
and remains TODO; final TASK-040 gates remain unchanged. Delivery bookkeeping
was reviewed by the main thread. GPT-6 Luna prepared the delivery;
completed-task caches `.cache/task037` and `.cache/task038` were removed after
preserving command evidence under `.cache/evaluation/qw38-language-v2/`.
Commit and push results are reported with delivery.

## Blocked execution record — 2026-09-27

Historical record before the user-directed acceptance amendment above.

Implemented the Q32/K64/D256 pipelined prefill and grouped six-head decode
candidate in the main thread (GPT-6 Codex), based on TASK-037 commit
`a719984c60f3ae8cbcaff61579d157c98539604e`. The candidate remains uncommitted
for inspection. Its one matched integrated prefill execution regressed, so
the explicit no-phase-regression completion requirement above is unmet.
No downstream task was activated and no acceptance criterion was amended.

| Required operation | Parent ms | Candidate ms | Result |
| --- | ---: | ---: | --- |
| Complete attention, M=32, prefix 32768 | 10.054 | 2.03597 | PASS |
| Complete attention, M=1, prefix 32768 | 0.593824 | 0.106208 | PASS |
| Matched 256-token prefill with final readout | 247.119085 | 270.714379 | FAIL: +9.55% |
| Matched eight fixed decode inputs with final readout | 169.045804 | 153.170365 | PASS |
| Complete short request | 416.164909 | 423.884774 | Regression |

These are single executions with zero warmups, including first use. The
matched arms share policy 1030, manifest
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`,
the authenticated TASK-027 256-token prompt and its first eight frozen
continuation inputs. Generated outputs do not select subsequent inputs.
No unchanged acceptance timing was repeated.

A targeted instrumented diagnostic found short-prefill attention GPU time
improved from 1.839010 to 1.240609 ms; MLP MMQ and FP8 GEMM costs were nearly
unchanged. Its favorable total (258.194287 → 232.380667 ms) is diagnostic
only and does not replace the failed matched acceptance result. The evidence
does not establish a concrete attention regression to repair. Repeatedly
sampling unchanged code for a favorable timing is not a resolution.

### Passed checks and implementation evidence

Independent asymmetric/strong-logit/causal/tail tests, all six GQA siblings,
masked padding, a fixed launch bucket with empty partitions, guarded scratch,
bitwise replay, 255/256/257 session transitions, and reset/restore/poison
checks passed. A 4095-prefix/3-row independent reference case passed with
zero maximum BF16 difference. Policy-1030 full-language integration passed
same-schedule state/logit replay and recovery. FP8 producer packing exactly
matches separate materialization, including padding.

Real-artifact mixer comparisons against the existing independent FP32
attention reference on identical prepared operands passed: prefill maximum
difference 0, decode 0.000205994, within the unchanged
`0.02 + 0.002*abs(reference)` bound. Both production phases exercised the
actual FP8 output projection with no hot allocations. The original Q4
artifact continuation/state fixture also passed without relaxed bounds.

The frozen eight-window TASK-035 precision screen passed over 1024 target
tokens: candidate NLL 1.7745875424981379, comparator 1.773838532533603,
delta **+0.0007490099645348636**, below +0.03. This is development screening,
not core-54 or production promotion.

The selected kernels use 128 threads and separately staged 64×264 BF16 K/V
buffers totaling 67,584 shared bytes. Q and running PV stay in registers;
score fragments convert to both BF16 probability components through shuffles.
Prefill uses 171 registers/thread; decode uses 96; both have zero local bytes
and one resident block per SM. Decode uses
`min(ceil(active_length/256), ceil(2*SM_count/4), 128)` partitions, also
bounded by capacity at binding. On the 170-SM RTX 5090 this is 85 partitions
at 32769 keys, using 2,105,280 partial bytes. Ranges divide active 64-key
tiles contiguously; empty ranges write neutral data, and merge uses increasing
partition order. Existing session allocation geometry remains unchanged.

Projected free VRAM at 32,768+128 capacity is 8,755,609,600 bytes, exceeding
the 2 GiB reserve by 6,608,125,952 bytes. This derives from the short request's
post-first-use free memory and additional KV bytes, not a measured final
capacity gate. TASK-040 retains final quality/replay/capacity obligations.

### Commands, failures and review

Exact commands, raw logs, source/binary identities, pinned upstream sources,
input identities, resource results, capacity calculation and diagnostics are
preserved in `.cache/evaluation/qw38-language-v2/task038-support/`, indexed by
`evidence-notes.md`. The entry commands were:

```sh
bash .cache/task038/baseline.sh
bash .cache/task038/check.sh
bash .cache/task038/validate.sh
bash .cache/task038/continue-validation.sh
bash .cache/task038/diagnose-artifact.sh
bash .cache/task038/finish-validation.sh
bash .cache/task038/diagnose-prefill.sh
git diff --check
```

`baseline.sh` and `check.sh` passed, including four focused CTest tests.
`validate.sh` passed the 4K and full-model checks but exposed two stale
fixture assumptions: missing FP8 engine scratch and use of a policy-1030
artifact for a policy-1029 mixed-layout binder test. `continue-validation.sh`
passed FP8 packing/consumer tests but exposed an inherited cross-phase
numerical mismatch. `diagnose-artifact.sh` reproduced that mismatch with
both parent and candidate attention. The first `finish-validation.sh` exposed
a reference-fixture lifetime mistake (Q captured after merge reused it as Y);
the corrected final execution passed the independent FP8 reference, original
Q4 continuation fixture, intended binder fixture, matched request execution
and quality screen. Successful execution of the matched request does not
mean its comparative prefill gate passed. All failure logs remain preserved.
`diagnose-prefill.sh` passed as a targeted diagnostic. `git diff --check` passed.

Independent **GPT-6 Astra, high reasoning, review pass 1** returned
`CHANGES_REQUIRED`: no additional material code defect, no further evidence
request, and the single blocking matched-prefill acceptance failure. Review
is preserved in `astra-review-1.md`. No second review was warranted without
a substantive correction or explicit change in authority. The task's
required-path blocker rule therefore applies. Luna delivery was not started;
no TASK-038 commit or push was performed. Task caches remain available.

**FOLLOW_UP_REQUIRED:** the pre-existing FP8-prefill/BF16-input-GEMV cache
rounding difference fails the legacy Q4/MLP cross-phase fixture on both parent
and candidate (mixer discrepancies 0.0312395 and 0.0313032). The new independent
same-operand oracle isolates attention, while the original Q4 and separate
whole-model state checks remain. This does not amend whole-model acceptance.

The original unblock requirement was an evidenced scoped correction or an
explicit acceptance amendment. The user-directed amendment above resolves
this blocker while preserving the failed timing and unresolved cause.
