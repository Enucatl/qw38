# TASK-047 — Context-aware scheduling and prefill kernel efficiency

## Status

TODO

## Milestone and dependency

M26 — Implement the agreed context regimes and improve prefill. Depends on
TASK-046. [FAST-04](../task_ledger.md#optimization-amendment--fast-04-2026-09-27)
governs the batch. Start from TASK-046's selected candidate and keep its
decode projection improvements.

## Evidence and delivered behavior

Today every prompt uses up to 512-token chunks; a full chunk always selects
paired-head attention. This is not full-prompt context dispatch. Decode uses
one grouped attention algorithm with length-dependent partitions and graph
buckets. TASK-043's 32K prefill spends 9.667 s in projections (8.018 s J128
MMQ) and 3.764 s in attention. J128 uses 220 registers/thread and 64 KiB shared
memory; paired attention uses 179 registers/thread and 67,584 shared bytes.
Neither resource count alone identifies a performance bottleneck.

Implement explicit small (<=4096) and large (>4096) selection, chosen once
from full prompt length for prefill and from visible populated length for
decode. Exactly 4096 is small. Preserve a maximum 512-token execution chunk
and existing smaller/tail support. Measure concrete schedule choices and
retain an improved prefill consumer, not a dispatcher that only changes labels.

## Implementation guide: selection and state

1. Trace `LanguageModelPlan::prefill_tokens`, prefill layer binders,
   `cuda/q4k_q8.cu` and `launch_attention_prefill_scan`. Carry a small immutable
   per-call scheduling value from the prompt boundary to consumers, rather
   than inferring the regime from `valid_tokens` inside each chunk.
   For an ordinary public call, full prompt length means the complete input
   span before internal chunking. For explicit partition/replay of one
   logical prompt, the caller must supply that logical prompt's total length
   consistently; use a narrowly scoped optional argument if needed. A new
   continuation prefill call is a new prompt span. Absolute prefix position
   remains a separate attention input, never a substitute for prompt length.
   Validate a supplied total against the submitted span and capacity; it is
   an incoming-token count, not `current_position + submitted_count`. No
   persistent ABI change or unsnapshotted session-latched regime is needed.
   Update `src/runtime/state_replay.cpp` and other partitioned callers to pass the
   logical total and report actual selected kernels/head ownership; replace
   existing labels inferred solely from `count == 512`.
2. Start with M512 capacity in both regimes. For small-regime full chunks,
   evaluate existing per-head attention against the paired path; for large
   prompts retain paired heads as the starting point. Actual prefix length
   may select a measured short-prefix specialization within either regime.
   Do not revert all <=4096 prompts to M256: the existing 4K M512 chain already
   improved matched prefill by 19%. Retain existing smaller/tail dispatch.
3. Decode retains six-head grouped MMA and deterministic partition merge.
   Evaluate a bounded short-context split rule with a 128-key target against
   the current 256-key target; retain the current long-context target and
   two-wave/device/capacity cap initially. These are tuning choices, not new
   precision. Record the selected rule per regime; identical choices may
   remain where the experiment shows no benefit, with an explicit rejection.
   Graph validity must include every schedule parameter that changes captured
   launches. A transition across 4096 must select/rebuild the proper graph
   even when a capacity bucket alone would not distinguish it. Active extents
   and neutral empty partitions remain separate from launch capacity.
   The present power-of-two buckets already separate this pure boundary;
   proving the existing key sufficient is preferable to redundant key fields.
   Position 4095 exposes 4096 keys (small); position 4096 exposes 4097 (large).
   Launch grid, device partition calculation, partial indexing and merge must
   agree under every retained rule and graph replay.

## Implementation guide: bounded kernel work

Inspect existing Nsight timelines first. Take targeted counter diagnostics
only for missing questions about occupancy, memory transactions, barrier or
instruction stalls; isolate representative MMQ and long-prefix attention
shapes. Counter replay is not performance acceptance. Select one justified
first candidate per family, not a Cartesian tile sweep.

- MMQ: investigate reducing live preread fragments/metadata in J128 before
  changing its tile. Preserve cooperative affine staging and weight reuse;
  consider a smaller token tile only for demonstrably underfilled short/tail
  shapes. Existing M256 down launches only 80 blocks. Explicitly measure the
  tradeoff between reuse, registers and enough blocks. Do not simply double
  buffer the existing 64 KiB shared allocation or add unmeasured async staging.
- Attention: retain the existing real asynchronous K/V pipeline. Investigate
  register ownership or a bounded alternate Q/K tile when counters identify
  exposed stalls. Current paired M512 has 192 blocks on 170 SMs; sharing all
  six heads at unchanged Q32 would leave only 64 blocks. More grouping is
  not automatically better. Keep two-component BF16 probabilities and FP32
  statistics/PV. New long-prefill key splitting, its extra merge and workspace
  are outside this bounded first round; record a follow-up if evidence makes
  that necessary rather than expanding the task.

Retain winners independently; a rejected family can keep its parent consumer
if its diagnostic, implemented candidate and comparison are recorded. At least
one kernel family must improve, in addition to explicit regime plumbing and
the measured schedule decisions. No new quantizer, artifact layout, GDN
algorithm, giant fusion, prefill graph or larger-than-512 arena.

## Verification and completion

- Preserve policy 1030's K32 affine equation/producer packs, FP32 residuals
  and recurrence, two-component BF16 P, BF16 KV/history and all state/stream
  lifetimes. Use existing independent MMQ/attention references and tolerances.
  Extend focused tests for selected tiles, tails, asymmetric GQA siblings,
  causal populated prefixes, empty partitions and scratch guards.
- Prove full-prompt selection at 4095/4096/4097 independent of internal chunk
  and explicit replay partition; cover M511/512/513 with a populated prefix.
  Exercise visible decode lengths 4095/4096/4097, unchanged-capacity graph
  transitions, actual graph/eager equality, reset/restore across regimes and
  late failure without premature host commit. Record kernel/partition choices,
  not just output agreement. Same-schedule replay is bitwise; different
  schedules use declared numerical checks, not arbitrary full-model tolerances.
- Once per arm, measure complete MLP at M256/M512 and complete attention
  at M512 with prefixes 0 and 32256, plus M1 attention at visible lengths
  4096 and 32768. Include packs, preparation, merge/gating, epilogues and
  completion. Reuse unchanged component evidence only with authenticated
  code/inputs/timing. Every retained changed case must not regress; at least
  one targeted prefill component must improve. Report resources and actual
  workspace, including graph transitions and the 2 GiB reserve projection.
- Run matched development requests once per arm at prompt lengths 256,
  4096 and 32768, each followed by eight fixed continuation inputs. Use the
  frozen TASK-027 token files and their first eight continuation inputs;
  minimally extend the existing development driver for 32768. Include first
  use and synchronized final logits. These are three bounded comparisons,
  not a new 128-token PERF-01 suite. Require no prefill/decode phase regression
  at any length and strict prefill improvement at 32768. The 4095/4097 checks
  above are correctness/dispatch checks, not extra performance rows.
- Run the frozen eight-window NLL screen when changed arithmetic/reduction
  order affects outputs. No core-54 or comparator rerun here.

DONE requires implemented and tested regime propagation, explicit measured
decisions for prefill/decode schedules, a retained prefill kernel improvement,
passing affected numerical/state/quality checks and the integrated results
above. A no-op dispatcher, rejection of all kernel changes, unresolved
required regression or insufficient capacity is BLOCKED pending repair/replan.
Publish the final small/large schedule table for TASK-048, including any
shared kernel choices justified by measurement.
