# TASK-047 — Context-aware scheduling and prefill kernel efficiency

## Status

BLOCKED

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

## Blocked implementation report — 2026-09-27

The main thread (Codex/GPT-6) implemented and tested the candidate below from
TASK-046 revision `8399e4c68181f39e8c9132d1c78253a9444eaba0`. It is **not
accepted or delivered**: mandatory final-candidate cost gates fail. Preserve
the working tree and `.cache/task047/` for a concrete repair/replan; do not
activate TASK-048. TASK-043 remains production. No commit or push was made.
Independent Astra review (gpt-6-astra, high) returned **BLOCKED**, pass 1,
with the performance failures and the M1 benchmark-boundary finding below.
No additional runtime correctness defect was found. Delivery through Luna
has not started. Review: `task047-support/astra-review-pass1.md` in the
preserved evidence directory below. No second review pass was used.

### Implemented candidate and rejected experiments

`LanguageModelPlan::prefill_tokens` classifies the complete incoming span once,
or a validated optional logical total supplied consistently by partitioned
callers. It passes an immutable small/large regime to attention consumers;
prefix position remains separate. `state_replay` supplies the logical total
and reports selected head ownership and key tile. No session-latched regime,
persistent ABI, artifact, precision policy or arena-capacity change was added.
Decode explicitly selects the measured shared 256-key partition target for
both visible-length regimes. Existing power-of-two graph buckets distinguish
4096 and 4097 at unchanged session capacity.

The preserved candidate schedule is **experimental, not approved**:

| Consumer | Small (<=4096) | Large (>4096) |
| --- | --- | --- |
| Maximum prefill chunk | 512 | 512 |
| M512 at prefix zero | paired heads, Q32/K32 | paired heads, Q32/K32 |
| Later M512 | paired heads, Q32/K64 | per-head Q32/K64 |
| Smaller/tail chunks | per-head Q32/K64 | per-head Q32/K64 |
| Decode | six-head MMA, 256-key target, existing two-wave/device/capacity cap and fixed merge | same |
| MLP | unchanged parent J128/J32 and Q8 producer | same |

Saved TASK-043 timelines were inspected first. Targeted parent counters found
MMQ long-scoreboard stalls (48.1% of cycles between issued instructions) and
attention math-pipeline stalls (36.6%), without spills. These diagnostics are
not timing acceptance or proof of an attainable speedup.

- MMQ's implemented single-K32-group preread experiment kept affine staging
  and each accumulator's K32 order. Exact complete outputs and independent
  references passed, but complete MLP M256 changed 2.110240 → 2.148512 ms,
  and M512 3.037622 → 3.055276 ms. Full-tile registers rose 220 → 255;
  shared memory stayed 64 KiB, no spills. Rejected; `cuda/q4k_q8.cu` is
  identical to the parent.
- Per-head K64 M512 changed prefix-zero cost 0.642702 → 0.647080 ms
  (rejected for small prompts), and prefix-32256 cost 7.472821 → 6.987227 ms
  (selected for later large-prompt chunks). Component outputs were exact.
- The 128-key short decode experiment changed visible-4096 measured host
  cost 0.100642 → 0.124356 ms, with 16 → 32 partitions. Rejected; retain
  the parent's 256-key rule in both regimes. This provisional rejection is
  based on the partial boundary described in R047-5, not complete attention.
- Paired K32 reduced registers 179 → 132 and shared bytes 67584 → 33792,
  with no local bytes and still one resident CTA/SM. The first experiment
  changed prefix-zero cost 0.642702 → 0.631410 ms, but prefix-32256 cost
  7.472821 → 7.918407 ms. K32 was restricted to prefix zero. Its final
  combined-candidate prefix-zero observation regressed, as recorded below;
  the earlier isolated win does not override it.

### Failed acceptance gates

Every timing is one synchronized complete operation/request per arm, zero
warmups, with actual first use. No statistical significance or noise claim is
made. Causes of the regressions remain unresolved.

| ID | Mandatory observation | Parent ms | Final candidate ms | Change |
| --- | --- | ---: | ---: | ---: |
| R047-1 | M512 complete attention, prefix 0 | 0.642702 | 0.897225 | +39.6020% |
| R047-2 | 4096-token prefill | 1567.058900 | 1572.748819 | +0.3631% |
| R047-3 | 32768-token prefill, strict improvement required | 15274.162059 | 15333.359371 | +0.3876% |
| R047-4 | 32768-context eight-step decode | 142.655745 | 142.893687 | +0.1668% |

**R047-5 — Incomplete M1 evidence (Astra medium finding):**
`benchmarks/attention_bench.cpp:103` times preparation, scan and merge/gating,
but omits production output projection and `ResidualAddFp32`. Its results
and the comment in `cuda/attention.hpp:34` cannot establish complete-attention
cost or complete-cost rejection of K128. Before renewed acceptance, extend
the driver through the real projection/residual/completion boundary and run
the required parent/candidate M1 cases and K128 experiment. No additional
measurement is needed to establish the current BLOCKED outcome. The review
groups timing failures under its R047-1 and this evidence gap under its
R047-2; the task's stable IDs remain R047-1–5.

Final long-prefix complete attention improved 7.472821 → 7.274885 ms
(2.6487%). This cannot waive R047-1 or the integrated failures. Final M1
attention host costs were 0.100642 → 0.103156 ms at 4096 and 0.191763 →
0.195282 ms at 32768; the retained decode kernel/rule is unchanged.

| Prompt | Prefill parent → candidate ms | First decode ms | Remaining seven ms | Whole eight ms | Complete inference ms |
| --- | --- | --- | --- | --- | --- |
| 256 | 164.314613 → 161.818820 | 19.149520 → 18.890357 | 109.563961 → 109.625787 | 128.714362 → 128.517057 | 293.029005 → 290.335907 |
| 4096 | 1567.058900 → 1572.748819 | 19.462384 → 19.309232 | 112.630357 → 112.736139 | 132.093594 → 132.046193 | 1699.152524 → 1704.795082 |
| 32768 | 15274.162059 → 15333.359371 | 20.619845 → 20.615625 | 122.035078 → 122.277170 | 142.655745 → 142.893687 | 15416.817914 → 15476.253088 |

Cold setup (separately timed) was 5419.307785 → 5488.327187 ms,
5438.327789 → 5487.200926 ms and 5414.322271 → 5489.401689 ms respectively.
CUDA initialization and all raw timings remain in the request JSONL records.
All nine output IDs agree for each pair. Final logits are exact at 256;
4096/32768 logits change, without an arbitrary full-model equality tolerance.

### Passing checks and evidence

Five final Release tests pass: attention prefill, artifact prefill, core unit,
reference, and integration. The independent long-prefix reference passes.
The full-model integration passes logical 4095/4096/4097 internal versus
explicit M512 partition replay bitwise, populated-prefix M511/512/513,
visible decode 4095/4096/4097 graph/eager equality at capacity 4100,
reset/restore, graph/capture/poison/error cases and late failure after safe
chunk commits. Nsight launch correlations identify real paired K32, paired
K64 and per-head K64 consumers; the trace is used only for schedule evidence.
`comparison.json` contains the per-logical-prompt counts and kernel names.

The frozen TASK-035 eight-window NLL screen passes: 1024 targets, aggregate
delta `+0.00045418668087737757` against `+0.03`. Inputs and frozen comparator
scores are authenticated. The current `.qw38` manifest identity remains
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`;
no checkpoint or artifact payload/scale digest was computed.

Capacity 32896, first decode and maximum graph upload pass with
8,319,336,448 free bytes under the diagnostic trace, above the 2 GiB reserve.
Model bytes 21,013,686,400, arena 513,802,240, Q8 workspace 82,444,292 and
prefill workspace 254,248,960 are unchanged. Component hot allocations are
zero. Sampled free memory is not a continuous peak-memory claim.

All exact commands, logs, source/binary identities and preserved experiments
are in `.cache/evaluation/qw38-language-v2/task047-support/`:

- `bash .cache/evaluation/qw38-language-v2/task047-support/parent.sh` — exit 0;
  pinned parent build, all parent cases, requests and targeted counters.
- `bash .cache/evaluation/qw38-language-v2/task047-support/first-components.sh`
  — exit 0; MMQ and attention references, MMQ/per-head/decode experiments.
- `bash .cache/evaluation/qw38-language-v2/task047-support/k32-components.sh`
  — exit 0; K32 references including long prefix and isolated comparisons.
- `bash .cache/evaluation/qw38-language-v2/task047-support/candidate.sh` —
  final exit 0; focused tests, profiled state integration, final component and
  matched-request measurements, NLL. Two earlier compile failures from a
  missing/misplaced include were repaired before any final acceptance runs;
  `candidate-build-attempt1.log` and `candidate-build-attempt2.log` are retained.
- `uv run --no-project --python 3.12 python .cache/evaluation/qw38-language-v2/task047-support/analyze.py`
  — exit 0; authenticated results and launch-correlation analysis produce
  `comparison.json` with status BLOCKED and R047-1–4 observations.

Environment: pinned image
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`,
CUDA 13.4.1 / nvcc 13.4.59, GCC 14.2, CMake 3.28.3, RTX 5090 sm_120,
driver 590.48.01, configured 400 W power limit. Correctness/profiler runs are
separate from acceptance timings. No comparator or core-54 rerun occurred.

FOLLOW_UP_REQUIRED: resolve R047-1–5 through a concrete authorized repair or
replan. Do not infer that the earlier isolated gains, unchanged decode
arithmetic, or historical task exceptions waive these observations.
