# Performance recovery after OPT-069

Status: proposed implementation batch OPT-070 through OPT-080. Source review
at Quartz `ba071925e6864d354c773db6bba863ba8b1807d3`, 2026-09-11. No new GPU
sitting, kernel implementation, numerical admission, or production keep is
claimed by this plan. Authority remains llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Authenticate the artifact rather than infer identity from its historical
`Qwen3.8-27B-Q4_K_M.gguf` filename.

Read [the shared testing strategy](TASK-TESTING-STRATEGY.md) and
[the preceding batch protocol](PERFORMANCE-RECOVERY-2026-09-11.md).
The sections below specify this batch's decisions and evidence requirements;
they do not lower OPT-056 or OPT-016.

## Diagnosis and measurement limits

OPT-069's measured fixture and dossier are the completed sitting. Its current
`REPORT.md` still says unmeasured; OPT-070 must repair that reporting mismatch
from authenticated retained data, without rerunning the sitting just to write
Markdown.

Equivalent latency calculated from OPT-069 mean throughput:

| Workload | Quartz ms | Same-sitting llama ms | Parity saving needed, Tq - Tl | Saving for +5% throughput, Tq - Tl/1.05 |
|---|---:|---:|---:|---:|
| P4096, whole prompt | 1405.31 | 1303.41 | 101.90 | 163.96 |
| D128, per token | 26.90 | 14.52 | 12.38 | 13.07 |
| D2048, per token | 28.21 | 14.85 | 13.36 | 14.07 |

These reciprocals of mean throughput are equivalent latencies, not recomputed
means of the raw latency distribution. Decode p95 is independently gated.
Use same-sitting llama P=3142.52, not historical OPT-056 llama P=3263.52.
The 2K point lead of 0.83 tok/s is too small to describe as a robust speedup.

1. OPT-066 changes prompt MMQ, so there is no direct decode saving. Its
   layer-0 complete FFN screen saves 0.745 ms; multiplying by 64 predicts
   about 47.7 ms/prompt only if representative. OPT-056 to OPT-069 P improves
   by about 53.1 ms/prompt. That is consistent in scale, not causal attribution.
2. OPT-064's 0.667 ms/token mixer estimate would predict only about 2.4%
   decode improvement at the old D2048 latency. It cannot close a 13–14 ms gap.
   `cuda/opt064_q8_rows_test.cu::run_screen` uses synthetic activations, three
   selected layers per group and group-by-group synchronization, extrapolates
   frequencies, and omits attention output projection. This is not the complete
   64-layer production mixer or OPT-061 capture replay. Its hot estimate even
   favors the old layout. Fresh whole-engine A/B must distinguish cache/scope
   error, other changes, hardware variation, and an actual regression.
3. OPT-064 and OPT-066 are installed keeps, not numerical rejections. Both
   iteration contracts' acceptance mode runs smoke/correctness/screen. OPT-064
   hard-codes 1+3 samples even for its native acceptance entry; OPT-066 keeps
   from layer-0 means. The stipulated ten paired rounds and short E2E guard
   were not established. A successful runner exit is not production admission.
4. OPT-060 is a suspect ranking, not a valid matched family gap. Llama calls
   `begin_run()` before warmup/prefix and only resets the epoch before measured
   work; warmup/prefix records contaminate the table and fill its 8192 slots.
   Quartz epochs restart per token and nested scopes remain double-counted.
   Eager events also differ from shipping FFN-graph execution. OPT-061 replay
   synchronizes per layer, uses layer-0 input across the weight walk, and its
   capture key omits source/selector identity. Repair these boundaries first.
5. OPT-062/063 are admission-blocked candidates, not performance losers.
   Screens show packed complete FFN about 15.8 ms, integer unfused about
   12.3–12.5 ms, paired integer about 11.27 ms. None establishes a production
   keep. Even a transferable 4.55 ms saving only projects about 42.3 tok/s
   from OPT-069 D2048; more work would still be needed.

Sources: [OPT-059](../evidence/optimization/opt059-gpu-numerics/REPORT.md),
[OPT-060](../evidence/optimization/opt060-engine-attribution/REPORT.md),
[OPT-061](../evidence/optimization/opt061-component-replay/REPORT.md),
[OPT-062](../evidence/optimization/opt062-q4-admission/REPORT.md),
[OPT-063](../evidence/optimization/opt063-integer-ffn/REPORT.md),
[OPT-064](../evidence/optimization/opt064-q8-rows/REPORT.md),
[OPT-066](../evidence/optimization/opt066-mmq-x-pipeline/REPORT.md),
and [OPT-069](OPT-069.md).

## Arithmetic concessions and hard boundaries

The user authorizes reassessing correctness-based rejections with documented
accuracy concessions comparable to llama/ds4. A looser assertion alone does
not accelerate code. It can admit an existing faster kernel, or allow a
different reduction, FMA or staging implementation. Measure that implementation.

Use OPT-059's already specified versioned policy first: abs/RMS ceiling
`max(strict_ceiling, 1.25 * measured_llama_gpu_error + 1e-6)` and its cosine
rule. Freeze reference-derived budgets before candidate results; retain old
envelopes as reported historical diagnostics. Do not infer a ds4 allowance
from a comment or copy an unrelated tolerance. A further policy change needs
an explicit version, measured authority comparison, and rationale; it is not
automatic when 1.25 fails.

Primitive layout, signedness, bounds, finite outputs, correct sum semantics,
full vocabulary, causal state, checkpoint/cancellation and same-path replay
remain hard requirements. Q8_1 half(sum(q)) is not half(sum(x)); Q4 MMV in
pinned llama recomputes integer sums rather than consuming that stored field.
Test each typed staging/consumer pair. Never loosen a tolerance to admit stale
staging, integer overflow, an alias error, NaNs or a wrong layout.

Retain model-level PPL ratio <=1.01 on both 1024-target spans and recurrence
incremental NLL <=0.02. OPT-073 distinguishes absolute task accuracy from
engine regression when both authorities answer arithmetic incorrectly. Neither
agreeing on a wrong answer nor changing a prompt makes old OPT-056 quality pass.

## Common implementation acceptance

Each new task creates `pins/optNNN_iteration_contract.json`, a focused task
contract, host evidence tests, and one owned fixture/report with raw sidecars.
Commands in the dossiers are interfaces to implement, not commands available
before task execution. Reuse OPT-057 and extend existing diagnostics where
possible. Normal pytest stays read-only and GPU-free.

- Feedback includes incremental build and all children within 300 seconds.
  A named phase selects a predeclared bounded workload, not an escape hatch
  for automatically relaunching a timed-out sweep. Record actual native counts,
  epoch start/end, monotonic duration, cache/load/reference work and failure.
- Screen at most control plus two named candidates, one warmup + three rounds,
  unless a dossier narrows it. Only one survivor gets acceptance. Real captures
  and a distinct-weight walk exceeding 2x queried L2 are required; record layer
  order and bytes. Host references use at most 16 sampled rows x 4 activations
  per family/case, full K, decode each row once. Small boundary cases are full.
- Component acceptance is 3 warmups + 10 independently restored AB/BA rounds.
  One timed round covers the entire declared layer set; no per-layer host
  synchronization. Require a positive two-sided 95% paired Student-t interval
  for control minus candidate (df=9, critical 2.262) and >=0.10 ms/token or
  >=5 ms/P4096 estimated saving using authenticated frequencies.
- Target E2E acceptance is five uninstrumented AB/BA whole-run pairs, each
  D2048+32 predetermined tokens or one P4096. Require the one-sided 95% upper
  bound on paired latency regression <=2% of mean control (df=4, 2.132).
  Measure one short pair on each other workload with the existing 95%
  throughput floor / 105% observed p95 guard. Short p95 is diagnostic only.
- State/reset and graph capture are separate per configuration. Both sides
  return full logits/hidden and perform the same commit/copy work. A host
  selector override must take effect in the linked production translation unit;
  prove actual eager/capture launches, not just header constants.
- New arithmetic passes calibrated primitive checks and full model quality
  before production promotion. Thirty-two teacher-forced targets are an alarm,
  not admission. Reuse immutable authority data; run full quality once per
  surviving arithmetic configuration, not per edit. Partition that work into
  named phases with complete provenance. Do not reuse candidate outputs across
  changed source, flags, inputs, state or selectors.
- Record separate `screened_in`, `component_accepted`, `production_kept`,
  `quality_blocked`, `performance_rejected`, `inconclusive`, `release_passed`.
  All scheduled phases must finish for their corresponding verdict. Ten
  samples cannot be relabeled as the historical 30-run protocol. Inconclusive
  means retain the control; never keep sampling until significance.

Keep power/clock policy unchanged (last observed cap 400 W); record actual
readback and temperatures on both engines. Optional counters cover one selected
steady-state launch, queried metric names, no `--set full`. No GPU concurrency.

## Priorities and no-repeat decisions

First OPT-071 attribution/replay repair and OPT-072 rejection inventory; prepare
OPT-073/074 reference work independently. OPT-070 then revalidates current keeps
on repaired replay. OPT-075 decides the existing Q4 paths before OPT-076 rewrites
their inner loop. Next choose OPT-077/078 by updated removable decode time.
OPT-079 is the single lower-priority prompt experiment, conditional on remaining
attention time and redundant conversion instructions. OPT-080 is one combined
long validation after cheap quality checks pass, not a long run per candidate.

Do not repeat unchanged OPT-067 prompt pairing, OPT-065/037 tile sweeps,
OPT-068 O3/FMA flags, OPT-024 repacking, OPT-028 stream-K, or OPT-054 batches
merely because the outcome gate failed. OPT-068 passed primitive numerics and
lost on complete cost: relaxing correctness cannot rescue that measurement.
OPT-029 prompt fusion is not evidence that the still-sequential decode GDN
implementation is optimal. Historical correctness rejections all receive an
explicit disposition in OPT-072, including variants inside otherwise kept tasks.

Broader graph capture remains conditional on fresh unhidden idle/launch time:
OPT-055 found about 0.1 ms/token. Add a separate concrete graph task only if
OPT-071 finds >=0.5 ms/token of removable overhead with state semantics intact.
Do not assume every `cudaDeviceSynchronize` runs in the benchmark: polling and
trace synchronizations are conditional. No unconditional Q6 logits retuning,
weight requantization, speculative decoding or vocabulary reduction in this batch.

Specialization makes these choices possible; it does not itself outperform the
specialized CUDA dispatches inside a generic engine. Keep/reject is measured.
