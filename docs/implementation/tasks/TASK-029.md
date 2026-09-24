# TASK-029 — Scheduling, dispatch and fusion refinement

## Status

TODO

## Milestone

M10 — Measured refinement and promotion

## Purpose

Reduce measured scheduling, conversion, fusion and launch costs in both phases without changing the selected weights.

## Depends on

- [TASK-028](TASK-028.md)

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
| M-01, T-01–03, G-02 | Tune chunks/crossovers, scratch boundaries and scheduling | OVERALL-01 |
| Projection part of P-02 | Preserve selected activation policy; track fusion/rounding effects | Candidate control |
| Q-01/Q-02, S-01/S-02, P-01 | Fixed weights/state and retained numerical semantics | Experimental control |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-028 records accepted precision/representation choices and remaining whole-request gaps.

## Scope

Address measured chunk-size, small-M crossover, activation reuse, normalization
plus quantization, epilogue, launch and synchronization costs. Consider CUDA
graphs only if launch overhead warrants them and stable-address/state/error
contracts remain valid. Include redundant normalization, occupancy, scale
generation and workspace costs when deciding fusion. Keep weights fixed.

**Exit:** justified scheduling/fusion decisions and verified full-request gains
or a measured keep decision. Rerun affected numerical/continuation gates and
full behavioral gates for arithmetic changes; replay covers graph/dispatch
boundaries and failure recovery. Optimize both prefill and populated decode.

## Out of scope

Weight requantization, unrelated state changes, mandatory CUDA graph adoption, fusion without complete-path measurement and bypassing commit/error semantics.

## Required interfaces and data representation

Version execution-plan identity with chunk/dispatch thresholds, quantization reuse, fusion and graph mode. Typed scratch lifetimes and any stable-address graph buffers remain bounded. Runtime errors preserve complete-token commit, poisoning, reset and restore contracts.

## Required semantics and constraints

Measure normalization recomputation, scaling/packing, epilogues and saved materialization together. Keep selected weights fixed and record any changed activation rounding. Test dispatch/chunk thresholds and ensure padded work does not alter scale/state semantics. CUDA graphs, if warranted, must respect session storage lifetime, input updates and failure recovery.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Address measured gaps in priority order. Compare chunk sizes and small-M crossovers at fixed policies before combining changes; use identical workloads and output semantics. A measured keep decision is sufficient where tuning has no useful effect.

## Expected files/modules

Affected runtime plans/dispatch, normalization/quantization/epilogue kernels, scratch ownership and optional graph launch path; focused regressions and performance report.

## Tests required

### Unit and contract checks

Crossover/tail selection, scratch aliases/lifetimes, scale reuse, stable graph bindings if introduced and error/commit propagation.

### Reference and numerical checks

Fused versus separate numerical paths including rounding and repeated normalization; retain precision-specific epilogue and projection checks.

### Integration checks

Continuation, snapshot/reset/interleave and failure recovery across selected chunk/dispatch/graph boundaries. Arithmetic changes require the full applicable EVAL-01 behavioral gates before promotion.

## Benchmark required

Complete conversion-inclusive projection/layer costs and matched prefill, populated decode and requests. Include launch/wait time, workspace, occupancy and redundant arithmetic; validate combined changes against the accepted control.

## Acceptance criteria

- [ ] Changes or keep decisions address measured scheduling/fusion/launch costs in both execution phases.
- [ ] Weights and unrelated state contracts remain fixed and numerical effects are explicit.
- [ ] Boundary, memory-lifetime and failure/replay checks pass, including graph mode if added.
- [ ] Arithmetic changes pass full applicable behavioral/context gates.
- [ ] Complete-request measurements justify decisions with workspace/resource costs and remaining regressions reported.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

TODO — no execution or acceptance evidence recorded for this revised task.

### Changes made

Record the concrete changes or measured keep decision, including decision and artifact/layout identities.

### Tests run

Record exact commands, outcomes, reference tolerances, covered boundaries and
limits. Do not infer runtime correctness from documentation checks.

### Benchmark results

Record raw evidence paths, timing boundaries, quality context, memory and
uncertainty, or the reason a benchmark is not required by this task.

### Architecture blocker

Record none or the complete ledger-defined blocker report.

### Follow-up observations

Record remaining coverage and performance gaps and their downstream owners.
