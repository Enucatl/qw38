# TASK-023 — Production prefill projections and workspace

## Status

TODO

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Deliver conversion-inclusive, bounded prefill projections for the accepted candidate, with reusable activation operands and correct output precision.

## Depends on

- [TASK-022](TASK-022.md)

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
| Q-01/Q-02, projection part of P-02, A-01/L-01 | Use the accepted candidate representation and precision policy | TASK-020–022 selection |
| M-01, T-02, G-02 | Bounded workspace and a distinct measured prefill schedule | OVERALL-01 |
| P-01, S-01/S-02 | FP32 residual/accumulation and unchanged state controls | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-022 provides full candidate decode and a passing core quality gate. TASK-019 has real-shape GEMM and conversion measurements; TASK-021 defines the consumer layout.

## Scope

Integrate the selected native block-scaled GEMMs, or the measured fallback,
into bounded token-major prefill. Include activation scale/pack generation,
reuse across compatible projections, paired gate/up organization, proper
SwiGLU and residual epilogues, and requested-row vocabulary handling. Preserve
the precision contract of each consumer; BF16-output example kernels alone
do not satisfy FP32 residual/logit obligations.

Select chunk sizes and GEMM tiles from TASK-019 results rather than freezing
the old 256-token/32×64×64 defaults. Validate dynamic tails, padding masks,
workspace bounds, epilogue equivalence and per-layer projection error.
**Exit:** complete projection paths with end-to-end conversion-inclusive
timings, reusable workspace and correct generation/evaluation modes. Repeated
decode and full-model BF16 expansion are not production prefill paths.

## Out of scope

Full-model prefill/handoff, GDN recurrence and attention cores, whole-model BF16 expansion, unmeasured duplicate weights and repeated decode as production prefill.

## Required interfaces and data representation

Typed projection plans describe valid token count, absolute positions, family/shape, scales/layout, dispatch, output precision and epilogue. A preallocated token-major arena accounts for live activation/scale buffers, GEMM workspace and tails. Head interfaces distinguish final-position generation logits from requested evaluation rows in bounded tiles.

## Required semantics and constraints

Include scale reductions and activation packing in execution and reuse compatible quantized inputs. Preserve FP32 accumulation/residual/logit obligations and the declared rounding of ordinary staging. Paired gate/up performs the correct SiLU/product; padded rows must not influence outputs, scales or future state. Candidate BF16-output examples require appropriate epilogues before satisfying runtime contracts.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Choose chunks, tiles, buffering and dispatch from TASK-019 measurements within the frozen memory reserve. The former 256-token/32×64×64 geometry is not a requirement. Freeze comparison settings when measuring a candidate.

## Expected files/modules

Prefill projection CUDA consumers/wrappers, activation quantization/reuse, chunk/workspace plans, MLP prefill integration and projection/epilogue checks.

## Tests required

### Unit and contract checks

Valid token counts and tails around actual tile/chunk sizes, padding/scales, requested-row sets, buffer lifetime/overlap and precision-specific epilogues.

### Reference and numerical checks

Projection and complete MLP outputs against independent reconstructed-operand references and the accepted candidate decode control with declared tolerances; separate activation-policy effects from arithmetic errors.

### Integration checks

Complete GDN/attention projection sets and MLP prefill without their recurrence/attention cores. Exercise generation/evaluation head modes, bounded workspace and stable allocation behavior.

## Benchmark required

Required for each selected family/shape/chunk: complete quantize/pack/GEMM/epilogue time, activation reuse, resources and peak workspace, plus kernel-only attribution. Confirm native Tensor Core execution or explicitly identify the measured fallback.

## Acceptance criteria

- [ ] All selected projection families, MLP and head modes consume the candidate representation correctly.
- [ ] Activation scaling/packing/reuse and required output precision are integrated and included in timings.
- [ ] Tails, masks, epilogues and requested rows satisfy independent numerical checks.
- [ ] Workspace is bounded, preallocated and accounts for scale buffers and library requirements.
- [ ] Chunk/tile/dispatch choices have conversion-inclusive measurements and fit the frozen memory budget.

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
