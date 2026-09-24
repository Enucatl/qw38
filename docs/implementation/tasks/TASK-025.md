# TASK-025 — Causal attention prefill and layer integration

## Status

TODO

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Integrate causal attention prefill around the accepted projection path, with correct cache/position behavior and measured SM120 resource use.

## Depends on

- [TASK-024](TASK-024.md)

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
| G-02, T-03, M-01 | Distinct tiled attention schedule with measured geometry and bounded workspace | OVERALL-01 |
| P-01/P-02 | FP32 softmax/reductions and BF16 KV, with accepted projection operand policy | Retained controls and selected projection policy |
| G-01 | Model-native gated attention, GQA, QK norm and RoPE semantics | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-024 has completed GDN layer integration; TASK-023 supplies projection/MLP prefill. Decode attention preparation, cache and numerical controls exist.

## Scope

Integrate q/g split, QK norm, partial RoPE, BF16 KV append, tiled causal GQA,
gate and output projection with the new projection paths. Respect 24 query
heads, four KV heads and head width 256. Reuse KV without materializing a
quadratic score matrix or six persistent GQA copies. Select tiles on SM120
from resources and measurements, preserving FP32 softmax/reductions.

Validate causal masking, existing cache prefixes, absolute positions, tail
tiles, capacity/population distinctions and complete attention+MLP layers.
**Exit:** numerical/continuation checks and short/long-context attention timing
with actual scratch/KV memory, ready for full-model prefill.

## Out of scope

Quantized KV, paging/eviction semantics, quadratic global score tensors, six persistent GQA KV copies, full-model acceptance and unrelated precision changes.

## Required interfaces and data representation

An attention prefill plan declares valid query rows, absolute positions, cache capacity/population, BF16 KV views, tile geometry and bounded workspace. Projection preparation preserves per-head q/g binding, QK normalization and partial RoPE. Compose gating/output residual with the complete attention+MLP layer.

## Required semantics and constraints

Preserve 24 query heads, four KV heads and width 256 with correct GQA mapping. Attend only causal populated entries, including existing prefixes; cache writes and position advances cover valid tokens only. Use FP32 online softmax/reductions and accumulators, with bounded staging and no quadratic score materialization.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Choose tiles using SM120 occupancy/shared-memory/register measurements at short and long contexts. The old 32-query×64-key tile is a control, not a required optimum.

## Expected files/modules

Attention prefill preparation/cache append, tiled causal attention and gating, complete attention+MLP layer integration, numerical/cache tests and profiling records.

## Tests required

### Unit and contract checks

q/g split and GQA mapping, partial RoPE positions, causal/tail masks, capacity versus populated length and workspace bounds.

### Reference and numerical checks

Independent small causal attention and gating controls, stable softmax stress cases, accepted projection-policy numerical checks and existing cache-prefix references.

### Integration checks

Complete attention+MLP with empty/nonempty caches, irregular chunks, short tails, reset/replay and prefill-to-decode continuation. Verify no padded cache append or position advance.

## Benchmark required

Required: short/long-context attention and full attention+MLP timing, KV traffic/reuse, resources and measured scratch/cache bytes.

## Acceptance criteria

- [ ] Preparation, QK norm, RoPE, GQA, gating and output residual preserve model semantics.
- [ ] Causal masking and cache/position updates pass empty/prefixed/tail/capacity checks.
- [ ] Numerical and continuation checks pass for the complete attention+MLP layer.
- [ ] No quadratic global score matrix or persistent replicated KV is required.
- [ ] Measured SM120 tile/resource choices and bounded scratch/KV usage are recorded.

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
