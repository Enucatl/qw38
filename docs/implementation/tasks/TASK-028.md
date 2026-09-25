# TASK-028 — Conversion-minimized native FP4 path

## Status

TODO

## Milestone

M10 — Measured refinement and promotion

## Purpose

Test whether native NVFP4 or MXFP4 computation can improve complete-request
speed without losing the selected model's quality or exceeding its memory
budget. Compare a real GPU conversion path with the accepted Q4_K/Q8 runtime;
the TASK-019 CPU reference packer is not a production performance result.

## Depends on

- [TASK-027](TASK-027.md)

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| Q-01/Q-02, projection part of P-02 | Test FP4 weight and activation precision against the accepted candidate | FP4-01 / OVERALL-01 |
| A-01/A-02/L-01 | Version any new packed view and account for its complete memory cost | FP4-01 / OVERALL-01 |
| P-01, S-01/S-02 | Preserve FP32 arithmetic/state and unrelated semantics | Retained control |

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, FP4-01 and the task contract,
  quality and performance decision rules.
- [TASK-019](TASK-019.md) — demonstrated native SM120 instructions and the limits
  of reference conversion timings.
- [TASK-020](TASK-020.md) — FP4 family/activation error screening.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) and the
  [54-case core amendment](../../architecture/evaluation-policy-core-54.md).
- [Technology baseline](../technology-baseline.md) and
  [code standards](../code-standards.md).

## Starting point

TASK-027 provides an accepted Q4_K/Q8 full-request baseline and ranked costs.
TASK-019 established native NVFP4/MXFP4 support, but its FP4 activation
packing ran on the CPU. TASK-020 found quality losses in several FP4 families.

## Scope

Build a bounded, executable native FP4 variant for at least one real projection
family chosen from TASK-020 quality evidence and TASK-027 bottlenecks. Evaluate
NVFP4 and MXFP4 for that family; document any format ruled out by its measured
quality, memory or conversion cost. Quantize weights once from the pinned BF16
source into a versioned resident native format. Do not repack full weights in
the request hot path or silently retain duplicate weight views.

Generate FP4 activation values and block scales on the GPU. Reuse a packed
activation across compatible projections in the same decode step or prefill
chunk where its scale and rounding contract permits. Compare separate conversion
with a fused producer
where a fused path is valid; count scale reductions, packing, launches, staging,
padding, native MMA, epilogues and any BF16/GEMV decode fallback. Avoid CPU
conversion in timed requests. Test native prefill and native small-M decode
against a BF16-activation GEMV using the **same FP4 weight bytes**. Choose
dispatch from conversion-inclusive measurements, including the case where GEMV
is faster for one-token decode.

## Out of scope

CPU packing as a production path, synthetic-kernel speed as promotion evidence,
hot-path full-weight repacking, unbudgeted duplicate resident weights and
changes to unrelated state or scheduling policy.

## Required interfaces and data representation

Record the candidate's quantizer, scale, physical layout and activation policy
identities. A production binding must identify the resident view, native and
GEMV consumers, scratch lifetimes and output precision. Any retained extra view
has an explicit size and load lifetime.

## Tests and measurements required

- Independent reconstruction and contraction checks for FP4 weights and
  activations, including scales, tails, zero blocks and output precision;
  instruction evidence for the native path on the project RTX 5090.
- Weight-only, activation-only and joint error measurements for the selected
  family. Preserve FP32 residual/state/logit semantics and session replay.
- Per-stage GPU timing and matched prefill, populated-decode and complete-request
  timing against the same Q4_K/Q8 baseline. Report cold weight preparation and
  upload separately from steady-state conversion. Do not compare the CPU
  reference packer with a production GPU kernel as if they were equivalent.
- Artifact bytes, resident and peak transient memory, scale buffers,
  workspace, cold-load time and any extra view costs within the capacity budget.
- Complete applicable 54-case EVAL-01, selected-P100, long-context and
  continuation checks before any FP4 variant becomes the selected policy.
  The optional 216-case suite remains human-initiated only.

## Acceptance criteria

- [ ] At least one real model projection family runs a native NVFP4 or MXFP4
  path with GPU activation conversion in both prefill and populated decode;
  both formats receive an evidence-backed disposition.
- [ ] Same-FP4-weight native/GEMV dispatch and reuse/fusion choices have
  conversion-inclusive measurements, numerical checks and explicit identities.
- [ ] Cold and steady-state time, artifact/resident/peak memory and full-request
  effects are compared fairly against the accepted Q4_K/Q8 control.
- [ ] Any promoted variant passes the applicable quality, long-context and
  replay gates; rejected variants retain measured reasons.
- [ ] The FP4 keep/change decision and remaining conversion bottlenecks are
  recorded for TASK-029.

## Architecture blocker rule

A rejected FP4 candidate is a result and leaves the accepted Q4_K/Q8 control
in place. Missing required comparison evidence prevents completion. A conflict
outside the reopened decisions requires the ledger's full architecture-blocker
report. A native kernel speedup alone does not justify promotion.

## Completion report

### Result

TODO — no experiment or acceptance evidence recorded yet.
