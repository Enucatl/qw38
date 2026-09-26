# TASK-026 — Full-model prefill, handoff and quality gate

## Status

DONE

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Validate the complete quantized engine across production prefill, decode transitions and the full required quality/context envelope.

## Depends on

- [TASK-025](TASK-025.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — scoring
  criteria and measurement definitions.
- [54-case core amendment](../../architecture/evaluation-policy-core-54.md) —
  routine coverage and manual-only full-suite execution.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| G-01/G-02, M-01 | Compose all language layers with bounded chunks and correct state handoff | Retained semantics and OVERALL-01 schedule |
| Q-01/Q-02/P-02/L-01 | Accepted weights and explicit per-phase activation/dispatch policy | Candidate selection |
| EVAL-01 | 54-case core production-prefill rerun and one fixed 32768 retrieval case | Policy, coverage and ownership remapped |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-022 candidate decode core passes. TASK-023–025 supply bounded projections and both complete layer types. Long-context and full-model production-prefill quality are not yet accepted.

## Scope

Compose all 64 layers, bounded chunks and final-position logits for generation;
provide requested-row logits for evaluation without allocating T×vocabulary
for the entire prompt. Test prefill into nonempty sessions and continuation
across native GEMM/GEMV dispatch transitions with the same resident weights.

Run the complete 54-case EVAL-01 core through production prefill plus decode, and the
fixed `R-32768-s0-d0.1` case in both required comparison arms. Preserve
the original boundaries/partitions (including 1/63/64/65/255/256/257 and
alternating 63/65), adding boundaries around chosen chunks and dispatch
crossovers. Require bitwise replay only for identical schedules; compare
different schedules using component tolerances and the unchanged behavioral
gates. Test chunk-dependent activation scales when the selected path uses them;
record their absence for the accepted BF16-activation path. **Exit:** accepted
full candidate quality, correct positions/history/state, bounded measured
memory, and documented coverage. Missing mandatory long-context evidence
blocks acceptance; full prefill equivalence is not inferred from one token.
Historical six-case 32768 runs remain evidence. Only the fixed case may be
run on the final binary under EVAL-01.

## Out of scope

Lowering quality thresholds, substituting a subset of the frozen 54-case core, inferring acceptance from one-token agreement, MTP/vision/batching claims and total T×vocabulary allocation.

## Required interfaces and data representation

The full-model prefill API supports valid token chunks, absolute positions, empty/nonempty sessions and final-row or bounded requested-row logits. Reports bind artifact, calibration, per-phase precision/scales, chunk/dispatch schedule and reference/input identities. Reuse session commit, poison/reset and snapshot contracts.

## Required semantics and constraints

Run all 64 layers and preserve FP32 residual/logit and state contracts. Identical schedules require bitwise replay; distinct schedules use existing component tolerances and unchanged EVAL-01 gates. Explicitly test W4A4 activation changes and chunk-dependent scales when selected; record BF16 activation and GEMV handoff when the accepted path has no activation scales. Never advance positions/history for padded rows or score a teacher target twice.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Retain EVAL-01's nine checkpoints 1, 3, 4, 63, 64, 65, 255, 256, 257 and partition sizes 1, 63, 64, 65, 255, 256 plus alternating 63/65. Add boundaries around selected chunks, recurrence intervals and dispatch crossovers.

## Expected files/modules

Full-model prefill scheduler, session/workspace/handoff integration, bounded evaluation logits, EVAL-01 reports and continuation/context tests.

## Tests required

### Unit and contract checks

Valid-row ownership, bounded requested-logit tiles, workspace lifetime, commit/failure boundaries and dispatch policy identity.

### Reference and numerical checks

Component-level state/output checks across repeated decode, full/chunked prefill and native/GEMV transitions. Reuse semantic controls and distinguish schedule rounding from semantic bugs.

### Integration checks

Complete 54-case core through production prefill+decode, resolved selected-P100 adjudication and the fixed `R-32768-s0-d0.1` case for candidate and comparator. Exercise nonempty-session prefill, required partitions/checkpoints, interleave, snapshot/restore and late-failure recovery; document same-schedule and cross-schedule results separately. The optional 216-case suite is human-initiated interactive work only; agents must never launch it.

## Benchmark required

Diagnostic full-model prefill/handoff and memory measurements, including final-row generation versus requested-row evaluation. Matched external performance acceptance is TASK-027.

## Acceptance criteria

- [x] Both layer types compose through all 64 layers with bounded prefill workspace and correct head modes.
- [x] The complete 54-case EVAL-01 core through production prefill passes with authenticated references and resolved reviews.
- [x] The fixed `R-32768-s0-d0.1` case has passing paired 32768 evidence; missing mandatory coverage blocks acceptance.
- [x] Required partition/checkpoint, dispatch-transition, incoming-state and failure/replay checks pass.
- [x] Measured resident/transient memory fits the declared capacity/reserve; context and schedule coverage are explicit.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

PASS. Production prefill, handoff, quality, context and memory acceptance
criteria are met. The selected path uses 256-token chunks, BF16 activations
without activation scales, bounded unpack/cuBLAS prefill, GEMV decode and
one-row requested-logit readout. The accepted artifact manifest digest is
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43` (manifest
only; no tensor payload hashing). Candidate binary SHA-256 is
`ab7479715a658fb9072322f87dba4390aa0f1bcda2a38aeaf6052b0da7577c1c`; base
source revision `c4b6ead97d9f2650a2eb5daaab8da95bb908c5b3` plus the
reviewed working-tree changes; config hash
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`;
precision policy ID 1027. The candidate schema is Q4_K CandidateV2/Q8, BF16
operands and grouped GEMV decode. Runtime binary/policy identities: replay
SHA-256 `1d3b66fa9335ea705626b9bcff4138d4414813adf13471385dbb739992bd2235`,
benchmark SHA-256 `f7ea6ac782002d619d398fb6fd26c42c0f58e56576b5148675538f254f64a5fa`,
policy SHA-256 `7da1a1171984dc4761c86d87260a549f6462d9baf9ad14f5143ac0511a1cc679`,
and policy-rebind SHA-256
`c0412086338595ce10682b9734493ab2848f093fac50edd9eb946b0aa75f12f6`.

### Changes made

Implemented all-layer bounded prefill scheduling, generation/evaluation head
modes, session handoff and state/recovery integration. Sink exception poisoning
and moved-session coverage were repaired. The selected BF16 activation path has
no activation scales; decode continues with grouped GEMV against the same
resident weights. Evaluation reads requested logits one row at a time.

### Tests run

- Build: `.codex-wake-run/bfad62f44353.log` — final build completed.
- Focused integration: `.codex-wake-run/271ba3e095fd.log` — PASS 1/1 (8.08 s).
- Component tests: `.codex-wake-run/cf4d6e56ab2e.log` — PASS 3/3.
- Replay command log: `.codex-wake-run/1e03d1a8ab2d.log` — PASS for partition lengths 1, 63, 64, 65, 255, 256 and alternating 63/65; checkpoints 1, 3, 4, 63, 64, 65, 255, 256, 257; reset, snapshot/restore and interleave; metadata/argmax at every checkpoint; no nonfinite values. Final replay records are `.cache/evaluation/qw38-language-v2/task026-support/state-replay-*.json` (excluding preliminary `state-replay-cross-256.json`). Identical schedules replay bitwise. Full-model cross-schedule state deltas exceed isolated-component tolerances and are recorded as diagnostics; component numerical checks pass.
- 54-case candidate run: `.cache/evaluation/qw38-language-v2/runs/task026-prefill-core54-repaired-20260925T212502Z-879219/result.json` — COMPLETE 54/54. Pair report `.cache/evaluation/qw38-language-v2/paired/llama-20260925T103522Z-694370-task026-prefill-core54-repaired-20260925T212502Z-879219/summary.json` — PASS; authenticated references and selected-P100 owner reviews are recorded in `.cache/evaluation/qw38-language-v2/task026-support/p100-owner-review.jsonl`. C92 8/15 in each arm, L12 12/12, R-512 6/6 and R-4096 6/6 in each arm; teacher-target NLL delta +0.0015673756188502924 nats/token over 701 targets. Re-pair command log: `.codex-wake-run/0f2aacc5690f.log`.
- Fixed long case candidate: `.cache/evaluation/qw38-language-v2/runs/task026-prefill-r32768-smoke-final/result.json` — COMPLETE 1/1 for `R-32768-s0-d0.1`; run log `.codex-wake-run/0edb2bf54470.log`. Input preparation command `uv run --script scripts/task018_prepare_core.py --fixtures .cache/evaluation/qw38-language-v2 --output .cache/evaluation/qw38-language-v2/runs/task026-prefill-r32768-smoke-final/long.tsv --long-only`; output matched `task026-support/long-smoke.tsv`. Finalization command `bash scripts/run_task026_long_candidate.sh --finalize .cache/evaluation/qw38-language-v2/runs/task026-prefill-r32768-smoke-final` completed.
- Comparator: one fresh-request replay for the same fixed case; `.cache/evaluation/qw38-language-v2/runs/llama-task026-r32768-single-final/run_manifest.json` COMPLETE 1/1; log `.codex-wake-run/83f45777995d.log`. Paired command `uv run --script scripts/task026_long_pair.py --fixtures .cache/evaluation/qw38-language-v2 --llama-run .cache/evaluation/qw38-language-v2/runs/llama-task026-r32768-single-final --candidate-run .cache/evaluation/qw38-language-v2/runs/task026-prefill-r32768-smoke-final --output .cache/evaluation/qw38-language-v2/paired/task026-r32768-single/summary.json` — PASS both 1/1, matching answer `731204` at 32,757 prompt tokens. Negative check `uv run --script tests/task026_long_pair_test.py --fixtures .cache/evaluation/qw38-language-v2 --llama-run .cache/evaluation/qw38-language-v2/runs/llama-task026-r32768-single-final --candidate-run .cache/evaluation/qw38-language-v2/runs/task026-prefill-r32768-smoke-final` — PASS; all four altered-input cases were rejected.
- Roles: main implementation and evidence by GPT-6; independent Astra high review passed on pass 2; Luna medium completed documentation and delivery preparation.
- `git diff --check` — PASS.

The final 32768 policy SHA-256 is
`7da1a1171984dc4761c86d87260a549f6462d9baf9ad14f5143ac0511a1cc679`; replay
SHA-256 is `1d3b66fa9335ea705626b9bcff4138d4414813adf13471385dbb739992bd2235`.
Historical six-case runs predate the explicit single-case instruction and are
not final-binary evidence. No repeated long-context run or optional 216-case
suite is counted.

### Benchmark results

Diagnostic benchmark evidence: `.cache/evaluation/qw38-language-v2/task026-support/prefill-bench-final.txt`; command log `.codex-wake-run/35457cd9b234.log`. Hardware: NVIDIA RTX 5090, 32,607 MiB, driver 590.48.01, pinned CUDA 13.4.1. T=256 prefill 335.144 ms; four requested rows 268.635 ms; handoff 30.1028 ms. T=4096 prefill 6,284.73 ms. T=32768 prefill 202,714 ms (3m23s), handoff 43.0286 ms. Peak memory 24,761,073,664 bytes; free memory after prefill 8,338,276,352 bytes, above the 2 GiB reserve. The first T=256 row includes lazy workspace initialization, while requested-row timing reuses that workspace, so those timings are not a matched speed comparison. Matched performance acceptance remains TASK-027.

Evaluator binary SHA-256: `ab7479715a658fb9072322f87dba4390aa0f1bcda2a38aeaf6052b0da7577c1c`; benchmark binary SHA-256: `f7ea6ac782002d619d398fb6fd26c42c0f58e56576b5148675538f254f64a5fa`. The core run's generated/logit hashes match the original pre-repair core bitwise. Artifact identity is the manifest digest above; tensor payload hashing is intentionally not used.

### Architecture blocker

None.

### Follow-up observations

Matched whole-request performance, including comparative speed acceptance, is
owned by TASK-027. Full-model cross-schedule state deltas remain diagnostics;
same-schedule replay and isolated-component tolerances pass. No additional
coverage is required by this task.
