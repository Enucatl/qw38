# TASK-025 — Causal attention prefill and layer integration

## Status

DONE

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

- [x] Preparation, QK norm, RoPE, GQA, gating and output residual preserve model semantics.
- [x] Causal masking and cache/position updates pass empty/prefixed/tail/capacity checks.
- [x] Numerical and continuation checks pass for the complete attention+MLP layer.
- [x] No quadratic global score matrix or persistent replicated KV is required.
- [x] Measured SM120 tile/resource choices and bounded scratch/KV usage are recorded.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

Completed on 2026-09-25. Main-thread implementation used Codex GPT-6. The
independent Astra review returned `REVIEW: PASS` on pass 2, with no findings,
acceptance gaps, or evidence requests. The main thread confirmed the reviewed
candidate and evidence remained unchanged after review.

The accepted runtime uses token-major chunk preparation, reusing decode q/g
QK-normalization and partial-RoPE semantics, appends BF16 K/V, then runs FP32
causal online softmax and gating. It composes output projection/residual and
the complete MLP layer, and commits valid KV rows only after successful
synchronization. Failure poisons the session. The selected attention tile is
one query row by 32 keys; the four-row control was measured separately.
Workspace is bounded and the implementation does not materialize a quadratic
score matrix or persistent replicated GQA KV. The artifact is the accepted
CandidateV2 Q4_K MLP, Q8 other projections/head, BF16 activation/KV policy;
its manifest-only SHA-256 is
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`.
No artifact payload digest is claimed.

### Changes made

Added production causal attention prefill preparation, BF16 KV append,
FP32 tiled online attention, gating, output residual, and full attention+MLP
layer integration. Cache capacity and populated length remain distinct;
valid-token positions and cache writes are preserved across empty, prefixed,
and irregular chunks. The measured SM120 choice is one query row by 32 keys,
with a four-row control retained for comparison. Candidate layout/policy is
the accepted CandidateV2 artifact identified above.

### Tests run

Pinned image: `qw38-dev:cuda13.4.1-pinned`, image ID
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`;
CUDA 13.4.1, GNU 14.2.0, C++23, `build/pinned-release`. Hardware was an RTX
5090 (SM120), driver 590.48.01, 32,607 MiB, 400 W power limit.

The final build command exited 0:

```sh
docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc 'cmake --build build/pinned-release --target qw38_attention_prefill_test qw38_attention_prefill_artifact_test -j 8'
```

The final targeted test command exited 0; CTest passed 2/2 in 3.78 seconds:

```sh
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc 'ctest --test-dir build/pinned-release --output-on-failure -V -R "^attention_prefill(_artifact)?$"'
```

Full test log: `.codex-wake-run/task025-review-pass2-tests.log` (duplicate
`.cache/task025/review_pass2_tests.log` is task cache slated for delivery
cleanup). The unit test uses an independent double-precision reference,
first positions 0/M3 and 33/M7, tile sizes 1 and 4, and softmax stress logits
at least 100 with a later maximum at least 103 after key 32. Maximum BF16
output difference was 0. The artifact test covers M5, empty/prefixed/irregular
chunks, guards/cache tail, capacity vs population, snapshots/reset, failure
poisoning/device mutation, and decode continuation. Maximum mid-layer
difference was 0.00031662, final output 0.00237715, and KV 0.0000610352;
chunk partitions were bitwise equal. Tighter tolerances and a +0.25 negative
control were checked. Final unit, artifact-test, and benchmark binary SHA-256
values are respectively
`663a3ef0c809cbe7db9c3bc74268d8685393c30d7690a80d7d0ac0f4ef064bdd`,
`41bfb321856dcffe51116e4d60656157f444c9de805ecf14db414fd905b4916f`, and
`7b1a0bcf34402ae924145dea24ad8890e5846fe73deb7511ee5bbf456e10a319`.
The focused regression run also passed 7/7 in 20.94 seconds; its log is
`.codex-wake-run/6ec9988d5f5c.log`. Exact command:

```sh
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc 'ctest --test-dir build/pinned-release --output-on-failure -V -R "^(attention_(unit|core_unit|reference|integration|prefill|prefill_artifact)|language_layer_integration)$"'
```

### Benchmark results

The benchmark command exited 0; complete log:
`.codex-wake-run/c4fe378e31c1.log`.

```sh
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc 'build/pinned-release/benchmarks/qw38_bench_attention_prefill'
```

At M32, after warmup, three GPU-event samples using zero-filled synthetic
prefixes, full attention+MLP layer medians were 2.39747 ms at prefix 512 and
5.05267 ms at prefix 4096. Attention-only Q1 medians were 0.434816, 3.13635,
and 24.2972 ms at prefixes 512, 4096, and 32768; Q4 control medians were
0.514112, 3.80755, and 30.4537 ms. These are engineering timings, not
whole-request PERF-01 parity. Q1 used 55 registers, 18,572 shared bytes, no
local memory, and 5 blocks/SM occupancy; Q4 used 48 registers, 25,136 shared
bytes, no local memory, and 3 blocks/SM. Engine plus attention workspace was
26,017,792 bytes at M32; KV allocation was 2,149,580,800 bytes at capacity
32,800. Artifact device bytes were 21,462,812,800; measured free/total device
memory after allocations was 8,403,288,064 / 33,664,794,624 bytes.

Unique-KV-read lower bounds at prefixes 512/4096/32768 were 2,228,224 /
16,908,288 / 134,348,800 bytes. Estimated logical Q1 K/V reads were
415,629,312 / 3,234,201,600 / 25,782,779,904 bytes versus Q4 estimates
104,202,240 / 808,845,312 / 6,445,989,888 bytes. These are logical estimates,
not measured DRAM traffic. The benchmark uses the accepted artifact manifest
identity above and zero-filled synthetic prefixes; no quality or full-model
prefill claim is made.

### Architecture blocker

None.

### Follow-up observations

Full-model prefill, quality gate, and full-model prefill-to-decode handoff
remain owned by TASK-026; TASK-025's complete attention-layer continuation is
already tested above. The synthetic-prefix timings here establish the TASK-025
attention engineering envelope only; whole-request PERF-01 comparison remains
with TASK-027.
