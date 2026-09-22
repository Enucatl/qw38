# QW38 Architecture V0 Implementation Task Ledger

## Goal

Implement Architecture V0 as a sequence of small, testable increments that reaches batch-one primary-language decode early, then adds the distinct production prefill schedule, reproducible measurement, and only afterward the eight authorized architecture experiments.

## Normative authority

- `docs/architecture/architecture-v0.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`

## Execution policy

- Execute tasks sequentially; accept every listed prerequisite before starting a task.
- Completion requires every acceptance criterion and required test in the task file.
- A locked-decision conflict stops the sequence with `ARCHITECTURE_BLOCKER`; it never authorizes an implicit redesign.
- Every blocker report contains: `Decision ID`, `Attempted implementation`, `Observed problem`, `Evidence`, `Why this is architectural rather than tuning`, `Smallest plausible alternative`, and `Affected downstream tasks`.
- Ordinary tasks implement V0 as written. Only TASK-023 through TASK-030 may challenge the named selected hypotheses, and each must keep unrelated variables fixed.
- Keep benchmarks separate from correctness tests. Preserve exact commands, results, artifact/binary identities, and hardware context in each completion report.
- Status values are `TODO`, `IN_PROGRESS`, `BLOCKED`, and `DONE`; all tasks begin `TODO`.

## Milestones

- **M0 — Reproducible implementation foundation:** the pinned CUDA 13.4.x container builds C++23/CUDA C++23 for `sm_120`, runs a GPU smoke test, and runs the normal test command.
- **M1 — Runtime artifact and compiler foundation:** `.qw38` has an independently tested ABI, writer, validator, metadata model, and BF16 identity compiler path.
- **M2 — Quantized contraction foundation:** Q4G64/Q8G32 have deterministic logical quantizers, V0 packers, independent decoders, reference contractions, and correct decode CUDA consumers.
- **M3 — Core runtime and MLP:** typed CUDA ownership, stable session storage, core numerical primitives, and the first complete MLP execute correctly.
- **M4 — GDN execution:** convolution/preparation, recurrence, and the complete GDN mixer execute with continuation-correct persistent state.
- **M5 — Attention execution:** preparation/cache and segmented online decode attention execute with deterministic merging.
- **M6 — Integrated language decode:** both layer kinds integrate through common state/scratch machinery and the complete 64-layer primary-language model produces FP32 logits.
- **M7 — Behavioral correctness baseline:** frozen language-only NLL, distribution, generation, continuation, capability, and feasible long-context checks distinguish BF16 control from V0 quantization effects.
- **M8 — Production V0 prefill:** tensor-core projections, chunked GDN, tiled causal attention, and exact prefill-to-decode handoff work end to end.
- **M9 — End-to-end performance baseline:** declared decode, prefill, request, memory, and kernel measurements are reproducible and identity-bound.
- **M10 — Architecture-V0 experiments:** EXP-A through EXP-H produce keep/change recommendations without opportunistic redesign.

## Task ledger

| Task | Name | Milestone | Depends on | Primary output/capability | Status |
| ---- | ---- | --------- | ---------- | ------------------------- | ------ |
| TASK-001 | Reproducible CUDA build foundation | M0 | — | Pinned container, native build/test skeleton, `sm_120` GPU smoke | DONE |
| TASK-002 | `.qw38` format constants and schema | M1 | TASK-001 | Explicit little-endian ABI types and schema validation primitives | DONE |
| TASK-003 | `.qw38` writer and integrity records | M1 | TASK-002 | Deterministic aligned artifact emission with SHA-256 records | DONE |
| TASK-004 | `.qw38` reader, metadata, and corruption validation | M1 | TASK-003 | Safe parser for tensors, graph bindings, policy, hashes, and state schema | DONE |
| TASK-005 | BF16 identity compiler path | M1 | TASK-004 | Streaming HF-to-`.qw38` identity compiler and exact reconstruction control | DONE |
| TASK-006 | Logical quantizers and CUDA V0 packers | M2 | TASK-005 | Q4G64/Q8G32/BF16 reference quantize, pack, unpack, and contraction | DONE |
| TASK-007 | CUDA runtime ownership and session storage | M3 | TASK-004 | Typed CUDA RAII, artifact upload, persistent state, and scratch arena | DONE |
| TASK-008 | Core reference math and activation kernels | M3 | TASK-007 | Embedding, RMS variants, nonlinearities, RoPE, and reference dense math | DONE |
| TASK-009 | Decode dense contraction consumers | M2 | TASK-006, TASK-008 | BF16/Q4/Q8 CUDA MMV consumers with declared epilogues | DONE |
| TASK-010 | Complete decode MLP | M3 | TASK-009 | Architecture-V0 RMS → paired SwiGLU → down/residual path | DONE |
| TASK-011 | GDN convolution and preparation | M4 | TASK-009 | Reference and CUDA FIR/history, q/k normalization, and gate preparation | DONE |
| TASK-012 | GDN recurrence and continuation | M4 | TASK-011 | Independently validated FP32 `[head,value,key]` recurrent state update | DONE |
| TASK-013 | Complete decode GDN mixer | M4 | TASK-010, TASK-012 | Eight-region GDN mixer plus MLP-compatible residual transition | DONE |
| TASK-014 | Attention preparation and KV cache | M5 | TASK-009 | Per-head q/g, QK norm, partial RoPE, and BF16 cache append | DONE |
| TASK-015 | Segmented online decode attention | M5 | TASK-014 | Causal GQA segment scan, fixed-order merge, gating, output residual | DONE |
| TASK-016 | One-layer integration checkpoint | M6 | TASK-013, TASK-015 | One GDN-style and one attention-style layer through common runtime | TODO |
| TASK-017 | Complete primary-language decode | M6 | TASK-016 | Embedding, 64 layers, persistent state, final norm, Q8 head, logits | TODO |
| TASK-018 | Behavioral correctness baseline | M7 | TASK-017 | Frozen language-only BF16/V0 behavioral and continuation validation | TODO |
| TASK-019 | Tensor-core prefill projections and chunk planning | M8 | TASK-018 | Bounded token-major scratch and common-view packed GEMM consumers | TODO |
| TASK-020 | Chunked prefill GDN | M8 | TASK-019 | Parallel FIR/history commit and ordered 64-token recurrence schedule | TODO |
| TASK-021 | Tiled causal prefill attention | M8 | TASK-019 | 32-query × 64-key online attention with correct cache/state semantics | TODO |
| TASK-022 | End-to-end prefill and decode handoff | M8 | TASK-020, TASK-021 | 256-token chunked full-model prefill and exact continuation into decode | TODO |
| TASK-023 | Reproducible V0 performance baseline | M9 | TASK-022 | Declared decode/prefill/request/memory/kernel baseline | TODO |
| TASK-024 | EXP-A main weight policy | M10 | TASK-023 | Quality/performance comparison of Q4 group sizes and failing families | TODO |
| TASK-025 | EXP-B vocabulary precision | M10 | TASK-024 | Q8G32 versus Q4G64 head decision | TODO |
| TASK-026 | EXP-C persistent-state precision | M10 | TASK-025 | FP32 versus BF16 GDN-state decision | TODO |
| TASK-027 | EXP-D GDN layout and ownership | M10 | TASK-026 | Selected versus `[head,key,value]` recurrence ABI decision | TODO |
| TASK-028 | EXP-E activation transport | M10 | TASK-027 | BF16 versus FP32 transport diagnosis and decision | TODO |
| TASK-029 | EXP-F normalization/projection boundary | M10 | TASK-028 | Separate RMS store versus decode GDN fusion decision | TODO |
| TASK-030 | EXP-G common weight view | M10 | TASK-029 | One versus additional MLP prefill weight view decision | TODO |
| TASK-031 | EXP-H prefill recurrence algorithm | M10 | TASK-030 | Serial 64-token versus chunkwise/WY recurrence decision | TODO |

## Critical path

```text
TASK-001 → 002 → 003 → 004 → 005 → 006
                         └→ 007 → 008 ─┐
                              006 ─────┴→ 009 → 010 → 011 → 012 → 013
                                                   009 → 014 → 015
                                      013 + 015 → 016 → 017  (complete decode)
                                                        ↓
                                      018 → 019 → 020 + 021 → 022  (complete V0 prefill)
                                                        ↓
                                      023 → 024 → 025 → 026 → 027 → 028 → 029 → 030 → 031
```

## Architecture blocker log

| Task | Decision | Summary | Resolution |
| ---- | -------- | ------- | ---------- |

## Architecture amendment log

| Amendment | Date | Decisions affected | Summary |
| --------- | ---- | ------------------ | ------- |
