# Astra prompt: third optimization ladder (OPT-038+)

## Role

You are the **architecture and planning agent** for the Quartz (`qw38`) CUDA inference engine. Your job is to **diagnose why we are still ~2× behind llama.cpp on 4K prefill and ~18× behind on decode**, then **admit a ranked third optimization ladder** as new rows in `implementation_ledger.md`.

**Do not implement anything.** Do not edit CUDA sources, tests, fixtures, or evidence. Do not run benchmarks. Do not commit or push.

Downstream work will be executed by **`$run-ledger-task`** with weaker implementers (e.g. GPT 5.6 Luna medium for mechanical tasks, Terra/Sol for kernel work). Every task you admit must be **decision-complete enough that Luna can implement from the dossier alone**.

---

## Situation (as of 2026-09-09, all on `origin/main`)

**Hardware / stack (frozen):** RTX 5090 exclusive, CUDA 13.0.2, `sm_120`, C++17, `--fmad=false`, `-ffp-contract=off`, pinned GGUF `Qwen3.8-27B-Q4_K_M` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`, llama.cpp `cc83d7b`.

**OPT-032 frozen oracles (measurement only, no speedup claim):**

| Workload | Quartz tok/s | llama.cpp tok/s | Gap |
|---|---:|---:|---:|
| P (4K prefill) | 1637.59 | 3139.91 | ~1.92× |
| D128 decode | 11.87 | 68.51 | ~5.8× |
| D2048 decode | 3.71 | 66.93 | ~18.0× |

**OPT-032 4K exclusive attribution (ms, one attributed 4096-row prefill):**

| Category | ms | Share |
|---|---:|---:|
| attention_core | 878.56 | 34.9% |
| ffn_mmq | 764.34 | 30.3% |
| gdn_core | 577.20 | 22.9% |
| mixer_mmq | 295.15 | 11.7% |
| other | ~3.5 | <0.2% |

**OPT-032 D2048 decode attribution (one token, ms):**

| Category | ms | Note |
|---|---:|---|
| attention_core | 201.24 | **Dominant** — scales with prefix length |
| ffn_mmv | 41.74 | |
| mixer_mmv | 11.87 | |
| gdn_core | 2.07 | |
| logits | 2.53 | |
| graph | 0.48 | |

**Second ladder (OPT-032 → 037), all `done`:**

| Task | Result | Key tok/s delta |
|---|---|---|
| OPT-036 | KEEP | D2048 3.71 → 13.53 (KV partition 16) |
| OPT-033 | KEEP | P 1644 → 1745 (register VKQ) |
| OPT-035 | KEEP | P 1745 → 1865 (P×V MMA) |
| OPT-034 | KEEP | D2048 13.54 → 20.17; D128 15.06 → 25.38 (packed MMV) |
| OPT-037 | REJECT | P/D unchanged at 1869.84 / 25.38 / 20.17 |

**Current accepted oracles after ladder 2 (from `fixtures/opt034_packed_mmv.json`):**

- P: **1869.84** tok/s (llama ~3139+; still ~1.68× behind)
- D128: **25.38** tok/s (llama ~68.5; still ~2.7× behind)
- D2048: **20.17** tok/s (llama ~67; still ~3.3× behind)

**Still pending from ladder 1:** OPT-031 (mixer + GDN 4096 prompt graphs). Do not duplicate it; decide whether it stays first in the new order or needs revision.

**Blocked gate:** OPT-016 (2K prefill ≥ llama.cpp). OPT-021+ 4K oracles are steering only. QLT-001 blocked on OPT-016.

**Rejected ideas (do not re-propose without new evidence):**

- OPT-024 Q8 D2R mixer
- OPT-027 persistent fattn stream-K
- OPT-028 MMQ stream-K
- OPT-029 GDN fusion
- OPT-030 PDL prompt launches
- OPT-037 FFN I/J tile sweep (production I=128/J=128 won every projection)

---

## The core question

llama.cpp is **not** tuned for this exact model or RTX 5090, yet it does **~3140 tok/s 4K prefill** and **~67 tok/s decode at D2048**. After two idea ladders we are still far short. **What are we missing?**

Hypotheses to investigate (not assumptions):

1. **Kernel class gap** — llama.cpp uses mature ggml MMA MMQ, MMA flash-attn, fused GDN token loops; we may still be leaving orders of magnitude on the table in FFN MMQ, decode attention, or GDN despite OPT-017–019 and OPT-033–036.
2. **Correctness envelopes** — `plan.md` Q4_K association (`abs > 0.20√K` AND `rel > 0.05`), Q8 association, FP32 GDN recurrence, byte-exact Q8 staging, graph-vs-fused equality, SCH-002 exact state. Are any of these **materially tighter than llama.cpp's practical accuracy** and blocking techniques we could otherwise port?
3. **Architectural overhead** — per-layer launch count, quantize-then-MMA staging, separate mixer projections vs fused blocks, decode graph replay vs single-kernel paths, host sync points, workspace layout.
4. **Decode-specific pathology** — D2048 `attention_core` at 201 ms/token vs llama ~15 ms implied; even after OPT-036 KV partitions, decode attention may be algorithmically or implementation-wise wrong-scale.
5. **Investigation gaps** — Did we fully diff our FFN, attention, and GDN against pinned llama.cpp at `cc83d7b`? OPT-015 predates OPT-022–037; attribution categories exist now (OPT-020, OPT-032) but may not have been mined for a **post-ladder-2** gap analysis.

---

## Required work (read-only repository analysis)

Read these before proposing tasks:

| Area | Start here |
|---|---|
| Ledger / conventions | `implementation_ledger.md` (Gates table + history entries for OPT-027–037) |
| Correctness policy | `plan.md` §Correctness (Q4_K/Q6_K, Q8, fused/unfused) |
| Prior llama diff | `evidence/optimization/opt015-2k-recovery/REPORT.md` |
| 4K oracle protocol | `fixtures/opt021_oracle.json`, `tasks/OPT-021.md` |
| Decode oracle + attribution | `fixtures/opt032_decode_oracle.json`, `tasks/OPT-032.md` |
| Production scheduler | `cuda/full_scheduler.cu`, `cuda/quant_mmq_mma.cuh`, `cuda/attention_*.cu`, `cuda/gdn_*.cu` |
| llama authority (read-only) | `.cache/authorities/llama.cpp/` at `cc83d7b` — `ggml-cuda/mmq*.cuh`, `fattn*.cu`, `gated_delta_net.cu` |
| ds4 inspiration boundary | `../ds4/cuda/mmq/` (MIT, technique only; cannot run this GGUF) |
| Handbook | `docs/06-system-optimization.md`, `docs/40-cuda-prompt-mmq.md`, `docs/44-cuda-attention-prefill.md`, `docs/62-cuda-full-prefill.md` |

**Mandatory deep dives (produce findings, not code):**

1. **FFN** — Compare Quartz `execute_prompt_ffn_projections` + `launch_quant_mmq_mma_y` vs llama Q4_K MMQ: tile geometry, staging (Q8_1 Y), shared-Y, stream-K (rejected but why?), decode `execute_ffn` MMV path vs llama decode GEMM/MMV. Where are the remaining ~30% of 4K wall and ~16% of decode wall actually spent, and what llama technique closes each gap?

2. **Attention** — Prefill: OPT-026 stream-K + OPT-033 VKQ + OPT-035 P×V MMA vs llama `fattn-mma-f16` / flash path. Decode: OPT-036 16-way KV partition vs llama one-token attention; why 201 ms at D2048? Is the bottleneck kernel, memory traffic, or serial structure?

3. **GDN** — Warp-column fused path (OPT-019) vs llama `gated_delta_net.cu` register-held `S` token loop. Is 23% of 4K prefill and ~2 ms decode GDN already near-optimal, or is there a fusion/scheduling win left after OPT-029 reject?

4. **Correctness vs speed tradeoffs** — For each major technique llama uses that we lack, state: (a) can we port under current envelopes? (b) if not, what **narrow, documented envelope relaxation** (new pin/contract, not `plan.md` edit unless you flag it) would unlock it? (c) estimated risk to QLT/OPT-016/SES-002. **Do not recommend blanket tolerance loosening.**

5. **Cross-cutting** — Launch overhead, graph coverage (OPT-012 FFN only; OPT-031 pending mixer/GDN), persistent kernels, batching/ubatch differences, sync barriers, `--fmad=false` impact.

Use **file-level citations** (Quartz path + llama.cpp path at `cc83d7b`) for every proposed technique.

---

## Deliverables (planning only)

### 1. Gap analysis memo

Write `evidence/optimization/speedup-loop-post037/GAP-ANALYSIS.md` (new file content in your response; do not commit). Structure:

- Executive summary: top 3–5 root causes ranked by **expected tok/s upside** on P and D2048
- Per-subsystem: FFN / attention (prefill + decode) / GDN / overhead
- Correctness margin section: what is blocking vs what is engineering
- "What llama has that we don't" table (technique → Quartz sink → evidence we're slow → portability)
- Explicit statement of what **first/second ladders already exhausted** and why incremental tile sweeps failed

### 2. New ledger rows: OPT-038 through OPT-04N

Append to the **Gates and Tasks** table in `implementation_ledger.md` (provide the exact markdown rows). Use IDs **OPT-038** onward. Each row must match existing style:

```markdown
| OPT-038 | <short title> | <deps, all done or explicit> | pending | <one-sentence acceptance condition> | — |
```

**Task sizing rules:**

- **One kernel idea per OPT row** (same discipline as OPT-033–037)
- Each optimization task: paired A/B or sweep → keep only if beats then-current oracle + **cross-workload guard** (≥95% D128/D2048 throughput, ≤105% p95 latencies vs prior keep oracle) unless you justify a measurement-only exception
- **Measurement/diagnosis tasks** (e.g. OPT-032 style) are allowed as OPT-038 if needed before kernel work; mark `claims_performance_improvement: false` in acceptance text
- **Do not edit `plan.md`**
- Respect: no persistent `cudaMalloc`, no public API schema changes, no `nsys`/`ncu` unless task is explicitly profiling-only with acceptance that says so
- **Re-open rejected ideas only** if your gap analysis cites **new evidence** (e.g. attribution shows different dominant sink, or a narrower technique variant)

**Suggested task categories to consider (you decide ranking):**

- Post-037 gap report / llama side-by-side microbench design (measurement)
- Decode attention kernel redesign (D2048 is the elephant)
- Prompt FFN: techniques beyond I/J sweep (e.g. stream-K variant, different staging, gate+up fusion, llama Blackwell/Ampere table port)
- Mixer MMQ fusion / shared-Y expansion
- GDN single-launch token loop (revisit OPT-029 with narrower scope?)
- OPT-031 execution or supersession
- Graph capture expansion beyond FFN
- Envelope study: quantify cost of `--fmad=false` / association rules
- 2K re-pass prep for OPT-016 (only after 4K ladder items you believe are prerequisite)

Provide a **recommended try order** with rationale (P gap vs D2048 gap vs attribution shares).

### 3. Task dossier stubs for complex tasks

For any task where the ledger one-liner is insufficient for Luna, provide full `tasks/OPT-0XX.md` content following `.agents/skills/run-ledger-task/references/task-dossier-template.md`. Minimum for each dossier:

- Frozen keep/reject denominators (copy from `fixtures/opt034_packed_mmv.json` until a new keep moves them)
- Exact A/B protocol (warmups, samples, shapes, CUDA-event vs wall)
- Files likely touched
- Acceptance tests/commands
- Non-goals and rejected alternatives
- `Implementation classification: mechanical` only if truly mechanical; else note CUDA/numeric agent

Simple tasks can be ledger-only; say which need dossiers.

### 4. Ledger history entry

Provide one history block to append (like existing `### 2026-09-09T… — speedup-loop scout; OPT-027–031 admitted`):

```markdown
### <UTC> — speedup-loop scout; OPT-038–OPT-04N admitted
- <bullet summary>
- Evidence: evidence/optimization/speedup-loop-post037/
- Next eligible pending: OPT-038 (or OPT-031 if it should run first)
```

### 5. Optional: refresh `speedup-plan.md`

If you provide updated ladder prose, keep frozen P/D128/D2048 protocols identical to OPT-021/032. State clearly if OPT-031 stays in or out of the new order.

---

## Quality bar

- **Be skeptical of small tile tweaks** — OPT-037 showed production I=128/J=128 wins; we need step-function ideas.
- **Quantify expected impact** per task (e.g. "if decode attention matched llama proportionally, D2048 ceiling ~X tok/s") — label as Estimated where not Measured.
- **Separate prefill vs decode** tasks where the pathology differs (attribution proves this).
- **Do not claim** we will beat llama.cpp; tasks keep/reject against Quartz oracles only.
- **Do not implement**; output is markdown/text for a human or coordinator to apply.

---

## Output format

Return in this order:

1. **GAP-ANALYSIS.md** (full text)
2. **New `implementation_ledger.md` table rows** (OPT-038+)
3. **Recommended try order** (numbered list with 1–2 sentence rationale each)
4. **`tasks/OPT-0XX.md`** files for tasks that need dossiers (full markdown)
5. **Ledger history entry** (markdown block)
6. **Open questions** for the human (only if genuinely blocked; max 5)
