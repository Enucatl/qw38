# TASK-16 — Build the CUDA hardware model

## Control

- Primary ID: `TASK-16`
- Coupled IDs: `none`
- Dependencies: `TASK-00` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Cover requested execution, memory, resource, synchronization, and instruction concepts; explain fusion versus occupancy tradeoffs mathematically; exclude current Qwen kernel inspection.

## Goal and boundaries

Produce `docs/architecture/cuda-hardware-model.md` as the Phase 1 **model-independent** CUDA resource and tradeoff reference: a parameterized vocabulary of execution, memory, occupancy limiters, synchronization, and instruction pipelines, plus algebraic fusion-versus-occupancy relations that TASK-17 will instantiate.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - This increment depends only on TASK-00. It is **model-independent**. Do not derive content from model equations, inventory, dataflow, quantization, layouts, or any kernel.
  - Allowed evidence: public CUDA C++ Programming Guide, PTX ISA, occupancy, and memory-hierarchy identities, plus this dossier. Plan.md explicitly allows CUDA hardware documentation.
  - Label claims `DERIVED` (occupancy/fusion/Little’s-law/roofline algebra), `OBSERVED` **only** for the locked list of published CUDA programming-guide / PTX identities below, or `UNKNOWN` for sitting-SKU limits and instruction availability. Do not apply `MEASURED` or `HYPOTHESIS` anywhere in the hardware-model deliverable.
  - GitHub Markdown math. Do not invent sitting-GPU numbers. Parameterize every SKU limit.
  - Do not inspect Quartz execution/CUDA code, llama.cpp/GGML construction or kernels, `models/` GGUF artifacts, production `OPT-*` kernels/pins/traces, or any `*.cu` / `*.cuh` kernel body.
- Non-goals:
  - No semantic-node CUDA mappings, ownership, or reduction choices (TASK-17).
  - No kernel source, launch configs, or SASS.
  - No microbenchmarks, `deviceQuery` dumps, or occupancy measurements (TASK-19).
  - No layout, quantization, graph, or compiler work.
  - No claim of a winning fusion policy, winning MMA shape, or winning occupancy target.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only).
  - Do not edit `docs/architecture/plan.md` or `docs/architecture/task_ledger.md` (coordinator/delivery own ledger status).
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib catalog checker used only as evidence tooling)

## Repository evidence

Planning inspected only the allowed sources. Implementation must not expand the evidence set.

- `docs/architecture/plan.md:9-33` — central path places CUDA mappings after semantic graph and schedules; this task supplies hardware vocabulary, not those mappings.
- `docs/architecture/plan.md:47-51` — CUDA hardware documentation is allowed; existing-engine CUDA kernels are not.
- `docs/architecture/plan.md:54-64` — evidence labels `DERIVED` / `OBSERVED` / `MEASURED` / `HYPOTHESIS` / `UNKNOWN`; this deliverable uses `DERIVED`, published-identity `OBSERVED`, and `UNKNOWN` only.
- `docs/architecture/plan.md:78-80` — TASK-16 runs in parallel with model inventory and must not look at model-specific kernels.
- `docs/architecture/plan.md:93-95` — TASK-17 maps semantic nodes into CUDA experiment spaces using this vocabulary; TASK-19 owns performance methodology.
- `docs/architecture/task_ledger.md` TASK-16 row — produces `docs/architecture/cuda-hardware-model.md`; purpose is a model-independent CUDA resource and tradeoff reference; open question is target-GPU-specific limits and instruction availability; completion is concept coverage, fusion-versus-occupancy math, and no current kernel inspection.
- `docs/architecture/task_ledger.md` TASK-17 row — **consumer only**. It will estimate work, storage, access, synchronization, occupancy, and mode suitability across mapping alternatives and must select no winner without measurements. This task does not perform those mappings.
- `docs/architecture/tasks/TASK-01.md`, `TASK-02.md`, `TASK-03.md` — peer dossier structure, stdlib checker pattern, JSON-fence embedding, and run-record shape. Do not copy their model-specific content.
- `scripts/check_model_semantics.py` and `scripts/check_dataflow.py` — tooling precedent only: Python 3.11+, stdlib `argparse` / `json` / `re` / `pathlib` / `sys`, Google docstrings, `--json` plus a markdown check flag, pretty-printed JSON, exit 0/1. TASK-16’s checker does **not** read `config.json` and does **not** import those scripts.

## Performance evidence

N/A — instruction-only hardware vocabulary; no prefill/decode/component timing, no keep/reject, no sink ranking, no GPU run.

## Implementation decisions

### Authority for the hardware model

Write a **hardware-general** reference parameterized by SM/SKU symbols. Public CUDA programming-guide / PTX ISA identities are sufficient. If a secondary blog or SKU datasheet disagrees with the programming-guide occupancy or barrier identities, the programming-guide identity wins and the datasheet is ignored here.

Do not transcribe kernels, SASS, or sitting `cudaGetDeviceProperties` output. Do not use production engine occupancy numbers as design authority.

**Published identities that may be labelled `OBSERVED`** (this list is exhaustive for the deliverable; do not mark other claims `OBSERVED`):

1. Warp size \(N_w=32\) on the CUDA devices this study considers.
2. Occupancy is the ratio of active resident warps on an SM to that SM’s maximum resident warps.
3. A CTA (thread block) is the domain of `__syncthreads`.
4. Shared memory is banked; the programming-guide identity is \(N_{\text{bank}}=32\) banks of 32-bit words.
5. Consecutive threads of a warp accessing consecutive aligned addresses in global memory coalesce into the minimum number of transactions for that request size.
6. Occupancy is jointly limited by registers per SM, shared memory per SM, threads per SM, and maximum CTAs per SM.

Everything algebraic (limiter mins, Little’s law, wave quantization, fusion footprints, arithmetic intensity, roofline inequality) is `DERIVED`. Every sitting-device numeric limit and optional instruction/capability is `UNKNOWN`.

### Deliverable structure (`docs/architecture/cuda-hardware-model.md`)

Use these **level-2 headings in this exact order**. Compact tables + numbered formulae. GitHub Markdown math (`$...$` / `$$...$$`). Do not leave `TBD` / `TODO` / `???`. Do not use `MEASURED` or `HYPOTHESIS`. The only `UNKNOWN` uses are SKU limits and optional-capability availability, concentrated in **SKU parameterization** and **Deferred SKU table** (those words may also appear next to optional instruction rows).

Title: `# CUDA hardware model` (not `TASK-16`).

1. **Authority** — this dossier, `plan.md` evidence policy, checker path; model-independent scope; in-scope (CUDA resource vocabulary + fusion/occupancy algebra) vs out-of-scope (mappings, kernels, measurements). State that the document specifies hardware concepts, not a kernel.
2. **SKU parameterization** — symbol table (locked below); canonical SKU sentence verbatim; warp size 32 as the sole numeric constant treated as `OBSERVED`.
3. **Execution hierarchy** — thread through SM/scheduler, SIMT divergence, optional thread-block cluster.
4. **Occupancy and latency hiding** — occupancy definition, theoretical occupancy, stalls, Little’s-law hiding test, wave quantization (formulae F7–F10, F8, F9, F10).
5. **Memory hierarchy** — registers, shared memory, L1, L2, HBM/device global, host pinned, local/spill; capacity vs bandwidth vs latency as distinct resources.
6. **Access patterns** — coalescing, alignment, bank conflicts.
7. **Resource limiters** — registers/thread, shared mem/CTA, threads/CTA, CTAs/SM, warp slots, barrier slots; limiter formulae F2–F6 and F1.
8. **Synchronization** — CTA barrier, warp sync, fences, streams, events, cooperative groups, async copy pipeline; **what each orders**.
9. **Instruction pipelines** — FMA/FFMA, load/store, tensor-core MMA family, async global-to-shared / TMA as optional SKU-UNKNOWN capability.
10. **Fusion versus occupancy** — algebraic tradeoff (F11, F12, intensity via F13); fused kernel raises register/shared footprint ⇒ \(O\) may fall ⇒ latency hiding may fail even if bytes/FLOPs per launch improve.
11. **Evaluation criteria** — numbered TASK-17 instantiation checklist (roofline F14 as a criterion, not a measured point); no winner.
12. **Deferred SKU table** — restates the open-question closure; lists SKU-UNKNOWN fields; forbids sitting-device fill-in in this document.
13. **Machine-checkable catalog** — exactly one fenced `json` code block, copied from a fresh `python3 scripts/check_cuda_hardware_model.py --json` run (pretty-printed, key order as emitted).

Immediately under **SKU parameterization**, include this **canonical SKU sentence verbatim** (checker substring match, outside the JSON fence):

> Target-GPU-specific limits and instruction availability remain UNKNOWN until a future measured SKU table. This document supplies the symbols and evaluation criteria TASK-17 will instantiate; it does not record sitting-device numbers and does not claim a winning fusion policy.

Immediately under **Fusion versus occupancy**, include this **canonical fusion sentence verbatim**:

> Fusion versus occupancy algebra is an evaluation criterion, not a selected mapping.

Immediately under **Evaluation criteria**, include this **canonical winner sentence verbatim**:

> This document selects no CUDA mapping winner.

No extra `##` headings. `###` subheadings are allowed. Documentation-stage `unverified` banner must be a blockquote or italic line **before** the first `##`, not a new `##` heading.

### SKU symbols (lock; all numeric values UNKNOWN except \(N_w\))

Table columns: `Symbol` | `JSON id` | `Meaning` | `Status`.

| Symbol | JSON id | Meaning | Status |
| --- | --- | --- | --- |
| \(N_w\) | `N_w` | threads per warp | `OBSERVED` \(=32\) |
| \(N_{\text{bank}}\) | `N_bank` | shared-memory banks | `OBSERVED` \(=32\) |
| \(N_{\text{SM}}\) | `N_SM` | SM count | `UNKNOWN` |
| \(W_{\max}\) | `W_max` | max resident warps per SM | `UNKNOWN` |
| \(S_{\text{reg}}\) | `S_reg` | 32-bit register-file size per SM | `UNKNOWN` |
| \(C_{\text{smem}}\) | `C_smem` | shared-memory capacity per SM (bytes) | `UNKNOWN` |
| \(T_{\max}\) | `T_max` | max threads per SM | `UNKNOWN` |
| \(B_{\max}\) | `B_max` | max CTAs per SM | `UNKNOWN` |
| \(N_{\text{bar}}\) | `N_bar` | hardware barrier slots per SM | `UNKNOWN` |
| \(N_{\text{sched}}\) | `N_sched` | warp schedulers per SM | `UNKNOWN` |
| \(G_{\text{reg}}\) | `G_reg` | register allocation granularity (32-bit regs) | `UNKNOWN` |
| \(G_{\text{smem}}\) | `G_smem` | shared-memory allocation granularity (bytes) | `UNKNOWN` |
| \(\Beta\) | `Beta_HBM` | device-global (HBM) peak bandwidth | `UNKNOWN` |
| \(\Pi_{\text{FMA}}\) | `Pi_FMA` | scalar FMA peak throughput | `UNKNOWN` |
| \(\Pi_{\text{TC}}\) | `Pi_TC` | tensor-core MMA peak throughput | `UNKNOWN` |
| \(L_{\text{issue}}\) | `L_issue` | dominant stall latency in scheduler cycles | `UNKNOWN` |
| \(\texttt{async\_copy\_cap}\) | `async_copy_cap` | `{absent, cp.async, TMA}` | `UNKNOWN` |
| \(\texttt{mma\_shapes}\) | `mma_shapes` | legal MMA \((M,N,K)\) and dtypes | `UNKNOWN` |
| \(\texttt{cluster\_cap}\) | `cluster_cap` | thread-block cluster availability | `UNKNOWN` |

JSON field `sku_unknown_symbols` is exactly this list of JSON ids **excluding** `N_w` and `N_bank` (those are published identities, not sitting-SKU unknowns):

`N_SM`, `W_max`, `S_reg`, `C_smem`, `T_max`, `B_max`, `N_bar`, `N_sched`, `G_reg`, `G_smem`, `Beta_HBM`, `Pi_FMA`, `Pi_TC`, `L_issue`, `async_copy_cap`, `mma_shapes`, `cluster_cap`

Do not instantiate any `UNKNOWN` symbol with a datasheet or sitting-GPU number. Close the ledger open question by the canonical SKU sentence: limits and instruction availability stay `UNKNOWN` until a future **measured** SKU table (a later task); this document supplies symbols and evaluation criteria only.

### Concept catalog (lock all 54 IDs)

Implementation must include **one catalog table per relevant section** with columns in this order:

`ID` | `Symbol` | `Must define`

Every `ID` appears in the markdown as a backtick-wrapped token (for example `` `thread` ``) **outside** the JSON fence. Do not add/remove IDs. Definitions below are normative: copy the meaning; do not substitute a different abstraction.

**Execution hierarchy** (IDs 1–10, 15 live here; occupancy IDs 11–14 live in the next section).

| ID | Symbol | Must define |
| --- | --- | --- |
| `thread` | scalar CUDA thread | Programmable scalar lane; has private registers and a thread index in the CTA. |
| `warp` | warp of \(N_w\) threads | SIMT scheduling unit issued together. |
| `warp_size` | \(N_w=32\) | Constant warp width; `OBSERVED` published identity. |
| `simt_divergence` | diverged warp | Taken/not-taken paths in a warp serialize; reconvergence is warp-scoped, not a CTA barrier. |
| `cta` | cooperative thread array | Launch unit that shares shared memory and `__syncthreads`; synonym of block. |
| `block` | thread block | Alias of `cta`; both IDs must appear; state they are the same object. |
| `grid` | \(N_{\text{grid}}\) CTAs | The kernel launch; CTAs of one grid do not share a CTA barrier. |
| `sm` | streaming multiprocessor | Occupancy, register file, shared memory, and scheduler domain. |
| `scheduler` | warp scheduler | Selects ready warps on an SM; count \(N_{\text{sched}}\) is `UNKNOWN`. |
| `thread_block_cluster` | optional cluster | Multi-CTA grouping with optional distributed shared memory; availability `cluster_cap` is `UNKNOWN`. Do not assume it exists. |

**Occupancy and latency hiding**

| ID | Symbol | Must define |
| --- | --- | --- |
| `stall` | scheduler stall | A warp is not issuable (scoreboard, memory, barrier, divergence). Distinct from capacity. |
| `occupancy` | \(O=W_{\text{active}}/W_{\max}\) | Fraction of max resident warps that are resident; F8. |
| `theoretical_occupancy` | same \(O\) from launch + footprints | Occupancy implied by resource limits (F1–F8). Achieved occupancy would be `MEASURED` and is out of scope; do not report it. |
| `latency_hiding` | \(W_{\text{active}}\ge W_{\text{need}}\) | Covering \(L_{\text{issue}}\) by switching warps (F9). May fail when fusion lowers \(O\). |
| `wave_quantization` | \(N_{\text{waves}},\eta_{\text{wave}}\) | Last wave of CTAs may underfill the device (F10). |

**Memory hierarchy** (capacity vs bandwidth vs latency must be named as distinct resources).

| ID | Symbol | Must define |
| --- | --- | --- |
| `registers` | per-thread register file slice | Fastest operand storage; private; sized by \(R_t\). |
| `shared_memory` | per-CTA scratch | Software-managed SM memory; capacity \(C_{\text{cta}}\) counts toward \(C_{\text{smem}}\). |
| `shared_memory_banks` | \(N_{\text{bank}}=32\) | Banked organization; `OBSERVED` published identity. |
| `l1` | per-SM cache | Hardware cache in front of L2/HBM; capacity/bandwidth `UNKNOWN`. Not a substitute for shared memory. |
| `l2` | device-wide cache | Shared among SMs; capacity/bandwidth `UNKNOWN`. |
| `hbm` | device global / HBM | Off-SM device memory; backing store for global loads/stores; bandwidth \(\Beta\) `UNKNOWN`. |
| `host_pinned` | page-locked host memory | Host staging for DMA; not device HBM; bandwidth/latency `UNKNOWN`. |
| `local_memory` | per-thread spill space | Lives in device memory; used when registers spill. |
| `capacity` | size (bytes or registers) | A distinct resource from bandwidth and latency. |
| `bandwidth` | bytes per second | A distinct resource from capacity and latency. |
| `latency_resource` | time per request | A distinct resource from capacity and bandwidth; feeds \(L_{\text{issue}}\). |

**Access patterns**

| ID | Symbol | Must define |
| --- | --- | --- |
| `coalescing` | warp-global transaction packing | Consecutive aligned addresses from a warp collapse transactions (`OBSERVED` identity 5). |
| `alignment` | address multiple of request size | Misalignment increases transactions; state it separately from coalescing. |
| `bank_conflicts` | multi-lane same-bank shared access | \(N\) distinct 32-bit words in one bank serialize \(N\)-way; broadcast of one word is not a conflict. |

**Resource limiters**

| ID | Symbol | Must define |
| --- | --- | --- |
| `registers_per_thread` | \(R_t\) | Static register footprint of the compiled kernel per thread. |
| `shared_mem_per_cta` | \(C_{\text{cta}}\) | Dynamic + static shared memory per CTA (bytes). |
| `threads_per_cta` | \(T_{\text{cta}}\) | Block size; must be a multiple of \(N_w\) to avoid wasted warp lanes (F7 still uses \(\lceil\cdot\rceil\)). |
| `ctas_per_sm` | \(B_{\text{SM}}\) | Resident CTAs per SM after all limiters (F6). |
| `warp_slots` | \(W_{\max}\) | Hardware warp residency slots per SM; `UNKNOWN` magnitude. |
| `barrier_slots` | \(N_{\text{bar}}\) | Hardware CTA-barrier slots; a CTA using `__syncthreads` consumes a slot; `UNKNOWN` magnitude. |
| `register_spilling` | spill to `local_memory` | When \(R_t\) exceeds what occupancy math can admit, extra live values go to local memory and raise \(L_{\text{issue}}\). |
| `littles_law` | \(W_{\text{need}}=N_{\text{sched}}\cdot L_{\text{issue}}\) | Independent warps needed to hide \(L_{\text{issue}}\) at one issue per scheduler per cycle (F9). |

**Synchronization** (each row must state **what is ordered**).

| ID | Symbol | Must define (ordering) |
| --- | --- | --- |
| `syncthreads` | `__syncthreads` / CTA barrier | All participating threads of **one CTA** reach the same barrier; subsequent shared-memory (and CUDA-defined global) accesses in that CTA see prior writes. Does **not** order other CTAs, other grids, or host-visible system memory. |
| `warp_sync` | `__syncwarp` / warp reconverge | Orders lanes of **one warp** only. Not a CTA barrier. Implicit reconvergence after divergence is warp-scoped. |
| `memory_fence` | `__threadfence_block` / `__threadfence` / `__threadfence_system` | Makes a thread’s writes visible to other threads in the CTA / device / system respectively. A fence is **not** a wait-for-others barrier. |
| `stream` | CUDA stream | FIFO of kernels and memcopies **on that stream**. Different streams are concurrent unless an event or other sync joins them. |
| `event` | CUDA event | Records a point in a stream; waiting on it orders another stream or the host after that point. |
| `cooperative_groups` | CG hierarchy | Conceptual groups: thread, warp, CTA, grid, and optional cluster. Grid-wide sync requires a cooperative launch; support is `UNKNOWN`. Do not assume grid sync is available. |
| `async_copy_pipeline` | commit/wait copy groups | Stages global→shared transfers against compute. Completion waits order copy visibility into shared memory. Capability `async_copy_cap` is `UNKNOWN`. |

**Instruction pipelines** (conceptual families, not kernels).

| ID | Symbol | Must define |
| --- | --- | --- |
| `fma` | scalar fused multiply-add | Arithmetic pipeline for \(d=a\cdot b+c\) on the scalar datapath; peak \(\Pi_{\text{FMA}}\) `UNKNOWN`. |
| `ffma` | FP FMA encoding | Same pipeline family as `fma` for floating dtypes; name the PTX-level FMA/FFMA family without picking a dtype mix. |
| `load_store` | memory pipeline | Global/shared/local ld/st; consumes bandwidth and latency of the addressed level. |
| `tensor_core_mma` | MMA pipeline | Matrix-multiply-accumulate pipeline with shape/dtype constraints; legal set `mma_shapes` is `UNKNOWN`. Distinct from scalar `fma`. Do not pick `mma` vs `wgmma` vs later PTX variants. |
| `async_gmem_to_smem` | async global→shared copy | Copy datapath overlapping compute; may be absent. |
| `tma` | Tensor Memory Accelerator | Optional dedicated copy/descriptor path. Availability is `UNKNOWN` (`async_copy_cap` may be `TMA`). Do not assume TMA exists. |

**Fusion versus occupancy / evaluation**

| ID | Symbol | Must define |
| --- | --- | --- |
| `occupancy_min` | F1 | \(O\) is the min of the four limiter occupancies. |
| `fusion_footprint` | \(R_f,C_f\) | Fused live ranges are at least the per-kernel maxima and typically larger (F11, F12). |
| `arithmetic_intensity` | \(I=F/B\) | FLOPs per byte at a named level (HBM unless stated). Evaluation criterion, not a measured point. |
| `roofline` | \(\Pi\le\min(\Pi_{\text{peak}},I\cdot\Beta)\) | Upper bound used as a TASK-17 criterion. \(\Pi_{\text{peak}}\) and \(\Beta\) stay `UNKNOWN`. |

### Formulae (lock F1–F14)

Number them `(F1)` … `(F14)` in the prose. Each **required substring** below must appear **outside** the JSON fence, character-for-character (inside `$...$` or `$$...$$` is required). Surrounding math spaces other than those in the substring are allowed only outside the substring.

**Register/shared allocation (prose next to F2/F3, DERIVED):**

\[
R_{\text{cta}}=G_{\text{reg}}\Bigl\lceil\frac{R_t\cdot T_{\text{cta}}}{G_{\text{reg}}}\Bigr\rceil,\qquad
C_{\text{alloc}}=G_{\text{smem}}\Bigl\lceil\frac{C_{\text{cta}}}{G_{\text{smem}}}\Bigr\rceil
\]

Implementation must include this granularity rounding. Because \(G_{\text{reg}}\) and \(G_{\text{smem}}\) are `UNKNOWN`, leave them symbolic.

| Tag | `formula_ids` entry | Required substring |
| --- | --- | --- |
| (F1) | `occupancy_min` | `O = \min(O_\text{reg}, O_\text{smem}, O_\text{threads}, O_\text{cta})` |
| (F2) | `limiter_reg` | `B_\text{reg} = \lfloor S_\text{reg} / R_\text{cta} \rfloor` |
| (F3) | `limiter_smem` | `B_\text{smem} = \lfloor C_\text{smem} / C_\text{alloc} \rfloor` |
| (F4) | `limiter_threads` | `B_\text{threads} = \lfloor T_\max / T_\text{cta} \rfloor` |
| (F5) | `limiter_cta` | `B_\text{cta} = B_\max` |
| (F6) | `resident_cta` | `B_\text{SM} = \min(B_\text{reg}, B_\text{smem}, B_\text{threads}, B_\text{cta})` |
| (F7) | `warps_per_cta` | `W_\text{cta} = \lceil T_\text{cta} / N_w \rceil` |
| (F8) | `occupancy_warps` | `O = W_\text{active} / W_\max` |
| (F9) | `littles_law` | `W_\text{need} = N_\text{sched} \cdot L_\text{issue}` |
| (F10) | `wave_quant` | `N_\text{waves} = \lceil N_\text{grid} / (N_\text{SM} \cdot B_\text{SM}) \rceil` |
| (F11) | `fusion_footprint` | `R_f \ge \max(R_1, R_2)` |
| (F12) | `fusion_smem` | `C_f \ge \max(C_1, C_2)` |
| (F13) | `arithmetic_intensity` | `I = F / B` |
| (F14) | `roofline` | `\Pi \le \min(\Pi_\text{peak}, I \cdot \Beta)` |

**Required companion identities in prose (DERIVED; not extra formula IDs):**

- \(W_{\text{active}}=B_{\text{SM}}\cdot W_{\text{cta}}\) next to F8.
- Component occupancies \(O_{\text{reg}}=\min(1,B_{\text{reg}}W_{\text{cta}}/W_{\max})\) and likewise for smem, threads, and cta, so F1 is the min of those four.
- Last-wave efficiency \(\eta_{\text{wave}}=N_{\text{grid}}/(N_{\text{waves}}\cdot N_{\text{SM}}\cdot B_{\text{SM}})\) next to F10.
- Hiding test: hiding is complete only if \(W_{\text{active}}\ge W_{\text{need}}\).
- Typical fused footprints when live ranges do not overlap: \(R_f\approx R_1+R_2-R_{\cap}\), \(C_f\approx C_1+C_2-C_{\cap}\) with \(R_{\cap},C_{\cap}\ge 0\).
- Fused intensity \(I_f=(F_1+F_2)/(B_1+B_2-B_{\text{int}})\) where \(B_{\text{int}}\) is the intermediate tensor traffic avoided (store+load of the intermediate in the unfused pair). Must state that if \(O_f\) falls so \(W_{\text{active},f}<W_{\text{need}}\), issue slots idle and wall time **may** worsen despite larger \(I_f\). That implication is `DERIVED` from F1+F9+F13, not a winner claim.

Place F1–F6 and the component-\(O\) identities in **Resource limiters**. Place F7–F10 in **Occupancy and latency hiding** (F7/F8 may be restated in Resource limiters; the required substrings need appear at least once outside the JSON fence). Place F11–F13 in **Fusion versus occupancy**. Place F14 in **Evaluation criteria**.

### Fusion versus occupancy (the tradeoff TASK-17 consumes)

Work a **symbolic** two-kernel example with no numeric SKU fill-in:

1. Unfused launches \(K_1,K_2\) with footprints \((R_1,C_1,T_{\text{cta},1})\) and \((R_2,C_2,T_{\text{cta},2})\), work \(F_1,F_2\), HBM bytes \(B_1,B_2\), intermediate bytes \(B_{\text{int}}\) written by \(K_1\) and read by \(K_2\).
2. Fused \(K_f\) with \(R_f\ge\max(R_1,R_2)\), \(C_f\ge\max(C_1,C_2)\) (F11, F12), \(F_f=F_1+F_2\), \(B_f=B_1+B_2-B_{\text{int}}\).
3. Compute \(O_1,O_2,O_f\) from F1–F8 using SKU symbols. State \(O_f\le\min(O_1,O_2)\) is **not** always true (block size may change) but **is** the expected direction when fusion adds live registers/shared memory at fixed \(T_{\text{cta}}\).
4. Intensity \(I_f\ge I_{\text{unfused}}\) when \(B_{\text{int}}>0\).
5. Roofline (F14) may therefore show a higher bound for \(K_f\), while F9 may show \(W_{\text{active},f}<W_{\text{need}}\) so the bound is not approachable.

Do not conclude which side wins. Do not pick a fusion policy.

### Evaluation criteria (lock the TASK-17 checklist)

Numbered list, this order. Each item is a **criterion**, not a performed mapping:

1. Occupancy \(O\) from F1–F8 given hypothesized \(R_t,C_{\text{cta}},T_{\text{cta}}\).
2. Latency-hiding test \(W_{\text{active}}\ge W_{\text{need}}\) (F9) with \(L_{\text{issue}}\) left symbolic or later-measured.
3. Wave quantization \(\eta_{\text{wave}}\) (F10) for the hypothesized grid.
4. Arithmetic intensity \(I\) (F13) versus roofline bound (F14); \(\Pi_{\text{peak}}\) is \(\Pi_{\text{FMA}}\) or \(\Pi_{\text{TC}}\) according to the hypothesized pipeline mix, both `UNKNOWN` here.
5. Synchronization class: none / `warp_sync` / `syncthreads` / grid-cooperative / `stream`+`event`.
6. Pipeline mix: `fma`/`ffma` vs `tensor_core_mma` vs `load_store` vs `async_gmem_to_smem`/`tma`.
7. Fusion candidate: sign of \(\Delta I\) versus sign of \(\Delta O\) (and whether F9 still holds).

TASK-17 instantiates these. This document does not.

### Tables versus prose

- **Tables:** SKU symbols; all 54 concept rows; evaluation checklist may be a numbered list rather than a table.
- **Prose:** Authority scope; what each sync primitive orders (short paragraph per ID is required, not only the table cell); fusion worked example; deferred SKU policy.
- **Math:** F1–F14 as displayed equations with tags `(F1)` … `(F14)`.

### Tooling

Create `scripts/check_cuda_hardware_model.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`; typed annotations; Google docstrings). No torch, CUDA Python, uv, Ruff, pytest, or import of other `scripts/check_*.py`.

CLI (cwd = repository root):

```text
python3 scripts/check_cuda_hardware_model.py [--json] [--check PATH]
```

Behavior:

- Build the catalog object (schema below). All fields are locked constants (no `config.json`, no GPU query).
- `--json`: print that object to stdout (pretty-printed `json.dumps(..., indent=2)` plus a trailing newline; key order = schema order); run internal asserts; exit 0.
- `--check PATH`: require PATH to exist as a file (else exit 1 with `hardware model file not found: …`). Then require:
  1. Level-2 headings (`^## `) **exactly** the 13 strings in `headings`, **in that order**, and no extras.
  2. The first fenced `json` block (` ```json ` … ` ``` `) parses equal to the live object (deep equality).
  3. After removing that first json fence from the search text: every `concept_ids` entry appears as a backtick-wrapped token; every `formula_ids` tag `(F1)` … `(F14)` appears; every `formula_substrings` entry appears; every `sku_unknown_symbols` entry appears; `warp_size` identity `N_w = 32` or `N_w=32` appears; the three canonical sentences appear verbatim.
  4. None of the placeholder tokens `TBD`, `TODO`, `???` (case-sensitive).
  5. None of the forbidden name tokens, matched case-insensitively as substrings: `qwen`, `quartz`, `llama.cpp`, `ggml`, `gguf`, `opt-`.
  6. None of the forbidden path tokens, matched as substrings: `.cu`, `.cuh`.
  7. None of the tokens `MEASURED` or `HYPOTHESIS` (case-sensitive).
  Exit 1 with a readable list of mismatches on any failure.
- No flags: same as `--json` (print object, exit 0), matching peer checkers.
- `--json` and `--check` together: check first, then print JSON on success.
- No exit 2 path (there is no missing-config gate). Internal assert failure: exit 1.

Internal asserts (fail closed): `warp_size == 32`, `n_headings == 13 == len(headings)`, `n_concepts == 54 == len(concept_ids)`, `n_formulae == 14 == len(formula_ids) == len(formula_substrings)`, `concept_ids` unique and equal the locked tuple, `sku_policy` equals `parameterized_unknown_until_measured_table`.

### Machine-checkable catalog JSON schema

Top-level keys in **this exact order** (all required):

| Key | Value |
| --- | --- |
| `authority` | string, exactly `docs/architecture/plan.md` |
| `deliverable` | string, exactly `docs/architecture/cuda-hardware-model.md` |
| `sku_policy` | string, exactly `parameterized_unknown_until_measured_table` |
| `warp_size` | int `32` |
| `n_bank` | int `32` |
| `n_headings` | int `13` |
| `headings` | array of the 13 heading strings below |
| `n_concepts` | int `54` |
| `concept_ids` | array of the 54 IDs in catalog order below |
| `n_formulae` | int `14` |
| `formula_ids` | array of the 14 ids in F1–F14 order |
| `formula_substrings` | array of the 14 required substrings in F1–F14 order |
| `sku_unknown_symbols` | array of the 17 JSON ids listed in SKU symbols |
| `forbidden_tokens` | array exactly `["qwen","quartz","llama.cpp","ggml","gguf","opt-",".cu",".cuh","TBD","TODO","???","MEASURED","HYPOTHESIS"]` |
| `canonical_sku_sentence` | the verbatim two-sentence SKU string |
| `canonical_fusion_sentence` | `Fusion versus occupancy algebra is an evaluation criterion, not a selected mapping.` |
| `canonical_winner_sentence` | `This document selects no CUDA mapping winner.` |

`headings` exact strings:

`Authority`, `SKU parameterization`, `Execution hierarchy`, `Occupancy and latency hiding`, `Memory hierarchy`, `Access patterns`, `Resource limiters`, `Synchronization`, `Instruction pipelines`, `Fusion versus occupancy`, `Evaluation criteria`, `Deferred SKU table`, `Machine-checkable catalog`

`concept_ids` exact order:

`thread`, `warp`, `warp_size`, `simt_divergence`, `cta`, `block`, `grid`, `sm`, `scheduler`, `stall`, `occupancy`, `theoretical_occupancy`, `latency_hiding`, `wave_quantization`, `thread_block_cluster`, `registers`, `shared_memory`, `shared_memory_banks`, `l1`, `l2`, `hbm`, `host_pinned`, `local_memory`, `coalescing`, `alignment`, `bank_conflicts`, `capacity`, `bandwidth`, `latency_resource`, `registers_per_thread`, `shared_mem_per_cta`, `threads_per_cta`, `ctas_per_sm`, `warp_slots`, `barrier_slots`, `register_spilling`, `littles_law`, `syncthreads`, `warp_sync`, `memory_fence`, `stream`, `event`, `cooperative_groups`, `async_copy_pipeline`, `fma`, `ffma`, `load_store`, `tensor_core_mma`, `async_gmem_to_smem`, `tma`, `occupancy_min`, `fusion_footprint`, `arithmetic_intensity`, `roofline`

`formula_ids` exact order:

`occupancy_min`, `limiter_reg`, `limiter_smem`, `limiter_threads`, `limiter_cta`, `resident_cta`, `warps_per_cta`, `occupancy_warps`, `littles_law`, `wave_quant`, `fusion_footprint`, `fusion_smem`, `arithmetic_intensity`, `roofline`

### Stage split

- **Implementation** writes `scripts/check_cuda_hardware_model.py` **and** `docs/architecture/cuda-hardware-model.md`. Runs `py_compile`, `--json`, and `--check` of the markdown after the document exists. Embeds the JSON fence from that live `--json` run. Records command outcomes in this dossier. Does not commit.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner before the first `##`, Authority links to this dossier / `plan.md` evidence policy / checker path, heading/JSON fence consistency. Must not change locked IDs, formulae, SKU policy, or canonical sentences. Does not invent mappings. Does not edit `plan.md` or the ledger.
- **Verification** independently re-runs the focused commands, reads the document against this dossier (all 54 concepts, F1–F14, SKU-UNKNOWN closure, no kernel inspection), and confirms no `plan.md` edit.
- **Delivery** marks TASK-16 `DONE` after a passing verification.

- Invariants:
  - Model-independent: no model-equation, inventory, or dataflow content.
  - Warp size 32 is the only sitting-numeric constant besides bank count 32, both programming-guide `OBSERVED`.
  - Occupancy \(O=\min(O_{\text{reg}},O_{\text{smem}},O_{\text{threads}},O_{\text{cta}})\).
  - Fusion may lower \(O\) enough to fail F9 even when \(I\) rises.
  - Sitting-SKU limits and TMA/MMA-shape/cluster availability remain `UNKNOWN`.
  - No `MEASURED` numbers; no winning fusion policy.
  - Checker is stdlib-only; JSON fence matches live `--json`.
- Rejected alternatives:
  - Filling symbols from one GPU datasheet or `deviceQuery`: rejected; sitting limits are `UNKNOWN` and the catalog is parameterized.
  - Copying production `OPT-*` occupancy or SASS: forbidden by plan.md; not design authority.
  - Deferring the concept catalog until TASK-17: rejected; ledger requires coverage here.
  - Performing TASK-17 mappings in this document: rejected; wrong dependency and would pick winners without measurements.
  - Microbenchmarks of occupancy or bandwidth: TASK-19.
  - Treating higher arithmetic intensity as a sufficient fusion win: rejected; F9 can fail.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01/02/03.
  - `--config` on the checker: rejected; there is no model config authority for this task.
  - Inspecting `*.cu` kernels to “confirm” the hardware model: forbidden.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/cuda-hardware-model.md` exists and follows the 13-heading list above.
  - All 54 catalog concepts are defined (execution, memory, resources, synchronization, instructions, fusion/eval).
  - Fusion versus occupancy is explained with F1–F14, including the explicit algebraic tradeoff that fused footprints can drop occupancy and fail latency hiding.
  - Ledger open question is closed as `UNKNOWN` pending a future measured SKU table; symbols and evaluation criteria are supplied; no sitting-GPU numbers invented.
  - No current model/engine kernel inspection; forbidden tokens absent from the hardware-model markdown.
  - JSON fence matches a live `--json` object.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_cuda_hardware_model.py` only (no pytest fixtures).
- Focused commands (repository root):

```sh
python3 -m py_compile scripts/check_cuda_hardware_model.py
python3 scripts/check_cuda_hardware_model.py --json
python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md
test -f docs/architecture/cuda-hardware-model.md
```

- Candidate quality: not required — no model execution or NLL; instruction-only hardware vocabulary.
- Repository-wide commands:

```sh
test -f docs/architecture/cuda-hardware-model.md
python3 -m py_compile scripts/check_cuda_hardware_model.py
python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (documentation plus stdlib checker; no GPU run).
- Documentation/evidence updates:
  - `docs/architecture/cuda-hardware-model.md` (create)
  - `scripts/check_cuda_hardware_model.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged)
- Definition of done: hardware-model document published with locked catalog, SKU parameterization, fusion/occupancy algebra, and JSON fence that verifies against live `--json`; ledger TASK-16 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high`
- UTC/time/tokens/cost: `2026-09-20T09:47:15Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-16.md`. Coupled IDs none. Locked 13 headings, 54-concept catalog, 14 formulae, SKU-UNKNOWN policy (open question closed as deferred measured table), stdlib checker CLI `--json` / `--check`, and acceptance commands. `docs/architecture/cuda-hardware-model.md` not written in this stage.
- Performance evidence applied: N/A (instruction-only hardware vocabulary; no timing)

### Implementation

- Agent/model: `cursor-grok-4.6-high`
- Changes:
  - created `scripts/check_cuda_hardware_model.py` (Python 3.11+ stdlib checker: argparse/json/re/sys/pathlib; `--json` / `--check PATH`; locked catalog; no import of other `scripts/check_*.py`; no GPU query)
  - created `docs/architecture/cuda-hardware-model.md` (title `# CUDA hardware model`; 13 locked `##` headings; 54 backtick-wrapped concept IDs; F1–F14 required substrings; SKU / fusion / winner sentences verbatim; JSON fence from live `--json`)
  - did not edit `docs/architecture/plan.md` or `docs/architecture/task_ledger.md`; no commit
- Commands (repository root):
  - `python3 -m py_compile scripts/check_cuda_hardware_model.py` — **pass** (exit 0; empty stdout/stderr)
  - `python3 scripts/check_cuda_hardware_model.py --json` — **pass** (exit 0). Salient object: `authority` `docs/architecture/plan.md`, `deliverable` `docs/architecture/cuda-hardware-model.md`, `sku_policy` `parameterized_unknown_until_measured_table`, `warp_size` 32, `n_bank` 32, `n_headings` 13, `n_concepts` 54, `n_formulae` 14, 17 `sku_unknown_symbols`. Pretty-printed JSON (indent 2, schema key order) embedded as the sole fenced `json` block in the hardware-model document.
  - `python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md` — **pass** (exit 0; headings, JSON deep equality, concept backticks, `(F1)`–`(F14)`, formula substrings, SKU symbols, warp-size identity, three canonical sentences, no placeholder/forbidden/MEASURED/HYPOTHESIS tokens outside the JSON fence)
  - `test -f docs/architecture/cuda-hardware-model.md` — **pass** (exit 0)
- UTC/time/tokens/cost: `2026-09-20T09:56:53Z`; `telemetry_unavailable`
- Commit: not created (implementation stage)

### Documentation

- Agent/model: requested `composer-2.5` was unavailable on this runtime; actually assigned parent inherit (`cursor-grok-4.6-high`)
- Changes and evidence:
  - `docs/architecture/cuda-hardware-model.md` — added draft-status banner (`unverified`) as a blockquote before the first `##`; Authority table and prose now link the TASK-16 dossier, `plan.md` evidence-and-notation policy, and `scripts/check_cuda_hardware_model.py`. Locked IDs, formulae, SKU policy, canonical sentences, and JSON fence content unchanged. No mappings, SKU numbers, or winners added. Performance evidence: N/A (instruction-only vocabulary; no timing claims).
- Commands:
  - `python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md` — **pass** (exit 0; banner is not a `##` heading; 13 headings, JSON fence, catalog IDs, formulae, and forbidden-token checks still hold)
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T09:59:02Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: requested `composer-2.5` unavailable; actually assigned parent inherit (`cursor-grok-4.6-high`); `telemetry_unavailable`
- Diff review:
  - `docs/architecture/plan.md`: unchanged (`git diff --exit-code` exit 0).
  - TASK-16 increment: untracked `docs/architecture/cuda-hardware-model.md`, `scripts/check_cuda_hardware_model.py`, `docs/architecture/tasks/TASK-16.md`; ledger TASK-16 `TODO` → `IN PROGRESS` only (completion checkboxes still open; status not marked `DONE`).
  - Checker is stdlib-only (`argparse` / `json` / `re` / `sys` / `pathlib`); does not import other `scripts/check_*.py`; `config.json` appears only as a negative docstring claim.
  - Worktree also has parallel TASK-04/TASK-05 artifacts (`docs/architecture/lifetime-and-state.md`, `docs/architecture/tasks/TASK-04.md`, `docs/architecture/tasks/TASK-05.md`, `scripts/check_lifetime_and_state.py`, `scripts/analyze_bf16_tensors.py`) and matching ledger `IN PROGRESS` flips. Those files are outside this increment and were not used as TASK-16 evidence.
- Independent raw-record checks (did not take implementation/documentation word for it):
  - Parsed the markdown JSON fence and compared the object to a fresh `python3 scripts/check_cuda_hardware_model.py --json` run: deep-equal; fence payload is byte-identical to live stdout (3816 bytes). Live object: `authority` `docs/architecture/plan.md`, `deliverable` `docs/architecture/cuda-hardware-model.md`, `sku_policy` `parameterized_unknown_until_measured_table`, `warp_size` 32, `n_bank` 32, `n_headings` 13, `n_concepts` 54, `n_formulae` 14, 17 `sku_unknown_symbols`, schema key order locked.
  - Level-2 headings: exactly the 13 locked strings in locked order; title `# CUDA hardware model`; `unverified` banner is a blockquote before the first `##`.
  - All 54 locked `concept_ids` appear as backtick-wrapped tokens outside the JSON fence.
  - `(F1)`–`(F14)` tags and all 14 required formula substrings present outside the fence. F7–F10 displayed in Occupancy; F1–F6 displayed in Resource limiters; F11–F13 displayed in Fusion; F14 displayed in Evaluation.
  - Three canonical sentences present verbatim under SKU parameterization / Fusion versus occupancy / Evaluation criteria.
  - SKU table: numeric constants only \(N_w=32\) and \(N_{\text{bank}}=32\) (`OBSERVED`); remaining 17 JSON ids stay `UNKNOWN`. Independent hunt for datasheet/SKU numbers (A100/H100/65536/…) in the body: none.
  - Fusion versus occupancy: algebraic two-kernel example with F11/F12 footprints, \(I_f\) vs unfused, and explicit `DERIVED` implication that if \(O_f\) falls so \(W_{\text{active},f}<W_{\text{need}}\), wall time **may** worsen despite larger \(I_f\); canonical fusion/winner sentences; no mapping winner.
  - Forbidden tokens `qwen` / `quartz` / `llama.cpp` / `ggml` / `gguf` / `opt-` / `.cu` / `.cuh` / `TBD` / `TODO` / `???` / `MEASURED` / `HYPOTHESIS`: absent from the hardware-model body (they appear only inside the JSON fence as the catalog `forbidden_tokens` list). No kernel-inspection artifacts in the TASK-16 file set.
  - Model-independent: hardware-model body has no model-equation / inventory / dataflow content; quantization/SASS/semantic-node mentions are out-of-scope exclusions only.
  - Performance evidence: N/A (instruction-only; no timing invented).
  - Documentation draft: conclusions labelled unverified; not prematurely marked verified. Candidate quality: not required.
- Commands (repository root):
  - Focused `python3 -m py_compile scripts/check_cuda_hardware_model.py` — **pass** (exit 0; empty stdout/stderr)
  - Focused `python3 scripts/check_cuda_hardware_model.py --json` — **pass** (exit 0). Salient: pretty-printed catalog, `n_headings` 13, `n_concepts` 54, `n_formulae` 14, `warp_size` 32, `n_bank` 32, 17 `sku_unknown_symbols`, `sku_policy` `parameterized_unknown_until_measured_table`
  - Focused `python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md` — **pass** (exit 0; empty stdout/stderr)
  - Focused `test -f docs/architecture/cuda-hardware-model.md` — **pass** (exit 0)
  - Repository-wide `test -f docs/architecture/cuda-hardware-model.md` — **pass** (exit 0)
  - Repository-wide `python3 -m py_compile scripts/check_cuda_hardware_model.py` — **pass** (exit 0)
  - Repository-wide `python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md` — **pass** (exit 0)
  - Native/CUDA/hardware gates: not applicable. Ruff / pytest / CMake / CUDA: not run (dossier forbids).
- Formatting changed files: none
- Verdict: `pass` — locked catalog, F1–F14 fusion/occupancy algebra, SKU-UNKNOWN closure, JSON fence identity, and repository boundaries hold; `plan.md` unchanged; TASK-16 not marked DONE.
- UTC/time/tokens/cost: `2026-09-20T10:02:12Z`; `telemetry_unavailable`

### Delivery

- Agent/model: requested `composer-2.5` was unavailable; actually assigned parent inherit (`cursor-grok-4.6-high`); `telemetry_unavailable`
- Scope: TASK-16 only; parallel TASK-04/TASK-05 artifacts left unstaged
- Outcome: TASK-16 marked `DONE` after verification pass (attempt 1, verdict `pass`)
- UTC/time/tokens/cost: `2026-09-20T10:06:09Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1, verdict `pass`) — hardware-model document with 13 headings, 54-concept catalog, F1–F14 fusion/occupancy algebra, SKU-UNKNOWN parameterization, stdlib checker + JSON fence. Open question closed: sitting-GPU limits and instruction availability remain UNKNOWN until a future measured SKU table; symbols and TASK-17 evaluation criteria are supplied.
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: N/A (no performance-evidence checks)
- Throughput delta: N/A — TASK-16 does not execute or time a model
- Commit: Publish CUDA hardware model reference
- Push: `origin/clean-sheet` (pending)
- First-pass acceptance: yes
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: none identified; sitting-SKU numeric limits remain UNKNOWN until a future measured table
