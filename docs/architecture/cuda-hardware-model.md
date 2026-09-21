# CUDA hardware model

> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.

Model-independent CUDA resource and tradeoff reference: a parameterized vocabulary of execution, memory, occupancy limiters, synchronization, and instruction pipelines, plus algebraic fusion-versus-occupancy relations. This document specifies hardware concepts, not a kernel.

## Authority

| Item | Value | Label |
| --- | --- | --- |
| Dossier | [`docs/architecture/tasks/TASK-16.md`](tasks/TASK-16.md) | — |
| Evidence policy | [`docs/architecture/plan.md`](plan.md#evidence-and-notation-policy) (DERIVED / OBSERVED / UNKNOWN labels) | OBSERVED / DERIVED / UNKNOWN |
| Checker | [`scripts/check_cuda_hardware_model.py`](../../scripts/check_cuda_hardware_model.py) | DERIVED |
| Scope | Model-independent CUDA resource vocabulary and fusion/occupancy algebra | — |

**In scope.** Execution hierarchy, memory hierarchy, access patterns, resource limiters, synchronization, instruction-pipeline families, and the algebraic fusion-versus-occupancy relations TASK-17 will instantiate.

**Out of scope.** Semantic-node CUDA mappings, ownership or reduction choices, kernel source, launch configurations, SASS, microbenchmarks, device-property dumps, occupancy measurements, layout, quantization, graph, and compiler work. No winning fusion policy, winning MMA shape, or winning occupancy target.

Allowed evidence is the public CUDA C++ Programming Guide, PTX ISA, occupancy, and memory-hierarchy identities, plus the [TASK-16 dossier](tasks/TASK-16.md). [`plan.md`](plan.md#evidence-and-notation-policy) is the evidence-policy authority. Catalog identity is checked by [`scripts/check_cuda_hardware_model.py`](../../scripts/check_cuda_hardware_model.py). If a secondary blog or SKU datasheet disagrees with a programming-guide occupancy or barrier identity, the programming-guide identity wins.

Evidence labels used here:

- `OBSERVED` — only the six published identities listed below.
- `DERIVED` — occupancy/fusion/Little’s-law/roofline algebra, limiter mins, and wave quantization.
- `UNKNOWN` — sitting-SKU numeric limits and optional instruction/capability availability.

The six `OBSERVED` published identities:

1. Warp size $N_w = 32$ on the CUDA devices this study considers.
2. Occupancy is the ratio of active resident warps on an SM to that SM’s maximum resident warps.
3. A CTA (thread block) is the domain of `__syncthreads`.
4. Shared memory is banked; the programming-guide identity is $N_{\text{bank}}=32$ banks of 32-bit words.
5. Consecutive threads of a warp accessing consecutive aligned addresses in global memory coalesce into the minimum number of transactions for that request size.
6. Occupancy is jointly limited by registers per SM, shared memory per SM, threads per SM, and maximum CTAs per SM.

## SKU parameterization

Target-GPU-specific limits and instruction availability remain UNKNOWN until a future measured SKU table. This document supplies the symbols and evaluation criteria TASK-17 will instantiate; it does not record sitting-device numbers and does not claim a winning fusion policy.

Warp size is the sole numeric constant treated as `OBSERVED` besides bank count: $N_w = 32$. Shared-memory bank count $N_{\text{bank}}=32$ is the other published identity. Every other SKU symbol stays `UNKNOWN`. Do not instantiate any `UNKNOWN` symbol with a datasheet or sitting-GPU number.

| Symbol | JSON id | Meaning | Status |
| --- | --- | --- | --- |
| $N_w$ | `N_w` | threads per warp | `OBSERVED` $=32$ |
| $N_{\text{bank}}$ | `N_bank` | shared-memory banks | `OBSERVED` $=32$ |
| $N_{\text{SM}}$ | `N_SM` | SM count | `UNKNOWN` |
| $W_{\max}$ | `W_max` | max resident warps per SM | `UNKNOWN` |
| $S_{\text{reg}}$ | `S_reg` | 32-bit register-file size per SM | `UNKNOWN` |
| $C_{\text{smem}}$ | `C_smem` | shared-memory capacity per SM (bytes) | `UNKNOWN` |
| $T_{\max}$ | `T_max` | max threads per SM | `UNKNOWN` |
| $B_{\max}$ | `B_max` | max CTAs per SM | `UNKNOWN` |
| $N_{\text{bar}}$ | `N_bar` | hardware barrier slots per SM | `UNKNOWN` |
| $N_{\text{sched}}$ | `N_sched` | warp schedulers per SM | `UNKNOWN` |
| $G_{\text{reg}}$ | `G_reg` | register allocation granularity (32-bit regs) | `UNKNOWN` |
| $G_{\text{smem}}$ | `G_smem` | shared-memory allocation granularity (bytes) | `UNKNOWN` |
| $\Beta$ | `Beta_HBM` | device-global (HBM) peak bandwidth | `UNKNOWN` |
| $\Pi_{\text{FMA}}$ | `Pi_FMA` | scalar FMA peak throughput | `UNKNOWN` |
| $\Pi_{\text{TC}}$ | `Pi_TC` | tensor-core MMA peak throughput | `UNKNOWN` |
| $L_{\text{issue}}$ | `L_issue` | dominant stall latency in scheduler cycles | `UNKNOWN` |
| $\texttt{async\_copy\_cap}$ | `async_copy_cap` | `{absent, cp.async, TMA}` | `UNKNOWN` |
| $\texttt{mma\_shapes}$ | `mma_shapes` | legal MMA $(M,N,K)$ and dtypes | `UNKNOWN` |
| $\texttt{cluster\_cap}$ | `cluster_cap` | thread-block cluster availability | `UNKNOWN` |

JSON field `sku_unknown_symbols` is exactly the JSON ids above excluding `N_w` and `N_bank`.

## Execution hierarchy

A kernel launch maps a `grid` of CTAs onto the device. Each `cta` (synonym `block`) is the software-managed sharing and `__syncthreads` domain. Threads of a CTA are grouped into warps of width `warp_size`. An `sm` holds the register file, shared memory, and warp `scheduler` that issue ready warps. A `thread_block_cluster` is optional and must not be assumed present.

| ID | Symbol | Must define |
| --- | --- | --- |
| `thread` | scalar CUDA thread | Programmable scalar lane; has private registers and a thread index in the CTA. |
| `warp` | warp of $N_w$ threads | SIMT scheduling unit issued together. |
| `warp_size` | $N_w=32$ | Constant warp width; `OBSERVED` published identity. |
| `simt_divergence` | diverged warp | Taken/not-taken paths in a warp serialize; reconvergence is warp-scoped, not a CTA barrier. |
| `cta` | cooperative thread array | Launch unit that shares shared memory and `__syncthreads`; synonym of block. |
| `block` | thread block | Alias of `cta`; both IDs must appear; they name the same object. |
| `grid` | $N_{\text{grid}}$ CTAs | The kernel launch; CTAs of one grid do not share a CTA barrier. |
| `sm` | streaming multiprocessor | Occupancy, register file, shared memory, and scheduler domain. |
| `scheduler` | warp scheduler | Selects ready warps on an SM; count $N_{\text{sched}}$ (`N_sched`) is `UNKNOWN`. |
| `thread_block_cluster` | optional cluster | Multi-CTA grouping with optional distributed shared memory; availability `cluster_cap` is `UNKNOWN`. Do not assume it exists. |

SIMT `simt_divergence`: lanes of one `warp` share an instruction stream. When a branch splits the warp, taken and not-taken paths serialize. Reconvergence is warp-scoped. It is not a `cta` barrier and does not order other warps.

## Occupancy and latency hiding

`occupancy` $O$ is the fraction of an SM’s maximum resident warps that are resident (`OBSERVED` identity 2). `theoretical_occupancy` is that same $O$ implied by launch geometry and resource footprints (F1–F8). Achieved occupancy is out of scope and is not reported. A `stall` is a not-issuable warp, distinct from capacity. `latency_hiding` tests whether enough resident warps cover $L_{\text{issue}}$. `wave_quantization` describes last-wave underfill of the device.

| ID | Symbol | Must define |
| --- | --- | --- |
| `stall` | scheduler stall | A warp is not issuable (scoreboard, memory, barrier, divergence). Distinct from capacity. |
| `occupancy` | $O=W_{\text{active}}/W_{\max}$ | Fraction of max resident warps that are resident; F8. |
| `theoretical_occupancy` | same $O$ from launch + footprints | Occupancy implied by resource limits (F1–F8). Achieved occupancy is out of scope; do not report it. |
| `latency_hiding` | $W_{\text{active}}\ge W_{\text{need}}$ | Covering $L_{\text{issue}}$ by switching warps (F9). May fail when fusion lowers $O$. |
| `wave_quantization` | $N_{\text{waves}},\eta_{\text{wave}}$ | Last wave of CTAs may underfill the device (F10). |

Warps per CTA (F7) still uses $\lceil\cdot\rceil$ even when $T_{\text{cta}}$ is a multiple of $N_w$:

$$
W_\text{cta} = \lceil T_\text{cta} / N_w \rceil
$$

(F7)

Occupancy from resident warps (F8), `OBSERVED` as a ratio and `DERIVED` in this algebraic form:

$$
O = W_\text{active} / W_\max
$$

(F8)

Companion identity (`DERIVED`) next to F8: $W_{\text{active}}=B_{\text{SM}}\cdot W_{\text{cta}}$.

Little’s-law hiding test (F9), `DERIVED`:

$$
W_\text{need} = N_\text{sched} \cdot L_\text{issue}
$$

(F9)

Hiding is complete only if $W_{\text{active}}\ge W_{\text{need}}$. $N_{\text{sched}}$ (`N_sched`) and $L_{\text{issue}}$ (`L_issue`) remain `UNKNOWN`.

Wave quantization (F10), `DERIVED`:

$$
N_\text{waves} = \lceil N_\text{grid} / (N_\text{SM} \cdot B_\text{SM}) \rceil
$$

(F10)

Last-wave efficiency (`DERIVED`) next to F10: $\eta_{\text{wave}}=N_{\text{grid}}/(N_{\text{waves}}\cdot N_{\text{SM}}\cdot B_{\text{SM}})$. $N_{\text{SM}}$ (`N_SM`) remains `UNKNOWN`.

## Memory hierarchy

Registers, shared memory, L1, L2, device global / HBM, host pinned memory, and local/spill space are distinct levels. `capacity`, `bandwidth`, and `latency_resource` are distinct resources: size, bytes per second, and time per request. Do not substitute one for another. L1 is not a substitute for shared memory. Host pinned memory is not device HBM.

| ID | Symbol | Must define |
| --- | --- | --- |
| `registers` | per-thread register file slice | Fastest operand storage; private; sized by $R_t$. |
| `shared_memory` | per-CTA scratch | Software-managed SM memory; capacity $C_{\text{cta}}$ counts toward $C_{\text{smem}}$. |
| `shared_memory_banks` | $N_{\text{bank}}=32$ | Banked organization; `OBSERVED` published identity. |
| `l1` | per-SM cache | Hardware cache in front of L2/HBM; capacity/bandwidth `UNKNOWN`. Not a substitute for shared memory. |
| `l2` | device-wide cache | Shared among SMs; capacity/bandwidth `UNKNOWN`. |
| `hbm` | device global / HBM | Off-SM device memory; backing store for global loads/stores; bandwidth $\Beta$ (`Beta_HBM`) `UNKNOWN`. |
| `host_pinned` | page-locked host memory | Host staging for DMA; not device HBM; bandwidth/latency `UNKNOWN`. |
| `local_memory` | per-thread spill space | Lives in device memory; used when registers spill. |
| `capacity` | size (bytes or registers) | A distinct resource from bandwidth and latency. |
| `bandwidth` | bytes per second | A distinct resource from capacity and latency. |
| `latency_resource` | time per request | A distinct resource from capacity and bandwidth; feeds $L_{\text{issue}}$. |

Shared-memory capacity per SM is $C_{\text{smem}}$ (`C_smem`), `UNKNOWN`. Register-file size per SM is $S_{\text{reg}}$ (`S_reg`), `UNKNOWN`.

## Access patterns

Coalescing, alignment, and bank conflicts are separate. Alignment is not coalescing. Broadcast of one shared-memory word is not a bank conflict.

| ID | Symbol | Must define |
| --- | --- | --- |
| `coalescing` | warp-global transaction packing | Consecutive aligned addresses from a warp collapse transactions (`OBSERVED` identity 5). |
| `alignment` | address multiple of request size | Misalignment increases transactions; state it separately from coalescing. |
| `bank_conflicts` | multi-lane same-bank shared access | $N$ distinct 32-bit words in one bank serialize $N$-way; broadcast of one word is not a conflict. |

`OBSERVED` identity 5: consecutive threads of a warp accessing consecutive aligned addresses in global memory coalesce into the minimum number of transactions for that request size. Misaligned addresses raise the transaction count even when the warp’s addresses are consecutive. On shared memory, $N_{\text{bank}}=32$ banks of 32-bit words (`OBSERVED` identity 4): $N$ distinct 32-bit words that map to one bank serialize $N$-way.

## Resource limiters

Occupancy is jointly limited by registers per SM, shared memory per SM, threads per SM, and maximum CTAs per SM (`OBSERVED` identity 6). The limiter formulae below are `DERIVED`. Granularities $G_{\text{reg}}$ (`G_reg`) and $G_{\text{smem}}$ (`G_smem`) stay symbolic (`UNKNOWN`).

| ID | Symbol | Must define |
| --- | --- | --- |
| `registers_per_thread` | $R_t$ | Static register footprint of the compiled kernel per thread. |
| `shared_mem_per_cta` | $C_{\text{cta}}$ | Dynamic + static shared memory per CTA (bytes). |
| `threads_per_cta` | $T_{\text{cta}}$ | Block size; must be a multiple of $N_w$ to avoid wasted warp lanes (F7 still uses $\lceil\cdot\rceil$). |
| `ctas_per_sm` | $B_{\text{SM}}$ | Resident CTAs per SM after all limiters (F6). |
| `warp_slots` | $W_{\max}$ | Hardware warp residency slots per SM; `UNKNOWN` magnitude (`W_max`). |
| `barrier_slots` | $N_{\text{bar}}$ | Hardware CTA-barrier slots; a CTA using `__syncthreads` consumes a slot; `UNKNOWN` magnitude (`N_bar`). |
| `register_spilling` | spill to `local_memory` | When $R_t$ exceeds what occupancy math can admit, extra live values go to local memory and raise $L_{\text{issue}}$. |
| `littles_law` | $W_{\text{need}}=N_{\text{sched}}\cdot L_{\text{issue}}$ | Independent warps needed to hide $L_{\text{issue}}$ at one issue per scheduler per cycle (F9). |

Register and shared-memory allocations are rounded up to hardware granularities (`DERIVED`):

$$
R_{\text{cta}}=G_{\text{reg}}\Bigl\lceil\frac{R_t\cdot T_{\text{cta}}}{G_{\text{reg}}}\Bigr\rceil,\qquad
C_{\text{alloc}}=G_{\text{smem}}\Bigl\lceil\frac{C_{\text{cta}}}{G_{\text{smem}}}\Bigr\rceil
$$

Register limiter (F2):

$$
B_\text{reg} = \lfloor S_\text{reg} / R_\text{cta} \rfloor
$$

(F2)

Shared-memory limiter (F3):

$$
B_\text{smem} = \lfloor C_\text{smem} / C_\text{alloc} \rfloor
$$

(F3)

Thread limiter (F4); $T_{\max}$ (`T_max`) is `UNKNOWN`:

$$
B_\text{threads} = \lfloor T_\max / T_\text{cta} \rfloor
$$

(F4)

CTA-slot limiter (F5); $B_{\max}$ (`B_max`) is `UNKNOWN`:

$$
B_\text{cta} = B_\max
$$

(F5)

Resident CTAs per SM (F6):

$$
B_\text{warp}=\lfloor W_\max/W_\text{cta}\rfloor,\qquad
B_\text{SM} = \min(B_\text{reg}, B_\text{smem}, B_\text{threads}, B_\text{cta}, B_\text{warp})
$$

(F6)

Component occupancies (`DERIVED`) so that F1 is the min of four limiter occupancies:

$O_{\text{reg}}=\min(1,B_{\text{reg}}W_{\text{cta}}/W_{\max})$, and likewise $O_{\text{smem}}$, $O_{\text{threads}}$, and $O_{\text{cta}}$.

Occupancy min (F1), `DERIVED` (the joint-limit identity is `OBSERVED`; this min is algebra):

$$
O = \min(O_\text{reg}, O_\text{smem}, O_\text{threads}, O_\text{cta})
$$

(F1)

F7 and F8 from the occupancy section also apply: $W_{\text{cta}}$ from F7 and $O=W_{\text{active}}/W_{\max}$ from F8 with $W_{\text{active}}=B_{\text{SM}}\cdot W_{\text{cta}}$. `occupancy_min` names F1.

## Synchronization

Each primitive below orders a stated domain. A fence is not a wait-for-others barrier. Grid-wide sync requires a cooperative launch; support is `UNKNOWN`.

| ID | Symbol | Must define (ordering) |
| --- | --- | --- |
| `syncthreads` | `__syncthreads` / CTA barrier | All participating threads of **one CTA** reach the same barrier; subsequent shared-memory (and CUDA-defined global) accesses in that CTA see prior writes. Does **not** order other CTAs, other grids, or host-visible system memory. |
| `warp_sync` | `__syncwarp` / warp reconverge | Orders lanes of **one warp** only. Not a CTA barrier. Implicit reconvergence after divergence is warp-scoped. |
| `memory_fence` | `__threadfence_block` / `__threadfence` / `__threadfence_system` | Makes a thread’s writes visible to other threads in the CTA / device / system respectively. A fence is **not** a wait-for-others barrier. |
| `stream` | CUDA stream | FIFO of kernels and memcopies **on that stream**. Different streams are concurrent unless an event or other sync joins them. |
| `event` | CUDA event | Records a point in a stream; waiting on it orders another stream or the host after that point. |
| `cooperative_groups` | CG hierarchy | Conceptual groups: thread, warp, CTA, grid, and optional cluster. Grid-wide sync requires a cooperative launch; support is `UNKNOWN`. Do not assume grid sync is available. |
| `async_copy_pipeline` | commit/wait copy groups | Stages global→shared transfers against compute. Completion waits order copy visibility into shared memory. Capability `async_copy_cap` is `UNKNOWN`. |

`syncthreads` (`OBSERVED` identity 3: a CTA is the domain of `__syncthreads`). All participating `thread`s of **one** `cta` reach the same barrier. Subsequent shared-memory accesses, and CUDA-defined global accesses, in that CTA see prior writes by threads of that CTA. The barrier does not order other CTAs, other grids, or host-visible system memory.

`warp_sync` orders lanes of **one** `warp` only (`__syncwarp` and implicit reconvergence after `simt_divergence`). It is not a CTA barrier and does not replace `syncthreads`.

`memory_fence` makes one thread’s writes visible at a stated scope: CTA (`__threadfence_block`), device (`__threadfence`), or system (`__threadfence_system`). Visibility is not a wait: other threads are not required to reach the fence. A fence is not a barrier.

`stream`: kernels and memcopies enqueued on one CUDA stream execute in FIFO order on that stream. Different streams are concurrent unless joined by an `event` or other synchronization.

`event`: recording an event marks a point in a `stream`. Waiting on that event orders another stream or the host after that point. An event does not, by itself, order unsynchronized streams.

`cooperative_groups` name conceptual groups: thread, warp, CTA, grid, and optional cluster (`cluster_cap` `UNKNOWN`). Grid-wide synchronization requires a cooperative launch; that support is `UNKNOWN`. Do not assume grid sync is available.

`async_copy_pipeline` stages global-to-shared transfers against compute. Commit/wait on copy groups orders copy visibility into `shared_memory`. Whether the datapath exists is `async_copy_cap` (`UNKNOWN`): `{absent, cp.async, TMA}`.

## Instruction pipelines

These are conceptual families, not kernels. Peak throughputs and legal MMA shapes stay `UNKNOWN`. Do not pick `mma` versus later PTX variants. Do not assume TMA exists.

| ID | Symbol | Must define |
| --- | --- | --- |
| `fma` | scalar fused multiply-add | Arithmetic pipeline for $d=a\cdot b+c$ on the scalar datapath; peak $\Pi_{\text{FMA}}$ (`Pi_FMA`) `UNKNOWN`. |
| `ffma` | FP FMA encoding | Same pipeline family as `fma` for floating dtypes; name the PTX-level FMA/FFMA family without picking a dtype mix. |
| `load_store` | memory pipeline | Global/shared/local ld/st; consumes bandwidth and latency of the addressed level. |
| `tensor_core_mma` | MMA pipeline | Matrix-multiply-accumulate pipeline with shape/dtype constraints; legal set `mma_shapes` is `UNKNOWN`. Distinct from scalar `fma`. Do not pick mma vs wgmma vs later PTX variants. |
| `async_gmem_to_smem` | async global→shared copy | Copy datapath overlapping compute; may be absent. |
| `tma` | Tensor Memory Accelerator | Optional dedicated copy/descriptor path. Availability is `UNKNOWN` (`async_copy_cap` may be `TMA`). Do not assume TMA exists. |

`fma` / `ffma` share the scalar fused multiply-add family. `tensor_core_mma` is a distinct MMA pipeline whose legal $(M,N,K)$ and dtypes are `mma_shapes` (`UNKNOWN`). `load_store` consumes the `bandwidth` and `latency_resource` of the addressed level. `async_gmem_to_smem` and `tma` are optional copy paths gated by `async_copy_cap` (`UNKNOWN`).

## Fusion versus occupancy

Fusion versus occupancy algebra is an evaluation criterion, not a selected mapping.

Fusing kernels raises register and shared-memory live ranges. Occupancy $O$ may fall. Latency hiding (F9) may fail even if bytes or FLOPs per launch improve. That implication is `DERIVED` from F1+F9+F13, not a winner claim.

| ID | Symbol | Must define |
| --- | --- | --- |
| `occupancy_min` | F1 | $O$ is the min of the four limiter occupancies. |
| `fusion_footprint` | $R_f,C_f$ | Fused live ranges are at least the per-kernel maxima and typically larger (F11, F12). |
| `arithmetic_intensity` | $I=F/B$ | FLOPs per byte at a named level (HBM unless stated). Evaluation criterion, not a measured point. |
| `roofline` | $\Pi\le\min(\Pi_{\text{peak}},I\cdot\Beta)$ | Upper bound used as a TASK-17 criterion. $\Pi_{\text{peak}}$ and $\Beta$ stay `UNKNOWN`. |

Register allocation is SKU-scoped:
\(R_\text{alloc,cta}=A_\text{reg}(R_t,T_\text{cta},G_\text{reg},\text{scope}_\text{reg})\),
and \(B_\text{reg}=\lfloor S_\text{reg}/R_\text{alloc,cta}\rfloor\).
Both allocation granularity and `scope_reg` are `UNKNOWN` until SKU fill.

Fused register footprint (F11), `DERIVED` only when constituent live
allocations remain live under the same accounting scope:

$$
R_f \ge \max(R_1, R_2)
$$

(F11)

Otherwise compiler recomputation or lifetime changes require re-deriving the
fused footprint; F11 is not universal. Fused shared-memory footprint (F12) is
likewise `DERIVED` only when constituent allocations remain live under the same
accounting scope:

$$
C_f \ge \max(C_1, C_2)
$$

(F12)

Otherwise the fused shared-memory footprint is re-derived; F12 is not universal.

Arithmetic intensity (F13), `DERIVED`:

$$
I = F / B
$$

(F13)

Typical fused footprints when live ranges do not overlap (`DERIVED`): $R_f\approx R_1+R_2-R_{\cap}$, $C_f\approx C_1+C_2-C_{\cap}$ with $R_{\cap},C_{\cap}\ge 0$. Fused intensity $I_f=(F_1+F_2)/(B_1+B_2-B_{\text{int}})$ where $B_{\text{int}}$ is the intermediate tensor traffic avoided (store+load of the intermediate in the unfused pair). If $O_f$ falls so $W_{\text{active},f}<W_{\text{need}}$, issue slots idle and wall time **may** worsen despite larger $I_f$.

**Symbolic two-kernel example** (`DERIVED`; no SKU fill-in).

1. Unfused launches $K_1,K_2$ with footprints $(R_1,C_1,T_{\text{cta},1})$ and $(R_2,C_2,T_{\text{cta},2})$, work $F_1,F_2$, HBM bytes $B_1,B_2$, and intermediate bytes $B_{\text{int}}$ written by $K_1$ and read by $K_2$.
2. Fused $K_f$ with $R_f\ge\max(R_1,R_2)$, $C_f\ge\max(C_1,C_2)$ (F11, F12), $F_f=F_1+F_2$, $B_f=B_1+B_2-B_{\text{int}}$.
3. Compute $O_1,O_2,O_f$ from F1–F8 using SKU symbols. $O_f\le\min(O_1,O_2)$ is **not** always true (block size may change) but **is** the expected direction when fusion adds live registers/shared memory at fixed $T_{\text{cta}}$.
4. Intensity $I_f\ge I_{\text{unfused}}$ when $B_{\text{int}}>0$.
5. Roofline (F14) may therefore show a higher bound for $K_f$, while F9 may show $W_{\text{active},f}<W_{\text{need}}$ so the bound is not approachable.

This example selects no side and picks no fusion policy. TASK-17 instantiates the symbols.

## Evaluation criteria

This document selects no CUDA mapping winner.

Numbered TASK-17 instantiation checklist. Each item is a **criterion**, not a performed mapping. TASK-17 instantiates these. This document does not.

1. Occupancy $O$ from F1–F8 given hypothesized $R_t,C_{\text{cta}},T_{\text{cta}}$.
2. Latency-hiding test $W_{\text{active}}\ge W_{\text{need}}$ (F9) with $L_{\text{issue}}$ left symbolic or later-measured.
3. Wave quantization $\eta_{\text{wave}}$ (F10) for the hypothesized grid.
4. Arithmetic intensity $I$ (F13) versus roofline bound (F14); $\Pi_{\text{peak}}$ is $\Pi_{\text{FMA}}$ or $\Pi_{\text{TC}}$ according to the hypothesized pipeline mix, both `UNKNOWN` here.
5. Synchronization class: none / `warp_sync` / `syncthreads` / grid-cooperative / `stream`+`event`.
6. Pipeline mix: `fma`/`ffma` vs `tensor_core_mma` vs `load_store` vs `async_gmem_to_smem`/`tma`.
7. Fusion candidate: sign of $\Delta I$ versus sign of $\Delta O$ (and whether F9 still holds).

Roofline (F14) is an upper-bound **criterion**, not a measured point (`DERIVED`). $\Pi_{\text{peak}}$ and $\Beta$ (`Pi_FMA` / `Pi_TC` / `Beta_HBM`) stay `UNKNOWN`:

$$
\Pi \le \min(\Pi_\text{peak}, I \cdot \Beta)
$$

(F14)

## Deferred SKU table

The ledger open question (target-GPU-specific limits and instruction availability) is closed here as `UNKNOWN` pending a future measured SKU table. This document supplies symbols and evaluation criteria only. Sitting-device fill-in in this document is forbidden.

| JSON id | Symbol | What stays UNKNOWN |
| --- | --- | --- |
| `N_SM` | $N_{\text{SM}}$ | SM count |
| `W_max` | $W_{\max}$ | max resident warps per SM |
| `S_reg` | $S_{\text{reg}}$ | 32-bit register-file size per SM |
| `C_smem` | $C_{\text{smem}}$ | shared-memory capacity per SM |
| `T_max` | $T_{\max}$ | max threads per SM |
| `B_max` | $B_{\max}$ | max CTAs per SM |
| `N_bar` | $N_{\text{bar}}$ | hardware barrier slots per SM |
| `N_sched` | $N_{\text{sched}}$ | warp schedulers per SM |
| `G_reg` | $G_{\text{reg}}$ | register allocation granularity |
| `G_smem` | $G_{\text{smem}}$ | shared-memory allocation granularity |
| `Beta_HBM` | $\Beta$ | device-global (HBM) peak bandwidth |
| `Pi_FMA` | $\Pi_{\text{FMA}}$ | scalar FMA peak throughput |
| `Pi_TC` | $\Pi_{\text{TC}}$ | tensor-core MMA peak throughput |
| `L_issue` | $L_{\text{issue}}$ | dominant stall latency in scheduler cycles |
| `async_copy_cap` | $\texttt{async\_copy\_cap}$ | `{absent, cp.async, TMA}` |
| `mma_shapes` | $\texttt{mma\_shapes}$ | legal MMA $(M,N,K)$ and dtypes |
| `cluster_cap` | $\texttt{cluster\_cap}$ | thread-block cluster availability |

Published identities that are **not** sitting-SKU unknowns: $N_w=32$ and $N_{\text{bank}}=32$.

## Machine-checkable catalog

Live catalog object from `python3 scripts/check_cuda_hardware_model.py --json`:

```json
{
  "authority": "docs/architecture/plan.md",
  "deliverable": "docs/architecture/cuda-hardware-model.md",
  "sku_policy": "parameterized_unknown_until_measured_table",
  "warp_size": 32,
  "n_bank": 32,
  "n_headings": 13,
  "headings": [
    "Authority",
    "SKU parameterization",
    "Execution hierarchy",
    "Occupancy and latency hiding",
    "Memory hierarchy",
    "Access patterns",
    "Resource limiters",
    "Synchronization",
    "Instruction pipelines",
    "Fusion versus occupancy",
    "Evaluation criteria",
    "Deferred SKU table",
    "Machine-checkable catalog"
  ],
  "n_concepts": 54,
  "concept_ids": [
    "thread",
    "warp",
    "warp_size",
    "simt_divergence",
    "cta",
    "block",
    "grid",
    "sm",
    "scheduler",
    "stall",
    "occupancy",
    "theoretical_occupancy",
    "latency_hiding",
    "wave_quantization",
    "thread_block_cluster",
    "registers",
    "shared_memory",
    "shared_memory_banks",
    "l1",
    "l2",
    "hbm",
    "host_pinned",
    "local_memory",
    "coalescing",
    "alignment",
    "bank_conflicts",
    "capacity",
    "bandwidth",
    "latency_resource",
    "registers_per_thread",
    "shared_mem_per_cta",
    "threads_per_cta",
    "ctas_per_sm",
    "warp_slots",
    "barrier_slots",
    "register_spilling",
    "littles_law",
    "syncthreads",
    "warp_sync",
    "memory_fence",
    "stream",
    "event",
    "cooperative_groups",
    "async_copy_pipeline",
    "fma",
    "ffma",
    "load_store",
    "tensor_core_mma",
    "async_gmem_to_smem",
    "tma",
    "occupancy_min",
    "fusion_footprint",
    "arithmetic_intensity",
    "roofline"
  ],
  "n_formulae": 14,
  "formula_ids": [
    "occupancy_min",
    "limiter_reg",
    "limiter_smem",
    "limiter_threads",
    "limiter_cta",
    "resident_cta",
    "warps_per_cta",
    "occupancy_warps",
    "littles_law",
    "wave_quant",
    "fusion_footprint",
    "fusion_smem",
    "arithmetic_intensity",
    "roofline"
  ],
  "formula_substrings": [
    "O = \\min(O_\\text{reg}, O_\\text{smem}, O_\\text{threads}, O_\\text{cta})",
    "B_\\text{reg}=\\lfloor S_\\text{reg}/R_\\text{alloc,cta}\\rfloor",
    "B_\\text{smem} = \\lfloor C_\\text{smem} / C_\\text{alloc} \\rfloor",
    "B_\\text{threads} = \\lfloor T_\\max / T_\\text{cta} \\rfloor",
    "B_\\text{cta} = B_\\max",
    "B_\\text{SM} = \\min(B_\\text{reg}, B_\\text{smem}, B_\\text{threads}, B_\\text{cta}, B_\\text{warp})",
    "W_\\text{cta} = \\lceil T_\\text{cta} / N_w \\rceil",
    "O = W_\\text{active} / W_\\max",
    "W_\\text{need} = N_\\text{sched} \\cdot L_\\text{issue}",
    "N_\\text{waves} = \\lceil N_\\text{grid} / (N_\\text{SM} \\cdot B_\\text{SM}) \\rceil",
    "R_f \\ge \\max(R_1, R_2)",
    "C_f \\ge \\max(C_1, C_2)",
    "I = F / B",
    "\\Pi \\le \\min(\\Pi_\\text{peak}, I \\cdot \\Beta)"
  ],
  "sku_unknown_symbols": [
    "N_SM",
    "W_max",
    "S_reg",
    "C_smem",
    "T_max",
    "B_max",
    "N_bar",
    "N_sched",
    "G_reg",
    "G_smem",
    "Beta_HBM",
    "Pi_FMA",
    "Pi_TC",
    "L_issue",
    "async_copy_cap",
    "mma_shapes",
    "cluster_cap"
  ],
  "forbidden_tokens": [
    "qwen",
    "quartz",
    "llama.cpp",
    "ggml",
    "gguf",
    "opt-",
    ".cu",
    ".cuh",
    "TBD",
    "TODO",
    "???",
    "MEASURED",
    "HYPOTHESIS"
  ],
  "canonical_sku_sentence": "Target-GPU-specific limits and instruction availability remain UNKNOWN until a future measured SKU table. This document supplies the symbols and evaluation criteria TASK-17 will instantiate; it does not record sitting-device numbers and does not claim a winning fusion policy.",
  "canonical_fusion_sentence": "Fusion versus occupancy algebra is an evaluation criterion, not a selected mapping.",
  "canonical_winner_sentence": "This document selects no CUDA mapping winner."
}
```
