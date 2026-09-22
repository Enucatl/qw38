# TASK-06 — Derive theoretical computation and traffic lower bounds

## Control

- Primary ID: `TASK-06`
- Coupled IDs: `none`
- Dependencies: `TASK-02`, `TASK-03`, `TASK-04` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Derive symbolic and instantiated work counts; calculate unavoidable weight, state, and activation traffic; label bottleneck classifications as hypotheses.

## Goal and boundaries

Produce `docs/architecture/work-and-traffic.md` as the Phase 1 decode/prefill **mathematical work** and **irreducible traffic** analysis for the Qwen3.8-27B language + MTP forward map. Close the ledger open question with **DERIVED** MAC/FLOP counts and byte lower bounds, plus **HYPOTHESIS** bottleneck labels (not measurements).

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Equations, ranks, and operator definitions come from `docs/architecture/model-semantics.md` (TASK-02). Catalog IDs and region cuts come from `docs/architecture/dataflow.md` (TASK-03). Persistent-state bytes come from `docs/architecture/lifetime-and-state.md` (TASK-04). Weight byte totals instantiate TASK-01 inventory / sitting `text_config` shapes × BF16.
  - Label claims `OBSERVED` (config/inventory already established), `DERIVED` (MAC, bytes, identities from ranks), or `HYPOTHESIS` (bottleneck class vs an UNKNOWN SKU ridge). `UNKNOWN` only for vision-encoder internals deferred here.
  - GitHub Markdown math. Cite TASK-02 equation tags `(1)`–`(24)` and TASK-03 region / catalog IDs. Do not rewrite forward math, redraw the TASK-03 DAG, or recompute TASK-04 state ranks.
  - Allowed evidence: TASK-01 inventory, TASK-02 semantics, TASK-03 dataflow, TASK-04 lifetime/state, sitting `config.json` `text_config`, plan evidence vocabulary, and TASK-16 intensity identity $I=F/B$ (F13) as a **formula citation only** (SKU $\Pi_\text{peak}$ and $\Beta$ stay UNKNOWN). No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`.
- Non-goals:
  - No CUDA fusion, buffer reuse, or physical allocation (TASK-12). Activation bytes are catalog-rank materialization, not kernel traffic.
  - No semantic-graph execution contracts (TASK-11) or decode/prefill **schedules** (TASK-13/14).
  - No layouts, quantization winners, or kernel names (TASK-08/09/15/17).
  - No sitting-GPU roofline numbers, Nsight, or tok/s (TASK-16 SKU table / TASK-19).
  - No new operators, algebraic alternatives as extra work, or vision-encoder internals.
  - No peak-memory claim and no summed CUDA live-set.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–04). Evidence scripts under `scripts/` are not a Python package; do not apply uv/`src` layout to this increment.
  - Do not edit `docs/architecture/plan.md`, `model-semantics.md`, `dataflow.md`, `lifetime-and-state.md`, or `model-inventory.md`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib work/traffic checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-13` — central path: mathematics → dataflow → lifetime → (this task) work and traffic bounds before numerical/quantization research.
- `docs/architecture/plan.md:54-64` — evidence labels; hypotheses remain hypotheses.
- `docs/architecture/plan.md:85-86` — TASK-06 sets mathematical work and irreducible traffic lower bounds.
- `docs/architecture/task_ledger.md` TASK-06 row — produces `docs/architecture/work-and-traffic.md`; completion is symbolic+instantiated work, unavoidable weight/state/activation traffic, bottleneck labels as hypotheses. Open question: region-level arithmetic intensity and bottleneck hypotheses.
- `docs/architecture/model-semantics.md` — equations `(1)`–`(24)`; prefill = map from zeros; decode = same map with incoming state; GDN definition `(17)`–`(18)`; `(19)` is the same map, not extra work; optional MTP omission is an algebraic equivalent, not the primary complete map.
- `docs/architecture/dataflow.md` — 52 catalog IDs, region IDs `embed`, `full_attn`, `linear_attn`, `mlp`, `primary_logits`, `mtp`; logical ≠ physical sentence.
- `docs/architecture/lifetime-and-state.md` / TASK-04 dossier — $T$ = stored KV length after append; decode $T_\text{new}=1$; KV/C/S bytes; decode read/write table; “TASK-06 may” expand prefill triangular KV-read; no FLOP bounds there.
- `docs/architecture/model-inventory.md` — BF16 payload; language+MTP 27,320,697,856 parameters / 54,641,395,712 bytes; family byte totals below; vision excluded from primary.
- `docs/architecture/cuda-hardware-model.md` F13 — $I=F/B$; F14 roofline with UNKNOWN $\Pi_\text{peak},\Beta$. Cite the identity; do not fill SKU peaks.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` only. Do not read safetensor payloads.
- `scripts/check_lifetime_and_state.py` — checker-style precedent. TASK-06’s checker is a sibling; do not import it.

## Performance evidence

N/A — theoretical work and traffic documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Byte and MAC totals are DERIVED arithmetic. Bottleneck **labels** are HYPOTHESIS versus an UNKNOWN ridge, not MEASURED attribution. Do not apply the performance-evidence checklist to rank kernels.

## Implementation decisions

### Authority for work and traffic

If a MAC or byte total would disagree with a TASK-02 operator, TASK-03 catalog rank, TASK-04 state byte, or TASK-01 BF16 family total, the earlier document wins and this one is wrong.

- Prefill and decode share **one** operator set. Only $T$ and whether incoming $(K,V,C,S)$ is zeros versus populated change.
- Primary totals **include MTP** (one full-attention block + `mtp.fc` + second `lm_head` contraction). Language-only is a secondary row.
- Algebraic equivalents (chunkwise GDN, SDPA, fused eval, omitting MTP, skipping non-final `lm_head`) are **not** the primary map. A secondary `inference_last_logits` row may omit $T-1$ vocabulary projections.
- Do not inspect Quartz or llama.cpp to “confirm” FLOPs or traffic.

### Deliverable structure (`docs/architecture/work-and-traffic.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Bottleneck class columns are `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, semantics, dataflow, lifetime, inventory, config, checker; evidence labels; in-scope (language + MTP work + three traffic channels) vs deferred (vision encoder). State that the document specifies mathematical work and byte lower bounds, not kernels.
2. **Work convention** — MAC/FLOP lock, decode vs prefill, $T$ convention, complete vs language-only, both canonical sentences below plus the activation sentence.
3. **Symbolic work** — per-region MAC formulas citing `(1)`–`(24)`.
4. **Instantiated work** — $C$, $A$, $T=1$ and $T=4096$ tables; prefill identity; secondary last-logits row.
5. **Weight traffic** — BF16 inventory unique bytes; embed as gather; decode vs prefill.
6. **State traffic** — TASK-04 decode volumes; prefill triangular KV; store after prefill.
7. **Activation traffic** — forced vs region-cut vs GEMM-IO; catalog ranks; not fusion.
8. **Intensity and bottleneck hypotheses** — $I=F/B$ (TASK-16 F13); region table; labels as HYPOTHESIS.
9. **Deferred vision** — residual-stream interface only.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Work convention**, include these **three canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Bottleneck classifications in this document are hypotheses, not measurements.

> Activation traffic is a DERIVED minimum materialization from catalog ranks, not a CUDA fusion or buffer-reuse claim.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_hypothesis` = sentence 2; `canonical_sentence_activation` = sentence 3.

Bullets required under that heading:

- Mathematical work uses TASK-02 operator definitions; chunkwise GDN `(19)` as a dense $128^3$ multiply is rejected as a work count.
- Weight traffic uses TASK-01 BF16 inventory bytes (not GGUF Q4).
- State traffic uses TASK-04 persistent-state bytes.
- Fan-out ≠ must-store still holds; activation lower bounds are named catalog cuts, not a peak working set.

### MAC / FLOP convention (lock)

| Symbol | Meaning |
| --- | --- |
| MAC | One multiply-add in a **contraction**: GEMM / matvec $y=Wx$, depthwise conv tap-sum `(14)`, GDN rank-1 update + state reads `(17)`–`(18)`, and attention $QK^\top$ / $AV$ `(9)`. No-bias maps add no extra MAC. |
| FLOP (primary) | $F=2\times\text{MAC}$ for those contractions (multiply and add). This $F$ is the numerator of $I=F/B$. |
| Embed `(1)` | Gather; **0 MAC**. Cost is weight-gather traffic. |
| Elementwise | RMSNorm `(2)`–`(3)`, SiLU, residual add `(4)`–`(5)`, softmax, RoPE `(12)`, $\sigma$, $\alpha/\beta$ `(15)`, L2 `(16)` are **not** in primary MAC and **not** in $F$ for intensity. |
| Outer product into $S$ | Counted as MAC ($d_k d_v$ per head) because it accumulates into $S$. |

Primary work is contraction MAC (and $F=2\times\text{MAC}$). Elementwise is lower order: a documented ceiling of $10^8$ FLOP for one complete decode at $T=4096$ is enough to prove it is $<1\%$ of $2C_\text{complete}$. Do not publish a tight elementwise schedule; do not use elementwise for bottleneck labels.

GQA: attention MAC uses $n_h=24$ query heads, not $n_\text{kv}=4$. Repeat-interleave of GDN Q/K is 0 MAC. GDN work uses definition `(17)`–`(18)`: $3\,n_v^\ell d_k^\ell d_v^\ell$ MAC per token per linear layer. Do **not** count `(19)` as $n_v^\ell (d_k^\ell)^3$.

JSON: `flop_per_mac` = 2; `mac_embed` = 0; `elementwise_not_in_primary` = true; `elementwise_upper_bound_decode_T4096` = 100000000; `gdn_uses_rank1_eq_17_18` = true.

### Decode vs prefill and $T$ (lock)

Let $T$ be the stored KV length **after** appending the current token (TASK-04). Decode of one new token has incoming KV length $T-1$ and attention contractions against **length $T$** (current token included). When $T=1$, incoming KV is empty.

- **Decode** = one position of the complete (or language-only) map at stored length $T$.
- **Prefill** of length $T$ = the same map at $t=1,\ldots,T$ from zero state. Causal full attention at step $t$ contracts against length $t$.

Identity (DERIVED):

$$
W_\text{decode}(T)=C+A T,\qquad
W_\text{prefill}(T)=\sum_{t=1}^{T}(C+A t)=T C+A\frac{T(T+1)}{2}.
$$

At $T=1$, prefill = decode. JSON: `T_is_stored_length_after_append` true; `decode_T_new` 1; `example_T` `[1, 4096]`.

MTP KV uses the same $T$ (TASK-02/04). Primary prefill therefore runs the MTP block at all $T$ positions (teacher-forced $e_{t+1}$). Secondary **inference last-logits**: subtract $(T-1)$ times each evaluated `lm_head` contraction; mixers/MLP/MTP block still scale with $T$ because MTP KV storage has rank $T$.

JSON: `primary_includes_mtp` true.

### Symbolic work by region (lock)

Layer counts: $n_\text{lin}=48$, $n_\text{full}=16$, $n_\text{mtp}=1$, $L=64$. Instantiated GEMM MAC $=d_\text{out}d_\text{in}$ for $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$.

| Region | Eqs | MAC per token (T-free) | T coefficient $A$ (MAC / stored-length unit) |
| --- | --- | --- | --- |
| `embed` | (1) | 0 | 0 |
| `linear_attn` (one layer) | (13)–(20) | $W_\text{qkv}+W_z+W_a+W_b+W_\text{out}$ + $d_\text{qkv}k_\text{conv}$ + $3 n_v d_k d_v$ | 0 |
| `full_attn` (one layer) | (6)–(10),(12) | $W_q+W_k+W_v+W_o$ | $2 n_h d_h=12288$ (QK + AV) |
| `mlp` (one layer) | (21) | $3\,I H$ | 0 |
| `primary_logits` / `lm_head` | (22) | $V H$ | 0 |
| `mtp` extras | (23)–(24) plus one full+MLP | $H\cdot 2H + (W_q+W_k+W_v+W_o) + 3IH + VH$ | $12288$ (MTP attention) |

Instantiated per-layer MAC (must appear as these integers):

| Quantity | MAC | How |
| --- | ---: | --- |
| Full proj one layer | 104857600 | $12288\cdot5120+2\cdot1024\cdot5120+5120\cdot6144$ |
| Linear proj $W_\text{qkv,z,a,b}$ | 84377600 | $10240\cdot5120+6144\cdot5120+2\cdot48\cdot5120$ |
| Linear conv | 40960 | $10240\cdot 4$ |
| GDN `(17)`–`(18)` | 2359296 | $3\cdot48\cdot128\cdot128$ |
| Linear `out_proj` | 31457280 | $5120\cdot6144$ |
| Linear token total | 118235136 | sum of the four linear rows |
| MLP one layer | 267386880 | $3\cdot17408\cdot5120$ |
| `lm_head` or one embed-table GEMM | 1271398400 | $248320\cdot5120$ |
| `mtp.fc` | 52428800 | $5120\cdot10240$ |

Stack coefficients (primary complete = language + MTP):

$$
\begin{aligned}
C_\text{lin}&=48\cdot 118235136=5675286528,\\
C_\text{full}&=16\cdot 104857600=1677721600,\\
C_\text{mlp}&=64\cdot 267386880=17112760320,\\
C_\text{lm}&=1271398400,\\
C_\text{mtp}&=52428800+104857600+267386880+1271398400=1696071680,\\
C_\text{language}&=25737166848,\qquad
C_\text{complete}=27433238528,\\
A_\text{language}&=16\cdot 12288=196608,\qquad
A_\text{complete}=208896.
\end{aligned}
$$

$C_\text{language}=C_\text{lin}+C_\text{full}+C_\text{mlp}+C_\text{lm}$. $C_\text{complete}=C_\text{language}+C_\text{mtp}$. $A_\text{complete}=A_\text{language}+12288$.

Prefill causal attention uses exact $T(T+1)/2$, not $T^2/2$.

Secondary inference last-logits:

$$
W^\text{last}_\text{language}(T)=W_\text{language}(T)-(T-1)C_\text{lm},\qquad
W^\text{last}_\text{complete}(T)=W_\text{complete}(T)-2(T-1)C_\text{lm}.
$$

### Instantiated work (lock)

| Mode | $T=1$ MAC | $T=4096$ MAC |
| --- | ---: | ---: |
| Decode language | 25737363456 | 26542473216 |
| Decode complete | 27433447424 | 28288876544 |
| Prefill language | 25737363456 | 107069105504256 |
| Prefill complete | 27433447424 | 114119319486464 |
| Prefill last-logits language | 25737363456 | 101862729056256 |
| Prefill last-logits complete | 27433447424 | 103706566590464 |

Companion FLOP = $2\times$ MAC. JSON arrays `mac_decode_language_at_example_T` etc. follow `example_T` order. Checker must recompute $C+AT$ and $TC+AT(T+1)/2$ from config shapes, not copy literals without asserts.

### Weight traffic (lock)

Element size BF16 = 2 bytes OBSERVED. Unique parameter bytes = TASK-01 family totals (checker **recomputes** from `text_config` shapes × counts × 2 and must match these inventory integers):

| Family | BF16 bytes | Label |
| --- | ---: | --- |
| `language_linear_attn` | 11124102144 | OBSERVED / DERIVED from shapes |
| `language_self_attn` | 3355459584 | includes q/k norm |
| `language_mlp` | 34225520640 | |
| `language_layer_norms` | 1310720 | input+post, 64 each |
| `language_final_norm` | 10240 | |
| `lm_head` | 2542796800 | |
| `mtp` | 849398784 | |
| Language+MTP excl. vision | 54641395712 | |
| `embed` table (not streamed) | 2542796800 | gather instead |

Decode/prefill **do not** stream the full embed table. Gather one row = $H\cdot 2=10240$ bytes.

- Unique non-embed (language+MTP minus $E$): $52098598912$ bytes.
- Decode gather: language $10240$; complete $20480$ ($e_t$ and $e_{t+1}$).
- Prefill gather: language $T\cdot 10240$; complete $2T\cdot 10240$ (prompt ids $1\ldots T$ plus MTP next-ids $2\ldots T+1$).
- Irreducible unique+gather decode complete: $52098619392$ bytes.
- A second physical read of shared $W_\text{lm}$ for `logits_1` is **not** in the unique lower bound (same tensor). Label any double-read as HYPOTHESIS, not DERIVED traffic.
- Vision family bytes are excluded from primary. Do not use GGUF Q4 sizes.

JSON: `weight_bytes_*` fields below; `weight_gather_bytes_per_row` = 10240.

### State traffic (lock)

Copy TASK-04 coefficients; do not re-derive ranks. Primary includes MTP KV (17 full layers).

Decode ($T_\text{new}=1$):

| Channel | Write bytes | Read bytes |
| --- | ---: | --- |
| KV | 69632 | $69632(T-1)$ |
| $C$ | 983040 | 2949120 |
| $S$ (F32) | 150994944 | 150994944 |
| Total | 152047616 | $69632(T-1)+153944064$ |

Instantiated decode read: $T=1$ → $153944064$; $T=4096$ → $439087104$. Storage after append: $B_\text{store}(T)=69632T+153944064$ (`154013696` at $T=1$, `439156736` at $T=4096$).

Prefill from zeros (this task expands the triangular KV schedule TASK-04 deferred):

$$
B^\text{KV,read}_\text{prefill}(T)=69632\cdot\frac{T(T-1)}{2},\qquad
B^\text{KV,write}_\text{prefill}(T)=69632\,T.
$$

Instantiated KV read: $T=1$ → $0$; $T=4096$ → $583972945920$. KV write at $T=4096$ → $285212672$.

C/S mathematical per-step volumes scale as TASK-04 × $T$ (recurrent definition). Initial-zero reads move 0 physical bytes; still list the mathematical TASK-04 per-token numbers. Surviving store after prefill is $B_\text{store}(T)$, not $T\times B_S$.

Do not use chunkwise GDN to claim zero $S$ traffic; the definition is recurrent `(17)`.

### Activation traffic (lock)

BF16 activations (not $S$). Three DERIVED views; **primary reported bound is region-cut**. None is a CUDA live-set or fusion claim.

1. **Forced** (TASK-04 must-survive non-state): `h`, `h_mid`, `g`, `z`, `logits_0`, and for complete also MTP `h`/`h_mid`/`g`/`logits_1`.
   - Decode language: $64\cdot 10240\cdot 2 + 16\cdot 12288 + 48\cdot 12288 + 496640 = 2593792$ bytes (`h`+`h_mid` + `g` + `z` + `logits_0`).
   - Decode complete: $2593792 + 2\cdot 10240 + 12288 + 496640 = 3123200$.
2. **Region-cut** (primary): forced plus catalog IDs at TASK-03 region boundaries: `e`, `h_tilde`, `mix_lin`/`mix_full`, `h_post`, `mlp_out`, `h_64`, `h_final`, and MTP `e_next`, `mtp_u`, `h_mtp`.
   - Decode language: $5245952$ bytes.
   - Decode complete: $5847040$ bytes.
   - Prefill = $T$ × decode region-cut (per-position map). $T=4096$: language $21487419392$; complete $23949475840$.
3. **GEMM-IO** (intensity denominator only, not the materialization floor): for each $y=Wx$, count $(d_\text{in}+d_\text{out})\times 2$ bytes. Unfused; double-counts a residual vector that feeds several maps.
   - One linear layer: $96448$; one full proj: $81920$; one MLP: $135168$; `lm_head`: $506880$; `mtp.fc`: $30720$.
   - Decode language all GEMMs: $15097856$; complete: $15852544$.

Omit intra-region ephemerals (`qkv`, `attn`, `swiglu`, …) from views 1–2; TASK-12 may fuse them. Do not add GEMM-IO into region-cut. Softmax score matrices are not catalog IDs; their traffic is the KV **state** channel.

### Intensity and bottleneck hypotheses (lock)

Use TASK-16 F13 $I=F/B$ with $F=2\times\text{MAC}$ and $B=$ weight + (GEMM-IO or region-cut, named) + state bytes in that region. Ridge $I_\text{ridge}=\Pi_\text{peak}/\Beta$ is UNKNOWN (no SKU fill-in). Every **class** below is HYPOTHESIS. Exact $I$ identities that do not need a ridge are DERIVED.

Locked DERIVED intensities (decode, named $B$):

| Region | Identity | Value |
| --- | --- | ---: |
| MLP vs weights only | $2\cdot 267386880 / 534773760$ | $1$ exactly |
| `lm_head` vs weights only | $2\cdot 1271398400 / 2542796800$ | $1$ exactly |
| GDN vs $S$ read+write | $2\cdot 2359296 / (2\cdot 3145728)$ | $0.75$ exactly |
| Full-attn core vs KV read$(T-1)$+write | $2\cdot 12288 T / (4096 T)$ | $6$ exactly ($T\ge 1$) |

Locked HYPOTHESIS labels (`bottleneck_labels` JSON, this order):

| ID | Applies when | Claim (must remain HYPOTHESIS) |
| --- | --- | --- |
| `weight_memory` | Decode GEMM regions with $I\approx 1$ FLOP/byte (MLP, projections, `lm_head`) | Memory-bound on any SKU whose ridge $\gg 1$ |
| `vocab_memory` | `lm_head` unique 2542796800 B | Vocabulary projection is a decode weight-traffic outlier |
| `state_memory` | GDN $I=0.75$ vs $S$; C/S dominate TASK-04 decode bytes | Linear-attn state traffic, not MAC, is the linear-mixer limiter |
| `kv_memory` | Full-attn core $I=6$ vs KV | KV movement, not QK FLOPs, limits decode full-attn until a ridge is known |
| `quadratic_attn` | Prefill $A T(T+1)/2$ MAC and triangular KV read | Prefill full-attn is the only quadratic region; linear-attn stays linear in $T$ |
| `compute` | Prefill reuses weights across $T$ tokens, $I\sim T$ vs decode | Large-$T$ prefill GEMMs may be compute-bound if $T>I_\text{ridge}$ |

Do not rank these by wall time. Do not name CUDA kernels. Do not claim a winner quantization or fusion.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 8 (Intensity and bottleneck hypotheses). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers.

Required IDs **inside that fence**: `embed`, `linear_attn`, `full_attn`, `mlp`, `lm_head`, `mtp`, `weight`, `state`, `activation`.

JSON `n_diagrams` is 1. `region_ids` is `["embed","linear_attn","full_attn","mlp","lm_head","mtp"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger work and traffic are **UNKNOWN**. Do not add vision MAC or vision weight bytes to primary totals. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_work_and_traffic.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for shapes, layer counts, and MAC/byte products.

CLI (cwd = repository root):

```text
python3 scripts/check_work_and_traffic.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --work-traffic docs/architecture/work-and-traffic.md \
  [--json]
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`. Derived: all MAC $C,A$, example-$T$ instantiations, weight bytes from shapes×2, activation bytes, triangular KV. Constant fields: canonical sentences, `flop_per_mac`, `example_T`, `n_diagrams`, bottleneck label list, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--work-traffic PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all three canonical sentences verbatim, (5) every `catalog_ids` entry present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer below present as a decimal substring, (10) the six bottleneck label ids present as substrings. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality.

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `flop_per_mac==2`, `mac_embed==0`, `gdn_uses_rank1_eq_17_18 is True`
- `mac_full_proj_per_layer==104857600`, `mac_lin_token_per_layer==118235136`, `mac_mlp_per_layer==267386880`, `mac_lm_head==1271398400`
- `mac_C_language==25737166848`, `mac_C_complete==27433238528`, `mac_A_language==196608`, `mac_A_complete==208896`
- `mac_C_complete == mac_C_language + mac_C_mtp`
- for each `example_T[i]`: decode $=C+AT$, prefill $=TC+AT(T+1)/2$, last-logits formulas, storage/decode-read match TASK-04 integers, prefill KV read $=69632 T(T-1)/2$
- `weight_bytes_language_linear_attn==11124102144` and other family bytes in the schema (recomputed from shapes)
- `weight_bytes_language_mtp_excl_vision==54641395712`
- `i_mlp_weight_only==1`, `i_lm_head_weight_only==1`, `i_gdn_vs_s_rw==0.75`, `i_attn_core_vs_kv==6`
- `elementwise_upper_bound_decode_T4096 < 0.01 * (2 * mac_decode_complete_at_example_T[1])`
- `primary_includes_mtp is True`, `n_diagrams==1`
- `bottleneck_labels` equals the locked six-id list
- `text_config.dtype == "bfloat16"` (weights); do not treat `mamba_ssm_dtype` as activation dtype

Locked document integers the `--work-traffic` check must find:

`104857600`, `118235136`, `267386880`, `1271398400`, `52428800`, `2359296`, `40960`, `5675286528`, `1677721600`, `17112760320`, `1696071680`, `25737166848`, `27433238528`, `196608`, `208896`, `25737363456`, `27433447424`, `26542473216`, `28288876544`, `107069105504256`, `114119319486464`, `101862729056256`, `103706566590464`, `11124102144`, `3355459584`, `34225520640`, `54641395712`, `52098598912`, `52098619392`, `2542796800`, `10240`, `69632`, `152047616`, `153944064`, `439087104`, `154013696`, `439156736`, `583972945920`, `285212672`, `2593792`, `3123200`, `5245952`, `5847040`, `21487419392`, `23949475840`, `15097856`, `15852544`

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `n_attn_heads`, `n_kv_heads`, `head_dim`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_qkv_width`, `linear_z_width`, `linear_conv_kernel_dim`, `bytes_bf16`, `flop_per_mac`, `T_is_stored_length_after_append`, `decode_T_new`, `primary_includes_mtp`, `gdn_uses_rank1_eq_17_18`, `elementwise_not_in_primary`, `elementwise_upper_bound_decode_T4096`, `example_T`,

`mac_embed`, `mac_full_proj_per_layer`, `mac_attn_coeff_per_full_layer`, `mac_lin_proj_per_layer`, `mac_lin_conv_per_layer`, `mac_gdn_per_layer`, `mac_lin_out_per_layer`, `mac_lin_token_per_layer`, `mac_mlp_per_layer`, `mac_lm_head`, `mac_mtp_fc`, `mac_C_linear_attn`, `mac_C_full_attn`, `mac_C_mlp`, `mac_C_lm_head`, `mac_C_mtp`, `mac_C_language`, `mac_C_complete`, `mac_A_language`, `mac_A_complete`,

`mac_decode_language_at_example_T`, `mac_decode_complete_at_example_T`, `mac_prefill_language_at_example_T`, `mac_prefill_complete_at_example_T`, `mac_prefill_inference_last_logits_language_at_example_T`, `mac_prefill_inference_last_logits_complete_at_example_T`,

`weight_bytes_language_linear_attn`, `weight_bytes_language_self_attn`, `weight_bytes_language_mlp`, `weight_bytes_language_layer_norms`, `weight_bytes_language_final_norm`, `weight_bytes_lm_head`, `weight_bytes_mtp`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_embed_table`, `weight_bytes_unique_non_embed`, `weight_gather_bytes_per_row`, `weight_gather_bytes_decode_language`, `weight_gather_bytes_decode_complete`, `weight_unique_plus_gather_decode_complete`, `weight_gather_bytes_prefill_language_at_example_T`, `weight_gather_bytes_prefill_complete_at_example_T`,

`kv_bytes_all_per_token`, `c_bytes_all`, `c_write_bytes_per_token_all`, `s_bytes_all`, `storage_kv_bytes_coeff_T`, `storage_fixed_bytes`, `decode_write_bytes`, `decode_read_kv_bytes_coeff_Tm1`, `decode_read_fixed_bytes`, `storage_bytes_at_example_T`, `decode_read_bytes_at_example_T`, `prefill_kv_read_bytes_at_example_T`, `prefill_kv_write_bytes_at_example_T`,

`residual_vector_bytes`, `g_bytes_per_full_layer`, `z_bytes_per_linear_layer`, `logits_bytes`, `act_forced_decode_language_bytes`, `act_forced_decode_complete_bytes`, `act_region_cut_decode_language_bytes`, `act_region_cut_decode_complete_bytes`, `act_region_cut_prefill_language_at_example_T`, `act_region_cut_prefill_complete_at_example_T`, `act_gemm_io_lin_layer_bytes`, `act_gemm_io_full_proj_bytes`, `act_gemm_io_mlp_layer_bytes`, `act_gemm_io_lm_head_bytes`, `act_gemm_io_mtp_fc_bytes`, `act_gemm_io_decode_language_bytes`, `act_gemm_io_decode_complete_bytes`,

`i_mlp_weight_only`, `i_lm_head_weight_only`, `i_gdn_vs_s_rw`, `i_attn_core_vs_kv`,

`bottleneck_labels` `["weight_memory","vocab_memory","state_memory","kv_memory","quadratic_attn","compute"]`, `region_ids` `["embed","linear_attn","full_attn","mlp","lm_head","mtp"]`, `n_catalog_nodes` (52), `catalog_ids` (TASK-03/04 order, same 52 IDs), `n_diagrams` (1), `canonical_sentence_logical`, `canonical_sentence_hypothesis`, `canonical_sentence_activation`.

Integer JSON fields that are intensities may be JSON numbers `1`, `0.75`, `6` (not strings). Arrays of MAC/bytes at `example_T` are JSON arrays of ints.

`catalog_ids` exact order (copy TASK-04):

`token_id`, `e`, `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `h_final`, `logits_0`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `K_state`, `V_state`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `S`, `o`, `u_gdn`, `mix_lin`, `C_state`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp`, `logits_1`

### Stage split

- **Implementation** writes `scripts/check_work_and_traffic.py` **and** `docs/architecture/work-and-traffic.md` (convention, symbolic+instantiated tables, three traffic channels, hypothesis labels, JSON fence). Runs `--json` and `--work-traffic` after the document exists. Records command outcomes in this dossier. Does not commit.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner, Authority table links to this dossier / semantics / dataflow / lifetime / inventory / plan evidence policy, heading/JSON fence consistency. Must not change locked MAC integers, traffic integers, canonical sentences, bottleneck ids, or Mermaid node IDs. Does not edit TASK-02/03/04 artifacts.
- **Verification** independently re-runs focused commands, recomputes $C,A,W(T)$ and byte products from sitting `text_config` ranks (not from JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF evidence, no `plan.md` edit, and that bottleneck rows are labelled HYPOTHESIS.

- Invariants:
  - Ten level-2 headings in the locked order; 52 catalog IDs mentioned; six bottleneck ids; three canonical sentences verbatim.
  - Prefill identity $TC+AT(T+1)/2$; $T$ after-append; primary includes MTP.
  - $F=2\times\text{MAC}$; GDN rank-1; embed 0 MAC; elementwise not in primary.
  - Weight BF16 inventory; state TASK-04; activation region-cut from catalog ranks.
  - Logical values do not imply allocation; bottlenecks are hypotheses.
  - Vision encoder remains unexpanded.
- Rejected alternatives:
  - Counting `(19)` as $48\times 128^3$ MAC: rejected; definition is `(17)`–`(18)`.
  - FLOP=MAC (not $2\times$): rejected; intensity needs add+mul.
  - Streaming the full embed table as decode weight traffic: rejected; gather only.
  - GGUF Q4 or CUDA dtype sizes: rejected; BF16 inventory + conceptual F32 $S$.
  - Summing all 52 catalog tensors as activation traffic: rejected; that is not a minimum and contradicts logical ≠ physical.
  - Using CUDA fusion to cut activation bytes: TASK-12.
  - Filling TASK-16 SKU $\Pi_\text{peak}$ or measuring tok/s: forbidden; labels stay HYPOTHESIS.
  - Inspecting Quartz or llama.cpp for “real” FLOPs: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–04.
  - Editing frozen TASK-01–04 docs or `plan.md`.
  - Omitting MTP from primary: rejected; complete map and TASK-04 MTP KV.
  - Prefill attention $T^2/2$ approximation: rejected; exact $T(T+1)/2$.
  - Importing other `check_*.py` as a library.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/work-and-traffic.md` exists and follows the heading list above.
  - Symbolic MAC formulas per region cite TASK-02 tags; instantiated $C,A$ and $T=1,4096$ tables match the locked integers.
  - Weight traffic uses TASK-01 BF16 family totals with embed-as-gather; state traffic uses TASK-04 bytes plus triangular prefill KV; activation traffic reports forced, region-cut (primary), and GEMM-IO with locked integers.
  - Bottleneck section lists the six labels as HYPOTHESIS and closes the ledger open question (not left UNKNOWN except vision).
  - Three canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp/GGUF evidence; no SKU peak fill-in.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_work_and_traffic.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_work_and_traffic.py
python3 scripts/check_work_and_traffic.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_work_and_traffic.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --work-traffic docs/architecture/work-and-traffic.md
```

- Candidate quality: not required — no model execution or NLL; this increment is work/traffic documentation.
- Repository-wide commands:

```sh
test -f docs/architecture/work-and-traffic.md
python3 -m py_compile scripts/check_work_and_traffic.py
python3 scripts/check_work_and_traffic.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --work-traffic docs/architecture/work-and-traffic.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/work-and-traffic.md` (create)
  - `scripts/check_work_and_traffic.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-02/03/04 deliverables unchanged)
- Definition of done: work-and-traffic document published with locked MAC convention, symbolic+instantiated counts, three traffic channels, and hypothesis bottleneck labels; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-06 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T12:49:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-06.md`. Coupled IDs `none`. Document structure (10 headings), MAC/FLOP convention, decode/prefill $T$ identity, region MAC formulas citing TASK-02, weight/state/activation traffic floors, six HYPOTHESIS bottleneck labels, stdlib checker `scripts/check_work_and_traffic.py`, JSON schema, and acceptance commands are closed. `docs/architecture/work-and-traffic.md` and the checker were **not** written in this stage.
- Performance evidence applied: N/A — derived work/traffic documentation; bottleneck labels are hypotheses, not measured sink ranking

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_work_and_traffic.py` (stdlib checker: `text_config` arithmetic for MAC $C,A$, example-$T$ instantiations, BF16 family bytes, TASK-04 state volumes + triangular prefill KV, activation forced/region-cut/GEMM-IO, F13 intensities; `--json` asserts; `--work-traffic` heading/JSON/mermaid/canonical/catalog/integer checks).
  - Created `docs/architecture/work-and-traffic.md` (ten locked headings, three canonical sentences, symbolic+instantiated MAC tables, three traffic channels, six HYPOTHESIS bottleneck labels, one `flowchart TB` mermaid, JSON fence copied from live `--json`).
  - Did not edit `docs/architecture/plan.md`, `model-semantics.md`, `dataflow.md`, `lifetime-and-state.md`, or `model-inventory.md`. Did not commit.
- Commands:
  - `python3 -m py_compile scripts/check_work_and_traffic.py` — pass (exit 0)
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); live object matches locked $C,A$, family bytes, example-$T$ MAC/bytes, intensities $1$, $0.75$, $6$
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md` — pass (exit 0)
- UTC/time/tokens/cost: `2026-09-20T13:05:51Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `cursor-grok-4.6-high` (documentation subagent; parent/inherit)
- Changes and evidence:
  - `docs/architecture/work-and-traffic.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-04 / [`dataflow.md`](../dataflow.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`model-semantics.md`](../model-semantics.md), [`dataflow.md`](../dataflow.md), [`lifetime-and-state.md`](../lifetime-and-state.md), [`model-inventory.md`](../model-inventory.md), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_work_and_traffic.py`](../../../scripts/check_work_and_traffic.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the first JSON fence left unchanged. No locked MAC integers, traffic integers, canonical sentences, bottleneck ids, or Mermaid node IDs edited. Frozen TASK-02/03/04 artifacts and `plan.md` not edited (no dossier-assigned upstream cross-links).
- Commands:
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md` — pass (exit 0; headings, JSON fence, one flowchart, canonical sentences, catalog IDs, diagram node IDs; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T13:06:35Z`; `telemetry_unavailable`

### Verification

- Attempt: 1 (first pass)
- Agent/model: `cursor-composer-2.5-fast` (integration verifier subagent)
- Diff review:
  - Deliverables match dossier scope: `docs/architecture/work-and-traffic.md` (10 locked `##` headings in order; three canonical sentences verbatim; symbolic+instantiated MAC tables with TASK-02 equation cites `(1)`–`(24)`; weight/state/activation channels with locked integers; six bottleneck rows labelled HYPOTHESIS; one `flowchart TB` mermaid with required node IDs; JSON fence equals live `--json`).
  - `scripts/check_work_and_traffic.py` is stdlib-only (`argparse`, `json`, `re`, `sys`, `pathlib`); no import of other `check_*.py`; no Quartz/llama.cpp/GGUF references in deliverable or checker.
  - Frozen upstream docs unchanged: `plan.md`, `model-semantics.md`, `dataflow.md`, `lifetime-and-state.md`, `model-inventory.md` (no git diff).
  - `docs/architecture/task_ledger.md` only reflects `IN PROGRESS` (delivery-stage update; completion checkboxes not yet marked — expected before delivery agent).
  - `UNKNOWN` appears only under **Deferred vision**; no `TBD` / `TODO` / `???`.
- Independent raw-record checks (recomputed from sitting `text_config` ranks, not JSON echo):
  - Layer counts and `full_attention_indices` from `layer_types`: PASS.
  - Per-layer MAC, $C_\text{language}$, $C_\text{complete}$, $A_\text{language}$, $A_\text{complete}$: PASS (locked integers).
  - Decode $C+AT$, prefill $TC+AT(T+1)/2$, last-logits formulas at $T\in\{1,4096\}$: PASS.
  - TASK-04 state bytes (conv delay $k-1$, F32 $S$), triangular prefill KV read, storage/decode-read at $T=4096$: PASS.
  - Intensities $I=1$ (MLP/`lm_head`), $0.75$ (GDN vs $S$), $6$ (full-attn core vs KV); elementwise ceiling $<1\%$ of complete decode FLOP at $T=4096$: PASS.
  - JSON fence byte-identical to fresh `--json` stdout (key order included): PASS.
- Commands:
  - `python3 -m py_compile scripts/check_work_and_traffic.py` — **pass** (exit 0)
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — **pass** (exit 0)
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md` — **pass** (exit 0)
  - `test -f docs/architecture/work-and-traffic.md` — **pass** (exit 0)
  - Repository-wide repeat (`py_compile` + `--work-traffic`) — **pass** (exit 0)
- Formatting changed files: none (verification appended this run record only)
- Verdict: **PASS** — dossier acceptance conditions, ledger completion criteria, and `plan.md` scope/evidence policy satisfied; all focused and repository-wide commands green.
- UTC/time/tokens/cost: `2026-09-20T13:07:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1) — `docs/architecture/work-and-traffic.md` (10 locked headings, three canonical sentences, symbolic+instantiated MAC tables, three traffic channels, six HYPOTHESIS bottleneck labels, one `flowchart TB` mermaid, JSON fence); `scripts/check_work_and_traffic.py` stdlib checker; independent $C,A$ and byte recomputation from sitting `text_config`; frozen upstream docs unchanged; open question closed with DERIVED intensities and HYPOTHESIS bottleneck classes
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (theoretical work/traffic bounds; no tok/s evidence)
- Throughput delta: N/A — TASK-06 does not execute or time the model
- Commit: Publish Qwen3.8 work and traffic bounds
- Push: `origin/clean-sheet` (pending delivery)
- First-pass acceptance: yes (verification attempt 1 pass; no repair loop)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands
