# TASK-04 — Analyze lifetime and persistent state

## Control

- Primary ID: `TASK-04`
- Coupled IDs: `none`
- Dependencies: `TASK-03` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Classify every significant value by lifetime and recomputability; calculate persistent-state storage and per-token read/write volumes; identify semantic storage candidates without CUDA decisions.

## Goal and boundaries

Produce `docs/architecture/lifetime-and-state.md` as the Phase 1 lifetime + persistent-state analysis for the Qwen3.8-27B language + MTP forward map, derived from TASK-02 equations and the TASK-03 52-node catalog. Close the ledger open question with **DERIVED** element counts and byte volumes for \(K,V,C,S\), plus a semantic (not CUDA) statement of which catalog values must survive which named boundaries.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Do not invent operators or catalog IDs. Use TASK-03’s 52 catalog IDs in the same order. Equations, ranks, and state kinds come from `docs/architecture/model-semantics.md` (TASK-02). Dimensions instantiate sitting `text_config` / TASK-01 inventory.
  - Label claims `OBSERVED` (config/inventory already established) or `DERIVED` (lifetime class, recomputability, and bytes from ranks × multiplicity × element size). `UNKNOWN` only for vision-encoder internals deferred here.
  - GitHub Markdown. Cite TASK-02 equation tags `(1)`–`(24)` and TASK-03 catalog IDs. Do not rewrite forward math or redraw the TASK-03 DAG. Exactly one small lifetime/state summary Mermaid flowchart is required (lock below).
  - Allowed evidence: TASK-02 semantics, TASK-03 dataflow, TASK-01 inventory, sitting `config.json` `text_config`, and the plan evidence vocabulary. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`.
  - Semantic storage candidates are not physical allocation. A “must survive token boundary” claim is mathematical state, not a cache layout.
- Non-goals:
  - No FLOP / arithmetic-intensity bounds (TASK-06), except using ranks already in TASK-02/03 for byte math.
  - No semantic-graph execution contracts (TASK-11).
  - No fusion / buffer-reuse / physical allocation decisions (TASK-12). Identifying a semantic candidate is allowed; choosing CUDA buffers is not.
  - No layouts, schedules, kernel names, or decode/prefill **execution** plans (TASK-13+). Prefill and decode share one lifetime model; \(T\) differs; incoming state is zeros versus populated.
  - No new equations, algebraic alternatives, or vision-encoder internals.
  - No peak-memory claim: if a residual-vector byte size is mentioned, it is DERIVED rank × dtype, not a physical working-set peak.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01/02/03).
  - Do not edit `docs/architecture/dataflow.md`, `docs/architecture/model-semantics.md`, `docs/architecture/model-inventory.md`, or `docs/architecture/plan.md`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib lifetime/state checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-13` — central path: mathematics → logical dataflow → **tensor lifetime and state**.
- `docs/architecture/plan.md:81-83` — TASK-04 determines lifetimes and token-persistent state after TASK-03’s DAG.
- `docs/architecture/task_ledger.md` TASK-04 row — produces `docs/architecture/lifetime-and-state.md`; completion is lifetime/recomputability classification of every significant value, persistent-state storage and per-token read/write volumes, and semantic storage candidates without CUDA decisions. Open question to close: exact state sizes and which values require materialization across boundaries.
- `docs/architecture/task_ledger.md` TASK-03 established results — 52-ID catalog, token-crossing \(K,V,C,S\), live-across `g`/`z`, canonical logical-≠-physical sentence, fan-out ≠ must-store.
- `docs/architecture/task_ledger.md` TASK-06/07/11/12 rows — downstream consumers (traffic bounds, numerical risk, semantic contracts, layouts). Do not perform those analyses here.
- `docs/architecture/tasks/TASK-03.md` — structural precedent: stdlib checker, JSON fence, heading order, 52-ID order, stage split.
- `docs/architecture/model-semantics.md` Persistent state — ranks \(K,V\in\mathbb{R}^{4\times T\times 256}\) each (RoPE baked into stored \(K\)); \(C\) last 3 vectors in \(\mathbb{R}^{10240}\); \(S\in\mathbb{R}^{48\times 128\times 128}\) conceptual float32 from `mamba_ssm_dtype`; 16 language full + 1 MTP KV; 48 linear; initial zeros; prefill vs decode is the same map.
- `docs/architecture/dataflow.md` — 52 catalog IDs, fan-out, token-crossing, live-across, sharing rank. Fan-out is not a store requirement.
- `docs/architecture/model-inventory.md` — `text_config.dtype` `"bfloat16"` OBSERVED; `mamba_ssm_dtype` `"float32"` OBSERVED; all checkpoint tensors BF16; 48 linear + 16 full at indices 3,7,…,63; MTP one full-attention block.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` only (`hidden_size` 5120, `num_hidden_layers` 64, `full_attention_interval` 4, `num_key_value_heads` 4, `head_dim` 256, `linear_conv_kernel_dim` 4, `linear_num_value_heads` 48, `linear_key_head_dim` / `linear_value_head_dim` 128, `mamba_ssm_dtype` `"float32"`, `dtype` `"bfloat16"`, `mtp_num_hidden_layers` 1, `use_cache` true). Do not read safetensor payloads.
- `scripts/check_dataflow.py` and `scripts/check_model_semantics.py` — checker-style precedent. TASK-04’s checker is a sibling script; do not import them.

## Performance evidence

N/A — lifetime and persistent-state documentation; no prefill/decode/component timing, no keep/reject, no sink ranking. Byte volumes are DERIVED arithmetic, not MEASURED traffic.

## Implementation decisions

### Authority for lifetimes and bytes

The lifetime model is the TASK-03 catalog classified by mathematical survival, plus TASK-02 state ranks converted to element counts and bytes. If a class or byte total would disagree with a TASK-02 rank or TASK-03 catalog ID, the earlier document wins and this one is wrong.

- Node set = the locked 52 catalog IDs below, **this exact order**. Do not add/remove IDs. Shared weights \(E\), \(W_\text{lm}\) are not catalog intermediates and are not lifetime-classified here (they are parameters).
- Algebraic equivalents in TASK-02 are not extra nodes and have no extra lifetime.
- Prefill and decode share **one** lifetime model. Incoming \((K,V,C,S)\) is zeros versus populated; \(T>1\) versus \(T=1\). Do not duplicate tables per mode.
- Primary persistent-state totals **include MTP KV** (17 full-attention KV instances = 16 language + 1 MTP). Language-only KV is a secondary row, not the primary total.
- Fan-out ≠ must-store (TASK-03). High-fan-out IDs are not automatically semantic storage candidates.

### Deliverable structure (`docs/architecture/lifetime-and-state.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric rank is TASK-02 / config `OBSERVED` or `DERIVED`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, `model-semantics.md`, `dataflow.md`, inventory, config, checker; evidence labels; in-scope (language + MTP lifetime + persistent-state bytes) vs deferred (vision encoder). State that the document specifies lifetimes and mathematical state sizes, not kernels.
2. **Logical versus physical** — both canonical sentences (exact text below) plus three bullets: fan-out ≠ must-store; token-crossing names mathematical state, not cache layout; “must survive a boundary” is not CUDA malloc.
3. **Lifetime taxonomy** — the five disjoint classes, recomputability labels, prefill/decode identity, and the \(T\) convention.
4. **Catalog lifetime table** — all 52 IDs in TASK-03 order (table below).
5. **Persistent state storage** — formulas, per-layer table, all-layers totals including MTP, language-only KV secondary row, instantiated \(T=1\) and \(T=4096\).
6. **Per-token read/write volumes** — decode \(T_\text{new}=1\); KV read uses \(T-1\); C/S conventions below.
7. **Semantic storage candidates** — ranked list that closes the ledger open question; one Mermaid summary (diagram 1 of 1).
8. **Deferred vision** — residual-stream interface only.
9. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Logical versus physical**, include these **two canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> A value that must survive a named boundary is mathematical state for that boundary, not a cache layout or CUDA allocation.

JSON field `canonical_sentence_logical` is the first sentence-pair (the two-sentence TASK-03 string). JSON field `canonical_sentence` is the second (lifetime-specific) sentence.

Bullets required under that heading:

- Fan-out of a named value is not a requirement to store that value.
- Token-crossing edges name mathematical state \((K,V,C,S)\), not a cache layout.
- Identifying a semantic storage candidate is not a CUDA buffer, fusion, or layout decision (TASK-12).

### Lifetime taxonomy (lock)

Five classes, complete and disjoint over the 52 catalog IDs. JSON array `lifetime_classes` in this order:

| Class | JSON array | Meaning |
| --- | --- | --- |
| `ephemeral` | `ephemeral_ids` | Produced and consumed inside the current-token map; does not mathematically survive a named residual-add, live-across-mixer, token, or output boundary as the surviving object. |
| `live-across` | `live_across_ids` | Fan-out 1; unique consumer is not the immediate successor operator (TASK-03). Locked IDs: `g`, `z`. |
| `layer-residual` | `layer_residual_ids` | Residual stream that both the mixer/MLP **read** and **add into**. Locked IDs: `h`, `h_mid`. |
| `token-persistent` | `token_persistent_ids` | Mathematical state across the token boundary. Locked IDs: `K_state`, `V_state`, `C_state`, `S`. |
| `output-sink` | `output_sink_ids` | Forward-map outputs consumed as `output`. Locked IDs: `logits_0`, `logits_1`. |

`h_64` is the last residual-stream vector but is **not** `layer-residual`: it does not live across a Mix/MLP add (it is the result of the last add). Class `ephemeral`. `h_mtp` is the MTP block output, class `ephemeral`. Current-token writes `k_rope` → `K_state`, `v_full` → `V_state`, `qkv` → `C_state` do not make those activations token-persistent; the surviving objects are the state IDs.

Recomputability (two labels, disjoint):

| Label | JSON | Meaning |
| --- | --- | --- |
| `recomputable` | every catalog ID except the four state IDs | If dropped, the value can be rebuilt from its current-token catalog parents (parents may include live persistent state). |
| `requires-prior-state` | `requires_prior_state_ids` = `state_ids` | The value **is** prior-token mathematical state. Reconstructing it without that state requires replaying tokens \(1\ldots t-1\). |

Boundary column (exactly one per catalog ID):

| Boundary | IDs |
| --- | --- |
| `none` | all IDs not listed below |
| `residual-add` | `h`, `h_mid` |
| `live-across-mixer` | `g`, `z` |
| `token` | `K_state`, `V_state`, `C_state`, `S` |
| `output` | `logits_0`, `logits_1` |

Prefill of length \(T\) and decode of one new token share this taxonomy. Only \(T\) and whether incoming state is zeros versus populated change.

Do not introduce a sixth class. Intra-equation reuse of `k_hat` (TASK-03) is not a lifetime class.

### \(T\) convention (lock)

Let \(T\) be the stored KV length **after** appending the current token (TASK-02 rank \((4,T,256)\)). Decode of one new token has incoming KV length \(T-1\). When \(T=1\), incoming KV is empty and KV read volume is 0. \(C\) and \(S\) have \(T\)-independent stored ranks (initial zeros still occupy the full \(C\) and \(S\) ranks).

JSON: `T_is_stored_length_after_append` is `true`; `decode_T_new` is `1`; `example_T` is `[1, 4096]`.

### Element sizes (lock)

| Quantity | Bytes | Label | Applies to |
| --- | --- | --- | --- |
| BF16 | 2 | OBSERVED `text_config.dtype` `"bfloat16"`; inventory all checkpoint tensors BF16 | Conceptual \(K,V,C\) (and any residual-vector illustration) |
| F32 | 4 | OBSERVED `text_config.mamba_ssm_dtype` `"float32"` (TASK-02: conceptual \(S\); weights remain BF16) | Conceptual \(S\) |

JSON: `bytes_bf16` = 2, `bytes_f32` = 4. These are IEEE widths used as conceptual element sizes, not CUDA allocation dtypes and not GGUF Q4 sizes. Byte volume = element count × element size (DERIVED). Report elements and bytes separately. No “approximate” without the exact integer.

### Persistent-state formulas (lock)

Live config identities the checker must recompute (do not invent):

- \(n_\text{kv}=\) `num_key_value_heads` = 4; \(d_h=\) `head_dim` = 256
- \(n_\text{full}=\) 16 from `layer_types`; \(n_\text{mtp}=\) `mtp_num_hidden_layers` = 1; \(n_\text{full}^\text{KV}=n_\text{full}+n_\text{mtp}=17\)
- \(n_\text{lin}=\) 48 from `layer_types`
- \(d_\text{qkv}=(2\cdot\texttt{linear_num_key_heads}+\texttt{linear_num_value_heads})\cdot\texttt{linear_key_head_dim}=10240\)
- \(d_C=\texttt{linear_conv_kernel_dim}-1=3\)
- \(S\) rank \((n_v,d_k,d_v)=(\texttt{linear_num_value_heads},\texttt{linear_key_head_dim},\texttt{linear_value_head_dim})=(48,128,128)\)

**KV** (grows with \(T\); RoPE baked into stored \(K\)):

\[
N_{KV}^{(\ell)}/\text{token}=2\cdot n_\text{kv}\cdot d_h=2048,\qquad
B_{KV}^{(\ell)}/\text{token}=4096
\]

\[
N_{KV}^{\text{all}}/\text{token}=17\cdot 2048=34816,\qquad
B_{KV}^{\text{all}}/\text{token}=69632
\]

Language-only (secondary): \(16\cdot 2048=32768\) elements/token, \(65536\) bytes/token.

Storage: \(N_{KV}^{\text{all}}(T)=34816\,T\) elements, \(B_{KV}^{\text{all}}(T)=69632\,T\) bytes.

**C** (does not grow with \(T\); last 3 QKV vectors):

\[
N_C^{(\ell)}=3\cdot 10240=30720,\qquad B_C^{(\ell)}=61440
\]

\[
N_C^{\text{all}}=48\cdot 30720=1474560,\qquad B_C^{\text{all}}=2949120
\]

**S** (does not grow with \(T\); conceptual F32):

\[
N_S^{(\ell)}=48\cdot 128\cdot 128=786432,\qquad B_S^{(\ell)}=3145728
\]

\[
N_S^{\text{all}}=48\cdot 786432=37748736,\qquad B_S^{\text{all}}=150994944
\]

\(B_S^{\text{all}}=144\times 2^{20}\) exactly (144 MiB). State this as an exact identity, not an approximation.

**Primary storage** (includes MTP KV):

\[
N_\text{store}(T)=34816\,T+39223296,\qquad
B_\text{store}(T)=69632\,T+153944064
\]

Fixed part \(39223296=1474560+37748736\) elements, \(153944064=2949120+150994944\) bytes.

Instantiated (must appear as these integers):

| \(T\) | KV elems | KV bytes | Store elems | Store bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34816 | 69632 | 39258112 | 154013696 |
| 4096 | 142606336 | 285212672 | 181829632 | 439156736 |

Language-only storage (secondary; exclude MTP KV, still include all \(C\) and \(S\)): \(N=32768\,T+39223296\), \(B=65536\,T+153944064\).

### Per-layer and all-layers table (lock; copy into heading 5)

| State | Instances | Elems / instance | Bytes / instance | All-instances elems | All-instances bytes | vs \(T\) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| \(K\) | 17 | \(1024\,T\) | \(2048\,T\) | \(17408\,T\) | \(34816\,T\) | grows |
| \(V\) | 17 | \(1024\,T\) | \(2048\,T\) | \(17408\,T\) | \(34816\,T\) | grows |
| \(K{+}V\) | 17 | \(2048\,T\) | \(4096\,T\) | \(34816\,T\) | \(69632\,T\) | grows |
| \(C\) | 48 | 30720 | 61440 | 1474560 | 2949120 | fixed |
| \(S\) | 48 | 786432 | 3145728 | 37748736 | 150994944 | fixed |
| Primary total | — | — | — | \(34816\,T+39223296\) | \(69632\,T+153944064\) | mixed |

### Per-token read/write volumes (lock; decode \(T_\text{new}=1\))

Write volume is new mathematical state produced this token. Read volume is prior-token state consumed this token.

| Channel | Write elems | Write bytes | Read elems | Read bytes |
| --- | ---: | ---: | --- | --- |
| KV (17 layers, incl. MTP) | 34816 | 69632 | \(34816(T-1)\) | \(69632(T-1)\) |
| \(C\) (48 layers) | 491520 | 983040 | 1474560 | 2949120 |
| \(S\) (48 layers) | 37748736 | 150994944 | 37748736 | 150994944 |
| Primary total | 38275072 | 152047616 | \(34816(T-1)+39223296\) | \(69632(T-1)+153944064\) |

Conventions:

- **KV write:** append current `k_rope` and `v_full` (one token) per full layer including MTP. Current `k_rope`/`v_full` are produced this step; they are not a state read.
- **KV read:** past \(T-1\) tokens of stored \(K\) and \(V\). At \(T=1\), KV read is 0.
- **C write:** one new QKV vector of width 10240 per linear layer (the delay drops the oldest tap). Do **not** count rewriting the two retained taps as new writes. Stored \(C\) remains 3 vectors. \(48\times 10240=491520\) elements, \(983040\) bytes.
- **C read:** all 3 stored taps per linear layer (\(1474560\) elements, \(2949120\) bytes), including when they are zeros.
- **S read and write:** the full matrix \(S_{t-1}\) and \(S_t\) (\(37748736\) elements, \(150994944\) bytes), including when \(S_0=0\).

Instantiated decode reads:

| \(T\) | Read elems | Read bytes |
| ---: | ---: | ---: |
| 1 | 39223296 | 153944064 |
| 4096 | 181794816 | 439087104 |

Do not expand prefill into a triangular KV-read schedule (TASK-06 may). Prefill of length \(T\) from zeros **writes** the same per-token KV/C/S as the table, \(T\) times for KV and once-per-token for C/S, and ends at storage \(B_\text{store}(T)\).

Do not report a summed activation live-set. Optional one-line illustration only: one residual vector is \(H=5120\) BF16 elements = 10240 bytes (DERIVED), **not** a physical peak-memory claim.

### Catalog lifetime table (lock all 52 IDs; copy into heading 4)

Table columns, in this order: `ID` | `Lifetime class` | `Recomputability` | `Boundary` | `Eq`

Implementation copies these rows; do not add/remove IDs or reorder.

| ID | Lifetime class | Recomputability | Boundary | Eq |
| --- | --- | --- | --- | --- |
| `token_id` | ephemeral | recomputable | none | (1) |
| `e` | ephemeral | recomputable | none | (1) |
| `h` | layer-residual | recomputable | residual-add | (4) |
| `h_tilde` | ephemeral | recomputable | none | (4) |
| `h_mid` | layer-residual | recomputable | residual-add | (4)–(5) |
| `h_post` | ephemeral | recomputable | none | (5),(21) |
| `h_64` | ephemeral | recomputable | none | (5),(22),(23) |
| `h_final` | ephemeral | recomputable | none | (22) |
| `logits_0` | output-sink | recomputable | output | (22) |
| `u_q` | ephemeral | recomputable | none | (6)–(7) |
| `q_prime` | ephemeral | recomputable | none | (7) |
| `g` | live-across | recomputable | live-across-mixer | (7),(10) |
| `k_raw` | ephemeral | recomputable | none | (6) |
| `v_full` | ephemeral | recomputable | none | (6),(9) |
| `q_n` | ephemeral | recomputable | none | (8) |
| `k_n` | ephemeral | recomputable | none | (8) |
| `q_rope` | ephemeral | recomputable | none | (9),(12) |
| `k_rope` | ephemeral | recomputable | none | (9),(12) |
| `attn` | ephemeral | recomputable | none | (9) |
| `y_gate` | ephemeral | recomputable | none | (10) |
| `mix_full` | ephemeral | recomputable | none | (10) |
| `K_state` | token-persistent | requires-prior-state | token | state |
| `V_state` | token-persistent | requires-prior-state | token | state |
| `qkv` | ephemeral | recomputable | none | (13)–(14) |
| `z` | live-across | recomputable | live-across-mixer | (13),(20) |
| `a` | ephemeral | recomputable | none | (13),(15) |
| `b` | ephemeral | recomputable | none | (13),(15) |
| `c_tilde` | ephemeral | recomputable | none | (14) |
| `c` | ephemeral | recomputable | none | (14) |
| `q_lin` | ephemeral | recomputable | none | (14),(16) |
| `k_lin` | ephemeral | recomputable | none | (14),(16) |
| `v_lin` | ephemeral | recomputable | none | (14),(17) |
| `q_hat` | ephemeral | recomputable | none | (16),(18) |
| `k_hat` | ephemeral | recomputable | none | (16),(17) |
| `alpha` | ephemeral | recomputable | none | (15) |
| `beta` | ephemeral | recomputable | none | (15) |
| `S` | token-persistent | requires-prior-state | token | (17)–(19) |
| `o` | ephemeral | recomputable | none | (18) |
| `u_gdn` | ephemeral | recomputable | none | (20) |
| `mix_lin` | ephemeral | recomputable | none | (20) |
| `C_state` | token-persistent | requires-prior-state | token | (14) |
| `g_mlp` | ephemeral | recomputable | none | (21) |
| `up` | ephemeral | recomputable | none | (21) |
| `swiglu` | ephemeral | recomputable | none | (21) |
| `mlp_out` | ephemeral | recomputable | none | (5),(21) |
| `e_next` | ephemeral | recomputable | none | (1),(23) |
| `e_next_n` | ephemeral | recomputable | none | (23) |
| `h64_n` | ephemeral | recomputable | none | (23) |
| `mtp_cat` | ephemeral | recomputable | none | (23) |
| `mtp_u` | ephemeral | recomputable | none | (23) |
| `h_mtp` | ephemeral | recomputable | none | (24) |
| `logits_1` | output-sink | recomputable | output | (24) |

Locked class arrays (catalog order, disjoint union = `catalog_ids`):

- `ephemeral_ids`: `token_id`, `e`, `h_tilde`, `h_post`, `h_64`, `h_final`, `u_q`, `q_prime`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `qkv`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `o`, `u_gdn`, `mix_lin`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp` (42)
- `live_across_ids`: `g`, `z` (2)
- `layer_residual_ids`: `h`, `h_mid` (2)
- `token_persistent_ids` / `state_ids` / `requires_prior_state_ids`: `K_state`, `V_state`, `C_state`, `S` (4)
- `output_sink_ids`: `logits_0`, `logits_1` (2)

### Semantic storage candidates (closes ledger open question)

This section is **DERIVED** from the taxonomy. It is not a HYPOTHESIS about runtime, fusion, or speed. “Require materialization across boundaries” means mathematical survival, not CUDA malloc.

Ranked must-survive list (this order is required in the prose):

1. **Token boundary** — `K_state`, `V_state`, `C_state`, `S`. Not recomputable without prior-token state or prefix replay. Exact sizes in Persistent state storage. Primary totals include MTP KV.
2. **Residual-add boundary** — `h` (must still exist when Mix is added, Eq. (4)); `h_mid` (must still exist when MLP is added, Eq. (5)).
3. **Live-across-mixer boundary** — `g` (produced at the `q_proj` split, consumed only in Eq. (10) after attention); `z` (produced at `W_z`, consumed only in GatedRMSNorm after the recurrence, Eq. (20)).

Do not say which ephemeral tensors “should” be stored for speed. High-fan-out `h_tilde`, `h_post`, `k_rope`, `v_full`, `qkv`, `h_64` remain recomputable from current-token parents; TASK-12 owns physical materialization. Shared \(E\) / \(W_\text{lm}\) are parameters, not activation lifetime.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 7 (Semantic storage candidates). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers. This is a lifetime-class summary, not a redraw of the TASK-03 DAG.

Required IDs **inside that fence**: `h`, `h_mid`, `g`, `z`, `K_state`, `V_state`, `C_state`, `S`, `logits_0`, `logits_1`.

JSON `n_diagrams` is 1.

### Deferred vision

Visual tokens may replace placeholders in the residual stream as vectors in \(\mathbb{R}^{H}\) (`out_hidden_size` 5120 OBSERVED). Encoder, patch embed, and merger internals are **UNKNOWN** / out of scope. Do not classify vision-encoder activations. Do not assign vision KV/C/S. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_lifetime_and_state.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`, Google docstrings). No torch, safetensors, mermaid parser package, uv, Ruff, or pytest. Do not import `scripts/check_dataflow.py` or `scripts/check_model_semantics.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, QKV width, conv delay, and \(S\) rank.

CLI (cwd = repository root):

```text
python3 scripts/check_lifetime_and_state.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --lifetime docs/architecture/lifetime-and-state.md \
  [--json]
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `n_kv_heads`, `head_dim`, `linear_qkv_width`, `linear_conv_delay`, `linear_state_heads`, `linear_state_dk`, `linear_state_dv`. Derived fields: layer-count sums, element/byte coefficients, example-\(T\) instantiations. Constant fields: catalog IDs, class arrays, canonical sentences, `bytes_bf16`/`bytes_f32`, `example_T`, `n_diagrams`, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--lifetime PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) both canonical sentences verbatim, (5) every `catalog_ids` entry present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer below present as a decimal substring. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require `dataflow.md` / `model-semantics.md` JSON equality (those remain TASK-03 / TASK-02 checkers).

`--json` internal asserts (all required):

- `n_catalog_nodes == 52` and `len(catalog_ids) == 52`
- `n_full_layers_with_kv == n_full_layers + n_mtp_blocks == 17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `bytes_bf16 == 2` and `bytes_f32 == 4`
- `kv_elems_per_full_layer_per_token == 2 * n_kv_heads * head_dim == 2048`
- `kv_elems_all_per_token == n_full_layers_with_kv * 2048 == 34816`
- `c_elems_all == n_linear_layers * linear_conv_delay * linear_qkv_width == 1474560`
- `s_elems_all == n_linear_layers * linear_state_heads * linear_state_dk * linear_state_dv == 37748736`
- corresponding byte fields equal element fields × 2 (KV, C) or × 4 (S)
- `storage_fixed_elems == c_elems_all + s_elems_all`
- `decode_write_elems == kv_elems_all_per_token + c_write_elems_per_token_all + s_elems_all`
- `primary_includes_mtp_kv is True`
- the five class arrays are pairwise disjoint and their union equals `catalog_ids`
- `token_persistent_ids == state_ids == requires_prior_state_ids`
- `n_diagrams == 1`
- for each `example_T[i]`: storage and decode-read arrays equal coefficient arithmetic

Locked document integers the `--lifetime` check must find:

`2048`, `34816`, `32768`, `1474560`, `37748736`, `4096`, `69632`, `2949120`, `150994944`, `39223296`, `153944064`, `38275072`, `152047616`, `491520`, `983040`, `30720`, `61440`, `786432`, `3145728`, `39258112`, `154013696`, `142606336`, `285212672`, `181829632`, `439156736`, `181794816`, `439087104`

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices` (int array, must equal `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`), `n_kv_heads`, `head_dim`, `linear_qkv_width`, `linear_conv_delay`, `linear_state_heads`, `linear_state_dk`, `linear_state_dv`, `bytes_bf16`, `bytes_f32`, `n_catalog_nodes` (52), `catalog_ids` (array, **this exact order**):

`token_id`, `e`, `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `h_final`, `logits_0`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `K_state`, `V_state`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `S`, `o`, `u_gdn`, `mix_lin`, `C_state`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp`, `logits_1`

then `lifetime_classes` `["ephemeral","live-across","layer-residual","token-persistent","output-sink"]`, `ephemeral_ids` (the 42 IDs above), `live_across_ids` `["g","z"]`, `layer_residual_ids` `["h","h_mid"]`, `token_persistent_ids` `["K_state","V_state","C_state","S"]`, `output_sink_ids` `["logits_0","logits_1"]`, `state_ids` `["K_state","V_state","C_state","S"]`, `requires_prior_state_ids` `["K_state","V_state","C_state","S"]`, `boundary_token_ids` `["K_state","V_state","C_state","S"]`, `boundary_residual_add_ids` `["h","h_mid"]`, `boundary_live_across_ids` `["g","z"]`, `boundary_output_ids` `["logits_0","logits_1"]`, `primary_includes_mtp_kv` (`true`), `T_is_stored_length_after_append` (`true`), `decode_T_new` (1),

`kv_elems_per_full_layer_per_token` (2048), `kv_bytes_per_full_layer_per_token` (4096), `kv_elems_all_per_token` (34816), `kv_bytes_all_per_token` (69632), `kv_elems_language_per_token` (32768), `kv_bytes_language_per_token` (65536), `c_elems_per_layer` (30720), `c_bytes_per_layer` (61440), `c_elems_all` (1474560), `c_bytes_all` (2949120), `c_write_elems_per_token_all` (491520), `c_write_bytes_per_token_all` (983040), `s_elems_per_layer` (786432), `s_bytes_per_layer` (3145728), `s_elems_all` (37748736), `s_bytes_all` (150994944), `storage_kv_elems_coeff_T` (34816), `storage_kv_bytes_coeff_T` (69632), `storage_fixed_elems` (39223296), `storage_fixed_bytes` (153944064), `decode_write_elems` (38275072), `decode_write_bytes` (152047616), `decode_read_kv_elems_coeff_Tm1` (34816), `decode_read_kv_bytes_coeff_Tm1` (69632), `decode_read_fixed_elems` (39223296), `decode_read_fixed_bytes` (153944064), `example_T` `[1, 4096]`, `storage_elems_at_example_T` `[39258112, 181829632]`, `storage_bytes_at_example_T` `[154013696, 439156736]`, `decode_read_elems_at_example_T` `[39223296, 181794816]`, `decode_read_bytes_at_example_T` `[153944064, 439087104]`, `n_diagrams` (1), `canonical_sentence` (the lifetime-specific sentence), `canonical_sentence_logical` (the TASK-03 two-sentence string).

`n_mtp_blocks` is 1 from `mtp_num_hidden_layers`. `n_linear_layers` / `n_full_layers` from `layer_types` or equivalently 48 / 16 from interval 4. `n_full_layers_with_kv` is 17.

### Stage split

- **Implementation** writes `scripts/check_lifetime_and_state.py` **and** `docs/architecture/lifetime-and-state.md` (taxonomy, 52-row lifetime table, byte formulas, semantic candidates, JSON fence). Runs `--json` and `--lifetime` after the document exists. Records command outcomes in this dossier. Does not commit.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner, Authority table links to this dossier / semantics / dataflow / inventory / plan evidence policy, heading/JSON fence consistency. Must not change locked IDs, class assignments, byte integers, canonical sentences, or Mermaid node IDs. Does not invent classes. Does not edit TASK-02/03 artifacts.
- **Verification** independently re-runs focused commands, reads the document against this dossier (all 52 catalog rows, both canonical sentences, byte coefficients and instantiated totals, semantic-candidate ranking, one Mermaid fence), and confirms no Quartz/llama.cpp/GGUF evidence and no `plan.md` edit.

- Invariants:
  - Nine level-2 headings in the locked order; catalog has exactly the 52 IDs above in TASK-03 order; five lifetime classes are disjoint and complete.
  - Both canonical sentences present verbatim.
  - Prefill and decode are one lifetime model; state kinds are only \(K,V,C,S\); primary totals include MTP KV.
  - Logical values do not imply allocation; semantic candidates are mathematical survivors, not CUDA buffers.
  - No new operators beyond TASK-02; no kernel/layout/schedule/FLOP text except as things this document does **not** decide.
  - Vision encoder remains unexpanded.
- Rejected alternatives:
  - Treating TASK-03 fan-out as must-store: rejected; TASK-03 canonical sentence still holds.
  - CUDA cache layouts, kernel names, or buffer reuse: TASK-12 / later; forbidden here.
  - Inspecting Quartz or llama.cpp to “confirm” lifetimes or bytes: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01/02/03.
  - Editing frozen TASK-02/03 docs (`model-semantics.md`, `dataflow.md`, `model-inventory.md`): one-way citations suffice.
  - Peak-memory as a physical claim, or a summed activation live-set: not this task.
  - FLOP / arithmetic-intensity bounds: TASK-06.
  - Skipping MTP KV in primary totals: rejected; ledger asks for persistent-state storage of the complete language+MTP map.
  - Using GGUF Q4 sizes for byte volumes: rejected; conceptual BF16/F32 from config/inventory.
  - Importing `check_dataflow.py` or `check_model_semantics.py` as a library: keep evidence scripts standalone.
  - Zero Mermaid fences or a redraw of the eight TASK-03 DAGs: rejected; exactly one lifetime summary diagram.
  - A sixth lifetime class, or classifying `h_64` as `layer-residual`: rejected; `h_64` does not live across a Mix/MLP add.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/lifetime-and-state.md` exists and follows the heading list above.
  - Every significant catalog value (all 52 IDs) is classified by lifetime class, recomputability, and boundary.
  - Persistent-state storage formulas and instantiated totals (elements and bytes, including MTP KV) are present; per-token decode read/write volumes use the locked \(T\) / \(T-1\) / C-write / S-full conventions.
  - Semantic storage candidates section lists the ranked DERIVED items 1–3 and closes the ledger open question (not left UNKNOWN).
  - Both canonical sentences are verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp/GGUF evidence; no FLOP bounds.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_lifetime_and_state.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_lifetime_and_state.py
python3 scripts/check_lifetime_and_state.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_lifetime_and_state.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --lifetime docs/architecture/lifetime-and-state.md
```

- Candidate quality: not required — no model execution or NLL; this increment is lifetime/state documentation.
- Repository-wide commands:

```sh
test -f docs/architecture/lifetime-and-state.md
python3 -m py_compile scripts/check_lifetime_and_state.py
python3 scripts/check_lifetime_and_state.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --lifetime docs/architecture/lifetime-and-state.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate.
- Documentation/evidence updates:
  - `docs/architecture/lifetime-and-state.md` (create)
  - `scripts/check_lifetime_and_state.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; `dataflow.md` unchanged; `model-semantics.md` unchanged; `model-inventory.md` unchanged)
- Definition of done: lifetime-and-state document published with locked taxonomy, 52-row catalog classification, persistent-state storage and per-token volumes, and semantic storage candidates; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-04 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T09:48:33Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-04.md`. Coupled IDs `none`. Document structure (9 headings), five disjoint lifetime classes, 52-ID catalog assignments, persistent-state byte coefficients, decode read/write convention, semantic storage ranking, one Mermaid summary, stdlib checker `scripts/check_lifetime_and_state.py`, JSON schema, and acceptance commands are closed. `docs/architecture/lifetime-and-state.md` and the checker were **not** written in this stage. Coordinator admitted the dossier (no placeholders; matches ledger row and `plan.md`); marked ledger TASK-04 `IN PROGRESS`.
- Performance evidence applied: N/A — lifetime/state documentation; no timing

### Implementation

- Agent/model: `cursor-grok-4.6-high`
- Changes:
  - `scripts/check_lifetime_and_state.py` — stdlib checker; `--json` instantiates the locked 68-key summary from sitting `text_config` (layer counts, QKV width, conv delay, \(S\) rank, BF16/F32 bytes, MTP-inclusive KV coefficients, example \(T=1,4096\)) and runs dossier internal asserts; `--lifetime PATH` checks heading order, first json fence equality, one mermaid flowchart with required IDs, both canonical sentences, catalog-ID substrings, locked integers, and UNKNOWN only in Deferred vision. Missing config exits 2. Does not import other `check_*.py`.
  - `docs/architecture/lifetime-and-state.md` — nine locked headings; 52-row catalog lifetime table; persistent-state formulas and per-layer table; per-token read/write volumes with \(T-1\) KV read; ranked semantic candidates 1–3; one mermaid flowchart (`h`, `h_mid`, `g`, `z`, `K_state`, `V_state`, `C_state`, `S`, `logits_0`, `logits_1`); first json fence copied from a live `--json` run.
- Commands:
  - `python3 -m py_compile scripts/check_lifetime_and_state.py` — pass (exit 0)
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); 68-key object; `n_catalog_nodes=52`; `n_full_layers_with_kv=17`; `storage_elems_at_example_T=[39258112, 181829632]`; `storage_bytes_at_example_T=[154013696, 439156736]`
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md` — pass (exit 0); empty stdout
- UTC/time/tokens/cost: `2026-09-20T09:57:08Z`; `telemetry_unavailable`
- Commit: not created (implementation stage)

### Documentation

- Agent/model: `cursor-grok-4.6-high` (parent/inherit; Task surface rejected `composer-2.5`)
- Changes and evidence:
  - `docs/architecture/lifetime-and-state.md` — mechanical pass only. Draft-status banner (`unverified`) already present in TASK-03 [`dataflow.md`](../dataflow.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already links this dossier, [`model-semantics.md`](../model-semantics.md), [`dataflow.md`](../dataflow.md), [`model-inventory.md`](../model-inventory.md), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_lifetime_and_state.py`](../../../scripts/check_lifetime_and_state.py), and plan evidence policy in [`plan.md`](../plan.md). Nine required `##` headings and the first JSON fence left unchanged. No locked IDs, class assignments, byte integers, canonical sentences, or Mermaid node IDs edited. Frozen TASK-02/03 artifacts and `plan.md` not edited.
- Commands:
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0 (JSON fence source unchanged).
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md` — exit 0 (heading order, JSON fence, one flowchart, canonical sentences, catalog IDs, diagram node IDs; existing banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T09:58:33Z`; `telemetry_unavailable`
- Commit: not created (documentation stage)

### Verification

- Attempt: 1
- Agent/model: `cursor-grok-4.6-high` (parent/inherit; Task surface rejected `composer-2.5`)
- Diff review: TASK-04-owned artifacts are `docs/architecture/lifetime-and-state.md` (new), `scripts/check_lifetime_and_state.py` (new), `docs/architecture/tasks/TASK-04.md` (dossier), and the TASK-04 ledger row `IN PROGRESS`. `git diff -- docs/architecture/plan.md` is empty. `dataflow.md`, `model-semantics.md`, and `model-inventory.md` are unmodified. No Quartz, llama.cpp/GGML Qwen, or GGUF evidence in TASK-04 files; no CUDA/layout/fusion decisions; no FLOP bounds. The worktree also contains concurrent TASK-05 / TASK-16 untracked files and matching ledger `IN PROGRESS` rows; those are outside this increment and were not treated as TASK-04 content. Acceptance conditions for TASK-04 hold.
- Independent raw-record checks: recomputed from sitting `text_config` ranks (not from JSON echo): KV \(17\times 2\times 4\times 256=34816\) elems/token, \(69632\) bytes/token; C \(48\times 3\times 10240=1474560\) elems, \(2949120\) bytes; S \(48\times 48\times 128\times 128=37748736\) elems, \(150994944\) bytes (\(144\times 2^{20}\) exactly); \(B_\text{store}(T)=69632T+153944064\); decode write \(34816+491520+37748736=38275072\) elems / \(69632+983040+150994944=152047616\) bytes. Instantiated storage \(T=1\): elems \(39258112\), bytes \(154013696\); \(T=4096\): KV elems \(142606336\), KV bytes \(285212672\), store elems \(181829632\), store bytes \(439156736\). Decode read \(T=1\): \(39223296\) elems / \(153944064\) bytes; \(T=4096\): \(181794816\) elems / \(439087104\) bytes. All match the document tables and the live 68-key `--json` object. First fenced `json` block is byte-identical to live `--json` stdout. Catalog table: 52 IDs in locked order; five disjoint classes (42 ephemeral, 2 live-across, 2 layer-residual, 4 token-persistent, 2 output-sink); both canonical sentences verbatim as own paragraphs; one `flowchart TB` mermaid fence with required IDs; ranked candidates 1–3 present; primary totals include MTP KV; \(T\) after-append / KV read \(T-1\) stated. `UNKNOWN` only in Deferred vision; no TBD/TODO/`???`. Checker stdlib-only (`argparse`, `json`, `re`, `sys`, `pathlib`); does not import other `check_*.py`. Performance evidence: N/A — lifetime/state documentation, not a throughput/keep-reject task; byte volumes are DERIVED arithmetic, not MEASURED traffic (no coverage identity to check).
- Commands:
  - `python3 -m py_compile scripts/check_lifetime_and_state.py` — pass (exit 0)
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); 68-key object; `n_catalog_nodes=52`; `n_full_layers_with_kv=17`; `primary_includes_mtp_kv=true`; `storage_bytes_at_example_T=[154013696, 439156736]`; `decode_write_bytes=152047616`
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md` — pass (exit 0); empty stdout
  - `test -f docs/architecture/lifetime-and-state.md` — pass (exit 0)
  - repository-wide `python3 -m py_compile scripts/check_lifetime_and_state.py` — pass (exit 0)
  - repository-wide `--lifetime` re-run — pass (exit 0)
  - `git diff -- docs/architecture/plan.md` — pass (empty; exit 0)
- Formatting changed files: none
- Verdict: `pass` — focused and repository-wide gates exited 0; independent rank arithmetic matches document/JSON; JSON fence equals live `--json`; 52-row taxonomy, canonical sentences, mermaid IDs, semantic-candidate ranking, and scope boundaries are satisfied; `plan.md` untouched.
- UTC/time/tokens/cost: `2026-09-20T10:01:42Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1) — `python3 -m py_compile scripts/check_lifetime_and_state.py`; `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json`; `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md`; `test -f docs/architecture/lifetime-and-state.md`; `git diff -- docs/architecture/plan.md` empty
- Candidate measured delta: N/A
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: N/A
- Throughput delta: N/A
- Commit: Publish Qwen3.8 lifetime and state
- Push: `origin/clean-sheet` (pending)
- First-pass acceptance: yes
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk: local `.cache/` config must remain present for focused commands
- Delivery: `cursor-grok-4.6-high` (parent/inherit; Task surface rejected `composer-2.5`)
- UTC/time/tokens/cost: `2026-09-20T10:04:34Z`; `telemetry_unavailable`
