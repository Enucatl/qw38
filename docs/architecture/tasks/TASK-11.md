# TASK-11 — Derive the specialized semantic graph

## Control

- Primary ID: `TASK-11`
- Coupled IDs: `none`
- Dependencies: `TASK-03`, `TASK-04`, `TASK-06`, `TASK-07` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Derive node boundaries from semantic evidence rather than framework primitives; Specify operations, I/O, state, internal values, and flexibilities per node; Keep contracts hardware-independent.

## Goal and boundaries

Produce `docs/architecture/semantic-graph.md` as the Phase 1 **hardware-independent Qwen-specific semantic graph**: execution-region contracts for language+MTP, derived from TASK-02 equations, the TASK-03 catalog, TASK-04 lifetimes/state, TASK-06 work/traffic identities, and TASK-07 precision-risk citations. Close the ledger open question (natural boundaries, internal values, and synchronization/materialization edges) with **DERIVED** node types and inter-node I/O. Do **not** select fusion, a decode/prefill schedule, a CUDA mapping, or a layout.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Equations, ranks, and algebraic equivalents come from `docs/architecture/model-semantics.md` (TASK-02). Catalog IDs, fan-out, live-across, and model-region names come from `docs/architecture/dataflow.md` (TASK-03). Lifetime classes, must-survive boundaries, and \(K,V,C,S\) bytes come from `docs/architecture/lifetime-and-state.md` (TASK-04). Region MAC and traffic identities are **cited** from `docs/architecture/work-and-traffic.md` (TASK-06), not recopied as a new work study. Sensitive-op ids and precision roles are **cited** from `docs/architecture/numerical-sensitivity.md` (TASK-07), not re-ranked.
  - Label claims `OBSERVED` (config/inventory already established), `DERIVED` (node cuts, I/O, instance counts, catalog partition, MAC citations from ranks), or `HYPOTHESIS` (every “fusion would help”, every split-candidate usefulness, every TASK-06 bottleneck **class** restated as a citation). `UNKNOWN` only for vision-encoder internals deferred here. No `MEASURED` tok/s or NLL. No selected fusion or kernel.
  - GitHub Markdown math. Cite TASK-02 equation tags `(1)`–`(24)`, TASK-03 catalog IDs, TASK-04 lifetime/state IDs, TASK-06 MAC integers, TASK-07 sensitive-op ids. Do not rewrite forward math, redraw the eight TASK-03 DAGs, recompute TASK-04 bytes as a new lifetime study, or recopy TASK-06 MAC/byte tables as a new traffic study.
  - Allowed evidence: TASK-02 semantics, TASK-03 dataflow, TASK-04 lifetime/state, TASK-06 work/traffic (citations), TASK-07 numerical-sensitivity (citations), sitting `config.json` `text_config`, plan evidence vocabulary. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No TASK-16 SKU fill-in (not a dependency). No TASK-08/09/10 recipe, packing, or compiler-stage decisions (not dependencies).
  - Contracts are hardware-independent: catalog IDs, equation tags, state kinds, and cited MAC/byte identities. No thread geometry, warps, SMs, CUDA dtypes, kernel names, streams, or sitting-GPU numbers.
- Non-goals:
  - No fusion, buffer reuse, or physical-allocation **winners** (TASK-12). Internals and split candidates are named so TASK-12 can hypothesize; listing them is not a fusion decision.
  - No decode/prefill **schedules** (TASK-13/14). Prefill and decode share **one** semantic graph; only \(T\) and incoming \((K,V,C,S)\) change.
  - No physical layouts (TASK-15) and no CUDA mapping alternatives (TASK-17).
  - No quantization recipes, runtime-format packing, or compiler stages (TASK-08/09/10). Nodes consume parameters; bit width is not a node boundary.
  - No quality/NLL experiments (TASK-18) and no tok/s (TASK-19).
  - No new operators, no extra catalog IDs, no 64-layer unrolling, and no vision-encoder internals.
  - No generic GEMM / softmax / RMSNorm / `nn.Linear` / Hugging Face module graph. Those are framework primitives; this task rejects them as node types.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–10). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `model-semantics.md`, `dataflow.md`, `lifetime-and-state.md`, `work-and-traffic.md`, `numerical-sensitivity.md`, `model-inventory.md`, or any TASK-08–10 deliverable.
  - Do not import other `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib semantic-graph checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places the **Qwen-specific semantic graph** after compiler/format work in the study narrative and before decode/prefill plans and CUDA mappings.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:89-92` — TASK-11 derives semantic regions from dependencies, state, and lifetimes; TASK-12 fusion; TASK-13/14 schedules; TASK-17 CUDA maps **nodes**.
- `docs/architecture/task_ledger.md` TASK-11 row — produces `docs/architecture/semantic-graph.md`; purpose is hardware-independent Qwen-specific execution regions and contracts; open question is natural boundaries, internal values, and synchronization/materialization edges (**closed here by DERIVED cuts, not by selecting fusion**); completion is node boundaries from semantic evidence (not framework primitives), per-node ops/I/O/state/internals/flexibilities, hardware-independent contracts.
- `docs/architecture/task_ledger.md` TASK-03 established results — 52-ID catalog; model regions `embed`, `full_attn`, `linear_attn`, `mlp`, `primary_logits`, `mtp` are **logical DAG regions**, not execution contracts; live-across `g`/`z`; sharing rank 1–11; logical ≠ physical.
- `docs/architecture/task_ledger.md` TASK-04 established results — five lifetime classes; must-survive: token \(K,V,C,S\); residual-add `h`/`h_mid`; live-across `g`/`z`; \(B_\text{store}(T)=69632T+153944064\); fan-out ≠ must-store.
- `docs/architecture/task_ledger.md` TASK-06 established results — embed 0 MAC gather; linear token MAC `118235136`; full proj `104857600` plus \(A=12288\) per full layer; MLP `267386880`; `lm_head` `1271398400`; GDN rank-1 `2359296`; `mtp.fc` `52428800`; intensities \(I=1\), \(0.75\), \(6\); six HYPOTHESIS bottleneck labels.
- `docs/architecture/task_ledger.md` TASK-07 established results — four precision roles; 20 sensitive ops; GDN primary recurrent `(17)`–`(18)`; conceptual BF16 \(K,V,C\) and F32 \(S\); risks remain HYPOTHESIS.
- `docs/architecture/task_ledger.md` TASK-12/13/14/17 — consumers of this graph; do not perform those designs here.
- `docs/architecture/model-semantics.md` — equations `(1)`–`(24)`; residual wrapper `(4)`–`(5)`; two RMSNorm roles must not be collapsed; GatedRMSNorm `(3)` is GDN-only; algebraic equivalents are the same real map; MTP optional omission is equivalent only when \(\ell^{(0)}\) alone is required.
- `docs/architecture/dataflow.md` — 52 catalog IDs in locked order; `g` live-across after attention; `z` live-across after recurrence; `h_64` fans out to primary logits and MTP; shared \(E\)/`W_\text{lm}`.
- `docs/architecture/lifetime-and-state.md` — `h`/`h_mid` residual-add; `g`/`z` live-across-mixer; token-persistent state IDs; `h_64` is ephemeral (not layer-residual).
- `docs/architecture/work-and-traffic.md` — MAC/traffic identities to cite; region-cut activation bound is not a fusion claim; bottleneck classes stay HYPOTHESIS.
- `docs/architecture/numerical-sensitivity.md` — sensitive-op ids to attach per node; `activation_dtype_decided` false; chunkwise FP gap HYPOTHESIS.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for layer counts, shapes, and MAC products. Do not read safetensor payloads.
- `scripts/check_dataflow.py` / `scripts/check_lifetime_and_state.py` / `scripts/check_work_and_traffic.py` / `scripts/check_numerical_sensitivity.py` — checker-style precedent. TASK-11’s checker is a sibling; do not import them.

## Performance evidence

N/A — semantic-graph **contract** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. MAC and byte figures are TASK-06/04 **citations** (DERIVED arithmetic already established), not new measurements. Fusion/split usefulness is HYPOTHESIS for TASK-12, not MEASURED. Do not apply the performance-evidence checklist to rank kernels or claim a winning node split.

- Measurement identity: N/A (no engine binary). TASK-04/06/07 integers are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Semantic-graph coverage is 6 node types, 52 catalog IDs partitioned into boundary/internal/state, 14 inter-node sync edges, 5 flexibility kinds.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` layer count or MAC product disagrees with TASK-01/06, or a catalog ID disagrees with TASK-03/04, stop and fail closed (do not invent nodes).
- Claim types: node cuts `derived`; instance counts `derived`; MAC/byte cells `derived` (citations); every fusion/split **usefulness** and every restated bottleneck class `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Graph completeness is 6 types, 135 complete instances, 52-ID partition, 14 sync edges, hardware-independent contracts.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for semantic-graph claims

If a node I/O, state kind, catalog ID, layer count, or cited MAC/byte would disagree with TASK-02/03/04/06/07 or sitting `text_config`, the earlier document / config wins and this one is wrong.

- Prefill and decode share **one** semantic graph. Only \(T\) (stored KV length after append) and whether incoming \((K,V,C,S)\) is zeros versus populated change. Do not duplicate node types per mode.
- Primary graph **includes MTP** (second embed, `mtp_mix`, one `gated_attn`, one `mlp`, second `lm_head`). Language-only instance counts are secondary. Omitting MTP when only \(\ell^{(0)}\) is required is a TASK-02 algebraic equivalent, not a second graph and not the primary contract.
- Algebraic equivalents in TASK-02 (chunkwise GDN, SDPA, GQA-as-repeat, RoPE complex form, \(S\) vs \(S^\top\), FIR vs delay-line conv) are the same real map **inside** the owning node. They are flexibilities, not extra node types. Finite-precision chunkwise-vs-recurrent gap remains TASK-07 HYPOTHESIS; GDN primary definition stays `(17)`–`(18)`.
- TASK-03 **model regions** (`full_attn`, `linear_attn`, `mlp`, …) name logical DAG cuts. Semantic **node types** are execution contracts derived from those cuts plus lifetime/work/precision evidence. Do not copy Hugging Face module names (`Qwen3NextGatedDeltaNet`, `nn.Linear`, …) as node ids.
- Fan-out ≠ must-store still holds. Declaring a catalog ID as node I/O is not a CUDA store. TASK-04 must-survive IDs (`h`, `h_mid`, `K,V,C,S`, `g`, `z`, logits) constrain **mathematical** visibility; `g`/`z` stay **internal** unless TASK-12 splits a node.
- Do not inspect Quartz or llama.cpp to “confirm” graphs or kernels.
- Do not select fusion, schedules, tiles, or CUDA mappings.

### Deliverable structure (`docs/architecture/semantic-graph.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Fusion/split usefulness cells are `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, semantics, dataflow, lifetime, work/traffic, numerical-sensitivity, inventory, config, checker; evidence labels; in-scope (language+MTP semantic contracts) vs deferred (vision encoder; TASK-12 fusion winners; TASK-13/14 schedules; TASK-17 CUDA). State that the document specifies **hardware-independent execution contracts**, not kernels.
2. **Semantic contract convention** — the three canonical sentences plus the boundary-question sentence (exact text below); what a node is; I/O ≠ must-store; prefill/decode identity.
3. **Boundary derivation** — the locked cut rules; rejected framework primitives; this heading **closes** the ledger open question.
4. **Node types** — the six types, multiplicity, equation ranges, residual-add-inside and RMS-inside-consumer locks.
5. **Per-node contracts** — one table per type: ops, I/O, state, internals, flexibilities, TASK-06/07 citations.
6. **Inter-node edges** — the 14 sync edges; catalog partition (boundary / internal / state); one Mermaid summary (diagram 1 of 1).
7. **Work, traffic, and precision citations** — cited MAC/byte/sensitive-op attachments; no new work study; bottleneck classes remain HYPOTHESIS citations.
8. **Flexibilities and non-decisions** — five flexibility kinds; ten split candidates; what TASK-12/13/14/17 own.
9. **Deferred vision** — residual-stream interface only; not a node type.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Semantic contract convention**, include these **three canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Semantic nodes in this document are hardware-independent execution contracts, not kernels, CUDA graphs, or framework modules.

> Internal values and flexible splits are TASK-12 materialization and fusion candidates; listing them is not a fusion or schedule decision.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_contracts` = sentence 2; `canonical_sentence_internals` = sentence 3.

Immediately after those three, include this **boundary-question sentence verbatim** (closes the ledger open question in checker-substring form):

> Natural node boundaries follow mixer kind and state, residual-add I/O, embed gather, vocabulary projection, and MTP mix; internal values stay inside nodes; synchronization edges are the declared node I/O, state, and shared weights.

JSON: `boundary_question_sentence` = that sentence. JSON `ledger_open_question_boundaries_closed` true.

Bullets required under that heading:

- Six semantic node types are complete for this task: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`.
- A node is a named execution contract: operations (equation tags), I/O (catalog IDs), state read/write, internal catalog IDs, and flexibilities.
- Prefill and decode share one graph; `decode_prefill_share_graph` true.
- Residual add of Mix lives **inside** mixer nodes; residual add of MLP lives **inside** `mlp`. Inputs `h` / `h_mid` must remain mathematically available until that add (`residual_input_live_until_add` true).
- Residual-stream RMSNorm `(2)` lives **inside** the consuming node. The two RMSNorm roles and GatedRMSNorm `(3)` must not be collapsed (`rms_inside_consumer` true, `rms_roles_not_collapsed` true).
- Live-across `g` and `z` are **internal** to `gated_attn` and `gated_delta_net` (`live_across_are_internal` true). They become hypothesized sync edges only if TASK-12 splits the node.
- Fan-out ≠ must-store and node I/O ≠ must-store still hold.
- Primary graph includes MTP. Hardware mapping is TASK-17. Fusion winners are TASK-12.

### Boundary derivation (lock; closes ledger open question)

Cuts are **DERIVED** from prior evidence, not from framework modules or CUDA kernels. Apply these rules in this order; JSON `boundary_rule_ids` in this exact order (6 ids). JSON `n_boundary_rules` = 6.

| id | Cut | Evidence | Why it is a node type |
| --- | --- | --- | --- |
| `cut_mixer_kind` | Gated Attention vs Gated DeltaNet | TASK-02 `(6)`–`(12)` vs `(13)`–`(20)`; state KV vs \(C,S\); TASK-06 \(A T\) vs \(T\)-free GDN; TASK-07 softmax/RoPE/KV vs recurrent \(S\) | Different equations, state, work class, and precision risks. One “attention” node would hide Qwen’s hybrid contract. |
| `cut_mixer_mlp` | Mixer vs MLP | TASK-02 `(4)` vs `(5)`; TASK-04 `h_mid` residual-add; TASK-06 distinct MAC | `h_mid` must survive between Mix add and MLP add. Different work (attn/GDN vs SwiGLU). |
| `cut_embed_gather` | Embed vs contractions | TASK-02 `(1)`; TASK-06 0 MAC gather 10240 B | Gather is not a GEMM. Distinct traffic class. |
| `cut_vocab` | `lm_head` vs hidden GEMMs | TASK-02 `(22)` `(24)`; TASK-06 `1271398400` MAC and `vocab_memory` | Vocabulary projection is a distinct contraction and traffic outlier. |
| `cut_mtp_mix` | MTP concat+`fc` vs a decoder layer | TASK-02 `(23)`; TASK-03 `h_64` fan-out | Mix of `e_next` and `h_64` is not mixer or MLP. |
| `cut_not_module` | Do **not** cut on RMS modules, `nn.Linear`, qkv/core/out kernels, or per-catalog-ID | TASK-03 live-across `g`/`z`; TASK-02 two RMS roles; plan.md anti-anchoring | Framework primitives and CUDA-shaped splits are not semantic evidence. |

JSON `framework_primitives_rejected` true.

**Not node types** (explicit):

- Hugging Face / paper class names, GGML ops, CUDA kernels, GEMM/softmax/RMS as generic ops.
- One node per catalog ID (that is the TASK-03 DAG).
- One node per decoder layer wrapping mixer+MLP (would hide `h_mid`).
- Required split of `gated_attn` into proj / core / `o_proj` (would promote live-across `g` to a required sync edge without TASK-12 evidence).
- Required split of `gated_delta_net` into conv / recurrence / `out_proj` (would promote `z` / `qkv` the same way).
- Residual-add as its own type (the add is the TASK-04 boundary **inside** mixer/`mlp`; I/O already exposes `h` / `h_mid`).
- `vision_if` (deferred; not a node).

This heading **closes** the open question:

- **Natural boundaries** = the six types from the six rules.
- **Internal values** = the 38 ephemeral/live-across catalog IDs owned inside nodes (including `g`, `z`, `mix_*`, `mlp_out`, RMS outputs `h_tilde`/`h_post`/`h_final`).
- **Synchronization/materialization edges** = the 14 declared inter-node edges (residual stream, MTP fan-out, outputs, token state, shared weights). Intra-node live-across is **not** an inter-node edge until a TASK-12 split.

Closing the question does **not** select fusion, materialization, or a schedule.

### Node types (lock)

JSON array `node_type_ids` in this exact order (6 ids). JSON `n_node_types` = 6.

| id | Multiplicity (complete) | Eqs | Consumes | Produces | State |
| --- | ---: | --- | --- | --- | --- |
| `embed` | 2 | `(1)` | `token_id` | `e` or `e_next` | none |
| `gated_attn` | 17 | `(2)` in-RMS, `(4)` Mix+add, `(6)`–`(12)` | `h` (MTP: `mtp_u` as `h`) | `h_mid` | `K_state`, `V_state` RW |
| `gated_delta_net` | 48 | `(2)` in-RMS, `(4)` Mix+add, `(3)`, `(13)`–`(20)` | `h` | `h_mid` | `C_state`, `S` RW |
| `mlp` | 65 | `(2)` post-RMS, `(5)` add, `(21)` | `h_mid` | next `h` / `h_64` / `h_mtp` | none |
| `lm_head` | 2 | `(2)` final or `mtp.norm`, `(22)` or `(24)` | `h_64` or `h_mtp` | `logits_0` or `logits_1` | none |
| `mtp_mix` | 1 | `(23)` mix | `h_64`, `e_next` | `mtp_u` | none |

JSON instance counts (complete primary = language + MTP):

| JSON key | Value | How |
| --- | ---: | --- |
| `n_embed_instances` | 2 | `e_t` and `e_{t+1}` |
| `n_gated_attn_instances` | 17 | 16 language + 1 MTP |
| `n_gated_delta_net_instances` | 48 | `n_linear_layers` |
| `n_mlp_instances` | 65 | 64 language + 1 MTP |
| `n_lm_head_instances` | 2 | primary + MTP |
| `n_mtp_mix_instances` | 1 | one mix |
| `n_node_instances_complete` | 135 | \(2+17+48+65+2+1\) |
| `n_node_instances_language` | 130 | \(1+16+48+64+1+0\) |

JSON: `primary_includes_mtp` true; `mtp_omission_is_algebraic_equivalent` true (not the primary graph).

Layer mixer xor: language layer \(\ell\) uses `gated_attn` iff \(\ell\in\mathcal{L}_\text{full}\) (`full_attention_indices` \(\{3,7,\ldots,63\}\)), else `gated_delta_net`. Then `mlp`. Do not unroll 64 layers in prose beyond this rule. JSON `mixer_xor_by_layer_types` true.

JSON booleans (lock):

- `decode_prefill_share_graph` = true
- `residual_add_inside_mixer` = true
- `residual_add_inside_mlp` = true
- `residual_input_live_until_add` = true
- `rms_inside_consumer` = true
- `rms_roles_not_collapsed` = true
- `live_across_are_internal` = true
- `hardware_independent` = true
- `cuda_mapping_deferred` = true
- `fusion_selected` = false
- `schedule_selected` = false
- `layout_selected` = false
- `framework_primitives_rejected` = true
- `fanout_neq_must_store` = true
- `node_io_neq_must_store` = true
- `gdn_primary_is_recurrent_eq_17` = true
- `chunkwise_fp_gap_is_hypothesis` = true
- `activation_dtype_decided` = false
- `vision_interface_is_not_a_node` = true
- `ledger_open_question_boundaries_closed` = true
- `catalog_partition_complete` = true

### Per-node contracts (lock)

Each type has: operations (equation tags + short op list), inputs, outputs, state read/write, internals (catalog IDs), flexibilities (kinds + named split candidates), TASK-06 MAC citation, TASK-07 sensitive-op ids. Do not add catalog IDs. Do not name CUDA kernels.

**`embed`**

- Ops: gather row of shared \(E\); 0 MAC (TASK-06). No RMS. No residual add.
- Inputs: `token_id`. Outputs: `e` (language) or `e_next` (MTP instance).
- State: none. Internals: none (`embed_internal_ids` empty).
- Weights: shared `E` (`shared_weight_ids` includes `E`).
- Flexibilities: `shared_weight_reuse` (same payload for `e` and `e_next`); `recompute_ephemeral` of `e` from `token_id`. No algebraic mixer equivalent. No split candidate.
- TASK-06 cite: `mac_embed` 0; `weight_gather_bytes_per_row` 10240. Complete decode gather 20480 B (`e_t` and `e_{t+1}`).
- TASK-07 cite: `param_bf16` only (table gather; no reduction).

**`gated_attn`**

- Ops: residual-stream RMSNorm `(2)` of `h` → `h_tilde`; projections `(6)`–`(7)` including `q\|g` split; QK-RMSNorm `(8)` then partial mRoPE `(11)`–`(12)` (order required); causal GQA softmax `(9)` against stored \(K,V\) plus current `k_rope`/`v_full`; sigmoid gate `(10)` (**not** SiLU); `W_o`; residual add `(4)` into `h_mid`.
- Inputs: `h` (MTP block: `mtp_u` identified as residual `h`). Outputs: `h_mid`.
- State RW: `K_state`, `V_state` (append `k_rope`, `v_full`; RoPE baked into stored \(K\); 17 instances including MTP). Reads past \(T-1\) then attends length \(T\) (TASK-04).
- Internals (`gated_attn_internal_ids`, this order): `h_tilde`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`. Live-across internal: `g`. High-fan-out internals: `k_rope`, `v_full` (attn + KV write).
- Contract: `h` remains available until the residual add. `g` remains available from the `q_proj` split until `(10)`.
- Flexibilities: `algebraic_equivalent` (exact SDPA; GQA-as-repeat; RoPE complex form; text mRoPE vs ordinary RoPE when \(p^T=p^H=p^W\)); `fuse_internals`; `split_at_internal` candidates `g`, `h_tilde`, `k_rope`, `v_full` (TASK-12 HYPOTHESIS cuts, not extra types); `recompute_ephemeral` of all internals and of output `h_mid` from `h` plus state.
- TASK-06 cite: T-free proj MAC `104857600` per instance; T-coefficient `12288`; KV 4096 B/token/instance.
- TASK-07 cite: `param_bf16`, `residual_stream`, `live_across_gates`, `silu_sigmoid` (sigmoid gate), `rms_hidden`, `rms_head`, `softmax_over_T`, `attn_av_over_T`, `gemm_k5120`, `rope_phase`, `state_kv_bf16`.

**`gated_delta_net`**

- Ops: residual-stream RMSNorm `(2)` of `h` → `h_tilde`; projections `(13)` (`W_qkv`, `W_z`, `W_a`, `W_b`); depthwise causal conv `(14)` reading `C_state`; SiLU and QKV split; \(\alpha/\beta\) `(15)`; L2 `(16)`; recurrence `(17)`–`(18)` as **definition** (not dense `(19)` as extra work); GatedRMSNorm `(3)` with `z`; `W_out`; residual add `(4)` into `h_mid`. Write `qkv` into `C_state`; write `S_t`.
- Inputs: `h`. Outputs: `h_mid`.
- State RW: `C_state` (3×10240), `S` (48×128×128 conceptual F32).
- Internals (`gated_delta_net_internal_ids`, this order): `h_tilde`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `o`, `u_gdn`, `mix_lin`. Live-across internal: `z`. Intra-equation reuse: `k_hat` (not a second catalog consumer; not a split candidate).
- Contract: `h` live until residual add; `z` live from `W_z` until `(20)`; GDN eval primary is left-to-right `(17)`–`(18)`.
- Flexibilities: `algebraic_equivalent` (chunkwise/WY of the **same** map; FIR vs delay-line conv; \(S\) vs \(S^\top\)); `fuse_internals`; `split_at_internal` candidates `z`, `h_tilde`, `qkv`; `recompute_ephemeral`. Chunkwise FP gap stays HYPOTHESIS; do not treat chunkwise as a second node or as zero \(S\) traffic (TASK-06).
- TASK-06 cite: `mac_lin_token_per_layer` `118235136` (includes conv `40960`, GDN `2359296`, projs, `out_proj`); \(S\) 3145728 B/instance F32; \(C\) 61440 B/instance; GDN vs \(S\) \(I=0.75\) identity cited, class `state_memory` remains HYPOTHESIS.
- TASK-07 cite: `param_bf16`, `residual_stream`, `live_across_gates`, `silu_sigmoid`, `rms_hidden`, `rms_head`, `l2_gdn`, `gemm_k5120`, `gdn_S_recurrent`, `gdn_inner_d128`, `gdn_alpha_beta`, `state_c_bf16`, `s_below_f32`, `conv_fir`.

**`mlp`**

- Ops: post-attn RMSNorm `(2)` of `h_mid` → `h_post`; SwiGLU `(21)` (`W_gate`, `W_up`, SiLU, `W_down`); residual add `(5)` of `mlp_out` into next residual (`h` / `h_64` / `h_mtp`).
- Inputs: `h_mid`. Outputs: next `h` (hidden layers), `h_64` (last language layer), or `h_mtp` (MTP instance).
- State: none. Internals (`mlp_internal_ids`, this order): `h_post`, `g_mlp`, `up`, `swiglu`, `mlp_out`.
- Contract: `h_mid` live until the residual add.
- Flexibilities: `fuse_internals`; `split_at_internal` candidates `h_post`, `swiglu`; `recompute_ephemeral`. No GDN/SDPA equivalent.
- TASK-06 cite: `mac_mlp_per_layer` `267386880`; \(I=1\) vs weights (DERIVED identity); class `weight_memory` remains HYPOTHESIS.
- TASK-07 cite: `param_bf16`, `residual_stream`, `silu_sigmoid`, `rms_hidden`, `gemm_k5120`, `gemm_k17408`.

**`lm_head`**

- Ops: residual-stream RMSNorm `(2)` (`model.language_model.norm` or `mtp.norm`) then \(W_\text{lm}\) `(22)` or `(24)`. Shared weight `W_lm`. Sampling softmax over \(V\) is **out of scope**.
- Inputs: `h_64` (primary) or `h_mtp` (MTP). Outputs: `logits_0` or `logits_1`.
- State: none. Internals (`lm_head_internal_ids`): `h_final` for the primary instance. The MTP instance’s post-`mtp.norm` vector has **no extra catalog ID**; do not invent one. JSON `mtp_norm_has_no_extra_catalog_id` true.
- Flexibilities: `shared_weight_reuse` (one `W_lm` payload; a second physical read for `logits_1` is TASK-06 HYPOTHESIS traffic, not DERIVED unique bytes); `fuse_internals`; `split_at_internal` candidate `h_final`; `recompute_ephemeral`.
- TASK-06 cite: `mac_lm_head` `1271398400`; weight bytes `2542796800`; \(I=1\) vs weights.
- TASK-07 cite: `param_bf16`, `rms_hidden`, `gemm_lm_head`.

**`mtp_mix`**

- Ops: RMSNorm of `e_next` and of `h_64`; concat embed-then-hidden; \(W_\text{fc}\) `(23)` → `mtp_u` (MTP block residual input).
- Inputs: `h_64`, `e_next`. Outputs: `mtp_u`.
- State: none. Internals (`mtp_mix_internal_ids`, this order): `e_next_n`, `h64_n`, `mtp_cat`.
- Flexibilities: `fuse_internals`; `split_at_internal` candidate `mtp_cat`; `recompute_ephemeral`.
- TASK-06 cite: `mac_mtp_fc` `52428800`.
- TASK-07 cite: `param_bf16`, `rms_hidden`, `gemm_k5120`.

JSON object `node_inputs` / `node_outputs` / `node_state` / `node_internals` keyed in `node_type_ids` order. `node_state` values: embed/`mlp`/`lm_head`/`mtp_mix` empty arrays; `gated_attn` `["K_state","V_state"]`; `gated_delta_net` `["C_state","S"]`.

`h_tilde` appears in **both** mixer internal lists (xor instances). Checker: unique union of all internals plus boundary plus state equals `catalog_ids`.

### Catalog partition (lock)

JSON `n_catalog_nodes` = 52. `catalog_ids` exact TASK-03/04 order (copy):

`token_id`, `e`, `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `h_final`, `logits_0`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `K_state`, `V_state`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `S`, `o`, `u_gdn`, `mix_lin`, `C_state`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp`, `logits_1`

Three disjoint classes whose union is `catalog_ids`:

| Class | JSON | IDs | Count |
| --- | --- | --- | ---: |
| `boundary` | `boundary_ids` | `token_id`, `e`, `h`, `h_mid`, `h_64`, `logits_0`, `e_next`, `mtp_u`, `h_mtp`, `logits_1` | 10 |
| `state` | `state_ids` | `K_state`, `V_state`, `C_state`, `S` | 4 |
| `internal` | `internal_ids` | all remaining, this order: `h_tilde`, `h_post`, `h_final`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `o`, `u_gdn`, `mix_lin`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next_n`, `h64_n`, `mtp_cat` | 38 |

JSON `n_boundary_ids` = 10, `n_state_ids` = 4, `n_internal_ids` = 38. `catalog_partition_complete` true. `live_across_ids` `["g","z"]` subset of internals. `intra_equation_reuse_ids` `["k_hat"]` subset of internals. `shared_weight_ids` `["E","W_lm"]` (not catalog IDs).

`mix_full`, `mix_lin`, and `mlp_out` are **internal** because residual add is inside the producing node. `h_64` is **boundary** because of fan-out to `lm_head` and `mtp_mix`, even though TASK-04 classifies it ephemeral (I/O ≠ must-store).

### Inter-node edges (lock)

JSON array `sync_edge_ids` in this exact order (14 ids). JSON `n_sync_edges` = 14. Parallel `sync_edge_classes`. JSON array `sync_edge_class_ids` in this order (7 ids): `identity`, `residual`, `fanout`, `mtp`, `output`, `state`, `shared_weight`. JSON `n_sync_edge_classes` = 7.

| id | Class | From → to | Catalog IDs |
| --- | --- | --- | --- |
| `identity_e_h0` | `identity` | `embed` → first mixer | `e` identified as \(h^{(0)}\) |
| `residual_h` | `residual` | `mlp` → next mixer or `h_64` | `h` / `h_64` |
| `residual_h_mid` | `residual` | mixer → `mlp` | `h_mid` |
| `fanout_h64` | `fanout` | last language `mlp` → `lm_head` and `mtp_mix` | `h_64` |
| `embed_e_next` | `mtp` | MTP `embed` → `mtp_mix` | `e_next` |
| `mtp_u_to_block` | `mtp` | `mtp_mix` → MTP `gated_attn` | `mtp_u` as `h` |
| `h_mtp_to_logits` | `mtp` | MTP `mlp` → MTP `lm_head` | `h_mtp` |
| `output_logits_0` | `output` | primary `lm_head` → `output` | `logits_0` |
| `output_logits_1` | `output` | MTP `lm_head` → `output` | `logits_1` |
| `state_kv` | `state` | `gated_attn` token-crossing | `K_state`, `V_state` |
| `state_c` | `state` | `gated_delta_net` token-crossing | `C_state` |
| `state_s` | `state` | `gated_delta_net` token-crossing | `S` |
| `shared_E` | `shared_weight` | both `embed` instances | `E` |
| `shared_W_lm` | `shared_weight` | both `lm_head` instances | `W_lm` |

JSON `sync_edge_classes` parallel to `sync_edge_ids`: `["identity","residual","residual","fanout","mtp","mtp","mtp","output","output","state","state","state","shared_weight","shared_weight"]`.

These 14 edges **are** the synchronization/materialization edges named by the ledger. TASK-12 may fuse **across** a non-state, non-output edge only as a HYPOTHESIS that merges or bypasses node types; this document does not authorize that merge. State edges cannot be dropped: omitting a KV/\(C\)/\(S\) write changes the map.

Optional vision replace into \(h^{(0)}\) annotates `identity_e_h0`; it is not a 15th edge type and not a node (`vision_interface_is_not_a_node` true).

### Work, traffic, and precision citations (lock)

Cite; do not recopy TASK-06 tables or TASK-07’s 20-row severity table. Checker **recomputes** the cited MAC integers from `text_config` with the same identities as TASK-06 (full proj \(12288\cdot5120+2\cdot1024\cdot5120+5120\cdot6144\); linear token sum; MLP \(3IH\); `lm_head` \(VH\); GDN \(3\cdot48\cdot128\cdot128\); `mtp.fc` \(H\cdot 2H\)).

JSON keys (ints unless noted):

`mac_embed` 0, `mac_full_proj_per_layer` 104857600, `mac_attn_coeff_per_full_layer` 12288, `mac_lin_token_per_layer` 118235136, `mac_lin_conv_per_layer` 40960, `mac_gdn_per_layer` 2359296, `mac_mlp_per_layer` 267386880, `mac_lm_head` 1271398400, `mac_mtp_fc` 52428800, `weight_gather_bytes_per_row` 10240, `kv_bytes_per_full_layer_per_token` 4096, `kv_bytes_all_per_token` 69632, `c_bytes_per_layer` 61440, `s_bytes_per_layer` 3145728, `s_bytes_all` 150994944, `weight_bytes_lm_head` 2542796800.

JSON numbers: `i_mlp_weight_only` 1, `i_lm_head_weight_only` 1, `i_gdn_vs_s_rw` 0.75, `i_attn_core_vs_kv` 6.

JSON `bottleneck_labels` copied from TASK-06 in TASK-06 order: `weight_memory`, `vocab_memory`, `state_memory`, `kv_memory`, `quadratic_attn`, `compute`. Restating a label here is a **citation**, still HYPOTHESIS, not a new ranking. JSON `n_bottleneck_labels` = 6.

JSON object `node_sensitive_ops` keyed by `node_type_ids`:

- `embed`: `["param_bf16"]`
- `gated_attn`: `["param_bf16","residual_stream","live_across_gates","silu_sigmoid","rms_hidden","rms_head","softmax_over_T","attn_av_over_T","gemm_k5120","rope_phase","state_kv_bf16"]`
- `gated_delta_net`: `["param_bf16","residual_stream","live_across_gates","silu_sigmoid","rms_hidden","rms_head","l2_gdn","gemm_k5120","gdn_S_recurrent","gdn_inner_d128","gdn_alpha_beta","state_c_bf16","s_below_f32","conv_fir"]`
- `mlp`: `["param_bf16","residual_stream","silu_sigmoid","rms_hidden","gemm_k5120","gemm_k17408"]`
- `lm_head`: `["param_bf16","rms_hidden","gemm_lm_head"]`
- `mtp_mix`: `["param_bf16","rms_hidden","gemm_k5120"]`

Do not add sensitive-op ids. Do not claim validation survival (TASK-18). JSON `precision_roles` `["param","activation","accum","state"]`. Config dtypes remain conceptual (`param_dtype` `"bfloat16"`, `s_conceptual_dtype` `"float32"`, `kv_conceptual_dtype` `"bfloat16"`). `activation_dtype_decided` false.

### Flexibilities and non-decisions (lock)

JSON array `flexibility_kind_ids` in this exact order (5 ids). JSON `n_flexibility_kinds` = 5.

| id | Meaning | Selected here? |
| --- | --- | --- |
| `algebraic_equivalent` | TASK-02 same real map inside the node | allowed, not a new node |
| `fuse_internals` | TASK-12 may fuse listed internals (HYPOTHESIS usefulness) | not selected |
| `split_at_internal` | TASK-12 may cut at a named internal (HYPOTHESIS) | not selected |
| `recompute_ephemeral` | TASK-04 recomputable values may be dropped and rebuilt | not a store winner |
| `shared_weight_reuse` | one payload for `E` / `W_lm` | required sharing, not a layout |

JSON array `split_candidate_ids` in this exact order (10 ids). JSON `n_split_candidates` = 10. Every id is an **internal** catalog ID. Usefulness of each split is HYPOTHESIS.

`g`, `z`, `h_tilde`, `k_rope`, `v_full`, `qkv`, `h_post`, `swiglu`, `h_final`, `mtp_cat`

JSON `split_selected` is an object keyed in that order with every value `false`. JSON `n_split_candidates_selected` = 0. JSON `fusion_selected` false. JSON `schedule_selected` false.

Non-decisions (prose required): TASK-12 owns which internals materialize and which split/fuse hypotheses to try; TASK-13/14 own decode vs prefill **order of the same nodes** and extra boundary traffic; TASK-15 owns tiles; TASK-17 owns CUDA mappings **per node type**; TASK-08–10 own element codes. This graph does not change when a TASK-08 recipe is later applied.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 6 (Inter-node edges). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers or 135 instances. Annotate 3:1 mixer xor in prose, not as 64 subgraphs.

Required IDs **inside that fence**: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`, `h`, `h_mid`, `h_64`, `K_state`, `V_state`, `C_state`, `S`, `logits_0`, `logits_1`.

JSON `n_diagrams` is 1. `diagram_ids` is `["embed","gated_attn","gated_delta_net","mlp","lm_head","mtp_mix","h","h_mid","h_64","K_state","V_state","C_state","S","logits_0","logits_1"]`.

### Deferred vision

Visual tokens may replace placeholders on the `identity_e_h0` edge (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger execution contracts are **UNKNOWN**. Do not add a vision node type. `vision_interface_is_not_a_node` true. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_semantic_graph.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, and the cited MAC/byte products (same identities as TASK-06/04). Duplicate TASK-03 catalog id order and TASK-06/07 id lists as constants; do not import them.

CLI (cwd = repository root):

```text
python3 scripts/check_semantic_graph.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_semantic_graph.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantic-graph docs/architecture/semantic-graph.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: instance counts, cited MAC/byte products, intensities. Constant fields: canonical sentences, boundary-question sentence, node-type/edge/flexibility/split lists, catalog partition, sensitive-op attachments, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--semantic-graph PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all three canonical sentences verbatim, (5) `boundary_question_sentence` verbatim, (6) every `node_type_ids`, `sync_edge_ids`, `flexibility_kind_ids`, `split_candidate_ids`, `boundary_rule_ids`, `catalog_ids`, `boundary_ids`, `internal_ids`, `state_ids`, and `bottleneck_labels` id present as a substring, (7) the diagram’s required IDs present **inside that mermaid fence**, (8) none of `TBD`, `TODO`, `???`, (9) no `UNKNOWN` except inside the Deferred vision section, (10) every locked document integer/decimal below present as a decimal or integer substring, (11) the words `HYPOTHESIS` and `hardware-independent` present, (12) none of the forbidden winner phrases: `selected fusion`, `selected schedule`, `selected kernel`, `selected split`, `should fuse`, `recommend fusion`, `framework module is the node`, `GGUF is the graph`, `Quartz graph`, `llama.cpp graph` (allow the substring only inside `not a selected fusion` / `not a fusion or schedule decision` / `not kernels`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks cited TASK-06/07 integers against those documents).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `n_node_types==6`, `n_node_instances_complete==135`, `n_node_instances_language==130`
- `n_embed_instances==2`, `n_gated_attn_instances==17`, `n_gated_delta_net_instances==48`, `n_mlp_instances==65`, `n_lm_head_instances==2`, `n_mtp_mix_instances==1`
- `n_node_instances_complete == n_embed_instances+n_gated_attn_instances+n_gated_delta_net_instances+n_mlp_instances+n_lm_head_instances+n_mtp_mix_instances`
- `n_catalog_nodes==52`, `n_boundary_ids==10`, `n_internal_ids==38`, `n_state_ids==4`
- `n_boundary_ids+n_internal_ids+n_state_ids==52`
- the three partition arrays are pairwise disjoint and their union equals `catalog_ids`
- `live_across_ids == ["g","z"]` and both are in `internal_ids`
- `n_sync_edges==14`, `n_sync_edge_classes==7`, `n_flexibility_kinds==5`, `n_split_candidates==10`, `n_split_candidates_selected==0`
- `n_boundary_rules==6`, `n_diagrams==1`, `n_bottleneck_labels==6`
- `mac_embed==0`, `mac_full_proj_per_layer==104857600`, `mac_attn_coeff_per_full_layer==12288`
- `mac_lin_token_per_layer==118235136`, `mac_mlp_per_layer==267386880`, `mac_lm_head==1271398400`
- `mac_gdn_per_layer==2359296`, `mac_mtp_fc==52428800`, `mac_lin_conv_per_layer==40960`
- `mac_full_proj_per_layer == 24*256*hidden_size + 2*4*256*hidden_size + hidden_size*(24*256)`
- `mac_mlp_per_layer == 3 * intermediate_size * hidden_size`
- `mac_lm_head == vocab_size * hidden_size`
- `mac_gdn_per_layer == 3 * 48 * 128 * 128`
- `i_mlp_weight_only==1`, `i_lm_head_weight_only==1`, `i_gdn_vs_s_rw==0.75`, `i_attn_core_vs_kv==6`
- `kv_bytes_all_per_token==69632`, `s_bytes_all==150994944`, `weight_gather_bytes_per_row==10240`
- `decode_prefill_share_graph is True`, `hardware_independent is True`, `fusion_selected is False`
- `schedule_selected is False`, `live_across_are_internal is True`, `rms_inside_consumer is True`
- `rms_roles_not_collapsed is True`, `residual_add_inside_mixer is True`, `residual_add_inside_mlp is True`
- `framework_primitives_rejected is True`, `ledger_open_question_boundaries_closed is True`
- `gdn_primary_is_recurrent_eq_17 is True`, `chunkwise_fp_gap_is_hypothesis is True`
- `activation_dtype_decided is False`, `vision_interface_is_not_a_node is True`
- `primary_includes_mtp is True`, `mixer_xor_by_layer_types is True`
- every `split_selected` value is False
- `node_type_ids` equals the locked 6-id list; `sync_edge_ids` equals the locked 14-id list
- `"h_tilde" in node_internals["gated_attn"]` and `"h_tilde" in node_internals["gated_delta_net"]`
- `"g" in node_internals["gated_attn"]` and `"z" in node_internals["gated_delta_net"]`
- `"mix_full" in node_internals["gated_attn"]` and `"mlp_out" in node_internals["mlp"]`

Locked document integers/decimals the `--semantic-graph` check must find:

`5120`, `17408`, `248320`, `135`, `130`, `17`, `65`, `104857600`, `118235136`, `267386880`, `1271398400`, `52428800`, `2359296`, `12288`, `40960`, `10240`, `69632`, `150994944`, `2542796800`, `3145728`, `61440`, `4096`, `0.75`

(The integer `48` / `16` / `64` / `6` / `52` / `14` / `10` / `38` will appear from counts and types; requiring the large MAC/byte set is the hard check. Do not require `1e-06`.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `n_attn_heads`, `n_kv_heads`, `head_dim`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `bytes_bf16`, `bytes_f32`,

`node_type_ids`, `n_node_types`, `n_embed_instances`, `n_gated_attn_instances`, `n_gated_delta_net_instances`, `n_mlp_instances`, `n_lm_head_instances`, `n_mtp_mix_instances`, `n_node_instances_complete`, `n_node_instances_language`,

`boundary_rule_ids`, `n_boundary_rules`,

`n_catalog_nodes`, `catalog_ids`, `boundary_ids`, `internal_ids`, `state_ids`, `n_boundary_ids`, `n_internal_ids`, `n_state_ids`, `live_across_ids`, `intra_equation_reuse_ids`, `shared_weight_ids`, `catalog_partition_complete`,

`node_inputs`, `node_outputs`, `node_state`, `node_internals`,

`sync_edge_ids`, `sync_edge_classes`, `sync_edge_class_ids`, `n_sync_edges`, `n_sync_edge_classes`,

`mac_embed`, `mac_full_proj_per_layer`, `mac_attn_coeff_per_full_layer`, `mac_lin_token_per_layer`, `mac_lin_conv_per_layer`, `mac_gdn_per_layer`, `mac_mlp_per_layer`, `mac_lm_head`, `mac_mtp_fc`, `weight_gather_bytes_per_row`, `kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`, `c_bytes_per_layer`, `s_bytes_per_layer`, `s_bytes_all`, `weight_bytes_lm_head`, `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_gdn_vs_s_rw`, `i_attn_core_vs_kv`, `bottleneck_labels`, `n_bottleneck_labels`,

`precision_roles`, `node_sensitive_ops`, `param_dtype`, `kv_conceptual_dtype`, `s_conceptual_dtype`,

`flexibility_kind_ids`, `n_flexibility_kinds`, `split_candidate_ids`, `n_split_candidates`, `split_selected`, `n_split_candidates_selected`,

`decode_prefill_share_graph`, `residual_add_inside_mixer`, `residual_add_inside_mlp`, `residual_input_live_until_add`, `rms_inside_consumer`, `rms_roles_not_collapsed`, `live_across_are_internal`, `hardware_independent`, `cuda_mapping_deferred`, `fusion_selected`, `schedule_selected`, `layout_selected`, `framework_primitives_rejected`, `fanout_neq_must_store`, `node_io_neq_must_store`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_fp_gap_is_hypothesis`, `activation_dtype_decided`, `vision_interface_is_not_a_node`, `primary_includes_mtp`, `mtp_omission_is_algebraic_equivalent`, `mtp_norm_has_no_extra_catalog_id`, `mixer_xor_by_layer_types`, `ledger_open_question_boundaries_closed`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_contracts`, `canonical_sentence_internals`, `boundary_question_sentence`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. Intensities `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_attn_core_vs_kv` are JSON numbers `1`, `1`, `6`. `i_gdn_vs_s_rw` is JSON number `0.75`. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. `node_inputs`, `node_outputs`, `node_state`, `node_internals`, `node_sensitive_ops` are JSON objects keyed in `node_type_ids` order. `split_selected` is a JSON object keyed in `split_candidate_ids` order with JSON `false` values.

`node_inputs` values (lock): `embed` `["token_id"]`; `gated_attn` `["h"]`; `gated_delta_net` `["h"]`; `mlp` `["h_mid"]`; `lm_head` `["h_64"]`; `mtp_mix` `["h_64","e_next"]`. MTP `lm_head` input `h_mtp` and MTP `gated_attn` input `mtp_u` are instance identifications stated in prose and `mtp_u_to_block` / `h_mtp_to_logits`; do not add a second JSON key.

`node_outputs` values (lock): `embed` `["e"]`; `gated_attn` `["h_mid"]`; `gated_delta_net` `["h_mid"]`; `mlp` `["h"]`; `lm_head` `["logits_0"]`; `mtp_mix` `["mtp_u"]`. Secondary outputs `e_next`, `h_64`, `h_mtp`, `logits_1` are instance identifications in prose and sync edges.

### Stage split

- **Implementation** writes `scripts/check_semantic_graph.py` **and** `docs/architecture/semantic-graph.md` (six node types, catalog partition, 14 sync edges that close the ledger open question, per-node contracts, 5 flexibility kinds, 10 unselected split candidates, JSON fence). Runs `--json` and `--semantic-graph` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-07 / `numerical-sensitivity.md` style), Authority table links to this dossier / semantics / dataflow / lifetime / work-and-traffic / numerical-sensitivity / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, node-type ids, sync-edge ids, split candidates, or Mermaid node IDs. Does not edit TASK-02/03/04/06/07 artifacts.
- **Verification** independently re-runs focused commands, recomputes instance counts and cited MAC products from sitting `text_config` (not from JSON echo), spot-checks cited TASK-06/07 integers against `docs/architecture/work-and-traffic.md` and `docs/architecture/numerical-sensitivity.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-graph, no winner phrases, no `plan.md` or ledger edit, no payload I/O, that every split remains unselected, that contracts are hardware-independent (no CUDA kernel/thread/SKU decisions), that live-across `g`/`z` are internal, and that the boundary open question is closed by the six types + 14 edges. Fusion, schedule, layout, and CUDA mapping remain open.

- Invariants:
  - Ten level-2 headings in the locked order; three canonical sentences plus `boundary_question_sentence` verbatim; 6 node types; 135 complete instances; 52 catalog IDs partitioned 10/38/4; 14 sync edges; 5 flexibility kinds; 10 split candidates all unselected; one Mermaid flowchart with required IDs.
  - Prefill/decode share one graph; primary includes MTP; mixer xor by `layer_types`; residual add inside mixer and `mlp`; RMS inside consumer; two RMS roles plus GatedRMSNorm not collapsed; `g`/`z` internal.
  - GDN numerical definition is recurrent `(17)`–`(18)`; chunkwise FP gap stays HYPOTHESIS.
  - Logical values do not imply allocation; node I/O ≠ must-store; contracts are hardware-independent.
  - Vision encoder remains unexpanded and is not a node type.
- Rejected alternatives:
  - Copying Hugging Face modules, GGML ops, or CUDA kernels as node types: rejected; ledger requires semantic evidence, not framework primitives; plan.md forbids Quartz/llama.cpp inspection.
  - One node per catalog ID: rejected; that is TASK-03; contracts would not coarsen the DAG.
  - One `layer` node wrapping mixer+MLP: rejected; TASK-04 `h_mid` residual-add and TASK-06 distinct work.
  - One `attention` node for GDN and Gated Attention: rejected; different state, \(T\)-scaling, and TASK-07 risks.
  - Required proj/core/out split of `gated_attn` or conv/recurrence/out split of GDN: rejected as **required** types; those are TASK-12 `split_at_internal` hypotheses because `g`/`z` are live-across internals.
  - Residual-add or RMSNorm as their own required types: rejected; residual add is the I/O contract inside mixer/`mlp`; RMS is inside the consumer; collapsing RMS roles is forbidden by TASK-02.
  - Separate decode and prefill graphs: rejected; same map; TASK-13/14 schedule \(T\) and incoming state.
  - Selecting any fusion, split, schedule, tile, or CUDA mapping: TASK-12/13/14/15/17.
  - Treating chunkwise GDN as a second node or as zero \(S\) traffic: rejected; TASK-02/06/07.
  - Inventing catalog IDs (including a post-`mtp.norm` activation): rejected; MTP RMS has no extra catalog ID.
  - Using GGUF Q4 or CUDA dtypes as contract dtypes: rejected; conceptual BF16/F32 from TASK-04/07.
  - Filling TASK-16 SKU peaks or measuring tok/s: forbidden; not this task.
  - Inspecting Quartz or llama.cpp for “real” graphs: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–10.
  - Editing frozen TASK-02/03/04/06/07 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py`.
  - Leaving the ledger open question unclassified: rejected; the six cuts + internals + 14 edges are the completion criterion. Closing it by picking a fusion: also rejected.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/semantic-graph.md` exists and follows the heading list above.
  - Node boundaries are derived from the six locked semantic cut rules, not from framework primitives; six types `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix` are specified.
  - Each type specifies operations (equation tags), I/O, state, internal values, and flexibilities.
  - Contracts are hardware-independent (no CUDA kernel/thread/SKU decisions; `hardware_independent` true; `cuda_mapping_deferred` true).
  - Ledger open question is closed by the boundary-derivation heading, `boundary_question_sentence`, catalog partition (10/38/4), and 14 sync edges; `n_split_candidates_selected` = 0; `fusion_selected` false; `schedule_selected` false.
  - Three canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **winners**; no Quartz/llama.cpp; GGUF is not the graph; no payload re-stream; no `plan.md` or ledger edit.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_semantic_graph.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_semantic_graph.py
python3 scripts/check_semantic_graph.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_semantic_graph.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantic-graph docs/architecture/semantic-graph.md
```

- Candidate quality: not required — no model execution or NLL; this increment is semantic-graph documentation. Fusion/split **usefulness** is HYPOTHESIS prose, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/semantic-graph.md
python3 -m py_compile scripts/check_semantic_graph.py
python3 scripts/check_semantic_graph.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantic-graph docs/architecture/semantic-graph.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run other `scripts/check_*.py` as a requirement of this task (verifier may spot-check TASK-06/07 markdown integers independently).

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/semantic-graph.md` (create)
  - `scripts/check_semantic_graph.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-02/03/04/06/07 deliverables unchanged)
- Definition of done: semantic-graph document published with locked six node types, per-node ops/I/O/state/internals/flexibilities, 14 sync edges, and boundary derivation that closes the TASK-11 open question; contracts hardware-independent; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-11 completion checkboxes can be marked at delivery; fusion, schedule, layout, and CUDA mapping remain open.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T14:16:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-11.md`. Coupled IDs `none`. Document structure (10 headings), three canonical sentences plus `boundary_question_sentence`, six node types (135 complete instances), catalog partition 10/38/4, 14 sync edges, five flexibility kinds, ten unselected split candidates, stdlib checker `scripts/check_semantic_graph.py`, JSON schema, and acceptance commands are closed. Ledger open question closed **by DERIVED node cuts, internals, and sync edges**, not by selecting fusion or a schedule. `docs/architecture/semantic-graph.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit.
- Performance evidence applied: N/A — semantic-graph contract documentation; fusion/split usefulness is hypothesis, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_semantic_graph.py` (stdlib checker: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`; no imports of other `scripts/check_*.py` or payload streamers). Instantiates the locked summary from sitting `text_config` plus TASK-02/03/04/06/07 constants; `--json` asserts six node types, 135 complete / 130 language instances, catalog partition 10/38/4, 14 sync edges, five flexibility kinds, ten unselected split candidates, cited MAC/byte identities, and hardware-independent booleans; `--semantic-graph` checks headings, JSON fence equality, one Mermaid flowchart, three canonical sentences plus `boundary_question_sentence`, locked integers, and forbidden winner phrases.
  - Created `docs/architecture/semantic-graph.md` with the ten locked level-2 headings, three canonical sentences plus `boundary_question_sentence`, six node types with per-node ops/I/O/state/internals/flexibilities, catalog partition and 14 sync edges that close the ledger open question, five flexibility kinds, ten unselected split candidates, one Mermaid flowchart, and a JSON fence copied from a live `--json` run. No fusion, schedule, layout, or CUDA mapping selected. Did not edit `plan.md`, `task_ledger.md`, TASK-02/03/04/06/07 deliverables, or stream safetensor payloads. No commit.
- Commands:
  - `python3 -m py_compile scripts/check_semantic_graph.py` — exit 0
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; live object matches locked identities (`n_node_instances_complete=135`, `n_node_instances_language=130`, `n_sync_edges=14`, `n_split_candidates_selected=0`, `mac_full_proj_per_layer=104857600`, `mac_lin_token_per_layer=118235136`, `mac_mlp_per_layer=267386880`, `mac_lm_head=1271398400`, `i_gdn_vs_s_rw=0.75`, `fusion_selected=false`, `ledger_open_question_boundaries_closed=true`)
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md` — exit 0
- UTC/time/tokens/cost: `2026-09-20T14:31:40Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/semantic-graph.md` — mechanical pass only. Draft-status banner (`unverified`) already present in TASK-07 / [`numerical-sensitivity.md`](../numerical-sensitivity.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`); left unchanged. Authority table already cross-links this dossier, [`model-semantics.md`](../model-semantics.md) (TASK-02), [`dataflow.md`](../dataflow.md) (TASK-03), [`lifetime-and-state.md`](../lifetime-and-state.md) (TASK-04), [`work-and-traffic.md`](../work-and-traffic.md) (TASK-06), [`numerical-sensitivity.md`](../numerical-sensitivity.md) (TASK-07), [`model-inventory.md`](../model-inventory.md) (TASK-01), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_semantic_graph.py`](../../../scripts/check_semantic_graph.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings in locked order and the first JSON fence verified unchanged (byte-identical to live `--json`). No locked integers, canonical sentences, `node_type_ids`, `sync_edge_ids`, `split_candidate_ids`, or Mermaid node IDs modified. Frozen TASK-02/03/04/06/07 artifacts and `plan.md` not edited.
- Commands:
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md` — pass (exit 0; headings, JSON fence, one flowchart, three canonical sentences plus `boundary_question_sentence`, locked ids/integers, `HYPOTHESIS` / `hardware-independent`; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T14:32:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `composer-2.5` (verifier subagent)
- Diff review:
  - New: `docs/architecture/semantic-graph.md`, `scripts/check_semantic_graph.py`, `docs/architecture/tasks/TASK-11.md`.
  - Modified: `docs/architecture/task_ledger.md` — `TASK-11` status only (`TODO` → `IN PROGRESS`); no completion checkboxes, established results, or open-question edits. Acceptable admission/coordinator mark; not a delivery ledger advance.
  - Unchanged: `docs/architecture/plan.md`, TASK-02/03/04/06/07 deliverables, frozen upstream docs.
  - Scope matches dossier: six DERIVED node types, per-node ops/I/O/state/internals/flexibilities, catalog partition 10/38/4, 14 sync edges, five flexibility kinds, ten unselected split candidates, one Mermaid flowchart, hardware-independent contracts. No fusion/schedule/layout/CUDA winners. Stdlib checker only; no imports of sibling `scripts/check_*.py`.
- Independent raw-record checks:
  - Instance counts from sitting `text_config` `layer_types`: `full_attention_indices` `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`; `n_linear_layers=48`, `n_full_layers=16`; complete instances `2+17+48+65+2+1=135`; language `1+16+48+64+1=130`. Matches checker JSON (not JSON-echo alone).
  - MAC/byte products recomputed via TASK-06 documented identities (`12288·5120+2·1024·5120+5120·6144`, `3·17408·5120`, `248320·5120`, `3·48·128·128`, `5120·10240`, `69632`, `150994944`, `2542796800`, gather `10240`, intensities `i_gdn_vs_s_rw=0.75`, `i_attn_core_vs_kv=6`). Spot-checked against `docs/architecture/work-and-traffic.md` table rows and JSON fence values — agreement.
  - TASK-07 sensitive-op ids (`param_bf16`, `softmax_over_T`, `gdn_S_recurrent`, `gemm_lm_head`, node attachments) spot-checked against `docs/architecture/numerical-sensitivity.md` — agreement.
  - First fenced `json` block byte-identical to live `--json` output. One `mermaid` fence with `flowchart` and all required diagram IDs present.
  - `UNKNOWN` appears only under **Deferred vision**. No `TBD`/`TODO`/`???`. Forbidden winner phrases absent (allowed negations only). `g`/`z` internal; `n_split_candidates_selected=0`; `fusion_selected`/`schedule_selected` false.
  - Candidate quality: not required (no GPU/NLL gate).
- Commands:
  - `python3 -m py_compile scripts/check_semantic_graph.py` — exit 0
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md` — exit 0
  - `test -f docs/architecture/semantic-graph.md` — pass
  - Repository-wide duplicate of `--semantic-graph` check — exit 0
  - Ruff/pytest/CUDA/other `check_*.py` — not run (out of scope per dossier)
- Formatting changed files: none (`uv run ruff format .` not required for this increment)
- Verdict: **PASS** (first attempt)
- UTC/time/tokens/cost: `2026-09-20T14:35:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-11 only; coupled IDs `none`
- Outcome: TASK-11 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T14:36:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/semantic-graph.md` (10 locked headings, three canonical sentences plus `boundary_question_sentence`, six DERIVED node types with per-node ops/I/O/state/internals/flexibilities, 52-ID catalog partition 10/38/4, 14 sync edges, five flexibility kinds, ten unselected split candidates, one Mermaid flowchart, JSON fence); `scripts/check_semantic_graph.py` stdlib checker; independent `text_config` recomputation and TASK-06/07 citation spot-checks pass; frozen upstream docs unchanged; fusion, schedule, layout, and CUDA mapping remain open; boundary open question closed by DERIVED cuts
- Candidate measured delta: N/A — semantic-graph documentation
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (graph contract locks; no tok/s evidence)
- Throughput delta: N/A — TASK-11 does not execute or time the model
- Commit: Publish Qwen3.8 semantic graph
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification PASS; ledger update at delivery)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/authorities/qwen3.8-27b-transformers/config.json` must remain present for focused commands
