# TASK-12 — Determine materialization and fusion opportunities

## Control

- Primary ID: `TASK-12`
- Coupled IDs: `none`
- Dependencies: `TASK-03`, `TASK-04`, `TASK-11` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Classify important intermediates with required justification; List reuse, recomputation, local-working-set, and synchronization tradeoffs; Label every fusion proposal as a hypothesis.

## Goal and boundaries

Produce `docs/architecture/materialization-and-fusion.md` as the Phase 1 **hardware-independent physical-materialization and fusion-hypothesis** analysis for the Qwen3.8-27B language+MTP map. Classify every TASK-03 catalog ID’s physical-materialization **need** with a TASK-04/11 justification (DERIVED). Enumerate reuse, recompute, local-working-set, and synchronization **tactics** (availability DERIVED; usefulness HYPOTHESIS). List every fusion/split/fuse-across-edge proposal as a **HYPOTHESIS**; select none. Close the ledger open question (which apparent fusions improve total behavior after working-set and synchronization costs) by enumerating those hypotheses with cost **identities**, not by picking a winner.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Catalog IDs, fan-out, live-across, and high-fan-out ranking come from `docs/architecture/dataflow.md` (TASK-03). Lifetime classes, must-survive boundaries, recomputability, and \(K,V,C,S\) bytes come from `docs/architecture/lifetime-and-state.md` (TASK-04). Node types, internals, 14 sync edges, ten split candidates, and five flexibility kinds come from `docs/architecture/semantic-graph.md` (TASK-11). Do not add catalog IDs, node types, or sync edges as **required** contracts.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established), `DERIVED` (physical-need class, byte identities from ranks × TASK-04 element size, extra-sync-edge counts), or `HYPOTHESIS` (every fusion/split/reuse/recompute **usefulness**, every “this fusion improves total behavior”). `UNKNOWN` only for vision-encoder internals deferred here. No `MEASURED` tok/s or NLL. No selected fusion, schedule, layout, or kernel.
  - GitHub Markdown math. Cite TASK-02 equation tags via TASK-11 contracts, TASK-03 catalog IDs, TASK-04 lifetime/state IDs, TASK-11 node types / sync-edge ids / split candidates. Do not rewrite forward math, redraw the eight TASK-03 DAGs, recompute TASK-04 persistent-state totals as a new lifetime study, or recopy TASK-06 MAC/traffic tables as a new work study (TASK-06 is not a dependency).
  - Allowed evidence: TASK-03 dataflow, TASK-04 lifetime/state, TASK-11 semantic-graph, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier. Byte identities instantiate TASK-03 ranks × TASK-04 conceptual BF16/F32 widths. TASK-11 already-cited MAC/byte integers may be **cited** from `semantic-graph.md`, not recopied as a new ranking. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No TASK-16 occupancy/fusion-vs-occupancy algebra (not a dependency; this document is hardware-independent). No TASK-08/09/10 recipe, packing, or compiler-stage decisions. No TASK-13/14 schedules.
  - Hardware-independent: catalog IDs, node types, sync-edge ids, rank×dtype bytes. No thread geometry, warps, SMs, CUDA dtypes, kernel names, streams, shared-memory tiles, or sitting-GPU numbers.
- Non-goals:
  - No fusion, split, reuse, or recomputation **winner**. Listing a hypothesis is not selecting it. `n_fusion_hypotheses_selected` = 0.
  - No decode/prefill **schedules** (TASK-13/14). Prefill and decode share **one** physical-need taxonomy; only \(T\) and incoming \((K,V,C,S)\) change. Extra boundary traffic of a schedule is TASK-13/14.
  - No physical layouts (TASK-15) and no CUDA mapping alternatives (TASK-17). Working-set means catalog-rank × conceptual dtype bytes, not a CUDA live-set or occupancy limit.
  - No quantization recipes, runtime-format packing, or compiler stages (TASK-08/09/10).
  - No quality/NLL experiments (TASK-18) and no tok/s (TASK-19).
  - No new operators, extra catalog IDs, extra required node types, or 64-layer unrolling.
  - No generic GEMM / softmax / RMSNorm kernel-fusion cookbook. Intra-node `fuse_internals` names catalog IDs, not CUDA kernels.
  - No peak-memory claim and no summed CUDA live-set. A residual-vector byte size is DERIVED rank × dtype, not a physical working-set peak.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–11). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `dataflow.md`, `lifetime-and-state.md`, `semantic-graph.md`, `model-semantics.md`, `work-and-traffic.md`, `numerical-sensitivity.md`, `model-inventory.md`, `cuda-hardware-model.md`, or any TASK-08–10 deliverable.
  - Do not import other `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib materialization/fusion checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places materialization/fusion after the Qwen-specific semantic graph and before decode/prefill plans and CUDA mappings.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:89-92` — TASK-11 derives semantic regions; TASK-12 identifies materialization and fusion hypotheses; TASK-13/14 schedule the same nodes; TASK-17 CUDA-maps nodes.
- `docs/architecture/task_ledger.md` TASK-12 row — produces `docs/architecture/materialization-and-fusion.md`; purpose is classify physical-materialization need and fusion experiment opportunities; open question is which apparent fusions improve total behavior after working-set and synchronization costs (**closed here by enumerating HYPOTHESIS proposals with cost identities, not by selecting a winner**); completion is classify important intermediates with required justification, list reuse/recompute/local-working-set/synchronization tradeoffs, label every fusion proposal as a hypothesis.
- `docs/architecture/task_ledger.md` TASK-03 established results — 52-ID catalog; live-across `g`/`z`; sharing rank 1–11; logical ≠ physical; fan-out ≠ must-store.
- `docs/architecture/task_ledger.md` TASK-04 established results — five lifetime classes; must-survive: token \(K,V,C,S\); residual-add `h`/`h_mid`; live-across `g`/`z`; \(B_\text{store}(T)=69632T+153944064\); high-fan-out ephemerals remain recomputable; physical materialization deferred to TASK-12.
- `docs/architecture/task_ledger.md` TASK-11 established results — six node types; 14 sync edges; five flexibility kinds; ten unselected split candidates; `g`/`z` internal; fuse-across a non-state, non-output edge is allowed only as a HYPOTHESIS; state edges cannot be dropped.
- `docs/architecture/task_ledger.md` TASK-13/14/17 — consumers of this classification; do not perform those designs here.
- `docs/architecture/dataflow.md` — 52 catalog IDs in locked order; ranks; high-fan-out IDs `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `k_rope`, `v_full`, `qkv`, `S`; live-across `g`/`z`; intra-equation `k_hat`; shared \(E\)/`W_\text{lm}`; canonical logical-≠-physical sentence.
- `docs/architecture/lifetime-and-state.md` — lifetime taxonomy; recomputable vs requires-prior-state; residual vector 10240 B; conceptual BF16 \(K,V,C\) and F32 \(S\); semantic storage candidates are mathematical, not CUDA.
- `docs/architecture/semantic-graph.md` — six node types; catalog partition 10/38/4; 14 sync edges; split candidates `g`, `z`, `h_tilde`, `k_rope`, `v_full`, `qkv`, `h_post`, `swiglu`, `h_final`, `mtp_cat` all unselected; `fuse_internals` not selected; residual add inside mixer/`mlp`; RMS inside consumer.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for layer counts, shapes, and byte products. Do not read safetensor payloads.
- `scripts/check_dataflow.py` / `scripts/check_lifetime_and_state.py` / `scripts/check_semantic_graph.py` — checker-style precedent. TASK-12’s checker is a sibling; do not import them.

## Performance evidence

N/A — materialization **classification** and fusion-**hypothesis** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Byte figures are TASK-03 ranks × TASK-04 conceptual dtypes (DERIVED arithmetic), not MEASURED traffic. Every fusion/split/reuse usefulness is HYPOTHESIS, not MEASURED. Do not apply the performance-evidence checklist to rank kernels or claim a winning fusion.

- Measurement identity: N/A (no engine binary). TASK-04/11 integers are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Classification coverage is 52 catalog IDs, 7 physical-need classes, 4 tradeoff tactics, 22 fusion hypotheses all unselected, 14 sync edges partitioned fusible/nonfusible/shared-weight.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` shape or a catalog ID disagrees with TASK-03/04/11, stop and fail closed (do not invent classes or hypotheses).
- Claim types: physical-need class `derived`; byte identities `derived`; extra-sync-edge counts `derived`; every fusion/split/reuse/recompute **usefulness** `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Classification completeness is 52-ID partition, 22 unselected hypotheses, open question closed as remaining-HYPOTHESIS.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for materialization and fusion claims

If a class, catalog ID, rank, byte identity, node type, or sync-edge id would disagree with TASK-03/04/11 or sitting `text_config`, the earlier document / config wins and this one is wrong.

- Prefill and decode share **one** physical-need taxonomy and **one** hypothesis list. Only \(T\) (stored KV length after append) and whether incoming \((K,V,C,S)\) is zeros versus populated change. Do not duplicate tables per mode.
- Primary classification **includes MTP**. Omitting MTP when only \(\ell^{(0)}\) is required is a TASK-02/11 algebraic equivalent, not a second taxonomy.
- TASK-04 mathematical must-survive is **not** a CUDA store. TASK-12 physical-need says which values **must** have a representation at a named boundary for the map to be evaluable without prefix replay or dropping I/O, versus which values may stay inside a node, be reused, be recomputed, or be promoted only under a HYPOTHESIS split.
- Fan-out ≠ must-store and node I/O ≠ must-store still hold. Declaring a catalog ID as node I/O is not a CUDA store. High-fan-out is a reuse **opportunity**, not a store requirement.
- Live-across `g` and `z` stay **internal** (TASK-11). They become hypothesized sync edges only under `split_g` / `split_z`. They are not required physical inter-node buffers.
- State edges cannot be dropped: omitting a KV/\(C\)/\(S\) write changes the map. Output edges cannot be dropped: omitting `logits_0` / `logits_1` drops the forward result. Shared-weight edges name required payload sharing, not node-type merges.
- TASK-12 may fuse **across** a non-state, non-output data edge only as a HYPOTHESIS that merges or bypasses node types. That hypothesis does not delete residual-add liveness, RMS-inside-consumer, or mixer xor.
- Algebraic equivalents (chunkwise GDN, SDPA, GQA-as-repeat, omitting MTP) are the same real map **inside** the owning node. They are not extra fusion hypotheses and not extra node types. Chunkwise FP gap stays TASK-11 HYPOTHESIS; GDN primary stays `(17)`–`(18)`; do not treat chunkwise as zero \(S\) traffic.
- Do not inspect Quartz or llama.cpp to “confirm” fusions or buffers.
- Do not select a fusion, split, schedule, tile, or CUDA mapping. Do not import TASK-16 occupancy algebra.

### Deliverable structure (`docs/architecture/materialization-and-fusion.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Fusion/split/reuse **usefulness** cells are `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, dataflow, lifetime, semantic-graph, inventory via those docs, config, checker; evidence labels; in-scope (language+MTP physical-need classes + fusion hypotheses) vs deferred (vision encoder; TASK-13/14 schedules; TASK-15 layouts; TASK-17 CUDA). State that the document specifies mathematical physical-need and experiment hypotheses, not kernels.
2. **Physical versus mathematical** — the four canonical sentences (exact text below) plus bullets: fan-out ≠ must-store; node I/O ≠ must-store; must-survive ≠ CUDA malloc; hypotheses remain hypotheses.
3. **Physical-need taxonomy** — the seven disjoint classes; assignment rules in order; default tactics; prefill/decode identity. This heading **starts** the ledger completion “classify with required justification.”
4. **Catalog physical-need table** — all 52 IDs in TASK-03 order (table below). Completes classification coverage.
5. **Forced stores and forbidden drops** — token state, inputs/outputs, residual-add liveness, RMS roles, nonfusible edges.
6. **Reuse, recompute, local-working-set, and synchronization tradeoffs** — four tactics; important intermediates; shared weights; intra-equation `k_hat`. Usefulness HYPOTHESIS.
7. **Fusion hypotheses** — the locked 22-id list, all HYPOTHESIS, none selected; extra-sync and working-set identities; one Mermaid summary (diagram 1 of 1). This heading **closes** the ledger open question together with sentence 3.
8. **Working-set and synchronization cost identities** — rank × dtype bytes cited; extra-sync-edge arithmetic; not a ranking and not TASK-06 recopied.
9. **Deferred vision** — residual-stream interface only.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Physical versus mathematical**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Physical-materialization need in this document is a mathematical survival class for evaluating the map, not a CUDA allocation, cache layout, or selected fusion.

> Every fusion, split, reuse, and recomputation proposal in this document is a HYPOTHESIS; none is a selected winner.

> Apparent fusions are enumerated with working-set and synchronization tradeoffs; which of them improve total behavior remains a HYPOTHESIS.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_physical` = sentence 2; `canonical_sentence_hypothesis` = sentence 3; `canonical_sentence_open_question` = sentence 4.

Sentence 4 **closes** the ledger open question in checker-substring form. JSON `ledger_open_question_fusions_remain_hypothesis` true.

Bullets required under that heading:

- Fan-out of a named value is not a requirement to store that value (`fanout_neq_must_store` true).
- Node I/O is not a requirement to allocate a CUDA buffer (`node_io_neq_must_store` true).
- TASK-04 “must survive a named boundary” is mathematical liveness, not physical malloc (`must_survive_neq_cuda_malloc` true).
- Prefill and decode share this taxonomy (`decode_prefill_share_taxonomy` true).
- Primary classification includes MTP (`primary_includes_mtp` true).
- Live-across `g`/`z` remain internal unless a split hypothesis is tried (`live_across_are_internal` true).
- Hardware mapping is TASK-17. Schedules are TASK-13/14. No fusion winner is selected (`fusion_winner_selected` false).

### Physical-need taxonomy (lock)

Seven classes, complete and disjoint over the 52 catalog IDs. JSON array `physical_need_class_ids` in this exact order (7 ids). JSON `n_physical_need_classes` = 7.

Assign **exactly one** class per catalog ID by applying these rules **in this order**. JSON `physical_need_rule_ids` in this exact order (7 ids). JSON `n_physical_need_rules` = 7.

| Rule id | If | Class | Default tactic |
| --- | --- | --- | --- |
| `rule_token_store` | TASK-04 `token-persistent` / `requires-prior-state` | `forced_token_store` | `must_store` |
| `rule_output` | TASK-04 `output-sink` | `forced_output` | `must_expose` |
| `rule_input` | catalog ID `token_id` | `forced_input` | `must_present` |
| `rule_boundary` | remaining TASK-11 `boundary_ids` | `boundary_tradeoff` | `keep_live_or_fuse` |
| `rule_live_across` | TASK-03/04/11 live-across `g`, `z` | `live_across_tradeoff` | `keep_inside_or_split` |
| `rule_reuse` | remaining TASK-03 high-fan-out activations (`h_tilde`, `h_post`, `k_rope`, `v_full`, `qkv`) | `reuse_tradeoff` | `reuse_or_recompute` |
| `rule_fuse_default` | all remaining catalog IDs | `fuse_default` | `fuse_or_recompute` |

JSON array `default_tactic_ids` in this exact order (7 ids), parallel to `physical_need_class_ids`: `must_store`, `must_expose`, `must_present`, `keep_live_or_fuse`, `keep_inside_or_split`, `reuse_or_recompute`, `fuse_or_recompute`. JSON `n_default_tactics` = 7.

Class meanings (required prose):

| Class | Meaning | Forced physical object? |
| --- | --- | --- |
| `forced_token_store` | Next-token evaluation without prefix replay requires a physical representation of this state. Omitting the store changes the map. | yes — the state IDs |
| `forced_output` | Forward-map outputs consumed as `output`. Dropping them drops the result. | yes — expose logits |
| `forced_input` | Graph input that must be presented. | yes — present `token_id` |
| `boundary_tradeoff` | TASK-11 node I/O that must remain mathematically available at a sync edge. A named CUDA buffer is not forced. Fuse-across, reuse, or recompute are HYPOTHESIS tactics that must still preserve residual-add / fan-out consumers. | no |
| `live_across_tradeoff` | Live-across internals. Physical inter-node buffer exists only if a split hypothesis is tried. Default: keep inside the owning node as a local working set. | no (unsplit) |
| `reuse_tradeoff` | High-fan-out ephemerals. Reuse of one copy vs recompute vs streaming into a forced state write is HYPOTHESIS. Not a token store. | no |
| `fuse_default` | Intra-node ephemerals whose unique consumer is the immediate successor (or a disjoint split). Default is fuse inside the node or recompute. Split candidates in this class remain unselected hypotheses. | no |

`S` is high-fan-out **and** token-persistent: rule 1 wins → `forced_token_store`, not `reuse_tradeoff`. `h` / `h_mid` / `h_64` are high-fan-out **and** TASK-11 boundary: rule 4 wins → `boundary_tradeoff`. `k_rope` / `v_full` / `qkv` write into forced state; the **activation** is `reuse_tradeoff` and the **surviving store** is the state ID.

JSON class arrays (catalog order inside each array; disjoint union = `catalog_ids`):

| JSON array | IDs | Count |
| --- | --- | ---: |
| `forced_token_store_ids` | `K_state`, `V_state`, `C_state`, `S` | 4 |
| `forced_output_ids` | `logits_0`, `logits_1` | 2 |
| `forced_input_ids` | `token_id` | 1 |
| `boundary_tradeoff_ids` | `e`, `h`, `h_mid`, `h_64`, `e_next`, `mtp_u`, `h_mtp` | 7 |
| `live_across_tradeoff_ids` | `g`, `z` | 2 |
| `reuse_tradeoff_ids` | `h_tilde`, `h_post`, `k_rope`, `v_full`, `qkv` | 5 |
| `fuse_default_ids` | `h_final`, `u_q`, `q_prime`, `k_raw`, `q_n`, `k_n`, `q_rope`, `attn`, `y_gate`, `mix_full`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `o`, `u_gdn`, `mix_lin`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next_n`, `h64_n`, `mtp_cat` | 31 |

JSON `n_forced_token_store` = 4, `n_forced_output` = 2, `n_forced_input` = 1, `n_boundary_tradeoff` = 7, `n_live_across_tradeoff` = 2, `n_reuse_tradeoff` = 5, `n_fuse_default` = 31. Sum 52. `physical_need_partition_complete` true.

JSON `important_ids` = forced + boundary + live-across + reuse_tradeoff, this exact catalog-relative order (21 ids):

`token_id`, `e`, `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `logits_0`, `g`, `v_full`, `k_rope`, `K_state`, `V_state`, `qkv`, `z`, `S`, `C_state`, `e_next`, `mtp_u`, `h_mtp`, `logits_1`

JSON `n_important_ids` = 21. These 21 are the “important intermediates” named by the ledger; the other 31 still appear in the catalog table as `fuse_default` with justification.

JSON `high_fanout_boundary_ids` `["h","h_mid","h_64"]` — residual/fan-out liveness plus reuse discussion. JSON `n_high_fanout_boundary_ids` = 3.

### Catalog physical-need table (lock all 52 IDs; copy into heading 4)

Table columns, in this order: `ID` | `Physical need` | `Default tactic` | `Justification`

Implementation copies these rows; do not add/remove IDs or reorder.

| ID | Physical need | Default tactic | Justification |
| --- | --- | --- | --- |
| `token_id` | `forced_input` | `must_present` | Graph input. |
| `e` | `boundary_tradeoff` | `keep_live_or_fuse` | TASK-11 `identity_e_h0`; identified as \(h^{(0)}\). |
| `h` | `boundary_tradeoff` | `keep_live_or_fuse` | TASK-04 residual-add; TASK-11 `residual_h`; fan-out 2. |
| `h_tilde` | `reuse_tradeoff` | `reuse_or_recompute` | TASK-03 high-fan-out 3 or 4; TASK-04 ephemeral. |
| `h_mid` | `boundary_tradeoff` | `keep_live_or_fuse` | TASK-04 residual-add Mix→MLP; TASK-11 `residual_h_mid`. |
| `h_post` | `reuse_tradeoff` | `reuse_or_recompute` | Fan-out to `g_mlp` and `up`. |
| `h_64` | `boundary_tradeoff` | `keep_live_or_fuse` | Fan-out to `lm_head` and `mtp_mix`; TASK-04 ephemeral. |
| `h_final` | `fuse_default` | `fuse_or_recompute` | `lm_head` internal; `split_h_final` HYPOTHESIS. |
| `logits_0` | `forced_output` | `must_expose` | Primary output sink. |
| `u_q` | `fuse_default` | `fuse_or_recompute` | Disjoint split parent, not reuse. |
| `q_prime` | `fuse_default` | `fuse_or_recompute` | Immediate successor `q_n`. |
| `g` | `live_across_tradeoff` | `keep_inside_or_split` | Live-across internal; `split_g` HYPOTHESIS. |
| `k_raw` | `fuse_default` | `fuse_or_recompute` | Immediate successor `k_n`. |
| `v_full` | `reuse_tradeoff` | `reuse_or_recompute` | Attn + `V_state` write; store is `V_state`. |
| `q_n` | `fuse_default` | `fuse_or_recompute` | Immediate successor `q_rope`. |
| `k_n` | `fuse_default` | `fuse_or_recompute` | Immediate successor `k_rope`. |
| `q_rope` | `fuse_default` | `fuse_or_recompute` | Immediate successor `attn`. |
| `k_rope` | `reuse_tradeoff` | `reuse_or_recompute` | Attn + `K_state` write; store is `K_state`. |
| `attn` | `fuse_default` | `fuse_or_recompute` | Immediate successor `y_gate`. |
| `y_gate` | `fuse_default` | `fuse_or_recompute` | Immediate successor `mix_full`. |
| `mix_full` | `fuse_default` | `fuse_or_recompute` | Residual add lives inside `gated_attn`. |
| `K_state` | `forced_token_store` | `must_store` | Requires prior-token state. |
| `V_state` | `forced_token_store` | `must_store` | Requires prior-token state. |
| `qkv` | `reuse_tradeoff` | `reuse_or_recompute` | FIR + `C_state` write; store is `C_state`. |
| `z` | `live_across_tradeoff` | `keep_inside_or_split` | Live-across internal; `split_z` HYPOTHESIS. |
| `a` | `fuse_default` | `fuse_or_recompute` | Immediate successor `alpha`. |
| `b` | `fuse_default` | `fuse_or_recompute` | Immediate successor `beta`. |
| `c_tilde` | `fuse_default` | `fuse_or_recompute` | Immediate successor `c`. |
| `c` | `fuse_default` | `fuse_or_recompute` | Disjoint QKV split parent, not reuse. |
| `q_lin` | `fuse_default` | `fuse_or_recompute` | Immediate successor `q_hat`. |
| `k_lin` | `fuse_default` | `fuse_or_recompute` | Immediate successor `k_hat`. |
| `v_lin` | `fuse_default` | `fuse_or_recompute` | Immediate successor `S`. |
| `q_hat` | `fuse_default` | `fuse_or_recompute` | Immediate successor `o`. |
| `k_hat` | `fuse_default` | `fuse_or_recompute` | Intra-equation two reads; not a split candidate. |
| `alpha` | `fuse_default` | `fuse_or_recompute` | Immediate successor `S`. |
| `beta` | `fuse_default` | `fuse_or_recompute` | Immediate successor `S`. |
| `S` | `forced_token_store` | `must_store` | Token-persistent F32 state; also high-fan-out. |
| `o` | `fuse_default` | `fuse_or_recompute` | Immediate successor `u_gdn`. |
| `u_gdn` | `fuse_default` | `fuse_or_recompute` | Immediate successor `mix_lin`. |
| `mix_lin` | `fuse_default` | `fuse_or_recompute` | Residual add lives inside `gated_delta_net`. |
| `C_state` | `forced_token_store` | `must_store` | Requires prior-token conv delay. |
| `g_mlp` | `fuse_default` | `fuse_or_recompute` | Immediate successor `swiglu`. |
| `up` | `fuse_default` | `fuse_or_recompute` | Immediate successor `swiglu`. |
| `swiglu` | `fuse_default` | `fuse_or_recompute` | `mlp` internal; `split_swiglu` HYPOTHESIS. |
| `mlp_out` | `fuse_default` | `fuse_or_recompute` | Residual add lives inside `mlp`. |
| `e_next` | `boundary_tradeoff` | `keep_live_or_fuse` | TASK-11 `embed_e_next`. |
| `e_next_n` | `fuse_default` | `fuse_or_recompute` | Immediate successor `mtp_cat`. |
| `h64_n` | `fuse_default` | `fuse_or_recompute` | Immediate successor `mtp_cat`. |
| `mtp_cat` | `fuse_default` | `fuse_or_recompute` | `mtp_mix` internal; `split_mtp_cat` HYPOTHESIS. |
| `mtp_u` | `boundary_tradeoff` | `keep_live_or_fuse` | TASK-11 `mtp_u_to_block` as residual `h`. |
| `h_mtp` | `boundary_tradeoff` | `keep_live_or_fuse` | TASK-11 `h_mtp_to_logits`. |
| `logits_1` | `forced_output` | `must_expose` | MTP output sink. |

JSON object `physical_need_by_id` keyed in `catalog_ids` order with the class strings above. JSON object `default_tactic_by_id` keyed in the same order with the tactic strings above.

### Forced stores and forbidden drops (lock)

JSON array `forbidden_drop_ids` in this exact order (9 ids). JSON `n_forbidden_drops` = 9.

`K_state`, `V_state`, `C_state`, `S`, `token_id`, `logits_0`, `logits_1`, `residual_add_mix`, `residual_add_mlp`

`residual_add_mix` and `residual_add_mlp` are **operations**, not catalog IDs. They appear in this list and in prose so a fuse-across-`h`/`h_mid` hypothesis cannot be read as deleting Eq. `(4)` / `(5)`. They are **not** catalog IDs and **not** in `catalog_ids`. JSON `forbidden_drop_ops` `["residual_add_mix","residual_add_mlp"]`. JSON `n_forbidden_drop_ops` = 2. JSON `forbidden_drop_catalog_ids` = the first 7 catalog IDs. JSON `n_forbidden_drop_catalog_ids` = 7.

Nonfusible TASK-11 sync edges (cannot be removed even as a hypothesis that “drops the write”):

JSON `nonfusible_sync_edge_ids` in this exact order (5 ids). JSON `n_nonfusible_sync_edges` = 5.

`state_kv`, `state_c`, `state_s`, `output_logits_0`, `output_logits_1`

Fusible-across data edges (HYPOTHESIS node-type merge only; does not delete liveness):

JSON `fusible_sync_edge_ids` in this exact order (7 ids). JSON `n_fusible_sync_edges` = 7.

`identity_e_h0`, `residual_h`, `residual_h_mid`, `fanout_h64`, `embed_e_next`, `mtp_u_to_block`, `h_mtp_to_logits`

Shared-weight edges (required payload sharing, not a node-type merge):

JSON `shared_weight_sync_edge_ids` in this exact order (2 ids). JSON `n_shared_weight_sync_edges` = 2.

`shared_E`, `shared_W_lm`

Checker: `n_nonfusible_sync_edges + n_fusible_sync_edges + n_shared_weight_sync_edges == 14`. JSON `n_sync_edges` = 14. JSON `sync_edge_ids` copied from TASK-11 in TASK-11 order.

Additional forbidden collapses (booleans, lock true):

- `rms_roles_not_collapsed` true (TASK-11; two residual-stream RMS roles plus GatedRMSNorm `(3)`).
- `residual_add_inside_mixer` true; `residual_add_inside_mlp` true; `residual_input_live_until_add` true.
- `gdn_primary_is_recurrent_eq_17` true; `chunkwise_not_zero_s_traffic` true.
- `fanout_h64_cannot_hide_from_one_consumer` true — fusing last `mlp` into `lm_head` still must expose `h_64` to `mtp_mix` unless MTP is omitted as an algebraic equivalent (not the primary map).
- `state_write_not_optional` true.

Prose required: fusing Mix and MLP across `residual_h_mid` keeps `h_mid` in a **local working set** until the MLP add; it does not authorize deleting the add. Splitting `gated_attn` at `g` **adds** a hypothesized sync edge; it does not make `g` token-persistent.

### Tradeoff tactics (lock; heading 6)

JSON array `tradeoff_tactic_ids` in this exact order (4 ids). JSON `n_tradeoff_tactics` = 4.

| id | Meaning | Availability (DERIVED) | Usefulness |
| --- | --- | --- | --- |
| `reuse` | One physical copy, several consumers | high-fan-out catalog IDs and shared \(E\)/`W_\text{lm}` | HYPOTHESIS |
| `recompute` | Drop and rebuild from current-token parents (parents may include live state) | every catalog ID except `K_state`,`V_state`,`C_state`,`S` | HYPOTHESIS |
| `local_working_set` | Keep the value only for the duration of the owning node; not a named inter-node buffer | all TASK-11 internals, including live-across `g`/`z` | HYPOTHESIS |
| `synchronize` | Promote a value to a named sync edge (existing 14, or a split candidate) | 14 TASK-11 edges; 10 split candidates | HYPOTHESIS |

JSON `n_recomputable_ids` = 48. JSON `recomputable_ids` = `catalog_ids` minus the four state IDs, catalog order. JSON `requires_prior_state_ids` `["K_state","V_state","C_state","S"]`.

JSON `reuse_opportunity_ids` in this exact order (11 ids). JSON `n_reuse_opportunity_ids` = 11.

`h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `k_rope`, `v_full`, `qkv`, `S`, `E`, `W_lm`

(`S` is forced store **and** a reuse opportunity of that stored object: it produces `o` and the next-token `S`. `E`/`W_lm` are shared weights, not catalog IDs.)

JSON `intra_equation_reuse_ids` `["k_hat"]`. Two reads inside Eq. `(17)` to produce one consumer `S`. Not a split candidate. Tactic: `local_working_set`. Usefulness of keeping `k_hat` vs recomputing the second read is HYPOTHESIS.

JSON `shared_weight_ids` `["E","W_lm"]`. `shared_weight_reuse` is required sharing of one payload (TASK-11), not a layout and not a selected fusion. A second physical read of `W_lm` for `logits_1` remains HYPOTHESIS traffic (citation of TASK-11), not DERIVED unique bytes.

Heading 6 must include a compact table of the 21 `important_ids` with columns: `ID` | `Physical need` | `Available tactics` | `Usefulness`. Available-tactics cells list a subset of the four tactic ids. Every usefulness cell is the word `HYPOTHESIS`. Forced rows list only the forced tactic (`must_store` / `must_expose` / `must_present` may appear in the physical-need column; tactics column for forced_token_store is `synchronize` only in the sense of the existing state edge — do **not** offer `recompute` for the four state IDs). For `K,V,C,S`, available tactics = `synchronize` (the existing state edge) plus `reuse` of the stored object; not `recompute` of the state itself.

### Fusion hypotheses (lock; heading 7; all HYPOTHESIS)

JSON array `fusion_kind_ids` in this exact order (3 ids). JSON `n_fusion_kinds` = 3.

`fuse_internals`, `split_at_internal`, `fuse_across_edge`

JSON array `fusion_hypothesis_ids` in this exact order (22 ids). JSON `n_fusion_hypotheses` = 22. Parallel `fusion_hypothesis_kinds`. JSON `n_fusion_hypotheses_selected` = 0. JSON object `fusion_selected` keyed in `fusion_hypothesis_ids` order with every value `false`. JSON `fusion_winner_selected` false.

**Intra-node `fuse_internals` (5).** One hypothesis per node type that owns internals. `embed` has no internals; do not invent `fuse_embed`. Extra sync edges 0. Working-set bytes 0 (no new named buffer; no dropped forced store). KV/\(C\)/\(S\) writes still occur inside the fused node.

| id | Kind | Node | Extra sync | Working-set bytes |
| --- | --- | --- | ---: | ---: |
| `fuse_gated_attn_internals` | `fuse_internals` | `gated_attn` | 0 | 0 |
| `fuse_gated_delta_net_internals` | `fuse_internals` | `gated_delta_net` | 0 | 0 |
| `fuse_mlp_internals` | `fuse_internals` | `mlp` | 0 | 0 |
| `fuse_lm_head_internals` | `fuse_internals` | `lm_head` | 0 | 0 |
| `fuse_mtp_mix_internals` | `fuse_internals` | `mtp_mix` | 0 | 0 |

**`split_at_internal` (10).** Exact TASK-11 `split_candidate_ids` order. Each split **adds** one hypothesized sync edge and names the working-set bytes of that internal if materialized at the new edge. Does not add a required node type.

| id | Kind | Catalog ID | Node | Extra sync | Working-set bytes |
| --- | --- | --- | --- | ---: | ---: |
| `split_g` | `split_at_internal` | `g` | `gated_attn` | +1 | 12288 |
| `split_z` | `split_at_internal` | `z` | `gated_delta_net` | +1 | 12288 |
| `split_h_tilde` | `split_at_internal` | `h_tilde` | mixer xor | +1 | 10240 |
| `split_k_rope` | `split_at_internal` | `k_rope` | `gated_attn` | +1 | 2048 |
| `split_v_full` | `split_at_internal` | `v_full` | `gated_attn` | +1 | 2048 |
| `split_qkv` | `split_at_internal` | `qkv` | `gated_delta_net` | +1 | 20480 |
| `split_h_post` | `split_at_internal` | `h_post` | `mlp` | +1 | 10240 |
| `split_swiglu` | `split_at_internal` | `swiglu` | `mlp` | +1 | 34816 |
| `split_h_final` | `split_at_internal` | `h_final` | `lm_head` | +1 | 10240 |
| `split_mtp_cat` | `split_at_internal` | `mtp_cat` | `mtp_mix` | +1 | 20480 |

`split_h_tilde` covers both mixer xor instances of the same catalog ID; do not emit two hypothesis ids.

**`fuse_across_edge` (7).** One hypothesis per fusible TASK-11 data edge. Each **removes** that edge as inter-node I/O (`extra_sync_edges` = −1). Residual-add / fan-out liveness still holds as a local working set. State and output edges have **no** hypothesis that drops them.

| id | Kind | Edge | Hidden I/O | Extra sync | Working-set bytes |
| --- | --- | --- | --- | ---: | ---: |
| `fuse_across_identity_e_h0` | `fuse_across_edge` | `identity_e_h0` | `e` | −1 | 10240 |
| `fuse_across_residual_h` | `fuse_across_edge` | `residual_h` | `h` | −1 | 10240 |
| `fuse_across_residual_h_mid` | `fuse_across_edge` | `residual_h_mid` | `h_mid` | −1 | 10240 |
| `fuse_across_fanout_h64` | `fuse_across_edge` | `fanout_h64` | `h_64` | −1 | 10240 |
| `fuse_across_embed_e_next` | `fuse_across_edge` | `embed_e_next` | `e_next` | −1 | 10240 |
| `fuse_across_mtp_u_to_block` | `fuse_across_edge` | `mtp_u_to_block` | `mtp_u` | −1 | 10240 |
| `fuse_across_h_mtp_to_logits` | `fuse_across_edge` | `h_mtp_to_logits` | `h_mtp` | −1 | 10240 |

JSON arrays parallel to `fusion_hypothesis_ids`: `fusion_hypothesis_kinds`, `fusion_extra_sync_edges` (JSON ints 0, 1, or −1), `fusion_working_set_bytes` (JSON ints). JSON `fusion_usefulness_label` exactly `HYPOTHESIS`.

Heading 7 table columns: `id` | `kind` | `target` | `extra_sync_edges` | `working_set_bytes` | `usefulness` | `selected`. Usefulness is always `HYPOTHESIS`. Selected is always `false`. Target is the node type, catalog ID, or sync-edge id as in the tables above.

JSON `split_candidate_ids` copied from TASK-11: `g`, `z`, `h_tilde`, `k_rope`, `v_full`, `qkv`, `h_post`, `swiglu`, `h_final`, `mtp_cat`. JSON `n_split_candidates` = 10. JSON `n_split_candidates_selected` = 0. JSON `split_selected` object, all false. Every `split_*` hypothesis id is `split_` + the candidate id.

Closing paragraph (required): this heading **closes** the ledger open question. The 22 proposals are the apparent fusions. Working-set bytes and extra-sync-edge counts are DERIVED identities. Which proposals improve total behavior after those costs remains a HYPOTHESIS. TASK-13/14 may consume this list when scheduling; TASK-17 may consume it when mapping. This document does not authorize a merge, a split, or a store winner.

### Working-set byte identities (lock; heading 8)

Conceptual element sizes from TASK-04: BF16 = 2 B (`bytes_bf16`), F32 = 4 B (`bytes_f32`). Not CUDA allocation dtypes. Checker recomputes from sitting `text_config`.

| JSON key | Formula | Value |
| --- | --- | ---: |
| `residual_elems` | `hidden_size` | 5120 |
| `residual_bytes` | \(5120\times 2\) | 10240 |
| `g_elems` | `num_attention_heads * head_dim` | 6144 |
| `g_bytes` | \(6144\times 2\) | 12288 |
| `z_elems` | `linear_num_value_heads * linear_value_head_dim` | 6144 |
| `z_bytes` | \(6144\times 2\) | 12288 |
| `k_rope_elems` | `num_key_value_heads * head_dim` | 1024 |
| `k_rope_bytes` | \(1024\times 2\) | 2048 |
| `v_full_elems` | same as `k_rope_elems` | 1024 |
| `v_full_bytes` | 2048 | 2048 |
| `qkv_elems` | \((2\cdot\texttt{linear_num_key_heads}+\texttt{linear_num_value_heads})\cdot\texttt{linear_key_head_dim}\) | 10240 |
| `qkv_bytes` | \(10240\times 2\) | 20480 |
| `swiglu_elems` | `intermediate_size` | 17408 |
| `swiglu_bytes` | \(17408\times 2\) | 34816 |
| `mtp_cat_elems` | \(2\cdot\texttt{hidden_size}\) | 10240 |
| `mtp_cat_bytes` | \(10240\times 2\) | 20480 |
| `logits_elems` | `vocab_size` | 248320 |
| `logits_bytes` | \(248320\times 2\) | 496640 |
| `kv_bytes_per_full_layer_per_token` | TASK-04 citation | 4096 |
| `kv_bytes_all_per_token` | TASK-04 citation | 69632 |
| `c_bytes_per_layer` | TASK-04 citation | 61440 |
| `s_bytes_per_layer` | TASK-04 citation | 3145728 |
| `s_bytes_all` | TASK-04 citation | 150994944 |
| `storage_fixed_bytes` | TASK-04 \(B_\text{store}\) fixed part | 153944064 |
| `storage_kv_bytes_coeff_T` | TASK-04 | 69632 |

Cite TASK-04 \(B_\text{store}(T)=69632T+153944064\). Do not re-derive the persistent-state table as a new study. Do not recopy TASK-06 GEMM-IO or region-cut activation totals; those remain TASK-06. Do not fill a SKU ridge.

Synchronization-cost identity (DERIVED, not MEASURED): `net_sync_edges_under_hypothesis = 14 + extra_sync_edges` for a single hypothesis applied in isolation. JSON `n_baseline_sync_edges` = 14. Applying two hypotheses together is **out of scope** (no pairwise table). Usefulness of any net-sync change is HYPOTHESIS.

JSON `hardware_independent` true; `cuda_mapping_deferred` true; `schedule_selected` false; `layout_selected` false; `activation_dtype_decided` false.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 7 (Fusion hypotheses). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers or 22 subgraphs of instances.

Required IDs **inside that fence**: `token_id`, `h`, `h_mid`, `h_64`, `h_tilde`, `g`, `z`, `k_rope`, `v_full`, `qkv`, `K_state`, `V_state`, `C_state`, `S`, `logits_0`, `logits_1`.

Use subgraphs for physical-need classes of those IDs (forced store / boundary / live-across / reuse / I/O). Dashed edges may annotate hypothesized splits of `g`/`z`; they must not be readable as selected winners. Caption must contain the word `HYPOTHESIS`.

JSON `n_diagrams` is 1. `diagram_ids` is `["token_id","h","h_mid","h_64","h_tilde","g","z","k_rope","v_full","qkv","K_state","V_state","C_state","S","logits_0","logits_1"]`.

### Deferred vision

Visual tokens may replace placeholders on the `identity_e_h0` edge (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger materialization is **UNKNOWN**. Do not classify vision-encoder activations. Do not add a vision fusion hypothesis. `vision_interface_is_not_a_node` true. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_materialization_and_fusion.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, QKV width, residual/`g`/`z`/`k_rope`/`qkv`/`swiglu`/`mtp_cat`/`logits` element counts, and TASK-04 KV/C/S byte citations. Duplicate TASK-03 catalog id order and TASK-11 node-type / sync-edge / split-candidate lists as constants; do not import them.

CLI (cwd = repository root):

```text
python3 scripts/check_materialization_and_fusion.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_materialization_and_fusion.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --materialization docs/architecture/materialization-and-fusion.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: residual/`g`/`z`/`k_rope`/`qkv`/`swiglu`/`mtp_cat`/`logits` elems and bytes; cited KV/C/S bytes. Constant fields: canonical sentences, class/hypothesis/tactic lists, catalog partition, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--materialization PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all four canonical sentences verbatim, (5) every `physical_need_class_ids`, `fusion_hypothesis_ids`, `fusion_kind_ids`, `tradeoff_tactic_ids`, `catalog_ids`, `important_ids`, `split_candidate_ids`, `sync_edge_ids`, `fusible_sync_edge_ids`, `nonfusible_sync_edge_ids` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer below present as a decimal or integer substring, (10) the words `HYPOTHESIS` and `hardware-independent` present, (11) none of the forbidden winner phrases: `selected fusion`, `selected winner`, `winning fusion`, `should fuse`, `recommend fusion`, `must fuse`, `chosen fusion`, `CUDA kernel fusion is required`, `framework module is the node`, `GGUF is the graph`, `Quartz graph`, `llama.cpp graph` (allow the substring only inside `not a selected fusion` / `not a CUDA allocation, cache layout, or selected fusion` / `none is a selected winner`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks cited TASK-04 bytes against `lifetime-and-state.md` and split/sync lists against `semantic-graph.md`).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `n_catalog_nodes==52`, `len(catalog_ids)==52`
- `n_physical_need_classes==7`, `n_physical_need_rules==7`, `n_default_tactics==7`
- `n_forced_token_store==4`, `n_forced_output==2`, `n_forced_input==1`, `n_boundary_tradeoff==7`, `n_live_across_tradeoff==2`, `n_reuse_tradeoff==5`, `n_fuse_default==31`
- `n_forced_token_store+n_forced_output+n_forced_input+n_boundary_tradeoff+n_live_across_tradeoff+n_reuse_tradeoff+n_fuse_default==52`
- the seven class arrays are pairwise disjoint and their union equals `catalog_ids`
- `forced_token_store_ids == ["K_state","V_state","C_state","S"]`
- `live_across_tradeoff_ids == ["g","z"]`
- `n_important_ids==21`, `n_tradeoff_tactics==4`, `n_recomputable_ids==48`
- `n_fusion_kinds==3`, `n_fusion_hypotheses==22`, `n_fusion_hypotheses_selected==0`
- `n_split_candidates==10`, `n_split_candidates_selected==0`
- `n_sync_edges==14`, `n_fusible_sync_edges==7`, `n_nonfusible_sync_edges==5`, `n_shared_weight_sync_edges==2`
- `n_fusible_sync_edges+n_nonfusible_sync_edges+n_shared_weight_sync_edges==14`
- `n_diagrams==1`, `n_forbidden_drop_catalog_ids==7`
- `residual_elems==hidden_size==5120`, `residual_bytes==10240`
- `g_elems==24*256==6144`, `g_bytes==12288`
- `z_elems==48*128==6144`, `z_bytes==12288`
- `k_rope_elems==4*256==1024`, `k_rope_bytes==2048`
- `qkv_elems==10240`, `qkv_bytes==20480`
- `swiglu_elems==intermediate_size==17408`, `swiglu_bytes==34816`
- `mtp_cat_elems==2*hidden_size==10240`, `mtp_cat_bytes==20480`
- `logits_elems==vocab_size==248320`, `logits_bytes==496640`
- `kv_bytes_all_per_token==69632`, `s_bytes_all==150994944`, `storage_fixed_bytes==153944064`
- `fusion_working_set_bytes` equals the locked 22-int list; `fusion_extra_sync_edges` equals the locked 22-int list
- every `fusion_selected` value is False; every `split_selected` value is False
- `fusion_winner_selected is False`, `ledger_open_question_fusions_remain_hypothesis is True`
- `decode_prefill_share_taxonomy is True`, `hardware_independent is True`
- `live_across_are_internal is True`, `fanout_h64_cannot_hide_from_one_consumer is True`
- `state_write_not_optional is True`, `chunkwise_not_zero_s_traffic is True`
- `vision_interface_is_not_a_node is True`, `schedule_selected is False`, `layout_selected is False`
- `physical_need_class_ids` equals the locked 7-id list; `fusion_hypothesis_ids` equals the locked 22-id list
- `"g" in live_across_tradeoff_ids` and `"S" in forced_token_store_ids`
- `"h_mid" in boundary_tradeoff_ids` and `"h_tilde" in reuse_tradeoff_ids`

Locked `fusion_extra_sync_edges` (22 ints, this order): `0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, -1,-1,-1,-1,-1,-1,-1`

Locked `fusion_working_set_bytes` (22 ints, this order): `0,0,0,0,0, 12288,12288,10240,2048,2048,20480,10240,34816,10240,20480, 10240,10240,10240,10240,10240,10240,10240`

Locked document integers/decimals the `--materialization` check must find:

`5120`, `17408`, `248320`, `10240`, `12288`, `2048`, `20480`, `34816`, `496640`, `6144`, `69632`, `150994944`, `153944064`, `3145728`, `61440`, `4096`, `22`, `21`, `31`

(The integer `48` / `16` / `64` / `7` / `52` / `14` / `10` / `4` / `5` will appear from counts and types; requiring the byte/MAC-scale set plus `22` / `21` / `31` is the hard check. Do not require `1e-06`.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `n_attn_heads`, `n_kv_heads`, `head_dim`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `bytes_bf16`, `bytes_f32`,

`n_catalog_nodes`, `catalog_ids`,

`physical_need_class_ids`, `n_physical_need_classes`, `physical_need_rule_ids`, `n_physical_need_rules`, `default_tactic_ids`, `n_default_tactics`,

`forced_token_store_ids`, `forced_output_ids`, `forced_input_ids`, `boundary_tradeoff_ids`, `live_across_tradeoff_ids`, `reuse_tradeoff_ids`, `fuse_default_ids`, `n_forced_token_store`, `n_forced_output`, `n_forced_input`, `n_boundary_tradeoff`, `n_live_across_tradeoff`, `n_reuse_tradeoff`, `n_fuse_default`, `physical_need_partition_complete`, `physical_need_by_id`, `default_tactic_by_id`,

`important_ids`, `n_important_ids`, `high_fanout_boundary_ids`, `n_high_fanout_boundary_ids`,

`tradeoff_tactic_ids`, `n_tradeoff_tactics`, `recomputable_ids`, `n_recomputable_ids`, `requires_prior_state_ids`, `reuse_opportunity_ids`, `n_reuse_opportunity_ids`, `intra_equation_reuse_ids`, `shared_weight_ids`,

`forbidden_drop_catalog_ids`, `n_forbidden_drop_catalog_ids`, `forbidden_drop_ops`, `n_forbidden_drop_ops`, `forbidden_drop_ids`, `n_forbidden_drops`,

`sync_edge_ids`, `n_sync_edges`, `fusible_sync_edge_ids`, `n_fusible_sync_edges`, `nonfusible_sync_edge_ids`, `n_nonfusible_sync_edges`, `shared_weight_sync_edge_ids`, `n_shared_weight_sync_edges`, `n_baseline_sync_edges`,

`fusion_kind_ids`, `n_fusion_kinds`, `fusion_hypothesis_ids`, `fusion_hypothesis_kinds`, `fusion_extra_sync_edges`, `fusion_working_set_bytes`, `n_fusion_hypotheses`, `fusion_selected`, `n_fusion_hypotheses_selected`, `fusion_usefulness_label`, `fusion_winner_selected`,

`split_candidate_ids`, `n_split_candidates`, `split_selected`, `n_split_candidates_selected`,

`residual_elems`, `residual_bytes`, `g_elems`, `g_bytes`, `z_elems`, `z_bytes`, `k_rope_elems`, `k_rope_bytes`, `v_full_elems`, `v_full_bytes`, `qkv_elems`, `qkv_bytes`, `swiglu_elems`, `swiglu_bytes`, `mtp_cat_elems`, `mtp_cat_bytes`, `logits_elems`, `logits_bytes`, `kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`, `c_bytes_per_layer`, `s_bytes_per_layer`, `s_bytes_all`, `storage_fixed_bytes`, `storage_kv_bytes_coeff_T`,

`decode_prefill_share_taxonomy`, `fanout_neq_must_store`, `node_io_neq_must_store`, `must_survive_neq_cuda_malloc`, `live_across_are_internal`, `residual_add_inside_mixer`, `residual_add_inside_mlp`, `residual_input_live_until_add`, `rms_roles_not_collapsed`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_not_zero_s_traffic`, `fanout_h64_cannot_hide_from_one_consumer`, `state_write_not_optional`, `hardware_independent`, `cuda_mapping_deferred`, `schedule_selected`, `layout_selected`, `activation_dtype_decided`, `primary_includes_mtp`, `vision_interface_is_not_a_node`, `ledger_open_question_fusions_remain_hypothesis`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_physical`, `canonical_sentence_hypothesis`, `canonical_sentence_open_question`.

Integer JSON fields that are counts/widths/bytes are JSON ints. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. `physical_need_by_id` and `default_tactic_by_id` are JSON objects keyed in `catalog_ids` order. `fusion_selected` and `split_selected` are JSON objects with JSON `false` values. `fusion_extra_sync_edges` and `fusion_working_set_bytes` are JSON arrays of ints parallel to `fusion_hypothesis_ids`. `fusion_usefulness_label` is the JSON string `HYPOTHESIS`.

`node_type_ids` are **not** a required top-level JSON key (avoid recopying the TASK-11 contract object). Node type names appear in fusion-hypothesis prose and as substrings in the markdown.

### Stage split

- **Implementation** writes `scripts/check_materialization_and_fusion.py` **and** `docs/architecture/materialization-and-fusion.md` (seven physical-need classes over 52 IDs, four tradeoff tactics, 22 unselected fusion hypotheses that close the ledger open question as remaining-HYPOTHESIS, JSON fence). Runs `--json` and `--materialization` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-07 / `numerical-sensitivity.md` / `semantic-graph.md` style), Authority table links to this dossier / dataflow / lifetime / semantic-graph / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, class ids, hypothesis ids, or Mermaid node IDs. Does not edit TASK-03/04/11 artifacts.
- **Verification** independently re-runs focused commands, recomputes residual/`g`/`z`/`qkv`/`swiglu` bytes and instance-related config fields from sitting `text_config` (not from JSON echo), spot-checks cited TASK-04 bytes against `docs/architecture/lifetime-and-state.md` and split/sync lists against `docs/architecture/semantic-graph.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-graph, no winner phrases, no `plan.md` or ledger edit, no payload I/O, that every fusion remains unselected, that contracts are hardware-independent (no CUDA kernel/thread/SKU decisions), that live-across `g`/`z` are not forced inter-node stores, that state/output edges are nonfusible, and that the open question is closed by remaining-HYPOTHESIS enumeration. Schedules, layouts, and CUDA mapping remain open.

- Invariants:
  - Ten level-2 headings in the locked order; four canonical sentences verbatim; 7 physical-need classes; 52 catalog IDs partitioned 4/2/1/7/2/5/31; 21 important IDs; 4 tradeoff tactics; 22 fusion hypotheses all unselected; 14 sync edges partitioned 7/5/2; one Mermaid flowchart with required IDs.
  - Prefill/decode share one taxonomy; primary includes MTP; `g`/`z` internal unless split; residual add inside mixer and `mlp`; state writes not optional; `h_64` cannot be hidden from one of its two primary consumers.
  - GDN numerical definition remains recurrent `(17)`–`(18)`; chunkwise is not zero \(S\) traffic.
  - Logical values do not imply allocation; physical-need is not CUDA malloc; every fusion proposal is HYPOTHESIS.
  - Vision encoder remains unexpanded and is not a fusion target.
- Rejected alternatives:
  - Selecting any fusion, split, or store winner because it “looks faster”: rejected; ledger and plan.md require hypotheses to remain hypotheses; no MEASURED evidence exists in Phase 1.
  - Treating TASK-04 must-survive as CUDA malloc: rejected; TASK-04 already forbade that; this task classifies need vs tradeoff.
  - Promoting `g`/`z` to required inter-node buffers: rejected; TASK-11 keeps them internal; `split_g`/`split_z` are unselected hypotheses.
  - One physical buffer per catalog ID: rejected; fan-out ≠ must-store; most internals are `fuse_default`.
  - Dropping KV/\(C\)/\(S\) writes by “fusing them away”: rejected; that changes the map.
  - Fusing Mix+MLP as a **required** node: rejected; TASK-11 `cut_mixer_mlp` and TASK-04 `h_mid`; `fuse_across_residual_h_mid` is only a HYPOTHESIS that still keeps the add.
  - Pairwise or all-subsets fusion search: rejected; 22 isolated hypotheses are the locked list; combinations are TASK-13/14/17 experiment design.
  - Importing TASK-16 occupancy / shared-memory fusion algebra: rejected; not a dependency; this document is hardware-independent.
  - Recopying TASK-06 GEMM-IO / region-cut tables as the materialization floor: rejected; TASK-06 is not a dependency; activation views there are not CUDA live-sets; cite TASK-03 ranks × TASK-04 dtypes instead.
  - Separate decode and prefill taxonomies: rejected; same map; TASK-13/14 schedule \(T\) and incoming state.
  - Inventing catalog IDs or extra required node types (proj/core/out, conv/recurrence/out): rejected; those remain split hypotheses.
  - Using GGUF Q4 or CUDA dtypes as working-set dtypes: rejected; conceptual BF16/F32 from TASK-04.
  - Filling TASK-16 SKU peaks or measuring tok/s: forbidden; not this task.
  - Inspecting Quartz or llama.cpp for “real” fusions: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–11.
  - Editing frozen TASK-03/04/11 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py`.
  - Leaving the ledger open question unclassified: rejected; the 22 hypotheses plus sentence 4 are the completion criterion. Closing it by picking a fusion: also rejected.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/materialization-and-fusion.md` exists and follows the heading list above.
  - Every catalog ID has a physical-need class with a TASK-03/04/11 justification; seven classes partition the 52 IDs; 21 important IDs are tabulated with tactics.
  - Reuse, recompute, local-working-set, and synchronization tactics are listed; usefulness of choosing a tactic is HYPOTHESIS.
  - Every fusion proposal is labelled `HYPOTHESIS`; `n_fusion_hypotheses` = 22; `n_fusion_hypotheses_selected` = 0; `fusion_winner_selected` false.
  - Ledger open question is closed by `canonical_sentence_open_question` and the 22-id enumeration with DERIVED extra-sync and working-set identities, not by a selected winner; `ledger_open_question_fusions_remain_hypothesis` true.
  - State and output edges are nonfusible; `g`/`z` are not forced inter-node stores; `fanout_h64_cannot_hide_from_one_consumer` true.
  - Four canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - Contracts are hardware-independent (no CUDA kernel/thread/SKU decisions; `hardware_independent` true; `cuda_mapping_deferred` true).
  - No kernel/layout/fusion/allocation **winners**; no Quartz/llama.cpp; GGUF is not the graph; no payload re-stream; no `plan.md` or ledger edit.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_materialization_and_fusion.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_materialization_and_fusion.py
python3 scripts/check_materialization_and_fusion.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_materialization_and_fusion.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --materialization docs/architecture/materialization-and-fusion.md
```

- Candidate quality: not required — no model execution or NLL; this increment is materialization/fusion documentation. Fusion **usefulness** is HYPOTHESIS prose, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/materialization-and-fusion.md
python3 -m py_compile scripts/check_materialization_and_fusion.py
python3 scripts/check_materialization_and_fusion.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --materialization docs/architecture/materialization-and-fusion.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run other `scripts/check_*.py` as a requirement of this task (verifier may spot-check TASK-04/11 markdown integers independently).

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/materialization-and-fusion.md` (create)
  - `scripts/check_materialization_and_fusion.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-03/04/11 deliverables unchanged)
- Definition of done: materialization-and-fusion document published with locked seven physical-need classes over 52 IDs, four tradeoff tactics, 22 unselected fusion hypotheses, and open-question closure that every apparent fusion remains HYPOTHESIS after working-set and sync-cost identities; contracts hardware-independent; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-12 completion checkboxes can be marked at delivery; no fusion winner, schedule, layout, or CUDA mapping is selected.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T14:45:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-12.md`. Coupled IDs `none`. Document structure (10 headings), four canonical sentences, seven physical-need classes (52-ID partition 4/2/1/7/2/5/31), 21 important intermediates, four tradeoff tactics, 22 unselected fusion hypotheses (5 fuse-internals + 10 split + 7 fuse-across-edge), sync-edge partition 7/5/2, stdlib checker `scripts/check_materialization_and_fusion.py`, JSON schema, and acceptance commands are closed. Ledger open question closed **by enumerating HYPOTHESIS proposals with DERIVED working-set and extra-sync identities**, not by selecting a fusion. `docs/architecture/materialization-and-fusion.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit.
- Performance evidence applied: N/A — classification/hypothesis documentation; fusion usefulness is hypothesis, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_materialization_and_fusion.py` (stdlib checker: `argparse`, `json`, `re`, `sys`, `pathlib`; no imports of other `scripts/check_*.py` or payload streamers). Instantiates the locked summary from sitting `text_config` plus TASK-03/04/11 constants; `--json` asserts seven physical-need classes partitioning 52 IDs 4/2/1/7/2/5/31, 21 important IDs, four tradeoff tactics, 22 unselected fusion hypotheses, 14 sync edges partitioned 7/5/2, DERIVED residual/`g`/`z`/`qkv`/`swiglu`/`mtp_cat`/`logits` bytes, cited TASK-04 KV/C/S bytes, and hardware-independent booleans; `--materialization` checks headings, JSON fence equality, one Mermaid flowchart, four canonical sentences, locked integers, and forbidden winner phrases.
  - Created `docs/architecture/materialization-and-fusion.md` with the ten locked level-2 headings, four canonical sentences, seven physical-need classes over 52 IDs, 21 important intermediates with tactics, four tradeoff tactics, 22 unselected fusion hypotheses that close the ledger open question as remaining-HYPOTHESIS, one Mermaid flowchart, and a JSON fence copied from a live `--json` run. No fusion, schedule, layout, or CUDA mapping selected. Did not edit `plan.md`, `task_ledger.md`, TASK-03/04/11 deliverables, or stream safetensor payloads. No commit.
- Commands:
  - `python3 -m py_compile scripts/check_materialization_and_fusion.py` — exit 0
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; live object matches locked identities (`n_catalog_nodes=52`, partition `4/2/1/7/2/5/31`, `n_important_ids=21`, `n_fusion_hypotheses=22`, `n_fusion_hypotheses_selected=0`, `fusion_winner_selected=false`, `n_sync_edges=14` fusible/nonfusible/shared `7/5/2`, `residual_bytes=10240`, `g_bytes=12288`, `qkv_bytes=20480`, `swiglu_bytes=34816`, `mtp_cat_bytes=20480`, `logits_bytes=496640`, `kv_bytes_all_per_token=69632`, `storage_fixed_bytes=153944064`, `ledger_open_question_fusions_remain_hypothesis=true`, `hardware_independent=true`, `live_across_are_internal=true`)
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --materialization docs/architecture/materialization-and-fusion.md` — exit 0
- UTC/time/tokens/cost: `2026-09-20T14:48:14Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/materialization-and-fusion.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-07 / [`numerical-sensitivity.md`](../numerical-sensitivity.md) / [`semantic-graph.md`](../semantic-graph.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`dataflow.md`](../dataflow.md) (TASK-03), [`lifetime-and-state.md`](../lifetime-and-state.md) (TASK-04), [`semantic-graph.md`](../semantic-graph.md) (TASK-11), [`model-inventory.md`](../model-inventory.md) (TASK-01) via those docs, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_materialization_and_fusion.py`](../../../scripts/check_materialization_and_fusion.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the JSON fence left unchanged. No locked integers, canonical sentences, physical-need class ids, fusion-hypothesis ids, or Mermaid node IDs edited. Frozen TASK-03/04/11 artifacts and `plan.md` not edited.
- Commands:
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --materialization docs/architecture/materialization-and-fusion.md` — pass (exit 0; headings, JSON fence, one flowchart, four canonical sentences, locked ids/integers, `HYPOTHESIS` / `hardware_independent`; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T14:49:12Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `composer-2.5` (verifier subagent)
- Diff review:
  - New: `docs/architecture/materialization-and-fusion.md` (10 required `##` headings, four canonical sentences verbatim, 52-ID physical-need table, 21 important intermediates, four tradeoff tactics, 22 unselected fusion hypotheses, one Mermaid flowchart, JSON fence).
  - New: `scripts/check_materialization_and_fusion.py` (stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`; no imports of other `scripts/check_*.py`).
  - New: `docs/architecture/tasks/TASK-12.md` (dossier).
  - Modified: `docs/architecture/task_ledger.md` (`TODO` → `IN PROGRESS` only; expected at admission; completion criteria still unchecked — delivery stage).
  - Unchanged: `plan.md`, `dataflow.md`, `lifetime-and-state.md`, `semantic-graph.md`, and all other upstream TASK deliverables.
- Independent raw-record checks:
  - 52 catalog IDs partition 4/2/1/7/2/5/31 (sum 52); 21 important IDs; 4 tradeoff tactics; 22 fusion hypotheses (5 fuse-internals + 10 split + 7 fuse-across-edge), all `selected: false`, `fusion_winner_selected: false`.
  - 14 sync edges partitioned 7 fusible / 5 nonfusible / 2 shared-weight.
  - Byte identities independently verified: `residual_bytes=10240`, `g_bytes=12288`, `qkv_bytes=20480`, `swiglu_bytes=34816`, `mtp_cat_bytes=20480`, `logits_bytes=496640`, `kv_bytes_all_per_token=69632`, `storage_fixed_bytes=153944064` — match sitting `text_config` ranks × TASK-04 constants and `lifetime-and-state.md` JSON fence.
  - JSON fence in document equals live `--json` output (full object equality).
  - All 16 `diagram_ids` present in the single Mermaid flowchart; four canonical sentences present verbatim.
  - `hardware_independent=true`, `cuda_mapping_deferred=true`, `ledger_open_question_fusions_remain_hypothesis=true`, `live_across_are_internal=true`, `fanout_h64_cannot_hide_from_one_consumer=true`.
  - No fusion/schedule/layout/CUDA winner selected; forbidden-winner phrases not triggered by checker.
- Commands:
  - `test -f docs/architecture/materialization-and-fusion.md` — exit 0
  - `python3 -m py_compile scripts/check_materialization_and_fusion.py` — exit 0
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --materialization docs/architecture/materialization-and-fusion.md` — exit 0
- Formatting changed files: none (Ruff not run per dossier scope)
- Verdict: **PASS** (first attempt)
- UTC/time/tokens/cost: `2026-09-20T14:52:00Z`; `telemetry_unavailable`

### Retries and escalation

none.

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-12 only; coupled IDs `none`
- Outcome: TASK-12 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T14:50:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/materialization-and-fusion.md` (10 locked headings, four canonical sentences, seven physical-need classes over 52 IDs, 21 important intermediates, four tradeoff tactics, 22 unselected fusion hypotheses, one Mermaid flowchart, JSON fence); `scripts/check_materialization_and_fusion.py` stdlib checker; independent `text_config` recomputation and TASK-03/04/11 citation spot-checks pass; frozen upstream docs unchanged; fusion, schedule, layout, and CUDA mapping remain open; ledger open question closed by remaining-HYPOTHESIS enumeration
- Candidate measured delta: N/A — materialization/fusion documentation
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (materialization/fusion locks; no tok/s evidence)
- Throughput delta: N/A — TASK-12 does not execute or time the model
- Commit: Publish Qwen3.8 materialization fusion
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification PASS; ledger update at delivery)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/authorities/qwen3.8-27b-transformers/config.json` must remain present for focused commands
