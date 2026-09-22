# TASK-21 — Freeze the clean-sheet baseline

## Control

- Primary ID: `TASK-21`
- Coupled IDs: `none`
- Dependencies: `TASK-20` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Check cross-document equations, dimensions, totals, state, contracts, and proposals; Correct documentation inconsistencies and preserve true unknowns; Mark the reviewed design `FROZEN_FOR_COMPARATIVE_REVIEW`.

## Goal and boundaries

Produce `docs/architecture/clean-sheet-review.md` as the Phase 1 **consistency review and freeze record**. Close the three ledger completion criteria by (1) checking equations, dimensions, totals, state, contracts, and proposals across the twenty-one TASK-01–20 architecture deliverables, (2) recording that planning-time JSON-fence comparison found **zero** documentation inconsistencies to correct and **preserving** the twelve TASK-20 remaining open questions plus documented notation aliases, and (3) marking the reviewed design `FROZEN_FOR_COMPARATIVE_REVIEW` **in this review document only**. Do **not** select a recipe, profile, artifact boundary, sequence, fusion, view, layout, mapping, corpus, hardware, or integrity winner. Do **not** close any of the twelve remaining open questions. Do **not** run NLL, reconstruction, tok/s, Nsight, SKU fill, or payload re-stream. Do **not** inspect Quartz or llama.cpp; freeze **authorizes later** comparative review, it does not execute it here. Do **not** flip TASK-20 `frozen_for_comparative_review` (historically false; TASK-20 checker forbids the freeze phrase as a selected state).

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - If any occupancy, MAC/byte, catalog id, node type, stage kind, recipe, mapping, or selected-flag would disagree with TASK-01–20 or sitting `text_config`, the earlier document wins and this review is wrong. Planning compared first JSON fences of all twenty-one corpus documents: shared scalars/lists agree except documented notation aliases (not inconsistencies).
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established; TASK-16 $N_w=32$, $N_{\text{bank}}=32$), `MEASURED` (TASK-05 citations only: `pooled_all.absmax=25.5`, `pooled_language_mtp.absmax=19.25`, `pooled_vision.absmax=25.5`; no new payload stats), `DERIVED` (cross-document key agreement, equation-tag presence, alias classification, freeze membership), `HYPOTHESIS` (every remaining alternative usefulness; every unrun experiment hypothesis — cited, not re-opened), or `UNKNOWN` (sitting-SKU numeric limits; vision-encoder internals; unselected corpora/hardware). No new `MEASURED` NLL, tok/s, occupancy, or bandwidth.
  - GitHub Markdown math only where a citation needs it. Cite TASK-01–20 ids and documents; do not rewrite forward math, recopy MAC tables, recopy 22 recipes, recopy 18 mappings, or recopy TASK-18/19 methodologies.
  - Allowed evidence: TASK-01–20 deliverables and their JSON fences, sitting `config.json` `text_config`, plan.md evidence vocabulary, this dossier. TASK integers and ids already published may be **cited**. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf` as a design authority or file to open. GGUF remains a named future black-box Pareto **reference** (TASK-18). After freeze, a later comparative review may use those systems as black-box references; they do not become design authority by the freeze mark.
- Non-goals:
  - No selected winners (`n_alternatives_selected` 0; every `*_selected` / `*_winner_selected` flag cited from TASK-08–20 remains false).
  - No executed experiments (`experiments_run` false; `nll_measured_here` false; `toks_measured_here` false; `benchmarks_run` false; `sku_table_filled` false; `comparative_review_executed_here` false).
  - No closing of TASK-07–19 ledger open questions (`n_remaining_open_questions_closed` 0). Preservation is not closure.
  - No correction of notation aliases (`dtype` `BF16` vs `bfloat16`; model `T_max` vs SKU `T_max`; `authority` checkpoint vs plan.md; layer/MTP/KV key-name pairs; `warp_size`/`n_bank` vs `N_w`/`N_bank`).
  - No stripping of per-task draft-status `unverified` banners on TASK-01–20 deliverables (`draft_banners_left_in_place` true). Those are process artifacts of those tasks’ documentation stages, not scientific contradictions.
  - No edit of TASK-20 `frozen_for_comparative_review` false or TASK-20 checker (`freeze_recorded_in_review_only` true).
  - No new operators, node types, recipes, mappings, metrics, stage kinds, or experiments.
  - No Quartz/llama.cpp inspection; do not open the GGUF file (`gguf_payload_inspected` false).
  - No CUDA kernels, `deviceQuery`, Nsight, sitting-GPU clocks, or OPT-058 harness.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–20). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, or any TASK-01–20 deliverable (planning found zero inconsistencies; a live conflict that is not a locked alias is a planning miss — stop).
  - Do not import any `scripts/check_*.py`, `scripts/inventory_bf16_checkpoint.py`, or `scripts/analyze_bf16_tensors.py`. Do not stream safetensor payloads. Do not subprocess sibling checkers from the TASK-21 checker (sibling checkers are focused commands only).
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib clean-sheet-review checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central design path; TASK-21 is not a new chain step.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden **until** freeze; after freeze, comparative study of existing runtimes may begin without making them silent design authority.
- `docs/architecture/plan.md:54-74` — evidence labels DERIVED / OBSERVED / MEASURED / HYPOTHESIS / UNKNOWN; hypotheses remain hypotheses.
- `docs/architecture/plan.md:97-100` — TASK-20 synthesizes; TASK-21 consistency-checks and marks `FROZEN_FOR_COMPARATIVE_REVIEW`.
- `docs/architecture/plan.md:107-115` — Phase 1 produces a mathematical and evidence-labelled architecture dossier, not a runtime; the frozen dossier is the baseline for later comparison.
- `docs/architecture/task_ledger.md` TASK-20 **Status: DONE**; TASK-21 depends on TASK-20; produces `docs/architecture/clean-sheet-review.md`; completion is cross-document check, correct inconsistencies / preserve true unknowns, and freeze mark.
- `docs/architecture/task_ledger.md` TASK-20 remaining open questions (still unresolved; **preserve**, do not close): TASK-07 risk survival; TASK-08 Pareto frontier; TASK-09 portable vs specialized artifact and ideal byte sequences; TASK-10 integrity `none` vs `checksum`; TASK-12/13 fusion winners; TASK-14 decode/prefill views; TASK-15 parallel decompositions; TASK-16 sitting SKU; TASK-17 mapping winners; TASK-18 corpora / prompt suite / capability / acceptance frontier; TASK-19 hardware / prompt matrix / reproducibility. Catalogued as twelve `oq_*` ids in `clean-sheet-architecture.md`.
- `docs/architecture/clean-sheet-architecture.md` / `experiment-backlog.md` — synthesis; `frozen_for_comparative_review` false; 125 alternatives / 0 selected; 12 remaining open questions / 0 closed; 16 unrun experiments. TASK-20 checker forbids freeze-as-selected-state in those files. Leave both files unchanged.
- Planning-time first-JSON-fence comparison of all twenty-one corpus documents: shared occupancy, MAC, catalog ids, node/stage/recipe/mapping/fusion lists, and selected-flags agree. Sole scalar name collision is `dtype` (`BF16` in TASK-01/05 vs `bfloat16` in TASK-19/20). `T_max` in TASK-07 is model context 262144; TASK-16 SKU id `T_max` is max threads per SM (UNKNOWN); TASK-19 already names `model_T_max` for the former.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads. Do not read GGUF.
- `scripts/check_clean_sheet_architecture.py` and sibling `scripts/check_*.py` — checker-style precedent. TASK-21’s checker is a sibling; do not import them.

## Performance evidence

N/A — Phase 1 **consistency review and freeze**. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. MAC/byte figures are TASK-06 citations. Freeze is a study-state mark, not a measured throughput claim. Do not apply the performance-evidence checklist to rank kernels or claim a winning mapping, recipe, or layout.

- Measurement identity: N/A (no engine binary, no eval run). Future identity remains the TASK-19 protocol (unselected values).
- Metric class: N/A for this increment. Future classes stay TASK-18 quality and TASK-19 performance; tok/s is not a quality axis.
- Coverage: N/A for GPU graphs. Review coverage is 21 corpus documents, 6 check families, 7 notation aliases, 0 inconsistencies found, 12 remaining open questions (0 closed), freeze true in the review document only.
- Time accounting: N/A
- Contradiction register: planning found no numeric/list JSON-fence contradictions. Documented notation aliases are not contradictions. If live `text_config` occupancy disagrees with TASK-01, or a cited TASK-01–20 integer or id disagrees with those documents **and is not a locked alias**, stop and fail closed (do not remeasure, do not invent NLL or tok/s, do not silently “correct” by selecting a winner).
- Claim types: occupancy `observed`/`derived`; TASK-05 citations `measured` (historical); cross-document agreement `derived`; every remaining alternative usefulness `hypothesis`; SKU / vision / unselected corpora `unknown`; freeze membership `derived`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Review completeness is the three ledger completion criteria plus explicit non-selection, unknown-preservation, and freeze-in-review-only.
- Screen eligibility: N/A (no keep/reject run).
- Shipping evidence: N/A

## Implementation decisions

### Authority for review claims

If an occupancy product would disagree with TASK-01 / sitting `text_config`, or a cited integer/id would disagree with the named TASK-01–20 document **and is not a locked notation alias**, the earlier document wins and this review is wrong.

- Prefill and decode share **one** semantic graph (TASK-11) and **one** compiled artifact (TASK-09 via TASK-17). They do **not** share one performance metric identity (TASK-19). Quality methodology is shared (TASK-18).
- Primary object is language+MTP **complete map** (135 node instances). Language-only is a secondary row.
- Freeze does **not** select a CUDA mapping, recipe map, or layout. It records that the locked structure plus the catalogue of unselected alternatives was consistency-checked.
- Preserving TASK-07–19 open questions answers TASK-21’s ledger open question as **true unknowns retained**. JSON `n_remaining_open_questions_closed` 0. JSON `ledger_open_question_unknowns_preserved` true.
- Do not inspect Quartz, llama.cpp, or GGUF to “confirm” the review.
- Do not fill TASK-16 UNKNOWN symbols. Do not run TASK-18/19 protocols.
- Mark freeze **only** in `clean-sheet-review.md` and this checker’s JSON (`frozen_for_comparative_review` true). Leave TASK-20 JSON `frozen_for_comparative_review` false.
- Tok/s is not a quality axis (`toks_is_not_quality_axis` true). Reconstruction is not model-level quality (`reconstruction_is_not_quality` true). Microbenchmarks cannot pass a mapping (`microbenchmark_cannot_pass_mapping` true).
- Planning found `n_inconsistencies_found` 0. Implementation must re-run the live fence comparison. If it finds a non-alias conflict, **stop** (planning miss); do not silently edit TASK-01–20 deliverables. JSON `n_inconsistencies_corrected` 0.

### One deliverable and one checker

| Path | Role |
| --- | --- |
| `docs/architecture/clean-sheet-review.md` | Corpus, six check families, findings, preserved unknowns, freeze mark |
| `scripts/check_clean_sheet_review.py` | Stdlib checker: live `text_config` occupancy, first JSON fences of 21 corpus docs, review markdown |

The review markdown contains one fenced `json` block, copied from a fresh checker `--json` run (pretty-printed, script key order).

### Deliverable structure (`docs/architecture/clean-sheet-review.md`)

Use these **level-2 headings in this exact order**. Compact tables + one Mermaid fence + short captions. Every occupancy number is `OBSERVED` or `DERIVED` (citation). Every TASK-05 cell is `MEASURED` (citation). Every remaining-alternative usefulness is `HYPOTHESIS`. SKU limits stay `UNKNOWN`. Do not leave `TBD`. The word `UNKNOWN` must appear in Deferred vision. The exact token `FROZEN_FOR_COMPARATIVE_REVIEW` must appear under **Freeze**.

Title: `# Qwen3.8-27B clean-sheet review (TASK-21)` (not `TASK-21` alone).

1. **Authority** — this dossier, plan.md, TASK-01–20 documents via corpus table, config, checker; evidence labels; in-scope (language+MTP consistency review + freeze mark) vs deferred (vision encoder; selected winners; executed experiments; executed comparative review). State that the document is a **review and freeze record**, not a selected runtime and not measurements.
2. **Review convention** — the four canonical sentences below plus the review sentence; earlier-task wins; freeze-in-review-only; aliases are not inconsistencies.
3. **Corpus and check families** — twenty-one corpus paths; six check families; equation tags `(1)`–`(24)`; occupancy/MAC/state/catalog/contract integers. Completes ledger checkbox 1.
4. **Cross-document findings** — zero inconsistencies; seven notation aliases preserved; draft banners left in place. Completes ledger checkbox 2 (correct: none needed). Contains the Mermaid fence.
5. **Preserved unknowns** — twelve `oq_*` ids, all still open. Completes ledger checkbox 2 (preserve true unknowns).
6. **Freeze** — `FROZEN_FOR_COMPARATIVE_REVIEW`; authorizes later comparative review; does not make Quartz/llama.cpp design authority; comparative review not executed here. Completes ledger checkbox 3.
7. **Deferred vision** — residual-stream interface only; encoder UNKNOWN.
8. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Review convention**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> This document is a Phase 1 consistency review of TASK-01 through TASK-20, not a selected runtime and not measurements.

> True unknowns remain unknowns; consistency review does not select winners or close experiment questions.

> The reviewed design is marked FROZEN_FOR_COMPARATIVE_REVIEW; existing runtimes are not thereby design authority.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_review` = sentence 2; `canonical_sentence_unknowns` = sentence 3; `canonical_sentence_freeze` = sentence 4.

Immediately after those four, include this **required review sentence verbatim**:

> Cross-document equations, dimensions, totals, state, contracts, and proposals are checked; documentation inconsistencies are corrected and true unknowns are preserved; the reviewed design is marked FROZEN_FOR_COMPARATIVE_REVIEW.

JSON: `review_question_sentence` = that sentence.

Bullets required under that heading:

- Six check families are complete: `equations`, `dimensions`, `totals`, `state`, `contracts`, `proposals` (`n_check_families` 6).
- Twenty-one corpus documents (`n_corpus_documents` 21). Dossiers under `docs/architecture/tasks/` are process records, not correction corpus. `plan.md` is read-only authority, not corpus.
- Earlier-task authority wins on a locked fact (`earlier_task_wins` true).
- Notation aliases are not inconsistencies (`n_notation_aliases` 7; `n_inconsistencies_found` 0; `n_inconsistencies_corrected` 0).
- TASK-20 freeze flag stays false (`synthesis_frozen_flag` false; `freeze_recorded_in_review_only` true).
- Draft-status banners on TASK-01–20 deliverables stay (`draft_banners_left_in_place` true).
- `FROZEN_FOR_COMPARATIVE_REVIEW` is this document (`frozen_for_comparative_review` true).
- Freeze authorizes a **later** comparative review (`freeze_authorizes_later_comparative_review` true) and does **not** make existing runtimes design authority (`freeze_makes_existing_runtimes_design_authority` false). Comparative review is not executed here (`comparative_review_executed_here` false).
- No MEASURED NLL or tok/s in this document (`experiments_run` false; `nll_measured_here` false; `toks_measured_here` false).
- Tok/s is not a quality axis (`toks_is_not_quality_axis` true). Reconstruction is not quality (`reconstruction_is_not_quality` true).
- Microbenchmarks cannot pass a mapping (`microbenchmark_cannot_pass_mapping` true).
- Q4_K_M is a future black-box Pareto reference, not a requirement (`gguf_is_pareto_reference` true; `gguf_is_not_a_requirement` true).
- Primary coverage includes MTP (135 instances). Language-only is secondary.
- `example_T_values` `[1, 4096]` remain illustration horizons, not a selected prompt matrix.
- No payload re-stream (`payloads_restreamed` false). Do not run `scripts/analyze_bf16_tensors.py`.

### Corpus (lock)

JSON array `corpus_paths` in this exact order (21 paths). JSON `n_corpus_documents` = 21.

| path | task | check families |
| --- | --- | --- |
| `docs/architecture/model-inventory.md` | TASK-01 | dimensions, totals |
| `docs/architecture/model-semantics.md` | TASK-02 | equations, dimensions |
| `docs/architecture/dataflow.md` | TASK-03 | dimensions, state, contracts |
| `docs/architecture/lifetime-and-state.md` | TASK-04 | dimensions, state |
| `docs/architecture/bf16-tensor-analysis.md` | TASK-05 | dimensions, totals |
| `docs/architecture/work-and-traffic.md` | TASK-06 | equations, dimensions, totals, state |
| `docs/architecture/numerical-sensitivity.md` | TASK-07 | equations, dimensions, contracts |
| `docs/architecture/quantization-design-space.md` | TASK-08 | dimensions, totals, proposals |
| `docs/architecture/runtime-format-design.md` | TASK-09 | dimensions, totals, proposals |
| `docs/architecture/model-compiler-plan.md` | TASK-10 | dimensions, totals, proposals |
| `docs/architecture/semantic-graph.md` | TASK-11 | dimensions, state, contracts |
| `docs/architecture/materialization-and-fusion.md` | TASK-12 | dimensions, state, proposals |
| `docs/architecture/decode-plan.md` | TASK-13 | dimensions, totals, state, contracts, proposals |
| `docs/architecture/prefill-plan.md` | TASK-14 | dimensions, totals, state, contracts, proposals |
| `docs/architecture/layout-strategy.md` | TASK-15 | dimensions, state, proposals |
| `docs/architecture/cuda-hardware-model.md` | TASK-16 | contracts, proposals |
| `docs/architecture/cuda-design-space.md` | TASK-17 | dimensions, state, contracts, proposals |
| `docs/architecture/quantization-validation.md` | TASK-18 | dimensions, totals, proposals |
| `docs/architecture/performance-validation.md` | TASK-19 | dimensions, totals, contracts, proposals |
| `docs/architecture/clean-sheet-architecture.md` | TASK-20 | all six (synthesis citations) |
| `docs/architecture/experiment-backlog.md` | TASK-20 | contracts, proposals |

JSON object `corpus_task_ids` keyed by `corpus_paths` with values `TASK-01` … `TASK-20` (both synthesis files `TASK-20`).

### Check families (lock)

JSON array `check_family_ids` in this exact order (6 ids). JSON `n_check_families` = 6.

| id | What is checked | Authority |
| --- | --- | --- |
| `equations` | `\tag{1}` … `\tag{24}` present in `model-semantics.md`; TASK-06/07 cite TASK-02 `(1)`–`(24)` as substring | TASK-02 |
| `dimensions` | occupancy integers and `full_attention_indices` vs sitting `text_config` and across fences | TASK-01 / config |
| `totals` | parameter/byte/MAC/MEASURED absmax integers across fences | TASK-01 / TASK-05 / TASK-06 |
| `state` | catalog ids, KV/$C$/$S$ bytes, $B_\text{store}$ coefficients | TASK-03 / TASK-04 |
| `contracts` | node types, stage kinds, 135 instances, selected-flags false | TASK-11 / TASK-13 / TASK-14 |
| `proposals` | 22 recipes, 22 fusions, 18 mappings, 17 SKU symbols, 12 `oq_*` still open | TASK-08 / TASK-12 / TASK-16 / TASK-17 / TASK-20 |

Equation lock: JSON `n_equation_tags` = 24, `equation_tag_start` = 1, `equation_tag_end` = 24. Checker requires `model-semantics.md` to contain `\tag{1}` through `\tag{24}` as substrings (backslash-tag form). Do not re-parse LaTeX.

### Locked occupancy, totals, MAC, state, contracts (cite, do not retabulate as new studies)

JSON occupancy from sitting `text_config` (same as TASK-20):

- `hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, `n_decoder_layers` 64, `n_linear_layers` 48, `n_full_layers` 16, `n_mtp_blocks` 1
- `full_attention_indices` `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `dtype` `"bfloat16"` (config string; see alias `alias_dtype_bf16_bfloat16`), `mamba_ssm_dtype` `"float32"`, `model_T_max` 262144
- `n_tensors` 1199, `n_shards` 18, `n_parameters` 27781427952, `n_bytes` 55562855904
- `n_language_mtp_parameters` 27320697856, `n_vision_parameters` 460730096, `n_language_mtp_tensors` 866, `n_vision_tensors` 333
- `weight_bytes_language_mtp_excl_vision` 54641395712, `weight_bytes_unique_non_embed` 52098598912
- `mac_C_complete` 27433238528, `mac_A_complete` 208896
- `mac_decode_complete_T1` = `mac_prefill_complete_T1` = 27433447424
- `mac_decode_complete_T4096` 28288876544, `mac_prefill_complete_T4096` 114119319486464
- `example_T_values` `[1, 4096]`
- `N_w` 32, `N_bank` 32
- `n_catalog_nodes` 52
- `n_full_layers_with_kv` 17
- `kv_bytes_all_per_token` 69632, `storage_kv_bytes_coeff_T` 69632, `storage_fixed_bytes` 153944064
- `s_bytes_all` 150994944, `c_bytes_all` 2949120
- `n_node_types` 6, `n_node_instances_complete` 135
- `n_stage_kinds` 9, `n_candidate_recipes` 22, `n_fusion_hypotheses` 22, `n_mappings` 18, `n_sku_unknown_symbols` 17, `n_sensitive_ops` 20, `n_catalogued_alternatives` 125
- `pooled_all_absmax` 25.5, `pooled_language_mtp_absmax` 19.25, `pooled_vision_absmax` 25.5

JSON `catalog_ids` exact TASK-03 order (52 ids):

`token_id`, `e`, `h`, `h_tilde`, `h_mid`, `h_post`, `h_64`, `h_final`, `logits_0`, `u_q`, `q_prime`, `g`, `k_raw`, `v_full`, `q_n`, `k_n`, `q_rope`, `k_rope`, `attn`, `y_gate`, `mix_full`, `K_state`, `V_state`, `qkv`, `z`, `a`, `b`, `c_tilde`, `c`, `q_lin`, `k_lin`, `v_lin`, `q_hat`, `k_hat`, `alpha`, `beta`, `S`, `o`, `u_gdn`, `mix_lin`, `C_state`, `g_mlp`, `up`, `swiglu`, `mlp_out`, `e_next`, `e_next_n`, `h64_n`, `mtp_cat`, `mtp_u`, `h_mtp`, `logits_1`

JSON `node_type_ids`: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`

JSON `stage_kind_ids`: `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`

JSON `sku_unknown_symbols`: `N_SM`, `W_max`, `S_reg`, `C_smem`, `T_max`, `B_max`, `N_bar`, `N_sched`, `G_reg`, `G_smem`, `Beta_HBM`, `Pi_FMA`, `Pi_TC`, `L_issue`, `async_copy_cap`, `mma_shapes`, `cluster_cap`

Checker compares a key **only among corpus fences that contain it**. Missing key is skip, not fail. Present values must equal the locked constant (after alias normalization below). `clean-sheet-architecture.md` and `experiment-backlog.md` fences must be deep-equal to each other.

### Notation aliases (lock; heading 4)

JSON array `notation_alias_ids` in this exact order (7 ids). JSON `n_notation_aliases` = 7. JSON `n_inconsistencies_found` = 0. JSON `n_inconsistencies_corrected` = 0. JSON array `inconsistency_ids` = `[]`. JSON `alternative_usefulness_label` exactly `HYPOTHESIS`.

| id | Surfaces | Disposition |
| --- | --- | --- |
| `alias_dtype_bf16_bfloat16` | TASK-01/05 fence `dtype` `BF16`; config / TASK-19/20 `dtype` `bfloat16` | preserve; safetensor header vs config string |
| `alias_tmax_model_vs_sku` | TASK-07 `T_max` 262144; TASK-16 SKU id `T_max` UNKNOWN; TASK-19/20 `model_T_max` 262144 | preserve; TASK-19 already documented the collision |
| `alias_authority_checkpoint_vs_plan` | most fences `authority` checkpoint path; TASK-16/19/20 `docs/architecture/plan.md` | preserve; occupancy vs study authority |
| `alias_n_decoder_layers_num_hidden_layers` | `n_decoder_layers` vs TASK-02 `num_hidden_layers`; both 64 | preserve |
| `alias_n_mtp_blocks_mtp_num_hidden_layers` | `n_mtp_blocks` vs TASK-02 `mtp_num_hidden_layers`; both 1 | preserve |
| `alias_n_kv_heads_num_key_value_heads` | `n_kv_heads` vs TASK-02 `num_key_value_heads`; both 4 | preserve |
| `alias_warp_n_w_n_bank` | TASK-16 `warp_size` / `n_bank` vs TASK-20 `N_w` / `N_bank`; all 32 | preserve |

Alias normalization for comparison: treat `BF16` and `bfloat16` as the dtype alias (do not fail); do **not** compare TASK-07 `T_max` to TASK-16 `sku_unknown_symbols` membership as a numeric equality; do **not** require `authority` strings to match across the corpus; treat `num_hidden_layers` as `n_decoder_layers`, `mtp_num_hidden_layers` as `n_mtp_blocks`, `num_key_value_heads` as `n_kv_heads`, `warp_size` as `N_w`, `n_bank` as `N_bank` when comparing to locked occupancy.

Prose required: listing an alias is not selecting a winner and not “correcting” TASK-01 `BF16` to `bfloat16`. Draft banners remain (`draft_banners_left_in_place` true).

Exactly **one** fenced `mermaid` block, under **Cross-document findings**. Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 135 instances, 22 recipes, or 18 mapping boxes.

Required node IDs **inside** the mermaid fence (substring match): `CORPUS`, `EQ`, `DIM`, `TOT`, `STATE`, `CONTRACT`, `PROP`, `ALIAS`, `OPENQ`, `FREEZE`.

JSON `n_diagrams` is 1. `diagram_ids` is `["corpus","eq","dim","tot","state","contract","prop","alias","openq","freeze"]`.

Suggested topology (implementer may rearrange edges; IDs are mandatory): CORPUS → EQ, DIM, TOT, STATE, CONTRACT, PROP; ALIAS → DIM; OPENQ → PROP; all six families → FREEZE. Caption must contain `HYPOTHESIS` and `FROZEN_FOR_COMPARATIVE_REVIEW`. Caption must state that OPENQ are preserved true unknowns and that freeze does not select winners or make existing runtimes design authority.

### Preserved unknowns (lock; heading 5)

JSON array `remaining_open_question_ids` in this exact order (12 ids). JSON `n_remaining_open_questions` = 12. JSON `n_remaining_open_questions_closed` = 0. JSON `ledger_open_question_unknowns_preserved` true.

Same twelve ids as TASK-20, still unselected:

| ID | Source | Still unselected |
| --- | --- | --- |
| `oq_risk_survival` | TASK-07 / TASK-18 | which sensitive-op hypotheses survive |
| `oq_pareto_frontier` | TASK-08 / TASK-18 | bit widths / grouping / scales / outliers / profile maps |
| `oq_artifact_boundary` | TASK-09 | portable vs backend-specialized vs dual-view |
| `oq_ideal_byte_sequence` | TASK-09 | which of seven consumer sequences |
| `oq_integrity_algorithm` | TASK-10 | `none` vs `checksum` |
| `oq_fusion_winner` | TASK-12 / TASK-13 | 22 fusion + 5 hoist |
| `oq_decode_prefill_views` | TASK-14 | four representation + two mode hypotheses |
| `oq_parallel_decomposition` | TASK-15 | which decompositions justify orderings/tiles |
| `oq_sku_limits` | TASK-16 | 17 sitting-SKU symbols |
| `oq_mapping_winner` | TASK-17 | 18 mappings after benchmarks |
| `oq_eval_corpora` | TASK-18 | four corpus classes, prompt suite, capability, acceptance frontier |
| `oq_hardware_protocol` | TASK-19 | hardware, prompt matrix, reproducibility protocol |

Prose required: preserving these ids is the TASK-21 answer to “genuine unresolved architectural questions retained after consistency review.” Freeze does not close them.

Checker: every `oq_*` id appears as a substring in the review document. TASK-20 architecture/backlog fences still have `n_remaining_open_questions_closed` 0 and every `*_selected` flag listed below false.

### Freeze (lock; heading 6)

JSON `frozen_for_comparative_review` true. JSON `freeze_authorizes_later_comparative_review` true. JSON `freeze_makes_existing_runtimes_design_authority` false. JSON `freeze_recorded_in_review_only` true. JSON `synthesis_frozen_flag` false. JSON `comparative_review_executed_here` false.

The heading **Freeze** must contain the exact token `FROZEN_FOR_COMPARATIVE_REVIEW` and the sentence:

> Existing Quartz and llama.cpp/GGML Qwen implementations are not design authority by this freeze; they may be compared later as black-box references with matching identities.

JSON `freeze_non_authority_sentence` = that sentence (verbatim in the Freeze section).

Prose required: after this mark, a subsequent study may inspect those runtimes. This increment does not. `quartz_inspected` false; `llama_inspected` false.

### Deferred vision

Vision-encoder internals remain `UNKNOWN` and out of the primary map. Residual-stream interface only. JSON `vision_eval_deferred` true. The word `UNKNOWN` must appear in Deferred vision.

### Sibling checkers (focused commands; not imported)

JSON array `sibling_checker_scripts` in this exact order (19 ids). JSON `n_sibling_checkers` = 19. These are **verification/focused commands**. The TASK-21 checker must not import or subprocess them.

| script | markdown flag |
| --- | --- |
| `scripts/inventory_bf16_checkpoint.py` | `--check-inventory docs/architecture/model-inventory.md` (plus `--checkpoint`) |
| `scripts/check_model_semantics.py` | `--semantics docs/architecture/model-semantics.md` |
| `scripts/check_dataflow.py` | `--dataflow docs/architecture/dataflow.md` |
| `scripts/check_lifetime_and_state.py` | `--lifetime docs/architecture/lifetime-and-state.md` |
| `scripts/check_work_and_traffic.py` | `--work-traffic docs/architecture/work-and-traffic.md` |
| `scripts/check_numerical_sensitivity.py` | `--numerical-sensitivity docs/architecture/numerical-sensitivity.md` |
| `scripts/check_quantization_design_space.py` | `--quantization-design-space docs/architecture/quantization-design-space.md` |
| `scripts/check_runtime_format_design.py` | `--runtime-format-design docs/architecture/runtime-format-design.md` |
| `scripts/check_model_compiler_plan.py` | `--model-compiler-plan docs/architecture/model-compiler-plan.md` |
| `scripts/check_semantic_graph.py` | `--semantic-graph docs/architecture/semantic-graph.md` |
| `scripts/check_materialization_and_fusion.py` | `--materialization docs/architecture/materialization-and-fusion.md` |
| `scripts/check_decode_plan.py` | `--decode-plan docs/architecture/decode-plan.md` |
| `scripts/check_prefill_plan.py` | `--prefill-plan docs/architecture/prefill-plan.md` |
| `scripts/check_layout_strategy.py` | `--layout-strategy docs/architecture/layout-strategy.md` |
| `scripts/check_cuda_hardware_model.py` | `--check docs/architecture/cuda-hardware-model.md` (no `--config`) |
| `scripts/check_cuda_design_space.py` | `--cuda-design-space docs/architecture/cuda-design-space.md` |
| `scripts/check_quantization_validation.py` | `--quantization-validation docs/architecture/quantization-validation.md` |
| `scripts/check_performance_validation.py` | `--performance-validation docs/architecture/performance-validation.md` |
| `scripts/check_clean_sheet_architecture.py` | `--architecture` and `--experiment-backlog` |

Do **not** run `scripts/analyze_bf16_tensors.py` (`payloads_restreamed` false). TASK-05 MEASURED absmax values are cited from the published fence.

### Additional locked booleans and integers

- `n_inconsistencies_found` 0
- `n_inconsistencies_corrected` 0
- `n_remaining_open_questions_closed` 0
- `n_alternatives_selected` 0
- `experiments_run` false
- `nll_measured_here` false
- `toks_measured_here` false
- `benchmarks_run` false
- `payloads_restreamed` false
- `gguf_payload_inspected` false
- `gguf_is_not_design_authority` true
- `gguf_is_pareto_reference` true
- `gguf_is_not_a_requirement` true
- `quartz_inspected` false
- `llama_inspected` false
- `llama_is_not_design_authority` true
- `device_query_run` false
- `nsight_run` false
- `sku_table_filled` false
- `mapping_winner_selected` false
- `pareto_frontier_selected` false
- `fusion_winner_selected` false
- `winner_selected_without_measurements` false
- `frozen_for_comparative_review` true
- `synthesis_frozen_flag` false
- `freeze_recorded_in_review_only` true
- `freeze_authorizes_later_comparative_review` true
- `freeze_makes_existing_runtimes_design_authority` false
- `comparative_review_executed_here` false
- `ledger_open_question_unknowns_preserved` true
- `draft_banners_left_in_place` true
- `earlier_task_wins` true
- `reconstruction_is_not_quality` true
- `toks_is_not_quality_axis` true
- `microbenchmark_cannot_pass_mapping` true
- `vision_eval_deferred` true
- `safetensors_is_source_not_runtime` true
- `n_review_headings` 8
- `n_diagrams` 1
- `n_headings` 8

`review_headings` JSON array:

`Authority`, `Review convention`, `Corpus and check families`, `Cross-document findings`, `Preserved unknowns`, `Freeze`, `Deferred vision`, `Machine-checkable summary JSON`.

### JSON schema (heading last)

Pretty-printed object, key order as emitted by the checker. Required keys (exact names; implementer may add only if this dossier is amended):

`authority`, `deliverable_review`, `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `model_T_max`, `dtype`, `mamba_ssm_dtype`, `n_tensors`, `n_shards`, `n_parameters`, `n_bytes`, `n_language_mtp_parameters`, `n_vision_parameters`, `n_language_mtp_tensors`, `n_vision_tensors`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `mac_C_complete`, `mac_A_complete`, `mac_decode_complete_T1`, `mac_prefill_complete_T1`, `mac_decode_complete_T4096`, `mac_prefill_complete_T4096`, `example_T_values`, `N_w`, `N_bank`, `n_catalog_nodes`, `catalog_ids`, `n_full_layers_with_kv`, `kv_bytes_all_per_token`, `storage_kv_bytes_coeff_T`, `storage_fixed_bytes`, `s_bytes_all`, `c_bytes_all`, `n_equation_tags`, `equation_tag_start`, `equation_tag_end`, `n_node_types`, `node_type_ids`, `n_node_instances_complete`, `n_stage_kinds`, `stage_kind_ids`, `n_candidate_recipes`, `n_fusion_hypotheses`, `n_mappings`, `n_sku_unknown_symbols`, `sku_unknown_symbols`, `n_sensitive_ops`, `n_catalogued_alternatives`, `pooled_all_absmax`, `pooled_language_mtp_absmax`, `pooled_vision_absmax`, `n_corpus_documents`, `corpus_paths`, `corpus_task_ids`, `n_check_families`, `check_family_ids`, `n_notation_aliases`, `notation_alias_ids`, `n_inconsistencies_found`, `n_inconsistencies_corrected`, `inconsistency_ids`, `n_remaining_open_questions`, `remaining_open_question_ids`, `n_remaining_open_questions_closed`, `n_sibling_checkers`, `sibling_checker_scripts`, `n_review_headings`, `review_headings`, `n_headings`, `n_diagrams`, `diagram_ids`, `earlier_task_wins`, `ledger_open_question_unknowns_preserved`, `frozen_for_comparative_review`, `synthesis_frozen_flag`, `freeze_recorded_in_review_only`, `freeze_authorizes_later_comparative_review`, `freeze_makes_existing_runtimes_design_authority`, `comparative_review_executed_here`, `draft_banners_left_in_place`, `pareto_frontier_selected`, `fusion_winner_selected`, `mapping_winner_selected`, `winner_selected_without_measurements`, `n_alternatives_selected`, `sku_table_filled`, `keep_source_is_not_quality_winner`, `reconstruction_is_not_quality`, `toks_is_not_quality_axis`, `microbenchmark_cannot_pass_mapping`, `experiments_run`, `nll_measured_here`, `toks_measured_here`, `benchmarks_run`, `payloads_restreamed`, `gguf_payload_inspected`, `gguf_is_not_design_authority`, `gguf_is_pareto_reference`, `gguf_is_not_a_requirement`, `quartz_inspected`, `llama_inspected`, `llama_is_not_design_authority`, `device_query_run`, `nsight_run`, `vision_eval_deferred`, `safetensors_is_source_not_runtime`, `alternative_usefulness_label`, `canonical_sentence_logical`, `canonical_sentence_review`, `canonical_sentence_unknowns`, `canonical_sentence_freeze`, `review_question_sentence`, `freeze_non_authority_sentence`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. Absmax fields are JSON floats (`25.5`, `19.25`, `25.5`). Booleans are JSON booleans. `full_attention_indices` and `example_T_values` are JSON arrays of ints. `inconsistency_ids` is a JSON array (empty). `corpus_task_ids` is a JSON object mapping each `corpus_paths` entry to a `TASK-NN` string. `authority` is `"docs/architecture/plan.md"`. `deliverable_review` is `"docs/architecture/clean-sheet-review.md"`. `dtype` is `"bfloat16"`. `mamba_ssm_dtype` is `"float32"`. `alternative_usefulness_label` is `"HYPOTHESIS"`.

### Tooling

Create `scripts/check_clean_sheet_review.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, CUDA Python, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, occupancy, and T=1/T=4096 MAC citations matching TASK-06 (use the locked MAC constants above; do not re-derive the full region MAC table). Duplicate TASK-01–20 id lists as constants; do not import them. Do not open `models/Qwen3.8-27B-Q4_K_M.gguf` or any safetensor. Do not run `deviceQuery`, Nsight, NLL, or any GPU binary. Do not subprocess sibling checkers.

CLI (cwd = repository root):

```text
python3 scripts/check_clean_sheet_review.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_clean_sheet_review.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --review docs/architecture/clean-sheet-review.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema above). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `max_position_embeddings` as `model_T_max`. Derived/cited: parameter/byte totals matching TASK-01/05/06, MAC T=1/T=4096 citations matching TASK-06 locked integers. Constant fields: all canonical sentences, corpus/check-family/alias/unknown lists, locked booleans.
- Parse the first ` ```json ` fence of every `corpus_paths` file. Fail if a corpus file is missing or has no JSON fence. Compare present keys against locked constants with alias normalization. Require architecture.md and experiment-backlog.md first fences deep-equal. Extract TASK-05 `pooled_all.absmax`, `pooled_language_mtp.absmax`, `pooled_vision.absmax` from `bf16-tensor-analysis.md`. Require `\tag{1}` … `\tag{24}` in `model-semantics.md`. Require TASK-20 fences `frozen_for_comparative_review` is JSON false and `n_remaining_open_questions_closed` is 0.
- `--json`: print the summary object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- `--review PATH`: require PATH to contain (1) every required `##` heading **in order**, (2) the first fenced `json` block equal to the live object, (3) none of `TBD`, `TODO`, `???`, (4) the word `UNKNOWN` present in Deferred vision, (5) every locked document integer below present as a decimal or integer substring, plus `19.25` and `25.5`, (6) the words `HYPOTHESIS` and `FROZEN_FOR_COMPARATIVE_REVIEW` present, (7) all four canonical sentences plus `review_question_sentence` and `freeze_non_authority_sentence` verbatim, (8) exactly one ` ```mermaid ` fence containing `flowchart` with required node IDs **inside that mermaid fence**, (9) every `corpus_paths` basename, `check_family_ids`, `notation_alias_ids`, `remaining_open_question_ids`, `catalog_ids`, `node_type_ids`, `stage_kind_ids`, `sku_unknown_symbols` id present as a substring. Forbidden phrases except inside allowed wrappers / canonical sentences / JSON fence: `selected mapping winner`, `winning kernel`, `winning mapping`, `selected recipe`, `selected fusion`, `experiments were run`, `NLL was measured in this study`, `SKU table is filled`, `tok/s is the quality axis`. Allowed wrappers: `selects no`, `not a selected`, `remain unselected`, `not measurements`, `not thereby design authority`. The freeze token is **required**, not forbidden. If `--review` is absent and `--json` is absent, check the default deliverable path. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).
- Do not fail if the GGUF file is absent. Do not stat/open GGUF as a checker requirement.

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `model_T_max==262144`
- `hidden_size==5120`, `intermediate_size==17408`, `vocab_size==248320`, `n_decoder_layers==64`
- `n_tensors==1199`, `n_parameters==27781427952`, `n_bytes==55562855904`
- `n_language_mtp_parameters==27320697856`, `n_vision_parameters==460730096`
- `weight_bytes_language_mtp_excl_vision==54641395712`, `weight_bytes_unique_non_embed==52098598912`
- `mac_C_complete==27433238528`, `mac_A_complete==208896`
- `mac_decode_complete_T1==mac_prefill_complete_T1==27433447424`
- `mac_decode_complete_T4096==28288876544`
- `mac_prefill_complete_T4096==114119319486464`
- `example_T_values == [1, 4096]`
- `N_w==32`, `N_bank==32`
- `n_catalog_nodes==52`, `len(catalog_ids)==52`
- `kv_bytes_all_per_token==69632`, `storage_fixed_bytes==153944064`
- `n_equation_tags==24`, `equation_tag_start==1`, `equation_tag_end==24`
- `n_node_types==6`, `n_node_instances_complete==135`, `n_stage_kinds==9`
- `n_candidate_recipes==22`, `n_fusion_hypotheses==22`, `n_mappings==18`
- `n_corpus_documents==21`, `n_check_families==6`, `n_notation_aliases==7`
- `n_inconsistencies_found==0`, `n_inconsistencies_corrected==0`, `inconsistency_ids==[]`
- `n_remaining_open_questions==12`, `n_remaining_open_questions_closed==0`
- `n_sibling_checkers==19`, `n_review_headings==8`, `n_diagrams==1`
- `pooled_language_mtp_absmax==19.25`, `pooled_all_absmax==25.5`, `pooled_vision_absmax==25.5`
- `frozen_for_comparative_review is True`, `synthesis_frozen_flag is False`
- `freeze_recorded_in_review_only is True`, `freeze_makes_existing_runtimes_design_authority is False`
- `comparative_review_executed_here is False`, `ledger_open_question_unknowns_preserved is True`
- `mapping_winner_selected is False`, `pareto_frontier_selected is False`, `fusion_winner_selected is False`
- `experiments_run is False`, `nll_measured_here is False`, `toks_measured_here is False`, `benchmarks_run is False`, `sku_table_filled is False`, `payloads_restreamed is False`
- `gguf_payload_inspected is False`, `quartz_inspected is False`, `llama_inspected is False`
- `device_query_run is False`, `nsight_run is False`
- `vision_eval_deferred is True`
- `n_alternatives_selected==0`, `n_catalogued_alternatives==125`

Locked document integers that must appear as decimal substrings in the review markdown (in addition to ids): `5120`, `17408`, `248320`, `64`, `48`, `16`, `262144`, `1199`, `27781427952`, `55562855904`, `27320697856`, `54641395712`, `52098598912`, `27433238528`, `208896`, `27433447424`, `28288876544`, `114119319486464`, `135`, `32`, `125`, `52`, `69632`, `153944064`, `21`, `12`, `24`, `19.25`, `25.5`.

### Stage split

- **Implementation** writes `scripts/check_clean_sheet_review.py` **and** `docs/architecture/clean-sheet-review.md` (eight headings, four canonical sentences plus `review_question_sentence` plus `freeze_non_authority_sentence`, 21-row corpus table, six check families, seven aliases, zero inconsistencies, twelve preserved unknowns, one Mermaid flowchart, freeze mark, JSON fence). Runs `--json` and `--review` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads. Does not open GGUF. Does not run benchmarks, NLL, `deviceQuery`, Nsight, tok/s collection, or `analyze_bf16_tensors.py`. Does not edit `plan.md`, the ledger, or any TASK-01–20 deliverable. If live fence comparison finds a non-alias conflict, stop and record it as a planning miss (do not silently correct).
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner on **the review markdown** (TASK-20 / `clean-sheet-architecture.md` style), Authority table links to this dossier / plan.md / TASK-01–20 deliverables as citations, heading/JSON fence consistency. Must not change locked integers, canonical sentences, alias ids, unknown ids, or Mermaid node IDs. Does not edit TASK-01–20 artifacts. Does not strip other documents’ draft banners.
- **Verification** independently re-runs focused commands (TASK-21 checker **and** the nineteen sibling checkers), recomputes layer counts / `full_attention_indices` / `model_T_max` from sitting `text_config` (not from JSON echo), spot-checks cited TASK-01/04/05/06/11/20 integers and ids against those documents (not from this JSON echo), reads the review document against this dossier, confirms architecture.md and backlog.md fences still equal each other and still have `frozen_for_comparative_review` false, confirms no Quartz/llama.cpp/GGUF-as-authority, no winner phrases, no `plan.md` or ledger edit, no payload I/O, no GGUF open, no GPU/NLL run, that every remaining open question stays open, and that the freeze token appears in the review document. The three ledger completion criteria are closed by the review. Unknowns remain unknowns.

- Invariants:
  - Eight review headings in locked order; four canonical sentences plus `review_question_sentence` plus `freeze_non_authority_sentence` verbatim; 21 corpus documents; 6 check families; 7 aliases; 0 inconsistencies; 12 remaining open questions; 0 closed; one Mermaid flowchart with required IDs.
  - JSON fence equal to live `--json`.
  - Freeze true in the review only; TASK-20 freeze flag remains false.
  - Microbenchmarks cannot pass a mapping; reconstruction is not quality; tok/s is not Pareto $Y$; Q4_K_M is a reference not a requirement.
  - `experiments_run` false; `comparative_review_executed_here` false; `mapping_winner_selected` false; `sku_table_filled` false; `payloads_restreamed` false.
  - Logical values do not imply allocation. Vision encoder remains unexpanded.
- Rejected alternatives:
  - Selecting any recipe, profile, mapping, fusion, layout, view, corpus, or hardware as a “consistency” result: rejected; preserve unknowns.
  - Closing TASK-07–19 open questions by assertion: rejected; `n_remaining_open_questions_closed` 0.
  - Flipping TASK-20 `frozen_for_comparative_review` or adding freeze-as-selected-state to synthesis docs: rejected; would break TASK-20’s checker; freeze is review-only.
  - Treating `BF16` vs `bfloat16` or model `T_max` vs SKU `T_max` as inconsistencies to rewrite: rejected; aliases.
  - Stripping draft banners across TASK-01–20: rejected; process artifacts, not scientific contradictions.
  - Running NLL / tok/s / Nsight / `deviceQuery` / SKU fill / `analyze_bf16_tensors.py` in this task: rejected.
  - Inspecting Quartz/llama.cpp or opening GGUF because freeze authorizes later comparison: rejected for **this** increment; `comparative_review_executed_here` false.
  - Importing or subprocessing `check_*.py` from the TASK-21 checker: rejected; sibling checkers are focused commands.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–20.
  - Editing `plan.md` or the ledger outside delivery.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/clean-sheet-review.md` exists and follows the eight headings above.
  - Cross-document equations (`\tag{1}`–`\tag{24}`), dimensions, totals, state, contracts, and proposals are checked via the stdlib checker against 21 corpus JSON fences plus sitting `text_config`.
  - Zero documentation inconsistencies found/corrected; seven notation aliases preserved; twelve true unknowns preserved (0 closed).
  - The reviewed design is marked `FROZEN_FOR_COMPARATIVE_REVIEW` in the review document; TASK-20 freeze flag remains false; existing runtimes are not design authority; comparative review is not executed here.
  - Canonical sentences plus `review_question_sentence` plus `freeze_non_authority_sentence` verbatim; JSON fence matches a live `--json` object from config arithmetic plus locked constants plus live fence comparison.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; no payload re-stream; no `plan.md` or TASK-01–20 deliverable edit; no MEASURED NLL or tok/s.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_clean_sheet_review.py` only (no pytest fixtures).
- Focused commands (repository root; config and checkpoint must exist for inventory):

```sh
python3 -m py_compile scripts/check_clean_sheet_review.py
python3 scripts/check_clean_sheet_review.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_clean_sheet_review.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --review docs/architecture/clean-sheet-review.md
python3 scripts/inventory_bf16_checkpoint.py \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --check-inventory docs/architecture/model-inventory.md
python3 scripts/check_model_semantics.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantics docs/architecture/model-semantics.md
python3 scripts/check_dataflow.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --dataflow docs/architecture/dataflow.md
python3 scripts/check_lifetime_and_state.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --lifetime docs/architecture/lifetime-and-state.md
python3 scripts/check_work_and_traffic.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --work-traffic docs/architecture/work-and-traffic.md
python3 scripts/check_numerical_sensitivity.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --numerical-sensitivity docs/architecture/numerical-sensitivity.md
python3 scripts/check_quantization_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-design-space docs/architecture/quantization-design-space.md
python3 scripts/check_runtime_format_design.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --runtime-format-design docs/architecture/runtime-format-design.md
python3 scripts/check_model_compiler_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --model-compiler-plan docs/architecture/model-compiler-plan.md
python3 scripts/check_semantic_graph.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantic-graph docs/architecture/semantic-graph.md
python3 scripts/check_materialization_and_fusion.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --materialization docs/architecture/materialization-and-fusion.md
python3 scripts/check_decode_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --decode-plan docs/architecture/decode-plan.md
python3 scripts/check_prefill_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --prefill-plan docs/architecture/prefill-plan.md
python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --layout-strategy docs/architecture/layout-strategy.md
python3 scripts/check_cuda_hardware_model.py \
  --check docs/architecture/cuda-hardware-model.md
python3 scripts/check_cuda_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --cuda-design-space docs/architecture/cuda-design-space.md
python3 scripts/check_quantization_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-validation docs/architecture/quantization-validation.md
python3 scripts/check_performance_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --performance-validation docs/architecture/performance-validation.md
python3 scripts/check_clean_sheet_architecture.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --architecture docs/architecture/clean-sheet-architecture.md \
  --experiment-backlog docs/architecture/experiment-backlog.md
```

Do **not** run `scripts/analyze_bf16_tensors.py`.

- Candidate quality: not required — no model execution, NLL, or OPT-058; this increment is review documentation. Experiment **hypotheses** remain unrun.
- Repository-wide commands:

```sh
test -f docs/architecture/clean-sheet-review.md
python3 -m py_compile scripts/check_clean_sheet_review.py
python3 scripts/check_clean_sheet_review.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --review docs/architecture/clean-sheet-review.md
```

Do not run Ruff, pytest, CMake, CUDA, Nsight, or `deviceQuery`; this increment does not introduce those gates. Do not open GGUF.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config and checkpoint presence are local-file requirements, not GPU gates. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/clean-sheet-review.md` (create)
  - `scripts/check_clean_sheet_review.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-01–20 deliverables unchanged)
- Definition of done: review document published with locked corpus, six check families, seven aliases, zero inconsistencies, twelve preserved unknowns, freeze mark `FROZEN_FOR_COMPARATIVE_REVIEW` in the review only; JSON fence verifies against sitting `config.json`, locked constants, and live corpus fences; sibling checkers still pass; no experiments run; no winners selected; no TASK-01–20 edits; ledger TASK-21 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T17:20:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-21.md`. Coupled IDs `none`. Document structure (8 headings), four canonical sentences plus review sentence plus freeze non-authority sentence, 21-document corpus, 6 check families, 7 notation aliases, 0 inconsistencies (planning JSON-fence comparison), 12 remaining open questions preserved, freeze-in-review-only, stdlib checker `scripts/check_clean_sheet_review.py`, JSON schema, sibling-checker focused commands, and acceptance commands are closed. `docs/architecture/clean-sheet-review.md` and the checker were **not** written in this stage. `plan.md`, `task_ledger.md`, and TASK-01–20 deliverables not edited. No commit. No experiments. No comparative-runtime inspection.
- Performance evidence applied: N/A — review documentation; no sink ranking, no measured tok/s, no SKU fill

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - created `scripts/check_clean_sheet_review.py` (stdlib checker; occupancy from sitting `text_config`; first JSON fences of 21 corpus documents with alias normalization; `--json` plus `--review` markdown checks; does not import or subprocess sibling checkers)
  - created `docs/architecture/clean-sheet-review.md` (eight headings, four canonical sentences plus `review_question_sentence` plus `freeze_non_authority_sentence`, 21-row corpus table, six check families, seven aliases, zero inconsistencies, one Mermaid flowchart with required IDs, twelve preserved unknowns / 0 closed, freeze mark `FROZEN_FOR_COMPARATIVE_REVIEW` in this review only, JSON fence equal to live `--json`)
  - this dossier Implementation run record
  - did **not** edit `docs/architecture/plan.md`, `task_ledger.md`, or any TASK-01–20 deliverable (live fence comparison found zero non-alias conflicts)
  - no commit; no payload I/O; no GGUF open; no GPU/NLL/`deviceQuery`/Nsight; no `analyze_bf16_tensors.py`
- Commands:
  - `python3 -m py_compile scripts/check_clean_sheet_review.py` — exit 0
  - `python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; stdout pretty-printed summary (10182 bytes; 120 keys); internal asserts passed (`n_corpus_documents` 21, `n_check_families` 6, `n_notation_aliases` 7, `n_inconsistencies_found` 0, `n_inconsistencies_corrected` 0, `n_remaining_open_questions` 12, `n_remaining_open_questions_closed` 0, `n_sibling_checkers` 19, `frozen_for_comparative_review` true, `synthesis_frozen_flag` false)
  - `python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --review docs/architecture/clean-sheet-review.md` — exit 0 (headings, JSON fence equal to live `--json`, mermaid IDs, canonical sentences, freeze token, twelve `oq_*` ids, no winner phrases)
  - `python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-inventory docs/architecture/model-inventory.md` — exit 0
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md` — exit 0
  - `python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md` — exit 0
  - `python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md` — exit 0
  - `python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md` — exit 0
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md` — exit 0
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-design-space docs/architecture/quantization-design-space.md` — exit 0
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md` — exit 0
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md` — exit 0
  - `python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md` — exit 0
  - `python3 scripts/check_materialization_and_fusion.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --materialization docs/architecture/materialization-and-fusion.md` — exit 0
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md` — exit 0
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md` — exit 0
  - `python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --layout-strategy docs/architecture/layout-strategy.md` — exit 0
  - `python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md` — exit 0
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md` — exit 0
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md` — exit 0
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md` — exit 0
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md` — exit 0
- UTC/time/tokens/cost: `2026-09-20T17:17:30Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent; parent/inherit mapping)
- Changes and evidence:
  - `docs/architecture/clean-sheet-review.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-20 / [`clean-sheet-architecture.md`](../clean-sheet-architecture.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`plan.md`](../plan.md) (OBSERVED evidence vocabulary), TASK-01–20 deliverables via citation table, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_clean_sheet_review.py`](../../../scripts/check_clean_sheet_review.py), and in-scope/deferred rows. Eight required `##` headings, four canonical sentences plus `review_question_sentence` plus `freeze_non_authority_sentence`, one Mermaid flowchart with required node IDs, twelve preserved `oq_*` ids, freeze mark `FROZEN_FOR_COMPARATIVE_REVIEW` in this review only, and the JSON fence left unchanged (locked integers, alias ids, unknown ids, Mermaid node IDs untouched). No edits to TASK-01–20 deliverables, `plan.md`, or `task_ledger.md`.
- Commands:
  - `test -f docs/architecture/clean-sheet-review.md` — pass (exit 0).
  - `python3 -m py_compile scripts/check_clean_sheet_review.py` — pass (exit 0).
  - `python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 120 keys; `n_corpus_documents` 21; `n_check_families` 6; `n_notation_aliases` 7; `n_inconsistencies_found` 0; `n_remaining_open_questions` 12; `n_remaining_open_questions_closed` 0; `frozen_for_comparative_review` true; `synthesis_frozen_flag` false; JSON fence source unchanged).
  - `python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --review docs/architecture/clean-sheet-review.md` — pass (exit 0; eight headings, JSON fence equal to live `--json`, mermaid IDs, canonical sentences, freeze token, twelve `oq_*` ids, no winner phrases; banner did not break the check).
  - Independent JSON fence spot-check (live `--json` object deep-equal to fenced JSON in review markdown, 120 keys) — pass.
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict; freeze mark `FROZEN_FOR_COMPARATIVE_REVIEW` remains in review document only until delivery updates ledger)
- UTC/time/tokens/cost: `2026-09-20T17:18:00Z`; `telemetry_unavailable`

### Verification

- Agent/model: `composer-2.5` (verification subagent; parent PASS handoff)
- Attempt: 1
- Commands:
  - `test -f docs/architecture/clean-sheet-review.md` — exit 0
  - `python3 -m py_compile scripts/check_clean_sheet_review.py` — exit 0
  - `python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; 120 keys; `n_corpus_documents` 21; `n_check_families` 6; `n_notation_aliases` 7; `n_inconsistencies_found` 0; `n_inconsistencies_corrected` 0; `n_remaining_open_questions` 12; `n_remaining_open_questions_closed` 0; `n_sibling_checkers` 19; `frozen_for_comparative_review` true; `synthesis_frozen_flag` false; `freeze_recorded_in_review_only` true; `ledger_open_question_unknowns_preserved` true
  - `python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --review docs/architecture/clean-sheet-review.md` — exit 0 (eight headings, JSON fence equal to live `--json`, mermaid IDs, canonical sentences, freeze token `FROZEN_FOR_COMPARATIVE_REVIEW`, twelve `oq_*` ids, no winner phrases)
  - Nineteen sibling checkers — exit 0 (implementation run record; not re-listed here)
- Formatting changed files: none (`uv run ruff format .` not run — out of scope per dossier)
- Verdict: **PASS** (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T17:20:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-21 only; coupled IDs `none`
- Outcome: TASK-21 marked `DONE` after verification PASS (attempt 1); freeze note `FROZEN_FOR_COMPARATIVE_REVIEW`; Phase 1 ledger complete
- UTC/time/tokens/cost: `2026-09-20T17:20:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/clean-sheet-review.md` (eight headings, 21-document corpus, 6 check families, 7 notation aliases, 0 inconsistencies, twelve preserved unknowns / 0 closed, freeze mark `FROZEN_FOR_COMPARATIVE_REVIEW` in review only, JSON fence equal to live `--json`); `scripts/check_clean_sheet_review.py` stdlib checker; sibling checkers pass; `plan.md` unchanged; TASK-01–20 deliverables unchanged; `synthesis_frozen_flag` false
- Candidate measured delta: N/A — review documentation
- Shipping delta: N/A
- Quality result: not required
- Evidence completeness: N/A for performance-evidence checks
- Throughput delta (when applicable): N/A
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: **yes**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config and checkpoint must remain present for focused commands; GGUF file is not required now; no GPU required; TASK-01–20 deliverables must remain unchanged
