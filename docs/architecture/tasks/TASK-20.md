# TASK-20 — Synthesize the clean-sheet architecture

## Control

- Primary ID: `TASK-20`
- Coupled IDs: `none`
- Dependencies: `TASK-01` through `TASK-19` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Synthesize the required architecture chain and evidence classes; Include an architecture diagram and alternatives; Create dependency-ordered experiment entries with all requested fields.

## Goal and boundaries

Produce `docs/architecture/clean-sheet-architecture.md` and `docs/architecture/experiment-backlog.md` as the Phase 1 **synthesis**: connect TASK-01–19 evidence into one candidate architecture (locked structure plus named unselected alternatives) and one dependency-ordered validation backlog. Close the three ledger completion criteria by (1) publishing the plan.md architecture chain with the five evidence classes attached to every synthesized task, (2) including one architecture diagram plus a catalogue of remaining alternatives, and (3) publishing sixteen experiment entries that each contain all twelve requested fields, in a valid topological order. Do **not** select a recipe, profile, artifact boundary, sequence, fusion, view, layout, mapping, corpus, hardware, or integrity winner. Do **not** run NLL, reconstruction, tok/s, Nsight, or SKU fill. Do **not** mark `FROZEN_FOR_COMPARATIVE_REVIEW` (TASK-21). **Catalogue** remaining TASK-07–19 open questions; do **not** close them.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - The candidate architecture is the **locked Phase 1 structure** (equations, graph, contracts, bounds, methodologies) plus **unselected experiment spaces**. Listing an alternative is not selecting it.
  - If any occupancy, MAC/byte, node type, stage kind, recipe, mapping, or id would disagree with TASK-01–19 or sitting `text_config`, the earlier document wins and this synthesis is wrong.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established; TASK-16 $N_w=32$, $N_{\text{bank}}=32$), `MEASURED` (TASK-05 citations only; no new payload stats), `DERIVED` (chain membership, occupancy/MAC citations, experiment DAG), `HYPOTHESIS` (every remaining alternative usefulness; every experiment hypothesis), or `UNKNOWN` (sitting-SKU numeric limits; vision-encoder internals; unselected corpora/hardware). No new `MEASURED` NLL, tok/s, occupancy, or bandwidth.
  - GitHub Markdown math only where a citation needs it. Cite TASK-01–19 ids and documents; do not rewrite forward math, recopy MAC tables, recopy 22 recipes as a new quantization space, recopy 18 mappings as a new design space, or recopy TASK-18/19 methodologies.
  - Allowed evidence: TASK-01–19 deliverables and their JSON fences, sitting `config.json` `text_config`, plan.md evidence vocabulary, this dossier. TASK integers and ids already published may be **cited**. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf` as a design authority or file to open. GGUF remains a named future black-box Pareto **reference** (TASK-18).
- Non-goals:
  - No selected winners (`n_alternatives_selected` 0; every `*_selected` / `*_winner_selected` flag cited from TASK-08–19 remains false).
  - No executed experiments (`experiments_run` false; `nll_measured_here` false; `toks_measured_here` false; `benchmarks_run` false; `sku_table_filled` false).
  - No freeze (`frozen_for_comparative_review` false). TASK-21 owns consistency review and the freeze mark.
  - No closing of TASK-07–19 ledger open questions (`n_remaining_open_questions_closed` 0). Cataloguing is not closing.
  - No new operators, node types, recipes, mappings, metrics, or stage kinds.
  - No Quartz/llama.cpp inspection; do not open the GGUF file (`gguf_payload_inspected` false).
  - No CUDA kernels, `deviceQuery`, Nsight, sitting-GPU clocks, or OPT-058 harness.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–19). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, or any TASK-01–19 deliverable.
  - Do not import any `scripts/check_*.py` or `scripts/analyze_bf16_tensors.py`. Do not stream safetensor payloads.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib clean-sheet-architecture checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central design path (mathematics → dataflow → lifetime → numerical sensitivity → precision strategy → custom quantization → custom physical layouts → offline compiler → semantic graph → decode/prefill plans → CUDA mappings → future experimental validation). TASK-20 synthesizes this chain.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels DERIVED / OBSERVED / MEASURED / HYPOTHESIS / UNKNOWN; hypotheses remain hypotheses.
- `docs/architecture/plan.md:97-100` — TASK-20 connects all findings into a candidate architecture and dependency-ordered experiment backlog; TASK-21 consistency-checks and marks `FROZEN_FOR_COMPARATIVE_REVIEW`.
- `docs/architecture/plan.md:107-115` — Phase 1 produces a mathematical and evidence-labelled architecture dossier, not a runtime; experiment backlog is a study deliverable.
- `docs/architecture/task_ledger.md` TASK-01–19 **Status: DONE**; TASK-20 depends on TASK-01 through TASK-19; produces `clean-sheet-architecture.md` and `experiment-backlog.md`; completion is chain+evidence classes, diagram+alternatives, and dependency-ordered experiment entries with all requested fields; open question (candidate alternatives and unknowns remaining) is **catalogued here, not closed by selecting**.
- `docs/architecture/task_ledger.md` remaining open questions (still unresolved; catalogue only): TASK-07 risk survival; TASK-08 Pareto frontier; TASK-09 portable vs specialized artifact and ideal byte sequences; TASK-10 integrity `none` vs `checksum`; TASK-12/13 fusion winners; TASK-14 decode/prefill views; TASK-15 parallel decompositions; TASK-16 sitting SKU; TASK-17 mapping winners; TASK-18 corpora / prompt suite / capability / acceptance frontier; TASK-19 hardware / prompt matrix / reproducibility.
- Established-result citations (do not re-derive): TASK-01 inventory 1199 tensors / 27320697856 language+MTP parameters / 54641395712 BF16 bytes; TASK-02 equations; TASK-03 52-ID DAG; TASK-04 state $B_\text{store}(T)=69632T+153944064$; TASK-05 MEASURED `pooled_language_mtp.absmax=19.25`; TASK-06 $C_\text{complete}=27433238528$, $A_\text{complete}=208896$; TASK-07 20 sensitive ops / four precision roles; TASK-08 22 recipes / 16 families; TASK-09 4 artifact approaches / 7 sequences / 8 open decisions; TASK-10 six compiler stages / three profiles; TASK-11 six node types / 135 instances; TASK-12 22 fusion hypotheses; TASK-13/14 nine stage kinds; TASK-14 four representation + two mode hypotheses; TASK-15 14 orderings / 9 tiles / 9 decompositions; TASK-16 17 SKU-UNKNOWN / $N_w=32$; TASK-17 18 mappings; TASK-18 quality methodology; TASK-19 performance methodology.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads. Do not read GGUF.
- `scripts/check_performance_validation.py` / `check_quantization_validation.py` / `check_cuda_design_space.py` — checker-style precedent. TASK-20’s checker is a sibling; do not import them.

## Performance evidence

N/A — Phase 1 **synthesis and unrun experiment backlog**. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. MAC/byte figures are TASK-06 citations. Experiment hypotheses are HYPOTHESIS. Do not apply the performance-evidence checklist to rank kernels or claim a winning mapping, recipe, or layout.

- Measurement identity: N/A (no engine binary, no eval run). Future identity remains the TASK-19 protocol (unselected values).
- Metric class: N/A for this increment. Future classes stay TASK-18 quality and TASK-19 performance; tok/s is not a quality axis.
- Coverage: N/A for GPU graphs. Synthesis coverage is 19 tasks, 12 chain steps, 5 evidence classes, 14 alternative families (125 catalogued alternatives, 0 selected), 12 remaining open questions (0 closed), 16 experiments, 12 requested fields.
- Time accounting: N/A
- Contradiction register: none at planning. If live `text_config` occupancy disagrees with TASK-01, or a cited TASK-01–19 integer or id disagrees with those documents, stop and fail closed (do not remeasure, do not invent NLL or tok/s).
- Claim types: occupancy `observed`/`derived`; TASK-05 citations `measured` (historical); chain membership `derived`; every experiment hypothesis and alternative usefulness `hypothesis`; SKU / vision / unselected corpora `unknown`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Synthesis completeness is the three ledger completion criteria plus explicit non-selection and non-freeze.
- Screen eligibility: N/A (no keep/reject run).
- Shipping evidence: N/A

## Implementation decisions

### Authority for synthesis claims

If an occupancy product would disagree with TASK-01 / sitting `text_config`, or a cited integer/id would disagree with the named TASK-01–19 document, the earlier document wins and this synthesis is wrong.

- Prefill and decode share **one** semantic graph (TASK-11) and **one** compiled artifact (TASK-09 via TASK-17). They do **not** share one performance metric identity (TASK-19). Quality methodology is shared (TASK-18).
- Primary object is language+MTP **complete map** (135 node instances). Language-only is a secondary row.
- The candidate architecture is **not** a selected CUDA mapping, recipe map, or layout. It is the locked structure plus the catalogue of alternatives.
- Cataloguing TASK-07–19 open questions answers TASK-20’s ledger open question as a **catalogue**, not as selected winners. JSON `ledger_open_question_alternatives_catalogued` true. JSON `n_remaining_open_questions_closed` 0.
- Do not inspect Quartz, llama.cpp, or GGUF to “confirm” the synthesis.
- Do not fill TASK-16 UNKNOWN symbols. Do not run TASK-18/19 protocols.
- Do not mark freeze. TASK-21 owns `FROZEN_FOR_COMPARATIVE_REVIEW`.
- Tok/s is not a quality axis (`toks_is_not_quality_axis` true). Reconstruction is not model-level quality (`reconstruction_is_not_quality` true). Microbenchmarks cannot pass a mapping (`microbenchmark_cannot_pass_mapping` true).

### Two deliverables and one checker

| Path | Role |
| --- | --- |
| `docs/architecture/clean-sheet-architecture.md` | Chain, evidence classes, locked structure, diagram, alternatives, remaining unknowns |
| `docs/architecture/experiment-backlog.md` | Requested fields + sixteen dependency-ordered unrun entries |
| `scripts/check_clean_sheet_architecture.py` | Stdlib checker for both |

Both markdown files contain **the same** first fenced `json` block, copied from a fresh checker `--json` run (pretty-printed, script key order). Completes cross-document machine check without two schemas.

### Deliverable structure (`docs/architecture/clean-sheet-architecture.md`)

Use these **level-2 headings in this exact order**. Compact tables + one Mermaid fence + short captions. Every occupancy number is `OBSERVED` or `DERIVED` (citation). Every TASK-05 cell is `MEASURED` (citation). Every remaining-alternative usefulness is `HYPOTHESIS`. SKU limits stay `UNKNOWN`. Do not leave `TBD`. The word `UNKNOWN` must appear in Deferred vision.

Title: `# Qwen3.8-27B clean-sheet architecture (TASK-20)` (not `TASK-20` alone).

1. **Authority** — this dossier, plan.md, TASK-01–19 documents via citation table, config, checker; evidence labels; in-scope (language+MTP synthesis + alternative catalogue + unrun backlog) vs deferred (vision encoder; freeze; selected winners; executed experiments). State that the document is a **synthesis**, not a selected runtime and not measurements.
2. **Synthesis convention** — the four canonical sentences below plus the synthesis sentence; chain vs ledger execution order; no freeze.
3. **Architecture chain and evidence classes** — twelve chain steps from plan.md; five evidence classes; 19-row TASK citation table. Completes ledger checkbox 1.
4. **Locked structure** — occupancy, node/stage/recipe/mapping counts, MAC identities, what is locked vs unselected. No new operators.
5. **Architecture diagram and alternatives** — one Mermaid flowchart; fourteen alternative families; 125 catalogued / 0 selected. Completes ledger checkbox 2. Contains the Mermaid fence.
6. **Remaining unknowns** — twelve `oq_*` ids, all still open; TASK-20 catalogue flag true; freeze false.
7. **Deferred vision** — residual-stream interface only; encoder UNKNOWN.
8. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Synthesis convention**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> This document is a Phase 1 synthesis of TASK-01 through TASK-19, not a selected runtime and not measurements.

> Unselected alternatives remain alternatives; listing them is not selecting a winner.

> The experiment backlog is dependency-ordered and unrun; no implementation benchmarks or NLL are collected in this study task.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_synthesis` = sentence 2; `canonical_sentence_alternatives` = sentence 3; `canonical_sentence_unrun` = sentence 4.

Immediately after those four, include this **required synthesis sentence verbatim**:

> The required architecture chain and evidence classes are synthesized; an architecture diagram and alternatives are included; dependency-ordered experiment entries list all requested fields; no winners are selected and no experiments are run in this study task.

JSON: `synthesis_question_sentence` = that sentence.

Bullets required under that heading:

- The twelve `chain_ids` follow plan.md’s central path, not ledger execution order. TASK-16 ran beside TASK-01; TASK-11 preceded TASK-15. The diagram is the design path (`chain_follows_plan_path` true; `chain_is_not_ledger_order` true).
- Five evidence classes are complete: `DERIVED`, `OBSERVED`, `MEASURED`, `HYPOTHESIS`, `UNKNOWN` (`n_evidence_classes` 5), matching plan.md order.
- Fan-out ≠ must-store still holds; a synthesized box is not a CUDA allocation and not a selected kernel.
- TASK-20 catalogues remaining alternatives and unknowns; it does not close TASK-07–19 open questions (`n_remaining_open_questions_closed` 0).
- `FROZEN_FOR_COMPARATIVE_REVIEW` is TASK-21 (`frozen_for_comparative_review` false).
- No MEASURED NLL or tok/s in this document (`experiments_run` false; `nll_measured_here` false; `toks_measured_here` false).
- Tok/s is not a quality axis (`toks_is_not_quality_axis` true). Reconstruction is not quality (`reconstruction_is_not_quality` true).
- Microbenchmarks cannot pass a mapping (`microbenchmark_cannot_pass_mapping` true). End-to-end is required alongside them (`e2e_required_alongside_microbenchmarks` true).
- Q4_K_M is a future black-box Pareto reference, not a requirement (`gguf_is_pareto_reference` true; `gguf_is_not_a_requirement` true).
- Primary coverage includes MTP (135 instances). Language-only is secondary.
- `example_T_values` `[1, 4096]` remain illustration horizons, not a selected prompt matrix.

### Architecture chain (lock)

JSON array `chain_ids` in this exact order (12 ids). JSON `n_chain_steps` = 12. JSON object `chain_source_task_groups` keyed by `chain_ids` with array values in the table below (citation completeness; checker asserts every listed TASK id is in `synthesized_task_ids` except `TASK-20` which is not synthesized-as-source). Do **not** put `TASK-20` inside `chain_source_task_groups`; the backlog is this increment’s output, cited in prose as the `experimental_validation` leaf. JSON `chain_source_task_groups` values:

- `mathematics`: `["TASK-01","TASK-02"]`
- `logical_dataflow`: `["TASK-03"]`
- `lifetime_and_state`: `["TASK-04"]`
- `numerical_sensitivity`: `["TASK-07"]`
- `precision_strategy`: `["TASK-07"]`
- `custom_quantization`: `["TASK-08"]`
- `custom_physical_layouts`: `["TASK-15"]`
- `offline_model_compiler`: `["TASK-09","TASK-10"]`
- `semantic_graph`: `["TASK-11","TASK-12"]`
- `decode_prefill_plans`: `["TASK-13","TASK-14"]`
- `cuda_mappings`: `["TASK-16","TASK-17"]`
- `experimental_validation`: `["TASK-18","TASK-19"]`

| id | Plan.md step | Source tasks (cite, do not rewrite) |
| --- | --- | --- |
| `mathematics` | Qwen3.8 mathematics | TASK-01, TASK-02 |
| `logical_dataflow` | logical dataflow | TASK-03 |
| `lifetime_and_state` | tensor lifetime and state | TASK-04 |
| `numerical_sensitivity` | numerical sensitivity | TASK-07 |
| `precision_strategy` | precision strategy | TASK-07 |
| `custom_quantization` | custom quantization | TASK-08 |
| `custom_physical_layouts` | custom physical layouts | TASK-15 |
| `offline_model_compiler` | offline model compiler | TASK-09, TASK-10 |
| `semantic_graph` | Qwen-specific semantic graph | TASK-11, TASK-12 |
| `decode_prefill_plans` | decode/prefill execution plans | TASK-13, TASK-14 |
| `cuda_mappings` | CUDA mappings | TASK-16, TASK-17 |
| `experimental_validation` | future experimental validation | TASK-18, TASK-19 (backlog is this increment) |

JSON array `synthesized_task_ids` in this exact order (19 ids): `TASK-01` … `TASK-19`. JSON `n_synthesized_tasks` = 19.

JSON array `satellite_task_ids` in this exact order (5 ids): `TASK-05`, `TASK-06`, `TASK-09`, `TASK-12`, `TASK-16`. These attach to the chain (distributions → quantization; work/traffic → all later bounds; format → compiler/layouts; fusion → graph/schedules; hardware model → CUDA) and are also members of `synthesized_task_ids`. JSON `n_satellite_tasks` = 5.

Prose required: TASK-05 MEASURED distributions shape TASK-08 candidate sets and do not pick winners. TASK-06 MAC/byte identities are lower bounds, not measured traffic. TASK-09 artifact approaches remain unselected. TASK-12 fusion hypotheses remain HYPOTHESIS. TASK-16 SKU symbols remain UNKNOWN.

Under **Architecture chain and evidence classes**, include a 19-row table with columns `task`, `deliverable`, `primary_evidence_classes`, `open_question_status`. Locked `primary_evidence_classes` strings (substring match via the class tokens; exact JSON object `task_evidence_classes` keyed by `synthesized_task_ids` with array values):

| task | primary_evidence_classes |
| --- | --- |
| TASK-01 | OBSERVED, DERIVED |
| TASK-02 | DERIVED |
| TASK-03 | DERIVED |
| TASK-04 | DERIVED |
| TASK-05 | MEASURED |
| TASK-06 | DERIVED, HYPOTHESIS |
| TASK-07 | DERIVED, HYPOTHESIS |
| TASK-08 | OBSERVED, MEASURED, DERIVED, HYPOTHESIS |
| TASK-09 | DERIVED, HYPOTHESIS |
| TASK-10 | DERIVED, HYPOTHESIS |
| TASK-11 | DERIVED |
| TASK-12 | DERIVED, HYPOTHESIS |
| TASK-13 | DERIVED, HYPOTHESIS |
| TASK-14 | DERIVED, HYPOTHESIS |
| TASK-15 | DERIVED, HYPOTHESIS |
| TASK-16 | DERIVED, UNKNOWN |
| TASK-17 | DERIVED, HYPOTHESIS, UNKNOWN |
| TASK-18 | DERIVED, HYPOTHESIS |
| TASK-19 | DERIVED, HYPOTHESIS, UNKNOWN |

JSON `task_open_question_closed` object keyed by `synthesized_task_ids`: true for TASK-01, TASK-02 (language math), TASK-03, TASK-04, TASK-06, TASK-11 (boundaries defined), TASK-12 (fusion remains hypothesis — the *classification* question closed), TASK-13 (boundary traffic closed); false for TASK-05 (policy winners deferred — treat as false because quantization winners remain), TASK-07, TASK-08, TASK-09, TASK-10, TASK-14, TASK-15, TASK-16, TASK-17, TASK-18, TASK-19.

To avoid a planner contradiction: TASK-05’s open question was “quantization policy winners remain for TASK-08”; TASK-08 left Pareto open. TASK-05 measurements themselves are complete. JSON `task_open_question_closed` = true for TASK-01, TASK-02, TASK-03, TASK-04, TASK-05, TASK-06, TASK-11, TASK-12, TASK-13; false for TASK-07, TASK-08, TASK-09, TASK-10, TASK-14, TASK-15, TASK-16, TASK-17, TASK-18, TASK-19. JSON `n_tasks_open_question_closed` = 9. JSON `n_tasks_open_question_open` = 10.

Vision deferred is not a closed language-math question; TASK-02 language math is closed.

### Evidence classes (lock)

JSON array `evidence_class_ids` in this exact order (5 ids). JSON `n_evidence_classes` = 5: `DERIVED`, `OBSERVED`, `MEASURED`, `HYPOTHESIS`, `UNKNOWN`.

Prose required: TASK-20 itself adds no new MEASURED points. TASK-05 remains the only payload-MEASURED source. Hypotheses remain hypotheses. UNKNOWN is sitting SKU, vision encoder, and unselected corpora/hardware/prompt/repro values.

### Locked structure (lock)

Cite, do not retabulate as new studies. JSON occupancy:

- `hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, `n_decoder_layers` 64, `n_linear_layers` 48, `n_full_layers` 16, `n_mtp_blocks` 1
- `full_attention_indices` `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `dtype` `"bfloat16"`, `mamba_ssm_dtype` `"float32"`, `model_T_max` 262144
- `n_language_mtp_parameters` 27320697856, `weight_bytes_language_mtp_excl_vision` 54641395712, `weight_bytes_unique_non_embed` 52098598912
- `mac_C_complete` 27433238528, `mac_A_complete` 208896
- `mac_decode_complete_T1` = `mac_prefill_complete_T1` = 27433447424
- `mac_decode_complete_T4096` 28288876544, `mac_prefill_complete_T4096` 114119319486464
- `example_T_values` `[1, 4096]`
- `N_w` 32, `N_bank` 32
- `n_node_types` 6, `n_node_instances_complete` 135 (`n_embed_instances` 2, `n_gated_attn_instances` 17, `n_gated_delta_net_instances` 48, `n_mlp_instances` 65, `n_lm_head_instances` 2, `n_mtp_mix_instances` 1)
- `n_stage_kinds` 9, `n_candidate_recipes` 22, `n_policy_families` 16, `n_compiler_stages` 6, `n_compiler_profiles` 3, `n_mappings` 18, `n_fusion_hypotheses` 22, `n_consumer_sequences` 7, `n_artifact_approaches` 4, `n_orderings` 14, `n_tile_families` 9, `n_parallel_decompositions` 9, `n_sku_unknown_symbols` 17, `n_sensitive_ops` 20

JSON arrays duplicated as checker constants (do not import other scripts); every id must appear as a substring in architecture.md:

- `node_type_ids`: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`
- `stage_kind_ids`: `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`
- `candidate_recipe_ids`: `keep_source`, `narrow_bf16`, `fp8_tensor`, `i8_tensor`, `i8_row`, `i8_row_asym`, `i8_g32`, `i6_row`, `i4_row`, `i4_col`, `i4_g32`, `i4_g64`, `i4_g128`, `i4_g128_p99`, `i4_g128_rms`, `i4_g128_extract`, `i4_g32_mixed`, `i4_row_extract`, `i4_clip`, `i3_g32`, `i3_g32_extract`, `i2_g32_extract`
- `policy_families`: `norm_gamma`, `gdn_time_param`, `gdn_gate_proj`, `conv1d`, `linear_large_proj`, `attn_qkv`, `attn_out`, `mlp_up_gate`, `mlp_down`, `embed_table`, `lm_head`, `mtp_fc`, `vision_deferred`, `state_kv`, `state_c`, `state_s`
- `compiler_stage_ids`: `validation`, `analysis`, `quantization`, `packing`, `metadata`, `integrity`
- `compiler_profile_ids`: `quality`, `balanced`, `compression`
- `control_profile_id`: `"control"`
- `artifact_approach_ids`: `portable_only`, `backend_specialized_only`, `portable_plus_specialized_views`, `manifest_plus_backend_blobs`
- `consumer_sequence_ids`: `seq_gemm_codes_then_scales`, `seq_gemm_interleaved_group`, `seq_gather_row`, `seq_lm_head_full`, `seq_outlier_extra`, `seq_state_s_dense`, `seq_specialized_tile`
- `integrity_algorithm_candidates`: `none`, `checksum`
- `fusion_hypothesis_ids`: `fuse_gated_attn_internals`, `fuse_gated_delta_net_internals`, `fuse_mlp_internals`, `fuse_lm_head_internals`, `fuse_mtp_mix_internals`, `split_g`, `split_z`, `split_h_tilde`, `split_k_rope`, `split_v_full`, `split_qkv`, `split_h_post`, `split_swiglu`, `split_h_final`, `split_mtp_cat`, `fuse_across_identity_e_h0`, `fuse_across_residual_h`, `fuse_across_residual_h_mid`, `fuse_across_fanout_h64`, `fuse_across_embed_e_next`, `fuse_across_mtp_u_to_block`, `fuse_across_h_mtp_to_logits`
- `hoist_hypothesis_ids`: `hoist_embed_next`, `overlap_fanout_h64`, `reuse_E`, `reuse_W_lm`, `reuse_h64`
- `representation_hypothesis_ids`: `distinct_prefill_gemm_view`, `distinct_decode_gemv_view`, `shared_view_both_modes`, `dual_view_binding`
- `mode_hypothesis_ids`: `token_serial_prefill`, `inference_last_logits`
- `ordering_ids`: `ord_gemm_out_major`, `ord_gemm_in_major`, `ord_embed_vocab_major`, `ord_embed_hidden_major`, `ord_conv_channel_tap`, `ord_conv_tap_channel`, `ord_vec_width`, `ord_kv_n_t_dh`, `ord_kv_n_dh_t`, `ord_kv_t_n_dh`, `ord_c_delay_channel`, `ord_c_channel_delay`, `ord_s_n_dk_dv`, `ord_s_n_dv_dk`
- `tile_family_ids`: `tile_none`, `tile_2d_mn`, `tile_1d_row`, `tile_conv_channel`, `tile_kv_t`, `tile_kv_dh`, `tile_s_head`, `tile_s_block`, `tile_mma_shaped`
- `parallel_decomposition_ids`: `par_gemm_d_out`, `par_gemm_d_in`, `par_gemm_T`, `par_attn_head`, `par_attn_T`, `par_gdn_head`, `par_conv_channel`, `par_kv_head`, `par_embed_row`
- `conversion_hypothesis_ids`: `conv_compile_pack`, `conv_load_repack`, `conv_inkernel_unpack`, `conv_dual_view`
- `mapping_ids`: `map_embed_thread_element`, `map_embed_warp_row`, `map_embed_cta_vector`, `map_attn_cta_head`, `map_attn_warp_t`, `map_attn_cta_splitk`, `map_gdn_cta_head`, `map_gdn_warp_recurrent`, `map_gdn_cta_chunk`, `map_mlp_cta_dout`, `map_mlp_cta_splitk`, `map_mlp_grid_T`, `map_lm_cta_vocab`, `map_lm_cta_splitk`, `map_lm_warp_gemv`, `map_mtp_cta_fc`, `map_mtp_cta_fused`, `map_mtp_split_norm_gemm`
- `sku_unknown_symbols`: `N_SM`, `W_max`, `S_reg`, `C_smem`, `T_max`, `B_max`, `N_bar`, `N_sched`, `G_reg`, `G_smem`, `Beta_HBM`, `Pi_FMA`, `Pi_TC`, `L_issue`, `async_copy_cap`, `mma_shapes`, `cluster_cap`
- `sensitive_ops`: `param_bf16`, `residual_stream`, `live_across_gates`, `silu_sigmoid`, `rms_hidden`, `rms_head`, `l2_gdn`, `softmax_over_T`, `attn_av_over_T`, `gemm_k5120`, `gemm_k17408`, `gemm_lm_head`, `gdn_S_recurrent`, `gdn_inner_d128`, `gdn_alpha_beta`, `rope_phase`, `state_kv_bf16`, `state_c_bf16`, `s_below_f32`, `conv_fir`
- `quality_high_ids`: `q_norm_gamma`, `q_gdn_time`, `q_gdn_gate`, `q_attn_out`, `q_mlp_down`, `q_state_kv`, `q_state_s`, `q_int2_mass`
- `bottleneck_labels`: `weight_memory`, `vocab_memory`, `state_memory`, `kv_memory`, `quadratic_attn`, `compute`
- `evaluation_criterion_ids`: `crit_occupancy`, `crit_latency_hiding`, `crit_wave_quant`, `crit_intensity_roofline`, `crit_sync_class`, `crit_pipeline_mix`, `crit_fusion_delta`
- `pareto_axis_ids`: `axis_quality`, `axis_compression`, `axis_reference_q4km`
- `corpus_class_ids`: `corpus_calibration`, `corpus_eval_nll`, `corpus_prompt`, `corpus_capability`
- `reconstruction_diagnostic_ids`: `r_param_mse`, `r_param_maxabs`, `r_param_cosine`, `r_clip_frac`, `r_act_residual`, `r_act_logits`, `r_state_kv`, `r_state_s`
- `teacher_forced_metric_ids`: `tf_nll_language`, `tf_nll_mtp`, `tf_nll_complete`, `tf_delta_vs_control`, `tf_kl_vs_control`
- `decode_metric_ids`: `dec_toks`, `dec_step_ms`, `dec_p50_step_ms`, `dec_p99_step_ms`
- `prefill_metric_ids`: `pre_toks`, `pre_ms`, `pre_ttft_ms`, `pre_mac_cite`
- `kernel_metric_ids`: `k_node_ms`, `k_mapping_ms`, `k_leaf_union_ms`, `k_graph_envelope_ms`, `k_occupancy_achieved`
- `memory_metric_ids`: `mem_hbm_gbps`, `mem_weight_bytes`, `mem_state_bytes`, `mem_act_bytes`, `mem_working_set`
- `e2e_metric_ids`: `e2e_latency_ms`, `e2e_output_toks`, `e2e_ttft_ms`

Locked what-is-selected flags (all false except catalogue/methodology flags):

- `pareto_frontier_selected` false, `compiler_profile_selected` false, `artifact_boundary_selected` false, `ideal_byte_sequence_selected` false, `integrity_algorithm_selected` false, `fusion_winner_selected` false, `decode_prefill_distinct_views_selected` false, `layout_winner_selected` false, `ordering_selected` false, `tile_size_selected` false, `mapping_winner_selected` false, `hardware_selected` false, `prompt_matrix_selected` false, `reproducibility_protocol_selected` false, `calibration_corpus_selected` false, `eval_corpus_selected` false, `acceptance_frontier_selected` false, `sku_table_filled` false, `hypothesis_survival_selected` false, `n_mappings_selected` 0, `n_fusion_hypotheses_selected` 0, `n_alternatives_selected` 0

Prose required: algebraic equivalents in TASK-02 remain the same real map. Chunkwise GDN is not zero $S$ traffic. At $T=1$, MAC prefill = decode; C/S physical reads still differ. Unique weight bytes are a TASK-06 lower bound, not measured HBM traffic.

### Alternative families (lock; heading 5)

JSON array `alternative_family_ids` in this exact order (14 ids). JSON `n_alternative_families` = 14. Parallel `alternative_family_counts` (ints summing to 125). JSON `n_catalogued_alternatives` = 125. JSON `n_alternatives_selected` = 0. JSON `alternative_usefulness_label` exactly `HYPOTHESIS`.

| id | count | source list |
| --- | --- | --- |
| `alt_recipes` | 22 | `candidate_recipe_ids` |
| `alt_profiles` | 3 | `compiler_profile_ids` |
| `alt_artifact_approaches` | 4 | `artifact_approach_ids` |
| `alt_consumer_sequences` | 7 | `consumer_sequence_ids` |
| `alt_integrity` | 2 | `integrity_algorithm_candidates` |
| `alt_representation` | 4 | `representation_hypothesis_ids` |
| `alt_mode` | 2 | `mode_hypothesis_ids` |
| `alt_fusion` | 22 | `fusion_hypothesis_ids` |
| `alt_hoist` | 5 | `hoist_hypothesis_ids` |
| `alt_orderings` | 14 | `ordering_ids` |
| `alt_tiles` | 9 | `tile_family_ids` |
| `alt_decompositions` | 9 | `parallel_decomposition_ids` |
| `alt_conversions` | 4 | `conversion_hypothesis_ids` |
| `alt_mappings` | 18 | `mapping_ids` |

Control `keep_source` / profile `control` is the identity compile, not a fourth intent axis and not a quality winner. Q4_K_M is a Pareto **reference**, not an `alt_*` member.

JSON `alternative_family_counts` = `[22,3,4,7,2,4,2,22,5,14,9,9,4,18]`.

Exactly **one** fenced `mermaid` block, under this heading. Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 135 instances, 22 recipes, or 18 mapping boxes.

Required node IDs **inside** the mermaid fence (substring match): `MATH`, `DATAFLOW`, `LIFETIME`, `SENS`, `QUANT`, `LAYOUT`, `COMPILER`, `GRAPH`, `DECODE`, `PREFILL`, `CUDA`, `BACKLOG`, `ALTS`, `OPENQ`.

JSON `n_diagrams` is 1. `diagram_ids` is `["math","dataflow","lifetime","sens","quant","layout","compiler","graph","decode","prefill","cuda","backlog","alts","openq"]`.

Suggested topology (implementer may rearrange edges; IDs are mandatory): MATH → DATAFLOW → LIFETIME → SENS → QUANT → LAYOUT → COMPILER → GRAPH → DECODE and PREFILL → CUDA → BACKLOG; ALTS → QUANT, LAYOUT, CUDA, GRAPH; BACKLOG → OPENQ. Caption must contain `HYPOTHESIS` and `unselected` (or `not a selected winner`). Caption must state that OPENQ is the remaining TASK-07–19 unknowns, not a freeze, and that the chain is the plan.md design path.

### Remaining unknowns (lock)

JSON array `remaining_open_question_ids` in this exact order (12 ids). JSON `n_remaining_open_questions` = 12. JSON `n_remaining_open_questions_closed` = 0. JSON `ledger_open_question_alternatives_catalogued` true. JSON `frozen_for_comparative_review` false.

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

Prose required: cataloguing these ids is the TASK-20 answer to “candidate alternatives and unknowns remaining.” It is not a selected architecture. TASK-21 may preserve them as true unknowns after consistency review.

### Deliverable structure (`docs/architecture/experiment-backlog.md`)

Use these **level-2 headings in this exact order**. Compact tables. No Mermaid fence in this file (`n_diagrams_backlog` 0; the architecture diagram lives in `clean-sheet-architecture.md`). Do not leave `TBD`. The word `UNKNOWN` must appear in Deferred vision.

Title: `# Qwen3.8-27B experiment backlog (TASK-20)` (not `TASK-20` alone).

1. **Authority** — this dossier, architecture synthesis, TASK-18/19 methodologies, checker; in-scope (sixteen unrun entries with twelve fields) vs deferred (executed results; freeze; selected winners).
2. **Backlog convention** — the four backlog canonical sentences below plus a pointer to `synthesis_question_sentence` (must appear verbatim in this file as well); dependency-order rule; quality vs performance split.
3. **Requested fields** — twelve field ids and meanings. Completes ledger checkbox 3 (field schema).
4. **Dependency-ordered experiment entries** — sixteen rows/tables in locked id order. Completes ledger checkbox 3 (entries).
5. **Deferred vision** — no vision-encoder experiments in this backlog.
6. **Machine-checkable summary JSON** — the **same** fenced `json` object as `clean-sheet-architecture.md`.

Immediately under **Backlog convention**, include these **four canonical sentences verbatim**, in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Experiment entries in this document are a backlog, not results.

> Dependency order is required; an entry may not run before every id in depends_on has a measured status.

> Microbenchmarks cannot pass a CUDA mapping; reconstruction is not model-level quality; Q4_K_M is a future black-box Pareto reference, not a requirement.

JSON: `canonical_sentence_logical` (same as architecture); `canonical_sentence_backlog` = sentence 2; `canonical_sentence_depends` = sentence 3; `canonical_sentence_eval_split` = sentence 4.

Also include `synthesis_question_sentence` verbatim in this file (same string as architecture).

Bullets required:

- Sixteen experiments, twelve requested fields, listed order is a valid topological order (`experiment_order_is_topological` true).
- Every `status` is `unrun` (`n_experiments_run` 0; `experiments_run` false).
- Quality experiments use TASK-18 metrics; performance experiments use TASK-19 metrics; tok/s is not Pareto $Y$.
- `exp_mapping_micro` cannot declare `mapping_winner_selected`; only a later measured `exp_mapping_e2e` may, and this study does not run it.
- Hardware, corpora, prompt matrix, and reproducibility remain unselected blocking unknowns, not extra experiments that select them.
- Control `keep_source` is the identity compile, not a quality winner.

### Requested fields (lock)

JSON array `experiment_field_ids` in this exact order (12 ids). JSON `n_experiment_fields` = 12.

| ID | JSON type | Meaning |
| --- | --- | --- |
| `id` | string | Stable experiment id from `experiment_ids` |
| `title` | string | Short name |
| `depends_on` | array of ids | Must be a subset of `experiment_ids`; empty allowed |
| `source_tasks` | array of `TASK-NN` | Evidence this experiment consumes |
| `chain_step` | string | Exactly one of `chain_ids` |
| `hypothesis` | string | HYPOTHESIS under test; listing is not proof |
| `alternatives` | array of ids | Unselected candidates compared; empty only where locked below |
| `eval_family` | string | Exactly one of `experiment_eval_family_ids` |
| `metric_ids` | array of ids | TASK-18 and/or TASK-19 (or TASK-09 comparison) metric/diagnostic ids; empty only where locked below |
| `disconfirming_outcome` | string | Result that rejects the hypothesis or the methodology |
| `blocking_unknowns` | array of `oq_*` | Subset of `remaining_open_question_ids`; empty allowed |
| `status` | string | Must equal `unrun` in this study |

JSON array `experiment_eval_family_ids` in this exact order (4 ids): `infrastructure`, `quality`, `format`, `performance`. JSON `n_experiment_eval_families` = 4.

JSON `experiment_status_unrun` exactly `"unrun"`.

Empty `metric_ids` allowed only for `exp_sku_fill`, `exp_identity_harness`, `exp_integrity_algorithm` (`n_empty_metric_experiments` 3). Empty `alternatives` allowed only for `exp_identity_harness` (`n_empty_alternative_experiments` 1). Empty `depends_on` allowed only for `exp_sku_fill` and `exp_identity_harness`. Empty `blocking_unknowns` allowed.

### Experiment entries (lock; sixteen; this order is topological)

JSON array `experiment_ids` in this exact order (16 ids). JSON `n_experiments` = 16. JSON array `experiments` of 16 objects, each with **all twelve** `experiment_field_ids` keys in that key order. JSON `n_experiments_run` = 0. JSON object `experiment_selected` keyed by `experiment_ids` with all `false`.

Checker asserts: unique ids; every `depends_on` member is in `experiment_ids`; no cycles; for each entry, every dependency appears **earlier** in `experiment_ids` (`experiment_order_is_topological` true); every `chain_step` is in `chain_ids`; every `eval_family` is in `experiment_eval_family_ids`; every `status` is `unrun`; `source_tasks` members match `TASK-0[1-9]` or `TASK-1[0-9]`.

Locked rows (implementer must copy title, hypothesis, disconfirming_outcome, arrays, eval_family, chain_step **verbatim** into JSON and as table cells / prose ids):

1. **`exp_sku_fill`**
   - title: `Fill sitting SKU table`
   - depends_on: `[]`
   - source_tasks: `["TASK-16","TASK-19"]`
   - chain_step: `cuda_mappings`
   - hypothesis: `Sitting-device capacity and peak symbols can be filled from a declared sitting identity; a datasheet is not a measurement.`
   - alternatives: the 17 `sku_unknown_symbols`
   - eval_family: `infrastructure`
   - metric_ids: `[]`
   - disconfirming_outcome: `Filling sku_unknown_symbols from an unnamed datasheet or treating datasheet peaks as mem_hbm_gbps.`
   - blocking_unknowns: `["oq_hardware_protocol"]`
   - status: `unrun`

2. **`exp_identity_harness`**
   - title: `Identity coverage and NLL harness`
   - depends_on: `[]`
   - source_tasks: `["TASK-18","TASK-19"]`
   - chain_step: `experimental_validation`
   - hypothesis: `Matching-identity capture is implementable without mixing decode-only, prefill, and complete-request windows.`
   - alternatives: `[]`
   - eval_family: `infrastructure`
   - metric_ids: `[]`
   - disconfirming_outcome: `Mixed (T+N)/t reported as decode-only, or missing coverage treated as zero excess.`
   - blocking_unknowns: `["oq_hardware_protocol","oq_eval_corpora"]`
   - status: `unrun`

3. **`exp_control_keep_source`**
   - title: `Control keep_source baseline`
   - depends_on: `["exp_identity_harness"]`
   - source_tasks: `["TASK-08","TASK-10","TASK-18","TASK-19"]`
   - chain_step: `experimental_validation`
   - hypothesis: `keep_source control compile reproduces the TASK-02 map at conceptual BF16 parameters and F32 S; it is an identity control, not a quality winner.`
   - alternatives: `["control"]`
   - eval_family: `quality`
   - metric_ids: `["tf_nll_complete","tf_nll_language","tf_nll_mtp","e2e_latency_ms","dec_toks","pre_toks"]`
   - disconfirming_outcome: `Undefined control NLL, or ranking keep_source as a quality winner.`
   - blocking_unknowns: `["oq_eval_corpora","oq_hardware_protocol"]`
   - status: `unrun`

4. **`exp_reconstruction_screen`**
   - title: `Reconstruction screens over candidate recipes`
   - depends_on: `["exp_control_keep_source"]`
   - source_tasks: `["TASK-08","TASK-18"]`
   - chain_step: `custom_quantization`
   - hypothesis: `Local reconstruction can fail catastrophic recipes and cannot place a Pareto quality point.`
   - alternatives: the 22 `candidate_recipe_ids`
   - eval_family: `quality`
   - metric_ids: the 8 `reconstruction_diagnostic_ids`
   - disconfirming_outcome: `Placing a Pareto point from r_* diagnostics, or treating reconstruction as NLL.`
   - blocking_unknowns: `[]`
   - status: `unrun`

5. **`exp_family_nll`**
   - title: `Teacher-forced family NLL`
   - depends_on: `["exp_reconstruction_screen"]`
   - source_tasks: `["TASK-07","TASK-08","TASK-18"]`
   - chain_step: `custom_quantization`
   - hypothesis: `Family-specific candidate sets, especially quality_high_ids families, change complete-map NLL versus keep_source.`
   - alternatives: the 16 `policy_families`
   - eval_family: `quality`
   - metric_ids: `["tf_nll_complete","tf_delta_vs_control","tf_kl_vs_control"]`
   - disconfirming_outcome: `Language-only NLL substituting for the complete map, or sampling used as NLL.`
   - blocking_unknowns: `["oq_eval_corpora"]`
   - status: `unrun`

6. **`exp_state_precision`**
   - title: `Persistent-state precision`
   - depends_on: `["exp_family_nll"]`
   - source_tasks: `["TASK-04","TASK-07","TASK-08","TASK-18"]`
   - chain_step: `precision_strategy`
   - hypothesis: `Narrowing conceptual F32 S or BF16 K,V,C changes NLL and/or state reconstruction.`
   - alternatives: `["state_kv","state_c","state_s","keep_source","narrow_bf16"]`
   - eval_family: `quality`
   - metric_ids: `["r_state_kv","r_state_s","tf_nll_complete"]`
   - disconfirming_outcome: `Treating chunkwise GDN as zero S traffic, or dropping MTP from the complete map.`
   - blocking_unknowns: `[]`
   - status: `unrun`

7. **`exp_risk_survival`**
   - title: `Numerical-risk hypothesis survival`
   - depends_on: `["exp_family_nll","exp_state_precision"]`
   - source_tasks: `["TASK-07","TASK-18"]`
   - chain_step: `numerical_sensitivity`
   - hypothesis: `Some of the twenty sensitive-op HYPOTHESIS severities survive model-level validation.`
   - alternatives: the 20 `sensitive_ops`
   - eval_family: `quality`
   - metric_ids: `["tf_nll_complete","r_act_residual","r_act_logits"]`
   - disconfirming_outcome: `Declaring hypothesis_survival_selected without complete-map NLL.`
   - blocking_unknowns: `["oq_eval_corpora"]`
   - status: `unrun`

8. **`exp_pareto_profiles`**
   - title: `Profile Pareto versus Q4_K_M reference`
   - depends_on: `["exp_risk_survival"]`
   - source_tasks: `["TASK-08","TASK-10","TASK-18"]`
   - chain_step: `custom_quantization`
   - hypothesis: `Legal quality, balanced, and compression recipe maps form a Pareto surface on quality and compression with Q4_K_M as a reference point, not a requirement.`
   - alternatives: `["quality","balanced","compression","control"]`
   - eval_family: `quality`
   - metric_ids: `["axis_quality","axis_compression","axis_reference_q4km","tf_nll_complete"]`
   - disconfirming_outcome: `Requiring Q4_K_M to be beaten, or using tok/s as Pareto Y.`
   - blocking_unknowns: `["oq_eval_corpora"]`
   - status: `unrun`

9. **`exp_artifact_boundary`**
   - title: `Portable versus backend-specialized artifact`
   - depends_on: `["exp_reconstruction_screen"]`
   - source_tasks: `["TASK-09","TASK-10"]`
   - chain_step: `offline_model_compiler`
   - hypothesis: `Artifact-boundary choice changes load/convert cost and consumer portability without changing the TASK-02 map.`
   - alternatives: the 4 `artifact_approach_ids`
   - eval_family: `format`
   - metric_ids: `["compile_once","load_convert","byte_sequence","tile_freedom","store_amplification","consumer_portability"]`
   - disconfirming_outcome: `Selecting artifact_boundary without a six-dimension comparison.`
   - blocking_unknowns: `[]`
   - status: `unrun`

10. **`exp_consumer_sequence`**
    - title: `Ideal consumer byte sequences`
    - depends_on: `["exp_artifact_boundary"]`
    - source_tasks: `["TASK-09","TASK-13","TASK-14"]`
    - chain_step: `custom_physical_layouts`
    - hypothesis: `One of seven consumer sequences is ideal per access class after a boundary is chosen.`
    - alternatives: the 7 `consumer_sequence_ids`
    - eval_family: `format`
    - metric_ids: `["mem_weight_bytes","k_node_ms"]`
    - disconfirming_outcome: `Ranking sequences by wall time without coverage, or selecting seq_specialized_tile extents here.`
    - blocking_unknowns: `["oq_hardware_protocol"]`
    - status: `unrun`

11. **`exp_integrity_algorithm`**
    - title: `Integrity algorithm none versus checksum`
    - depends_on: `["exp_artifact_boundary"]`
    - source_tasks: `["TASK-09","TASK-10"]`
    - chain_step: `offline_model_compiler`
    - hypothesis: `checksum versus none is an integrity and load-safety choice, not a quality or NLL decision.`
    - alternatives: `["none","checksum"]`
    - eval_family: `format`
    - metric_ids: `[]`
    - disconfirming_outcome: `Treating checksum as a quality winner.`
    - blocking_unknowns: `[]`
    - status: `unrun`

12. **`exp_decode_prefill_views`**
    - title: `Decode/prefill representation views`
    - depends_on: `["exp_consumer_sequence"]`
    - source_tasks: `["TASK-13","TASK-14"]`
    - chain_step: `decode_prefill_plans`
    - hypothesis: `Distinct GEMV/GEMM views, a shared view, or dual-view binding changes traffic versus a single sequence; mode hypotheses remain unselected until measured.`
    - alternatives: the 4 `representation_hypothesis_ids` plus the 2 `mode_hypothesis_ids` (6 ids, representation first)
    - eval_family: `performance`
    - metric_ids: `["pre_toks","dec_toks","mem_act_bytes","e2e_latency_ms"]`
    - disconfirming_outcome: `Reporting T=1 prefill as decode-only, or selecting views without end-to-end measurement.`
    - blocking_unknowns: `["oq_hardware_protocol"]`
    - status: `unrun`

13. **`exp_fusion_hoist`**
    - title: `Fusion and hoist hypotheses`
    - depends_on: `["exp_control_keep_source"]`
    - source_tasks: `["TASK-12","TASK-13","TASK-14","TASK-16","TASK-17"]`
    - chain_step: `semantic_graph`
    - hypothesis: `Some of twenty-two fusion and five hoist hypotheses improve end-to-end latency without violating TASK-16 occupancy algebra.`
    - alternatives: the 22 `fusion_hypothesis_ids` plus the 5 `hoist_hypothesis_ids` (fusion first)
    - eval_family: `performance`
    - metric_ids: `["e2e_latency_ms","k_node_ms","k_occupancy_achieved","mem_working_set"]`
    - disconfirming_outcome: `Selecting fusion from microbenchmarks only, or occupancy-unaware fusion.`
    - blocking_unknowns: `["oq_hardware_protocol"]`
    - status: `unrun`

14. **`exp_layout_decomp`**
    - title: `Parallel decompositions justifying layouts`
    - depends_on: `["exp_decode_prefill_views","exp_sku_fill"]`
    - source_tasks: `["TASK-15","TASK-17"]`
    - chain_step: `custom_physical_layouts`
    - hypothesis: `A planned parallel decomposition justifies a candidate ordering and tile family per layout object.`
    - alternatives: the 9 `parallel_decomposition_ids` plus the 14 `ordering_ids` plus the 9 `tile_family_ids` plus the 4 `conversion_hypothesis_ids` (that order)
    - eval_family: `performance`
    - metric_ids: `["k_mapping_ms","mem_hbm_gbps","k_occupancy_achieved"]`
    - disconfirming_outcome: `Claiming layout optimality before CUDA analysis, or selecting tiles without a decomposition.`
    - blocking_unknowns: `["oq_sku_limits"]`
    - status: `unrun`

15. **`exp_mapping_micro`**
    - title: `CUDA mapping microbenchmarks`
    - depends_on: `["exp_layout_decomp","exp_fusion_hoist","exp_sku_fill"]`
    - source_tasks: `["TASK-17","TASK-19"]`
    - chain_step: `cuda_mappings`
    - hypothesis: `Eighteen ownership and reduction mappings differ on kernel and memory component metrics; none may be declared winner from this experiment alone.`
    - alternatives: the 18 `mapping_ids`
    - eval_family: `performance`
    - metric_ids: the 5 `kernel_metric_ids` plus the 5 `memory_metric_ids` (kernel first)
    - disconfirming_outcome: `mapping_winner_selected from microbenchmarks, or ranking graph envelopes as leaf kernel time.`
    - blocking_unknowns: `["oq_hardware_protocol"]`
    - status: `unrun`

16. **`exp_mapping_e2e`**
    - title: `CUDA mapping end-to-end`
    - depends_on: `["exp_mapping_micro","exp_control_keep_source"]`
    - source_tasks: `["TASK-17","TASK-19","TASK-18"]`
    - chain_step: `experimental_validation`
    - hypothesis: `Identity-matched end-to-end latency plus decode-only and prefill metrics, with complete-map NLL as a quality guard, can fail mappings; tok/s is not the quality axis.`
    - alternatives: the 18 `mapping_ids`
    - eval_family: `performance`
    - metric_ids: `["e2e_latency_ms","e2e_output_toks","dec_toks","pre_toks","tf_nll_complete"]`
    - disconfirming_outcome: `Mixed (T+N)/t as decode-only, skipping e2e, or using tok/s as Pareto Y.`
    - blocking_unknowns: `["oq_hardware_protocol","oq_eval_corpora"]`
    - status: `unrun`

JSON `comparison_dimension_ids` = `["compile_once","load_convert","byte_sequence","tile_freedom","store_amplification","consumer_portability"]` (TASK-09 citation for `exp_artifact_boundary` metrics).

Backlog prose required: do not add a seventeenth experiment that selects hardware or corpora. Those stay `blocking_unknowns`. Do not run any entry. Do not promote `exp_mapping_micro` to a winner.

### Deferred vision (both documents)

Vision-encoder internals remain `UNKNOWN` and out of the primary map. Residual-stream interface only. No vision experiments in the backlog. JSON `vision_eval_deferred` true. The word `UNKNOWN` must appear in each document’s Deferred vision section.

### Additional locked booleans and integers

- `experiments_run` false
- `n_experiments_run` 0
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
- `winner_selected_without_measurements` false
- `frozen_for_comparative_review` false
- `ledger_open_question_alternatives_catalogued` true
- `n_remaining_open_questions_closed` 0
- `reconstruction_is_not_quality` true
- `toks_is_not_quality_axis` true
- `microbenchmark_cannot_pass_mapping` true
- `e2e_required_alongside_microbenchmarks` true
- `keep_source_is_not_quality_winner` true
- `control_profile_is_keep_source` true
- `decode_prefill_share_graph` true
- `decode_prefill_share_artifact` true
- `decode_prefill_share_metric_identity` false
- `chain_follows_plan_path` true
- `chain_is_not_ledger_order` true
- `experiment_order_is_topological` true
- `vision_eval_deferred` true
- `safetensors_is_source_not_runtime` true
- `architecture_headings` exactly the 8 names below
- `backlog_headings` exactly the 6 names below
- `n_architecture_headings` 8
- `n_backlog_headings` 6
- `n_headings` 14 (8+6, documented as split)

`architecture_headings` JSON array:

`Authority`, `Synthesis convention`, `Architecture chain and evidence classes`, `Locked structure`, `Architecture diagram and alternatives`, `Remaining unknowns`, `Deferred vision`, `Machine-checkable summary JSON`.

`backlog_headings` JSON array:

`Authority`, `Backlog convention`, `Requested fields`, `Dependency-ordered experiment entries`, `Deferred vision`, `Machine-checkable summary JSON`.

### JSON schema (both documents, heading last)

Pretty-printed object, key order as emitted by the checker. Required keys (exact names; implementer may add only if this dossier is amended):

`authority`, `deliverable_architecture`, `deliverable_backlog`, `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `model_T_max`, `dtype`, `mamba_ssm_dtype`, `n_language_mtp_parameters`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `mac_C_complete`, `mac_A_complete`, `mac_decode_complete_T1`, `mac_prefill_complete_T1`, `mac_decode_complete_T4096`, `mac_prefill_complete_T4096`, `example_T_values`, `N_w`, `N_bank`, `n_node_types`, `node_type_ids`, `n_embed_instances`, `n_gated_attn_instances`, `n_gated_delta_net_instances`, `n_mlp_instances`, `n_lm_head_instances`, `n_mtp_mix_instances`, `n_node_instances_complete`, `n_stage_kinds`, `stage_kind_ids`, `n_candidate_recipes`, `candidate_recipe_ids`, `n_policy_families`, `policy_families`, `n_compiler_stages`, `compiler_stage_ids`, `n_compiler_profiles`, `compiler_profile_ids`, `control_profile_id`, `n_artifact_approaches`, `artifact_approach_ids`, `n_consumer_sequences`, `consumer_sequence_ids`, `integrity_algorithm_candidates`, `n_fusion_hypotheses`, `fusion_hypothesis_ids`, `n_hoist_hypotheses`, `hoist_hypothesis_ids`, `n_representation_hypotheses`, `representation_hypothesis_ids`, `n_mode_hypotheses`, `mode_hypothesis_ids`, `n_orderings`, `ordering_ids`, `n_tile_families`, `tile_family_ids`, `n_parallel_decompositions`, `parallel_decomposition_ids`, `n_conversion_hypotheses`, `conversion_hypothesis_ids`, `n_mappings`, `mapping_ids`, `n_sku_unknown_symbols`, `sku_unknown_symbols`, `n_sensitive_ops`, `sensitive_ops`, `quality_high_ids`, `bottleneck_labels`, `evaluation_criterion_ids`, `pareto_axis_ids`, `corpus_class_ids`, `reconstruction_diagnostic_ids`, `teacher_forced_metric_ids`, `decode_metric_ids`, `prefill_metric_ids`, `kernel_metric_ids`, `memory_metric_ids`, `e2e_metric_ids`, `comparison_dimension_ids`, `n_synthesized_tasks`, `synthesized_task_ids`, `n_satellite_tasks`, `satellite_task_ids`, `n_chain_steps`, `chain_ids`, `chain_source_task_groups`, `n_evidence_classes`, `evidence_class_ids`, `task_evidence_classes`, `task_open_question_closed`, `n_tasks_open_question_closed`, `n_tasks_open_question_open`, `n_alternative_families`, `alternative_family_ids`, `alternative_family_counts`, `n_catalogued_alternatives`, `n_alternatives_selected`, `alternative_usefulness_label`, `n_remaining_open_questions`, `remaining_open_question_ids`, `n_remaining_open_questions_closed`, `n_experiment_fields`, `experiment_field_ids`, `n_experiment_eval_families`, `experiment_eval_family_ids`, `experiment_status_unrun`, `n_experiments`, `experiment_ids`, `experiments`, `experiment_selected`, `n_experiments_run`, `n_empty_metric_experiments`, `n_empty_alternative_experiments`, `n_diagrams`, `diagram_ids`, `n_architecture_headings`, `architecture_headings`, `n_backlog_headings`, `backlog_headings`, `n_headings`, `chain_follows_plan_path`, `chain_is_not_ledger_order`, `experiment_order_is_topological`, `ledger_open_question_alternatives_catalogued`, `frozen_for_comparative_review`, `pareto_frontier_selected`, `compiler_profile_selected`, `artifact_boundary_selected`, `ideal_byte_sequence_selected`, `integrity_algorithm_selected`, `fusion_winner_selected`, `decode_prefill_distinct_views_selected`, `layout_winner_selected`, `ordering_selected`, `tile_size_selected`, `mapping_winner_selected`, `winner_selected_without_measurements`, `hardware_selected`, `prompt_matrix_selected`, `reproducibility_protocol_selected`, `calibration_corpus_selected`, `eval_corpus_selected`, `acceptance_frontier_selected`, `sku_table_filled`, `hypothesis_survival_selected`, `n_mappings_selected`, `n_fusion_hypotheses_selected`, `keep_source_is_not_quality_winner`, `control_profile_is_keep_source`, `reconstruction_is_not_quality`, `toks_is_not_quality_axis`, `microbenchmark_cannot_pass_mapping`, `e2e_required_alongside_microbenchmarks`, `decode_prefill_share_graph`, `decode_prefill_share_artifact`, `decode_prefill_share_metric_identity`, `experiments_run`, `nll_measured_here`, `toks_measured_here`, `benchmarks_run`, `payloads_restreamed`, `gguf_payload_inspected`, `gguf_is_not_design_authority`, `gguf_is_pareto_reference`, `gguf_is_not_a_requirement`, `quartz_inspected`, `llama_inspected`, `llama_is_not_design_authority`, `device_query_run`, `nsight_run`, `vision_eval_deferred`, `safetensors_is_source_not_runtime`, `canonical_sentence_logical`, `canonical_sentence_synthesis`, `canonical_sentence_alternatives`, `canonical_sentence_unrun`, `canonical_sentence_backlog`, `canonical_sentence_depends`, `canonical_sentence_eval_split`, `synthesis_question_sentence`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. Booleans are JSON booleans. `full_attention_indices` and `example_T_values` are JSON arrays of ints. `experiments` is a JSON array of 16 objects. `task_evidence_classes` is a JSON object mapping each `synthesized_task_ids` entry to an array of evidence-class strings. `task_open_question_closed` and `experiment_selected` are JSON objects of booleans. `authority` is `"docs/architecture/plan.md"`. `deliverable_architecture` is `"docs/architecture/clean-sheet-architecture.md"`. `deliverable_backlog` is `"docs/architecture/experiment-backlog.md"`. `dtype` is `"bfloat16"`. `mamba_ssm_dtype` is `"float32"`. `control_profile_id` is `"control"`. `experiment_status_unrun` is `"unrun"`. `alternative_usefulness_label` is `"HYPOTHESIS"`. `n_catalogued_alternatives` is 125. `n_empty_metric_experiments` is 3. `n_empty_alternative_experiments` is 1. `n_headings` is 14.

### Tooling

Create `scripts/check_clean_sheet_architecture.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, CUDA Python, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, occupancy, and T=1/T=4096 MAC citations matching TASK-06 (use the locked MAC constants above; do not re-derive the full region MAC table). Duplicate TASK-01–19 id lists as constants; do not import them. Do not open `models/Qwen3.8-27B-Q4_K_M.gguf` or any safetensor. Do not run `deviceQuery`, Nsight, NLL, or any GPU binary.

CLI (cwd = repository root):

```text
python3 scripts/check_clean_sheet_architecture.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_clean_sheet_architecture.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --architecture docs/architecture/clean-sheet-architecture.md \
  --experiment-backlog docs/architecture/experiment-backlog.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema above). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `max_position_embeddings` as `model_T_max`. Derived/cited: parameter/byte totals matching TASK-01/06, MAC T=1/T=4096 citations matching TASK-06 locked integers. Constant fields: all canonical sentences, chain/evidence/alternative/experiment lists, TASK-01–19 id catalogues, locked booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- `--architecture PATH` and/or `--experiment-backlog PATH`: require each given PATH to contain (1) every required `##` heading listed for that file **in order**, (2) the first fenced `json` block equal to the live object, (3) none of `TBD`, `TODO`, `???`, (4) the word `UNKNOWN` present in that file’s Deferred vision section, (5) every locked document integer below present as a decimal or integer substring, (6) the words `HYPOTHESIS` and `not measurements` present (architecture) / `backlog` and `unrun` present (backlog). Architecture additionally requires: exactly one ` ```mermaid ` fence containing `flowchart`; all four architecture canonical sentences plus `synthesis_question_sentence` verbatim; every `chain_ids`, `evidence_class_ids`, `synthesized_task_ids`, `alternative_family_ids`, `remaining_open_question_ids`, `experiment_ids`, `node_type_ids`, `stage_kind_ids`, `candidate_recipe_ids`, `mapping_ids`, `fusion_hypothesis_ids`, `sku_unknown_symbols` id present as a substring; the diagram’s required IDs present **inside that mermaid fence**. Backlog additionally requires: **zero** mermaid fences; all four backlog canonical sentences plus `synthesis_question_sentence` verbatim; every `experiment_ids` and `experiment_field_ids` id present as a substring; every experiment `title` present as a substring. Both files: none of the forbidden phrases: `selected mapping winner`, `winning kernel`, `winning mapping`, `selected recipe`, `selected fusion`, `FROZEN_FOR_COMPARATIVE_REVIEW`, `should use this GPU`, `experiments were run`, `NLL was measured in this study`, `SKU table is filled`, `tok/s is the quality axis` (allow those substrings only inside `selects no` / `not a selected` / `remain unselected` / canonical sentences / `not measurements` / `unrun`). If neither path flag is passed and `--json` is absent, check both default deliverable paths. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).
- Do not fail if the GGUF file is absent. Do not stat/open GGUF as a checker requirement.

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks TASK-01–19 integers and ids against those documents).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `model_T_max==262144`
- `hidden_size==5120`, `intermediate_size==17408`, `vocab_size==248320`, `n_decoder_layers==64`
- `n_language_mtp_parameters==27320697856`
- `weight_bytes_language_mtp_excl_vision==54641395712`, `weight_bytes_unique_non_embed==52098598912`
- `mac_C_complete==27433238528`, `mac_A_complete==208896`
- `mac_decode_complete_T1==mac_prefill_complete_T1==27433447424`
- `mac_decode_complete_T4096==28288876544`
- `mac_prefill_complete_T4096==114119319486464`
- `example_T_values == [1, 4096]`
- `N_w==32`, `N_bank==32`
- `n_node_types==6`, `n_node_instances_complete==135`, `n_stage_kinds==9`
- `n_candidate_recipes==22`, `n_policy_families==16`, `n_mappings==18`, `n_fusion_hypotheses==22`
- `n_synthesized_tasks==19`, `n_chain_steps==12`, `n_evidence_classes==5`
- `chain_source_task_groups` keys equal `chain_ids`; `experimental_validation` value equals `["TASK-18","TASK-19"]`; no value contains `TASK-20`
- `n_alternative_families==14`, `n_catalogued_alternatives==125`, `n_alternatives_selected==0`
- `sum(alternative_family_counts)==125`
- `n_remaining_open_questions==12`, `n_remaining_open_questions_closed==0`
- `n_experiment_fields==12`, `n_experiments==16`, `n_experiments_run==0`
- `n_tasks_open_question_closed==9`, `n_tasks_open_question_open==10`
- `n_architecture_headings==8`, `n_backlog_headings==6`, `n_headings==14`, `n_diagrams==1`
- `node_type_ids` / `mapping_ids` / `candidate_recipe_ids` / `fusion_hypothesis_ids` / `sku_unknown_symbols` / `stage_kind_ids` equal the locked lists
- `len(experiments)==16` and each object has exactly the twelve `experiment_field_ids` keys in that order
- topological `depends_on` check; all `status=="unrun"`
- empty `metric_ids` only for the three locked ids; empty `alternatives` only for `exp_identity_harness`
- `ledger_open_question_alternatives_catalogued is True`, `frozen_for_comparative_review is False`
- `mapping_winner_selected is False`, `pareto_frontier_selected is False`, `fusion_winner_selected is False`
- `experiments_run is False`, `nll_measured_here is False`, `toks_measured_here is False`, `benchmarks_run is False`, `sku_table_filled is False`
- `toks_is_not_quality_axis is True`, `reconstruction_is_not_quality is True`, `microbenchmark_cannot_pass_mapping is True`
- `gguf_payload_inspected is False`, `quartz_inspected is False`, `llama_inspected is False`
- `device_query_run is False`, `nsight_run is False`
- `decode_prefill_share_metric_identity is False`
- `vision_eval_deferred is True`
- `n_embed_instances==2`, `n_gated_attn_instances==17`, `n_gated_delta_net_instances==48`, `n_mlp_instances==65`, `n_lm_head_instances==2`, `n_mtp_mix_instances==1`

Locked document integers that must appear as decimal substrings in **both** markdown files (in addition to ids): `5120`, `17408`, `248320`, `64`, `48`, `16`, `262144`, `27320697856`, `54641395712`, `52098598912`, `27433238528`, `208896`, `27433447424`, `28288876544`, `114119319486464`, `135`, `32`, `125`, `19`, `12`, `16`.

### Stage split

- **Implementation** writes `scripts/check_clean_sheet_architecture.py` **and** `docs/architecture/clean-sheet-architecture.md` **and** `docs/architecture/experiment-backlog.md` (eight architecture headings, six backlog headings, four+four canonical sentences plus `synthesis_question_sentence`, twelve chain steps, five evidence classes, 19-task citation table, fourteen alternative families, twelve remaining open questions, sixteen experiments with twelve fields, one Mermaid flowchart, identical JSON fences). Runs `--json` and the dual-path check after both documents exist. Records command outcomes in this dossier. Does not commit. Does not stream payloads. Does not open GGUF. Does not run benchmarks, NLL, `deviceQuery`, Nsight, or tok/s collection. Does not edit `plan.md` or the ledger.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner on **both** markdown files (TASK-19 / `performance-validation.md` style), Authority table links to this dossier / plan.md / TASK-01–19 deliverables as citations, heading/JSON fence consistency. Must not change locked integers, canonical sentences, experiment titles/hypotheses, alternative counts, or Mermaid node IDs. Does not edit TASK-01–19 artifacts.
- **Verification** independently re-runs focused commands, recomputes layer counts / `full_attention_indices` / `model_T_max` from sitting `text_config` (not from JSON echo), spot-checks cited TASK-01/06/08/11/12/15/17/18/19 integers and ids against those documents (not from this JSON echo), reads both documents against this dossier, confirms the experiment DAG is topological, confirms no Quartz/llama.cpp/GGUF-as-authority, no winner phrases, no freeze mark, no `plan.md` or ledger edit, no payload I/O, no GGUF open, no GPU/NLL run, that every alternative family usefulness is HYPOTHESIS, that experiments are unrun, and that remaining open questions stay open. The three ledger completion criteria are closed by the synthesis. TASK-21 freeze remains unselected.

- Invariants:
  - Eight architecture headings and six backlog headings in locked order; four architecture + four backlog canonical sentences plus `synthesis_question_sentence` verbatim; 12 chain steps; 5 evidence classes; 19 synthesized tasks; 14 alternative families; 125 catalogued alternatives; 0 selected; 12 remaining open questions; 0 closed; 16 experiments; 12 fields; one Mermaid flowchart with required IDs in architecture.md only.
  - Identical JSON fences in both markdown files, equal to live `--json`.
  - Prefill/decode share one graph and one artifact, not one metric identity.
  - Microbenchmarks cannot pass a mapping; reconstruction is not quality; tok/s is not Pareto $Y$; Q4_K_M is a reference not a requirement.
  - `experiments_run` false; `frozen_for_comparative_review` false; `mapping_winner_selected` false; `sku_table_filled` false.
  - Logical values do not imply allocation. Vision encoder remains unexpanded.
- Rejected alternatives:
  - Selecting any recipe, profile, mapping, fusion, layout, view, corpus, or hardware: rejected; catalogue only.
  - Running NLL / tok/s / Nsight / `deviceQuery` / SKU fill in Phase 1: rejected; `experiments_run` false.
  - Marking `FROZEN_FOR_COMPARATIVE_REVIEW`: rejected; TASK-21.
  - Closing TASK-07–19 open questions by assertion: rejected; `n_remaining_open_questions_closed` 0.
  - One backlog row per catalogued alternative (125 rows): rejected; sixteen dependency-ordered entries that **cite** alternative lists.
  - Two checkers or importing `check_*.py`: rejected; one stdlib sibling.
  - Recopying TASK-06 MAC tables or TASK-17 mapping definitions as new studies: rejected; cite.
  - Opening GGUF or inspecting Quartz/llama.cpp: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–19.
  - Editing frozen TASK-01–19 docs, the ledger, or `plan.md`.
  - Treating T=1 prefill as decode-only, reconstruction as NLL, or tok/s as quality.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/clean-sheet-architecture.md` exists and follows the eight headings above.
  - `docs/architecture/experiment-backlog.md` exists and follows the six headings above.
  - Architecture chain (12 steps) and evidence classes (5) are synthesized and cite TASK-01–19.
  - One architecture diagram (Mermaid flowchart with required IDs) plus fourteen alternative families (125 catalogued, 0 selected).
  - Sixteen experiment entries in topological `depends_on` order, each with all twelve requested fields; all `unrun`.
  - No winners selected; no experiments run; no freeze mark; remaining twelve open questions catalogued not closed.
  - Canonical sentences plus `synthesis_question_sentence` verbatim in both files; identical JSON fences match a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; no payload re-stream; no `plan.md` or ledger edit; no MEASURED NLL or tok/s.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_clean_sheet_architecture.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_clean_sheet_architecture.py
python3 scripts/check_clean_sheet_architecture.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_clean_sheet_architecture.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --architecture docs/architecture/clean-sheet-architecture.md \
  --experiment-backlog docs/architecture/experiment-backlog.md
```

- Candidate quality: not required — no model execution, NLL, or OPT-058; this increment is synthesis documentation. Experiment **hypotheses** are unrun.
- Repository-wide commands:

```sh
test -f docs/architecture/clean-sheet-architecture.md
test -f docs/architecture/experiment-backlog.md
python3 -m py_compile scripts/check_clean_sheet_architecture.py
python3 scripts/check_clean_sheet_architecture.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --architecture docs/architecture/clean-sheet-architecture.md \
  --experiment-backlog docs/architecture/experiment-backlog.md
```

Do not run Ruff, pytest, CMake, CUDA, Nsight, or `deviceQuery`; this increment does not introduce those gates. Do not run `scripts/analyze_bf16_tensors.py`. Do not open GGUF.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/clean-sheet-architecture.md` (create)
  - `docs/architecture/experiment-backlog.md` (create)
  - `scripts/check_clean_sheet_architecture.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-01–19 deliverables unchanged)
- Definition of done: both synthesis documents published with locked chain, evidence classes, diagram, alternative catalogue, and sixteen twelve-field topological experiment entries; JSON fences verify against sitting `config.json` plus locked constants; no experiments run; no winners selected; no freeze; ledger TASK-20 completion checkboxes can be marked at delivery; TASK-07–19 open questions remain catalogued and unselected.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T16:50:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-20.md`. Coupled IDs `none`. Document structure (8 architecture + 6 backlog headings), four+four canonical sentences plus synthesis sentence, 12 chain steps, 5 evidence classes, 19-task citation table, 14 alternative families (125 catalogued / 0 selected), 12 remaining open questions (0 closed), 16 topological experiments with 12 requested fields, stdlib checker `scripts/check_clean_sheet_architecture.py`, JSON schema, and acceptance commands are closed. Remaining TASK-07–19 open questions catalogued, not selected. Freeze left to TASK-21. `docs/architecture/clean-sheet-architecture.md`, `docs/architecture/experiment-backlog.md`, and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit. No experiments.
- Performance evidence applied: N/A — synthesis documentation; no sink ranking, no measured tok/s, no SKU fill

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - created `scripts/check_clean_sheet_architecture.py` (stdlib checker; occupancy from sitting `text_config`; locked chain/evidence/alternative/experiment catalogues; `--json` plus dual-path markdown checks)
  - created `docs/architecture/clean-sheet-architecture.md` (eight headings, four architecture canonical sentences plus `synthesis_question_sentence`, 12 chain steps, 5 evidence classes, 19-task citation table, 14 alternative families / 125 catalogued / 0 selected, one Mermaid flowchart with required IDs, 12 remaining open questions / 0 closed, identical JSON fence)
  - created `docs/architecture/experiment-backlog.md` (six headings, four backlog canonical sentences plus `synthesis_question_sentence`, 12 requested fields, 16 topological `unrun` experiments, identical JSON fence, no Mermaid)
  - this dossier Implementation run record
  - did **not** edit `docs/architecture/plan.md`, `task_ledger.md`, or any TASK-01–19 deliverable
  - no commit; no payload I/O; no GGUF open; no GPU/NLL/`deviceQuery`/Nsight
- Commands:
  - `python3 -m py_compile scripts/check_clean_sheet_architecture.py` — exit 0
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; stdout pretty-printed summary (37047 bytes); internal asserts passed (`n_chain_steps` 12, `n_evidence_classes` 5, `n_synthesized_tasks` 19, `n_catalogued_alternatives` 125, `n_alternatives_selected` 0, `n_remaining_open_questions` 12, `n_remaining_open_questions_closed` 0, `n_experiments` 16, `n_experiments_run` 0, `frozen_for_comparative_review` false)
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md` — exit 0 (headings, identical JSON fences equal to live `--json`, mermaid IDs, canonical sentences, unrun DAG, no winner/freeze phrases)
- UTC/time/tokens/cost: `2026-09-20T16:57:49Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent; parent/inherit mapping)
- Changes and evidence:
  - `docs/architecture/clean-sheet-architecture.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-19 / [`performance-validation.md`](../performance-validation.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`plan.md`](../plan.md) (OBSERVED evidence vocabulary), TASK-01–19 deliverables via citation table, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_clean_sheet_architecture.py`](../../../scripts/check_clean_sheet_architecture.py), and in-scope/deferred rows. Eight required `##` headings, four architecture canonical sentences plus `synthesis_question_sentence`, one Mermaid flowchart with required node IDs, and the JSON fence left unchanged (locked integers, chain/evidence/alternative counts, experiment ids, Mermaid node IDs untouched). No edits to TASK-01–19 deliverables, `plan.md`, or `task_ledger.md`.
  - `docs/architecture/experiment-backlog.md` — mechanical pass only. Added the same draft-status banner. Authority table already cross-links this dossier, [`clean-sheet-architecture.md`](../clean-sheet-architecture.md), [`quantization-validation.md`](../quantization-validation.md) (TASK-18), [`performance-validation.md`](../performance-validation.md) (TASK-19), sitting config, checker, and in-scope/deferred rows. Six required `##` headings, four backlog canonical sentences plus `synthesis_question_sentence`, sixteen topological `unrun` experiment entries, identical JSON fence, no Mermaid. Locked field schema, experiment titles/hypotheses, and integers untouched.
- Commands:
  - `test -f docs/architecture/clean-sheet-architecture.md` — pass (exit 0).
  - `test -f docs/architecture/experiment-backlog.md` — pass (exit 0).
  - `python3 -m py_compile scripts/check_clean_sheet_architecture.py` — pass (exit 0).
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 187 keys; `n_chain_steps` 12; `n_catalogued_alternatives` 125; `n_alternatives_selected` 0; `n_remaining_open_questions` 12; `n_remaining_open_questions_closed` 0; `n_experiments` 16; `n_experiments_run` 0; `frozen_for_comparative_review` false; JSON fence source unchanged).
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md` — pass (exit 0; eight + six headings, identical JSON fences, one Mermaid flowchart in architecture only, canonical sentences, unrun DAG, no winner/freeze phrases; banner did not break the check).
  - Independent JSON fence spot-check (live `--json` object deep-equal to fenced JSON in both markdown files, 187 keys) — pass.
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T17:00:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `composer-2.5` (delivery subagent; parent/inherit mapping)
- Diff review:
  - New: `docs/architecture/clean-sheet-architecture.md`, `docs/architecture/experiment-backlog.md`, `scripts/check_clean_sheet_architecture.py`, `docs/architecture/tasks/TASK-20.md`.
  - Modified: `docs/architecture/task_ledger.md` — status `TODO` → `IN PROGRESS` at admission; delivery flips to `DONE`.
  - Unchanged: `docs/architecture/plan.md`, TASK-01–19 deliverables, frozen upstream docs.
  - Scope matches dossier: synthesis documents + stdlib checker only; no experiments, GPU, GGUF, Quartz/llama inspection, winner selection, or freeze.
- Independent raw-record checks:
  - Recomputed sitting `text_config` from `.cache/authorities/qwen3.8-27b-transformers/config.json`: `hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, 64 decoder layers, 48 linear + 16 full, `full_attention_indices` `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`, `mtp_num_hidden_layers` 1, `model_T_max` 262144 — match live `--json`.
  - MAC/byte citations spot-checked against [`work-and-traffic.md`](../work-and-traffic.md): `mac_C_complete` 27433238528, `mac_A_complete` 208896, T=1/T=4096 decode/prefill MACs, `weight_bytes_unique_non_embed` 52098598912, `weight_bytes_language_mtp_excl_vision` 54641395712 — match.
  - Node instances 2+17+48+65+2+1=135 match TASK-11 complete counts; `n_mappings` 18 and `mapping_ids` match TASK-17 fence.
  - First fenced JSON in both deliverables deep-equal fresh `--json` (187 keys).
  - Eight architecture + six backlog headings, four+four canonical sentences plus `synthesis_question_sentence`, one Mermaid flowchart with required node IDs, 14 alternative families / 125 catalogued / 0 selected, 12 remaining open questions / 0 closed, 16 topological `unrun` experiments with 12 fields — confirmed.
  - Forbidden states absent: `n_alternatives_selected` 0, `frozen_for_comparative_review` false, `experiments_run` false, `nll_measured_here` false, `toks_measured_here` false, `benchmarks_run` false, `sku_table_filled` false, `gguf_payload_inspected` false.
- Commands:
  - `test -f docs/architecture/clean-sheet-architecture.md` — exit 0
  - `test -f docs/architecture/experiment-backlog.md` — exit 0
  - `python3 -m py_compile scripts/check_clean_sheet_architecture.py` — exit 0
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; 187 keys; `n_chain_steps` 12; `n_catalogued_alternatives` 125; `n_alternatives_selected` 0; `n_remaining_open_questions` 12; `n_remaining_open_questions_closed` 0; `n_experiments` 16; `n_experiments_run` 0; `frozen_for_comparative_review` false
  - `python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md` — exit 0
- Formatting changed files: none (`uv run ruff format .` not run — out of scope per dossier)
- Verdict: **PASS** (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T17:05:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-20 only; coupled IDs `none`
- Outcome: TASK-20 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T17:05:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/clean-sheet-architecture.md` (eight headings, 12 chain steps, 5 evidence classes, 19-task citation table, 14 alternative families / 125 catalogued / 0 selected, one Mermaid flowchart, 12 remaining open questions / 0 closed, JSON fence equal to live `--json`); `docs/architecture/experiment-backlog.md` (six headings, 16 topological `unrun` experiments with 12 fields, identical JSON fence); `scripts/check_clean_sheet_architecture.py` stdlib checker; `plan.md` unchanged; alternatives and unknowns catalogued not selected; freeze deferred to TASK-21 (`frozen_for_comparative_review` false; `n_remaining_open_questions_closed` 0)
- Candidate measured delta: N/A — synthesis documentation
- Shipping delta: N/A
- Quality result: not required
- Evidence completeness: N/A for performance-evidence checks
- Throughput delta (when applicable): N/A
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: **yes**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands; GGUF file is not required now; no GPU required; TASK-01–19 deliverables must remain unchanged
