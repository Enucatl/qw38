# TASK-17 — Map semantic nodes into CUDA design spaces

## Control

- Primary ID: `TASK-17`
- Coupled IDs: `none`
- Dependencies: `TASK-11`, `TASK-13`, `TASK-14`, `TASK-15`, `TASK-16` (all DONE at admission)
- Status: `IN PROGRESS`
- Ledger acceptance: Analyze multiple mapping alternatives per semantic node; Estimate work, storage, access, synchronization, occupancy, and mode suitability; Select no winner without measurements.

## Goal and boundaries

Produce `docs/architecture/cuda-design-space.md` as the Phase 1 **CUDA mapping experiment space**: multiple plausible **ownership and reduction** alternatives for every TASK-11 semantic node type, with TASK-16 evaluation algebra instantiated symbolically and TASK-15 candidate layouts attached as unselected experiment bindings. Close the three ledger completion criteria by (1) publishing at least two alternatives per node type (locked: three each, eighteen total), (2) estimating work, storage, access, synchronization, occupancy, and mode suitability for every alternative, and (3) selecting **no** mapping, kernel, launch config, MMA shape, layout, or fusion winner. **Keep** the ledger open question (which mappings win on target hardware and profiles after benchmarks) **unresolved**. Do **not** fill sitting-SKU numbers. Do **not** run TASK-19 measurements.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Node types, I/O, state, internals, and instance counts come from `docs/architecture/semantic-graph.md` (TASK-11). Decode GEMV consumers, one-token setting, serial stage kinds, and unselected fusion/packing attachments come from `docs/architecture/decode-plan.md` (TASK-13). Prefill GEMM consumers, five fundamental differences, layer-serial stages, and unselected views come from `docs/architecture/prefill-plan.md` (TASK-14). Candidate orderings, tile families, parallel decompositions, and justification hypotheses come from `docs/architecture/layout-strategy.md` (TASK-15). Execution/memory/sync/pipeline vocabulary, F1–F14, SKU-UNKNOWN symbols, and the seven evaluation criteria come from `docs/architecture/cuda-hardware-model.md` (TASK-16). Do not add node types, catalog IDs, layout objects, or hardware concepts.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established; TASK-16 published identities \(N_w=32\), \(N_{\text{bank}}=32\)), `DERIVED` (MAC/byte citations and occupancy/intensity algebra instantiated from locked ranks and F1–F14), `HYPOTHESIS` (every mapping **usefulness**, every mode-fit, every coalescing/bank-conflict **usefulness**, every fusion \(\Delta I\) vs \(\Delta O\) sign, every layout-decomposition **justification**), or `UNKNOWN` (sitting-SKU numeric limits and optional capabilities `async_copy_cap`, `mma_shapes`, `cluster_cap`; vision-encoder internals). No `MEASURED` tok/s, occupancy, or NLL.
  - GitHub Markdown math. Cite TASK-02 equation tags via TASK-11/13/14 contracts, TASK-06 MAC/byte integers via those docs, TASK-15 ordering/tile/decomposition ids, TASK-16 concept ids and formulae. Do not rewrite forward math, recopy TASK-06 MAC tables as a new work study, recopy TASK-15 as a new layout study, or recopy TASK-16 as a new hardware-model study.
  - Allowed evidence: TASK-11 semantic-graph, TASK-13 decode-plan, TASK-14 prefill-plan, TASK-15 layout-strategy, TASK-16 cuda-hardware-model, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier. TASK-01/02/03/04/06/09/12 integers and ids already cited by those documents may be **cited** through them. TASK-12 fusion hypothesis ids are cited through TASK-13/14 attachments (22 ids, all unselected); do not treat TASK-12 as a silent extra dependency. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No `deviceQuery`, datasheet SKU fill-in, SASS, or `*.cu` / `*.cuh` kernel bodies.
  - CUDA mappings name ownership, reduction, hypothesized launch symbols \((R_t,C_{\text{cta}},T_{\text{cta}})\), sync class, and pipeline mix. They are not selected kernels and not sitting-device recipes. Naming `tensor_core_mma` as a hypothesized pipeline is not selecting `mma_shapes`.
- Non-goals:
  - No mapping, kernel, launch-config, occupancy-target, MMA-shape, layout, fusion, packing, or view **winner**. Listing an alternative is not selecting it. `n_mappings_selected` = 0. `mapping_winner_selected` false. `launch_config_selected` false. `mma_shapes_selected` false. `layout_winner_selected` false. `fusion_winner_selected` false. `ledger_open_question_mapping_winner_closed` false.
  - No TASK-19 microbenchmarks, end-to-end tok/s, or achieved occupancy. Occupancy estimates are F1–F8 templates with UNKNOWN SKU symbols.
  - No sitting-device fill-in of TASK-16 UNKNOWN symbols.
  - No closing of TASK-15’s parallel-decomposition justification question. Instantiating a decomposition as a CUDA ownership axis does not justify an ordering or tile (`parallel_decomposition_justifies_layout_selected` false).
  - No decode or prefill **schedule** rewrite (TASK-13/14). No new node types (TASK-11). No new layout objects (TASK-15). No new hardware concepts (TASK-16).
  - No quantization recipe winners, artifact-boundary winner, or compiler-stage rewrite (TASK-08/09/10). In-kernel unpack remains a TASK-15 conversion hypothesis attachment.
  - No activation working-buffer **layouts** (`activations_in_layout_scope` false). Hypothesized register/shared live ranges are occupancy symbols, not TASK-15 objects.
  - No quality/NLL experiments (TASK-18).
  - No generic cuBLAS / CUTLASS / FlashAttention cookbook and no GGUF or Quartz kernel as authority.
  - No peak-memory claim and no summed CUDA live-set presented as a measured working set.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–16). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `semantic-graph.md`, `decode-plan.md`, `prefill-plan.md`, `layout-strategy.md`, `cuda-hardware-model.md`, or any TASK-08/09/10/11/12 deliverable.
  - Do not import other `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib CUDA-design-space checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places CUDA mappings after the Qwen-specific semantic graph and independent decode/prefill plans.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze; CUDA hardware documentation is allowed (consumed here via TASK-16, not by inspecting kernels).
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:93-95` — TASK-17 maps semantic nodes **and layouts** into multiple CUDA experiment spaces; TASK-19 owns performance methodology.
- `docs/architecture/task_ledger.md` TASK-17 row — produces `docs/architecture/cuda-design-space.md`; purpose is multiple ownership and reduction mappings per semantic node; open question is which mappings win after benchmarks (**kept unresolved here**); completion is multiple alternatives, six-way estimates, and no winner without measurements.
- `docs/architecture/task_ledger.md` TASK-11 established results — six node types, 135 complete instances, 14 sync edges; hardware mapping deferred there.
- `docs/architecture/task_ledger.md` TASK-13/14 established results — nine stage kinds; decode GEMV vs prefill GEMM; five fundamental differences; fusion/packing/views unselected; CUDA mapping deferred there.
- `docs/architecture/task_ledger.md` TASK-15 established results — seven layout objects; 14 orderings; 9 tile families; 9 parallel decompositions; 10 justification hypotheses including `j_mma_shaped_unselected_par` (MMA tiles need TASK-17 to **name** a decomposition); open question kept unresolved; no optimality.
- `docs/architecture/task_ledger.md` TASK-16 established results — 54-concept catalog, F1–F14, SKU-UNKNOWN, seven TASK-17 evaluation criteria; sitting limits remain UNKNOWN; this task instantiates those criteria and does not fill the SKU table.
- `docs/architecture/semantic-graph.md` — six `node_type_ids`; per-node ops/I/O/state; catalog partition; flexibilities.
- `docs/architecture/decode-plan.md` — decode consumers; `seq_*` attachments; 22 unselected fusion hypothesis ids.
- `docs/architecture/prefill-plan.md` — prefill consumers; `diff_matrix_matrix` / `diff_tiling` / `diff_reuse` / `diff_state` / `diff_temporary_storage`.
- `docs/architecture/layout-strategy.md` — orderings, tiles, `parallel_decomposition_ids`, justification hypotheses; `tile_mma_shaped` extents unselected.
- `docs/architecture/cuda-hardware-model.md` — ownership hierarchy, reduction/sync/pipeline ids, F1–F14, evaluation checklist, SKU symbols.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for ranks, layer counts, and cited MAC/byte products. Do not read safetensor payloads.
- `scripts/check_semantic_graph.py` / `check_decode_plan.py` / `check_prefill_plan.py` / `check_layout_strategy.py` / `check_cuda_hardware_model.py` — checker-style precedent. TASK-17’s checker is a sibling; do not import them.

## Performance evidence

N/A — CUDA **mapping-space** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. MAC and byte figures are TASK-06/04/11/13/14/15 citations plus DERIVED occupancy/intensity algebra, not MEASURED traffic or tok/s. Every mapping **usefulness** is HYPOTHESIS, not MEASURED. Do not apply the performance-evidence checklist to rank kernels or claim a winning mapping.

- Measurement identity: N/A (no engine binary). TASK-11/13/14/15/16 integers and formulae are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Mapping coverage is 6 node types, 18 unselected mappings (3 per type), 6 estimate dimensions, 7 TASK-16 evaluation criteria instantiated, 9 TASK-15 decompositions attached, 8 mapping-risk hypotheses.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` rank or a cited TASK-06/11/13/14/15/16 identity disagrees with those documents, stop and fail closed (do not invent a mapping).
- Claim types: ranks and MAC/bytes `observed`/`derived`; F1–F14 instantiations `derived` with SKU symbols `unknown`; every mapping/mode/coalescing/fusion-delta **usefulness** `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Mapping-space completeness is multiple alternatives per node, six estimate dimensions, and the open question left unresolved without a winner.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for mapping claims

If a node type, stage kind, layout object, parallel decomposition, MAC/byte, formula, or SKU symbol would disagree with TASK-11/13/14/15/16 or sitting `text_config`, the earlier document / config wins and this one is wrong.

- Prefill and decode share **one** semantic graph (TASK-11) and **one** compiled artifact (TASK-09 via TASK-15). This document maps **both** schedule consumers. Distinct **views** remain unselected (`decode_prefill_distinct_views_selected` false). Listing a decode-primary mapping and a prefill-primary mapping is not selecting dual views or a winner.
- Primary mapped objects are the six TASK-11 node types. Layout objects and parallel decompositions are **attachments**, not extra node types.
- Algebraic equivalents in TASK-02 remain the same real map: GQA-as-repeat does **not** store repeated KV; paper \(S^\top\) (19) is the same map as (17)–(18); chunkwise GDN is not zero \(S\) traffic; conv delay is 3 stored vectors.
- A CUDA mapping is not a kernel binary, not a CUDA graph, and not a selected launch. Hypothesized \(T_{\text{cta}}\) candidates are warp-multiple thread counts, not a selected block size.
- Unique weight bytes are counted **once** per complete decode/prefill. Mapping does not restream the model.
- Do not inspect Quartz, llama.cpp, or GGUF kernels to “confirm” mappings.
- Do not select a mapping, MMA shape, occupancy target, or layout. Do not fill TASK-16 UNKNOWN symbols from a datasheet.
- Name a CUDA decomposition for TASK-15 `j_mma_shaped_unselected_par` (`par_gemm_d_out` + `tile_mma_shaped` + hypothesized `tensor_core_mma` pipeline). That **names** the missing decomposition; it does not select MMA extents (`mma_shapes` stays `UNKNOWN`; `mma_tile_extents_selected` false).
- Instantiate TASK-16 evaluation criteria 1–7 as **criteria applied to hypothesized symbols**, not as measured points. Roofline F14 remains an upper bound with \(\Pi_{\text{peak}}\) and \(\Beta\) `UNKNOWN`.

### Deliverable structure (`docs/architecture/cuda-design-space.md`)

Use these **level-2 headings in this exact order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Mapping/mode/coalescing/fusion-delta **usefulness** cells are `HYPOTHESIS`. SKU limits and optional capabilities are `UNKNOWN`. Do not leave `TBD`. The word `UNKNOWN` may appear in SKU/evaluation/pipeline-optional rows and in Deferred vision; it must appear in Deferred vision.

Title: `# CUDA design space` (not `TASK-17`).

1. **Authority** — this dossier, semantic-graph, decode-plan, prefill-plan, layout-strategy, cuda-hardware-model, inventory via those docs, config, checker; evidence labels; in-scope (language+MTP per-node CUDA ownership/reduction alternatives + six-way estimates + TASK-16 criteria instantiation + TASK-15 layout attachments) vs deferred (vision encoder; TASK-19 measurements; sitting SKU table). State that the document specifies a **CUDA mapping experiment space**, not selected kernels and not measured winners.
2. **Mapping convention** — the five canonical sentences plus the layout sentence (exact text below); what a mapping / ownership / reduction is; open question stays open; no winner.
3. **Mapping vocabulary** — ownership, reduction, estimate dimensions, sync classes, pipeline mixes, hypothesized \(T_{\text{cta}}\) candidates; SKU symbols remain UNKNOWN.
4. **Per-node mapping alternatives** — eighteen mappings, three per node type, all unselected. Completes ledger checkbox 1. Contains the Mermaid fence.
5. **Work, storage, and access estimates** — per-mapping work (cited MAC + DERIVED partition), storage (cited HBM bytes + symbolic \(R_t,C_{\text{cta}}\)), access (TASK-15 ordering/tile attachments; coalescing usefulness HYPOTHESIS). Completes estimate dimensions `work`, `storage`, `access`.
6. **Synchronization, occupancy, and mode suitability** — per-mapping sync class, F1–F10 occupancy/hiding/wave templates with UNKNOWN SKU, decode vs prefill fit. Completes estimate dimensions `synchronization`, `occupancy`, `mode_suitability`. Completes ledger checkbox 2.
7. **Layout instantiations** — nine TASK-15 parallel decompositions attached to mappings; MMA decomposition named; TASK-15 justification question stays unresolved.
8. **Evaluation instantiation** — seven TASK-16 criteria applied; F13/F14; no winner.
9. **Fusion versus occupancy** — TASK-13/14’s 22 fusion hypotheses remain unselected; F11/F12 \(\Delta I\) vs \(\Delta O\) is HYPOTHESIS; mapping-risk hypotheses.
10. **Non-decisions** — what TASK-19/15/12/09/08/14 still own; open question stays open.
11. **Deferred vision** — residual-stream interface only; encoder mappings UNKNOWN.
12. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Mapping convention**, include these **five canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> CUDA mappings in this document are ownership and reduction alternatives per semantic node, not selected kernels and not measured winners.

> Occupancy, intensity, and roofline figures in this document instantiate TASK-16 algebra with symbolic SKU limits; they are not sitting-device measurements.

> Which mappings win on target hardware and profiles remains open until TASK-19 measurements.

> This document selects no CUDA mapping winner.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_mappings` = sentence 2; `canonical_sentence_symbolic` = sentence 3; `canonical_sentence_open_question` = sentence 4; `canonical_sentence_winner` = sentence 5.

Sentence 4 **keeps** the ledger open question unresolved in checker-substring form. JSON `ledger_open_question_mapping_winner_closed` false. Sentence 5 **closes** the third completion criterion as a prohibition, not as a selected mapping. Sentence 5 equals TASK-16 `canonical_winner_sentence`.

Immediately after those five, include this **layout sentence verbatim**:

> Instantiating a TASK-15 parallel decomposition as a CUDA ownership axis does not select a layout and does not justify an ordering or tile.

JSON `canonical_sentence_layout` equals that sentence. JSON `parallel_decomposition_justifies_layout_selected` false. JSON `mma_decomposition_named` true. JSON `mma_tile_extents_selected` false.

Bullets required under that heading:

- Six semantic node types are complete for this task (`n_node_types` 6). Eighteen mappings are complete (`n_mappings` 18); three per type (`n_mappings_per_node` 3).
- A mapping is a named ownership + reduction alternative of one node type with hypothesized \((R_t,C_{\text{cta}},T_{\text{cta}})\), sync class, pipeline mix, and six estimate dimensions. Listing a mapping is not selecting it (`n_mappings_selected` 0, `mapping_winner_selected` false).
- Estimate dimensions are complete: work, storage, access, synchronization, occupancy, mode suitability (`n_estimate_dimensions` 6).
- TASK-16 evaluation criteria are instantiated (`n_evaluation_criteria` 7) and not measured (`evaluation_measured` false).
- Prefill and decode share one graph and one artifact; distinct views remain unselected (`decode_prefill_distinct_views_selected` false).
- Activations are out of layout scope (`activations_in_layout_scope` false). Register/shared footprints are occupancy symbols (`activation_live_is_occupancy_symbol` true).
- Unique weights are counted once (`weight_unique_counted_once` true).
- Fan-out ≠ must-store and logical ≠ physical still hold.
- Primary coverage includes MTP (135 instances). Fusion winners remain TASK-12/13/14 hypotheses. Ideal byte sequence remains TASK-09 open. TASK-15 orderings/tiles remain unselected. Sitting SKU limits remain UNKNOWN.
- Thread/warp/CTA/grid **ownership** is named (`thread_geometry_absent` false). No launch config is selected (`launch_config_selected` false).
- No mapping winner (`mapping_winner_selected` false).

JSON booleans (lock true unless noted):

- `decode_prefill_share_artifact` = true
- `decode_prefill_share_graph` = true
- `decode_prefill_distinct_views_selected` = false
- `activations_in_layout_scope` = false
- `activation_live_is_occupancy_symbol` = true
- `hardware_independent` = false
- `cuda_mapping_deferred` = false
- `thread_geometry_absent` = false
- `launch_config_selected` = false
- `mapping_winner_selected` = false
- `kernel_named` = false
- `layout_winner_selected` = false
- `ordering_selected` = false
- `tile_size_selected` = false
- `mma_tile_extents_selected` = false
- `mma_shapes_selected` = false
- `mma_decomposition_named` = true
- `alignment_grain_selected` = false
- `conversion_pipeline_selected` = false
- `parallel_decomposition_selected` = false
- `parallel_decomposition_justifies_layout_selected` = false
- `ideal_byte_sequence_selected` = false
- `artifact_boundary_selected` = false
- `fusion_winner_selected` = false
- `ledger_open_question_mapping_winner_closed` = false
- `ledger_open_question_parallel_decomposition_closed` = false
- `evaluation_measured` = false
- `sku_limits_unknown` = true
- `async_copy_cap_unknown` = true
- `mma_shapes_unknown` = true
- `cluster_cap_unknown` = true
- `occupancy_formulae_instantiated` = true
- `gdn_primary_is_recurrent_eq_17` = true
- `chunkwise_not_zero_s_traffic` = true
- `paper_s_transpose_same_map` = true
- `state_write_not_optional` = true
- `kv_rope_baked_into_k` = true
- `gqa_repeat_not_stored` = true
- `conv_z_does_not_enter_conv` = true
- `weight_unique_counted_once` = true
- `weight_second_w_lm_read_is_hypothesis` = true
- `gguf_is_not_the_runtime_format` = true
- `safetensors_is_source_not_runtime` = true
- `vision_interface_is_not_a_node` = true
- `payloads_restreamed` = false
- `analyzes_work` = true
- `analyzes_storage` = true
- `analyzes_access` = true
- `analyzes_synchronization` = true
- `analyzes_occupancy` = true
- `analyzes_mode_suitability` = true
- `multiple_alternatives_per_node` = true
- `winner_selected_without_measurements` = false
- `cluster_assumed_present` = false
- `tma_assumed_present` = false
- `red_grid_selected` = false

JSON `T_is_stored_length_after_append` true. JSON `example_T` `[1, 4096]`. JSON `primary_includes_mtp` true. JSON `sku_policy` exactly `parameterized_unknown_until_measured_table`.

### Mapping vocabulary (lock; heading 3)

JSON array `ownership_ids` in this exact order (4 ids). JSON `n_ownership_classes` = 4. These are TASK-16 concept ids. Do not add `thread_block_cluster` as an ownership class (`cluster_cap` is `UNKNOWN`; `cluster_assumed_present` false). `block` is an alias of `cta` (TASK-16); ownership uses `cta` only.

| id | TASK-16 concept | Owns |
| --- | --- | --- |
| `thread` | `thread` | one output element or one inner-loop step |
| `warp` | `warp` | \(N_w=32\) lanes issued together |
| `cta` | `cta` | one thread block; domain of `syncthreads` |
| `grid` | `grid` | one kernel launch; no CTA barrier across blocks |

JSON array `reduction_ids` in this exact order (5 ids). JSON `n_reduction_classes` = 5.

| id | Meaning | Sync implied |
| --- | --- | --- |
| `red_none` | owner writes complete outputs; no partial reduction | `sync_none` or local barrier only |
| `red_warp` | warp shuffle / `warp_sync` reduction of partials | `sync_warp` |
| `red_cta` | shared-memory reduction across the CTA | `sync_cta` |
| `red_splitk_cta` | split along a contraction axis; CTA reduces partials | `sync_cta` |
| `red_grid` | grid-wide reduction (atomics or cooperative groups) | `sync_grid`; capability `UNKNOWN` |

JSON `red_grid_requires_cooperative_or_atomics` true. JSON `n_mappings_using_red_grid` = 0. JSON `red_grid_selected` false. Grid-wide reduction is named as unselected variant hypothesis `var_splitk_grid` (see heading 9), not as a nineteenth mapping.

JSON array `estimate_dimension_ids` in this exact order (6 ids). JSON `n_estimate_dimensions` = 6: `work`, `storage`, `access`, `synchronization`, `occupancy`, `mode_suitability`.

JSON array `sync_class_ids` in this exact order (5 ids). JSON `n_sync_classes` = 5. These instantiate TASK-16 evaluation criterion 5:

`sync_none`, `sync_warp`, `sync_cta`, `sync_grid`, `sync_stream_event`

Map to TASK-16: none / `warp_sync` / `syncthreads` / grid-cooperative / `stream`+`event`. JSON `sync_grid_requires_cooperative` true. No mapping uses `sync_grid` as a selected class (`n_mappings_using_sync_grid` 0). `map_mlp_grid_T` and `map_mtp_split_norm_gemm` use `sync_stream_event` (independent CTAs joined by stream/event), not grid-cooperative.

JSON array `pipeline_ids` in this exact order (6 ids). JSON `n_pipeline_ids` = 6. TASK-16 instruction families:

`fma`, `ffma`, `load_store`, `tensor_core_mma`, `async_gmem_to_smem`, `tma`

JSON `async_copy_optional` true. Do not assume TMA exists (`tma_assumed_present` false). Do not pick `mma` vs `wgmma` vs later PTX variants. `tensor_core_mma` is the hypothesized MMA family; legal shapes stay `UNKNOWN`.

JSON array `cta_T_candidates` `[32, 64, 128, 256]`. JSON `n_cta_T_candidates` = 4. JSON `n_cta_T_selected` = 0. These are hypothesized threads-per-CTA and **must** be multiples of \(N_w=32\) (DERIVED from TASK-16 F7). They are not TASK-15 tile extents (those remain layout candidates). Naming 256 is not selecting it. JSON `cta_T_all_multiples_of_warp` true.

JSON array `mode_fit_ids` in this exact order (3 ids). JSON `n_mode_fit_ids` = 3: `decode_primary`, `prefill_primary`, `both`. JSON `mode_suitability_label` exactly `HYPOTHESIS`. JSON `n_mode_winners_selected` = 0. A `decode_primary` label is hypothesized fit, not a prohibition on prefill use.

JSON array `consumer_mode_ids` `["decode_gemv","prefill_gemm"]` (TASK-15). JSON `n_consumer_modes` = 2. JSON `n_consumer_modes_selected` = 0.

JSON `N_w` = 32. JSON `N_bank` = 32. JSON `sku_unknown_symbols` exactly the TASK-16 list of 17 ids:

`N_SM`, `W_max`, `S_reg`, `C_smem`, `T_max`, `B_max`, `N_bar`, `N_sched`, `G_reg`, `G_smem`, `Beta_HBM`, `Pi_FMA`, `Pi_TC`, `L_issue`, `async_copy_cap`, `mma_shapes`, `cluster_cap`

Do not instantiate any of those 17 with a datasheet or sitting-GPU number.

### Per-node mapping alternatives (lock; heading 4)

JSON array `node_type_ids` in TASK-11 order (6 ids): `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`. JSON `n_node_types` = 6. JSON instance counts cited from TASK-11: `n_embed_instances` 2, `n_gated_attn_instances` 17, `n_gated_delta_net_instances` 48, `n_mlp_instances` 65, `n_lm_head_instances` 2, `n_mtp_mix_instances` 1, `n_node_instances_complete` 135.

JSON array `mapping_ids` in this exact order (18 ids). JSON `n_mappings` = 18. JSON `n_mappings_per_node` = 3. JSON `n_mappings_selected` = 0. JSON object `mapping_selected` keyed in that order with every value `false`. JSON `mapping_usefulness_label` exactly `HYPOTHESIS`. Parallel arrays `mapping_node_types`, `mapping_ownership_ids`, `mapping_reduction_ids`, `mapping_sync_class_ids`, `mapping_primary_pipeline_ids`, `mapping_mode_fit_ids`.

| id | Node | Ownership | Reduction | Sync | Primary pipeline | Mode fit |
| --- | --- | --- | --- | --- | --- | --- |
| `map_embed_thread_element` | `embed` | `thread` | `red_none` | `sync_none` | `load_store` | `both` |
| `map_embed_warp_row` | `embed` | `warp` | `red_none` | `sync_warp` | `load_store` | `both` |
| `map_embed_cta_vector` | `embed` | `cta` | `red_none` | `sync_cta` | `load_store` | `both` |
| `map_attn_cta_head` | `gated_attn` | `cta` | `red_none` | `sync_cta` | `fma` | `both` |
| `map_attn_warp_t` | `gated_attn` | `warp` | `red_warp` | `sync_warp` | `fma` | `decode_primary` |
| `map_attn_cta_splitk` | `gated_attn` | `cta` | `red_splitk_cta` | `sync_cta` | `fma` | `both` |
| `map_gdn_cta_head` | `gated_delta_net` | `cta` | `red_none` | `sync_cta` | `fma` | `both` |
| `map_gdn_warp_recurrent` | `gated_delta_net` | `warp` | `red_none` | `sync_warp` | `fma` | `decode_primary` |
| `map_gdn_cta_chunk` | `gated_delta_net` | `cta` | `red_cta` | `sync_cta` | `fma` | `prefill_primary` |
| `map_mlp_cta_dout` | `mlp` | `cta` | `red_none` | `sync_cta` | `fma` | `both` |
| `map_mlp_cta_splitk` | `mlp` | `cta` | `red_splitk_cta` | `sync_cta` | `fma` | `both` |
| `map_mlp_grid_T` | `mlp` | `grid` | `red_none` | `sync_stream_event` | `tensor_core_mma` | `prefill_primary` |
| `map_lm_cta_vocab` | `lm_head` | `cta` | `red_none` | `sync_cta` | `fma` | `both` |
| `map_lm_cta_splitk` | `lm_head` | `cta` | `red_splitk_cta` | `sync_cta` | `fma` | `both` |
| `map_lm_warp_gemv` | `lm_head` | `warp` | `red_warp` | `sync_warp` | `fma` | `decode_primary` |
| `map_mtp_cta_fc` | `mtp_mix` | `cta` | `red_none` | `sync_cta` | `fma` | `both` |
| `map_mtp_cta_fused` | `mtp_mix` | `cta` | `red_none` | `sync_cta` | `fma` | `both` |
| `map_mtp_split_norm_gemm` | `mtp_mix` | `grid` | `red_none` | `sync_stream_event` | `load_store` | `both` |

JSON `mapping_node_types` `["embed","embed","embed","gated_attn","gated_attn","gated_attn","gated_delta_net","gated_delta_net","gated_delta_net","mlp","mlp","mlp","lm_head","lm_head","lm_head","mtp_mix","mtp_mix","mtp_mix"]`.

JSON `mapping_ownership_ids` `["thread","warp","cta","cta","warp","cta","cta","warp","cta","cta","cta","grid","cta","cta","warp","cta","cta","grid"]`.

JSON `mapping_reduction_ids` `["red_none","red_none","red_none","red_none","red_warp","red_splitk_cta","red_none","red_none","red_cta","red_none","red_splitk_cta","red_none","red_none","red_splitk_cta","red_warp","red_none","red_none","red_none"]`.

JSON `mapping_sync_class_ids` `["sync_none","sync_warp","sync_cta","sync_cta","sync_warp","sync_cta","sync_cta","sync_warp","sync_cta","sync_cta","sync_cta","sync_stream_event","sync_cta","sync_cta","sync_warp","sync_cta","sync_cta","sync_stream_event"]`.

JSON `mapping_primary_pipeline_ids` `["load_store","load_store","load_store","fma","fma","fma","fma","fma","fma","fma","fma","tensor_core_mma","fma","fma","fma","fma","fma","load_store"]`.

JSON `mapping_mode_fit_ids` `["both","both","both","both","decode_primary","both","both","decode_primary","prefill_primary","both","both","prefill_primary","both","both","decode_primary","both","both","both"]`.

JSON `n_decode_primary_mappings` = 3. JSON `n_prefill_primary_mappings` = 2. JSON `n_both_mode_mappings` = 13. Sum 18.

Required prose per node type (compact; do not unroll 135 instances):

**`embed`.** Gather of shared \(E\); TASK-06 `mac_embed` 0; `weight_gather_bytes_per_row` 10240. No RMS. No residual add. Three alternatives differ by who issues the row copy (thread / warp / CTA). Optional `async_gmem_to_smem` may attach to `map_embed_cta_vector` only as HYPOTHESIS (`async_copy_cap` UNKNOWN). Hidden-major gather (`ord_embed_hidden_major`) is an access hypothesis, not a fourth mapping.

**`gated_attn`.** Residual RMS `(2)`, projections `(6)`–`(7)`, QK-RMS + mRoPE, causal GQA softmax `(9)`, sigmoid gate `(10)`, `W_o`, residual add `(4)`, KV append. Decode: one query vs length \(T\). Prefill: causal matrix-matrix with exact \(T(T+1)/2\). `map_attn_cta_head` owns a query head (`par_attn_head`); `map_attn_warp_t` splits stored length (`par_attn_T`) with warp reduction of scores; `map_attn_cta_splitk` splits \(T\) or \(d_\text{in}\) then CTA-reduces. Prefill GEMM of projections may hypothesize `tensor_core_mma` as a **secondary** pipeline on `map_attn_cta_head` without changing the primary `fma` id (softmax/AV stay scalar). Secondary pipeline usefulness is HYPOTHESIS. JSON `attn_prefill_mma_is_secondary_hypothesis` true.

**`gated_delta_net`.** Residual RMS, projections `(13)`, depthwise conv `(14)` on `C_state`, SiLU/QKV, \(\alpha/\beta\), L2, recurrence `(17)`–`(18)` as definition, GatedRMSNorm `(3)`, `W_out`, residual add. FIR channels use `par_conv_channel` **inside** these mappings (conv is not a seventh node type). `map_gdn_cta_head` / `map_gdn_warp_recurrent` keep left-to-right recurrence per head (`par_gdn_head`). `map_gdn_cta_chunk` is the chunkwise/WY **algebraic equivalent** of the same map (TASK-11 flexibility); it is not zero \(S\) traffic (`chunkwise_not_zero_s_traffic` true) and is not a second node. JSON `gdn_chunk_is_algebraic_equivalent` true.

**`mlp`.** Post-RMS `(2)`, SwiGLU `(21)`, residual add `(5)`. Decode GEMV; prefill GEMM over \(T\). `map_mlp_cta_dout` splits \(d_\text{out}\) (`par_gemm_d_out`); `map_mlp_cta_splitk` splits \(d_\text{in}\) (`par_gemm_d_in`); `map_mlp_grid_T` splits sequence \(T\) (`par_gemm_T`) and is `prefill_primary` because decode \(T_\text{new}=1\) makes that split degenerate. `tile_mma_shaped` attaches to `map_mlp_cta_dout` / `map_mlp_grid_T` as an unselected layout family with `mma_shapes` UNKNOWN.

**`lm_head`.** Final RMS then \(W_\text{lm}\) `(22)`/`(24)`; \(V=248320\); `mac_lm_head` 1271398400; `weight_bytes_lm_head` 2542796800. Sampling softmax over \(V\) remains out of scope. `map_lm_cta_vocab` owns vocab-out tiles; `map_lm_cta_splitk` splits \(H\); `map_lm_warp_gemv` is decode-primary warp-owned GEMV. A second physical \(W_\text{lm}\) read for MTP logits stays HYPOTHESIS extra, not unique bytes.

**`mtp_mix`.** RMS of `e_next` and `h_64`, concat, \(W_\text{fc}\) `(23)`; `mac_mtp_fc` 52428800. `map_mtp_cta_fc` is the unfused-looking CTA GEMM of \(W_\text{fc}\). `map_mtp_cta_fused` attaches unselected `fuse_mtp_mix_internals`. `map_mtp_split_norm_gemm` attaches unselected `split_mtp_cat` as two launches joined by `sync_stream_event`. Attaching a TASK-12/13/14 fusion id is not selecting it.

Required: every node type has three alternatives; at least two distinct `ownership_ids` per node type; at least two distinct `reduction_ids` **or** `sync_class_ids` per node type. Checker asserts that partition.

Secondary pipelines (optional; JSON object `mapping_secondary_pipeline_ids`, values `null` or a `pipeline_ids` entry):

- `map_embed_cta_vector`: `async_gmem_to_smem` (HYPOTHESIS; may be absent)
- `map_attn_cta_head`: `tensor_core_mma` (HYPOTHESIS prefill projections)
- `map_mlp_cta_dout`: `tensor_core_mma` (HYPOTHESIS prefill GEMM)
- `map_lm_cta_vocab`: `tensor_core_mma` (HYPOTHESIS prefill GEMM)
- `map_mtp_cta_fc`: `tensor_core_mma` (HYPOTHESIS)
- all others: `null`

JSON `n_secondary_pipeline_hypotheses` = 5. JSON `n_secondary_pipelines_selected` = 0.

### Work, storage, and access estimates (lock; heading 5)

Cite TASK-06 MAC/bytes through TASK-11/13/14/15. Checker **recomputes** the cited MAC integers from `text_config` with the same identities as TASK-06/11 (full proj \(24\cdot 256\cdot H + 2\cdot 4\cdot 256\cdot H + H\cdot 24\cdot 256\); linear token sum; MLP \(3IH\); `lm_head` \(VH\); GDN \(3\cdot 48\cdot 128\cdot 128\); conv \(d_\text{qkv}\cdot 4\); `mtp.fc` \(H\cdot 2H\)).

JSON keys (ints unless noted):

`mac_embed` 0, `mac_full_proj_per_layer` 104857600, `mac_attn_coeff_per_full_layer` 12288, `mac_lin_token_per_layer` 118235136, `mac_lin_conv_per_layer` 40960, `mac_gdn_per_layer` 2359296, `mac_mlp_per_layer` 267386880, `mac_lm_head` 1271398400, `mac_mtp_fc` 52428800, `weight_gather_bytes_per_row` 10240, `kv_bytes_per_full_layer_per_token` 4096, `kv_bytes_all_per_token` 69632, `c_bytes_per_layer` 61440, `s_bytes_per_layer` 3145728, `s_f32_bytes` 150994944, `weight_bytes_lm_head` 2542796800.

JSON numbers: `i_mlp_weight_only` 1, `i_lm_head_weight_only` 1, `i_gdn_vs_s_rw` 0.75, `i_attn_core_vs_kv` 6.

JSON `bottleneck_labels` copied from TASK-06 in TASK-06 order: `weight_memory`, `vocab_memory`, `state_memory`, `kv_memory`, `quadratic_attn`, `compute`. Restating a label is a **citation**, still HYPOTHESIS. JSON `n_bottleneck_labels` = 6.

**Work partition (DERIVED, not a new work study).** For a mapping that splits an axis of length \(L\) into hypothesized tiles of extent \(e\) (unselected; \(e\) from TASK-15 `tile_extent_candidates` or head counts), work per owner is \((e/L)\) of the cited MAC when `red_none`, and the full cited MAC per reduction tree when `red_splitk_cta` / `red_warp` (partials sum to the same MAC). Do not pick \(e\). Write the identity:

\[
F_{\text{owner}} = (e/L)\,F_{\text{node}}\quad\text{when reduction is }\texttt{red\_none.}
\]

JSON `work_partition_identity_present` true. Head counts used as \(L\) when the split is heads: \(L=24\) for `par_attn_head`, \(L=48\) for `par_gdn_head`, \(L=4\) for `par_kv_head`, \(L=10240\) for `par_conv_channel`, \(L=V=248320\) for embed rows (independent rows, not a split of one row).

Per-mapping work cite (JSON `mapping_work_mac_ids` parallel to `mapping_ids`):

`mac_embed`, `mac_embed`, `mac_embed`, `mac_full_proj_per_layer`, `mac_attn_coeff_per_full_layer`, `mac_attn_coeff_per_full_layer`, `mac_lin_token_per_layer`, `mac_gdn_per_layer`, `mac_gdn_per_layer`, `mac_mlp_per_layer`, `mac_mlp_per_layer`, `mac_mlp_per_layer`, `mac_lm_head`, `mac_lm_head`, `mac_lm_head`, `mac_mtp_fc`, `mac_mtp_fc`, `mac_mtp_fc`

`map_attn_cta_head` cites projection MAC as the large contraction and still must mention `mac_attn_coeff_per_full_layer` \(T\)-scaled core in prose. `map_gdn_cta_head` cites the full linear-token MAC including conv `40960` and GDN `2359296`. `map_gdn_warp_recurrent` and `map_gdn_cta_chunk` cite `mac_gdn_per_layer` for the recurrence core; prose must still name conv as `par_conv_channel` sub-work.

**Storage.** HBM backing is the cited unique-weight / state bytes (not a live-set sum). Register and shared footprints stay symbols \(R_t\), \(C_{\text{cta}}\) (TASK-16). Do not invent numeric register counts. JSON `storage_rt_numeric` false. JSON `storage_hbm_cited` true. Per-mapping HBM cite ids (JSON `mapping_hbm_cite_ids`):

`weight_gather_bytes_per_row` ×3 for embed; `kv_bytes_per_full_layer_per_token` for the three attn maps; `s_bytes_per_layer` for the three GDN maps (prose also names `c_bytes_per_layer`); `mac_mlp` maps cite weight-memory class (bytes via \(3IH\times\) dtype; do not recopy a full weight table); lm_head maps cite `weight_bytes_lm_head`; mtp maps cite `mac_mtp_fc` work with \(H\cdot 2H\cdot 2\) BF16 unique \(W_\text{fc}\) identity stated in prose as DERIVED \(2\cdot 5120\cdot 10240=104857600\) B if conceptual BF16 — checker key `weight_bytes_mtp_fc` = \(2\cdot H\cdot 2H=104857600\).

JSON `weight_bytes_mtp_fc` 104857600 (DERIVED \(2\cdot H\cdot 2H\) BF16). This is a citation identity, not a selected store dtype (`activation_dtype_decided` false remains).

**Access.** Attach TASK-15 orderings/tiles as HYPOTHESIS bindings. Coalescing usefulness is HYPOTHESIS (TASK-16 identity 5 is OBSERVED as a hardware fact; whether a given ordering realizes it is not measured). Bank-conflict usefulness is HYPOTHESIS. JSON `access_usefulness_label` exactly `HYPOTHESIS`. JSON `coalescing_identity_observed` true (the programming-guide identity, cited from TASK-16). JSON `n_access_winners_selected` = 0.

JSON object `mapping_layout_object_ids` (arrays):

- embed maps: `["gather_row"]`
- attn maps: `["dense_gemm","state_kv"]`
- GDN maps: `["dense_gemm","depthwise_conv","vector_param","state_c","state_s"]`
- mlp maps: `["dense_gemm"]`
- lm_head maps: `["dense_gemm"]`
- mtp maps: `["dense_gemm"]`

JSON object `mapping_decomposition_ids` (arrays; attachments, not selected decompositions):

- `map_embed_thread_element`, `map_embed_warp_row`, `map_embed_cta_vector`: `["par_embed_row"]`
- `map_attn_cta_head`: `["par_attn_head","par_kv_head"]`
- `map_attn_warp_t`: `["par_attn_T"]`
- `map_attn_cta_splitk`: `["par_attn_T","par_gemm_d_in"]`
- `map_gdn_cta_head`, `map_gdn_warp_recurrent`: `["par_gdn_head","par_conv_channel"]`
- `map_gdn_cta_chunk`: `["par_gdn_head","par_conv_channel"]`
- `map_mlp_cta_dout`: `["par_gemm_d_out"]`
- `map_mlp_cta_splitk`: `["par_gemm_d_in"]`
- `map_mlp_grid_T`: `["par_gemm_T"]`
- `map_lm_cta_vocab`, `map_lm_warp_gemv`: `["par_gemm_d_out"]`
- `map_lm_cta_splitk`: `["par_gemm_d_in"]`
- `map_mtp_cta_fc`, `map_mtp_cta_fused`, `map_mtp_split_norm_gemm`: `["par_gemm_d_out"]`

Every TASK-15 `parallel_decomposition_ids` entry appears in at least one mapping attachment (checker). That instantiates the nine decompositions as CUDA ownership axes without selecting them.

### Synchronization, occupancy, and mode suitability (lock; heading 6)

**Synchronization.** Use the `mapping_sync_class_ids` column. Prose must state **what is ordered** for each used class (cite TASK-16): `sync_none` orders nothing beyond the issuing thread; `sync_warp` orders one warp; `sync_cta` is `__syncthreads` within one CTA; `sync_stream_event` orders kernels/memcopies on a stream via events; `sync_grid` is named in vocabulary but unused (`cooperative_groups` grid sync `UNKNOWN`). Do not assume cluster barriers.

**Occupancy.** Instantiate TASK-16 F1–F10 with hypothesized \((R_t,C_{\text{cta}},T_{\text{cta}})\) and UNKNOWN SKU symbols. Required formula substrings **outside** the JSON fence (character-for-character, inside `$...$` or `$$...$$`):

- `O = \min(O_\text{reg}, O_\text{smem}, O_\text{threads}, O_\text{cta})` (F1)
- `W_\text{cta} = \lceil T_\text{cta} / N_w \rceil` (F7)
- `O = W_\text{active} / W_\max` (F8)
- `W_\text{need} = N_\text{sched} \cdot L_\text{issue}` (F9)
- `N_\text{waves} = \lceil N_\text{grid} / (N_\text{SM} \cdot B_\text{SM}) \rceil` (F10)

Companion identities (DERIVED; required in prose): \(W_{\text{active}}=B_{\text{SM}}\cdot W_{\text{cta}}\); hiding complete only if \(W_{\text{active}}\ge W_{\text{need}}\); last-wave \(\eta_{\text{wave}}=N_{\text{grid}}/(N_{\text{waves}}\cdot N_{\text{SM}}\cdot B_{\text{SM}})\). \(N_{\text{SM}}\), \(W_{\max}\), \(S_{\text{reg}}\), \(C_{\text{smem}}\), \(T_{\max}\), \(B_{\max}\), \(N_{\text{sched}}\), \(L_{\text{issue}}\) stay `UNKNOWN`. Do not report achieved occupancy. Do not print a numeric occupancy fraction as if measured.

JSON `occupancy_sku_symbols_unknown` true. JSON `achieved_occupancy_reported` false.

Hypothesized \(T_{\text{cta}}\) is taken from `cta_T_candidates` and is **not selected**. Prose may say “a hypothesized \(T_{\text{cta}}\in\{32,64,128,256\}\)” without picking one. \(R_t\) and \(C_{\text{cta}}\) remain symbols. Fusion may raise both (F11, F12) and fail F9 even if intensity rises.

**Mode suitability.** Use `mapping_mode_fit_ids`. Completes decode vs prefill (TASK-13 GEMV / TASK-14 GEMM; `diff_matrix_matrix`, `diff_tiling`, `diff_reuse`). Usefulness of acting on a `decode_primary` or `prefill_primary` label remains HYPOTHESIS. Do not select distinct views. At \(T=1\), `map_mlp_grid_T` degenerates (DERIVED); that identity does not select a decode mapping.

JSON `mode_fit_at_T1_grid_T_degenerates` true.

### Layout instantiations (lock; heading 7)

JSON array `parallel_decomposition_ids` in TASK-15 order (9 ids). JSON `n_parallel_decompositions` = 9. JSON `n_parallel_decompositions_selected` = 0.

JSON object `decomposition_mapping_ids` keyed in that order (each value an array of `mapping_ids`):

- `par_gemm_d_out`: `["map_mlp_cta_dout","map_lm_cta_vocab","map_lm_warp_gemv","map_mtp_cta_fc","map_mtp_cta_fused","map_mtp_split_norm_gemm"]`
- `par_gemm_d_in`: `["map_mlp_cta_splitk","map_lm_cta_splitk","map_attn_cta_splitk"]`
- `par_gemm_T`: `["map_mlp_grid_T"]`
- `par_attn_head`: `["map_attn_cta_head"]`
- `par_attn_T`: `["map_attn_warp_t","map_attn_cta_splitk"]`
- `par_gdn_head`: `["map_gdn_cta_head","map_gdn_warp_recurrent","map_gdn_cta_chunk"]`
- `par_conv_channel`: `["map_gdn_cta_head","map_gdn_warp_recurrent","map_gdn_cta_chunk"]`
- `par_kv_head`: `["map_attn_cta_head"]`
- `par_embed_row`: `["map_embed_thread_element","map_embed_warp_row","map_embed_cta_vector"]`

JSON array `justification_hypothesis_ids` copied from TASK-15 (10 ids). JSON `n_justification_hypotheses` = 10. JSON `n_justification_hypotheses_selected` = 0. JSON `justification_usefulness_label` exactly `HYPOTHESIS`.

**MMA decomposition name (required prose).** TASK-15 `j_mma_shaped_unselected_par` left the CUDA decomposition unspecified. This document **names** it: CTA-owned \(d_\text{out}\) tiles (`par_gemm_d_out`) whose hypothesized pipeline mix includes `tensor_core_mma`, consuming unselected layout family `tile_mma_shaped`, with legal \((M,N,K)\) and dtypes equal to TASK-16 `mma_shapes` (`UNKNOWN`). Mapped alternatives that may bind that name: `map_mlp_cta_dout`, `map_mlp_grid_T`, `map_lm_cta_vocab`, `map_mtp_cta_fc`. Naming is not selecting extents, not selecting MMA vs wgmma, and not selecting a winner. JSON `mma_decomposition_named` true. JSON `mma_decomposition_id` exactly `par_gemm_d_out`. JSON `mma_pipeline_id` exactly `tensor_core_mma`. JSON `mma_tile_family_id` exactly `tile_mma_shaped`.

JSON array `tile_family_ids` copied from TASK-15 (9 ids) as citations; `tile_size_selected` false; `mma_tile_extents_selected` false.

Required closing sentence under this heading (already locked as `canonical_sentence_layout`; must appear here as well, verbatim, so the checker substring is present in this section’s source — it already appears under Mapping convention; repeating here is allowed and required):

> Instantiating a TASK-15 parallel decomposition as a CUDA ownership axis does not select a layout and does not justify an ordering or tile.

Do not rank attachments by wall time. Do not treat this pairing table as measured CUDA evidence.

### Evaluation instantiation (lock; heading 8)

Immediately under this heading, repeat the winner sentence verbatim:

> This document selects no CUDA mapping winner.

JSON array `evaluation_criterion_ids` in this exact order (7 ids). JSON `n_evaluation_criteria` = 7. JSON `evaluation_usefulness_label` exactly `HYPOTHESIS` for criterion outcomes; the algebra is `DERIVED`; SKU peaks are `UNKNOWN`; nothing is `MEASURED`.

| id | TASK-16 item | Instantiation here |
| --- | ---: | --- |
| `crit_occupancy` | 1 | F1–F8 with hypothesized \(R_t,C_{\text{cta}},T_{\text{cta}}\); SKU UNKNOWN |
| `crit_latency_hiding` | 2 | F9; \(L_{\text{issue}}\) symbolic |
| `crit_wave_quant` | 3 | F10; \(N_{\text{SM}}\) UNKNOWN |
| `crit_intensity_roofline` | 4 | F13 vs F14; \(\Pi_{\text{FMA}}\) / \(\Pi_{\text{TC}}\) / \(\Beta\) UNKNOWN |
| `crit_sync_class` | 5 | `mapping_sync_class_ids` |
| `crit_pipeline_mix` | 6 | `mapping_primary_pipeline_ids` + secondary hypotheses |
| `crit_fusion_delta` | 7 | sign \(\Delta I\) vs sign \(\Delta O\); F9 still holds? HYPOTHESIS |

Required F13/F14 substrings outside the JSON fence:

- `I = F / B` (F13)
- `\Pi \le \min(\Pi_\text{peak}, I \cdot \Beta)` (F14)

Cite intensities `i_mlp_weight_only` 1, `i_lm_head_weight_only` 1, `i_gdn_vs_s_rw` 0.75, `i_attn_core_vs_kv` 6 as **node-level** F13 identities (TASK-06 via TASK-11). Mapping-level \(I\) may differ when split-K rereads weights or fusion drops intermediate bytes; the **sign** of that change is HYPOTHESIS. \(\Pi_{\text{peak}}\) is \(\Pi_{\text{FMA}}\) or \(\Pi_{\text{TC}}\) according to the hypothesized pipeline mix, both `UNKNOWN`. F14 is an upper bound, not a measured point. JSON `roofline_is_bound_not_measurement` true. JSON `n_evaluation_winners_selected` = 0.

### Fusion versus occupancy (lock; heading 9)

JSON array `fusion_hypothesis_ids` copied from TASK-13/14 (22 ids, TASK-12 order). JSON `n_fusion_hypotheses` = 22. JSON `n_fusion_hypotheses_selected` = 0. JSON `fusion_usefulness_label` exactly `HYPOTHESIS`. JSON `fusion_winner_selected` false.

Required F11/F12 substrings outside the JSON fence:

- `R_f \ge \max(R_1, R_2)` (F11)
- `C_f \ge \max(C_1, C_2)` (F12)

Prose required: fused internals raise live ranges (F11, F12); \(O\) may fall; F9 may fail even if \(I\) rises. That implication is `DERIVED` from TASK-16, not a fusion winner. `map_mtp_cta_fused` and intra-node `fuse_*_internals` attachments illustrate criterion 7; they do not select fusion.

JSON `var_splitk_grid_id` exactly `var_splitk_grid`. JSON `n_reduction_variant_hypotheses` = 1. JSON `n_reduction_variant_hypotheses_selected` = 0. `var_splitk_grid` is the HYPOTHESIS that a `red_splitk_cta` mapping instead reduces with `red_grid` if cooperative groups or atomics are available; gated by UNKNOWN capabilities; not a nineteenth mapping.

Mapping-risk hypotheses (every severity is HYPOTHESIS). JSON array `mapping_risk_ids` in this exact order (8 ids). Parallel `mapping_risk_severities`. JSON `n_mapping_risks` = 8.

| ID | Ties to | Severity | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `m_occ_fusion` | F9/F11 | high | Fused internals may drop \(O\) so \(W_{\text{active}}<W_{\text{need}}\) |
| `m_splitk_sync` | `red_splitk_cta` | medium | Split-K adds reduction traffic and barriers |
| `m_mma_sku` | `mma_shapes` | medium | `tensor_core_mma` may be unavailable or shape-mismatched |
| `m_async_absent` | `async_copy_cap` | low | Async copy / TMA may be absent |
| `m_cluster_absent` | `cluster_cap` | low | Grid-cooperative / cluster may be absent |
| `m_decode_prefill_same_map` | TASK-14 views | medium | One mapping may not fit GEMV and GEMM equally; this does **not** select distinct views |
| `m_gdn_serial` | `(17)`–`(18)` | medium | Per-head serial recurrence limits \(T\)-parallelism |
| `m_lm_vocab_wave` | F10, \(V\) | medium | Vocab-out grids may wave-quantize |

JSON `mapping_high_ids`: `m_occ_fusion`. `mapping_medium_ids`: `m_splitk_sync`, `m_mma_sku`, `m_decode_prefill_same_map`, `m_gdn_serial`, `m_lm_vocab_wave`. `mapping_low_ids`: `m_async_absent`, `m_cluster_absent`. JSON `n_mapping_high` = 1, `n_mapping_medium` = 5, `n_mapping_low` = 2. JSON `mapping_risk_severities` `["high","medium","medium","low","low","medium","medium","medium"]`.

### Non-decisions (lock; heading 10)

Prose required: TASK-19 owns measurements and the sitting SKU table. TASK-15 still owns layout **selection** and still leaves the parallel-decomposition justification question unresolved. TASK-12/13/14 still own fusion **winners**. TASK-09 still owns artifact-boundary, ideal-sequence, scale-storage, scale-placement, alignment-grain, and code-bit-order **selection**. TASK-14 still owns decode/prefill **view** selection. TASK-08 still owns recipe **winners**. TASK-07 `activation_dtype_decided` remains false; hypothesized `fma` vs `tensor_core_mma` is a pipeline mix, not a dtype recipe. This mapping space does not change when a TASK-08 recipe is later applied. The ledger open question (which mappings win after benchmarks) remains unresolved. State writes are not optional. Chunkwise GDN is not zero \(S\) traffic.

JSON `activation_dtype_decided` false.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 4 (Per-node mapping alternatives). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 135 instances, or 18 mapping boxes as separate subgraphs beyond the six node types plus an open-winner node. Caption must contain `HYPOTHESIS` and `no winner` (or the winner sentence’s `selects no CUDA mapping winner` as surrounding prose; the fence caption itself must contain `HYPOTHESIS` and `winner`).

Required IDs **inside that fence**: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`, `open`.

JSON `n_diagrams` is 1. `diagram_ids` is `["embed","gated_attn","gated_delta_net","mlp","lm_head","mtp_mix","open"]`.

Suggested topology (not a selected schedule): each node type points at `open`. Caption: HYPOTHESIS CUDA mappings per node; winner unresolved.

### Deferred vision

Visual tokens may replace placeholders on the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger CUDA mappings are **UNKNOWN**. Do not add a vision node type or vision mapping. `vision_interface_is_not_a_node` true. The word `UNKNOWN` must appear in this section.

### Tooling

Create `scripts/check_cuda_design_space.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, CUDA Python, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, \(d_\text{qkv}\), GQA, MAC/byte products, KV/\(C\)/\(S\) bytes, gather/lm-head/mtp-fc bytes. Duplicate TASK-11 `node_type_ids`, TASK-13/14 `stage_kind_ids` and `fusion_hypothesis_ids`, TASK-15 `parallel_decomposition_ids` / `justification_hypothesis_ids` / `tile_family_ids`, and TASK-16 `sku_unknown_symbols` / formula substrings as constants; do not import them.

The ledger **Produces** line names only `docs/architecture/cuda-design-space.md`. The checker is stdlib evidence tooling matching TASK-01–16 and the user-required stdlib checker; it is in scope for this increment.

CLI (cwd = repository root):

```text
python3 scripts/check_cuda_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_cuda_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --cuda-design-space docs/architecture/cuda-design-space.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `linear_conv_kernel_dim`. Derived: \(d_\text{qkv}\), \(g_\text{qa}\), instance counts, cited MAC/byte products, intensities, `weight_bytes_mtp_fc`. Constant fields: canonical sentences, mapping/ownership/reduction/sync/pipeline/estimate/evaluation/risk lists, booleans, SKU-unknown symbols.
- `--json`: print that object to stdout (pretty-printed `json.dumps(..., indent=2)` plus a trailing newline; key order = schema order); run internal asserts listed below; exit 0.
- Default / `--cuda-design-space PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all five canonical sentences plus `canonical_sentence_layout` verbatim, (5) every `node_type_ids`, `mapping_ids`, `ownership_ids`, `reduction_ids`, `estimate_dimension_ids`, `sync_class_ids`, `pipeline_ids`, `mode_fit_ids`, `evaluation_criterion_ids`, `parallel_decomposition_ids`, `justification_hypothesis_ids`, `fusion_hypothesis_ids`, `mapping_risk_ids`, `stage_kind_ids`, `tile_family_ids`, `bottleneck_labels`, and `sku_unknown_symbols` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) the word `UNKNOWN` present in the Deferred vision section, (9) every locked document integer/decimal below present as a decimal or integer substring, (10) the words `HYPOTHESIS` and `unresolved` present, (11) formula tags `(F1)`, `(F7)`, `(F8)`, `(F9)`, `(F10)`, `(F11)`, `(F12)`, `(F13)`, `(F14)` and the required formula substrings present outside the JSON fence, (12) none of the forbidden winner phrases: `winning mapping`, `selected kernel`, `best occupancy`, `should use tensor cores`, `recommend this mapping`, `deviceQuery`, `Quartz kernel`, `llama.cpp kernel`, `GGUF kernel`, `FlashAttention`, `cuBLAS`, `CUTLASS`, `selected winner`, `MMA shape is`, `achieved occupancy`, `this mapping is required`, `selected distinct views`, `prefill requires a distinct view` (allow the substring only inside `not selected kernels` / `not measured winners` / `selects no CUDA mapping winner` / `does not select a layout` / `does not justify`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).
- `--json` and `--cuda-design-space` together: check first, then print JSON on success.
- No flags besides `--config`: same as `--json` (print object, exit 0), matching peer checkers that print JSON by default when the markdown path is omitted.

Do not read safetensor payloads. Do not query a GPU. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks cited TASK-06/11/13/14/15/16 integers and ids against those documents).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `hidden_size==5120`, `intermediate_size==17408`, `vocab_size==248320`, `head_dim==256`
- `num_attention_heads==24`, `num_key_value_heads==4`, `g_qa==6`
- `linear_num_value_heads==48`, `linear_key_head_dim==128`, `linear_value_head_dim==128`
- `linear_conv_kernel_dim==4`, `d_qkv==10240`
- `n_node_types==6`, `n_node_instances_complete==135`
- `n_embed_instances==2`, `n_gated_attn_instances==17`, `n_gated_delta_net_instances==48`, `n_mlp_instances==65`, `n_lm_head_instances==2`, `n_mtp_mix_instances==1`
- `n_mappings==18`, `n_mappings_per_node==3`, `n_mappings_selected==0`
- every node type appears exactly 3 times in `mapping_node_types`
- `n_ownership_classes==4`, `n_reduction_classes==5`, `n_estimate_dimensions==6`, `n_sync_classes==5`, `n_pipeline_ids==6`, `n_mode_fit_ids==3`
- `n_evaluation_criteria==7`, `n_parallel_decompositions==9`, `n_justification_hypotheses==10`, `n_fusion_hypotheses==22`
- `n_mapping_risks==8`, `n_mapping_high==1`, `n_mapping_medium==5`, `n_mapping_low==2`
- `n_cta_T_candidates==4`, `n_cta_T_selected==0`, `cta_T_all_multiples_of_warp is True`
- `N_w==32`, `N_bank==32`
- `n_mappings_using_red_grid==0`, `n_mappings_using_sync_grid==0`
- `n_decode_primary_mappings==3`, `n_prefill_primary_mappings==2`, `n_both_mode_mappings==13`
- `mac_embed==0`, `mac_full_proj_per_layer==104857600`, `mac_attn_coeff_per_full_layer==12288`
- `mac_lin_token_per_layer==118235136`, `mac_mlp_per_layer==267386880`, `mac_lm_head==1271398400`
- `mac_gdn_per_layer==2359296`, `mac_mtp_fc==52428800`, `mac_lin_conv_per_layer==40960`
- `mac_full_proj_per_layer == 24*256*hidden_size + 2*4*256*hidden_size + hidden_size*(24*256)`
- `mac_mlp_per_layer == 3 * intermediate_size * hidden_size`
- `mac_lm_head == vocab_size * hidden_size`
- `mac_gdn_per_layer == 3 * 48 * 128 * 128`
- `mac_mtp_fc == hidden_size * (2 * hidden_size)`
- `mac_lin_conv_per_layer == d_qkv * 4`
- `i_mlp_weight_only==1`, `i_lm_head_weight_only==1`, `i_gdn_vs_s_rw==0.75`, `i_attn_core_vs_kv==6`
- `kv_bytes_all_per_token==69632`, `s_f32_bytes==150994944`, `weight_gather_bytes_per_row==10240`
- `weight_bytes_lm_head==2542796800`, `weight_bytes_mtp_fc==104857600`
- `mapping_winner_selected is False`, `launch_config_selected is False`, `kernel_named is False`
- `ledger_open_question_mapping_winner_closed is False`
- `parallel_decomposition_justifies_layout_selected is False`
- `mma_decomposition_named is True`, `mma_tile_extents_selected is False`, `mma_shapes_unknown is True`
- `evaluation_measured is False`, `sku_limits_unknown is True`
- `cluster_assumed_present is False`, `tma_assumed_present is False`
- `fusion_winner_selected is False`, `decode_prefill_distinct_views_selected is False`
- `analyzes_work and analyzes_storage and analyzes_access and analyzes_synchronization and analyzes_occupancy and analyzes_mode_suitability`
- `multiple_alternatives_per_node is True`, `winner_selected_without_measurements is False`
- `thread_geometry_absent is False`, `cuda_mapping_deferred is False`, `hardware_independent is False`
- `n_diagrams==1`, `n_secondary_pipelines_selected==0`, `n_fusion_hypotheses_selected==0`
- every `parallel_decomposition_ids` entry appears in at least one `mapping_decomposition_ids` list
- `mma_decomposition_id=="par_gemm_d_out"`, `mma_pipeline_id=="tensor_core_mma"`, `mma_tile_family_id=="tile_mma_shaped"`
- `sku_policy=="parameterized_unknown_until_measured_table"`
- `len(sku_unknown_symbols)==17`

Locked document integers/decimals the `--cuda-design-space` check must find:

`5120`, `17408`, `248320`, `10240`, `256`, `128`, `48`, `24`, `4`, `16`, `17`, `64`, `135`, `104857600`, `12288`, `118235136`, `40960`, `2359296`, `267386880`, `1271398400`, `52428800`, `4096`, `69632`, `61440`, `3145728`, `150994944`, `2542796800`, `0.75`

(`10240` is both gather-row bytes and \(d_\text{qkv}\); `104857600` is both full-proj MAC and `weight_bytes_mtp_fc`; substring once is enough.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `head_dim`, `num_attention_heads`, `num_key_value_heads`, `g_qa`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_value_heads`, `linear_conv_kernel_dim`, `d_qkv`, `bytes_bf16`, `bytes_f32`,

`mac_embed`, `mac_full_proj_per_layer`, `mac_attn_coeff_per_full_layer`, `mac_lin_token_per_layer`, `mac_lin_conv_per_layer`, `mac_gdn_per_layer`, `mac_mlp_per_layer`, `mac_lm_head`, `mac_mtp_fc`, `weight_gather_bytes_per_row`, `kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`, `c_bytes_per_layer`, `s_bytes_per_layer`, `s_f32_bytes`, `weight_bytes_lm_head`, `weight_bytes_mtp_fc`, `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_gdn_vs_s_rw`, `i_attn_core_vs_kv`, `bottleneck_labels`, `n_bottleneck_labels`,

`N_w`, `N_bank`, `sku_policy`, `sku_unknown_symbols`, `cta_T_candidates`, `n_cta_T_candidates`, `n_cta_T_selected`, `cta_T_all_multiples_of_warp`,

`node_type_ids`, `n_node_types`, `n_embed_instances`, `n_gated_attn_instances`, `n_gated_delta_net_instances`, `n_mlp_instances`, `n_lm_head_instances`, `n_mtp_mix_instances`, `n_node_instances_complete`,

`ownership_ids`, `n_ownership_classes`, `reduction_ids`, `n_reduction_classes`, `estimate_dimension_ids`, `n_estimate_dimensions`, `sync_class_ids`, `n_sync_classes`, `pipeline_ids`, `n_pipeline_ids`, `mode_fit_ids`, `n_mode_fit_ids`, `consumer_mode_ids`, `n_consumer_modes`, `n_consumer_modes_selected`, `stage_kind_ids`, `n_stage_kinds`,

`mapping_ids`, `n_mappings`, `n_mappings_per_node`, `n_mappings_selected`, `mapping_selected`, `mapping_node_types`, `mapping_ownership_ids`, `mapping_reduction_ids`, `mapping_sync_class_ids`, `mapping_primary_pipeline_ids`, `mapping_secondary_pipeline_ids`, `mapping_mode_fit_ids`, `mapping_work_mac_ids`, `mapping_layout_object_ids`, `mapping_decomposition_ids`, `mapping_usefulness_label`, `n_decode_primary_mappings`, `n_prefill_primary_mappings`, `n_both_mode_mappings`, `n_mappings_using_red_grid`, `n_mappings_using_sync_grid`, `n_secondary_pipeline_hypotheses`, `n_secondary_pipelines_selected`, `attn_prefill_mma_is_secondary_hypothesis`, `gdn_chunk_is_algebraic_equivalent`, `work_partition_identity_present`, `storage_rt_numeric`, `storage_hbm_cited`, `access_usefulness_label`, `coalescing_identity_observed`, `n_access_winners_selected`, `mode_suitability_label`, `n_mode_winners_selected`, `mode_fit_at_T1_grid_T_degenerates`,

`evaluation_criterion_ids`, `n_evaluation_criteria`, `evaluation_usefulness_label`, `n_evaluation_winners_selected`, `roofline_is_bound_not_measurement`, `occupancy_sku_symbols_unknown`, `achieved_occupancy_reported`,

`parallel_decomposition_ids`, `n_parallel_decompositions`, `n_parallel_decompositions_selected`, `decomposition_mapping_ids`, `justification_hypothesis_ids`, `n_justification_hypotheses`, `n_justification_hypotheses_selected`, `justification_usefulness_label`, `tile_family_ids`, `n_tile_families`, `mma_decomposition_id`, `mma_pipeline_id`, `mma_tile_family_id`,

`fusion_hypothesis_ids`, `n_fusion_hypotheses`, `n_fusion_hypotheses_selected`, `fusion_usefulness_label`, `var_splitk_grid_id`, `n_reduction_variant_hypotheses`, `n_reduction_variant_hypotheses_selected`, `red_grid_requires_cooperative_or_atomics`, `sync_grid_requires_cooperative`, `async_copy_optional`,

`mapping_risk_ids`, `mapping_risk_severities`, `mapping_high_ids`, `mapping_medium_ids`, `mapping_low_ids`, `n_mapping_risks`, `n_mapping_high`, `n_mapping_medium`, `n_mapping_low`,

`example_T`, `T_is_stored_length_after_append`, `primary_includes_mtp`,

`decode_prefill_share_artifact`, `decode_prefill_share_graph`, `decode_prefill_distinct_views_selected`, `activations_in_layout_scope`, `activation_live_is_occupancy_symbol`, `hardware_independent`, `cuda_mapping_deferred`, `thread_geometry_absent`, `launch_config_selected`, `mapping_winner_selected`, `kernel_named`, `layout_winner_selected`, `ordering_selected`, `tile_size_selected`, `mma_tile_extents_selected`, `mma_shapes_selected`, `mma_decomposition_named`, `alignment_grain_selected`, `conversion_pipeline_selected`, `parallel_decomposition_selected`, `parallel_decomposition_justifies_layout_selected`, `ideal_byte_sequence_selected`, `artifact_boundary_selected`, `fusion_winner_selected`, `ledger_open_question_mapping_winner_closed`, `ledger_open_question_parallel_decomposition_closed`, `evaluation_measured`, `sku_limits_unknown`, `async_copy_cap_unknown`, `mma_shapes_unknown`, `cluster_cap_unknown`, `occupancy_formulae_instantiated`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_not_zero_s_traffic`, `paper_s_transpose_same_map`, `state_write_not_optional`, `kv_rope_baked_into_k`, `gqa_repeat_not_stored`, `conv_z_does_not_enter_conv`, `weight_unique_counted_once`, `weight_second_w_lm_read_is_hypothesis`, `gguf_is_not_the_runtime_format`, `safetensors_is_source_not_runtime`, `vision_interface_is_not_a_node`, `payloads_restreamed`, `analyzes_work`, `analyzes_storage`, `analyzes_access`, `analyzes_synchronization`, `analyzes_occupancy`, `analyzes_mode_suitability`, `multiple_alternatives_per_node`, `winner_selected_without_measurements`, `cluster_assumed_present`, `tma_assumed_present`, `red_grid_selected`, `activation_dtype_decided`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_mappings`, `canonical_sentence_symbolic`, `canonical_sentence_open_question`, `canonical_sentence_winner`, `canonical_sentence_layout`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. `i_gdn_vs_s_rw` is JSON number `0.75`. `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_attn_core_vs_kv` are JSON ints `1`, `1`, `6`. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. `mapping_secondary_pipeline_ids` is a JSON object keyed by `mapping_ids` with values `null` or a pipeline id. `mapping_layout_object_ids` and `mapping_decomposition_ids` are JSON objects keyed by `mapping_ids` with array values. `decomposition_mapping_ids` is a JSON object keyed by `parallel_decomposition_ids`. `mapping_selected` is a JSON object keyed by `mapping_ids` with all `false`. `stage_kind_ids` match TASK-13/14 exactly. `fusion_hypothesis_ids` match TASK-13/14 exactly. `parallel_decomposition_ids`, `justification_hypothesis_ids`, and `tile_family_ids` match TASK-15 exactly. `sku_unknown_symbols` match TASK-16 exactly.

TASK-13/14 `stage_kind_ids` order (9): `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`. JSON `n_stage_kinds` = 9.

TASK-15 `tile_family_ids` order (9): `tile_none`, `tile_2d_mn`, `tile_1d_row`, `tile_conv_channel`, `tile_kv_t`, `tile_kv_dh`, `tile_s_head`, `tile_s_block`, `tile_mma_shaped`. JSON `n_tile_families` = 9.

Draft-status banner (documentation stage): a blockquote or italic line **before** the first `##`, matching TASK-13–16:

> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.

### Stage split

- **Implementation** writes `scripts/check_cuda_design_space.py` **and** `docs/architecture/cuda-design-space.md`. Runs `py_compile`, `--json`, and `--cuda-design-space` of the markdown after the document exists. Embeds the JSON fence from that live `--json` run. Records command outcomes in this dossier. Does not commit.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner before the first `##`, Authority links to this dossier / `plan.md` evidence policy / checker path / the five dependency documents, heading/JSON fence consistency. Must not change locked IDs, canonical sentences, mapping counts, or SKU policy. Does not invent a winner. Does not edit `plan.md` or the ledger.
- **Verification** independently re-runs the focused commands, reads the document against this dossier (18 mappings, six estimate dimensions, seven criteria, no winner, SKU-UNKNOWN, no kernel inspection), and confirms no `plan.md` edit.
- **Delivery** marks TASK-17 `DONE` after a passing verification.

### Invariants

- Eighteen mappings, three per each of six node types; zero selected.
- Every estimate dimension is present and JSON `analyzes_*` all true.
- Occupancy/intensity/roofline use TASK-16 formulae with UNKNOWN SKU symbols; no sitting-device numbers; no achieved occupancy.
- `red_grid` and `sync_grid` stay in vocabulary and unused as selected mapping classes; cluster/TMA not assumed present.
- MMA decomposition is **named** (`par_gemm_d_out` + `tile_mma_shaped` + `tensor_core_mma`) without selecting `mma_shapes`.
- TASK-15 justification question remains unresolved; TASK-17 mapping-winner question remains unresolved.
- State writes are not optional; chunkwise GDN is not zero \(S\) traffic; GQA repeat is not stored.
- Checker is stdlib-only; JSON fence matches live `--json`.
- If live `text_config` disagrees with a cited integer, the earlier document / config wins.

### Rejected alternatives

- Selecting a mapping because GEMM “looks different” from GEMV: rejected; no MEASURED evidence; TASK-14 left views open; mode-fit labels stay HYPOTHESIS.
- Filling SKU symbols from one GPU datasheet or `deviceQuery`: rejected; sitting limits stay `UNKNOWN`; occupancy stays algebraic.
- Copying Quartz / llama.cpp / CUTLASS / FlashAttention kernels as the mapping catalog: forbidden by `plan.md` until freeze; not design authority.
- Treating higher arithmetic intensity as a sufficient mapping win: rejected; F9 can fail (TASK-16).
- Using grid-cooperative or TMA as a required mapping: rejected; capabilities UNKNOWN.
- Adding a seventh node type for conv or softmax: rejected; conv and softmax stay inside `gated_delta_net` / `gated_attn`.
- Closing TASK-15’s justification question by treating the attachment table as measured evidence: rejected; naming a CUDA axis is not justifying a layout.
- Selecting `tile_mma_shaped` extents from a datasheet: rejected; `mma_shapes` UNKNOWN; `mma_tile_extents_selected` false.
- Reporting numeric occupancy or tok/s: rejected; TASK-19 owns measurements.
- Activation working-buffer layouts as first-class objects: rejected; live ranges are occupancy symbols.
- Importing other `scripts/check_*.py` or restreaming safetensors: rejected.
- uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–16.
- Making TASK-08/09/10/12 silent extra dependencies: rejected; cite packing/fusion through TASK-13/14/15.
- Inspecting `*.cu` kernels to “confirm” mappings: forbidden.

### Discovered ledger work

none.

### Unresolved decisions

`none` (implementation choices for this increment are closed). The ledger open question — which mappings win on target hardware and profiles after benchmarks — is **intentionally unresolved** (`ledger_open_question_mapping_winner_closed` false). That is an acceptance outcome, not a missing dossier decision. TASK-15 parallel-decomposition justification, decode/prefill distinct views, ideal byte sequence, artifact boundary, alignment grain, fusion winners, MMA shapes, and sitting SKU limits remain owned by their tasks.

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/cuda-design-space.md` exists with the twelve locked headings in order, five canonical sentences plus the layout sentence, one Mermaid flowchart, and a JSON fence equal to live checker `--json`.
  - Multiple mapping alternatives per semantic node: 18 mappings, 3 per each of 6 types; `multiple_alternatives_per_node` true; `n_mappings_selected` 0.
  - Six estimate dimensions present; JSON `analyzes_*` all true.
  - TASK-16 criteria 1–7 instantiated symbolically; F1/F7–F14 substrings present; SKU symbols UNKNOWN; `evaluation_measured` false.
  - No selected mapping/kernel/launch/MMA-shape/layout/fusion winner; `mapping_winner_selected` false; `winner_selected_without_measurements` false; forbidden winner phrases absent.
  - MMA decomposition named without selecting extents.
  - Ledger open question remains unresolved.
  - Coupled IDs: none.
- Tests/fixtures to add or change:
  - `scripts/check_cuda_design_space.py` (create)
  - `docs/architecture/cuda-design-space.md` (create; JSON fence is the fixture)
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_cuda_design_space.py
python3 scripts/check_cuda_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_cuda_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --cuda-design-space docs/architecture/cuda-design-space.md
```

- Candidate quality: not required — no model execution or NLL; this increment is CUDA-mapping-space documentation. Mapping **usefulness** is HYPOTHESIS prose, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/cuda-design-space.md
python3 -m py_compile scripts/check_cuda_design_space.py
python3 scripts/check_cuda_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --cuda-design-space docs/architecture/cuda-design-space.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run other `scripts/check_*.py` as a requirement of this task (verifier may spot-check TASK-06/11/13/14/15/16 integers and ids independently).

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/cuda-design-space.md` (create)
  - `scripts/check_cuda_design_space.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-11/13/14/15/16 deliverables unchanged)
- Definition of done: CUDA-design-space document published with locked eighteen unselected mappings (three per node type), six estimate dimensions, seven TASK-16 criteria instantiated with UNKNOWN SKU symbols, named MMA decomposition without selected extents, one Mermaid flowchart, and **no mapping winner**; ledger open question **remains unresolved**; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-17 completion checkboxes can be marked at delivery **with the open question still recorded as unresolved**; fusion, packing selection, distinct views, layout selection, and TASK-19 measurements remain open.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T16:10:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-17.md`. Coupled IDs `none`. Document structure (12 headings), five canonical sentences plus layout sentence, six node types with 18 unselected mappings (3 each), six estimate dimensions, seven TASK-16 evaluation criteria instantiated symbolically, 9 TASK-15 decompositions attached, MMA decomposition named (`par_gemm_d_out` + `tile_mma_shaped` + `tensor_core_mma`) without selecting `mma_shapes`, 22 unselected fusion hypotheses, 8 mapping-risk hypotheses, stdlib checker `scripts/check_cuda_design_space.py`, JSON schema, and acceptance commands are closed. Ledger open question **kept unresolved** (`ledger_open_question_mapping_winner_closed` false; `mapping_winner_selected` false; `winner_selected_without_measurements` false). TASK-15 justification question left unresolved. `docs/architecture/cuda-design-space.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit. No implementation.
- Performance evidence applied: N/A — CUDA mapping-space documentation; mapping usefulness is hypothesis, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_cuda_design_space.py` (stdlib checker: `argparse`/`json`/`math`/`re`/`sys`/`pathlib`; no import of other `scripts/check_*.py`). Instantiates the locked 219-key summary from sitting `text_config`, asserts 18 unselected mappings (3 per node type), six estimate dimensions, seven TASK-16 criteria, named MMA decomposition without extents, SKU-UNKNOWN symbols, and markdown/JSON/Mermaid/canonical-sentence gates.
  - Created `docs/architecture/cuda-design-space.md` with the twelve locked headings, five canonical sentences plus the layout sentence, eighteen unselected mappings, six-way estimates, F1/F7–F14 instantiations with UNKNOWN SKU symbols, nine TASK-15 decomposition attachments, 22 unselected fusion hypotheses, eight mapping-risk hypotheses, one Mermaid flowchart, Deferred vision UNKNOWN, and a JSON fence equal to live `--json`. `n_mappings_selected` 0; `mapping_winner_selected` false; ledger open question left unresolved. No `plan.md` / ledger / TASK-11–16 deliverable edits. No commit.
- Commands:
  - `python3 -m py_compile scripts/check_cuda_design_space.py` — pass
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass; `n_mappings=18`, `n_mappings_per_node=3`, `n_mappings_selected=0`, all six `analyzes_*` true, `mma_decomposition_named=true`, `mma_tile_extents_selected=false`, `sku_limits_unknown=true` (17 symbols), `mapping_winner_selected=false`, `winner_selected_without_measurements=false`, `ledger_open_question_mapping_winner_closed=false`
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md` — pass (headings, live JSON fence equality, one Mermaid flowchart, canonical sentences, locked ids/numbers/formulae, no winner phrases)
  - Combined `--json --cuda-design-space` — pass then print JSON; exit 0
- UTC/time/tokens/cost: `2026-09-20T16:07:37Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent; parent/inherit mapping)
- Changes and evidence:
  - `docs/architecture/cuda-design-space.md` — mechanical pass only. Draft-status banner (`unverified`) already present in TASK-13–16 style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`); left unchanged. Authority table already cross-links this dossier, [`semantic-graph.md`](../semantic-graph.md) (TASK-11), [`decode-plan.md`](../decode-plan.md) (TASK-13), [`prefill-plan.md`](../prefill-plan.md) (TASK-14), [`layout-strategy.md`](../layout-strategy.md) (TASK-15), [`cuda-hardware-model.md`](../cuda-hardware-model.md) (TASK-16), [`model-inventory.md`](../model-inventory.md) (TASK-01) via those docs, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_cuda_design_space.py`](../../../scripts/check_cuda_design_space.py), and plan evidence policy in [`plan.md`](../plan.md) (OBSERVED / DERIVED / HYPOTHESIS / UNKNOWN). Twelve required `##` headings in locked order, five canonical sentences plus layout sentence, one Mermaid flowchart, and the JSON fence left unchanged. No locked IDs, canonical sentences, mapping counts, SKU policy, or winner prose edited. `plan.md` and `task_ledger.md` not edited. No commit.
- Commands:
  - `test -f docs/architecture/cuda-design-space.md` — pass (exit 0).
  - `python3 -m py_compile scripts/check_cuda_design_space.py` — pass (exit 0).
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 219 keys; `n_mappings` 18; `n_mappings_per_node` 3; `n_mappings_selected` 0; `n_estimate_dimensions` 6; all six `analyzes_*` true; `n_evaluation_criteria` 7; `n_parallel_decompositions` 9; `n_fusion_hypotheses` 22; `n_mapping_risks` 8; `n_diagrams` 1; `mma_decomposition_named` true; `mma_tile_extents_selected` false; `sku_limits_unknown` true; `mapping_winner_selected` false; `winner_selected_without_measurements` false; `ledger_open_question_mapping_winner_closed` false; `evaluation_measured` false; `multiple_alternatives_per_node` true; JSON fence source unchanged).
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md` — pass (exit 0; twelve headings, JSON fence, one Mermaid flowchart, five canonical sentences plus layout sentence, locked ids/integers/formulae, `HYPOTHESIS` / `unresolved`; banner did not break the check).
  - Independent JSON fence spot-check (live `--json` object equals fenced JSON in `cuda-design-space.md`, 219 keys deep-equal after parse) — pass.
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T16:08:27Z`; `telemetry_unavailable`

### Verification

- Attempt: 1 (first pass)
- Agent/model: `composer-2.5` (verifier subagent; parent/inherit mapping)
- Diff review:
  - New: `docs/architecture/cuda-design-space.md`, `scripts/check_cuda_design_space.py`, `docs/architecture/tasks/TASK-17.md` (untracked).
  - Modified tracked: `docs/architecture/task_ledger.md` only (`TODO` → `IN PROGRESS`; expected at admission, not delivery).
  - `docs/architecture/plan.md` unchanged. TASK-11/13/14/15/16 deliverables unchanged.
  - Checker is stdlib-only (`argparse`, `json`, `math`, `re`, `sys`, `pathlib`); no import of other `scripts/check_*.py`.
  - Document: twelve locked `##` headings in order; draft-status `unverified` banner before first `##`; five canonical sentences plus layout sentence; eighteen unselected mappings (3 per node type); six estimate dimensions; seven TASK-16 criteria with F1/F7–F14 formulae; nine TASK-15 decomposition attachments; named MMA decomposition (`par_gemm_d_out` + `tile_mma_shaped` + `tensor_core_mma`) without selected extents; 22 unselected fusion hypotheses; 8 mapping-risk hypotheses; one Mermaid flowchart; Deferred vision `UNKNOWN`; no forbidden winner phrases (checker gate).
- Independent raw-record checks:
  - Live `--json` vs fenced JSON in `cuda-design-space.md`: 219 keys, deep-equal after parse.
  - Mapping distribution: 3 per each of 6 node types (`embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`).
  - `mapping_selected`: all 18 values `false`; `n_mappings_selected` 0; `mapping_winner_selected` false; `winner_selected_without_measurements` false; `ledger_open_question_mapping_winner_closed` false.
  - `analyzes_work/storage/access/synchronization/occupancy/mode_suitability`: all true.
  - `mma_decomposition_named` true; `mma_tile_extents_selected` false; `mma_shapes_selected` false; `sku_limits_unknown` true (17 symbols); `evaluation_measured` false; `achieved_occupancy_reported` false.
  - `n_parallel_decompositions` 9; `n_fusion_hypotheses` 22; `n_mapping_risks` 8; `n_diagrams` 1.
- Commands:
  - `test -f docs/architecture/cuda-design-space.md` — pass (exit 0)
  - `python3 -m py_compile scripts/check_cuda_design_space.py` — pass (exit 0)
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 219 keys; critical flags match acceptance)
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md` — pass (exit 0)
  - `python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json --cuda-design-space docs/architecture/cuda-design-space.md` — pass (exit 0)
  - Independent JSON fence spot-check (live `--json` object equals fenced JSON, 219 keys deep-equal) — pass
  - `git diff docs/architecture/plan.md` — empty (unchanged)
- Formatting changed files: none (`uv run ruff format` not required for this increment)
- Verdict: **PASS**
- UTC/time/tokens/cost: `2026-09-20T16:10:00Z`; `telemetry_unavailable`

### Retries and escalation

none.

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-17 only; coupled IDs `none`
- Outcome: TASK-17 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T16:12:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/cuda-design-space.md` (twelve locked headings, five canonical sentences plus layout sentence, eighteen unselected mappings, six estimate dimensions, seven TASK-16 criteria with UNKNOWN SKU symbols, named MMA decomposition without extents, nine TASK-15 decomposition attachments, 22 unselected fusion hypotheses, eight mapping-risk hypotheses, one Mermaid flowchart, JSON fence equal to live `--json`); `scripts/check_cuda_design_space.py` stdlib checker; `plan.md` unchanged; fusion, packing selection, distinct views, layout selection, and TASK-19 measurements remain open; ledger open question kept unresolved (`ledger_open_question_mapping_winner_closed` false; `mapping_winner_selected` false; `winner_selected_without_measurements` false)
- Candidate measured delta: N/A — mapping-space documentation
- Shipping delta: N/A
- Quality result: not required
- Evidence completeness: N/A for performance-evidence checks
- Throughput delta (when applicable): N/A
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: **yes**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: none
