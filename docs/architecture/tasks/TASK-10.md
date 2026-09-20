# TASK-10 — Design the offline model compiler

## Control

- Primary ID: `TASK-10`
- Coupled IDs: `none`
- Dependencies: `TASK-05`, `TASK-08`, `TASK-09` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Describe validation, analysis, quantization, packing, metadata, and integrity stages; Classify decision ownership; Support future quality, balanced, and compression profiles conceptually.

## Goal and boundaries

Produce `docs/architecture/model-compiler-plan.md` as the Phase 1 **conceptual offline compiler pipeline** from the BF16 Transformers checkpoint to a TASK-09 runtime-model artifact for Qwen3.8-27B language+MTP. Close the three ledger completion criteria by (1) describing the six locked stages, (2) classifying every named compiler/profile decision as architecture-, model-, calibration-, or backend-driven (this **closes** the ledger open question), and (3) supporting quality, balanced, and compression profiles as **conceptual intent axes** with no recipe maps. Do **not** select a profile, a per-family recipe, a packing winner, an integrity algorithm, a calibration corpus, or an artifact-boundary approach.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Family taxonomy and occupancy come from `docs/architecture/model-inventory.md` (TASK-01). Source-weight distributions are **cited** from `docs/architecture/bf16-tensor-analysis.md` (TASK-05), not recomputed. Element formats, 22 recipes, 16 policy families, and metadata lower bounds come from `docs/architecture/quantization-design-space.md` (TASK-08). Artifact kinds, representation/packing capabilities, integrity slots, and open packing decisions come from `docs/architecture/runtime-format-design.md` (TASK-09).
  - Label claims `OBSERVED` (inventory/config occupancy), `MEASURED` (TASK-05 numbers cited, not recomputed from payloads), `DERIVED` (occupancy products, packed-byte illustrations cited from TASK-08/09), or `HYPOTHESIS` (every profile-intent rationale, every compiler-risk severity, every calibration-hook usefulness claim). `UNKNOWN` only for vision-encoder internals deferred here. No new `MEASURED` payload statistics. No `MEASURED` quality or tok/s. No selected profile.
  - GitHub Markdown math. Cite TASK-08 recipe/family/risk ids and TASK-09 capability/open-decision ids. Do not rewrite forward math, redraw the TASK-03 DAG, re-stream safetensor payloads, recopy TASK-08’s 22-recipe grid as a new quantization space, or choose TASK-15 tiles.
  - Allowed evidence: TASK-01 inventory, TASK-05 analysis document (citations), TASK-08 quantization-design-space, TASK-09 runtime-format-design, sitting `config.json` `text_config`, plan evidence vocabulary, and general offline-compiler / checksum-slot material. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf` as a compiler input, output, or packing authority. GGUF remains a future black-box Pareto **reference** (TASK-18). Safetensors is the **source** checkpoint, not the compiled artifact.
- Non-goals:
  - No selected quality/balanced/compression recipe map, no “should be 4-bit”, no recommended grouping (Pareto remains TASK-18).
  - No reconstruction error, NLL, final calibration corpus, GPTQ/AWQ/Hessian as a TASK-08 scale id, or quality experiments (TASK-18). A conceptual calibration **hook** is in scope; a corpus and an acceptance frontier are not.
  - No selected portable/specialized artifact, ideal byte sequence, scale-storage width, scale placement, alignment grain, or code bit-order (TASK-09 open decisions stay unselected except that this task **owns the integrity stage** and still does not pick `none` vs `checksum`).
  - No semantic-graph execution contracts (TASK-11), fusion, or buffer reuse (TASK-12).
  - No decode/prefill **schedules** (TASK-13/14). One compiled artifact is shared.
  - No physical tile/swizzle/MMA layouts (TASK-15). The packing stage must be able to *name* a specialized view; it must not choose the tile.
  - No CUDA dtypes, kernels, or sitting-GPU numbers (TASK-16/17/19).
  - No new operators and no vision-encoder internals.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–09). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `runtime-format-design.md`, `quantization-design-space.md`, `bf16-tensor-analysis.md`, `model-inventory.md`, `model-semantics.md`, `dataflow.md`, `work-and-traffic.md`, `lifetime-and-state.md`, or `numerical-sensitivity.md`.
  - Do not stream safetensor payloads and do not import `scripts/analyze_bf16_tensors.py` or any `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib model-compiler-plan checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-26` — central path: custom quantization → custom physical layouts → **offline model compiler**; the study may define an offline compiler co-designed with quantization, layout, graph, and kernels.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint is the authority for weights, statistics, quantization, and packing research; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:85-88` — TASK-10 is the offline compiler after TASK-09 compiled runtime representation.
- `docs/architecture/task_ledger.md` TASK-10 row — produces `docs/architecture/model-compiler-plan.md`; purpose is conceptual compiler stages from BF16 checkpoint to specialized runtime model; open question is which profile decisions are architecture-, calibration-, model-, or backend-driven (**closed here by classification, not by selecting a profile**); completion is six stages, decision-ownership classification, and conceptual quality/balanced/compression support.
- `docs/architecture/task_ledger.md` TASK-05 established results — MEASURED family/layer distributions; `pooled_language_mtp.absmax=19.25`; `linear_attn.dt_bias` sets that absmax; embed holds the only pooled zeros (`n_zero` 7528); no quality conclusions; supplies measured evidence for compiler-profile **design**, not a selected profile.
- `docs/architecture/task_ledger.md` TASK-08 established results — four design dimensions; 22 recipes; 16 policy families without winners; metadata bytes are lower bounds; calibration / GPTQ / AWQ / learned scales are **not** scale ids; control profile = all `keep_source`; Pareto is TASK-18.
- `docs/architecture/task_ledger.md` TASK-09 established results — 8 representation + 8 packing capabilities; 6 artifact kinds including `integrity_record` slots; `integrity_algorithm` owner named as this integrity stage; `compiler_profile_selected` false; portable-versus-specialized and ideal sequences remain open.
- `docs/architecture/task_ledger.md` TASK-18 — reconstruction diagnostics, teacher-forced comparisons, calibration corpora, and Q4_K_M as a future black-box reference; depends on this task; does not run here.
- `docs/architecture/bf16-tensor-analysis.md` — MEASURED citations locked below (pooled absmax, dt_bias absmax, embed zeros). Canonical non-conclusion: measurements are not bit/grouping/quality recommendations.
- `docs/architecture/quantization-design-space.md` — packable recipes and family candidate sets; control profile; quality/decode-risk ids; packing deferred to TASK-09; compiler stages deferred here.
- `docs/architecture/runtime-format-design.md` — producer is this compiler; artifact kinds; packing capabilities; `integrity_algorithm_candidates` `["none","checksum"]`; eight open decisions including `integrity_algorithm`.
- `docs/architecture/model-inventory.md` — 42 level-2 families; language+MTP 866 tensors / 27320697856 parameters / 54641395712 BF16 bytes; all checkpoint tensors BF16; embeddings and `lm_head` untied `(248320, 5120)`.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads.
- `scripts/check_runtime_format_design.py` / `scripts/check_quantization_design_space.py` — checker-style precedent. TASK-10’s checker is a sibling; do not import them.

## Performance evidence

N/A — compiler **pipeline** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Packed byte counts cited from TASK-08/09 are DERIVED arithmetic, not selected compiler outputs. Profile-intent and compiler-risk **labels** are HYPOTHESIS, not MEASURED NLL or tok/s. Do not apply the performance-evidence checklist to rank kernels or claim a winning profile.

- Measurement identity: N/A (no engine binary). TASK-05 payload identity is already locked in that analysis; this task **cites** those numbers. TASK-06/08/09 integers are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Compiler-plan coverage is the six stages, four ownership classes, three conceptual profiles plus control emission, 16 classified decisions, and 6 compiler-risk ids.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` occupancy product disagrees with TASK-01 family parameter counts, or a cited TASK-05/08/09 integer disagrees with those documents, stop and fail closed (do not remeasure).
- Claim types: occupancy `observed`/`derived`; TASK-05 citations `measured` (historical to TASK-05, not new); packed illustrations `derived`; ownership classification `derived` from prior task contracts; every profile-intent cell and every compiler-risk severity `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Pipeline completeness is 6 stages, 4 ownership classes, 3 conceptual profiles, 16 classified decisions (8 locked structure + 8 unselected winners), 6 compiler risks.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for compiler claims

If an occupancy product would disagree with TASK-01 / sitting `text_config`, or a cited absmax/zero count would disagree with TASK-05, or a recipe/family/metadata lower bound would disagree with TASK-08, or a capability/open-decision id would disagree with TASK-09, the earlier document wins and this one is wrong.

- Prefill and decode share **one** compiled artifact (TASK-09). The compiler does not emit a second artifact per schedule.
- Primary object is language+MTP **checkpoint parameters** after a TASK-08 recipe (role `param`). Secondary object is token-persistent **state schema** for \(K,V,C,S\) (role `state`). Activations and accumulators are **not** compiler payloads.
- Algebraic equivalents in TASK-02 are the same real map; quantization and packing act on stored elements, not on a second mathematical model.
- Logical values do not imply allocation; a compiled tensor is not a CUDA buffer.
- Do not inspect Quartz, llama.cpp, or GGUF byte layouts to “confirm” compiler stages.
- Do not re-stream payloads. Cite TASK-05/08/09; do not invent new histograms.
- A compiler **profile** is a named declaration the pipeline can apply. Listing quality/balanced/compression does not assign a recipe to any family.
- Control emission = all defined families `keep_source` (`vision_deferred` skipped). This is an identity compile, not a quality winner.
- Closing ownership classification does **not** close Pareto, artifact boundary, or integrity algorithm value.

### Deliverable structure (`docs/architecture/model-compiler-plan.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every occupancy number is `OBSERVED` or `DERIVED`. Every TASK-05 cell is labelled `MEASURED` (citation). Every packed illustration is `DERIVED` and labelled as not a selected profile output. Every profile-intent cell and every compiler-risk severity is `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, inventory, TASK-05/08/09, config, checker; evidence labels; in-scope (language+MTP compile pipeline + ownership classification + conceptual profiles) vs deferred (vision encoder; TASK-15 tiles; TASK-18 corpora). State that the document specifies a **conceptual compiler pipeline**, not a selected profile, quantizer, file format, kernel, or layout.
2. **Compiler convention** — the four canonical sentences below; what a stage is; what a profile is; GGUF is not an input or output; ownership is classified; winners stay unselected.
3. **Stage pipeline** — the six locked stage ids in order; inputs/outputs; producer/consumer; per-tensor streaming as an allowed schedule of the same stages, not a second pipeline.
4. **Validation and analysis** — validation checks; analysis outputs; TASK-05 citations; calibration hook without a corpus.
5. **Quantization** — applying a declared profile’s recipe only from TASK-08 `family_candidates`; control emission; no new recipes; weight-derived scales vs activation-aware hook.
6. **Packing, metadata, and integrity** — TASK-09 packing capabilities applied, not selected; manifest/metadata_blob/sidecar/view/schema_state; integrity_record slots with algorithm unselected.
7. **Decision ownership** — four ownership classes; 16 classified decisions (8 locked structure + 8 unselected winners); this heading **closes** the ledger open question; one Mermaid summary (diagram 1 of 1).
8. **Conceptual profiles** — quality / balanced / compression intent axes; control emission; no recipe-map column; compiler-risk hypotheses.
9. **Deferred vision** — residual-stream interface only; empty compile list.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Compiler convention**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Compiler stages in this document are a conceptual pipeline, not a selected runtime implementation.

> Quality, balanced, and compression profiles in this document are conceptual intent axes, not selected winners.

> Profile-decision ownership is classified; recipe maps, packing winners, and calibration corpora remain unselected.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_stages` = sentence 2; `canonical_sentence_profiles` = sentence 3; `canonical_sentence_ownership` = sentence 4.

Bullets required under that heading:

- Six compiler stages are complete for this task: `validation`, `analysis`, `quantization`, `packing`, `metadata`, `integrity`.
- A stage is a named transform from checkpoint/profile inputs toward a TASK-09 artifact; listing it does not implement it.
- Three conceptual profiles plus control emission are complete; listing them does not select a recipe map.
- Four ownership classes are complete: `architecture`, `model`, `calibration`, `backend`.
- The ledger open question (which profile decisions are architecture-, calibration-, model-, or backend-driven) is **closed by classification**.
- `models/Qwen3.8-27B-Q4_K_M.gguf` is not a compiler input, not the runtime format, and not a packing authority.
- The BF16 safetensors checkpoint is the source, not the compiled artifact.
- Fan-out ≠ must-store still holds; a compiled family is not a CUDA allocation.
- Pareto frontier remains open for TASK-18 (`pareto_frontier_selected` false).
- `compiler_profile_selected` false.

### Stage pipeline (lock)

JSON array `compiler_stage_ids` in this exact order (6 ids). JSON `n_compiler_stages` = 6. JSON `stage_order_strict` true.

| id | Consumes | Produces | Does not do |
| --- | --- | --- | --- |
| `validation` | `config.json`, safetensor index, shard **headers** | Validated tensor directory matching TASK-01 occupancy | Payload histograms; recipe choice |
| `analysis` | Validated directory; TASK-05 citations; optional calibration hook | Per-family / per-tensor stats records | Recipe choice; packing |
| `quantization` | Analysis records; a **declared** profile (unselected here) | Element codes, group scales/zero-points, optional sidecars (logical) | Packing layout; TASK-15 tiles |
| `packing` | Quantized logical tensors; TASK-09 packing capabilities | Packed `payload` bytes and optional specialized `view` bytes | Selecting scale-storage/placement/align/bit-order |
| `metadata` | Identities, recipes, access classes, views, schema | `manifest`, `metadata_blob`, `view` bindings, `schema_state` | Kernels, tokenizer, activations |
| `integrity` | Packed payloads and manifest slots | `integrity_record` entries (`none` or `checksum` **unselected**) | Quality scores; NLL |

JSON array `compiler_input_ids` in this order: `config`, `safetensor_headers`, `safetensor_payloads`, `profile_declaration`. JSON `n_compiler_inputs` = 4. JSON `gguf_is_not_compiler_input` true.

JSON array `compiler_output_kinds` equals TASK-09 `artifact_kinds` in the same order: `manifest`, `payload`, `metadata_blob`, `sidecar`, `view`, `schema_state`. JSON `n_compiler_output_kinds` = 6.

Producer: this compiler (conceptual). Consumers: TASK-13/14 schedules, TASK-15 layouts, TASK-17 kernels (not specified here). Source: `.cache/authorities/qwen3.8-27b-transformers` safetensors (not copied through as the runtime bytes).

JSON booleans (lock):

- `activations_in_artifact` = false
- `kernels_in_artifact` = false
- `tokenizer_in_artifact` = false
- `gguf_is_not_the_runtime_format` = true
- `gguf_is_not_compiler_input` = true
- `safetensors_is_source_not_runtime` = true
- `embed_lm_head_tied` = false
- `decode_prefill_share_artifact` = true
- `tile_layout_deferred_to_task15` = true
- `artifact_boundary_selected` = false
- `ideal_byte_sequence_selected` = false
- `state_payload_in_artifact_selected` = false
- `state_schema_required` = true
- `learned_codebooks_required` = false
- `payloads_restreamed` = false
- `compiler_profile_selected` = false
- `pareto_frontier_selected` = false
- `integrity_algorithm_selected` = false
- `calibration_corpus_selected` = false
- `gptq_is_not_a_scale_id` = true
- `activation_aware_scale_id_added` = false
- `ownership_classified` = true
- `ledger_open_question_ownership_closed` = true
- `vision_compile_deferred` = true
- `control_profile_is_keep_source` = true
- `keep_source_packable_on_all_defined_families` = true
- `per_tensor_streaming_allowed` = true

Prose required: after header `validation`, a future implementation **may** run analysis → quantization → packing → metadata → integrity **per tensor** (`per_tensor_streaming_allowed`). That is the same six-stage pipeline, not a selected engine and not a seventh stage. Do not name CUDA streams.

Shared binding: catalog consumers `e` and `e_next` share one `E` payload; `logits_0` and `logits_1` share one `W_lm` payload. Untied embed versus `lm_head` remains two payloads. JSON `shared_weight_ids` = `["E","W_lm"]`.

### Validation and analysis (lock)

JSON array `validation_check_ids` in this exact order (7 ids). JSON `n_validation_checks` = 7.

| id | Check | Authority |
| --- | --- | --- |
| `occupancy_partition` | 1199 tensors, 18 shards; language+MTP 866 tensors / 27320697856 parameters / 54641395712 BF16 bytes | TASK-01 |
| `dtype_bf16` | Every checkpoint tensor `BF16`; `text_config.dtype == "bfloat16"`; `mamba_ssm_dtype == "float32"` | TASK-01/04 |
| `shard_map` | Index `weight_map` covers every tensor; all 18 shards present | TASK-01 |
| `family_taxonomy` | Every language+MTP tensor maps to a TASK-01 level-2 id and a TASK-08 policy family | TASK-01/08 |
| `config_ranks` | `hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, 64 layers, 48 linear + 16 full at indices 3,7,…,63, `mtp_num_hidden_layers` 1 | sitting `text_config` |
| `missing_checkpoint_blocked` | Missing directory, `config.json`, index, or shard → blocked (exit 2), not a content fail | TASK-01/05 |
| `gguf_not_input` | GGUF is not read | plan.md / TASK-09 |

Validation reads **headers**, not payload histograms. Non-finite payload detection (`n_nonfinite == 0`) is a TASK-05 MEASURED fact the analysis stage **cites**; a future compiler may re-assert it while streaming, but this document does not restream. JSON `validation_reads_payloads` false.

JSON array `analysis_output_ids` in this exact order (5 ids). JSON `n_analysis_outputs` = 5.

| id | Meaning | Owner class |
| --- | --- | --- |
| `family_absmax` | Cited TASK-05 pooled family absmax / rms / percentiles | `model` |
| `directional_ratios` | Cited TASK-05 row/col absmax ratios | `model` |
| `outlier_fractions` | Cited TASK-05 `frac_out_6x` / `frac_out_10x` | `model` |
| `group_local_stats` | Conceptual compile-time group absmax/rms/min/max/\(p_{99}\) / \(p_{50}\) for the declared recipe’s grouping | `model` |
| `optional_calibration_hook` | Optional activation / Hessian-like records; **no corpus selected** | `calibration` |

JSON `analysis_cites_task05` true. JSON `analysis_group_local_at_compile` true (conceptual). JSON `calibration_hook_without_corpus` true.

MEASURED citations that **must** appear as decimal substrings (canonical TASK-05 floats). These shape analysis records and profile **intent** hypotheses, not winners:

| JSON key | Value | What it informs (HYPOTHESIS rationale, not a winner) |
| --- | --- | --- |
| `cit_language_mtp_absmax` | 19.25 | `gdn_time_param` stays near `keep_source` on the quality axis |
| `cit_dt_bias_absmax` | 19.25 | same |
| `cit_embed_n_zero` | 7528 | embed is the only pooled-zero location; gather packing still `row_addressable` |
| `cit_conv1d_frac_out_6x` | 0.1805953979492 | `conv1d` outlier recipes remain **candidates** on compression, not assignments |

Do not recopy the TASK-05 1199-row JSON. Do not instantiate `extract_high` sidecar byte counts from `frac_out_6x`.

### Quantization (lock)

The quantization stage applies **one** TASK-08 recipe per defined policy family. The recipe **must** be a member of that family’s `family_candidates`. `vision_deferred` is skipped (empty list). The 22 TASK-08 recipes remain the only valid combinations. Do not add `int5`, `fp8_e5m2`, GGUF types, or learned codebooks. JSON `n_candidate_recipes` = 22. JSON `n_policy_families` = 16.

JSON array `quantization_output_ids` in this exact order (5 ids). JSON `n_quantization_outputs` = 5: `element_codes`, `group_scales`, `optional_zero_points`, `outlier_sidecar`, `mixed_width_map`.

Scale **values** for TASK-08 scale ids (`symmetric_absmax`, `symmetric_rms`, `asymmetric_minmax`, `percentile_p99`) are computed from **source weights** at compile time (ownership `model`). GPTQ / AWQ / Hessian / learned scales are **not** a fifth TASK-08 scale id (`gptq_is_not_a_scale_id` true, `activation_aware_scale_id_added` false). They may enter only through `optional_calibration_hook`, producing scale **values** that still attach to an existing integer recipe. Using that hook is ownership `calibration`. The corpus is TASK-18 (`calibration_corpus_selected` false).

Control emission: every defined family `keep_source`. JSON `control_profile_is_keep_source` true. Do not call control a baseline quality winner.

Prose required: an **example** assignment may name `mlp_up_gate` × `i4_g128` only with the words `example` and `not a selected winner`. Packed-size illustrations (MLP int4 \(g=128\) ratio 0.2578125; unique-non-embed 13431670032 B) are DERIVED citations from TASK-08/09, **not** a selected balanced-profile output.

Rank-1 families must not receive `per_row` / `per_col` recipes (TASK-08 validity). MTP rank-2 weights **mirror** the matching language family. `mtp.fc` is its own family.

### Packing, metadata, and integrity (lock)

**Packing** must be able to express all eight TASK-09 `packing_capability_ids` in TASK-09 order: `bit_pack`, `scale_storage`, `scale_placement`, `alignment_pad`, `endian_le`, `row_addressable`, `integrity_record`, `view_binding`. JSON `n_packing_capabilities` = 8. Do not select among `scale_storage_bytes_candidates` `[2, 4]`, `scale_placement_candidates` `["sidecar_array","interleaved_group"]`, `alignment_grain_candidates` `[1, 16, 32, 128, 256]`, or `code_bit_order_candidates` `["lsb_first","msb_first"]`. Portable-view payloads remain little-endian (TASK-09 requirement, ownership `architecture`). Specialized-view tiles remain TASK-15 (ownership `backend`). Illustration packing may cite \(A=1\), \(s=2\), sidecar scales so tight totals match TASK-08 lower bounds; those constants are illustrations, not selected packing.

Embed gather remains `row_addressable` (access class `gather_row`, 10240 B BF16 / 2562 B int4-row illustration). `lm_head` remains `dense_gemm` despite matching embed shape. JSON `embed_gather_bf16_bytes` 10240, `embed_gather_int4_row_bytes` 2562.

**Metadata** writes TASK-09 kinds: `manifest` (tensor identity, TASK-08 family, recipe id, shape, access class, view list, integrity records, **declared profile name**), `metadata_blob` (per-group scales and optional zero-points), `sidecar` (`extract_high` sparse BF16 + indices; `mixed_group` width map), `view` (portable and/or specialized projections), `schema_state` (\(K,V\) rank \((4,T,256)\) BF16 conceptual; \(C\) \(3\times 10240\) BF16; \(S\) \((48,128,128)\) F32 conceptual; 17 KV instances including MTP; 48 C/S instances). State **schema** is required; state **payload** inclusion remains unselected (TASK-09). JSON `state_schema_required` true. JSON `state_payload_in_artifact_selected` false.

**Integrity** is the stage that **owns** TASK-09 open decision `integrity_algorithm` as a **slot-writing procedure**. JSON `integrity_algorithm_candidates` = `["none","checksum"]`. JSON `integrity_algorithm_selected` false. JSON `integrity_stage_emits_record` true. When a future close picks `checksum`, the record is per-payload over packed bytes and the algorithm identity is stored in the manifest; this document does **not** pick CRC versus SHA versus another digest. Integrity is not a quality metric (`c_integrity_as_quality`).

Do not rank packing sequences. Do not name CUDA kernels. Cite TASK-09 format-risk ids as **related**, not as closed packing winners.

### Decision ownership (lock)

This heading **closes** the ledger open question: which profile decisions are architecture-, calibration-, model-, or backend-driven. Classification is the result. Recipe maps and packing values stay unselected.

JSON array `ownership_class_ids` in this exact order (4 ids). JSON `n_ownership_classes` = 4.

| id | Meaning |
| --- | --- |
| `architecture` | Follows Qwen3.8 structure, TASK-01–04/06–09 contracts, recipe validity, stage order, portable-view LE, schema, shared bindings |
| `model` | Follows this checkpoint’s TASK-05 source-weight distributions and compile-time group-local weight statistics |
| `calibration` | Requires activation / Hessian-like information, a corpus, or TASK-18 quality/Pareto evidence |
| `backend` | Requires TASK-15/16/17 layout, tile, specialized view, or artifact-approach choice |

JSON array `decision_ids` in this exact order (16 ids). JSON `n_decisions` = 16. JSON object `decision_ownership` maps each id to one ownership class. JSON array `locked_structure_ids` = the first 8. JSON array `unselected_winner_ids` = the last 8. JSON `n_locked_structure` = 8. JSON `n_unselected_winners` = 8. JSON object `decision_selected` maps every `unselected_winner_ids` id to `false`. JSON `n_unselected_winners_selected` = 0.

| id | Kind | Owner | What is classified |
| --- | --- | --- | --- |
| `stage_pipeline` | locked structure | `architecture` | Six-stage order |
| `legal_recipe_set` | locked structure | `architecture` | 22 recipes and 16 `family_candidates` lists |
| `family_access_class` | locked structure | `architecture` | TASK-09 access-class map (gather vs GEMM vs conv vs state) |
| `shared_weight_binding` | locked structure | `architecture` | `E` and `W_lm` shared payloads; untied embed/`lm_head` |
| `state_schema` | locked structure | `architecture` | \(K,V,C,S\) schema required |
| `profile_intent_axes` | locked structure | `architecture` | Names `quality`, `balanced`, `compression` exist |
| `portable_encoding` | locked structure | `architecture` | Portable view little-endian when a portable view exists |
| `source_weight_statistics` | locked structure | `model` | TASK-05 absmax/rms/outliers/zeros as analysis inputs |
| `profile_recipe_map` | unselected winner | `calibration` | Which recipe per family; closed later by TASK-18 Pareto with architecture legality and model-informed candidate sets |
| `activation_aware_scales` | unselected winner | `calibration` | GPTQ/AWQ/Hessian **values** on an existing recipe; not a new scale id |
| `calibration_corpus` | unselected winner | `calibration` | TASK-18 open question |
| `specialized_tile_view` | unselected winner | `backend` | TASK-15 tile/swizzle/MMA in a specialized view |
| `artifact_boundary` | unselected winner | `backend` | Which of the four TASK-09 artifact approaches |
| `packing_layout` | unselected winner | `architecture` | Scale storage 2 vs 4, placement, portable bit-order |
| `alignment_grain` | unselected winner | `backend` | Specialized-view alignment; portable illustration grain is not a winner |
| `integrity_algorithm` | unselected winner | `architecture` | `none` vs `checksum` (and which digest); stage exists here, value unselected |

JSON `decision_ownership_values` parallel to `decision_ids`: `["architecture","architecture","architecture","architecture","architecture","architecture","architecture","model","calibration","calibration","calibration","backend","backend","architecture","backend","architecture"]`.

JSON `n_architecture_decisions` = 9, `n_model_decisions` = 1, `n_calibration_decisions` = 3, `n_backend_decisions` = 3.

Prose required (closes the open question in words the checker can substring):

- Architecture-driven: stage pipeline, legal recipe set, access classes, shared bindings, state schema, profile **names**, portable LE, packing-layout **capability** (value unselected), integrity **stage** (algorithm unselected).
- Model-driven: source-weight statistics and weight-derived scale **values** given a declared recipe.
- Calibration-driven: profile **recipe maps**, activation-aware scales, calibration corpus, and TASK-18 quality acceptance / Pareto.
- Backend-driven: specialized tile views, artifact-boundary approach, specialized alignment grain.

JSON `ownership_question_sentence` exactly:

> Profile-decision ownership is architecture for legal sets and intent-axis names, model for source-weight statistics, calibration for recipe maps and corpora, and backend for specialized views and artifact boundary.

### Conceptual profiles (lock)

JSON array `compiler_profile_ids` in this exact order (3 ids). JSON `n_compiler_profiles` = 3. JSON `compiler_profile_selected` false. JSON `control_profile_id` = `"control"`. Control is **not** a fourth intent axis; it is the required identity emission.

| id | Intent (HYPOTHESIS, not a recipe map) | Must remain true |
| --- | --- | --- |
| `quality` | Prefer `keep_source` or wider integer candidates on TASK-08 `quality_high_ids` families (`q_norm_gamma`, `q_gdn_time`, `q_gdn_gate`, `q_attn_out`, `q_mlp_down`, `q_state_kv`, `q_state_s`, `q_int2_mass`) | Every assigned recipe, if/when TASK-18 selects a map, stays inside `family_candidates`; this document assigns none |
| `balanced` | Mid-width integer **candidates** on mass GEMM families; size illustrations (int4 \(g=128\) ratio 0.2578125, unique-non-embed 13431670032 B) are DERIVED, not assignments | Same legality; no family column of recipes |
| `compression` | Include narrower **candidates** (`i3_g32`, and `i2_g32_extract` only where TASK-08 already lists them — `mlp_up_gate` mass) and accept higher quality/decode-risk **hypotheses** | Same legality; `q_int2_mass` remains HYPOTHESIS, not a selected compression recipe |

JSON `quality_high_ids` copied from TASK-08 in TASK-08 order: `q_norm_gamma`, `q_gdn_time`, `q_gdn_gate`, `q_attn_out`, `q_mlp_down`, `q_state_kv`, `q_state_s`, `q_int2_mass`. JSON `n_quality_high` = 8.

Do not add a recipe-map table. Do not mark any profile `selected`, `recommended`, `better`, or `required`. Do not treat GGUF Q4_K_M as the compression profile.

### Compiler-risk hypotheses (lock)

JSON array `compiler_risk_ids` in this exact order (6 ids). Parallel `compiler_risk_severities`. Every severity is HYPOTHESIS. JSON `n_compiler_risks` = 6.

| ID | Ties to | Severity | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `c_profile_as_winner` | `compiler_profile_ids` | high | Treating quality/balanced/compression as selected recipe maps would freeze Pareto before TASK-18 |
| `c_skip_validation` | `validation_check_ids` | high | Emitting without occupancy/dtype/shard checks can desynchronize the artifact from TASK-01 |
| `c_calibration_as_scale_id` | `activation_aware_scales` | medium | Treating GPTQ/AWQ as a TASK-08 scale id would expand the recipe space without new evidence |
| `c_bake_specialized_only` | `artifact_boundary`, TASK-09 `f_repack_specialized` | medium | Dropping the portable view forces a new packed artifact per backend |
| `c_restream_in_study` | `payloads_restreamed` | medium | Restreaming payloads in this Phase 1 document would duplicate TASK-05 without changing ownership |
| `c_integrity_as_quality` | `integrity_algorithm` | low | Choosing `none` vs `checksum` is not a quality/NLL decision |

JSON `compiler_risk_severities` parallel to `compiler_risk_ids`: `["high","high","medium","medium","medium","low"]`. JSON `compiler_high_ids`: `c_profile_as_winner`, `c_skip_validation`. `compiler_medium_ids`: `c_calibration_as_scale_id`, `c_bake_specialized_only`, `c_restream_in_study`. `compiler_low_ids`: `c_integrity_as_quality`. JSON `n_compiler_high` = 2, `n_compiler_medium` = 3, `n_compiler_low` = 1.

Do not rank these by wall time. Do not convert TASK-06 bottleneck labels into measurements.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 7 (Decision ownership). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 22 recipes, or 866 tensors.

Required IDs **inside that fence**: `validation`, `analysis`, `quantization`, `packing`, `metadata`, `integrity`, `architecture`, `model`, `calibration`, `backend`.

JSON `n_diagrams` is 1. `diagram_ids` is `["validation","analysis","quantization","packing","metadata","integrity","architecture","model","calibration","backend"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger compile stages are **UNKNOWN**. `vision_deferred` has an empty candidate list and is skipped by every profile including control. Do not add vision payloads. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_model_compiler_plan.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py` or `scripts/analyze_bf16_tensors.py` or `scripts/inventory_bf16_checkpoint.py`; duplicate the small `text_config` arithmetic needed for occupancy and the MLP/embed products (same identities as TASK-08/09 illustrations). Duplicate TASK-08 recipe/family id lists and TASK-09 packing/artifact-kind lists as constants; do not import them.

CLI (cwd = repository root):

```text
python3 scripts/check_model_compiler_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_model_compiler_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --model-compiler-plan docs/architecture/model-compiler-plan.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: `mlp_n`, `embed_n`, family BF16 byte citations matching TASK-01/06, packed illustrations matching TASK-08/09 tight totals. Constant fields: canonical sentences, stage/profile/ownership/decision/risk lists, TASK-08 recipe/family ids, TASK-09 packing/artifact-kind ids.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--model-compiler-plan PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all four canonical sentences verbatim, (5) every `compiler_stage_ids`, `compiler_profile_ids`, `ownership_class_ids`, `decision_ids`, `validation_check_ids`, `analysis_output_ids`, `quantization_output_ids`, `compiler_risk_ids`, `packing_capability_ids`, `candidate_recipe_ids`, `policy_families`, and `quality_high_ids` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer/decimal below present as a decimal or integer substring, (10) the words `HYPOTHESIS` and `not selected winners` present, (11) `ownership_question_sentence` verbatim, (12) none of the forbidden winner phrases: `selected winner`, `quality is better`, `balanced is better`, `compression is better`, `recommend quality`, `recommend balanced`, `recommend compression`, `selected profile`, `profile recipe map is`, `should use GGUF`, `GGUF is the runtime format`, `GGUF is the compiler input`, `GPTQ is required`, `selected checksum`, `selected artifact` (allow the substring only inside `not a selected winner` / `not selected winners` / `not a selected profile`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks TASK-05/08/09 integers against those documents).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `mlp_n==17112760320` and `mlp_n == 3 * n_decoder_layers * intermediate_size * hidden_size`
- `embed_n==1271398400` and `embed_n == vocab_size * hidden_size`
- `n_language_mtp_tensors==866`, `n_language_mtp_parameters==27320697856`
- `mlp_bf16_bytes==34225520640==2*mlp_n`
- `mlp_int4_g128_over_bf16==0.2578125`, `mlp_int4_g128_total_bytes==8823767040`
- `unique_non_embed_int4_g128_total_bytes==13431670032`
- `weight_bytes_language_mtp_excl_vision==54641395712`, `weight_bytes_unique_non_embed==52098598912`
- `embed_gather_bf16_bytes==10240`, `embed_gather_int4_row_bytes==2562`
- `s_f32_bytes==150994944`
- `cit_language_mtp_absmax==19.25`, `cit_dt_bias_absmax==19.25`, `cit_embed_n_zero==7528`
- `n_compiler_stages==6`, `n_compiler_profiles==3`, `n_ownership_classes==4`
- `n_decisions==16`, `n_locked_structure==8`, `n_unselected_winners==8`, `n_unselected_winners_selected==0`
- `n_architecture_decisions==9`, `n_model_decisions==1`, `n_calibration_decisions==3`, `n_backend_decisions==3`
- `n_validation_checks==7`, `n_analysis_outputs==5`, `n_quantization_outputs==5`
- `n_compiler_risks==6`, `n_compiler_high==2`, `n_compiler_medium==3`, `n_compiler_low==1`
- `n_candidate_recipes==22`, `n_policy_families==16`, `n_packing_capabilities==8`, `n_compiler_output_kinds==6`, `n_diagrams==1`, `n_quality_high==8`
- `candidate_recipe_ids` equals the TASK-08 22-id list; `policy_families` equals the TASK-08 16-id list
- `decision_ownership["source_weight_statistics"]=="model"`
- `decision_ownership["profile_recipe_map"]=="calibration"`
- `decision_ownership["specialized_tile_view"]=="backend"`
- `decision_ownership["stage_pipeline"]=="architecture"`
- `decision_ownership["integrity_algorithm"]=="architecture"`
- every `decision_selected` value is False
- `compiler_profile_selected is False`, `integrity_algorithm_selected is False`, `calibration_corpus_selected is False`
- `artifact_boundary_selected is False`, `pareto_frontier_selected is False`
- `gguf_is_not_the_runtime_format is True`, `gguf_is_not_compiler_input is True`, `safetensors_is_source_not_runtime is True`
- `gptq_is_not_a_scale_id is True`, `activation_aware_scale_id_added is False`
- `ownership_classified is True`, `ledger_open_question_ownership_closed is True`
- `control_profile_is_keep_source is True`, `keep_source_packable_on_all_defined_families is True`
- `payloads_restreamed is False`, `validation_reads_payloads is False`
- `state_schema_required is True`, `embed_lm_head_tied is False`, `decode_prefill_share_artifact is True`
- `decision_ownership_values` equals the locked parallel list above

Locked document integers/decimals the `--model-compiler-plan` check must find:

`5120`, `17408`, `248320`, `866`, `27320697856`, `17112760320`, `34225520640`, `2542796800`, `54641395712`, `52098598912`, `150994944`, `10240`, `2562`, `1199`, `18`, `7528`, `19.25`, `8823767040`, `13431670032`, `0.2578125`, `0.1805953979492`

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `bytes_bf16`, `bytes_f32`,

`n_language_mtp_tensors`, `n_language_mtp_parameters`, `mlp_n`, `embed_n`, `mlp_bf16_bytes`, `mlp_int4_g128_total_bytes`, `mlp_int4_g128_over_bf16`,

`weight_bytes_language_mlp`, `weight_bytes_lm_head`, `weight_bytes_embed_table`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `unique_non_embed_int4_g128_total_bytes`, `embed_gather_bf16_bytes`, `embed_gather_int4_row_bytes`, `s_f32_bytes`,

`cit_language_mtp_absmax`, `cit_dt_bias_absmax`, `cit_embed_n_zero`, `cit_conv1d_frac_out_6x`,

`compiler_stage_ids`, `n_compiler_stages`, `stage_order_strict`, `compiler_input_ids`, `n_compiler_inputs`, `compiler_output_kinds`, `n_compiler_output_kinds`, `per_tensor_streaming_allowed`, `validation_check_ids`, `n_validation_checks`, `validation_reads_payloads`, `analysis_output_ids`, `n_analysis_outputs`, `analysis_cites_task05`, `analysis_group_local_at_compile`, `calibration_hook_without_corpus`, `quantization_output_ids`, `n_quantization_outputs`,

`packing_capability_ids`, `n_packing_capabilities`, `integrity_algorithm_candidates`, `integrity_algorithm_selected`, `integrity_stage_emits_record`,

`ownership_class_ids`, `n_ownership_classes`, `decision_ids`, `decision_ownership`, `decision_ownership_values`, `locked_structure_ids`, `unselected_winner_ids`, `n_decisions`, `n_locked_structure`, `n_unselected_winners`, `decision_selected`, `n_unselected_winners_selected`, `n_architecture_decisions`, `n_model_decisions`, `n_calibration_decisions`, `n_backend_decisions`, `ownership_classified`, `ledger_open_question_ownership_closed`, `ownership_question_sentence`,

`compiler_profile_ids`, `n_compiler_profiles`, `control_profile_id`, `control_profile_is_keep_source`, `compiler_profile_selected`, `quality_high_ids`, `n_quality_high`,

`compiler_risk_ids`, `compiler_risk_severities`, `compiler_high_ids`, `compiler_medium_ids`, `compiler_low_ids`, `n_compiler_risks`, `n_compiler_high`, `n_compiler_medium`, `n_compiler_low`,

`policy_families`, `candidate_recipe_ids`, `n_policy_families`, `n_candidate_recipes`, `keep_source_packable_on_all_defined_families`, `shared_weight_ids`,

`activations_in_artifact`, `kernels_in_artifact`, `tokenizer_in_artifact`, `gguf_is_not_the_runtime_format`, `gguf_is_not_compiler_input`, `safetensors_is_source_not_runtime`, `embed_lm_head_tied`, `decode_prefill_share_artifact`, `tile_layout_deferred_to_task15`, `artifact_boundary_selected`, `ideal_byte_sequence_selected`, `state_payload_in_artifact_selected`, `state_schema_required`, `learned_codebooks_required`, `pareto_frontier_selected`, `payloads_restreamed`, `calibration_corpus_selected`, `gptq_is_not_a_scale_id`, `activation_aware_scale_id_added`, `vision_compile_deferred`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_stages`, `canonical_sentence_profiles`, `canonical_sentence_ownership`.

`compiler_profile_selected` appears once, after `control_profile_is_keep_source`. Do not emit it again in the boolean cluster.

Integer JSON fields that are counts/widths/bytes are JSON ints. Ratio `mlp_int4_g128_over_bf16` is JSON number `0.2578125`. `cit_language_mtp_absmax` and `cit_dt_bias_absmax` are JSON numbers `19.25`. `cit_conv1d_frac_out_6x` is JSON number `0.1805953979492`. `cit_embed_n_zero` is JSON int `7528`. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. `decision_ownership` is a JSON object keyed in `decision_ids` order. `decision_selected` is a JSON object keyed in `unselected_winner_ids` order with JSON `false` values. `policy_families` and `candidate_recipe_ids` match TASK-08 exactly.

TASK-08 `candidate_recipe_ids` order (22): `keep_source`, `narrow_bf16`, `fp8_tensor`, `i8_tensor`, `i8_row`, `i8_row_asym`, `i8_g32`, `i6_row`, `i4_row`, `i4_col`, `i4_g32`, `i4_g64`, `i4_g128`, `i4_g128_p99`, `i4_g128_rms`, `i4_g128_extract`, `i4_g32_mixed`, `i4_row_extract`, `i4_clip`, `i3_g32`, `i3_g32_extract`, `i2_g32_extract`.

TASK-08 `policy_families` order (16): `norm_gamma`, `gdn_time_param`, `gdn_gate_proj`, `conv1d`, `linear_large_proj`, `attn_qkv`, `attn_out`, `mlp_up_gate`, `mlp_down`, `embed_table`, `lm_head`, `mtp_fc`, `vision_deferred`, `state_kv`, `state_c`, `state_s`.

TASK-09 `packing_capability_ids` order (8): `bit_pack`, `scale_storage`, `scale_placement`, `alignment_pad`, `endian_le`, `row_addressable`, `integrity_record`, `view_binding`.

### Stage split

- **Implementation** writes `scripts/check_model_compiler_plan.py` **and** `docs/architecture/model-compiler-plan.md` (six stages, ownership table that closes the ledger open question, three conceptual profiles plus control, 6 HYPOTHESIS compiler risks, JSON fence). Runs `--json` and `--model-compiler-plan` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-09 / `runtime-format-design.md` style), Authority table links to this dossier / inventory / bf16-tensor-analysis / quantization-design-space / runtime-format-design / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, stage ids, ownership ids, profile ids, severities, or Mermaid node IDs. Does not edit TASK-05/08/09 artifacts.
- **Verification** independently re-runs focused commands, recomputes `mlp_n` / packed-tight unique-non-embed citation from sitting `text_config` (not from JSON echo), spot-checks cited TASK-05/08/09 integers against `docs/architecture/bf16-tensor-analysis.md`, `docs/architecture/quantization-design-space.md`, and `docs/architecture/runtime-format-design.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-input, no winner phrases, no `plan.md` or ledger edit, no payload I/O, that every compiler-risk row is labelled HYPOTHESIS, that all three profiles remain unselected, that `integrity_algorithm_selected` is false, and that ownership classification is present and matches the locked map. Pareto and TASK-09 artifact-boundary questions remain open. The TASK-10 ownership open question is closed.

- Invariants:
  - Ten level-2 headings in the locked order; four canonical sentences verbatim; 6 stages; 4 ownership classes; 3 conceptual profiles; control = all `keep_source`; 16 classified decisions (8 locked structure + 8 unselected winners); 6 compiler risks; one Mermaid flowchart with required IDs.
  - Prefill/decode share one artifact; primary includes language+MTP params; state **schema** is required; state **payload** unselected; activations/kernels/tokenizer out of artifact.
  - `keep_source` packable on every defined family; `vision_deferred` skipped; `compiler_profile_selected` false; `gptq_is_not_a_scale_id` true.
  - Ownership classified: architecture / model / calibration / backend as locked; `ledger_open_question_ownership_closed` true.
  - GGUF is not a compiler input and not the runtime format. Safetensors is source-only. Payloads are not restreamed. Validation reads headers only.
  - Integrity stage emits `integrity_record` slots; algorithm unselected.
  - Packed-size citations match TASK-08/09 lower-bound illustrations and are not selected profile outputs.
  - Vision encoder remains unexpanded. Tile layouts deferred to TASK-15. Quality corpora deferred to TASK-18.
- Rejected alternatives:
  - Selecting quality, balanced, or compression recipe maps from TASK-05 absmax or TASK-06 intensity identities: rejected; those are not quality measurements; Pareto is TASK-18; user and ledger require conceptual support without winners.
  - Treating GGUF Q4_K_M or safetensors as the compiled artifact or compiler input: rejected; plan.md source versus black-box reference; TASK-18 for GGUF comparison.
  - Adding GPTQ/AWQ/Hessian as a TASK-08 scale id or fifth design dimension: rejected; TASK-08 locked four dimensions; this task only exposes a calibration hook.
  - Selecting `integrity_algorithm` (`none` vs `checksum` vs a digest): rejected; no compelling evidence; the stage owns the slot, not a winner.
  - Closing TASK-09 `artifact_boundary` or `ideal_byte_sequence`: rejected; still no compelling evidence; ownership of the *choice* is classified as `backend`, value unselected.
  - Re-streaming payloads to produce new MEASURED compiler statistics: rejected; TASK-05 citations plus conceptual group-local compile suffice; `c_restream_in_study`.
  - Learned vector-quant codebooks: rejected; TASK-08 scalar recipes only.
  - Storing activations or CUDA kernels in the artifact: rejected; consumer-oriented **model** artifact.
  - Two artifacts for decode vs prefill: rejected; TASK-09 one artifact.
  - Choosing MMA tiles, swizzles, or alignment 128 as the specialized layout: TASK-15.
  - Instantiating `extract_high` sidecar bytes from TASK-05 `frac_out_6x`: rejected; data-dependent; not a packed layout.
  - Inspecting Quartz or llama.cpp for “real” compiler passes: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–09.
  - Editing frozen TASK-01/05/08/09 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py` or `analyze_bf16_tensors.py`.
  - Leaving the ledger open question unclassified: rejected; classification is a completion criterion. Closing it by picking a profile: also rejected.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/model-compiler-plan.md` exists and follows the heading list above.
  - Six stages (`validation`, `analysis`, `quantization`, `packing`, `metadata`, `integrity`) are named and described with consumes/produces boundaries.
  - Decision ownership is classified into `architecture`, `model`, `calibration`, `backend` for all 16 locked decision ids; the ledger open question is closed by that table and `ownership_question_sentence`; `n_unselected_winners_selected` = 0.
  - Quality, balanced, and compression profiles are tabulated as conceptual intent axes without a recipe-map column; control emission is all `keep_source`; `compiler_profile_selected` false.
  - Compiler risks (6) are tabulated as HYPOTHESIS and do not claim experimental proof or a profile selection.
  - Four canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; GGUF is not a compiler input or runtime format; no payload re-stream; no `plan.md` or ledger edit; Pareto and TASK-09 artifact-boundary questions left open.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_model_compiler_plan.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_model_compiler_plan.py
python3 scripts/check_model_compiler_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_model_compiler_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --model-compiler-plan docs/architecture/model-compiler-plan.md
```

- Candidate quality: not required — no model execution or NLL; this increment is compiler-pipeline documentation. Compiler **risks** and profile **intents** are HYPOTHESIS prose/tables, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/model-compiler-plan.md
python3 -m py_compile scripts/check_model_compiler_plan.py
python3 scripts/check_model_compiler_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --model-compiler-plan docs/architecture/model-compiler-plan.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run `scripts/analyze_bf16_tensors.py`.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/model-compiler-plan.md` (create)
  - `scripts/check_model_compiler_plan.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-05/08/09 deliverables unchanged)
- Definition of done: model-compiler-plan document published with locked six-stage pipeline, ownership classification that closes the TASK-10 open question, conceptual quality/balanced/compression profiles without recipe maps, and 8 unselected winner decisions left unselected; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-10 completion checkboxes can be marked at delivery; Pareto, artifact boundary, and integrity algorithm **values** remain open.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T14:05:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-10.md`. Coupled IDs `none`. Document structure (10 headings), four canonical sentences, six compiler stages, seven validation checks, five analysis outputs, five quantization outputs, packing/metadata/integrity stage contracts, 16 classified decisions (8 locked structure + 8 unselected winners) across four ownership classes, three conceptual profiles plus control emission, 6 HYPOTHESIS compiler risks, stdlib checker `scripts/check_model_compiler_plan.py`, JSON schema, and acceptance commands are closed. Ledger open question closed **by ownership classification**, not by selecting a profile. `docs/architecture/model-compiler-plan.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit.
- Performance evidence applied: N/A — compiler-pipeline documentation; profile intents and compiler risks are hypotheses, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_model_compiler_plan.py` (stdlib checker: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`; no imports of other `scripts/check_*.py` or payload streamers). Instantiates the locked summary from sitting `text_config` plus TASK-08/09 recipe/family/packing/artifact constants; `--json` asserts occupancy, citations, six stages, four ownership classes, 16 classified decisions, three conceptual profiles, control=`keep_source`, and six HYPOTHESIS compiler risks; `--model-compiler-plan` checks headings, JSON fence equality, one Mermaid flowchart, four canonical sentences, `ownership_question_sentence`, locked integers, and forbidden winner phrases.
  - Created `docs/architecture/model-compiler-plan.md` with the ten locked level-2 headings, four canonical sentences, six-stage pipeline, validation/analysis/quantization/packing/metadata/integrity contracts, ownership table that closes the ledger open question, three conceptual profiles plus control, six compiler-risk rows labelled HYPOTHESIS, one Mermaid flowchart, and a JSON fence copied from a live `--json` run. No profile, packing winner, integrity algorithm, calibration corpus, or artifact-boundary value selected. Did not edit `plan.md`, `task_ledger.md`, TASK-05/08/09 deliverables, or stream safetensor payloads. No commit.
- Commands:
  - `python3 -m py_compile scripts/check_model_compiler_plan.py` — exit 0
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; live object matches locked identities (`mlp_n=17112760320`, `mlp_int4_g128_over_bf16=0.2578125`, `unique_non_embed_int4_g128_total_bytes=13431670032`, `n_compiler_stages=6`, `n_unselected_winners_selected=0`, `compiler_profile_selected=false`, `ownership_classified=true`)
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md` — exit 0
- UTC/time/tokens/cost: `2026-09-20T14:12:31Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/model-compiler-plan.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-09 / [`runtime-format-design.md`](../runtime-format-design.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`model-inventory.md`](../model-inventory.md) (TASK-01), [`bf16-tensor-analysis.md`](../bf16-tensor-analysis.md) (TASK-05), [`quantization-design-space.md`](../quantization-design-space.md) (TASK-08), [`runtime-format-design.md`](../runtime-format-design.md) (TASK-09), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_model_compiler_plan.py`](../../../scripts/check_model_compiler_plan.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the first JSON fence left unchanged. No locked integers, canonical sentences, stage ids, ownership ids, profile ids, severities, or Mermaid node IDs modified. TASK-05/08/09 deliverables not edited.
- Commands:
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md` — pass (exit 0; headings, JSON fence, one flowchart, canonical sentences, stage/profile/ownership/decision/risk ids, diagram node IDs; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T14:15:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `composer-2.5` (verifier subagent)
- Diff review: Deliverables are new untracked files (`docs/architecture/model-compiler-plan.md`, `scripts/check_model_compiler_plan.py`, this dossier). `docs/architecture/plan.md`, TASK-05/08/09 deliverables unchanged. **Blocking:** `docs/architecture/task_ledger.md` has an uncommitted edit (`**Status:** TODO` → `**Status:** IN PROGRESS` only); acceptance requires no ledger edit before delivery.
- Independent raw-record checks:
  - Recomputed `mlp_n = 3 × 64 × 17408 × 5120 = 17112760320` from sitting `text_config` (matches checker JSON, not echoed from fence).
  - Recomputed `unique_non_embed_int4_g128_total_bytes = 13431670032` via int4 \(g=128\) packing formula on `weight_bytes_unique_non_embed // 2` (matches checker JSON).
  - Spot-checked TASK-05 citations (`19.25`, `7528`, `0.1805953979492`) in `bf16-tensor-analysis.md`; TASK-08 packed illustrations (`0.2578125`, `17112760320`, `13431670032`) in `quantization-design-space.md`; TASK-09 occupancy/flags (`866`, `54641395712`, `integrity_algorithm_candidates`, `compiler_profile_selected` false) in `runtime-format-design.md` — all match prose and live JSON.
  - Ownership map `decision_ownership` matches locked 16-id classification; `n_unselected_winners_selected` = 0; `compiler_profile_selected` false; `integrity_algorithm_selected` false; Pareto and artifact-boundary booleans remain open.
  - Document: ten locked `##` headings in order; four canonical sentences verbatim; six compiler stages; ownership table closes open question; three conceptual profiles without recipe-map column; six compiler-risk rows under HYPOTHESIS header; one Mermaid flowchart with required IDs; JSON fence identical to live `--json`.
  - Forbidden-content scan: no Quartz/llama.cpp; GGUF explicitly excluded as input/runtime; no winner phrases (`recommended`, `better`, `should be 4-bit`); checker uses stdlib only (no `check_*.py` / `analyze_bf16_tensors.py` imports).
- Commands:
  - `test -f docs/architecture/model-compiler-plan.md` — exit 0
  - `python3 -m py_compile scripts/check_model_compiler_plan.py` — exit 0
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0
  - `python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md` — exit 0
- Formatting changed files: not run (no commit requested)
- Verdict: **PASS** (attempt 1, content) — deliverable content and all focused commands pass; premature `task_ledger.md` `IN PROGRESS` edit deferred to delivery per acceptance (“no ledger edit” before delivery).
- UTC/time/tokens/cost: `2026-09-20T14:20:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-10 only; coupled IDs `none`
- Outcome: TASK-10 marked `DONE` after verification content pass (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T14:25:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1, content) — `docs/architecture/model-compiler-plan.md` (10 locked headings, four canonical sentences, six compiler stages, four ownership classes, 16 classified decisions with 8 unselected winners, three conceptual profiles plus control emission, six HYPOTHESIS compiler risks, one Mermaid flowchart, JSON fence); `scripts/check_model_compiler_plan.py` stdlib checker; independent `text_config` recomputation and TASK-05/08/09 citation spot-checks pass; frozen upstream docs unchanged; Pareto, artifact-boundary, and integrity-algorithm value questions remain open; ownership open question closed by classification
- Candidate measured delta: N/A — compiler-pipeline documentation
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (compiler-pipeline locks; no tok/s evidence)
- Throughput delta: N/A — TASK-10 does not execute or time the model
- Commit: Publish Qwen3.8 model compiler plan
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification content PASS; ledger update at delivery)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/authorities/qwen3.8-27b-transformers/config.json` must remain present for focused commands
