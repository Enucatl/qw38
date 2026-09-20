# TASK-15 — Design physical tensor layouts from consumers

## Control

- Primary ID: `TASK-15`
- Coupled IDs: `none`
- Dependencies: `TASK-09`, `TASK-13`, `TASK-14` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Analyze logical dimensions, consumers, access, tiling, alignment, and conversion; Include GDN, convolution, and KV persistent state; Avoid claims of optimality before CUDA analysis.

## Goal and boundaries

Produce `docs/architecture/layout-strategy.md` as the Phase 1 **consumer-driven candidate physical layouts** for Qwen3.8-27B language+MTP **weights and persistent state**. Close the three ledger completion criteria by (1) analyzing logical dimensions, consumers, access, tiling, alignment, and conversion for every layout object, (2) including GDN \(S\), depthwise convolution \(W^{\text{conv}}\) / \(C\), and KV \(K,V\) as first-class objects, and (3) making **no** optimality claim before CUDA analysis. Fill TASK-09 `seq_specialized_tile` with a **named candidate space** (orderings, tile families, conversion hypotheses). **Keep** the ledger open question (which planned parallel decompositions justify each candidate ordering and tile) **unresolved**. Do **not** select an ordering, tile extent, alignment grain, conversion pipeline, MMA shape, decode/prefill view split, or ideal byte sequence.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Artifact capabilities, seven consumer sequences, `view_binding`, portable little-endian, alignment-grain candidates, and `seq_specialized_tile` as a named-but-unfilled sequence come from `docs/architecture/runtime-format-design.md` (TASK-09); do not close portable-versus-specialized, ideal-sequence, scale-storage, scale-placement, alignment-grain, or artifact-boundary. Decode consumers, one-token GEMV, populated incoming state, and stage sequence attachments come from `docs/architecture/decode-plan.md` (TASK-13). Prefill consumers, GEMM-over-\(T\), triangular KV, zeros incoming, named tiling **axes**, and unselected distinct views come from `docs/architecture/prefill-plan.md` (TASK-14). Do not add node types, catalog IDs, sequences, or access classes as **required** contracts.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established), `DERIVED` (axis names and ranks from TASK-02 via TASK-09/13/14, byte identities cited from TASK-04/06, group-payload grains cited from TASK-09, divisibility of locked widths by tile-extent candidates), or `HYPOTHESIS` (every ordering/tile **usefulness**, every conversion-cost **usefulness**, every parallel-decomposition **justification**, every layout-risk severity). `UNKNOWN` only for vision-encoder internals deferred here. No `MEASURED` tok/s or NLL. No selected layout, tile, view split, or kernel.
  - GitHub Markdown math. Cite TASK-02 equation tags via TASK-13/14 contracts, TASK-06 MAC/byte integers, TASK-09 sequence / access-class / packing ids, TASK-13/14 stage-kind ids. Do not rewrite forward math, recopy TASK-06 MAC tables as a new work study, recopy TASK-09 packing illustrations as a new format study, recopy TASK-13/14 serial orders as a new schedule, or instantiate TASK-16 occupancy algebra.
  - Allowed evidence: TASK-09 runtime-format, TASK-13 decode-plan, TASK-14 prefill-plan, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier. TASK-01/02/03/04/06/08/11/12 integers and ids already cited by those three documents may be **cited** through them. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No TASK-16 occupancy/fusion-versus-occupancy algebra (not a listed dependency; this document names candidate layouts, not CUDA mappings). No TASK-17 CUDA mappings.
  - Hardware-independent layouts: axis orderings, named tile **families**, alignment grains, conversion points. No thread geometry, warps, SMs, CUDA dtypes, kernel names, streams, shared-memory tiles as selected objects, or sitting-GPU numbers. Naming `tile_mma_shaped` as an unselected family is not selecting an MMA instruction.
- Non-goals:
  - No layout, ordering, tile, alignment, conversion, or view **winner**. Listing a candidate or attaching a TASK-09 sequence is not selecting it. `n_orderings_selected` = 0. `tile_size_selected` false. `n_justification_hypotheses_selected` = 0. `ideal_byte_sequence_selected` false. `decode_prefill_distinct_views_selected` false. `ledger_open_question_parallel_decomposition_closed` false.
  - No optimality, “best tile”, or “this ordering is required for coalescing” claim. CUDA analysis is TASK-17. SKU MMA shapes remain TASK-16 `UNKNOWN`.
  - No CUDA mapping alternatives (TASK-17) and no sitting-device fill-in (TASK-16).
  - No decode or prefill **schedule** rewrite (TASK-13/14 already published). Cite their consumers; do not change serial order.
  - No quantization recipe winners, artifact-boundary winner, or compiler-stage rewrite (TASK-08/09/10). This document may **constrain** specialized-view alignment as a capability restatement; it must not close TASK-09 `alignment_grain`.
  - No activation working-buffer layouts (`activations_in_layout_scope` false). Ledger purpose is weights and persistent state. Prefill’s \(T\) axis on activations is a **consumer** of weight/state layouts, not a new artifact payload.
  - No quality/NLL experiments (TASK-18) and no tok/s (TASK-19).
  - No new operators, extra catalog IDs, extra access classes, or extra consumer sequences.
  - No generic cuBLAS / CUTLASS cookbook and no GGUF block layout as authority.
  - No peak-memory claim and no summed CUDA live-set.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–14). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `runtime-format-design.md`, `decode-plan.md`, `prefill-plan.md`, `work-and-traffic.md`, `quantization-design-space.md`, `semantic-graph.md`, `materialization-and-fusion.md`, `dataflow.md`, `lifetime-and-state.md`, `model-semantics.md`, `model-inventory.md`, `cuda-hardware-model.md`, or any TASK-08/10/11/12 deliverable.
  - Do not import other `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib layout-strategy checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places custom physical layouts after quantization and the compiled format, and consumer-driven candidate layouts after independent decode/prefill plans and before CUDA mappings.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:89-92` — TASK-13 and TASK-14 independently schedule decode and prefill, **leading to** consumer-driven candidate layouts in TASK-15; TASK-17 CUDA-maps nodes **and layouts**.
- `docs/architecture/task_ledger.md` TASK-15 row — produces `docs/architecture/layout-strategy.md`; purpose is candidate consumer-driven layouts for weights and persistent state; open question is which planned parallel decompositions justify each candidate ordering and tile (**kept unresolved here**); completion is the six-way analysis, GDN/conv/KV inclusion, and no optimality before CUDA.
- `docs/architecture/task_ledger.md` TASK-09 established results — seven consumer sequences without an ideal sequence; `ideal_byte_sequence_selected` false; one compiled artifact for prefill and decode; `view_binding` capability; `seq_specialized_tile` named with layout deferred here; portable LE required for the portable view; specialized views may swizzle; `alignment_grain_candidates` `[1, 16, 32, 128, 256]` unselected; `tile_layout_deferred_to_task15` true **there**.
- `docs/architecture/task_ledger.md` TASK-13 established results — nine decode stage kinds; GEMV consumers; populated incoming \((K,V,C,S)\); stage sequence attachments; `layout_selected` false **there**.
- `docs/architecture/task_ledger.md` TASK-14 established results — same nine stage kinds over length-\(T\); GEMM consumers; zeros incoming; five fundamental differences including `diff_tiling`; `tile_size_selected` false; `decode_prefill_distinct_views_selected` false; representation tradeoffs unresolved.
- `docs/architecture/task_ledger.md` TASK-16/17 — TASK-16 is DONE and **not** a listed dependency; occupancy algebra must not rank tiles. TASK-17 consumes this candidate space; do not perform CUDA mappings here.
- `docs/architecture/runtime-format-design.md` — seven `consumer_sequence_ids`; seven `access_class_ids`; `family_access_class`; packing grains; dual-view size illustration F; format risks `f_gather_stride` / `f_unpack_portable` / `f_3bit_shift`.
- `docs/architecture/decode-plan.md` — decode GEMV consumers; `seq_state_s_dense` on GDN; `seq_lm_head_full`; `seq_gather_row`; incoming populated state.
- `docs/architecture/prefill-plan.md` — prefill GEMM / causal-MM / triangular-KV consumers; tiling **axes** named; extents unselected; views unselected.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for ranks, layer counts, and cited byte products. Do not read safetensor payloads.
- `scripts/check_runtime_format_design.py` / `scripts/check_decode_plan.py` / `scripts/check_prefill_plan.py` — checker-style precedent. TASK-15’s checker is a sibling; do not import them.

## Performance evidence

N/A — layout **candidate-space** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Byte figures are TASK-04/06/09 citations plus DERIVED rank×dtype and divisibility arithmetic, not MEASURED traffic. Every ordering/tile/conversion/justification **usefulness** is HYPOTHESIS, not MEASURED. Do not apply the performance-evidence checklist to rank kernels or claim a winning layout.

- Measurement identity: N/A (no engine binary). TASK-09/13/14 integers are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Layout coverage is 7 layout objects (= TASK-09 access classes), 6 analysis dimensions, 3 persistent-state coverages (GDN / conv / KV), 14 unselected orderings, 9 unselected tile families, 9 unselected parallel decompositions, 4 unselected conversion hypotheses, 10 unselected justification hypotheses, 8 layout-risk hypotheses.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` rank or a cited TASK-04/06/09 byte disagrees with TASK-09/13/14, stop and fail closed (do not invent axes).
- Claim types: ranks and grains `observed`/`derived`; divisibility `derived`; every ordering/tile/conversion/justification **usefulness** `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Layout-space completeness is the six analysis dimensions, GDN/conv/KV inclusion, and the open question left unresolved without an optimality claim.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for layout claims

If a rank, access class, sequence id, stage kind, family mapping, or cited byte would disagree with TASK-09/13/14 or sitting `text_config`, the earlier document / config wins and this one is wrong.

- Prefill and decode share **one** compiled artifact (TASK-09) and **one** semantic graph (TASK-11 via the schedules). This document names **candidate layouts** of weights and persistent state for **both** schedule consumers. Distinct **views** remain unselected (`decode_prefill_distinct_views_selected` false). Listing a decode-GEMV ordering and a prefill-GEMM ordering is not selecting dual views.
- Primary objects are language+MTP **checkpoint parameters** after a TASK-08 recipe (still unselected) and token-persistent **state** \(K,V,C,S\). Activations and accumulators are **not** layout objects here (`activations_in_layout_scope` false).
- Algebraic equivalents in TASK-02 are the same real map: GQA-as-repeat does **not** store repeated KV; paper \(S^\top\) (19) is the same map as (17)–(18); chunkwise GDN is not zero \(S\) traffic; conv delay is 3 stored vectors, not a length-4 buffer with a current-token slot as extra math.
- Logical values do not imply allocation; a candidate layout is not a CUDA buffer. Checkpoint orientation \((d_\text{out},d_\text{in})\) is the **source** rank, not a required physical order (`logical ≠ physical` still holds).
- Unique weight bytes are counted **once** per complete decode/prefill. Layout does not restream the model.
- Do not inspect Quartz, llama.cpp, or GGUF byte layouts to “confirm” tiles.
- Do not select an ordering, tile extent, MMA shape, alignment grain, conversion pipeline, or parallel decomposition. Do not import TASK-16 occupancy algebra to rank candidates.
- Fill `seq_specialized_tile` as a **capability space** the format can store. Do not select it as the ideal byte sequence.

### Deliverable structure (`docs/architecture/layout-strategy.md`)

Use these **level-2 headings in this exact order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Ordering/tile/conversion/justification **usefulness** cells are `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, runtime-format, decode-plan, prefill-plan, inventory via those docs, config, checker; evidence labels; in-scope (language+MTP weight + persistent-state candidate layouts + six-way analysis + GDN/conv/KV) vs deferred (vision encoder; TASK-17 CUDA; activation working buffers). State that the document specifies a **hardware-independent candidate layout space**, not kernels and not a selected layout.
2. **Layout convention** — the five canonical sentences (exact text below); what a layout object / ordering / tile family is; open question stays open; no optimality.
3. **Logical dimensions** — seven layout objects, axes, ranks, family mapping. Completes analysis dimension `logical_dimensions`.
4. **Consumers** — decode vs prefill consumers per object; stage attachments cited. Completes analysis dimension `consumers`.
5. **Access** — TASK-09 access classes and sequences; `row_addressable`; GQA vs stored KV. Completes analysis dimension `access`.
6. **Candidate orderings and tiling** — 14 orderings and 9 tile families, all unselected; extent candidates; divisibility identities. Completes analysis dimensions `tiling` (and names orderings).
7. **Alignment and conversion** — TASK-09 grains cited unselected; four conversion hypotheses unselected; specialized-view constraint capability without closing `alignment_grain`. Completes analysis dimensions `alignment` and `conversion`.
8. **GDN, convolution, and KV persistent state** — three required `###` subsections. Completes ledger checkbox 2.
9. **Parallel decompositions** — nine decompositions and ten justification hypotheses, all unselected; this heading **keeps** the ledger open question unresolved.
10. **Work citations and non-decisions** — cited bytes and bottleneck labels; what TASK-17/09/12/14 still own.
11. **Deferred vision** — residual-stream interface only.
12. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Layout convention**, include these **five canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Candidate layouts in this document are consumer-driven orderings and tiles, not selected winners and not CUDA mappings.

> Alignment and conversion in this document are named capabilities and hypotheses, not a selected pack-to-layout pipeline.

> Which planned parallel decompositions justify each candidate ordering and tile remains open until CUDA analysis.

> This document makes no optimality claim before CUDA analysis.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_candidates` = sentence 2; `canonical_sentence_conversion` = sentence 3; `canonical_sentence_open_question` = sentence 4; `canonical_sentence_optimality` = sentence 5.

Sentence 4 **keeps** the ledger open question unresolved in checker-substring form. JSON `ledger_open_question_parallel_decomposition_closed` false. Sentence 5 **closes** the third completion criterion as a prohibition, not as a selected layout.

Bullets required under that heading:

- Seven layout objects are complete for this task and equal the seven TASK-09 `access_class_ids` (`n_layout_objects` 7).
- A layout object is a named rank + axis set with candidate orderings and tile families. Listing a candidate is not selecting it (`n_orderings_selected` 0, `tile_size_selected` false).
- Analysis dimensions are complete: logical dimensions, consumers, access, tiling, alignment, conversion (`n_analysis_dimensions` 6).
- GDN, convolution, and KV are included (`n_persistent_state_coverages` 3).
- Prefill and decode share one artifact; distinct views remain unselected (`decode_prefill_distinct_views_selected` false).
- Activations are out of layout scope (`activations_in_layout_scope` false). \(T\) on prefill GEMM is a consumer axis of weight layouts.
- Unique weights are counted once (`weight_unique_counted_once` true).
- Fan-out ≠ must-store and logical ≠ physical still hold.
- Primary coverage includes MTP KV (17 instances) and 48 GDN/conv instances. Hardware mapping is TASK-17. Fusion winners remain TASK-12 hypotheses. Ideal byte sequence remains TASK-09 open. `seq_specialized_tile` candidates are named here without selecting that sequence.
- No thread geometry (`thread_geometry_absent` true). No optimality claim (`optimality_claim_absent` true).

JSON booleans (lock true unless noted):

- `decode_prefill_share_artifact` = true
- `decode_prefill_share_graph` = true
- `decode_prefill_distinct_views_selected` = false
- `activations_in_layout_scope` = false
- `hardware_independent` = true
- `cuda_mapping_deferred` = true
- `thread_geometry_absent` = true
- `layout_winner_selected` = false
- `ordering_selected` = false
- `tile_size_selected` = false
- `mma_tile_extents_selected` = false
- `alignment_grain_selected` = false
- `specialized_alignment_grain_selected` = false
- `specialized_view_may_constrain_alignment` = true
- `conversion_pipeline_selected` = false
- `parallel_decomposition_selected` = false
- `ideal_byte_sequence_selected` = false
- `artifact_boundary_selected` = false
- `seq_specialized_tile_candidates_named` = true
- `tile_layout_deferred_to_task15` = false (this task owns the candidate space)
- `ledger_open_question_parallel_decomposition_closed` = false
- `parallel_decomposition_justifies_layout_selected` = false
- `optimality_claim_absent` = true
- `gdn_primary_is_recurrent_eq_17` = true
- `chunkwise_not_zero_s_traffic` = true
- `paper_s_transpose_same_map` = true
- `state_write_not_optional` = true
- `kv_rope_baked_into_k` = true
- `gqa_repeat_not_stored` = true
- `conv_z_does_not_enter_conv` = true
- `portable_payload_little_endian` = true
- `gguf_is_not_the_runtime_format` = true
- `safetensors_is_source_not_runtime` = true
- `embed_lm_head_tied` = false
- `lm_head_uses_dense_gemm_object` = true
- `embed_uses_gather_row_object` = true
- `weight_unique_counted_once` = true
- `analyzes_logical_dimensions` = true
- `analyzes_consumers` = true
- `analyzes_access` = true
- `analyzes_tiling` = true
- `analyzes_alignment` = true
- `analyzes_conversion` = true
- `includes_gdn` = true
- `includes_convolution` = true
- `includes_kv` = true
- `vision_interface_is_not_a_node` = true
- `payloads_restreamed` = false
- `fusion_winner_selected` = false
- `state_payload_in_artifact_selected` = false

JSON `T_is_stored_length_after_append` true. JSON `example_T` `[1, 4096]`. JSON `primary_includes_mtp` true.

### Logical dimensions (lock; heading 3)

JSON array `layout_object_ids` in this exact order (7 ids) — **same as** TASK-09 `access_class_ids`. JSON `n_layout_objects` = 7. Parallel `layout_object_ranks`. JSON object `layout_object_axes` keyed in that order.

| id | Rank | Axes (logical, checkpoint-facing) | Source rank |
| --- | ---: | --- | --- |
| `gather_row` | 2 | `vocab`, `hidden` | \(E\in\mathbb{R}^{V\times H}=(248320,5120)\) |
| `dense_gemm` | 2 | `d_out`, `d_in` | \(W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}\), \(y=Wx\) |
| `depthwise_conv` | 2 | `channel`, `tap` | squeezed \((10240,4)\); checkpoint \((10240,1,4)\) |
| `vector_param` | 1 | `width` | \(\gamma\in\mathbb{R}^{H}\) or \(A_\log,d_t\in\mathbb{R}^{48}\) or \(\gamma^z\in\mathbb{R}^{128}\) or \(\gamma_{q,k}\in\mathbb{R}^{256}\) |
| `state_kv` | 3 | `n_kv`, `T`, `d_h` | \(K,V\in\mathbb{R}^{4\times T\times 256}\) each |
| `state_c` | 2 | `delay`, `channel` | \(C\in\mathbb{R}^{3\times 10240}\) |
| `state_s` | 3 | `n_v`, `d_k`, `d_v` | \(S\in\mathbb{R}^{48\times 128\times 128}\) conceptual F32 |

JSON `layout_object_ranks` `[2,2,2,1,3,2,3]`. JSON `layout_object_axes`:

- `gather_row`: `["vocab","hidden"]`
- `dense_gemm`: `["d_out","d_in"]`
- `depthwise_conv`: `["channel","tap"]`
- `vector_param`: `["width"]`
- `state_kv`: `["n_kv","T","d_h"]`
- `state_c`: `["delay","channel"]`
- `state_s`: `["n_v","d_k","d_v"]`

JSON array `analysis_dimension_ids` in this exact order (6 ids). JSON `n_analysis_dimensions` = 6:

`logical_dimensions`, `consumers`, `access`, `tiling`, `alignment`, `conversion`

JSON array `persistent_state_coverage_ids` in this exact order (3 ids). JSON `n_persistent_state_coverages` = 3:

`gdn`, `convolution`, `kv`

Family mapping: copy TASK-09 `policy_families` and `family_access_class` / `family_access_class_values` exactly. JSON `family_layout_object` equals `family_access_class` (same 16 keys; `vision_deferred` is JSON `null`). Do not recopy TASK-08 recipe lists as a new quantization grid; cite that every defined family remains packable as `keep_source` and as that family’s candidates. `lm_head` is `dense_gemm` despite matching embed shape. Embed is `gather_row`, not GEMM. Conv1d uses squeezed rank-2.

Instantiated widths (OBSERVED / DERIVED from `text_config`): `hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, `head_dim` 256, `num_attention_heads` 24, `num_key_value_heads` 4, `g_qa` 6, `linear_key_head_dim` / `linear_value_head_dim` 128, `linear_num_value_heads` 48, `linear_conv_kernel_dim` 4, `linear_conv_delay` 3, `d_qkv` 10240, `n_full_layers_with_kv` 17, `n_linear_layers` 48. JSON `g_qa` = `num_attention_heads // num_key_value_heads` = 6. JSON `d_qkv` = 10240. JSON `linear_conv_delay` = `linear_conv_kernel_dim - 1` = 3.

Prose required: checkpoint orientation is the **logical** rank. Physical order is an unselected candidate. GQA repeat is **not** a stored axis of `state_kv`. Paper \(S^\top\) is not a second state object; it is an unselected ordering of `state_s`.

### Consumers (lock; heading 4)

JSON array `consumer_mode_ids` in this exact order (2 ids). JSON `n_consumer_modes` = 2: `decode_gemv`, `prefill_gemm`. Neither mode selects a layout. JSON `n_consumer_modes_selected` = 0.

JSON array `stage_kind_ids` copied from TASK-13/14 in TASK-13/14 order (9 ids). JSON `n_stage_kinds` = 9. Cite; do not rewrite the serial order.

Table columns: `Object` | `Decode consumer (TASK-13)` | `Prefill consumer (TASK-14)` | `Primary sequence attachment`

Implementation copies these rows; do not add objects.

| Object | Decode consumer | Prefill consumer | Primary sequence |
| --- | --- | --- | --- |
| `gather_row` | One (language) or two (complete) rows of shared \(E\); table not streamed | \(T\) or \(2T\) rows | `seq_gather_row` |
| `dense_gemm` | GEMV \(x\in\mathbb{R}^{H}\); unique weights once; `lm_head` streams 2542796800 B-class table (`seq_lm_head_full`) | GEMM with sequence axis \(T\); unique weights once per prompt, reused across \(T\) | `seq_gemm_codes_then_scales`; `lm_head` stages use `seq_lm_head_full` |
| `depthwise_conv` | FIR `(14)` on one token; 4 taps, delay 3 | FIR over the length-\(T\) stream; delay still 3 stored | none of the seven sequences is conv-specific; `seq_specialized_tile` is an unselected view |
| `vector_param` | Elementwise \(\gamma\), \(A_\log\), \(d_t\) | Same, rank \(T\) on the activation, not on the parameter | none |
| `state_kv` | Read length \(T-1\), write 1 token; GQA consume with 24 queries | Triangular read, write \(T\) tokens; causal MM | none dedicated |
| `state_c` | Read 3 taps, write 1 QKV vector | From zeros (0 physical on first token); write each step | none dedicated |
| `state_s` | Read/write full \(S\) 150994944 B/step conceptual F32 | From zeros; write \(S_t\) each step; surviving store is not \(T\times B_S\) | `seq_state_s_dense` |

JSON `layout_object_primary_sequence_ids` in `layout_object_ids` order:

`["seq_gather_row","seq_gemm_codes_then_scales",null,null,null,null,"seq_state_s_dense"]`

JSON `lm_head_sequence_id` = `seq_lm_head_full`. JSON `gated_delta_net_state_sequence_id` = `seq_state_s_dense`. Attaching a sequence is **not** selecting an ideal byte sequence.

JSON `n_layout_object_primary_sequences` = 3 (the non-null attachments).

Prose required: mixer xor still holds — `language_mixer` consumes `state_kv` **xor** `(state_c, state_s)` by layer type. MTP mixer consumes only `state_kv`. Depthwise conv is a GDN-layer weight consumer, not a full-attention consumer.

### Access (lock; heading 5)

JSON array `access_class_ids` copied from TASK-09 in TASK-09 order (7 ids). JSON `n_access_classes` = 7. JSON `layout_object_ids` **equals** `access_class_ids`.

JSON array `consumer_sequence_ids` copied from TASK-09 in TASK-09 order (7 ids). JSON `n_consumer_sequences` = 7. JSON `ideal_byte_sequence_selected` false.

Access identities (DERIVED citations, not a new traffic study):

- Embed gather is 0 MAC, 10240 B/row BF16 (TASK-06). `row_addressable` remains a required packing **capability**; stride/pad remain open (TASK-09).
- Decode GEMMs have TASK-06 \(I=1\) vs unique weights for MLP/`lm_head` (citation; bottleneck label `weight_memory` / `vocab_memory` remains HYPOTHESIS).
- Prefill reuses those unique bytes across \(T\) (`weight_unique_counted_once`; TASK-06 `compute` label remains HYPOTHESIS).
- GQA: 24 query heads read 4 stored KV heads with \(g_\text{qa}=6\). Stored rank stays \((4,T,256)\); repeated KV is not stored (`gqa_repeat_not_stored` true).
- RoPE is baked into stored \(K\) before the cache (`kv_rope_baked_into_k` true). Stored \(K\) has no extra rotary axis.
- GDN \(S\) access is the dense recurrent update (17) plus \(S^\top \tilde q\) (18). `seq_state_s_dense` names that dense read/write; it is not a selected layout.
- Conv access is per-channel FIR over 4 taps. \(z\) does not enter the convolution.

Prose required: a packing convenient for `seq_gemm_interleaved_group` may break `seq_gather_row` (HYPOTHESIS, cite TASK-09 `f_gather_stride`). Do not rank sequences by wall time. Do not name CUDA kernels. `seq_specialized_tile` is a **view** sequence, not an eighth access class; this document names candidates it could store (`seq_specialized_tile_candidates_named` true) without selecting it.

### Candidate orderings and tiling (lock; heading 6)

JSON array `ordering_ids` in this exact order (14 ids). JSON `n_orderings` = 14. Parallel `ordering_object_ids`. JSON `n_orderings_selected` = 0. JSON `ordering_selected` false. JSON `ordering_usefulness_label` exactly `HYPOTHESIS`.

| id | Object | Axis order (fastest last) | Meaning |
| --- | --- | --- | --- |
| `ord_gemm_out_major` | `dense_gemm` | `d_out`, `d_in` | Checkpoint orientation; row of \(W\) contiguous |
| `ord_gemm_in_major` | `dense_gemm` | `d_in`, `d_out` | Transpose pack; \(d_\text{in}\) contiguous |
| `ord_embed_vocab_major` | `gather_row` | `vocab`, `hidden` | One vocab row contiguous (`row_addressable`-friendly) |
| `ord_embed_hidden_major` | `gather_row` | `hidden`, `vocab` | Hidden-major; gather may be strided (pairs with `f_gather_stride`) |
| `ord_conv_channel_tap` | `depthwise_conv` | `channel`, `tap` | 4 taps of one channel contiguous (FIR inner sum) |
| `ord_conv_tap_channel` | `depthwise_conv` | `tap`, `channel` | Tap-major across 10240 channels |
| `ord_vec_width` | `vector_param` | `width` | Sole rank-1 order |
| `ord_kv_n_t_dh` | `state_kv` | `n_kv`, `T`, `d_h` | TASK-02 rank |
| `ord_kv_n_dh_t` | `state_kv` | `n_kv`, `d_h`, `T` | Token index last |
| `ord_kv_t_n_dh` | `state_kv` | `T`, `n_kv`, `d_h` | Sequence-major |
| `ord_c_delay_channel` | `state_c` | `delay`, `channel` | TASK-02 \(3\times 10240\) |
| `ord_c_channel_delay` | `state_c` | `channel`, `delay` | 3 taps of one channel contiguous |
| `ord_s_n_dk_dv` | `state_s` | `n_v`, `d_k`, `d_v` | TASK-02 \(S\) (17)–(18) |
| `ord_s_n_dv_dk` | `state_s` | `n_v`, `d_v`, `d_k` | Paper \(S^\top\) store (19); same map |

JSON `ordering_object_ids` `["dense_gemm","dense_gemm","gather_row","gather_row","depthwise_conv","depthwise_conv","vector_param","state_kv","state_kv","state_kv","state_c","state_c","state_s","state_s"]`.

Every object has at least one ordering. Do not mark any `selected`, `required`, or `optimal`. Do not drop `ord_embed_hidden_major` because it is gather-hostile; listing it is the candidate space.

JSON array `tile_family_ids` in this exact order (9 ids). JSON `n_tile_families` = 9. JSON object `tile_family_objects`. JSON `tile_size_selected` false. JSON `mma_tile_extents_selected` false. JSON `tile_usefulness_label` exactly `HYPOTHESIS`.

| id | Objects | Meaning |
| --- | --- | --- |
| `tile_none` | all seven | No blocking beyond the ordering |
| `tile_2d_mn` | `dense_gemm` | 2D tiles on \((d_\text{out},d_\text{in})\) |
| `tile_1d_row` | `gather_row` | Tiles along `hidden` inside one vocab row |
| `tile_conv_channel` | `depthwise_conv`, `state_c` | Tiles along 10240 channels |
| `tile_kv_t` | `state_kv` | Tiles along stored length \(T\) |
| `tile_kv_dh` | `state_kv` | Tiles along `d_h` |
| `tile_s_head` | `state_s` | One \(128\times 128\) head as the tile |
| `tile_s_block` | `state_s` | Blocks inside a head matrix |
| `tile_mma_shaped` | `dense_gemm` | MMA-ready extents; **extents unselected** (SKU MMA shapes are TASK-16 UNKNOWN / TASK-17) |

JSON `tile_family_objects`:

- `tile_none`: `["gather_row","dense_gemm","depthwise_conv","vector_param","state_kv","state_c","state_s"]`
- `tile_2d_mn`: `["dense_gemm"]`
- `tile_1d_row`: `["gather_row"]`
- `tile_conv_channel`: `["depthwise_conv","state_c"]`
- `tile_kv_t`: `["state_kv"]`
- `tile_kv_dh`: `["state_kv"]`
- `tile_s_head`: `["state_s"]`
- `tile_s_block`: `["state_s"]`
- `tile_mma_shaped`: `["dense_gemm"]`

JSON array `tile_extent_candidates` `[16, 32, 64, 128, 256]`. These are the TASK-09 `alignment_grain_candidates` excluding `1`. They are **candidates**, not selected tiles. JSON `n_tile_extent_candidates` = 5.

Divisibility (DERIVED; checker recomputes from `text_config`; do not use to pick a winner):

| Extent | \(H=5120\) | \(I=17408\) | \(d_\text{qkv}=10240\) | \(d_h=256\) | \(d_k=128\) |
| ---: | --- | --- | --- | --- | --- |
| 16 | yes | yes | yes | yes | yes |
| 32 | yes | yes | yes | yes | yes |
| 64 | yes | yes | yes | yes | yes |
| 128 | yes | yes | yes | yes | yes |
| 256 | yes | yes | yes | yes | **no** |

JSON `tile_extent_divides_hidden` all true. JSON `tile_extent_divides_intermediate` all true. JSON `tile_extent_divides_d_qkv` all true. JSON `tile_extent_divides_head_dim` all true. JSON `tile_extent_divides_dk` `[true,true,true,true,false]`. JSON `n_v_divides_none_of_extents_as_head_count` true (48 is not 16/32/64/128/256); `tile_s_head` uses one head, not an extent from that list.

Prose required: naming an extent that divides a width is not selecting it. Prefill’s sequence axis \(T\) is a **tile axis** of activations and of `state_kv`; it is not a weight-tensor axis. Decode GEMV has no sequence tile on weights (`diff_tiling` citation). `tile_mma_shaped` must not instantiate an MMA \((M,N,K)\) from a datasheet.

### Alignment and conversion (lock; heading 7)

JSON `alignment_grain_candidates` copied from TASK-09: `[1, 16, 32, 128, 256]`. JSON `alignment_grain_selected` false. JSON `specialized_alignment_grain_selected` false. JSON `specialized_view_may_constrain_alignment` true (TASK-09 already allowed TASK-15 to constrain specialized views). This document **does not** close TASK-09 `alignment_grain`.

Cited group-payload grains (TASK-09; not a new packing study):

| JSON key | Value |
| --- | ---: |
| `int4_g32_group_payload_bytes` | 16 |
| `int3_g32_group_payload_bytes` | 12 |
| `int2_g32_group_payload_bytes` | 8 |
| `int4_row_hidden_payload_bytes` | 2560 |
| `int6_row_hidden_payload_bytes` | 3840 |
| `embed_gather_int4_row_bytes` | 2562 |
| `embed_gather_bf16_bytes` | 10240 |

JSON `scale_storage_bytes_candidates` `[2, 4]` unselected. JSON `scale_placement_candidates` `["sidecar_array","interleaved_group"]` unselected. JSON `code_bit_order_candidates` `["lsb_first","msb_first"]` unselected. Portable view payloads remain little-endian. Specialized views may swizzle (candidates above); swizzle is not selected.

JSON array `conversion_hypothesis_ids` in this exact order (4 ids). JSON `n_conversion_hypotheses` = 4. JSON `n_conversion_hypotheses_selected` = 0. JSON `conversion_pipeline_selected` false. JSON `conversion_usefulness_label` exactly `HYPOTHESIS`.

| id | Meaning | Pairs with TASK-09 |
| --- | --- | --- |
| `conv_compile_pack` | Offline pack emits a specialized layout into the artifact | `backend_specialized_only` / specialized view of `portable_plus_specialized_views` |
| `conv_load_repack` | Portable view converted at load | `portable_only` `load_convert`; TASK-08 `d_weight_dequant` |
| `conv_inkernel_unpack` | Consumer unpacks/dequants in the contraction | `portable_only`; format risk `f_unpack_portable` |
| `conv_dual_view` | Store portable and specialized copies | Illustration F size DERIVED 26863340064 B if two full int4-g128 unique-non-embed copies; **impact** HYPOTHESIS |

Do not select a conversion pipeline. Dual-view **size** may be cited from TASK-09 illustration F; do not recommend storing two views. JSON `dual_view_int4_g128_unique_non_embed_bytes` 26863340064 (citation).

Align identity (citation of TASK-09, not a selected grain):

\[
B_{\text{aligned}}=\Bigl\lceil B_{\text{payload,pack}}/A\Bigr\rceil A.
\]

Prose required: int3 g32 is 12 bytes, not a power of two (DERIVED). Whether that forbids a specialized grain is HYPOTHESIS (`l_int3_grain`). int4 g32 is 16 bytes (DERIVED). Whether specialized views must pad to a multiple of 16 is HYPOTHESIS, not a close of `alignment_grain`.

### GDN, convolution, and KV persistent state (lock; heading 8)

Required `###` subheadings in this exact order: `GDN`, `Convolution`, `KV persistent state`. Checker substring-matches those three heading titles. Completes ledger checkbox 2. JSON `includes_gdn` / `includes_convolution` / `includes_kv` true.

**GDN.** Consumers: 48 `gated_delta_net` language mixers (TASK-13/14 xor). Weights: \(W_\text{qkv}\in\mathbb{R}^{10240\times 5120}\) (`dense_gemm`), \(W_z\in\mathbb{R}^{6144\times 5120}\) (`dense_gemm`), \(W_a,W_b\in\mathbb{R}^{48\times 5120}\) (`dense_gemm`, family `gdn_gate_proj`), \(A_\log,d_t\in\mathbb{R}^{48}\) (`vector_param`, family `gdn_time_param`), \(W^{\text{conv}}\) (`depthwise_conv`), \(W_\text{out}\in\mathbb{R}^{5120\times 6144}\) (`dense_gemm`). State \(S\): rank \((48,128,128)\) F32, 3145728 B/layer, 150994944 B all-48 (`s_bytes_per_layer`, `s_f32_bytes`). Decode reads and writes the full matrix each step; prefill starts from zeros (0 physical on the first token) and writes \(S_t\) each step. Primary recurrence is (17)–(18). Paper (19) \(S^\top\) is the same map (`paper_s_transpose_same_map`). Chunkwise GDN is not zero \(S\) traffic. Candidate orderings `ord_s_n_dk_dv`, `ord_s_n_dv_dk`. Candidate tiles `tile_s_head`, `tile_s_block`. Sequence attachment `seq_state_s_dense`. TASK-06 `i_gdn_vs_s_rw` 0.75 and bottleneck label `state_memory` remain HYPOTHESIS citations.

**Convolution.** Weight \(W^{\text{conv}}\in\mathbb{R}^{10240\times 1\times 4}\) squeezed \((10240,4)\), family `conv1d`, access `depthwise_conv`. FIR (14) over \(k_\text{conv}=4\) taps; stored delay \(k_\text{conv}-1=3\). \(z\) does not enter the convolution. State \(C\): rank \((3,10240)\) BF16, 61440 B/layer, 2949120 B all-48 (`c_bytes_per_layer`, `c_bytes_all`). Decode reads 3 taps and writes one new QKV vector (do not count rewriting retained taps as new writes). Prefill from zeros: first-token physical read 0. Candidate orderings `ord_conv_channel_tap`, `ord_conv_tap_channel`, `ord_c_delay_channel`, `ord_c_channel_delay`. Candidate tile `tile_conv_channel`. No dedicated TASK-09 sequence.

**KV persistent state.** \(K,V\) each \((4,T,256)\) BF16; 4096 B/token/instance; 69632 B/token across 17 instances including MTP (`kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`). Full-attention layers only; linear layers have no KV. RoPE baked into stored \(K\). GQA repeat not stored. Decode: read \(T-1\), write 1. Prefill: triangular read \(69632\cdot T(T-1)/2\), write \(69632T\). Attention consumer is (9) with 24 queries against 4 KV heads. Candidate orderings `ord_kv_n_t_dh`, `ord_kv_n_dh_t`, `ord_kv_t_n_dh`. Candidate tiles `tile_kv_t`, `tile_kv_dh`. Writes are not optional.

Prose required: omitting a KV/\(C\)/\(S\) write changes the map. Surviving store after prefill is TASK-04 \(B_\text{store}(T)=69632T+153944064\), not \(T\times B_S\). Layout of state is independent of whether zero templates live in the artifact (`state_payload_in_artifact_selected` false remains TASK-09 open).

### Parallel decompositions (lock; heading 9)

This heading enumerates planned parallel decompositions and **unselected** pairing hypotheses. It does **not** close the ledger open question. JSON `ledger_open_question_parallel_decomposition_closed` false. JSON `parallel_decomposition_justifies_layout_selected` false. JSON `justification_usefulness_label` exactly `HYPOTHESIS`.

JSON array `parallel_decomposition_ids` in this exact order (9 ids). JSON `n_parallel_decompositions` = 9. JSON `n_parallel_decompositions_selected` = 0.

| id | Split axis | Consumer |
| --- | --- | --- |
| `par_gemm_d_out` | `d_out` | decode GEMV / prefill GEMM |
| `par_gemm_d_in` | `d_in` (partial reduction) | same |
| `par_gemm_T` | sequence \(T\) | prefill GEMM only |
| `par_attn_head` | 24 query heads | gated attention core |
| `par_attn_T` | stored length \(T\) | attention vs KV |
| `par_gdn_head` | 48 value heads | GDN (17)–(18) |
| `par_conv_channel` | 10240 channels | FIR (14) |
| `par_kv_head` | 4 KV heads | KV layout vs GQA consume |
| `par_embed_row` | independent vocab rows | gather; not a split of one row |

JSON array `justification_hypothesis_ids` in this exact order (10 ids). JSON `n_justification_hypotheses` = 10. JSON `n_justification_hypotheses_selected` = 0.

| id | Ordering or tile | Decomposition | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `j_gemm_out_major_par_d_out` | `ord_gemm_out_major` | `par_gemm_d_out` | Out-major matches a \(d_\text{out}\) split |
| `j_gemm_in_major_par_d_in` | `ord_gemm_in_major` | `par_gemm_d_in` | In-major matches a \(K\)-split |
| `j_gemm_tile_2d_par_T` | `tile_2d_mn` | `par_gemm_T` | 2D weight tiles match a prefill \(T\) split |
| `j_embed_vocab_major_par_row` | `ord_embed_vocab_major` | `par_embed_row` | Vocab-major matches row gather |
| `j_conv_channel_tap_par_channel` | `ord_conv_channel_tap` | `par_conv_channel` | Channel-tap matches a channel split |
| `j_kv_n_t_dh_par_head` | `ord_kv_n_t_dh` | `par_kv_head` | TASK-02 KV order matches a KV-head split |
| `j_kv_n_dh_t_par_T` | `ord_kv_n_dh_t` | `par_attn_T` | Token-last KV matches a sequence split |
| `j_s_n_dk_dv_par_head` | `ord_s_n_dk_dv` | `par_gdn_head` | \(S\) as (17) matches a head split |
| `j_s_n_dv_dk_par_head` | `ord_s_n_dv_dk` | `par_gdn_head` | \(S^\top\) store matches a head split |
| `j_mma_shaped_unselected_par` | `tile_mma_shaped` | (unspecified CUDA) | MMA tiles need TASK-17 to name a decomposition |

Required closing sentence (checker substring):

> Listing a justification hypothesis is not selecting a parallel decomposition and does not justify a candidate ordering or tile.

JSON `canonical_sentence_justification` equals that sentence.

Do not rank hypotheses by wall time. Do not treat a pairing table as CUDA evidence. TASK-17 may instantiate these hypotheses; this task must not.

Layout-risk hypotheses (every severity is HYPOTHESIS). JSON array `layout_risk_ids` in this exact order (8 ids). Parallel `layout_risk_severities`. JSON `n_layout_risks` = 8.

| ID | Ties to | Severity | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `l_gemm_vs_gather` | `f_gather_stride` | medium | A GEMM-friendly order may make embed-row gather non-contiguous |
| `l_decode_vs_prefill_view` | TASK-14 views | medium | One order may not serve GEMV and GEMM equally; this does **not** select distinct views |
| `l_s_transpose` | (17) vs (19) | medium | Storing \(S\) vs \(S^\top\) changes inner-loop axes |
| `l_kv_append` | KV orderings | medium | \(T\)-last vs \(T\)-middle changes append versus scan |
| `l_conv_fir_stride` | conv orderings | medium | Tap-major vs channel-major changes FIR stride |
| `l_convert_cost` | `f_unpack_portable` | high | Compile/load/in-kernel conversion may add decode traffic |
| `l_int3_grain` | `f_3bit_shift` | medium | int3 12-byte groups may not match power-of-two tiles |
| `l_mma_sku_unknown` | `tile_mma_shaped` | low | MMA-shaped tiles cannot be justified without TASK-16/17 |

JSON `layout_high_ids`: `l_convert_cost`. `layout_medium_ids`: `l_gemm_vs_gather`, `l_decode_vs_prefill_view`, `l_s_transpose`, `l_kv_append`, `l_conv_fir_stride`, `l_int3_grain`. `layout_low_ids`: `l_mma_sku_unknown`. JSON `n_layout_high` = 1, `n_layout_medium` = 6, `n_layout_low` = 1.

JSON `layout_risk_severities` `["medium","medium","medium","medium","medium","high","medium","low"]`.

### Work citations and non-decisions (lock; heading 10)

Cite; do not recopy TASK-06 symbolic tables. Checker **recomputes** state bytes from `text_config` with the same identities as TASK-04/06:

- `kv_bytes_per_full_layer_per_token` = \(2\cdot n_\text{kv}\cdot d_h\cdot 2=4096\)
- `kv_bytes_all_per_token` = \(4096\times 17=69632\)
- `c_bytes_per_layer` = \(3\cdot d_\text{qkv}\cdot 2=61440\)
- `c_bytes_all` = \(61440\times 48=2949120\)
- `s_bytes_per_layer` = \(n_v\cdot d_k\cdot d_v\cdot 4=3145728\)
- `s_f32_bytes` = \(3145728\times 48=150994944\)
- `s_bf16_bytes` = 75497472 (TASK-09 illustration D; **not** a selected store dtype)
- `weight_bytes_lm_head` = \(V\cdot H\cdot 2=2542796800\)
- `embed_gather_bf16_bytes` = \(H\cdot 2=10240\)
- `dual_view_int4_g128_unique_non_embed_bytes` = 26863340064 (TASK-09 illustration F citation)

JSON `bottleneck_labels` copied from TASK-06: `weight_memory`, `vocab_memory`, `state_memory`, `kv_memory`, `quadratic_attn`, `compute`. Restating a label here is a **citation**, still HYPOTHESIS. JSON `n_bottleneck_labels` = 6. JSON `i_mlp_weight_only` 1, `i_lm_head_weight_only` 1, `i_gdn_vs_s_rw` 0.75, `i_attn_core_vs_kv` 6 (decode identity cited; do not invent a layout-ridge winner).

Non-decisions (prose required): TASK-17 owns CUDA mappings per node type **and** which (if any) candidate layout to instantiate. TASK-16 SKU limits stay UNKNOWN and are not filled from a datasheet. TASK-09 still owns artifact-boundary, ideal-sequence, scale-storage, scale-placement, alignment-grain, and code-bit-order **selection**. TASK-12 still owns fusion **winners**. TASK-14 still owns decode/prefill **view** selection (`decode_prefill_distinct_views_selected` false). TASK-08 still owns recipe **winners**. This layout space does not change when a TASK-08 recipe is later applied; packing grains stay citations. The ledger open question remains unresolved.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 9 (Parallel decompositions). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 866 tensors, or \(T\) positions. Caption must contain the word `HYPOTHESIS` and the word `unresolved`.

Required IDs **inside that fence**: `gemm`, `gather`, `conv`, `kv`, `gdn`, `c_state`, `portable`, `specialized`, `open`.

JSON `n_diagrams` is 1. `diagram_ids` is `["gemm","gather","conv","kv","gdn","c_state","portable","specialized","open"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger layouts are **UNKNOWN**. `vision_deferred` has `family_access_class` null and no layout object. Do not add vision payloads. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_layout_strategy.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, \(d_\text{qkv}\), GQA, KV/\(C\)/\(S\) bytes, gather/lm-head bytes, group-payload grains, and tile-extent divisibility. Duplicate TASK-09 `consumer_sequence_ids` / `access_class_ids` / `policy_families` / `family_access_class_values` and TASK-13/14 `stage_kind_ids` as constants; do not import them.

The ledger **Produces** line names only `docs/architecture/layout-strategy.md`. The checker is stdlib evidence tooling matching TASK-01–14 and the user-required stdlib checker; it is in scope for this increment.

CLI (cwd = repository root):

```text
python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --layout-strategy docs/architecture/layout-strategy.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `linear_conv_kernel_dim`. Derived: \(d_\text{qkv}\), \(g_\text{qa}\), conv delay, instance counts, KV/\(C\)/\(S\) bytes, gather/lm-head bytes, group grains, tile-extent divisibility. Constant fields: canonical sentences, layout/ordering/tile/decomposition/conversion/justification/risk lists, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--layout-strategy PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all five canonical sentences plus `canonical_sentence_justification` verbatim, (5) every `layout_object_ids`, `ordering_ids`, `tile_family_ids`, `parallel_decomposition_ids`, `conversion_hypothesis_ids`, `justification_hypothesis_ids`, `layout_risk_ids`, `analysis_dimension_ids`, `persistent_state_coverage_ids`, `consumer_sequence_ids`, `access_class_ids`, `policy_families`, `stage_kind_ids`, `bottleneck_labels` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) `### GDN`, `### Convolution`, and `### KV persistent state` present, (8) none of `TBD`, `TODO`, `???`, (9) no `UNKNOWN` except inside the Deferred vision section, (10) every locked document integer/decimal below present as a decimal or integer substring, (11) the words `HYPOTHESIS`, `hardware-independent`, `unresolved`, and `no optimality claim` present, (12) none of the forbidden winner/optimality phrases: `optimal layout`, `best tile`, `winning layout`, `selected tile is`, `selected ordering is`, `this layout is optimal`, `CUDA occupancy selects`, `should use this tile`, `recommend this layout`, `parallel decomposition justifies`, `MMA shape is required`, `ideal byte sequence is`, `artifact boundary is`, `selected winner`, `thread block`, `warp shuffle`, `Quartz layout`, `llama.cpp layout`, `GGUF is the layout`, `selected distinct views`, `prefill requires a distinct view` (allow the substring only inside `not a selected winner` / `not selected winners` / `not a CUDA mapping` / `makes no optimality claim` / `does not justify a candidate`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks cited TASK-04/06/09 integers against those documents, sequence/access-class ids against TASK-09 markdown, and stage-kind ids against TASK-13/14 markdown).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `hidden_size==5120`, `intermediate_size==17408`, `vocab_size==248320`, `head_dim==256`
- `num_attention_heads==24`, `num_key_value_heads==4`, `g_qa==6`
- `linear_num_value_heads==48`, `linear_key_head_dim==128`, `linear_value_head_dim==128`
- `linear_conv_kernel_dim==4`, `linear_conv_delay==3`, `d_qkv==10240`
- `kv_bytes_per_full_layer_per_token==4096`, `kv_bytes_all_per_token==69632`
- `c_bytes_per_layer==61440`, `c_bytes_all==2949120`
- `s_bytes_per_layer==3145728`, `s_f32_bytes==150994944`, `s_bf16_bytes==75497472`
- `embed_gather_bf16_bytes==10240`, `embed_gather_int4_row_bytes==2562`, `weight_bytes_lm_head==2542796800`
- `int4_g32_group_payload_bytes==16`, `int3_g32_group_payload_bytes==12`, `int2_g32_group_payload_bytes==8`
- `int4_row_hidden_payload_bytes==2560`, `int6_row_hidden_payload_bytes==3840`
- `dual_view_int4_g128_unique_non_embed_bytes==26863340064`
- `n_layout_objects==7`, `layout_object_ids==access_class_ids`
- `n_orderings==14`, `n_orderings_selected==0`, `n_tile_families==9`, `tile_size_selected is False`
- `mma_tile_extents_selected is False`, `n_tile_extent_candidates==5`
- `tile_extent_divides_dk[-1] is False` and the first four are True
- `n_parallel_decompositions==9`, `n_parallel_decompositions_selected==0`
- `n_conversion_hypotheses==4`, `n_conversion_hypotheses_selected==0`
- `n_justification_hypotheses==10`, `n_justification_hypotheses_selected==0`
- `n_layout_risks==8`, `n_layout_high==1`, `n_layout_medium==6`, `n_layout_low==1`
- `n_analysis_dimensions==6`, `n_persistent_state_coverages==3`
- `n_consumer_sequences==7`, `n_access_classes==7`, `n_stage_kinds==9`, `n_diagrams==1`
- `n_consumer_modes==2`, `n_consumer_modes_selected==0`
- `family_access_class["vision_deferred"] is None`, `family_access_class["embed_table"]=="gather_row"`, `family_access_class["lm_head"]=="dense_gemm"`, `family_access_class["conv1d"]=="depthwise_conv"`, `family_access_class["state_s"]=="state_s"`
- `layout_object_primary_sequence_ids[0]=="seq_gather_row"`, `[1]=="seq_gemm_codes_then_scales"`, `[6]=="seq_state_s_dense"`, and indices `[2]` through `[5]` (`depthwise_conv`, `vector_param`, `state_kv`, `state_c`) are `None`
- `lm_head_sequence_id=="seq_lm_head_full"`, `gated_delta_net_state_sequence_id=="seq_state_s_dense"`
- `ideal_byte_sequence_selected is False`, `artifact_boundary_selected is False`
- `decode_prefill_distinct_views_selected is False`, `ledger_open_question_parallel_decomposition_closed is False`
- `layout_winner_selected is False`, `optimality_claim_absent is True`
- `activations_in_layout_scope is False`, `thread_geometry_absent is True`, `hardware_independent is True`
- `gguf_is_not_the_runtime_format is True`, `payloads_restreamed is False`
- `seq_specialized_tile_candidates_named is True`, `tile_layout_deferred_to_task15 is False`
- `includes_gdn and includes_convolution and includes_kv`
- all six `analyzes_*` booleans true
- `gqa_repeat_not_stored is True`, `paper_s_transpose_same_map is True`, `chunkwise_not_zero_s_traffic is True`

Locked document integers/decimals the `--layout-strategy` check must find:

`5120`, `17408`, `248320`, `10240`, `6144`, `256`, `128`, `48`, `24`, `4`, `3`, `6`, `16`, `17`, `64`, `4096`, `69632`, `61440`, `2949120`, `3145728`, `150994944`, `75497472`, `2542796800`, `10240`, `2562`, `2560`, `3840`, `12`, `8`, `26863340064`, `0.75`

(`10240` appears twice in this list as gather-row bytes and \(d_\text{qkv}\); the checker searches substrings, so once is enough in the document.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `head_dim`, `num_attention_heads`, `num_key_value_heads`, `g_qa`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_value_heads`, `linear_conv_kernel_dim`, `linear_conv_delay`, `d_qkv`, `bytes_bf16`, `bytes_f32`,

`kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`, `c_bytes_per_layer`, `c_bytes_all`, `s_bytes_per_layer`, `s_f32_bytes`, `s_bf16_bytes`, `embed_gather_bf16_bytes`, `embed_gather_int4_row_bytes`, `weight_bytes_lm_head`, `int4_g32_group_payload_bytes`, `int3_g32_group_payload_bytes`, `int2_g32_group_payload_bytes`, `int4_row_hidden_payload_bytes`, `int6_row_hidden_payload_bytes`, `dual_view_int4_g128_unique_non_embed_bytes`,

`tile_extent_candidates`, `n_tile_extent_candidates`, `alignment_grain_candidates`, `scale_storage_bytes_candidates`, `scale_placement_candidates`, `code_bit_order_candidates`, `tile_extent_divides_hidden`, `tile_extent_divides_intermediate`, `tile_extent_divides_d_qkv`, `tile_extent_divides_head_dim`, `tile_extent_divides_dk`, `n_v_divides_none_of_extents_as_head_count`,

`layout_object_ids`, `n_layout_objects`, `layout_object_ranks`, `layout_object_axes`, `layout_object_primary_sequence_ids`, `n_layout_object_primary_sequences`, `lm_head_sequence_id`, `gated_delta_net_state_sequence_id`,

`access_class_ids`, `n_access_classes`, `policy_families`, `family_access_class`, `family_access_class_values`, `family_layout_object`, `consumer_sequence_ids`, `n_consumer_sequences`,

`analysis_dimension_ids`, `n_analysis_dimensions`, `persistent_state_coverage_ids`, `n_persistent_state_coverages`, `consumer_mode_ids`, `n_consumer_modes`, `n_consumer_modes_selected`, `stage_kind_ids`, `n_stage_kinds`, `bottleneck_labels`, `n_bottleneck_labels`, `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_gdn_vs_s_rw`, `i_attn_core_vs_kv`,

`ordering_ids`, `ordering_object_ids`, `n_orderings`, `n_orderings_selected`, `ordering_usefulness_label`, `tile_family_ids`, `n_tile_families`, `tile_family_objects`, `tile_size_selected`, `mma_tile_extents_selected`, `tile_usefulness_label`,

`parallel_decomposition_ids`, `n_parallel_decompositions`, `n_parallel_decompositions_selected`, `conversion_hypothesis_ids`, `n_conversion_hypotheses`, `n_conversion_hypotheses_selected`, `conversion_usefulness_label`, `justification_hypothesis_ids`, `n_justification_hypotheses`, `n_justification_hypotheses_selected`, `justification_usefulness_label`,

`layout_risk_ids`, `layout_risk_severities`, `layout_high_ids`, `layout_medium_ids`, `layout_low_ids`, `n_layout_risks`, `n_layout_high`, `n_layout_medium`, `n_layout_low`,

`example_T`, `T_is_stored_length_after_append`, `primary_includes_mtp`,

`decode_prefill_share_artifact`, `decode_prefill_share_graph`, `decode_prefill_distinct_views_selected`, `activations_in_layout_scope`, `hardware_independent`, `cuda_mapping_deferred`, `thread_geometry_absent`, `layout_winner_selected`, `ordering_selected`, `alignment_grain_selected`, `specialized_alignment_grain_selected`, `specialized_view_may_constrain_alignment`, `conversion_pipeline_selected`, `parallel_decomposition_selected`, `ideal_byte_sequence_selected`, `artifact_boundary_selected`, `seq_specialized_tile_candidates_named`, `tile_layout_deferred_to_task15`, `ledger_open_question_parallel_decomposition_closed`, `parallel_decomposition_justifies_layout_selected`, `optimality_claim_absent`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_not_zero_s_traffic`, `paper_s_transpose_same_map`, `state_write_not_optional`, `kv_rope_baked_into_k`, `gqa_repeat_not_stored`, `conv_z_does_not_enter_conv`, `portable_payload_little_endian`, `gguf_is_not_the_runtime_format`, `safetensors_is_source_not_runtime`, `embed_lm_head_tied`, `lm_head_uses_dense_gemm_object`, `embed_uses_gather_row_object`, `weight_unique_counted_once`, `analyzes_logical_dimensions`, `analyzes_consumers`, `analyzes_access`, `analyzes_tiling`, `analyzes_alignment`, `analyzes_conversion`, `includes_gdn`, `includes_convolution`, `includes_kv`, `vision_interface_is_not_a_node`, `payloads_restreamed`, `fusion_winner_selected`, `state_payload_in_artifact_selected`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_candidates`, `canonical_sentence_conversion`, `canonical_sentence_open_question`, `canonical_sentence_optimality`, `canonical_sentence_justification`.

Integer JSON fields that are counts/widths/bytes are JSON ints. `i_gdn_vs_s_rw` is JSON number `0.75`. `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_attn_core_vs_kv` are JSON ints `1`, `1`, `6`. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. `family_access_class["vision_deferred"]` and `family_layout_object["vision_deferred"]` are JSON `null`. `layout_object_primary_sequence_ids[2:6]` are JSON `null`. `tile_extent_divides_*` are JSON arrays of booleans. `policy_families` match TASK-08/09 exactly. `stage_kind_ids` match TASK-13/14 exactly. `consumer_sequence_ids` and `access_class_ids` match TASK-09 exactly.

TASK-09 `access_class_ids` order (7): `gather_row`, `dense_gemm`, `depthwise_conv`, `vector_param`, `state_kv`, `state_c`, `state_s`.

TASK-09 `consumer_sequence_ids` order (7): `seq_gemm_codes_then_scales`, `seq_gemm_interleaved_group`, `seq_gather_row`, `seq_lm_head_full`, `seq_outlier_extra`, `seq_state_s_dense`, `seq_specialized_tile`.

TASK-09 `policy_families` order (16): `norm_gamma`, `gdn_time_param`, `gdn_gate_proj`, `conv1d`, `linear_large_proj`, `attn_qkv`, `attn_out`, `mlp_up_gate`, `mlp_down`, `embed_table`, `lm_head`, `mtp_fc`, `vision_deferred`, `state_kv`, `state_c`, `state_s`.

TASK-09 `family_access_class_values`: `["vector_param","vector_param","dense_gemm","depthwise_conv","dense_gemm","dense_gemm","dense_gemm","dense_gemm","dense_gemm","gather_row","dense_gemm","dense_gemm",null,"state_kv","state_c","state_s"]`.

TASK-13/14 `stage_kind_ids` order (9): `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`.

Draft-status banner (documentation stage): a blockquote or italic line **before** the first `##`, matching TASK-13/14:

> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.

Title: `# Physical tensor layouts from consumers` (not `TASK-15`).

### Invariants

- Seven layout objects equal seven access classes; no eighth access class.
- Every TASK-08 defined family except `vision_deferred` maps to one layout object.
- GDN \(S\), conv \(W^{\text{conv}}\) / \(C\), and KV \(K,V\) each have named axes, orderings, tiles, and consumers.
- Zero selected orderings, tiles, conversions, decompositions, justifications, views, or sequences.
- No MEASURED tok/s, NLL, or layout ranking.
- No thread geometry and no sitting-GPU / MMA-shape instantiation.
- Logical ≠ physical; checkpoint orientation is not a required store order.
- GQA repeat not stored; paper \(S^\top\) is an ordering, not a second state; chunkwise GDN is not zero \(S\) traffic.
- If live `text_config` disagrees with a cited integer, the earlier document / config wins.

### Rejected alternatives

- Selecting any ordering, tile, or conversion because GEMM “looks different” from GEMV: rejected; no MEASURED evidence; TASK-14 left views open; this task **must** leave views open.
- Closing the parallel-decomposition question by treating the pairing table as justification: rejected; listing a hypothesis is not justification; CUDA analysis is TASK-17.
- Using TASK-16 occupancy or a datasheet MMA shape to pick tiles: rejected; TASK-16 is not a dependency; optimality before CUDA is forbidden; `mma_tile_extents_selected` false.
- Two compiled artifacts for decode vs prefill: rejected by TASK-09; one artifact, `view_binding` capability.
- Treating checkpoint \((d_\text{out},d_\text{in})\) as the required physical layout: rejected; logical ≠ physical.
- Storing GQA-repeated KV: rejected by TASK-03/02; stored rank stays \((4,T,256)\).
- Treating length-4 conv buffers with a current-token slot as extra math: rejected by TASK-02; delay is 3.
- Treating paper \(S^\top\) as extra state traffic: rejected; same map, unselected ordering.
- Treating chunkwise GDN as zero \(S\) traffic: rejected by TASK-13/14.
- Activation working-buffer layouts as first-class objects: rejected; ledger purpose is weights and persistent state.
- Adding an eighth consumer sequence for conv or KV: rejected; TASK-09 sequence list is locked; use `seq_specialized_tile` as an unselected view.
- Importing other `scripts/check_*.py` or restreaming safetensors: rejected.
- Inspecting Quartz, llama.cpp, or GGUF layouts: rejected until freeze.
- Selecting `alignment_grain` for the portable view: rejected; TASK-09 open decision stays open.
- Ruff/pytest/uv packaging for the checker: rejected; stdlib only.
- Making TASK-11/12/16 silent extra dependencies that block independence: rejected; cite node types / fusion / occupancy only through TASK-09/13/14 or as deferred CUDA.

### Discovered ledger work

none.

### Unresolved decisions

`none` (implementation choices for this increment are closed). The ledger open question — which planned parallel decompositions justify each candidate ordering and tile — is **intentionally unresolved** (`ledger_open_question_parallel_decomposition_closed` false). That is an acceptance outcome, not a missing dossier decision. Decode/prefill distinct views, ideal byte sequence, artifact boundary, alignment grain, fusion winners, and CUDA mappings remain owned by their tasks.

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/layout-strategy.md` exists with the twelve locked headings in order, five canonical sentences plus the justification sentence, GDN / Convolution / KV subsections, one Mermaid flowchart, and a JSON fence equal to live checker `--json`.
  - Six analysis dimensions are present and JSON `analyzes_*` all true.
  - GDN, convolution, and KV are included; JSON `includes_gdn` / `includes_convolution` / `includes_kv` true.
  - No selected layout/ordering/tile/conversion/decomposition; `optimality_claim_absent` true; forbidden winner/optimality phrases absent.
  - Ledger open question remains unresolved.
  - Coupled IDs: none.
- Tests/fixtures to add or change:
  - `scripts/check_layout_strategy.py` (create)
  - `docs/architecture/layout-strategy.md` (create; JSON fence is the fixture)
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_layout_strategy.py
python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --layout-strategy docs/architecture/layout-strategy.md
```

- Candidate quality: not required — no model execution or NLL; this increment is layout-candidate documentation. Ordering/tile/conversion **usefulness** is HYPOTHESIS prose, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/layout-strategy.md
python3 -m py_compile scripts/check_layout_strategy.py
python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --layout-strategy docs/architecture/layout-strategy.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run other `scripts/check_*.py` as a requirement of this task (verifier may spot-check TASK-04/06/09 integers and TASK-09/13/14 ids independently).

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/layout-strategy.md` (create)
  - `scripts/check_layout_strategy.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-09/13/14 deliverables unchanged)
- Definition of done: layout-strategy document published with locked seven layout objects, six analysis dimensions, GDN/conv/KV coverage, 14 unselected orderings, 9 unselected tile families, 9 unselected parallel decompositions, 4 unselected conversion hypotheses, 10 unselected justification hypotheses, no thread geometry, and **no optimality claim**; ledger open question **remains unresolved**; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-15 completion checkboxes can be marked at delivery **with the open question still recorded as unresolved**; fusion, packing selection, distinct views, and CUDA mapping remain open except for the named candidate layout space that fills `seq_specialized_tile` without selecting it.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T15:40:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-15.md`. Coupled IDs `none`. Document structure (12 headings), five canonical sentences plus justification sentence, seven layout objects (= access classes), six analysis dimensions, GDN/conv/KV coverage, 14 unselected orderings, 9 unselected tile families with extent candidates `[16,32,64,128,256]` and \(d_k\)-divisibility `256` false, 9 unselected parallel decompositions, 4 unselected conversion hypotheses, 10 unselected justification hypotheses, 8 layout-risk hypotheses, stdlib checker `scripts/check_layout_strategy.py`, JSON schema, and acceptance commands are closed. Ledger open question **kept unresolved** (`ledger_open_question_parallel_decomposition_closed` false; `parallel_decomposition_justifies_layout_selected` false; `optimality_claim_absent` true). `docs/architecture/layout-strategy.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit. No implementation.
- Performance evidence applied: N/A — layout-candidate documentation; ordering/tile/conversion/justification usefulness is hypothesis, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T15:48:12Z`; `telemetry_unavailable`
- Changed files (uncommitted):
  - `scripts/check_layout_strategy.py` (create; stdlib checker)
  - `docs/architecture/layout-strategy.md` (create; twelve headings, five canonical sentences plus justification sentence, GDN/Convolution/KV subsections, one Mermaid flowchart, JSON fence)
  - this dossier run record
- Outcome: Published the hardware-independent candidate layout space for seven layout objects (= TASK-09 access classes), six analysis dimensions, GDN/conv/KV coverage, 14 unselected orderings, 9 unselected tile families, 9 unselected parallel decompositions, 4 unselected conversion hypotheses, 10 unselected justification hypotheses, 8 layout-risk hypotheses. Ledger open question kept unresolved (`ledger_open_question_parallel_decomposition_closed` false; `n_orderings_selected` 0; `tile_size_selected` false; `optimality_claim_absent` true). `plan.md`, `task_ledger.md`, TASK-09/13/14 deliverables unchanged. No commit.
- Focused commands (repository root; cwd `/home/user/qw38`):

```text
python3 -m py_compile scripts/check_layout_strategy.py
# exit 0

python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
# exit 0; 171-key summary; JSON fence equals live object;
# n_layout_objects 7; ledger_open_question_parallel_decomposition_closed false;
# optimality_claim_absent true

python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --layout-strategy docs/architecture/layout-strategy.md
# exit 0
```

- Missing-config smoke (not a dossier focused command): `--config /no/such/config.json --json` exit 2.
- Performance evidence applied: N/A — layout-candidate documentation; no GPU/quality gates.

### Documentation

- Agent/model: `composer-2.5` (documentation subagent; parent/inherit mapping)
- Changes and evidence:
  - `docs/architecture/layout-strategy.md` — mechanical pass only. Draft-status banner (`unverified`) already present in TASK-07 / [`numerical-sensitivity.md`](../numerical-sensitivity.md) / [`semantic-graph.md`](../semantic-graph.md) / [`decode-plan.md`](../decode-plan.md) / [`prefill-plan.md`](../prefill-plan.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`); left unchanged. Authority table already cross-links this dossier, [`runtime-format-design.md`](../runtime-format-design.md) (TASK-09), [`decode-plan.md`](../decode-plan.md) (TASK-13), [`prefill-plan.md`](../prefill-plan.md) (TASK-14), [`model-inventory.md`](../model-inventory.md) (TASK-01) via those docs, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_layout_strategy.py`](../../../scripts/check_layout_strategy.py), and plan evidence policy in [`plan.md`](../plan.md). Twelve required `##` headings, five canonical sentences plus justification sentence, GDN / Convolution / KV `###` subsections, one Mermaid flowchart, and the JSON fence left unchanged. No locked integers, canonical sentences, layout-object ids, ordering ids, tile-family ids, parallel-decomposition ids, conversion ids, justification ids, layout-risk ids, or Mermaid node IDs edited. Ledger open question not closed (`ledger_open_question_parallel_decomposition_closed` false; `parallel_decomposition_justifies_layout_selected` false; `optimality_claim_absent` true). Frozen TASK-09/13/14 artifacts and `plan.md` not edited. `task_ledger.md` not edited (delivery stage).
- Commands:
  - `test -f docs/architecture/layout-strategy.md` — pass (exit 0).
  - `python3 -m py_compile scripts/check_layout_strategy.py` — pass (exit 0).
  - `python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 171 keys; `n_layout_objects` 7; `n_analysis_dimensions` 6; `n_persistent_state_coverages` 3; `n_orderings` 14; `n_orderings_selected` 0; `n_tile_families` 9; `tile_size_selected` false; `n_parallel_decompositions` 9; `n_conversion_hypotheses` 4; `n_justification_hypotheses` 10; `ledger_open_question_parallel_decomposition_closed` false; `optimality_claim_absent` true; `thread_geometry_absent` true; `hardware_independent` true; `decode_prefill_distinct_views_selected` false; `seq_specialized_tile_candidates_named` true; `tile_layout_deferred_to_task15` false; `n_layout_risks` 8; `n_diagrams` 1; `fusion_winner_selected` false; JSON fence source unchanged).
  - `python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --layout-strategy docs/architecture/layout-strategy.md` — pass (exit 0; twelve headings, JSON fence, one flowchart, five canonical sentences plus justification sentence, locked ids/integers, `HYPOTHESIS` / `hardware-independent` / `unresolved`; banner did not break the check).
  - Independent JSON fence spot-check (live `--json` object equals fenced JSON in `layout-strategy.md`, 171 keys byte-for-byte after parse) — pass.
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T15:49:13Z`; `telemetry_unavailable`

### Verification

- Agent/model: `composer-2.5` (verifier; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T15:49:00Z`; `telemetry_unavailable`
- Scope review (independent):
  - Deliverables: `docs/architecture/layout-strategy.md`, `scripts/check_layout_strategy.py` present; dossier updated; no other semantic edits.
  - Frozen artifacts unchanged: `docs/architecture/plan.md`, `runtime-format-design.md`, `decode-plan.md`, `prefill-plan.md`, TASK-09/13/14 checkers not modified.
  - `docs/architecture/task_ledger.md` shows only admission `TODO` → `IN PROGRESS` (expected; delivery stage not run).
  - Title `# Physical tensor layouts from consumers`; draft-status banner before first `##`; twelve locked `##` headings in order; `### GDN`, `### Convolution`, `### KV persistent state`; one Mermaid flowchart; five canonical sentences plus justification sentence present verbatim.
  - JSON fence byte-for-byte equals live `--json` (171 keys); locked counts verified: `n_layout_objects` 7, `n_analysis_dimensions` 6, `n_orderings` 14 / `n_orderings_selected` 0, `n_tile_families` 9 / `tile_size_selected` false, `n_parallel_decompositions` 9, `n_conversion_hypotheses` 4, `n_justification_hypotheses` 10, `n_layout_risks` 8, `n_diagrams` 1; `includes_gdn` / `includes_convolution` / `includes_kv` true; `analyzes_*` all true; `ledger_open_question_parallel_decomposition_closed` false; `optimality_claim_absent` true; `thread_geometry_absent` true; no forbidden winner phrases outside allowed negations; `UNKNOWN` only in Deferred vision.
  - Spot-check vs TASK-09/13: `access_class_ids`, `consumer_sequence_ids`, `policy_families`, `stage_kind_ids` match peer checkers.
  - `scripts/check_layout_strategy.py` stdlib-only (no imports of sibling `check_*.py`).
  - Candidate NLL: not required (documentation increment).
  - Performance evidence: N/A.
- Commands (repository root `/home/user/qw38`):

```text
python3 -m py_compile scripts/check_layout_strategy.py
# exit 0

python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
# exit 0; 171-key summary; n_layout_objects 7; ledger_open_question_parallel_decomposition_closed false; optimality_claim_absent true

python3 scripts/check_layout_strategy.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --layout-strategy docs/architecture/layout-strategy.md
# exit 0

test -f docs/architecture/layout-strategy.md
# exit 0

python3 scripts/check_layout_strategy.py --config /no/such/config.json --json
# exit 2 (missing-config smoke; not a dossier gate)

# Independent JSON fence parity (live --json vs first ```json fence in layout-strategy.md)
# exit 0; 171 keys equal

# Independent spot-check vs TASK-09/13 checkers (access_class_ids, consumer_sequence_ids, policy_families, stage_kind_ids)
# all PASS
```

- Verdict: **PASS** — all dossier acceptance conditions and repository-wide gates satisfied; ready for delivery (no commit in this stage).

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-15 only; coupled IDs `none`
- Outcome: TASK-15 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T15:51:00Z`; `telemetry_unavailable`

### Retries and escalation

none.

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/layout-strategy.md` (twelve locked headings, five canonical sentences plus justification sentence, seven layout objects, six analysis dimensions, GDN/conv/KV coverage, 14 unselected orderings, 9 unselected tile families, 9 unselected parallel decompositions, 4 unselected conversion hypotheses, 10 unselected justification hypotheses, 8 layout-risk hypotheses, one Mermaid flowchart, JSON fence); `scripts/check_layout_strategy.py` stdlib checker; independent JSON-fence equality confirmed; frozen upstream docs unchanged; fusion, packing selection, distinct views, and CUDA mapping remain open; ledger open question kept unresolved (`ledger_open_question_parallel_decomposition_closed` false; `optimality_claim_absent` true)
- Candidate measured delta: N/A — layout-candidate documentation
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (no GPU/quality gates)
- Throughput delta: N/A — TASK-15 does not execute or time the model
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification attempt 1 pass)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: none
