# TASK-09 — Design the custom runtime model format

## Control

- Primary ID: `TASK-09`
- Coupled IDs: `none`
- Dependencies: `TASK-03`, `TASK-06`, `TASK-08` (all DONE at admission)
- Status: `IN PROGRESS`
- Ledger acceptance: Analyze all requested representation and packing capabilities; Compare portable and backend-specific artifact approaches; Keep decisions open absent compelling evidence.

## Goal and boundaries

Produce `docs/architecture/runtime-format-design.md` as the Phase 1 **compiler-produced, consumer-oriented runtime model artifact** requirements and capability space for Qwen3.8-27B language+MTP. Close the three ledger completion criteria by (1) analyzing the locked representation and packing capabilities, (2) comparing portable versus backend-specific artifact approaches, and (3) leaving the ledger open question open: portable versus backend-specialized artifact boundaries and ideal consumer byte sequences. Do **not** select an on-disk winner, an ideal byte sequence, a scale-storage width, or a specialized tile.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Logical producers/consumers and the logical-≠-physical rule come from `docs/architecture/dataflow.md` (TASK-03). Weight/state traffic and access identities (gather versus full GEMM; unique weight read versus \(T\)-reuse) come from `docs/architecture/work-and-traffic.md` (TASK-06). Element formats, grouping, scales, outlier policies, 22 recipes, 16 policy families, and metadata **lower bounds** come from `docs/architecture/quantization-design-space.md` (TASK-08). Family occupancy comes from `docs/architecture/model-inventory.md` (TASK-01).
  - Label claims `OBSERVED` (inventory/config occupancy), `DERIVED` (packed-byte ceilings, header illustrations, occupancy products, dual-view size products), or `HYPOTHESIS` (every portable-versus-specialized tradeoff cell, every format-risk severity, every “ideal sequence” rationale). `UNKNOWN` only for vision-encoder internals deferred here. No new `MEASURED` payload statistics. No `MEASURED` quality or tok/s. No selected artifact.
  - GitHub Markdown math. Cite TASK-03 catalog/region IDs, TASK-06 traffic identities, TASK-08 recipe/family/risk ids. Do not rewrite forward math, redraw the TASK-03 DAG, re-stream safetensor payloads, recopy TASK-08’s 22-recipe research space as a new quantization grid, or choose TASK-15 tiles.
  - Allowed evidence: TASK-01 inventory, TASK-03 dataflow, TASK-06 work/traffic, TASK-08 quantization-design-space, sitting `config.json` `text_config`, plan evidence vocabulary, and general bit-packing / little-endian / checksum material. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf` as a format constraint. GGUF is not the runtime format; it remains a future black-box Pareto **reference** (TASK-18). Safetensors is the **source** checkpoint, not the runtime artifact.
- Non-goals:
  - No selected portable/specialized winner, no “ideal byte sequence is …”, no recommended alignment or scale-storage width (ledger open question stays open).
  - No compiler stages, profile ownership, or calibration corpora (TASK-10).
  - No semantic-graph execution contracts (TASK-11), fusion, or buffer reuse (TASK-12).
  - No decode/prefill **schedules** (TASK-13/14). One compiled artifact is shared; whether those schedules need distinct **views** stays open for TASK-14.
  - No physical tile/swizzle/MMA layouts (TASK-15). The format must be able to *name* a specialized view; it must not choose the tile.
  - No CUDA dtypes, kernels, or sitting-GPU numbers (TASK-16/17/19).
  - No quantization recipe winners, new bit widths, or learned vector-quant codebooks (TASK-08 scalar recipes remain the packable set; Pareto is TASK-18).
  - No activation working-dtype payloads in the artifact.
  - No new operators and no vision-encoder internals.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–08). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `dataflow.md`, `work-and-traffic.md`, `quantization-design-space.md`, `model-inventory.md`, `model-semantics.md`, `lifetime-and-state.md`, `numerical-sensitivity.md`, or `bf16-tensor-analysis.md`.
  - Do not stream safetensor payloads and do not import `scripts/analyze_bf16_tensors.py` or any `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib runtime-format checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-22` — central path: custom quantization → layouts/compiler; the study may define a runtime model format co-designed with quantization, layout, graph, and kernels; none is a fixed generic format.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint is the authority for weights, statistics, quantization, and **packing** research; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:85-88` — TASK-09 is the compiled runtime representation after TASK-08, before TASK-10.
- `docs/architecture/task_ledger.md` TASK-09 row — produces `docs/architecture/runtime-format-design.md`; purpose is requirements for a compiler-produced, consumer-oriented model artifact; open question (portable versus backend-specialized boundaries and ideal consumer byte sequences) is **not** closed here; completion is representation/packing capability analysis, portable-versus-backend comparison, and keeping decisions open absent compelling evidence.
- `docs/architecture/task_ledger.md` TASK-03 established results — 52 catalog IDs; logical ≠ physical; sharing rank is DERIVED, not a cache layout; `E` and `W_lm` are shared-weight stadium nodes with fan-out 2 each; embeddings and `lm_head` are **untied**.
- `docs/architecture/task_ledger.md` TASK-06 established results — language+MTP unique weight bytes `54641395712`; unique non-embed `52098598912`; embed gather 10240 B/row (0 MAC); `lm_head` 2542796800 B `vocab_memory`; MLP vs weights \(I=1\); \(S\) 150994944 B/step; bottleneck **labels** HYPOTHESIS.
- `docs/architecture/task_ledger.md` TASK-08 established results — four design dimensions; 22 recipes; 16 policy families without winners; metadata bytes are **lower bounds, not packed layouts**; `scale_storage_bytes_candidates` `[2, 4]`; `d_3bit_pack` defers non-byte-aligned layout to this task.
- `docs/architecture/dataflow.md` — canonical logical-≠-physical sentence; region IDs `embed`, `full_attn`, `linear_attn`, `mlp`, `primary_logits`, `mtp`; catalog consumers of packed weights and of state \(K,V,C,S\).
- `docs/architecture/work-and-traffic.md` — DERIVED unique weight bytes; embed is gather; decode unique+gather complete `52098619392`; state volumes for \(K,V,C,S\).
- `docs/architecture/quantization-design-space.md` — packable recipes and family candidate sets; metadata formulas; packing, scale-storage 2 vs 4, and runtime format explicitly deferred here.
- `docs/architecture/model-inventory.md` — 42 level-2 families; language+MTP 866 tensors / 27320697856 parameters / 54641395712 BF16 bytes; all checkpoint tensors BF16; embeddings and `lm_head` untied `(248320, 5120)`.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads.
- `scripts/check_quantization_design_space.py` / `scripts/check_work_and_traffic.py` / `scripts/check_dataflow.py` — checker-style precedent. TASK-09’s checker is a sibling; do not import them.
- `docs/architecture/task_ledger.md` TASK-10/13/14/15 — consumers of this format (compiler stages, schedules, physical layouts); do not perform those designs here.

## Performance evidence

N/A — runtime-format **requirements** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Packed byte counts are DERIVED arithmetic from TASK-08 lower bounds plus ceil/align illustrations. Portable-versus-specialized **labels** are HYPOTHESIS, not MEASURED load time or tok/s. Do not apply the performance-evidence checklist to rank kernels or claim a winning artifact.

- Measurement identity: N/A (no engine binary). TASK-06/08 integers are cited, not remeasured from payloads.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Format coverage is all 16 TASK-08 policy families (42 level-2 + 4 state ids) plus the locked capability and approach lists.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` occupancy product disagrees with TASK-01 family parameter counts, or a cited TASK-06/08 byte disagrees with those documents, stop and fail closed (do not remeasure).
- Claim types: occupancy `observed`/`derived`; packed illustrations `derived`; every portable/specialized tradeoff and format-risk severity `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Format-space completeness is 8 representation + 8 packing capabilities, 4 artifact approaches, 7 consumer sequences, 8 open decisions left unselected.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for format claims

If an occupancy product would disagree with TASK-01 / sitting `text_config`, or a cited weight/state byte would disagree with TASK-06, or a recipe/family/metadata lower bound would disagree with TASK-08, or a catalog/region id would disagree with TASK-03, the earlier document wins and this one is wrong.

- Prefill and decode share **one** compiled artifact. Distinct **views** of the same logical tensor are a capability (`view_binding`), not a selected decode/prefill split (TASK-14).
- Primary object is language+MTP **checkpoint parameters** after a TASK-08 recipe (role `param`). Secondary object is token-persistent **state schema** for \(K,V,C,S\) (role `state`): ranks and conceptual dtypes so a consumer can allocate. Whether zero-filled state **payloads** live in the artifact is an open decision.
- Activations and accumulators are **not** artifact payloads.
- Algebraic equivalents in TASK-02 are the same real map; packing acts on stored elements, not on a second mathematical model.
- Logical values do not imply allocation; a packed tensor is not a CUDA buffer.
- Do not inspect Quartz, llama.cpp, or GGUF byte layouts to “confirm” packing.
- Do not re-stream payloads. Cite TASK-06/08; do not invent new histograms.
- Consumer orientation: packing capabilities are shaped by TASK-03 consumers and TASK-06 access (gather row, dense GEMM, depthwise conv, vector param, state), not by safetensors shard order.

### Deliverable structure (`docs/architecture/runtime-format-design.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every occupancy number is `OBSERVED` or `DERIVED`. Every packed illustration is `DERIVED`. Every portable/specialized tradeoff cell and every format-risk severity is `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, inventory, dataflow, work/traffic, quantization-design-space, config, checker; evidence labels; in-scope (language+MTP compiled params + state schema + packing capabilities) vs deferred (vision encoder; TASK-15 tiles; TASK-10 compiler stages). State that the document specifies artifact **requirements and a capability space**, not a selected file format, kernel, or layout.
2. **Format convention** — the five canonical sentences below; what a capability is; GGUF is not the runtime format; safetensors is source-only; open question stays open.
3. **Artifact object model** — producer/consumer; kinds; what is in versus out of the artifact; shared bindings; one artifact for prefill and decode.
4. **Representation capabilities** — the eight locked representation ids; mapping of 16 policy families to access classes; TASK-08 recipes remain the packable set.
5. **Packing capabilities** — the eight locked packing ids; ceil/align formulas; instantiated illustrations (MLP, unique-non-embed, embed gather, \(S\), int3/int2/int6 grains, header overhead, dual-view size); scale-storage/placement/align/bit-order remain unselected.
6. **Consumer access and byte sequences** — seven named sequences from TASK-03/06 access; no ideal sequence.
7. **Portable versus backend-specific artifacts** — four approaches; six comparison dimensions; no winner.
8. **Open decisions** — the eight locked open ids, all unselected; format-risk hypotheses; one Mermaid summary (diagram 1 of 1).
9. **Deferred vision** — residual-stream interface only; empty packing list.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Format convention**, include these **five canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Representation and packing capabilities in this document are a research space, not selected winners.

> Portable versus backend-specialized artifact boundaries remain open absent compelling evidence.

> Packed byte counts in this document are DERIVED illustrations, not selected consumer layouts.

> Ideal consumer byte sequences remain an open question.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_capabilities` = sentence 2; `canonical_sentence_portable` = sentence 3; `canonical_sentence_packed` = sentence 4; `canonical_sentence_sequences` = sentence 5.

Bullets required under that heading:

- Eight representation capabilities and eight packing capabilities are complete for this task.
- Listing a capability, sequence, or artifact approach does not select it.
- Fan-out ≠ must-store still holds; a packed family is not a CUDA allocation.
- The ledger open question (portable versus backend-specialized boundaries and ideal consumer byte sequences) remains open.
- `models/Qwen3.8-27B-Q4_K_M.gguf` is not the runtime format and not a packing authority.
- The BF16 safetensors checkpoint is the source, not the compiled artifact.

### Artifact object model (lock)

JSON array `artifact_kinds` in this order: `manifest`, `payload`, `metadata_blob`, `sidecar`, `view`, `schema_state`. JSON `n_artifact_kinds` = 6.

| Kind | Meaning |
| --- | --- |
| `manifest` | Tensor directory: identity, TASK-08 family, recipe id, shape, access class, view list, integrity records. |
| `payload` | Packed element codes (or IEEE bytes for `keep_source` / `narrow_bf16` / `fp8_tensor`). |
| `metadata_blob` | Per-group scales and optional zero-points. |
| `sidecar` | `extract_high` sparse BF16 values plus indices; `mixed_group` width map. |
| `view` | One portable and/or one-or-more specialized projections of the same logical tensor. |
| `schema_state` | Ranks and conceptual dtypes for \(K,V,C,S\) allocation; not a cache layout. |

Producer: future TASK-10 compiler (not specified here). Consumers: TASK-13/14 schedules, TASK-15 layouts, TASK-17 kernels (not specified here). Source: `.cache/authorities/qwen3.8-27b-transformers` safetensors (not copied through as the runtime bytes).

JSON booleans (lock):

- `activations_in_artifact` = false
- `kernels_in_artifact` = false
- `tokenizer_in_artifact` = false
- `gguf_is_not_the_runtime_format` = true
- `safetensors_is_source_not_runtime` = true
- `embed_lm_head_tied` = false
- `decode_prefill_share_artifact` = true
- `decode_prefill_distinct_views_selected` = false
- `tile_layout_deferred_to_task15` = true
- `artifact_boundary_selected` = false
- `ideal_byte_sequence_selected` = false
- `state_payload_in_artifact_selected` = false
- `state_schema_required` = true
- `learned_codebooks_required` = false
- `payloads_restreamed` = false

Shared binding: catalog consumers `e` and `e_next` share one `E` payload; `logits_0` and `logits_1` share one `W_lm` payload. Untied embed versus `lm_head` remains two payloads. JSON `shared_weight_ids` = `["E","W_lm"]`.

### Representation capabilities (lock)

JSON array `representation_capability_ids` in this exact order (8 ids). JSON `n_representation_capabilities` = 8.

| id | Must express | Authority |
| --- | --- | --- |
| `tensor_identity` | Name, rank, TASK-08 policy family, recipe id, source level-2 id | TASK-01/08 |
| `element_encoding` | All eight TASK-08 formats: `source`, `bf16`, `fp8_e4m3`, `int8`, `int6`, `int4`, `int3`, `int2` | TASK-08 `formats` |
| `group_metadata` | Grouping `none` / `per_tensor` / `per_row` / `per_col` / `block_g32` / `block_g64` / `block_g128`; ragged last group; group-clips-to-axis; scales; optional zero-point | TASK-08 |
| `outlier_sidecar` | `extract_high` sparse BF16 + index; `clip` stores no extra elements | TASK-08 `outlier_ids` |
| `mixed_width_map` | Per-group wider/narrower integer width for `mixed_group` | TASK-08 |
| `shared_binding` | Single payload with multiple catalog consumers (`E`, `W_lm`) | TASK-03 |
| `access_class` | Consumer access class per family (table below) | TASK-03/06 |
| `state_schema` | \(K,V\) rank \((4,T,256)\) BF16 conceptual; \(C\) \(3\times 10240\) BF16; \(S\) \((48,128,128)\) F32 conceptual; 17 KV instances including MTP; 48 C/S instances | TASK-03/04/06 |

Do not add a ninth representation capability (`codebook`, `kernel`, `activation_payload`, `gguf_type`). The 22 TASK-08 recipes are the **only** packable combinations this document must support. Families list subsets there; this document does not add recipes.

JSON array `access_class_ids` in this order: `gather_row`, `dense_gemm`, `depthwise_conv`, `vector_param`, `state_kv`, `state_c`, `state_s`. JSON `n_access_classes` = 7.

JSON object `family_access_class` maps each of the 16 TASK-08 `policy_families` (same order as TASK-08) to one access class or `null` for `vision_deferred`:

| family | `family_access_class` |
| --- | --- |
| `norm_gamma` | `vector_param` |
| `gdn_time_param` | `vector_param` |
| `gdn_gate_proj` | `dense_gemm` |
| `conv1d` | `depthwise_conv` |
| `linear_large_proj` | `dense_gemm` |
| `attn_qkv` | `dense_gemm` |
| `attn_out` | `dense_gemm` |
| `mlp_up_gate` | `dense_gemm` |
| `mlp_down` | `dense_gemm` |
| `embed_table` | `gather_row` |
| `lm_head` | `dense_gemm` |
| `mtp_fc` | `dense_gemm` |
| `vision_deferred` | `null` |
| `state_kv` | `state_kv` |
| `state_c` | `state_c` |
| `state_s` | `state_s` |

JSON `family_access_class_values` parallel to `policy_families`: `["vector_param","vector_param","dense_gemm","depthwise_conv","dense_gemm","dense_gemm","dense_gemm","dense_gemm","dense_gemm","gather_row","dense_gemm","dense_gemm",null,"state_kv","state_c","state_s"]`.

Embed gather is **not** a GEMM packing class (TASK-06: 0 MAC, 10240 B/row BF16). `lm_head` **is** `dense_gemm` despite matching embed shape `(248320, 5120)` (TASK-06 `vocab_memory`). Conv1d uses TASK-05 squeezed rank-2 `(10240, 4)`.

Every defined family except `vision_deferred` must be packable as `keep_source` (BF16 LE elements, no scale) **and** as every recipe in that family’s TASK-08 `family_candidates`. JSON `keep_source_packable_on_all_defined_families` true. Do not recopy `family_candidates` arrays; cite TASK-08 and require every `candidate_recipe_ids` id as a markdown substring.

### Packing capabilities (lock)

JSON array `packing_capability_ids` in this exact order (8 ids). JSON `n_packing_capabilities` = 8.

| id | Must express | Selected here? |
| --- | --- | --- |
| `bit_pack` | Integer codes packed into bytes at the group grain (tensor grain if `per_tensor` / IEEE-like). Formula below. | No winner among `code_bit_order_candidates` |
| `scale_storage` | Scale (and zp) widths from TASK-08 `scale_storage_bytes_candidates` `[2, 4]` | No (2 vs 4 open) |
| `scale_placement` | `sidecar_array` versus `interleaved_group` | No |
| `alignment_pad` | Payload padded to a grain in `alignment_grain_candidates` | No |
| `endian_le` | Portable view payloads are little-endian bytes (source BF16 and TASK-05 decode are LE). Specialized views may swizzle (TASK-15). | Portable LE is a **requirement** for the portable view only, not a specialized-layout winner |
| `row_addressable` | A consumer can address one embed row without unpacking the whole table | Capability required; stride/pad open |
| `integrity_record` | Per-payload checksum/version slots in the manifest | Algorithm open |
| `view_binding` | Same logical tensor may have a portable view and zero or more specialized views | Which views exist is open |

JSON `code_bit_order_candidates` = `["lsb_first","msb_first"]`. JSON `scale_placement_candidates` = `["sidecar_array","interleaved_group"]`. JSON `alignment_grain_candidates` = `[1, 16, 32, 128, 256]`. JSON `scale_storage_bytes_candidates` = `[2, 4]`. JSON `integrity_algorithm_candidates` = `["none","checksum"]`. JSON `scale_storage_illustration_bytes` = 2. JSON `alignment_illustration_grain` = 1. JSON `scale_placement_illustration` = `"sidecar_array"`.

Packed payload at integer bit-width \(b\) on \(n\) elements:

\[
B_{\text{payload,pack}}=\Bigl\lceil\frac{n\,b}{8}\Bigr\rceil.
\]

Group grain: for grouping size \(g\), pack each group with \(\lceil g_{\text{eff}} b/8\rceil\) then concatenate; last group may be ragged (`ragged_last_group` true, inherited from TASK-08). IEEE-like: \(B_{\text{payload,pack}}=n\times\) element bytes (`bytes_bf16` = 2, `bytes_f32` = 4, `bytes_fp8` = 1).

Align:

\[
B_{\text{aligned}}=\Bigl\lceil B_{\text{payload,pack}}/A\Bigr\rceil A.
\]

Metadata lower bound unchanged from TASK-08: \(n_g=\lceil n/g\rceil\), \(B_{\text{meta}}=n_g(s+z)\). Illustration packing uses \(A=1\), \(s=2\), \(z=0\), sidecar scales, so **tight packed totals equal TASK-08 lower bounds** for the named MLP / unique-non-embed / embed-gather illustrations. JSON `packing_illustration_matches_task08_lower_bound` true. JSON `payload_formula_uses_ceil` true.

**Grain identities** (DERIVED; must appear as these integers):

| JSON key | Value | Meaning |
| --- | ---: | --- |
| `int3_g32_group_payload_bytes` | 12 | \(32\times 3/8=12\) exact; TASK-08 `d_3bit_pack` layout grain |
| `int2_g32_group_payload_bytes` | 8 | \(32\times 2/8=8\) |
| `int6_row_hidden_payload_bytes` | 3840 | \(5120\times 6/8=3840\) one hidden-width row |
| `int4_row_hidden_payload_bytes` | 2560 | \(5120\times 4/8=2560\) |
| `embed_gather_int4_row_bytes` | 2562 | 2560 + one 2-byte scale (TASK-08 illustration C) |
| `embed_gather_bf16_bytes` | 10240 | TASK-06 gather row |

**Illustration A — language MLP** (same as TASK-08, now labelled packed-tight): \(n=17112760320\), int4, \(s=2\), \(A=1\):

| \(g\) | \(B_{\text{payload,pack}}\) | \(B_{\text{meta}}\) | \(B\) | \(B/B_{\text{bf16}}\) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 8556380160 | 1069547520 | 9625927680 | 0.28125 |
| 128 | 8556380160 | 267386880 | 8823767040 | 0.2578125 |

JSON keys match TASK-08: `mlp_n`, `mlp_bf16_bytes`, `mlp_int4_payload_bytes`, `mlp_int4_g32_total_bytes`, `mlp_int4_g32_over_bf16`, `mlp_int4_g128_total_bytes`, `mlp_int4_g128_over_bf16`.

**Illustration B — unique non-embed language+MTP:** \(n=26049299456\), \(B_{\text{bf16}}=52098598912\), int4 \(g=128\) \(s=2\) \(A=1\): \(B=13431670032\), ratio \(0.2578125\). JSON keys `unique_non_embed_n`, `unique_non_embed_bf16_bytes`, `unique_non_embed_int4_g128_total_bytes`, `unique_non_embed_int4_g128_over_bf16`. Label: illustration, **not** a selected profile.

**Illustration C — embed gather:** as TASK-08 / table above. Access class `gather_row` requires `row_addressable` packing so decode need not stream 2542796800 B.

**Illustration D — \(S\):** conceptual F32 150994944 B/step; BF16 store 75497472 B; FP8 37748736 B. JSON `s_f32_bytes`, `s_bf16_bytes`, `s_fp8_bytes`. Schema required; payload-in-artifact unselected.

**Illustration E — manifest header overhead:** 64 bytes/tensor × 866 language+MTP tensors = 55424 B. JSON `n_language_mtp_tensors` 866, `container_header_illustration_bytes_per_tensor` 64, `container_header_illustration_total_bytes` 55424. 64 is an illustration constant, not a selected header size.

**Illustration F — dual-view store amplification:** two copies of illustration B = \(2\times 13431670032=26863340064\) B, still below BF16 unique-non-embed 52098598912. JSON `dual_view_int4_g128_unique_non_embed_bytes` 26863340064. DERIVED size only; **not** a recommendation to store two views.

Sidecar `extract_high` packed size is data-dependent (\(n_{\text{out}}\times(2+\text{index bytes})\)); do **not** instantiate a tensor-level sidecar byte count from TASK-05 fractions (those fractions shaped TASK-08 candidates, not packed layouts). JSON `outlier_sidecar_bytes_instantiated` false.

### Consumer access and byte sequences (lock)

JSON array `consumer_sequence_ids` in this exact order (7 ids). JSON `n_consumer_sequences` = 7. Parallel `consumer_sequence_access_classes`. None is selected as ideal. JSON `ideal_byte_sequence_selected` false.

| id | Access | Sequence (capability, not a winner) |
| --- | --- | --- |
| `seq_gemm_codes_then_scales` | `dense_gemm` | Read packed codes for a contraction, then sidecar scales (TASK-06 \(I=1\) MLP/`lm_head` vs weights) |
| `seq_gemm_interleaved_group` | `dense_gemm` | For each group: scale then codes |
| `seq_gather_row` | `gather_row` | One vocab row of codes + one scale (10240 B BF16 or 2562 B int4-row illustration) |
| `seq_lm_head_full` | `dense_gemm` | Entire `lm_head` unique 2542796800 B-class table every decode (TASK-06 `vocab_memory`; TASK-08 `d_lm_head_unpack`) |
| `seq_outlier_extra` | any `extract_high` | Additional irregular BF16 sidecar gathers (TASK-08 `d_outlier_gather`) |
| `seq_state_s_dense` | `state_s` | Dense \(S\) 150994944 B/step conceptual F32 (TASK-06 `state_memory`) |
| `seq_specialized_tile` | specialized view | Backend tile/swizzle order; **layout is TASK-15**; named only as a sequence the format must be able to store |

JSON `consumer_sequence_access_classes` = `["dense_gemm","dense_gemm","gather_row","dense_gemm","dense_gemm","state_s","dense_gemm"]` with the last meaning “specialized view of a dense family” (do not invent an eighth access class). Prose must say `seq_specialized_tile` is a **view** sequence, not a new access class.

Prose required: a packing that is convenient for `seq_gemm_interleaved_group` may break `seq_gather_row` (HYPOTHESIS `f_gather_stride`). Do not rank sequences by wall time. Do not name CUDA kernels.

### Portable versus backend-specific artifacts (lock)

JSON array `artifact_approach_ids` in this exact order (4 ids). JSON `n_artifact_approaches` = 4. JSON `artifact_boundary_selected` false.

| id | Meaning |
| --- | --- |
| `portable_only` | One backend-neutral packed view per logical tensor; a backend unpacks or converts at load or in-kernel |
| `backend_specialized_only` | Only a backend-specific packed view; recompile/repack per backend |
| `portable_plus_specialized_views` | Portable view plus optional specialized views in the same artifact (`view_binding`) |
| `manifest_plus_backend_blobs` | Portable manifest; payloads may be backend blobs referenced by the manifest without duplicating a portable payload |

JSON array `comparison_dimension_ids` in this order (6 ids). JSON `n_comparison_dimensions` = 6.

| id | Portable-only (HYPOTHESIS) | Specialized-only (HYPOTHESIS) | Dual-view / manifest+blobs (HYPOTHESIS except size DERIVED) |
| --- | --- | --- | --- |
| `compile_once` | One compiler emission serves every backend after convert | Emission is per backend | Manifest once; blobs may be per backend |
| `load_convert` | May pay unpack/convert (pairs with TASK-08 `d_weight_dequant`) | Map/load may skip convert | Convert only if the needed view is absent |
| `byte_sequence` | Codes+meta in a backend-neutral order | Tile/swizzle/MMA-ready (TASK-15 fills) | Both sequences may coexist |
| `tile_freedom` | TASK-15 layouts applied at load or by a view builder | Layout baked into payload | Specialized view bakes a layout; portable view does not |
| `store_amplification` | One copy | One copy per backend | Illustration F DERIVED 26863340064 B if two full int4-g128 unique-non-embed copies |
| `consumer_portability` | Any consumer that understands the portable view | One kernel/layout family | Portable consumers ignore specialized blobs |

Do not mark any approach `selected`, `recommended`, `better`, or `required`. Do not treat GGUF as `portable_only`. Safetensors is source, not an approach id.

### Open decisions (lock)

JSON array `open_decision_ids` in this exact order (8 ids). JSON `n_open_decisions` = 8. JSON object `open_decision_selected` maps each id to `false`. JSON `n_open_decisions_selected` = 0.

| id | What stays unselected | Owner of a future close |
| --- | --- | --- |
| `artifact_boundary` | Which of the four `artifact_approach_ids` | This ledger open question; not TASK-18 quality |
| `ideal_byte_sequence` | Which of the seven `consumer_sequence_ids` | This ledger open question; TASK-13/14/15 may inform |
| `scale_storage_bytes` | 2 vs 4 | Packing; TASK-08 left it here |
| `scale_placement` | sidecar vs interleaved | Packing |
| `alignment_grain` | 1 / 16 / 32 / 128 / 256 | Packing; TASK-15 may constrain specialized views |
| `code_bit_order` | `lsb_first` vs `msb_first` | Packing |
| `integrity_algorithm` | `none` vs `checksum` (and which checksum) | TASK-10 integrity stage |
| `state_payload_inclusion` | Schema-only vs storing zero templates | TASK-04/13; schema remains required |

JSON `pareto_frontier_selected` false (TASK-08/18; restated so this document cannot close it). JSON `compiler_profile_selected` false (TASK-10).

### Format-risk hypotheses (lock)

JSON array `format_risk_ids` in this exact order (6 ids). Parallel `format_risk_severities`. Every severity is HYPOTHESIS. JSON `n_format_risks` = 6.

| ID | Ties to | Severity | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `f_unpack_portable` | `portable_only`, TASK-08 `d_weight_dequant` | high | A portable-only artifact may add unpack/convert work on decode GEMMs |
| `f_repack_specialized` | `backend_specialized_only` | medium | Specialized-only forces a new packed artifact per backend |
| `f_dual_view_size` | Illustration F | low | Storing portable+specialized copies amplifies store (size is DERIVED; **impact** is HYPOTHESIS) |
| `f_gather_stride` | `row_addressable` vs GEMM sequences | medium | A GEMM-friendly pack may make embed-row gather non-contiguous |
| `f_3bit_shift` | TASK-08 `d_3bit_pack`, grain 12 B | medium | int3/int6 still need shift/mask even when the g32 grain is an integer number of bytes |
| `f_outlier_index` | `outlier_sidecar` | medium | Sparse sidecar indices are irregular extra traffic (pairs with `d_outlier_gather`) |

JSON `format_high_ids`: `f_unpack_portable`. `format_medium_ids`: `f_repack_specialized`, `f_gather_stride`, `f_3bit_shift`, `f_outlier_index`. `format_low_ids`: `f_dual_view_size`. JSON `n_format_high` = 1, `n_format_medium` = 4, `n_format_low` = 1.

Do not rank these by wall time. Do not convert TASK-06 bottleneck labels into measurements. Cite TASK-08 decode-risk ids as **related**, not as closed packing winners.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 8 (Open decisions). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 22 recipes, or 866 tensors.

Required IDs **inside that fence**: `representation`, `packing`, `portable`, `specialized`, `gather`, `gemm`, `state`, `open`.

JSON `n_diagrams` is 1. `diagram_ids` is `["representation","packing","portable","specialized","gather","gemm","state","open"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger packing is **UNKNOWN**. `vision_deferred` has `family_access_class` null and no packing recipes. Do not add vision payloads. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_runtime_format_design.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py` or `scripts/analyze_bf16_tensors.py`; duplicate the small `text_config` arithmetic needed for occupancy and the MLP/embed/metadata products (same identities as TASK-08 illustrations). Duplicate TASK-08 recipe/family id lists as constants; do not import them.

CLI (cwd = repository root):

```text
python3 scripts/check_runtime_format_design.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_runtime_format_design.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --runtime-format-design docs/architecture/runtime-format-design.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: `mlp_n`, `embed_n`, family BF16 byte citations matching TASK-01/06, packed illustrations matching TASK-08 tight totals, header/dual-view products. Constant fields: canonical sentences, capability/approach/sequence/open/risk lists, TASK-08 recipe/family ids.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--runtime-format-design PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all five canonical sentences verbatim, (5) every `representation_capability_ids`, `packing_capability_ids`, `artifact_approach_ids`, `consumer_sequence_ids`, `open_decision_ids`, `format_risk_ids`, `access_class_ids`, `policy_families`, and `candidate_recipe_ids` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer/decimal below present as a decimal or integer substring, (10) the words `HYPOTHESIS` and `not selected winners` present, (11) none of the forbidden winner phrases: `portable is better`, `specialized is better`, `recommend portable`, `recommend specialized`, `selected artifact`, `ideal byte sequence is`, `artifact boundary is`, `should use GGUF`, `GGUF is the runtime format`, `selected winner` (allow the substring only inside `not a selected winner` / `not selected winners`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks TASK-06/08 integers against those documents).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `mlp_n==17112760320` and `mlp_n == 3 * n_decoder_layers * intermediate_size * hidden_size`
- `embed_n==1271398400` and `embed_n == vocab_size * hidden_size`
- `n_language_mtp_tensors==866`, `n_language_mtp_parameters==27320697856`
- `mlp_bf16_bytes==34225520640==2*mlp_n`
- `mlp_int4_g32_over_bf16==0.28125` and `mlp_int4_g128_over_bf16==0.2578125`
- `mlp_int4_g32_total_bytes==9625927680`, `mlp_int4_g128_total_bytes==8823767040`
- `unique_non_embed_int4_g128_total_bytes==13431670032`, `unique_non_embed_int4_g128_over_bf16==0.2578125`
- `weight_bytes_language_mtp_excl_vision==54641395712`, `weight_bytes_unique_non_embed==52098598912`
- `embed_gather_bf16_bytes==10240`, `embed_gather_int4_row_bytes==2562`
- `s_f32_bytes==150994944`, `kv_bytes_all_per_token==69632`, `c_bytes_all==2949120`
- `int3_g32_group_payload_bytes==12`, `int2_g32_group_payload_bytes==8`, `int6_row_hidden_payload_bytes==3840`
- `container_header_illustration_total_bytes==55424==866*64`
- `dual_view_int4_g128_unique_non_embed_bytes==26863340064==2*13431670032`
- `n_representation_capabilities==8`, `n_packing_capabilities==8`, `n_artifact_approaches==4`
- `n_consumer_sequences==7`, `n_open_decisions==8`, `n_open_decisions_selected==0`
- `n_format_risks==6`, `n_format_high==1`, `n_format_medium==4`, `n_format_low==1`
- `n_access_classes==7`, `n_artifact_kinds==6`, `n_diagrams==1`, `n_comparison_dimensions==6`
- `candidate_recipe_ids` equals the TASK-08 22-id list; `policy_families` equals the TASK-08 16-id list
- `family_access_class["vision_deferred"] is None`, `family_access_class["embed_table"]=="gather_row"`, `family_access_class["lm_head"]=="dense_gemm"`
- every `open_decision_selected` value is False
- `artifact_boundary_selected is False`, `ideal_byte_sequence_selected is False`
- `gguf_is_not_the_runtime_format is True`, `safetensors_is_source_not_runtime is True`
- `activations_in_artifact is False`, `kernels_in_artifact is False`, `learned_codebooks_required is False`
- `decode_prefill_share_artifact is True`, `state_schema_required is True`, `embed_lm_head_tied is False`
- `packing_illustration_matches_task08_lower_bound is True`, `payload_formula_uses_ceil is True`
- `portable_payload_little_endian is True`, `outlier_sidecar_bytes_instantiated is False`, `payloads_restreamed is False`

Locked document integers/decimals the `--runtime-format-design` check must find:

`5120`, `17408`, `248320`, `866`, `27320697856`, `17112760320`, `34225520640`, `2542796800`, `54641395712`, `52098598912`, `150994944`, `69632`, `2949120`, `10240`, `2562`, `2560`, `3840`, `12`, `8`, `64`, `55424`, `8556380160`, `9625927680`, `8823767040`, `13431670032`, `26863340064`, `0.28125`, `0.2578125`

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `bytes_bf16`, `bytes_f32`, `bytes_fp8`,

`n_language_mtp_tensors`, `n_language_mtp_parameters`, `mlp_n`, `embed_n`, `mlp_bf16_bytes`, `mlp_int4_payload_bytes`, `mlp_int4_g32_meta_bytes`, `mlp_int4_g32_total_bytes`, `mlp_int4_g32_over_bf16`, `mlp_int4_g128_meta_bytes`, `mlp_int4_g128_total_bytes`, `mlp_int4_g128_over_bf16`,

`weight_bytes_language_mlp`, `weight_bytes_lm_head`, `weight_bytes_embed_table`, `weight_bytes_language_linear_attn`, `weight_bytes_language_self_attn`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `unique_non_embed_n`, `unique_non_embed_bf16_bytes`, `unique_non_embed_int4_g128_total_bytes`, `unique_non_embed_int4_g128_over_bf16`,

`embed_gather_bf16_bytes`, `embed_gather_int4_row_bytes`, `int4_row_hidden_payload_bytes`, `int6_row_hidden_payload_bytes`, `int3_g32_group_payload_bytes`, `int2_g32_group_payload_bytes`, `s_f32_bytes`, `s_bf16_bytes`, `s_fp8_bytes`, `kv_bytes_all_per_token`, `c_bytes_all`,

`scale_storage_illustration_bytes`, `scale_storage_bytes_candidates`, `alignment_illustration_grain`, `alignment_grain_candidates`, `scale_placement_illustration`, `scale_placement_candidates`, `code_bit_order_candidates`, `integrity_algorithm_candidates`, `container_header_illustration_bytes_per_tensor`, `container_header_illustration_total_bytes`, `dual_view_int4_g128_unique_non_embed_bytes`,

`ragged_last_group`, `group_clips_to_axis`, `payload_formula_uses_ceil`, `packing_illustration_matches_task08_lower_bound`, `portable_payload_little_endian`, `outlier_sidecar_bytes_instantiated`,

`artifact_kinds`, `n_artifact_kinds`, `representation_capability_ids`, `n_representation_capabilities`, `packing_capability_ids`, `n_packing_capabilities`, `access_class_ids`, `n_access_classes`, `policy_families`, `family_access_class`, `family_access_class_values`, `candidate_recipe_ids`, `n_candidate_recipes`, `keep_source_packable_on_all_defined_families`, `shared_weight_ids`,

`consumer_sequence_ids`, `consumer_sequence_access_classes`, `n_consumer_sequences`, `artifact_approach_ids`, `n_artifact_approaches`, `comparison_dimension_ids`, `n_comparison_dimensions`,

`open_decision_ids`, `open_decision_selected`, `n_open_decisions`, `n_open_decisions_selected`, `format_risk_ids`, `format_risk_severities`, `format_high_ids`, `format_medium_ids`, `format_low_ids`, `n_format_risks`, `n_format_high`, `n_format_medium`, `n_format_low`,

`activations_in_artifact`, `kernels_in_artifact`, `tokenizer_in_artifact`, `gguf_is_not_the_runtime_format`, `safetensors_is_source_not_runtime`, `embed_lm_head_tied`, `decode_prefill_share_artifact`, `decode_prefill_distinct_views_selected`, `tile_layout_deferred_to_task15`, `artifact_boundary_selected`, `ideal_byte_sequence_selected`, `state_payload_in_artifact_selected`, `state_schema_required`, `learned_codebooks_required`, `compiler_profile_selected`, `pareto_frontier_selected`, `payloads_restreamed`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_capabilities`, `canonical_sentence_portable`, `canonical_sentence_packed`, `canonical_sentence_sequences`.

Integer JSON fields that are counts/widths/bytes are JSON ints. Ratios `mlp_int4_g32_over_bf16`, `mlp_int4_g128_over_bf16`, `unique_non_embed_int4_g128_over_bf16` are JSON numbers `0.28125` and `0.2578125`. Booleans are JSON booleans. Candidate arrays of ints are JSON arrays of ints. `full_attention_indices` is a JSON array of ints. `family_access_class["vision_deferred"]` is JSON `null`. `open_decision_selected` is a JSON object keyed in `open_decision_ids` order with JSON `false` values. `policy_families` and `candidate_recipe_ids` match TASK-08 exactly.

TASK-08 `candidate_recipe_ids` order (22): `keep_source`, `narrow_bf16`, `fp8_tensor`, `i8_tensor`, `i8_row`, `i8_row_asym`, `i8_g32`, `i6_row`, `i4_row`, `i4_col`, `i4_g32`, `i4_g64`, `i4_g128`, `i4_g128_p99`, `i4_g128_rms`, `i4_g128_extract`, `i4_g32_mixed`, `i4_row_extract`, `i4_clip`, `i3_g32`, `i3_g32_extract`, `i2_g32_extract`.

TASK-08 `policy_families` order (16): `norm_gamma`, `gdn_time_param`, `gdn_gate_proj`, `conv1d`, `linear_large_proj`, `attn_qkv`, `attn_out`, `mlp_up_gate`, `mlp_down`, `embed_table`, `lm_head`, `mtp_fc`, `vision_deferred`, `state_kv`, `state_c`, `state_s`.

### Stage split

- **Implementation** writes `scripts/check_runtime_format_design.py` **and** `docs/architecture/runtime-format-design.md` (object model, 8+8 capabilities, packing illustrations, 7 sequences, 4 approaches, 8 open decisions, 6 HYPOTHESIS format risks, JSON fence). Runs `--json` and `--runtime-format-design` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-04 / `work-and-traffic.md` style), Authority table links to this dossier / inventory / dataflow / work-and-traffic / quantization-design-space / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, capability ids, approach ids, open-decision ids, severities, or Mermaid node IDs. Does not edit TASK-03/06/08 artifacts.
- **Verification** independently re-runs focused commands, recomputes `mlp_n` / packed-tight totals / unique-non-embed bytes / header product / dual-view product from sitting `text_config` (not from JSON echo), spot-checks cited TASK-06/08 integers against `docs/architecture/work-and-traffic.md` and `docs/architecture/quantization-design-space.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-format, no winner phrases, no `plan.md` edit, no payload I/O, that every format-risk row is labelled HYPOTHESIS, and that all eight open decisions remain unselected. Ledger open question left open.

- Invariants:
  - Ten level-2 headings in the locked order; five canonical sentences verbatim; 8 representation + 8 packing capabilities; 4 artifact approaches; 7 consumer sequences; 8 open decisions all unselected; 6 format risks; one Mermaid flowchart with required IDs.
  - Prefill/decode share one artifact; primary includes language+MTP params; state **schema** is required; state **payload** unselected; activations/kernels/tokenizer out of artifact.
  - Tight packing illustrations match TASK-08 lower bounds; `payload_formula_uses_ceil` true; portable view little-endian.
  - `keep_source` packable on every defined family; `vision_deferred` empty; no selected artifact or ideal sequence; `artifact_boundary_selected` false.
  - GGUF is not the runtime format. Safetensors is source-only. Payloads are not restreamed.
  - Logical values do not imply allocation; capabilities are not winners; packed bytes are not selected layouts; ideal sequences remain open.
  - Vision encoder remains unexpanded. Tile layouts deferred to TASK-15. Compiler stages deferred to TASK-10.
- Rejected alternatives:
  - Selecting `portable_only` or `backend_specialized_only` from TASK-06 intensity identities: rejected; those identities are not format measurements; the ledger requires decisions stay open absent compelling evidence (none exists here).
  - Treating GGUF Q4_K_M or safetensors as the runtime artifact: rejected; plan.md source versus black-box reference; TASK-18 for GGUF comparison.
  - Re-streaming payloads to produce new MEASURED packed sizes: rejected; TASK-08 lower bounds plus ceil/align arithmetic suffice.
  - Learned vector-quant codebooks as a representation capability: rejected; TASK-08 scalar recipes only.
  - Storing activations or CUDA kernels in the artifact: rejected; consumer-oriented **model** artifact, not a fatbin or working set.
  - Tying embed and `lm_head` because they share shape: rejected; TASK-01/03 untied; different access classes (`gather_row` vs `dense_gemm`).
  - Closing TASK-14’s decode/prefill view question by baking two artifacts: rejected; one artifact, `view_binding` capability, view split unselected.
  - Choosing MMA tiles, swizzles, or alignment 128 as the specialized layout: TASK-15.
  - Designing compiler stages or quality/balanced/compression profiles: TASK-10.
  - Instantiating `extract_high` sidecar bytes from TASK-05 `frac_out_6x`: rejected; data-dependent; not a packed layout.
  - Using CUDA dtypes or GGUF sizes as packed element sizes: rejected; TASK-08 formats + conceptual F32 \(S\).
  - Inspecting Quartz or llama.cpp for “real” packing: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–08.
  - Editing frozen TASK-01/03/06/08 docs or `plan.md`.
  - Importing other `check_*.py` or `analyze_bf16_tensors.py`.
  - Closing the ledger open question “portable versus backend-specialized artifact boundaries and ideal consumer byte sequences”.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/runtime-format-design.md` exists and follows the heading list above.
  - Eight representation and eight packing capabilities are named and analyzed; packing formulas instantiated (MLP \(g=32/128\) ratios 0.28125 / 0.2578125 matching TASK-08; unique-non-embed illustration; embed gather 2562; int3 g32 grain 12; header 55424; dual-view 26863340064).
  - Four artifact approaches are compared on six dimensions without a winner; seven consumer sequences are tabulated without an ideal sequence; eight open decisions are listed and unselected.
  - Format risks (6) are tabulated as HYPOTHESIS and do not claim experimental proof or an artifact selection.
  - Five canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; GGUF is not the runtime format; no payload re-stream; no `plan.md` edit; ledger open question left open.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_runtime_format_design.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_runtime_format_design.py
python3 scripts/check_runtime_format_design.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_runtime_format_design.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --runtime-format-design docs/architecture/runtime-format-design.md
```

- Candidate quality: not required — no model execution or NLL; this increment is runtime-format requirements documentation. Format **risks** are HYPOTHESIS prose/tables, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/runtime-format-design.md
python3 -m py_compile scripts/check_runtime_format_design.py
python3 scripts/check_runtime_format_design.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --runtime-format-design docs/architecture/runtime-format-design.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run `scripts/analyze_bf16_tensors.py`.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/runtime-format-design.md` (create)
  - `scripts/check_runtime_format_design.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-03/06/08 deliverables unchanged)
- Definition of done: runtime-format-design document published with locked object model, 8+8 capabilities, packing illustrations, 7 consumer sequences, 4 artifact approaches compared without a winner, and 8 open decisions left unselected; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-09 completion checkboxes can be marked at delivery; the portable-versus-specialized / ideal-sequence open question remains open.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T13:50:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-09.md`. Coupled IDs `none`. Document structure (10 headings), five canonical sentences, artifact object model (6 kinds), 8 representation + 8 packing capabilities, 7 consumer sequences, 4 artifact approaches × 6 comparison dimensions, 8 open decisions left unselected, 6 HYPOTHESIS format risks, stdlib checker `scripts/check_runtime_format_design.py`, JSON schema, and acceptance commands are closed. `docs/architecture/runtime-format-design.md` and the checker were **not** written in this stage. `plan.md` not edited. No commit.
- Performance evidence applied: N/A — format-requirements documentation; portable/specialized labels and format risks are hypotheses, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_runtime_format_design.py` (stdlib checker: `text_config` occupancy arithmetic matching TASK-01/06 family BF16 bytes, MLP/embed/metadata/header/dual-view illustrations with `ceil(n b/8)` packing, locked 8+8 capabilities / 6 artifact kinds / 7 sequences / 4 approaches / 8 open decisions / 6 HYPOTHESIS format risks; `--json` internal asserts; `--runtime-format-design` heading/JSON/mermaid/canonical/id/token/winner-phrase checks). Does not import other `check_*.py` or `analyze_bf16_tensors.py`. Does not stream safetensor payloads.
  - Created `docs/architecture/runtime-format-design.md` (ten locked headings, five canonical sentences, artifact object model, representation/packing capability tables, MLP \(g=32/128\) ratios 0.28125 / 0.2578125 plus unique-non-embed / embed-gather / \(S\) / header 55424 / dual-view 26863340064 illustrations, seven consumer sequences without an ideal sequence, four artifact approaches compared on six dimensions without a winner, eight open decisions left unselected, six HYPOTHESIS format-risk rows, one `flowchart TB` mermaid with required IDs, JSON fence copied from a live `--json` run).
  - Did not edit `docs/architecture/plan.md`, `dataflow.md`, `work-and-traffic.md`, `quantization-design-space.md`, `model-inventory.md`, `model-semantics.md`, `lifetime-and-state.md`, `numerical-sensitivity.md`, or `bf16-tensor-analysis.md`. Did not commit.
- Commands:
  - `python3 -m py_compile scripts/check_runtime_format_design.py` — pass (exit 0)
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); live object `n_representation_capabilities=8`, `n_packing_capabilities=8`, `n_artifact_approaches=4`, `n_consumer_sequences=7`, `n_open_decisions=8`, `n_open_decisions_selected=0`, `n_format_risks=6`, `mlp_n=17112760320`, `mlp_int4_g32_over_bf16=0.28125`, `mlp_int4_g128_over_bf16=0.2578125`, `unique_non_embed_int4_g128_total_bytes=13431670032`, `dual_view_int4_g128_unique_non_embed_bytes=26863340064`, `container_header_illustration_total_bytes=55424`, `artifact_boundary_selected=false`, `ideal_byte_sequence_selected=false`, `gguf_is_not_the_runtime_format=true`
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md` — first attempt fail (exit 1): intro used the word `UNKNOWN` outside Deferred vision; removed that mention and re-ran — pass (exit 0)
- UTC/time/tokens/cost: `2026-09-20T13:56:51Z`; `telemetry_unavailable`
- Commit: not created (implementation stage)

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/runtime-format-design.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-04 / [`lifetime-and-state.md`](../lifetime-and-state.md) / [`work-and-traffic.md`](../work-and-traffic.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`model-inventory.md`](../model-inventory.md) (TASK-01), [`dataflow.md`](../dataflow.md) (TASK-03), [`work-and-traffic.md`](../work-and-traffic.md) (TASK-06), [`quantization-design-space.md`](../quantization-design-space.md) (TASK-08), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_runtime_format_design.py`](../../../scripts/check_runtime_format_design.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the first JSON fence left unchanged. No locked integers, canonical sentences, capability ids, approach ids, open-decision ids, severities, or Mermaid node IDs edited. Frozen TASK-03/06/08 artifacts and `plan.md` not edited.
- Commands:
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md` — pass (exit 0; headings, JSON fence, one flowchart, canonical sentences, capability/approach/sequence/open/risk ids, diagram node IDs; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T14:00:00Z`; `telemetry_unavailable`

### Verification

- Agent/model: `composer-2.5` (verification subagent)
- UTC/time/tokens/cost: `2026-09-20T14:05:00Z`; `telemetry_unavailable`
- Verdict: **PASS** (first attempt)
- Focused commands (repository root):
  - `python3 -m py_compile scripts/check_runtime_format_design.py` — pass (exit 0)
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); internal asserts satisfied (`n_representation_capabilities=8`, `n_packing_capabilities=8`, `n_artifact_approaches=4`, `n_consumer_sequences=7`, `n_open_decisions=8`, `n_open_decisions_selected=0`, `n_format_risks=6`, `mlp_n=17112760320`, `mlp_int4_g32_over_bf16=0.28125`, `mlp_int4_g128_over_bf16=0.2578125`, `artifact_boundary_selected=false`, `ideal_byte_sequence_selected=false`, `gguf_is_not_the_runtime_format=true`)
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md` — pass (exit 0)
- Repository-wide commands:
  - `test -f docs/architecture/runtime-format-design.md` — pass
  - `python3 -m py_compile scripts/check_runtime_format_design.py` — pass (exit 0)
  - `python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md` — pass (exit 0)
- Independent review:
  - Recomputed `mlp_n`, MLP int4 g32/g128 packed-tight totals, unique-non-embed int4 g128 total, header product (`866×64`), and dual-view product from sitting `text_config` (`num_hidden_layers=64`, `hidden_size=5120`, `intermediate_size=17408`) — matches live `--json` and TASK-08 illustrations (`9625927680` / `0.28125`, `8823767040` / `0.2578125`, `13431670032`, `55424`, `26863340064`).
  - Spot-checked cited TASK-06 integers in `work-and-traffic.md` (`54641395712`, `52098598912`, `10240`, `150994944`) and TASK-08 integers in `quantization-design-space.md` (`17112760320`, packed illustrations, ratios) against live `--json` — all match; not JSON-echo only.
  - First fenced `json` block in `runtime-format-design.md` equals live `--json` output (byte-identical object).
  - Ten required `##` headings in locked order; five canonical sentences verbatim; one `flowchart TB` mermaid with required node IDs (`representation`, `packing`, `portable`, `specialized`, `gather`, `gemm`, `state`, `open`).
  - Eight representation + eight packing capabilities, seven consumer sequences, four artifact approaches × six comparison dimensions, eight open decisions all unselected (`n_open_decisions_selected=0`), six format-risk rows labelled HYPOTHESIS.
  - No forbidden winner phrases; `UNKNOWN` only under Deferred vision; no `TBD`/`TODO`/`???`.
  - `scripts/check_runtime_format_design.py` uses stdlib only (`argparse`, `json`, `math`, `re`, `sys`, `pathlib`); does not import other `check_*.py` or `analyze_bf16_tensors.py`; does not stream safetensor payloads.
  - `docs/architecture/plan.md` and frozen TASK-03/06/08 deliverables unchanged. Ledger open question (portable versus backend-specialized boundaries and ideal consumer byte sequences) remains open.
- Performance evidence checklist: N/A (format-requirements documentation; no GPU/timing claims)
- Candidate quality gate: not required (no model execution or NLL)
- Formatting: no `uv run ruff format` gate for this increment (dossier forbids Ruff/pytest/CMake/CUDA)

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-09 only; coupled IDs `none`
- Outcome: TASK-09 marked `DONE` after verification pass (attempt 1, verdict PASS)
- UTC/time/tokens/cost: `2026-09-20T14:10:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1) — `docs/architecture/runtime-format-design.md` (10 locked headings, five canonical sentences, six artifact object kinds, 8+8 capabilities, seven consumer sequences without an ideal sequence, four artifact approaches compared without winner, eight open decisions unselected, six HYPOTHESIS format risks, one `flowchart TB` mermaid, JSON fence); `scripts/check_runtime_format_design.py` stdlib checker; independent `text_config` recomputation and TASK-06/08 citation spot-checks pass; frozen upstream docs unchanged; portable-versus-specialized open question remains open
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (runtime-format requirements locks; no tok/s evidence)
- Throughput delta: N/A — TASK-09 does not execute or time the model
- Commit: Publish Qwen3.8 runtime format design
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification attempt 1 PASS; no repair loop)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/authorities/qwen3.8-27b-transformers/config.json` must remain present for focused commands
