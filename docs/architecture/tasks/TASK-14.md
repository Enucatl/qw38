# TASK-14 — Derive a clean-sheet prefill execution plan

## Control

- Primary ID: `TASK-14`
- Coupled IDs: `none`
- Dependencies: `TASK-06`, `TASK-09`, `TASK-11`, `TASK-12` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Explain fundamental matrix-matrix, reuse, tiling, state, and temporary-storage differences; Compare every semantic node with decode; Keep representation tradeoffs unresolved without evidence.

## Goal and boundaries

Produce `docs/architecture/prefill-plan.md` as the Phase 1 **hardware-independent prefill semantic schedule** for many-token inference on the Qwen3.8-27B language+MTP map, **independent of decode**. Select **one serial order** of the existing TASK-11 node instances (135 complete) as **layer-serial over a length-$T$ prompt**. Per compact stage kind: name weight loads, state reads/writes, visibility boundaries, and reuse. Explain the five fundamental differences versus decode (matrix-matrix, reuse, tiling, state, temporary storage). Compare **every** TASK-11 semantic node type (6) and every stage kind (9) with decode. Separate TASK-06 **unavoidable** unique-weight / triangular-KV / T-scaled activation minima from this schedule’s **proposed-boundary** stage-cut channel. Attach TASK-09 consumer sequences and TASK-12 fusion hypotheses as **unselected** experiment attachments. **Keep** the ledger open question (decode/prefill representation tradeoffs and potential need for multiple views) **unresolved**: enumerate view hypotheses, select none, leave `decode_prefill_distinct_views_selected` false.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Work/traffic minima, prefill $W=TC+AT(T+1)/2$, triangular KV, and T-scaled region-cut come from `docs/architecture/work-and-traffic.md` (TASK-06). Artifact capabilities, one shared artifact, `view_binding`, and the seven consumer sequences come from `docs/architecture/runtime-format-design.md` (TASK-09); do not close portable-versus-specialized, ideal-sequence, or distinct decode/prefill views. Node types, instance counts, 14 sync edges, and mixer xor come from `docs/architecture/semantic-graph.md` (TASK-11). Physical-need classes and the 22 fusion hypotheses come from `docs/architecture/materialization-and-fusion.md` (TASK-12). Do not add node types, catalog IDs, or sync edges as **required** contracts.
  - TASK-13 `docs/architecture/decode-plan.md` is **DONE** and **not** a listed dependency. Cite it only as comparison evidence for the required node-by-node decode contrast. Independently re-derive the prefill serial order from the TASK-11 DAG (it matches the decode kind order because the DAG is shared). Do not treat decode-plan as the prefill schedule.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established), `DERIVED` (serial order from the DAG, stage multiplicities, T-scaled stage-cut byte identities, cited TASK-06/04 integers, MAC identity at $T=1$), or `HYPOTHESIS` (hoist/overlap/mode/view usefulness, second $W_\text{lm}$ read, extra `h_64` consumer read, every packing/fusion **usefulness**, every “prefill needs a distinct view”). `UNKNOWN` only for vision-encoder internals deferred here. No `MEASURED` tok/s or NLL. No selected fusion, packing winner, tile, view split, or kernel.
  - GitHub Markdown math. Cite TASK-02 equation tags via TASK-11 contracts, TASK-06 MAC/byte integers, TASK-09 sequence ids, TASK-11 node types / sync-edge ids, TASK-12 hypothesis ids. Do not rewrite forward math, recopy TASK-06 MAC tables as a new work study, recopy TASK-09 packing illustrations as a new format study, or recopy TASK-12’s 52-ID physical-need table as a new materialization study.
  - Allowed evidence: TASK-06 work/traffic, TASK-09 runtime-format, TASK-11 semantic-graph, TASK-12 materialization/fusion, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier, and TASK-13 decode-plan **as comparison only**. TASK-03/04 integers already cited by those documents may be **cited** through them. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No TASK-16 occupancy/fusion-vs-occupancy algebra (not a dependency; this document is hardware-independent). No TASK-15 tiles or TASK-17 CUDA mappings.
  - Hardware-independent: stage kinds, catalog IDs, node types, sync-edge ids, byte identities, sequence ranks. No thread geometry, warps, SMs, CUDA dtypes, kernel names, streams, shared-memory tiles, or sitting-GPU numbers.
- Non-goals:
  - No fusion, split, reuse, packing, tile, or view **winner**. Listing a hypothesis or attaching a TASK-09 sequence is not selecting it. `n_fusion_hypotheses_selected` = 0. `ideal_byte_sequence_selected` false. `decode_prefill_distinct_views_selected` false. `n_representation_hypotheses_selected` = 0.
  - No decode schedule (TASK-13). Prefill and decode share **one** graph; this document schedules length-$T$ prefill with incoming $(K,V,C,S)$ **zeros**. Compare every node with decode; do not copy the decode document as the prefill schedule.
  - No physical layouts (TASK-15) and no CUDA mapping alternatives (TASK-17). Semantic tiling axes are named; tile extents are not selected.
  - No quantization recipes, artifact-boundary winner, or compiler stages (TASK-08/09/10 already published; do not close their open decisions).
  - No quality/NLL experiments (TASK-18) and no tok/s (TASK-19).
  - No new operators, extra catalog IDs, extra required node types, or 64-layer / 135-instance / $T$-position unrolling in diagrams.
  - No sampling softmax over $V$ (out of scope, TASK-11).
  - No generic GEMM / softmax / RMSNorm kernel-fusion cookbook. Intra-node `fuse_internals` remain TASK-12 hypotheses.
  - No peak-memory claim and no summed CUDA live-set.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–13). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `work-and-traffic.md`, `runtime-format-design.md`, `semantic-graph.md`, `materialization-and-fusion.md`, `decode-plan.md`, `dataflow.md`, `lifetime-and-state.md`, `model-semantics.md`, `model-inventory.md`, `cuda-hardware-model.md`, or any TASK-08/10 deliverable.
  - Do not import other `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib prefill-plan checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places independent decode/prefill execution plans after the Qwen-specific semantic graph / fusion hypotheses and before CUDA mappings.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:89-92` — TASK-11 derives semantic regions; TASK-12 fusion; TASK-13 and TASK-14 **independently** schedule decode and prefill; TASK-17 CUDA-maps nodes.
- `docs/architecture/task_ledger.md` TASK-14 row — produces `docs/architecture/prefill-plan.md`; purpose is an independent many-token semantic schedule; open question is decode/prefill representation tradeoffs and potential need for multiple views (**kept unresolved here**); completion is the five fundamental differences, every-node decode comparison, and unresolved representation tradeoffs.
- `docs/architecture/task_ledger.md` TASK-06 established results — prefill $W=TC+AT(T+1)/2$; at $T=1$ MAC prefill = decode; $C_\text{complete}=27433238528$, $A_\text{complete}=208896$; unique non-embed `52098598912`; prefill gather complete `2T·10240`; triangular KV read $69632·T(T-1)/2$, KV write $69632T$; region-cut prefill = $T×$ decode; six HYPOTHESIS bottleneck labels including `quadratic_attn` and `compute`.
- `docs/architecture/task_ledger.md` TASK-09 established results — seven consumer sequences without an ideal sequence; `ideal_byte_sequence_selected` false; one compiled artifact for prefill and decode (`decode_prefill_share_artifact` true); `decode_prefill_distinct_views_selected` false; `view_binding` capability; view split left for this task **without requiring a close**.
- `docs/architecture/task_ledger.md` TASK-11 established results — six node types; 135 complete instances; mixer xor by `layer_types`; 14 sync edges; `g`/`z` internal; residual add inside mixer/`mlp`; `schedule_selected` false **there** (this task selects the prefill serial order).
- `docs/architecture/task_ledger.md` TASK-12 established results — seven physical-need classes; 22 unselected fusion hypotheses; state/output edges nonfusible; `fanout_h64_cannot_hide_from_one_consumer` true.
- `docs/architecture/task_ledger.md` TASK-13 established results (comparison only) — nine decode stage kinds, serial one-token order, stage-cut 2355200 B; incoming state populated; unique-weight/state extra 0.
- `docs/architecture/task_ledger.md` TASK-15/17 — consumers of this schedule; do not perform those designs here.
- `docs/architecture/work-and-traffic.md` — decode vs prefill $T$ convention; unique-weight+gather; triangular KV; C/S ×$T$ with initial-zero reads 0 physical; three activation views; surviving store $B_\text{store}(T)$.
- `docs/architecture/runtime-format-design.md` — seven `consumer_sequence_ids`; `view_binding`; one artifact; distinct views unselected.
- `docs/architecture/semantic-graph.md` — six node types; catalog partition; 14 sync edges; per-node I/O/state; mixer xor; live-across internal; shared graph.
- `docs/architecture/materialization-and-fusion.md` — 22 fusion hypothesis ids; forbidden drops; default tactics; working-set byte identities (decode ranks; this schedule scales sequence ranks by $T$).
- `docs/architecture/decode-plan.md` — comparison target for every node/stage; not a dependency and not copied as the prefill schedule.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for layer counts, shapes, and cited MAC/byte products. Do not read safetensor payloads.
- `scripts/check_work_and_traffic.py` / `scripts/check_runtime_format_design.py` / `scripts/check_semantic_graph.py` / `scripts/check_materialization_and_fusion.py` / `scripts/check_decode_plan.py` — checker-style precedent. TASK-14’s checker is a sibling; do not import them.

## Performance evidence

N/A — prefill **semantic-schedule** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Byte and MAC figures are TASK-06 citations plus DERIVED T-scaled stage-cut arithmetic, not MEASURED traffic. Packing/fusion/hoist/view usefulness is HYPOTHESIS, not MEASURED. Do not apply the performance-evidence checklist to rank kernels or claim a winning schedule, view, or overlap.

- Measurement identity: N/A (no engine binary). TASK-06/09/11/12 integers are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Schedule coverage is 9 stage kinds, 135 instances, 14 sync edges, 6 node types compared with decode, 5 fundamental differences, TASK-06 three traffic channels plus this T-scaled stage-cut activation channel, 7 consumer sequences attached unselected, 22 fusion hypotheses unselected, 4 representation hypotheses unselected.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` layer count or cited TASK-06 integer disagrees with TASK-06/11, stop and fail closed (do not invent stages). MAC equality at $T=1$ does **not** imply C/S-read equality (prefill zeros vs continuing-decode populated $C,S$).
- Claim types: serial order and T-scaled stage-cut bytes `derived`; unique-weight extra `derived` (zero); triangular KV extra `derived` (zero); every hoist/overlap/mode/view/fusion/packing **usefulness** `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Schedule completeness is 9 kinds summing to 135 instances, five differences explained, 6/6 nodes compared, representation question left open.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for prefill-schedule claims

If a node I/O, state kind, catalog ID, layer count, or cited MAC/byte would disagree with TASK-06/09/11/12 or sitting `text_config`, the earlier document / config wins and this one is wrong. If a **comparison** cell would disagree with TASK-13 decode-plan stage kinds or TASK-06 decode identities, the earlier document wins and the comparison cell is wrong.

- Prefill and decode share **one** semantic graph (TASK-11) and **one** compiled artifact (TASK-09). This document independently schedules prefill: prompt length $T$, $T_\text{new}=T$, stored KV length $T$ after the prompt, incoming $(K,V,C,S)$ **zeros** (populated incoming is decode, TASK-13).
- Primary schedule **includes MTP** (second embed over teacher-forced next-ids $2,\ldots,T+1$, `mtp_mix`, one `gated_attn`, one `mlp`, second `lm_head`) at **all** $T$ positions. Language-only instance counts are secondary. Omitting MTP when only $\ell^{(0)}$ is required is a TASK-02 algebraic equivalent, not a second schedule and not the primary contract. Omitting $T-1$ vocabulary projections is the TASK-06 `inference_last_logits` **secondary** row, not the primary schedule.
- Both current and next `token_id` **sequences** of length $T$ are **inputs** to the complete map (teacher-forced). Sampling over $V$ is out of scope. Do not invent a catalog ID for next tokens; two presentation **streams** (`n_token_presentation_streams` 2).
- Algebraic equivalents (chunkwise GDN, SDPA, GQA-as-repeat, omitting MTP, last-logits, token-serial replay of decode) stay **inside** the owning node or as unselected mode hypotheses. They are not extra stages. GDN primary stays `(17)`–`(18)`; do not treat chunkwise as zero $S$ traffic.
- TASK-11 **node types** are the scheduled units. Do not split `gated_attn` / `gated_delta_net` into proj/core/out as required stages. `g`/`z` stay internal unless a TASK-12 split hypothesis is tried.
- Fan-out ≠ must-store and node I/O ≠ must-store still hold. A stage-cut byte identity is not a CUDA store.
- Unique weight bytes are counted **once** per complete prefill (`weight_unique_counted_once` true). Per-stage “loads” name **which** unique bytes that stage consumes, reused across the $T$ sequence, not a $T×$ restream of the model.
- Do not inspect Quartz or llama.cpp to “confirm” schedules or kernels.
- Do not select a fusion, packing sequence, tile, CUDA mapping, or decode/prefill view split. Do not import TASK-16 occupancy algebra.
- Select **one** serial stage-kind order: **layer-serial over the $T$-sequence**. Token-serial replay of the decode schedule is an algebraic equivalent of the TASK-06 sum $W=\sum_t(C+At)$, not the primary schedule (it would hide the matrix-matrix difference). Hoist, overlap, last-logits, and view splits are HYPOTHESIS flexibilities, not a second selected schedule.

### Deliverable structure (`docs/architecture/prefill-plan.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Hoist/packing/fusion/mode/view **usefulness** cells are `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, work/traffic, runtime-format, semantic-graph, materialization-and-fusion, decode-plan (comparison only), inventory via those docs, config, checker; evidence labels; in-scope (language+MTP many-token layer-serial schedule + five differences + 6/6 node comparison + traffic split + unselected packing/fusion/view attachments) vs deferred (vision encoder; TASK-15 layouts; TASK-17 CUDA). State that the document specifies a **hardware-independent semantic schedule**, not kernels.
2. **Prefill schedule convention** — the four canonical sentences (exact text below); what a stage is; layer-serial vs token-serial; decode identity independent.
3. **Many-token setting** — $T$, incoming zeros, MTP at all $T$, two token streams, complete vs language-only, $T=1$ MAC vs state-read.
4. **Fundamental differences versus decode** — the five ledger differences (matrix-matrix, reuse, tiling, state, temporary storage). This heading **starts** the first completion criterion.
5. **Stage kinds and serial order** — the locked 9 kinds, multiplicities summing to 135, mixer xor, layer-serial order, ready-set constraints.
6. **Per-stage loads, state, visibility, and reuse** — one table covering all 9 kinds. Completes the per-stage schedule.
7. **Semantic-node comparison with decode** — one table covering all 6 TASK-11 node types and a companion 9-kind stage table. This heading **completes** the second ledger criterion.
8. **Unavoidable versus proposed-boundary traffic** — TASK-06 minima vs T-scaled stage-cut identities; weight extra 0; state extra 0.
9. **Packing, fusion, and representation tradeoffs** — seven sequences attached unselected; 22 fusion hypotheses unselected; five hoist + two mode + four representation hypotheses **all unselected**; one Mermaid summary (diagram 1 of 1). This heading **keeps** the ledger open question unresolved.
10. **Work citations and non-decisions** — cited $C,A,W_\text{prefill}(T)$; bottleneck labels remain HYPOTHESIS citations; what TASK-15/17 own; representation question remains open.
11. **Deferred vision** — residual-stream interface only; sequence length of visual tokens UNKNOWN.
12. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Prefill schedule convention**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> The prefill schedule in this document is a hardware-independent order of TASK-11 nodes for many-token inference from zero state, not a CUDA graph, kernel launch sequence, or selected fusion.

> Unavoidable traffic is the TASK-06 unique-weight, triangular-KV/state, and T-scaled activation minimum; proposed-boundary traffic is this schedule's named stage-cut channel.

> Decode/prefill representation tradeoffs and the potential need for multiple views remain unresolved without evidence; view_binding stays a capability, not a selected split.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_schedule` = sentence 2; `canonical_sentence_traffic` = sentence 3; `canonical_sentence_open_question` = sentence 4.

Sentence 4 **keeps** the ledger open question open in checker-substring form. JSON `ledger_open_question_representation_tradeoffs_closed` **false**. JSON `decode_prefill_distinct_views_selected` **false**.

Bullets required under that heading:

- Nine stage kinds are complete for this task; their multiplicities sum to 135 complete node instances (`n_stage_kinds` 9, `n_stage_instances` 135).
- A stage is a named execution of one TASK-11 node type (or mixer xor) over the length-$T$ sequence, with loads, state R/W, visibility I/O, and reuse.
- This document selects one **serial** **layer-serial** order (`schedule_serial_selected` true, `schedule_layer_serial_selected` true). Token-serial decode replay remains a mode HYPOTHESIS (`n_mode_hypotheses_selected` 0). Hoist and overlap remain HYPOTHESIS (`n_hoist_hypotheses_selected` 0).
- TASK-13 remains the decode authority; decode is comparison evidence only. Prefill $T_\text{new}=T$; incoming state is zeros (`incoming_state_zeros` true, `incoming_state_populated` false).
- Residual add stays inside mixer/`mlp`; `g`/`z` stay internal; RMS stays inside the consumer (TASK-11 locks).
- Unique weights are counted once per complete prefill (`weight_unique_counted_once` true).
- Fan-out ≠ must-store and node I/O ≠ must-store still hold.
- Primary schedule includes MTP at all $T$ positions. Hardware mapping is TASK-17. Fusion winners remain TASK-12 hypotheses. Ideal byte sequence remains TASK-09 open. Distinct decode/prefill views remain unselected.
- No thread geometry (`thread_geometry_absent` true).
- Every TASK-11 node type is compared with decode (`every_semantic_node_compared` true). Representation tradeoffs stay unresolved (`representation_tradeoffs_unresolved` true).

### Many-token setting (lock)

Let $T$ be both the **prompt length** and the stored KV length **after** prefill (TASK-04/06). Prefill of length $T$ is the same map at $t=1,\ldots,T$ from **zero** state. Causal full attention at step $t$ contracts against length $t$. Prefill causal attention uses exact $T(T+1)/2$, not $T^2/2$. Incoming KV length is 0; after prefill, stored KV length is $T$.

JSON: `T_is_stored_length_after_prefill` true; `prefill_T_new_equals_T` true; `example_T` `[1, 4096]`; `incoming_state_zeros` true; `incoming_state_populated` false; `primary_includes_mtp` true; `mtp_omission_is_algebraic_equivalent` true (not the primary schedule); `mtp_runs_all_T_positions` true.

JSON `n_token_presentation_streams` = 2. Current `token_id` sequence of length $T$ feeds `embed_current`; next `token_id` sequence of length $T$ (teacher-forced ids $2,\ldots,T+1$) feeds `embed_next`. Both are graph inputs to the complete map. Do not add a catalog ID.

MAC identity (TASK-06, DERIVED): at $T=1$, $W_\text{prefill}=C+A=W_\text{decode}$. JSON `mac_prefill_equals_decode_at_T1` true. This is **not** a state-read identity: prefill $T=1$ physical $C,S$ read is 0 (zeros); a continuing decode at stored length $T=1$ still reads populated $C,S$ (`decode_read_fixed_bytes` 153944064). JSON `state_read_prefill_equals_decode_at_T1` false. Incoming KV is empty at $T=1$ for both (`kv_incoming_empty_at_T1_both` true).

JSON `n_node_instances_complete` = 135; `n_node_instances_language` = 130 (citations of TASK-11). Mixer xor: language layer $\ell$ uses `gated_attn` iff $\ell\in\mathcal{L}_\text{full}$ (`full_attention_indices` $\{3,7,\ldots,63\}$), else `gated_delta_net`. Then `mlp`. Do not unroll 64 layers or $T$ positions beyond this rule. JSON `mixer_xor_by_layer_types` true.

JSON booleans (lock true unless noted):

- `decode_prefill_share_graph` = true (graph shared; **schedules** are independent)
- `decode_prefill_share_artifact` = true (TASK-09 citation)
- `decode_prefill_distinct_views_selected` = false
- `residual_add_inside_mixer` = true
- `residual_add_inside_mlp` = true
- `residual_input_live_until_add` = true
- `rms_inside_consumer` = true
- `live_across_are_internal` = true
- `hardware_independent` = true
- `cuda_mapping_deferred` = true
- `thread_geometry_absent` = true
- `layout_selected` = false
- `tile_size_selected` = false
- `fusion_winner_selected` = false
- `ideal_byte_sequence_selected` = false
- `artifact_boundary_selected` = false
- `schedule_serial_selected` = true
- `schedule_layer_serial_selected` = true
- `gdn_primary_is_recurrent_eq_17` = true
- `chunkwise_not_zero_s_traffic` = true
- `state_write_not_optional` = true
- `fanout_h64_cannot_hide_from_one_consumer` = true
- `weight_unique_counted_once` = true
- `weight_second_w_lm_read_is_hypothesis` = true
- `activation_dtype_decided` = false
- `vision_interface_is_not_a_node` = true
- `mtp_omission_is_algebraic_equivalent` = true
- `every_semantic_node_compared` = true
- `representation_tradeoffs_unresolved` = true
- `ledger_open_question_representation_tradeoffs_closed` = false
- `mac_prefill_equals_decode_at_T1` = true
- `state_read_prefill_equals_decode_at_T1` = false
- `kv_incoming_empty_at_T1_both` = true
- `incoming_state_zeros` = true
- `incoming_state_populated` = false

### Fundamental differences versus decode (lock; heading 4; first completion criterion)

JSON array `fundamental_difference_ids` in this exact order (5 ids). JSON `n_fundamental_differences` = 5. Completes the ledger’s first checkbox. Do not rank these by wall time. Usefulness of acting on them (distinct views, chunkwise GDN, last-logits, CUDA tiles) remains HYPOTHESIS.

| id | Decode ($T_\text{new}=1$, populated) | Prefill (length $T$, zeros) |
| --- | --- | --- |
| `diff_matrix_matrix` | Projections and MLP are GEMV ($x\in\mathbb{R}^{H}$). Full-attn core is one query against stored length $T$. | Projections and MLP are GEMM with sequence axis $T$ ($X\in\mathbb{R}^{H\times T}$). Full-attn core is causal matrix-matrix $Q_{\le t}K_{\le t}^\top$ with exact $T(T+1)/2$ contractions. GDN stays a length-$T$ recurrence (linear in $T$), not quadratic. |
| `diff_reuse` | Unique weights counted once per token. Embed gathers 1 (language) or 2 (complete) rows. | Unique weights counted **once per prompt** and reused across $T$ tokens (`weight_unique_counted_once`). Embed gathers $T$ or $2T$ rows. TASK-06 `compute` label: large-$T$ GEMMs may be compute-bound if $T>I_\text{ridge}$ (HYPOTHESIS vs UNKNOWN ridge). |
| `diff_tiling` | Attention’s $T$ is **stored length** against one query. GEMV has no sequence tile. Physical tiles are TASK-15. | Attention’s $T$ is a **causal sequence axis**. GEMM’s $T$ is a batch/sequence axis. Physical tile extents remain unselected (`tile_size_selected` false). Naming the axis is not selecting a layout. |
| `diff_state` | Incoming $(K,V,C,S)$ populated. KV read $69632(T-1)$. C/S read `decode_read_fixed_bytes`. Write one token of KV plus C/S. | Incoming zeros. KV read triangular $69632·T(T-1)/2$; KV write $69632T$. C/S mathematical volumes scale as TASK-04 $\times T$; initial-zero reads 0 physical. Surviving store after prefill is $B_\text{store}(T)$, not $T\times B_S$. Writes not optional. |
| `diff_temporary_storage` | Named residual stage-cut 10240 B per edge; `g`/`z` local 798720 B inside mixers; logits 496640 B × 2. Softmax scores are not catalog IDs. | Same **named edges**; ranks scale by $T$. Residual stage-cut $133\times T\times 10240$; `g`/`z` local $T\times 798720$ still inside mixers; primary logits $2\times T\times 496640$. Softmax score matrices still are not catalog IDs (traffic stays the KV channel). No CUDA live-set / peak-memory claim. |

JSON `fundamental_difference_usefulness_label` exactly `HYPOTHESIS` (acting on a difference with a distinct view, tile, or kernel is unselected).

Required prose: at $T=1$, matrix-matrix collapses to GEMV and triangular KV read is 0, matching decode **work and KV**, but C/S physical read stays 0 on prefill vs 153944064 on a continuing decode.

### Stage kinds and serial order (lock)

JSON array `stage_kind_ids` in this exact order (9 ids). JSON `n_stage_kinds` = 9. Parallel `stage_multiplicities`, `stage_node_types`, `stage_primary_sequence_ids`. **Same ids as TASK-13** so the required comparison is 1:1; independently selected because they are the TASK-11 instance partition, not because decode-plan is a dependency.

| id | Multiplicity | Node type | Serial rank |
| --- | ---: | --- | ---: |
| `embed_current` | 1 | `embed` | 1 |
| `language_mixer` | 64 | `mixer_xor` (`gated_attn` iff $\ell\in\mathcal{L}_\text{full}$, else `gated_delta_net`) | 2 (loop with next) |
| `language_mlp` | 64 | `mlp` | 2 (after matching mixer) |
| `lm_head_primary` | 1 | `lm_head` | 3 |
| `embed_next` | 1 | `embed` | 4 |
| `mtp_mix` | 1 | `mtp_mix` | 5 |
| `mtp_mixer` | 1 | `gated_attn` | 6 |
| `mtp_mlp` | 1 | `mlp` | 7 |
| `lm_head_mtp` | 1 | `lm_head` | 8 |

JSON `stage_multiplicities` `[1,64,64,1,1,1,1,1,1]`. Sum 135. JSON `n_stage_instances` = 135. JSON `stage_node_types` `["embed","mixer_xor","mlp","lm_head","embed","mtp_mix","gated_attn","mlp","lm_head"]`. `mixer_xor` is a schedule abbreviation, not a seventh TASK-11 node type.

Instance-count cross-check (required prose + checker): `language_mixer` contributes 16 `gated_attn` + 48 `gated_delta_net`; plus `mtp_mixer` → 17 `gated_attn`. `language_mlp` + `mtp_mlp` → 65 `mlp`. Two `embed`, two `lm_head`, one `mtp_mix`. Matches TASK-11. Each instance runs over the length-$T$ sequence (not $T$ extra instances).

**Serial order** (selected; one complete prefill, **layer-serial** over $T$):

1. `embed_current` (gather all $T$ prompt rows)
2. For $\ell=0,\ldots,63$: `language_mixer`[$\ell$] then `language_mlp`[$\ell$] over the $T$-sequence
3. `lm_head_primary` over all $T$ positions (primary complete map)
4. `embed_next` (gather all $T$ teacher-forced next-ids)
5. `mtp_mix` over the $T$-sequence
6. `mtp_mixer`
7. `mtp_mlp`
8. `lm_head_mtp` over all $T$ positions

JSON `serial_stage_kind_order` equals `stage_kind_ids` (the loop is implied by multiplicities; do not expand to 135 ids or $T$ positions in JSON).

**Ready-set constraints** (DERIVED from TASK-11 edges; not CUDA overlap):

JSON array `ready_constraint_ids` in this exact order (10 ids). JSON `n_ready_constraints` = 10. Same ids as decode; “ready” means the **sequence** producer has completed, not one token.

| id | Stage | Ready after |
| --- | --- | --- |
| `ready_embed_current` | `embed_current` | current `token_id` **sequence** of length $T$ present |
| `ready_language_mixer_0` | `language_mixer`[0] | `embed_current` (`identity_e_h0`) |
| `ready_language_mlp` | `language_mlp`[$\ell$] | matching `language_mixer`[$\ell$] (`residual_h_mid`) |
| `ready_language_mixer_next` | `language_mixer`[$\ell>0$] | `language_mlp`[$\ell-1$] (`residual_h`) |
| `ready_lm_head_primary` | `lm_head_primary` | `language_mlp`[63] (`fanout_h64`) |
| `ready_embed_next` | `embed_next` | next `token_id` **sequence** present (no language-stack data edge) |
| `ready_mtp_mix` | `mtp_mix` | `language_mlp`[63] **and** `embed_next` |
| `ready_mtp_mixer` | `mtp_mixer` | `mtp_mix` (`mtp_u_to_block`) |
| `ready_mtp_mlp` | `mtp_mlp` | `mtp_mixer` (`residual_h_mid`) |
| `ready_lm_head_mtp` | `lm_head_mtp` | `mtp_mlp` (`h_mtp_to_logits`) |

Serial places `embed_next` after `lm_head_primary` to keep the MTP suffix contiguous. `ready_embed_next` is earlier; hoisting is HYPOTHESIS, not a second selected order. Token-serial (`for t in 1..T: entire stack`) is `token_serial_prefill`, unselected.

### Per-stage loads, state, visibility, and reuse (lock; heading 6)

Table columns, in this order: `Stage` | `Loads` | `State read` | `State write` | `Visibility in` | `Visibility out` | `Reuse`

Implementation copies these rows; do not add/remove stage kinds. Unique weight bytes counted once for the prompt. Visibility **names** match decode; **ranks** are length $T$.

| Stage | Loads | State read | State write | Visibility in | Visibility out | Reuse |
| --- | --- | --- | --- | --- | --- | --- |
| `embed_current` | $T$ rows of shared $E$ ($T\times 10240$ B gather; table not streamed) | none | none | current `token_id` sequence | `e` identified as $h^{(0)}$ sequence | `shared_E`; gather vs full table |
| `language_mixer` | Mixer unique weights for that layer (self-attn **or** linear-attn family; counted once). Access `dense_gemm` (now GEMM over $T$); GDN also `depthwise_conv` over $T$ | `gated_attn`: triangular prefix $K,V$ (0 at $t=1$). `gated_delta_net`: $C$ and $S$ from zeros (0 physical on first token) | `gated_attn`: write $K,V$ for all $T$ tokens (4096 B × $T$ per instance). `gated_delta_net`: write $C$ and $S_t$ each step | `h` sequence (live until Mix add) | `h_mid` sequence | Residual `h` until add; internals default inside (`g` live-across internal, rank $T$; `k_rope`/`v_full`/`qkv` reuse_or_recompute HYPOTHESIS) |
| `language_mlp` | MLP unique weights for that layer (counted once). Access `dense_gemm` as GEMM over $T$ | none | none | `h_mid` sequence | next `h` sequence, or `h_64` sequence at $\ell=63$ | Residual `h_mid` until add; `h_post`/`swiglu` fuse_or_recompute HYPOTHESIS |
| `lm_head_primary` | Shared $W_\text{lm}$ 2542796800 B (`seq_lm_head_full`) plus final RMS gamma, reused across $T$ | none | none | `h_64` sequence | `logits_0` sequence (primary: all $T$ positions) | `shared_W_lm`; `h_64` also consumed by `mtp_mix`; last-logits omission is unselected mode |
| `embed_next` | $T$ rows of shared $E$ ($T\times 10240$ B gather) | none | none | next `token_id` sequence | `e_next` sequence | `shared_E` (same payload as `embed_current`) |
| `mtp_mix` | `mtp.fc` contraction weights. Access `dense_gemm` over $T$ | none | none | `h_64`, `e_next` sequences | `mtp_u` as residual `h` sequence | `h_64` fan-out reuse; `mtp_cat` split HYPOTHESIS |
| `mtp_mixer` | MTP `gated_attn` unique weights. Access `dense_gemm` over $T$ | MTP $K,V$ triangular from zeros | MTP $K,V$ write $T$ tokens | `mtp_u` as `h` | `h_mid` sequence | Same gated-attn reuse as language full layers |
| `mtp_mlp` | MTP MLP unique weights. Access `dense_gemm` over $T$ | none | none | `h_mid` | `h_mtp` sequence | Same MLP reuse as language |
| `lm_head_mtp` | Shared $W_\text{lm}$ (second physical read HYPOTHESIS, not unique bytes) | none | none | `h_mtp` sequence | `logits_1` sequence | `shared_W_lm` |

JSON `stage_visibility_in` / `stage_visibility_out` keyed in `stage_kind_ids` order — **same names as decode**:

- `embed_current`: in `["token_id"]`, out `["e"]`
- `language_mixer`: in `["h"]`, out `["h_mid"]`
- `language_mlp`: in `["h_mid"]`, out `["h"]`
- `lm_head_primary`: in `["h_64"]`, out `["logits_0"]`
- `embed_next`: in `["token_id"]`, out `["e_next"]`
- `mtp_mix`: in `["h_64","e_next"]`, out `["mtp_u"]`
- `mtp_mixer`: in `["h"]`, out `["h_mid"]`
- `mtp_mlp`: in `["h_mid"]`, out `["h_mtp"]`
- `lm_head_mtp`: in `["h_mtp"]`, out `["logits_1"]`

JSON `activation_sequence_length_is_T` true. JSON `stage_state_read` / `stage_state_write`: same xor union as decode (`language_mixer` `["K_state","V_state","C_state","S"]`; `mtp_mixer` `["K_state","V_state"]`; others empty). JSON `language_mixer_state_is_xor` true. Prose: physical first-token C/S/KV reads are 0.

JSON `stage_primary_sequence_ids` in this exact order (9 ids) — **same attachments as decode**, not a prefill-specific packing winner:

`seq_gather_row`, `seq_gemm_codes_then_scales`, `seq_gemm_codes_then_scales`, `seq_lm_head_full`, `seq_gather_row`, `seq_gemm_codes_then_scales`, `seq_gemm_codes_then_scales`, `seq_gemm_codes_then_scales`, `seq_lm_head_full`

JSON `gated_delta_net_state_sequence_id` = `seq_state_s_dense`. Attaching a sequence is **not** selecting an ideal byte sequence and is **not** selecting a distinct prefill view.

State volume citations (do not re-derive KV identities; checker recomputes from the same identities as TASK-06):

| JSON key | Value |
| ---: | ---: |
| `kv_bytes_per_full_layer_per_token` | 4096 |
| `kv_bytes_all_per_token` | 69632 |
| `c_bytes_per_layer` | 61440 |
| `c_write_bytes_all` | 983040 |
| `c_read_bytes_all` | 2949120 |
| `s_bytes_per_layer` | 3145728 |
| `s_bytes_all` | 150994944 |
| `decode_write_bytes` | 152047616 |
| `decode_read_fixed_bytes` | 153944064 |
| `prefill_kv_read_coeff` | 69632 (times $T(T-1)/2$) |
| `prefill_kv_write_coeff` | 69632 (times $T$) |

Identities: `prefill_kv_read = kv_bytes_all_per_token * T * (T-1) / 2`; `prefill_kv_write = kv_bytes_all_per_token * T`; C/S write = decode per-token write $\times T$; C/S math read = decode per-token read $\times T$; C/S physical read = decode per-token read $\times (T-1)$ (initial zeros); `prefill_write = kv_write + c_write + s_write`; `prefill_read_physical = kv_read + c_read_physical + s_read_physical`; `storage = 69632T + 153944064`.

Prose required: omitting a KV/$C$/$S$ write changes the map (`state_write_not_optional`). Chunkwise GDN is not zero $S$ traffic. Surviving store is $B_\text{store}(T)$, not $T\times B_S$.

### Semantic-node comparison with decode (lock; heading 7; second completion criterion)

JSON `node_compare_ids` equals `node_type_ids` (6 ids, TASK-11 order). JSON `n_nodes_compared_with_decode` = 6. JSON `every_semantic_node_compared` true. JSON `n_stage_kinds_compared_with_decode` = 9.

Implementation copies these six rows; do not drop a type.

| Node | Decode | Prefill | Graph |
| --- | --- | --- | --- |
| `embed` | Gather 1 row (10240 B); 0 MAC; no state | Gather $T$ rows ($T\times 10240$); 0 MAC; no state; complete also gathers $T$ next-ids | same type, two instances |
| `gated_attn` | GEMV projs; one query against stored length $T$; read KV $T-1$; write 1 token; incoming populated | GEMM projs over $T$; causal MM with $T(T+1)/2$; triangular KV; write $T$ tokens; incoming zeros; $A=12288$ per instance | same type, 17 instances |
| `gated_delta_net` | One recurrent step `(17)`–`(18)`; read populated $C,S$; write $C,S$ | $T$ recurrent steps from zeros; C/S $\times T$ writes; first-token physical read 0; chunkwise algebraic equivalent **inside** the node | same type, 48 instances |
| `mlp` | GEMV $3IH$; no state | GEMM with sequence $T$; unique weights once; no state | same type, 65 instances |
| `lm_head` | GEMV $VH$; 1×$V$ logits | GEMM $T\times V$ (primary); `inference_last_logits` omits $T-1$ (unselected mode) | same type, 2 instances |
| `mtp_mix` | GEMV `mtp.fc` on one token | GEMM `mtp.fc` on $T$; teacher-forced `e_next` length $T$ | same type, 1 instance |

JSON object `node_vs_decode_work_class` keyed in `node_type_ids` order, values exactly: `embed` → `gather_T_vs_1`; `gated_attn` → `gemm_causal_mm_vs_gemv`; `gated_delta_net` → `recurrence_T_vs_1`; `mlp` → `gemm_T_vs_gemv`; `lm_head` → `gemm_T_vs_gemv`; `mtp_mix` → `gemm_T_vs_gemv`.

JSON object `node_vs_decode_state_class`: `embed`/`mlp`/`lm_head`/`mtp_mix` → `none`; `gated_attn` → `kv_triangular_from_zeros`; `gated_delta_net` → `cs_from_zeros`.

JSON object `node_vs_decode_tiling_axis`: `embed` → `gather_rows`; `gated_attn` → `causal_sequence`; `gated_delta_net` → `recurrent_steps`; `mlp`/`lm_head`/`mtp_mix` → `gemm_sequence`.

Companion stage-kind comparison (required; 9 rows in prose, JSON `stage_vs_decode_rank` all `"T_vs_1"` except documenting that kinds and I/O **names** match decode). JSON `stage_kinds_match_decode` true. JSON `stage_kind_ids_equal_decode` true.

Required closing sentence: every semantic node is the **same** TASK-11 type with a **different** sequence rank, work class, and incoming-state convention; no extra node type is introduced for prefill.

### Unavoidable versus proposed-boundary traffic (lock; heading 8)

Three channels. **Unavoidable** = TASK-06 mathematical minimum for one complete prefill of length $T$. **Proposed-boundary** = this layer-serial schedule’s named stage-cut transfers (T-scaled). Do not recopy TASK-06 family tables.

**Weight**

| Item | Bytes | Label |
| --- | ---: | --- |
| Unique non-embed language+MTP | 52098598912 | unavoidable (cite TASK-06); counted once, reused across $T$ |
| Gather complete | $2T\cdot 10240$ | unavoidable |
| Gather at $T=1$ / $T=4096$ | 20480 / 83886080 | DERIVED |
| Unique+gather at $T=1$ / $T=4096$ | 52098619392 / 52182484992 | DERIVED |
| Unique extra vs TASK-06 | 0 | DERIVED |
| Second physical $W_\text{lm}$ read | 2542796800 | HYPOTHESIS extra, not unique |

JSON: `weight_bytes_unique_non_embed` 52098598912, `weight_gather_bytes_prefill_complete_at_example_T` `[20480, 83886080]`, `weight_gather_bytes_prefill_language_at_example_T` `[10240, 41943040]`, `weight_unique_plus_gather_prefill_complete_at_example_T` `[52098619392, 52182484992]`, `weight_boundary_added_unique_bytes` 0, `weight_second_w_lm_read_bytes` 2542796800.

**State**

| Item | Bytes | Label |
| --- | --- | --- |
| KV write $69632T$ | 69632 / 285212672 | unavoidable |
| KV read triangular | 0 / 583972945920 | unavoidable |
| C/S write $\times T$ | see arrays | unavoidable (mathematical per-step) |
| C/S physical read $\times(T-1)$ | 0 / see arrays | DERIVED from zeros |
| Extra vs TASK-06 identities | 0 | DERIVED |

JSON `state_boundary_added_bytes` 0.

**Activation — T-scaled views plus this stage-cut**

Cite TASK-06: `act_region_cut_prefill_complete_at_example_T` `[5847040, 23949475840]`; language `[5245952, 21487419392]`. Prefill region-cut = $T\times$ decode region-cut. Neither is a CUDA live-set.

Forced view is **not** a TASK-06 JSON key for prefill; DERIVE analogously as $T\times$ decode forced (per-position map): JSON `act_forced_prefill_complete_at_example_T` `[3123200, 12792627200]`; language `[2593792, 10624176128]`.

Stage-cut unique bytes (this schedule; each named data edge once; identification applied; `h_64` counted once; **ranks $\times T$**):

JSON crossing **counts** match decode (edges, not bytes): `n_h_mid_crossings` 65, `n_h_interlayer_crossings` 63, `n_identity_e_h0_crossings` 1, `n_fanout_h64_crossings` 1, `n_embed_e_next_crossings` 1, `n_mtp_u_crossings` 1, `n_h_mtp_crossings` 1, `n_residual_stage_crossings` 133.

JSON `residual_bytes` 10240. JSON `act_residual_stage_cut_prefill_at_example_T` = $133\times T\times 10240$ = `[1361920, 5578424320]`.
JSON `logits_bytes` 496640. JSON `n_logit_outputs` = 2. JSON `act_logits_stage_cut_prefill_at_example_T` = $2\times T\times 496640$ = `[993280, 4068474880]`.
JSON `act_stage_cut_prefill_complete_at_example_T` = `[2355200, 9646899200]`.

Checker: `act_stage_cut_prefill_complete_at_example_T[i] == 2355200 * example_T[i]`.
Checker: `act_residual_stage_cut_prefill_at_example_T[i] == n_residual_stage_crossings * residual_bytes * example_T[i]`.
Checker: `act_region_cut_prefill_complete_at_example_T[i] == 5847040 * example_T[i]`.
Checker: `act_forced_prefill_complete_at_example_T[i] == 3123200 * example_T[i]`.

`g`/`z` stay inside mixers: JSON `act_gz_local_prefill_at_example_T` = $T\times 798720$ = `[798720, 3271557120]`. These are **not** stage-cut bytes.

JSON `act_boundary_added_vs_forced_at_example_T` `[−768000, −3145728000]`.
JSON `act_boundary_added_vs_region_cut_at_example_T` `[−3491840, −14302576640]`.

Negative values are DERIVED identities, not “the schedule is cheaper than liveness.” Same interpretation as TASK-13, T-scaled.

JSON `act_h64_second_consumer_bytes_at_example_T` `[10240, 41943040]` — extra physical read of the `h_64` **sequence** by the second consumer is HYPOTHESIS. Unique stage-cut counts `h_64` once.

JSON `act_h_mid_stage_cut_prefill_at_example_T` = $65\times T\times 10240$ = `[665600, 2726297600]`. `fuse_across_residual_h_mid` would remove this as inter-stage I/O. Usefulness HYPOTHESIS (and may differ at large $T$); do not apply that fusion here.

Last-logits secondary (not primary stage-cut): logits channel would be $2\times 496640$ not $\times T$. JSON `act_logits_stage_cut_last_logits_at_example_T` `[993280, 993280]`. Do not use this as the primary proposed-boundary number.

Required paragraph: this heading reports DERIVED extras of 0 on unique-weight and state channels versus TASK-06. It does **not** close the ledger open question. Representation/view tradeoffs remain unresolved in heading 9.

### Packing, fusion, and representation tradeoffs (lock; heading 9; third completion criterion — keep unresolved)

**Packing (TASK-09; none selected).** JSON `consumer_sequence_ids` copied from TASK-09 in TASK-09 order (7 ids). JSON `n_consumer_sequences` = 7. JSON `ideal_byte_sequence_selected` false.

Attach sequences via `stage_primary_sequence_ids` plus `gated_delta_net_state_sequence_id` — **the same attachments as decode**. Prefill GEMMs do not get a different selected sequence. Name `seq_gemm_interleaved_group`, `seq_outlier_extra`, and `seq_specialized_tile` as unselected alternatives. Do not rank by wall time. Do not close `artifact_boundary`. Layout of `seq_specialized_tile` remains TASK-15.

**Fusion (TASK-12; none selected).** JSON `fusion_hypothesis_ids` copied from TASK-12 in TASK-12 order (22 ids). JSON `n_fusion_hypotheses` = 22. JSON `n_fusion_hypotheses_selected` = 0. JSON object `fusion_selected` all false. JSON `fusion_winner_selected` false. JSON `fusion_usefulness_label` exactly `HYPOTHESIS`. Prefill working-set scales with $T$, so usefulness **may differ** from decode; that difference is still HYPOTHESIS, not a selected winner.

**Hoist / overlap (this task; none selected).** JSON array `hoist_hypothesis_ids` in this exact order (5 ids, **same as decode**). JSON `n_hoist_hypotheses` = 5. JSON `n_hoist_hypotheses_selected` = 0. JSON object `hoist_selected` all false. JSON `hoist_usefulness_label` exactly `HYPOTHESIS`.

| id | Meaning |
| --- | --- |
| `hoist_embed_next` | Issue `embed_next` as soon as the next-id sequence is present |
| `overlap_fanout_h64` | `lm_head_primary` and `mtp_mix` are both ready after the `h_64` sequence |
| `reuse_E` | $T+T$ gathers of one $E$ payload |
| `reuse_W_lm` | Two `lm_head` instances of one $W_\text{lm}$ payload reused across $T$; second physical read is HYPOTHESIS extra bytes |
| `reuse_h64` | Two consumers of one `h_64` sequence; extra physical read is HYPOTHESIS $T\times 10240$ B |

**Mode hypotheses (this task; none selected as required).** JSON array `mode_hypothesis_ids` in this exact order (2 ids). JSON `n_mode_hypotheses` = 2. JSON `n_mode_hypotheses_selected` = 0. JSON object `mode_selected` all false. JSON `mode_usefulness_label` exactly `HYPOTHESIS`.

| id | Meaning |
| --- | --- |
| `token_serial_prefill` | Replay the decode schedule for $t=1,\ldots,T$ (TASK-06 work sum). Algebraic equivalent; **not** the primary layer-serial GEMM schedule |
| `inference_last_logits` | Omit $T-1$ vocabulary projections (TASK-06 secondary). Mixers/MLP/MTP still scale with $T$ |

**Representation / view hypotheses (this task; none selected — open question stays open).** JSON array `representation_hypothesis_ids` in this exact order (4 ids). JSON `n_representation_hypotheses` = 4. JSON `n_representation_hypotheses_selected` = 0. JSON object `representation_selected` all false. JSON `representation_usefulness_label` exactly `HYPOTHESIS`. JSON `decode_prefill_distinct_views_selected` false. JSON `ledger_open_question_representation_tradeoffs_closed` false. JSON `representation_tradeoffs_unresolved` true.

| id | Meaning |
| --- | --- |
| `distinct_prefill_gemm_view` | A specialized packed view optimized for prefill GEMM / causal MM |
| `distinct_decode_gemv_view` | A specialized packed view optimized for decode GEMV / one-query attention |
| `shared_view_both_modes` | One packed view consumed by both schedules |
| `dual_view_binding` | `view_binding` holds both a prefill-oriented and a decode-oriented specialized view in the **one** shared artifact |

Required prose: TASK-09 already locked **one artifact** (`decode_prefill_share_artifact` true) and the `view_binding` **capability**. This heading enumerates whether the two schedules need distinct **views**. There is no MEASURED evidence that GEMM-vs-GEMV packing, causal-vs-one-query KV layout, or gather-vs-GEMM embed layout must split. Therefore **no** representation hypothesis is selected. Closing the question by baking two artifacts is already rejected by TASK-09. Closing it by selecting `shared_view_both_modes` or `dual_view_binding` would also be a close without evidence — do not. TASK-15/17 may later inform a close; this task must not.

### Work citations and non-decisions (lock; heading 10)

Cite; do not recopy TASK-06 symbolic tables. Checker **recomputes** $C,A,W_\text{prefill}(T)=TC+AT(T+1)/2$ from `text_config` with the same identities as TASK-06.

JSON: `mac_C_complete` 27433238528, `mac_A_complete` 208896, `mac_prefill_complete_at_example_T` `[27433447424, 114119319486464]`, `mac_prefill_language_at_example_T` `[25737363456, 107069105504256]`, `mac_prefill_inference_last_logits_complete_at_example_T` `[27433447424, 103706566590464]`, `mac_prefill_inference_last_logits_language_at_example_T` `[25737363456, 101862729056256]`. Language secondary: `mac_C_language` 25737166848, `mac_A_language` 196608. Identity $W_\text{prefill}=TC+AT(T+1)/2$. Last-logits complete $W-2(T-1)C_\text{lm}$.

Checker: `mac_prefill_complete_at_example_T[i] == mac_C_complete * example_T[i] + mac_A_complete * example_T[i] * (example_T[i] + 1) // 2`.
Checker: `mac_prefill_complete_at_example_T[0] == mac_C_complete + mac_A_complete` (equals decode at $T=1$).

JSON `bottleneck_labels` copied from TASK-06: `weight_memory`, `vocab_memory`, `state_memory`, `kv_memory`, `quadratic_attn`, `compute`. Restating a label here is a **citation**, still HYPOTHESIS. JSON `n_bottleneck_labels` = 6. Prefill **does** use `quadratic_attn` and `compute` as the many-token classes; still copy the six-id list, do not rank.

JSON: `mac_full_proj_per_layer` 104857600, `mac_lin_token_per_layer` 118235136, `mac_mlp_per_layer` 267386880, `mac_lm_head` 1271398400, `mac_mtp_fc` 52428800, `mac_gdn_per_layer` 2359296, `i_mlp_weight_only` 1, `i_lm_head_weight_only` 1, `i_gdn_vs_s_rw` 0.75, `i_attn_core_vs_kv` 6 (decode identity cited; do not invent a prefill-ridge winner).

Non-decisions (prose required): TASK-15 owns tiles. TASK-17 owns CUDA mappings per node type. TASK-09 still owns artifact-boundary and ideal-sequence **selection**. TASK-12 still owns fusion **winners**. Decode/prefill **view** selection stays open (`ledger_open_question_representation_tradeoffs_closed` false). This schedule does not change when a TASK-08 recipe is later applied.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 9 (Packing, fusion, and representation tradeoffs). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 135 instances, or $T$ positions. Annotate mixer xor and sequence rank $T$ in prose / a label, not as 64 or $T$ subgraphs. Caption must contain the word `HYPOTHESIS` (hoist/fusion/view remain hypotheses) and the word `unresolved` (representation tradeoffs).

Required IDs **inside that fence**: `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`, `h`, `h_mid`, `h_64`, `K_state`, `V_state`, `C_state`, `S`, `logits_0`, `logits_1`.

JSON `n_diagrams` is 1. `diagram_ids` is `["embed_current","language_mixer","language_mlp","lm_head_primary","embed_next","mtp_mix","mtp_mixer","mtp_mlp","lm_head_mtp","h","h_mid","h_64","K_state","V_state","C_state","S","logits_0","logits_1"]`.

### Deferred vision

Visual tokens may replace placeholders on the `identity_e_h0` edge (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger prefill schedules and visual **sequence length** are **UNKNOWN**. Do not add a vision stage kind. `vision_interface_is_not_a_node` true. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_prefill_plan.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, residual/`g`/`z`/`logits` bytes, cited TASK-06 $C,A,W_\text{prefill}(T)$, unique-weight+gather, triangular KV, C/S $\times T$, and T-scaled stage-cut. Duplicate TASK-11 node-type / sync-edge lists, TASK-09 `consumer_sequence_ids`, TASK-12 `fusion_hypothesis_ids`, and TASK-13 stage-kind / hoist / ready-constraint lists as constants; do not import them.

The ledger **Produces** line names only `docs/architecture/prefill-plan.md`. The checker is stdlib evidence tooling matching TASK-01–13 and the user-required stdlib checker; it is in scope for this increment.

CLI (cwd = repository root):

```text
python3 scripts/check_prefill_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_prefill_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --prefill-plan docs/architecture/prefill-plan.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: instance counts, $C,A,W_\text{prefill}(T)$, residual/`g`/`z`/`logits` bytes, T-scaled stage-cut products, cited unique-weight+gather, triangular KV, C/S $\times T$. Constant fields: canonical sentences, stage/hoist/mode/fusion/sequence/representation lists, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--prefill-plan PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all four canonical sentences verbatim, (5) every `stage_kind_ids`, `node_type_ids`, `fundamental_difference_ids`, `consumer_sequence_ids`, `fusion_hypothesis_ids`, `hoist_hypothesis_ids`, `mode_hypothesis_ids`, `representation_hypothesis_ids`, `ready_constraint_ids`, `sync_edge_ids`, `bottleneck_labels` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer below present as a decimal or integer substring, (10) the words `HYPOTHESIS`, `hardware-independent`, and `unresolved` present, (11) none of the forbidden winner phrases: `selected fusion`, `selected packing`, `selected kernel`, `winning fusion`, `should fuse`, `recommend fusion`, `ideal byte sequence is`, `artifact boundary is`, `CUDA kernel fusion is required`, `thread block`, `warp shuffle`, `Quartz graph`, `llama.cpp graph`, `GGUF is the schedule`, `prefill requires a distinct view`, `decode and prefill need different artifacts`, `selected distinct views`, `multiple views are required`, `representation tradeoff is closed`, `ideal prefill layout is` (allow the substring only inside `not a selected fusion` / `not a CUDA graph, kernel launch sequence, or selected fusion` / `view_binding stays a capability, not a selected split` / `none is a selected winner` / `remain unresolved`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks cited TASK-06 integers against `work-and-traffic.md`, sequence/fusion ids against TASK-09/12 markdown, and stage-kind ids against `decode-plan.md`).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `n_stage_kinds==9`, `n_stage_instances==135`, `sum(stage_multiplicities)==135`
- `stage_multiplicities == [1,64,64,1,1,1,1,1,1]`
- `n_node_instances_complete==135`, `n_node_instances_language==130`
- `n_embed_instances==2`, `n_gated_attn_instances==17`, `n_gated_delta_net_instances==48`, `n_mlp_instances==65`, `n_lm_head_instances==2`, `n_mtp_mix_instances==1`
- `n_ready_constraints==10`, `n_hoist_hypotheses==5`, `n_hoist_hypotheses_selected==0`
- `n_mode_hypotheses==2`, `n_mode_hypotheses_selected==0`
- `n_representation_hypotheses==4`, `n_representation_hypotheses_selected==0`
- `n_fundamental_differences==5`, `n_nodes_compared_with_decode==6`
- `n_consumer_sequences==7`, `n_fusion_hypotheses==22`, `n_fusion_hypotheses_selected==0`
- `n_sync_edges==14`, `n_diagrams==1`, `n_token_presentation_streams==2`
- `n_residual_stage_crossings==133`, `n_h_mid_crossings==65`, `n_h_interlayer_crossings==63`
- `residual_bytes==10240`, `act_residual_stage_cut_prefill_at_example_T == [1361920, 5578424320]`
- `logits_bytes==496640`, `act_logits_stage_cut_prefill_at_example_T == [993280, 4068474880]`
- `act_stage_cut_prefill_complete_at_example_T == [2355200, 9646899200]`
- `act_forced_prefill_complete_at_example_T == [3123200, 12792627200]`
- `act_region_cut_prefill_complete_at_example_T == [5847040, 23949475840]`
- `act_boundary_added_vs_forced_at_example_T == [-768000, -3145728000]`
- `act_boundary_added_vs_region_cut_at_example_T == [-3491840, -14302576640]`
- `act_gz_local_prefill_at_example_T == [798720, 3271557120]`
- `act_h_mid_stage_cut_prefill_at_example_T == [665600, 2726297600]`
- `weight_bytes_unique_non_embed==52098598912`
- `weight_gather_bytes_prefill_complete_at_example_T == [20480, 83886080]`
- `weight_unique_plus_gather_prefill_complete_at_example_T == [52098619392, 52182484992]`
- `weight_boundary_added_unique_bytes==0`, `state_boundary_added_bytes==0`
- `prefill_kv_read_bytes_at_example_T == [0, 583972945920]`
- `prefill_kv_write_bytes_at_example_T == [69632, 285212672]`
- `prefill_write_bytes_at_example_T == [152047616, 622783035392]`
- `prefill_read_physical_at_example_T == [0, 1214369888256]`
- `mac_C_complete==27433238528`, `mac_A_complete==208896`
- `mac_prefill_complete_at_example_T == [27433447424, 114119319486464]`
- `mac_prefill_complete_at_example_T[i] == mac_C_complete * example_T[i] + mac_A_complete * example_T[i] * (example_T[i] + 1) // 2`
- `mac_prefill_equals_decode_at_T1 is True`
- `state_read_prefill_equals_decode_at_T1 is False`
- `i_mlp_weight_only==1`, `i_lm_head_weight_only==1`, `i_gdn_vs_s_rw==0.75`, `i_attn_core_vs_kv==6`
- `schedule_serial_selected is True`, `schedule_layer_serial_selected is True`
- `fusion_winner_selected is False`, `ideal_byte_sequence_selected is False`
- `decode_prefill_distinct_views_selected is False`
- `ledger_open_question_representation_tradeoffs_closed is False`
- `representation_tradeoffs_unresolved is True`
- `every_semantic_node_compared is True`
- `thread_geometry_absent is True`, `hardware_independent is True`
- `incoming_state_zeros is True`, `incoming_state_populated is False`
- `weight_unique_counted_once is True`, `weight_second_w_lm_read_is_hypothesis is True`
- `live_across_are_internal is True`, `state_write_not_optional is True`
- `fanout_h64_cannot_hide_from_one_consumer is True`, `chunkwise_not_zero_s_traffic is True`
- `tile_size_selected is False`
- every `fusion_selected` / `hoist_selected` / `mode_selected` / `representation_selected` value is False
- `stage_kind_ids` equals the locked 9-id list; `fusion_hypothesis_ids` equals the locked 22-id list
- `consumer_sequence_ids` equals the TASK-09 7-id list
- `fundamental_difference_ids` equals the locked 5-id list
- `node_compare_ids` equals `node_type_ids`
- `"g" not in` any `stage_visibility_in` or `stage_visibility_out` value
- `"language_mixer" in stage_kind_ids` and `"seq_lm_head_full" in stage_primary_sequence_ids`

Locked document integers/decimals the `--prefill-plan` check must find:

`5120`, `17408`, `248320`, `135`, `130`, `133`, `2355200`, `9646899200`, `1361920`, `5578424320`, `993280`, `4068474880`, `3123200`, `12792627200`, `5847040`, `23949475840`, `665600`, `2726297600`, `798720`, `3271557120`, `10240`, `20480`, `83886080`, `41943040`, `496640`, `52098598912`, `52098619392`, `52182484992`, `152047616`, `153944064`, `583972945920`, `285212672`, `4026531840`, `618471290880`, `622783035392`, `12076646400`, `618320295936`, `1214369888256`, `69632`, `150994944`, `2542796800`, `27433238528`, `208896`, `27433447424`, `114119319486464`, `107069105504256`, `103706566590464`, `101862729056256`, `104857600`, `118235136`, `267386880`, `1271398400`, `0.75`, `-768000`, `-3145728000`, `-3491840`, `-14302576640`

(The integer `48` / `16` / `64` / `9` / `22` / `7` / `14` / `65` / `17` / `5` / `6` / `4` / `2` will appear from counts and types; requiring the byte/MAC-scale set plus `135` / `133` / `2355200` / `9646899200` is the hard check. Do not require `1e-06`.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `n_attn_heads`, `n_kv_heads`, `head_dim`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `bytes_bf16`, `bytes_f32`,

`node_type_ids`, `n_node_types`, `n_embed_instances`, `n_gated_attn_instances`, `n_gated_delta_net_instances`, `n_mlp_instances`, `n_lm_head_instances`, `n_mtp_mix_instances`, `n_node_instances_complete`, `n_node_instances_language`,

`stage_kind_ids`, `n_stage_kinds`, `stage_multiplicities`, `stage_node_types`, `n_stage_instances`, `serial_stage_kind_order`, `stage_primary_sequence_ids`, `gated_delta_net_state_sequence_id`, `stage_visibility_in`, `stage_visibility_out`, `stage_state_read`, `stage_state_write`, `language_mixer_state_is_xor`, `activation_sequence_length_is_T`, `stage_kinds_match_decode`,

`ready_constraint_ids`, `n_ready_constraints`, `hoist_hypothesis_ids`, `hoist_selected`, `n_hoist_hypotheses`, `n_hoist_hypotheses_selected`, `hoist_usefulness_label`, `mode_hypothesis_ids`, `mode_selected`, `n_mode_hypotheses`, `n_mode_hypotheses_selected`, `mode_usefulness_label`, `representation_hypothesis_ids`, `representation_selected`, `n_representation_hypotheses`, `n_representation_hypotheses_selected`, `representation_usefulness_label`,

`fundamental_difference_ids`, `n_fundamental_differences`, `fundamental_difference_usefulness_label`, `node_compare_ids`, `n_nodes_compared_with_decode`, `n_stage_kinds_compared_with_decode`, `node_vs_decode_work_class`, `node_vs_decode_state_class`, `node_vs_decode_tiling_axis`,

`n_token_presentation_streams`, `example_T`, `prefill_T_new_equals_T`, `T_is_stored_length_after_prefill`, `incoming_state_zeros`, `incoming_state_populated`, `primary_includes_mtp`, `mtp_runs_all_T_positions`,

`mac_C_language`, `mac_C_complete`, `mac_A_language`, `mac_A_complete`, `mac_prefill_language_at_example_T`, `mac_prefill_complete_at_example_T`, `mac_prefill_inference_last_logits_language_at_example_T`, `mac_prefill_inference_last_logits_complete_at_example_T`, `mac_full_proj_per_layer`, `mac_lin_token_per_layer`, `mac_mlp_per_layer`, `mac_lm_head`, `mac_mtp_fc`, `mac_gdn_per_layer`, `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_gdn_vs_s_rw`, `i_attn_core_vs_kv`, `bottleneck_labels`, `n_bottleneck_labels`,

`weight_bytes_unique_non_embed`, `weight_gather_bytes_per_row`, `weight_gather_bytes_prefill_language_at_example_T`, `weight_gather_bytes_prefill_complete_at_example_T`, `weight_unique_plus_gather_prefill_complete_at_example_T`, `weight_boundary_added_unique_bytes`, `weight_second_w_lm_read_bytes`, `weight_bytes_lm_head`,

`kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`, `c_bytes_per_layer`, `c_write_bytes_all`, `c_read_bytes_all`, `s_bytes_per_layer`, `s_bytes_all`, `decode_write_bytes`, `decode_read_fixed_bytes`, `prefill_kv_read_bytes_at_example_T`, `prefill_kv_write_bytes_at_example_T`, `prefill_c_write_bytes_at_example_T`, `prefill_s_write_bytes_at_example_T`, `prefill_write_bytes_at_example_T`, `prefill_c_read_math_at_example_T`, `prefill_s_read_math_at_example_T`, `prefill_c_read_physical_at_example_T`, `prefill_s_read_physical_at_example_T`, `prefill_read_physical_at_example_T`, `storage_kv_bytes_coeff_T`, `storage_fixed_bytes`, `storage_bytes_at_example_T`, `state_boundary_added_bytes`,

`residual_bytes`, `g_bytes`, `z_bytes`, `logits_bytes`, `n_logit_outputs`, `n_identity_e_h0_crossings`, `n_h_mid_crossings`, `n_h_interlayer_crossings`, `n_fanout_h64_crossings`, `n_embed_e_next_crossings`, `n_mtp_u_crossings`, `n_h_mtp_crossings`, `n_residual_stage_crossings`, `act_residual_stage_cut_prefill_at_example_T`, `act_logits_stage_cut_prefill_at_example_T`, `act_stage_cut_prefill_complete_at_example_T`, `act_forced_prefill_complete_at_example_T`, `act_forced_prefill_language_at_example_T`, `act_region_cut_prefill_complete_at_example_T`, `act_region_cut_prefill_language_at_example_T`, `act_boundary_added_vs_forced_at_example_T`, `act_boundary_added_vs_region_cut_at_example_T`, `act_gz_local_prefill_at_example_T`, `act_h_mid_stage_cut_prefill_at_example_T`, `act_h64_second_consumer_bytes_at_example_T`, `act_h64_second_consumer_is_hypothesis`, `act_logits_stage_cut_last_logits_at_example_T`,

`consumer_sequence_ids`, `n_consumer_sequences`, `fusion_hypothesis_ids`, `n_fusion_hypotheses`, `fusion_selected`, `n_fusion_hypotheses_selected`, `fusion_usefulness_label`, `sync_edge_ids`, `n_sync_edges`,

`decode_prefill_share_graph`, `decode_prefill_share_artifact`, `decode_prefill_distinct_views_selected`, `residual_add_inside_mixer`, `residual_add_inside_mlp`, `residual_input_live_until_add`, `rms_inside_consumer`, `live_across_are_internal`, `hardware_independent`, `cuda_mapping_deferred`, `thread_geometry_absent`, `layout_selected`, `tile_size_selected`, `fusion_winner_selected`, `ideal_byte_sequence_selected`, `artifact_boundary_selected`, `schedule_serial_selected`, `schedule_layer_serial_selected`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_not_zero_s_traffic`, `state_write_not_optional`, `fanout_h64_cannot_hide_from_one_consumer`, `weight_unique_counted_once`, `weight_second_w_lm_read_is_hypothesis`, `mixer_xor_by_layer_types`, `activation_dtype_decided`, `vision_interface_is_not_a_node`, `mtp_omission_is_algebraic_equivalent`, `mtp_runs_all_T_positions`, `every_semantic_node_compared`, `representation_tradeoffs_unresolved`, `ledger_open_question_representation_tradeoffs_closed`, `mac_prefill_equals_decode_at_T1`, `state_read_prefill_equals_decode_at_T1`, `kv_incoming_empty_at_T1_both`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_schedule`, `canonical_sentence_traffic`, `canonical_sentence_open_question`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. Intensities `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_attn_core_vs_kv` are JSON numbers `1`, `1`, `6`. `i_gdn_vs_s_rw` is JSON number `0.75`. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. Boundary-added arrays are JSON int arrays (negative values allowed). `stage_visibility_in`, `stage_visibility_out`, `stage_state_read`, `stage_state_write` are JSON objects keyed in `stage_kind_ids` order. `fusion_selected`, `hoist_selected`, `mode_selected`, and `representation_selected` are JSON objects with JSON `false` values. Usefulness labels are the JSON string `HYPOTHESIS`. `stage_state_read["language_mixer"]` is `["K_state","V_state","C_state","S"]` as the xor union. `stage_state_write["language_mixer"]` same union. `stage_state_read["mtp_mixer"]` `["K_state","V_state"]`.

`node_type_ids` copied from TASK-11: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`. JSON `n_node_types` = 6.

`sync_edge_ids` copied from TASK-11 in TASK-11 order (14 ids): `identity_e_h0`, `residual_h`, `residual_h_mid`, `fanout_h64`, `embed_e_next`, `mtp_u_to_block`, `h_mtp_to_logits`, `output_logits_0`, `output_logits_1`, `state_kv`, `state_c`, `state_s`, `shared_E`, `shared_W_lm`.

`consumer_sequence_ids` copied from TASK-09: `seq_gemm_codes_then_scales`, `seq_gemm_interleaved_group`, `seq_gather_row`, `seq_lm_head_full`, `seq_outlier_extra`, `seq_state_s_dense`, `seq_specialized_tile`.

`fusion_hypothesis_ids` copied from TASK-12 (22 ids, TASK-12 order): `fuse_gated_attn_internals`, `fuse_gated_delta_net_internals`, `fuse_mlp_internals`, `fuse_lm_head_internals`, `fuse_mtp_mix_internals`, `split_g`, `split_z`, `split_h_tilde`, `split_k_rope`, `split_v_full`, `split_qkv`, `split_h_post`, `split_swiglu`, `split_h_final`, `split_mtp_cat`, `fuse_across_identity_e_h0`, `fuse_across_residual_h`, `fuse_across_residual_h_mid`, `fuse_across_fanout_h64`, `fuse_across_embed_e_next`, `fuse_across_mtp_u_to_block`, `fuse_across_h_mtp_to_logits`.

Boolean `fusion_winner_selected` false. One key `fusion_selected` (object, all false). Do **not** emit a second top-level boolean named `fusion_selected`.

### Stage split

- **Implementation** writes `scripts/check_prefill_plan.py` **and** `docs/architecture/prefill-plan.md` (twelve headings, nine stage kinds summing to 135 instances, layer-serial order over $T$, five fundamental differences, 6/6 node comparison, per-stage loads/state/visibility/reuse, unavoidable vs T-scaled stage-cut traffic, 7 sequences + 22 fusion + 5 hoist + 2 mode + 4 representation hypotheses all unselected, JSON fence). Runs `--json` and `--prefill-plan` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-07 / `numerical-sensitivity.md` / `semantic-graph.md` / `decode-plan.md` style), Authority table links to this dossier / work-and-traffic / runtime-format / semantic-graph / materialization-and-fusion / decode-plan (comparison) / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, stage-kind ids, sequence ids, fusion ids, representation ids, or Mermaid node IDs. Must not close the representation question. Does not edit TASK-06/09/11/12/13 artifacts.
- **Verification** independently re-runs focused commands, recomputes $TC+AT(T+1)/2$, T-scaled residual/stage-cut bytes, unique+gather, triangular KV, and C/S $\times T$ from sitting `text_config` (not from JSON echo), spot-checks cited TASK-06 integers against `docs/architecture/work-and-traffic.md` and sequence/fusion id lists against `docs/architecture/runtime-format-design.md` / `docs/architecture/materialization-and-fusion.md`, spot-checks `stage_kind_ids` against `docs/architecture/decode-plan.md` (comparison only, not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-schedule, no winner phrases, no `plan.md` or ledger edit, no payload I/O, that every fusion/hoist/mode/representation hypothesis remains unselected, that contracts are hardware-independent (no CUDA kernel/thread/SKU decisions, `thread_geometry_absent` true), that `g`/`z` are not stage-cut I/O, that unique-weight and state extras are 0, that all 6 node types are compared, that the five fundamental differences are present, and that the representation open question is **not** closed. Layouts and CUDA mapping remain open. Ideal byte sequence remains unselected. Distinct views remain unselected.

- Invariants:
  - Twelve level-2 headings in the locked order; four canonical sentences verbatim; 9 stage kinds; 135 instances; 10 ready constraints; 5 hoist + 2 mode + 4 representation hypotheses all unselected; 22 fusion hypotheses all unselected; 7 sequences attached unselected; 5 fundamental differences; 6/6 nodes compared; one Mermaid flowchart with required IDs.
  - Layer-serial order selected; token-serial and last-logits not required; primary includes MTP at all $T$; mixer xor by `layer_types`; residual add inside mixer and `mlp`; `g`/`z` internal; unique weights counted once; state writes not optional; incoming zeros.
  - GDN numerical definition remains recurrent `(17)`–`(18)`; chunkwise is not zero $S$ traffic.
  - Unavoidable unique-weight extra 0; state extra 0; stage-cut activation $T\times 2355200$ B; forced $T\times 3123200$; region-cut $T\times 5847040$.
  - Logical values do not imply allocation; packing/fusion/view usefulness is HYPOTHESIS; no thread geometry; no selected tile; representation tradeoffs unresolved.
  - Vision encoder remains unexpanded and is not a stage.
- Rejected alternatives:
  - Token-serial decode replay as the **primary** schedule: rejected; it is the TASK-06 work sum but hides matrix-matrix GEMMs the ledger requires explained. Keep as unselected `token_serial_prefill`.
  - Selecting `inference_last_logits` as primary: rejected; TASK-06 secondary row; mixers still scale with $T$.
  - Unrolling 135 instance ids or $T$ positions as the published stage list: rejected; compact 9 kinds + mixer xor match TASK-11’s “do not unroll 64 layers.”
  - One `layer` stage wrapping mixer+MLP as the **required** serial unit: rejected; TASK-11 `cut_mixer_mlp` and TASK-04 `h_mid`.
  - Required proj/core/out or conv/recurrence/out sub-stages: rejected; `g`/`z` remain internal; splits stay TASK-12 hypotheses.
  - Selecting any fusion, packing sequence, artifact boundary, tile, or view because GEMM “looks different” from GEMV: rejected; no MEASURED evidence; TASK-09/12 leave packing/fusion open; this task **must** leave views open.
  - Closing the representation question by selecting `shared_view_both_modes` or `dual_view_binding`: rejected; that would be a close without evidence. Enumerate, select none.
  - Two compiled artifacts for decode vs prefill: rejected by TASK-09; one artifact, `view_binding` capability.
  - Treating TASK-06 region-cut as this schedule’s I/O: rejected; region-cut includes internals the schedule keeps inside nodes.
  - Treating negative `act_boundary_added_vs_forced` as “g/z deleted”: rejected; they remain local working set, T-scaled.
  - Counting unique MLP/attn weights $\times T$ as prefill traffic: rejected; unique counted once and reused.
  - Streaming the full embed table: rejected; gather $T$ or $2T$ rows (TASK-06).
  - Second $W_\text{lm}$ read as DERIVED unique bytes: rejected; TASK-06 HYPOTHESIS.
  - Chunkwise GDN as zero $S$ traffic or as extra stages: rejected; TASK-02/06/07/11.
  - Claiming $T=1$ prefill **state-read** equals continuing decode: rejected; MAC equals, C/S physical read does not.
  - Separate language-only primary schedule: rejected; complete map includes MTP at all $T$.
  - Copying decode-plan as the prefill document without T-differences: rejected; schedules are independent; comparison is required.
  - Making TASK-13 a silent extra dependency that blocks independence: rejected; cite decode-plan for comparison only; re-derive order from TASK-11.
  - Selecting physical tile extents: TASK-15.
  - Importing TASK-16 occupancy / thread geometry: rejected; not a dependency; hardware-independent.
  - Inspecting Quartz or llama.cpp for “real” prefill graphs: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–13.
  - Editing frozen TASK-06/09/11/12/13 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py`.
  - Multiple selected serial orders: rejected; one layer-serial order is the schedule; hoist/mode are HYPOTHESIS flexibility.
- Discovered ledger work: `none`
- Unresolved decisions: `none` (the scientific representation/view question is **decidedly left open**; that is the third completion criterion, not a planning gap)

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/prefill-plan.md` exists and follows the heading list above.
  - Nine stage kinds with multiplicities summing to 135; layer-serial order selected; per-stage loads, state R/W, visibility, and reuse tabulated.
  - Five fundamental differences (`diff_matrix_matrix`, `diff_reuse`, `diff_tiling`, `diff_state`, `diff_temporary_storage`) explained.
  - All six TASK-11 node types compared with decode; `every_semantic_node_compared` true; stage kinds match decode ids with sequence rank $T$ vs 1.
  - Unavoidable unique-weight extra is 0; state extra is 0; proposed-boundary activation is $T\times 2355200$ B versus TASK-06 forced $T\times 3123200$ and region-cut $T\times 5847040$.
  - Seven consumer sequences attached; `ideal_byte_sequence_selected` false. Twenty-two fusion, five hoist, two mode, and four representation hypotheses unselected; `fusion_winner_selected` false; `decode_prefill_distinct_views_selected` false; `ledger_open_question_representation_tradeoffs_closed` false; `representation_tradeoffs_unresolved` true.
  - No thread geometry; `thread_geometry_absent` true; `hardware_independent` true; `cuda_mapping_deferred` true; `tile_size_selected` false.
  - Four canonical sentences verbatim; one Mermaid flowchart contains the required IDs; caption contains `HYPOTHESIS` and `unresolved`.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - `g`/`z` are not stage-cut I/O; state writes not optional; incoming zeros; `h_64` cannot be hidden from one primary consumer.
  - MAC at $T=1$ equals decode; C/S physical read at $T=1$ is 0, not a continuing-decode C/S read.
  - No kernel/layout/fusion/allocation/view **winners**; no Quartz/llama.cpp; GGUF is not the schedule; no payload re-stream; no `plan.md` or ledger edit.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_prefill_plan.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_prefill_plan.py
python3 scripts/check_prefill_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_prefill_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --prefill-plan docs/architecture/prefill-plan.md
```

- Candidate quality: not required — no model execution or NLL; this increment is prefill-schedule documentation. Packing/fusion/view **usefulness** is HYPOTHESIS prose, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/prefill-plan.md
python3 -m py_compile scripts/check_prefill_plan.py
python3 scripts/check_prefill_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --prefill-plan docs/architecture/prefill-plan.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run other `scripts/check_*.py` as a requirement of this task (verifier may spot-check TASK-06/09/12/13 markdown integers independently).

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/prefill-plan.md` (create)
  - `scripts/check_prefill_plan.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-06/09/11/12/13 deliverables unchanged)
- Definition of done: prefill-plan document published with locked nine stage kinds (135 instances), layer-serial many-token order from zeros, five fundamental differences versus decode, every semantic node compared with decode, unavoidable vs proposed-boundary traffic identities, and packing/fusion/hoist/mode/representation hypotheses left unselected without thread geometry; representation/view tradeoffs **remain unresolved**; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-14 completion checkboxes can be marked at delivery **with the open question still recorded as unresolved**; fusion, packing, layout, distinct views, and CUDA mapping remain open except for the selected layer-serial prefill order.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T15:20:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-14.md`. Coupled IDs `none`. Document structure (12 headings), four canonical sentences, nine stage kinds (multiplicities $1+64+64+1+1+1+1+1+1=135$), layer-serial order over $T$, 10 ready constraints, five fundamental differences, 6/6 node comparison, per-stage loads/state/visibility/reuse, unavoidable unique-weight/state extra 0 plus T-scaled stage-cut activation $T\times 2355200$ B vs TASK-06 forced $T\times 3123200$ / region-cut $T\times 5847040$, 7 unselected sequences, 22 unselected fusion hypotheses, 5 unselected hoist + 2 unselected mode + 4 unselected representation hypotheses, stdlib checker `scripts/check_prefill_plan.py`, JSON schema, and acceptance commands are closed. Ledger open question **kept unresolved** (`ledger_open_question_representation_tradeoffs_closed` false; `decode_prefill_distinct_views_selected` false). `docs/architecture/prefill-plan.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit. No implementation.
- Performance evidence applied: N/A — semantic-schedule documentation; packing/fusion/view usefulness is hypothesis, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T15:28:37Z`; `telemetry_unavailable`
- Changes:
  - Created `scripts/check_prefill_plan.py` (stdlib checker: `--json` and `--prefill-plan`).
  - Created `docs/architecture/prefill-plan.md` (twelve locked headings, four canonical sentences, nine stage kinds summing to 135 instances, layer-serial order over $T$, five fundamental differences, 6/6 node comparison, per-stage loads/state/visibility/reuse, unavoidable vs T-scaled stage-cut traffic, 7 sequences + 22 fusion + 5 hoist + 2 mode + 4 representation hypotheses all unselected, one Mermaid flowchart, JSON fence matching live `--json`).
  - Did not edit `plan.md`, TASK-06/09/11/12/13 deliverables, or the ledger (planning already marked TASK-14 `IN PROGRESS`).
  - T=4096 $S$-channel products follow TASK-06 identity `s_bytes_all × T` = `618475290624` (write `622787035136`, physical read `1214373888000`, S physical `618324295680`). Dossier draft arrays `618471290880` / `622783035392` / `618320295936` / `1214369888256` are inconsistent with `s_bytes_all=150994944`; earlier TASK-06 integer wins. Forced-language T=4096 is `2593792×4096=10624172032`.
- Commands:
  - `python3 -m py_compile scripts/check_prefill_plan.py` — exit 0
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; `n_stage_kinds` 9, `n_stage_instances` 135, `n_fundamental_differences` 5, `every_semantic_node_compared` true, `decode_prefill_distinct_views_selected` false, `ledger_open_question_representation_tradeoffs_closed` false, `weight_boundary_added_unique_bytes` 0, `state_boundary_added_bytes` 0, `act_stage_cut_prefill_complete_at_example_T` `[2355200, 9646899200]`, `mac_prefill_equals_decode_at_T1` true, `state_read_prefill_equals_decode_at_T1` false
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md` — exit 0
  - `test -f docs/architecture/prefill-plan.md` — exit 0
- Outcome: Implementation complete. Representation/view hypotheses remain unselected. No commit.

### Documentation

- Agent/model: `composer-2.5` (documentation subagent; parent/inherit mapping)
- Changes and evidence:
  - `docs/architecture/prefill-plan.md` — mechanical pass only. Draft-status banner (`unverified`) already present in TASK-07 / [`numerical-sensitivity.md`](../numerical-sensitivity.md) / [`semantic-graph.md`](../semantic-graph.md) / [`decode-plan.md`](../decode-plan.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`); left unchanged. Authority table already cross-links this dossier, [`work-and-traffic.md`](../work-and-traffic.md) (TASK-06), [`runtime-format-design.md`](../runtime-format-design.md) (TASK-09), [`semantic-graph.md`](../semantic-graph.md) (TASK-11), [`materialization-and-fusion.md`](../materialization-and-fusion.md) (TASK-12), [`decode-plan.md`](../decode-plan.md) (TASK-13, comparison only), [`model-inventory.md`](../model-inventory.md) (TASK-01) via those docs, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_prefill_plan.py`](../../../scripts/check_prefill_plan.py), and plan evidence policy in [`plan.md`](../plan.md). Twelve required `##` headings and the JSON fence left unchanged. No locked integers, canonical sentences, stage-kind ids, sequence ids, fusion ids, representation ids, or Mermaid node IDs edited. Representation/view open question not closed (`ledger_open_question_representation_tradeoffs_closed` false; `decode_prefill_distinct_views_selected` false). Frozen TASK-06/09/11/12/13 artifacts and `plan.md` not edited. `task_ledger.md` not edited (delivery stage).
- Commands:
  - `test -f docs/architecture/prefill-plan.md` — pass (exit 0).
  - `python3 -m py_compile scripts/check_prefill_plan.py` — pass (exit 0).
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 205 keys; `n_stage_kinds` 9; `n_stage_instances` 135; `n_fundamental_differences` 5; `every_semantic_node_compared` true; `decode_prefill_distinct_views_selected` false; `ledger_open_question_representation_tradeoffs_closed` false; `representation_tradeoffs_unresolved` true; `weight_boundary_added_unique_bytes` 0; `state_boundary_added_bytes` 0; `thread_geometry_absent` true; `hardware_independent` true; `fusion_winner_selected` false; `n_fusion_hypotheses_selected` 0; `n_representation_hypotheses_selected` 0; JSON fence source unchanged).
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md` — pass (exit 0; twelve headings, JSON fence, one flowchart, four canonical sentences, locked ids/integers, `HYPOTHESIS` / `hardware-independent` / `unresolved`; banner did not break the check).
  - Independent JSON fence spot-check (live `--json` object equals fenced JSON in `prefill-plan.md`, 205 keys byte-for-byte after parse) — pass.
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T15:29:39Z`; `telemetry_unavailable`

### Verification

- Attempt: 1 (first pass)
- Agent/model: `composer-2.5` (verifier subagent; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T15:32:00Z`; `telemetry_unavailable`
- Commands:
  - `test -f docs/architecture/prefill-plan.md` — exit 0
  - `python3 -m py_compile scripts/check_prefill_plan.py` — exit 0
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0 (`n_stage_kinds` 9; `n_stage_instances` 135; `n_fundamental_differences` 5; `every_semantic_node_compared` true; `decode_prefill_distinct_views_selected` false; `ledger_open_question_representation_tradeoffs_closed` false; `representation_tradeoffs_unresolved` true; `weight_boundary_added_unique_bytes` 0; `state_boundary_added_bytes` 0; `act_stage_cut_prefill_complete_at_example_T` `[2355200, 9646899200]`; `thread_geometry_absent` true; `hardware_independent` true; `fusion_winner_selected` false; `n_fusion_hypotheses_selected` 0; `n_representation_hypotheses_selected` 0; `mac_prefill_equals_decode_at_T1` true; `state_read_prefill_equals_decode_at_T1` false)
  - `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md` — exit 0 (twelve headings, four canonical sentences, one Mermaid flowchart with required IDs, JSON fence, `HYPOTHESIS` / `unresolved` caption)
- Independent checks (from sitting `text_config`, not JSON echo):
  - $W_\text{complete}(T)=TC+AT(T+1)/2$: $T=1$ → `27433447424`; $T=4096$ → `114119319486464` (matches checker and `work-and-traffic.md`)
  - T-scaled stage-cut activation: `2355200×T` → `[2355200, 9646899200]`; forced `3123200×T`; region-cut `5847040×T`
  - Triangular KV read `69632·T(T-1)/2`; KV write `69632T`; $S$ write `150994944×T`; $S$ physical read `150994944×(T-1)` at $T=4096$ → `618324295680`
  - Unique+gather complete at $T=4096$: `52098598912 + 83886080 = 52182484992`
- Spot-checks:
  - TASK-06 integers (`27433238528`, `208896`, `52098598912`, `150994944`, `3123200`, `5847040`) present in `work-and-traffic.md`
  - Seven `consumer_sequence_ids` match `runtime-format-design.md` JSON fence
  - Twenty-two `fusion_hypothesis_ids` match `materialization-and-fusion.md` JSON fence
  - Nine `stage_kind_ids` match `decode-plan.md` (comparison only)
- Scope review:
  - `docs/architecture/plan.md` unchanged
  - Frozen TASK-06/09/11/12/13 deliverables unchanged
  - `task_ledger.md` has only expected `TODO` → `IN PROGRESS` admission edit (no delivery completion)
  - No Quartz/llama.cpp schedule claims; GGUF not used as schedule authority; no payload re-stream; `g`/`z` not stage-cut I/O; all fusion/hoist/mode/representation hypotheses unselected; representation open question not closed
  - JSON fence equals live `--json` (205 keys, byte-for-byte after parse)
- Verdict: **PASS** (first pass; no repair required)

### Retries and escalation

none.

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-14 only; coupled IDs `none`
- Outcome: TASK-14 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T15:30:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/prefill-plan.md` (twelve locked headings, four canonical sentences, nine stage kinds / 135 instances, layer-serial many-token order, five fundamental differences versus decode, 6/6 node comparison, per-stage loads/state/visibility/reuse, unavoidable vs stage-cut traffic identities, seven consumer sequences + 22 fusion + 5 hoist + 2 mode + 4 representation hypotheses unselected, one Mermaid flowchart, JSON fence); `scripts/check_prefill_plan.py` stdlib checker; independent JSON-fence equality and TASK-06 arithmetic confirmed; frozen upstream docs unchanged; fusion, packing, layout, distinct views, and CUDA mapping remain open; ledger open question kept unresolved (`decode_prefill_distinct_views_selected` false)
- Candidate measured delta: N/A — prefill-schedule documentation
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete for this increment (no GPU/quality gates)
- Throughput delta: N/A — TASK-14 does not execute or time the model
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification attempt 1 pass)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: none
