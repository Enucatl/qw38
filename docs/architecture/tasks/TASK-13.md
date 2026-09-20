# TASK-13 — Derive a clean-sheet decode execution plan

## Control

- Primary ID: `TASK-13`
- Coupled IDs: `none`
- Dependencies: `TASK-06`, `TASK-09`, `TASK-11`, `TASK-12` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Describe loads, state reads/writes, visibility boundaries, and reuse per stage; Separate unavoidable traffic from proposed-boundary traffic; Identify packing and fusion hypotheses without thread geometry.

## Goal and boundaries

Produce `docs/architecture/decode-plan.md` as the Phase 1 **hardware-independent decode semantic schedule** for one-token inference on the Qwen3.8-27B language+MTP map. Select **one serial order** of the existing TASK-11 node instances (135 complete). Per compact stage kind: name weight loads, state reads/writes, visibility boundaries, and reuse. Separate TASK-06 **unavoidable** unique-weight / state / forced-activation minima from this schedule’s **proposed-boundary** stage-cut channel. Attach TASK-09 consumer sequences and TASK-12 fusion hypotheses as **unselected** experiment attachments. Close the ledger open question (boundary-added traffic relative to mathematical minimum traffic) with **DERIVED** identities on matching channels, not by selecting a fusion, packing sequence, layout, or CUDA mapping.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Work/traffic minima and decode \(W=C+AT\) come from `docs/architecture/work-and-traffic.md` (TASK-06). Artifact capabilities and the seven consumer sequences come from `docs/architecture/runtime-format-design.md` (TASK-09); do not close portable-versus-specialized or ideal-sequence. Node types, instance counts, 14 sync edges, and mixer xor come from `docs/architecture/semantic-graph.md` (TASK-11). Physical-need classes and the 22 fusion hypotheses come from `docs/architecture/materialization-and-fusion.md` (TASK-12). Do not add node types, catalog IDs, or sync edges as **required** contracts.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established), `DERIVED` (serial order from the DAG, stage multiplicities, stage-cut byte identities, cited TASK-06/04 integers), or `HYPOTHESIS` (hoist/overlap usefulness, second \(W_\text{lm}\) read, extra `h_64` consumer read, every packing/fusion **usefulness**). `UNKNOWN` only for vision-encoder internals deferred here. No `MEASURED` tok/s or NLL. No selected fusion, packing winner, tile, or kernel.
  - GitHub Markdown math. Cite TASK-02 equation tags via TASK-11 contracts, TASK-06 MAC/byte integers, TASK-09 sequence ids, TASK-11 node types / sync-edge ids, TASK-12 hypothesis ids. Do not rewrite forward math, recopy TASK-06 MAC tables as a new work study, recopy TASK-09 packing illustrations as a new format study, or recopy TASK-12’s 52-ID physical-need table as a new materialization study.
  - Allowed evidence: TASK-06 work/traffic, TASK-09 runtime-format, TASK-11 semantic-graph, TASK-12 materialization/fusion, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier. TASK-03/04 integers already cited by those documents may be **cited** through them. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No TASK-16 occupancy/fusion-vs-occupancy algebra (not a dependency; this document is hardware-independent). No TASK-14 prefill schedule, TASK-15 tiles, or TASK-17 CUDA mappings.
  - Hardware-independent: stage kinds, catalog IDs, node types, sync-edge ids, byte identities. No thread geometry, warps, SMs, CUDA dtypes, kernel names, streams, shared-memory tiles, or sitting-GPU numbers.
- Non-goals:
  - No fusion, split, reuse, or packing **winner**. Listing a hypothesis or attaching a TASK-09 sequence is not selecting it. `n_fusion_hypotheses_selected` = 0. `ideal_byte_sequence_selected` false.
  - No prefill schedule (TASK-14). Prefill and decode share **one** graph; this document schedules \(T_\text{new}=1\) with populated incoming \((K,V,C,S)\). Do not compare every node with prefill.
  - No physical layouts (TASK-15) and no CUDA mapping alternatives (TASK-17).
  - No quantization recipes, artifact-boundary winner, or compiler stages (TASK-08/09/10 already published; do not close their open decisions).
  - No quality/NLL experiments (TASK-18) and no tok/s (TASK-19).
  - No new operators, extra catalog IDs, extra required node types, or 64-layer / 135-instance unrolling in diagrams.
  - No sampling softmax over \(V\) (out of scope, TASK-11).
  - No generic GEMM / softmax / RMSNorm kernel-fusion cookbook. Intra-node `fuse_internals` remain TASK-12 hypotheses.
  - No peak-memory claim and no summed CUDA live-set.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–12). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `work-and-traffic.md`, `runtime-format-design.md`, `semantic-graph.md`, `materialization-and-fusion.md`, `dataflow.md`, `lifetime-and-state.md`, `model-semantics.md`, `model-inventory.md`, `cuda-hardware-model.md`, or any TASK-08/10 deliverable.
  - Do not import other `scripts/check_*.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib decode-plan checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places independent decode/prefill execution plans after the Qwen-specific semantic graph / fusion hypotheses and before CUDA mappings.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:89-92` — TASK-11 derives semantic regions; TASK-12 fusion; TASK-13/14 independently schedule decode and prefill; TASK-17 CUDA-maps nodes.
- `docs/architecture/task_ledger.md` TASK-13 row — produces `docs/architecture/decode-plan.md`; purpose is the ideal semantic schedule for one-token inference independent of current engines; open question is boundary-added traffic relative to mathematical minimum traffic (**closed here by DERIVED stage-cut identities versus TASK-06 minima, not by selecting a fusion or packing**); completion is loads/state/visibility/reuse per stage, unavoidable vs proposed-boundary traffic, packing and fusion hypotheses without thread geometry.
- `docs/architecture/task_ledger.md` TASK-06 established results — decode \(W=C+AT\); \(C_\text{complete}=27433238528\), \(A_\text{complete}=208896\); unique non-embed `52098598912`; unique+gather decode complete `52098619392`; state write `152047616`; state read \(69632(T-1)+153944064\); forced activation complete `3123200`; region-cut complete `5847040`; six HYPOTHESIS bottleneck labels.
- `docs/architecture/task_ledger.md` TASK-09 established results — seven consumer sequences without an ideal sequence; `ideal_byte_sequence_selected` false; one compiled artifact for prefill and decode; `seq_gather_row` / `seq_lm_head_full` / `seq_state_s_dense` access identities.
- `docs/architecture/task_ledger.md` TASK-11 established results — six node types; 135 complete instances; mixer xor by `layer_types`; 14 sync edges; `g`/`z` internal; residual add inside mixer/`mlp`; `schedule_selected` false **there** (this task selects the decode serial order).
- `docs/architecture/task_ledger.md` TASK-12 established results — seven physical-need classes; 22 unselected fusion hypotheses; state/output edges nonfusible; `fanout_h64_cannot_hide_from_one_consumer` true.
- `docs/architecture/task_ledger.md` TASK-14/15/17 — consumers of this schedule; do not perform those designs here.
- `docs/architecture/work-and-traffic.md` — decode vs prefill \(T\) convention; unique-weight+gather; state volumes; three activation views; second \(W_\text{lm}\) read is HYPOTHESIS.
- `docs/architecture/runtime-format-design.md` — seven `consumer_sequence_ids`; access classes; state schema required; state payload inclusion unselected.
- `docs/architecture/semantic-graph.md` — six node types; catalog partition; 14 sync edges; per-node I/O/state; mixer xor; live-across internal.
- `docs/architecture/materialization-and-fusion.md` — 22 fusion hypothesis ids; forbidden drops; default tactics; working-set byte identities.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for layer counts, shapes, and cited MAC/byte products. Do not read safetensor payloads.
- `scripts/check_work_and_traffic.py` / `scripts/check_runtime_format_design.py` / `scripts/check_semantic_graph.py` / `scripts/check_materialization_and_fusion.py` — checker-style precedent. TASK-13’s checker is a sibling; do not import them.

## Performance evidence

N/A — decode **semantic-schedule** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Byte and MAC figures are TASK-06 citations plus DERIVED stage-cut arithmetic, not MEASURED traffic. Packing/fusion/hoist usefulness is HYPOTHESIS, not MEASURED. Do not apply the performance-evidence checklist to rank kernels or claim a winning schedule overlap.

- Measurement identity: N/A (no engine binary). TASK-06/09/11/12 integers are cited, not remeasured.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Schedule coverage is 9 stage kinds, 135 instances, 14 sync edges, TASK-06 three traffic channels plus this stage-cut activation channel, 7 consumer sequences attached unselected, 22 fusion hypotheses unselected.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` layer count or cited TASK-06 integer disagrees with TASK-06/11, stop and fail closed (do not invent stages).
- Claim types: serial order and stage-cut bytes `derived`; unique-weight/state extra `derived` (zero); every hoist/overlap/second-read/fusion/packing **usefulness** `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Schedule completeness is 9 kinds summing to 135 instances, unavoidable vs stage-cut split, open question closed as DERIVED identities + remaining-HYPOTHESIS packing/fusion.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for decode-schedule claims

If a node I/O, state kind, catalog ID, layer count, or cited MAC/byte would disagree with TASK-06/09/11/12 or sitting `text_config`, the earlier document / config wins and this one is wrong.

- Prefill and decode share **one** semantic graph (TASK-11). This document schedules decode only: \(T_\text{new}=1\), stored KV length \(T\) after append, incoming \((K,V,C,S)\) **populated** (zeros is prefill, TASK-14).
- Primary schedule **includes MTP** (second embed, `mtp_mix`, one `gated_attn`, one `mlp`, second `lm_head`). Language-only instance counts are secondary. Omitting MTP when only \(\ell^{(0)}\) is required is a TASK-02 algebraic equivalent, not a second schedule and not the primary contract.
- Both current and next `token_id` presentations are **inputs** to the complete map (teacher-forced or otherwise provided). Sampling over \(V\) is out of scope. Do not invent a catalog ID for the next token; two presentations of `token_id`.
- Algebraic equivalents (chunkwise GDN, SDPA, GQA-as-repeat, omitting MTP) stay **inside** the owning node. They are not extra stages. GDN primary stays `(17)`–`(18)`; do not treat chunkwise as zero \(S\) traffic.
- TASK-11 **node types** are the scheduled units. Do not split `gated_attn` / `gated_delta_net` into proj/core/out as required stages. `g`/`z` stay internal unless a TASK-12 split hypothesis is tried.
- Fan-out ≠ must-store and node I/O ≠ must-store still hold. A stage-cut byte identity is not a CUDA store.
- Unique weight bytes are counted **once** per complete decode. Per-stage “loads” name **which** unique bytes that stage consumes, not a 64× restream of the model.
- Do not inspect Quartz or llama.cpp to “confirm” schedules or kernels.
- Do not select a fusion, packing sequence, tile, or CUDA mapping. Do not import TASK-16 occupancy algebra.
- Select **one** serial stage-kind order. Hoist and fan-out overlap are HYPOTHESIS flexibilities, not a second selected schedule.

### Deliverable structure (`docs/architecture/decode-plan.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Hoist/packing/fusion **usefulness** cells are `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, work/traffic, runtime-format, semantic-graph, materialization-and-fusion, inventory via those docs, config, checker; evidence labels; in-scope (language+MTP one-token serial schedule + traffic split + unselected packing/fusion attachments) vs deferred (vision encoder; TASK-14 prefill; TASK-15 layouts; TASK-17 CUDA). State that the document specifies a **hardware-independent semantic schedule**, not kernels.
2. **Decode schedule convention** — the four canonical sentences (exact text below); what a stage is; serial vs hoist; prefill identity deferred.
3. **One-token setting** — \(T\), incoming populated state, MTP, two token presentations, complete vs language-only.
4. **Stage kinds and serial order** — the locked 9 kinds, multiplicities summing to 135, mixer xor, serial order, ready-set constraints. This heading **starts** the ledger completion “per stage.”
5. **Per-stage loads, state, visibility, and reuse** — one table covering all 9 kinds (loads / state R/W / visibility / reuse). Completes the per-stage criterion.
6. **Unavoidable versus proposed-boundary traffic** — TASK-06 minima vs stage-cut identities; weight extra 0; state extra 0; activation three-view comparison; this heading **closes** the ledger open question.
7. **Packing and fusion hypotheses** — seven sequences attached unselected; 22 fusion hypotheses unselected; five hoist hypotheses; one Mermaid summary (diagram 1 of 1).
8. **Work citations and non-decisions** — cited \(C,A,W(T)\); bottleneck labels remain HYPOTHESIS citations; what TASK-14/15/17 own.
9. **Deferred vision** — residual-stream interface only.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Decode schedule convention**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> The decode schedule in this document is a hardware-independent order of TASK-11 nodes for one-token inference, not a CUDA graph, kernel launch sequence, or selected fusion.

> Unavoidable traffic is the TASK-06 unique-weight, state, and forced-activation minimum; proposed-boundary traffic is this schedule's named stage-cut channel.

> Boundary-added traffic relative to the mathematical minimum is the named stage-cut channel compared with TASK-06 unique-weight, state, and activation views; which packing or fusion hypotheses change that extra remains a HYPOTHESIS.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_schedule` = sentence 2; `canonical_sentence_traffic` = sentence 3; `canonical_sentence_open_question` = sentence 4.

Sentence 4 **closes** the ledger open question in checker-substring form. JSON `ledger_open_question_boundary_traffic_closed` true.

Bullets required under that heading:

- Nine stage kinds are complete for this task; their multiplicities sum to 135 complete node instances (`n_stage_kinds` 9, `n_stage_instances` 135).
- A stage is a named execution of one TASK-11 node type (or mixer xor) with loads, state R/W, visibility I/O, and reuse.
- This document selects one **serial** order (`schedule_serial_selected` true). Hoist and overlap remain HYPOTHESIS (`n_hoist_hypotheses_selected` 0).
- Prefill is not scheduled here (`prefill_schedule_deferred` true). Decode \(T_\text{new}=1\); incoming state is populated (`incoming_state_populated` true).
- Residual add stays inside mixer/`mlp`; `g`/`z` stay internal; RMS stays inside the consumer (TASK-11 locks).
- Unique weights are counted once per complete decode (`weight_unique_counted_once` true).
- Fan-out ≠ must-store and node I/O ≠ must-store still hold.
- Primary schedule includes MTP. Hardware mapping is TASK-17. Fusion winners remain TASK-12 hypotheses. Ideal byte sequence remains TASK-09 open.
- No thread geometry (`thread_geometry_absent` true).

### One-token setting (lock)

Let \(T\) be the stored KV length **after** appending the current token (TASK-04/06). Decode of one new token has incoming KV length \(T-1\) and attention contractions against **length \(T\)**. When \(T=1\), incoming KV is empty but \(C,S\) are still populated on a continuing decode; the **first** generated token after prefill has incoming KV length \(T-1\) from the prompt. This document does not schedule that prefill.

JSON: `T_is_stored_length_after_append` true; `decode_T_new` 1; `example_T` `[1, 4096]`; `incoming_state_populated` true; `primary_includes_mtp` true; `mtp_omission_is_algebraic_equivalent` true (not the primary schedule).

JSON `n_token_presentations` = 2. Current `token_id` feeds `embed_current`; next `token_id` feeds `embed_next`. Both are graph inputs to the complete map. Do not add a catalog ID.

JSON `n_node_instances_complete` = 135; `n_node_instances_language` = 130 (citations of TASK-11). Mixer xor: language layer \(\ell\) uses `gated_attn` iff \(\ell\in\mathcal{L}_\text{full}\) (`full_attention_indices` \(\{3,7,\ldots,63\}\)), else `gated_delta_net`. Then `mlp`. Do not unroll 64 layers beyond this rule. JSON `mixer_xor_by_layer_types` true.

JSON booleans (lock true unless noted):

- `decode_prefill_share_graph` = true (graph shared; **schedules** are independent)
- `residual_add_inside_mixer` = true
- `residual_add_inside_mlp` = true
- `residual_input_live_until_add` = true
- `rms_inside_consumer` = true
- `live_across_are_internal` = true
- `hardware_independent` = true
- `cuda_mapping_deferred` = true
- `thread_geometry_absent` = true
- `prefill_schedule_deferred` = true
- `layout_selected` = false
- `fusion_winner_selected` = false
- `ideal_byte_sequence_selected` = false
- `artifact_boundary_selected` = false
- `schedule_serial_selected` = true
- `gdn_primary_is_recurrent_eq_17` = true
- `chunkwise_not_zero_s_traffic` = true
- `state_write_not_optional` = true
- `fanout_h64_cannot_hide_from_one_consumer` = true
- `weight_unique_counted_once` = true
- `weight_second_w_lm_read_is_hypothesis` = true
- `activation_dtype_decided` = false
- `vision_interface_is_not_a_node` = true
- `ledger_open_question_boundary_traffic_closed` = true

### Stage kinds and serial order (lock)

JSON array `stage_kind_ids` in this exact order (9 ids). JSON `n_stage_kinds` = 9. Parallel `stage_multiplicities`, `stage_node_types`, `stage_primary_sequence_ids`.

| id | Multiplicity | Node type | Serial rank |
| --- | ---: | --- | ---: |
| `embed_current` | 1 | `embed` | 1 |
| `language_mixer` | 64 | `mixer_xor` (`gated_attn` iff \(\ell\in\mathcal{L}_\text{full}\), else `gated_delta_net`) | 2 (loop with next) |
| `language_mlp` | 64 | `mlp` | 2 (after matching mixer) |
| `lm_head_primary` | 1 | `lm_head` | 3 |
| `embed_next` | 1 | `embed` | 4 |
| `mtp_mix` | 1 | `mtp_mix` | 5 |
| `mtp_mixer` | 1 | `gated_attn` | 6 |
| `mtp_mlp` | 1 | `mlp` | 7 |
| `lm_head_mtp` | 1 | `lm_head` | 8 |

JSON `stage_multiplicities` `[1,64,64,1,1,1,1,1,1]`. Sum 135. JSON `n_stage_instances` = 135. JSON `stage_node_types` `["embed","mixer_xor","mlp","lm_head","embed","mtp_mix","gated_attn","mlp","lm_head"]`. `mixer_xor` is a schedule abbreviation, not a seventh TASK-11 node type.

Instance-count cross-check (required prose + checker): `language_mixer` contributes 16 `gated_attn` + 48 `gated_delta_net`; plus `mtp_mixer` → 17 `gated_attn`. `language_mlp` + `mtp_mlp` → 65 `mlp`. Two `embed`, two `lm_head`, one `mtp_mix`. Matches TASK-11.

**Serial order** (selected; one complete decode):

1. `embed_current`
2. For \(\ell=0,\ldots,63\): `language_mixer`[\(\ell\)] then `language_mlp`[\(\ell\)]
3. `lm_head_primary`
4. `embed_next`
5. `mtp_mix`
6. `mtp_mixer`
7. `mtp_mlp`
8. `lm_head_mtp`

JSON `serial_stage_kind_order` equals `stage_kind_ids` (the loop is implied by multiplicities; do not expand to 135 ids in JSON).

**Ready-set constraints** (DERIVED from TASK-11 edges; not CUDA overlap):

JSON array `ready_constraint_ids` in this exact order (10 ids). JSON `n_ready_constraints` = 10.

| id | Stage | Ready after |
| --- | --- | --- |
| `ready_embed_current` | `embed_current` | current `token_id` present |
| `ready_language_mixer_0` | `language_mixer`[0] | `embed_current` (`identity_e_h0`) |
| `ready_language_mlp` | `language_mlp`[\(\ell\)] | matching `language_mixer`[\(\ell\)] (`residual_h_mid`) |
| `ready_language_mixer_next` | `language_mixer`[\(\ell>0\)] | `language_mlp`[\(\ell-1\)] (`residual_h`) |
| `ready_lm_head_primary` | `lm_head_primary` | `language_mlp`[63] (`fanout_h64`) |
| `ready_embed_next` | `embed_next` | next `token_id` present (no language-stack data edge) |
| `ready_mtp_mix` | `mtp_mix` | `language_mlp`[63] **and** `embed_next` |
| `ready_mtp_mixer` | `mtp_mixer` | `mtp_mix` (`mtp_u_to_block`) |
| `ready_mtp_mlp` | `mtp_mlp` | `mtp_mixer` (`residual_h_mid`) |
| `ready_lm_head_mtp` | `lm_head_mtp` | `mtp_mlp` (`h_mtp_to_logits`) |

Serial places `embed_next` after `lm_head_primary` to keep the MTP suffix contiguous. `ready_embed_next` is earlier; hoisting is HYPOTHESIS (below), not a second selected order.

### Per-stage loads, state, visibility, and reuse (lock; heading 5)

Table columns, in this order: `Stage` | `Loads` | `State read` | `State write` | `Visibility in` | `Visibility out` | `Reuse`

Implementation copies these rows; do not add/remove stage kinds.

| Stage | Loads | State read | State write | Visibility in | Visibility out | Reuse |
| --- | --- | --- | --- | --- | --- | --- |
| `embed_current` | One row of shared \(E\) (10240 B gather; table not streamed) | none | none | current `token_id` | `e` identified as \(h^{(0)}\) | `shared_E`; gather vs full table |
| `language_mixer` | Mixer unique weights for that layer (self-attn **or** linear-attn family; counted once in the unique-weight total). Access `dense_gemm`; GDN also `depthwise_conv` | `gated_attn`: \(K,V\) of length \(T-1\) (4096 B/token/instance). `gated_delta_net`: \(C\) 61440 B and \(S\) 3145728 B per instance | `gated_attn`: append \(K,V\) 4096 B. `gated_delta_net`: write \(C\) and \(S_t\) | `h` (live until Mix add) | `h_mid` | Residual `h` until add; internals default inside (`g` live-across internal; `k_rope`/`v_full`/`qkv` reuse_or_recompute HYPOTHESIS; `k_hat` intra-equation) |
| `language_mlp` | MLP unique weights for that layer (counted once in the unique-weight total). Access `dense_gemm` | none | none | `h_mid` (live until MLP add) | next `h`, or `h_64` at \(\ell=63\) | Residual `h_mid` until add; `h_post`/`swiglu` fuse_or_recompute HYPOTHESIS |
| `lm_head_primary` | Shared \(W_\text{lm}\) 2542796800 B (`seq_lm_head_full`) plus final RMS gamma | none | none | `h_64` | `logits_0` | `shared_W_lm`; `h_64` also consumed by `mtp_mix` |
| `embed_next` | One row of shared \(E\) (10240 B gather) | none | none | next `token_id` | `e_next` | `shared_E` (same payload as `embed_current`) |
| `mtp_mix` | `mtp.fc` contraction weights. Access `dense_gemm` | none | none | `h_64`, `e_next` | `mtp_u` as residual `h` | `h_64` fan-out reuse; `mtp_cat` split HYPOTHESIS |
| `mtp_mixer` | MTP `gated_attn` unique weights. Access `dense_gemm` | MTP \(K,V\) of length \(T-1\) | MTP \(K,V\) append 4096 B | `mtp_u` as `h` | `h_mid` | Same gated-attn reuse as language full layers |
| `mtp_mlp` | MTP MLP unique weights. Access `dense_gemm` | none | none | `h_mid` | `h_mtp` | Same MLP reuse as language |
| `lm_head_mtp` | Shared \(W_\text{lm}\) (second physical read HYPOTHESIS, not unique bytes) | none | none | `h_mtp` | `logits_1` | `shared_W_lm` |

JSON `stage_visibility_in` / `stage_visibility_out` keyed in `stage_kind_ids` order:

- `embed_current`: in `["token_id"]`, out `["e"]`
- `language_mixer`: in `["h"]`, out `["h_mid"]`
- `language_mlp`: in `["h_mid"]`, out `["h"]` (last instance identified as `h_64` in prose / `fanout_h64`)
- `lm_head_primary`: in `["h_64"]`, out `["logits_0"]`
- `embed_next`: in `["token_id"]`, out `["e_next"]`
- `mtp_mix`: in `["h_64","e_next"]`, out `["mtp_u"]`
- `mtp_mixer`: in `["h"]`, out `["h_mid"]`
- `mtp_mlp`: in `["h_mid"]`, out `["h_mtp"]`
- `lm_head_mtp`: in `["h_mtp"]`, out `["logits_1"]`

JSON `stage_state_read` / `stage_state_write`:

- mixers as TASK-11: `language_mixer` xor `["K_state","V_state"]` or `["C_state","S"]`; `mtp_mixer` `["K_state","V_state"]`; all other kinds empty arrays.
- JSON `language_mixer_state_is_xor` true.

JSON `stage_primary_sequence_ids` in this exact order (9 ids):

`seq_gather_row`, `seq_gemm_codes_then_scales`, `seq_gemm_codes_then_scales`, `seq_lm_head_full`, `seq_gather_row`, `seq_gemm_codes_then_scales`, `seq_gemm_codes_then_scales`, `seq_gemm_codes_then_scales`, `seq_lm_head_full`

JSON `gated_delta_net_state_sequence_id` = `seq_state_s_dense`. Attaching a sequence is **not** selecting an ideal byte sequence.

State volume citations (do not re-derive; checker recomputes from the same identities as TASK-06):

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
| `decode_read_kv_bytes_coeff_Tm1` | 69632 |

Prose required: omitting a KV/\(C\)/\(S\) write changes the map (`state_write_not_optional`). Chunkwise GDN is not zero \(S\) traffic.

### Unavoidable versus proposed-boundary traffic (lock; heading 6; closes open question)

Three channels. **Unavoidable** = TASK-06 mathematical minimum for one complete decode. **Proposed-boundary** = this serial schedule’s named stage-cut transfers. Do not recopy TASK-06 family tables.

**Weight**

| Item | Bytes | Label |
| --- | ---: | --- |
| Unique non-embed language+MTP | 52098598912 | unavoidable (cite TASK-06) |
| Gather complete (`e_t` + `e_{t+1}`) | 20480 | unavoidable |
| Unique+gather decode complete | 52098619392 | unavoidable |
| Unique extra vs TASK-06 | 0 | DERIVED |
| Second physical \(W_\text{lm}\) read | 2542796800 | HYPOTHESIS extra, not unique |

JSON: `weight_bytes_unique_non_embed` 52098598912, `weight_gather_bytes_decode_complete` 20480, `weight_unique_plus_gather_decode_complete` 52098619392, `weight_boundary_added_unique_bytes` 0, `weight_second_w_lm_read_bytes` 2542796800.

**State**

| Item | Bytes | Label |
| --- | --- | --- |
| Write all \(K,V,C,S\) | 152047616 | unavoidable |
| Read | \(69632(T-1)+153944064\) | unavoidable |
| Read at \(T=1\) | 153944064 | DERIVED |
| Read at \(T=4096\) | 439087104 | DERIVED |
| Extra vs TASK-06 | 0 | DERIVED |

JSON arrays `decode_read_bytes_at_example_T`, `storage_bytes_at_example_T` follow `example_T`. Storage \(B_\text{store}(T)=69632T+153944064\) (154013696 at \(T=1\), 439156736 at \(T=4096\)). JSON `state_boundary_added_bytes` 0.

**Activation — three views plus this stage-cut**

Cite TASK-06: `act_forced_decode_complete_bytes` 3123200; `act_region_cut_decode_complete_bytes` 5847040. Neither is a CUDA live-set.

Stage-cut unique bytes (this schedule; each named data edge once; identification applied so `e` is the first mixer input and `mtp_u` is the MTP mixer input; `h_64` counted once, not twice):

JSON `n_h_mid_crossings` = 65 (64 language + 1 MTP).
JSON `n_h_interlayer_crossings` = 63.
JSON `n_identity_e_h0_crossings` = 1.
JSON `n_fanout_h64_crossings` = 1.
JSON `n_embed_e_next_crossings` = 1.
JSON `n_mtp_u_crossings` = 1.
JSON `n_h_mtp_crossings` = 1.
JSON `n_residual_stage_crossings` = 133. Sum: 1 (`e` as \(h^{(0)}\)) + 65 (`h_mid`) + 63 (inter-layer `h`) + 1 (`h_64`) + 1 (`e_next`) + 1 (`mtp_u`) + 1 (`h_mtp`). Do **not** add a second MTP `h_mid` line; it is already inside `n_h_mid_crossings`.

JSON `residual_bytes` 10240. JSON `act_residual_stage_cut_bytes` = \(133\times 10240\) = 1361920.
JSON `logits_bytes` 496640. JSON `n_logit_outputs` = 2. JSON `act_logits_stage_cut_bytes` = 993280.
JSON `act_stage_cut_decode_complete_bytes` = 2355200.

Checker: `n_residual_stage_crossings == n_identity_e_h0_crossings + n_h_mid_crossings + n_h_interlayer_crossings + n_fanout_h64_crossings + n_embed_e_next_crossings + n_mtp_u_crossings + n_h_mtp_crossings`.
Checker: `act_residual_stage_cut_bytes == n_residual_stage_crossings * residual_bytes`.
Checker: `act_stage_cut_decode_complete_bytes == act_residual_stage_cut_bytes + act_logits_stage_cut_bytes`.

`g`/`z` stay inside mixers: JSON `act_gz_local_bytes` = \(16\times 12288 + 48\times 12288 + 12288\) = 798720 (language `g` + language `z` + MTP `g`). These are **not** stage-cut bytes.

JSON `act_boundary_added_vs_forced` = \(2355200-3123200\) = \(-768000\).
JSON `act_boundary_added_vs_region_cut` = \(2355200-5847040\) = \(-3491840\).

Negative values are DERIVED identities, not “the schedule is cheaper than liveness.” Forced includes live-across `g`/`z` as mathematical must-survive; region-cut includes intra-node internals (`h_tilde`, `mix_*`, `h_post`, …). The schedule keeps those **inside** stages. Proposed-boundary traffic is the **named stage-cut channel** 2355200 B, not a claim that `g`/`z` disappeared.

JSON `act_h64_second_consumer_bytes` 10240 — extra physical read of `h_64` by the second of (`lm_head_primary`, `mtp_mix`) is HYPOTHESIS (`act_h64_second_consumer_is_hypothesis` true). Unique stage-cut counts `h_64` once.

JSON `act_h_mid_stage_cut_bytes` = \(65\times 10240\) = 665600. `fuse_across_residual_h_mid` would remove this as **inter-stage** I/O (TASK-12 extra_sync \(-1\)) while keeping the add in a local working set. Usefulness HYPOTHESIS. Do not apply that fusion here.

Closing paragraph (required): this heading **closes** the ledger open question. Unavoidable unique-weight extra is 0. Unavoidable state extra is 0. Proposed-boundary activation is the stage-cut identity 2355200 B, compared with TASK-06 forced 3123200 B and region-cut 5847040 B. A second \(W_\text{lm}\) read, an extra `h_64` consumer read, hoist/overlap, and every TASK-12 fusion/split change to those extras remain HYPOTHESIS. This document does not authorize a merge, a split, a packing winner, or a CUDA mapping.

### Packing and fusion hypotheses (lock; heading 7)

**Packing (TASK-09; none selected).** JSON `consumer_sequence_ids` copied from TASK-09 in TASK-09 order (7 ids). JSON `n_consumer_sequences` = 7. JSON `ideal_byte_sequence_selected` false.

Attach sequences to stages via `stage_primary_sequence_ids` plus `gated_delta_net_state_sequence_id`. Also name `seq_gemm_interleaved_group`, `seq_outlier_extra`, and `seq_specialized_tile` as **unselected alternatives** that the format must be able to express. Do not rank by wall time. Do not close `artifact_boundary`. Layout of `seq_specialized_tile` remains TASK-15.

**Fusion (TASK-12; none selected).** JSON `fusion_hypothesis_ids` copied from TASK-12 in TASK-12 order (22 ids). JSON `n_fusion_hypotheses` = 22. JSON `n_fusion_hypotheses_selected` = 0. JSON object `fusion_selected` all false. JSON `fusion_winner_selected` false. JSON `fusion_usefulness_label` exactly `HYPOTHESIS`.

Default schedule **keeps TASK-11 node cuts** as stages. That is compatible with internals staying inside a node; it is **not** a selected `fuse_internals` winner. `split_*` would add a hypothesized sub-stage. `fuse_across_*` would merge adjacent stage kinds. `fuse_across_fanout_h64` cannot hide `h_64` from one of its two primary consumers (TASK-12 lock). State and output edges have no drop hypothesis.

**Hoist / overlap (this task; none selected as required).** JSON array `hoist_hypothesis_ids` in this exact order (5 ids). JSON `n_hoist_hypotheses` = 5. JSON `n_hoist_hypotheses_selected` = 0. JSON object `hoist_selected` all false. Usefulness HYPOTHESIS.

| id | Meaning |
| --- | --- |
| `hoist_embed_next` | Issue `embed_next` as soon as next `token_id` is present (serial places it after `lm_head_primary`) |
| `overlap_fanout_h64` | `lm_head_primary` and `mtp_mix` are both ready after `h_64` |
| `reuse_E` | Two gathers of one \(E\) payload |
| `reuse_W_lm` | Two `lm_head` instances of one \(W_\text{lm}\) payload; second physical read is HYPOTHESIS extra bytes |
| `reuse_h64` | Two consumers of one `h_64`; extra physical read is HYPOTHESIS 10240 B |

JSON `hoist_usefulness_label` exactly `HYPOTHESIS`.

### Work citations and non-decisions (lock; heading 8)

Cite; do not recopy TASK-06 symbolic tables. Checker **recomputes** \(C,A,W(T)\) from `text_config` with the same identities as TASK-06.

JSON: `mac_C_complete` 27433238528, `mac_A_complete` 208896, `mac_decode_complete_at_example_T` `[27433447424, 28288876544]`. Language secondary: `mac_C_language` 25737166848, `mac_A_language` 196608. Identity \(W=C+AT\).

JSON `bottleneck_labels` copied from TASK-06: `weight_memory`, `vocab_memory`, `state_memory`, `kv_memory`, `quadratic_attn`, `compute`. Restating a label here is a **citation**, still HYPOTHESIS. JSON `n_bottleneck_labels` = 6. Decode does not use `quadratic_attn` as a one-token class (that label is prefill); still copy the six-id list, do not rank.

Non-decisions (prose required): TASK-14 owns prefill order of the **same** nodes and many-token differences. TASK-15 owns tiles. TASK-17 owns CUDA mappings per node type. TASK-09 still owns artifact-boundary and ideal-sequence **selection**. TASK-12 still owns fusion **winners**. This schedule does not change when a TASK-08 recipe is later applied.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 7 (Packing and fusion hypotheses). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers or 135 instances. Annotate mixer xor in prose / a label, not as 64 subgraphs. Caption must contain the word `HYPOTHESIS` (hoist/fusion remain hypotheses).

Required IDs **inside that fence**: `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`, `h`, `h_mid`, `h_64`, `K_state`, `V_state`, `C_state`, `S`, `logits_0`, `logits_1`.

JSON `n_diagrams` is 1. `diagram_ids` is `["embed_current","language_mixer","language_mlp","lm_head_primary","embed_next","mtp_mix","mtp_mixer","mtp_mlp","lm_head_mtp","h","h_mid","h_64","K_state","V_state","C_state","S","logits_0","logits_1"]`.

### Deferred vision

Visual tokens may replace placeholders on the `identity_e_h0` edge (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger decode schedules are **UNKNOWN**. Do not add a vision stage kind. `vision_interface_is_not_a_node` true. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_decode_plan.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, residual/`g`/`z`/`logits` bytes, cited TASK-06 \(C,A,W(T)\), unique-weight+gather, and TASK-04/06 state volumes. Duplicate TASK-11 node-type / sync-edge lists, TASK-09 `consumer_sequence_ids`, and TASK-12 `fusion_hypothesis_ids` as constants; do not import them.

CLI (cwd = repository root):

```text
python3 scripts/check_decode_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_decode_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --decode-plan docs/architecture/decode-plan.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: instance counts, \(C,A,W(T)\), residual/`g`/`z`/`logits` bytes, stage-cut products, cited unique-weight+gather and state volumes. Constant fields: canonical sentences, stage/hoist/fusion/sequence lists, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--decode-plan PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all four canonical sentences verbatim, (5) every `stage_kind_ids`, `consumer_sequence_ids`, `fusion_hypothesis_ids`, `hoist_hypothesis_ids`, `ready_constraint_ids`, `sync_edge_ids`, `node_type_ids`, `bottleneck_labels` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer below present as a decimal or integer substring, (10) the words `HYPOTHESIS` and `hardware-independent` present, (11) none of the forbidden winner phrases: `selected fusion`, `selected packing`, `selected kernel`, `winning fusion`, `should fuse`, `recommend fusion`, `ideal byte sequence is`, `artifact boundary is`, `CUDA kernel fusion is required`, `thread block`, `warp shuffle`, `Quartz graph`, `llama.cpp graph`, `GGUF is the schedule` (allow the substring only inside `not a selected fusion` / `not a CUDA graph, kernel launch sequence, or selected fusion` / `none is a selected winner`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks cited TASK-06 integers against `work-and-traffic.md` and fusion/sequence ids against TASK-09/12 markdown).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `n_stage_kinds==9`, `n_stage_instances==135`, `sum(stage_multiplicities)==135`
- `stage_multiplicities == [1,64,64,1,1,1,1,1,1]`
- `n_node_instances_complete==135`, `n_node_instances_language==130`
- `n_embed_instances==2`, `n_gated_attn_instances==17`, `n_gated_delta_net_instances==48`, `n_mlp_instances==65`, `n_lm_head_instances==2`, `n_mtp_mix_instances==1`
- `n_ready_constraints==10`, `n_hoist_hypotheses==5`, `n_hoist_hypotheses_selected==0`
- `n_consumer_sequences==7`, `n_fusion_hypotheses==22`, `n_fusion_hypotheses_selected==0`
- `n_sync_edges==14`, `n_diagrams==1`, `n_token_presentations==2`
- `n_residual_stage_crossings==133`, `n_h_mid_crossings==65`, `n_h_interlayer_crossings==63`
- `residual_bytes==10240`, `act_residual_stage_cut_bytes==1361920`
- `logits_bytes==496640`, `act_logits_stage_cut_bytes==993280`
- `act_stage_cut_decode_complete_bytes==2355200`
- `act_forced_decode_complete_bytes==3123200`, `act_region_cut_decode_complete_bytes==5847040`
- `act_boundary_added_vs_forced==-768000`, `act_boundary_added_vs_region_cut==-3491840`
- `act_gz_local_bytes==798720`, `act_h_mid_stage_cut_bytes==665600`
- `weight_bytes_unique_non_embed==52098598912`, `weight_gather_bytes_decode_complete==20480`
- `weight_unique_plus_gather_decode_complete==52098619392`, `weight_boundary_added_unique_bytes==0`
- `decode_write_bytes==152047616`, `state_boundary_added_bytes==0`
- `mac_C_complete==27433238528`, `mac_A_complete==208896`
- `mac_decode_complete_at_example_T == [27433447424, 28288876544]`
- `mac_decode_complete_at_example_T[i] == mac_C_complete + mac_A_complete * example_T[i]`
- `i_mlp_weight_only==1`, `i_lm_head_weight_only==1`, `i_gdn_vs_s_rw==0.75`, `i_attn_core_vs_kv==6`
- `schedule_serial_selected is True`, `fusion_winner_selected is False`
- `ideal_byte_sequence_selected is False`, `thread_geometry_absent is True`, `hardware_independent is True`
- `incoming_state_populated is True`, `prefill_schedule_deferred is True`
- `weight_unique_counted_once is True`, `weight_second_w_lm_read_is_hypothesis is True`
- `live_across_are_internal is True`, `state_write_not_optional is True`
- `fanout_h64_cannot_hide_from_one_consumer is True`, `chunkwise_not_zero_s_traffic is True`
- `ledger_open_question_boundary_traffic_closed is True`
- every `fusion_selected` value is False; every `hoist_selected` value is False
- `stage_kind_ids` equals the locked 9-id list; `fusion_hypothesis_ids` equals the locked 22-id list
- `consumer_sequence_ids` equals the TASK-09 7-id list
- `"g" not in` any `stage_visibility_in` or `stage_visibility_out` value
- `"language_mixer" in stage_kind_ids` and `"seq_lm_head_full" in stage_primary_sequence_ids`

Locked document integers/decimals the `--decode-plan` check must find:

`5120`, `17408`, `248320`, `135`, `130`, `133`, `2355200`, `1361920`, `993280`, `3123200`, `5847040`, `665600`, `798720`, `10240`, `20480`, `496640`, `52098598912`, `52098619392`, `152047616`, `153944064`, `439087104`, `69632`, `150994944`, `2542796800`, `27433238528`, `208896`, `27433447424`, `28288876544`, `104857600`, `118235136`, `267386880`, `1271398400`, `0.75`

(The integer `48` / `16` / `64` / `9` / `22` / `7` / `14` / `65` / `17` will appear from counts and types; requiring the byte/MAC-scale set plus `135` / `133` / `2355200` is the hard check. Do not require `1e-06`.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `n_attn_heads`, `n_kv_heads`, `head_dim`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `bytes_bf16`, `bytes_f32`,

`node_type_ids`, `n_node_types`, `n_embed_instances`, `n_gated_attn_instances`, `n_gated_delta_net_instances`, `n_mlp_instances`, `n_lm_head_instances`, `n_mtp_mix_instances`, `n_node_instances_complete`, `n_node_instances_language`,

`stage_kind_ids`, `n_stage_kinds`, `stage_multiplicities`, `stage_node_types`, `n_stage_instances`, `serial_stage_kind_order`, `stage_primary_sequence_ids`, `gated_delta_net_state_sequence_id`, `stage_visibility_in`, `stage_visibility_out`, `stage_state_read`, `stage_state_write`, `language_mixer_state_is_xor`,

`ready_constraint_ids`, `n_ready_constraints`, `hoist_hypothesis_ids`, `hoist_selected`, `n_hoist_hypotheses`, `n_hoist_hypotheses_selected`, `hoist_usefulness_label`,

`n_token_presentations`, `example_T`, `decode_T_new`, `T_is_stored_length_after_append`, `incoming_state_populated`, `primary_includes_mtp`,

`mac_C_language`, `mac_C_complete`, `mac_A_language`, `mac_A_complete`, `mac_decode_complete_at_example_T`, `mac_full_proj_per_layer`, `mac_lin_token_per_layer`, `mac_mlp_per_layer`, `mac_lm_head`, `mac_mtp_fc`, `mac_gdn_per_layer`, `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_gdn_vs_s_rw`, `i_attn_core_vs_kv`, `bottleneck_labels`, `n_bottleneck_labels`,

`weight_bytes_unique_non_embed`, `weight_gather_bytes_per_row`, `weight_gather_bytes_decode_complete`, `weight_unique_plus_gather_decode_complete`, `weight_boundary_added_unique_bytes`, `weight_second_w_lm_read_bytes`, `weight_bytes_lm_head`,

`kv_bytes_per_full_layer_per_token`, `kv_bytes_all_per_token`, `c_bytes_per_layer`, `c_write_bytes_all`, `c_read_bytes_all`, `s_bytes_per_layer`, `s_bytes_all`, `decode_write_bytes`, `decode_read_fixed_bytes`, `decode_read_kv_bytes_coeff_Tm1`, `decode_read_bytes_at_example_T`, `storage_kv_bytes_coeff_T`, `storage_fixed_bytes`, `storage_bytes_at_example_T`, `state_boundary_added_bytes`,

`residual_bytes`, `g_bytes`, `z_bytes`, `logits_bytes`, `n_logit_outputs`, `n_identity_e_h0_crossings`, `n_h_mid_crossings`, `n_h_interlayer_crossings`, `n_fanout_h64_crossings`, `n_embed_e_next_crossings`, `n_mtp_u_crossings`, `n_h_mtp_crossings`, `n_residual_stage_crossings`, `act_residual_stage_cut_bytes`, `act_logits_stage_cut_bytes`, `act_stage_cut_decode_complete_bytes`, `act_forced_decode_complete_bytes`, `act_region_cut_decode_complete_bytes`, `act_boundary_added_vs_forced`, `act_boundary_added_vs_region_cut`, `act_gz_local_bytes`, `act_h_mid_stage_cut_bytes`, `act_h64_second_consumer_bytes`, `act_h64_second_consumer_is_hypothesis`,

`consumer_sequence_ids`, `n_consumer_sequences`, `fusion_hypothesis_ids`, `n_fusion_hypotheses`, `fusion_selected`, `n_fusion_hypotheses_selected`, `fusion_usefulness_label`, `sync_edge_ids`, `n_sync_edges`,

`decode_prefill_share_graph`, `residual_add_inside_mixer`, `residual_add_inside_mlp`, `residual_input_live_until_add`, `rms_inside_consumer`, `live_across_are_internal`, `hardware_independent`, `cuda_mapping_deferred`, `thread_geometry_absent`, `prefill_schedule_deferred`, `layout_selected`, `fusion_winner_selected`, `ideal_byte_sequence_selected`, `artifact_boundary_selected`, `schedule_serial_selected`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_not_zero_s_traffic`, `state_write_not_optional`, `fanout_h64_cannot_hide_from_one_consumer`, `weight_unique_counted_once`, `weight_second_w_lm_read_is_hypothesis`, `mixer_xor_by_layer_types`, `activation_dtype_decided`, `vision_interface_is_not_a_node`, `mtp_omission_is_algebraic_equivalent`, `ledger_open_question_boundary_traffic_closed`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_schedule`, `canonical_sentence_traffic`, `canonical_sentence_open_question`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. Intensities `i_mlp_weight_only`, `i_lm_head_weight_only`, `i_attn_core_vs_kv` are JSON numbers `1`, `1`, `6`. `i_gdn_vs_s_rw` is JSON number `0.75`. Booleans are JSON booleans. `full_attention_indices` is a JSON array of ints. `act_boundary_added_vs_forced` and `act_boundary_added_vs_region_cut` are JSON ints (negative). `stage_visibility_in`, `stage_visibility_out`, `stage_state_read`, `stage_state_write` are JSON objects keyed in `stage_kind_ids` order. `fusion_selected` and `hoist_selected` are JSON objects with JSON `false` values. `fusion_usefulness_label` and `hoist_usefulness_label` are the JSON string `HYPOTHESIS`. `stage_state_read["language_mixer"]` is `["K_state","V_state","C_state","S"]` as the xor union (prose says xor per layer; JSON stores the union so the checker can require both families without duplicating 64 keys). `stage_state_write["language_mixer"]` same union. `stage_state_read["mtp_mixer"]` `["K_state","V_state"]`.

`node_type_ids` copied from TASK-11: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`. JSON `n_node_types` = 6.

`sync_edge_ids` copied from TASK-11 in TASK-11 order (14 ids).

`consumer_sequence_ids` copied from TASK-09: `seq_gemm_codes_then_scales`, `seq_gemm_interleaved_group`, `seq_gather_row`, `seq_lm_head_full`, `seq_outlier_extra`, `seq_state_s_dense`, `seq_specialized_tile`.

`fusion_hypothesis_ids` copied from TASK-12 (22 ids, TASK-12 order).

Duplicate key `fusion_selected` appears once as the boolean `false` at the boolean block **and** once as the object keyed by hypothesis ids. To avoid JSON key collision, the object is `fusion_selected` and the boolean is `fusion_winner_selected` plus `fusion_selected` **only as the object**. Do **not** emit a second top-level boolean named `fusion_selected`. The boolean block above lists `fusion_selected` as documentation of the lock; the schema key list uses the **object** `fusion_selected` and boolean `fusion_winner_selected` only. Implementation: one key `fusion_selected` (object, all false). Boolean `fusion_winner_selected` false. Checker asserts `fusion_winner_selected is False` and every object value False.

### Stage split

- **Implementation** writes `scripts/check_decode_plan.py` **and** `docs/architecture/decode-plan.md` (nine stage kinds summing to 135 instances, serial order, per-stage loads/state/visibility/reuse, unavoidable vs stage-cut traffic that closes the ledger open question, 7 sequences + 22 fusion hypotheses + 5 hoist hypotheses all unselected, JSON fence). Runs `--json` and `--decode-plan` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-07 / `numerical-sensitivity.md` / `semantic-graph.md` style), Authority table links to this dossier / work-and-traffic / runtime-format / semantic-graph / materialization-and-fusion / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, stage-kind ids, sequence ids, fusion ids, or Mermaid node IDs. Does not edit TASK-06/09/11/12 artifacts.
- **Verification** independently re-runs focused commands, recomputes \(C+AT\), residual/stage-cut bytes, unique+gather, and state volumes from sitting `text_config` (not from JSON echo), spot-checks cited TASK-06 integers against `docs/architecture/work-and-traffic.md` and sequence/fusion id lists against `docs/architecture/runtime-format-design.md` / `docs/architecture/materialization-and-fusion.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-schedule, no winner phrases, no `plan.md` or ledger edit, no payload I/O, that every fusion/hoist remains unselected, that contracts are hardware-independent (no CUDA kernel/thread/SKU decisions, `thread_geometry_absent` true), that `g`/`z` are not stage-cut I/O, that unique-weight and state extras are 0, and that the open question is closed by the stage-cut vs TASK-06 comparison. Prefill, layouts, and CUDA mapping remain open. Ideal byte sequence remains unselected.

- Invariants:
  - Ten level-2 headings in the locked order; four canonical sentences verbatim; 9 stage kinds; 135 instances; 10 ready constraints; 5 hoist hypotheses all unselected; 22 fusion hypotheses all unselected; 7 sequences attached unselected; one Mermaid flowchart with required IDs.
  - Serial order selected; hoist/overlap not required; primary includes MTP; mixer xor by `layer_types`; residual add inside mixer and `mlp`; `g`/`z` internal; unique weights counted once; state writes not optional.
  - GDN numerical definition remains recurrent `(17)`–`(18)`; chunkwise is not zero \(S\) traffic.
  - Unavoidable unique-weight extra 0; state extra 0; stage-cut activation 2355200 B; forced 3123200; region-cut 5847040.
  - Logical values do not imply allocation; packing/fusion usefulness is HYPOTHESIS; no thread geometry.
  - Vision encoder remains unexpanded and is not a stage.
- Rejected alternatives:
  - Unrolling 135 instance ids as the published stage list: rejected; compact 9 kinds + mixer xor match TASK-11’s “do not unroll 64 layers.”
  - One `layer` stage wrapping mixer+MLP as the **required** serial unit: rejected; TASK-11 `cut_mixer_mlp` and TASK-04 `h_mid`; `fuse_across_residual_h_mid` stays an unselected hypothesis.
  - Required proj/core/out or conv/recurrence/out sub-stages: rejected; `g`/`z` remain internal; splits stay TASK-12 hypotheses.
  - Selecting any fusion, packing sequence, or artifact boundary because it “looks faster”: rejected; no MEASURED evidence; TASK-09/12 leave those open.
  - Treating TASK-06 region-cut as this schedule’s I/O: rejected; region-cut includes internals the schedule keeps inside nodes.
  - Treating negative `act_boundary_added_vs_forced` as “g/z deleted”: rejected; they remain local working set.
  - Counting unique MLP/attn weights × 64 as decode traffic: rejected; unique counted once.
  - Streaming the full embed table: rejected; gather only (TASK-06).
  - Second \(W_\text{lm}\) read as DERIVED unique bytes: rejected; TASK-06 HYPOTHESIS.
  - Chunkwise GDN as zero \(S\) traffic or as extra stages: rejected; TASK-02/06/07/11.
  - Separate language-only primary schedule: rejected; complete map includes MTP.
  - Scheduling prefill here or comparing every node with prefill: TASK-14.
  - Importing TASK-16 occupancy / thread geometry: rejected; not a dependency; hardware-independent.
  - Inspecting Quartz or llama.cpp for “real” decode graphs: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–12.
  - Editing frozen TASK-06/09/11/12 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py`.
  - Leaving the ledger open question unclassified: rejected; stage-cut vs TASK-06 identities plus sentence 4 are the completion criterion. Closing it by picking a fusion or packing: also rejected.
  - Multiple selected serial orders: rejected; one serial order is the schedule; hoist is HYPOTHESIS flexibility.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/decode-plan.md` exists and follows the heading list above.
  - Nine stage kinds with multiplicities summing to 135; serial order selected; per-stage loads, state R/W, visibility, and reuse tabulated.
  - Unavoidable unique-weight extra is 0; state extra is 0; proposed-boundary activation is the stage-cut identity 2355200 B versus TASK-06 forced 3123200 and region-cut 5847040; `ledger_open_question_boundary_traffic_closed` true.
  - Seven consumer sequences attached; `ideal_byte_sequence_selected` false. Twenty-two fusion hypotheses and five hoist hypotheses unselected; `fusion_winner_selected` false; `n_hoist_hypotheses_selected` 0.
  - No thread geometry; `thread_geometry_absent` true; `hardware_independent` true; `cuda_mapping_deferred` true.
  - Four canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - `g`/`z` are not stage-cut I/O; state writes not optional; `h_64` cannot be hidden from one primary consumer.
  - No kernel/layout/fusion/allocation **winners**; no Quartz/llama.cpp; GGUF is not the schedule; no payload re-stream; no `plan.md` or ledger edit.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_decode_plan.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_decode_plan.py
python3 scripts/check_decode_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_decode_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --decode-plan docs/architecture/decode-plan.md
```

- Candidate quality: not required — no model execution or NLL; this increment is decode-schedule documentation. Packing/fusion **usefulness** is HYPOTHESIS prose, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/decode-plan.md
python3 -m py_compile scripts/check_decode_plan.py
python3 scripts/check_decode_plan.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --decode-plan docs/architecture/decode-plan.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run other `scripts/check_*.py` as a requirement of this task (verifier may spot-check TASK-06/09/12 markdown integers independently).

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/decode-plan.md` (create)
  - `scripts/check_decode_plan.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-06/09/11/12 deliverables unchanged)
- Definition of done: decode-plan document published with locked nine stage kinds (135 instances), serial one-token order, per-stage loads/state/visibility/reuse, unavoidable vs proposed-boundary traffic identities that close the TASK-13 open question, and packing/fusion/hoist hypotheses left unselected without thread geometry; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-13 completion checkboxes can be marked at delivery; fusion, packing, layout, prefill, and CUDA mapping remain open except for the selected serial decode order.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T14:50:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-13.md`. Coupled IDs `none`. Document structure (10 headings), four canonical sentences, nine stage kinds (multiplicities \(1+64+64+1+1+1+1+1+1=135\)), serial order, 10 ready constraints, per-stage loads/state/visibility/reuse, unavoidable unique-weight/state extra 0 plus stage-cut activation 2355200 B vs TASK-06 forced 3123200 / region-cut 5847040, 7 unselected sequences, 22 unselected fusion hypotheses, 5 unselected hoist hypotheses, stdlib checker `scripts/check_decode_plan.py`, JSON schema, and acceptance commands are closed. Ledger open question closed **by DERIVED stage-cut identities versus TASK-06 minima**, not by selecting a fusion or packing. `docs/architecture/decode-plan.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit.
- Performance evidence applied: N/A — semantic-schedule documentation; packing/fusion/hoist usefulness is hypothesis, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - `scripts/check_decode_plan.py` (create) — stdlib checker; instantiates the locked 158-key summary from sitting `text_config`; `--json` asserts nine stage kinds / 135 instances, stage-cut 2355200 B versus TASK-06 forced 3123200 / region-cut 5847040, unique-weight extra 0, state extra 0, 22 fusion and 5 hoist hypotheses all unselected; `--decode-plan` checks headings, canonical sentences, JSON fence equality, one Mermaid flowchart, ids, locked integers, and forbidden winner phrases.
  - `docs/architecture/decode-plan.md` (create) — ten level-2 headings; four canonical sentences verbatim; nine stage kinds (multiplicities \(1+64+64+1+1+1+1+1+1=135\)); selected serial order; per-stage loads/state/visibility/reuse; unavoidable unique-weight/state extra 0 plus stage-cut activation 2355200 B closing the ledger open question as DERIVED identities; 7 sequences + 22 fusion hypotheses + 5 hoist hypotheses all unselected; one Mermaid flowchart with required IDs; live JSON fence.
  - `docs/architecture/plan.md`, `task_ledger.md`, TASK-06/09/11/12 deliverables, and other `scripts/check_*.py` were not edited in this stage.
- Commands:
  - `python3 -m py_compile scripts/check_decode_plan.py` — pass
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass; 158 keys; `n_stage_kinds` 9; `n_stage_instances` 135; `act_stage_cut_decode_complete_bytes` 2355200; `act_forced_decode_complete_bytes` 3123200; `act_region_cut_decode_complete_bytes` 5847040; `weight_boundary_added_unique_bytes` 0; `state_boundary_added_bytes` 0; `ledger_open_question_boundary_traffic_closed` true; `schedule_serial_selected` true; `fusion_winner_selected` false; `n_fusion_hypotheses_selected` 0; `n_hoist_hypotheses_selected` 0; `thread_geometry_absent` true
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md` — pass (exit 0)
- UTC/time/tokens/cost: `2026-09-20T15:06:11Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/decode-plan.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-07 / [`numerical-sensitivity.md`](../numerical-sensitivity.md) / [`semantic-graph.md`](../semantic-graph.md) / [`materialization-and-fusion.md`](../materialization-and-fusion.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`work-and-traffic.md`](../work-and-traffic.md) (TASK-06), [`runtime-format-design.md`](../runtime-format-design.md) (TASK-09), [`semantic-graph.md`](../semantic-graph.md) (TASK-11), [`materialization-and-fusion.md`](../materialization-and-fusion.md) (TASK-12), [`model-inventory.md`](../model-inventory.md) (TASK-01) via those docs, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_decode_plan.py`](../../../scripts/check_decode_plan.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the JSON fence left unchanged. No locked integers, canonical sentences, stage-kind ids, fusion-hypothesis ids, hoist-hypothesis ids, or Mermaid node IDs edited. Frozen TASK-06/09/11/12 artifacts and `plan.md` not edited. `task_ledger.md` not edited (delivery stage).
- Commands:
  - `python3 -m py_compile scripts/check_decode_plan.py` — pass (exit 0).
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 158 keys; JSON fence source unchanged).
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md` — pass (exit 0; headings, JSON fence, one flowchart, four canonical sentences, locked ids/integers, `HYPOTHESIS` / `hardware-independent`; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T15:10:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `composer-2.5` (verifier subagent; parent/inherit mapping)
- Diff review:
  - New: `docs/architecture/decode-plan.md`, `scripts/check_decode_plan.py`, `docs/architecture/tasks/TASK-13.md`.
  - Modified tracked: `docs/architecture/task_ledger.md` only (`TODO` → `IN PROGRESS` at admission; no completion checkboxes or history entry; acceptable pre-delivery status flip).
  - Frozen deliverables unchanged: `plan.md`, `work-and-traffic.md`, `runtime-format-design.md`, `semantic-graph.md`, `materialization-and-fusion.md`, other `scripts/check_*.py`.
  - Scope matches dossier: hardware-independent decode schedule, nine stage kinds / 135 instances, serial order selected, per-stage table, unavoidable vs stage-cut traffic, seven sequences + 22 fusion + 5 hoist hypotheses all unselected, one Mermaid diagram, JSON fence, stdlib checker only. No thread geometry, fusion/packing winners, Quartz/llama.cpp inspection, or plan/ledger delivery edits.
- Independent raw-record checks:
  - Live `--json` object (158 keys) equals fenced JSON in `decode-plan.md` (byte-for-byte after parse).
  - Stage-cut arithmetic: `n_residual_stage_crossings` 133 × `residual_bytes` 10240 = `act_residual_stage_cut_bytes` 1361920; `n_logit_outputs` 2 × `logits_bytes` 496640 = `act_logits_stage_cut_bytes` 993280; sum = `act_stage_cut_decode_complete_bytes` 2355200.
  - TASK-06 cited minima match checker/live JSON: `act_forced_decode_complete_bytes` 3123200, `act_region_cut_decode_complete_bytes` 5847040, `weight_unique_plus_gather_decode_complete` 52098619392, `decode_write_bytes` 152047616 (confirmed in `work-and-traffic.md` prose/JSON).
  - `weight_boundary_added_unique_bytes` 0; `state_boundary_added_bytes` 0; `ledger_open_question_boundary_traffic_closed` true; closed by DERIVED stage-cut identities, not fusion/packing selection.
  - Ten required `##` headings in locked order; four canonical sentences present verbatim; one ` ```mermaid ` fence with required `diagram_ids`; `check_decode_plan.py` imports stdlib only (no sibling `check_*.py`).
- Commands:
  - `test -f docs/architecture/decode-plan.md` — pass
  - `python3 -m py_compile scripts/check_decode_plan.py` — pass (exit 0)
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 158 keys; `n_stage_kinds` 9; `n_stage_instances` 135; `act_stage_cut_decode_complete_bytes` 2355200; `act_forced_decode_complete_bytes` 3123200; `act_region_cut_decode_complete_bytes` 5847040; `weight_boundary_added_unique_bytes` 0; `state_boundary_added_bytes` 0; `ledger_open_question_boundary_traffic_closed` true; `fusion_winner_selected` false; `n_fusion_hypotheses_selected` 0; `n_hoist_hypotheses_selected` 0; `thread_geometry_absent` true)
  - `python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md` — pass (exit 0)
  - Independent Python fence/arithmetic spot-check (not in dossier focused list) — pass
- Formatting changed files: none (`uv run ruff format .` not required for this increment)
- Verdict: **PASS**
- UTC/time/tokens/cost: `2026-09-20T15:07:00Z`; `telemetry_unavailable`

### Retries and escalation

none yet.

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-13 only; coupled IDs `none`
- Outcome: TASK-13 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T15:08:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/decode-plan.md` (10 locked headings, four canonical sentences, nine stage kinds / 135 instances, serial one-token order, per-stage loads/state/visibility/reuse, unavoidable vs stage-cut traffic identities, seven consumer sequences + 22 fusion + 5 hoist hypotheses unselected, one Mermaid flowchart, JSON fence); `scripts/check_decode_plan.py` stdlib checker; independent JSON-fence equality and stage-cut arithmetic confirmed against TASK-06 cited minima; frozen upstream docs unchanged; fusion, packing, layout, prefill, and CUDA mapping remain open; ledger open question closed by DERIVED stage-cut identities versus TASK-06 minima
- Candidate measured delta: N/A — decode-schedule documentation
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete for this increment (no GPU/quality gates)
- Throughput delta: N/A — TASK-13 does not execute or time the model
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification attempt 1 pass)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: none
