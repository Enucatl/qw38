# TASK-19 — Design the future performance methodology

## Control

- Primary ID: `TASK-19`
- Coupled IDs: `none`
- Dependencies: `TASK-06`, `TASK-13`, `TASK-14`, `TASK-17` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Specify requested decode, prefill, and kernel metrics; Require end-to-end measurement alongside microbenchmarks; Do not run implementation benchmarks in this study task.

## Goal and boundaries

Produce `docs/architecture/performance-validation.md` as the Phase 1 **performance evaluation methodology** for Qwen3.8-27B language+MTP. Close the three ledger completion criteria by (1) naming decode, prefill, and kernel metrics with locked identities, (2) requiring end-to-end measurement alongside microbenchmarks (memory metrics included as the fourth named family), and (3) running **no** implementation benchmarks, SKU fill-in, Nsight, `deviceQuery`, or tok/s collection in this study task. **Keep** the ledger open question (exact hardware, prompt matrix, and reproducibility protocol for the implementation phase) **unresolved**. Do **not** select a CUDA mapping winner. Do **not** fill the TASK-16 sitting SKU table. Do **not** close TASK-17’s mapping-winner question.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Work/traffic minima, decode \(W=C+AT\), prefill \(W=TC+AT(T+1)/2\), unique-weight / state / activation channels, and example horizons come from `docs/architecture/work-and-traffic.md` (TASK-06). Decode one-token schedule, stage kinds, and GEMV consumers come from `docs/architecture/decode-plan.md` (TASK-13). Prefill many-token schedule, five fundamental differences, and GEMM consumers come from `docs/architecture/prefill-plan.md` (TASK-14). Eighteen unselected mappings, six node types, seven evaluation criteria, and SKU-UNKNOWN instantiation come from `docs/architecture/cuda-design-space.md` (TASK-17). TASK-16 SKU symbols and F1–F14 are cited through TASK-17 / `cuda-hardware-model.md`; do not recopy the hardware-model catalog as a new vocabulary study. TASK-18 tok/s-is-not-quality is cited; do not rewrite NLL methodology.
  - Label claims `OBSERVED` (sitting `text_config` / inventory already established; TASK-16 \(N_w=32\), \(N_{\text{bank}}=32\)), `DERIVED` (MAC/byte citations, tok/s and latency **formulas**, identity/coverage contracts), `HYPOTHESIS` (every methodology-risk severity), or `UNKNOWN` (sitting-SKU numeric limits; vision-encoder internals). No new `MEASURED` tok/s, occupancy, bandwidth, latency, or NLL. Protocol limits are **methodology contracts**, not measured tables.
  - GitHub Markdown math. Cite TASK-06 MAC/byte integers, TASK-13/14 stage kinds and \(T\) convention, TASK-11 node types via TASK-17, TASK-16 SKU JSON ids, TASK-18 `toks_is_not_quality_axis`. Do not rewrite forward math, recopy TASK-06 MAC tables as a new work study, recopy TASK-13/14 schedules, or recopy TASK-17’s eighteen mappings as a new design space.
  - Allowed evidence: TASK-06 work-and-traffic, TASK-13 decode-plan, TASK-14 prefill-plan, TASK-17 cuda-design-space, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier, and the performance-evidence identity/coverage/time-accounting **contracts** as methodology (not as executed traces). TASK-01/04/11/16 integers and ids already cited by those documents may be **cited** through them. TASK-18 is cited only for the tok/s-is-not-quality split. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`. No `deviceQuery`, Nsight, CUPTI, SASS, or `*.cu` / `*.cuh` kernel bodies.
- Non-goals:
  - No implementation benchmarks (`benchmarks_run` false, `toks_measured_here` false). No Nsight, ncu, nsys, CUPTI, `deviceQuery`, occupancy dumps, or sitting-GPU clocks.
  - No sitting SKU fill-in (`sku_table_filled` false). Datasheet numbers are not measurements.
  - No selected hardware, prompt matrix, warmup count, clock-lock policy, or reproducibility sitting (`hardware_selected` false, `prompt_matrix_selected` false, `reproducibility_protocol_selected` false).
  - No CUDA mapping, kernel, launch-config, occupancy-target, MMA-shape, layout, fusion, packing, or view **winner**. `mapping_winner_selected` false. TASK-17’s ledger open question stays open (`ledger_open_question_mapping_winner_closed` false).
  - No quality/NLL experiments (TASK-18). Tok/s is not a quality axis (`toks_is_not_quality_axis` true).
  - No new operators, node types, mappings, stage kinds, or hardware concepts.
  - No Quartz/llama.cpp inspection; do not open the GGUF file (`gguf_payload_inspected` false). After freeze, those engines may be future black-box **performance** references with matching identities; they are not design authorities here.
  - No peak-memory claim presented as a measured live-set. Working-set high-water is a named metric identity, not a number in this document.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–18). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `work-and-traffic.md`, `decode-plan.md`, `prefill-plan.md`, `cuda-design-space.md`, `cuda-hardware-model.md`, `quantization-validation.md`, or any TASK-01–18 deliverable.
  - Do not import any `scripts/check_*.py`. Do not stream safetensor payloads.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib performance-validation checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places **future experimental validation** after CUDA mappings; TASK-18 is quality/quantization validation; TASK-19 is performance.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint authority; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden until freeze; CUDA hardware documentation is allowed (consumed here via TASK-16/17 citations, not by inspecting kernels or running `deviceQuery`).
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:93-96` — TASK-17 maps semantic nodes and layouts into CUDA experiment spaces; TASK-18 defines quality validation; TASK-19 defines end-to-end and kernel performance methodology.
- `docs/architecture/task_ledger.md` TASK-06 established results — decode \(W=C+AT\); prefill \(W=TC+AT(T+1)/2\); \(C_\text{complete}=27433238528\), \(A_\text{complete}=208896\); unique non-embed `52098598912`; six HYPOTHESIS bottleneck labels; example horizons \(T\in\{1,4096\}\).
- `docs/architecture/task_ledger.md` TASK-13 established results — nine stage kinds; decode GEMV; one-token \(T_\text{new}=1\) with populated \((K,V,C,S)\); unique-weight/state extra 0; fusion/packing unselected.
- `docs/architecture/task_ledger.md` TASK-14 established results — nine stage kinds; prefill GEMM; five fundamental differences; incoming state zeros; distinct views unselected; at \(T=1\) MAC prefill = decode (traffic still differs).
- `docs/architecture/task_ledger.md` TASK-16 established results — SKU-UNKNOWN table; \(N_w=32\), \(N_{\text{bank}}=32\); F1–F14; sitting limits remain UNKNOWN until a future measured SKU table (this task defines the fill **protocol**, not the numbers).
- `docs/architecture/task_ledger.md` TASK-17 established results — 18 unselected mappings; seven criteria with UNKNOWN SKU; mapping winner remains open until TASK-19 **measurements** (this increment specifies the methodology and does not run those measurements or select a winner).
- `docs/architecture/task_ledger.md` TASK-18 established results — teacher-forced NLL is the quality axis; `toks_is_not_quality_axis` true; `v_toks_as_quality` named.
- `docs/architecture/task_ledger.md` TASK-19 row — produces `docs/architecture/performance-validation.md`; purpose is decode, prefill, kernel, memory, and end-to-end performance measurements; open question (exact hardware, prompt matrix, reproducibility protocol) is **not** closed here; completion is named decode/prefill/kernel metrics, e2e required alongside microbenchmarks, and no implementation benchmarks in this study task. Downstream: architectural experiments evaluated fairly.
- `docs/architecture/work-and-traffic.md` — MAC identities; three traffic channels; \(T\) convention; bottleneck labels HYPOTHESIS.
- `docs/architecture/decode-plan.md` — decode consumers; serial one-token order; stage-cut vs unavoidable.
- `docs/architecture/prefill-plan.md` — prefill consumers; `diff_matrix_matrix` / `diff_tiling` / `diff_reuse` / `diff_state` / `diff_temporary_storage`.
- `docs/architecture/cuda-design-space.md` — six `node_type_ids`; 18 `mapping_ids` unselected; SKU still UNKNOWN; “TASK-19 owns measurements and the sitting SKU table.”
- `docs/architecture/cuda-hardware-model.md` — `sku_unknown_symbols` (17 ids); theoretical occupancy F8; achieved occupancy out of scope **there**; this methodology names achieved occupancy as a future kernel metric without measuring it.
- `docs/architecture/quantization-validation.md` — tok/s is not Pareto \(Y\); do not mix TASK-18 quality with TASK-19 performance.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads. Do not read GGUF.
- `scripts/check_work_and_traffic.py` / `check_decode_plan.py` / `check_prefill_plan.py` / `check_cuda_design_space.py` / `check_quantization_validation.py` — checker-style precedent. TASK-19’s checker is a sibling; do not import them.

## Performance evidence

N/A — performance **evaluation-methodology** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. MAC and byte figures are TASK-06 citations, not MEASURED traffic or tok/s. Metric **formulas** are DERIVED protocol. Methodology-risk **labels** are HYPOTHESIS. This increment **defines** identity, coverage, and time-accounting contracts for later measured work; it does not apply the performance-evidence checklist to rank kernels or claim a winning mapping. Do not treat a named metric as a measured point.

- Measurement identity: N/A (no engine binary, no eval run). Future identity fields are specified as a protocol in heading 8. That protocol is not executed here.
- Metric class: N/A for this increment. Future classes are locked: `prefill`, `decode_only`, `complete_request`, `component`, `memory`. A ratio requires matching identities on both arms.
- Coverage: N/A for GPU graphs. Methodology coverage is 5 metric families, 4 decode metrics, 4 prefill metrics, 5 kernel metrics, 5 memory metrics, 3 end-to-end metrics, 15 identity fields, 9 coverage fields, 17 SKU-UNKNOWN symbols, 8 methodology risks.
- Time accounting: N/A (no intervals). Future disjoint-union rules are locked as contracts in heading 8.
- Contradiction register: none at planning. If a live `text_config` occupancy product disagrees with TASK-01, or a cited TASK-06/13/14/17 integer or id disagrees with those documents, stop and fail closed (do not remeasure, do not invent tok/s).
- Claim types: occupancy/MAC/byte citations `observed`/`derived`; tok/s and latency formulas `derived` methodology contracts; every methodology-risk severity `hypothesis`; sitting SKU and vision encoder `unknown`.
- Target/guard roles: not opted in. No numeric tok/s guard.
- Evidence completeness: N/A for performance-evidence checks. Methodology completeness is the three ledger completion criteria plus explicit non-closure of hardware / prompt matrix / reproducibility protocol.
- Screen eligibility: N/A (no keep/reject experiment).
- Shipping evidence: N/A

## Implementation decisions

### Authority for performance-methodology claims

If a MAC or byte total would disagree with TASK-06 / sitting `text_config`, or a stage kind / \(T\) convention would disagree with TASK-13/14, or a node type / mapping id / evaluation criterion / SKU symbol would disagree with TASK-17/16, the earlier document wins and this one is wrong.

- Prefill and decode share **one** semantic graph (TASK-11) and **one** compiled artifact (TASK-09 via TASK-17). They do **not** share one performance metric identity. Decode-only tok/s is not prefill tok/s and not complete-request latency.
- Primary object is language+MTP **complete map** (135 node instances). Language-only is a secondary row, not a substitute identity.
- Algebraic equivalents in TASK-02 remain the same real map. Chunkwise GDN is not zero \(S\) traffic. At \(T=1\), MAC prefill = decode; C/S physical reads still differ (zeros vs populated). That equality does **not** license reporting T=1 prefill as decode-only.
- A named metric is not a measured table. Listing `dec_toks` does not produce a tok/s number.
- Unique weight bytes are counted **once** per complete decode/prefill as a TASK-06 lower-bound **identity**, not as measured HBM traffic.
- Do not inspect Quartz, llama.cpp, or GGUF kernels to “confirm” timers, graph capture, or tok/s harnesses.
- Do not fill TASK-16 UNKNOWN symbols from a datasheet or sitting `cudaGetDeviceProperties`.
- Do not select a mapping winner. Closing the three completion criteria does **not** close hardware, prompt matrix, reproducibility protocol, or TASK-17’s mapping-winner question.
- TASK-18 quality remains disjoint: tok/s is not Pareto \(Y\).

### Deliverable structure (`docs/architecture/performance-validation.md`)

Use these **level-2 headings in this exact order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED` (citation). Every protocol-limit row is labelled as a methodology contract, not a measured error. Every methodology-risk severity is `HYPOTHESIS`. SKU limits stay `UNKNOWN`. Do not leave `TBD`. The word `UNKNOWN` may appear in SKU-fill rows and must appear in Deferred vision.

Title: `# Qwen3.8-27B performance methodology (TASK-19)` (not `TASK-19` alone).

1. **Authority** — this dossier, work-and-traffic, decode-plan, prefill-plan, cuda-design-space, cuda-hardware-model via TASK-17, config, checker; evidence labels; in-scope (language+MTP decode/prefill/kernel/memory/end-to-end methodology + identity/coverage/SKU-fill protocol) vs deferred (vision encoder; sitting SKU numbers; selected hardware/prompts/repro; mapping winner). State that the document specifies an **evaluation methodology**, not measurements, and not a selected mapping.
2. **Measurement convention** — the five canonical sentences below plus the methodology sentence; five metric families; five metric classes; first-token convention; open question stays open.
3. **Decode metrics** — four metric ids; decode-only identity; \(W=C+AT\) citation. Completes ledger checkbox 1 (decode).
4. **Prefill metrics** — four metric ids; prefill identity; \(W=TC+AT(T+1)/2\) citation; T=1 MAC equality does not imply equal traffic. Completes ledger checkbox 1 (prefill).
5. **Kernel metrics** — five metric ids; graph envelope ≠ leaf; coverage before ranking; microbenchmarks cannot pass a mapping. Completes ledger checkbox 1 (kernel).
6. **Memory metrics** — five metric ids; three TASK-06 channels plus achieved HBM and working-set high-water; lower bounds are not measured bandwidth.
7. **End-to-end metrics** — three metric ids; complete-request primary is latency; e2e required alongside microbenchmarks. Completes ledger checkbox 2.
8. **Identity, coverage, and time accounting** — 15 identity fields; 9 coverage fields; disjoint-union contracts; no parent+child; synchronize is wait not GPU idle.
9. **SKU-fill protocol and unselected hardware** — 17 SKU-UNKNOWN symbols; fill protocol defined; table not filled; hardware / prompt matrix / reproducibility unselected. Contains the Mermaid fence.
10. **Methodology risks** — eight risks, all HYPOTHESIS.
11. **Deferred vision** — residual-stream interface only; encoder performance UNKNOWN.
12. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Measurement convention**, include these **five canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Decode, prefill, kernel, memory, and end-to-end metrics in this document are a methodology, not measurements.

> Microbenchmarks cannot pass a CUDA mapping; end-to-end measurement on a matching identity is required alongside them.

> Exact hardware, prompt matrix, and reproducibility protocol for the implementation phase remain unselected.

> This document fills no sitting SKU numbers and selects no mapping winner.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_methodology` = sentence 2; `canonical_sentence_e2e` = sentence 3; `canonical_sentence_open_question` = sentence 4; `canonical_sentence_winner` = sentence 5.

Sentence 4 **keeps** the ledger open question unresolved in checker-substring form. JSON `ledger_open_question_hardware_closed` false. Sentence 5 **closes** the third completion criterion as a prohibition, not as a selected mapping, and keeps SKU unfilled. Sentence 5 does **not** close TASK-17’s mapping-winner question; JSON `ledger_open_question_mapping_winner_closed` false.

Immediately after those five, include this **required methodology sentence verbatim**:

> Decode, prefill, and kernel metrics are specified; end-to-end measurement is required alongside microbenchmarks; no implementation benchmarks are run in this study task; exact hardware, prompt matrix, and reproducibility protocol remain unselected.

JSON: `methodology_question_sentence` = that sentence.

Bullets required under that heading:

- Five metric families are complete for this task: `decode`, `prefill`, `kernel`, `memory`, `end_to_end` (`n_metric_families` 5).
- Five metric classes are complete: `prefill`, `decode_only`, `complete_request`, `component`, `memory` (`n_metric_classes` 5). A ratio requires matching classes and windows on both arms (`ratio_requires_matching_identity` true).
- First-token convention is `ttft_in_prefill`: the first generated token belongs to the prefill / TTFT window, not to the decode-only numerator (`first_token_in_prefill` true; `decode_only_excludes_first_token` true).
- Decode-only tok/s is not prefill tok/s and not complete-request latency (`decode_is_not_prefill` true; `decode_is_not_complete_request` true).
- Microbenchmarks (kernel + memory component windows) cannot pass a mapping (`microbenchmark_cannot_pass_mapping` true). End-to-end measurement is required alongside them (`e2e_required_alongside_microbenchmarks` true).
- `example_T_values` `[1, 4096]` are illustration horizons for MAC/byte citations, not a selected prompt matrix (`example_T_is_not_prompt_matrix` true).
- Model context horizon is `max_position_embeddings` \(T_{\text{ctx max}}=262144\). Do not confuse it with TASK-16 SKU `T_max` (max threads per SM). JSON `model_T_max` = 262144. JSON SKU id `T_max` remains a `sku_unknown_symbols` entry.
- Primary coverage includes MTP (135 instances). Language-only is secondary.
- Fan-out ≠ must-store still holds; naming a metric is not a CUDA timer and not a measured table.
- No MEASURED tok/s, occupancy, or bandwidth in this document (`toks_measured_here` false; `benchmarks_run` false; `experiments_run` false).
- Tok/s is not a quality axis (`toks_is_not_quality_axis` true).
- Hardware, prompt matrix, and reproducibility protocol remain unselected. SKU table remains unfilled. Mapping winner remains unselected.

### Metric families and classes (lock)

JSON array `metric_family_ids` in this exact order (5 ids). JSON `n_metric_families` = 5.

| id | Role | Can fail a mapping | Can declare a mapping winner |
| --- | --- | --- | --- |
| `decode` | Decode-only rate and per-step latency | yes (identity-matched regression) | no, not alone |
| `prefill` | Prefill rate and window latency | yes | no, not alone |
| `kernel` | Component / leaf / envelope / occupancy | yes (local) | no (`microbenchmark_cannot_pass_mapping`) |
| `memory` | HBM / weight / state / activation / working-set | yes (local) | no |
| `end_to_end` | Complete-request latency (primary) plus declared output rate | yes | required alongside kernel/memory; still not a winner by itself in this document |

JSON: `microbenchmark_cannot_pass_mapping` true; `e2e_required_alongside_microbenchmarks` true; `kernel_cannot_pass_mapping` true.

JSON array `metric_class_ids` in this exact order (5 ids). JSON `n_metric_classes` = 5: `prefill`, `decode_only`, `complete_request`, `component`, `memory`.

JSON `public_sample_eval_output_is_quality` true (TASK-18 owns that class; this document must not report it as tok/s). JSON `toks_is_not_quality_axis` true.

### Decode metrics (lock)

JSON array `decode_metric_ids` in this exact order (4 ids). JSON `n_decode_metrics` = 4. Parallel `decode_metric_classes` all `decode_only`.

| ID | Class | Meaning |
| --- | --- | --- |
| `dec_toks` | decode_only | \((N_{\mathrm{gen}}-1)/t_{\mathrm{decode}}\) with \(N_{\mathrm{gen}}\ge 2\); \(t_{\mathrm{decode}}\) excludes setup, graph creation, warmup, prefill/TTFT, and checkpoint restore |
| `dec_step_ms` | decode_only | Mean GPU time per subsequent generated token in the decode-only window |
| `dec_p50_step_ms` | decode_only | Median per-step GPU time in that window |
| `dec_p99_step_ms` | decode_only | 99th-percentile per-step GPU time in that window |

Decode-only formula (DERIVED methodology; not measured here). After a length-\(T\) prompt with incoming \((K,V,C,S)\) **populated** (TASK-13), \(T_{\text{new}}=1\) per step:

\[
\mathrm{tok/s}_{\mathrm{dec}}=\frac{N_{\mathrm{gen}}-1}{t_{\mathrm{decode}}},\qquad N_{\mathrm{gen}}\ge 2.
\]

Cite TASK-06 decode work as the **bound identity**, not a timer:

\[
W_{\mathrm{decode}}=C+AT,
\]

with \(C_\text{complete}=27433238528\), \(A_\text{complete}=208896\) (DERIVED citation). JSON `mac_C_complete` = 27433238528. JSON `mac_A_complete` = 208896. JSON `decode_work_identity` = `"C+AT"`. JSON `decode_T_new` = 1. JSON `decode_incoming_state` = `"populated"`.

Prose required: setup, graph creation, warmup, and checkpoint restore stay outside the decode-only denominator. TTFT is not decode-only. Do not report a single-token “decode” as `dec_toks`.

### Prefill metrics (lock)

JSON array `prefill_metric_ids` in this exact order (4 ids). JSON `n_prefill_metrics` = 4. Parallel `prefill_metric_classes`: `prefill`, `prefill`, `prefill`, `prefill`.

| ID | Class | Meaning |
| --- | --- | --- |
| `pre_toks` | prefill | \(T/t_{\mathrm{prefill}}\) for a length-\(T\) prompt with incoming \((K,V,C,S)\) **zeros** (TASK-14) |
| `pre_ms` | prefill | Prefill-window wall (same exclusions as decode-only: setup, graph create, warmup, restore) |
| `pre_ttft_ms` | prefill | Time to first generated token; equals the prefill window under `ttft_in_prefill` |
| `pre_mac_cite` | prefill | TASK-06 \(W=TC+AT(T+1)/2\) citation for the same \(T\); not a timer |

Prefill formula (DERIVED methodology; not measured here):

\[
\mathrm{tok/s}_{\mathrm{pre}}=\frac{T}{t_{\mathrm{prefill}}}.
\]

Prefill work identity:

\[
W_{\mathrm{prefill}}=TC+\frac{AT(T+1)}{2}.
\]

JSON `prefill_work_identity` = `"TC+AT(T+1)/2"`. JSON `prefill_incoming_state` = `"zeros"`. JSON `t1_mac_equal_does_not_imply_equal_traffic` true. JSON `mac_decode_complete_T1` = 27433447424. JSON `mac_prefill_complete_T1` = 27433447424 (TASK-06; T=1 MAC equality). JSON `mac_decode_complete_T4096` = 28288876544. JSON `mac_prefill_complete_T4096` = 114119319486464.

Prose required: at \(T=1\), MAC prefill equals MAC decode; C/S physical reads still differ. Do not report T=1 `pre_toks` as `dec_toks`. Prefill numerator is prompt length \(T\), not \(T+1\) first-token padding.

### Kernel metrics (lock)

JSON array `kernel_metric_ids` in this exact order (5 ids). JSON `n_kernel_metrics` = 5. Parallel `kernel_metric_classes` all `component`.

| ID | Class | Meaning |
| --- | --- | --- |
| `k_node_ms` | component | GPU time attributed to one TASK-11 node type in a covered window |
| `k_mapping_ms` | component | GPU time for one TASK-17 mapping alternative under test; listing is not selecting |
| `k_leaf_union_ms` | component | Disjoint union of **leaf** kernel intervals in the window |
| `k_graph_envelope_ms` | component | CUDA-graph / parent envelope duration; **not** leaf kernel time |
| `k_occupancy_achieved` | component | Achieved occupancy; theoretical occupancy remains TASK-16 F8 and is not this metric |

JSON `graph_envelope_is_not_leaf` true. JSON `missing_coverage_is_not_zero` true. JSON `kernel_cannot_pass_mapping` true. JSON `achieved_occupancy_measured_here` false. JSON `n_node_types` = 6. JSON `node_type_ids` in TASK-11 order: `embed`, `gated_attn`, `gated_delta_net`, `mlp`, `lm_head`, `mtp_mix`. JSON `n_mappings` = 18. JSON `n_mappings_selected` = 0. JSON `mapping_winner_selected` false.

JSON array `mapping_ids` equals the TASK-17 18-id list (citation completeness; do not retabulate ownership/reduction as a new design space). Duplicate as checker constants; do not import TASK-17’s script.

TASK-17 `mapping_ids` order (18): `map_embed_thread_element`, `map_embed_warp_row`, `map_embed_cta_vector`, `map_attn_cta_head`, `map_attn_warp_t`, `map_attn_cta_splitk`, `map_gdn_cta_head`, `map_gdn_warp_recurrent`, `map_gdn_cta_chunk`, `map_mlp_cta_dout`, `map_mlp_cta_splitk`, `map_mlp_grid_T`, `map_lm_cta_vocab`, `map_lm_cta_splitk`, `map_lm_warp_gemv`, `map_mtp_cta_fc`, `map_mtp_cta_fused`, `map_mtp_split_norm_gemm`.

Prose required: graph envelopes are not leaf kernel time. Missing node tracing forbids graph-internal idle and component-completeness claims. Unclassified work is not zero excess. A kernel microbenchmark may **fail** a mapping locally; it must not **pass** a mapping onto a winner without matching-identity end-to-end measurement. Do not rank TASK-17 mappings in this document.

Cite TASK-17 evaluation criterion ids as substrings (7): `crit_occupancy`, `crit_latency_hiding`, `crit_wave_quant`, `crit_intensity_roofline`, `crit_sync_class`, `crit_pipeline_mix`, `crit_fusion_delta`. JSON `n_evaluation_criteria` = 7. JSON `evaluation_measured` false.

### Memory metrics (lock)

JSON array `memory_metric_ids` in this exact order (5 ids). JSON `n_memory_metrics` = 5. Parallel `memory_metric_classes` all `memory`.

| ID | Class | Meaning |
| --- | --- | --- |
| `mem_hbm_gbps` | memory | Achieved device-global bandwidth in a covered window; compare later to SKU \(\Beta\) only after SKU fill |
| `mem_weight_bytes` | memory | Unique weight bytes moved in the window; TASK-06 unique non-embed `52098598912` is the lower-bound **identity**, not measured traffic |
| `mem_state_bytes` | memory | \(K,V,C,S\) traffic versus TASK-04/06 decode-read / triangular-prefill identities |
| `mem_act_bytes` | memory | Activation traffic versus TASK-06 forced / region-cut / GEMM-IO views |
| `mem_working_set` | memory | Allocated high-water with an explicit identity; not a proven live-set without coverage |

JSON `weight_bytes_unique_non_embed` = 52098598912. JSON `weight_bytes_language_mtp_excl_vision` = 54641395712. JSON `lower_bound_is_not_measured_bandwidth` true. JSON `peak_memory_is_not_live_set` true. JSON `n_traffic_channels` = 3. JSON `traffic_channel_ids` = `["weight","state","activation"]`.

Prose required: TASK-06 bytes are DERIVED minima. Bytes divided by assumed \(\Beta\) is a conditional estimate, not measured `mem_hbm_gbps`. Do not fill \(\Beta\) (`Beta_HBM`) here.

### End-to-end metrics (lock)

JSON array `e2e_metric_ids` in this exact order (3 ids). JSON `n_e2e_metrics` = 3.

| ID | Class | Primary? | Meaning |
| --- | --- | --- | --- |
| `e2e_latency_ms` | complete_request | **primary** | Wall from first in-window prompt ingest to last generated token; setup/graph-create/warmup/restore excluded |
| `e2e_output_toks` | complete_request | secondary | \(N_{\mathrm{gen}}/t_{\mathrm{e2e}}\); **not** decode-only; **not** \((T+N_{\mathrm{gen}})/t_{\mathrm{e2e}}\) unless that mixed identity is declared separately and never compared to `dec_toks` |
| `e2e_ttft_ms` | complete_request | diagnostic | Same window as `pre_ttft_ms` under `ttft_in_prefill`; still complete-request family when reported beside `e2e_latency_ms` |

JSON `e2e_primary_id` = `"e2e_latency_ms"`. JSON `e2e_mixed_toks_is_not_decode_only` true. JSON `e2e_required_alongside_microbenchmarks` true. JSON `complete_request_vs_decode_only_forbidden` true.

Prose required: refuse a complete-request versus decode-only comparison. Mixed \((T+N_{\mathrm{gen}})/t_{\mathrm{e2e}}\) is not `dec_toks` and is not `pre_toks`. End-to-end measurement is **required alongside** kernel and memory microbenchmarks; it is not optional colour. This document still selects no winner (`mapping_winner_selected` false).

### Identity, coverage, and time accounting (lock)

JSON array `identity_field_ids` in this exact order (15 ids). JSON `n_identity_fields` = 15.

| ID | Must record |
| --- | --- |
| `engine_commit` | Engine / study commit |
| `loaded_binary_hash` | Actual loaded binary hash |
| `build_flags` | Build flags / image / tool versions |
| `artifact_selectors` | Compiled artifact or future black-box selectors (GGUF not opened here) |
| `input_token_hash` | Input token hash |
| `prefix` | Prefix identity |
| `output_eval_counts` | Output / eval counts |
| `first_token_convention` | Must equal `ttft_in_prefill` unless a later task records a different declared convention |
| `allocated_capacity` | Allocated KV/state capacity |
| `populated_length` | Populated length \(T\) |
| `sampling_output_policy` | Sampling / output policy (unselected here) |
| `graph_mode` | Graph / eager mode |
| `clocks_residents` | Clock and residency policy |
| `warmups_samples` | Warmup count and sample count |
| `numerator_start_end_events` | Exact numerator and start/end events |

JSON `identity_protocol_defined` true. JSON `identity_values_selected` false. JSON `sampling_policy_selected` false. JSON `warmup_count_selected` false. JSON `clock_policy_selected` false.

JSON array `coverage_field_ids` in this exact order (9 ids). JSON `n_coverage_fields` = 9: `captured_tables`, `units`, `clock_domain`, `capture_bounds`, `streams`, `expected_graph_node_counts`, `observed_graph_node_counts`, `family_mapping`, `unclassified_work`.

JSON `coverage_protocol_defined` true. JSON `coverage_values_selected` false. JSON `graph_envelope_is_not_leaf` true. JSON `missing_coverage_is_not_zero` true.

JSON array `time_accounting_rule_ids` in this exact order (5 ids). JSON `n_time_accounting_rules` = 5.

| ID | Contract |
| --- | --- |
| `ta_disjoint_union` | Union overlapping intervals before summing; reconstruct a disjoint union if a leaf sum exceeds its envelope |
| `ta_no_parent_child` | Do not sum graph parents and children, or nested NVTX and kernels, into one total |
| `ta_no_cpu_gpu_double` | Do not add overlapping CPU waits and GPU work |
| `ta_sync_is_wait` | CPU `cudaEventSynchronize` duration is wait duration, not removable GPU idle |
| `ta_setup_outside_rate` | Setup, graph creation, warmup, and checkpoint restore stay outside rate denominators |

JSON `time_accounting_protocol_defined` true.

Prose required: a ratio requires matching metric identities. Unmapped families stay `null`/`unknown`. Missing hardware later is `incomplete`/`blocked`, not a measured rejection. Historical numbers are references, not current denominators.

### SKU-fill protocol and unselected hardware (lock)

JSON array `sku_unknown_symbols` equals the TASK-16 17-id list. JSON `n_sku_unknown_symbols` = 17. Duplicate as checker constants; do not import TASK-16’s script.

TASK-16 `sku_unknown_symbols` order (17): `N_SM`, `W_max`, `S_reg`, `C_smem`, `T_max`, `B_max`, `N_bar`, `N_sched`, `G_reg`, `G_smem`, `Beta_HBM`, `Pi_FMA`, `Pi_TC`, `L_issue`, `async_copy_cap`, `mma_shapes`, `cluster_cap`.

JSON `N_w` = 32. JSON `N_bank` = 32. JSON `sku_table_filled` false. JSON `sku_fill_protocol_defined` true. JSON `sku_from_datasheet_is_not_measurement` true. JSON `hardware_selected` false. JSON `prompt_matrix_selected` false. JSON `reproducibility_protocol_selected` false. JSON `ledger_open_question_hardware_closed` false.

SKU-fill protocol (future; not executed here):

- Capacity symbols (`N_SM`, `W_max`, `S_reg`, `C_smem`, `T_max` threads-per-SM, `B_max`, `N_bar`, `N_sched`, `G_reg`, `G_smem`, `cluster_cap`, `async_copy_cap`, `mma_shapes`) are filled from a **declared sitting device** with recorded identity, not from an unnamed datasheet.
- Peak symbols (`Beta_HBM`, `Pi_FMA`, `Pi_TC`, `L_issue`) are filled from identity-matched microbenchmarks on that same sitting, not from a marketing peak.
- Filling the table is a later measured increment. This document only names the protocol.

`example_T_values` `[1, 4096]` remain illustration horizons. Do not name WikiText, ShareGPT, or any other corpus as **the** prompt matrix. Do not name a GPU SKU as **the** hardware. Do not lock warmup counts or clock locks.

JSON `example_T_values` = `[1, 4096]`. JSON `example_T_is_not_prompt_matrix` true.

Exactly **one** fenced `mermaid` block, under this heading. Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 18 mappings, or token sequences.

Required node IDs **inside** the mermaid fence (substring match): `DECODE`, `PREFILL`, `KERNEL`, `MEMORY`, `IDENTITY`, `E2E`, `SKU`, `OPENQ`.

JSON `n_diagrams` is 1. `diagram_ids` is `["decode","prefill","kernel","memory","identity","e2e","sku","openq"]`.

Suggested topology (implementer may rearrange edges; IDs are mandatory): decode/prefill/kernel/memory → identity → e2e; sku → openq; e2e → openq. Caption must state that OPENQ is the unselected hardware / prompt matrix / reproducibility protocol, not a selected winner.

### Methodology risks (lock)

JSON array `methodology_risk_ids` in this exact order (8 ids). JSON `n_methodology_risks` = 8. Parallel `methodology_risk_severities` (`high`/`medium`/`low`). All usefulness claims beyond the locked contracts are `HYPOTHESIS`.

| ID | Deferred or pair | Severity | Why it can mislead |
| --- | --- | --- | --- |
| `v_mixed_identity` | class mismatch | high | Comparing decode-only tok/s to complete-request latency (or mixed \((T+N)/t\)) |
| `v_envelope_as_leaf` | `k_graph_envelope_ms` | high | Ranking sinks from graph envelopes without leaf / node tracing |
| `v_micro_as_winner` | kernel/memory only | high | Declaring a TASK-17 mapping winner from microbenchmarks without e2e |
| `v_missing_coverage_zero` | coverage | high | Treating unclassified work as zero excess |
| `v_bound_as_measured` | TASK-06 | medium | Treating MAC/byte lower bounds as measured traffic or tok/s |
| `v_datasheet_sku` | SKU table | medium | Filling `sku_unknown_symbols` from a datasheet without sitting identity |
| `v_t1_prefill_as_decode` | \(T=1\) | medium | Equating T=1 prefill with decode because MAC matches |
| `v_toks_as_quality` | TASK-18 | low | Using tok/s as Pareto \(Y\) |

JSON `n_methodology_high` = 4. JSON `n_methodology_medium` = 3. JSON `n_methodology_low` = 1. JSON `methodology_risk_usefulness_label` exactly `HYPOTHESIS`. JSON `methodology_risk_severities` = `["high","high","high","high","medium","medium","medium","low"]`.

Prose required: these risks do not claim experimental proof. They do not select hardware or a mapping. `v_toks_as_quality` restates TASK-18; it does not add a quality metric here.

### Deferred vision

Vision-encoder internals remain `UNKNOWN` and out of the primary map. Residual-stream interface only. No vision tok/s, encoder kernel metrics, or encoder working-set. JSON `vision_eval_deferred` true. The word `UNKNOWN` must appear in this section.

### Non-decisions (prose under SKU heading or a short paragraph at end of heading 9)

TASK-19 owns the **methodology** for measurements and for filling the sitting SKU table; it does not own a filled table in this increment. TASK-17 still owns mapping **selection** after future measurements (`ledger_open_question_mapping_winner_closed` false). TASK-15 still owns layout selection. TASK-12/13/14 still own fusion winners. TASK-14 still owns decode/prefill view selection. TASK-09 still owns artifact-boundary and ideal-sequence selection. TASK-08 still owns recipe winners. TASK-18 still owns quality/NLL. TASK-16 published identities \(N_w=32\), \(N_{\text{bank}}=32\) remain the only numeric SKU constants. State writes are not optional. Chunkwise GDN is not zero \(S\) traffic.

### Additional locked booleans and integers

- `benchmarks_run` false
- `toks_measured_here` false
- `experiments_run` false
- `methodology_defined` true
- `nll_measured_here` false
- `payloads_restreamed` false
- `gguf_payload_inspected` false
- `gguf_is_not_design_authority` true
- `quartz_inspected` false
- `llama_inspected` false
- `llama_is_not_design_authority` true
- `device_query_run` false
- `nsight_run` false
- `sku_table_filled` false
- `sku_fill_protocol_defined` true
- `hardware_selected` false
- `prompt_matrix_selected` false
- `reproducibility_protocol_selected` false
- `identity_protocol_defined` true
- `coverage_protocol_defined` true
- `time_accounting_protocol_defined` true
- `mapping_winner_selected` false
- `winner_selected_without_measurements` false
- `ledger_open_question_mapping_winner_closed` false
- `ledger_open_question_hardware_closed` false
- `decode_prefill_share_graph` true
- `decode_prefill_share_artifact` true
- `decode_prefill_share_metric_identity` false
- `activations_in_layout_scope` false
- `vision_eval_deferred` true
- `safetensors_is_source_not_runtime` true
- `hidden_size` 5120
- `intermediate_size` 17408
- `vocab_size` 248320
- `n_decoder_layers` 64
- `n_linear_layers` 48
- `n_full_layers` 16
- `n_mtp_blocks` 1
- `n_node_instances_complete` 135
- `n_stage_kinds` 9
- `headings` exactly the 12 names above
- `n_headings` 12

JSON `stage_kind_ids` equals the TASK-13/14 nine-id list: `embed_current`, `language_mixer`, `language_mlp`, `lm_head_primary`, `embed_next`, `mtp_mix`, `mtp_mixer`, `mtp_mlp`, `lm_head_mtp`.

JSON `full_attention_indices` = `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`.

JSON `bottleneck_labels` citation (TASK-06, still HYPOTHESIS; do not promote to MEASURED): include as substrings `weight`, `state`, `activation` in memory-channel prose; do not retabulate the six bottleneck classes as new measurements.

### JSON schema (heading 12)

Pretty-printed object, key order as emitted by the checker. Required keys (exact names; implementer may add only if this dossier is amended):

`authority`, `deliverable`, `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `model_T_max`, `dtype`, `mamba_ssm_dtype`, `n_language_mtp_parameters`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `mac_C_complete`, `mac_A_complete`, `mac_decode_complete_T1`, `mac_prefill_complete_T1`, `mac_decode_complete_T4096`, `mac_prefill_complete_T4096`, `example_T_values`, `example_T_is_not_prompt_matrix`, `N_w`, `N_bank`, `n_node_types`, `node_type_ids`, `n_embed_instances`, `n_gated_attn_instances`, `n_gated_delta_net_instances`, `n_mlp_instances`, `n_lm_head_instances`, `n_mtp_mix_instances`, `n_node_instances_complete`, `n_stage_kinds`, `stage_kind_ids`, `n_mappings`, `mapping_ids`, `n_mappings_selected`, `n_evaluation_criteria`, `evaluation_criterion_ids`, `evaluation_measured`, `n_metric_families`, `metric_family_ids`, `n_metric_classes`, `metric_class_ids`, `n_decode_metrics`, `decode_metric_ids`, `decode_metric_classes`, `n_prefill_metrics`, `prefill_metric_ids`, `n_kernel_metrics`, `kernel_metric_ids`, `n_memory_metrics`, `memory_metric_ids`, `n_e2e_metrics`, `e2e_metric_ids`, `e2e_primary_id`, `n_identity_fields`, `identity_field_ids`, `n_coverage_fields`, `coverage_field_ids`, `n_time_accounting_rules`, `time_accounting_rule_ids`, `n_sku_unknown_symbols`, `sku_unknown_symbols`, `n_traffic_channels`, `traffic_channel_ids`, `n_methodology_risks`, `methodology_risk_ids`, `methodology_risk_severities`, `methodology_risk_usefulness_label`, `n_methodology_high`, `n_methodology_medium`, `n_methodology_low`, `n_diagrams`, `diagram_ids`, `n_headings`, `headings`, `decode_work_identity`, `prefill_work_identity`, `decode_T_new`, `decode_incoming_state`, `prefill_incoming_state`, `first_token_in_prefill`, `decode_only_excludes_first_token`, `first_token_convention_id`, `ratio_requires_matching_identity`, `decode_is_not_prefill`, `decode_is_not_complete_request`, `t1_mac_equal_does_not_imply_equal_traffic`, `graph_envelope_is_not_leaf`, `missing_coverage_is_not_zero`, `kernel_cannot_pass_mapping`, `microbenchmark_cannot_pass_mapping`, `e2e_required_alongside_microbenchmarks`, `e2e_mixed_toks_is_not_decode_only`, `complete_request_vs_decode_only_forbidden`, `lower_bound_is_not_measured_bandwidth`, `peak_memory_is_not_live_set`, `public_sample_eval_output_is_quality`, `toks_is_not_quality_axis`, `identity_protocol_defined`, `identity_values_selected`, `coverage_protocol_defined`, `coverage_values_selected`, `time_accounting_protocol_defined`, `sampling_policy_selected`, `warmup_count_selected`, `clock_policy_selected`, `sku_table_filled`, `sku_fill_protocol_defined`, `sku_from_datasheet_is_not_measurement`, `hardware_selected`, `prompt_matrix_selected`, `reproducibility_protocol_selected`, `ledger_open_question_hardware_closed`, `mapping_winner_selected`, `winner_selected_without_measurements`, `ledger_open_question_mapping_winner_closed`, `n_mappings_per_node`, `decode_prefill_share_graph`, `decode_prefill_share_artifact`, `decode_prefill_share_metric_identity`, `activations_in_layout_scope`, `benchmarks_run`, `toks_measured_here`, `experiments_run`, `methodology_defined`, `nll_measured_here`, `payloads_restreamed`, `gguf_payload_inspected`, `gguf_is_not_design_authority`, `quartz_inspected`, `llama_inspected`, `llama_is_not_design_authority`, `device_query_run`, `nsight_run`, `achieved_occupancy_measured_here`, `vision_eval_deferred`, `safetensors_is_source_not_runtime`, `canonical_sentence_logical`, `canonical_sentence_methodology`, `canonical_sentence_e2e`, `canonical_sentence_open_question`, `canonical_sentence_winner`, `methodology_question_sentence`.

Integer JSON fields that are counts/widths/bytes/MAC are JSON ints. Booleans are JSON booleans. `full_attention_indices` and `example_T_values` are JSON arrays of ints. `first_token_convention_id` is the string `"ttft_in_prefill"`. `authority` is `"docs/architecture/plan.md"`. `deliverable` is `"docs/architecture/performance-validation.md"`. `dtype` is `"bfloat16"`. `mamba_ssm_dtype` is `"float32"`. `n_embed_instances` 2, `n_gated_attn_instances` 17, `n_gated_delta_net_instances` 48, `n_mlp_instances` 65, `n_lm_head_instances` 2, `n_mtp_mix_instances` 1. `n_language_mtp_parameters` 27320697856. `n_mappings_per_node` 3.

`headings` JSON array in exact order:

`Authority`, `Measurement convention`, `Decode metrics`, `Prefill metrics`, `Kernel metrics`, `Memory metrics`, `End-to-end metrics`, `Identity, coverage, and time accounting`, `SKU-fill protocol and unselected hardware`, `Methodology risks`, `Deferred vision`, `Machine-checkable summary JSON`.

### Tooling

Create `scripts/check_performance_validation.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, CUDA Python, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for layer counts, `full_attention_indices`, occupancy citations, and T=1/T=4096 MAC citations matching TASK-06 (use the locked MAC constants above; do not re-derive the full region MAC table). Duplicate TASK-11 `node_type_ids`, TASK-13/14 `stage_kind_ids`, TASK-17 `mapping_ids` / `evaluation_criterion_ids`, and TASK-16 `sku_unknown_symbols` as constants; do not import them. Do not open `models/Qwen3.8-27B-Q4_K_M.gguf` or any safetensor. Do not run `deviceQuery`, Nsight, or any GPU binary.

CLI (cwd = repository root):

```text
python3 scripts/check_performance_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_performance_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --performance-validation docs/architecture/performance-validation.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema above). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `max_position_embeddings` as `model_T_max`. Derived/cited: parameter/byte totals matching TASK-01/06, MAC T=1/T=4096 citations matching TASK-06 locked integers. Constant fields: canonical sentences, methodology sentence, metric-family / metric-class / metric-id / identity / coverage / time-accounting / SKU / risk lists, TASK-11 node types, TASK-13/14 stage kinds, TASK-17 mapping ids and criterion ids, TASK-16 sku_unknown_symbols.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--performance-validation PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all five canonical sentences verbatim, (5) `methodology_question_sentence` verbatim, (6) every `metric_family_ids`, `metric_class_ids`, `decode_metric_ids`, `prefill_metric_ids`, `kernel_metric_ids`, `memory_metric_ids`, `e2e_metric_ids`, `identity_field_ids`, `coverage_field_ids`, `time_accounting_rule_ids`, `methodology_risk_ids`, `node_type_ids`, `stage_kind_ids`, `mapping_ids`, `evaluation_criterion_ids`, `sku_unknown_symbols`, and `traffic_channel_ids` id present as a substring, (7) the diagram’s required IDs present **inside that mermaid fence**, (8) none of `TBD`, `TODO`, `???`, (9) the word `UNKNOWN` present in the Deferred vision section, (10) every locked document integer below present as a decimal or integer substring, (11) the words `HYPOTHESIS` and `not measurements` present, (12) none of the forbidden phrases: `selected mapping winner`, `winning kernel`, `winning mapping`, `SKU table is filled`, `selected hardware`, `selected prompt matrix`, `selected reproducibility protocol`, `should use this GPU`, `datasheet SKU is measured`, `microbenchmarks pass the mapping`, `reconstruction is sufficient`, `tok/s is the quality axis` (allow those substrings only inside `selects no mapping winner` / `not measurements` / `remain unselected` / canonical sentences / `not a selected winner`). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).
- Do not fail if the GGUF file is absent. Do not stat/open GGUF as a checker requirement.

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks TASK-06/13/14/17 integers and ids against those documents).

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
- `n_mappings==18`, `n_mappings_per_node==3`, `n_mappings_selected==0`
- `n_evaluation_criteria==7`, `evaluation_measured is False`
- `n_metric_families==5`, `n_metric_classes==5`
- `n_decode_metrics==4`, `n_prefill_metrics==4`, `n_kernel_metrics==5`, `n_memory_metrics==5`, `n_e2e_metrics==3`
- `n_identity_fields==15`, `n_coverage_fields==9`, `n_time_accounting_rules==5`
- `n_sku_unknown_symbols==17`, `n_traffic_channels==3`
- `n_methodology_risks==8`, `n_methodology_high==4`, `n_methodology_medium==3`, `n_methodology_low==1`
- `n_diagrams==1`, `n_headings==12`
- `node_type_ids` equals the TASK-11 six-id list; `mapping_ids` equals the TASK-17 18-id list; `sku_unknown_symbols` equals the TASK-16 17-id list; `stage_kind_ids` equals the TASK-13/14 nine-id list; `evaluation_criterion_ids` equals the TASK-17 seven-id list
- `first_token_in_prefill is True`, `decode_only_excludes_first_token is True`
- `t1_mac_equal_does_not_imply_equal_traffic is True`
- `graph_envelope_is_not_leaf is True`, `missing_coverage_is_not_zero is True`
- `microbenchmark_cannot_pass_mapping is True`, `e2e_required_alongside_microbenchmarks is True`, `kernel_cannot_pass_mapping is True`
- `e2e_primary_id == "e2e_latency_ms"`, `complete_request_vs_decode_only_forbidden is True`
- `example_T_is_not_prompt_matrix is True`
- `sku_table_filled is False`, `hardware_selected is False`, `prompt_matrix_selected is False`, `reproducibility_protocol_selected is False`, `ledger_open_question_hardware_closed is False`
- `mapping_winner_selected is False`, `winner_selected_without_measurements is False`, `ledger_open_question_mapping_winner_closed is False`
- `benchmarks_run is False`, `toks_measured_here is False`, `experiments_run is False`, `methodology_defined is True`
- `toks_is_not_quality_axis is True`, `nll_measured_here is False`
- `gguf_payload_inspected is False`, `quartz_inspected is False`, `llama_inspected is False`
- `device_query_run is False`, `nsight_run is False`, `achieved_occupancy_measured_here is False`
- `decode_prefill_share_metric_identity is False`
- `vision_eval_deferred is True`
- `n_embed_instances==2`, `n_gated_attn_instances==17`, `n_gated_delta_net_instances==48`, `n_mlp_instances==65`, `n_lm_head_instances==2`, `n_mtp_mix_instances==1`

Locked document integers that must appear as decimal substrings in the markdown (in addition to ids): `5120`, `17408`, `248320`, `64`, `48`, `16`, `262144`, `27320697856`, `54641395712`, `52098598912`, `27433238528`, `208896`, `27433447424`, `28288876544`, `114119319486464`, `135`, `32`.

### Stage split

- **Implementation** writes `scripts/check_performance_validation.py` **and** `docs/architecture/performance-validation.md` (five metric families, four decode / four prefill / five kernel / five memory / three e2e metrics, 15 identity fields, 9 coverage fields, 5 time-accounting rules, 17 SKU-UNKNOWN symbols with fill protocol and unfilled table, eight HYPOTHESIS methodology risks, JSON fence). Runs `--json` and `--performance-validation` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads. Does not open GGUF. Does not run benchmarks, `deviceQuery`, Nsight, or tok/s collection.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-18 / `quantization-validation.md` / TASK-17 / `cuda-design-space.md` style), Authority table links to this dossier / work-and-traffic / decode-plan / prefill-plan / cuda-design-space / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, metric ids, severities, or Mermaid node IDs. Does not edit TASK-06/13/14/17 artifacts.
- **Verification** independently re-runs focused commands, recomputes layer counts / `full_attention_indices` / `model_T_max` from sitting `text_config` (not from JSON echo), spot-checks cited TASK-06/13/14/17 integers and ids against `docs/architecture/work-and-traffic.md`, `docs/architecture/decode-plan.md`, `docs/architecture/prefill-plan.md`, and `docs/architecture/cuda-design-space.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-authority, no winner phrases, no `plan.md` or ledger edit, no payload I/O, no GGUF open, no GPU run, that every methodology-risk row is labelled HYPOTHESIS, that microbenchmarks cannot pass a mapping, that e2e is required alongside microbenchmarks, that SKU / hardware / prompt matrix / reproducibility remain unselected, and that no implementation benchmarks were run. The three ledger completion criteria are closed by the methodology. TASK-17 mapping **selection** and the hardware/prompt/repro open question remain open.

- Invariants:
  - Twelve level-2 headings in the locked order; five canonical sentences plus `methodology_question_sentence` verbatim; 5 metric families; 5 metric classes; 4+4+5+5+3 metrics; 15 identity fields; 9 coverage fields; 5 time-accounting rules; 17 SKU-UNKNOWN symbols; 8 methodology risks; one Mermaid flowchart with required IDs.
  - Prefill/decode share one graph and one artifact, not one metric identity; first token in prefill; T=1 MAC equality does not imply equal traffic.
  - Microbenchmarks cannot pass a mapping; e2e is required alongside them; e2e primary is latency.
  - Graph envelopes are not leaf time; missing coverage is not zero excess; TASK-06 bounds are not measured bandwidth.
  - `sku_table_filled` false; all hardware/prompt/repro selected flags false; `toks_measured_here` false; `mapping_winner_selected` false.
  - Tok/s is not a quality axis. Logical values do not imply allocation. Vision encoder remains unexpanded.
- Rejected alternatives:
  - Running nsys/ncu/`deviceQuery`/tok/s collection in Phase 1: rejected; methodology only; `benchmarks_run` false.
  - Filling SKU from a datasheet or sitting device now: rejected; open question stays open; `sku_table_filled` false; `v_datasheet_sku`.
  - Selecting a GPU, prompt matrix, warmup count, or clock lock: rejected; ledger open question stays open.
  - Selecting a TASK-17 mapping winner from named metrics: rejected; metrics are not measurements; `microbenchmark_cannot_pass_mapping`; `mapping_winner_selected` false.
  - Treating kernel microbenchmarks as sufficient acceptance: rejected; ledger checkbox 2 requires e2e alongside microbenchmarks.
  - Reporting mixed \((T+N_{\mathrm{gen}})/t\) as decode-only tok/s: rejected; `v_mixed_identity`; `complete_request_vs_decode_only_forbidden`.
  - Equating T=1 prefill with decode: rejected; `t1_mac_equal_does_not_imply_equal_traffic`.
  - Ranking from graph envelopes without leaf coverage: rejected; `graph_envelope_is_not_leaf`.
  - Using tok/s as TASK-18 Pareto \(Y\): rejected; `toks_is_not_quality_axis`.
  - Recopying TASK-06 MAC tables or TASK-17 eighteen mapping definitions as new studies: rejected; cite, do not replace.
  - Opening GGUF or inspecting Quartz/llama.cpp timers: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–18.
  - Editing frozen TASK-06/13/14/16/17/18 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py`.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/performance-validation.md` exists and follows the heading list above.
  - Decode metrics (4), prefill metrics (4), and kernel metrics (5) are named with locked identities and formulas; memory metrics (5) are named; end-to-end metrics (3) include latency as primary.
  - End-to-end measurement is required alongside microbenchmarks; microbenchmarks cannot pass a mapping.
  - No implementation benchmarks, SKU numbers, tok/s tables, `deviceQuery`, or Nsight in this study task (`benchmarks_run` false; `toks_measured_here` false; `sku_table_filled` false).
  - Exact hardware, prompt matrix, and reproducibility protocol remain unselected; TASK-17 mapping winner remains unselected.
  - Methodology risks (8) are tabulated as HYPOTHESIS and do not claim experimental proof or a selected winner.
  - Five canonical sentences plus `methodology_question_sentence` verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; no payload re-stream; no `plan.md` or ledger edit; no MEASURED tok/s.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_performance_validation.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_performance_validation.py
python3 scripts/check_performance_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_performance_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --performance-validation docs/architecture/performance-validation.md
```

- Candidate quality: not required — no model execution, NLL, or OPT-058; this increment is performance-methodology documentation. Decode/prefill **metrics** are protocol, not measured tok/s.
- Repository-wide commands:

```sh
test -f docs/architecture/performance-validation.md
python3 -m py_compile scripts/check_performance_validation.py
python3 scripts/check_performance_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --performance-validation docs/architecture/performance-validation.md
```

Do not run Ruff, pytest, CMake, CUDA, Nsight, or `deviceQuery`; this increment does not introduce those gates. Do not run `scripts/analyze_bf16_tensors.py`. Do not open GGUF.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/performance-validation.md` (create)
  - `scripts/check_performance_validation.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-06/13/14/17 deliverables unchanged)
- Definition of done: performance-validation document published with locked decode, prefill, kernel, memory, and end-to-end metric identities; e2e required alongside microbenchmarks; no implementation benchmarks run; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-19 completion checkboxes can be marked at delivery; exact hardware, prompt matrix, reproducibility protocol, sitting SKU numbers, and mapping winner remain unselected.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T16:25:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-19.md`. Coupled IDs `none`. Document structure (12 headings), five canonical sentences plus methodology sentence, five metric families, five metric classes, four decode / four prefill / five kernel / five memory / three e2e metrics, 15 identity fields, 9 coverage fields, 5 time-accounting rules, 17 SKU-UNKNOWN symbols with fill protocol and unfilled table, eight HYPOTHESIS methodology risks, stdlib checker `scripts/check_performance_validation.py`, JSON schema, and acceptance commands are closed. Ledger open question left open (exact hardware / prompt matrix / reproducibility protocol). TASK-17 mapping **selection** remains unselected. `docs/architecture/performance-validation.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit. No benchmarks.
- Performance evidence applied: N/A — evaluation-methodology documentation; no sink ranking, no measured tok/s, no SKU fill

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_performance_validation.py` (stdlib checker: argparse/json/math/re/sys/pathlib; live `text_config` occupancy + locked MAC/metric/SKU/risk constants; `--json` and `--performance-validation` gates; no other `check_*.py` imports; no GGUF/safetensor/GPU).
  - Created `docs/architecture/performance-validation.md` (12 locked headings; five canonical sentences plus `methodology_question_sentence`; 5 metric families / 5 classes; 4 decode / 4 prefill / 5 kernel / 5 memory / 3 e2e metrics; 15 identity fields; 9 coverage fields; 5 time-accounting rules; 17 SKU-UNKNOWN symbols with fill protocol and unfilled table; 8 HYPOTHESIS methodology risks; one Mermaid flowchart with DECODE/PREFILL/KERNEL/MEMORY/IDENTITY/E2E/SKU/OPENQ; JSON fence equal to live `--json`).
  - Did not edit `plan.md`, `task_ledger.md`, TASK-01–18 deliverables, or run benchmarks / `deviceQuery` / Nsight / tok/s collection. No commit.
- Commands:
  - `python3 -m py_compile scripts/check_performance_validation.py` — exit 0
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; 148-key object; `benchmarks_run` false; `sku_table_filled` false; `mapping_winner_selected` false; `methodology_defined` true
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md` — exit 0
- UTC/time/tokens/cost: `2026-09-20T16:38:33Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent; parent/inherit mapping)
- Changes and evidence:
  - `docs/architecture/performance-validation.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-18 / [`quantization-validation.md`](../quantization-validation.md) / TASK-17 / [`cuda-design-space.md`](../cuda-design-space.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`work-and-traffic.md`](../work-and-traffic.md) (TASK-06), [`decode-plan.md`](../decode-plan.md) (TASK-13), [`prefill-plan.md`](../prefill-plan.md) (TASK-14), [`cuda-design-space.md`](../cuda-design-space.md) (TASK-17), [`cuda-hardware-model.md`](../cuda-hardware-model.md) (TASK-16) via TASK-17, sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_performance_validation.py`](../../../scripts/check_performance_validation.py), and plan evidence policy in [`plan.md`](../plan.md) (OBSERVED / DERIVED / HYPOTHESIS / UNKNOWN). Twelve required `##` headings, five canonical sentences plus `methodology_question_sentence`, one Mermaid flowchart with DECODE/PREFILL/KERNEL/MEMORY/IDENTITY/E2E/SKU/OPENQ, and the JSON fence left unchanged (locked integers, metric ids, severities, Mermaid node IDs untouched). No edits to TASK-06/13/14/17 deliverables, `plan.md`, or `task_ledger.md`.
- Commands:
  - `test -f docs/architecture/performance-validation.md` — pass (exit 0).
  - `python3 -m py_compile scripts/check_performance_validation.py` — pass (exit 0).
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 148 keys; `benchmarks_run` false; `sku_table_filled` false; `mapping_winner_selected` false; `methodology_defined` true; JSON fence source unchanged).
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md` — pass (exit 0; twelve headings, JSON fence, one Mermaid flowchart, five canonical sentences plus methodology sentence, locked ids/integers, `HYPOTHESIS` / `not measurements` / `unresolved`; banner did not break the check).
  - Independent JSON fence spot-check (live `--json` object equals fenced JSON in `performance-validation.md`, 148 keys deep-equal after parse) — pass.
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T16:39:16Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `composer-2.5` (verifier subagent; parent/inherit mapping)
- Diff review:
  - New: `docs/architecture/performance-validation.md`, `scripts/check_performance_validation.py`, `docs/architecture/tasks/TASK-19.md`.
  - Modified: `docs/architecture/task_ledger.md` — status only `TODO` → `IN PROGRESS` (workflow admission; no completion checkboxes, history, or criteria text changed). Acceptable.
  - Unchanged: `docs/architecture/plan.md`, TASK-06/13/14/17/18 deliverables, frozen upstream docs.
  - Scope matches dossier: methodology document + stdlib checker only; no benchmarks, GPU, GGUF, Quartz/llama inspection, mapping winner, or SKU fill.
- Independent raw-record checks:
  - Recomputed sitting `text_config` from `.cache/authorities/qwen3.8-27b-transformers/config.json`: `hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, 64 decoder layers, 48 linear + 16 full, `full_attention_indices` `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`, `mtp_num_hidden_layers` 1, `model_T_max` 262144 — all match live `--json` (not JSON echo).
  - MAC/byte citations spot-checked against [`work-and-traffic.md`](../work-and-traffic.md): `mac_C_complete` 27433238528, `mac_A_complete` 208896, T=1/T=4096 decode/prefill MACs, `weight_bytes_unique_non_embed` 52098598912, `weight_bytes_language_mtp_excl_vision` 54641395712 — match.
  - Node instances 2+17+48+65+2+1=135 match [`decode-plan.md`](../decode-plan.md) TASK-11 complete counts.
  - `mapping_ids` (18) deep-equal TASK-17 JSON fence in [`cuda-design-space.md`](../cuda-design-space.md).
  - `stage_kind_ids` (9) match decode/prefill-plan locked lists.
  - First fenced JSON in deliverable deep-equal fresh `--json` (148 keys).
  - Twelve locked `##` headings, five canonical sentences + `methodology_question_sentence`, one Mermaid flowchart with DECODE/PREFILL/KERNEL/MEMORY/IDENTITY/E2E/SKU/OPENQ, eight methodology risks all labelled HYPOTHESIS — confirmed in document and checker gates.
  - Forbidden states absent: no MEASURED tok/s tables, `benchmarks_run`/`toks_measured_here`/`sku_table_filled`/`mapping_winner_selected` false; hardware/prompt/repro/mapping winner remain unselected; `device_query_run`/`nsight_run` false.
  - Checker uses stdlib only (`argparse`, `json`, `math`, `re`, `sys`, `pathlib`); no imports from other `check_*.py`.
- Commands:
  - `test -f docs/architecture/performance-validation.md` — exit 0
  - `python3 -m py_compile scripts/check_performance_validation.py` — exit 0
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0; 148-key object; `benchmarks_run` false; `sku_table_filled` false; `mapping_winner_selected` false; `methodology_defined` true; `ledger_open_question_hardware_closed` false; `ledger_open_question_mapping_winner_closed` false; `e2e_required_alongside_microbenchmarks` true; `microbenchmark_cannot_pass_mapping` true; metric counts 4/4/5/5/3; identity/coverage/time fields 15/9/5; SKU symbols 17; methodology risks 8
  - `python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md` — exit 0
  - Independent Python deep-equal: fenced JSON == live `--json` — pass
- Formatting changed files: none (`uv run ruff format .` not run — out of scope per dossier)
- Verdict: **PASS** (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T16:42:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-19 only; coupled IDs `none`
- Outcome: TASK-19 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T16:45:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/performance-validation.md` (twelve locked headings, five canonical sentences plus methodology sentence, five metric families and five metric classes, four decode / four prefill / five kernel / five memory / three end-to-end metrics, fifteen identity fields, nine coverage fields, five time-accounting rules, seventeen SKU-UNKNOWN symbols with fill protocol and unfilled table, eight HYPOTHESIS methodology risks, one Mermaid flowchart, JSON fence equal to live `--json`); `scripts/check_performance_validation.py` stdlib checker; `plan.md` unchanged; exact hardware, prompt matrix, reproducibility protocol, sitting SKU numbers, and mapping winner remain unselected (`hardware_selected` false; `prompt_matrix_selected` false; `reproducibility_protocol_selected` false; `ledger_open_question_hardware_closed` false; `mapping_winner_selected` false; `ledger_open_question_mapping_winner_closed` false; `benchmarks_run` false; `sku_table_filled` false; `methodology_defined` true; `e2e_required_alongside_microbenchmarks` true)
- Candidate measured delta: N/A — evaluation-methodology documentation
- Shipping delta: N/A
- Quality result: not required
- Evidence completeness: N/A for performance-evidence checks
- Throughput delta (when applicable): N/A
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: **yes**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands; GGUF file is not required now; no GPU required
