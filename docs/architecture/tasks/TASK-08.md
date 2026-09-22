# TASK-08 — Design the custom quantization research space

## Control

- Primary ID: `TASK-08`
- Coupled IDs: `none`
- Dependencies: `TASK-05`, `TASK-06`, `TASK-07` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Cover requested design dimensions and metadata/compute implications; Propose family-specific candidate policies without selecting winners; Record quality and decode-complexity risks.

## Goal and boundaries

Produce `docs/architecture/quantization-design-space.md` as the Phase 1 **tensor-specific custom quantization research space** for Qwen3.8-27B. Close the three ledger completion criteria with **OBSERVED** inventory/config occupancy, **MEASURED** TASK-05 citations (not remeasured here), **DERIVED** metadata/byte illustrations, and **HYPOTHESIS** candidate-policy rationale plus quality and decode-complexity risks. Do **not** select bit-width, grouping, scale, or outlier winners, and do **not** close the ledger open question (Pareto frontier remains for TASK-18).

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - BF16 Transformers checkpoint at `.cache/authorities/qwen3.8-27b-transformers` remains the high-precision **source**. Family taxonomy and occupancy come from `docs/architecture/model-inventory.md` (TASK-01). Source-weight distributions come from `docs/architecture/bf16-tensor-analysis.md` (TASK-05). Weight/state traffic and $I=F/B$ identities come from `docs/architecture/work-and-traffic.md` (TASK-06). Precision roles and sensitive-op hypotheses come from `docs/architecture/numerical-sensitivity.md` (TASK-07).
  - Label claims `OBSERVED` (inventory/config/family occupancy), `MEASURED` (TASK-05 numbers cited, not recomputed from payloads), `DERIVED` (metadata byte formulas, occupancy products, illustration ratios), or `HYPOTHESIS` (every candidate-set rationale, every quality risk, every decode-complexity risk). `UNKNOWN` only for vision-encoder internals deferred here. No new `MEASURED` payload statistics. No `MEASURED` quality or tok/s.
  - GitHub Markdown math. Cite TASK-02 equation tags where a family feeds a map, TASK-05 family ids, TASK-06 traffic identities, TASK-07 sensitive-op ids. Do not rewrite forward math, re-stream safetensor payloads, recopy TASK-05 1199-row JSON, or redraw the TASK-03 DAG.
  - Allowed evidence: TASK-01 inventory, TASK-05 analysis document (citations), TASK-06 work/traffic, TASK-07 numerical sensitivity, sitting `config.json` `text_config`, plan evidence vocabulary, and general scalar-quantization / IEEE field-width material. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf` as a design constraint. GGUF Q4_K_M remains a future black-box Pareto **reference** (TASK-18), not a candidate recipe.
- Non-goals:
  - No selected winners, Pareto frontier, “should be 4-bit”, or recommended grouping (open question stays open).
  - No reconstruction error, NLL, calibration corpora, GPTQ/AWQ/Hessian scales, or quality experiments (TASK-18).
  - No runtime artifact byte sequences, packing layouts, or portable-vs-backend format (TASK-09).
  - No compiler stages or profile ownership (TASK-10).
  - No activation working-dtype or GEMM-accumulator recipe (TASK-07 left those undecided; mixed-precision CUDA recipes are TASK-17).
  - No kernels, MMA shapes, CUDA dtypes, or sitting-GPU numbers (TASK-16/17/19).
  - No fusion, buffer reuse, physical allocation, or semantic-graph contracts (TASK-11/12).
  - No decode/prefill **schedules** (TASK-13/14) and no physical layouts (TASK-15).
  - No new operators and no vision-encoder internals.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–07). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `bf16-tensor-analysis.md`, `work-and-traffic.md`, `numerical-sensitivity.md`, `model-inventory.md`, `model-semantics.md`, `dataflow.md`, or `lifetime-and-state.md`.
  - Do not stream safetensor payloads and do not import `scripts/analyze_bf16_tensors.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib quantization-design-space checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-22` — central path: numerical sensitivity → precision strategy → **custom quantization** → layouts/compiler.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint is the authority for weights, tensor statistics, and quantization research; GGUF is not an architectural constraint; Quartz/llama.cpp inspection forbidden.
- `docs/architecture/plan.md:41-45` — the study may define a new tensor-specific quantization system, bit widths, grouping, and outlier treatment; co-design with layout/graph/kernels, none treated as a fixed generic format.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:85-88` — TASK-08 develops tensor-specific quantization experiments after TASK-06/07, followed by TASK-09/10.
- `docs/architecture/task_ledger.md` TASK-08 row — produces `docs/architecture/quantization-design-space.md`; purpose is tensor-specific custom quantization experiments from BF16 source evidence; open question (Pareto frontier) is **not** closed here; completion is design dimensions and metadata/compute implications, family-specific candidate policies without winners, quality and decode-complexity risks.
- `docs/architecture/task_ledger.md` TASK-05 established results — MEASURED family/layer distributions; `pooled_language_mtp.absmax=19.25`; `linear_attn.dt_bias` sets that absmax; embed holds the only pooled zeros; directional row/col ratios documented; no quality conclusions.
- `docs/architecture/task_ledger.md` TASK-06 established results — language+MTP unique weight bytes `54641395712`; unique non-embed `52098598912`; MLP vs weights $I=1$; `lm_head` 2542796800 B `vocab_memory`; GDN vs $S$ $I=0.75$; $S$ 150994944 B/step; bottleneck **labels** HYPOTHESIS.
- `docs/architecture/task_ledger.md` TASK-07 established results — four precision roles; `param_bf16` further narrowing is this task; `gdn_alpha_beta` / `residual_stream` / `s_below_f32` / `state_kv_bf16` high (HYPOTHESIS); activation dtype not decided.
- `docs/architecture/bf16-tensor-analysis.md` — MEASURED citations locked below (pooled absmax, family absmax/rms/percentiles/outlier fractions, directional max/median ratios). Canonical non-conclusion: measurements are not bit/grouping/quality recommendations.
- `docs/architecture/work-and-traffic.md` — DERIVED unique weight bytes by family; embed is gather (0 MAC, 10240 B/row); decode GEMM $I=1$ vs weights; state volumes for $K,V,C,S$.
- `docs/architecture/numerical-sensitivity.md` — roles `param`/`activation`/`accum`/`state`; 20 sensitive ops; further weight narrowing/grouping/outlier policy assigned here; `mamba_ssm_dtype` is not an activation dtype.
- `docs/architecture/model-inventory.md` — 42 level-2 families; language+MTP 866 tensors / 27320697856 parameters / 54641395712 BF16 bytes; all checkpoint tensors BF16; embeddings and `lm_head` untied `(248320, 5120)`.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads.
- `scripts/check_numerical_sensitivity.py` / `scripts/check_work_and_traffic.py` — checker-style precedent. TASK-08’s checker is a sibling; do not import them.
- `docs/architecture/task_ledger.md` TASK-18 — reconstruction diagnostics, teacher-forced comparisons, and Q4_K_M as a future black-box reference; not this increment.

## Performance evidence

N/A — quantization **research-space** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Metadata byte counts are DERIVED arithmetic. Quality and decode-complexity **labels** are HYPOTHESIS, not MEASURED error or tok/s. Do not apply the performance-evidence checklist to rank kernels or claim quality impact.

- Measurement identity: N/A (no engine binary). TASK-05 payload identity is already locked in that analysis; this task **cites** those numbers.
- Metric class: N/A
- Coverage: N/A for GPU graphs. Research-space coverage is all 42 level-2 families plus four token-persistent state ids.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` occupancy product disagrees with TASK-01 family parameter counts, stop and fail closed. If a cited TASK-05 number disagrees with `bf16-tensor-analysis.md`, stop and fail closed (do not remeasure).
- Claim types: occupancy `observed`/`derived`; TASK-05 citations `measured` (historical to TASK-05, not new); metadata illustrations `derived`; candidate rationale and all risks `hypothesis`.
- Target/guard roles: not opted in.
- Evidence completeness: N/A for performance-evidence checks. Design-space completeness is the four dimensions, metadata formulas, 16 policy families, 22 recipes, 17 quality risks, 9 decode-complexity risks.
- Screen eligibility: N/A
- Shipping evidence: N/A

## Implementation decisions

### Authority for quantization claims

If an occupancy product would disagree with TASK-01 / sitting `text_config`, or a cited absmax/ratio/outlier fraction would disagree with TASK-05, or a weight/state byte would disagree with TASK-06, or a sensitive-op id would disagree with TASK-07, the earlier document wins and this one is wrong.

- Prefill and decode share **one** research space. Only access pattern (gather vs full GEMM; unique weight read vs $T$-reuse) changes decode-complexity hypotheses, not the candidate recipe lists.
- Primary object is language+MTP **checkpoint parameters** (role `param`). Secondary object is token-persistent **state** $K,V,C,S$ (role `state`), because TASK-06 traffic and TASK-07 `s_below_f32` / `state_kv_bf16` make state a quantization experiment surface. Activations and accumulators are **not** family policies here.
- Algebraic equivalents in TASK-02 are the same real map; quantization acts on stored elements, not on a second mathematical model.
- Do not inspect Quartz or llama.cpp to “confirm” GGUF groupings.
- Do not re-stream payloads. Cite TASK-05; do not invent new histograms.
- Layer absmax variation (TASK-05) is absorbed by `per_tensor` / finer grouping recipes, not by per-layer format winners.

### Deliverable structure (`docs/architecture/quantization-design-space.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every occupancy number is `OBSERVED` or `DERIVED`. Every TASK-05 cell is labelled `MEASURED` (citation). Every metadata illustration is `DERIVED`. Every candidate-set rationale cell and every risk severity is `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, inventory, TASK-05/06/07, config, checker; evidence labels; in-scope (language+MTP params + state experiments) vs deferred (vision encoder; activation dtype). State that the document specifies a research space, not a selected quantizer, kernel, or runtime format.
2. **Research-space convention** — the three canonical sentences below plus the metadata sentence; what a candidate is; GGUF is not a recipe; Pareto stays open.
3. **Design dimensions** — the four locked dimensions and their vocabularies; recipe table (22 ids); validity rules (IEEE vs integer).
4. **Metadata and compute implications** — byte formulas; instantiated MLP and unique-non-embed illustrations; scale-storage candidates; dequant cost as a count (DERIVED) vs decode impact (HYPOTHESIS); packing deferred to TASK-09.
5. **Source evidence** — compact TASK-05/06/07 citation tables used to **shape** family candidate sets, not to pick winners.
6. **Policy families** — 16 families; complete mapping of 42 level-2 ids + `K_state`/`V_state`/`C_state`/`S`.
7. **Family-specific candidate policies** — `family_candidates` lists; no winner column; `keep_source` on every family except `vision_deferred`.
8. **Quality and decode-complexity risks** — 17 quality rows + 9 decode-complexity rows (all HYPOTHESIS); one Mermaid summary (diagram 1 of 1).
9. **Deferred vision** — residual-stream interface only; empty candidate list.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Research-space convention**, include these **three canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Candidate policies in this document are a research space, not selected winners.

> Quality and decode-complexity risks in this document are hypotheses, not measurements.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_candidates` = sentence 2; `canonical_sentence_hypothesis` = sentence 3.

Immediately after those three, include this **fourth required sentence verbatim**:

> Metadata byte counts in this document are DERIVED lower bounds on scale and code storage, not a packed runtime layout.

JSON: `canonical_sentence_metadata` = that sentence.

Bullets required under that heading:

- Four design dimensions are complete for this task: `bit_width`, `grouping`, `scale`, `outlier_policy`.
- A candidate recipe is one valid 4-tuple from those dimensions; listing it does not select it.
- Fan-out ≠ must-store still holds; a quantized family is not a CUDA allocation.
- The ledger open question (Pareto frontier) remains open for TASK-18.
- `models/Qwen3.8-27B-Q4_K_M.gguf` is not a candidate recipe and not a grouping authority.

### Design dimensions (lock)

JSON array `design_dimensions` in this order: `bit_width`, `grouping`, `scale`, `outlier_policy`. JSON `n_design_dimensions` = 4.

#### `bit_width` / element formats

JSON array `formats` in this order:

| JSON | Kind | Meaning |
| --- | --- | --- |
| `source` | control | Family conceptual source: BF16 for every checkpoint tensor and for $K,V,C$; F32 for $S$. Recipe `keep_source`. |
| `bf16` | IEEE-like | Explicit BF16 store. Used only by recipe `narrow_bf16` on `state_s` (F32 conceptual → BF16). |
| `fp8_e4m3` | IEEE-like | 8-bit float with 4-bit exponent / 3-bit trailing significand (de facto E4M3). Not a CUDA dtype decision. |
| `int8` | integer | Signed 8-bit codes with a scale. |
| `int6` | integer | Signed 6-bit codes with a scale. |
| `int4` | integer | Signed 4-bit codes with a scale. |
| `int3` | integer | Signed 3-bit codes with a scale. |
| `int2` | integer | Signed 2-bit codes with a scale. |

JSON `ieee_like_formats` = `["source", "bf16", "fp8_e4m3"]`. JSON `integer_formats` = `["int8", "int6", "int4", "int3", "int2"]`.

Do not add `int5`, `fp8_e5m2`, or GGUF type names. `fp8_e5m2` is a rejected alternative (redundant with `fp8_e4m3` for shaping this space; TASK-18 may add it). Integer codes are two’s-complement around zero for symmetric scales; asymmetric recipes add a zero-point.

#### `grouping`

JSON array `grouping_ids` in this order:

| JSON | Meaning |
| --- | --- |
| `none` | No groups; IEEE-like / `keep_source` only. |
| `per_tensor` | One scale (and optional zero-point) for the whole tensor. |
| `per_row` | One scale per row of $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ (TASK-05 axis 0 = $d_\text{out}$, $y=Wx$). Rank-1: not used. |
| `per_col` | One scale per column (axis 1 = $d_\text{in}$). Rank-1: not used. |
| `block_g32` | Contiguous groups of 32 elements in **row-major** storage order. |
| `block_g64` | Groups of 64, same order. |
| `block_g128` | Groups of 128, same order. |

Last group may be shorter (`ragged_last_group` true). If $g$ exceeds the spanned axis length, that span is one group (`group_clips_to_axis` true). `linear_attn.conv1d` uses the TASK-05 squeezed rank-2 `(10240, 4)`: `per_row` = one scale per output channel.

JSON `block_group_sizes` = `[32, 64, 128]`.

#### `scale`

JSON array `scale_ids` in this order:

| JSON | Meaning |
| --- | --- |
| `none` | No extra scale; IEEE-like formats only. |
| `symmetric_absmax` | $s = \mathrm{absmax}/q_{\max}$ on the group; signed symmetric. |
| `symmetric_rms` | $s$ from TASK-05 RMS (group RMS × a locked constant 3 is **not** introduced; use $s=\mathrm{rms}/q_{\max}$ as the named candidate). |
| `asymmetric_minmax` | Affine scale + zero-point from group min and max. |
| `percentile_p99` | $s = p_{99}(|w|)/q_{\max}$ using the TASK-05 nearest-rank definition on that group at experiment time; family-level $p_{99}$ citations below only **motivate** including this id. |

JSON `scale_storage_bytes_candidates` = `[2, 4]` (F16 vs F32 scale storage). Illustrations below use 2-byte scales. Choosing 2 vs 4 is TASK-09 packing, not a winner here.

Calibration / GPTQ / AWQ / learned scales are **not** scale ids (TASK-10/18).

#### `outlier_policy`

JSON array `outlier_ids` in this order:

| JSON | Meaning |
| --- | --- |
| `none` | All elements use the group code. |
| `clip` | Clamp $\lvert w\rvert$ to the representable range (destroys the tail). |
| `extract_high` | Store elements with $\lvert w\rvert > 6\times p_{50}$ (TASK-05 6× definition, group-local at experiment time) in a sparse BF16 sidecar; remaining codes use the integer recipe. |
| `mixed_group` | Groups whose absmax exceeds $10\times$ the tensor-median group absmax use the next wider integer format (`int4`→`int8`, `int3`→`int4`); others stay narrow. |

#### Validity rules (lock)

- IEEE-like format ⇒ `grouping` ∈ {`none`, `per_tensor`}, `scale` = `none`, `outlier` = `none`. Exception: `fp8_e4m3` uses `per_tensor` + `none` + `none` (recipe `fp8_tensor`). `keep_source` and `narrow_bf16` use `none` + `none` + `none`.
- Integer format ⇒ `grouping` ≠ `none`, `scale` ≠ `none`.
- `clip` and `extract_high` and `mixed_group` apply only to integer recipes.
- Rank-1 families must not use `per_row` or `per_col`.

The 22 recipes below are the **only** valid combinations this document enumerates. Families list subsets. The Cartesian product of the dimension vocabularies is **not** the experiment grid; the named recipes are.

### Candidate recipes (lock)

JSON array `candidate_recipe_ids` in this exact order (22 ids). Parallel arrays `recipe_formats`, `recipe_groupings`, `recipe_scales`, `recipe_outliers`. JSON `n_candidate_recipes` = 22.

| id | format | grouping | scale | outlier |
| --- | --- | --- | --- | --- |
| `keep_source` | `source` | `none` | `none` | `none` |
| `narrow_bf16` | `bf16` | `none` | `none` | `none` |
| `fp8_tensor` | `fp8_e4m3` | `per_tensor` | `none` | `none` |
| `i8_tensor` | `int8` | `per_tensor` | `symmetric_absmax` | `none` |
| `i8_row` | `int8` | `per_row` | `symmetric_absmax` | `none` |
| `i8_row_asym` | `int8` | `per_row` | `asymmetric_minmax` | `none` |
| `i8_g32` | `int8` | `block_g32` | `symmetric_absmax` | `none` |
| `i6_row` | `int6` | `per_row` | `symmetric_absmax` | `none` |
| `i4_row` | `int4` | `per_row` | `symmetric_absmax` | `none` |
| `i4_col` | `int4` | `per_col` | `symmetric_absmax` | `none` |
| `i4_g32` | `int4` | `block_g32` | `symmetric_absmax` | `none` |
| `i4_g64` | `int4` | `block_g64` | `symmetric_absmax` | `none` |
| `i4_g128` | `int4` | `block_g128` | `symmetric_absmax` | `none` |
| `i4_g128_p99` | `int4` | `block_g128` | `percentile_p99` | `none` |
| `i4_g128_rms` | `int4` | `block_g128` | `symmetric_rms` | `none` |
| `i4_g128_extract` | `int4` | `block_g128` | `symmetric_absmax` | `extract_high` |
| `i4_g32_mixed` | `int4` | `block_g32` | `symmetric_absmax` | `mixed_group` |
| `i4_row_extract` | `int4` | `per_row` | `symmetric_absmax` | `extract_high` |
| `i4_clip` | `int4` | `block_g128` | `symmetric_absmax` | `clip` |
| `i3_g32` | `int3` | `block_g32` | `symmetric_absmax` | `none` |
| `i3_g32_extract` | `int3` | `block_g32` | `symmetric_absmax` | `extract_high` |
| `i2_g32_extract` | `int2` | `block_g32` | `symmetric_absmax` | `extract_high` |

Shared list `C_GEMM_LARGE` (15 ids, this order) reused by `linear_large_proj`, `attn_qkv`, `attn_out`, `mlp_down`:

`keep_source`, `fp8_tensor`, `i8_row`, `i8_row_asym`, `i6_row`, `i4_row`, `i4_col`, `i4_g32`, `i4_g64`, `i4_g128`, `i4_g128_p99`, `i4_g128_rms`, `i4_g128_extract`, `i4_g32_mixed`, `i3_g32`

JSON `c_gemm_large` equals that list.

### Metadata and compute implications (lock)

Let $n$ be the element count, $b$ the integer code width in bits, $g$ the group size, $s$ the scale storage bytes, $z$ the zero-point bytes ($z=0$ for symmetric, $z=1$ for asymmetric illustrations).

$$
B_\text{payload}=\frac{n\,b}{8},\qquad
n_g=\Bigl\lceil\frac{n}{g}\Bigr\rceil,\qquad
B_\text{meta}=n_g(s+z),\qquad
B=B_\text{payload}+B_\text{meta}.
$$

IEEE-like recipes: $B = n \times$ element bytes (`bytes_bf16` = 2, `bytes_f32` = 4, `bytes_fp8` = 1). No $B_\text{meta}$.

These are **storage lower bounds**, not packed layouts (TASK-09). 3-bit and 6-bit payloads use exact $nb/8$ in the formula even when that is not an integer byte per element; `d_3bit_pack` records the HYPOTHESIS unpack cost. JSON `payload_formula_allows_fractional_bytes` true.

Dequant arithmetic **count** if dequant-then-MAC (DERIVED): one multiply per element ($n$) plus $n_g$ scale loads. Fused-dequant-in-contraction extra MAC is **not** claimed; that is a TASK-17 mapping. Decode impact of those counts is HYPOTHESIS (`d_weight_dequant`).

**Illustration A — language MLP**, $n=17112760320$ (DERIVED: $3\times 64\times 17408\times 5120$), int4, $s=2$, $z=0$, vs BF16 $B_\text{bf16}=34225520640$:

| $g$ | $n_g$ | $B_\text{payload}$ | $B_\text{meta}$ | $B$ | $B/B_\text{bf16}$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 534773760 | 8556380160 | 1069547520 | 9625927680 | 0.28125 |
| 128 | 133693440 | 8556380160 | 267386880 | 8823767040 | 0.2578125 |

JSON: `mlp_n` 17112760320, `mlp_bf16_bytes` 34225520640, `mlp_int4_payload_bytes` 8556380160, `mlp_int4_g32_meta_bytes` 1069547520, `mlp_int4_g32_total_bytes` 9625927680, `mlp_int4_g32_over_bf16` 0.28125, `mlp_int4_g128_meta_bytes` 267386880, `mlp_int4_g128_total_bytes` 8823767040, `mlp_int4_g128_over_bf16` 0.2578125.

Exactness: $B_\text{payload}=B_\text{bf16}/4$; $B_\text{meta}/B_\text{bf16}=1/g$ when $s=2$ and BF16 is 2 bytes/element; $g=32$ ⇒ $0.25+1/32=0.28125$; $g=128$ ⇒ $0.25+1/128=0.2578125$.

**Illustration B — unique non-embed language+MTP**, $n=26049299456$, $B_\text{bf16}=52098598912$ (TASK-06), int4 $g=128$ $s=2$: $B_\text{payload}=13024649728$, $n_g=203510152$, $B_\text{meta}=407020304$, $B=13431670032$, ratio $0.2578125$. JSON keys `unique_non_embed_n`, `unique_non_embed_bf16_bytes`, `unique_non_embed_int4_g128_total_bytes`, `unique_non_embed_int4_g128_over_bf16`. Label in prose: illustration of metadata arithmetic, **not** a selected profile.

**Illustration C — embed gather** (TASK-06: decode reads one row, 10240 B BF16). int4 `per_row`: payload $5120\times 0.5=2560$ plus one 2-byte scale ⇒ 2562 B. JSON `embed_gather_bf16_bytes` 10240, `embed_gather_int4_row_bytes` 2562. Access pattern differs from `lm_head` full GEMM (HYPOTHESIS `d_embed_gather` vs `d_lm_head_unpack`).

**Illustration D — $S$** (TASK-04/06): conceptual F32 150994944 B/step. BF16 store 75497472 B; FP8 37748736 B; int8+per_tensor 2-byte scale 37748738 B. JSON `s_f32_bytes` 150994944, `s_bf16_bytes` 75497472, `s_fp8_bytes` 37748736. Not a winner; pairs with `q_state_s` / `d_state_s_narrow`.

JSON `scale_storage_illustration_bytes` = 2.

### Source evidence citations (lock)

Do not recopy TASK-05 family tables. The deliverable **must** include these MEASURED citations as decimal substrings (canonical TASK-05 floats) and these OBSERVED/DERIVED occupancy/traffic integers.

MEASURED (from `docs/architecture/bf16-tensor-analysis.md`):

| JSON key | Value | What it shapes (HYPOTHESIS rationale, not a winner) |
| --- | --- | --- |
| `cit_language_mtp_absmax` | 19.25 | `gdn_time_param` keeps `keep_source` and only `i8_tensor` |
| `cit_dt_bias_absmax` | 19.25 | same |
| `cit_a_log_absmax` | 5.5625 | same |
| `cit_conv1d_frac_out_6x` | 0.1805953979492 | `conv1d` includes `i4_clip` and `extract_high` |
| `cit_embed_row_ratio` | 11.74384236453 | `embed_table` includes `per_row` / `extract_high` |
| `cit_lm_head_row_ratio` | 5.223880597015 | `lm_head` includes `per_row` |
| `cit_out_proj_mean_row_ratio` | 15.3244687679 | `linear_large_proj` includes `per_row` and `per_col` |
| `cit_o_proj_mean_row_ratio` | 13.59971022497 | `attn_out` uses `C_GEMM_LARGE` |
| `cit_down_proj_max_row_ratio` | 25.73544973545 | `mlp_down` keeps `per_row`/`per_col`; omits `int2` |
| `cit_mlp_gate_frac_out_6x` | 0.0003818664480658 | `mlp_up_gate` may include `int2` extract (mass, not tail) |
| `cit_embed_n_zero` | 7528 | embed is the only pooled-zero location (TASK-05) |

OBSERVED/DERIVED occupancy and traffic (must match TASK-01/06 / `text_config`):

| JSON key | Value |
| --- | ---: |
| `weight_bytes_language_mlp` | 34225520640 |
| `weight_bytes_lm_head` | 2542796800 |
| `weight_bytes_embed_table` | 2542796800 |
| `weight_bytes_language_linear_attn` | 11124102144 |
| `weight_bytes_language_self_attn` | 3355459584 |
| `weight_bytes_language_mtp_excl_vision` | 54641395712 |
| `weight_bytes_unique_non_embed` | 52098598912 |
| `s_f32_bytes` | 150994944 |
| `kv_bytes_all_per_token` | 69632 |
| `c_bytes_all` | 2949120 |
| `mlp_n` | 17112760320 |
| `embed_n` | 1271398400 |
| `n_language_mtp_parameters` | 27320697856 |

JSON booleans: `gguf_is_not_a_recipe` true; `pareto_frontier_selected` false; `activation_quant_in_family_policies` false; `payloads_restreamed` false.

### Policy families (lock)

JSON array `policy_families` in this exact order (16 ids). JSON `n_policy_families` = 16.

| id | role | Members (level-2 or state ids) |
| --- | --- | --- |
| `norm_gamma` | param | `input_layernorm`, `post_attention_layernorm`, `final_norm`, `linear_attn.norm`, `self_attn.q_norm`, `self_attn.k_norm`, `mtp.norm`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.input_layernorm`, `mtp.post_attention_layernorm`, `mtp.self_attn.q_norm`, `mtp.self_attn.k_norm` |
| `gdn_time_param` | param | `linear_attn.A_log`, `linear_attn.dt_bias` |
| `gdn_gate_proj` | param | `linear_attn.in_proj_a`, `linear_attn.in_proj_b` |
| `conv1d` | param | `linear_attn.conv1d` |
| `linear_large_proj` | param | `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj` |
| `attn_qkv` | param | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `mtp.self_attn.q_proj`, `mtp.self_attn.k_proj`, `mtp.self_attn.v_proj` |
| `attn_out` | param | `self_attn.o_proj`, `mtp.self_attn.o_proj` |
| `mlp_up_gate` | param | `mlp.gate_proj`, `mlp.up_proj`, `mtp.mlp.gate_proj`, `mtp.mlp.up_proj` |
| `mlp_down` | param | `mlp.down_proj`, `mtp.mlp.down_proj` |
| `embed_table` | param | `embed` |
| `lm_head` | param | `lm_head` |
| `mtp_fc` | param | `mtp.fc` |
| `vision_deferred` | param | `vision.patch_embed`, `vision.pos_embed`, `vision.blocks`, `vision.merger` |
| `state_kv` | state | `K_state`, `V_state` |
| `state_c` | state | `C_state` |
| `state_s` | state | `S` |

JSON object `policy_family_members` maps each family id to that member list (arrays in the order above). Union of param members = all 42 `LEVEL2_IDS`. Union of state members = `["K_state", "V_state", "C_state", "S"]`. JSON `n_level2_mapped` = 42, `n_state_mapped` = 4.

MTP rank-2 weights **mirror** the matching language family (same candidate list), because occupancy is 1 and TASK-05 did not show a separate policy-shape. `mtp.fc` is its own family (unique $5120\times 10240$ contraction).

### Family-specific candidate policies (lock)

JSON object `family_candidates` maps each policy family to a JSON array of recipe ids. These arrays are the research space. There is **no** preferred/default/winner field. Rationale column in the markdown is labelled HYPOTHESIS.

| family | `family_candidates` (this order) | $n$ | HYPOTHESIS rationale (not a selection) |
| --- | --- | ---: | --- |
| `norm_gamma` | `keep_source`, `i8_tensor`, `i8_g32` | 3 | Rank-1 $\gamma$; TASK-07 `rms_hidden` high; small $n$; many families have MEASURED `frac_out_6x=0`. Omit `per_row`/`int4`. |
| `gdn_time_param` | `keep_source`, `i8_tensor` | 2 | $n=2304$ each; MEASURED absmax 19.25 / 5.5625; TASK-07 `gdn_alpha_beta` high. No grouping finer than tensor: vectors length 48. |
| `gdn_gate_proj` | `keep_source`, `i8_tensor`, `i8_row`, `i6_row`, `i4_row`, `i4_g128`, `i4_g128_extract` | 7 | Feeds $\alpha/\beta$; modest 11796480 params each; TASK-05 col ratios ~6. Include extract; omit `int2`. |
| `conv1d` | `keep_source`, `i8_row`, `i4_row`, `i4_g32`, `i4_g128_extract`, `i4_clip` | 6 | MEASURED `frac_out_6x` 0.1805953979492 on squeezed `(10240, 4)`. Clip is a **candidate**, not a recommendation. |
| `linear_large_proj` | `C_GEMM_LARGE` | 15 | Dominant linear-attn bytes; `out_proj` mean row ratio 15.3244687679. |
| `attn_qkv` | `C_GEMM_LARGE` | 15 | Same GEMM-shaped space; K also feeds `state_kv` (quality risk, not a different recipe list). |
| `attn_out` | `C_GEMM_LARGE` | 15 | Residual path (TASK-07 `residual_stream` high) plus high directional ratios; still the same GEMM grid, **without** `int2`. |
| `mlp_up_gate` | `C_GEMM_LARGE` + `i3_g32_extract` + `i2_g32_extract` | 17 | 11.4 GiB each; TASK-06 $I=1$; low MEASURED 6× fractions. `int2` is an extreme **candidate** on this mass only. |
| `mlp_down` | `C_GEMM_LARGE` | 15 | Residual + $K=17408$ + max row ratio 25.73544973545; omit `int2`. |
| `embed_table` | `keep_source`, `i8_row`, `i4_row`, `i4_g128`, `i4_row_extract` | 5 | Gather, not GEMM (0 MAC). Row ratio 11.74384236453; `n_zero` 7528. |
| `lm_head` | `keep_source`, `i8_row`, `i8_row_asym`, `i6_row`, `i4_row`, `i4_g128`, `i4_g128_extract` | 7 | Full GEMM; TASK-06 `vocab_memory` 2542796800 B. Not a gather recipe set. |
| `mtp_fc` | `keep_source`, `i8_row`, `i4_row`, `i4_col`, `i4_g128` | 5 | Unique shape; MEASURED row ratio 1.225 vs col 4.95, so `per_col` stays in the set. |
| `vision_deferred` | `[]` | 0 | UNKNOWN internals; no experiments defined here. |
| `state_kv` | `keep_source`, `fp8_tensor`, `i8_tensor` | 3 | Conceptual BF16; TASK-07 `state_kv_bf16` high; TASK-06 69632 B/token. |
| `state_c` | `keep_source`, `i8_tensor` | 2 | TASK-07 `state_c_bf16` low; 2949120 B. |
| `state_s` | `keep_source`, `narrow_bf16`, `fp8_tensor`, `i8_tensor` | 4 | Conceptual F32; `keep_source` = F32; `narrow_bf16` **is** the `s_below_f32` experiment. |

JSON `family_candidate_counts` parallel to `policy_families`: `[3, 2, 7, 6, 15, 15, 15, 17, 15, 5, 7, 5, 0, 3, 2, 4]`.

Every family except `vision_deferred` contains `keep_source`. JSON `keep_source_on_all_defined_families` true.

Prose required: an **example grid cell** may be named (`mlp_up_gate` × `i4_g128`) only with the words `example` and `not a selected winner`. Control profile = all `keep_source` (and `vision_deferred` skipped). Do not call the control a baseline quality winner.

### Quality risks (lock)

JSON array `quality_risk_ids` in this exact order (17 ids). Parallel `quality_risk_families`, `quality_risk_sensitive_ops` (primary TASK-07 id), `quality_risk_severities`. Every severity is HYPOTHESIS. JSON `n_quality_risks` = 17.

| ID | Family / scope | Primary TASK-07 op | Severity |
| --- | --- | --- | --- |
| `q_norm_gamma` | `norm_gamma` | `rms_hidden` | high |
| `q_gdn_time` | `gdn_time_param` | `gdn_alpha_beta` | high |
| `q_gdn_gate` | `gdn_gate_proj` | `gdn_alpha_beta` | high |
| `q_conv` | `conv1d` | `conv_fir` | low |
| `q_linear_proj` | `linear_large_proj` | `gemm_k5120` | medium |
| `q_attn_qkv` | `attn_qkv` | `gemm_k5120` | medium |
| `q_attn_out` | `attn_out` | `residual_stream` | high |
| `q_mlp_up` | `mlp_up_gate` | `gemm_k5120` | medium |
| `q_mlp_down` | `mlp_down` | `residual_stream` | high |
| `q_embed` | `embed_table` | `param_bf16` | medium |
| `q_lm_head` | `lm_head` | `gemm_lm_head` | medium |
| `q_mtp_fc` | `mtp_fc` | `gemm_k5120` | medium |
| `q_state_kv` | `state_kv` | `state_kv_bf16` | high |
| `q_state_c` | `state_c` | `state_c_bf16` | low |
| `q_state_s` | `state_s` | `s_below_f32` | high |
| `q_clip_tail` | any `clip` recipe | `param_bf16` | medium |
| `q_int2_mass` | `mlp_up_gate` × `i2_g32_extract` | `param_bf16` | high |

Locked partitions:

- `quality_high_ids`: `q_norm_gamma`, `q_gdn_time`, `q_gdn_gate`, `q_attn_out`, `q_mlp_down`, `q_state_kv`, `q_state_s`, `q_int2_mass`
- `quality_medium_ids`: `q_linear_proj`, `q_attn_qkv`, `q_mlp_up`, `q_embed`, `q_lm_head`, `q_mtp_fc`, `q_clip_tail`
- `quality_low_ids`: `q_conv`, `q_state_c`

JSON `n_quality_high` = 8, `n_quality_medium` = 7, `n_quality_low` = 2.

Prose required (not extra ids): `q_attn_qkv` primary is `gemm_k5120`; RoPE-baked $K$ also feeds `state_kv_bf16`. `q_mlp_down` primary is residual_stream; `gemm_k17408` is the matching accum path. Survival of any row is TASK-18.

### Decode-complexity risks (lock)

JSON array `decode_risk_ids` in this exact order (9 ids). Parallel `decode_risk_severities`. Every severity is HYPOTHESIS versus TASK-06 UNKNOWN SKU ridge. JSON `n_decode_risks` = 9.

| ID | Ties to | Severity | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `d_weight_dequant` | TASK-06 `weight_memory`, $I=1$ | high | Extra dequant ALU on decode GEMMs may compete with a memory-bound region |
| `d_fine_group_meta` | Illustration A $g=32$ vs $g=128$ | medium | Fine-group scale traffic is a material decode bandwidth term |
| `d_outlier_gather` | `extract_high` | medium | Sparse BF16 sidecar is irregular extra traffic |
| `d_mixed_width` | `mixed_group` | low | Width switching prevents uniform unpack |
| `d_embed_gather` | Illustration C | medium | Per-row dequant of one vocab row differs from table-wide unpack |
| `d_lm_head_unpack` | TASK-06 `vocab_memory` | high | 2542796800 B unique; unpack every decode |
| `d_3bit_pack` | `int3` / `int6` | medium | Non-byte-aligned codes add shift/mask work (layout is TASK-09) |
| `d_state_s_narrow` | Illustration D, TASK-06 `state_memory` | high | Narrowing $S$ cuts 150994944 B/step but pairs with `q_state_s` |
| `d_kv_narrow` | TASK-06 `kv_memory` | medium | Narrowing KV scales with $T$ via 69632 B/token |

JSON `decode_high_ids`: `d_weight_dequant`, `d_lm_head_unpack`, `d_state_s_narrow`. `decode_medium_ids`: `d_fine_group_meta`, `d_outlier_gather`, `d_embed_gather`, `d_3bit_pack`, `d_kv_narrow`. `decode_low_ids`: `d_mixed_width`. JSON `n_decode_high` = 3, `n_decode_medium` = 5, `n_decode_low` = 1.

Do not rank these by wall time. Do not name CUDA kernels. Do not convert TASK-06 bottleneck labels into measurements.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 8 (Quality and decode-complexity risks). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers or 22 recipes.

Required IDs **inside that fence**: `bit_width`, `grouping`, `scale`, `outlier`, `param`, `state`, `quality`, `decode`.

JSON `n_diagrams` is 1. `diagram_ids` is `["bit_width","grouping","scale","outlier","param","state","quality","decode"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger quantization experiments are **UNKNOWN**. `vision_deferred` has an empty candidate list. Do not add vision recipes. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_quantization_design_space.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py` or `scripts/analyze_bf16_tensors.py` or `scripts/inventory_bf16_checkpoint.py`; duplicate the small `text_config` arithmetic needed for occupancy and the MLP/embed products.

CLI (cwd = repository root):

```text
python3 scripts/check_quantization_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_quantization_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-design-space docs/architecture/quantization-design-space.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`. Derived: `mlp_n`, `embed_n`, family BF16 byte citations matching TASK-01/06, metadata illustrations. Constant fields: canonical sentences, dimension/recipe/family/risk lists, MEASURED citation floats (locked, not live-scanned from TASK-05).
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--quantization-design-space PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all four canonical sentences verbatim, (5) every `policy_families`, `candidate_recipe_ids`, `quality_risk_ids`, and `decode_risk_ids` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer/decimal below present as a decimal or integer substring, (10) `design_dimensions` ids present as substrings, (11) the words `HYPOTHESIS` and `not selected winners` present (the latter via the canonical candidates sentence), (12) none of the forbidden winner phrases: `should be 4-bit`, `should be 8-bit`, `quantization winner`, `recommend 4-bit`, `recommend 8-bit`, `selected winner`, `Pareto frontier is`. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks TASK-05 citations against `bf16-tensor-analysis.md`). Do not require `1e-06` as a raw substring.

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `mlp_n==17112760320` and `mlp_n == 3 * n_decoder_layers * intermediate_size * hidden_size`
- `embed_n==1271398400` and `embed_n == vocab_size * hidden_size`
- `mlp_bf16_bytes==34225520640==2*mlp_n`
- `mlp_int4_g32_over_bf16==0.28125` and `mlp_int4_g128_over_bf16==0.2578125`
- `unique_non_embed_int4_g128_over_bf16==0.2578125`
- `weight_bytes_language_mtp_excl_vision==54641395712`
- `weight_bytes_unique_non_embed==52098598912`
- `s_f32_bytes==150994944`, `kv_bytes_all_per_token==69632`, `c_bytes_all==2949120`
- `n_design_dimensions==4`, `n_candidate_recipes==22`, `n_policy_families==16`
- `n_quality_risks==17`, `n_quality_high==8`, `n_quality_medium==7`, `n_quality_low==2`
- `n_decode_risks==9`, `n_decode_high==3`, `n_decode_medium==5`, `n_decode_low==1`
- `n_level2_mapped==42`, `n_state_mapped==4`, `n_diagrams==1`
- `candidate_recipe_ids` equals the locked 22-id list; parallel format/grouping/scale/outlier arrays match the table
- `policy_families` equals the locked 16-id list; `family_candidates` arrays equal the locked lists; `family_candidate_counts` equals `[3, 2, 7, 6, 15, 15, 15, 17, 15, 5, 7, 5, 0, 3, 2, 4]`
- union of `policy_family_members` param lists equals the locked 42 level-2 ids in `LEVEL2_IDS` order when flattened by family-table order (assert set equality plus length 42)
- `family_candidates["vision_deferred"]==[]`
- `keep_source` is in every `family_candidates` list except `vision_deferred`
- `narrow_bf16` appears only on `state_s`
- `i2_g32_extract` appears only on `mlp_up_gate`
- `i4_clip` appears only on `conv1d`
- `gguf_is_not_a_recipe is True`, `pareto_frontier_selected is False`, `activation_quant_in_family_policies is False`, `payloads_restreamed is False`
- `cit_language_mtp_absmax==19.25`, `cit_dt_bias_absmax==19.25`, `cit_conv1d_frac_out_6x==0.1805953979492`

Locked document integers/decimals the `--quantization-design-space` check must find:

`5120`, `17408`, `248320`, `17112760320`, `34225520640`, `2542796800`, `54641395712`, `52098598912`, `150994944`, `69632`, `2949120`, `19.25`, `0.28125`, `0.2578125`, `8556380160`, `9625927680`, `8823767040`, `10240`, `2562`, `0.1805953979492`, `11.74384236453`, `25.73544973545`, `15.3244687679`, `7528`

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `bytes_bf16`, `bytes_f32`, `bytes_fp8`,

`mlp_n`, `embed_n`, `n_language_mtp_parameters`, `mlp_bf16_bytes`, `mlp_int4_payload_bytes`, `mlp_int4_g32_meta_bytes`, `mlp_int4_g32_total_bytes`, `mlp_int4_g32_over_bf16`, `mlp_int4_g128_meta_bytes`, `mlp_int4_g128_total_bytes`, `mlp_int4_g128_over_bf16`,

`weight_bytes_language_mlp`, `weight_bytes_lm_head`, `weight_bytes_embed_table`, `weight_bytes_language_linear_attn`, `weight_bytes_language_self_attn`, `weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `unique_non_embed_n`, `unique_non_embed_bf16_bytes`, `unique_non_embed_int4_g128_total_bytes`, `unique_non_embed_int4_g128_over_bf16`,

`embed_gather_bf16_bytes`, `embed_gather_int4_row_bytes`, `s_f32_bytes`, `s_bf16_bytes`, `s_fp8_bytes`, `kv_bytes_all_per_token`, `c_bytes_all`, `scale_storage_illustration_bytes`, `scale_storage_bytes_candidates`, `ragged_last_group`, `group_clips_to_axis`, `payload_formula_allows_fractional_bytes`,

`cit_language_mtp_absmax`, `cit_dt_bias_absmax`, `cit_a_log_absmax`, `cit_conv1d_frac_out_6x`, `cit_embed_row_ratio`, `cit_lm_head_row_ratio`, `cit_out_proj_mean_row_ratio`, `cit_o_proj_mean_row_ratio`, `cit_down_proj_max_row_ratio`, `cit_mlp_gate_frac_out_6x`, `cit_embed_n_zero`,

`design_dimensions`, `formats`, `ieee_like_formats`, `integer_formats`, `grouping_ids`, `block_group_sizes`, `scale_ids`, `outlier_ids`,

`candidate_recipe_ids`, `recipe_formats`, `recipe_groupings`, `recipe_scales`, `recipe_outliers`, `c_gemm_large`, `n_candidate_recipes`, `n_design_dimensions`,

`policy_families`, `policy_family_members`, `family_candidates`, `family_candidate_counts`, `n_policy_families`, `n_level2_mapped`, `n_state_mapped`, `keep_source_on_all_defined_families`,

`quality_risk_ids`, `quality_risk_families`, `quality_risk_sensitive_ops`, `quality_risk_severities`, `quality_high_ids`, `quality_medium_ids`, `quality_low_ids`, `n_quality_risks`, `n_quality_high`, `n_quality_medium`, `n_quality_low`,

`decode_risk_ids`, `decode_risk_severities`, `decode_high_ids`, `decode_medium_ids`, `decode_low_ids`, `n_decode_risks`, `n_decode_high`, `n_decode_medium`, `n_decode_low`,

`gguf_is_not_a_recipe`, `pareto_frontier_selected`, `activation_quant_in_family_policies`, `payloads_restreamed`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_candidates`, `canonical_sentence_hypothesis`, `canonical_sentence_metadata`.

Integer JSON fields that are counts/widths/bytes are JSON ints. Ratios `mlp_int4_g32_over_bf16`, `mlp_int4_g128_over_bf16`, `unique_non_embed_int4_g128_over_bf16` are JSON numbers `0.28125` and `0.2578125`. MEASURED citations are JSON numbers using the canonical floats above (`cit_conv1d_frac_out_6x` `0.1805953979492`, `cit_embed_row_ratio` `11.74384236453`, `cit_lm_head_row_ratio` `5.223880597015`, `cit_out_proj_mean_row_ratio` `15.3244687679`, `cit_o_proj_mean_row_ratio` `13.59971022497`, `cit_down_proj_max_row_ratio` `25.73544973545`, `cit_mlp_gate_frac_out_6x` `0.0003818664480658`, `cit_a_log_absmax` `5.5625`, absmax citations `19.25`). Booleans are JSON booleans. `scale_storage_bytes_candidates` and `block_group_sizes` are JSON arrays of ints. `full_attention_indices` is a JSON array of ints.

`recipe_formats` / `recipe_groupings` / `recipe_scales` / `recipe_outliers` are JSON arrays of strings parallel to `candidate_recipe_ids`. `quality_risk_*` parallel arrays match `quality_risk_ids`. `decode_risk_severities` parallels `decode_risk_ids`.

`policy_family_members` and `family_candidates` are JSON objects keyed in `policy_families` order (script `json.dumps` with that key order).

LEVEL2_IDS for the set-equality assert (this order):

`embed`, `final_norm`, `lm_head`, `input_layernorm`, `post_attention_layernorm`, `linear_attn.A_log`, `linear_attn.conv1d`, `linear_attn.dt_bias`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.norm`, `linear_attn.out_proj`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `self_attn.q_norm`, `self_attn.k_norm`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `mtp.fc`, `mtp.norm`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.input_layernorm`, `mtp.post_attention_layernorm`, `mtp.self_attn.q_proj`, `mtp.self_attn.k_proj`, `mtp.self_attn.v_proj`, `mtp.self_attn.o_proj`, `mtp.self_attn.q_norm`, `mtp.self_attn.k_norm`, `mtp.mlp.gate_proj`, `mtp.mlp.up_proj`, `mtp.mlp.down_proj`, `vision.patch_embed`, `vision.pos_embed`, `vision.blocks`, `vision.merger`.

### Stage split

- **Implementation** writes `scripts/check_quantization_design_space.py` **and** `docs/architecture/quantization-design-space.md` (dimensions, recipes, metadata illustrations, 16 family candidate lists, 17+9 HYPOTHESIS risks, JSON fence). Runs `--json` and `--quantization-design-space` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-04 / `work-and-traffic.md` style), Authority table links to this dossier / inventory / TASK-05 / TASK-06 / TASK-07 / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, recipe ids, family candidate lists, severities, or Mermaid node IDs. Does not edit TASK-05/06/07 artifacts.
- **Verification** independently re-runs focused commands, recomputes `mlp_n` / metadata ratios / unique-non-embed bytes from sitting `text_config` (not from JSON echo), spot-checks MEASURED citation floats against `docs/architecture/bf16-tensor-analysis.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-recipe, no winner phrases, no `plan.md` edit, no payload I/O, and that every risk row is labelled HYPOTHESIS. Pareto open question left open.

- Invariants:
  - Ten level-2 headings in the locked order; four dimensions; 22 recipes; 16 policy families; 42 level-2 + 4 state ids mapped; 17 quality risks; 9 decode-complexity risks; four canonical sentences verbatim.
  - Prefill/decode share one research space; primary includes language+MTP params; state is secondary; activations are not family policies.
  - `keep_source` on every defined family; `vision_deferred` empty; no selected winners; `pareto_frontier_selected` false.
  - GGUF Q4_K_M is not a recipe. Payloads are not restreamed.
  - Logical values do not imply allocation; candidates are not winners; risks are hypotheses; metadata bytes are not packed layouts.
  - Vision encoder remains unexpanded.
- Rejected alternatives:
  - Selecting a 4-bit MLP / 8-bit residual winner from TASK-05 tables: rejected; those tables forbid quality conclusions; Pareto is TASK-18.
  - Treating GGUF Q4_K / Q4_K_M grouping as a candidate recipe: rejected; plan.md black-box reference only (TASK-18).
  - Re-streaming payloads to produce new MEASURED stats: rejected; TASK-05 already measured.
  - Activation-aware / GPTQ / AWQ / Hessian scales as a design dimension: rejected; TASK-10/18.
  - Learned vector-quant codebooks as a dimension: rejected for this space; scalar recipes only; TASK-09/10 may add packing/codebooks later.
  - `int5` or `fp8_e5m2` or `block_g256`: rejected as redundant for shaping this grid.
  - Per-layer bit-width switching as a dimension: rejected; layer variation is absorbed by `per_tensor` / finer groups, not layer winners.
  - Activation working-dtype family policies: rejected; TASK-07 `activation_dtype_decided` is false; TASK-17 owns mixed-precision CUDA recipes.
  - Counting chunkwise GDN `(19)` as a reason to skip `state_s` experiments: rejected; definition remains `(17)`–`(18)` (TASK-06/07).
  - Using CUDA dtypes or GGUF sizes as conceptual element sizes: rejected; BF16 inventory + conceptual F32 $S$.
  - Inspecting Quartz or llama.cpp for “real” quantizers: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–07.
  - Editing frozen TASK-01–07 docs or `plan.md`.
  - Importing other `check_*.py` or `analyze_bf16_tensors.py`.
  - A fifth design dimension (`codebook`, `calibration`, `kernel`): rejected; four ledger dimensions only; metadata/compute is implications, not a fifth selector.
  - Closing the ledger open question “which bit widths, grouping, scales, and outlier policies form the Pareto frontier”.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/quantization-design-space.md` exists and follows the heading list above.
  - Four design dimensions are named; 22 recipes tabulated; metadata formulas instantiated (MLP $g=32/128$ ratios 0.28125 / 0.2578125; unique-non-embed illustration; embed gather; $S$ bytes).
  - Sixteen policy families map all 42 level-2 ids plus $K,V,C,S$; family candidate lists match the lock; no winner column; `vision_deferred` empty.
  - Quality (17) and decode-complexity (9) risks are tabulated as HYPOTHESIS and do not claim experimental proof or Pareto selection.
  - Four canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants/citations.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; GGUF is not a recipe; no payload re-stream; no `plan.md` edit.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_quantization_design_space.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_quantization_design_space.py
python3 scripts/check_quantization_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_quantization_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-design-space docs/architecture/quantization-design-space.md
```

- Candidate quality: not required — no model execution or NLL; this increment is research-space documentation. Quality **risks** are HYPOTHESIS prose/tables, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/quantization-design-space.md
python3 -m py_compile scripts/check_quantization_design_space.py
python3 scripts/check_quantization_design_space.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-design-space docs/architecture/quantization-design-space.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run `scripts/analyze_bf16_tensors.py`.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/quantization-design-space.md` (create)
  - `scripts/check_quantization_design_space.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-05/06/07 deliverables unchanged)
- Definition of done: quantization-design-space document published with locked dimensions, recipes, metadata illustrations, 16 family candidate sets without winners, and HYPOTHESIS quality plus decode-complexity risks; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-08 completion checkboxes can be marked at delivery; the Pareto-frontier open question remains open.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T13:35:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-08.md`. Coupled IDs `none`. Document structure (10 headings), four design dimensions, 22 candidate recipes, 16 policy families with locked `family_candidates`, metadata formulas and MLP/unique-non-embed/embed/$S$ illustrations, 17 quality + 9 decode-complexity HYPOTHESIS risks, stdlib checker `scripts/check_quantization_design_space.py`, JSON schema, and acceptance commands are closed. `docs/architecture/quantization-design-space.md` and the checker were **not** written in this stage. `plan.md` not edited. No commit.
- Performance evidence applied: N/A — research-space documentation; risk labels are hypotheses, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_quantization_design_space.py` (stdlib checker: `text_config` occupancy arithmetic matching TASK-01/06 family BF16 bytes, MLP/embed/metadata illustrations, locked 4 dimensions / 22 recipes / 16 policy families / 17+9 HYPOTHESIS risks; `--json` internal asserts; `--quantization-design-space` heading/JSON/mermaid/canonical/id/token/winner-phrase checks). Does not import other `check_*.py` or `analyze_bf16_tensors.py`. Does not stream safetensor payloads.
  - Created `docs/architecture/quantization-design-space.md` (ten locked headings, four canonical sentences, 22-recipe table, MLP $g=32/128$ ratios 0.28125 / 0.2578125 plus unique-non-embed / embed-gather / $S$ illustrations, 16 family candidate lists without a winner column, 17 quality + 9 decode-complexity HYPOTHESIS rows, one `flowchart TB` mermaid with required IDs, JSON fence copied from a live `--json` run).
  - Did not edit `docs/architecture/plan.md`, `bf16-tensor-analysis.md`, `work-and-traffic.md`, `numerical-sensitivity.md`, `model-inventory.md`, `model-semantics.md`, `dataflow.md`, or `lifetime-and-state.md`. Did not commit.
- Commands:
  - `python3 -m py_compile scripts/check_quantization_design_space.py` — pass (exit 0)
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); live object `n_candidate_recipes=22`, `n_policy_families=16`, `n_quality_risks=17`, `n_decode_risks=9`, `mlp_n=17112760320`, `mlp_int4_g32_over_bf16=0.28125`, `mlp_int4_g128_over_bf16=0.2578125`, `weight_bytes_language_mtp_excl_vision=54641395712`, `weight_bytes_unique_non_embed=52098598912`, `pareto_frontier_selected=false`, `gguf_is_not_a_recipe=true`
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-design-space docs/architecture/quantization-design-space.md` — pass (exit 0)
- UTC/time/tokens/cost: `2026-09-20T13:41:03Z`; `telemetry_unavailable`
- Commit: not created (implementation stage)

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/quantization-design-space.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-04 / [`lifetime-and-state.md`](../lifetime-and-state.md) / [`work-and-traffic.md`](../work-and-traffic.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`model-inventory.md`](../model-inventory.md) (TASK-01), [`bf16-tensor-analysis.md`](../bf16-tensor-analysis.md) (TASK-05), [`work-and-traffic.md`](../work-and-traffic.md) (TASK-06), [`numerical-sensitivity.md`](../numerical-sensitivity.md) (TASK-07), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_quantization_design_space.py`](../../../scripts/check_quantization_design_space.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the first JSON fence left unchanged. No locked integers, canonical sentences, recipe ids, family candidate lists, severities, or Mermaid node IDs edited. Frozen TASK-05/06/07 artifacts and `plan.md` not edited.
- Commands:
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-design-space docs/architecture/quantization-design-space.md` — pass (exit 0; headings, JSON fence, one flowchart, canonical sentences, recipe ids, diagram node IDs; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T13:42:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1 (first pass)
- Agent/model: `composer-2.5` (verifier subagent)
- UTC/time/tokens/cost: `2026-09-20T13:45:00Z`; `telemetry_unavailable`
- Focused commands (repository root):
  - `python3 -m py_compile scripts/check_quantization_design_space.py` — pass (exit 0)
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); live object `n_candidate_recipes=22`, `n_policy_families=16`, `n_quality_risks=17`, `n_decode_risks=9`, `mlp_n=17112760320`, `mlp_int4_g32_over_bf16=0.28125`, `mlp_int4_g128_over_bf16=0.2578125`, `weight_bytes_language_mtp_excl_vision=54641395712`, `weight_bytes_unique_non_embed=52098598912`, `full_attention_indices=[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`, `pareto_frontier_selected=false`, `gguf_is_not_a_recipe=true`
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-design-space docs/architecture/quantization-design-space.md` — pass (exit 0)
- Repository-wide commands:
  - `test -f docs/architecture/quantization-design-space.md` — pass
  - `python3 -m py_compile scripts/check_quantization_design_space.py` — pass (exit 0)
  - `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-design-space docs/architecture/quantization-design-space.md` — pass (exit 0)
- Independent review (not from JSON echo):
  - Recomputed from sitting `text_config` (`hidden_size=5120`, `intermediate_size=17408`, `vocab_size=248320`, 64 decoder layers): `mlp_n=17112760320`, `embed_n=1271398400`, `mlp_bf16_bytes=34225520640`, int4 $g=32$ ratio `0.28125`, int4 $g=128$ ratio `0.2578125` — all match dossier locks.
  - TASK-06 traffic integers (`54641395712`, `52098598912`, `150994944`, `69632`, `2949120`) present in deliverable and match [`work-and-traffic.md`](../work-and-traffic.md).
  - MEASURED citation floats spot-checked against [`bf16-tensor-analysis.md`](../bf16-tensor-analysis.md) (not from checker JSON): `19.25` (`pooled_language_mtp` / `linear_attn.dt_bias`), `5.5625` (`linear_attn.A_log`), `0.1805953979492` (`linear_attn.conv1d` `frac_out_6x`), `11.74384236453` (`embed` row ratio), `5.223880597015` (`lm_head`), `15.3244687679` (`linear_attn.out_proj`), `13.59971022497` (`self_attn.o_proj`), `25.73544973545` (`mlp.down_proj`), `0.0003818664480658` (`mlp.gate_proj` `frac_out_6x`), `7528` (`embed.n_zero`) — all match source document.
  - First fenced `json` block in deliverable byte-equal to live `--json` output — pass.
  - Ten required `##` headings in locked order; four canonical sentences verbatim; exactly one `flowchart TB` mermaid fence with required node ids (`bit_width`, `grouping`, `scale`, `outlier`, `param`, `state`, `quality`, `decode`).
  - 22 recipes, 16 policy families, 42 level-2 + 4 state ids mapped; `family_candidate_counts` and exclusivity locks (`narrow_bf16` only `state_s`, `i2_g32_extract` only `mlp_up_gate`, `i4_clip` only `conv1d`, `keep_source` on all defined families) — pass.
  - 17 quality + 9 decode-complexity risks tabulated; severities and claims labelled HYPOTHESIS; no experimental proof or Pareto selection language.
  - No forbidden winner phrases (`should be 4-bit`, `selected winner`, `Pareto frontier is`, etc.); GGUF cited only as not-a-recipe / TASK-18 reference; Quartz/llama.cpp only in forbidden-inspection prose.
  - `UNKNOWN` appears only under **Deferred vision**; no `TBD` / `TODO` / `???`.
  - `scripts/check_quantization_design_space.py` uses stdlib only (`argparse`, `json`, `math`, `re`, `sys`, `pathlib`); does not import other `check_*.py` or `analyze_bf16_tensors.py`; no safetensor payload I/O observed.
  - Frozen artifacts unchanged: `docs/architecture/plan.md`, `bf16-tensor-analysis.md`, `work-and-traffic.md`, `numerical-sensitivity.md`, `model-inventory.md` — no edits in this increment (git: only new TASK-08 deliverables plus ledger `IN PROGRESS` row).
  - Candidate NLL: not required (research-space documentation); quality risks are HYPOTHESIS prose only.
  - Ruff / pytest / CMake / CUDA: not run (dossier explicitly excludes these gates for TASK-08).
- Verdict: **PASS** (first-pass acceptance)

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-08 only; coupled IDs `none`
- Outcome: TASK-08 marked `DONE` after verification pass (attempt 1, verdict PASS)
- UTC/time/tokens/cost: `2026-09-20T13:46:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1) — `docs/architecture/quantization-design-space.md` (10 locked headings, four canonical sentences, four design dimensions, 22 recipes, 16 policy families without winners, metadata illustrations, 17 quality + 9 decode-complexity HYPOTHESIS risks, one `flowchart TB` mermaid, JSON fence); `scripts/check_quantization_design_space.py` stdlib checker; independent `text_config` recomputation and TASK-05 citation spot-checks pass; frozen upstream docs unchanged; Pareto-frontier open question remains for TASK-18
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (research-space design locks; no tok/s evidence)
- Throughput delta: N/A — TASK-08 does not execute or time the model
- Commit: Publish Qwen3.8 quantization design space
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification attempt 1 PASS; no repair loop)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands
