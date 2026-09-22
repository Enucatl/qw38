# TASK-18 — Design quantization evaluation methodology

## Control

- Primary ID: `TASK-18`
- Coupled IDs: `none`
- Dependencies: `TASK-07`, `TASK-08`, `TASK-10` (all DONE at admission)
- Status: `DONE`
- Ledger acceptance: Define reconstruction diagnostics and their limits; Define teacher-forced and behavioral comparisons; Position Q4_K_M as a future black-box Pareto reference, not a requirement.

## Goal and boundaries

Produce `docs/architecture/quantization-validation.md` as the Phase 1 **quantization evaluation methodology** for Qwen3.8-27B language+MTP. Close the three ledger completion criteria by (1) naming local reconstruction diagnostics and locking their limits, (2) naming teacher-forced and behavioral comparison classes, and (3) positioning `models/Qwen3.8-27B-Q4_K_M.gguf` as a future black-box Pareto **reference**, not a recipe and not a requirement. Do **not** run NLL or reconstruction experiments, do **not** select a recipe map or Pareto hull, and do **not** close the ledger open question (final calibration corpora, prompt suite, capability benchmarks, and acceptance frontier remain unselected).

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Precision roles and sensitive-op hypotheses come from `docs/architecture/numerical-sensitivity.md` (TASK-07). Element formats, 22 recipes, 16 policy families, quality-risk ids, and metadata lower bounds come from `docs/architecture/quantization-design-space.md` (TASK-08). Compiler stages, conceptual profiles, calibration hook, and ownership of recipe maps / corpora come from `docs/architecture/model-compiler-plan.md` (TASK-10). Occupancy integers may be cited from TASK-01/06 through those documents.
  - Label claims `OBSERVED` (inventory/config occupancy cited), `MEASURED` (TASK-05 numbers cited only if needed to refuse quality conclusions; none required as new measurements), `DERIVED` (NLL event counts from sequence length, occupancy/byte illustrations cited from TASK-08, residual-add counts cited from TASK-07), or `HYPOTHESIS` (every methodology-risk severity, every “reconstruction will miss NLL” usefulness claim that is not a locked protocol limit). Protocol limits in heading 3 are **methodology contracts** (DERIVED from the nonlinear map from local error to NLL, plus TASK-07 residual/GDN structure), not measured error. `UNKNOWN` only for vision-encoder internals deferred here. No new `MEASURED` payload statistics. No `MEASURED` NLL, reconstruction error, or tok/s.
  - GitHub Markdown math. Cite TASK-02 equation tags for logits/MTP alignment, TASK-07 sensitive-op ids, TASK-08 family/quality-risk ids, TASK-10 profile/calibration ids. Do not rewrite forward math, re-stream safetensor payloads, recopy TASK-08’s 22-recipe grid as a new quantization space, or choose TASK-15 tiles.
  - Allowed evidence: TASK-07 numerical-sensitivity, TASK-08 quantization-design-space, TASK-10 model-compiler-plan, sitting `config.json` `text_config`, plan evidence vocabulary, this dossier, and general evaluation / teacher-forced NLL / reconstruction-error material. TASK-01/02/04/05/06 integers and ids already cited by TASK-07/08/10 may be **cited** through them. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf` as a recipe, grouping authority, compiler input, or file to open. GGUF remains a named future black-box Pareto **reference**. Safetensors is the **source** checkpoint, not the eval artifact.
- Non-goals:
  - No selected quality/balanced/compression recipe map, no “should be 4-bit”, no recommended grouping, no selected Pareto hull (`pareto_frontier_selected` false). TASK-08’s Pareto **question** stays open; this task supplies the evaluation method, not the frontier.
  - No executed reconstruction error, NLL, KL, greedy match, or capability scores (`nll_measured_here` false).
  - No final calibration corpus, eval corpus, prompt suite, capability benchmark list, or numeric acceptance frontier (ledger open question stays open).
  - No claim that any TASK-07 sensitive-op or TASK-08 quality-risk hypothesis **survives** (`hypothesis_survival_selected` false).
  - No tok/s, kernel timing, or end-to-end latency (TASK-19). Decode-complexity risks stay TASK-08 HYPOTHESIS citations.
  - No new TASK-08 recipes, bit widths, GGUF types, or `fp8_e5m2`. TASK-18 evaluates the existing 22 recipes; it does not enlarge the space.
  - No GPTQ/AWQ/Hessian as a TASK-08 scale id (`gptq_is_not_a_scale_id` true). The TASK-10 `optional_calibration_hook` may be used later with a disjoint corpus; it is not required.
  - No Quartz/llama.cpp inspection; do not open the GGUF file even if present (`gguf_payload_inspected` false; `q4km_file_required_now` false).
  - No CUDA dtypes, kernels, sitting-GPU numbers, or OPT-058 harness execution.
  - No semantic-graph, fusion, decode/prefill **schedule**, or tile **decisions** (TASK-11–17 already published where DONE; do not close their open decisions).
  - No new operators and no vision-encoder internals.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–17). Evidence scripts under `scripts/` are not a Python package.
  - Do not edit `docs/architecture/plan.md`, `task_ledger.md`, `numerical-sensitivity.md`, `quantization-design-space.md`, `model-compiler-plan.md`, `bf16-tensor-analysis.md`, `model-inventory.md`, `model-semantics.md`, `dataflow.md`, `work-and-traffic.md`, `lifetime-and-state.md`, `runtime-format-design.md`, or any TASK-09/11–17 deliverable.
  - Do not stream safetensor payloads and do not import any `scripts/check_*.py`, `scripts/analyze_bf16_tensors.py`, or `scripts/inventory_bf16_checkpoint.py`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib quantization-validation checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-32` — central path places **future experimental validation** after CUDA mappings; TASK-18 is quality/quantization validation, TASK-19 is performance.
- `docs/architecture/plan.md:35-52` — BF16 checkpoint is the authority for weights, statistics, quantization, and packing research; `models/Qwen3.8-27B-Q4_K_M.gguf` is not an architectural constraint; it is reserved as a future black-box practical baseline; Quartz/llama.cpp inspection forbidden until freeze.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:93-96` — TASK-18 defines quality and quantization validation; TASK-19 defines end-to-end and kernel performance methodology.
- `docs/architecture/task_ledger.md` TASK-00 — Q4_K_M GGUF reserved for future black-box comparison only.
- `docs/architecture/task_ledger.md` TASK-07 established results — four precision roles; 20 sensitive ops; severities HYPOTHESIS; open question (which qualitative risk hypotheses survive model-level validation) is **not** closed here; this task defines the protocol only.
- `docs/architecture/task_ledger.md` TASK-08 established results — four design dimensions; 22 recipes; 16 policy families without winners; 17 quality + 9 decode-complexity HYPOTHESIS risks; GGUF Q4_K_M is not a recipe; Pareto frontier remains an open **selection** question.
- `docs/architecture/task_ledger.md` TASK-10 established results — six compiler stages; calibration hook without a corpus; `profile_recipe_map` / `calibration_corpus` ownership `calibration`; `pareto_frontier_selected` false; `calibration_corpus_selected` false.
- `docs/architecture/task_ledger.md` TASK-18 row — produces `docs/architecture/quantization-validation.md`; purpose is local diagnostics and model-level quality comparisons; open question (final calibration corpora, prompt suite, capability benchmarks, and acceptance frontier) is **not** closed here; completion is reconstruction diagnostics and limits, teacher-forced and behavioral comparisons, and Q4_K_M as a future black-box Pareto reference, not a requirement. Downstream: future quantizer selection becomes evidence-driven rather than reconstruction-only.
- `docs/architecture/task_ledger.md` TASK-19 — tok/s / kernel / end-to-end performance methodology; not this increment.
- `docs/architecture/numerical-sensitivity.md` — 20 sensitive ops; high-risk ids include `residual_stream`, `gdn_S_recurrent`, `gdn_alpha_beta`, `state_kv_bf16`, `s_below_f32`; `n_residual_adds_language` 128 / `n_residual_adds_complete` 130; `example_T` `[1, 4096]`; `T_max` 262144; validation survival not decided there.
- `docs/architecture/quantization-design-space.md` — 22 recipes; 16 policy families; `quality_high_ids` eight ids; reconstruction/NLL/Pareto deferred here; GGUF cited only as not-a-recipe / TASK-18 reference.
- `docs/architecture/model-compiler-plan.md` — control emission = all `keep_source`; three conceptual profiles unselected; `optional_calibration_hook` without corpus; recipe maps closed later by TASK-18 **Pareto methodology** with architecture legality; this task still does not select a map.
- `docs/architecture/model-semantics.md` (via TASK-07/10 citations) — language logits $\ell^{(0)}_t$ predict token $t+1$; MTP $\ell^{(1)}_t$ consumes embedding of $t+1$ and predicts token $t+2$; complete map includes MTP; sampling over $V$ is not the forward map.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` for occupancy arithmetic. Do not read safetensor payloads. Do not read GGUF.
- `scripts/check_numerical_sensitivity.py` / `scripts/check_quantization_design_space.py` / `scripts/check_model_compiler_plan.py` — checker-style precedent. TASK-18’s checker is a sibling; do not import them.

## Performance evidence

N/A — quantization **evaluation-methodology** documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Byte counts cited from TASK-08 are DERIVED arithmetic, not selected profile outputs. NLL formulas are DERIVED event counts, not MEASURED quality. Methodology-risk **labels** are HYPOTHESIS. Do not apply the performance-evidence checklist to rank kernels or claim a winning quantizer.

- Measurement identity: N/A (no engine binary, no eval run). Future NLL identity is specified as a protocol in heading 4 (same token sequence, same $T$ horizon, keep_source control vs candidate, MTP included). That protocol is not executed here.
- Metric class: N/A for this increment. Future primary quality metric class is teacher-forced complete-map NLL (not decode-only tok/s, not complete-request latency).
- Coverage: N/A for GPU graphs. Methodology coverage is 3 eval layers, 8 reconstruction diagnostics, 6 limits, 5 teacher-forced metrics, 3 behavioral classes, 4 corpus classes, 3 Pareto axes, 6 methodology risks, and the 8 TASK-08 `quality_high_ids` survival **screens**.
- Time accounting: N/A
- Contradiction register: none at planning. If a live `text_config` occupancy product disagrees with TASK-01 family parameter counts, or a cited TASK-07/08/10 integer or id disagrees with those documents, stop and fail closed (do not remeasure, do not invent NLL).
- Claim types: occupancy/byte illustrations `observed`/`derived` (citations); NLL event-count formulas `derived`; protocol limits `derived` methodology contracts; every methodology-risk severity `hypothesis`; every “this diagnostic will catch that failure mode” usefulness beyond the locked limit table `hypothesis`.
- Target/guard roles: not opted in. No numeric ΔNLL guard.
- Evidence completeness: N/A for performance-evidence checks. Methodology completeness is the three ledger completion criteria plus explicit non-closure of the four-part open question.
- Screen eligibility: N/A (no keep/reject experiment).
- Shipping evidence: N/A

## Implementation decisions

### Authority for evaluation claims

If an occupancy product would disagree with TASK-01 / sitting `text_config`, or a sensitive-op / residual-add / $T$ citation would disagree with TASK-07, or a recipe/family/quality-risk / metadata lower bound would disagree with TASK-08, or a profile/calibration/ownership flag would disagree with TASK-10, the earlier document wins and this one is wrong.

- Prefill and decode share **one** quality methodology. Only $T$ (context horizon) and whether incoming $(K,V,C,S)$ is zeros versus populated change. Do not duplicate metric ids per mode.
- Primary object is language+MTP **checkpoint parameters** after a TASK-08 recipe (still unselected) versus control `keep_source`. Secondary object is token-persistent **state** $K,V,C,S$ when a state recipe is under test. Activations are reconstruction **probes**, not family policies.
- Algebraic equivalents in TASK-02 are the same real map; quantization acts on stored elements. Log-softmax over $V$ is an **evaluation readout** of $\ell^{(0)}$ / $\ell^{(1)}$, not a new graph node and not a sampling policy.
- Control emission = all defined families `keep_source` (`vision_deferred` skipped). Control is the identity compile for ΔNLL, not a quality winner.
- A candidate under test is a **declared** TASK-10 profile recipe map whose every assignment stays inside TASK-08 `family_candidates`. This document assigns none.
- Do not inspect Quartz, llama.cpp, or GGUF byte layouts to “confirm” K-quants, NLL harnesses, or grouping.
- Do not re-stream payloads. Cite TASK-07/08/10; do not invent histograms or error tables.
- Closing the three completion criteria does **not** close Pareto selection, corpora, prompt suite, capability benchmarks, or acceptance numbers.

### Deliverable structure (`docs/architecture/quantization-validation.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every occupancy/byte number is `OBSERVED` or `DERIVED` (citation). Every NLL event count is `DERIVED`. Every protocol-limit row is labelled as a methodology contract, not a measured error. Every methodology-risk severity is `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, TASK-07/08/10, config, checker; evidence labels; in-scope (language+MTP reconstruction screens + teacher-forced/behavioral methodology + Q4_K_M reference positioning) vs deferred (vision encoder; TASK-19 tok/s; selected corpora/frontier). State that the document specifies an **evaluation methodology**, not measurements, not a selected quantizer, and not a selected Pareto hull.
2. **Evaluation convention** — the four canonical sentences below plus the methodology sentence; three eval layers; control vs candidate; GGUF is a reference not a requirement; open question stays open.
3. **Reconstruction diagnostics and limits** — eight diagnostic ids; six limits; reconstruction may fail a candidate early and must not pass it onto the quality axis.
4. **Teacher-forced comparisons** — five metric ids; complete-map NLL primary; control-delta identity; MTP included; log-softmax is a readout.
5. **Behavioral comparisons** — three classes; secondary to NLL; suites unselected.
6. **Black-box Pareto reference** — Q4_K_M positioning; three Pareto axes; control point; compression axis uses TASK-08 lower bounds; no selected hull.
7. **Corpora, splits, and unselected acceptance** — four corpus classes; disjoint calibration vs eval; all selected flags false.
8. **Hypothesis-survival protocol and methodology risks** — screens for eight `quality_high_ids`; six methodology risks (all HYPOTHESIS); one Mermaid summary (diagram 1 of 1).
9. **Deferred vision** — residual-stream interface only; no vision eval.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Evaluation convention**, include these **four canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Reconstruction diagnostics in this document are local screens, not model-level quality.

> Teacher-forced and behavioral comparisons in this document are a methodology, not measurements.

> Q4_K_M is a future black-box Pareto reference, not a requirement.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_reconstruction` = sentence 2; `canonical_sentence_comparisons` = sentence 3; `canonical_sentence_q4km` = sentence 4.

Immediately after those four, include this **required methodology sentence verbatim**:

> Reconstruction is a local screen with named limits; teacher-forced complete-map NLL versus keep_source is the primary quality comparison; Q4_K_M is a future black-box Pareto reference, not a requirement; final corpora, prompt suite, capability benchmarks, and acceptance frontier remain unselected.

JSON: `methodology_question_sentence` = that sentence.

Bullets required under that heading:

- Three evaluation layers are complete for this task: `reconstruction`, `teacher_forced`, `behavioral`.
- Reconstruction is a local screen; it cannot place a point on the quality axis.
- Teacher-forced complete-map NLL versus `keep_source` is the primary quality comparison.
- Behavioral classes are secondary; their suites remain unselected.
- Control emission is all `keep_source`; control is an identity, not a quality winner.
- `models/Qwen3.8-27B-Q4_K_M.gguf` is not a candidate recipe, not a grouping authority, not a compiler input, not the runtime format, and not a requirement that custom profiles match or beat it.
- The ledger open question (final calibration corpora, prompt suite, capability benchmarks, and acceptance frontier) remains open (`calibration_corpus_selected` false, `eval_corpus_selected` false, `prompt_suite_selected` false, `capability_benchmark_selected` false, `acceptance_frontier_selected` false).
- Pareto hull remains unselected (`pareto_frontier_selected` false). Hypothesis survival remains unselected (`hypothesis_survival_selected` false).
- Fan-out ≠ must-store still holds; naming a diagnostic is not a CUDA store and not a measured table.
- No MEASURED NLL or reconstruction error in this document (`nll_measured_here` false).

### Evaluation layers (lock)

JSON array `eval_layer_ids` in this exact order (3 ids). JSON `n_eval_layers` = 3.

| id | Role | Can fail a candidate | Can place a Pareto quality point |
| --- | --- | --- | --- |
| `reconstruction` | Local param/activation/state screens versus source or control | yes (catastrophic local error) | no |
| `teacher_forced` | Complete-map NLL / ΔNLL / KL versus `keep_source` on teacher tokens | yes | yes (primary) |
| `behavioral` | Greedy/prompt/capability classes | yes (suite unselected) | secondary only, after a suite is selected later |

JSON: `reconstruction_is_not_quality` true; `teacher_forced_is_primary_quality` true; `behavioral_is_secondary_quality` true.

JSON `control_profile_id` = `"control"`. JSON `control_profile_is_keep_source` true. JSON `keep_source_is_identity_control` true. JSON `keep_source_is_not_quality_winner` true.

### Reconstruction diagnostics and limits (lock)

JSON array `reconstruction_diagnostic_ids` in this exact order (8 ids). JSON `n_reconstruction_diagnostics` = 8. Parallel `reconstruction_diagnostic_roles` (one of `param`, `activation`, `state`).

| ID | Role | Target | Meaning |
| --- | --- | --- | --- |
| `r_param_mse` | param | source BF16 (F32 for `state_s` `keep_source`) | Mean squared error of dequantized family elements versus source |
| `r_param_maxabs` | param | source | Max-abs dequant versus source, per defined policy family |
| `r_param_cosine` | param | source | Cosine of flattened dequant versus source, per family |
| `r_clip_frac` | param | source | Fraction of elements/groups saturating under a `clip` recipe |
| `r_act_residual` | activation | control forward | MSE at residual-add catalog points versus `keep_source` |
| `r_act_logits` | activation | control forward | MSE / max-abs of $\ell^{(0)}$ (and $\ell^{(1)}$ when MTP is in the map) versus control; **not** NLL |
| `r_state_kv` | state | conceptual BF16 $K,V$ | Store/load versus control conceptual KV (RoPE-baked $K$) |
| `r_state_s` | state | conceptual F32 $S$ | Store/load versus control conceptual $S$ (`s_below_f32` screen) |

JSON `reconstruction_diagnostic_roles` = `["param","param","param","param","activation","activation","state","state"]`.

Do not add a ninth diagnostic that is NLL. `r_act_logits` is a local output screen. Per-family reconstruction covers every TASK-08 defined policy family except `vision_deferred`. JSON `n_policy_families` = 16. JSON `vision_eval_deferred` true.

JSON array `reconstruction_limit_ids` in this exact order (6 ids). JSON `n_reconstruction_limits` = 6. These are methodology contracts, not measured tables.

| ID | Contract |
| --- | --- |
| `lim_local_not_nll` | Family MSE / cosine / max-abs does not determine teacher-forced NLL; the forward map from local error to logits is nonlinear |
| `lim_residual_accum` | Residual-path families (`attn_out`, `mlp_down`) can accumulate across `n_residual_adds_language` 128 language adds (130 complete); a small per-layer MSE can be large at logits |
| `lim_gdn_horizon` | GDN $S$ error can grow with $T$; pairs TASK-07 `gdn_S_recurrent` / `s_below_f32` |
| `lim_identity_control` | `keep_source` reconstruction is identity (error ~0 up to exact dequant of source); that is not a ranking |
| `lim_gguf_not_target` | Reconstruction target is the BF16 (F32 $S$) **source**, never Q4_K_M codes or GGUF grouping |
| `lim_single_family` | Screening one family in isolation can miss interactions with other families on the same residual stream |

Prose required: reconstruction **may** reject a candidate that is catastrophically wrong (non-finite codes, near-zero cosine, huge max-abs). Reconstruction **must not** accept a candidate onto `axis_quality`. Survival of TASK-07/08 rows is not decided by these screens alone.

Cite TASK-07 integers as substrings: `n_residual_adds_language` 128, `n_residual_adds_complete` 130, `example_T_values` `[1, 4096]`, `T_max` 262144.

### Teacher-forced comparisons (lock)

JSON array `teacher_forced_metric_ids` in this exact order (5 ids). JSON `n_teacher_forced_metrics` = 5.

| ID | Primary? | Meaning |
| --- | --- | --- |
| `tf_nll_language` | secondary | Mean NLL of $\ell^{(0)}$ versus teacher token $x_{t+1}$ |
| `tf_nll_mtp` | secondary | Mean NLL of $\ell^{(1)}$ versus teacher token $x_{t+2}$ |
| `tf_nll_complete` | **primary absolute** | Event-count mixture of language and MTP NLL |
| `tf_delta_vs_control` | **primary quality axis** | Candidate `tf_nll_complete` minus `keep_source` `tf_nll_complete` on the **same** token sequence |
| `tf_kl_vs_control` | diagnostic | Mean $\mathrm{KL}(p_{\text{control}}\Vert p_{\text{candidate}})$ over teacher positions; does not replace ΔNLL |

NLL (DERIVED methodology; not measured here). For a teacher token sequence $x_{1:T}$ with $T\ge 3$:

$$
\mathrm{NLL}_{\mathrm{lang}}=\frac{1}{T-1}\sum_{t=1}^{T-1}-\log p_{\theta}(x_{t+1}\mid x_{1:t})
\quad\text{from }\ell^{(0)},
$$

$$
\mathrm{NLL}_{\mathrm{mtp}}=\frac{1}{T-2}\sum_{t=1}^{T-2}-\log p_{\theta}^{(1)}(x_{t+2}\mid \ldots)
\quad\text{from }\ell^{(1)},
$$

$$
\mathrm{NLL}_{\mathrm{complete}}=\frac{(T-1)\,\mathrm{NLL}_{\mathrm{lang}}+(T-2)\,\mathrm{NLL}_{\mathrm{mtp}}}{(T-1)+(T-2)}.
$$

JSON: `n_language_nll_events_offset` 1 (events = $T-1$); `n_mtp_nll_events_offset` 2 (events = $T-2$); `min_eval_tokens_language` 2; `min_eval_tokens_mtp` 3; `min_eval_tokens_complete` 3; `mtp_in_primary_nll` true; `sampling_out_of_nll` true; `log_softmax_is_eval_readout` true.

JSON `example_T_values` = `[1, 4096]` matching TASK-07 context horizons (decode-style $T=1$ means one new token with populated state **and** a teacher target, not a 1-token corpus). JSON `T_max` = 262144. JSON `example_T_is_not_prompt_matrix` true.

Identity for `tf_delta_vs_control` (protocol, not executed):

- Same tokenizer and same teacher token ids (including MTP next-ids $t+1$ as graph inputs).
- Same $T$ horizon class and same empty-versus-populated incoming state convention.
- Control = TASK-10 control emission (`keep_source` on every defined family).
- Candidate = a legal TASK-08 recipe per family; this document assigns none.
- Prefill and decode share the metric; do not report decode-only tok/s as NLL.

Omitting MTP from the primary number is methodology risk `v_nll_without_mtp`. Language-only NLL may be tabulated as a **secondary** row, not the Pareto $Y$ value.

Softmax over $V$ for sampling remains out of the TASK-02 forward map. Log-softmax used here is an evaluation readout of logits already specified by TASK-02 `(22)`/`(24)` and MTP $\ell^{(1)}$.

### Behavioral comparisons (lock)

JSON array `behavioral_class_ids` in this exact order (3 ids). JSON `n_behavioral_classes` = 3.

| ID | Meaning | Suite selected? |
| --- | --- | --- |
| `beh_greedy_prefix` | Greedy argmax continuation match versus control on a prompt prefix (readout of the same logits; not sampling) | no |
| `beh_prompt_suite` | Named prompt-suite family for qualitative / long-form checks | no (`prompt_suite_selected` false) |
| `beh_capability` | Capability-benchmark family | no (`capability_benchmark_selected` false) |

Behavioral classes are **secondary**. They must not replace `tf_delta_vs_control` on `axis_quality`. Do not name WikiText, MMLU, HumanEval, or any other specific suite as **the** suite (that would close the open question). Candidate **classes** above are the research space.

JSON `prompt_suite_selected` false; `capability_benchmark_selected` false.

### Black-box Pareto reference (lock)

JSON array `pareto_axis_ids` in this exact order (3 ids). JSON `n_pareto_axes` = 3.

| ID | Axis | What is plotted | Selected? |
| --- | --- | --- | --- |
| `axis_quality` | $Y$ | `tf_delta_vs_control` (primary); behavioral only as a later secondary overlay | method locked; values unmeasured |
| `axis_compression` | $X$ | TASK-08 payload+metadata **lower bound** bytes for a **declared** legal recipe map, not a packed TASK-09 layout | method locked; no map declared |
| `axis_reference_q4km` | reference point | Future black-box Q4_K_M (file size + quality metric with explicit identity) | not a recipe; not required |

JSON booleans (lock):

- `gguf_is_not_a_recipe` true
- `gguf_is_not_a_requirement` true
- `gguf_is_pareto_reference` true
- `gguf_is_not_the_runtime_format` true
- `gguf_is_not_compiler_input` true
- `gguf_payload_inspected` false
- `q4km_must_be_beaten` false
- `q4km_must_be_matched` false
- `q4km_file_required_now` false
- `q4km_nll_required` false
- `q4km_nll_identity_matched_when_logits_available` true
- `pareto_frontier_selected` false
- `compiler_profile_selected` false

JSON `q4km_named_path` exactly `models/Qwen3.8-27B-Q4_K_M.gguf`.

Q4_K_M protocol (future, after freeze; not run here):

- Treat the GGUF runtime as a **black box**. Do not copy K-quant grouping into TASK-08 recipes.
- If that runtime can emit teacher-forced logits on the **same** eval token sequences, plot NLL with a matching identity and label the point `axis_reference_q4km`.
- If logits are unavailable, NLL for that point stays `unknown`; a behavioral black-box metric may still be plotted and must be labelled a different identity (do not ratio it against `tf_delta_vs_control`).
- Custom profiles are **not** required to beat or match Q4_K_M quality or size.
- Control point: `keep_source` at unique-non-embed BF16 bytes `52098598912`, `tf_delta_vs_control` = 0 by definition.

Compression-axis **illustration** (DERIVED citation from TASK-08, **not** a selected map; must appear as substrings): unique-non-embed int4 $g=128$ total `13431670032` B, ratio `0.2578125`; MLP $n=17112760320$, MLP BF16 `34225520640`. Label every use with `example` and `not a selected winner`.

Do not rank decode-complexity risks on this plot. Tok/s is TASK-19 (`toks_is_not_quality_axis` true).

### Corpora, splits, and unselected acceptance (lock)

JSON array `corpus_class_ids` in this exact order (4 ids). JSON `n_corpus_classes` = 4.

| ID | Use | Owner (TASK-10 class) | Selected? |
| --- | --- | --- | --- |
| `corpus_calibration` | TASK-10 `optional_calibration_hook` (activation-aware scale **values** on an existing recipe) | `calibration` | false |
| `corpus_eval_nll` | Teacher-forced NLL / KL | `calibration` (eval split) | false |
| `corpus_prompt` | `beh_prompt_suite` | `calibration` | false |
| `corpus_capability` | `beh_capability` | `calibration` | false |

JSON: `calibration_corpus_selected` false; `eval_corpus_selected` false; `prompt_suite_selected` false; `capability_benchmark_selected` false; `acceptance_frontier_selected` false; `ledger_open_question_corpora_closed` false.

Protocol (lock, no datasets named as winners):

- Calibration ∩ eval = empty. Using eval tokens in `optional_calibration_hook` is `v_calib_eval_leak`.
- GPTQ/AWQ/Hessian remain **not** a TASK-08 scale id. They may only produce scale **values** attached to an existing integer recipe via the hook (`gptq_is_not_a_scale_id` true, `activation_aware_scale_id_added` false, `gptq_required` false).
- Do not name a numeric ΔNLL / perplexity / match-rate threshold. Acceptance frontier is an open question.
- Candidate corpus **shapes** may be described as `held_out_text` and `long_context_text` in prose (horizon $T$ up to `T_max`) without selecting a named dataset.

JSON `calibration_eval_must_be_disjoint` true.

### Hypothesis-survival protocol and methodology risks (lock)

A TASK-07 sensitive-op or TASK-08 quality-risk hypothesis **survives** only after a future measured `tf_delta_vs_control` (and the attached reconstruction screen) shows a quality-visible effect relative to an acceptance frontier that is **not** selected here. This document declares **no** survivor. JSON `hypothesis_survival_selected` false.

JSON array `quality_high_ids` copied from TASK-08 in TASK-08 order (8 ids). JSON `n_quality_high` = 8:

`q_norm_gamma`, `q_gdn_time`, `q_gdn_gate`, `q_attn_out`, `q_mlp_down`, `q_state_kv`, `q_state_s`, `q_int2_mass`.

JSON array `survival_reconstruction_ids` parallel to `quality_high_ids`:

`r_param_mse`, `r_state_s`, `r_state_s`, `r_act_residual`, `r_act_residual`, `r_state_kv`, `r_state_s`, `r_param_mse`.

JSON `survival_requires_teacher_forced` true (every high-risk screen still needs `tf_nll_complete` / `tf_delta_vs_control`).

JSON array `sensitive_high_ids` copied from TASK-07 in TASK-07 order (9 ids) as citations, not as extra eval metrics:

`residual_stream`, `rms_hidden`, `softmax_over_T`, `attn_av_over_T`, `gdn_S_recurrent`, `gdn_alpha_beta`, `rope_phase`, `state_kv_bf16`, `s_below_f32`.

JSON `n_sensitive_high` = 9. Do not claim any of these survive.

JSON array `methodology_risk_ids` in this exact order (6 ids). Parallel `methodology_risk_severities`. Every severity is HYPOTHESIS. JSON `n_methodology_risks` = 6.

| ID | Ties to | Severity | Claim (must remain HYPOTHESIS) |
| --- | --- | --- | --- |
| `v_recon_as_quality` | `reconstruction` layer | high | Treating reconstruction MSE/cosine as sufficient for Pareto $Y$ would freeze selection without model-level evidence |
| `v_gguf_as_requirement` | `axis_reference_q4km` | high | Requiring custom profiles to match or beat Q4_K_M would turn a black-box reference into a design constraint |
| `v_calib_eval_leak` | `corpus_calibration` / `corpus_eval_nll` | high | Calibrating on eval tokens would inflate teacher-forced comparisons |
| `v_control_as_winner` | `keep_source` | medium | Treating identity control as a quality winner would confuse ΔNLL with a selected profile |
| `v_nll_without_mtp` | `tf_nll_complete` | medium | Omitting MTP from the primary NLL would drop the complete map |
| `v_toks_as_quality` | TASK-19 | low | Using tok/s as a quality axis would mix TASK-19 performance into TASK-18 Pareto $Y$ |

JSON `methodology_risk_severities` = `["high","high","high","medium","medium","low"]`. JSON `methodology_high_ids`: `v_recon_as_quality`, `v_gguf_as_requirement`, `v_calib_eval_leak`. `methodology_medium_ids`: `v_control_as_winner`, `v_nll_without_mtp`. `methodology_low_ids`: `v_toks_as_quality`. JSON `n_methodology_high` = 3, `n_methodology_medium` = 2, `n_methodology_low` = 1.

Do not rank these by wall time. Do not convert TASK-06 bottleneck labels into measurements. Do not convert TASK-07/08 HYPOTHESIS rows into MEASURED survivors.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 8 (Hypothesis-survival protocol and methodology risks). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers, 22 recipes, or eval sequences.

Required IDs **inside that fence**: `reconstruction`, `teacher_forced`, `behavioral`, `q4km`, `control`, `pareto`, `calibration`, `eval`.

JSON `n_diagrams` is 1. `diagram_ids` is `["reconstruction","teacher_forced","behavioral","q4km","control","pareto","calibration","eval"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger reconstruction and NLL are **UNKNOWN**. `vision_deferred` has an empty TASK-08 candidate list and is skipped by control and by every eval layer here. Do not add vision diagnostics. The word `UNKNOWN` may appear only in this section of the deliverable.

### Occupancy and citation integers (lock)

Live from `text_config` (checker recomputes; do not hardcode without asserting against config):

| JSON key | Value | How |
| --- | ---: | --- |
| `hidden_size` | 5120 | config |
| `intermediate_size` | 17408 | config |
| `vocab_size` | 248320 | config |
| `n_decoder_layers` | 64 | config |
| `n_linear_layers` | 48 | `layer_types` |
| `n_full_layers` | 16 | `layer_types` |
| `n_mtp_blocks` | 1 | `mtp_num_hidden_layers` |
| `T_max` | 262144 | `max_position_embeddings` |
| `mlp_n` | 17112760320 | $3\times 64\times 17408\times 5120$ |
| `embed_n` | 1271398400 | $248320\times 5120$ |
| `n_language_mtp_tensors` | 866 | TASK-01 citation |
| `n_language_mtp_parameters` | 27320697856 | TASK-01 citation |
| `mlp_bf16_bytes` | 34225520640 | $2\times mlp_n$ |
| `weight_bytes_language_mtp_excl_vision` | 54641395712 | TASK-06/08 citation |
| `weight_bytes_unique_non_embed` | 52098598912 | TASK-06/08 citation |
| `unique_non_embed_int4_g128_total_bytes` | 13431670032 | TASK-08 illustration |
| `mlp_int4_g128_over_bf16` | 0.2578125 | TASK-08 illustration |
| `s_f32_bytes` | 150994944 | TASK-04/06/07 citation |
| `n_residual_adds_language` | 128 | TASK-07 |
| `n_residual_adds_complete` | 130 | TASK-07 |

JSON `full_attention_indices` = `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`. JSON `text_config.dtype` via live config `"bfloat16"`; `mamba_ssm_dtype` `"float32"`. JSON `bytes_bf16` 2, `bytes_f32` 4.

JSON `authority` exactly `.cache/authorities/qwen3.8-27b-transformers`.

JSON `policy_families` equals the TASK-08 16-id list. JSON `candidate_recipe_ids` equals the TASK-08 22-id list. JSON `n_candidate_recipes` 22. Duplicate those id lists as checker constants; do not import TASK-08’s script.

TASK-08 `candidate_recipe_ids` order (22): `keep_source`, `narrow_bf16`, `fp8_tensor`, `i8_tensor`, `i8_row`, `i8_row_asym`, `i8_g32`, `i6_row`, `i4_row`, `i4_col`, `i4_g32`, `i4_g64`, `i4_g128`, `i4_g128_p99`, `i4_g128_rms`, `i4_g128_extract`, `i4_g32_mixed`, `i4_row_extract`, `i4_clip`, `i3_g32`, `i3_g32_extract`, `i2_g32_extract`.

TASK-08 `policy_families` order (16): `norm_gamma`, `gdn_time_param`, `gdn_gate_proj`, `conv1d`, `linear_large_proj`, `attn_qkv`, `attn_out`, `mlp_up_gate`, `mlp_down`, `embed_table`, `lm_head`, `mtp_fc`, `vision_deferred`, `state_kv`, `state_c`, `state_s`.

JSON `quality_risk_ids` equals the TASK-08 17-id list (citation completeness; do not retabulate severities as new measurements). JSON `n_quality_risks` 17.

### Additional locked booleans

- `nll_measured_here` false
- `payloads_restreamed` false
- `experiments_run` false
- `methodology_defined` true
- `toks_is_not_quality_axis` true
- `gptq_is_not_a_scale_id` true
- `gptq_required` false
- `activation_aware_scale_id_added` false
- `activation_quant_in_family_policies` false
- `keep_source_packable_on_all_defined_families` true
- `vision_eval_deferred` true
- `safetensors_is_source_not_runtime` true

### Tooling

Create `scripts/check_quantization_validation.py` (Python 3.11+, stdlib only: `argparse`, `json`, `math`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py` or `scripts/analyze_bf16_tensors.py` or `scripts/inventory_bf16_checkpoint.py`; duplicate the small `text_config` arithmetic needed for occupancy and the MLP/embed products (same identities as TASK-08/10 illustrations). Duplicate TASK-07 residual-add / `example_T` / `T_max` / `sensitive_high_ids` constants and TASK-08 recipe/family/quality-high/quality-risk id lists as constants; do not import them. Do not open `models/Qwen3.8-27B-Q4_K_M.gguf` or any safetensor.

CLI (cwd = repository root):

```text
python3 scripts/check_quantization_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  [--json]

python3 scripts/check_quantization_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-validation docs/architecture/quantization-validation.md
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `max_position_embeddings`. Derived: `mlp_n`, `embed_n`, family BF16 byte citations matching TASK-01/06, packed/metadata illustrations matching TASK-08 tight totals. Constant fields: canonical sentences, methodology sentence, eval-layer / diagnostic / limit / metric / corpus / axis / risk lists, TASK-08 recipe/family/quality ids, TASK-07 high-risk ids and residual-add counts.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--quantization-validation PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all four canonical sentences verbatim, (5) `methodology_question_sentence` verbatim, (6) every `eval_layer_ids`, `reconstruction_diagnostic_ids`, `reconstruction_limit_ids`, `teacher_forced_metric_ids`, `behavioral_class_ids`, `pareto_axis_ids`, `corpus_class_ids`, `methodology_risk_ids`, `quality_high_ids`, `policy_families`, and `candidate_recipe_ids` id present as a substring, (7) the diagram’s required IDs present **inside that mermaid fence**, (8) none of `TBD`, `TODO`, `???`, (9) no `UNKNOWN` except inside the Deferred vision section, (10) every locked document integer/decimal below present as a decimal or integer substring, (11) the words `HYPOTHESIS` and `not a selected winner` present, (12) none of the forbidden phrases: `selected winner`, `Pareto frontier is`, `Q4_K_M is required`, `must beat Q4_K_M`, `must match Q4_K_M`, `should use GGUF`, `GGUF is the runtime format`, `GGUF is the compiler input`, `reconstruction is sufficient`, `selected calibration corpus`, `selected prompt suite`, `selected capability benchmark`, `acceptance frontier is`, `should be 4-bit`, `GPTQ is required`, `quality winner` (allow those substrings only inside `not a selected winner` / `not selected winners` / `not a quality winner` / `not a requirement` / the canonical Q4_K_M sentence). Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).
- Do not fail if the GGUF file is absent. Do not stat/open GGUF as a checker requirement.

Do not read safetensor payloads. Do not require other architecture markdown JSON equality (verifier, not this checker, spot-checks TASK-07/08/10 integers and ids against those documents).

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `T_max==262144`
- `mlp_n==17112760320` and `mlp_n == 3 * n_decoder_layers * intermediate_size * hidden_size`
- `embed_n==1271398400` and `embed_n == vocab_size * hidden_size`
- `n_language_mtp_tensors==866`, `n_language_mtp_parameters==27320697856`
- `mlp_bf16_bytes==34225520640==2*mlp_n`
- `mlp_int4_g128_over_bf16==0.2578125`
- `unique_non_embed_int4_g128_total_bytes==13431670032`
- `weight_bytes_language_mtp_excl_vision==54641395712`, `weight_bytes_unique_non_embed==52098598912`
- `s_f32_bytes==150994944`
- `n_residual_adds_language==128`, `n_residual_adds_complete==130`
- `example_T_values == [1, 4096]`
- `n_eval_layers==3`, `n_reconstruction_diagnostics==8`, `n_reconstruction_limits==6`
- `n_teacher_forced_metrics==5`, `n_behavioral_classes==3`, `n_pareto_axes==3`, `n_corpus_classes==4`
- `n_methodology_risks==6`, `n_methodology_high==3`, `n_methodology_medium==2`, `n_methodology_low==1`
- `n_quality_high==8`, `n_quality_risks==17`, `n_sensitive_high==9`
- `n_candidate_recipes==22`, `n_policy_families==16`, `n_diagrams==1`
- `candidate_recipe_ids` equals the TASK-08 22-id list; `policy_families` equals the TASK-08 16-id list
- `quality_high_ids` equals the TASK-08 eight-id list
- `survival_reconstruction_ids` equals the locked parallel list
- `min_eval_tokens_complete==3`, `mtp_in_primary_nll is True`
- `gguf_is_not_a_recipe is True`, `gguf_is_not_a_requirement is True`, `gguf_is_pareto_reference is True`
- `gguf_payload_inspected is False`, `q4km_must_be_beaten is False`, `q4km_file_required_now is False`
- `pareto_frontier_selected is False`, `compiler_profile_selected is False`
- `calibration_corpus_selected is False`, `eval_corpus_selected is False`, `prompt_suite_selected is False`, `capability_benchmark_selected is False`, `acceptance_frontier_selected is False`
- `hypothesis_survival_selected is False`, `ledger_open_question_corpora_closed is False`
- `nll_measured_here is False`, `experiments_run is False`, `methodology_defined is True`
- `reconstruction_is_not_quality is True`, `teacher_forced_is_primary_quality is True`
- `keep_source_is_identity_control is True`, `keep_source_is_not_quality_winner is True`
- `toks_is_not_quality_axis is True`, `gptq_is_not_a_scale_id is True`, `gptq_required is False`
- `q4km_named_path == "models/Qwen3.8-27B-Q4_K_M.gguf"`
- `methodology_risk_severities` equals the locked parallel list

Locked document integers/decimals the `--quantization-validation` check must find:

`5120`, `17408`, `248320`, `866`, `27320697856`, `17112760320`, `34225520640`, `54641395712`, `52098598912`, `150994944`, `13431670032`, `0.2578125`, `262144`, `4096`, `128`, `130`

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `full_attention_indices`, `bytes_bf16`, `bytes_f32`, `T_max`,

`n_language_mtp_tensors`, `n_language_mtp_parameters`, `mlp_n`, `embed_n`, `mlp_bf16_bytes`, `mlp_int4_g128_over_bf16`,

`weight_bytes_language_mtp_excl_vision`, `weight_bytes_unique_non_embed`, `unique_non_embed_int4_g128_total_bytes`, `s_f32_bytes`, `n_residual_adds_language`, `n_residual_adds_complete`, `example_T_values`, `example_T_is_not_prompt_matrix`,

`eval_layer_ids`, `n_eval_layers`, `reconstruction_diagnostic_ids`, `reconstruction_diagnostic_roles`, `n_reconstruction_diagnostics`, `reconstruction_limit_ids`, `n_reconstruction_limits`,

`teacher_forced_metric_ids`, `n_teacher_forced_metrics`, `n_language_nll_events_offset`, `n_mtp_nll_events_offset`, `min_eval_tokens_language`, `min_eval_tokens_mtp`, `min_eval_tokens_complete`, `mtp_in_primary_nll`, `sampling_out_of_nll`, `log_softmax_is_eval_readout`,

`behavioral_class_ids`, `n_behavioral_classes`,

`pareto_axis_ids`, `n_pareto_axes`, `q4km_named_path`,

`corpus_class_ids`, `n_corpus_classes`, `calibration_eval_must_be_disjoint`,

`quality_high_ids`, `n_quality_high`, `survival_reconstruction_ids`, `survival_requires_teacher_forced`, `quality_risk_ids`, `n_quality_risks`, `sensitive_high_ids`, `n_sensitive_high`,

`methodology_risk_ids`, `methodology_risk_severities`, `methodology_high_ids`, `methodology_medium_ids`, `methodology_low_ids`, `n_methodology_risks`, `n_methodology_high`, `n_methodology_medium`, `n_methodology_low`,

`policy_families`, `candidate_recipe_ids`, `n_policy_families`, `n_candidate_recipes`, `keep_source_packable_on_all_defined_families`, `control_profile_id`, `control_profile_is_keep_source`,

`reconstruction_is_not_quality`, `teacher_forced_is_primary_quality`, `behavioral_is_secondary_quality`, `keep_source_is_identity_control`, `keep_source_is_not_quality_winner`,

`gguf_is_not_a_recipe`, `gguf_is_not_a_requirement`, `gguf_is_pareto_reference`, `gguf_is_not_the_runtime_format`, `gguf_is_not_compiler_input`, `gguf_payload_inspected`, `q4km_must_be_beaten`, `q4km_must_be_matched`, `q4km_file_required_now`, `q4km_nll_required`, `q4km_nll_identity_matched_when_logits_available`,

`pareto_frontier_selected`, `compiler_profile_selected`, `calibration_corpus_selected`, `eval_corpus_selected`, `prompt_suite_selected`, `capability_benchmark_selected`, `acceptance_frontier_selected`, `hypothesis_survival_selected`, `ledger_open_question_corpora_closed`,

`nll_measured_here`, `payloads_restreamed`, `experiments_run`, `methodology_defined`, `toks_is_not_quality_axis`, `gptq_is_not_a_scale_id`, `gptq_required`, `activation_aware_scale_id_added`, `activation_quant_in_family_policies`, `vision_eval_deferred`, `safetensors_is_source_not_runtime`,

`diagram_ids`, `n_diagrams`, `canonical_sentence_logical`, `canonical_sentence_reconstruction`, `canonical_sentence_comparisons`, `canonical_sentence_q4km`, `methodology_question_sentence`.

Integer JSON fields that are counts/widths/bytes/offsets are JSON ints. Ratio `mlp_int4_g128_over_bf16` is JSON number `0.2578125`. Booleans are JSON booleans. `full_attention_indices` and `example_T_values` are JSON arrays of ints. `survival_reconstruction_ids` is a JSON array of strings parallel to `quality_high_ids`.

### Stage split

- **Implementation** writes `scripts/check_quantization_validation.py` **and** `docs/architecture/quantization-validation.md` (three eval layers, eight reconstruction diagnostics, six limits, five teacher-forced metrics, three behavioral classes, Q4_K_M reference positioning, four unselected corpus classes, eight high-risk survival **screens** without survivors, six HYPOTHESIS methodology risks, JSON fence). Runs `--json` and `--quantization-validation` after the document exists. Records command outcomes in this dossier. Does not commit. Does not stream payloads. Does not open GGUF. Does not run NLL.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-07 / `numerical-sensitivity.md` / TASK-08 / `quantization-design-space.md` / TASK-10 / `model-compiler-plan.md` style), Authority table links to this dossier / numerical-sensitivity / quantization-design-space / model-compiler-plan / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, diagnostic ids, metric ids, severities, or Mermaid node IDs. Does not edit TASK-07/08/10 artifacts.
- **Verification** independently re-runs focused commands, recomputes `mlp_n` / unique-non-embed citation from sitting `text_config` (not from JSON echo), spot-checks cited TASK-07/08/10 integers and ids against `docs/architecture/numerical-sensitivity.md`, `docs/architecture/quantization-design-space.md`, and `docs/architecture/model-compiler-plan.md` (not from this JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF-as-recipe-or-requirement, no winner phrases, no `plan.md` or ledger edit, no payload I/O, no GGUF open, that every methodology-risk row is labelled HYPOTHESIS, that reconstruction cannot place Pareto $Y$, that teacher-forced complete-map NLL versus `keep_source` is primary, that Q4_K_M is a reference not a requirement, and that corpora / prompt suite / capability benchmarks / acceptance frontier / Pareto hull / hypothesis survival remain unselected. The three ledger completion criteria are closed by the methodology. TASK-07 survival and TASK-08 Pareto **selection** questions remain open.

- Invariants:
  - Ten level-2 headings in the locked order; four canonical sentences plus `methodology_question_sentence` verbatim; 3 eval layers; 8 reconstruction diagnostics; 6 limits; 5 teacher-forced metrics; 3 behavioral classes; 3 Pareto axes; 4 corpus classes; 6 methodology risks; one Mermaid flowchart with required IDs.
  - Prefill/decode share one methodology; primary NLL includes language+MTP; control is `keep_source` identity; activations are probes not family policies.
  - Reconstruction is a screen with named limits; it is not model-level quality.
  - Q4_K_M is a future black-box Pareto reference, not a recipe and not a requirement. GGUF is not opened in this increment.
  - `pareto_frontier_selected` false; all corpus/suite/frontier/survival selected flags false; `nll_measured_here` false.
  - GPTQ is not a scale id and not required. Tok/s is not a quality axis.
  - Logical values do not imply allocation. Vision encoder remains unexpanded.
- Rejected alternatives:
  - Selecting a Pareto hull or recipe map from TASK-05 absmax, TASK-06 intensities, or TASK-08 candidate lists: rejected; those are not quality measurements; this task defines how a future frontier would be measured.
  - Treating reconstruction MSE as sufficient quality: rejected; ledger purpose is evidence-driven rather than reconstruction-only; `lim_local_not_nll` / `v_recon_as_quality`.
  - Requiring custom profiles to match or beat Q4_K_M: rejected; plan.md black-box baseline, not a constraint; completion criterion 3.
  - Copying GGUF K-quant grouping into TASK-08 recipes: rejected; GGUF is not a recipe; `lim_gguf_not_target`.
  - Running NLL / OPT-058 / reconstruction experiments in Phase 1: rejected; methodology only; `nll_measured_here` false.
  - Selecting WikiText or any named eval/calibration dataset as **the** corpus: rejected; ledger open question stays open.
  - Locking a numeric ΔNLL / perplexity acceptance threshold: rejected; acceptance frontier unselected.
  - Omitting MTP from primary NLL: rejected; complete map includes MTP; language-only is secondary.
  - Using greedy/capability scores as the primary quality axis: rejected; teacher-forced ΔNLL is primary; behavioral suites unselected.
  - Using TASK-19 tok/s as Pareto $Y$: rejected; `v_toks_as_quality`.
  - Adding `fp8_e5m2`, `int5`, or GGUF types to the recipe space: rejected; TASK-18 evaluates the locked 22 recipes.
  - Adding GPTQ/AWQ as a fifth TASK-08 scale id: rejected; TASK-10 hook only; not required.
  - Opening the GGUF file or inspecting Quartz/llama.cpp NLL harnesses: forbidden by plan.md.
  - Claiming any TASK-07/08 hypothesis survives: rejected; protocol only.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–17.
  - Editing frozen TASK-07/08/10 docs, the ledger, or `plan.md`.
  - Importing other `check_*.py` or `analyze_bf16_tensors.py`.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/quantization-validation.md` exists and follows the heading list above.
  - Reconstruction diagnostics (8) and limits (6) are named; reconstruction is a local screen that cannot place Pareto $Y$.
  - Teacher-forced metrics (5) include complete-map NLL and ΔNLL versus `keep_source` as primary; MTP is in the primary number; behavioral classes (3) are secondary with suites unselected.
  - Q4_K_M is positioned as a future black-box Pareto reference, not a recipe and not a requirement; GGUF is not opened.
  - Corpora, prompt suite, capability benchmarks, and acceptance frontier remain unselected; Pareto hull and hypothesis survival remain unselected.
  - Methodology risks (6) are tabulated as HYPOTHESIS and do not claim experimental proof or a selected frontier.
  - Four canonical sentences plus `methodology_question_sentence` verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp; no payload re-stream; no `plan.md` or ledger edit; no MEASURED NLL.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_quantization_validation.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_quantization_validation.py
python3 scripts/check_quantization_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_quantization_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-validation docs/architecture/quantization-validation.md
```

- Candidate quality: not required — no model execution or NLL; this increment is evaluation-methodology documentation. Teacher-forced **metrics** are protocol, not OPT-058 measurements.
- Repository-wide commands:

```sh
test -f docs/architecture/quantization-validation.md
python3 -m py_compile scripts/check_quantization_validation.py
python3 scripts/check_quantization_validation.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --quantization-validation docs/architecture/quantization-validation.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates. Do not run `scripts/analyze_bf16_tensors.py`. Do not open GGUF.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/quantization-validation.md` (create)
  - `scripts/check_quantization_validation.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-07/08/10 deliverables unchanged)
- Definition of done: quantization-validation document published with locked reconstruction diagnostics and limits, teacher-forced and behavioral comparison classes, and Q4_K_M as a future black-box Pareto reference not a requirement; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-18 completion checkboxes can be marked at delivery; final corpora, prompt suite, capability benchmarks, acceptance frontier, Pareto hull, and hypothesis survival remain unselected.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T16:20:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-18.md`. Coupled IDs `none`. Document structure (10 headings), four canonical sentences plus methodology sentence, three eval layers, eight reconstruction diagnostics, six limits, five teacher-forced metrics, three behavioral classes, three Pareto axes with Q4_K_M as reference-not-requirement, four unselected corpus classes, eight high-risk survival screens without survivors, six HYPOTHESIS methodology risks, stdlib checker `scripts/check_quantization_validation.py`, JSON schema, and acceptance commands are closed. Ledger open question left open (corpora / prompt suite / capability benchmarks / acceptance frontier). TASK-07 survival and TASK-08 Pareto **selection** remain unselected. `docs/architecture/quantization-validation.md` and the checker were **not** written in this stage. `plan.md` and `task_ledger.md` not edited. No commit.
- Performance evidence applied: N/A — evaluation-methodology documentation; no sink ranking, no measured NLL, no tok/s

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_quantization_validation.py` (stdlib checker: live `text_config` occupancy, locked methodology constants, `--json` asserts, `--quantization-validation` document checks). Does not import other `scripts/check_*.py`. Does not open GGUF or safetensor payloads.
  - Created `docs/architecture/quantization-validation.md` (10 locked headings, four canonical sentences plus `methodology_question_sentence`, 3 eval layers, 8 reconstruction diagnostics, 6 limits, 5 teacher-forced metrics, 3 behavioral classes, Q4_K_M as future black-box Pareto reference not a requirement, 4 unselected corpus classes, 8 high-risk survival screens without survivors, 6 HYPOTHESIS methodology risks, one `flowchart TB` mermaid with required node IDs, JSON fence equal to live `--json`).
  - No edits to `plan.md`, `task_ledger.md`, TASK-07/08/10 deliverables, or other frozen docs. No NLL/reconstruction experiments. No commit.
- Commands:
  - `python3 -m py_compile scripts/check_quantization_validation.py` — exit 0
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — exit 0 (internal asserts pass; pretty-printed summary object)
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md` — exit 0 (headings, verbatim sentences, mermaid IDs, JSON fence equality, locked integers, forbidden-phrase scan)
- UTC/time/tokens/cost: `2026-09-20T16:23:29Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/quantization-validation.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-07 / [`numerical-sensitivity.md`](../numerical-sensitivity.md) / TASK-08 / [`quantization-design-space.md`](../quantization-design-space.md) / TASK-10 / [`model-compiler-plan.md`](../model-compiler-plan.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`numerical-sensitivity.md`](../numerical-sensitivity.md) (TASK-07), [`quantization-design-space.md`](../quantization-design-space.md) (TASK-08), [`model-compiler-plan.md`](../model-compiler-plan.md) (TASK-10), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_quantization_validation.py`](../../../scripts/check_quantization_validation.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the first JSON fence left unchanged (live `--json` equality confirmed). Locked integers, canonical sentences, diagnostic/metric ids, severities, and Mermaid node IDs untouched. TASK-07/08/10 artifacts not edited.
- Commands:
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md` — pass (exit 0; headings, JSON fence, one flowchart, canonical sentences, diagnostic/metric ids, diagram node IDs; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T16:24:08Z`; `telemetry_unavailable`

### Verification

- Attempt: 1 (first pass)
- Agent/model: `composer-2.5` (verifier subagent; parent/inherit mapping)
- Diff review:
  - New: `docs/architecture/quantization-validation.md`, `scripts/check_quantization_validation.py`, `docs/architecture/tasks/TASK-18.md` (untracked).
  - Modified tracked: `docs/architecture/task_ledger.md` only (`TODO` → `IN PROGRESS`; expected at admission, not delivery).
  - `docs/architecture/plan.md` unchanged. TASK-07/08/10 deliverables unchanged.
  - Checker is stdlib-only (`argparse`, `json`, `math`, `re`, `sys`, `pathlib`); no import of other `scripts/check_*.py`.
  - Document: draft-status `unverified` banner before first `##`; ten locked `##` headings in order; four canonical sentences plus `methodology_question_sentence` verbatim; three eval layers; eight reconstruction diagnostics; six limits; five teacher-forced metrics; three behavioral classes; Q4_K_M as future black-box Pareto reference not a requirement; four unselected corpus classes; eight high-risk survival screens without survivors; six HYPOTHESIS methodology risks; one Mermaid flowchart with required node IDs; Deferred vision `UNKNOWN`; no forbidden winner phrases outside allowed negations.
- Independent raw-record checks:
  - Live `--json` vs fenced JSON in `quantization-validation.md`: 118 keys, deep-equal after parse.
  - Locked counts: `n_eval_layers` 3; `n_reconstruction_diagnostics` 8; `n_reconstruction_limits` 6; `n_teacher_forced_metrics` 5; `n_behavioral_classes` 3; `n_pareto_axes` 3; `n_corpus_classes` 4; `n_quality_high` 8; `n_methodology_risks` 6; `n_policy_families` 16; `n_candidate_recipes` 22; `n_diagrams` 1.
  - Selection flags all false as required: `pareto_frontier_selected`, `hypothesis_survival_selected`, `calibration_corpus_selected`, `eval_corpus_selected`, `prompt_suite_selected`, `capability_benchmark_selected`, `acceptance_frontier_selected`, `ledger_open_question_corpora_closed`, `compiler_profile_selected`, `q4km_must_be_beaten`, `q4km_must_be_matched`, `gguf_payload_inspected`, `nll_measured_here`.
  - Primary quality axis: `teacher_forced_is_primary_quality` true; `reconstruction_is_not_quality` true; `behavioral_is_secondary_quality` true; `gguf_is_not_a_requirement` true; `gguf_is_pareto_reference` true; `methodology_defined` true.
  - Occupancy from sitting `text_config`: `hidden_size` 5120; `n_decoder_layers` 64; `n_language_mtp_parameters` 27320697856; `weight_bytes_language_mtp_excl_vision` 54641395712; TASK-07 integers `n_residual_adds_language` 128 / `n_residual_adds_complete` 130 / `T_max` 262144 / `example_T_values` `[1, 4096]` present as substrings.
- Commands:
  - `test -f docs/architecture/quantization-validation.md` — pass (exit 0)
  - `python3 -m py_compile scripts/check_quantization_validation.py` — pass (exit 0)
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; 118 keys; critical flags match acceptance)
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md` — pass (exit 0)
  - `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json --quantization-validation docs/architecture/quantization-validation.md` — pass (exit 0)
  - Independent JSON fence spot-check (live `--json` object equals fenced JSON, 118 keys deep-equal) — pass
  - `git diff docs/architecture/plan.md` — empty (unchanged)
- Formatting changed files: none (`uv run ruff format` not required for this increment)
- Verdict: **PASS**
- UTC/time/tokens/cost: `2026-09-20T16:25:00Z`; `telemetry_unavailable`

### Retries and escalation

none

### Delivery

- Agent/model: `composer-2.5` (delivery subagent)
- Scope: TASK-18 only; coupled IDs `none`
- Outcome: TASK-18 marked `DONE` after verification PASS (attempt 1)
- UTC/time/tokens/cost: `2026-09-20T16:26:00Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification PASS (attempt 1) — `docs/architecture/quantization-validation.md` (ten locked headings, four canonical sentences plus methodology sentence, three eval layers, eight reconstruction diagnostics, six limits, five teacher-forced metrics, three behavioral classes, Q4_K_M as future black-box Pareto reference not a requirement, four unselected corpus classes, eight high-risk survival screens without survivors, six HYPOTHESIS methodology risks, one Mermaid flowchart, JSON fence equal to live `--json`); `scripts/check_quantization_validation.py` stdlib checker; `plan.md` unchanged; Pareto selection, corpora, prompt suite, capability benchmarks, acceptance frontier, and hypothesis survival remain unselected (`pareto_frontier_selected` false; `calibration_corpus_selected` false; `eval_corpus_selected` false; `prompt_suite_selected` false; `capability_benchmark_selected` false; `acceptance_frontier_selected` false; `hypothesis_survival_selected` false; `nll_measured_here` false)
- Candidate measured delta: N/A — evaluation-methodology documentation
- Shipping delta: N/A
- Quality result: not required
- Evidence completeness: N/A for performance-evidence checks
- Throughput delta (when applicable): N/A
- Commit: delivery commit on `clean-sheet` (see git log)
- Push: `origin/clean-sheet`
- First-pass acceptance: **yes**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands; GGUF file is not required now
