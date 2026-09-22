# TASK-07 — Study numerical sensitivity from the mathematics

## Control

- Primary ID: `TASK-07`
- Coupled IDs: `none`
- Dependencies: `TASK-02`, `TASK-04` (all DONE at admission); TASK-06 `work-and-traffic.md` is a citation for contraction widths, not a ledger dependency
- Status: `DONE`
- Ledger acceptance: Analyze requested sensitive operations and accumulation paths; distinguish all relevant precision roles; classify risk without claiming experimental proof.

## Goal and boundaries

Produce `docs/architecture/numerical-sensitivity.md` as the Phase 1 **mathematical precision-risk** analysis for the Qwen3.8-27B language + MTP forward map. Close the ledger completion criteria with **OBSERVED** config/inventory dtypes, **DERIVED** reduction lengths and IEEE field widths, and **HYPOTHESIS** risk classes. Do **not** claim that any hypothesis survives model-level validation (that open question remains for TASK-18).

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Equations, ranks, and operator definitions come from `docs/architecture/model-semantics.md` (TASK-02). Persistent-state conceptual dtypes and token-boundary survival come from `docs/architecture/lifetime-and-state.md` (TASK-04). Contraction K-dimensions may be cited from `docs/architecture/work-and-traffic.md` (TASK-06) only as the same $d_\text{in}$ already in TASK-02; do not recopy MAC/byte tables.
  - Label claims `OBSERVED` (config/inventory dtypes and constants), `DERIVED` (reduction lengths, layer counts, IEEE widths, comparisons such as $\varepsilon <$ BF16 ulp at 1), or `HYPOTHESIS` (every risk class, severity, and “quality-visible error” claim). `UNKNOWN` only for vision-encoder internals deferred here. No `MEASURED` claims.
  - GitHub Markdown math. Cite TASK-02 equation tags `(1)`–`(24)` and TASK-04 state IDs. Do not rewrite forward math, recompute TASK-04 byte totals as a traffic analysis, or redraw the TASK-03 DAG.
  - Allowed evidence: TASK-01 inventory (dtype/shape facts, not payloads), TASK-02 semantics, TASK-04 lifetime/state, sitting `config.json` `text_config`, TASK-06 contraction widths as citations, plan evidence vocabulary, and general numerical-analysis / IEEE field-width material. No Quartz, llama.cpp/GGML Qwen, or `models/Qwen3.8-27B-Q4_K_M.gguf`.
  - Config `dtype` and `mamba_ssm_dtype` name **conceptual** element types for parameters and persistent state. They are not CUDA accumulation dtypes, kernel dtypes, or an activation working-precision decision.
- Non-goals:
  - No quantization winners, bit widths, grouping, or outlier policies (TASK-08). Do not consume TASK-05 payload statistics or absmax/RMS tables.
  - No CUDA dtypes, MMA shapes, mixed-precision recipes, or kernel accumulator choices (TASK-16/17).
  - No quality/NLL experiments, reconstruction diagnostics, or “which hypotheses survive” verdicts (TASK-18).
  - No fusion, buffer reuse, physical allocation, or semantic-graph contracts (TASK-11/12).
  - No decode/prefill **schedules** (TASK-13/14) and no layouts (TASK-15).
  - No sampling / vocab-softmax as part of the forward map (logits are the output; softmax over $V$ is out of scope).
  - No new operators and no vision-encoder internals.
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only, matching TASK-01–06). Evidence scripts under `scripts/` are not a Python package; do not apply uv/`src` layout to this increment.
  - Do not edit `docs/architecture/plan.md`, `model-semantics.md`, `lifetime-and-state.md`, `work-and-traffic.md`, `model-inventory.md`, `dataflow.md`, or `bf16-tensor-analysis.md`.
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib numerical-sensitivity checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:9-17` — central path: mathematics → dataflow → lifetime → **numerical sensitivity** → precision/quantization research.
- `docs/architecture/plan.md:54-74` — evidence labels; hypotheses remain hypotheses; GitHub Markdown math.
- `docs/architecture/plan.md:85-87` — TASK-07 identifies numerical-risk hypotheses after TASK-06 work/traffic bounds.
- `docs/architecture/task_ledger.md` TASK-07 row — produces `docs/architecture/numerical-sensitivity.md`; purpose is precision risks across weights, activations, reductions, and persistent state; completion is sensitive ops/accumulation paths, precision roles, risk classified without experimental proof. Open question (validation survival) is **not** closed here.
- `docs/architecture/model-semantics.md` — equations `(1)`–`(24)`; RMS `(2)`–`(3)` with $\varepsilon=10^{-6}$; causal softmax `(9)` scale $d_h^{-1/2}=1/16$; RoPE `(11)`–`(12)` $\theta=10^7$; GDN $\alpha/\beta$ `(15)`, L2 `(16)`, recurrence `(17)`–`(18)` as the definition; `mamba_ssm_dtype: float32` applies to the $\alpha/\beta$ parameterization and to $S$ (weights remain BF16); algebraic equivalents are the same real map.
- `docs/architecture/lifetime-and-state.md` — conceptual BF16 for $K,V,C$; conceptual F32 for $S$; token-persistent IDs `K_state`, `V_state`, `C_state`, `S`; live-across `g`, `z`; layer-residual `h`, `h_mid`; $T$ after-append; example $T\in\{1,4096\}$.
- `docs/architecture/work-and-traffic.md` — contraction MAC uses $d_\text{out}d_\text{in}$; GEMM K-dimension equals TASK-02 $d_\text{in}$ (5120 hidden, 17408 MLP-down, 5120 `lm_head`); GDN definition is rank-1 `(17)`–`(18)` not dense `(19)`; do not treat `mamba_ssm_dtype` as activation dtype.
- `docs/architecture/model-inventory.md` — `text_config.dtype` `"bfloat16"`; `mamba_ssm_dtype` `"float32"`; all checkpoint tensors BF16; `max_position_embeddings` 262144; `rms_norm_eps` `1e-06`; `rope_theta` 10000000.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — live `text_config` only. Do not read safetensor payloads.
- `scripts/check_work_and_traffic.py` / `scripts/check_lifetime_and_state.py` — checker-style precedent. TASK-07’s checker is a sibling; do not import them.
- `docs/architecture/bf16-tensor-analysis.md` (TASK-05) — **not** an input. Measured weight distributions are reserved for TASK-08.

## Performance evidence

N/A — mathematical numerical-sensitivity documentation. No prefill/decode/component timing, no keep/reject, no GPU sink ranking. Reduction lengths are DERIVED arithmetic. Risk **labels** are HYPOTHESIS, not MEASURED error. Do not apply the performance-evidence checklist to rank kernels or claim quality impact.

## Implementation decisions

### Authority for numerical claims

If a reduction length, layer count, or state dtype would disagree with a TASK-02 operator, TASK-04 conceptual dtype, or sitting `text_config` key, the earlier document / config wins and this one is wrong.

- Prefill and decode share **one** sensitivity model. Only $T$ (softmax/AV/GDN horizon, RoPE phase) and whether incoming $(K,V,C,S)$ is zeros versus populated change. Do not duplicate tables per mode.
- Primary op-counts **include MTP** (one extra full-attention block and a second `lm_head` contraction). Language-only counts are secondary rows where a count differs.
- Algebraic equivalents in TASK-02 (chunkwise GDN, SDPA, RoPE complex form, $S$ vs $S^\top$) are the same **real-valued** map. Finite-precision evaluation order is **not** guaranteed identical; the primary numerical definition is the recurrent left-to-right map `(17)`–`(18)`. A chunkwise-vs-recurrent FP gap is a HYPOTHESIS note, not a second model and not an extra `sensitive_ops` row.
- Do not inspect Quartz or llama.cpp to “confirm” dtypes or accumulators.
- Do not ingest TASK-05 histograms, absmax, or RMS. Weight **role** is the OBSERVED BF16 source format plus DERIVED contraction widths.

### Deliverable structure (`docs/architecture/numerical-sensitivity.md`)

Use these **level-2 headings in this order**. Compact tables + one Mermaid fence + short captions. Every numeric instantiation is `OBSERVED` or `DERIVED`. Every risk class/severity cell is `HYPOTHESIS`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in Deferred vision.

1. **Authority** — this dossier, semantics, lifetime, inventory, config, checker; optional citation of TASK-06 for contraction $d_\text{in}$; evidence labels; in-scope (language + MTP precision roles + sensitive ops) vs deferred (vision encoder). State that the document specifies mathematical precision risks, not kernels or a mixed-precision recipe.
2. **Precision roles** — the four locked roles, both canonical hypothesis/config sentences plus the logical-≠-physical sentence (exact text below), and a role table.
3. **IEEE widths and config dtypes** — BF16/F32 field widths, ulp at 1, $\varepsilon$ comparison; config `dtype` vs `mamba_ssm_dtype`; activation working dtype is **not** decided.
4. **Weights** — `param` role; all checkpoint tensors BF16; `A_log`/`dt_bias` remain BF16 parameters used in the F32-conceptual $\alpha/\beta$ path; embed is gather (no reduction); further narrowing is TASK-08.
5. **Activations** — residual stream, live-across `g`/`z`, SiLU/sigmoid, RoPE phase; catalog IDs named, not a new catalog.
6. **Reductions and accumulation paths** — table of every `sensitive_ops` row whose role is `accum`, with DERIVED widths/horizons and equation cites; GDN primary = recurrent `(17)`; chunkwise FP gap as HYPOTHESIS prose.
7. **Persistent state** — $K,V,C$ conceptual BF16, $S$ conceptual F32; roundtrip; `s_below_f32` is the hypothesized narrowing of $S$, not a claim that F32 $S$ is itself risky.
8. **Risk classification** — complete 20-row table (all severities HYPOTHESIS); one Mermaid summary (diagram 1 of 1).
9. **Deferred vision** — residual-stream interface only.
10. **Machine-checkable summary JSON** — one fenced `json` object copied from a fresh checker `--json` run.

Immediately under **Precision roles**, include these **three canonical sentences verbatim** (checker substring match), in this order, each as its own paragraph:

> Logical values in this document are graph nodes. They do not imply physical allocation, materialization, buffer reuse, or kernel fusion.

> Risk classifications in this document are hypotheses, not measurements.

> Config dtypes name conceptual element types for parameters and persistent state; they are not CUDA accumulation or kernel dtypes.

JSON: `canonical_sentence_logical` = sentence 1; `canonical_sentence_hypothesis` = sentence 2; `canonical_sentence_dtype` = sentence 3.

Bullets required under that heading:

- Four precision roles are complete for this task: `param`, `activation`, `accum`, `state`.
- A config dtype is not an activation working-precision or GEMM-accumulator decision.
- Fan-out ≠ must-store still holds; naming a sensitive catalog ID is not a CUDA store.

### Precision roles (lock)

JSON array `precision_roles` in this order:

| Role | JSON | Meaning |
| --- | --- | --- |
| `param` | `precision_roles[0]` | Source parameter element type. OBSERVED: every checkpoint tensor is BF16, including `A_log`, `dt_bias`, and all RMSNorm $\gamma$. |
| `activation` | `precision_roles[1]` | Working/transport values of the TASK-03 catalog except token-persistent state. No OBSERVED CUDA/activation dtype; `text_config.dtype` does **not** decide this role. |
| `accum` | `precision_roles[2]` | Inner sums and normalizers of multi-term maps (GEMM K-reduction, RMS/L2, softmax/AV over $T$, GDN recurrence and $d_k$ inner products, conv taps). No config accumulator dtype. |
| `state` | `precision_roles[3]` | Token-persistent $K,V,C,S$. Conceptual dtypes from TASK-02/04: BF16 for $K,V,C$; F32 for $S$. |

Do not add a fifth role. RoPE frequencies, softmax scale, and $\alpha/\beta$ sit under `activation` / `accum` via mechanism `exp_range`. Mixed-precision **recipes** are TASK-17/08, not a role.

### Risk mechanisms (lock)

JSON array `risk_mechanisms` in this order:

| Mechanism | Meaning |
| --- | --- |
| `long_sum` | Combining $N>1$ addends (GEMM K, AV over $T$, conv taps, GDN inner $d_k$). |
| `normalize` | Divide by RMS or L2 (possibly after a long sum of squares). |
| `exp_range` | `exp` / `softplus` / `sigmoid` / `SiLU` / softmax / $\cos,\sin$ of a large phase. |
| `recurrent` | $S_t$ depends on $S_{t-1}$ with multiplicative $\alpha_t\in(0,1]$; horizon scales with $T$. |
| `roundtrip` | Write then read token-persistent state. |
| `residual_add` | $h \leftarrow h + \mathrm{Mix}$ or $h \leftarrow h + \mathrm{MLP}$. |
| `narrow_element` | Carrying a value in a 7-bit-mantissa format (BF16), or hypothesizing a store narrower than the conceptual state dtype. |

Each `sensitive_ops` row names **one** primary mechanism. Prose may mention a second mechanism; JSON has one.

### IEEE widths and config dtypes (lock)

General numerical-analysis facts (DERIVED from IEEE 754 binary32 and the de facto BF16 layout; not measured):

| Format | Sign | Exp | Trailing significand | ULP at 1 | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 1 | 8 | 7 | $2^{-7}=0.0078125$ | `bf16_mantissa_bits` 7, `bf16_exp_bits` 8, `bf16_ulp_at_1` 0.0078125 |
| F32 | 1 | 8 | 23 | $2^{-23}$ | `f32_mantissa_bits` 23, `f32_exp_bits` 8 |

JSON: `bytes_bf16` = 2, `bytes_f32` = 4 (same conceptual sizes as TASK-04). These are IEEE widths, not CUDA allocation dtypes and not GGUF Q4 sizes.

Config (OBSERVED):

| Key | Value | Applies to |
| --- | --- | --- |
| `text_config.dtype` | `"bfloat16"` | `param`; conceptual $K,V,C$ |
| `text_config.mamba_ssm_dtype` | `"float32"` | conceptual $S$ and the $\alpha/\beta$ parameterization (TASK-02); **not** activations |
| `rms_norm_eps` | `1e-06` | $\varepsilon$ in `(2)`, `(3)`, `(16)` |
| `max_position_embeddings` | 262144 | $T$ horizon ceiling for softmax/AV/RoPE/GDN hypotheses |
| `rope_theta` | 10000000 | $\theta$ in `(11)` |
| `head_dim` | 256 | softmax scale $d_h^{-1/2}=1/16=0.0625$ |

JSON booleans: `mamba_ssm_dtype_is_not_activation_dtype` true; `activation_dtype_decided` false; `eps_lt_bf16_ulp_at_1` true (`1e-6 < 0.0078125`).

Do not claim that RMS, softmax, or GEMM **run** in BF16. That would be a HYPOTHESIS about the `accum` role, and the risk table already encodes it without asserting a kernel dtype.

### Sensitive operations (lock)

JSON array `sensitive_ops` in this exact order (20 IDs). Parallel arrays `sensitive_op_roles`, `sensitive_op_mechanisms`, `sensitive_op_severities` have the same length. Every severity is HYPOTHESIS.

| ID | Role | Mechanism | Width / horizon (DERIVED) | Eqs | Severity |
| --- | --- | --- | --- | --- | --- |
| `param_bf16` | param | narrow_element | all checkpoint tensors BF16 | weights | medium |
| `residual_stream` | activation | residual_add | 128 language adds / 130 complete | (4)(5) | high |
| `live_across_gates` | activation | narrow_element | `g`: $24\times256$; `z`: $48\times128$ | (7)(10)(13)(20) | medium |
| `silu_sigmoid` | activation | exp_range | elementwise | (3)(10)(14)(21) | low |
| `rms_hidden` | accum | normalize | $N=5120$, 129 language / 134 complete maps | (2)(4)(5)(22)(23)(24) | high |
| `rms_head` | accum | normalize | 256 (QK) or 128 (GDN gated) | (3)(8) | medium |
| `l2_gdn` | accum | normalize | 128 per Q/K head | (16) | medium |
| `softmax_over_T` | accum | exp_range | $T\in\{1,4096\}$ examples; $T\le 262144$ | (9) | high |
| `attn_av_over_T` | accum | long_sum | same $T$ | (9) | high |
| `gemm_k5120` | accum | long_sum | $K=5120$ | (6)(10)(13)(20)(21-up/gate)(22)(23) | medium |
| `gemm_k17408` | accum | long_sum | $K=17408$ | (21) down | medium |
| `gemm_lm_head` | accum | long_sum | $K=5120$, $V=248320$ outputs | (22)(24) | medium |
| `gdn_S_recurrent` | accum | recurrent | $T$ steps, $48\times128\times128$ per layer | (17)(18) | high |
| `gdn_inner_d128` | accum | long_sum | 128 | (17)(18) | medium |
| `gdn_alpha_beta` | accum | exp_range | nested exp / softplus / sigmoid; 48 heads | (15) | high |
| `rope_phase` | activation | exp_range | $p\cdot\omega_0=T$ with $\omega_0=1$; $T\le 262144$ | (11)(12) | high |
| `state_kv_bf16` | state | roundtrip | 17 KV instances; RoPE baked into $K$ | state | high |
| `state_c_bf16` | state | roundtrip | 3 delay vectors $\times 10240$ | (14) | low |
| `s_below_f32` | state | narrow_element | conceptual $S$ is F32; narrowing is the risk | (17) | high |
| `conv_fir` | accum | long_sum | 4 taps | (14) | low |

Locked severity partitions (JSON arrays, this order):

- `high_risk_ids`: `residual_stream`, `rms_hidden`, `softmax_over_T`, `attn_av_over_T`, `gdn_S_recurrent`, `gdn_alpha_beta`, `rope_phase`, `state_kv_bf16`, `s_below_f32`
- `medium_risk_ids`: `param_bf16`, `live_across_gates`, `rms_head`, `l2_gdn`, `gemm_k5120`, `gemm_k17408`, `gemm_lm_head`, `gdn_inner_d128`
- `low_risk_ids`: `silu_sigmoid`, `state_c_bf16`, `conv_fir`

JSON: `n_sensitive_ops` = 20; `n_high_risk` = 9; `n_medium_risk` = 8; `n_low_risk` = 3.

Prose required (not extra IDs):

- `param_bf16` medium means the BF16 checkpoint is the study’s high-precision **source**, and further narrowing is TASK-08 — not that BF16 weights are an experimental defect.
- `s_below_f32` is the hypothesized risk of storing $S$ narrower than conceptual F32. Conceptual F32 $S$ is the TASK-02/04 reference, not a measured requirement.
- Softmax over $V$ (sampling) is out of scope; `gemm_lm_head` is the $H$-reduction into logits.
- Chunkwise GDN vs `(17)` may disagree in finite precision (HYPOTHESIS). Primary path remains `(17)`–`(18)`.

### Reduction lengths and op counts (lock)

Live from `text_config` (checker recomputes; do not hardcode without asserting against config):

| JSON key | Value | How |
| --- | ---: | --- |
| `reduction_rms_hidden` | 5120 | `hidden_size` |
| `reduction_rms_qk` | 256 | `head_dim` |
| `reduction_rms_gdn` | 128 | `linear_value_head_dim` |
| `reduction_l2_gdn` | 128 | `linear_key_head_dim` |
| `reduction_gemm_hidden` | 5120 | `hidden_size` |
| `reduction_gemm_mlp_down` | 17408 | `intermediate_size` |
| `reduction_lm_head_k` | 5120 | `hidden_size` |
| `reduction_lm_head_outputs` | 248320 | `vocab_size` (output rank, not a forward-map reduction) |
| `reduction_conv` | 4 | `linear_conv_kernel_dim` |
| `reduction_gdn_inner` | 128 | `linear_key_head_dim` |
| `reduction_attn_over_T_max` | 262144 | `max_position_embeddings` |
| `softmax_scale` | 0.0625 | $d_h^{-1/2}$ |
| `rope_theta` | 10000000 | config |
| `rope_phase_at_T_max_j0` | 262144 | $T_\max\cdot\omega_0$ with $\omega_0=1$ |
| `eps` | 1e-06 | `rms_norm_eps` |
| `n_decoder_layers` | 64 | config |
| `n_linear_layers` | 48 | `layer_types` |
| `n_full_layers` | 16 | `layer_types` |
| `n_mtp_blocks` | 1 | `mtp_num_hidden_layers` |
| `n_full_layers_with_kv` | 17 | 16+1 |
| `n_residual_adds_language` | 128 | $64\times 2$ |
| `n_residual_adds_complete` | 130 | 128+2 MTP Mix/MLP |
| `n_rms_hidden_language` | 129 | $64\times 2+1$ final |
| `n_rms_hidden_complete` | 134 | 129+5 MTP hidden RMS (`pre_fc` $\times 2$, layer in+post, `mtp.norm`) |
| `n_rms_qk` | 34 | $17\times 2$ (`q_norm`+`k_norm`) |
| `n_rms_gdn` | 48 | one GatedRMSNorm per linear layer |
| `example_T` | `[1, 4096]` | TASK-04 convention |
| `T_max` | 262144 | config |
| `T_is_stored_length_after_append` | true | TASK-04 |
| `decode_T_new` | 1 | TASK-04 |

`reduction_attn_over_T_at_example_T` = `[1, 4096]` (same as `example_T`). Attention softmax/AV length at stored $T$ includes the current token (TASK-04/06).

State element/byte citations (recompute like TASK-04; must match these integers):

| JSON key | Value |
| --- | ---: |
| `kv_bytes_all_per_token` | 69632 |
| `c_bytes_all` | 2949120 |
| `s_bytes_all` | 150994944 |
| `s_elems_per_layer` | 786432 |
| `linear_qkv_width` | 10240 |
| `linear_conv_delay` | 3 |

JSON strings: `param_dtype` `"bfloat16"`, `kv_conceptual_dtype` `"bfloat16"`, `c_conceptual_dtype` `"bfloat16"`, `s_conceptual_dtype` `"float32"`.

JSON: `gdn_primary_is_recurrent_eq_17` true; `chunkwise_fp_gap_is_hypothesis` true; `primary_includes_mtp` true.

### Diagram format (lock)

Exactly **one** fenced `mermaid` block, under heading 8 (Risk classification). Fence body starts with `flowchart TB` or `flowchart LR`. Caption sits in markdown above the fence. Do not use `sequenceDiagram`, `stateDiagram-v2`, or `%%{init:...}%%`. Do not unroll 64 layers.

Required IDs **inside that fence**: `param`, `activation`, `accum`, `state`, `S`, `KV`, `rms`, `softmax`, `gdn`.

JSON `n_diagrams` is 1. `diagram_ids` is `["param","activation","accum","state","S","KV","rms","softmax","gdn"]`.

### Deferred vision

Visual tokens may replace placeholders in the residual stream (`out_hidden_size` 5120 OBSERVED). Encoder / patch embed / merger numerical sensitivity is **UNKNOWN**. Do not add vision-sensitive ops to `sensitive_ops`. The word `UNKNOWN` may appear only in this section of the deliverable.

### Tooling

Create `scripts/check_numerical_sensitivity.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`, Google docstrings, type annotations on public functions). No torch, safetensors, numpy, mermaid parser, uv, Ruff, or pytest. Do not import other `scripts/check_*.py`; duplicate the small `text_config` arithmetic needed for shapes, layer counts, and reduction lengths.

CLI (cwd = repository root):

```text
python3 scripts/check_numerical_sensitivity.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --numerical-sensitivity docs/architecture/numerical-sensitivity.md \
  [--json]
```

Behavior:

- Read `text_config` from `--config`. Build the summary object (schema below). Live fields from config: `hidden_size`, `intermediate_size`, `vocab_size`, layer counts, head dims, linear widths, `full_attention_indices`, `dtype`, `mamba_ssm_dtype`, `rms_norm_eps`, `max_position_embeddings`, `rope_theta`. Derived: reduction lengths, op counts, softmax scale, ulp comparison, state byte citations, example-$T$ attention horizons. Constant fields: canonical sentences, IEEE bit widths, role/mechanism/op lists, severities, booleans.
- `--json`: print that object to stdout (pretty-printed, script key order); run internal asserts listed below; exit 0.
- Default / `--numerical-sensitivity PATH`: also require PATH to contain (1) every required `##` heading listed above **in order**, (2) the first fenced `json` block equal to the live object, (3) exactly one ` ```mermaid ` fence containing `flowchart`, (4) all three canonical sentences verbatim, (5) every `sensitive_ops` id present as a substring, (6) the diagram’s required IDs present **inside that mermaid fence**, (7) none of `TBD`, `TODO`, `???`, (8) no `UNKNOWN` except inside the Deferred vision section, (9) every locked document integer/decimal below present as a decimal or integer substring, (10) `precision_roles` and `risk_mechanisms` ids present as substrings, (11) the word `HYPOTHESIS` present. Exit 1 with a readable list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads. Do not require other architecture markdown JSON equality. Do not require TASK-05 numbers.

`--json` internal asserts (all required):

- `n_linear_layers==48`, `n_full_layers==16`, `n_mtp_blocks==1`, `n_full_layers_with_kv==17`
- `full_attention_indices == [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`
- `text_config.dtype == "bfloat16"` and `mamba_ssm_dtype == "float32"`
- `mamba_ssm_dtype_is_not_activation_dtype is True`, `activation_dtype_decided is False`
- `bf16_mantissa_bits==7`, `f32_mantissa_bits==23`, `bf16_ulp_at_1==0.0078125`
- `eps_lt_bf16_ulp_at_1 is True` (compare live `rms_norm_eps` to `bf16_ulp_at_1`)
- `softmax_scale==0.0625` (assert `head_dim==256` and `1/sqrt(256)==0.0625`)
- `reduction_rms_hidden==hidden_size==5120`, `reduction_gemm_mlp_down==17408`, `reduction_lm_head_outputs==248320`
- `reduction_attn_over_T_max==262144==max_position_embeddings`
- `rope_phase_at_T_max_j0==262144`
- `n_residual_adds_language==128`, `n_residual_adds_complete==130`
- `n_rms_hidden_language==129`, `n_rms_hidden_complete==134`, `n_rms_qk==34`, `n_rms_gdn==48`
- `n_sensitive_ops==20`, `n_high_risk==9`, `n_medium_risk==8`, `n_low_risk==3`
- `sensitive_ops` equals the locked 20-id list; parallel role/mechanism/severity arrays match the table; `high_risk_ids` / `medium_risk_ids` / `low_risk_ids` equal the locked partitions
- `gdn_primary_is_recurrent_eq_17 is True`, `chunkwise_fp_gap_is_hypothesis is True`, `primary_includes_mtp is True`
- `kv_bytes_all_per_token==69632`, `c_bytes_all==2949120`, `s_bytes_all==150994944` (recomputed from ranks × 2 or 4)
- `n_diagrams==1`
- `s_conceptual_dtype=="float32"`, `kv_conceptual_dtype=="bfloat16"`

Locked document integers/decimals the `--numerical-sensitivity` check must find:

`5120`, `17408`, `248320`, `256`, `128`, `262144`, `10000000`, `0.0078125`, `0.0625`, `129`, `134`, `130`, `69632`, `2949120`, `150994944`, `786432`, `10240`

(The integer `128` covers both `n_residual_adds_language` and GDN $d_k$. Do not require `1e-06` as a raw substring; $\varepsilon$ may appear as $10^{-6}$. JSON carries `eps` as `1e-06`.)

### Instantiated summary JSON schema

Top-level keys (all required; script key order locked as this list):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `n_decoder_layers`, `n_linear_layers`, `n_full_layers`, `n_mtp_blocks`, `n_full_layers_with_kv`, `full_attention_indices`, `n_attn_heads`, `n_kv_heads`, `head_dim`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_qkv_width`, `linear_conv_kernel_dim`, `linear_conv_delay`, `bytes_bf16`, `bytes_f32`, `bf16_mantissa_bits`, `bf16_exp_bits`, `f32_mantissa_bits`, `f32_exp_bits`, `bf16_ulp_at_1`, `eps`, `eps_lt_bf16_ulp_at_1`, `softmax_scale`, `rope_theta`, `T_max`, `rope_phase_at_T_max_j0`, `T_is_stored_length_after_append`, `decode_T_new`, `primary_includes_mtp`, `example_T`,

`param_dtype`, `kv_conceptual_dtype`, `c_conceptual_dtype`, `s_conceptual_dtype`, `mamba_ssm_dtype_is_not_activation_dtype`, `activation_dtype_decided`, `gdn_primary_is_recurrent_eq_17`, `chunkwise_fp_gap_is_hypothesis`,

`reduction_rms_hidden`, `reduction_rms_qk`, `reduction_rms_gdn`, `reduction_l2_gdn`, `reduction_gemm_hidden`, `reduction_gemm_mlp_down`, `reduction_lm_head_k`, `reduction_lm_head_outputs`, `reduction_conv`, `reduction_gdn_inner`, `reduction_attn_over_T_max`, `reduction_attn_over_T_at_example_T`,

`n_residual_adds_language`, `n_residual_adds_complete`, `n_rms_hidden_language`, `n_rms_hidden_complete`, `n_rms_qk`, `n_rms_gdn`,

`kv_bytes_all_per_token`, `c_bytes_all`, `s_bytes_all`, `s_elems_per_layer`,

`precision_roles` `["param","activation","accum","state"]`, `risk_mechanisms` `["long_sum","normalize","exp_range","recurrent","roundtrip","residual_add","narrow_element"]`,

`sensitive_ops` (20 ids in table order), `sensitive_op_roles`, `sensitive_op_mechanisms`, `sensitive_op_severities`, `high_risk_ids`, `medium_risk_ids`, `low_risk_ids`, `n_sensitive_ops`, `n_high_risk`, `n_medium_risk`, `n_low_risk`,

`state_ids` `["K_state","V_state","C_state","S"]`, `diagram_ids` `["param","activation","accum","state","S","KV","rms","softmax","gdn"]`, `n_diagrams` (1), `canonical_sentence_logical`, `canonical_sentence_hypothesis`, `canonical_sentence_dtype`.

Integer JSON fields that are counts/widths are JSON ints. `softmax_scale` and `bf16_ulp_at_1` are JSON numbers `0.0625` and `0.0078125`. `eps` is JSON number `1e-06`. Booleans are JSON booleans. Arrays of ints at `example_T` are JSON arrays of ints.

`sensitive_op_roles` / `sensitive_op_mechanisms` / `sensitive_op_severities` are JSON arrays of strings parallel to `sensitive_ops`.

### Stage split

- **Implementation** writes `scripts/check_numerical_sensitivity.py` **and** `docs/architecture/numerical-sensitivity.md` (roles, IEEE/config dtypes, weights/activations/reductions/state, 20-row HYPOTHESIS table, JSON fence). Runs `--json` and `--numerical-sensitivity` after the document exists. Records command outcomes in this dossier. Does not commit.
- **Documentation** performs a mechanical pass only: draft-status `unverified` banner (TASK-04 / `work-and-traffic.md` style), Authority table links to this dossier / semantics / lifetime / inventory / plan evidence policy, heading/JSON fence consistency. Must not change locked integers, canonical sentences, sensitive-op ids, severities, or Mermaid node IDs. Does not edit TASK-02/04/06 artifacts.
- **Verification** independently re-runs focused commands, recomputes reduction lengths and op counts from sitting `text_config` (not from JSON echo), reads the document against this dossier, and confirms no Quartz/llama.cpp/GGUF/TASK-05 evidence, no `plan.md` edit, and that every risk row is labelled HYPOTHESIS.

- Invariants:
  - Ten level-2 headings in the locked order; 20 sensitive-op ids; four roles; seven mechanisms; three canonical sentences verbatim.
  - Prefill/decode share one sensitivity model; $T$ after-append; primary includes MTP.
  - GDN numerical definition is recurrent `(17)`–`(18)`; chunkwise FP gap stays HYPOTHESIS.
  - Conceptual $K,V,C$ BF16 and $S$ F32 from TASK-02/04; `mamba_ssm_dtype` is not an activation dtype.
  - Logical values do not imply allocation; risks are hypotheses; config dtypes are not CUDA accumulators.
  - Vision encoder remains unexpanded.
- Rejected alternatives:
  - Consuming TASK-05 absmax/RMS to rank weight sensitivity: rejected; wrong dependency; TASK-08 owns BF16 source evidence plus these hypotheses.
  - Treating `text_config.dtype` as activation or GEMM-accumulator dtype: rejected; role `activation`/`accum` are undecided.
  - Treating `mamba_ssm_dtype` as activation dtype: rejected (TASK-02/06).
  - Claiming a mixed-precision or kernel-accumulator winner: TASK-17/08.
  - Measuring NLL / reconstruction error to “prove” a risk: TASK-18; would illegally convert HYPOTHESIS into MEASURED.
  - Closing the ledger open question “which hypotheses survive validation”: that question stays open.
  - Counting `(19)` / chunkwise GDN as a second primary numerical map: rejected; definition is `(17)`–`(18)`.
  - Including sampling softmax over $V$ as a forward-map reduction: rejected; logits are the output.
  - GGUF Q4 or CUDA dtype sizes as conceptual element sizes: rejected; BF16 inventory + conceptual F32 $S$.
  - Inspecting Quartz or llama.cpp for “real” dtypes: forbidden by plan.md.
  - uv / Ruff / pytest for this increment: rejected; stdlib checker matches TASK-01–06.
  - Editing frozen TASK-01–06 docs or `plan.md`.
  - Omitting MTP from primary op counts.
  - A fifth precision role (`scale` / `transcendental`): rejected; those maps sit under `activation`/`accum` via `exp_range`.
  - Importing other `check_*.py` as a library.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/numerical-sensitivity.md` exists and follows the heading list above.
  - Four precision roles are distinguished; seven mechanisms are named; all 20 sensitive ops and accumulation paths are tabulated with DERIVED widths/horizons and TASK-02 equation cites.
  - Risk classification lists the locked high/medium/low partitions as HYPOTHESIS and does not claim experimental proof or validation survival.
  - Three canonical sentences verbatim; one Mermaid flowchart contains the required IDs.
  - JSON fence matches a live `--json` object from config arithmetic plus locked constants.
  - No kernel/layout/fusion/allocation **decisions**; no Quartz/llama.cpp/GGUF/TASK-05 evidence; no mixed-precision winner.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_numerical_sensitivity.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_numerical_sensitivity.py
python3 scripts/check_numerical_sensitivity.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_numerical_sensitivity.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --numerical-sensitivity docs/architecture/numerical-sensitivity.md
```

- Candidate quality: not required — no model execution or NLL; this increment is numerical-sensitivity documentation.
- Repository-wide commands:

```sh
test -f docs/architecture/numerical-sensitivity.md
python3 -m py_compile scripts/check_numerical_sensitivity.py
python3 scripts/check_numerical_sensitivity.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --numerical-sensitivity docs/architecture/numerical-sensitivity.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate. Performance evidence: N/A.
- Documentation/evidence updates:
  - `docs/architecture/numerical-sensitivity.md` (create)
  - `scripts/check_numerical_sensitivity.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; TASK-02/04/06 deliverables unchanged)
- Definition of done: numerical-sensitivity document published with locked precision roles, IEEE/config dtypes, 20 sensitive ops/accumulation paths, and HYPOTHESIS risk classes; JSON fence verifies against sitting `config.json` plus locked constants; ledger TASK-07 completion checkboxes can be marked at delivery; the validation-survival open question remains open.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- UTC/time/tokens/cost: `2026-09-20T13:08:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-07.md`. Coupled IDs `none`. Document structure (10 headings), four precision roles, seven risk mechanisms, 20 sensitive ops with locked severities, IEEE/config dtype facts, stdlib checker `scripts/check_numerical_sensitivity.py`, JSON schema, and acceptance commands are closed. `docs/architecture/numerical-sensitivity.md` and the checker were **not** written in this stage.
- Performance evidence applied: N/A — derived numerical-sensitivity documentation; risk labels are hypotheses, not measured sink ranking or quality impact

### Implementation

- Agent/model: `cursor-grok-4.6-high` (this stage; parent/inherit mapping)
- Changes:
  - Created `scripts/check_numerical_sensitivity.py` (stdlib checker: `text_config` arithmetic for reduction lengths, layer/op counts, IEEE widths, state-byte citations; locked roles/mechanisms/20-op table; `--json` internal asserts; `--numerical-sensitivity` heading/JSON/mermaid/canonical/id/integer checks). Does not import other `check_*.py`.
  - Created `docs/architecture/numerical-sensitivity.md` (ten locked headings, three canonical sentences, four precision roles, seven mechanisms, 20-row HYPOTHESIS risk table, one `flowchart TB` mermaid with required IDs, JSON fence copied from a live `--json` run).
  - Did not edit `docs/architecture/plan.md`, `model-semantics.md`, `lifetime-and-state.md`, `work-and-traffic.md`, `model-inventory.md`, `dataflow.md`, or `bf16-tensor-analysis.md`. Did not commit.
- Commands:
  - `python3 -m py_compile scripts/check_numerical_sensitivity.py` — pass (exit 0)
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0); live object `n_sensitive_ops=20`, `n_high_risk=9`, `n_medium_risk=8`, `n_low_risk=3`, `softmax_scale=0.0625`, `bf16_ulp_at_1=0.0078125`, `eps_lt_bf16_ulp_at_1=true`, `kv_bytes_all_per_token=69632`, `c_bytes_all=2949120`, `s_bytes_all=150994944`, `n_rms_hidden_complete=134`, `n_residual_adds_complete=130`
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md` — pass (exit 0)
- UTC/time/tokens/cost: `2026-09-20T13:19:55Z`; `telemetry_unavailable`
- Commit: not created (implementation stage)

### Documentation

- Agent/model: `composer-2.5` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/numerical-sensitivity.md` — mechanical pass only. Added draft-status banner (`unverified`) in TASK-04 / [`lifetime-and-state.md`](../lifetime-and-state.md) / [`work-and-traffic.md`](../work-and-traffic.md) style (`> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.`). Authority table already cross-links this dossier, [`model-semantics.md`](../model-semantics.md), [`lifetime-and-state.md`](../lifetime-and-state.md), [`model-inventory.md`](../model-inventory.md), [`work-and-traffic.md`](../work-and-traffic.md) (contraction $d_\text{in}$ citation only), sitting [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../../.cache/authorities/qwen3.8-27b-transformers/config.json), [`scripts/check_numerical_sensitivity.py`](../../../scripts/check_numerical_sensitivity.py), and plan evidence policy in [`plan.md`](../plan.md). Ten required `##` headings and the first JSON fence left unchanged. No locked integers, canonical sentences, sensitive-op ids, severities, or Mermaid node IDs edited. Frozen TASK-02/04/06 artifacts and `plan.md` not edited.
- Commands:
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — pass (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md` — pass (exit 0; headings, JSON fence, one flowchart, canonical sentences, sensitive-op ids, diagram node IDs; banner did not break the check).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T13:21:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1 (first pass)
- Agent/model: `cursor-composer-2.5` (integration verifier subagent)
- Diff review:
  - Deliverables match dossier scope: `docs/architecture/numerical-sensitivity.md` (10 locked `##` headings in order; three canonical sentences verbatim; four precision roles and seven risk mechanisms; 20 sensitive-op rows with DERIVED widths/horizons and TASK-02 equation cites; complete risk table with every severity cell labelled HYPOTHESIS; one `flowchart TB` mermaid with required node IDs `param`, `activation`, `accum`, `state`, `S`, `KV`, `rms`, `softmax`, `gdn`; JSON fence equals live `--json`).
  - `scripts/check_numerical_sensitivity.py` is stdlib-only (`argparse`, `json`, `re`, `sys`, `pathlib`); no import of other `check_*.py`; no Quartz/llama.cpp/GGUF/TASK-05 payload references in deliverable or checker (only dossier-permitted negations such as “not GGUF sizes” and “No MEASURED claims”).
  - Frozen upstream docs unchanged: `plan.md`, `model-semantics.md`, `lifetime-and-state.md`, `work-and-traffic.md`, `model-inventory.md`, `dataflow.md`, `bf16-tensor-analysis.md` (no git diff).
  - Prefill/decode share one sensitivity model; primary op-counts include MTP; GDN primary is recurrent `(17)`–`(18)` with chunkwise FP gap as HYPOTHESIS prose only; conceptual $K,V,C$ BF16 and $S$ F32; `mamba_ssm_dtype` not treated as activation dtype; validation-survival open question left open (TASK-18).
  - `UNKNOWN` appears only under **Deferred vision**; no `TBD` / `TODO` / `???`.
- Independent raw-record checks (recomputed from sitting `text_config` ranks, not JSON echo):
  - Layer counts (`n_linear_layers` 48, `n_full_layers` 16, `n_mtp_blocks` 1, `n_full_layers_with_kv` 17) and reduction lengths (`hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320, `head_dim` 256, `softmax_scale` 0.0625, `eps` $<$ BF16 ulp at 1): PASS.
  - Op counts (`n_residual_adds_language` 128/130 complete, `n_rms_hidden_language` 129/134 complete, `n_rms_qk` 34, `n_rms_gdn` 48): PASS.
  - State-byte citations (`kv_bytes_all_per_token` 69632, `c_bytes_all` 2949120, `s_bytes_all` 150994944, `s_elems_per_layer` 786432, `linear_qkv_width` 10240, `linear_conv_delay` 3): PASS.
  - Locked 20-op table partitions (`n_high_risk` 9, `n_medium_risk` 8, `n_low_risk` 3) and role/mechanism arrays: PASS (checker `--json` asserts).
  - JSON fence byte-identical to fresh `--json` stdout (key order included): PASS.
- Commands:
  - `python3 -m py_compile scripts/check_numerical_sensitivity.py` — **pass** (exit 0)
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — **pass** (exit 0)
  - `python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md` — **pass** (exit 0)
  - `test -f docs/architecture/numerical-sensitivity.md` — **pass** (exit 0)
  - Repository-wide repeat (`py_compile` + `--numerical-sensitivity`) — **pass** (exit 0)
- Formatting changed files: none (verification appended this run record only)
- Verdict: **PASS** — dossier acceptance conditions and `plan.md` scope/evidence policy satisfied; all focused and repository-wide commands green; independent `text_config` recomputation matches locked integers.
- UTC/time/tokens/cost: `2026-09-20T13:22:30Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass (attempt 1) — `docs/architecture/numerical-sensitivity.md` (10 locked headings, three canonical sentences, four precision roles, seven risk mechanisms, 20 HYPOTHESIS sensitive-op rows, one `flowchart TB` mermaid, JSON fence); `scripts/check_numerical_sensitivity.py` stdlib checker; independent `text_config` recomputation from sitting config; frozen upstream docs unchanged; validation-survival open question remains for TASK-18
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: complete (mathematical precision-risk hypotheses; no tok/s evidence)
- Throughput delta: N/A — TASK-07 does not execute or time the model
- Commit: Publish Qwen3.8 numerical sensitivity
- Push: `origin/clean-sheet`
- First-pass acceptance: yes (verification attempt 1 pass; no repair loop)
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands
