# TASK-02 — Derive the complete mathematical model

## Control

- Primary ID: `TASK-02`
- Coupled IDs: `none`
- Dependencies: `TASK-01` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Specify every required operation and recurrent transition with equations; state dimensions for inputs, outputs, weights, and state; exclude kernel, graph-layout, and implementation detail.

## Goal and boundaries

Produce `docs/architecture/model-semantics.md` as the Phase 1 mathematical authority for the Qwen3.8-27B language stack: a dimensioned forward specification from token ids and persistent state to primary logits, optional MTP logits, and next state.

- Constraints:
  - `docs/architecture/plan.md` is authoritative for study scope; do not modify it.
  - Dimensions and tensor names come from `docs/architecture/model-inventory.md` and `.cache/authorities/qwen3.8-27b-transformers/config.json`. Do not re-inventory the checkpoint.
  - Label claims `OBSERVED` (config/inventory), `DERIVED` (arithmetic or algebra from those plus published architecture), or `UNKNOWN` only for items this task explicitly defers (vision-encoder internals).
  - Use GitHub Markdown math and the plan’s column-vector convention: \(W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}\), \(y=Wx\).
  - Allowed evidence: BF16 config/inventory, architectural documentation and papers (Qwen3-Next / Qwen3.5 hybrid, Gated DeltaNet Yang et al. ICLR 2025 arXiv:2412.06464, Gated Attention, Qwen2-VL mRoPE, DeepSeek-style MTP). Hugging Face Transformers `Qwen3_5*` modeling matching `transformers_version` `5.8.0.dev0` is architectural documentation for this checkpoint, not a runtime to copy.
  - Do not inspect Quartz execution/CUDA code, llama.cpp/GGML Qwen construction or kernels, or `models/Qwen3.8-27B-Q4_K_M.gguf`.
- Non-goals:
  - No kernel, fusion, graph, layout, tiling, dtype-cast schedules, or CUDA mapping (TASK-11+).
  - No logical DAG drawing (TASK-03) beyond naming producers/consumers implied by the equations.
  - No lifetime/traffic byte counts (TASK-04) beyond defining state tensors and ranks.
  - No work/FLOP bounds (TASK-06) beyond writing the operations.
  - No vision-encoder, preprocessor, chat-template, or tokenizer-merge mathematics.
  - No sampling, loss, or training-only dropout schedules (config `attention_dropout` is 0).
  - No `pyproject.toml` / uv package / Ruff / pytest suite (stdlib checker only).
- Plan impact: `none`
- Affected interfaces: none (documentation plus a stdlib section/dimension checker used only as evidence tooling)

## Repository evidence

- `docs/architecture/plan.md:65-72` — GitHub Markdown math; column-vector \(W,x,y=Wx\) example.
- `docs/architecture/plan.md:81-83` — TASK-02 writes the full forward mathematical specification before engine design.
- `docs/architecture/task_ledger.md` TASK-02 row — produces `docs/architecture/model-semantics.md`; completion is equations, dimensions, and exclusion of kernel/graph/layout detail. Open question to close: exact equations and algebraic alternatives.
- `docs/architecture/task_ledger.md` TASK-01 established results — 64 language layers, 48 linear + 16 full at indices 3,7,…,63; MTP one full-attention block; output gate via doubled `q_proj`; forward math deferred here.
- `docs/architecture/model-inventory.md` — all `text_config` keys, layer_types pattern, tensor shapes, and TASK-01 UNKNOWNs this document must resolve: output-gate application, linear-attention recurrence, `mrope_section` vs rotary dim.
- `.cache/authorities/qwen3.8-27b-transformers/config.json` — `Qwen3_5ForConditionalGeneration` / `qwen3_5`; `attn_output_gate: true`, `output_gate_type: "swish"`; `partial_rotary_factor: 0.25`; `rope_parameters.mrope_section: [11,11,10]`, `mrope_interleaved: true`, `rope_theta: 10000000`; `mamba_ssm_dtype: "float32"`; `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`.
- Published architecture (allowed by plan.md): Qwen3.5 reuses Qwen3-Next’s 3:1 Gated DeltaNet / Gated Attention decoder; Gated DeltaNet paper Eq. (10); Hugging Face Qwen3.5 docs (hybrid stack, mRoPE split of rotary frequencies, `Qwen3NextGatedDeltaNet` name).
- Shape identities from TASK-01 that the equations must consume (do not re-derive from payloads):
  - `q_proj` `(12288, 5120)` = \(2\cdot 24\cdot 256\times H\); `o_proj` `(5120, 6144)` = \(H\times 24\cdot 256\).
  - Linear `in_proj_qkv` `(10240, 5120)` = \((16+16+48)\cdot 128\times H\); `in_proj_z` `(6144, 5120)`; `in_proj_a`/`in_proj_b` `(48, 5120)`; `conv1d` `(10240, 1, 4)`; `A_log`/`dt_bias` `(48,)`; `linear_attn.norm` `(128,)`.
  - `mtp.fc` `(5120, 10240)` = \(H\times 2H\); MTP self-attn matches one language full-attention layer.

## Performance evidence

N/A — mathematical specification; no prefill/decode/component timing, no keep/reject, no sink ranking.

## Implementation decisions

### Authority for equations

Write the **inference-time** language+MTP map. Treat the following as jointly sufficient and consistent; if a secondary source disagrees, prefer checkpoint shapes + this dossier:

1. TASK-01 inventory (names, shapes, layer order).
2. `config.json` `text_config`.
3. Gated DeltaNet (Yang et al., ICLR 2025, arXiv:2412.06464) Eq. (10) for the recurrence, with Qwen3.5 parameterization of \(\alpha_t,\beta_t\), short conv, L2-normalized Q/K, and output gate.
4. Qwen3-Next / Qwen3.5 Gated Attention: QK-RMSNorm, partial RoPE, GQA, causal softmax, sigmoid channel gate from the extra `q_proj` half.
5. Qwen2-VL / Qwen3.5 interleaved mRoPE on the **frequency** axis of length \(d_\text{rot}/2\).
6. DeepSeek-style one-layer MTP: concat of RMSNorm(next-token embedding) and RMSNorm(hidden), linear mix, one full-attention decoder block, shared `lm_head`.

Do not transcribe Python, CUDA, FLA chunk kernels, or cache object layouts. Recurrent form is the definition; chunk-parallel evaluation is an algebraic equivalent (see below).

### Deliverable structure (`docs/architecture/model-semantics.md`)

Use these **level-2 headings in this order**. Compact tables + numbered equations. Every numeric instantiation is `OBSERVED` or `DERIVED`. Do not leave `TBD`. The only `UNKNOWN` allowed is vision-encoder internals, isolated in the deferred section.

1. **Authority** — this dossier, inventory, config; evidence labels; in-scope (language + MTP) vs deferred (vision encoder). State that the document specifies mathematics, not kernels.
2. **Notation** — symbol table (below) and instantiated `text_config` dimensions. Column-vector convention. Sequence index \(t=1\ldots T\); decode is the same map with \(T=1\) plus incoming state.
3. **Embedding** — untied lookup; no embedding scale; vision placeholders noted only as an interface.
4. **Normalization** — two RMSNorm roles (must not be collapsed).
5. **Residual decoder layer** — pre-norm, mixer, residual, pre-norm MLP, residual; `layer_types[i]`.
6. **Full attention (Gated Attention)** — projections, QK-norm, partial RoPE, GQA, causal softmax, sigmoid output gate, `o_proj`.
7. **Rotary embeddings (partial mRoPE)** — `mrope_section` vs rotary dim (closes TASK-01 UNKNOWN).
8. **Linear attention (Gated DeltaNet)** — projections, causal conv, \(\alpha/\beta\), L2 Q/K, recurrence, gated RMSNorm + SiLU \(z\), `out_proj`.
9. **MLP** — SwiGLU.
10. **Persistent state** — KV, conv delay, GDN matrix \(S\); dtypes from config; initial zeros.
11. **Primary logits** — final RMSNorm and untied `lm_head`.
12. **MTP** — mix, one full-attention block, shared embeddings/head, MTP KV state, token alignment.
13. **Algebraic equivalents** — same mathematics, different evaluation (not alternative models).
14. **Deferred vision** — residual-stream interface only.
15. **Instantiated dimensions** — one fenced `json` object (schema below) copied from a fresh checker run.

### Symbol table and instantiated ranks (lock these names)

| Symbol | Meaning | Instantiation (DERIVED unless noted OBSERVED) |
| --- | --- | --- |
| \(H\) | `hidden_size` | 5120 OBSERVED |
| \(I\) | `intermediate_size` | 17408 OBSERVED |
| \(V\) | `vocab_size` | 248320 OBSERVED |
| \(L\) | `num_hidden_layers` | 64 OBSERVED |
| \(n_h\) | `num_attention_heads` | 24 OBSERVED |
| \(n_\text{kv}\) | `num_key_value_heads` | 4 OBSERVED |
| \(d_h\) | `head_dim` | 256 OBSERVED |
| \(g_\text{qa}\) | GQA group size \(n_h/n_\text{kv}\) | 6 |
| \(d_\text{rot}\) | rotary dims \(d_h\cdot\) `partial_rotary_factor` | 64 |
| \(n_\omega\) | frequency count \(d_\text{rot}/2\) | 32 |
| \(\theta\) | `rope_theta` | \(10^7\) OBSERVED |
| \(\varepsilon\) | `rms_norm_eps` | \(10^{-6}\) OBSERVED |
| \(n_k^\ell,n_v^\ell\) | linear key/value heads | 16, 48 OBSERVED |
| \(d_k^\ell,d_v^\ell\) | linear key/value head dim | 128, 128 OBSERVED |
| \(r^\ell\) | \(n_v^\ell/n_k^\ell\) | 3 |
| \(d_\text{qkv}\) | conv/QKV width \((2n_k^\ell+n_v^\ell)d_k^\ell\) | 10240 |
| \(d_z\) | \(n_v^\ell d_v^\ell\) | 6144 |
| \(k_\text{conv}\) | `linear_conv_kernel_dim` | 4 OBSERVED |
| \(\mathcal{L}_\text{full}\) | full-attention indices | \(\{i: i\bmod 4=3\}=\{3,7,\ldots,63\}\) |
| \(\mathcal{L}_\text{lin}\) | linear-attention indices | \(\{0,\ldots,63\}\setminus\mathcal{L}_\text{full}\) (48 layers) |

Weight matrices use checkpoint orientation \((d_\text{out},d_\text{in})\). No language Linear/Conv bias (`attention_bias: false`; conv1d bias absent).

### Residual-stream RMSNorm (zero-centered)

Used for `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `model.language_model.norm`, and all MTP `*.norm` / `pre_fc_norm_*` (weight shape matches the normalized axis).

\[
\operatorname{RMS}(x)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon},\qquad
\operatorname{RMSNorm}_{1+\gamma}(x)=(1+\gamma)\odot \frac{x}{\operatorname{RMS}(x)}.
\]

\(\gamma\) is the stored tensor (initialized at 0 in the architecture; at inference use the checkpoint values as-is). **Do not** write Llama-style \(\gamma\odot x/\mathrm{RMS}\) for these layers.

### Gated head-wise RMSNorm (GDN output)

Used only for `linear_attn.norm.weight` \(\gamma^z\in\mathbb{R}^{128}\):

\[
\operatorname{GatedRMSNorm}(o,z;\,\gamma^z)=\bigl(\gamma^z\odot \tfrac{o}{\operatorname{RMS}(o)}\bigr)\odot \operatorname{SiLU}(z).
\]

This is multiplicative \(\gamma^z\) (architecture init 1), **not** \(1+\gamma\). `output_gate_type: "swish"` refers to this \(\operatorname{SiLU}(z)=z\,\sigma(z)\), not to full-attention gating.

### Embedding

\[
e_t=E_{:,\,\mathrm{id}_t}\in\mathbb{R}^{H},\qquad E\in\mathbb{R}^{H\times V}\ \text{(store as }(V,H)\text{)}.
\]

Untied: `lm_head` \(W_\text{lm}\in\mathbb{R}^{V\times H}\) is a distinct matrix. No \(\sqrt{H}\) embedding scale. `mtp_use_dedicated_embeddings: false` ⇒ MTP reuses \(E\) and \(W_\text{lm}\).

### Decoder layer \(\ell=0\ldots 63\)

Pre-norm residual (same for both mixer types):

\[
\tilde h^{(\ell)}=\operatorname{RMSNorm}_{1+\gamma_\text{in}^{(\ell)}}(h^{(\ell)}),\quad
h^{(\ell+1/2)}=h^{(\ell)}+\operatorname{Mix}^{(\ell)}(\tilde h^{(\ell)},\text{state}),
\]
\[
h^{(\ell+1)}=h^{(\ell+1/2)}+\operatorname{MLP}^{(\ell)}\bigl(\operatorname{RMSNorm}_{1+\gamma_\text{post}^{(\ell)}}(h^{(\ell+1/2)})\bigr).
\]

\(h^{(0)}=e\) (plus optional vision replacements; see deferred). \(\operatorname{Mix}^{(\ell)}\) is GDN if \(\ell\in\mathcal{L}_\text{lin}\), else Gated Attention.

### Full attention (closes output-gate UNKNOWN)

Let \(x\in\mathbb{R}^{H}\) be the pre-normed token (drop \(t,\ell\)).

1. **Projections** (no bias):
   \[
   u=W_q x\in\mathbb{R}^{2 n_h d_h},\quad
   k^\text{raw}=W_k x\in\mathbb{R}^{n_\text{kv}d_h},\quad
   v=W_v x\in\mathbb{R}^{n_\text{kv}d_h}.
   \]
   Reshape \(u\) to \((n_h,\,2d_h)\) and split **last axis** (per-head concat, not a global half-split):
   \[
   q'_h=u_h[0:d_h],\qquad g_h=u_h[d_h:2d_h]\in\mathbb{R}^{d_h},\quad h=1\ldots n_h.
   \]
   Instantiation: \(W_q\in\mathbb{R}^{12288\times 5120}\), \(W_k,W_v\in\mathbb{R}^{1024\times 5120}\), \(W_o\in\mathbb{R}^{5120\times 6144}\).

2. **QK-RMSNorm then RoPE** (order is required):
   \[
   q_h=\operatorname{RMSNorm}_{1+\gamma_q}(q'_h),\qquad
   k_i=\operatorname{RMSNorm}_{1+\gamma_k}(k^\text{raw}_i),\quad i=1\ldots n_\text{kv}.
   \]
   Apply partial mRoPE to the first \(d_\text{rot}=64\) coordinates of each \(q_h\) and \(k_i\); pass-through the remaining \(192\). \(\gamma_q,\gamma_k\in\mathbb{R}^{256}\).

3. **GQA + causal softmax** with scale \(d_h^{-1/2}=1/16\):
   \[
   \operatorname{Attn}(Q,K,V)_h=\sum_{s\le t}\operatorname{softmax}_s\Bigl(\frac{q_h^\top k_{\lceil h/g_\text{qa}\rceil,s}}{\sqrt{d_h}}\Bigr) v_{\lceil h/g_\text{qa}\rceil,s}.
   \]
   Inference dropout is 0. Causal mask is the only mask in the language-only spec (padding is a data-prep concern, not a new operator).

4. **Sigmoid output gate then \(W_o\)** (Gated Attention; **not** SiLU):
   \[
   y=\operatorname{Attn}(Q,K,V)\odot \sigma(g)\in\mathbb{R}^{n_h d_h},\qquad
   \operatorname{Mix}_\text{full}=W_o y\in\mathbb{R}^{H}.
   \]
   \(\sigma\) is logistic sigmoid. Flatten heads in head-major order matching the \(q\|g\) split.

MTP self-attention uses the same equations and the same doubled `q_proj`.

### Partial interleaved mRoPE (closes mrope_section UNKNOWN)

- Rotary width is \(d_\text{rot}=d_h\cdot 0.25=64\), **not** the full head. Cos/sin therefore have length 64; they are applied only to \(q,k[:, :64]\).
- Inverse frequencies are **32** real numbers (\(d_\text{rot}/2\)):
  \[
  \omega_j=\theta^{-2j/d_\text{rot}},\quad j=0\ldots 31,\quad \theta=10^{7}.
  \]
  No `inv_freq` tensor in the checkpoint; frequencies are computed.
- `mrope_section` \([11,11,10]\) **partitions those 32 frequencies**, not the 64 rotary coordinates. Sum \(11+11+10=32=n_\omega\). After `cat(freqs, freqs)` the 64-vector is the even/odd (rotate-half) layout.
- `mrope_interleaved: true` layout on frequency index \(j\in\{0,\ldots,31\}\):
  - temporal \(T\): \(j\equiv 0\pmod{3}\) and \(j\le 30\) → 11 bins;
  - height \(H\): \(j\equiv 1\pmod{3}\) (includes 31) → 11 bins;
  - width \(W\): \(j\equiv 2\pmod{3}\) and \(j\le 29\) → 10 bins.
  Equivalently: start from the \(T\) frequency row, then overwrite slice `(offset, mrope_section[axis]*3, 3)` for height (`offset=1`) and width (`offset=2`).
- Position ids \(p=(p^T,p^H,p^W)\). Text-only: \(p^T=p^H=p^W=t-1\) (0-based), which **reduces to ordinary RoPE** on the 64 rotary dims. Vision tokens use distinct \(T,H,W\) (interface only here).
- Apply rotate-half on the 64-vector:
  \[
  \operatorname{RoPE}(x,\cos,\sin)=(x_{0:32}\odot\cos-x_{32:64}\odot\sin,\; x_{32:64}\odot\cos+x_{0:32}\odot\sin)
  \]
  with \(\cos_j=\cos(p^{\text{axis}(j)}\omega_j)\) (and likewise sin), axis from the interleaved assignment. Algebraically equivalent: complex multiply of \(x_{0:32}+i x_{32:64}\) by \(e^{i\phi}\).

### Linear attention / Gated DeltaNet (closes SSM UNKNOWN)

Per linear layer, all projections bias-free.

1. **Input projections**
   \[
   \begin{aligned}
   \mathrm{qkv}&=W_\text{qkv}x\in\mathbb{R}^{10240},\\
   z&=W_z x\in\mathbb{R}^{6144},\\
   a&=W_a x\in\mathbb{R}^{48},\qquad
   b&=W_b x\in\mathbb{R}^{48}.
   \end{aligned}
   \]
   Split \(\mathrm{qkv}\) as \((q^\text{pre},k^\text{pre},v^\text{pre})\) with widths \(2048+2048+6144\). Reshape \(z\) to \((n_v^\ell,d_v^\ell)=(48,128)\). \(z\) does **not** enter the convolution.

2. **Depthwise causal conv + SiLU** on the 10240-wide QKV stream, kernel \(k_\text{conv}=4\), groups \(=10240\), no bias. With \(x^{\text{qkv}}_t=\mathrm{qkv}_t\) and \(x^{\text{qkv}}_{\le 0}=0\):
   \[
   \tilde c_{t,c}=\sum_{j=0}^{3} W^{\text{conv}}_{c,1,j}\, x^{\text{qkv}}_{t-3+j,c},\qquad
   c_t=\operatorname{SiLU}(\tilde c_t).
   \]
   Split \(c_t\) into \(q,k\in\mathbb{R}^{16\times 128}\) and \(v\in\mathbb{R}^{48\times 128}\). Repeat \(q,k\) heads by \(r^\ell=3\) (`repeat_interleave`) so both have 48 heads.

3. **Gates** (Mamba2-style \(\alpha\), sigmoid \(\beta\)):
   \[
   \beta_t=\sigma(b_t)\in(0,1)^{48},\qquad
   \alpha_t=\exp\bigl(-\,e^{A_\log}\odot \operatorname{softplus}(a_t+d_t)\bigr)\in(0,1]^{48},
   \]
   where \(A_\log,d_t\in\mathbb{R}^{48}\) are `A_log` and `dt_bias`, \(\operatorname{softplus}(u)=\log(1+e^u)\), and \(\exp/\operatorname{softplus}\) are elementwise. Config `mamba_ssm_dtype: float32` applies to this parameterization and to \(S\) (weights remain BF16).

4. **L2-normalize Q/K** (per head, \(\varepsilon=10^{-6}\)):
   \[
   \tilde q=\frac{q}{\|q\|_2+\text{stable}},\qquad
   \tilde k=\frac{k}{\|k\|_2+\text{stable}}
   \]
   (use \(x\mapsto x/\sqrt{\|x\|_2^2+\varepsilon}\)). **Do not** L2-normalize \(v\).

5. **Gated delta recurrence (definition).** For each of 48 heads, state \(S_{t-1}\in\mathbb{R}^{d_k^\ell\times d_v^\ell}=\mathbb{R}^{128\times 128}\), \(S_0=0\):
   \[
   S_t=\alpha_t S_{t-1}+\tilde k_t\otimes\bigl(\beta_t\bigl(v_t-(\alpha_t S_{t-1})^\top \tilde k_t\bigr)\bigr),
   \]
   \[
   o_t=S_t^\top \Bigl(\frac{\tilde q_t}{\sqrt{d_k^\ell}}\Bigr)\in\mathbb{R}^{128}.
   \]
   Paper form (Yang et al. Eq. 10) with \(S^\top\) stored is **the same map**:
   \[
   S^\top_t=S^\top_{t-1}\bigl(\alpha_t(I-\beta_t \tilde k_t\tilde k_t^\top)\bigr)+\beta_t v_t\tilde k_t^\top.
   \]
   Write both and state they are transposes of one matrix.

6. **Output**
   \[
   u_t=\operatorname{GatedRMSNorm}(o_t,z_t;\gamma^z)\in\mathbb{R}^{48\times 128},\qquad
   \operatorname{Mix}_\text{lin}=W_\text{out}\operatorname{vec}(u_t)\in\mathbb{R}^{H},
   \]
   \(W_\text{out}\in\mathbb{R}^{5120\times 6144}\). Flatten heads as \((48\cdot 128)\).

RoPE is **not** applied on linear layers. Local position is the causal conv.

### MLP (all 64 layers + MTP)

\[
\operatorname{MLP}(x)=W_\text{down}\bigl(\operatorname{SiLU}(W_\text{gate}x)\odot (W_\text{up}x)\bigr),
\]
\(W_\text{gate},W_\text{up}\in\mathbb{R}^{17408\times 5120}\), \(W_\text{down}\in\mathbb{R}^{5120\times 17408}\).

### Persistent state (ranks only; no byte traffic)

Define the token-to-token state that TASK-03/04 will consume. Initial state is zeros. `use_cache: true` is OBSERVED.

| State | Layers | Rank per layer (no batch) | Notes |
| --- | --- | --- | --- |
| \(K^{(\ell)},V^{(\ell)}\) | 16 language full + 1 MTP | \((n_\text{kv}, T, d_h)=(4,T,256)\) each | grows with \(T\); RoPE is baked into stored \(K\) (applied before cache) |
| \(C^{(\ell)}\) conv delay | 48 linear | last \(k_\text{conv}-1=3\) pre-activation QKV vectors, each \(\mathbb{R}^{10240}\) | equivalent length-\(4\) buffers that also reserve a current-token slot are implementation, not extra math |
| \(S^{(\ell)}\) | 48 linear | \((n_v^\ell,d_k^\ell,d_v^\ell)=(48,128,128)\) | conceptual dtype float32 from `mamba_ssm_dtype` |

Full-attention layers have no conv/\(S\) state. Linear layers have no KV cache.

Prefill of length \(T\) is the sequence map above starting from zeros. Decode of one new token is the same equations with incoming \((K,V,C,S)\).

### Primary logits

After layer 63:

\[
h^{\text{final}}=\operatorname{RMSNorm}_{1+\gamma_\text{final}}(h^{(64)}),\qquad
\ell^{(0)}_t=W_\text{lm}\,h^{\text{final}}_t\in\mathbb{R}^{V}.
\]

\(\ell^{(0)}_t\) scores the **next** token \(\mathrm{id}_{t+1}\). No bias, no tanh clip.

### MTP (one extra full-attention block)

Not required to produce \(\ell^{(0)}\). It is part of the complete model.

**Alignment:** at position \(t\), main hidden \(h^{(64)}_t\) has seen tokens \(1\ldots t\). MTP consumes the embedding of token \(t+1\) (teacher-forced or sampled from \(\ell^{(0)}\)) and predicts token \(t+2\):

\[
u_t=W_\text{fc}\begin{bmatrix}\operatorname{RMSNorm}_{1+\gamma_e}(e_{t+1})\\ \operatorname{RMSNorm}_{1+\gamma_h}(h^{(64)}_t)\end{bmatrix}\in\mathbb{R}^{H},
\]
\(W_\text{fc}\in\mathbb{R}^{5120\times 10240}\) (`mtp.fc`; concat embedding then hidden). Then one decoder layer `mtp.layers.0` with **full** Gated Attention + MLP (same equations as a language full-attention layer, including doubled `q_proj` and sigmoid gate), using mRoPE at the MTP token’s positions, with its own KV state. Then:

\[
\ell^{(1)}_t=W_\text{lm}\,\operatorname{RMSNorm}_{1+\gamma_\text{mtp}}(h^{\text{mtp}}_t)\in\mathbb{R}^{V}.
\]

Shared \(E\) and \(W_\text{lm}\). `mtp_num_hidden_layers: 1` ⇒ a single such block.

### Algebraic equivalents (include; do not pick a kernel)

State that these compute the **same** map:

- GDN recurrent step vs chunkwise/WY expansion of Eq. (10) (Yang et al. §3.3). Chunk size is not a model parameter.
- Causal conv as FIR vs a delay-line recurrence of length 3.
- GQA as repeating KV heads vs grouped matmul.
- RoPE rotate-half vs complex multiply; text mRoPE vs ordinary RoPE when \(p^T=p^H=p^W\).
- Softmax attention as explicit \(QK^\top V\) vs any exact SDPA.
- Paper \(S\in\mathbb{R}^{d_v\times d_k}\) vs stored \(S\in\mathbb{R}^{d_k\times d_v}\) (transpose).
- Optional omission of MTP when only \(\ell^{(0)}\) is required.

Forbidden as “alternatives”: fused kernels, flash-attn tiling, quantization, approximate attention.

### Deferred vision

Visual tokens enter as vectors in \(\mathbb{R}^{H}\) via `model.visual.merger` (`out_hidden_size` 5120), replacing `image_token_id` / `video_token_id` placeholders. Those positions use 3D mRoPE. Encoder, patch embed, and merger internals are **UNKNOWN** / out of scope (TASK-01 deferred). Do not invent vision equations.

### Tooling

Create `scripts/check_model_semantics.py` (Python 3.11+, stdlib only: `argparse`, `json`, `re`, `sys`, `pathlib`, Google docstring). No torch, no safetensors, no uv/Ruff/pytest.

CLI (cwd = repository root):

```text
python3 scripts/check_model_semantics.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantics docs/architecture/model-semantics.md \
  [--json]
```

Behavior:

- Read `text_config` from `--config`. Compute the instantiated table (including `d_rot=64`, `n_omega=32`, `mrope_section` sum 32, `d_qkv=10240`, `d_z=6144`, `r_ell=3`, full-attention index list, `g_qa=6`, `scale=d_h**-0.5`).
- `--json`: print the totals object (schema below) to stdout; run internal asserts; exit 0.
- Default / `--semantics PATH`: also require PATH to contain (1) every required `##` heading listed above, (2) the first fenced `json` block equal to the live object, (3) none of `TBD`, `TODO`, `???` in the file, (4) no `UNKNOWN` except inside the Deferred vision section. Exit 1 with a readable diff/list on mismatch.
- Missing config: exit 2 (blocked, not a content fail).

Do not read safetensor payloads.

### Instantiated-dimensions JSON schema

Top-level keys (all required, integers except `authority` string and `full_attention_indices` int array):

`authority` (exactly `.cache/authorities/qwen3.8-27b-transformers`), `hidden_size`, `intermediate_size`, `vocab_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `gqa_group_size`, `rotary_dim`, `n_rope_freq`, `mrope_section` (array `[11,11,10]`), `mrope_section_sum`, `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_kv_repeat`, `linear_qkv_width`, `linear_z_width`, `linear_conv_kernel_dim`, `linear_conv_delay`, `linear_state_heads`, `linear_state_dk`, `linear_state_dv`, `full_attention_interval`, `n_linear_layers`, `n_full_layers`, `full_attention_indices`, `mtp_num_hidden_layers`.

All values live from config arithmetic, not placeholders.

### Stage split

- **Implementation** writes `scripts/check_model_semantics.py` **and** `docs/architecture/model-semantics.md` (equations are the increment). Runs `--json` and `--semantics` after the document exists. Records command outcomes in this dossier.
- **Documentation** performs a mechanical pass only: links to `model-inventory.md` / `plan.md` evidence policy, heading/JSON fence consistency, draft conclusions labeled `unverified`. Must not change locked equations or gate/RoPE/GDN identities. Does not invent new operators.
- **Verification** independently re-runs focused commands, reads the document against this dossier (every operator, rank, and UNKNOWN closure), and confirms no Quartz/llama.cpp/GGUF evidence and no `plan.md` edit.

- Invariants:
  - Every mixer, norm, gate, recurrence, and logit map has an equation and ranks.
  - `output_gate_type: swish` ≠ full-attention sigmoid gate.
  - Residual-stream RMSNorm is \(1+\gamma\); GDN head RMSNorm is \(\gamma^z\odot\mathrm{RMS}\odot\mathrm{SiLU}(z)\).
  - `mrope_section` sums to \(d_\text{rot}/2\), not \(d_\text{rot}\).
  - No kernel/graph/layout text.
  - Vision encoder remains unexpanded.
- Rejected alternatives:
  - Sigmoid full-attention gate replaced by SiLU because `output_gate_type` is `swish`: rejected; that flag is the GDN \(z\)-gate.
  - RoPE on all 256 head dims: rejected; `partial_rotary_factor` 0.25.
  - Treating `mrope_section` as a split of 64 rotary coords: rejected; it splits 32 frequencies.
  - Fused `in_proj_qkvz` / `in_proj_ba` as in some Qwen3-Next writeups: rejected; this checkpoint has four separate projections.
  - Paper-only \(S\in\mathbb{R}^{d_v\times d_k}\) without the stored \((48,128,128)\) shape: rejected as incomplete for TASK-04.
  - Deferring MTP: rejected; ledger and inventory treat MTP as language-side.
  - Copying FLA/chunk Python as the spec: rejected; recurrence is the definition.
  - Pytest/Ruff/uv for this increment: rejected; stdlib checker suffices.
  - Inspecting Quartz or llama.cpp to “confirm” equations: forbidden by plan.md.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions:
  - `docs/architecture/model-semantics.md` exists and follows the heading list above.
  - Every required operation and recurrent transition is written as equations (embedding, both RMSNorms, residual layer, Gated Attention including sigmoid gate, partial interleaved mRoPE, GDN including conv/\(\alpha,\beta\)/L2/recurrence/gated RMSNorm, SwiGLU MLP, primary logits, MTP mix+block+logits).
  - Dimensions are stated for inputs, outputs, weights, and state, instantiated to this model.
  - TASK-01 UNKNOWNs are closed: output-gate rule, GDN recurrence, `mrope_section` vs rotary dim.
  - Kernel, graph-layout, and implementation detail are absent.
  - JSON fence matches a live `--json` object from config arithmetic.
  - Coupled IDs: none.
- Tests/fixtures to add or change: `scripts/check_model_semantics.py` only (no pytest fixtures).
- Focused commands (repository root; config must exist):

```sh
python3 -m py_compile scripts/check_model_semantics.py
python3 scripts/check_model_semantics.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --json
python3 scripts/check_model_semantics.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantics docs/architecture/model-semantics.md
```

- Candidate quality: not required — no model execution or NLL.
- Repository-wide commands:

```sh
test -f docs/architecture/model-semantics.md
python3 -m py_compile scripts/check_model_semantics.py
python3 scripts/check_model_semantics.py \
  --config .cache/authorities/qwen3.8-27b-transformers/config.json \
  --semantics docs/architecture/model-semantics.md
```

Do not run Ruff, pytest, CMake, or CUDA; this increment does not introduce those gates.

- Native/CUDA/hardware gates: not applicable (no kernels, no GPU work). Config presence is a local-file requirement, not a GPU gate.
- Documentation/evidence updates:
  - `docs/architecture/model-semantics.md` (create)
  - `scripts/check_model_semantics.py` (create)
  - this dossier run records
  - ledger status/history at delivery only (`plan.md` unchanged; `model-inventory.md` unchanged)
- Definition of done: semantics document published with locked equations and ranks, JSON fence verifies against sitting `config.json`, ledger TASK-02 completion checkboxes can be marked at delivery.

## Run record

### Planning

- Agent/model: `cursor-grok-4.6-high`
- UTC/time/tokens/cost: `2026-09-20T09:18:00Z`; `telemetry_unavailable`
- Outcome: Decision-complete dossier created at `docs/architecture/tasks/TASK-02.md`. Coupled IDs none. Document structure, notation, state variables, gate/RoPE/GDN/MTP identities, stdlib checker, and acceptance commands are closed. `model-semantics.md` not written in this stage.
- Performance evidence applied: N/A (mathematical specification; no timing)

### Implementation

- Agent/model: `cursor-grok-4.6-high`
- Changes:
  - created `docs/architecture/model-semantics.md` (15 required `##` headings, locked equations and ranks, instantiated-dimensions JSON fence)
  - created `scripts/check_model_semantics.py` (stdlib checker: config arithmetic, `--json`, `--semantics`)
- Commands:
  - `python3 -m py_compile scripts/check_model_semantics.py` → exit 0
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` → exit 0; printed the schema object (`hidden_size` 5120, `rotary_dim` 64, `n_rope_freq` 32, `mrope_section` `[11,11,10]`, `linear_qkv_width` 10240, `n_full_layers` 16, `full_attention_indices` `[3,7,…,63]`, `mtp_num_hidden_layers` 1)
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md` → exit 0 (headings, JSON fence, no TBD/TODO/`???`, UNKNOWN only in Deferred vision)
- UTC/time/tokens/cost: `2026-09-20T09:28:17Z`; `telemetry_unavailable`

### Documentation

- Agent/model: `cursor-composer` (documentation subagent)
- Changes and evidence:
  - `docs/architecture/model-semantics.md` — added draft-status banner (`unverified`); Authority table cross-links to dossier, inventory, config, checker script, and plan evidence policy; TASK-01 UNKNOWN closures link back to inventory State implications (labeled unverified); Instantiated dimensions links to checker script.
  - `docs/architecture/model-inventory.md` — State implications cross-links to `model-semantics.md` sections for forward math, mRoPE, GDN recurrence, and output gate (TASK-02, unverified); no equation or rank edits.
- Commands:
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — **pass** (exit 0; JSON fence source unchanged).
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md` — **pass** (exit 0; 15 headings, JSON fence match, no forbidden tokens, UNKNOWN only in Deferred vision).
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: `2026-09-20T09:30:00Z`; `telemetry_unavailable`

### Verification

- Attempt: 1
- Agent/model: `cursor-composer` (integration verifier subagent)
- Diff review:
  - `docs/architecture/model-semantics.md` — 15 required `##` headings in dossier order; numbered equations for embedding, both RMSNorm roles (\(1+\gamma\) residual vs \(\gamma^z\) GDN), residual decoder layer, Gated Attention (per-head \(q\|g\) split, QK-norm then RoPE, GQA softmax, **sigmoid** output gate), partial interleaved mRoPE (\(d_\text{rot}=64\), \(n_\omega=32\), `mrope_section` partitions frequencies), GDN (four projections, causal conv, \(\alpha/\beta\), L2 Q/K, recurrence + transpose form, GatedRMSNorm+SiLU), SwiGLU MLP, persistent state ranks, primary logits, MTP mix+block+logits, algebraic equivalents, deferred vision interface only.
  - `scripts/check_model_semantics.py` — stdlib-only (`argparse`, `json`, `re`, `sys`, `pathlib`); `--json` arithmetic and `--semantics` heading/JSON-fence/forbidden-token checks match dossier.
  - TASK-01 UNKNOWN closures confirmed: full-attention output gate = sigmoid on extra `q_proj` half (`output_gate_type: swish` names GDN \(z\)-gate only); GDN recurrence Eq. (10) with stored \((48,128,128)\) shape; `mrope_section` sums to \(d_\text{rot}/2\), not \(d_\text{rot}\).
  - No kernel/graph-layout/CUDA/fusion/tiling implementation detail in deliverables; `conv` kernel width is mathematical FIR only.
  - `docs/architecture/plan.md` — unchanged.
  - `docs/architecture/task_ledger.md` — status `TODO` → `IN PROGRESS` only (expected pre-delivery).
  - No Quartz/llama.cpp/GGUF inspection artifacts.
- Independent raw-record checks:
  - First fenced `json` block in `model-semantics.md` byte-matches live `--json` output (`hidden_size` 5120, `rotary_dim` 64, `mrope_section` `[11,11,10]`, `linear_qkv_width` 10240, `n_full_layers` 16, `full_attention_indices` `[3,7,…,63]`, `mtp_num_hidden_layers` 1).
  - `UNKNOWN` appears only in Deferred vision section; no `TBD`, `TODO`, or `???` in semantics file.
- Commands:
  - `python3 -m py_compile scripts/check_model_semantics.py` — **pass** (exit 0)
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --json` — **pass** (exit 0)
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md` — **pass** (exit 0; headings, JSON fence, forbidden tokens)
  - `test -f docs/architecture/model-semantics.md` — **pass** (exit 0)
  - `python3 -m py_compile scripts/check_model_semantics.py` (repository-wide duplicate) — **pass** (exit 0)
  - `python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md` (repository-wide duplicate) — **pass** (exit 0)
- Formatting changed files: none
- Verdict: **pass** — deliverables satisfy dossier acceptance and ledger completion criteria; equations and ranks cover the full language+MTP forward map; JSON fence verifies against sitting `config.json`.
- UTC/time/tokens/cost: `2026-09-20T09:28:53Z`; `telemetry_unavailable`

### Retries and escalation

none

### Final outcome

- Status: `DONE`
- Acceptance evidence: verification pass — `model-semantics.md` and JSON fence reconcile to sitting `config.json`; checker passes focused commands
- Candidate measured delta: N/A (no throughput work)
- Shipping delta: N/A (diagnostics/documentation)
- Quality result: not required
- Evidence completeness: N/A (no performance-evidence checks)
- Throughput delta: N/A — TASK-02 does not execute or time the model
- Commit: Establish Qwen3.8 forward math specification
- Push: `origin/clean-sheet` (success)
- First-pass acceptance: **verified**
- Total elapsed/tokens/cost: `telemetry_unavailable`
- Remaining risk or recovery condition: local `.cache/` config must remain present for focused commands
