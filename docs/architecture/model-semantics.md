# Qwen3.8-27B model semantics (TASK-02)

> **Draft status:** conclusions in this document are **unverified** until the verification stage completes.

Inference-time language + MTP forward map for the Qwen3.8-27B stack: token ids
and persistent state to primary logits, optional MTP logits, and next state.
This is the Phase 1 mathematical authority. Dimensions instantiate the sitting
checkpoint; they are not re-inventoried here.

## Authority

| Item | Value | Label |
| --- | --- | --- |
| Dossier | [`docs/architecture/tasks/TASK-02.md`](tasks/TASK-02.md) | — |
| Inventory | [`docs/architecture/model-inventory.md`](model-inventory.md) (TASK-01) | OBSERVED / DERIVED |
| Config | [`.cache/authorities/qwen3.8-27b-transformers/config.json`](../../.cache/authorities/qwen3.8-27b-transformers/config.json) `text_config` | OBSERVED |
| Checker | [`scripts/check_model_semantics.py`](../../scripts/check_model_semantics.py) | DERIVED |
| Evidence policy | [`docs/architecture/plan.md`](plan.md) (OBSERVED / DERIVED labels) | OBSERVED |
| In scope | Language stack (64 decoder layers) and one MTP full-attention block | — |
| Deferred | Vision encoder internals (residual-stream interface only) | — |
| Scope of this document | Mathematics of the forward map, not kernels | — |

Column-vector convention from the plan: \(W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}\), \(y=Wx\). Weight matrices are stored in checkpoint orientation \((d_\text{out},d_\text{in})\). No language Linear or Conv bias (`attention_bias: false` OBSERVED; conv1d bias absent OBSERVED).

Secondary sources used only as architecture (not as a runtime to copy): Qwen3-Next / Qwen3.5 hybrid (3:1 Gated DeltaNet / Gated Attention), Gated DeltaNet (Yang et al., ICLR 2025, arXiv:2412.06464) Eq. (10), Qwen2-VL / Qwen3.5 interleaved mRoPE, DeepSeek-style one-layer MTP. Hugging Face Transformers `Qwen3_5*` matching `transformers_version` `5.8.0.dev0` is architectural documentation for this checkpoint.

If a secondary source disagrees, checkpoint shapes + this document win.

## Notation

Sequence index \(t=1,\ldots,T\). Prefill is the map below from zero state. Decode is the same map with \(T=1\) plus incoming persistent state.

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

Elementwise helpers used below:

$$
\sigma(u)=\frac{1}{1+e^{-u}},\qquad
\operatorname{SiLU}(z)=z\,\sigma(z),\qquad
\operatorname{softplus}(u)=\log(1+e^u).
$$

## Embedding

Untied lookup; no embedding scale. Store \(E\) as \((V,H)=(248320,5120)\) OBSERVED.

$$
e_t=E_{:,\,\mathrm{id}_t}\in\mathbb{R}^{H},\qquad E\in\mathbb{R}^{H\times V}.
\tag{1}
$$

`lm_head` \(W_\text{lm}\in\mathbb{R}^{V\times H}\) is a distinct matrix (`tie_word_embeddings: false` OBSERVED). No \(\sqrt{H}\) scale. `mtp_use_dedicated_embeddings: false` ⇒ MTP reuses \(E\) and \(W_\text{lm}\) OBSERVED.

Vision placeholders (`image_token_id`, `video_token_id`) may be replaced by vectors in \(\mathbb{R}^{H}\) before the first decoder layer; that replacement is an interface, not an embedding equation (see Deferred vision).

## Normalization

Two RMSNorm roles. They must not be collapsed.

**Residual-stream RMSNorm** (zero-centered), used for `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `model.language_model.norm`, and all MTP `*.norm` / `pre_fc_norm_*`. Weight shape matches the normalized axis. \(\gamma\) is the stored tensor (architecture init 0; at inference use checkpoint values as-is). This is **not** Llama-style \(\gamma\odot x/\mathrm{RMS}\).

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon},\qquad
\operatorname{RMSNorm}_{1+\gamma}(x)=(1+\gamma)\odot \frac{x}{\operatorname{RMS}(x)}.
\tag{2}
$$

**Gated head-wise RMSNorm**, used only for `linear_attn.norm.weight` \(\gamma^z\in\mathbb{R}^{128}\) OBSERVED. Multiplicative \(\gamma^z\) (architecture init 1), **not** \(1+\gamma\). Config `output_gate_type: "swish"` OBSERVED refers to this \(\operatorname{SiLU}(z)\), not to full-attention gating.

$$
\operatorname{GatedRMSNorm}(o,z;\,\gamma^z)=\bigl(\gamma^z\odot \tfrac{o}{\operatorname{RMS}(o)}\bigr)\odot \operatorname{SiLU}(z).
\tag{3}
$$

## Residual decoder layer

Pre-norm residual, identical wrapper for both mixer types. Layer \(\ell=0,\ldots,63\); mixer selected by `layer_types[\(\ell\)]` OBSERVED: Gated DeltaNet if \(\ell\in\mathcal{L}_\text{lin}\), Gated Attention if \(\ell\in\mathcal{L}_\text{full}\).

$$
\tilde h^{(\ell)}=\operatorname{RMSNorm}_{1+\gamma_\text{in}^{(\ell)}}(h^{(\ell)}),\quad
h^{(\ell+1/2)}=h^{(\ell)}+\operatorname{Mix}^{(\ell)}(\tilde h^{(\ell)},\text{state}),
\tag{4}
$$

$$
h^{(\ell+1)}=h^{(\ell+1/2)}+\operatorname{MLP}^{(\ell)}\bigl(\operatorname{RMSNorm}_{1+\gamma_\text{post}^{(\ell)}}(h^{(\ell+1/2)})\bigr).
\tag{5}
$$

\(h^{(0)}=e\) (plus optional vision replacements). \(\gamma_\text{in}^{(\ell)},\gamma_\text{post}^{(\ell)}\in\mathbb{R}^{H}\). Inputs and outputs of each layer are \(\mathbb{R}^{H}\) per token.

## Full attention (Gated Attention)

Let \(x\in\mathbb{R}^{H}\) be the pre-normed token (drop \(t,\ell\)). Language layers \(\ell\in\mathcal{L}_\text{full}\) and MTP self-attention use the same equations, including doubled `q_proj` (closes TASK-01 [output-gate open question](model-inventory.md#state-implications); unverified).

**Projections** (no bias). Instantiation OBSERVED / DERIVED: \(W_q\in\mathbb{R}^{12288\times 5120}=2\cdot 24\cdot 256\times H\), \(W_k,W_v\in\mathbb{R}^{1024\times 5120}=n_\text{kv}d_h\times H\), \(W_o\in\mathbb{R}^{5120\times 6144}=H\times n_h d_h\).

$$
u=W_q x\in\mathbb{R}^{2 n_h d_h},\quad
k^\text{raw}=W_k x\in\mathbb{R}^{n_\text{kv}d_h},\quad
v=W_v x\in\mathbb{R}^{n_\text{kv}d_h}.
\tag{6}
$$

Reshape \(u\) to \((n_h,\,2d_h)=(24,512)\) and split the **last axis** (per-head concat, not a global half-split):

$$
q'_h=u_h[0:d_h],\qquad g_h=u_h[d_h:2d_h]\in\mathbb{R}^{d_h},\quad h=1,\ldots,n_h.
\tag{7}
$$

**QK-RMSNorm then RoPE** (this order is required). \(\gamma_q,\gamma_k\in\mathbb{R}^{256}\) OBSERVED.

$$
q_h=\operatorname{RMSNorm}_{1+\gamma_q}(q'_h),\qquad
k_i=\operatorname{RMSNorm}_{1+\gamma_k}(k^\text{raw}_i),\quad i=1,\ldots,n_\text{kv}.
\tag{8}
$$

Apply partial mRoPE to the first \(d_\text{rot}=64\) coordinates of each \(q_h\) and \(k_i\); pass-through the remaining \(192\).

**GQA + causal softmax** with scale \(d_h^{-1/2}=1/16\) DERIVED. Inference dropout is 0 (`attention_dropout: 0.0` OBSERVED). Causal mask is the only mask in the language-only spec.

$$
\operatorname{Attn}(Q,K,V)_h=\sum_{s\le t}\operatorname{softmax}_s\Bigl(\frac{q_h^\top k_{\lceil h/g_\text{qa}\rceil,s}}{\sqrt{d_h}}\Bigr) v_{\lceil h/g_\text{qa}\rceil,s}.
\tag{9}
$$

**Sigmoid output gate then \(W_o\)** (Gated Attention; **not** SiLU). Flatten heads in head-major order matching the \(q\|g\) split.

$$
y=\operatorname{Attn}(Q,K,V)\odot \sigma(g)\in\mathbb{R}^{n_h d_h},\qquad
\operatorname{Mix}_\text{full}=W_o y\in\mathbb{R}^{H}.
\tag{10}
$$

\(\sigma\) is logistic sigmoid. Config `attn_output_gate: true` OBSERVED is this channel gate from the extra `q_proj` half. Config `output_gate_type: "swish"` is **not** this gate (it names the GDN \(z\)-SiLU).

## Rotary embeddings (partial mRoPE)

Closes the TASK-01 [`mrope_section` open question](model-inventory.md#state-implications) (unverified).

- Rotary width is \(d_\text{rot}=d_h\cdot 0.25=64\) DERIVED, **not** the full head. Cos/sin therefore have length 64; they are applied only to \(q,k[:, :64]\).
- Inverse frequencies are **32** real numbers (\(d_\text{rot}/2\)). No `inv_freq` tensor in the checkpoint OBSERVED; frequencies are computed:

$$
\omega_j=\theta^{-2j/d_\text{rot}},\quad j=0,\ldots,31,\quad \theta=10^{7}.
\tag{11}
$$

- `mrope_section` \([11,11,10]\) OBSERVED **partitions those 32 frequencies**, not the 64 rotary coordinates. Sum \(11+11+10=32=n_\omega\) DERIVED. After `cat(freqs, freqs)` the 64-vector is the even/odd (rotate-half) layout.
- `mrope_interleaved: true` OBSERVED layout on frequency index \(j\in\{0,\ldots,31\}\):
  - temporal \(T\): \(j\equiv 0\pmod{3}\) and \(j\le 30\) → 11 bins;
  - height \(H\): \(j\equiv 1\pmod{3}\) (includes 31) → 11 bins;
  - width \(W\): \(j\equiv 2\pmod{3}\) and \(j\le 29\) → 10 bins.

  Equivalently: start from the \(T\) frequency row, then overwrite slice `(offset, mrope_section[axis]*3, 3)` for height (`offset=1`) and width (`offset=2`).
- Position ids \(p=(p^T,p^H,p^W)\). Text-only: \(p^T=p^H=p^W=t-1\) (0-based), which **reduces to ordinary RoPE** on the 64 rotary dims. Vision tokens use distinct \(T,H,W\) (interface only here).

Apply rotate-half on the 64-vector:

$$
\operatorname{RoPE}(x,\cos,\sin)=(x_{0:32}\odot\cos-x_{32:64}\odot\sin,\; x_{32:64}\odot\cos+x_{0:32}\odot\sin)
\tag{12}
$$

with \(\cos_j=\cos(p^{\text{axis}(j)}\omega_j)\) (and likewise sin), axis from the interleaved assignment. Algebraically equivalent: complex multiply of \(x_{0:32}+i x_{32:64}\) by \(e^{i\phi}\).

## Linear attention (Gated DeltaNet)

Closes the TASK-01 [linear-attention recurrence open question](model-inventory.md#state-implications) (unverified). Per linear layer \(\ell\in\mathcal{L}_\text{lin}\), all projections bias-free. RoPE is **not** applied; local position is the causal conv.

**Input projections.** Instantiation OBSERVED: \(W_\text{qkv}\in\mathbb{R}^{10240\times 5120}\), \(W_z\in\mathbb{R}^{6144\times 5120}\), \(W_a,W_b\in\mathbb{R}^{48\times 5120}\).

$$
\begin{aligned}
\mathrm{qkv}&=W_\text{qkv}x\in\mathbb{R}^{10240},\\
z&=W_z x\in\mathbb{R}^{6144},\\
a&=W_a x\in\mathbb{R}^{48},\qquad
b&=W_b x\in\mathbb{R}^{48}.
\end{aligned}
\tag{13}
$$

Split \(\mathrm{qkv}\) as \((q^\text{pre},k^\text{pre},v^\text{pre})\) with widths \(2048+2048+6144\). Reshape \(z\) to \((n_v^\ell,d_v^\ell)=(48,128)\). \(z\) does **not** enter the convolution.

**Depthwise causal conv + SiLU** on the 10240-wide QKV stream, kernel \(k_\text{conv}=4\), groups \(=10240\), no bias. Weight \(W^{\text{conv}}\in\mathbb{R}^{10240\times 1\times 4}\) OBSERVED. With \(x^{\text{qkv}}_t=\mathrm{qkv}_t\) and \(x^{\text{qkv}}_{\le 0}=0\):

$$
\tilde c_{t,c}=\sum_{j=0}^{3} W^{\text{conv}}_{c,1,j}\, x^{\text{qkv}}_{t-3+j,c},\qquad
c_t=\operatorname{SiLU}(\tilde c_t).
\tag{14}
$$

Split \(c_t\) into \(q,k\in\mathbb{R}^{16\times 128}\) and \(v\in\mathbb{R}^{48\times 128}\). Repeat \(q,k\) heads by \(r^\ell=3\) so both have 48 heads.

**Gates** (Mamba2-style \(\alpha\), sigmoid \(\beta\)). \(A_\log,d_t\in\mathbb{R}^{48}\) are `A_log` and `dt_bias` OBSERVED. Config `mamba_ssm_dtype: float32` OBSERVED applies to this parameterization and to \(S\) (weights remain BF16).

$$
\beta_t=\sigma(b_t)\in(0,1)^{48},\qquad
\alpha_t=\exp\bigl(-\,e^{A_\log}\odot \operatorname{softplus}(a_t+d_t)\bigr)\in(0,1]^{48}.
\tag{15}
$$

**L2-normalize Q/K** (per head, \(\varepsilon=10^{-6}\)). **Do not** L2-normalize \(v\).

$$
\tilde q=\frac{q}{\sqrt{\|q\|_2^2+\varepsilon}},\qquad
\tilde k=\frac{k}{\sqrt{\|k\|_2^2+\varepsilon}}.
\tag{16}
$$

**Gated delta recurrence (definition).** For each of 48 heads, state \(S_{t-1}\in\mathbb{R}^{d_k^\ell\times d_v^\ell}=\mathbb{R}^{128\times 128}\), \(S_0=0\):

$$
S_t=\alpha_t S_{t-1}+\tilde k_t\otimes\bigl(\beta_t\bigl(v_t-(\alpha_t S_{t-1})^\top \tilde k_t\bigr)\bigr),
\tag{17}
$$

$$
o_t=S_t^\top \Bigl(\frac{\tilde q_t}{\sqrt{d_k^\ell}}\Bigr)\in\mathbb{R}^{128}.
\tag{18}
$$

Paper form (Yang et al. Eq. 10) with \(S^\top\) stored is **the same map** (transposes of one matrix):

$$
S^\top_t=S^\top_{t-1}\bigl(\alpha_t(I-\beta_t \tilde k_t\tilde k_t^\top)\bigr)+\beta_t v_t\tilde k_t^\top.
\tag{19}
$$

**Output.** \(W_\text{out}\in\mathbb{R}^{5120\times 6144}\) OBSERVED. Flatten heads as \((48\cdot 128)\).

$$
u_t=\operatorname{GatedRMSNorm}(o_t,z_t;\gamma^z)\in\mathbb{R}^{48\times 128},\qquad
\operatorname{Mix}_\text{lin}=W_\text{out}\operatorname{vec}(u_t)\in\mathbb{R}^{H}.
\tag{20}
$$

## MLP

All 64 language layers and the MTP block. Instantiation OBSERVED: \(W_\text{gate},W_\text{up}\in\mathbb{R}^{17408\times 5120}\), \(W_\text{down}\in\mathbb{R}^{5120\times 17408}\). `hidden_act: "silu"` OBSERVED.

$$
\operatorname{MLP}(x)=W_\text{down}\bigl(\operatorname{SiLU}(W_\text{gate}x)\odot (W_\text{up}x)\bigr)\in\mathbb{R}^{H}.
\tag{21}
$$

## Persistent state

Ranks only; no byte traffic. Initial state is zeros. `use_cache: true` OBSERVED.

| State | Layers | Rank per layer (no batch) | Notes |
| --- | --- | --- | --- |
| \(K^{(\ell)},V^{(\ell)}\) | 16 language full + 1 MTP | \((n_\text{kv}, T, d_h)=(4,T,256)\) each | grows with \(T\); RoPE is baked into stored \(K\) (applied before cache) |
| \(C^{(\ell)}\) conv delay | 48 linear | last \(k_\text{conv}-1=3\) pre-activation QKV vectors, each \(\mathbb{R}^{10240}\) | length-4 buffers that also reserve a current-token slot are not extra math |
| \(S^{(\ell)}\) | 48 linear | \((n_v^\ell,d_k^\ell,d_v^\ell)=(48,128,128)\) | conceptual dtype float32 from `mamba_ssm_dtype` |

Full-attention layers have no conv/\(S\) state. Linear layers have no KV cache.

Prefill of length \(T\) is the sequence map above starting from zeros. Decode of one new token is the same equations with incoming \((K,V,C,S)\).

## Primary logits

After layer 63. \(\gamma_\text{final}\in\mathbb{R}^{H}\) OBSERVED (`model.language_model.norm`). \(W_\text{lm}\in\mathbb{R}^{V\times H}\) OBSERVED. No bias, no tanh clip.

$$
h^{\text{final}}=\operatorname{RMSNorm}_{1+\gamma_\text{final}}(h^{(64)}),\qquad
\ell^{(0)}_t=W_\text{lm}\,h^{\text{final}}_t\in\mathbb{R}^{V}.
\tag{22}
$$

\(\ell^{(0)}_t\) scores the **next** token \(\mathrm{id}_{t+1}\).

## MTP

Not required to produce \(\ell^{(0)}\). It is part of the complete model. `mtp_num_hidden_layers: 1` OBSERVED ⇒ a single block `mtp.layers.0`.

**Alignment:** at position \(t\), main hidden \(h^{(64)}_t\) has seen tokens \(1\ldots t\). MTP consumes the embedding of token \(t+1\) (teacher-forced or sampled from \(\ell^{(0)}\)) and predicts token \(t+2\).

\(W_\text{fc}\in\mathbb{R}^{5120\times 10240}\) OBSERVED (`mtp.fc`; concat embedding then hidden). \(\gamma_e,\gamma_h\in\mathbb{R}^{H}\) are `mtp.pre_fc_norm_embedding` and `mtp.pre_fc_norm_hidden`.

$$
u_t=W_\text{fc}\begin{bmatrix}\operatorname{RMSNorm}_{1+\gamma_e}(e_{t+1})\\ \operatorname{RMSNorm}_{1+\gamma_h}(h^{(64)}_t)\end{bmatrix}\in\mathbb{R}^{H}.
\tag{23}
$$

Then one decoder layer `mtp.layers.0` with **full** Gated Attention + MLP (equations (4)–(10) and (21), including doubled `q_proj` and sigmoid gate), using mRoPE at the MTP token’s positions, with its own KV state. Then \(\gamma_\text{mtp}\in\mathbb{R}^{H}\) (`mtp.norm`):

$$
\ell^{(1)}_t=W_\text{lm}\,\operatorname{RMSNorm}_{1+\gamma_\text{mtp}}(h^{\text{mtp}}_t)\in\mathbb{R}^{V}.
\tag{24}
$$

Shared \(E\) and \(W_\text{lm}\).

## Algebraic equivalents

These compute the **same** map (not alternative models):

- GDN recurrent step (17)–(19) vs chunkwise/WY expansion of Yang et al. Eq. (10) §3.3. Chunk size is not a model parameter.
- Causal conv (14) as an FIR vs a delay-line recurrence of length 3.
- GQA as repeating KV heads vs grouped matmul.
- RoPE rotate-half (12) vs complex multiply; text mRoPE vs ordinary RoPE when \(p^T=p^H=p^W\).
- Softmax attention (9) as explicit \(QK^\top V\) vs any exact SDPA.
- Paper \(S\in\mathbb{R}^{d_v\times d_k}\) vs stored \(S\in\mathbb{R}^{d_k\times d_v}\) (transpose).
- Optional omission of MTP when only \(\ell^{(0)}\) is required.

Approximate attention, quantization, and fused evaluation schedules are not equivalent maps and are out of scope.

## Deferred vision

Visual tokens enter as vectors in \(\mathbb{R}^{H}\) via `model.visual.merger` (`out_hidden_size` 5120 OBSERVED), replacing `image_token_id` / `video_token_id` placeholders. Those positions use 3D mRoPE. Encoder, patch embed, and merger internals are **UNKNOWN** / out of scope (TASK-01 deferred). This document does not expand vision-encoder equations.

## Instantiated dimensions

Live object from `text_config` arithmetic (copied from [`scripts/check_model_semantics.py --json`](../../scripts/check_model_semantics.py)):

```json
{
  "authority": ".cache/authorities/qwen3.8-27b-transformers",
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "vocab_size": 248320,
  "num_hidden_layers": 64,
  "num_attention_heads": 24,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "gqa_group_size": 6,
  "rotary_dim": 64,
  "n_rope_freq": 32,
  "mrope_section": [
    11,
    11,
    10
  ],
  "mrope_section_sum": 32,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 48,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_kv_repeat": 3,
  "linear_qkv_width": 10240,
  "linear_z_width": 6144,
  "linear_conv_kernel_dim": 4,
  "linear_conv_delay": 3,
  "linear_state_heads": 48,
  "linear_state_dk": 128,
  "linear_state_dv": 128,
  "full_attention_interval": 4,
  "n_linear_layers": 48,
  "n_full_layers": 16,
  "full_attention_indices": [
    3,
    7,
    11,
    15,
    19,
    23,
    27,
    31,
    35,
    39,
    43,
    47,
    51,
    55,
    59,
    63
  ],
  "mtp_num_hidden_layers": 1
}
```
