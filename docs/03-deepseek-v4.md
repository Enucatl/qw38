# 3. Exact Qwen3.8-27B text model contract

[Previous](02-execution-path.md) · [Index](README.md) · [Next](04-numerics.md)

## Why this matters

A plausible transformer is wrong if a packed slice, norm, residual, or state
update differs. This is the executable forward-pass specification.

## Validated constants

Read this table as a wiring diagram, not as tuning advice. A **hidden size** is
the width of the vector passed from layer to layer. A **head** is an independent
smaller vector space used by a mixer; splitting into heads lets the hardware
perform many small operations in parallel. “QK heads” produce queries and keys;
“value heads” produce the values that are accumulated. The counts need not be
equal, which is why the implementation has to repeat Q/K heads explicitly.

| Field | Value |
|---|---:|
| vocabulary / residual / FFN | 248,320 / 5,120 / 17,408 |
| layers | 64: `(GDN,GDN,GDN,attention) * 16` |
| GDN QK / value heads / dimensions | 16 / 48 / 128 |
| convolution width | 4 |
| attention Q / KV heads / dimension | 24 / 4 / 256 |
| rotary dimensions | 64 per head (`partial_rotary_factor=0.25`) |
| RMS epsilon / recurrence dtype | `1e-6` / FP32 |
| maximum positions | 262,144 |

Full-attention layers are zero-based `3,7,...,63`. Test transitions `2 -> 3`,
`3 -> 4`, and layer 63 explicitly.

The first four layers are therefore GDN 0, GDN 1, GDN 2, and attention 3;
layer 4 begins the next group. “48 GDN plus 16 attention” does not mean running
all GDN layers first. The types are interleaved, and every layer consumes the
residual vector produced by its immediate predecessor.

## Common block and weights

The model is a **pre-norm residual block**. `x` is the vector entering a layer;
`RMSNorm` rescales it using its root-mean-square magnitude, and `Mixer` means
either GDN or full attention depending on the layer number. The `x + ...` is a
residual connection: the layer learns a correction while preserving a direct
path for information and gradients. The second residual branch is a dense FFN.
RMSNorm computes in FP32 and scales by `(1 + weight)` (the parameter is stored
as an offset from one), which differs from implementations that store the full
scale directly. Each layer runs:

```text
x <- x + Mixer(RMSNorm(x))
x <- x + down_proj(SiLU(gate_proj(RMSNorm(x))) * up_proj(RMSNorm(x)))
```

There are two separate normalizations. The first feeds the mixer. After the
mixer correction is added, the second feeds the FFN. Reusing the first normalized
vector would be wrong because the first residual addition changed `x`. With one
decode token these are vectors; with prefill the operations apply to every row,
except where the mixer deliberately communicates across positions.

Every layer binds two `[5120]` norms and FFN weights `[17408,5120]`,
`[17408,5120]`, `[5120,17408]`. Global weights include embeddings and an
untied LM head, both `[248320,5120]`, plus final norm. Treat `[out,in]` as the
checkpoint convention and inventory actual keys before conversion.

| Layer kind | Learned tensor | Checkpoint shape `[out,in]` or vector |
|---|---|---:|
| GDN | `in_proj_qkv` / `in_proj_z` | `[10240,5120]` / `[6144,5120]` |
| GDN | `in_proj_a` / `in_proj_b` | `[48,5120]` each |
| GDN | depthwise convolution | `[10240,1,4]` |
| GDN | `A_log` / `dt_bias` / gated-norm weight | `[48]` / `[48]` / `[128]` |
| GDN | `out_proj` | `[5120,6144]` |
| attention | packed query+gate projection | `[12288,5120]` |
| attention | K / V projections | `[1024,5120]` each |
| attention | per-head Q / K norms | `[256]` each |
| attention | output projection | `[5120,6144]` |

Biases are disabled for these linear projections and the GDN convolution in the
pinned config. Tensor key spelling remains a manifest concern because wrapper
prefixes can differ across checkpoint tooling.

## Gated DeltaNet

GDN is a learned recurrent filter. Unlike full attention, it does not retain a
separate representation of every prior token. Instead, `S` is a summary matrix
that is updated when a token arrives. The projections below manufacture the
query, key, value, gate, and decay controls used by that update. The packed
projection is split by contiguous channel ranges; splitting after a convolution
would produce a different model.

From normalized `u[B,T,5120]`:

```text
packed = in_proj_qkv(u) -> [B,T,10240]
Q = packed[...,0:2048]       -> [B,T,16,128]
K = packed[...,2048:4096]    -> [B,T,16,128]
V = packed[...,4096:10240]   -> [B,T,48,128]
Z = in_proj_z(u)             -> [B,T,48,128]
b = in_proj_b(u)             -> [B,T,48]
a = in_proj_a(u)             -> [B,T,48]
```

`B` is the number of sequences processed together and `T` is the number of
positions per sequence in this call. The final dimension holds contiguous
channels. Packing Q, K, and V saves projection overhead, but their ranges still
have distinct meanings and must be sliced at the exact offsets shown.

Apply depthwise causal width-4 convolution and SiLU to packed QKV. “Causal” means
the output at position `t` can use the current row and the previous three rows,
but not a future row. The ring buffer is the compact way to provide those rows
during decode. L2-normalize Q/K and repeat each Q/K head three times: 16 source
heads become 48 heads so each value head has a matching query and key. Let
`beta=sigmoid(b)`,
`g=-exp(A_log)*softplus(a+dt_bias)`, and `q=Q/sqrt(128)`. Per head, with FP32
`S[128,128]`:

```text
S <- exp(g_t) * S
prediction <- k_t^T S
delta <- beta_t * (v_t - prediction)
S <- S + k_t outer delta
y_t <- q_t^T S
```

The dimensions make the update easier to follow. `k_t^T S` maps a 128-value key
through the 128-by-128 memory and produces a 128-value prediction. The outer
product `k_t outer delta` creates another 128-by-128 matrix, so it can update
`S`. Finally, `q_t^T S` reads a 128-value result. These operations happen
independently for each of 48 value heads.

The recurrence first decays old information, predicts what the current key would
retrieve from the old summary, and writes only the prediction error (`delta`).
`beta` controls how strongly that error is written. The query then reads the
updated summary. This order is essential: reading before writing or applying
the decay at a different point changes every later token. Apply headwise RMSNorm
to `y`, multiply by `SiLU(Z)`, flatten 48 heads to 6,144, and apply
`out_proj[5120,6144]`. Persistent state is FP32 `[48,128,128]` and
the official cache's last four packed QKV rows `[10240,4]` per GDN layer.

## Full attention and positions

Full attention keeps the familiar three roles: Q asks what this token wants to
retrieve, K describes each stored token for matching, and V contains the content
to retrieve. Qwen's 24 query heads share four K/V heads (six query heads per
KV head), reducing cache size. The extra half of `q_proj` is an output gate;
it is not another attention head.

`q_proj` emits `[T,24,512]`, split into query and gate halves of 256. K and V
are `[T,4,256]`. Apply per-head Q/K RMSNorm, RoPE to 64 dimensions, append K/V,
repeat KV heads sixfold conceptually, compute causal attention scaled by
`1/sqrt(256)`, multiply output by `sigmoid(gate)`, flatten to 6,144, and apply
`o_proj[5120,6144]`.

For one decode token, attention creates 24 score rows over every cached
position. Softmax turns each row into non-negative weights that sum to one, and
the weighted values become 24 output heads. “Causal” means position `t` may read
only positions through `t`. During one-token decode all cached rows are legal;
during prefill a triangular mask hides later rows in the same chunk.

RoPE rotates paired coordinates by a position-dependent angle so attention can
use relative order without adding a learned position vector. Qwen rotates only
64 of each 256-dimensional head. Text constructs four identical scalar-position
channels: channel 0 controls the causal mask and channels 1–3 supply rotary
positions. A future multimodal
processor supplies temporal/height/width positions and visual embeddings.

## State and transfer ledger

**Estimated, batch one:** recurrent matrices use
`48*48*128*128*4 = 144 MiB`; convolution state uses
`48*10240*4*4 = 7.5 MiB`; attention KV uses 64 KiB/token (2 GiB at 32K).

These state types serve different purposes. The recurrent matrix is a learned
summary and cannot reproduce individual old tokens. The convolution ring keeps
only the local history needed by the width-four filter. Full KV retains an
addressable row for every earlier position, so it grows with context. A session
must carry all three because different layers consume different state.

Reuse DwarfStar validation, ownership, allocation accounting, and differential
testing. Adapt serialization and hybrid scheduling. Reject compressed attention,
sparse indexing, MoE routing/streaming, mHC, and DSpark equations.

## Failure modes and exercise

Common errors are wrong head expansion, gate order, conventional RMS weights,
BF16 recurrence, convolution warm-up, rotating 256 dimensions, and misclassifying
layer 3. Trace embedding, GDN convolution/recurrence/output, layer-3 attention,
FFN, final norm, and logits. Expected: scalar results match pinned Transformers
and an independent llama.cpp trace at every boundary within declared tolerances.
