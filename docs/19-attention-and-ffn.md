# 19. Grouped causal attention, partial RoPE, and SwiGLU

[Index](README.md) · Implementation tasks: CPU-003 and EDU-005 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter explains the scalar primitives in
[`src/attention.cpp`](../src/attention.cpp) without assuming prior machine-
learning knowledge. The exact production contract is pinned in
[`pins/attention_ffn_contract.json`](../pins/attention_ffn_contract.json), and
small deterministic cases live in
[`fixtures/attention_ffn_authority.json`](../fixtures/attention_ffn_authority.json).
CPU-003 establishes inspectable arithmetic; it does not yet execute complete
real-weight layers.

## Attention as a memory lookup

Attention lets the current token retrieve useful information from earlier
tokens. Three learned projections describe the lookup:

- a **query** says what the current token is looking for;
- a **key** describes what an earlier token can be found by; and
- a **value** carries the information returned if that earlier token is useful.

For each earlier position, attention takes a dot product between the current
query and that position's key. Larger scores receive larger weights. The output
is a weighted sum of the corresponding values.

Qwen3.8 has a full-attention layer at indices 3, 7, 11, and every fourth layer
through 63. CPU-003 names layers 3, 7, and 63 in its fixtures to cover the first,
an early repeated layer, and the final boundary. The focused values are
synthetic; the layer names prove schedule selection, not real-model logits.

## Heads and grouped-query attention

Attention is divided into **heads**, independent groups that can learn different
relationships. Qwen3.8 uses:

| Quantity | Production value |
|---|---:|
| query heads | 24 |
| key/value heads | 4 |
| numbers per head | 256 |
| query heads per KV head | 6 |

Ordinary multi-head attention could store a separate key and value for every
query head. **Grouped-query attention (GQA)** instead lets six adjacent query
heads share one key/value head:

```text
kv_head = query_head / 6
```

Thus query heads 0–5 use KV head 0, heads 6–11 use KV head 1, and so on. Sharing
reduces persistent history while retaining more query perspectives. It must be
adjacent grouping, not round-robin assignment.

The query projection also produces a separate gate. Its production output has
`24 × 256 × 2 = 12,288` numbers: half are queries and half become elementwise
sigmoid gates. After attention, every output lane is multiplied by its gate.
The 24 attention heads concatenate to 6,144 numbers before the learned output
projection returns to the 5,120-wide residual stream.

## RMS normalization of queries and keys

Each projected query and key head is independently RMS-normalized:

```text
rms = sqrt(mean(x[i]²) + 0.000001)
normalized[i] = x[i] / rms × (1 + learned_weight[i])
```

RMS means “root mean square.” Squaring removes signs, the mean measures typical
squared magnitude, and the square root restores the original unit. The epsilon
prevents division by zero. Qwen's stored RMSNorm parameter is an offset from one,
which is why the formula uses `1 + weight`, not merely `weight`.

Qwen shares one 256-number query-normalization weight across query heads and one
256-number key-normalization weight across KV heads in a layer. Values are not
normalized by this operation.

## RoPE: putting position into query and key

A bare dot product does not know whether a key came from the previous token or
thousands of tokens ago. **Rotary Position Embedding (RoPE)** rotates pairs of
query and key coordinates by position-dependent angles. Applying the same
rotation scheme to queries and keys makes their dot product sensitive to
relative position without storing an added position vector in the KV value.

Qwen3.8 uses a head width of 256 but rotates only the first 64 coordinates. This
is **partial RoPE**, with fraction `64 / 256 = 0.25`. Coordinates 64–255 pass
through unchanged. The 64-coordinate region is split into two 32-coordinate
halves. For lane `i` in the first half and its partner `i + 32`:

```text
angle = position / 10,000,000^(2i / 64)
new_first  = first × cos(angle) − second × sin(angle)
new_second = second × cos(angle) + first × sin(angle)
```

Position zero has angle zero, so cosine is one and sine is zero: the first token
is unchanged. Higher frequency pairs rotate faster than lower frequency pairs.
Rotating values, rotating all 256 lanes, pairing adjacent lanes, or applying RoPE
after caching would define a different model.

The upstream configuration also contains multimodal RoPE sections. Quartz v1 is
text-only; ordinary text positions use the text frequency path. Vision position
axes remain outside the release boundary.

## The KV timeline and causality

After normalization and RoPE, the current key is appended to that layer's
**key/value cache (KV cache)**. The unrotated projected value is appended beside
it. A decode step at position `t` may read cache positions 0 through `t` only.
This is **causality**: output cannot depend on a future token.

CPU-003 initializes future fixture cache rows with conspicuous values near
positive and negative 1,000. Correct outputs remain small because the scalar
loop stops at the current position. Reading the allocated capacity rather than
the committed frontier would immediately contaminate the fixture.

With two-byte production KV storage, one token occupies:

```text
4 KV heads × 256 lanes × 2 (key and value) × 2 bytes = 4,096 bytes/layer
4,096 bytes × 16 attention layers = 65,536 bytes/token
65,536 × 131,072 tokens = 8 GiB
```

**Estimated:** this explains the plan's exact 8 GiB KV term. **Proposed:** it
does not prove allocation, graph overhead, or the required 1.5 GiB free reserve;
MEM-001 retains that measured gate.

## Scores and stable softmax

For one query head and every admitted context position:

```text
score[t] = dot(query, key[t]) / sqrt(256)
```

Dividing by the square root of head width keeps scores from growing simply
because a head has many lanes. **Softmax** turns the scores into non-negative
weights that sum to one:

```text
weight[t] = exp(score[t]) / sum(exp(score[j]))
```

Direct exponentiation can overflow for a large score. Subtracting the maximum
score from every score leaves the final ratios unchanged and ensures the largest
exponent is `exp(0) = 1`. Quartz uses this **stable softmax** in FP32. The
attention result is the weighted value sum, followed by the query projection's
sigmoid output gate.

During multi-token prefill, an explicit causal mask serves the same rule. The
focused CPU-003 primitive advances one token at a time; memory-bounded causal
prefill and the 131,072 boundary remain ATN-002 work.

## The feed-forward network and SwiGLU

Every one of the 64 layers contains a feed-forward network (FFN) after its token
mixer. The layer first RMS-normalizes the 5,120-wide residual vector. Two learned
matrix projections expand it independently to 17,408 numbers:

```text
gate = gate_projection(x)
up   = up_projection(x)
```

The **SiLU** activation is `silu(g) = g / (1 + exp(-g))`. Qwen uses the SwiGLU
combination:

```text
activated = silu(gate) × up
output = down_projection(activated)
```

The multiplication is elementwise: intermediate lane 10 multiplies lane 10,
not every other lane. The down projection returns from 17,408 to 5,120 numbers.
The decoder then adds that output to the residual stream.

[`swiglu_ffn`](../src/attention.cpp) retains gate, up, activated, and down-output
buffers as visible taps. Its small fixture includes non-square matrices, which
makes a transposed weight orientation easier to detect. Real Q4_K/Q6_K matrix
rows and full residual scheduling remain CPU-004 work.

## Fixtures and the failed relative gate

The fixture generator is an explicit FP32 scalar transcription of the pinned
Transformers eager attention, RoPE, RMSNorm, and MLP equations. It uses a small
six-query/two-KV-head shape with the same three-to-one grouped pattern, four of
eight lanes rotated, four causal positions, and distinct cases labeled 3, 7,
and 63. The frozen gates are:

| Metric | CPU-003 limit |
|---|---:|
| maximum absolute error | `3e-6` |
| maximum relative error | `3e-5` |
| RMS error | `1e-6` |
| minimum cosine similarity | `0.999999` |

The first run failed relative error even though absolute error was only about
`5.17e-8`. Diagnosis found that the Python generator called double-precision
transcendental functions and rounded afterward while claiming each operation was
FP32. It now calls the platform's `expf`, `sqrtf`, `powf`, `sinf`, and `cosf`
directly. The native implementation and frozen limits did not change. The
ledger preserves the failed value, diagnosis, and corrected result.

**External primary:** shapes, ordering, and equations come from Transformers
revision `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`, pinned with the source hash
and used symbols.

**Measured local:** layers 3/7/63 attention outputs and KV caches, causal sentinel
behavior, and FFN gate/up/activation/output taps pass the frozen metrics. The
worst measured relative error is about `7.80e-6`.

**Historical boundary:** as with CPU-002, local PyTorch was absent at this gate.
These remain transparent scalar transcriptions, not direct eager traces.
ORA-001 subsequently compared the primary eager authority, the independent
same-GGUF llama.cpp oracle, and Quartz as documented in
[Chapter 38](38-scalar-authority-tolerances.md).

## Proof boundary

CPU-003 proves the focused scalar interpretation of grouped head mapping,
per-head RMSNorm, partial RoPE, causal append/read order, stable softmax, output
gating, and SwiGLU tap order for deterministic small cases. It does not prove
real projection weights, all 16 production attention states, arbitrary prefill,
131,072-token capacity, the full residual layer, CUDA kernels, language quality,
memory fit, or speed. Each remains a named downstream gate.
