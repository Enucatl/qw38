# 28. Real grouped-query attention and KV state

[Index](README.md) · Implementation tasks: CPU-013 and EDU-014 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter follows two tokens through the real attention mixer in layer 3.
Two tokens are the smallest useful real-state example: at position zero, rotary
position encoding does nothing and attention has only one value to choose. At
position one, rotation changes Q and K, the cache contains history, and softmax
must choose a mixture of two positions.

Layer 3 is the first full-attention layer in the pinned 64-layer schedule. Its
mixer branch is:

```text
normalized = RMSNorm(residual)
Q, output_gate, K, V = learned projections(normalized)
context = causal_grouped_attention(Q, K, V, KV_cache)
correction = output_projection(context * sigmoid(output_gate))
output = residual + correction
```

The FFN follows this branch but is kept separate in the diagnostic. Joining
both branches across the real layer schedule remains CPU-004 work.

## Why attention needs Q, K, and V

Each token arrives as 5,120 learned features. Three projections give those
features different jobs:

- a **query** (Q) describes what the current token is looking for;
- a **key** (K) describes what each stored token can be matched by; and
- a **value** (V) carries the information retrieved from that stored token.

Quartz computes a query-key dot product for every allowed past position. A
larger score means that stored value should contribute more strongly. Softmax
turns all scores into nonnegative weights whose sum is one, and the weighted
values become the attention result.

The production dimensions are:

```text
residual:        5,120
query heads:        24 × 256 lanes = 6,144 values
key/value heads:     4 × 256 lanes = 1,024 values each
output gate:        24 × 256 lanes = 6,144 values
```

A head is simply one independently scored slice of a projection. “Lane” means
one number inside that head.

## Step 1: normalize and project the real GGUF rows

[`prepare_attention_scalar_parameters`](../src/mixer.cpp) decodes the layer's
5,120 input scales and its 256 query/key head scales. These are direct GGUF
scales: the converter has already added the source model's implicit one.
Applying the offset twice would silently change every score.

The input RMSNorm equation is:

```text
normalized[i] = residual[i] / sqrt(mean(residual²) + 1e-6)
normalized[i] *= input_scale[i]
```

The admitted layer-3 matrices then perform:

```text
packed Q+gate: 5,120 -> 12,288   Q8_0
K:             5,120 ->  1,024   Q8_0
V:             5,120 ->  1,024   Q8_0
```

Q and its output gate share one physical matrix but alternate by head: 256 Q
rows, then 256 gate rows, repeated 24 times. Chapter 24 explains why Quartz
must split this head-local packing rather than cut the matrix into two global
halves.

## Step 2: normalize heads and apply partial RoPE

Q and K are independently RMS-normalized within each 256-lane head. V is not.
Quartz then applies Rotary Position Embedding (RoPE) to only the first 64 lanes
of each normalized Q and K head.

RoPE treats the first 64 lanes as 32 pairs. Each pair rotates by an angle based
on the token position and its pair index:

```text
angle = position / 10,000,000^(2 * pair / 64)
[left', right'] = rotation(angle) × [left, right]
```

At position zero the angle is zero, so every pair is unchanged. At position
one the pairs rotate. Lanes 64 through 255 remain exactly unrotated; this is
what “partial RoPE” means. Applying RoPE to all 256 lanes would produce valid
floating-point numbers but the wrong model.

The fixture samples lanes 31, 32, 63, and 64 around both the pair-half and
rotary/non-rotary boundaries.

## Step 3: append K and V to the timeline

After normalization and RoPE, the current K is written into the key cache. The
unrotated raw V projection is written into the value cache:

```text
key_cache[position, kv_head, lane] = rotated_normalized_K
value_cache[position, kv_head, lane] = V
```

This cache is persistent session state. It differs from temporary workspace:
later tokens need the cache, and saving or reusing a prefix must preserve it.
The scalar diagnostic uses FP32 for inspectability. V1 production storage is
two bytes per cached value.

For one token in one attention layer, K and V together contain
`2 × 1,024 = 2,048` values. At two bytes each that is 4 KiB per token per
attention layer. Across 131,072 positions and 16 attention layers, the exact
capacity is 8 GiB—the KV allocation named in the v1 memory gate.

CPU-013 uses a two-position FP32 cache: 4,096 values or 16 KiB. That small test
capacity proves indexing and boundary rejection; it is not the 128K fit proof.

## Step 4: grouped-query head mapping

The model has 24 query heads but only four K/V heads. This is grouped-query
attention (GQA): six adjacent query heads share one K/V head.

```text
query heads  0..5  -> KV head 0
query heads  6..11 -> KV head 1
query heads 12..17 -> KV head 2
query heads 18..23 -> KV head 3
```

Sharing K/V greatly reduces persistent cache memory. It does not mean the six
query heads become identical; each keeps its own Q and output gate, so it can
score the shared history differently. The real fixture samples query heads 5
and 6 specifically to catch an off-by-one error at the first group boundary,
and head 23 to exercise the last group.

## Step 5: causal scores and stable softmax

For each query head, Quartz scores only positions from zero through the current
position:

```text
score[t] = dot(Q, key_cache[t]) / sqrt(256)
```

“Causal” means a token cannot read a future position. At position one there are
exactly two scores. Stable softmax subtracts the largest score before taking
exponentials:

```text
e[t] = exp(score[t] - max_score)
probability[t] = e[t] / sum(e)
context[lane] = sum(probability[t] * value_cache[t, lane])
```

Subtracting the maximum changes no probability ratio, but prevents large scores
from overflowing `exp`. The scalar path accumulates these operations in FP32 so
its order is explicit and comparable with fixtures.

## Step 6: output gate, projection, and residual

Each context lane is regulated by its corresponding projected gate:

```text
gated_context[i] = context[i] * sigmoid(output_gate[i])
```

Sigmoid maps any finite gate to a number between zero and one. The learned Q6_K
output matrix then combines all 6,144 gated head values into a 5,120-value
correction. Finally, the original activation is added back in FP32.

The four projection matrices perform 104,857,600 scalar weight products per
token: 73,400,320 for packed Q/gate, K, and V, plus 31,457,280 for the output
projection. This is exact dimension arithmetic, not a throughput measurement.

## Memory ownership

Prepared scalar parameters contain 5,632 FP32 values (22 KiB): input norm plus
query/key head norms. At the two-token diagnostic capacity, the explicit
workspace contains 43,010 FP32 values (172,040 bytes, about 168 KiB):

| Temporary buffer | Values |
|---|---:|
| normalized input | 5,120 |
| packed Q/gate | 12,288 |
| split Q and gate | 6,144 + 6,144 |
| K and V projections | 1,024 + 1,024 |
| gated attention result | 6,144 |
| score scratch | 2 |
| output correction | 5,120 |

The score scratch grows with the current context in this simple scalar oracle.
The production CUDA prefill design must remain memory-bounded rather than
materializing an entire square attention matrix.

## Independent evidence

[`tools/generate_real_attention_step_fixtures.py`](../tools/generate_real_attention_step_fixtures.py)
maps the pinned GGUF and separately decodes the selected Q8_0 rows. It
transcribes direct RMSNorm, partial RoPE, grouped causal scores, stable softmax,
and sigmoid gating with FP32 `libm` operations. The resulting
[`fixtures/real_attention_step.json`](../fixtures/real_attention_step.json)
contains:

- token-one normalized input and raw Q/gate projection taps;
- rotated normalized K and raw V cache taps at both positions;
- gated causal output taps for heads 0, 5, 6, and 23; and
- SHA-256 hashes for every physical row named by the displayed evidence.

[`tests/test_real_attention.py`](../tests/test_real_attention.py) applies frozen
absolute, relative, and RMS limits and rejects NaN or infinity. It also checks
the exact FP32 residual-add relationship.

The complete Q6_K output projection executes natively and every result is
finite, but the correction is currently a native regression tap rather than an
independently decoded semantic result. Direct trace/oracle admission remains
TRC-001, TRC-002, and ORA-001 work.

## Failure and atomicity boundary

Exact parameter, workspace, cache, capacity, position, and output counts are
validated before any cache write. One negative test shortens the attention
output buffer; another requests position two from a capacity-two cache. Both
prove that KV and final output remain untouched.

Once validation passes, attention appends KV before running the Q6_K output
projection. A hypothetical later failure could therefore leave temporary
session state changed. CPU-013 proves correct mutation order, not transactional
request commit. SES-002 must stage state or restore it before exposing a new
session frontier.

## Proof boundary

**External:** pinned Transformers defines per-head normalization, partial RoPE,
six-to-one GQA mapping, causal scaled scores, stable softmax, raw-V caching, and
sigmoid output gating.

**Measured:** two real layer-3 positions match independently decoded norm,
projection, RoPE/KV, grouping, causal-softmax, and gated-output taps; complete
native buffers are finite; residual addition is exact; malformed workspace and
capacity fail before KV mutation.

**Estimated:** projection work is 104,857,600 scalar weight products per token;
the explicit two-position scalar workspace is about 168 KiB; production
two-byte KV arithmetic yields exactly 8 GiB at 128K across 16 layers.

CPU-013 does not yet execute attention prefill, join the attention FFN, traverse
the hybrid 64-layer schedule, apply final norm, compute logits, or demonstrate
the production KV allocation. Those remain explicit later gates.
