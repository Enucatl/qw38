# 27. One complete real SwiGLU FFN branch

[Index](README.md) · Implementation tasks: CPU-012 and EDU-013 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter follows one 5,120-value activation through layer 0's real
feed-forward network (FFN). An FFN lets each token transform its own features;
unlike attention or a Gated Delta Network (GDN), it does not read earlier
tokens or keep persistent state.

CPU-012 implements the second branch of a decoder layer:

```text
x = x + mixer(RMSNorm(x))
x = x + down(SiLU(gate(RMSNorm(x))) * up(RMSNorm(x)))
```

Here `*` means element-by-element multiplication. It is not a matrix product.
The first line's mixer is GDN in layer 0 and was implemented in CPU-011. The
CPU-012 diagnostic deliberately starts from a deterministic standalone
activation, rather than the CPU-011 output, so each branch can be isolated when
a number changes. Joining both branches in the 64-layer scheduler remains
CPU-004 work.

## What “feed-forward” means

The input is a vector of 5,120 floating-point numbers. They are not words or
token IDs: each number is one learned feature of the current token. Three
learned matrices transform that vector:

```text
normalized: 5,120 values
gate:       5,120 -> 17,408 values
up:         5,120 -> 17,408 values
down:      17,408 ->  5,120 values
```

`gate` and `up` use different weights. Calling both with the same normalized
input does not make them redundant: training teaches one projection which
features to regulate and the other which content to carry. The temporary
17,408-value space is called the intermediate or FFN width. It gives the model
more room to combine features before returning to the 5,120-value residual
width.

[`execute_ffn_step`](../src/mixer.cpp) reads the three matrices through typed,
non-owning GGUF views established in Chapter 23. Their rows use Q4_K, a packed
roughly four-bit weight format explained in Chapter 17. The scalar path decodes
each row while computing its dot product; it does not first expand the whole
18.97 GB model into FP32.

## Step 1: direct-scale post-mixer RMSNorm

The learned tensor is named `blk.0.post_attention_norm.weight`. The historical
name says “attention” even though layer 0 uses GDN; it means the normalization
after whichever mixer that layer owns.

RMSNorm first measures the root-mean-square size of the activation, then scales
each feature:

```text
mean_square = sum(x[i]²) / 5120
normalized[i] = x[i] / sqrt(mean_square + 1e-6)
normalized[i] *= GGUF_scale[i]
```

The small `1e-6` prevents division by zero. “Direct scale” is important: the
GGUF converter has already changed the source checkpoint's offset convention,
so Quartz must not add one again. The prepared scalar parameter is 5,120 FP32
values, exactly 20 KiB.

The original `x` remains unchanged because the branch needs it for the final
residual addition.

## Step 2: independent gate and up projections

A matrix-vector projection computes one dot product per output row. The gate
and up matrices each have 17,408 rows of 5,120 weights:

```text
gate_values = ffn_gate_matrix × normalized
up_values   = ffn_up_matrix   × normalized
```

That is `17,408 × 5,120 = 89,128,960` scalar weight products for each
projection. The diagnostic computes every row, not only the values printed in
the fixture.

The independent fixture generator maps the pinned GGUF, hashes the exact
physical rows selected for evidence, decodes their Q4_K blocks in Python, and
computes their dot products separately. The row hashes prove that a matching
number did not come from accidentally testing a different row.

## Step 3: SiLU gate and elementwise product

SiLU, pronounced “sigh-loo,” is a smooth nonlinear function:

```text
SiLU(g) = g / (1 + exp(-g))
activated[i] = SiLU(gate_values[i]) * up_values[i]
```

For a large positive gate, SiLU is close to the gate itself. For a large
negative gate, its output approaches zero. Around zero it changes smoothly.
Multiplying by `up_values[i]` lets the gate amplify, reduce, or reverse the
corresponding carried feature. Because each index is paired only with the same
index, this stage performs 17,408 ordinary scalar multiplications rather than a
dot product.

“SwiGLU” is the conventional name for this combination of a SiLU-transformed
gate and a second linear projection. The fixture independently checks selected
gate, up, and activated values, which detects swapped projections, a missing
SiLU, or an incorrect elementwise pairing.

## Step 4: down projection and residual

The down matrix combines the 17,408 activated values into a 5,120-value
correction:

```text
correction = ffn_down_matrix × activated
output[i] = original_x[i] + correction[i]
```

The down projection performs another
`5,120 × 17,408 = 89,128,960` scalar weight products. Gate, up, and down
together therefore perform 267,386,880 products for one token in one FFN
branch. This arithmetic count is an estimate from the exact dimensions, not a
speed measurement.

The addition is called a residual connection. It gives the branch permission
to learn a correction instead of reconstructing the entire input. Pytest checks
selected output lanes against an explicit FP32 addition of the deterministic
input and the computed correction.

## Temporary memory and lifetime

The scalar workspace contains:

| Buffer | FP32 values | Meaning |
|---|---:|---|
| normalized | 5,120 | normalized branch input |
| gate | 17,408 | first wide projection |
| up | 17,408 | second wide projection |
| activated | 17,408 | `SiLU(gate) * up` |
| correction | 5,120 | down-projected result |
| **Total** | **62,464** | **249,856 bytes = 244 KiB** |

This is temporary memory: it can be reused after the token finishes and never
belongs in a checkpoint. The FFN has no convolution ring, KV cache, or
recurrent matrix. Its weights belong to the engine; its workspace belongs to
the current execution; only the final residual continues to the next branch.

CUDA will eventually fuse some of these stages or use BF16 storage, but it must
remain numerically within the tolerances frozen before that optimization.

## Evidence and fixture equality

[`tools/generate_real_ffn_step_fixtures.py`](../tools/generate_real_ffn_step_fixtures.py)
creates [`fixtures/real_ffn_step.json`](../fixtures/real_ffn_step.json) from the
pinned model. [`tests/test_real_ffn.py`](../tests/test_real_ffn.py) compares the
native diagnostic with that frozen file.

Fixture equality does not always mean identical bytes for floating-point
arithmetic. The test converts stored little-endian FP32 bytes back to numbers
and reports three bounded kinds of error:

- absolute error: the direct distance between two values;
- relative error: that distance compared with the expected value's size; and
- RMS error: a summary of the whole selected set.

It also rejects NaN and infinity. Exact equality is used where rounding is not
an acceptable explanation: model identity, selected physical-row SHA-256
hashes, workspace counts, and the explicit FP32 residual-add relationship.

The fixture independently covers the norm and selected gate/up/SwiGLU rows.
The native diagnostic executes the complete down matrix and verifies that all
62,464 workspace values are finite, but its final correction is currently a
native regression tap, not an independent semantic-authority result. A direct
Transformers or trace comparison is still required by TRC-001, TRC-002, and
ORA-001 before the full scalar scheduler can become the frozen CUDA oracle.

## Failure boundary

Every pointer and exact element count is checked before the function writes an
output. The negative test shortens the activated buffer by one value and proves
that the gate and final output remain untouched.

After validation, a hypothetical arithmetic failure could leave temporary
workspace values partially written. That is safe at this boundary because the
FFN has no persistent state and the caller must discard a failed result. Atomic
session frontier and checkpoint behavior remain SES-002 and SES-003 work.

## Proof boundary

**External:** pinned Transformers defines the direct branch order and SwiGLU
equation; the pinned GGUF identity and converter contract define the admitted
weight representation.

**Measured:** on the admitted layer-0 artifact, all complete native workspace
buffers are finite, selected independently decoded normalization/gate/up/SwiGLU
taps meet frozen numeric tolerances, exact row hashes match, residual addition
is exact in FP32, and malformed workspace fails before writes.

**Estimated:** the branch performs 267,386,880 scalar weight products and uses
244 KiB of explicit FP32 workspace, calculated from the production dimensions.

CPU-012 does not yet join GDN and FFN, execute a real attention layer, process a
second real token, schedule all 64 layers, apply final norm, or produce logits.
Those are explicit CPU-004 and later trace/oracle gates, not implied by this
single-branch result.
