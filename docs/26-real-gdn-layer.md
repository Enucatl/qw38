# 26. One complete real GDN mixer layer

[Index](README.md) · Implementation tasks: CPU-011 and EDU-012 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter follows one token through the real layer-0 Gated Delta Network
mixer. [`execute_gdn_mixer_step`](../src/mixer.cpp) now joins the learned
projections from Chapter 25 to convolution history, recurrent history, gated
normalization, output projection, and the first residual addition.

The FFN half of layer 0 remains separate. “Complete GDN mixer” therefore means
the first branch in this decoder-layer equation:

```text
x = x + GDN(RMSNorm(x))
x = x + FFN(RMSNorm(x))       # next boundary, not in CPU-011
```

## Prepared parameters, temporary work, and persistent state

Three kinds of memory participate, and confusing them causes checkpoint and
reuse bugs.

**Prepared parameters** are fixed model values decoded once for the scalar
path: the 5,120 input-norm scales, 10,240 × 4 convolution weights, 48 folded-A
values, 48 time biases, and 128 gated-norm scales. They occupy 46,304 FP32
values, or 180.875 KiB, per prepared GDN layer. CUDA will later use resident or
repacked representations rather than this scalar layout.

**Workspace** holds temporary results for the current token: normalized input,
projections, convolved QKV, both physical and semantic head orders, gates,
recurrent output, normalized/gated output, and the 5,120-value mixer correction.
The explicit CPU-011 workspace contains 78,208 FP32 values, or 305.5 KiB. It can
be reused by the next token after the step finishes.

**Persistent state** summarizes earlier tokens and must survive:

- convolution ring: `10,240 channels × 4 = 40,960` FP32 values;
- recurrent matrices: `48 heads × 128 × 128 = 786,432` FP32 values.

Together one GDN layer owns 827,392 state values, or 3.15625 MiB. Across 48 GDN
layers this yields the previously estimated 144 MiB recurrent matrices and 7.5
MiB convolution rings. A session—not the immutable engine—owns these values.

## Step 1: direct-scale input RMSNorm

The input is the deterministic 5,120-value residual used in Chapter 25. The
ordinary source checkpoint treats its learned norm weight as an offset, but the
GGUF converter has already added one. CPU-011 decodes `attn_norm.weight` and
calls the direct-scale `rms_norm_scale`:

```text
mean_square = sum(x[i]²) / 5120
normalized[i] = x[i] / sqrt(mean_square + 1e-6)
normalized[i] *= GGUF_scale[i]
```

The normalized activation, not the original residual, feeds all four GDN
projection matrices. The original residual remains available for the addition
at the end.

## Step 2: project, convolve, then split

The typed matrices produce packed QKV, value gate Z, alpha, and beta controls.
The packed 10,240-channel QKV result enters the causal depthwise convolution
before any split:

```text
ring = [three previous projected values, current projected value]
convolution = dot(ring, four weights)
convolved = SiLU(convolution)
```

The fixture begins with zero state, so the first token sees three zeros and its
current projection. After the call, every channel ring is `[0, 0, 0, current]`.
The convolved vector then splits into 2,048 Q, 2,048 K, and 6,144 V values.

Moving the split earlier would require separate convolution-state conventions
and risks applying the converter's value-head permutation at the wrong point.
Keeping the physical packed ring matches the GGUF convolution rows directly.

## Step 3: cross the physical/semantic head boundary

GGUF stores value-associated heads in tiled order:

```text
[replica][key head][lane]
```

The readable recurrence uses grouped order:

```text
[key head][replica][lane]
```

CPU-011 converts convolved V, projected Z, alpha, beta, folded A, and time bias
from tiled to grouped order. Q and K have only 16 key heads, so they remain in
their existing order. Grouped value heads 0, 1, and 2 reuse key head 0; heads 3,
4, and 5 reuse key head 1.

The convolution ring deliberately remains physical/tiled because it stores the
packed GGUF channels. The recurrent state deliberately remains semantic/grouped
because its head index defines which Q/K pair updates each matrix. A checkpoint
must preserve both conventions and its layout version.

## Step 4: folded gates and recurrent mutation

For each grouped value head:

```text
update_gate = sigmoid(beta_projection)
log_decay = GGUF_folded_A × softplus(alpha_projection + dt_bias)
```

The folded A is already `-exp(source A_log)`, so CPU-011 does not exponentiate it
again. Query and key are L2-normalized; query also receives `1/sqrt(128)`.

The zero-initialized first-token state then follows the admitted mutation order:

```text
S = exp(log_decay) × S
prediction = transpose(key) × S
delta = (value - prediction) × update_gate
S = S + key outer delta
recurrent_output = transpose(query) × S
```

Even though decay has no visible effect on an all-zero initial matrix, its
values are independently checked. Later-token fixtures will be needed to prove
decay against nonzero real state and chunked execution.

## Step 5: gated RMSNorm

Transformers defines the gated norm independently for each 128-lane value head:

```text
variance = mean(recurrent_output²)
normalized = recurrent_output / sqrt(variance + 1e-6)
scaled = normalized × direct_GGUF_weight
gated = scaled × SiLU(Z)
```

Normalization happens before the Z gate. The learned weight is a direct scale;
no extra one is added. Both recurrent values and the SiLU gate use FP32 in this
scalar boundary. [`gdn_gated_rms_norm`](../src/gdn.cpp) keeps this equation
separate from ordinary and attention RMSNorm so their weight conventions cannot
be mixed accidentally.

The gated grouped result converts back to tiled order because the GGUF
`ssm_out.weight` columns were converted to expect tiled value heads. That matrix
produces a 5,120-value mixer correction.

## Step 6: residual addition

The output of the GDN branch is:

```text
branch_output[i] = original_residual[i] + mixer_correction[i]
```

Pytest checks selected lanes with explicit FP32 addition. It does not feed this
branch output into the FFN yet, so it is not a complete decoder-layer output.

## Independent evidence

[`tools/generate_real_gdn_step_fixtures.py`](../tools/generate_real_gdn_step_fixtures.py)
maps the pinned GGUF independently, decodes the necessary Q8_0/F32 rows, and
transcribes the pinned Transformers operations with float `libm` functions. It
freezes selected taps in
[`fixtures/real_gdn_step.json`](../fixtures/real_gdn_step.json):

- input normalization endpoints;
- convolution values across Q, K, and multiple physical V heads;
- grouped V values proving the tiled/grouped permutation;
- alpha, log-decay, and update gates;
- recurrent and gated outputs for grouped heads 0, 1, and 3; and
- exact convolution-ring and recurrent-matrix state elements.

Heads 0 and 1 share key head 0 but occupy different physical tiled locations;
head 3 uses key head 1. This selection catches both replica and key-head mapping
errors. Numeric gates report absolute, relative, and RMS errors under frozen
tolerances. The native diagnostic executes every output and state element, while
the independent fixture targets the highest-risk boundaries.

The final 5,120-value output projection is currently checked as a deterministic
native tap plus the exact residual relationship. A later Transformers/trace
oracle remains mandatory before CPU-004 can close; CPU-011 does not promote its
own complete output to semantic authority.

## Failure and atomicity boundary

All pointers and exact counts are validated before state mutation. A deliberately
short gated-tiled workspace returns an error with both zero state buffers
unchanged.

After validation, convolution and recurrence update state in place. A hypothetical
arithmetic or output-projection failure after those mutations would not roll the
state back. Session-level atomic staging and commit remain SES-002 work. This is
an explicit boundary: CPU-011 proves correct one-step mutation, not request-level
transaction semantics.

## Proof boundary

**External:** the pinned Transformers source defines convolution-before-split,
folded gate inputs, recurrence order, and variance-mean gated RMSNorm with direct
scale followed by FP32 SiLU gating.

**Measured:** the real layer-0 first-token path matches independently generated
normalization, convolution, layout, gate, recurrence, gated-norm, and state taps;
the native output obeys exact FP32 residual addition; malformed workspace fails
before state mutation.

CPU-011 still excludes the layer FFN, real attention state, multiple real tokens,
all 64 layers, final norm, and logits. Those remaining CPU-004 increments must
preserve the state and layout contracts established here.
