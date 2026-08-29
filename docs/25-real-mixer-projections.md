# 25. Executing real mixer projections and sizing their workspaces

[Index](README.md) · Implementation tasks: CPU-010 and EDU-011 in
[`implementation_ledger.md`](../implementation_ledger.md)

Earlier chapters proved one stored matrix row at a time, then bound all model
weights, then proved how packed outputs are sliced. This chapter joins those
boundaries: [`src/mixer.cpp`](../src/mixer.cpp) runs complete layer-0 GDN and
layer-3 attention mixer projections from the real pinned GGUF.

This is the first gate that computes every output row of real learned mixer
matrices. It still stops before convolution, recurrence, attention lookup, and
residual addition.

## Activation, weight row, and projection output

An **activation** is the current numeric representation of a token. Qwen's
residual activation contains 5,120 FP32 values in the scalar oracle. A projection
matrix has one 5,120-weight row for every output value. Each output is one dot
product:

```text
output[row] = sum(weight[row, column] × activation[column])
```

Chapter 21 tested selected rows. A **complete matvec** repeats that operation for
every row. For example, layer 0's packed QKV matrix has 10,240 rows, so it emits
all 10,240 packed values in one call to `tensor_matvec`.

The diagnostic activation is deterministic rather than sampled:

```text
a[i] = (((i × 37) mod 101) - 50) / 32, rounded to FP32
```

This pattern contains positive, negative, and zero-adjacent values and repeats
only after 101 lanes. Anyone can regenerate it without storing 20 KiB of fixture
data. It is a projection test input, not a token embedding and not a claim about
natural model activations.

## Typed weights remove string lookup from execution

`project_gdn_mixer` receives `GdnLayerWeights`, not a GGUF filename or tensor
name. It uses four already checked fields:

- packed QKV: 10,240 outputs;
- value gate: 6,144 outputs;
- alpha controls: 48 outputs; and
- beta controls: 48 outputs.

`project_attention_mixer` uses the packed query/gate matrix for 12,288 outputs,
then splits it into 6,144 query and 6,144 gate values. It also projects 1,024 key
and 1,024 value outputs.

This is why typed binding was a prerequisite. Projection code expresses model
roles directly and delegates row decoding and dot arithmetic to the one admitted
`tensor_matvec` implementation.

## What a workspace is

A **workspace** is temporary memory used between stages of a forward pass. It is
not a learned weight and is not persistent session history. A caller owns the
memory; the small workspace structures contain pointers and exact element counts.

The GDN projection workspace is:

| Field | FP32 values | Purpose |
|---|---:|---|
| packed QKV | 10,240 | input to width-four convolution |
| value gate | 6,144 | later SiLU gate for recurrent output |
| alpha | 48 | decay control input |
| beta | 48 | update-strength control input |
| **total** | **16,480** | **65,920 bytes = 64.375 KiB** |

The attention projection workspace is:

| Field | FP32 values | Purpose |
|---|---:|---|
| packed query/gate | 12,288 | stable packed diagnostic boundary |
| query | 6,144 | per-head normalized/RoPE input |
| gate | 6,144 | later sigmoid output gate |
| key | 1,024 | four key heads |
| value | 1,024 | four value heads |
| **total** | **26,624** | **106,496 bytes = 104 KiB** |

The diagnostic allocates both simultaneously, so it fills 43,104 values or
172,416 bytes. A production single-layer schedule can reuse storage between GDN
and attention variants. That future reuse must be listed in the allocation
ledger; this gate fixes element counts, not a final CUDA allocation strategy.

Every pointer must be non-null and every count exact before the first matrix
multiplication. A 10,239-value packed GDN buffer is rejected rather than partly
filled. Once admitted, the diagnostic initializes every output to NaN and checks
that all 43,104 values became finite. This catches a loop that accidentally
stops before the final row.

## Why GDN remains packed here

The GDN packed QKV result feeds a depthwise causal convolution before it is
split. Splitting raw projection output in this function would encourage the
wrong scheduler order. Therefore `project_gdn_mixer` leaves its 10,240 values
packed.

Attention has no corresponding packed convolution, so its projection function
immediately applies the per-head query/gate split admitted in Chapter 24. It
retains the packed buffer as a diagnostic boundary and also writes separate
query and gate workspaces.

## Independent selected-row evidence

Checking all 30,816 learned output rows independently in Python would repeat
157,777,920 weight/activation products and make the test unnecessarily slow.
Instead, Quartz combines two forms of evidence:

1. Native code executes every row and proves every output slot is finite.
2. The independent Python decoder checks selected rows at every important
   boundary and freezes them in
   [`fixtures/mixer_projections.json`](../fixtures/mixer_projections.json).

Selected GDN QKV rows include the first and last Q row, first and last K row,
and first and last V row. Gate, alpha, and beta include endpoints. Attention
query/gate rows cover query and gate halves in head 0, query lanes in head 1,
and the final gate lane in head 23; K and V include both endpoints.

Each fixture entry stores the SHA-256 of every selected physical row as well as
its expected FP32 dot bytes. Pytest rereads and hashes the mapped row before
comparing the complete native projection tap. Thus a matching number cannot hide
a changed source row, and a matching row hash cannot hide wrong arithmetic.

The attention split taps are then reconstructed from the independently admitted
packed row values. This proves that real, rather than synthetic, output passes
through the per-head layout correctly.

## Scalar cost and measurement

The two mixer diagnostics evaluate 30,816 matrix rows of width 5,120, or
157,777,920 scalar weight/activation products plus accumulation. On this
development host the focused native command completed in approximately 0.29
seconds with about 185 MiB maximum resident host memory. This is a **measured
diagnostic value**, not an end-to-end token speed and not an RTX 5090 CUDA result.

## Proof boundary

**Measured:** complete real layer-0 and layer-3 mixer projections fill all
43,104 workspace values; independently decoded and hashed selected rows match
exact FP32 taps; attention real-output splitting matches its per-head layout;
and a short workspace fails before projection.

CPU-010 does not apply input RMSNorm, GDN convolution or recurrence, attention
RoPE/KV/softmax, mixer output projection, residual addition, FFN, final norm, or
logits. It also does not promise atomic session behavior after a later arithmetic
failure. Those remain CPU-004 and session gates. This chapter establishes the
real learned-matrix and temporary-memory boundary those stages will consume.
