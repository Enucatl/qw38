# 22. From source parameters to GGUF runtime values

[Index](README.md) · Implementation tasks: CPU-007 and EDU-008 in
[`implementation_ledger.md`](../implementation_ledger.md)

A model converter does more than put the same numbers in a different file. It
may precompute part of an equation, remove dimensions of size one, or rearrange
heads for a runtime's preferred memory order. Quartz must execute the numbers
that are actually in its pinned GGUF, not apply the original source equations to
already converted values a second time.

This chapter explains every conversion Quartz currently admits. The exact
llama.cpp revision, source-file hashes, and rules are machine-readable in
[`pins/gguf_conversion_contract.json`](../pins/gguf_conversion_contract.json).
The reversible layout code is in
[`src/conversion.cpp`](../src/conversion.cpp); the two gate conventions are
kept visibly separate in [`src/gdn.cpp`](../src/gdn.cpp).

## Three representations of one model

It helps to distinguish three things:

1. A **source checkpoint** stores parameters with names, shapes, and equations
   defined by the training framework.
2. A **converter** reads those parameters and writes a GGUF artifact. Some
   values and shapes change during this step.
3. The **runtime** reads the GGUF. It must know the converted meaning of every
   payload and must not pretend that the bytes still use source conventions.

The conversion is deterministic and happens before Quartz starts. Quartz does
not need the source checkpoint during inference, but the pinned converter is
part of the evidence that explains what its GGUF bytes mean.

## Folded GDN decay: why `A_log` becomes negative `A`

The source Gated Delta Network stores `A_log`. For each value head, the source
gate computes:

```text
source_A = -exp(A_log)
log_decay = source_A × softplus(alpha + dt_bias)
decay = exp(log_decay)
```

`exp` means the exponential function. It always returns a positive number, so
the leading minus sign makes `source_A` negative. `softplus(x) = log(1 +
exp(x))` is also positive. Therefore `log_decay` is negative and the final
`decay` lies between zero and one: old recurrent state fades instead of growing.

The converter calculates `-exp(A_log)` once and stores that result as
`ssm_a`. This is called **folding** because a fixed part of the runtime equation
has been folded into the saved parameter:

```text
GGUF stored_A = -exp(source A_log)
log_decay = GGUF stored_A × softplus(alpha + dt_bias)
```

Applying `-exp` again would be a serious double-conversion error. Quartz exposes
`gdn_gates_from_source` for oracle fixtures and `gdn_gates_from_gguf` for the
actual artifact. A test derives the folded values and requires both routes to
produce exactly the same little-endian FP32 bytes. The GGUF route also rejects a
zero, positive, NaN, or infinite stored A before changing output gates.

## RMSNorm offsets versus direct scales

RMSNorm first divides a vector by its root-mean-square magnitude. A learned
number then scales each lane. The ordinary source Qwen norm represents that
learned number as an offset:

```text
source output[i] = normalized[i] × (1 + source_weight[i])
```

The converter adds one and stores the finished scale:

```text
GGUF scale[i] = 1 + source_weight[i]
runtime output[i] = normalized[i] × GGUF scale[i]
```

For example, a source offset of `0.25` becomes a stored scale of `1.25`. Adding
one again at runtime would incorrectly use `2.25`. The functions `rms_norm` and
`rms_norm_scale` make the source-offset and GGUF-direct meanings explicit, and
the conversion diagnostic proves their results are bit-identical when the
weights are related by the converter rule.

Qwen's GDN gated norm is an important exception at the source boundary: its
`linear_attn.norm.weight` is already a direct scale, so the converter does not
add one. From Quartz's GGUF boundary both ordinary and GDN norm payloads are
therefore multiplied as direct scales, even though they arrived there by
different source rules.

## Squeezing the convolution dimension

The source depthwise convolution weight has shape `[channels, 1, width]`. The
middle dimension contains exactly one element, so it carries no choice. The
converter **squeezes** it—removes that size-one dimension—and writes a logical
`[channels, width]` matrix.

GGUF lists the fastest-changing physical dimension first, as Chapter 21
explains. The pinned layer-0 metadata is consequently `[4, 10240]`: each of
10,240 channels owns four contiguous FP32 coefficients. No values are discarded;
only a redundant shape axis disappears. Quartz's causal depthwise convolution
already accepts `channels × width` storage, so it does not reinsert the axis.

## Grouped heads versus tiled heads

The GDN has 16 key heads and 48 value heads. Each key head is reused by three
value heads. Two orders can describe those same 48 heads:

```text
source grouped: [key][replica within that key][lane]
GGUF tiled:      [replica][key][lane]
```

Consider two key heads, three replicas, and one lane. Give key 0's replicas the
labels `0, 1, 2` and key 1's replicas `100, 101, 102`:

```text
grouped = [0, 1, 2, 100, 101, 102]
tiled   = [0, 100, 1, 101, 2, 102]
```

Nothing is numerically transformed; only indices move. With lane width `W`, the
conversion implemented by Quartz is:

```text
tiled[(replica × key_heads + key) × W + lane]
  -> grouped[(key × replicas + replica) × W + lane]
```

The inverse uses the same pair of indices in the opposite direction. Tests use
two lanes per head so that accidentally moving individual lanes, rather than
whole heads, is also detected. The round trip must reproduce every original
FP32 bit. In-place conversion is rejected when there is more than one replica,
because writing an early destination could destroy an input still needed later.

The pinned converter applies tiled order to every value-associated part of the
GDN: V rows in the packed QKV projection, gate rows, alpha and beta rows, decay
and time-bias vector elements, V convolution channels, and input columns of the
output projection. Q and K remain key-head data and are not part of this
reordering.

Quartz's readable recurrent core uses grouped order because `value_head /
replicas` then identifies the reused key head directly. At the GGUF boundary,
projection results and parameter vectors move from tiled to grouped order before
recurrence. The recurrent output moves from grouped back to tiled before the
stored output projection. This is an explicit runtime-layout boundary, not a
change in model semantics.

## Real-artifact fixtures and “fixture equality”

A **fixture** is a small, checked-in expected result used by a repeatable test.
**Fixture equality** means a new run produced the same recorded value under the
fixture's stated comparison rule. It does not mean two mathematical methods are
universally identical.

[`fixtures/gguf_conversion.json`](../fixtures/gguf_conversion.json) freezes five
payloads from layer 0 of the exact admitted model: folded decay, time bias,
ordinary input norm, GDN norm, and convolution. It records the full-payload
SHA-256, first and last eight FP32 values as raw bytes, range, shape, and storage
size. Pytest rereads the model mapping and requires exact equality. It also
checks that all admitted folded decay values are negative and that convolution
metadata has the squeezed `[4, 10240]` shape.

These hashes prove the runtime is testing the intended bytes. The synthetic
gate, norm, and permutation cases prove what the code does with those
conventions. Together they are stronger than either alone: hashes cannot prove
an equation, and a correct equation tested on invented values cannot prove the
real model was bound correctly.

## Session and checkpoint ownership

Weights and fixed conversion rules belong to the engine. Token-dependent GDN
state belongs to a session. The proposed scheduler keeps recurrent state in the
semantic grouped order, while the mapped model stays in its immutable GGUF tiled
order. A future checkpoint must state its state-layout version and save every
semantic state element; restore must reject an incompatible layout rather than
guess. The checkpoint implementation remains SES-001 work and is not claimed by
this chapter.

## Evidence and proof boundary

**External:** the pinned llama.cpp converter and Qwen3.5 graph define the
GGUF-side transformations. Their file hashes are frozen in the conversion
contract and provenance ledger.

**Measured:** synthetic grouped/tiled conversion round-trips exactly; folded and
source gate paths match exact FP32 bytes; source-offset and stored-scale norm
paths match exact FP32 bytes; malformed folded decay fails closed; and the five
real parameter payloads match their frozen hashes.

CPU-007 establishes parameter semantics and reversible head layout. It does not
yet bind all tensor roles, run all 64 layers, or prove final logits. That remains
CPU-004. Keeping this boundary explicit prevents a conversion fixture from being
mistaken for a working inference engine.
