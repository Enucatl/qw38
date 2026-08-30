# 29. Composing a complete decoder layer

[Index](README.md) · Implementation tasks: CPU-014 and EDU-015 in
[`implementation_ledger.md`](../implementation_ledger.md)

The previous chapters proved each large branch separately. This chapter joins
them in the order the model actually uses. [`scheduler.cpp`](../src/scheduler.cpp)
now executes complete production-shaped layer 0 (GDN plus FFN) and layer 3
(attention plus FFN).

“Complete layer” is narrower than “complete model.” A layer transforms one
5,120-value token activation. The model still needs an embedding before layer
0, all 64 layers in schedule order, a final normalization, and a vocabulary
projection after layer 63.

## The two residual branches

Let `x0` be the activation entering a decoder layer. The exact branch order is:

```text
mixer_input = RMSNorm_1(x0)
mixer_correction = mixer(mixer_input, state)
x1 = x0 + mixer_correction

ffn_input = RMSNorm_2(x1)
ffn_correction = down(SiLU(gate(ffn_input)) * up(ffn_input))
x2 = x1 + ffn_correction
```

`x2` is the layer output and becomes the next layer's `x0`.

A residual connection is the `old value + learned correction` pattern. There
are two of them, not one. The mixer does not replace the activation, and the FFN
does not add its correction directly to the original `x0`.

That last distinction is the purpose of CPU-014. Running two individually
correct branches in the wrong order would still produce finite numbers, but it
would be a different neural network.

## Why there are two normalizations

The first learned RMSNorm belongs to the mixer branch. The second, stored as
`post_attention_norm.weight`, belongs to the FFN branch. Despite that historical
tensor name, GDN layers also use it after their mixer.

Both are **pre-norms**: each normalization happens before the branch it feeds.
The second norm must measure `x1`, which already includes the mixer correction:

```text
correct: RMSNorm_2(x0 + mixer_correction)
wrong:   RMSNorm_2(x0)
```

RMSNorm depends on the mean square of every lane, so even a correction in one
place can change the scale applied across the whole vector. CPU-014 retains the
full 5,120-value `x1` buffer; it does not try to reconstruct the second norm from
a few displayed taps.

## One scheduler boundary, two mixer variants

Most of the layer is shared:

```text
input -> mixer variant -> post-mixer residual -> shared SwiGLU FFN -> output
```

The mixer variant is selected by the pinned layer schedule:

- layers whose index is not `3 mod 4` use GDN and persistent convolution plus
  recurrent state;
- layers 3, 7, 11, ..., 63 use grouped-query attention and persistent KV state.

Quartz keeps two explicit entry points—`execute_gdn_layer_step` and
`execute_attention_layer_step`—instead of hiding state layouts behind
inheritance or a generic operator registry. The caller therefore cannot pass a
KV cache where recurrent matrices are required. Both entry points converge only
at the shared FFN function after producing a 5,120-value post-mixer residual.

## Parameters, state, workspace, and output

Four memory lifetimes meet at this boundary:

**Parameters** are immutable learned values prepared for a layer. A scalar GDN
layer prepares 51,424 FP32 values (200.875 KiB) including both mixer and FFN
norm-related parameters. An attention layer prepares 10,752 values (42 KiB).
The large quantized matrices remain mapped views rather than decoded copies.

**Persistent state** summarizes earlier tokens. A GDN layer owns its convolution
ring and recurrent matrices. An attention layer owns its K/V timeline. The FFN
owns no persistent state.

**Workspace** holds temporary values for the current call. CPU-014 uses a
separate 5,120-value post-mixer buffer so the second norm has an unambiguous
input. At the two-token attention diagnostic capacity, the explicit totals are:

| Complete layer | FP32 values | Bytes | KiB |
|---|---:|---:|---:|
| GDN mixer + post-mixer + FFN | 145,792 | 583,168 | 569.5 |
| Attention mixer + post-mixer + FFN | 110,594 | 442,376 | about 432.01 |

These totals describe the readable scalar implementation. A later CUDA
scheduler can reuse compatible buffers and fuse boundaries, but its visible
results must remain within the already frozen tolerances.

**Output** is the final 5,120-value `x2`. It is not persistent model state by
itself; the scheduler passes it to the next layer.

## Validation before state mutation

A layer wrapper could encounter an invalid FFN buffer only after its mixer had
already appended KV or updated GDN state. That would turn a simple caller error
into a partially mutated session.

CPU-014 therefore separates FFN validation from FFN execution:

```text
validate post-mixer, FFN workspace, and final output
execute and mutate the mixer
execute the already-validated FFN
```

The negative fixture shortens the FFN activated buffer by one FP32 value. Both
layer variants reject it before the mixer runs. Tests prove that convolution,
recurrent, or KV state and the final output remain untouched.

This preflight handles deterministic pointer and count errors. It is not a full
transaction. If a later admitted matrix operation unexpectedly failed after a
mixer mutation, CPU-014 would not restore the old state. SES-002 must stage or
roll back persistent state before advancing the public session frontier.

## What the composition fixture proves

[`tools/generate_real_layer_composition_fixtures.py`](../tools/generate_real_layer_composition_fixtures.py)
runs the native layer wrappers and freezes selected boundary taps in
[`fixtures/real_layer_composition.json`](../fixtures/real_layer_composition.json).
The fixture deliberately calls itself a **native composition regression**, not
an independent semantic authority.

[`tests/test_real_layer_composition.py`](../tests/test_real_layer_composition.py)
checks three separate properties for layer 0 and layer 3:

1. Frozen post-mixer, FFN norm, gate, up, activation, correction, and final taps
   have not changed unexpectedly.
2. The wrapper's post-mixer taps are bit-for-bit identical to the output of the
   already admitted standalone mixer diagnostic.
3. The final taps equal `FP32(post_mixer + FFN correction)` exactly.

The first check catches regressions. The second proves the handoff source. The
third proves the second residual target. Earlier independent GGUF fixtures
remain the numeric evidence for each underlying mixer and selected FFN rows.
A direct Transformers full-layer trace remains TRC-001/TRC-002/ORA-001 work.

## A worked four-number analogy

The production vector has 5,120 values, but the ordering is easier to see with
four. Suppose:

```text
x0                 = [10, 20, 30, 40]
mixer correction   = [ 1, -2,  3, -4]
x1                 = [11, 18, 33, 36]
FFN correction     = [-1,  5,  0,  2]
x2                 = [10, 23, 33, 38]
```

The second norm and FFN must inspect `[11, 18, 33, 36]`. Feeding `[10, 20, 30,
40]` instead would ignore the mixer when deciding the FFN correction. Adding the
FFN correction to `x0` would produce `[9, 25, 30, 42]`, also wrong.

## Proof boundary

**External:** pinned Transformers defines pre-norm mixer, first residual,
post-mixer pre-norm FFN, and second residual order for both layer variants.

**Measured:** complete real layer-0 and layer-3 calls execute finite full
workspaces; their post-mixer handoffs equal the admitted standalone mixer
results; their final selected lanes obey exact FP32 FFN residual addition; and
malformed FFN storage fails before persistent state mutation.

**Estimated:** scalar parameter and workspace byte totals are direct arithmetic
from the explicit production-sized buffers above.

CPU-014 still does not prove cross-layer scheduling, token embedding, all 48 GDN
and 16 attention state slots, final RMSNorm, vocabulary logits, multi-token
chunking, or atomic session commit. Those remain CPU-004 and later gates.
