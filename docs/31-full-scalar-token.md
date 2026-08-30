# 31. One token through the full scalar model

[Index](README.md) · Implementation tasks: CPU-016 and EDU-017 in
[`implementation_ledger.md`](../implementation_ledger.md)

CPU-016 is the first Quartz path that connects every major scalar component:

```text
token 42
  -> Q4_K embedding
  -> layers 0 through 63 in exact hybrid order
  -> final direct-scale RMSNorm
  -> all 248,320 FP32 logits
```

The real pinned weights and production dimensions are used throughout. The run
starts with zero persistent state and a context capacity of one token.

This is a structural full-model milestone, not yet a semantic continuation
authority. A stable native result proves that Quartz consistently executes its
own admitted components in the intended order. TRC-001, TRC-002, and ORA-001
must still compare full traces and logits with pinned Transformers and an
independent same-GGUF implementation.

## The exact hybrid schedule

Qwen3.8-27B has 64 decoder layers. Every fourth layer, starting at layer 3, uses
full attention:

```text
0 GDN       1 GDN       2 GDN       3 attention
4 GDN       5 GDN       6 GDN       7 attention
...
60 GDN     61 GDN      62 GDN      63 attention
```

The rule is `layer % 4 == 3`. It produces exactly 48 GDN layers and 16 attention
layers. This order is model semantics, not a performance choice. Swapping an
attention and GDN layer would apply different weights and read or mutate the
wrong kind of history.

[`prepare_scalar_model_parameters`](../src/scalar_runtime.cpp) checks every
typed layer against this rule before execution. The token loop checks it again
before any state mutation.

## Physical layer numbers versus state slots

Physical layer numbers always range from 0 to 63, but the two state families
have their own compact slot numbers:

```text
physical layers: 0  1  2  3  4  5  6  7 ... 63
kind:            G  G  G  A  G  G  G  A ...  A
GDN slot:        0  1  2  -  3  4  5  - ...  -
attention slot:  -  -  -  0  -  -  -  1 ... 15
```

Quartz exposes arrays indexed by physical layer so the execution loop cannot
lose its place. Each valid entry points into the corresponding compact storage
slab. A GDN layer view can only contain convolution and recurrent state; an
attention layer view can only contain K and V cache. This keeps state ownership
explicit without inheritance or a generic tensor registry.

## Engine-owned prepared parameters

Mapped quantized matrices remain immutable model views. Small F32 parameters
that the scalar equations use repeatedly are decoded once:

- 64 mixer input norms and 64 FFN norms;
- 48 GDN convolution kernels, folded-A vectors, time biases, and head norms;
- 16 attention query and key head norms; and
- one final output norm.

Together these contain 2,645,504 FP32 values, or 10,582,016 bytes
(10.091796875 MiB). They belong to the prepared engine, not to a session. Two
sessions using the same model should share these immutable values.

The prepared structures contain pointers into contiguous owning vectors. Moving
the owner transfers those vectors without decoding or copying the 18.97 GB GGUF.
Copying is disabled because copied internal pointers would otherwise refer to
the wrong owner's storage.

## Session-owned persistent state

The capacity-one scalar session allocates:

| State | FP32 values |
|---|---:|
| 48 convolution rings | 1,966,080 |
| 48 recurrent matrices | 37,748,736 |
| 16 one-token key rows | 16,384 |
| 16 one-token value rows | 16,384 |
| **Total** | **39,747,584** |

That is 158,990,336 bytes, or 151.625 MiB. The GDN state size does not grow with
context. The scalar KV rows do grow with capacity and use FP32 for inspection.
Production V1 instead stores two bytes per KV value, yielding the 8 GiB 128K
allocation explained in Chapter 28.

All 48 GDN slots and all 16 attention slots change during the admitted token-42
run. The diagnostic checks complete state vectors for finite values and records
selected state taps from early, middle, and final slots.

## Shared scratch instead of 64 workspaces

Layers execute sequentially for one token, so layer 1 no longer needs its
temporary values after producing its output. Quartz allocates one readable
workspace for the largest GDN branch, one for attention, and one shared FFN
workspace, then reuses them across all matching layers.

The scalar workspace contains 204,161 FP32 values, or 816,644 bytes
(about 797.504 KiB). It includes:

- two 5,120-value activation buffers;
- one 5,120-value post-mixer residual;
- complete GDN, attention, and FFN intermediate buffers;
- capacity-one attention score scratch; and
- the final normalized hidden vector.

The 970 KiB FP32 logits buffer belongs to the caller and is not included in that
workspace count. Persistent session state is also separate.

This reuse is a lifetime decision, not a fusion. Every intermediate still has a
named buffer and remains inspectable. CUDA can later reuse or fuse more memory
only after trace equivalence is frozen.

## Ping-pong residual buffers

Each layer needs an input and a distinct output because residual additions still
read the original layer input. Allocating 64 separate 5,120-value outputs would
be unnecessary. Quartz alternates two buffers:

```text
embedding -> A
layer 0: read A, write B
layer 1: read B, write A
layer 2: read A, write B
...
layer 63: read B, write A
```

This is called ping-pong or double buffering. Because there are an even 64
layers, the final hidden vector resides in A. The implementation still swaps
input/output pointers after every successful layer rather than relying on that
fact inside the loop.

Within each layer, the separate post-mixer buffer preserves the two-residual
order established in Chapter 29.

## One-token execution sequence

[`execute_scalar_token`](../src/scalar_runtime.cpp) performs:

1. validate global parameter, state, workspace, vocabulary, capacity, schedule,
   logits, and frontier contracts;
2. decode token 42 into activation A;
3. execute each physical layer from 0 through 63, choosing its explicit GDN or
   attention entry point;
4. count a layer complete only after both its mixer and FFN succeed;
5. apply final norm to the layer-63 output;
6. compute all 248,320 logits; and
7. advance the session frontier from 0 to 1 only after logits succeed.

At position zero each attention layer appends its first K/V row and attends to
that one row. GDN layers update their initially zero convolution rings and
recurrent matrices.

## What a frontier means

The frontier is the number of token positions the public session considers
committed. A capacity-one state begins at frontier 0. After a successful full
token, it becomes 1 and no second token fits.

CPU-016 advances the frontier only after final logits succeed. That prevents an
ordinary success from being exposed early. It does not yet provide transaction
rollback: an unexpected failure at layer 40 could leave layers 0–39 mutated even
though the frontier remains 0. SES-002 must stage state or restore the old
version before this runtime backs public `Session::eval`.

## Global preflight failure

Layer-local validation is too late for global caller mistakes. If the shared FFN
arena were short, discovering that at layer 0 would still be safe, but a malformed
attention arena discovered at layer 3 could follow three GDN mutations.

The scalar runtime therefore validates the exact size of every owning parameter,
state, and scratch vector before embedding lookup. The negative test removes one
FFN activation value. The call returns with:

```text
layers completed = 0
frontier = 0
all GDN and attention state = unchanged zero
all logits = untouched NaN sentinels
```

This proves deterministic structural failure before mutation. It does not widen
the later session atomicity claim.

## Structural fixture and its limit

[`tools/generate_real_scalar_token_fixture.py`](../tools/generate_real_scalar_token_fixture.py)
runs the complete native path and freezes selected final-hidden, final-norm,
logit, GDN-state, and attention-state taps plus exact counts in
[`fixtures/real_scalar_token.json`](../fixtures/real_scalar_token.json).

The fixture labels itself a **native structural 64-layer zero-state scalar
regression**. Exact equality catches changed layer order, wrong state-slot
mapping, buffer handoff errors, altered accumulation, or an accidental artifact
change. It cannot show that Quartz and Transformers agree if both have never
been compared at the same full-model taps.

The native run produced greedy token 3,649 for this artificial single token.
That value is useful as a regression identifier but is not an admitted model
answer. Quality, NLL, and continuation claims remain prohibited until the oracle
gate passes.

## Measured execution

**Measured:** the first host run completed all 64 layers and logits in 36.21
seconds with 18,013,440 KiB maximum RSS. A warm fixture-capture run completed in
22.81 seconds with the same maximum RSS. Much of that RSS is mapped model pages
touched by scalar row traversal; it is not the 151.625 MiB explicit session
state alone.

These measurements characterize an inspectable scalar oracle, not the intended
CUDA performance. The production target requires GPU quantized kernels and
resident weights.

## Proof boundary

**External:** the official contract fixes 64 physical layers, the 3-GDN/1-
attention repeating schedule, embedding/final boundaries, and state families.

**Measured:** one real token completes all 64 layers; exactly 48 GDN and 16
attention slots mutate; frontier advances once; complete final logits are finite;
selected native boundary/state/logit taps are stable; malformed global scratch
fails before any layer or state change.

**Estimated:** prepared, state, workspace, and logit byte counts are direct sums
of the exact owning vector lengths.

CPU-016 does not close CPU-004. It still lacks a direct semantic-authority trace,
multiple-token continuation equality, arbitrary chunked prefill, and token-wise
versus chunked comparison. Those are the next scalar trace/oracle increments
before CUDA admission.

**Status update (2026-08-30):** CPU-004 and ORA-001 subsequently closed those
scalar requirements. [Chapter 38](38-scalar-authority-tolerances.md) freezes the
resulting pre-CUDA numeric gates.
