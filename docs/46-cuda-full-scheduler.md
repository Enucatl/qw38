# 46. The first complete CUDA token

[Index](README.md) · Implementation tasks: SCH-001 and EDU-032 in
[`implementation_ledger.md`](../implementation_ledger.md)

Earlier chapters tested one operation at a time. SCH-001 is the first time a
real token enters as an integer ID, crosses every model layer on the GPU, and
leaves as 248,320 FP32 logits. It is a correctness-first decode scheduler, not
yet the transactional, graphed, or tuned runtime.

## From a mapped file to resident weights

“Resident” means the weights remain in GPU memory between tokens. The host first
memory-maps and validates the exact 18,973,870,432-byte GGUF. A memory map lets
the operating system expose file bytes at virtual addresses without Quartz
copying the file into a second host buffer.

[`ResidentModel::upload`](../cuda/full_scheduler.cu) allocates one equally sized
device byte range and copies the canonical file into it. Why copy metadata and
padding too? It gives every admitted tensor a simple checked translation:

```text
host tensor offset = host tensor address - mapped-file base
device tensor address = device-file base + host tensor offset
```

The offset and complete tensor byte range must fit the mapped file. All 851
typed tensor views are remapped before execution; a bad type or range fails the
upload. This one-allocation representation is easy to inspect and preserves the
authenticated row bytes. A later repack may be faster or smaller, but would need
its own versioned cache and correctness gate.

## The 64-layer hybrid clockwork

Qwen3.8-27B alternates by a fixed rule rather than reading a runtime operator
list:

```text
layers 0, 1, 2: GDN
layer 3:         attention
layers 4, 5, 6: GDN
layer 7:         attention
... repeat sixteen times ...
layer 63:        attention
```

That produces exactly 48 GDN and 16 attention layers. Every layer also contains
one SwiGLU feed-forward branch. The scheduler in
[`execute_token`](../cuda/full_scheduler.cu) follows this literal pattern:

```text
token ID
  -> decode quantized embedding row
  -> repeat 64 times:
       input RMSNorm
       GDN or attention mixer
       mixer residual addition
       FFN RMSNorm
       gate/up projections and SwiGLU
       down projection and FFN residual addition
  -> final RMSNorm
  -> 248,320-row output projection
  -> FP32 logits
```

There is no generic tensor graph, dynamic dispatch registry, or inheritance
hierarchy. Each named projection calls the already admitted quantized kernel;
each stateful mixer calls its admitted GDN or attention boundary.

## Storage is not the same as accumulation

Low precision is useful for values transported into large matrix operations,
but repeated residual additions are numerically sensitive. The admitted layout
therefore has explicit roles:

- the master residual accumulator is FP32;
- normalized projection inputs, GDN/attention output-projection inputs, and the
  17,408-value SwiGLU activation are stored as BF16;
- projection outputs, GDN recurrence, attention math, and final logits are
  FP32;
- attention K/V history is BF16;
- Q4_K and Q6_K matrix inputs use the CUD-001 temporary Q8 staging;
- resident Q8_0 matrices consume BF16 directly, avoiding a second eight-bit
  rounding of an already eight-bit weight.

The FP32 residual is an accumulator, not a hidden extra copy of every layer's
activations. One 5,120-value buffer carries the current residual and another
carries the candidate update; the scheduler reuses them throughout all layers.

This boundary was chosen from a recorded failure, not hidden experimentation.
The first complete version rounded the residual to BF16 after every mixer and
FFN and also quantized Q8_0 inputs to temporary Q8. It still chose the correct
tokens, but token 0 measured `0.230289` maximum and `0.0506235` RMS logit error,
outside the immutable `0.21` and `0.037` limits. No limit was widened. Direct
BF16 input for Q8_0 improved the result; retaining the sensitive residual sum in
FP32 brought every scalar-device boundary inside its pre-CUDA gate.

## Session state and reusable scratch

[`SchedulerSession`](../cuda/full_scheduler.h) owns prefix-dependent memory:

- 48 convolution rings;
- 48 FP32 recurrent matrices;
- 16 BF16 key caches and 16 BF16 value caches;
- a token-position frontier.

Layer number is not a state-array index. Separate `gdn_slot` and
`attention_slot` counters translate the hybrid schedule into dense state
arrays. For example, layer 3 uses attention slot 0 and layer 7 uses attention
slot 1. This avoids wasting 64-layer arrays for both state kinds.

[`SchedulerWorkspace`](../cuda/full_scheduler.h) owns values that can be reused
after a layer: normalized BF16 input, projection outputs, transient Q8 blocks,
candidate GDN state, one candidate K/V row, attention scores, and logits. At
capacity two it measured 4,769,472 bytes. Session state measured 158,990,336
bytes, almost all of it the 48 persistent GDN matrices.

This scheduler publishes each layer's candidate state as it proceeds. A failure
late in a token can therefore leave earlier layers changed. That limitation is
intentional and visible: SES-002 will add request-level candidate ownership and
one atomic commit across all state. SCH-001 proves execution order and values,
not rollback.

## What logits and greedy continuation mean

A **logit** is one unnormalized FP32 score for a vocabulary token. The output
matrix has 248,320 rows, so one decode step produces 248,320 logits. Greedy
sampling selects the index with the largest score. It does not apply
temperature, top-p, or randomness.

The test runs token 42 from zero state. Its largest logit selects token 3,649.
It then evaluates token 3,649 using the mutated recurrence and KV state; the
next largest logit selects token 1,277. Matching the second choice matters: it
proves the first token's hybrid state was used, not merely that a stateless
forward pass happened twice.

## Full-row and trace-tap comparisons

[`full_scheduler_test.cu`](../cuda/full_scheduler_test.cu) compares all 248,320
logits for both positions with the scalar oracle. It also compares all 5,120
values at layer residuals 0, 3, and 63 and at final norm. Those taps cover an
early GDN result, the first attention result, the last layer, and the final
projection input.

For each array the diagnostic reports:

- maximum absolute error: the largest raw distance;
- maximum relative error: distance divided by the reference magnitude;
- RMS error: a size-independent average error magnitude;
- cosine similarity: whether both complete vectors point in nearly the same
  direction;
- non-finite count: NaN or infinity values;
- first index beyond the frozen absolute limit;
- full-row greedy equality.

Relative error can look enormous when a correct reference value is almost zero,
so it is reported but not used alone for admission. Absolute, RMS, cosine,
finite-count, and greedy requirements all remain mandatory.

**Measured local:** both complete CUDA logit rows passed the immutable scalar
envelope. Token 0 measured `0.138223` maximum, `0.0272043` RMS, and `0.999872`
cosine. Token 1 measured `0.161953`, `0.0321326`, and `0.999864`. Every selected
residual/final-norm tap passed its frozen layer-specific limits, and both greedy
tokens matched exactly.

**Measured negative:** comparison with the pinned same-GGUF llama.cpp rows kept
the same greedy tokens and no non-finite values. Token 1 stayed inside absolute
and RMS limits but measured `0.999828237` cosine, about `0.0000028` below the
scalar-derived independent envelope. This result is retained and does not alter
the scalar-device admission. It remains useful evidence for later full quality
evaluation rather than being rounded away or mislabeled as a pass.

## Timing, memory, and proof boundary

**Measured local:** canonical upload took about `1.56 s`. The two correctness
decode calls took about `60.9 ms` and `60.4 ms`; the scalar reference took about
42.7 seconds for both. These are synchronized component measurements with
diagnostic tap copies, no CUDA graph, and no tuning, so they are not release
throughput claims.

After SES-002 added its reusable all-layer transaction candidate, a two-token
session left about 13.841 GB of 33.671 GB free after the resident model, state,
and 160.38 MB workspace allocations. This does not prove 128K fit:
the 8 GiB full attention cache, graph allocations, allocator overhead, and
required 1.5 GiB reserve must coexist in MEM-001.

The authenticated boundary is
[`cuda_scheduler_contract.json`](../pins/cuda_scheduler_contract.json), and the
retained measurements and failures are in
[`cuda_full_scheduler.json`](../fixtures/cuda_full_scheduler.json).

SCH-001 proves resident typed-weight remapping, the exact 48/16 schedule,
two-position state continuation, selected trace boundaries, full FP32 logits,
and greedy equality. It does not yet prove atomic failure, common-prefix reuse,
checkpointing, 128K simultaneous allocation, chunked full-model prefill, graph
capture, server behavior, quality-suite results, or competitive speed.
