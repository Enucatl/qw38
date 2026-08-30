# 35. Multi-token scalar chunks

[Index](README.md) · Implementation tasks: CPU-004 and EDU-021 in
[`implementation_ledger.md`](../implementation_ledger.md)

A prompt normally contains many tokens. CPU-016 proved one token could cross all
64 layers, but a usable semantic oracle must also preserve history and positions
across a sequence. CPU-004 adds the narrow scalar chunk operation and proves that
changing the caller's chunk boundary does not change any result.

This is correctness infrastructure, not optimized prompt processing. CUDA's
layer-major, tiled prefill path remains later GDN-002/ATN-002 work.

## Token-wise execution and a chunk

**Token-wise execution** calls the one-token function repeatedly:

```text
execute token 42 at position 0
execute token 3649 at position 1
```

A **chunk** presents both tokens in one call:

```text
execute chunk [42, 3649] starting at position 0
```

[`execute_scalar_chunk`](../src/scalar_runtime.cpp) deliberately loops over the
same admitted one-token implementation. It does not introduce a second set of
model equations. The chunk boundary is a scheduling boundary: it controls how
many tokens the caller submits, not what the model computes.

## Why token order and positions matter

Token 3649 is the greedy result after token 42 in the frozen native regression.
When it runs second:

- every GDN layer reads recurrence and convolution state left by token 42;
- every attention layer reads the position-0 KV row and writes position 1;
- partial RoPE uses position 1 rather than position 0; and
- the session frontier advances from one to two.

Running the same token from a new zero state would be a different calculation.
Chunk equivalence therefore compares complete persistent state as well as
logits.

## Whole-chunk preflight

The chunk validates all caller-controlled structure before executing token 0:

- token pointer exists and the count is nonzero;
- model parameters, session state, and workspace have exact admitted sizes;
- the entire token count fits after the current frontier;
- `token_count * 248,320` cannot overflow `size_t`;
- output storage has exactly that many FP32 elements; and
- every token ID is smaller than the vocabulary size.

Validating every token first is important. If `[42, invalid]` executed token 42
before discovering the invalid second ID, the call would return failure with a
mutated session and one written logit row. The negative test requires frontier
zero, every state value still zero, `layers_completed = 0`, and every output
sentinel still NaN for invalid token, insufficient capacity, and short logits.

This preflight protects predictable caller errors. It is not full transaction
rollback for an unexpected arithmetic or trace-sink failure after execution has
begun; public atomic semantics remain SES-002 work.

## Logit row layout

Each input token produces every vocabulary score. The output is a flat row-major
array:

```text
row 0: logits after token 42       248,320 FP32 values
row 1: logits after token 3649     248,320 FP32 values
```

The stride is exactly 248,320 values. The score for token ID `v` after input row
`r` is at:

```text
logits[r * 248320 + v]
```

Retaining every row lets the oracle compare intermediate continuation logits,
not only the last token. Production generation can later request or consume one
row at a time to reduce host storage.

## Exact equivalence experiment

[`check_real_scalar_chunk`](../src/eval.cpp) creates two independent capacity-2
sessions sharing one immutable prepared model:

```text
session A: execute one chunk [42, 3649]
session B: execute token 42, then execute token 3649
```

After both paths finish, it requires exact FP32 equality for:

- all 496,640 logits;
- all 48 GDN convolution rings;
- all 48 GDN recurrent matrices;
- every row in all 16 attention key caches;
- every row in all 16 attention value caches; and
- the committed frontier.

No tolerance is used because both paths invoke the same scalar operations in the
same order. A difference would identify a scheduling, position, stride, or state-
ownership bug rather than normal cross-backend rounding.

Both workspaces report 64 completed layers after their last token. That counter
describes the most recent token, not a cumulative 128-layer total.

## Frozen structural fixture

[`fixtures/real_scalar_chunk.json`](../fixtures/real_scalar_chunk.json) stores
the two input IDs, five selected logits per row, both greedy IDs, frontier,
stride, and exact-equivalence result. It labels itself a **native structural
two-token scalar chunk equivalence** fixture.

The first row again chooses token 3649. After consuming token 3649, the second
row chooses token 1277. At CPU-004 these IDs were useful regression identifiers,
not an admitted model answer. ORA-001 subsequently compared the same rendered
inputs and traces with independent authorities.

## Measured cost

**Measured:** the clean equivalence diagnostic executed four complete scalar
tokens—two in the chunk session and two in the repeated-token session—in 87.08
seconds. Maximum resident host memory was 18,172,800 KiB. Most of that is mapped
GGUF pages touched by scalar matvec; the two explicit capacity-2 state owners
also contribute roughly twice the scalar session allocation.

This is intentionally slow, inspectable oracle work. It is not a prediction of
CUDA prompt throughput.

## Arbitrary chunk boundaries

The exact test establishes the fundamental boundary:

```text
chunk length 2 == chunk length 1 followed by chunk length 1
```

Because each successful call starts at the session's current frontier and the
implementation accepts any positive count that fits, the same invariant applies
to partitions such as `[2, 1, 5]` versus `[1, 1, 1, 1, 1, 1, 1, 1]`. Later
quality fixtures will use longer partitions to detect recurrence drift; the
production optimized prefill paths must compare against this scalar definition.

## Proof boundary

**Measured:** two real tokens in one chunk exactly match two repeated one-token
calls across every output logit, every persistent state element, and frontier;
three whole-chunk preflight failures leave state and outputs untouched.

**Subsequently measured:** ORA-001 supplied independent full-model semantic
traces, per-tap tolerances, and greedy continuation admission with no near-tie
exception. CPU-004 by itself still proves only local scalar chunk invariance;
the external agreement is documented in
[Chapter 38](38-scalar-authority-tolerances.md).
