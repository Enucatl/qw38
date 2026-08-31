# 41. One CUDA GDN step and atomic state staging

[Index](README.md) · Implementation tasks: GDN-001 and EDU-027 in
[`implementation_ledger.md`](../implementation_ledger.md)

The quantized CUDA kernels in Chapters 39 and 40 produce learned projections.
GDN-001 begins immediately after that boundary and executes the stateful core of
one Gated Delta Network token on the RTX 5090. It adds a crucial rule: calculating
a possible next state is separate from accepting that state into the session.

## The one-token boundary

The input is one FP32 vector containing 10,240 projected channels. Width-four
**causal convolution** mixes each channel with its own three earlier values,
then SiLU activates the result. The channel layout is:

```text
2,048 query values | 2,048 key values | 6,144 value values
```

Those counts come from 16 query/key heads and 48 value heads, each 128 values
wide. Precomputed `log_decay` and `beta` values supply the two gates described
in [Chapter 18](18-gated-delta-network.md). GDN-001 deliberately starts after
projection and gate calculation so its state-mutation boundary stays small and
directly comparable with the admitted scalar functions.

## Two different kinds of state

Each GDN layer carries:

- 10,240 convolution histories, each four FP32 values; and
- 48 recurrent matrices, each `128 × 128` FP32 values.

The production counts are 40,960 convolution values and 786,432 recurrent
values. They depend on the token prefix. They are not immutable model weights
and cannot be shared between unrelated sessions.

The convolution kernel reads the old four-value history, shifts it logically,
appends the current projected value, and writes the new history elsewhere. The
recurrent kernel then follows the exact **delta rule** order:

1. normalize query and key and reuse one key head for three value heads;
2. decay the old recurrent matrix;
3. predict the value already represented by that state;
4. calculate the beta-scaled difference;
5. add the key-by-difference outer product; and
6. read the updated matrix with the normalized query.

One CUDA block owns one value head. One thread owns one value lane and walks all
key lanes in scalar order. This is a correctness-first layout; later profiling
may parallelize more of the matrix while preserving the frozen visible result.

## Committed state versus candidate state

**Committed state** represents the session prefix the user has actually
accepted. **Candidate state** represents what the next state would be if the
current token succeeds and is accepted.

[`launch_gdn_prepare`](../cuda/gdn_step.cu) requires different pointers for the
two. It reads committed convolution and recurrence, writes candidate buffers and
outputs, and never writes the committed buffers or position **frontier**. The
frontier is the number of committed tokens—the boundary between valid history
and work that has not been accepted.

After synchronization and error checking, the caller has two choices:

```text
accept token -> launch commit -> candidate becomes committed, frontier advances
cancel/error -> do not commit -> discard candidate, frontier stays unchanged
```

This prepare/commit split makes cancellation understandable. There is no
rollback algorithm: uncommitted work simply never becomes session state.

## Why commit uses one CUDA block

[`launch_gdn_commit`](../cuda/gdn_step.cu) launches one 256-thread block. Its
threads copy both candidate arrays, synchronize inside the block, and only then
does thread zero advance the frontier. A multi-block copy would need a separate
global synchronization before changing the frontier.

This is atomic at the focused GDN boundary: validation failure or cancellation
before commit leaves the old state/frontier visible, and a successful commit
publishes the exact candidate bytes before its new frontier. It is not yet the
whole product transaction. SES-002 must combine all 48 GDN layers, 16 attention
KV updates, the token, sampler data, and request cancellation into one session
commit policy.

## Failure injection and comparison

The device diagnostic deliberately passes the same state pointers as both
committed and candidate. The API rejects that alias before launching a kernel.
It then performs valid prepare work and copies the committed arrays and frontier
back to the host before commit. Every byte still equals the original state and
the frontier remains 41. This covers both invalid preparation and deliberate
cancellation. A later explicit commit must make committed bytes equal candidate
bytes exactly and advance the frontier to 42.

The small `2/6/8/8` case is easy to inspect. The production
`16/48/128/128` case covers the actual head reuse and 786,432-value recurrent
matrix. Both compare candidate convolution state, recurrent state, convolution
output, and recurrent output with the existing scalar authority.

The frozen CUDA admission limits are:

- maximum absolute error at `5e-8`;
- maximum RMS error at `5e-9`;
- zero NaN or infinity values;
- exact preservation during prepare/cancellation; and
- byte-exact candidate publication during commit.

Relative error remains report-only because a denominator near zero can magnify
an insignificant absolute difference. The shapes, protocol, limits, and local
source identities are pinned in
[`pins/cuda_gdn_contract.json`](../pins/cuda_gdn_contract.json).

## Measured evidence and limits

**Measured local:** both cases passed on the RTX 5090 with CUDA 13.0.2. Worst
absolute error was `3.7252903e-9`; aggregate RMS was below `2.76e-10`. The
production prepare averaged about `0.0205 ms` over 30 synchronized CUDA-event
samples after three warm-ups. [`fixtures/cuda_gdn_step.json`](../fixtures/cuda_gdn_step.json)
retains the exact diagnostic summary.

These timings exclude learned projection, gated RMSNorm, output projection,
FFN, allocator work, and the rest of the model, so they are not a layer or
tokens-per-second claim. GDN-001 proves the post-projection one-token state core
and its local publication protocol. Chunked 64-token scans, complete layers,
full-session atomicity, long-recurrence drift, and performance remain later
gates.
