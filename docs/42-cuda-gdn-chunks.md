# 42. Chunked CUDA GDN prefill in 64-token windows

[Index](README.md) · Implementation tasks: GDN-002 and EDU-028 in
[`implementation_ledger.md`](../implementation_ledger.md)

[Chapter 41](41-cuda-gdn-step.md) prepared one token of GDN state. A prompt has
many tokens, and processing those known input tokens is called **prefill**.
GDN-002 accepts an arbitrary positive prompt chunk while retaining exactly the
same causal convolution and recurrent mutation order as repeated one-token work.

## Chunk input and output

The post-projection input is token-major:

```text
[token 0's 10,240 channels]
[token 1's 10,240 channels]
...
```

`log_decay` and `beta` are likewise stored as 48 values per token. The recurrent
output is token-major `[token_count, 6,144]`. “Token-major” makes every token's
complete value-head result contiguous and matches the scheduler's causal order.

[`launch_gdn_prepare_chunk`](../cuda/gdn_step.cu) rejects a zero token count,
invalid shape, null pointer, or aliased committed/candidate state before launch.
It does not change the committed frontier.

## Why split at 64 tokens

Quartz divides an external chunk into internal **64-token windows**:

```text
3 tokens   -> [3]
64 tokens  -> [64]
65 tokens  -> [64, 1]
129 tokens -> [64, 64, 1]
```

This gives later optimized scan kernels a fixed internal unit and bounds how
much per-launch sequential work one block owns. It does not reset state at a
window boundary. Window zero reads committed state and writes candidate state;
every later window continues from that same candidate.

The current implementation is a correctness-first sequential recurrence, not
an associative parallel scan. “Scan” here names the bounded state-carrying
window. A future parallel formulation must still compare with these exact
visible results before replacing it.

## Strict token order inside a window

GDN recurrence cannot process token 10 before token 9 because token 10 reads the
matrix token 9 produced. Inside each value-head block, Quartz therefore repeats:

```text
normalize token t query/key
decay state from token t-1
predict, calculate delta, update state
write token t output
continue with that updated state
```

The causal convolution follows the same **strict token order**. Each channel
loads its four committed or candidate history values into a small local ring,
advances the ring for every token in the window, writes each activated output,
and finally stores the ending candidate history.

## Whole-chunk candidate state

Internal windows are not separate transactions. The candidate after window one
is merely working state for window two. Even if 129 tokens require three
windows, committed recurrence, committed convolution, and the frontier remain
unchanged until the caller explicitly commits the final candidate.

Cancellation is therefore simple:

```text
prepare 129 tokens -> candidate only
cancel             -> discard candidate
committed prefix   -> unchanged
```

This avoids exposing a 64-token partial prefix when the larger requested chunk
fails or is cancelled. Whole-model session atomicity still belongs to SES-002,
which must combine GDN candidates with attention KV, tokens, and sampler state.

## Chunk equivalence test

The diagnostic evaluates every case twice on the GPU:

1. one `launch_gdn_prepare_chunk` call; and
2. repeated one-token prepare followed by one-token commit.

It compares every convolution output, recurrent output, final convolution state,
and final recurrent matrix byte-for-byte. This is stronger than comparing only
the last output. Counts 64 and 65 specifically reveal accidental resets or
off-by-one errors at the internal boundary; 129 crosses two boundaries.

Both GPU paths are also compared with the readable scalar recurrence. The
frozen limits remain maximum absolute error `5e-8`, maximum RMS error `5e-9`,
and zero non-finite values. Relative error remains report-only near zero. The
complete protocol is pinned in
[`pins/cuda_gdn_chunk_contract.json`](../pins/cuda_gdn_chunk_contract.json).

## Measured evidence and limits

**Measured local:** small-shape chunks of 3, 64, 65, and 129 tokens and a
production-shape 65-token chunk passed on the RTX 5090 with CUDA 13.0.2. Chunk
and repeated-token CUDA bytes were identical in all cases. Worst scalar/device
absolute error was `2.23517418e-8`; aggregate RMS remained below `1.79e-9`.

After three warm-ups, 30 synchronized samples measured about `0.100 ms` for the
small 65-token case and `0.500 ms` for the production-state 65-token core.
[`fixtures/cuda_gdn_chunk.json`](../fixtures/cuda_gdn_chunk.json) retains every
case. These measurements exclude projections, normalization/output projection,
FFN, attention, and scheduler overhead and are not full prefill throughput.

GDN-002 proves arbitrary chunk sizes, internal-window continuity, exact CUDA
token-wise equivalence, and whole-chunk candidate isolation. It does not yet
prove a parallel scan, complete GDN layers, the 64-layer scheduler, long-context
quality, tuned dispatch, or request-level atomicity.
