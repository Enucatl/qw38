# 44. Memory-bounded CUDA attention prefill at 128K

[Index](README.md) · Implementation tasks: ATN-002 and EDU-030 in
[`implementation_ledger.md`](../implementation_ledger.md)

[Chapter 43](43-cuda-attention-decode.md) processed one new token. **Prefill**
processes a known prompt containing many tokens. The same causal rule applies:
prompt token 20 can use tokens 0 through 20, but not token 21.

ATN-002 extends the post-projection CUDA boundary to arbitrary positive chunks.
It prioritizes an inspectable state and memory contract. It is not yet the tuned
prefill kernel used for the final speed gate.

## Token-major input and output

Projected query, key, value, and gate arrays are **token-major**. All values for
token 0 come first, then all values for token 1. Output contexts follow the same
order:

```text
[token 0 query heads and lanes]
[token 1 query heads and lanes]
...
```

The normalization workspaces hold only the current token: 6,144 FP32 query
values and 1,024 FP32 key values at production shape. They are reused after each
token's kernels finish on the same CUDA stream.

## How causality crosses a chunk

Suppose the committed prefix ends at position 2 and a three-token chunk begins
there:

```text
token at position 2 reads committed 0..1 + candidate row 2
token at position 3 reads committed 0..1 + candidate rows 2..3
token at position 4 reads committed 0..1 + candidate rows 2..4
```

Later tokens must see earlier **candidate rows** even though none of those rows
is committed yet. [`grouped_attention`](../cuda/attention_decode.cu) chooses the
source by position: rows before `start_position` come from committed BF16 cache;
rows at or after it come from the candidate chunk.

This produces the same bytes as repeatedly preparing and committing one token,
while keeping the entire external chunk provisional.

## Memory-bounded does not mean constant memory

A naive prefill implementation may materialize a score rectangle with one row
for every prompt token and one column for every context token. For `T` new
tokens and context `C`, that consumes roughly `T × C` scores and becomes
quadratic when `T` and `C` grow together.

Quartz instead launches tokens in causal order and reuses one score slab:

```text
24 query heads × (start position + chunk tokens) FP32 scores
```

Memory therefore grows linearly with the largest context, not with
`chunk_tokens × context_tokens`. The current implementation still performs the
expected causal compute work, and its sequential token launches are explicitly
untuned. “Memory-bounded” describes the allocation shape, not fast execution.

## Whole-chunk prepare, commit, and cancellation

[`launch_attention_prepare_chunk`](../cuda/attention_decode.cu) fills every
candidate K/V row and token-major output while leaving the committed cache and
frontier unchanged. If any later layer fails or a request is cancelled, the
caller discards all candidate rows.

[`launch_attention_commit_chunk`](../cuda/attention_decode.cu) copies the
contiguous candidate range into its final cache positions. One block
synchronizes after copying both K and V, then advances the frontier. No partial
chunk becomes visible through that frontier.

Request-level atomicity still belongs to SES-002 because a real request must
publish GDN, all 16 attention caches, tokens, and sampler state together.

## What “128K” means here

The guaranteed context is 131,072 tokens, which is 128 × 1,024 rather than
128,000. Positions start at zero, so the last valid position is 131,071.

One production attention layer stores 1,024 BF16 keys and 1,024 BF16 values per
token:

```text
131,072 × 2,048 values × 2 bytes = 536,870,912 bytes = 512 MiB
```

The largest score slab is:

```text
24 heads × 131,072 positions × 4 bytes = 12,582,912 bytes = 12 MiB
```

All 16 attention layers' KV caches total 8 GiB. ATN-002 allocates and executes
one real 512 MiB layer at its final position. MEM-001 remains responsible for
showing that weights, all caches, GDN state, workspaces, graphs, allocator
overhead, and the 1.5 GiB reserve fit simultaneously.

## Evidence and the expensive final position

**Measured local:** 3-token and 9-token inspectable chunks plus a 9-token
production-shape chunk were byte-identical to repeated one-token CUDA execution.
Whole-chunk prepare preserved committed bytes/frontier, and whole-chunk commit
produced the same final cache and frontier as repeated commits.

After three warm-ups and 30 synchronized samples, the production nine-token
post-projection core averaged about `0.369 ms` at a short prefix. These timings
exclude learned projections, output projection, residual/FFN work, and graph
launch.

**Measured local:** the RTX 5090 successfully allocated the production
512 MiB one-layer cache plus the 12 MiB score workspace and executed position
131,071. The correctness-first final-position call took about `872 ms`, emitted
only finite values, preserved the frontier during prepare, committed frontier
131,072, and rejected positions or chunks beyond capacity. The long-context
time is a negative performance result to optimize later, not a release-speed
claim.

ATN-002 proves causal chunk continuity, exact token-wise equivalence, linear
score storage, whole-chunk state isolation, and the final legal one-layer
position. It does not prove complete-model memory fit, tuned long-context speed,
full-layer projections, or 64-layer scheduling.
