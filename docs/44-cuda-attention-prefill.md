# 44. Memory-bounded CUDA attention prefill at 128K

[Index](README.md) · Implementation tasks: ATN-002 and EDU-030 in
[`implementation_ledger.md`](../implementation_ledger.md)

[Chapter 43](43-cuda-attention-decode.md) processed one new token. **Prefill**
processes a known prompt containing many tokens. The same causal rule applies:
prompt token 20 can use tokens 0 through 20, but not token 21.

ATN-002 extends the post-projection CUDA boundary to arbitrary positive chunks.
It prioritizes an inspectable state and memory contract. OPT-005 replaces its
production per-row launch loop with a fixed-memory tiled path; the former loop
remains available as a test-only reference for differential measurements.

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

Quartz's OPT-005 attention grid streams contexts through compile-time 32-row KV
tiles. It keeps each tile's BF16 K/V and FP32 scores in shared memory and carries
running maximum, denominator, and weighted-value state in FP32. When a new tile
has a larger maximum, the previous state is rescaled before accumulation. No
prompt-sized score rectangle is allocated, and the production multi-row path
leaves global score scratch byte-for-byte untouched.

The production launch has exactly two kernel launches per positive chunk: KV
staging on a `(kv_head, token)` grid, followed by grouped attention on a
`(kv_head, ceil(token_count / 2))` grid. This is launch topology evidence, not
a complete model or end-to-end speed result.

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

## OPT-005 through OPT-007 tiled attention evidence

The implementation preserves strict causal visibility: committed rows are used
below `start_position`, candidate rows at or above it, and each query row masks
positions after its absolute position. Candidate rows and the frontier remain
provisional until the existing chunk commit. OPT-006 changed block ownership
from one query head to one KV head: with 24 query heads and four KV heads, it
handles the six query heads `kv_head * 6 .. kv_head * 6 + 5`. It loads each
causal 32-row BF16 K/V tile into shared memory once and consumes that tile for
all six mapped query heads.

OPT-007 extends that owner to `(KV head, query-row tile)`, with exactly two
consecutive query rows per attention block. Block `y` owns relative rows
`2*y` and `2*y + 1`; an odd-sized chunk activates only the first row of its
last block. The block stages each K/V tile through the later owned row, then
admits a context to a row only when the context is at or before that row's
absolute position. Thus the earlier row may see a staged later context but
never incorporates it. Each row keeps its own query, maximum, denominator, and
value accumulator, while its normalization/RoPE, lane-order dot reduction,
online-softmax update, and gated output retain the one-row order.

The fixed shared layout remains 33,792 dynamic bytes: FP32 scratch plus
32-by-256 BF16 K/V tiles. The row state is compile-time bounded, so it does not
grow with prompt length; global score scratch remains untouched. The measured
fixture captured two kernel nodes for every 1/2/3/63/64/65-row case, staging
grids/blocks `[4, rows, 1]` / `[256, 1, 1]`, and attention grids/blocks
`[4, ceil(rows / 2), 1]` / `[256, 1, 1]`. On its
pinned RTX 5090 record, the attention kernel used 38 registers, 16 static
shared bytes, 224 local bytes per thread, and 33,792 launch dynamic shared
bytes; CUDA reported two active blocks per SM (170 SMs) for the 64-row capture.

The same record requires byte-exact production versus retained one-row output
and candidate BF16 K/V for all six row cases, finite output, prepare isolation,
untouched score scratch, and invalid-input rejection. It also requires a
65-row prepare/commit sequence to be byte-exact to 64 rows followed by one row
for output, committed cache, and final frontier. This is component-only exact
semantic and launch evidence: it makes no throughput, speedup, end-to-end, or
complete-model memory claim. The contract and retained measured record are
[`pins/cuda_query_row_attention_contract.json`](../pins/cuda_query_row_attention_contract.json)
and [`fixtures/cuda_query_row_attention.json`](../fixtures/cuda_query_row_attention.json).

The regenerated OPT-005 fixture records finite 3-row output with
`max_abs=8.94069672e-08`, `rms=1.06907114e-08`, and cosine `1`, and finite
9-row output with `max_abs=1.1920929e-07`, `rms=1.41810235e-08`, and cosine
`1`. Captured production graphs contain two kernel nodes for 1, 3, 9, and 64
rows; the retained reference contains 3, 9, 27, and 192 nodes respectively.

On the pinned RTX 5090, the fixture's tiled/reference means at 2,048, 8,192,
and 32,768 committed-prefix rows are `14.7988598/466.828623 ms`,
`58.3420746/1843.24634 ms`, and `267.752901/13864.1842 ms`; the corresponding
speedups are `31.5449048x`, `31.5937743x`, and `51.7797723x`. The fixture
contains 30 tiled and three retained-reference samples for each case. These are
post-projection, production-shape component measurements only: they exclude
projections, scheduler work, and end-to-end recovery.

OPT-006's separate pinned RTX 5090 fixture compares the grouped production
kernel with the retained per-query-head tiled diagnostic. Their production-GQA
outputs are byte-exact; candidate BF16 rows, prepare/commit isolation,
causality, scratch preservation, invalid-input rejection, and the two-node
production graph also pass. Against the untiled reference, the maximum absolute
errors for 1, 3, 9, and 64 rows are `5.96046448e-08`, `1.1920929e-07`,
`1.49011612e-07`, and `2.08616257e-07`; the corresponding RMS errors are
`3.97332123e-09`, `1.1860859e-08`, `1.50660302e-08`, and
`1.23316877e-08`, with cosine `1` in every case.

For traffic evidence, instrumented diagnostic specializations increment a
device counter alongside every global BF16 K and V source load, then atomically
publish the block total. For a 64-row chunk, `contexts = sum(start + token + 1)`
and the expected requested values are
`24 * contexts * 256 * 2` for retained per-query tiles versus
`4 * contexts * 256 * 2` for grouped tiles. At prefixes 2,048, 8,192, and
32,768, the measured retained/grouped counts are
`1,636,171,776/272,695,296`, `6,468,009,984/1,078,001,664`, and
`25,795,362,816/4,299,227,136` values; multiplying by the two-byte BF16
element size gives `3,272,343,552/545,390,592`,
`12,936,019,968/2,156,003,328`, and
`51,590,725,632/8,598,454,272` requested bytes. Each case is exactly six to
one, matching the GQA group size.

These are executed kernel global-load *requests*, not physical DRAM
transactions: the counters do not establish cache behavior, coalescing, or
hardware bytes transferred. They also do not measure latency, throughput, or
end-to-end performance. The retained evidence is
[`fixtures/cuda_gqa_attention.json`](../fixtures/cuda_gqa_attention.json),
validated against
[`pins/cuda_gqa_attention_contract.json`](../pins/cuda_gqa_attention_contract.json).
