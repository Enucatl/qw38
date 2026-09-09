# 44. Memory-bounded CUDA attention prefill at 128K

[Index](README.md) · Implementation tasks: ATN-002, OPT-010, OPT-019, OPT-026, OPT-027, OPT-033, and EDU-030 in
[`implementation_ledger.md`](../implementation_ledger.md)
· Contracts:
[`pins/opt019_core_recovery_contract.json`](../pins/opt019_core_recovery_contract.json),
[`pins/opt026_fattn_streamk_contract.json`](../pins/opt026_fattn_streamk_contract.json),
[`pins/opt027_persistent_fattn_contract.json`](../pins/opt027_persistent_fattn_contract.json),
[`pins/opt033_register_vkq_contract.json`](../pins/opt033_register_vkq_contract.json)
· Evidence:
[`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json),
[`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md),
[`fixtures/opt026_fattn_streamk.json`](../fixtures/opt026_fattn_streamk.json),
[`evidence/optimization/opt026-fattn-streamk/REPORT.md`](../evidence/optimization/opt026-fattn-streamk/REPORT.md),
[`fixtures/opt027_persistent_fattn.json`](../fixtures/opt027_persistent_fattn.json),
[`evidence/optimization/opt027-persistent-fattn/REPORT.md`](../evidence/optimization/opt027-persistent-fattn/REPORT.md),
[`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](../evidence/optimization/opt027-persistent-fattn/REJECTION.md),
[`fixtures/opt033_register_vkq.json`](../fixtures/opt033_register_vkq.json),
[`evidence/optimization/opt033-register-vkq/REPORT.md`](../evidence/optimization/opt033-register-vkq/REPORT.md)

[Chapter 43](43-cuda-attention-decode.md) processed one new token. **Prefill**
processes a known prompt containing many tokens. The same causal rule applies:
prompt token 20 can use tokens 0 through 20, but not token 21.

ATN-002 extends the post-projection CUDA boundary to arbitrary positive chunks.
It prioritizes an inspectable state and memory contract. OPT-005 replaces its
per-row launch loop with a fixed-memory tiled path; the former loop remains
available as a test-only reference for differential measurements. OPT-019 makes
production `launch_attention_prepare_chunk` the fattn-mma-f16 analog when
`token_count >= 16`. The two-row tiled path remains the unloosened OPT-005
numeric reference and is still used when `token_count < 16`. Rank-3 warp-0 MMA
is retained for component A/B only.

## Token-major input and output

Projected query, key, value, and gate arrays are **token-major**. All values for
token 0 come first, then all values for token 1. Output contexts follow the same
order:

```text
[token 0 query heads and lanes]
[token 1 query heads and lanes]
...
```

Committed device K/V use the shared physical layout in
[Chapter 43](43-cuda-attention-decode.md). Candidate chunk rows stay token-major.

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
prompt-sized score rectangle is allocated, and both the tiled reference and
the quality MMA path leave global score scratch byte-for-byte untouched.

The tiled reference launch has exactly two kernel launches per positive chunk:
KV staging on a `(kv_head, token)` grid, followed by grouped attention on a
`(kv_head, ceil(token_count / 2))` grid. This is launch topology evidence, not
a complete model or end-to-end speed result. Production quality MMA
(`token_count >= 16`) uses the fattn-mma analog with pinned ncols1=16 and
ncols2=2. OPT-026 keeps Ada+ stream-K (KV bipartition plus softmax combine)
after a paired A/B win versus occupancy-1 whole-tile; occupancy-2 whole-tile
was eligible but slower than stream-K. The scheduler aliases existing
`prompt_projected_bf16_` and `prompt_q8_` for the combine buffers. Score
scratch stays untouched. OPT-027 measured llama.cpp Ada+ persistent
stream-K (`nsm × occupancy` linearized tiles plus 5% efficiency rounding)
against that production path and did not install it after an A/B loss.
OPT-033 then kept register-resident value accumulation on that same
stream-K grid: each thread holds the running FP32 value sum across 32-row
KV tiles and stores once after the KV loop. Production prompt fattn
therefore remains OPT-026 `stream_k` (`grid.z=2`) with the OPT-033
`registers` pin. Decode `launch_attention_prepare` partitions stay.
The two-row tiled path remains the unloosened OPT-005 numeric reference.
Live 4K keep/reject numbers stay in
[`evidence/optimization/opt026-fattn-streamk/REPORT.md`](../evidence/optimization/opt026-fattn-streamk/REPORT.md),
[`evidence/optimization/opt027-persistent-fattn/REPORT.md`](../evidence/optimization/opt027-persistent-fattn/REPORT.md),
and
[`evidence/optimization/opt033-register-vkq/REPORT.md`](../evidence/optimization/opt033-register-vkq/REPORT.md).

## Whole-chunk prepare, commit, and cancellation

[`launch_attention_prepare_chunk`](../cuda/attention_decode.cu) fills every
candidate K/V row and token-major output while leaving the committed cache and
frontier unchanged. If any later layer fails or a request is cancelled, the
caller discards all candidate rows.

[`launch_attention_commit_chunk`](../cuda/attention_decode.cu) scatters each
token-major candidate value into its physical cache address. One block
synchronizes after scattering both K and V, then advances the frontier. No
partial chunk becomes visible through that frontier.

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

## OPT-010 committed tile contiguity

OPT-007 reused each 32-row K/V tile for two query rows, but consecutive tokens
of one KV head still strode by 1,024 BF16 values in committed storage. OPT-010
stores committed K/V with the shared physical formula from
[Chapter 43](43-cuda-attention-decode.md). Decode and prefill read that same
layout.

When every row of a 32-row tile lies below `start_position`, the production
kernel takes one `tile_base` pointer per KV head and loads:

```text
keys[row * 256 + lane]   = tile_base[row * width + lane]
values[row * 256 + lane] = tile_base_v[row * width + lane]
```

That global span is `32 × 256` consecutive BF16 values. An instrumented
specialization of the production load helper records the device pointers of
`tile_base` and `tile_base + rows * width - 1` and requires
`last - first == (rows * width - 1) * sizeof(__nv_bfloat16)` with `rows == 32`
for those full committed tiles. Production
[`launch_attention_prepare_chunk`](../cuda/attention_decode.cu) stays
uninstrumented.

Mixed tiles, where some rows are already committed and some are still
candidates, load committed rows from the corresponding physical addresses and
candidate rows from token-major candidate storage. They are not one coalesced
region and are not counted as coalesced failures.

The retained diagnostic also requires a bijection of the physical index onto
the unchanged `attention_cache_values` product, byte-exact logical pack/unpack
round-trips, byte-exact production versus retained one-row GQA outputs and
candidate BF16 rows, commit-scatter plus pack equal to the logical candidate
prefix, two captured kernel nodes with 33,792 dynamic shared bytes, and a
16-layer K+V product of 8,589,934,592 bytes. This is executed pointer-span and
exact-value evidence, not Nsight DRAM transactions, latency, throughput, or
end-to-end recovery. The contract and retained record are
[`pins/cuda_kv_tile_layout_contract.json`](../pins/cuda_kv_tile_layout_contract.json)
and [`fixtures/cuda_kv_tile_layout.json`](../fixtures/cuda_kv_tile_layout.json).

## OPT-019 fattn-mma quality production

Production `launch_attention_prepare_chunk` for `token_count >= 16` is the
fattn-mma-f16 analog adapted from llama.cpp `fattn.cu` / `fattn-mma-f16.cuh` /
`mma.cuh` at `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors).
ds4 is mentioned only for multi-row online structure; this increment does not
copy `../ds4` and does not include ggml headers. Geometry is DKQ=DV=256 with
GQA group size 6: `ncols2 = 2` so each KV-head block walks the six query heads
as three subgroups and loads each KV tile once per tile for the group.
`ncols1` was swept at 2048 query rows in `{8, 16, 32}`; ncols1=8 and 16 met
OPT-005 versus tiled with dual-F16 KQ, ncols1=32 missed the envelope (single
F16), and the winner pin is `attention_mma_query_rows_2048 = 16`. That pin is
used for every production quality launch with `token_count >= 16`, including
4,096-row chunks. All warps participate in `mma.sync` m16n8k16. Online softmax
stays in registers. Decode `launch_attention_prepare` is unchanged.

Quality versus `launch_attention_prepare_chunk_tiled` stays inside OPT-005
BF16 max abs `5e-5`, RMS `5e-6`, zero non-finites, and exact candidate K/V.
Rank-3 `launch_attention_prepare_chunk_mma_rank3` remains for A/B; it is not
production. OPT-007 `query_rows_per_block = 2` is still the tiled pin, not the
quality MMA tile.

**Measured, RTX 5090, component-only:** at 2048 rows, quality versus tiled
recorded `max_abs=3.16649675e-06`, `rms=1.31638146e-07`, `passed=true`,
winner ncols1=16. CUDA-event means were quality `17.1602116 ms`, tiled
`168.421204 ms`, Rank-3 `59.7287598 ms`, occupancy 1. ncols1 sweep means were
8: `20.1653976 ms` (envelope true), 16: `17.1821537 ms` (envelope true), 32:
envelope false. Live exact-2048 category `attention` after this increment is
**475.116028** ms versus the locked before snapshot **1148.47461** ms. Tiled
attention remains the unloosened reference. This is not an end-to-end 2K tok/s
gate. Artifacts:
[`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json)
and
[`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md).

## OPT-027 persistent Ada+ fattn stream-K

OPT-027 measured llama.cpp Ada+ **persistent** stream-K for prompt
attention: launch `nsm × occupancy` blocks over linearized
KV×destination-tile work, apply 5% occupancy-loss efficiency rounding,
and combine fractional tiles with uniform or general softmax fixup plus
the existing output-gate. Persistent fixup would alias existing
`prompt_q8_` / `prompt_projected_bf16_` (no extra `cudaMalloc`). Score
scratch stays untouched. `kSelectedFattnPath` stays `"stream_k"` so the
OPT-026 host pin remains valid. Persistent selection is a separate
enumerator `kSelectedPersistentFattnPath`.

**Measured, RTX 5090:** paired CUDA-event A/B on production 4096-row fattn
among `stream_k` (occupancy-2 plus `grid.z=2` KV bipartition) and
`persistent`. Winner `stream_k`; `persistent` did not strictly beat that
baseline under frozen OPT-005 envelopes with score scratch untouched.
Raw A/B samples stay in
[`evidence/optimization/opt027-persistent-fattn/fattn-ab-raw.txt`](../evidence/optimization/opt027-persistent-fattn/fattn-ab-raw.txt).
Production therefore stays OPT-026 `stream_k` (`grid.z=2`). Persistent
kernels remain as non-production symbols. ncols1 stays 16. Tiled
`launch_attention_prepare_chunk_tiled` remains the unloosened OPT-005
reference. Decode attention is unchanged. Mixer Q8 quality, skinny
`mma_i32_j128`, and FFN `shared_y_swiglu_q8` stay. Live exclusive sitting
**reject:** Quartz mean is not strictly greater than the frozen
successor-oracle baseline **1746.71973**; `reverted` true;
`successor_oracle` false; `production_persistent_installed` false; A/B
winner `stream_k`; `ladder_exhausted` false. `quartz_meets_llama` is
informational and is not this gate. **The 2K parity owner remains the
blocked dedicated gate.** This reject does not substitute for that gate
and does not claim Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt027-persistent-fattn/REPORT.md`](../evidence/optimization/opt027-persistent-fattn/REPORT.md),
[`pins/opt027_persistent_fattn_contract.json`](../pins/opt027_persistent_fattn_contract.json),
[`fixtures/opt027_persistent_fattn.json`](../fixtures/opt027_persistent_fattn.json),
and
[`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](../evidence/optimization/opt027-persistent-fattn/REJECTION.md).

## Register-resident value sums (OPT-033)

Production stream-K quality still scores KV in 32-row tiles with scalar
probability×V. The previous path rescaled and stored the running value
sum through global `vkq` on every tile. OPT-033 gives each thread a
compile-time FP32 accumulator (production: 64 floats), rescales those
registers, and writes live lanes once after the KV loop. The stream-K
combine kernel is unchanged. Ncols1=16, Ncols2=2, KV tile 32, dual-F16 Q,
`__launch_bounds__(128, 2)`, and `grid.z=2` stay frozen. Score scratch
stays untouched. Combine buffers continue to alias existing
`prompt_projected_bf16_` / `prompt_q8_`. There is no extra persistent
`cudaMalloc`. MMA probability×V is out of scope. Decode attention stays
on its partitioned one-token path.

`launch_attention_prepare_chunk_stream_k` honors compile-time
`kSelectedVkqAccum`. Both `global` and `registers` remain launchable so
the paired A/B can compare them. Byte equality is versus current stream-K
`global`, not versus tiled; OPT-005 envelopes versus tiled remain the
numeric gate.

**Measured, RTX 5090:** paired 3-warm / 30-alternating CUDA-event A/B at
4096 rows including combine selected **registers**. Combined FP32 output
was byte-equal to `global`, occupancy met the eligibility floor, and the
component mean was strictly lower. Keep also required live P tok/s
strictly above the frozen then-current accepted P denominator and the
cross-workload D128/D2048 guard. D2048 tok/s improvement is not required
for this prefill keep. Production pin is `registers`. `reverted` is
false. Live means, p95s, and per-candidate samples stay in the report;
this chapter does not replace them:
[`evidence/optimization/opt033-register-vkq/REPORT.md`](../evidence/optimization/opt033-register-vkq/REPORT.md).
This is a register-resident prefill value-sum keep/reject under frozen
envelopes, not the 2K llama.cpp parity gate and not Quartz ≥ llama.cpp.
