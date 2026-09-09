# 42. Chunked CUDA GDN prefill in 64-token windows

[Index](README.md) · Implementation tasks: GDN-002, OPT-013, OPT-019, OPT-029, and EDU-028 in
[`implementation_ledger.md`](../implementation_ledger.md)
· Contracts:
[`pins/cuda_gdn_chunk_contract.json`](../pins/cuda_gdn_chunk_contract.json),
[`pins/cuda_gdn_scan_contract.json`](../pins/cuda_gdn_scan_contract.json),
[`pins/opt019_core_recovery_contract.json`](../pins/opt019_core_recovery_contract.json),
[`pins/opt029_gdn_fuse_contract.json`](../pins/opt029_gdn_fuse_contract.json)
· Evidence:
[`fixtures/cuda_gdn_chunk.json`](../fixtures/cuda_gdn_chunk.json),
[`fixtures/cuda_gdn_scan.json`](../fixtures/cuda_gdn_scan.json),
[`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json),
[`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md),
[`fixtures/opt029_gdn_fuse.json`](../fixtures/opt029_gdn_fuse.json),
[`evidence/optimization/opt029-gdn-fuse/REPORT.md`](../evidence/optimization/opt029-gdn-fuse/REPORT.md)

[Chapter 41](41-cuda-gdn-step.md) prepared one token of GDN state. A prompt has
many tokens, and processing those known input tokens is called **prefill**.
GDN-002 accepts an arbitrary positive prompt chunk while retaining exactly the
same causal convolution and recurrent mutation order as repeated one-token work.
OPT-013 keeps that internal **64-token window** and adds an associative parallel
scan as a retained overlay alternate. OPT-019 makes production prompt recurrence
(`GdnScanPath::kFusedTokenLoop`) the warp-column fused quality path. Sequential
64-token windows remain the unloosened GDN-002/OPT-013 numeric and exact-state
reference.

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

The sequential path is still the byte-exact reference: one convolution kernel
and one recurrence kernel per window, in **strict token order**, writing only
**candidate state**. The parallel path uses the same 64-token windows and must
stay inside the frozen GDN-002 envelopes against those visible sequential
results. It is not required to be byte-exact, because composing dense window
operators reorders FP32 reductions.

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

## Associative parallel prompt scan

One gated-delta step is an affine map of the recurrent matrix. For one value
head, normalized key `k`, scalar `α = exp(log_decay)`, scalar `β`, and value
row `v`:

```text
S_t = α (I − β k kᵀ) S_{t−1} + β k vᵀ
y_t = qᵀ S_t
```

Writing `(A, B) = (α(I − βkkᵀ), βkvᵀ)`, two maps compose as
`(A2, B2) ∘ (A1, B1) = (A2 A1, A2 B1 + B2)`. That associativity lets every
window of a prompt chunk run the existing gated-delta arithmetic from a **zero**
initial state in parallel, store its window operator `(A_w, B_w)`, then combine
those operators instead of walking every token serially across the chunk.

`GdnScanPath::kParallelAssociative` does four launches per overlay batch:

1. **Convolution.** One data-parallel FIR over the whole chunk
   (`grid.y == token_count`, production `grid.x == 40`, 256 threads). This is
   not a scan; each token's four causal inputs are the same values the
   sequential ring would see.
2. **Intra windows.** Grid `[value_heads, W]` (production with 64-window
   scratch: `[48, 64, 1]`, 128 threads). Each block runs the existing sequential
   gated-delta loop from zero and stores dense `A_w`
   (`value_heads × key_width × key_width` FP32; production `48 × 128 × 128`)
   plus `B_w` (that window's ending zero-state matrix).
3. **Prefix.** Grid `[value_heads]` (`[48, 1, 1]`). One `A_w S + B_w` matvec per
   window produces each window's incoming `S_in`. This is not a 4,096-token
   homogeneous replay of every token with `v = 0`.
4. **From-state replay.** Grid `[value_heads, W]` again. Each block replays the
   sequential window body from that `S_in` so outputs use sequential-window FMA
   order.

Scratch overlays `prompt_projected_bf16_` only during prepare. One `(A, B)` pair
is 1,572,864 FP32 values. At `R = 4096` the overlay holds `W_fit = 22` windows,
so a 4,096-token overlay layer batches `22 + 22 + 20`. A capacity-65 overlay
cannot hold one pair (`W_fit = 0`) and falls back to sequential windows. Tails
with a single window (`2 ≤ token_count ≤ 64`) keep the existing recurrence
kernel after the 2D convolution. Decode one-token GDN stays sequential. There is
no extra session `cudaMalloc`; the workspace byte formula is unchanged.

Prepare still does not publish committed convolution or recurrent bytes, or the
frontier. The first parallel batch reads committed recurrent state; later
batches continue from candidate. The overlay path remains
`GdnScanPath::kParallelAssociative`. It is not the production default after
OPT-019.

## Warp-column fused production recurrence

Production `kFusedTokenLoop` still launches the existing parallel convolution,
then a warp-column fused recurrence adapted from llama.cpp `gated_delta_net.cu`
at `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors). ds4 has
no GDN analog. Production shape `GdnConfig{16, 48, 128, 128, 4}` uses grid
`(48, 1, 32)` and block `(32, 4)`: each warp owns one value column with
`s_shard[4]` FP32 state. There is no `__syncthreads()` in the token loop.
Rank-2 fused (`float state[128]` per thread) remains `launch_gdn_fused_rank2`
for component A/B only; it is not production. Decode one-token GDN is unchanged.
Partial configs other than 128-wide stay on Rank-2 fused.

Quality versus sequential stays inside maximum absolute error `5e-8`, maximum
RMS `5e-9`, and zero non-finites. Association differs, so quality is not
memcmp-equal to sequential. OPT-008's 4,096-versus-64 memcmp stays on
`kSequentialWindows`. Prepare still does not mutate committed convolution or
recurrent bytes.

## OPT-029 fused conv / gated-output

OPT-029 measured collapsing tiled causal convolution and/or gated-output
into that warp-column token loop. Candidates were `off` (split parallel
conv + warp-column + `gdn_gated_output_rows`), `fuse_conv` (conv inside
the 128-thread loop), `fuse_gate` (gated-output inside a 1024-thread
one-block-per-head loop), and `fuse_both`. Sequential windows remain the
unloosened numeric and exact-state reference. Eligible fused ids met the
frozen `5e-8` / `5e-9` envelopes versus sequential on recurrent output
and candidate state; gated BF16 versus the split gated-output kernel on
the same recurrent also met that envelope. Occupancy was ≥ 1 on every
id. The paired CUDA-event A/B on production 4096-token GDN core
(conv + recurrence + gated-output, 0 warm-ups, 3 samples) did **not**
find an optimized id strictly faster than `off`. Production therefore
keeps the split sequence. `GdnScanPath::kFusedTokenLoop` stays the
production scan enum. `kSelectedGdnFusePath` is `off`. Fused kernels
remain as non-production symbols. Decode GDN is unchanged. Live 4K
keep/reject numbers stay in the report and are not restated here:
[`evidence/optimization/opt029-gdn-fuse/REPORT.md`](../evidence/optimization/opt029-gdn-fuse/REPORT.md).

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

**Measured, RTX 5090, component-only:** the OPT-013 diagnostic on CUDA 13.0.2
kept sequential GDN-002 chunk-versus-tokenwise byte equality, then compared
parallel versus sequential at production 64, 65, 129, 256, and 4,096 tokens
inside maximum absolute error `5e-8`, maximum RMS `5e-9`, and zero non-finite
values. Production 4,096 parallel versus sequential recorded
`max_abs = 1.49011612e-08` and `rms = 1.02587529e-10`, including the
4,096-versus-64-window split. Overlay arithmetic is `W_fit(4096) = 22` and
`W_fit(65) = 0`. Captured 4,096-token geometry with 64-window scratch is one
convolution node `[40, 4096, 256]`, one intra `[48, 64, 128]`, one prefix
`[48, 1, 128]`, and one from-state `[48, 64, 128]`. After three warm-ups, 30
paired CUDA-event samples at 4,096 tokens measured sequential mean
`38.7084427 ms` and parallel mean `31.2641716 ms`
(`measurement_utc=2026-09-08T08:09:51Z`). That A/B uses diagnostic scratch sized
for all 64 windows in one batch; production overlay batches `22 + 22 + 20`.
[`fixtures/cuda_gdn_scan.json`](../fixtures/cuda_gdn_scan.json) retains every
raw sample. The proof limit is component-only GDN-prepare evidence. Sequential
windows remain the byte-exact reference. Nsight Systems is not claimed.
End-to-end prefill/decode speedup and 128K quality are not claimed.

**Measured, RTX 5090, component-only (OPT-019):** quality fused versus sequential
at 2048 tokens recorded `max_abs=1.58324838e-08`, `rms=1.28841632e-10`, zero
non-finites, and prepare isolation. CUDA-event means (3 replicates, 0 warm-ups)
were quality `4.01747179 ms`, Rank-2 `13.941781 ms`, sequential `19.3479786 ms`,
and overlay `14.6024103 ms`, occupancy 9. Live exact-2048 category `gdn` after
this increment is **1205.38806** ms versus the locked before snapshot
**1643.46082** ms. Sequential windows remain the unloosened reference. This is
not an end-to-end 2K tok/s gate. Artifacts:
[`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json)
and
[`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md).

GDN-002 proves arbitrary chunk sizes, internal-window continuity, exact CUDA
token-wise equivalence, and whole-chunk candidate isolation. OPT-013 proves the
associative intra / `A S + B` prefix / from-state scan against those sequential
windows at the frozen envelopes, overlay `W_fit`, sequential fallback, launch
geometry, and the component 4,096-token CUDA-event gate. OPT-019 proves
production fused recurrence is the warp-column quality path inside those same
envelopes, plus the live exact-2048 `gdn` category drop under the locked
addressed rule. OPT-029 proves fused conv and/or gated-output into that
token loop under the same envelopes when a paired A/B wins, or retains
the split sequence with rejection evidence when it does not. None of
these tasks prove complete GDN layers as a speedup claim, the
comparative 5% prefill/decode gates, long-context quality,
request-level atomicity, or llama.cpp 2K tok/s parity.
