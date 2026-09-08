# Chunked full-model CUDA prefill

[Index](README.md) · Implementation tasks: SCH-002, MEM-002, OPT-008, OPT-009, OPT-011, OPT-012, OPT-013, OPT-014, OPT-015, and EDU-047 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contracts:
[`pins/cuda_prompt_scheduler_contract.json`](../pins/cuda_prompt_scheduler_contract.json),
[`pins/cuda_prompt_pipeline_contract.json`](../pins/cuda_prompt_pipeline_contract.json),
[`pins/cuda_prompt_graph_contract.json`](../pins/cuda_prompt_graph_contract.json),
[`pins/cuda_gdn_scan_contract.json`](../pins/cuda_gdn_scan_contract.json),
[`pins/cuda_prefill_attribution_contract.json`](../pins/cuda_prefill_attribution_contract.json),
[`pins/opt015_recovery_contract.json`](../pins/opt015_recovery_contract.json)
· Evidence: [`fixtures/cuda_prompt_scheduler.json`](../fixtures/cuda_prompt_scheduler.json),
[`fixtures/cuda_prompt_pipeline.json`](../fixtures/cuda_prompt_pipeline.json),
[`fixtures/cuda_prompt_graph.json`](../fixtures/cuda_prompt_graph.json),
[`fixtures/cuda_gdn_scan.json`](../fixtures/cuda_gdn_scan.json),
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json),
[`fixtures/opt015_recovery.json`](../fixtures/opt015_recovery.json),
[`evidence/optimization/opt015-2k-recovery/REPORT.md`](../evidence/optimization/opt015-2k-recovery/REPORT.md)

## Why prompt execution differs from decode

**Decode** generates one new token at a time. Token 12 cannot be chosen until
token 11 has produced logits, so the existing stable-address CUDA graph is
specialized for one row. **Prefill** already knows every token in the prompt.
Those rows still have causal dependencies, but their large matrix projections
can be calculated together.

Before SCH-002, `Session::sync` called the complete decode scheduler once per
prompt token. A 64-token prompt therefore read the same weight matrices 64
separate times and launched thousands of small operations. This was correct but
the BEN-001 smoke exposed it at only about 16.2 prompt tokens/s.

## Token-major storage and layer-major work

Quartz stores a chunk in **token-major** order: all 5,120 residual values for
token 0, then all values for token 1, and so on. If `R` is a 4,096 × 5,120
residual matrix, row `t` begins at `R[t * 5120]`.

Execution is **layer-major**. The scheduler carries all rows through layer 0,
then all rows through layer 1, continuing through layer 63. Contrast this with
the old token-major schedule, which carried token 0 through all 64 layers before
starting token 1. Layer-major execution lets each projection consume a matrix of
prompt rows while its weights are already being used.

For one chunk the path is:

1. Decode every token's embedding row. Production uses one batched kernel that
   still round-trips through BF16 into the FP32 residual. The retained unfused
   path launches one decode per token and a separate widen.
2. Normalize each residual row independently at layer 0. Later layers receive
   that input norm from the previous fused residual-add-norm.
3. Use MMQ to project all rows for the layer's mixer.
4. Apply either the GDN recurrence or causal grouped-query attention.
5. Fuse mixer residual-plus-FFN-norm, run the three prompt-row FFN projections,
   and fuse FFN residual-plus-next-input-norm (layer 63's last FFN add stays
   standalone).
6. After layer 63, compute final normalization and logits only for the last row.
7. Overlap last-row logits/hidden D2H with one all-layer KV scatter, join both
   prompt streams, then publish persistent state and advance the frontier.

Only the last logits are needed because `Session::sync` promises the state from
which generation continues, not one logit matrix for every prompt position.

## What MMQ changes

MMV means matrix-vector multiplication: one activation row. **MMQ** here means
the same quantized weight matrix multiplied by several prompt rows.

Q4_K and Q6_K projections use the admitted transient Q8 activation blocks and
`launch_quant_mmq`. OPT-009 extends those compile-time tiles through 64 and
selects them from a checked-in RTX 5090 sweep, so `grid.y` is
`ceil(prompt_rows / selected_tile)` rather than OPT-004's eight-row ceiling.

The first integration attempt sent Q8_0 weights through that same Q8-staged
path. That was wrong for exact scheduler equivalence: the one-token Q8_0 path
multiplies the weights directly by BF16 activations, while generic MMQ
requantized those activations. The result was numerically close but changed
persistent state and last logits. The failed `append_vs_fresh` run is retained
in the ledger.

Quartz therefore keeps a Q8_0-by-BF16 kernel. SCH-002's first version processed
multiple rows by mapping `blockIdx.y` to one prompt row (`grid.y = prompt_rows`).
That batched launches without reusing weights: each output-row warp reread the
entire packed matrix. OPT-009 replaces production with `launch_q8_mmq_bf16`.
`blockIdx.y` owns a prompt-row tile, one Q8_0 weight is decoded per column, and
that scalar is applied to every in-range prompt row with the decode
`__fmul_rn` / `__fadd_rn` walk. Activations stay BF16. The row-wise kernel is
retained only as `launch_q8_mmq_bf16_reference`. Captured graphs at 64 and 4,096
prompt rows show production `grid.y` of 16 and 1,024 (selected tile 4) versus
reference `grid.y` equal to the prompt-row count.

The two-token prefix test and the 65-token boundary test then returned
byte-equal state, hidden output, and logits. OPT-008's `[4096, 1]` exact-state
comparison remains the scheduler equality authority. OPT-009's own proof is
component-only: weight-tile reuse, measured SM120 selection, occupancy, Q8_0
byte equality to the retained reference, frozen CUD-002 envelopes, and the
component timing predicates in
[`fixtures/cuda_prompt_mmq.json`](../fixtures/cuda_prompt_mmq.json). It does not
claim end-to-end prefill or decode speedup, or 128K quality recovery.

## GDN and attention remain causal

Batching projections does not remove recurrence. Within each GDN layer the
internal scan window is still at most 64 tokens, carries the convolution ring
and FP32 recurrent matrix forward, and produces a final candidate state for that
layer.

Production prompt chunks use OPT-013's associative parallel scan when overlay
scratch on `prompt_projected_bf16_` can hold one `(A_w, B_w)` pair. At
`prompt_chunk_rows_ == 4096` that overlay fits `W_fit = 22` windows, so a
4,096-token layer batches `22 + 22 + 20`: zero-state intra windows, a 48-block
`A S + B` prefix, then parallel from-state replay of sequential window
arithmetic. Tails with one window (`2 ≤ token_count ≤ 64`) keep sequential
recurrence after a single 2D convolution. Capacity-65 cannot overlay one pair
(`W_fit = 0`) and falls back to sequential windows. Decode one-token GDN stays
sequential. Prompt FFN graphs still exclude GDN.

OPT-008's 4,096-versus-64-row memcmp remains a **sequential** GDN gate: that
comparison pins `GdnScanPath::kSequentialWindows` rather than loosening
byte equality to tolerances. Parallel cross-boundary proof is the OPT-013
diagnostic at the frozen GDN-002 envelopes (`5e-8` / `5e-9` / zero non-finite),
including 4,096 parallel tokens versus 64 sequential windows.

Attention also visits chunk rows in order. A row may read all committed KV rows
from earlier chunks and candidate rows earlier in its current chunk, never a
future row. Partial RoPE uses the absolute position `old frontier + row`.

OPT-008 sets the outer scheduler policy to 4,096 rows. A 4,097-token prompt
therefore executes as `[4096, 1]`, with the final single row using the
established decode arithmetic. For a smaller session, the reusable allocation
and selected prompt chunk are bounded by its capacity: a capacity-65 session
executes `[65]` as one prompt transaction rather than allocating or dispatching
4,096 rows.

## Prompt pipeline fusion

After OPT-008's 4,096-row transactions, OPT-009's weight-reusing MMQ, and
OPT-010's physical KV scatter, avoidable cost remained in prompt *orchestration*:
one embedding kernel per token, unfused residual/norm pairs, sixteen one-block
scatter launches, and a blocking last-row copy before those copies.

OPT-011 keeps MMQ, GDN, and attention arithmetic. It changes how
`execute_prompt_chunk` launches and commits them. Production is
`PromptPipelinePath::kFusedOverlapped`. `PromptPipelinePath::kUnfusedSerial`
retains the previous serial path as the exactness reference, not a second
supported product backend.

**Batched embedding.** Production copies the chunk's token IDs into unused
prompt-Q8 scratch (no extra `cudaMalloc`) and launches one
`launch_quant_rows_decode_widen` kernel. Captured graphs show `grid.y` equal to
the token count: `[20, 64, 1]` at 64 rows and `[20, 4096, 1]` at 4,096 rows,
each with 256 threads. Thread `(column, row)` decodes `token_ids[row]` with the
existing weight decoder, stores `__float2bfloat16_rn`, then writes
`__bfloat162float` of that BF16 into the FP32 residual. That is the old
decode-then-widen round-trip without a global BF16 embedding store. The unfused
path still loops `launch_quant_row_decode` and one `bf16_to_fp32`.

**Residual-add-norm on prompt rows.** Decode already fused FFN residual add with
the next layer's input RMSNorm (OPT-002). Production prompt now uses a row-wise
kernel of the same class: `token_count` blocks of 256 threads, parallel
`__fadd_rn` into the residual, the same ordered FP32 sum-of-squares on thread 0,
then scaled BF16 stores. After each mixer: fused add into `after_mixer` plus
FFN-norm. After FFN of layers 0–62: fused add into `residual` plus the next
layer's input-norm. That is **127** fused launches per 64-layer chunk plus **one**
standalone last FFN residual add on layer 63. Layer 0 still has a standalone
input RMSNorm. SwiGLU, GDN gate prep, attention split, and post-attention BF16
conversion stay unfused so OPT-009/OPT-007 kernels do not change.

**All-layer scatter.** A 4,096-row layer is 4,194,304 BF16 pairs. The old scatter
walked that in **one** 256-thread block. Production now launches
`launch_attention_scatter_layers` once: `grid.y` is the 16 attention layers, and
`grid.x` is `min(2048, ceil(values / 256))`. Captured graphs are `[256, 16, 1]`
at 64 rows and `[2048, 16, 1]` at 4,096 rows, each one kernel node. Physical
index math is unchanged. Decode `execute_token` also calls this launcher for one
token; it still uses blocking D2H and `cudaDeviceSynchronize`. The one-layer
wrapper remains so OPT-010 diagnostics stay valid.

**Copy/compute overlap.** After all 64 layers and last logits enqueue on the
prompt compute stream, the scheduler records a compute-done event, waits that
event on a second copy stream, and issues **two** asynchronous D2H copies of last
logits and last hidden on the copy stream while the all-layer scatter runs on
compute. Publication waits until both streams join, then host-swaps GDN
pointers, copies tokens and outputs, and advances the frontier last. The unfused
path keeps two blocking `cudaMemcpy`s, sixteen one-layer scatters, and one
`cudaDeviceSynchronize`. Workspace streams and the event are driver handles; the
OPT-008 byte formula `160,380,416 + 96*C + 404,992*R` is unchanged.

These launch and barrier counts are executed increments next to the CUDA calls.
The pinned image still has no Nsight Systems, so overlap is not claimed from a
Systems timeline.

## Prompt FFN CUDA graphs

After OPT-011 fusion, production 4,096-row fused chunks replay 64 stable-address
prompt FFN graphs on `prompt_compute_stream_`. Capture and ordinary fused FFN
share `execute_prompt_ffn`, so the graph records the same mixer residual-plus-
FFN-norm, gate/up/down MMQ, SwiGLU, and next-input residual-add-norm sequence.
Mixer GDN/attention, embedding, logits, D2H, scatter, and commit stay ordinary
launches because those pointers and extents still change per chunk.

Replay is exact-geometry only. A full 4,096-token fused chunk with graphs bound
to that workspace `cudaGraphLaunch`es each layer after the mixer writes
`prompt_mixer_output_`. Tails and any `token_count != 4096` keep ordinary fused
FFN on the same helper. Workspaces with `prompt_chunk_rows_ != 4096` never create
the prompt set, so their fused chunks also stay ordinary. One-row remainders
replay the existing decode FFN graphs through `execute_token`.

Equality is graph versus ordinary fused, not a new unfused gate. Graphs bound to
another workspace, or combined with the unfused serial path, fail before
publication. Cancellation still waits for the finished layer, including a graph
replay, then polls; a stop at poll 8 publishes nothing and launches no later
graphs, logits, D2H, or scatter.

**Measured, RTX 5090:** a 4,096-row fused graph chunk was byte-exact to a
4,096-row ordinary fused chunk in committed GDN/KV, tokens, frontier, last
hidden, and logits. Graph counters recorded 64 prompt graph launches; the
ordinary fused pair recorded zero, plus the inherited 127 fused residual-add-norm
launches and one last residual add. A 64-row fused fallback on the 4096-bound
object was also byte-exact. Capacity-65 create kept 64 decode graphs and skipped
prompt graphs. Proof is component-only FFN-subgraph evidence; it is not a
whole-chunk graph, Nsight Systems claim, end-to-end speedup, or 128K quality
recovery.

## Candidate state, committed state, and cancellation

**Committed** state is the conversation callers are allowed to observe.
**Candidate** state is temporary work that might still fail. Each prompt chunk
uses separate candidate storage for all 48 GDN layers and for up to 4,096 KV rows
in all 16 attention layers, bounded by session capacity. Tokens, last hidden
state, logits, and frontier also remain
unchanged during calculation.

After every layer, an optional cancellation callback is polled. The fused path
`cudaStreamSynchronize`s the compute stream at that boundary; the unfused path
keeps `cudaDeviceSynchronize`. If cancellation arrives, the function returns
`cancelled` and does not enqueue later layers, logits, D2H, or scatter, and does
not swap GDN state, copy KV rows, copy tokens, or advance the frontier. The
measured 4,096-row cancellation case remained byte-equal to an empty session
with frontier zero. A 64-row fused cancel at poll 8 likewise published nothing
and launched no scatter.

Only after all 64 layers and the last logits succeed does the chunk commit.
Async D2H overlapping scatter is still candidate work. Host GDN pointer swaps,
token memcpy, caller-buffer writes, and frontier advance run only after both
prompt streams join. If logits, the copies, scatter, or either join fails, the
function returns without that host publication. Staging buffers may hold
candidate bytes; that is not publication.

## Fixed scratch and the 128K budget

**Scratch** is reusable temporary memory whose contents have no meaning after an
operation. The workspace permanently owns buffers for
`min(4096, session capacity)` prompt rows: two FP32 residual matrices, BF16
normalized/projected rows, Q8 activations, projection and mixer outputs, GDN
intermediates, and per-layer candidate KV rows. This fixed, capacity-bounded
allocation avoids request-sized allocator activity and leaves the decode graph's
addresses unchanged.

At capacity 131,072, the diagnostic workspace is 1,831,810,560 bytes. That
requested size is unchanged by prompt graphs: they are additional CUDA graph
objects, not extra scratch. The live simultaneous 131,072-token session plus
resident model plus 128 uploaded graphs is in
[Chapter 54](54-post-graph-128k-memory.md). **Measured, RTX 5090:** 3,521,118,208
bytes remained free after 64 decode plus 64 prompt executables, leaving
1,910,505,472 bytes above the required 1.5 GiB reserve.

## Measured result and proof boundary

**Measured, RTX 5090:** a deterministic 4,097-token history executed as
`[4096, 1]` and was byte-equal in committed GDN/KV state, frontier, last hidden
vector, and logits to explicit 64-row chunks followed by one decode row. The
capacity-65 fallback executed as one 65-row transaction with 64 layer polls and
was byte-equal to an explicit `[64, 1]` reference. Cancelling a 4,096-row chunk
at a layer boundary left the empty session and caller outputs unchanged.

These focused native checks prove the chunk policy, its tail and capacity
fallbacks, exact differential, cancellation before commit, and physical 128K
allocation reserve. OPT-009 adds component-only MMQ evidence: production
weight-tile reuse, measured SM120 kind×bucket selection, and frozen numeric
envelopes.

OPT-011 adds component-only pipeline evidence in
[`fixtures/cuda_prompt_pipeline.json`](../fixtures/cuda_prompt_pipeline.json).
**Measured, RTX 5090:** fused versus unfused chunks at 2, 3, 64, and 65 rows
were byte-equal in committed GDN/KV, tokens, frontier, last hidden, and logits.
Fused 64-row counters were one embedding launch, zero widen, one scatter, zero
blocking D2H, two async D2H, zero `cudaDeviceSynchronize`, 127 fused
residual-add-norm launches, and one last FFN residual add. Unfused counters were
64 embeddings, one widen, 16 scatters, two blocking D2H, and one device
synchronize. A 64-row fused CUDA-event mean of 1297.02747 ms was strictly below
the unfused mean of 1298.96277 ms over 30 paired samples after three warm-ups.
That A/B is an orchestration predicate, not an end-to-end prefill claim. The
fixture states the proof limit: component-only orchestration evidence, no Nsight
Systems overlap screenshot, and no end-to-end prefill/decode speedup.

OPT-013 adds component-only GDN-prepare evidence in
[`fixtures/cuda_gdn_scan.json`](../fixtures/cuda_gdn_scan.json).
**Measured, RTX 5090:** parallel versus sequential production 4,096-token
prepare stayed inside `max_abs = 1.49011612e-08` and `rms = 1.02587529e-10`
with zero non-finite values, including the 4,096-versus-64-window split.
Overlay `W_fit(4096) = 22` and `W_fit(65) = 0`. After three warm-ups, 30 paired
CUDA-event samples measured sequential mean `38.7084427 ms` and parallel mean
`31.2641716 ms` on diagnostic 64-window scratch. That A/B is a component
GDN-prepare predicate, not an end-to-end prefill claim. Sequential 64-token
windows remain the byte-exact reference. Nsight Systems is not claimed.

OPT-014 adds live 2048-token category exposure in
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json).
**Measured, RTX 5090:** one cold empty-session 2048-token production
`sync_tokens` (capacity 131072, fused overlapped, parallel GDN scan, graphs
created) took 41963.8828 ms host wall (~48.80 tok/s). Exclusive CUDA events on
the prompt compute stream recorded FFN/MMQ 32294.9512 ms, GDN 5300.55371 ms,
attention 4364.3335 ms, logits 2.86684799 ms, commit/sync 0.576767981 ms, and
embedding 0.076063998 ms. Graph was measured 0 ms with zero prompt-graph
launches, because a 2048-token chunk does not replay 4096-row FFN graphs. The
other/idle remainder was 0.5234375 ms, so the eight named categories reconstruct
wall within `rel_tol = 1e-4` and `abs_tol_ms = 0.05`. Nsight Systems and Nsight
Compute were `not_used`. This is instrumentation of that timed run, not a
throughput gate and not llama.cpp parity. Chapter 51 owns the category
definitions.

OPT-015 maps those eight categories to a ranked recovery sequence. It copies
the OPT-014 exact-2048 attribution and the scaling `llama-bench` 2K object in
[`evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json`](../evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json).
**Measured:** Quartz 41963.8828 ms (~48.80 tok/s) versus llama.cpp `cc83d7b`
mean 3114.049476 tok/s (`n_ubatch` 512, `flash_attn` -1). **Estimated** gap:
`3114.049476 / 48.8038712 ≈ 63.8×`. **External:** llama.cpp reaches thousands
of tok/s on this same GGUF because Q4_K MMQ is MMA/tensor-core (J up to 128),
GDN is a fused per-head token loop, and `flash_attn` auto selects MMA F16.
Quartz 2K `graph` is measured 0 ms, so CUDA graphs are not the present gap.
`../ds4` is MIT inspiration only; ds4 cannot run this Qwen GGUF, and there is
no ds4 same-model baseline.

Exclusive category map (shares of OPT-014 `wall_ms` 41963.8828 ms):

| Category | Share | Faster path or keep | Rank |
|---|---:|---|---:|
| `ffn_mmq` | 77.0% | Q4_K/Q6_K MMA MMQ; keep CUD-002 reference | 1 `q4k_q6k_mma_mmq` |
| `gdn` | 12.6% | Fused per-head GDN token loop after Rank 1 | 2 `gdn_fused_token_loop` |
| `attention` | 10.4% | Full-causal MMA/tiled attention; not sparse | 3 `causal_mma_attention` |
| `logits` | ~0.007% | Keep | — |
| `commit_sync` | ~0.001% | Keep OPT-011 overlap | — |
| `embedding` | ~0.0002% | Keep | — |
| `graph` | 0% | Optional 2048-row FFN graphs after Rank 1 | 4 `optional_2048_prompt_graphs` |
| `other_idle` | ~0.001% | Keep remainder | — |

**Proposed** sequence: `q4k_q6k_mma_mmq`, `gdn_fused_token_loop`,
`causal_mma_attention`, `optional_2048_prompt_graphs`. Rank 1 is mandatory
because the **Estimated** FFN-only ceiling is ~63 tok/s. OPT-016 remains the
later 2K throughput gate; this report does not claim that sequence will pass
it. Artifacts:
[`evidence/optimization/opt015-2k-recovery/REPORT.md`](../evidence/optimization/opt015-2k-recovery/REPORT.md),
[`pins/opt015_recovery_contract.json`](../pins/opt015_recovery_contract.json),
and [`fixtures/opt015_recovery.json`](../fixtures/opt015_recovery.json).

The **proof boundary** excludes comparative speed claims, 2K/8K sustained
prefill throughput, execution of a 128K prefill, 128K retrieval quality, thermal
stability, superiority to llama.cpp/vLLM, and a Nsight Systems overlap timeline.
OPT-014 measures named categories on one cold 2048-token timed run; it does not
convert that instrumentation into a sustained-prefill or speed admission.
OPT-015 explains the 2K gap and ranks recoveries; it is not llama.cpp parity
and not a throughput gate. BEN-001
provides the harness; CMP-002/CMP-003 still own the 30-sample comparative gate.
QLT-001 remains blocked. OPT-012's prompt graphs are FFN subgraphs only: not a
whole-chunk graph, not a speedup gate, and not 128K quality recovery.
OPT-013 does not claim those gates either.
