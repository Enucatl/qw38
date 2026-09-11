# Chunked full-model CUDA prefill

[Index](README.md) · Implementation tasks: SCH-002, MEM-002, OPT-008, OPT-009, OPT-011, OPT-012, OPT-013, OPT-014, OPT-015, OPT-017, OPT-018, OPT-019, OPT-020, OPT-021, OPT-022, OPT-023, OPT-024, OPT-025, OPT-026, OPT-027, OPT-028, OPT-029, OPT-030, OPT-032, OPT-033, OPT-035, OPT-037, OPT-038, OPT-040, OPT-041, OPT-052, OPT-053, OPT-054, OPT-055, OPT-056, OPT-069, OPT-080, and EDU-047 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contracts:
[`pins/cuda_prompt_scheduler_contract.json`](../pins/cuda_prompt_scheduler_contract.json),
[`pins/cuda_prompt_pipeline_contract.json`](../pins/cuda_prompt_pipeline_contract.json),
[`pins/cuda_prompt_graph_contract.json`](../pins/cuda_prompt_graph_contract.json),
[`pins/cuda_gdn_scan_contract.json`](../pins/cuda_gdn_scan_contract.json),
[`pins/cuda_prefill_attribution_contract.json`](../pins/cuda_prefill_attribution_contract.json),
[`pins/opt015_recovery_contract.json`](../pins/opt015_recovery_contract.json),
[`pins/opt017_mixer_mma_contract.json`](../pins/opt017_mixer_mma_contract.json),
[`pins/opt018_ffn_mma_contract.json`](../pins/opt018_ffn_mma_contract.json),
[`pins/opt019_core_recovery_contract.json`](../pins/opt019_core_recovery_contract.json),
[`pins/opt020_prefill_split_contract.json`](../pins/opt020_prefill_split_contract.json),
[`pins/opt021_oracle_contract.json`](../pins/opt021_oracle_contract.json),
[`pins/opt022_mixer_q8_quality_contract.json`](../pins/opt022_mixer_q8_quality_contract.json),
[`pins/opt023_skinny_mixer_contract.json`](../pins/opt023_skinny_mixer_contract.json),
[`pins/opt024_mixer_q8_d2r_contract.json`](../pins/opt024_mixer_q8_d2r_contract.json),
[`pins/opt025_ffn_shared_y_contract.json`](../pins/opt025_ffn_shared_y_contract.json),
[`pins/opt026_fattn_streamk_contract.json`](../pins/opt026_fattn_streamk_contract.json),
[`pins/opt027_persistent_fattn_contract.json`](../pins/opt027_persistent_fattn_contract.json),
[`pins/opt028_mmq_streamk_contract.json`](../pins/opt028_mmq_streamk_contract.json),
[`pins/opt029_gdn_fuse_contract.json`](../pins/opt029_gdn_fuse_contract.json),
[`pins/opt030_pdl_launches_contract.json`](../pins/opt030_pdl_launches_contract.json),
[`pins/opt032_decode_oracle_contract.json`](../pins/opt032_decode_oracle_contract.json),
[`pins/opt033_register_vkq_contract.json`](../pins/opt033_register_vkq_contract.json),
[`pins/opt035_pv_mma_contract.json`](../pins/opt035_pv_mma_contract.json),
[`pins/opt040_gdn_shared_inverse_contract.json`](../pins/opt040_gdn_shared_inverse_contract.json),
[`pins/opt041_fattn_warp_qk_contract.json`](../pins/opt041_fattn_warp_qk_contract.json)
· Evidence: [`fixtures/cuda_prompt_scheduler.json`](../fixtures/cuda_prompt_scheduler.json),
[`fixtures/cuda_prompt_pipeline.json`](../fixtures/cuda_prompt_pipeline.json),
[`fixtures/cuda_prompt_graph.json`](../fixtures/cuda_prompt_graph.json),
[`fixtures/cuda_gdn_scan.json`](../fixtures/cuda_gdn_scan.json),
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json),
[`fixtures/opt015_recovery.json`](../fixtures/opt015_recovery.json),
[`fixtures/opt017_mixer_mma.json`](../fixtures/opt017_mixer_mma.json),
[`fixtures/opt018_ffn_mma.json`](../fixtures/opt018_ffn_mma.json),
[`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json),
[`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json),
[`fixtures/opt021_oracle.json`](../fixtures/opt021_oracle.json),
[`fixtures/opt022_mixer_q8_quality.json`](../fixtures/opt022_mixer_q8_quality.json),
[`fixtures/opt023_skinny_mixer.json`](../fixtures/opt023_skinny_mixer.json),
[`fixtures/opt024_mixer_q8_d2r.json`](../fixtures/opt024_mixer_q8_d2r.json),
[`fixtures/opt025_ffn_shared_y.json`](../fixtures/opt025_ffn_shared_y.json),
[`fixtures/opt026_fattn_streamk.json`](../fixtures/opt026_fattn_streamk.json),
[`fixtures/opt027_persistent_fattn.json`](../fixtures/opt027_persistent_fattn.json),
[`fixtures/opt028_mmq_streamk.json`](../fixtures/opt028_mmq_streamk.json),
[`fixtures/opt029_gdn_fuse.json`](../fixtures/opt029_gdn_fuse.json),
[`fixtures/opt030_pdl_launches.json`](../fixtures/opt030_pdl_launches.json),
[`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json),
[`fixtures/opt033_register_vkq.json`](../fixtures/opt033_register_vkq.json),
[`fixtures/opt035_pv_mma.json`](../fixtures/opt035_pv_mma.json),
[`evidence/optimization/opt015-2k-recovery/REPORT.md`](../evidence/optimization/opt015-2k-recovery/REPORT.md),
[`evidence/optimization/opt017-mixer-q8-mma/REPORT.md`](../evidence/optimization/opt017-mixer-q8-mma/REPORT.md),
[`evidence/optimization/opt018-ffn-mma-quality/REPORT.md`](../evidence/optimization/opt018-ffn-mma-quality/REPORT.md),
[`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md),
[`evidence/optimization/opt021-4k-oracle/REPORT.md`](../evidence/optimization/opt021-4k-oracle/REPORT.md),
[`evidence/optimization/opt022-mixer-q8-quality/REPORT.md`](../evidence/optimization/opt022-mixer-q8-quality/REPORT.md),
[`evidence/optimization/opt023-skinny-mixer/REPORT.md`](../evidence/optimization/opt023-skinny-mixer/REPORT.md),
[`evidence/optimization/opt024-mixer-q8-d2r/REPORT.md`](../evidence/optimization/opt024-mixer-q8-d2r/REPORT.md),
[`evidence/optimization/opt025-ffn-shared-y/REPORT.md`](../evidence/optimization/opt025-ffn-shared-y/REPORT.md),
[`evidence/optimization/opt026-fattn-streamk/REPORT.md`](../evidence/optimization/opt026-fattn-streamk/REPORT.md),
[`evidence/optimization/opt027-persistent-fattn/REPORT.md`](../evidence/optimization/opt027-persistent-fattn/REPORT.md),
[`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](../evidence/optimization/opt027-persistent-fattn/REJECTION.md),
[`evidence/optimization/opt028-mmq-streamk/REPORT.md`](../evidence/optimization/opt028-mmq-streamk/REPORT.md),
[`evidence/optimization/opt028-mmq-streamk/REJECTION.md`](../evidence/optimization/opt028-mmq-streamk/REJECTION.md),
[`evidence/optimization/opt029-gdn-fuse/REPORT.md`](../evidence/optimization/opt029-gdn-fuse/REPORT.md),
[`evidence/optimization/opt029-gdn-fuse/REJECTION.md`](../evidence/optimization/opt029-gdn-fuse/REJECTION.md),
[`evidence/optimization/opt030-pdl-launches/REPORT.md`](../evidence/optimization/opt030-pdl-launches/REPORT.md),
[`evidence/optimization/opt030-pdl-launches/REJECTION.md`](../evidence/optimization/opt030-pdl-launches/REJECTION.md),
[`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md),
[`evidence/optimization/opt033-register-vkq/REPORT.md`](../evidence/optimization/opt033-register-vkq/REPORT.md),
[`evidence/optimization/opt035-pv-mma/REPORT.md`](../evidence/optimization/opt035-pv-mma/REPORT.md),
[`fixtures/opt040_gdn_shared_inverse.json`](../fixtures/opt040_gdn_shared_inverse.json),
[`evidence/optimization/opt040-gdn-shared-inverse/REPORT.md`](../evidence/optimization/opt040-gdn-shared-inverse/REPORT.md),
[`fixtures/opt041_fattn_warp_qk.json`](../fixtures/opt041_fattn_warp_qk.json),
[`evidence/optimization/opt041-fattn-warp-qk/REPORT.md`](../evidence/optimization/opt041-fattn-warp-qk/REPORT.md)

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

Quartz therefore keeps a Q8_0-by-BF16 kernel family. SCH-002's first version
processed multiple rows by mapping `blockIdx.y` to one prompt row
(`grid.y = prompt_rows`). That batched launches without reusing weights: each
output-row warp reread the entire packed matrix. OPT-009 introduced
`launch_q8_mmq_bf16_variant`: `blockIdx.y` owns a prompt-row tile, one Q8_0
weight is decoded per column, and that scalar is applied to every in-range
prompt row with the decode `__fmul_rn` / `__fadd_rn` walk. Activations stay
BF16. The row-wise kernel is retained as `launch_q8_mmq_bf16_reference`.
Captured OPT-009 graphs at 64 and 4,096 prompt rows show tiled `grid.y` of 16
and 1,024 (selected tile 4) versus reference `grid.y` equal to the prompt-row
count.

OPT-017 admitted Rank-1 fused MMA behind `launch_q8_mmq_bf16` when
`prompt_rows >= 8`. OPT-022 replaces that production body with quality MMA
(D4 `quantize_mmq_q8_1`, packed load-tiles, `MMQ_ITER_K=256`, `dim3(32, 8)`,
J=128) and shares one residual D4 Y per layer for mixer GEMMs that read
`prompt_normalized_`. Mixer projections listed in
[`src/weights.cpp`](../src/weights.cpp) (GDN packed_qkv, value_gate, alpha/beta,
output; attention query_gate, key, value) follow that mixer Q8_0 path.
OPT-023 later templates quality I=32 for mixer Q8_0 `output_rows < 128`
(GDN α/β) when that A/B wins. OPT-024 A/B'd aligned-SoA D2R for large mixer
Q8_0 (`output_rows >= 128`) and did not install it; large mixer GEMMs stay
I=128 quality MMA.
Attention output remains Q6_K `launch_quant_mmq`. OPT-028 A/B'd
llama.cpp-style stream-K plus optional fixup for Q4_K/Q6_K quality MMA
and did not install it; production remains 2D tiling. Production still does not
send Q8_0 through `launch_quant_mmq` Q8Block staging. Quality MMA is admitted
under the Q8 association rule versus host CPU dequant GEMM. Rank-1 CUD-002
numbers stay the staged-Q8 Rank-1 gate and are not the quality-path gate.
The OPT-009 byte-exact pair stays variant versus reference.

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

Production prompt chunks use `GdnScanPath::kFusedTokenLoop`: the existing
parallel convolution, then optional hoisted per-(token, key_head) Q/K inverses
in the `prompt_projected_bf16_` overlay (OPT-040), then OPT-019 warp-column
fused quality recurrence (`dim3(32, 4)`, grid z=32 for width 128). Sequential
64-token windows remain
the unloosened GDN-002/OPT-013 reference. OPT-013's associative overlay
(`kParallelAssociative`) is retained: at `prompt_chunk_rows_ == 4096` that
overlay fits `W_fit = 22` windows, so a 4,096-token layer batches
`22 + 22 + 20` when that path is selected. Tails with one window
(`2 ≤ token_count ≤ 64`) keep sequential recurrence after a single 2D
convolution on the overlay/sequential paths. Rank-2 fused remains for
component A/B only. Decode one-token GDN stays sequential. Prompt FFN graphs
still exclude GDN.

OPT-008's 4,096-versus-64-row memcmp remains a **sequential** GDN gate: that
comparison pins `GdnScanPath::kSequentialWindows` rather than loosening
byte equality to tolerances. Parallel cross-boundary proof is the OPT-013
diagnostic at the frozen GDN-002 envelopes (`5e-8` / `5e-9` / zero non-finite),
including 4,096 parallel tokens versus 64 sequential windows.

Attention also visits chunk rows in order. Production
`launch_attention_prepare_chunk` uses fattn-mma quality when `token_count >= 16`
(ncols1=16, ncols2=2) and the two-row tiled path otherwise. Prompt scheduler
attention uses Ada+ stream-K after the OPT-026 A/B win, aliasing existing
workspace for the softmax combine, with register-resident value sums after
the OPT-033 keep, dual-F16 probability×V MMA after the OPT-035 keep, and
warp-owned QK microtiles after the OPT-041 keep. OPT-027 measured persistent
stream-K against that path and did not install it; production remains
`stream_k` (`grid.z=2`) with `kSelectedVkqAccum = "registers"`,
`kSelectedPvPath = "mma"`, and `kSelectedQKPath = "warp_microtile"`. The tiled
path remains the unloosened OPT-005 numeric reference. Rank-3 warp-0 MMA is
retained for A/B. A row may read all committed KV rows
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
launches because those pointers and extents still change per chunk. OPT-030
does not apply Hopper/Blackwell PDL inside those captured FFN graphs and does
not recapture them. While a stream is capturing, the PDL wrapper takes the
ordinary `<<<>>>` branch so graph nodes stay unattributed kernel launches.
Graph-versus-ordinary fused byte equality is unloosened. Production PDL
remains off after the 4K reject (`kSelectedPdlPath` `off`).

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

OPT-017 admits production mixer Q8_0 MMA behind the existing
`launch_q8_mmq_bf16` name. **Measured, RTX 5090:** CUD-002 versus
`launch_quant_mmq_variant` `kQ8_0` stays inside max abs `5e-4` and RMS
`2.5e-4` with zero non-finites; J is pinned at 128; OPT-009
variant-versus-reference byte equality is unloosened. A live unperturbed
exact-2048 remasurement mean was 375.743988 tok/s versus live llama.cpp
3203.276277 tok/s. Post-remasurement attribution wall 5383.53369 ms had
`gdn` 1647.58936 ms, `attention` 1207.48132 ms, and `ffn_mmq` 2525.01465 ms.
**OPT-016 remains the 2K parity owner.** This remasurement is not that gate
and is not an 8K/32K/128K throughput claim. Artifacts:
[`evidence/optimization/opt017-mixer-q8-mma/REPORT.md`](../evidence/optimization/opt017-mixer-q8-mma/REPORT.md),
[`pins/opt017_mixer_mma_contract.json`](../pins/opt017_mixer_mma_contract.json),
and [`fixtures/opt017_mixer_mma.json`](../fixtures/opt017_mixer_mma.json).

OPT-018 closes the Q4_K/Q6_K production MMQ quality path. **Measured, RTX 5090:**
admission is host CPU dequant GEMM under the ds4 Q4_K association rule
(option C); fixed abs/rms are retired for that Q4_K/Q6_K MMQ gate; the scalar
variant keeps exact Q8 staging. J is pinned at 128. Component Q4
`2048×17408×5120` MMA 1.64595187 ms versus variant 155.836014 ms. Live
exact-2048 `ffn_mmq` **399.287018** ms versus before **2524.67725** ms and live
llama.cpp 2K wall **636.184782** ms (`avg_ts` 3219.6604). Unperturbed Quartz
mean 625.792114 tok/s. `llama_competitive` is true; `would_pass_opt016` is
informational false. **OPT-016 remains the 2K parity owner.** This remasurement
is not that gate and is not an 8K/32K/128K throughput claim. Artifacts:
[`evidence/optimization/opt018-ffn-mma-quality/REPORT.md`](../evidence/optimization/opt018-ffn-mma-quality/REPORT.md),
[`pins/opt018_ffn_mma_contract.json`](../pins/opt018_ffn_mma_contract.json),
and [`fixtures/opt018_ffn_mma.json`](../fixtures/opt018_ffn_mma.json).

OPT-019 closes residual non-projection GDN scan/recurrence and causal attention
core time. **Measured, RTX 5090:** quality GDN versus sequential at 2048 tokens
stayed inside `max_abs=1.58324838e-08` and `rms=1.28841632e-10` with zero
non-finites; quality attention versus tiled stayed inside
`max_abs=3.16649675e-06` and `rms=1.31638146e-07`. ncols1 pin is 16. Live
exact-2048 `gdn` **1205.38806** ms and `attention` **475.116028** ms versus
locked befores **1643.46082** / **1148.47461** ms (combined **1680.504088** ms
versus **2791.93543** ms) meet the 85%/85%/70% addressed rule. Unperturbed
Quartz mean 978.151855 tok/s versus live llama.cpp 3197.246224 tok/s.
`owns_opt016_parity_gate` is false; `would_pass_opt016` is informational false.
**OPT-016 remains the 2K parity owner.** This remasurement is not that gate
and is not an 8K/32K/128K throughput claim. Artifacts:
[`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md),
[`pins/opt019_core_recovery_contract.json`](../pins/opt019_core_recovery_contract.json),
and [`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json).

OPT-020 splits mixer-projection MMQ out of the former composite GDN and
attention buckets. **Measured, RTX 5090:** one cold empty-session 2048-token
production `sync_tokens` (capacity 131072, fused overlapped, production fused
GDN, graphs created) took 2086.2561 ms host wall (~981.66 tok/s) at
2026-09-08T21:12:38Z. Exclusive CUDA events on the prompt compute stream
recorded mixer-projection MMQ 1199.25122 ms, FFN/MMQ 397.254333 ms,
attention_core 273.986725 ms, gdn_core 212.156006 ms, logits 2.33337593 ms,
commit/sync 0.590431988 ms, and embedding 0.0736320019 ms. Graph was measured
0 ms with zero prompt-graph launches. The other/idle remainder was
0.610351562 ms, so the nine named categories reconstruct wall within
`rel_tol = 1e-4` and `abs_tol_ms = 0.05`. Nsight Systems and Nsight Compute were
`not_used`. This is instrumentation of that timed run, not a throughput gate
and not llama.cpp parity. The historical eight-category snapshot remains
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json).
Chapter 51 owns the live category definitions. Artifacts:
[`pins/opt020_prefill_split_contract.json`](../pins/opt020_prefill_split_contract.json)
and [`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json).

OPT-021 pins the exclusive-RTX-5090 **4K keep/reject oracle**. Production
prompt FFN graphs replay only when `token_count == 4096` and created graphs
are passed into `sync_tokens`; exact-2048 does not replay those 4096-row
graphs. **Measured, RTX 5090:** one exclusive sitting, llama.cpp first then
Quartz, records three cold exact-4096 unperturbed `sync_tokens` walls
(attribution null, graphs created, production fused GDN and overlapped path)
and live `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99`. Quartz mean
**966.039062** tok/s versus llama.cpp **3224.522433** tok/s is the retained
baseline. Later production changes keep only when cold exact-4096 mean tok/s
is strictly greater than that Quartz mean. `quartz_meets_llama` is
informational and is not this gate. **OPT-016 remains the 2K parity owner.**
This oracle does not substitute for that gate and does not claim Quartz ≥
llama.cpp. Scout sitting 965.204895 / 3182.476587 tok/s is **not** the
retained fixture. Live numbers stay in the report:
[`evidence/optimization/opt021-4k-oracle/REPORT.md`](../evidence/optimization/opt021-4k-oracle/REPORT.md),
[`pins/opt021_oracle_contract.json`](../pins/opt021_oracle_contract.json),
and [`fixtures/opt021_oracle.json`](../fixtures/opt021_oracle.json).

OPT-022 lands production mixer Q8_0 quality MMQ with shared residual Y.
**Measured, RTX 5090:** D4 `quantize_mmq_q8_1`, packed load-tiles,
`MMQ_ITER_K=256`, `dim3(32, 8)`, J=128 when `prompt_rows >= 8`; mixer inputs
share one residual D4 Y per layer; GDN output requantizes at K=6144 into the
same `prompt_q8_`. Quality MMA versus CPU dequant GEMM uses the Q8 association
rule. The OPT-009 tiled-versus-reference pair remains byte-exact. Live
exclusive sitting keep: Quartz mean **1680.38025** tok/s versus the frozen
oracle baseline **967.267761**; `reverted` false; `successor_oracle` true.
`quartz_meets_llama` is informational and is not this gate. **OPT-016 remains
the 2K parity owner.** This keep does not substitute for that gate and does
not claim Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt022-mixer-q8-quality/REPORT.md`](../evidence/optimization/opt022-mixer-q8-quality/REPORT.md),
[`pins/opt022_mixer_q8_quality_contract.json`](../pins/opt022_mixer_q8_quality_contract.json),
and [`fixtures/opt022_mixer_q8_quality.json`](../fixtures/opt022_mixer_q8_quality.json).

OPT-023 lands skinny-M mixer dispatch for mixer Q8_0 `output_rows < 128`
(GDN α/β). **Measured, RTX 5090:** paired CUDA-event A/B on
`4096×48×5120` among `mma_i128_j128`, `mma_i32_j128`, `mma_i64_j128`, and
`mmv_tiled_j1`; winner `mma_i32_j128` is strictly faster than I=128
Fallback. Production `launch_q8_mmq_quality_mma` templates I=32 for those
skinny GEMMs on the shared residual D4 Y. Large mixer Q8 GEMMs remain
I=128 / J=128 quality MMA with shared Y. Decode MMV is unchanged. The
OPT-009 tiled-versus-reference pair remains byte-exact. Live exclusive
sitting keep: Quartz mean strictly greater than the frozen post-OPT-022
oracle baseline **1680.80627**; `reverted` false; `successor_oracle` true;
`production_skinny` true. `quartz_meets_llama` is informational and is not
this gate. **OPT-016 remains the 2K parity owner.** This keep does not
substitute for that gate and does not claim Quartz ≥ llama.cpp. Live
numbers stay in the report:
[`evidence/optimization/opt023-skinny-mixer/REPORT.md`](../evidence/optimization/opt023-skinny-mixer/REPORT.md),
[`pins/opt023_skinny_mixer_contract.json`](../pins/opt023_skinny_mixer_contract.json),
and [`fixtures/opt023_skinny_mixer.json`](../fixtures/opt023_skinny_mixer.json).

OPT-024 A/B'd a ds4-inspired aligned-SoA D2R / int8 MMA path for mixer Q8_0
`output_rows >= 128`. **Measured, RTX 5090:** paired CUDA-event A/B on the
five production large mixer shapes among `quality_mma` and `d2r_soa`; D2R
did not strictly beat quality MMA on every timed shape. Production large
mixers remain I=128 / J=128 quality MMA with
shared residual D4 Y. Skinny α/β stay `mma_i32_j128`. Decode MMV stays on
GGUF 34-byte blocks. The OPT-009 tiled-versus-reference pair remains
byte-exact. Live exclusive sitting **reject:** Quartz mean is not strictly
greater than the frozen successor-oracle baseline **1687.86169**;
`reverted` true; `successor_oracle` false; `production_d2r` false; A/B
winner `quality_mma`. `quartz_meets_llama` is informational and is not this
gate. **The 2K parity owner remains the blocked dedicated gate.** This
reject does not substitute for that gate and does not claim Quartz ≥
llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt024-mixer-q8-d2r/REPORT.md`](../evidence/optimization/opt024-mixer-q8-d2r/REPORT.md),
[`pins/opt024_mixer_q8_d2r_contract.json`](../pins/opt024_mixer_q8_d2r_contract.json),
[`fixtures/opt024_mixer_q8_d2r.json`](../fixtures/opt024_mixer_q8_d2r.json),
and
[`evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md`](../evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md).

OPT-025 lands dense prompt FFN shared-Y and SwiGLU-into-down Q8 for Q4_K
gate/up/down. **Measured, RTX 5090:** paired CUDA-event A/B on the production
FFN shapes among `baseline`, `shared_y`, `swiglu_q8`, and
`shared_y_swiglu_q8`; winner `shared_y_swiglu_q8` is strictly faster than
per-GEMM quantize plus BF16 SwiGLU. Production `execute_prompt_ffn` (fused
graphs and unfused) quantizes the FFN-norm activation once for gate and up
and writes down-leg DS4 Y from SwiGLU without a global BF16 mid store.
Mixer Q8 quality MMA, skinny `mma_i32_j128`, and the D2R reject stay.
Decode FFN stays MMV. The OPT-009 tiled-versus-reference pair remains
byte-exact. Live exclusive sitting keep: Quartz mean strictly greater than
the frozen successor-oracle baseline **1687.86169**; `reverted` false;
`successor_oracle` true; `production_ffn_optimized` true. `quartz_meets_llama`
is informational and is not this gate. **The 2K parity owner remains the
blocked dedicated gate.** This keep does not substitute for that gate and
does not claim Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt025-ffn-shared-y/REPORT.md`](../evidence/optimization/opt025-ffn-shared-y/REPORT.md),
[`pins/opt025_ffn_shared_y_contract.json`](../pins/opt025_ffn_shared_y_contract.json),
and [`fixtures/opt025_ffn_shared_y.json`](../fixtures/opt025_ffn_shared_y.json).

OPT-037 swept 4096-row Q4_K FFN quality MMA I∈{64,128} × J∈{32,64,128}
independently for gate, up, and down while keeping shared-Y /
SwiGLU-into-Q8 and 2D scheduling. **Measured, RTX 5090:** paired
CUDA-event A/B (3 warm-ups, 30 measured rounds, MMA only) selected
`i128_j128` on every projection (`any_win=false`). Production pins stay
I=128 / J=128. Prompt FFN graphs were not recaptured. Mixer Q8 quality,
skinny `mma_i32_j128`, packed decode MMV, and fattn stream-K stay.
Decode FFN stays MMV. Live exclusive sitting **reject:** no admitted
component win; tok/s sitting skipped; `reverted` true. Keep denominators
are the frozen OPT-034 P / D128 / D2048 oracles. `quartz_meets_llama` is
informational and is not this gate. **The 2K parity owner remains the
blocked dedicated gate.** Live numbers stay in the report:
[`evidence/optimization/opt037-ffn-tiles/REPORT.md`](../evidence/optimization/opt037-ffn-tiles/REPORT.md),
[`pins/opt037_ffn_tile_contract.json`](../pins/opt037_ffn_tile_contract.json),
[`fixtures/opt037_ffn_tiles.json`](../fixtures/opt037_ffn_tiles.json),
and
[`evidence/optimization/opt037-ffn-tiles/REJECTION.md`](../evidence/optimization/opt037-ffn-tiles/REJECTION.md).

OPT-026 lands Ada+ fattn stream-K occupancy for prompt attention. **Measured,
RTX 5090:** paired CUDA-event A/B on production 4096-row fattn among
`baseline`, `occ2`, and `stream_k`; winner `stream_k` is strictly faster than
occupancy-1 whole-tile under frozen OPT-005 envelopes with score scratch
untouched. Production scheduler attention aliases existing
`prompt_projected_bf16_` and `prompt_q8_` for the KV-bipartition combine.
ncols1 stays 16. Mixer Q8 quality, skinny `mma_i32_j128`, and FFN
`shared_y_swiglu_q8` stay. Decode attention stays one-token. The tiled path
remains the unloosened OPT-005 reference. Live exclusive sitting keep: Quartz
mean strictly greater than the frozen successor-oracle baseline
**1709.21912**; `reverted` false; `successor_oracle` true;
`production_fattn_optimized` true; `ladder_exhausted` true.
`quartz_meets_llama` is informational and is not this gate. **The 2K parity
owner remains the blocked dedicated gate.** This keep does not substitute for
that gate and does not claim Quartz ≥ llama.cpp. The 4K idea ladder is
exhausted. Live numbers stay in the report:
[`evidence/optimization/opt026-fattn-streamk/REPORT.md`](../evidence/optimization/opt026-fattn-streamk/REPORT.md),
[`pins/opt026_fattn_streamk_contract.json`](../pins/opt026_fattn_streamk_contract.json),
and [`fixtures/opt026_fattn_streamk.json`](../fixtures/opt026_fattn_streamk.json).

OPT-027 measured llama.cpp Ada+ persistent fattn stream-K
(`nsm × occupancy` linearized tiles, 5% efficiency rounding,
uniform/general fixup). **Measured, RTX 5090:** paired CUDA-event A/B on
production 4096-row fattn among `stream_k` and `persistent`; winner
`stream_k` (`win=false`). Production scheduler attention remains OPT-026
`stream_k` (`grid.z=2`). Persistent kernels remain as non-production
symbols. `kSelectedFattnPath` stays `"stream_k"`;
`kSelectedPersistentFattnPath` is `off`. ncols1 stays 16. Mixer Q8
quality, skinny `mma_i32_j128`, and FFN `shared_y_swiglu_q8` stay. Decode
attention stays one-token. The tiled path remains the unloosened OPT-005
reference. Live exclusive sitting **reject:** Quartz mean is not strictly
greater than the frozen successor-oracle baseline **1746.71973**;
`reverted` true; `successor_oracle` false;
`production_persistent_installed` false; A/B winner `stream_k`;
`ladder_exhausted` false. `quartz_meets_llama` is informational and is
not this gate. **The 2K parity owner remains the blocked dedicated
gate.** This reject does not substitute for that gate and does not claim
Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt027-persistent-fattn/REPORT.md`](../evidence/optimization/opt027-persistent-fattn/REPORT.md),
[`pins/opt027_persistent_fattn_contract.json`](../pins/opt027_persistent_fattn_contract.json),
[`fixtures/opt027_persistent_fattn.json`](../fixtures/opt027_persistent_fattn.json),
and
[`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](../evidence/optimization/opt027-persistent-fattn/REJECTION.md).

OPT-028 measured llama.cpp-style Q4_K/Q6_K MMQ stream-K (linearized `kbc`
tile walk plus optional fixup). **Measured, RTX 5090:** paired CUDA-event
A/B on one production 4096-row FFN among `off`, `stream_k`, and
`stream_k_nsm`; winner `off` (`win=false`). Production quality MMA remains
2D tiling. Stream-K kernels remain as non-production symbols.
`kSelectedMmqStreamKPath` is `off`. `kSelectedFfnPath` stays
`"shared_y_swiglu_q8"`. Mixer Q8 quality, skinny `mma_i32_j128`, and fattn
Ada+ stream-K stay. Decode FFN stays MMV. J stays 128. Scheduler fixup
aliases exist and are unused while the selected path is `off`. No extra
persistent `cudaMalloc`. Prompt FFN graphs were not recaptured. Live
exclusive sitting **reject:** Quartz mean is not strictly greater than the
frozen successor-oracle baseline **1746.71973**, and the A/B lost;
`reverted` true; `successor_oracle` false;
`production_mmq_stream_k_installed` false; A/B winner `off`;
`ladder_exhausted` false. `quartz_meets_llama` is informational and is
not this gate. **The 2K parity owner remains the blocked dedicated
gate.** This reject does not substitute for that gate and does not claim
Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt028-mmq-streamk/REPORT.md`](../evidence/optimization/opt028-mmq-streamk/REPORT.md),
[`pins/opt028_mmq_streamk_contract.json`](../pins/opt028_mmq_streamk_contract.json),
[`fixtures/opt028_mmq_streamk.json`](../fixtures/opt028_mmq_streamk.json),
and
[`evidence/optimization/opt028-mmq-streamk/REJECTION.md`](../evidence/optimization/opt028-mmq-streamk/REJECTION.md).

OPT-029 measured collapsing tiled causal convolution and/or gated-output
into the warp-column fused GDN token loop. Candidates were `off`,
`fuse_conv`, `fuse_gate`, and `fuse_both`. Eligible fused ids met frozen
GDN-002 envelopes versus sequential; gated BF16 versus split gated-output
on the same recurrent also met that envelope. Occupancy was ≥ 1. The
paired CUDA-event A/B on production 4096-token GDN core (conv +
recurrence + gated-output) did not find an optimized id strictly faster
than `off` (`win=false`). Production GDN core remains split parallel conv
+ warp-column + `gdn_gated_output_rows`. Fused kernels remain as
non-production symbols. `kSelectedGdnFusePath` is `off`.
`GdnScanPath::kFusedTokenLoop` stays the production scan enum. Mixer Q8
quality, skinny `mma_i32_j128`, FFN `shared_y_swiglu_q8`, and fattn
Ada+ stream-K stay. Decode GDN is unchanged. Sequential windows remain
the unloosened reference. No extra persistent `cudaMalloc`. Live
exclusive sitting **reject:** Quartz mean is not strictly greater than the
frozen successor-oracle baseline **1746.71973**, and the A/B lost;
`reverted` true; `successor_oracle` false;
`production_gdn_fuse_installed` false; A/B winner `off`;
`ladder_exhausted` false. `quartz_meets_llama` is informational and is
not this gate. **The 2K parity owner remains the blocked dedicated
gate.** This reject does not substitute for that gate and does not claim
Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt029-gdn-fuse/REPORT.md`](../evidence/optimization/opt029-gdn-fuse/REPORT.md),
[`pins/opt029_gdn_fuse_contract.json`](../pins/opt029_gdn_fuse_contract.json),
[`fixtures/opt029_gdn_fuse.json`](../fixtures/opt029_gdn_fuse.json),
and
[`evidence/optimization/opt029-gdn-fuse/REJECTION.md`](../evidence/optimization/opt029-gdn-fuse/REJECTION.md).

OPT-030 measured Hopper/Blackwell programmatic dependent launch (PDL)
serialization of successive ungraphed mixer/GDN/attention prompt kernels
on `prompt_compute_stream_`. Candidates were `off` (ordinary `<<<>>>`),
`pdl_host` (`cudaLaunchKernelEx` plus programmatic stream
serialization), and `pdl` (PSS plus device grid-dependency sync/LC).
Eligible candidates were byte-identical versus `off` on the ungraphed
chain. Occupancy was ≥ 1. Capture of `execute_prompt_ffn` stayed ordinary
kernel nodes (`used_ex=false`). The paired CUDA-event A/B on one
4096-token GDN layer plus one 4096-token attention layer ungraphed chain
won with `pdl` (`win=true`). Production did not install that path.
`kSelectedPdlPath` is `off`. The PDL wrapper remains as a non-production
symbol. Mixer Q8 quality, skinny `mma_i32_j128`, FFN `shared_y_swiglu_q8`,
fattn Ada+ stream-K, and split GDN core stay. Decode launches are
unchanged. FFN graphs were not recaptured and are not PDL-attributed.
No extra persistent `cudaMalloc`. Live exclusive sitting **reject:**
Quartz mean is not strictly greater than the frozen successor-oracle
baseline **1746.71973**; A/B won but keep requires both an A/B win and
a strictly greater 4K mean; `reverted` true; `successor_oracle` false;
`production_pdl_installed` false; A/B winner `pdl`;
`ladder_exhausted` false. `quartz_meets_llama` is informational and is
not this gate. **The 2K parity owner remains the blocked dedicated
gate.** This reject does not substitute for that gate and does not claim
Quartz ≥ llama.cpp. Live numbers stay in the report:
[`evidence/optimization/opt030-pdl-launches/REPORT.md`](../evidence/optimization/opt030-pdl-launches/REPORT.md),
[`pins/opt030_pdl_launches_contract.json`](../pins/opt030_pdl_launches_contract.json),
[`fixtures/opt030_pdl_launches.json`](../fixtures/opt030_pdl_launches.json),
and
[`evidence/optimization/opt030-pdl-launches/REJECTION.md`](../evidence/optimization/opt030-pdl-launches/REJECTION.md).

OPT-032 refreshes live exclusive 4K prefill attribution on one cold
exact-4096 production `sync_tokens` with graphs created. At 4096, prompt FFN
graphs replay, so `graph` is measured and `prompt_graph_launches == 64`. The
P throughput oracle in that sitting still uses attribution null and graphs
created (the OPT-021 protocol). Historical 2048-token mixer-versus-core
[`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json)
is unchanged. Exclusive decode categories on one production token after D128
and D2048 prefixes are owned by chapter 51; they are not tok/s oracles.
This increment **claims no performance improvement** and does not substitute
for the 2K llama.cpp parity gate. Live tok/s and exclusive milliseconds stay
in the report:
[`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md),
[`pins/opt032_decode_oracle_contract.json`](../pins/opt032_decode_oracle_contract.json),
and [`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json).

OPT-038 refreshes live exclusive 4K prefill attribution on one cold
exact-4096 production `sync_tokens` with graphs created on current production
objects after the OPT-033–OPT-037 ladder. At 4096, prompt FFN graphs replay, so
`graph` is measured and `prompt_graph_launches == 64`. The P throughput oracle
in that sitting still uses attribution null and graphs created (the OPT-021
protocol). New OPT-038 diagnostics additionally record independent
`raw_host_wall_ms` alongside the unchanged `finish_prefill_attribution`
adjusted reconstruction. Historical 2048-token mixer-versus-core
[`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json)
and OPT-032 attribution fixtures are unchanged. Exclusive decode categories on
one production token after D128 and D2048 prefixes are owned by chapter 51;
decode exclusive milliseconds for the live sitting live in the post-ladder gap
fixture. This increment **claims no performance improvement**, does not publish a
successor oracle, and does not substitute for the 2K llama.cpp parity gate.
Accepted keep denominators remain the frozen OPT-034 copies. Live tok/s and
exclusive milliseconds stay in the report:
[`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](../evidence/optimization/opt038-post-ladder-gap/REPORT.md),
[`pins/opt038_post_ladder_gap_contract.json`](../pins/opt038_post_ladder_gap_contract.json),
and
[`fixtures/opt038_post_ladder_gap.json`](../fixtures/opt038_post_ladder_gap.json).

OPT-033 keeps register-resident value accumulation on production 4096-row
fattn-mma stream-K including combine. **Measured, RTX 5090:** paired
CUDA-event A/B among `global` and `registers`; winner `registers` with
byte-equal combined output and strictly lower component time. Production
`kSelectedVkqAccum` is `registers`. Ncols1 stays 16, Ncols2 stays 2, KV
tile stays 32, dual-F16 Q stays, and `grid.z` stays 2. Scalar
probability×V and the combine kernel stay. Decode 16/16 partitions stay.
Mixer Q8 quality, skinny `mma_i32_j128`, FFN `shared_y_swiglu_q8`, and
GDN warp-column stay. Tiled `launch_attention_prepare_chunk_tiled`
remains the unloosened OPT-005 reference. Live exclusive sitting **keep:**
Quartz P strictly greater than the frozen then-current accepted
denominator **1644.04822**; D128/D2048 throughput and p95 guards held;
`reverted` false; `keep_sitting_skipped` false; `status` measured.
D2048 tok/s improvement is not required for this prefill keep.
`quartz_meets_llama` is informational and is not this gate. **The 2K
parity owner remains the blocked dedicated gate.** This keep does not
substitute for that gate and does not claim Quartz ≥ llama.cpp. Live
numbers stay in the report:
[`evidence/optimization/opt033-register-vkq/REPORT.md`](../evidence/optimization/opt033-register-vkq/REPORT.md),
[`pins/opt033_register_vkq_contract.json`](../pins/opt033_register_vkq_contract.json),
and [`fixtures/opt033_register_vkq.json`](../fixtures/opt033_register_vkq.json).

OPT-035 keeps dual-F16 probability×V MMA on production 4096-row fattn-mma
stream-K including combine, on top of register-resident VKQ. **Measured,
RTX 5090:** paired CUDA-event A/B among `scalar` and `mma`; winner `mma`
with frozen OPT-005 envelopes versus tiled and strictly lower component
time. Production `kSelectedPvPath` is `mma`. Ncols1 stays 16, Ncols2 stays
2, KV tile stays 32, dual-F16 Q stays, `grid.z` stays 2, and register VKQ
stays. Dual-F16 V is out of scope. The combine kernel stays. Decode 16/16
partitions stay. Mixer Q8 quality, skinny `mma_i32_j128`, FFN
`shared_y_swiglu_q8`, and GDN warp-column stay. Tiled
`launch_attention_prepare_chunk_tiled` remains the unloosened OPT-005
reference. Byte equality versus scalar is not this keep predicate. Live
exclusive sitting **keep:** Quartz P strictly greater than the frozen
then-current accepted denominator **1745.10315**; D128/D2048 throughput
and p95 guards held; `reverted` false; `keep_sitting_skipped` false;
`status` measured. D2048 tok/s improvement is not required for this
prefill keep. `quartz_meets_llama` is informational and is not this gate.
**The 2K parity owner remains the blocked dedicated gate.** This keep does
not substitute for that gate and does not claim Quartz ≥ llama.cpp. Live
numbers stay in the report:
[`evidence/optimization/opt035-pv-mma/REPORT.md`](../evidence/optimization/opt035-pv-mma/REPORT.md),
[`pins/opt035_pv_mma_contract.json`](../pins/opt035_pv_mma_contract.json),
and [`fixtures/opt035_pv_mma.json`](../fixtures/opt035_pv_mma.json).

OPT-040 hoists prompt GDN Q/K inverse normalization into existing overlay
scratch. **Measured, RTX 5090:** paired CUDA-event A/B on the complete
4096-token GDN component (parallel conv + optional shared-inverse kernel +
warp-column recurrence + gated-output) among `repeated` and `shared`; winner
`shared` with byte-equal quality versus `repeated` and frozen sequential
GDN-002 envelopes. Inverse scratch aliases `prompt_projected_bf16_` as a float
overlay between parallel conv and recurrence; gated-output overwrites that
buffer afterward. Required floats are `2 * token_count * key_heads`; the
OPT-008 workspace byte formula is unchanged. `token_count == 1` tails stay
`repeated`. Decode sequential GDN and OPT-029 fuse `off` stay. Mixer Q8 quality,
skinny `mma_i32_j128`, FFN `shared_y_swiglu_q8`, fattn stream-K, register VKQ,
and P×V MMA stay. Live exclusive sitting **keep:** Quartz P strictly greater
than the frozen then-current accepted P denominator; D128/D2048 throughput and
p95 guards held; `reverted` false; `keep_sitting_skipped` false; `status`
measured. `quartz_meets_llama` is informational and is not this gate. **The
2K parity owner remains the blocked dedicated gate.** This keep does not
substitute for that gate and does not claim Quartz ≥ llama.cpp. Live numbers
stay in the report:
[`evidence/optimization/opt040-gdn-shared-inverse/REPORT.md`](../evidence/optimization/opt040-gdn-shared-inverse/REPORT.md),
[`pins/opt040_gdn_shared_inverse_contract.json`](../pins/opt040_gdn_shared_inverse_contract.json),
and
[`fixtures/opt040_gdn_shared_inverse.json`](../fixtures/opt040_gdn_shared_inverse.json).

OPT-052 hoists scaled Q/K and decay into the same overlay and measures a
conversion-inclusive column-major state tile. **Measured, RTX 5090:** paired CUDA-event A/B on the complete 4096-token GDN component
(parallel conv + preprocessing + warp-column recurrence + gated-output)
among `shared`, `preproc`, `preproc_fma`, `approx_exp`, and `transpose`;
winner `transpose` with byte-equal quality versus `shared` and frozen
sequential GDN-002 envelopes. Production `kSelectedGdnPreprocPath` is
`transpose`. Shared inverses, fuse `off`, sequential windows, and decode
GDN stay. FMA and approximate exp remain separate measured variants, not
silent replacements. Live exclusive sitting **keep:** Quartz P strictly
greater than the copied OPT-051 keep; D128/D2048 throughput and p95
guards held; `reverted` false; `keep_sitting_skipped` false; `status`
measured. `quartz_meets_llama` is informational and is not this gate.
**The 2K parity owner remains the blocked dedicated gate.** Live numbers
stay in the report:
[`evidence/optimization/opt052-gdn-arithmetic/REPORT.md`](../evidence/optimization/opt052-gdn-arithmetic/REPORT.md),
[`pins/opt052_gdn_arithmetic_contract.json`](../pins/opt052_gdn_arithmetic_contract.json),
and
[`fixtures/opt052_gdn_arithmetic.json`](../fixtures/opt052_gdn_arithmetic.json).

OPT-053 keeps explicit FMA scale accumulation and two-stage packed-Y
`cp.async` on production quality MMA. **Measured, RTX 5090:** paired
CUDA-event A/B on the complete 4096-token FFN (quantize Y + gate + up +
SwiGLU-into-Q8 + down) among `off`, `fma`, `async_y`, and `fma_async`;
winner `fma_async` with like-arithmetic `async_y` byte-equal versus `off`
and FMA inside production-numerics budgets. Production
`kSelectedMmqPipelinePath` is `fma_async`. Integer MMA fragments, Q4_K min
corrections, Q6_K subscales, stream-K `off`, and I=128/J=128 tiles stay.
Tails keep the synchronous Y load. Extra Y-tile shared is 18432 bytes at
J=128. Occupancy remains 1. Live exclusive sitting **keep:** Quartz P
strictly greater than the copied prior keep; D128/D2048 throughput and p95
guards held; `reverted` false; `keep_sitting_skipped` false; `status`
measured. `quartz_meets_llama` is informational and is not this gate.
**The 2K parity owner remains the blocked dedicated gate.** Live numbers
stay in the report:
[`evidence/optimization/opt053-mmq-pipeline/REPORT.md`](../evidence/optimization/opt053-mmq-pipeline/REPORT.md),
[`pins/opt053_mmq_pipeline_contract.json`](../pins/opt053_mmq_pipeline_contract.json),
and
[`fixtures/opt053_mmq_pipeline.json`](../fixtures/opt053_mmq_pipeline.json).

OPT-054 measures internal 512/1024/2048/4096 physical batches inside one
atomic 4096-token `execute_prompt_chunk` transaction. Candidate KV uses an
outer staging range and explicit logical-versus-scratch indexing so earlier
uncommitted rows are not read as committed cache. GDN convolution/recurrent
state is carried privately between microbatches. State, frontier, tokens, and
logits publish only after the last internal batch; cancellation after internal
batches 1, 2, or the final batch leaves the outer transaction unpublished.
Graphs-off A/B isolates batch size, then the shipping path keeps matching FFN
graph shapes. **Measured, RTX 5090 keep 4096:** graphs-off 512 2476.38477 tok/s,
1024 2697.28589, 2048 2822.62891, 4096 2842.41333; graphs-on 4096 2840.70215.
No smaller size won complete wall versus 4096 eager or graphs-on shipping.
Copied P 2895.42773, D128 37.5605927, D2048 35.7286987. OPT-044 admits
cross-size arithmetic drift; isolation and restore stay exact. Live numbers
stay in the report:
[`evidence/optimization/opt054-prefill-microbatch/REPORT.md`](../evidence/optimization/opt054-prefill-microbatch/REPORT.md),
[`pins/opt054_prefill_microbatch_contract.json`](../pins/opt054_prefill_microbatch_contract.json),
and
[`fixtures/opt054_prefill_microbatch.json`](../fixtures/opt054_prefill_microbatch.json).

OPT-055 measures remaining decode and prompt launch/idle gaps after that
4096-row keep. `SchedulerGraphs` still ships 64 decode plus 64 prompt FFN
executables; layer-segment mixer/core slots stay empty because measured
`other_idle` stayed below noise on D128/D2048 and 4096-row prompt, including a
non-null poll path. Graph/eager fused outputs stay equal. Copied P 2895.42773,
D128 37.5605927, D2048 35.7286987. Extra device allocation is zero; the
128-graph 128K reserve is unchanged. Live numbers stay in the report:
[`evidence/optimization/opt055-execution-graphs/REPORT.md`](../evidence/optimization/opt055-execution-graphs/REPORT.md),
[`pins/opt055_execution_graphs_contract.json`](../pins/opt055_execution_graphs_contract.json),
and
[`fixtures/opt055_execution_graphs.json`](../fixtures/opt055_execution_graphs.json).

OPT-056 is the same-sitting exclusive RTX 5090 outcome gate versus pinned
llama.cpp on those combined production paths. **Measured unpassed:** P
**2808.49609** vs llama **3263.516321**, D128 **37.4816246** vs **68.9318767**,
D2048 **35.7208481** vs **67.3394327** tok/s; decode p95 worse than llama on both
prefixes; 2K **3012.69507** vs **3169.571249**. Combined production-optimization
quality failed the eight greedy tasks. `gate.passed` is false. Live numbers stay
in
[`evidence/optimization/opt056-performance-gate/REPORT.md`](../evidence/optimization/opt056-performance-gate/REPORT.md).

OPT-069 is the combined-batch re-sitting of that same exclusive P/D/2K protocol
on the frozen 062–068 keep/reject selectors (keeps: Q8 `r2_w2` and MMQ
`fma_async_x`). **Measured unpassed outcomes:** P **2914.65698** vs llama
**3142.517034**, D128 **37.1789093** vs **68.8708796**, D2048 **35.4498482** vs
**67.3394867** tok/s; decode p95 worse than llama; quality v2 fails
`task_arithmetic`; 2K **3132.88379** vs **3132.053862**. Parity gap is `Tq-Tl`
and is not the +5% bar `Tq-Tl/1.05`. The +5% outcome stays unpassed. Live
numbers stay in
[`evidence/optimization/opt069-batch-gate/REPORT.md`](../evidence/optimization/opt069-batch-gate/REPORT.md).

OPT-080 re-sits that protocol on the post-069 freeze (`kv_once` attention,
sequential decode GDN, packed Q4). Quality preflight must pass before original
P/D/2K timing. Preflight is not release evidence. The original +5% and 2K
gates stay blocked unless their owning conditions pass. Live numbers stay in
[`evidence/optimization/opt080-batch-gate/REPORT.md`](../evidence/optimization/opt080-batch-gate/REPORT.md).

OPT-041 keeps warp-owned 16×8 prompt QK microtiles on production 4096-row
fattn-mma stream-K including combine, on top of register-resident VKQ and
dual-F16 probability×V MMA. **Measured, RTX 5090:** paired CUDA-event A/B
among `cparts` and `warp_microtile`; winner `warp_microtile` with byte-equal
quality output versus `cparts`, frozen OPT-005 envelopes versus tiled, and
strictly lower component time. Production `kSelectedQKPath` is
`warp_microtile`. The `cparts` shared slab stays allocated; softmax, rescale,
P×V MMA, register VKQ, and the combine kernel stay. Ncols1 stays 16, Ncols2
stays 2, KV tile stays 32, dual-F16 Q stays, `grid.z` stays 2, register VKQ
stays, and P×V MMA stays. Decode 16/16 partitions stay. Mixer Q8 quality,
skinny `mma_i32_j128`, FFN `shared_y_swiglu_q8`, and GDN shared-inverse stay.
Tiled `launch_attention_prepare_chunk_tiled` remains the unloosened OPT-005
reference. Live exclusive sitting **keep:** Quartz P strictly greater than the
frozen then-current keep-oracle P denominator; D128/D2048 throughput and p95
guards held; `reverted` false; `keep_sitting_skipped` false; `status`
measured. D2048 tok/s improvement is not required for this prefill keep.
`quartz_meets_llama` is informational and is not this gate. **The 2K parity
owner remains the blocked dedicated gate.** This keep does not substitute for
that gate and does not claim Quartz ≥ llama.cpp. Live numbers stay in the
report:
[`evidence/optimization/opt041-fattn-warp-qk/REPORT.md`](../evidence/optimization/opt041-fattn-warp-qk/REPORT.md),
[`pins/opt041_fattn_warp_qk_contract.json`](../pins/opt041_fattn_warp_qk_contract.json),
and
[`fixtures/opt041_fattn_warp_qk.json`](../fixtures/opt041_fattn_warp_qk.json).

The **proof boundary** excludes comparative speed claims, 2K/8K sustained
prefill throughput, execution of a 128K prefill, 128K retrieval quality, thermal
stability, superiority to llama.cpp/vLLM, and a Nsight Systems overlap timeline.
OPT-014 measures named categories on one cold 2048-token timed run; it does not
convert that instrumentation into a sustained-prefill or speed admission.
OPT-015 explains the 2K gap and ranks recoveries; it is not llama.cpp parity
and not a throughput gate. OPT-017 records mixer Q8_0 MMA admission plus a
live exact-2048 remasurement and re-attribution. OPT-018 records Q4_K/Q6_K
quality MMA plus live FFN remasurement. OPT-019 records warp-column GDN and
fattn-mma attention core quality plus live GDN/attention remasurement and
re-attribution; it does not pass or own the 2K tok/s gate. OPT-020 records
the mixer versus core split of those composite GDN and attention buckets; it
is instrumentation, not a throughput gate, and not llama.cpp parity. OPT-021
records the frozen exact-4096 keep/reject protocol and same-sitting llama.cpp
4K `avg_ts`; it is not the 2K parity gate, not Quartz ≥ llama.cpp, and not a
BEN-001 `qw38-bench` result. OPT-022 records mixer Q8_0 quality MMQ plus
shared residual Y and a live 4K keep versus that frozen oracle baseline; it
does not own or pass the 2K tok/s gate. OPT-023 records skinny-M mixer
dispatch for small `output_rows` and a live 4K keep versus the post-OPT-022
oracle baseline; it does not own or pass the 2K tok/s gate. OPT-024 records
aligned-SoA D2R for large mixer Q8_0 and a live 4K reject versus the current
successor-oracle baseline; it does not own or pass the 2K tok/s gate. OPT-025
records dense FFN shared-Y / SwiGLU-into-down Q8 and a live 4K keep versus
the then-current oracle baseline; it does not own or pass the 2K tok/s gate.
OPT-026 records fattn Ada+ stream-K occupancy and a live 4K keep versus the
OPT-025 oracle baseline; it does not own or pass the 2K tok/s gate; the
first 4K idea ladder is exhausted. OPT-027 records persistent Ada+ fattn
stream-K and a live 4K reject versus the OPT-026 oracle baseline;
production fattn remains `stream_k` (`grid.z=2`); it does not own or pass
the 2K tok/s gate; the second 4K ladder is not exhausted. OPT-028 records
Q4_K/Q6_K MMQ stream-K and a live 4K reject versus the OPT-026 oracle
baseline; production quality MMA remains 2D tiling; it does not own or
pass the 2K tok/s gate; the second 4K ladder is not exhausted. OPT-029
records fused GDN conv/gated-output and a live 4K reject versus the
OPT-026 oracle baseline; production GDN core remains the split sequence;
it does not own or pass the 2K tok/s gate; the second 4K ladder is not
exhausted. OPT-030 records Hopper/Blackwell PDL serialization of
ungraphed mixer/GDN/attention prompt launches and a live 4K reject
versus the OPT-026 oracle baseline; production stays ordinary `<<<>>>`;
FFN graphs stay non-PDL; it does not own or pass the 2K tok/s gate; the
second 4K ladder is not exhausted. OPT-032 records frozen P/D128/D2048
oracles, exclusive decode categories, a 4K attribution refresh with graphs
created, matched llama.cpp public-API decode measurements, and a recorded
next-task order; it claims no performance improvement, is not the 2K parity
gate, not Quartz ≥ llama.cpp, and not a BEN-001 `qw38-bench` result. OPT-038
records the post-ladder refresh of those oracles and attributions on current
production objects, independent raw host-wall accounting, matched-component
experiment specifications, and a host-recomputed next-task order over pending
transferable IDs; it claims no performance improvement, does not publish a
successor oracle, leaves accepted keep denominators unchanged, is not the 2K
parity gate, not Quartz ≥ llama.cpp, and not a BEN-001 `qw38-bench` result.
OPT-033
records register-resident prefill attention value sums and a live 4K keep
versus the then-current accepted P denominator with the D128/D2048 guard;
production fattn remains `stream_k` (`grid.z=2`) with `registers` value
accumulation; it does not own or pass the 2K tok/s gate. OPT-035 records
dual-F16 probability×V MMA on that register-resident path and a live 4K
keep versus the then-current accepted P denominator with the D128/D2048
guard; production fattn remains `stream_k` (`grid.z=2`) with `registers`
value accumulation and `mma` probability×V; it does not own or pass the
2K tok/s gate. OPT-040 records hoisted prompt GDN Q/K inverses in existing
overlay scratch and a live 4K keep versus the then-current accepted P
denominator with the D128/D2048 guard; production inverse pin is `shared`;
decode sequential GDN and fuse `off` stay; it does not own or pass the 2K
tok/s gate. OPT-041 records warp-owned prompt QK microtiles and a live 4K keep
versus the then-current keep-oracle P denominator with the D128/D2048 guard;
production QK pin is `warp_microtile`; register VKQ and P×V MMA stay; it does
not own or pass the 2K tok/s gate. OPT-053 records explicit FMA scale accumulation and two-stage
packed-Y cp.async on quality MMA and a live 4K keep versus the prior keep
P denominator with the D128/D2048 guard; production MMQ pipeline pin is
`fma_async`; stream-K stays `off`; I=128/J=128 tiles stay; it does not own
or pass the 2K tok/s gate. OPT-054 records the 512/1024/2048/4096 internal
prefill microbatch sweep inside one atomic 4096-token transaction and keeps
4096 unless a smaller size wins complete P without regressing D; it does not
own or pass the 2K tok/s gate. OPT-055 records remaining decode/prompt launch
idle after that keep and retains FFN-only graphs when idle stays below noise;
it does not own or pass the 2K tok/s gate. OPT-056 records the combined
same-sitting P/D128/D2048 outcome versus pinned llama.cpp with a 5% margin,
decode p95, production-optimization quality, and original 2K evidence; the
measured sitting did not pass and does not redefine the 2K llama.cpp parity
gate. BEN-001
provides the harness; CMP-002/CMP-003 still own the 30-sample comparative gate.
QLT-001 remains blocked. OPT-012's prompt graphs are FFN subgraphs only: not a
whole-chunk graph, not a speedup gate, and not 128K quality recovery.
OPT-013 does not claim those gates either.
