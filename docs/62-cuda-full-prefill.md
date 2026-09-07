# Chunked full-model CUDA prefill

[Index](README.md) · Implementation tasks: SCH-002, MEM-002, OPT-008, OPT-009, and EDU-047 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contract:
[`pins/cuda_prompt_scheduler_contract.json`](../pins/cuda_prompt_scheduler_contract.json)
· Evidence: [`fixtures/cuda_prompt_scheduler.json`](../fixtures/cuda_prompt_scheduler.json)

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

1. Decode every token's embedding row into BF16 and widen the residuals to FP32.
2. Normalize each residual row independently.
3. Use MMQ to project all rows for the layer's mixer.
4. Apply either the GDN recurrence or causal grouped-query attention.
5. Add the mixer residual, run the three prompt-row FFN projections, and add the
   FFN residual.
6. After layer 63, compute final normalization and logits only for the last row.
7. Publish every persistent state row and advance the frontier.

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

Batching projections does not make recurrence parallel. Within each GDN layer,
the chunk primitive visits prompt rows in strict order. Its internal scan window
is at most 64 tokens, carries the convolution ring and FP32 recurrent matrix
forward, and produces a final candidate state for that layer.

Attention also visits chunk rows in order. A row may read all committed KV rows
from earlier chunks and candidate rows earlier in its current chunk, never a
future row. Partial RoPE uses the absolute position `old frontier + row`.

OPT-008 sets the outer scheduler policy to 4,096 rows. The GDN primitive still
uses its unchanged internal 64-row scan windows. A 4,097-token prompt therefore
executes as `[4096, 1]`, with the final single row using the established decode
arithmetic. For a smaller session, the reusable allocation and selected prompt
chunk are bounded by its capacity: a capacity-65 session executes `[65]` as one
prompt transaction rather than allocating or dispatching 4,096 rows.

## Candidate state, committed state, and cancellation

**Committed** state is the conversation callers are allowed to observe.
**Candidate** state is temporary work that might still fail. Each prompt chunk
uses separate candidate storage for all 48 GDN layers and for up to 4,096 KV rows
in all 16 attention layers, bounded by session capacity. Tokens, last hidden
state, logits, and frontier also remain
unchanged during calculation.

After every layer, an optional cancellation callback is polled. If cancellation
arrives, the function returns `cancelled` and does not swap GDN state, copy KV
rows, copy tokens, or advance the frontier. The measured 4,096-row cancellation
case remained byte-equal to an empty session with frontier zero. Only after all
64 layers and the last logits succeed does the chunk commit.

## Fixed scratch and the 128K budget

**Scratch** is reusable temporary memory whose contents have no meaning after an
operation. The workspace permanently owns buffers for
`min(4096, session capacity)` prompt rows: two FP32 residual matrices, BF16
normalized/projected rows, Q8 activations, projection and mixer outputs, GDN
intermediates, and per-layer candidate KV rows. This fixed, capacity-bounded
allocation avoids request-sized allocator activity and leaves the decode graph's
addresses unchanged.

At capacity 131,072, the diagnostic workspace is 1,831,810,560 bytes. MEM-002
reran the simultaneous 131,072-token session plus resident model plus 64 uploaded
graphs. **Measured, RTX 5090:** 3,573,809,152 bytes remained free, leaving
1,963,196,416 bytes above the required 1.5 GiB reserve.

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
envelopes. The **proof boundary** excludes comparative speed claims, 2K/8K
sustained prefill, execution of a 128K prefill, 128K retrieval quality,
thermal stability, and superiority to llama.cpp/vLLM. BEN-001 provides the
harness; CMP-002/CMP-003 still own the 30-sample comparative gate. QLT-001
remains blocked.
