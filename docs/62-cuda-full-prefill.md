# Chunked full-model CUDA prefill

[Index](README.md) · Implementation tasks: SCH-002, MEM-002, and EDU-047 in
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
token 0, then all values for token 1, and so on. If `R` is a 64 × 5,120
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
the same quantized weight matrix multiplied by several prompt rows. Q4_K and
Q6_K projections use the admitted transient Q8 activation blocks and the tuned
prompt-row tile.

The first integration attempt sent Q8_0 weights through that same path. That was
wrong for exact scheduler equivalence: the one-token Q8_0 path multiplies the
weights directly by BF16 activations, while generic MMQ requantized those
activations. The result was numerically close but changed persistent state and
last logits. The failed `append_vs_fresh` run is retained in the ledger.

Quartz now has a batched Q8_0-by-BF16 kernel. It processes multiple rows but
keeps the exact multiply, FP32 accumulation, and warp reduction order of the
one-token path. The two-token prefix test and the 65-token boundary test then
returned byte-equal state, hidden output, and logits.

## GDN and attention remain causal

Batching projections does not make recurrence parallel. Within each GDN layer,
the chunk primitive visits prompt rows in strict order. Its internal scan window
is at most 64 tokens, carries the convolution ring and FP32 recurrent matrix
forward, and produces a final candidate state for that layer.

Attention also visits chunk rows in order. A row may read all committed KV rows
from earlier chunks and candidate rows earlier in its current chunk, never a
future row. Partial RoPE uses the absolute position `old frontier + row`.

The scheduler chooses 64 rows because this matches the existing GDN scan window
and the tuned MMQ dispatch range. A 65-token prompt is deliberately split into
`[64, 1]`; the final single row uses the established decode arithmetic.

## Candidate state, committed state, and cancellation

**Committed** state is the conversation callers are allowed to observe.
**Candidate** state is temporary work that might still fail. Each prompt chunk
uses separate candidate storage for all 48 GDN layers and for 64 KV rows in all
16 attention layers. Tokens, last hidden state, logits, and frontier also remain
unchanged during calculation.

After every layer, an optional cancellation callback is polled. If cancellation
arrives, the function returns `cancelled` and does not swap GDN state, copy KV
rows, copy tokens, or advance the frontier. The measured 64-row cancellation
case remained byte-equal to an empty session with frontier zero. Only after all
64 layers and the last logits succeed does the chunk commit.

## Fixed scratch and the 128K budget

**Scratch** is reusable temporary memory whose contents have no meaning after an
operation. The workspace permanently owns buffers for at most 64 prompt rows:
two FP32 residual matrices, BF16 normalized/projected rows, Q8 activations,
projection and mixer outputs, GDN intermediates, and per-layer candidate KV
rows. Fixed allocation avoids allocator activity in each request and leaves the
decode graph's addresses unchanged.

This raises the complete workspace from 172,963,328 to 198,882,816 bytes. MEM-002
reran the simultaneous 131,072-token session plus resident model plus 64 uploaded
graphs. **Measured, RTX 5090:** 5,199,101,952 bytes remained free, leaving
3,588,489,216 bytes above the required 1.5 GiB reserve.

## Measured result and proof boundary

**Measured, RTX 5090:** a 65-token deterministic history crossed the chunk
boundary in 1,417.114 ms versus 4,204.656 ms through repeated one-token
execution, a 2.967× speedup. All committed GDN/KV bytes, the token frontier,
last hidden vector, and logits were byte equal. A separate 17-token benchmark
smoke improved from the earlier 16.20 to 40.96 prompt tokens/s.

These are focused local measurements, not the release performance matrix. They
prove exact native equivalence at the 64/65 boundary, cancellation before
commit, actual MMQ/chunk dispatch, improved wall time, and continued 128K memory
fit. The **proof boundary** does not include 2K/8K sustained prefill, 128K
retrieval quality, thermal stability, or superiority to llama.cpp/vLLM. BEN-001
provides the harness; CMP-002/CMP-003 still own the 30-sample comparative gate.
