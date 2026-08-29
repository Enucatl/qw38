# 9. Gated text-engine implementation roadmap

[Previous](08-rtx-5090.md) · [Index](README.md) · [Next](10-qwen-transfer.md)

## Why this matters

Each milestone produces a runnable artifact and prevents performance work from
hiding a semantic error. Do not begin a later gate while an earlier one has
unexplained drift.

## Milestones and acceptance gates

Each row has an input, a visible output, and a decision rule. For example, the
scalar milestone does not merely “run a prompt”: it emits intermediate tensors,
compares them with two authorities, and records the first failing boundary.
That makes a failed gate actionable and prevents a later speed result from
concealing an earlier semantic error.

The milestones form four larger phases:

1. **Freeze semantics (1–3):** decide exactly which model is being implemented
   and obtain a slow, observable result.
2. **Reach the GPU (4–7):** implement operations first, then stateful mixers and
   the complete scheduler, while retaining the scalar oracle.
3. **Meet the product constraint (8–11):** reduce weight size, prove the memory
   budget, optimize measured bottlenecks, and persist sessions.
4. **Extend deliberately (12):** add optional state machines only after the text
   core is stable.

| # | Deliverable | Input -> output | Gate |
|---:|---|---|---|
| 1 | pinned fixture | prompt/messages -> IDs, positions, logits | hashes and expected logits recorded |
| 2 | inventory/converter | checkpoint shards -> validated manifest/artifact | every tensor accounted; sample round trips |
| 3 | scalar text forward | IDs -> taps, state, logits | matches Transformers and llama.cpp |
| 4 | dense CUDA primitives | shape/dtype/tails -> projections/norm/FFN | scalar differential suite passes |
| 5 | one-token GDN | row + old state -> output + new state | warm-up and FP32 recurrence pass |
| 6 | hybrid scheduler | row -> 48 GDN + 16 attention layers | layers 3/4/63 and logits pass |
| 7 | chunked prefill | arbitrary chunks -> logits + final state | decode-equivalent final state |
| 8 | quantized path | calibrated artifact -> logits/text | high-precision comparison and quality suite pass |
| 9 | 32 GiB proof | artifact + 32K session -> allocation log | graphs included; reserve remains; run completes |
| 10 | measured optimization | baseline -> fused/graphed/scheduled path | profiler attribution, correctness, raw A/B data |
| 11 | persistence | arbitrary prefix -> saved/restored/forked sessions | all hybrid state continues identically |
| 12 | extensions | text core -> MTP, then vision boundary | Chapter 10 gates pass separately |

Milestone 1 includes one-token and multi-token prompts and the official chat
template. Milestone 6 tests the first attention layer and the return to GDN.
Milestone 7 includes chunk sizes around 4 and 64. Milestone 8 compares greedy
walks, held-out perplexity/NLL, long recurrence, and task fixtures; a short
perplexity win cannot excuse long-state degradation. Milestone 9 repeats at
larger contexts until the admitted maximum is established.

## Stable interfaces

`ModelSpec` and `ModelWeights` are created once; `SequenceInput` carries
embeddings and positions; `SessionState` contains all mutable prefix state;
`BackendOps` supplies reference or CUDA operations. Server and CLI code see
tokens/logits/session operations, never tensor names. This conceptual boundary
does not require changing DwarfStar's runtime API.

One possible call flow is:

```text
spec = validate(config, manifest)
weights = load_and_repack(spec, artifact)
backend = make_cuda_backend(spec, weights)
session = create_session(spec, context_capacity)
input = SequenceInput(token_embeddings, positions)
logits = backend.forward(weights, input, session.state)
```

The names are conceptual rather than a required public API. Their value is that
the converter does not know about HTTP messages, the CUDA backend does not parse
checkpoint names, and the server does not manipulate recurrent matrices. This
keeps a bug within a testable boundary.

### What “runnable” means

A milestone artifact should have one documented command, fixed small inputs,
machine-readable output, and a nonzero exit status on gate failure. A notebook
or debugger observation is useful during development but is not the gate until
another programmer can reproduce it without hidden state.

## DwarfStar lessons

Reuse lifetimes, allocation accounting, differential tests, MMV/MMQ, graph
discipline, benchmarking, and correctness gates. Adapt validation, binding,
quantization, serialization, hybrid scheduling, batching, and future MTP.
Reject compressed-attention schemas, indexers, MoE routing/streaming, mHC, and
DSpark semantics.

## Failure modes and exercise

Do not optimize before a high-precision oracle, accept only final token text,
introduce quantization and CUDA simultaneously, or call a nominal artifact size
a fit proof. Exercise: for each gate write the exact fixture, observable output,
tolerance, failure artifact, and rollback criterion. Expected: another engineer
can run every gate without an oral explanation.
