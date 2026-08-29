# Build a Qwen3.8-27B engine on one RTX 5090

This handbook has one outcome: help a systems programmer build a narrow,
text-first inference engine for Qwen3.8-27B on a 32 GiB RTX 5090. It is a design
and implementation guide, not Qwen support for DwarfStar and not a promise of a
particular speed. That is the primary optimization target; the architecture also
supports DGX Spark, Apple Silicon, and CPU-only hosts at lower support tiers.
DwarfStar is a worked systems case study.

## Authority and evidence

Pin all inputs before writing code. The model contract comes from the official
[configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json),
[checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B), tokenizer/chat template,
and the Transformers `qwen3_5` implementation. The checkpoint is named Qwen3.8,
but its configuration declares `model_type: qwen3_5`; follow the declared
implementation, not a guessed module name. Use a pinned llama.cpp revision as
an independent execution oracle. [q27](https://github.com/signalnine/q27) is an
attributed specialization case study; its measurements do not describe
DwarfStar or the engine proposed here.

## Two reading tracks

The fundamentals track is Chapters 1, 3, 5, 6, 7, and 8. The implementation
track is Chapters 2, 4, 9, and 10. Both use Chapter 11 for review and Chapter 12
for language and host portability decisions.

1. [Inference fundamentals](01-foundations.md): shapes, bytes, rooflines.
2. [A complete DwarfStar request](02-execution-path.md): reusable mechanics and model-specific traps.
3. [The Qwen model contract](03-deepseek-v4.md): the forward pass and state.
4. [Scalar oracle](04-numerics.md): reference implementation and differential tests.
5. [Weights and the 32 GiB ledger](05-gpu-implementation.md): conversion and quantization.
6. [CUDA decode and prefill](06-system-optimization.md): MMV, MMQ, GDN, attention.
7. [Hybrid sessions](07-engineering-method.md): checkpointing, reuse, batching, graphs.
8. [RTX 5090 optimization](08-rtx-5090.md): profiler-led sequencing.
9. [Gated implementation roadmap](09-engine-comparison.md): twelve acceptance milestones.
10. [Deferred MTP and vision](10-qwen-transfer.md): extensions without text-core redesign.
11. [Glossary, worksheets, and exercises](11-glossary-worksheets.md).
12. [Language and platform strategy](12-language-and-platforms.md): C++ boundaries and tiered backends.
13. [V1 allocation ledger](13-allocation-ledger.md): explicit 128K budget and proof boundary.
14. [Source and evidence ledger](sources.md).

Claims use four labels: **Measured** (this exact artifact and hardware),
**External** (a linked source), **Estimated** (reproducible arithmetic), and
**Proposed** (a design or experiment not yet demonstrated).

## Engine boundary

- `ModelSpec`: validated dimensions, layer kinds, tensor names, dtypes, and quantization policy.
- `ModelWeights`: immutable resident or repacked tensors.
- `SequenceInput`: token embeddings and position metadata; later visual embeddings.
- `SessionState`: 48 recurrent matrices and convolution rings, 16 KV caches, position frontier, and optional MTP state.
- `BackendOps`: coarse reference, CPU, CUDA, and Metal execution services, with
  semantic primitive operations retained for differential tests.

The text path is `template -> token IDs -> embeddings -> 64 hybrid layers ->
final norm -> logits -> sampler`. Weights belong to the engine; prefix-dependent
data belongs to the session. V1 excludes vision and MTP.

## Chapter contract

Each technical chapter covers motivation, concepts, worked shapes, a DwarfStar
example, transfer boundaries, concrete work, failure modes, and a verification
exercise with an expected result.
