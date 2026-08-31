# Quartz Watch 38

**Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.**

Quartz Watch 38 (`qw38`) is a deliberately narrow, work-in-progress inference
engine for the pinned Qwen3.8-27B Q4_K_M artifact on one RTX 5090. The approved
scope is in [plan.md](plan.md), and implementation claims and evidence are in
[implementation_ledger.md](implementation_ledger.md).

The repository is not yet a usable inference server. Current binaries fail
closed when an operation has not passed its delivery gate.

The beginner-oriented [implementation handbook](docs/README.md) explains each
admitted concept and links it to code, fixtures, failures, and evidence. The
latest chapter covers [GGUF parameter conversion](docs/22-gguf-conversion.md),
including folded decay, RMSNorm scales, squeezed shapes, and GDN head order.
The following [typed-weight chapter](docs/23-typed-model-weights.md) explains how
all 851 admitted ranges become safe global and per-layer fields.
The [packed-projection chapter](docs/24-packed-projections.md) then fixes the
different GDN and attention slicing rules used by scalar execution.
The [real-mixer chapter](docs/25-real-mixer-projections.md) executes complete
typed layer-0 and layer-3 projections and records their exact workspaces and
independently decoded evidence taps.
The [real GDN layer chapter](docs/26-real-gdn-layer.md) follows one layer-0 token
through normalization, both persistent states, gated output, and residual.
The [real FFN chapter](docs/27-real-ffn-layer.md) then explains the complete
layer-0 SwiGLU branch, including its three Q4_K projections, temporary memory,
numeric evidence, and remaining scheduler boundary.
The [real attention chapter](docs/28-real-attention-layer.md) follows two
layer-3 positions through partial RoPE, grouped causal lookup, KV mutation,
output gating, projection, and residual.
The [complete-layer chapter](docs/29-complete-decoder-layer.md) joins each mixer
variant to its SwiGLU branch and explains both residuals, validation order,
memory lifetimes, and the remaining full-scheduler boundary.
The [embedding and logits chapter](docs/30-embeddings-and-logits.md) explains how
token IDs become hidden vectors and how final normalized state becomes all
248,320 vocabulary scores.
The [full scalar token chapter](docs/31-full-scalar-token.md) follows one real
token through all 48 GDN and 16 attention layers, final logits, state ownership,
scratch reuse, and the remaining oracle boundary.
The [trace bundle chapter](docs/32-trace-bundles-and-metrics.md) explains how
diagnostic taps are stored, authenticated, and compared using absolute,
relative, RMS, cosine, non-finite, first-failure, and top-logit evidence.
The [diagnostic isolation chapter](docs/33-diagnostic-trace-isolation.md) then
defines stable tap names and exact filters while proving trace machinery is
absent from the normal binary.
The [real scalar trace chapter](docs/34-real-scalar-traces.md) maps every tap to
its execution moment and shape, then follows one filtered real tensor through a
model-identified, state-aware trace-v1 bundle.
The [scalar chunk chapter](docs/35-scalar-token-chunks.md) explains multi-token
positions, whole-request preflight, vocabulary-logit row layout, and exact
equivalence with repeated one-token execution.
The [independent-authority chapter](docs/36-independent-llama-authority.md)
explains why Quartz is checked against both llama.cpp and Transformers, how the
same GGUF and raw token history are enforced, and why numeric reporting precedes
the tolerance freeze.
The [Transformers authority chapter](docs/37-transformers-authority.md) then
explains original Safetensors checkpoints, eager execution, GPU/CPU/disk
offload, diagnostic hooks and taps, and exactly what the official-checkpoint
comparison does and does not prove.
The [scalar tolerance chapter](docs/38-scalar-authority-tolerances.md) aligns all
three authorities, explains exact layout normalization and numeric error
metrics, and freezes the immutable scalar gates CUDA implementations must pass.
The [CUDA MMV chapter](docs/39-cuda-quant-mmv.md) then explains the first SM120
kernel, transient BF16-to-Q8 staging, warp-per-row ownership, and the measured
scalar-versus-device admission boundary.
The [CUDA prompt MMQ chapter](docs/40-cuda-prompt-mmq.md) extends that boundary
to arbitrary prompt rows with explicit two-dimensional tiles, packed-weight
reuse, token-major output, and tail handling.
The [CUDA GDN step chapter](docs/41-cuda-gdn-step.md) adds the first stateful GPU
model core and explains committed versus candidate recurrence, cancellation,
and frontier-last publication.
The [chunked CUDA GDN chapter](docs/42-cuda-gdn-chunks.md) carries that state
through arbitrary prompt chunks using continuous 64-token windows and proves
byte-exact equivalence with repeated one-token GPU execution.
The [CUDA attention decode chapter](docs/43-cuda-attention-decode.md) explains
grouped query-to-KV sharing, partial RoPE, BF16 cache rows, causal reads, stable
softmax, and candidate-row commit on the GPU.
The [CUDA attention prefill chapter](docs/44-cuda-attention-prefill.md) extends
that state through atomic prompt chunks with linear score storage and executes
the final legal position in a real 131,072-row production cache.
The [CUDA scheduler-prerequisite chapter](docs/45-cuda-scheduler-primitives.md)
then explains resident Q8_0 weights versus temporary Q8 activations, embedding
row lookup, BF16 pointwise operations, and the packed-layout conversions needed
before the complete 64-layer scheduler can be assembled.

## Build

```sh
make
make test
```

The host-only build establishes and tests the public boundary. CUDA compilation
uses the immutable CUDA 13.0 container:

```sh
make cuda-image
make cuda-build
```

Run `build/qw38-eval --build-info` to inspect the compiled target and artifact
pin. No model is downloaded by the build.
