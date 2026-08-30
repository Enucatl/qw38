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
