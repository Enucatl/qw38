# 37. Official-checkpoint Transformers authority

[Index](README.md) · Implementation tasks: ORA-003 and EDU-023 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`transformers_scalar_authority.json`](../fixtures/transformers_scalar_authority.json)

Quartz computes with a compressed GGUF file. How do we know that our
interpretation still represents the model Qwen published? We run the same input
through Qwen's original checkpoint and the pinned upstream Transformers model.
That execution is the **primary semantic authority**: it tells us what the
published model equations and weights produce before GGUF quantization.

## Original checkpoint and GGUF are different artifacts

The original checkpoint stores almost all model weights as BF16 values. BF16 is
a 16-bit floating-point format: it has fewer precision bits than ordinary FP32,
but the same broad exponent range. The checkpoint's 18 weight files total
55,563,006,776 bytes.

Quartz's GGUF is 18,973,870,432 bytes because most weights were quantized into
small integer blocks such as Q4_K and Q6_K. Quantization is like recording a
smooth curve with fewer tick marks: it saves memory, but introduces small
numeric differences. Consequently, original-checkpoint logits are not expected
to be bit-for-bit equal to GGUF logits. Both artifacts must still choose the
same unambiguous greedy continuation and remain within tolerances that ORA-004
subsequently froze.

## What a Safetensors shard is

A **tensor** is a rectangular array of numbers. A **Safetensors** file stores
named tensors with a small metadata header and raw data ranges, without allowing
executable program objects inside the model file. A **shard** is one piece of a
checkpoint split across several files so that no single download is enormous.
The index maps each tensor name to one of the 18 shards.

[`transformers_authority_contract.json`](../pins/transformers_authority_contract.json)
pins every shard's filename, byte count, and SHA-256 digest. SHA-256 is a
content fingerprint: changing even one byte produces a different digest with
overwhelming probability. [`verify_transformers_authority.py`](../tools/verify_transformers_authority.py)
rejects a missing, extra, resized, or changed shard and rejects a changed source
checkout. This is **exact identity**, not approximate numeric comparison.

The checkpoint contains 1,199 tensors. The text generation wrapper loads 1,184.
The remaining 15 belong to MTP, an optional speculative next-token mechanism
excluded from V1. Vision parameters are present because the upstream wrapper is
multimodal, but the unused vision module is placed on disk and never enters this
text-only execution. These exclusions do not remove any weight used by the
normal text logits.

## What eager execution means

Frameworks can combine, reorder, or replace operations to run faster. **Eager
execution** means PyTorch performs the visible model operations directly as the
Python forward method requests them. The attention implementation is also
forced to `eager`, rather than Flash Attention or another fused kernel.

Eager does not mean scalar arithmetic or exact reproducibility across every
machine. PyTorch still uses optimized matrix multiplication kernels. Its value
here is inspectability: the recurrence, RoPE, attention, and residual boundaries
remain visible and can be captured. The exact source commit and Python package
environment are pinned in the contract and hash-locked requirements file.

## GPU, CPU, and disk offload

The 55.6 GB checkpoint cannot reside wholly in 32 GiB of GPU memory. **Offload**
means keeping a module's weights somewhere else until its layer executes:

- layers 0–24, embeddings, final norm, and output weights stay on the GPU;
- layers 25–44 stay in host RAM;
- layers 45–63 and the unused visual module remain backed by the original files
  on disk and are staged when needed.

This is slow but semantically straightforward. [`run_transformers_authority.py`](../tools/run_transformers_authority.py)
fails if Accelerate chooses a different map. **Measured:** the captured two-token
run peaked at 24,460,563,456 allocated GPU bytes, 24,490,541,056 reserved GPU
bytes, and 24,357,984 KiB maximum resident host memory. Loading took 20.72
seconds and execution took 46.36 seconds. These are feasibility measurements,
not production performance claims or benchmark samples.

[`setup_transformers_authority.sh`](../tools/setup_transformers_authority.sh)
recreates the exact checkout, Python environment, metadata, shards, and final
identity verification. It deliberately separates metadata and weight filters;
an earlier malformed download command fetched only metadata, a negative result
preserved in the ledger.

## Hooks and taps

A **hook** is a diagnostic callback invoked when a chosen operation runs. A
**tap** is the tensor copied at that boundary, analogous to attaching a probe to
a point in an electronic circuit. [`transformers_taps.py`](../tools/transformers_taps.py)
registers hooks on upstream modules and temporarily wraps three upstream
functions. Every wrapper calls the original function first and only observes
its inputs or result; it does not reimplement or edit the model equation.

The diagnostic records 238 taps over two positions at representative early and
late GDN layers (0 and 62) and attention layers (3, 7, and 63). They cover:

- embedding, input/post-mixer norms, mixer and layer residuals;
- GDN projections, post-convolution Q/K/V, decay, beta, core output, and the
  complete FP32 recurrent matrix;
- attention projections, normalized Q/K, values before and after partial RoPE,
  growing K/V caches, attention weights, and output;
- SwiGLU gate, activation, up, down, final norm, and complete logits.

Each tensor is converted to contiguous little-endian FP32 bytes. "Little
endian" defines the byte order of each 32-bit value, so another tool reads the
same bits. Its manifest stores name, shape, original dtype, blob offset, byte
count, SHA-256, and summaries including NaN/infinity counts. The ignored raw
blob is 20,160,352 bytes with digest
`1083ab56433026ac03128603dbed017391c98e3053b9169774e9654e8e85a031`;
the checked-in fixture retains all 238 records needed to authenticate it.

## Why raw token IDs come first

Text normally passes through Unicode handling, the chat template, and BPE
tokenization before it reaches the model. Those concepts and their exact tests
are explained in [Tokenizer authority](15-tokenizer-authority.md) and
[Chat templates](16-chat-template.md). This trace begins later, with raw token
IDs `[42, 3649]`, so a tokenizer difference cannot masquerade as a model-math
difference. Chapter 36 separately proves exact template bytes and token IDs
between Quartz and llama.cpp.

Position 0 consumes token 42 and predicts token 3649. Position 1 consumes token
3649 and predicts token 1277. A **greedy token** is simply the index of the
largest logit. A **logit** is an unnormalized score for one vocabulary token;
the model emits 248,320 scores at each position.

## Reading the comparison

The complete official logit rows have SHA-256
`9b64105a1c7262271c85054ef30cd116e0af4e85a497e6ae24c478007ed97947`.
Quartz and llama.cpp choose the same two greedy tokens. Their cosine similarities
against the official rows range from 0.99375 to 0.99557. RMS differences range
from 0.15852 to 0.21763. These larger differences than the same-GGUF comparison
are expected consequences of comparing compressed GGUF weights with original
BF16 weights, not evidence that any arbitrary difference is acceptable.

**Fixture equality** means a test compares newly produced fields with a saved,
reviewed fixture. Exact fields—identity, tokens, shapes, byte ranges, and
hashes—must match exactly. Floating-point taps require explicit per-boundary
tolerances because different admitted numeric formats do not have identical
bits. ORA-003 records values with zero tolerance only to show where differences
start; ORA-004 subsequently chose and froze the admission tolerances.

## What this proves

- The exact official checkpoint and exact Transformers source execute on this
  host using eager attention within the available GPU and host memory.
- The two-token run emits finite, authenticated taps across every required
  conceptual stage and complete finite logits.
- Official BF16, Quartz Q4_K_M, and llama.cpp Q4_K_M choose the same greedy
  tokens for this diagnostic history.
- The instrumentation exposes recurrence and KV state without replacing the
  pinned upstream equations.

## What this does not prove

- This chapter alone does not admit Quartz numeric accuracy. ORA-004
  subsequently compared matching taps, investigated layout mismatches, and
  froze the pre-CUDA tolerances in
  [Chapter 38](38-scalar-authority-tolerances.md).
- It is not a tokenizer, template, quality, long-context, speed, or 128K memory
  test.
- Two equal greedy tokens do not prove all logits or hidden states are close.
- Offloaded Transformers timings do not predict Quartz performance.
- The raw 20 MB tap blob is local evidence identified by its hash; the committed
  manifest alone cannot reconstruct its numeric values.

That narrow proof boundary was deliberate: ORA-003 established a trustworthy
primary source. ORA-004 then admitted Quartz at every mapped visible boundary
and froze the tolerances before any CUDA optimization could see or loosen them.
