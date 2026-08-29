# 11. Glossary, worksheets, and review

[Previous](10-qwen-transfer.md) · [Index](README.md) · [Next](12-language-and-platforms.md) · [Sources](sources.md)

## Glossary

- **Activation:** a temporary vector or tensor produced while evaluating the
  model. Activations are derived from input and weights; most can be overwritten
  after the next operation consumes them.
- **Arithmetic intensity:** useful arithmetic operations per byte moved from a
  chosen memory level. Higher intensity gives the GPU more opportunity to use
  compute hardware instead of waiting for data.
- **Backend:** an implementation of model operations for a particular execution
  environment, such as scalar CPU code or CUDA kernels.
- **Batch:** rows from one or more sequences processed together. Batching can
  improve weight reuse but may add queueing latency.
- **BF16 / FP16 / FP32:** floating-point formats occupying 16, 16, and 32 bits.
  They differ in exponent range and precision; accumulation may use a wider
  format than stored inputs.
- **Causal:** unable to use information from future sequence positions.
- **Checkpoint:** a versioned snapshot sufficient to continue a session without
  replaying its entire prefix.
- **Decode:** evaluate newly committed positions, commonly one token per session,
  using the persistent state created by the prefix.
- **Embedding:** the learned vector looked up for a token ID, or a compatible
  vector produced by the future vision encoder.
- **Engine:** long-lived owner of validated model metadata, immutable weights,
  backend resources, and compiled execution plans.
- **FLOP:** one floating-point operation. This guide counts a multiply and add as
  two FLOPs when estimating matrix multiplication.
- **Gated DeltaNet (GDN):** a recurrent token mixer that decays a fixed summary,
  writes a gated prediction error, and reads it with a query.
- **Head:** one independently indexed subspace within attention or GDN. Several
  heads let the model learn different retrieval or recurrence behavior.
- **ITL:** inter-token latency, the user-visible time between generated tokens.
- **Kernel:** one GPU program launch over many threads.
- **KV cache:** growing full-attention key and value rows retained for later
  queries. It is different from GDN recurrent and convolution state.
- **Logit:** an unnormalized score for one vocabulary item. A sampler converts
  the complete logit vector into a next-token choice.
- **MMV / MMQ:** matrix-vector and matrix-matrix kernel families, often operating
  directly on quantized weight blocks.
- **MTP:** multi-token prediction, used after v1 to propose several tokens for
  speculative target verification.
- **mRoPE:** modality-aware RoPE using temporal, height, and width coordinates.
- **Oracle:** a trusted reference implementation or trace used to judge another
  implementation's intermediate tensors.
- **Prefill:** evaluate an unmatched prompt suffix, usually many rows at once,
  to build persistent state and produce the first next-token logits.
- **Quantization:** approximate numeric storage using fewer bits plus scale or
  other reconstruction metadata.
- **Residual connection:** addition of a block's input to its computed correction.
- **RoPE:** rotary positional encoding; Qwen text attention rotates 64 dimensions
  of each 256-dimensional Q/K head.
- **Session:** mutable state for one sequence or conversation, including committed
  IDs, positions, GDN state, KV, and current logits.
- **Softmax:** conversion of scores to non-negative weights that sum to one.
- **State frontier:** the position through which every session-state component
  has committed consistently.
- **Tensor:** a multidimensional rectangular array. Its shape records the length
  of each logical axis but not necessarily its physical storage order.
- **TTFT:** time to first token, including request handling, prompt preparation,
  prefill, and the first sampling step.
- **Workspace:** temporary reusable memory sized for operations in an execution
  plan; unlike weights or session state, its contents do not survive logically.

## How to use the worksheets

Do not fill these tables from memory. Generate the tensor worksheet from the
converter manifest and runtime allocation log, then investigate differences.
Run the verification worksheet as executable fixtures. Each row should name the
exact model and source hashes, command, tolerance, output artifact, and owner
responsible for a failure.

## Tensor and state worksheet

| Item | Logical shape | Format | Lifetime | Bytes | Oracle tap |
|---|---|---|---|---:|---|
| embeddings / LM head | `[248320,5120]` each | | engine | | embedding/logits |
| GDN recurrence | `48 layers * [48,128,128]` | FP32 v1 | session | 144 MiB | each GDN output |
| convolution rings | `48 * [10240,4]` | FP32 v1 | session | 7.5 MiB | post-convolution |
| full K and V | `16 * 2 * [C,4,256]` | BF16 v1 | session | `65536*C` | attention output |
| logits | `[248320]` | FP32 | session | 993,280 B | final logits |
| workspace/repack/graphs | measured plan | | engine/session | | allocation log |

For each blank, record source key, orientation, block metadata, alignment,
owner, address stability, and checksum. Totals must reconcile with runtime logs.

`C` in the KV row is context capacity: the maximum number of positions allocated
for this session. Logical length may be smaller. “Lifetime” determines ownership:
engine allocations are shared across sessions, session allocations multiply by
concurrency, and workspace may be shared only when executions cannot overlap.

## Verification worksheet

| Scenario | Compare | Expected result |
|---|---|---|
| positions 0–5 | GDN convolution and recurrence | oracle agreement through warm-up |
| prompt lengths 1/4/5/63/64/65 | full tap set | tolerance gates pass |
| layers 2/3/4/63 | scheduler and mixer taps | correct kind and residual order |
| arbitrary prefill chunks | final state and logits vs decode | equivalent result |
| arbitrary save/restore | uninterrupted continuation | all hybrid state agrees |
| 32K and larger context | allocation ledger and completion | admitted sizes retain reserve |
| quant vs high precision | logits, NLL/PPL, long/task suite | declared quality budget passes |
| MTP accept/partial/reject | target-only run | same committed sequence/state |
| future visual rows | Transformers embeddings/positions/logits | multimodal boundary agrees |

## Reader labs

The labs are ordered so each introduces one new responsibility. Complete them
with small arrays before using the 27B checkpoint; a five-token, one-head fixture
is enough to expose indexing and state-order bugs without requiring large model
hardware.

1. Derive the 10,240 packed GDN width and label every slice. Expected:
   `16*128 + 16*128 + 48*128`.
2. State the recurrence update without consulting code. Expected: decay state,
   predict at K, beta-scaled correction, outer update, query readout.
3. Fill an allocation ledger from an artifact and runtime log. Expected: explain
   every byte category and prove reserve after graph creation.
4. Specify scalar milestone steps. Expected: pin, inventory, embed, 64 scheduled
   blocks, final norm/head, intermediate taps, two independent comparisons.
5. Review every claim label and link. Expected: measured, external, estimated,
   and proposed statements cannot be mistaken for one another.

The handbook passes reader review when someone unfamiliar with GDN can complete
labs 1–4 and state the implementation sequence without undocumented assumptions.
