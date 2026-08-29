# 1. Inference fundamentals

[Index](README.md) · [Next](02-execution-path.md)

## What this chapter builds

This chapter assumes that you have never implemented a language-model runtime.
By the end, you should be able to explain the complete path from prompt text to
generated text, identify which data is shared by all requests and which data
belongs to one conversation, and understand why prompt processing and token
generation need different GPU kernels.

The target model, Qwen3.8-27B, is a **hybrid autoregressive language model**.
“Autoregressive” means that it generates one token after another, with every
new token conditioned on the tokens before it. “Hybrid” means that its layers
use two different mechanisms for moving information between sequence
positions: full attention and Gated DeltaNet. No prior knowledge of either is
required here. Chapter 3 gives their exact Qwen tensor contract after this
chapter establishes the concepts.

## From text to one generated token

A model does not receive strings directly. A **tokenizer** turns text into a
sequence of integer **token IDs**. A token may represent a word, a piece of a
word, punctuation, or a control marker. For illustration only:

```text
"The cat sat" -> [token_id_0, token_id_1, token_id_2]
```

This illustration shows three tokens, not Qwen's actual segmentation or IDs.
The exact result depends on Qwen's tokenizer and chat template. Those are part
of the model contract: changing tokenization changes the input.

Inference then follows this loop:

1. Look up an **embedding** for each input token. An embedding is a learned row
   of numbers; in this model each row has 5,120 values.
2. Pass those rows through 64 model layers. Each layer lets positions exchange
   information and then transforms each position independently.
3. Normalize the last layer's output and project its final row to 248,320
   **logits**, one score per vocabulary token.
4. Apply sampling rules to choose one token ID. Softmax may first turn logits
   into probabilities; temperature, top-k, or top-p rules may restrict them.
5. Append the chosen ID and repeat until a stop token or another limit is met.

Only step 4 chooses a token. Everything before it computes scores. Therefore a
correct engine must reproduce both the model arithmetic and the tokenizer,
template, and sampling policy used by its reference implementation.

## The repeated layer: two jobs and a residual stream

The 5,120-value row carried from one layer to the next is the **residual
stream**. A layer reads it and adds two learned corrections:

```text
x <- x + token_mixer(normalize(x))
x <- x + feed_forward(normalize(x))
```

The additions are **residual connections**. They preserve a direct path for the
incoming information while each branch contributes a correction. `normalize`
keeps numeric magnitudes controlled. Qwen uses RMSNorm; Chapter 3 specifies its
exact formula and parameter convention.

The two branches have different purposes:

- The **token mixer** moves information between sequence positions. In Qwen it
  is either full attention or Gated DeltaNet (GDN), depending on the layer.
- The **feed-forward network** (FFN) transforms every position separately. It
  uses the same learned matrices at every sequence position, but an FFN row
  does not directly inspect another row.

Qwen interleaves three GDN layers and one full-attention layer, repeating that
group 16 times. It therefore has 48 GDN layers and 16 full-attention layers.
The layers still execute in order: the output of one is the input of the next.

### The feed-forward branch

A **linear layer** multiplies an input by a learned weight matrix. Qwen's FFN
uses three such matrices. Two expand a 5,120-value row to 17,408 values, and a
third contracts it again:

```text
gate = gate_proj(x)       # [T,5120] -> [T,17408]
up   = up_proj(x)         # [T,5120] -> [T,17408]
h    = SiLU(gate) * up    # elementwise product
y    = down_proj(h)       # [T,17408] -> [T,5120]
```

`SiLU` is a smooth nonlinear function. The `*` multiplies corresponding
elements, rather than performing matrix multiplication. The gate lets the
network regulate which expanded features pass to the output. The word “gate”
appears elsewhere in the model too; it always means a learned control over how
strongly information passes, but the controlling tensors and formulas differ.

## Full attention, from first principles

Suppose the prompt is `The cat sat`. When the model processes `sat`, useful
information may reside at either earlier position. Full attention gives the
current position a content-dependent way to retrieve it.

Each attention layer projects every input row into three roles:

- A **query** (Q) describes what the current position is looking for.
- A **key** (K) describes what a position offers or how it can be matched.
- A **value** (V) contains the information returned if that position matches.

For one query `q` and all available keys and values, attention conceptually
does this:

```text
scores[j]  = dot(q, K[j]) / sqrt(head_dimension)
weights    = softmax(scores)
output     = sum_j(weights[j] * V[j])
```

The dot product gives similar query/key vectors a larger score. **Softmax**
turns all scores into non-negative weights that sum to one. The result is a
weighted blend of the value rows. These operations are differentiable during
training; at inference time the learned projections are already fixed.

### Causality and “full” attention

The model must not see tokens it is supposed to predict. A **causal mask** says
that position `t` may attend only to positions `0` through `t`. During prompt
processing this forms a lower-triangular visibility pattern:

```text
query position 0: may read [0]
query position 1: may read [0, 1]
query position 2: may read [0, 1, 2]
```

“Full” does not mean non-causal, and it does not mean every layer is an
attention layer. It means that, within such a layer, a query can compare with
every permitted earlier position instead of only a local window or a compressed
summary. Consequently the arithmetic for a long prompt grows roughly with the
number of permitted query-key pairs, and stored history grows with its length.

### Heads, grouped queries, and position

Attention divides its projected channels into **heads**. A head is a smaller,
independently matched query/key/value space. Multiple heads let a layer learn
different retrieval patterns at the same time. Their outputs are concatenated
and projected back to the residual width.

Qwen uses 24 query heads but only four key/value heads in its attention layers.
Six query heads share each K/V head. This is **grouped-query attention**: it
preserves many query views while storing fewer keys and values. “Sharing” is a
logical mapping; a good implementation does not physically copy each cached
K/V row six times.

Attention by itself compares content, not order. **RoPE** (rotary positional
encoding) rotates pairs of query and key coordinates by angles derived from
their positions. This makes attention scores depend on relative placement.
Qwen applies RoPE to part of each head; the precise 64-of-256 dimensions matter
later, but the foundation is simply that position modifies Q and K before they
are compared.

### Why full attention needs a KV cache

When generating the next token, old keys and values have not changed. Recreating
them through all prior layers would waste work. Each session therefore appends
the new K and V rows to a **KV cache** and reuses the old rows on later steps:

```text
position 0: cache K0,V0
position 1: read K0,V0; append K1,V1
position 2: read K0..K1,V0..V1; append K2,V2
```

The cache is per session because two conversations have different prefixes.
It is also per attention layer because every layer produces different K and V.
For this model's 16 attention layers, four KV heads, head width 256, and BF16
(two-byte) storage:

```text
16 layers * (K + V) * 4 heads * 256 values * 2 bytes
= 65,536 bytes per token
```

Thus 32,768 cached tokens require 2 GiB and 262,144 require 16 GiB for one
session, before allocator rounding or other state. This is a linear growth law:
twice the cached context requires twice the KV memory.

## Gated DeltaNet, from first principles

Full attention retains individually addressable information for every previous
position. Gated DeltaNet takes another approach: it processes tokens in order
and continually updates a fixed-size summary. This makes it a **recurrent**
token mixer—its output for the current token depends on state left by the
previous token.

An analogy is a running notebook with a fixed number of cells. Full attention
keeps every old page and can revisit a specific one. GDN continually edits the
notebook; the notebook does not grow, but old details may be combined or
overwritten. The analogy describes the storage tradeoff, not the exact math.

For each head, GDN keeps a matrix `S`. A token produces a key `k`, value `v`,
query `q`, write strength `beta`, and decay control `g`. In simplified form:

```text
S <- exp(g) * S                     # decay some old information
prediction <- k^T S                 # what the old state predicts for this key
error <- beta * (v - prediction)    # new information worth writing
S <- S + outer(k, error)            # update the fixed-size memory
output <- q^T S                     # retrieve for the current query
```

An **outer product** of two 128-value vectors produces a 128-by-128 matrix, so
it has the same shape as `S`. The update associates the key with the part of the
value that the state did not already predict. `beta` gates write strength, and
the exponential term controls forgetting. Qwen runs this independently over
48 value heads, giving persistent state `[48,128,128]` in each GDN layer.

GDN also applies a width-four **causal convolution** before the recurrence. A
convolution here is a learned local filter over the current projected row and
the previous three rows. Decode retains those recent rows in a small circular,
or **ring**, buffer. This ring is separate from the large recurrent matrix.

The important contrast is:

| Property | Full attention | Gated DeltaNet |
|---|---|---|
| Representation of history | One K/V row per prior token | Updated summary matrix plus short ring |
| Can directly score every old position | Yes | No |
| Persistent memory vs. context length | Grows linearly | Fixed per layer |
| Decode dependency | Reads all cached K/V rows | Must update state in token order |

“Fixed” does not mean small or free. Across Qwen's 48 GDN layers, the recurrent
matrices occupy about 144 MiB per session in FP32, and the convolution state
adds more. They must be allocated, reset, copied, and serialized correctly.
Unlike KV memory, however, they do not become larger when context grows from
32K to 256K tokens.

## Prefill and decode are two phases of the same model

**Prefill** evaluates the known prompt. If it has `T` tokens, the engine can
submit many rows together. The causal mask still prevents future information
from leaking backward, while the GPU can reuse weights across prompt rows.
Prefill builds every layer's KV or GDN state and produces logits for the first
generated token.

**Decode** is the repeating generation phase. After sampling a token, the
engine normally evaluates that one new row for the session, reads and updates
the state, and produces the next logits. Decode does not mean a different model
or approximate arithmetic; it is a different shape and state-use pattern.

```text
prompt IDs -- prefill(T rows) --> session state + first logits
                                         |
sample one ID -- decode(1 row) -----------+--> updated state + next logits
                         repeat
```

A server can batch one new row from many sessions. That raises the total row
count of a GPU launch, but each row still reads and updates its own session
state. It may improve throughput while increasing an individual request's
queueing delay.

User-visible metrics reflect the split:

- **Time to first token (TTFT)** includes prompt preparation, prefill, and the
  first sampling step.
- **Inter-token latency (ITL)** is the delay between generated tokens.
- Prompt tokens/second measures prefill work. Aggregate output tokens/second
  measures server throughput; it is not the same as one user's ITL.

## Shapes and matrix multiplication

A **tensor** is a multidimensional array. Its **shape** names the length of each
axis. `[T,5120]` means `T` rows with 5,120 scalar values per row.
`[48,128,128]` means 48 independent 128-by-128 matrices. Axis order matters:
`[T,heads,dim]` and `[heads,T,dim]` have the same element count but different
logical indexing.

For a linear projection, let `d` be input width and `o` output width:

```text
X[T,d] @ W[d,o] -> Y[T,o]
```

Each output value is a dot product: multiply `d` corresponding pairs and add
the products. Counting one multiply and one add separately gives:

```text
FLOPs = 2*T*d*o
```

This is a work estimate, not a duration. It omits or simplifies loads, stores,
normalization, activation functions, quantization, cache behavior, and kernel
launches.

For the three Qwen FFN matrices, ignoring the inexpensive elementwise work:

```text
linear FLOPs = 3 * 2*T*5120*17408 = 534,773,760*T
BF16 weight storage = 3*5120*17408*2 bytes = 510 MiB per layer
```

At `T=1`, about 510 MiB of FFN weights may be streamed to compute one row. At
`T=128`, the same weight tile can contribute to 128 rows while it is resident
on chip. The operations increase by 128, but external weight traffic need not.
This difference is central to GPU kernel selection.

## MMV, MMQ, and why one kernel is not enough

**MMV** means matrix-vector multiplication: a large weight matrix is applied
to one vector, or sometimes a very small group treated similarly. This is the
common decode shape. There is little reuse of each loaded weight across rows,
so reducing weight traffic and fixed setup cost is crucial.

**MMQ** in this handbook means the quantized matrix-matrix kernel family used
when a weight matrix is applied to many activation rows. More generally this
operation is called **GEMM** (general matrix multiplication). This is the common
prefill shape. The kernel tiles both matrices so that a loaded weight tile can
serve multiple rows, often using GPU tensor-core instructions. MMQ is not a
different mathematical layer; it is an implementation specialized for another
shape and weight format.

The actual row count, data type, quantization format, dimensions, and hardware
determine the crossover. `T=1` strongly suggests MMV and a large `T` suggests
MMQ/GEMM, but the engine should benchmark intermediate sizes rather than encode
the phase name as the only dispatch rule.

**Quantization** stores approximate weights with fewer bits plus scales or
other metadata. It can reduce the bytes that MMV must stream, but kernels must
unpack or interpret the representation, and approximation can change model
outputs. Quantization is therefore both a performance format and a numerical
policy that must be tested.

## Compute, bandwidth, and launch limits

GPU peak arithmetic alone does not predict latency. Three lower bounds are
useful:

```text
compute time   = FLOPs / useful compute rate
transfer time  = bytes moved / sustained bandwidth
launch time    = fixed dispatch and coordination cost

latency >= max(compute time, transfer time, launch time)
```

A decode FFN often has low **arithmetic intensity**—few useful operations per
byte fetched—because a weight is used for only one row. It can be
bandwidth-bound. A prefill matrix multiply reuses weights and may become
compute-bound. Very small operations can be launch-bound even when both their
FLOP and byte counts are tiny. Fusing operations can remove launches and
intermediate memory traffic, but it is an optimization to measure after the
unfused operations are correct.

For example, reading 1 GiB at a sustained 1 TiB/s has an idealized floor of
about 1 ms:

```text
1 GiB / (1 TiB/s) = 1/1024 second ~= 0.98 ms
```

This does not promise 1 ms execution; it proves that an implementation which
must read those bytes cannot be faster than that transfer floor.

## What lives for how long

Keeping ownership explicit prevents both memory-planning errors and corrupted
conversations:

- **Weights** are learned, immutable tensors shared by every request. They live
  with the engine and dominate base model storage.
- **Activations** are intermediate results for the current execution. Most can
  be reused or discarded after their consumers finish.
- **Workspace** is reusable temporary storage required by kernels or execution
  plans. Its contents are not conversational state.
- **Session state** belongs to one token sequence: KV caches, GDN matrices,
  convolution rings, positions, committed token IDs, and current logits.
- **Allocator reserve and repacked weights** consume real VRAM even when they
  are not visible in the original checkpoint file size.

The **state frontier** is the last position for which all session components
have committed consistently. If a failure updates GDN state but not KV state,
the session cannot safely continue at that apparent position. This is why
checkpoint, rollback, and prefix reuse are correctness features as well as
memory-management features.

## DwarfStar transfer boundary

DwarfStar is the existing engine used later as a systems case study. Its
engine/session lifetimes, MMV-versus-MMQ dispatch idea, allocation accounting,
and TTFT/ITL reporting are reusable because those are general inference-engine
mechanics.

Its model-specific mechanisms are not Qwen mechanisms. Compressed-attention
rows, a sparse indexer, routed experts, expert streaming, and mHC streams must
not be copied into Qwen's forward pass. Qwen instead requires the hybrid GDN
and full-attention state described above. “Transfer” in this handbook means
reuse an engineering pattern only after separating it from the old model's
tensor shapes and mathematics.

## A first measurement record

Never report only “tokens/s.” Record enough context to reproduce and interpret
the number:

- model and weight/quantization identity;
- prompt length, generated length, batch size, and context frontier;
- GPU power mode, clocks, and relevant software revisions;
- TTFT, prompt tokens/s, p50/p95 ITL, and aggregate output tokens/s;
- peak allocated and reserved VRAM; and
- the correctness fixture and sampling policy used.

Two runs are not comparable if they use different templates, prefix reuse,
samplers, output quality, prompt lengths, or concurrency, even if both print a
single tokens-per-second value.

## Common first-project mistakes

- Estimating VRAM fit from checkpoint file size while omitting repacked
  weights, KV capacity, GDN state, workspaces, graphs, and allocator reserve.
- Predicting one-token decode speed from advertised peak FLOPs or TOPS.
- Treating prefill and decode as the same matrix shape.
- Applying KV's per-token growth law to GDN's fixed recurrent state, or assuming
  fixed GDN state is free.
- Assuming “full attention” is unmasked or present in every layer.
- Treating MMV and MMQ as model semantics instead of alternative kernels for
  the same learned projections.
- Resetting, sharing, or advancing session state at the wrong boundary.

## Exercises and expected results

1. Draw the lifecycle of a three-token prompt followed by two generated tokens.
   Mark when embeddings, logits, K/V rows, and GDN state are created or updated.
   **Expected:** prefill consumes all three prompt rows and builds state through
   position 2; each decode call adds exactly one committed position.
2. Recalculate attention KV storage for 1,024 and 32,768 tokens.
   **Expected:** 64 MiB and 2 GiB respectively, demonstrating linear growth.
3. For 1 GiB of weights and 1 TiB/s sustainable bandwidth, derive the read
   floor, then run copy and empty-kernel launch microbenchmarks.
   **Expected:** an idealized read floor near 1 ms; large copies approach a
   bandwidth plateau and tiny operations approach a launch floor. Neither
   benchmark alone predicts end-to-end inference.

At this point you should have the vocabulary and dataflow needed for the rest
of the handbook. Chapter 2 follows a complete request through an existing
engine; Chapter 3 then pins every Qwen-specific layer shape and update rule.
