# 18. Gated Delta Networks and persistent state

[Index](README.md) · Implementation tasks: CPU-002 and EDU-004 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter explains the recurrent half of Qwen3.8's hybrid model from no prior
knowledge. It covers the semantic core implemented in
[`src/gdn.cpp`](../src/gdn.cpp), the frozen contract in
[`pins/gdn_contract.json`](../pins/gdn_contract.json), and the focused fixtures
in [`fixtures/gdn_authority.json`](../fixtures/gdn_authority.json). Matrix
projections, the final gated RMS normalization, and the output projection are
part of the later complete-layer scheduler; CPU-002 does not claim those paths.

## Why a model needs memory

When a model processes a token, it needs information about earlier tokens.
Ordinary causal attention keeps a key and value for every earlier position and
looks back over that growing history. A recurrent layer takes another approach:
it condenses earlier information into a fixed-size **state**, updates that state
for each new token, and passes the state to the next token.

“Recurrent” means that a result from one step is fed back into the next step. If
`S_t` is the state after token `t`, then token `t + 1` starts from `S_t`. The
state is not a model weight: weights are immutable and shared by sessions, while
recurrent state depends on one session's token prefix and changes over time.

Qwen3.8 is **hybrid**. Its 64 text layers contain 48 Gated Delta Network (GDN)
layers and 16 full-attention layers. Every fourth layer is full attention. A
GDN's fixed-size state avoids adding a full KV row at that layer for every token,
but its recurrence must be executed in exact token order.

## The projections and their names

The input to a GDN layer is a 5,120-number residual vector for each token. Learned
matrix projections produce several working vectors:

- **query (`q`)** asks what information to read from the state;
- **key (`k`)** identifies where and how to write new information;
- **value (`v`)** is the information offered for writing;
- **`b`** becomes the update-strength gate `beta`;
- **`a`** contributes to the state-decay gate; and
- **`z`** gates the later normalized output.

The `q`, `k`, and `v` projections first pass through a depthwise causal
convolution and a SiLU activation. CPU-002 implements the inspectable
convolution/state primitive and recurrent rule. It does not yet multiply the
real 5,120-wide activation by all admitted model tensors; that belongs to the
full scalar layer path.

## Heads and the three-to-one mapping

A **head** is an independent group of channels that owns its own state and can
learn a different kind of relationship. Qwen3.8's GDN contract is:

| Quantity | Production value |
|---|---:|
| key/query heads | 16 |
| value heads | 48 |
| key width per head | 128 |
| value width per head | 128 |
| convolution width | 4 tokens |

There are three times as many value heads as key heads. The official model uses
`repeat_interleave`: key head 0 serves value heads 0, 1, and 2; key head 1 serves
value heads 3, 4, and 5; and so on. In integer arithmetic the mapping is:

```text
key_head = value_head / 3
```

where `/` discards the remainder. This is not round-robin mapping. Mapping value
head 1 to key head 1 would silently change the model. The CPU fixture uses two
distinct key heads and six value heads specifically so this mistake becomes
visible.

## The recurrent matrix

Each value head owns a `128 × 128` matrix. One axis corresponds to key lanes and
the other to value lanes. Across 48 heads, one production GDN layer stores:

```text
48 × 128 × 128 = 786,432 FP32 numbers
786,432 × 4 bytes = 3,145,728 bytes
```

**Estimated:** all 48 GDN layers therefore need 150,994,944 bytes, exactly
144 MiB, for recurrent matrices. **Proposed:** the final CUDA allocation and its
measured overhead remain BLD-003 and MEM-001 work. This arithmetic does not prove
that the 128K engine fits.

Quartz stores recurrence in FP32 because a small rounding difference is fed into
every future token and can accumulate over a long sequence. Other temporary
activations may use BF16 in the production engine, but CPU-002 keeps its focused
inputs and outputs in FP32 for inspection.

## Normalizing query and key

Before recurrence, each query and key head is L2-normalized:

```text
length = sqrt(x[0]² + x[1]² + ... + 0.000001)
normalized_x[i] = x[i] / length
```

The small epsilon prevents division by zero for an all-zero vector. The query is
then divided by `sqrt(key_width)`. With the production width that is
`sqrt(128)`. Normalization keeps vector magnitude from unintentionally dominating
the read/write geometry; query scaling controls the output magnitude.

The order and precision are part of the numeric contract. Quartz sums squares
from the first lane to the last in FP32 in the scalar oracle. Parallel CUDA may
round differently and must be checked against the frozen tolerances.

## The two gates

The update-strength gate is:

```text
beta = sigmoid(b) = 1 / (1 + exp(-b))
```

It lies between zero and one. Near zero, a token barely corrects the state; near
one, it makes a strong correction.

The model also computes a non-positive log decay:

```text
log_decay = -exp(A_log) × softplus(a + dt_bias)
decay     = exp(log_decay)
```

`softplus(x) = log(1 + exp(x))` is always positive, so `log_decay` is non-positive
and `decay` lies between zero and one. Multiplying the old state by this decay
forgets some earlier information. [`src/gdn.cpp`](../src/gdn.cpp) uses stable
branches for sigmoid and softplus so very large positive or negative inputs do
not cause avoidable overflow.

## The delta rule, one step at a time

For each value head and token, the official recurrent fallback performs these
operations in this order:

1. Decay the old state: `S = S × decay`.
2. Ask what value the decayed state already predicts for this key:
   `prediction = kᵀS`.
3. Measure the gated error: `delta = (v − prediction) × beta`.
4. Write that correction: `S = S + k × deltaᵀ`.
5. Read the updated state: `output = qᵀS`.

The outer product `k × deltaᵀ` is a matrix: every key lane is multiplied by
every value-error lane. The rule writes only the difference between the offered
value and the state's prediction. That is why it is a **delta** rule.

For a one-number teaching example, suppose normalization has already happened,
the old state is `2`, decay is `0.5`, key is `0.5`, value is `3`, beta is `0.25`,
and query is `0.25`:

```text
decayed state = 2 × 0.5 = 1
prediction    = 0.5 × 1 = 0.5
delta         = (3 − 0.5) × 0.25 = 0.625
updated state = 1 + 0.5 × 0.625 = 1.3125
output        = 0.25 × 1.3125 = 0.328125
```

Reading before the update would return `0.25`, not `0.328125`. Updating before
decay or omitting prediction subtraction would also differ. The explicit order
in [`gdn_recurrent_step`](../src/gdn.cpp) is therefore semantic, not merely an
implementation preference.

## Causal depthwise convolution and warm-up

A causal operation may use the current token and earlier tokens, never a future
token. The GDN projection convolution has width four. **Depthwise** means every
channel has its own four weights and history; channels are not mixed by this
convolution.

At a new session, missing history is zero. For one channel the four-position
state begins `[0, 0, 0, 0]`. Receiving values 10 and then 20 changes it to:

```text
after 10: [0, 0,  0, 10]
after 20: [0, 0, 10, 20]
```

With oldest-to-newest weights `[1, 2, 3, 4]`, the raw first result is `40` and
the second is `110`. SiLU, `x / (1 + exp(-x))`, is applied afterward. This
zero-filled beginning is **convolution warm-up**. Starting with uninitialized
memory, reversing the weights, or shifting after the dot product changes the
first tokens and all downstream recurrence.

The logical state is shown as a shifting row because that is easiest to audit.
A fast implementation may use a circular, or **ring**, buffer and move an index
instead of moving values. It must serialize the same four logical positions and
produce the same results.

## State mutation and atomic sessions

Both the convolution history and recurrent matrix mutate at every token. They
are part of a session's committed token prefix. A saved checkpoint must contain
every matrix and history row; restoring only token IDs would not reconstruct the
same next-token result without replay.

CPU-002 mutates its supplied arrays directly because it is a focused arithmetic
oracle. This does not establish product-level atomicity. SES-002 must stage all
48 GDN states, all convolution rings, 16 attention KV updates, the position, and
the token so an error or cancellation cannot expose a half-committed layer stack.

## Token-wise execution and chunk boundaries

A **chunk** is a consecutive group of tokens processed together for efficiency.
Recurrence still has a strict dependency:

```text
token 0 state → token 1 state → token 2 state → ...
```

Splitting five tokens as `[5]`, `[2, 1, 2]`, or `[1, 1, 1, 1, 1]` must not reset,
skip, or duplicate state. **Measured:** the CPU-002 diagnostic produces
byte-identical outputs and final states for all three splits. This focused scalar
test does not admit the future 64-token parallel scan; GDN-002 must prove that
optimized chunk algorithm against token-wise execution.

## Fixtures, tolerances, and authority labels

[`tools/generate_gdn_fixtures.py`](../tools/generate_gdn_fixtures.py) is an
explicit scalar transcription of the pinned Transformers fallback. It produces
gate values, five recurrent outputs, the final recurrent matrices, six
convolution outputs, and final warm-up histories. It freezes these limits:

| Metric | CPU-002 limit |
|---|---:|
| maximum absolute error | `2e-6` |
| maximum relative error | `2e-5` |
| RMS error | `1e-6` |
| minimum cosine similarity | `0.999999` |

Relative error uses `max(abs(expected), 1e-6)` as its denominator so values near
zero do not turn a tiny absolute difference into an unbounded ratio. Tests also
reject NaN and infinity and compare native chunkings exactly.

**External primary:** the equations, shapes, and mutation order come from
Transformers revision `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`; its source
hash and symbols are pinned in the contract.

**Measured local:** the native scalar implementation meets the checked-in
transcription fixtures with maximum observed differences on this run below
`1.2e-7`, and all chunkings are byte-identical.

**Not yet measured:** PyTorch is absent from the focused local environment, so
CPU-002 is not a direct Transformers eager/offloaded trace. ORA-001 still must
run that primary semantic authority, llama.cpp traces, and Quartz together before
freezing model-layer tolerances. The ledger preserves this boundary explicitly.

## What CPU-002 proves—and what it does not

CPU-002 proves a readable implementation of the pinned gate equations, head
reuse, FP32 recurrence order, causal convolution warm-up, persistent mutation,
and chunk-boundary continuity for deterministic small shapes. Count and shape
checks fail closed instead of indexing incompatible arrays.

It does not yet prove real-tensor projection orientation, gated RMS normalization,
the output projection, a complete 5,120-wide layer, 48-layer long-sequence drift,
CUDA equivalence, checkpoint atomicity, 128K fit, language quality, or speed.
Those claims retain separate tasks and evidence gates in the implementation
ledger.
