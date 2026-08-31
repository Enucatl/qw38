# 47. Exact token-prefix synchronization

[Index](README.md) · Implementation tasks: SES-001 and EDU-033 in
[`implementation_ledger.md`](../implementation_ledger.md)

A chat request does not send a GPU state. It sends a token history: a sequence
of integer token IDs such as `[42, 3649]`. The session already holds the model's
memory after some earlier history. Synchronization makes those two descriptions
agree before generation continues.

## Token history, frontier, and common prefix

A **token history** is the exact ordered list already committed to a session.
The **frontier** is the number of committed tokens, so it is also the position
where the next token would be evaluated. Token IDs are compared as integers;
similar-looking text is irrelevant at this boundary because tokenization has
already happened.

The **common prefix** is the identical run at the beginning of two histories.
For example:

```text
committed: [42, 3649]
requested: [42, 1219, 88]
common:    [42]             length 1
```

“Prefix” always starts at position zero. A matching token later in the request
does not count after the first mismatch.

## The deliberately narrow reuse rule

[`sync_tokens`](../cuda/full_scheduler.cu) admits three useful cases:

1. **Append:** committed `[42]`, requested `[42, 3649]`. The entire committed
   history is a prefix, so token `42` is reused and only `3649` is evaluated.
2. **No-op:** committed and requested are both `[42, 3649]`. No model layer is
   evaluated. The session returns its last committed hidden vector and logits.
3. **Empty on empty:** nothing is evaluated and the zero state remains zero.

The session owns a host copy of its last 5,120-value FP32 hidden vector and all
248,320 FP32 logits. This is about 1 MB. Keeping those outputs with the session
means a no-op is correct even if a caller supplies a different scratch
workspace. A workspace is temporary calculation memory; it is not the durable
meaning of a conversation.

If the histories **diverge**, as in the example above, Quartz clears every GDN
state and attention-cache byte and replays the requested history from zero. It
also replays when the requested history is **shorter** than the committed one.
For committed `[42, 3649]` and requested `[42]`, the common prefix has length
one, but the GPU cannot simply move its frontier backward: the 48 recurrent GDN
matrices already contain effects from token `3649`.

Why not retain a state snapshot at every token? One production GDN state is
about 159 MB. Multiplying that by 131,072 positions is vastly beyond a 32 GiB
GPU, before weights or attention cache are counted. SES-001 therefore chooses a
clear invariant: only the complete current history can be reused; arbitrary
rollback means deterministic replay. SES-003 will add explicit disk
checkpoints, not an impossible per-token snapshot array.

## Preflight happens before mutation

**Preflight** means checking the whole request before changing session state.
Quartz verifies the session/workspace capacities, output sizes, request length,
and every token ID. The vocabulary contains IDs `0` through `248319`; `248320`
is rejected. The diagnostic proves that this invalid token leaves the earlier
valid state byte-exact.

This is narrower than request atomicity. A CUDA error or cancellation after
valid execution begins could still leave partial state in SES-001. Candidate
state and frontier-last publication for the whole public request belong to
SES-002.

## What “exact” and fixture equality mean

Floating-point implementations often need error tolerances when two different
algorithms are compared. Prefix reuse is different: both routes execute the
same kernels in the same order from the same bytes. The acceptance gate can
therefore demand **byte-exact** equality for:

- the ordered token IDs and frontier;
- all 48 GDN convolution rings and recurrent matrices;
- all 16 attention key and value caches;
- the last hidden vector; and
- every one of the 248,320 last logits.

**Fixture equality** means a stored, reviewed example says exactly what a test
must observe. The fixture is not merely “the command exited successfully.” For
append `[42] -> [42, 3649]`, it records common-prefix length one, one reused
token, one evaluated token, no reset, and byte-exact equality with a fresh
two-token execution. It records corresponding facts for no-op, divergence,
shortening, empty reset, and invalid-input preservation in
[`cuda_prefix_sync.json`](../fixtures/cuda_prefix_sync.json).

The device comparison kernel checks persistent allocations without copying
roughly 159 MB back to the host. One mismatch flag is set if any byte differs.
Host token/output arrays use ordinary byte comparison. Exact equality is safe
here because replay deliberately preserves operation order; it is not a general
claim that mathematically equivalent GPU programs always have identical bits.

## Measured result and proof boundary

**Measured local:** on the RTX 5090 with CUDA 13.0.2, append reuse, no-op reuse,
divergent replay, shorter replay, empty reset, and invalid-token preflight all
passed. Append, divergent, and shorter final states and outputs were byte-exact
against fresh execution. SES-001 does not claim a performance result; OPT-001
owns synchronized timing and attribution.

The authenticated rules and source hashes are in
[`cuda_prefix_sync_contract.json`](../pins/cuda_prefix_sync_contract.json).
This gate proves exact current-prefix reuse and replay correctness at capacity
three. Its proof boundary excludes CUDA-failure/cancellation atomicity
(SES-002), checkpoint persistence (SES-003), the simultaneous 128K memory-fit
gate (MEM-001), and server-level response continuation.
