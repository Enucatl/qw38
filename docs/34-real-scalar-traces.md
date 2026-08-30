# 34. Real scalar taps and bundle capture

[Index](README.md) · Implementation tasks: TRC-002 and EDU-020 in
[`implementation_ledger.md`](../implementation_ledger.md)

TRC-002 connects the diagnostic boundary from Chapter 33 to the real 64-layer
scalar execution from Chapter 31. A requested tensor now travels through this
complete path:

```text
real mapped GGUF -> scalar stage -> filtered borrowed view
                 -> native FP32 copy -> Python typed reader
                 -> trace-v1 manifest + checked tensor blob
```

This proves that a trace describes an actual runtime value rather than a
synthetic test array. At TRC-002 it was still native structural evidence;
ORA-001 subsequently compared it with pinned Transformers and llama.cpp traces.

## Why the release entry point did not change

[`execute_scalar_token`](../src/scalar_runtime.cpp) remains the ordinary entry
point. Only a build with `QW38_DIAGNOSTIC_TRACE` also exposes
`execute_scalar_token_traced`. Both call the same internal loop. Preprocessor
guards remove the trace arguments, branches, extra attention RoPE buffers, tap
names, and callback calls from normal object code.

This matters more than merely passing a null callback. A null callback in a
normal binary would still leave conditional branches and trace-specific memory
layout in the production path. Compile-time removal lets later performance
measurements describe the actual release runtime.

## When each tap is offered

“When” is part of a tap's meaning. Quartz offers:

- `embedding` immediately after decoding the token row;
- mixer and FFN taps after that complete layer has executed, while its shared
  scratch still contains the just-produced intermediates;
- GDN convolution and recurrent state after their one-token mutation;
- attention KV rows after the current position is committed to those caches;
- `layer_residual` before the ping-pong buffers swap;
- `final_norm` after final RMSNorm; and
- `logits` after all 248,320 vocabulary rows have been projected but before the
  session frontier advances.

The sink copies a matching tensor during the callback. It cannot retain the
borrowed pointer because the next layer reuses the same workspace.

## Raw Q/K versus normalized RoPE Q/K

Attention has two distinct query/key observations:

```text
attention.query / attention.key
    raw projection results

attention.rope_query / attention.rope_key
    per-head normalized values after partial RoPE at this position
```

The attention implementation computes normalized RoPE Q/K in local arrays.
Diagnostic builds copy those exact arrays into diagnostic-only workspace before
attention scores are calculated. The KV key row equals the current RoPE key,
while the KV value row stores the unrotated value projection. Keeping all these
names separate makes a projection error distinguishable from normalization,
rotation, or cache-addressing errors.

`attention.context` is the grouped causal attention result after the output
gate and before the output matrix. `attention.output` is the 5,120-value output-
matrix correction before the mixer residual addition.

## The pinned tap table

[`pins/scalar_trace_contract.json`](../pins/scalar_trace_contract.json) freezes
every name, applicable layer kind, and shape. Important shaped state includes:

| Tap | Shape | Meaning |
|---|---:|---|
| `gdn.query` | 16 × 128 | key-head query channels |
| `gdn.recurrent_output` | 48 × 128 | value-head recurrence result |
| `gdn.recurrent_state` | 48 × 128 × 128 | persistent FP32 delta matrices |
| `gdn.convolution_state` | 4 × 10,240 | persistent causal convolution ring |
| `attention.query` | 24 × 256 | raw query heads |
| `attention.rope_key` | 4 × 256 | normalized, partially rotated KV heads |
| `attention.context` | 24 × 256 | grouped causal result after output gate |
| `ffn.gate` | 17,408 | wide gate projection |
| `layer_residual` | 5,120 | complete decoder-layer output |
| `logits` | 248,320 | vocabulary scores |

The source-coverage test requires every pinned spelling to appear in both the
registry and scalar runtime offer sites. Runtime shape validation independently
checks the dimension product against the offered element count.

## One exact filter per capture

The focused capture command accepts one physical layer and one exact tap. Global
taps use the word `global`:

```sh
build/qw38-eval-diagnostic \
  --capture-real-scalar-trace models/Qwen3.8-27B-Q4_K_M.gguf \
  42 global final_norm /tmp/final-norm.f32le.bin
```

Requiring one exact match bounds disk use and prevents an accidental `all *`
capture from writing every large state tensor. An unknown tap, wrong layer kind,
zero matches, multiple matches, existing output, or failed write returns an
explicit error and removes a partial raw file.

The native command emits only a temporary FP32 tensor and small metadata. It
does not implement JSON. This preserves the narrow C++ runtime boundary.

## Turning the native copy into trace v1

[`capture_scalar_trace.py`](../tools/capture_scalar_trace.py) orchestrates the
full evidence path. It:

1. hashes the complete GGUF and requires the pinned model identity;
2. runs the diagnostic executable with the exact filter;
3. validates the native byte count and unpacks little-endian FP32;
4. records the token, position, before/after frontiers, and hashes of all four
   state slabs;
5. identifies the executable by source revision and executable SHA-256;
6. writes the selected tensor using the Chapter 32 typed writer; and
7. re-reads and fully validates the completed bundle before returning success.

The whole-model hash is deliberately not skipped. It is slower than trusting a
filename, but a trace attributed to the wrong 19 GB artifact is unusable oracle
evidence. Later caching may optimize this only behind an equally fail-closed
identity check.

## State hashes and scope

The before/after session snapshots hash four complete owning slabs:

- all GDN convolution rings;
- all GDN recurrent matrices;
- all attention key rows; and
- all attention value rows.

For the zero-state token-42 capture, before and after hashes differ and the
frontier moves from zero to one. A focused tensor bundle therefore retains proof
of the surrounding session transition without duplicating roughly 152 MiB of
state into every small trace.

Checksums show exact mutation identity, not semantic correctness. A future
checkpoint trace can include selected full state tensors when an element-wise
comparison is needed.

## Failure and atomicity boundary

Trace callback failure can occur after a layer has mutated GDN or KV state. This
diagnostic entry point returns the failure and does not advance the frontier,
but it does not roll back earlier state. That is acceptable for a disposable
diagnostic session. It must not back the public atomic `Session::eval`; SES-002
will stage or restore public state on any execution or sink failure.

## Measured evidence

[`tests/test_real_scalar_trace.py`](../tests/test_real_scalar_trace.py) captures
the complete 5,120-value real final norm for token 42, writes and re-reads trace
v1, then compares indices 0, 1, 2,559, and 5,119 exactly with the existing frozen
native scalar fixture. It also checks pinned model identity, token/position,
frontiers, global layer encoding, and shape. A fake model is rejected by hash
before native execution.

The focused trace suite completed 21 tests in 32.33 seconds before the RoPE tap
retention refinement; the attention/scalar subset then completed 11 tests in
34.47 seconds. These timings include hashing and one real full-model scalar run
and are diagnostic measurements, not throughput claims.

The first diagnostic rebuild after adding RoPE-only fields failed because an
older attention fixture workspace initializer did not supply those new fields.
The fields were added under the same compile guard with dedicated buffers; both
normal and diagnostic builds then passed. This negative result is retained in
the implementation ledger.

## Proof boundary

**Measured:** every pinned scalar tap is registered and offered in diagnostic
code; normal code compiles without trace fields; one filtered real final-norm
tensor copies exactly, survives trace-v1 validation, and matches the frozen
native structural fixture at its admitted indices.

**Subsequently measured:** ORA-001 captured corresponding pinned Transformers
and llama.cpp tensors and froze per-tap tolerances in
[Chapter 38](38-scalar-authority-tolerances.md). Future CUDA kernels must offer
the same names at visible unfused boundaries and pass those immutable gates;
this scalar chapter alone still does not claim CUDA taps exist.
