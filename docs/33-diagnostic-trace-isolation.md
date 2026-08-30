# 33. Diagnostic trace isolation and filters

[Index](README.md) · Implementation tasks: TRC-003 and EDU-019 in
[`implementation_ledger.md`](../implementation_ledger.md)

Chapter 32 defined the evidence files. TRC-003 defines the smaller C++ boundary
that model code will use to offer a tensor to a trace writer. It deliberately
does not yet connect the real 64-layer scalar runtime; that remains TRC-002.

## Why there are two builds

Capturing an activation means inspecting it, deciding whether it matches a
filter, and often copying millions of bytes. Those operations would add code,
branches, strings, and cost to every normal token if compiled into production.

Quartz therefore has two binaries:

```text
build/qw38-eval             normal release-style diagnostic utility
build/qw38-eval-diagnostic  expensive tensor tracing enabled
```

`make diagnostic` compiles a separate object directory with
`QW38_DIAGNOSTIC_TRACE` defined. [`diagnostic_trace.h`](../src/diagnostic_trace.h)
refuses to compile without that definition, and the source is not in the normal
library source list. This is **compile-time isolation**: the release binary does
not merely turn tracing off at runtime; it contains neither the trace command
nor stable tap-name strings.

## A backend-neutral sink

A **sink** is a callback that receives one tensor view. The view contains:

- a stable semantic tap name;
- a physical layer number, or the global sentinel;
- a non-owning pointer and element count; and
- one to three exact shape dimensions.

“Non-owning” means the view borrows runtime memory only for the callback. The
sink must copy values it wants to retain. This avoids allocating or copying when
a tap is filtered out.

The callback returns an explicit `Status`. A future scalar or CUDA caller can
stop execution when the sink cannot write evidence. The interface contains no
JSON, filesystem, Python, CPU-workspace, or CUDA-buffer type, so both backends
can expose the same semantic boundary. CUDA will still need an explicit device-
to-host capture before calling a host file sink.

## Exact filters

A filter has a layer and tap name. The layer is `0` through `63` or `all`; the
name is one registered name or `*`. Matching uses exact text:

```text
3 attention.rope_query   matches only that layer and tap
all final_norm           matches the global final norm
all *                    matches every offered tap
```

`attention.queryish` is rejected rather than treated as an empty selection.
This catches spelling mistakes that could otherwise produce an apparently
successful but useless trace. A valid filter can still match zero tensors—for
example, requesting an attention tap at GDN layer 0—and the eventual manifest
makes the empty result visible.

## Stable names are an API

[`diagnostic_trace.cpp`](../src/diagnostic_trace.cpp) registers names for the
approved observations: embedding, input norms, GDN projection/convolution/
recurrence/state/output, attention packed Q/gate and RoPE Q/K/KV/output, both
residuals, FFN stages, final norm, and logits.

These names describe semantics, not temporary variable names. A fused CUDA
kernel may compute several stages together, but its visible admitted boundary
must still compare under the same name. Renaming a tap changes the versioned
diagnostic API and cannot happen silently.

## Shape validation before the sink

Before invoking a sink, Quartz verifies that:

- the name is registered and the layer is valid;
- the data pointer is non-null and count is nonzero;
- rank is between one and three;
- every dimension is positive and multiplication cannot overflow; and
- the dimension product equals the element count.

This is the tensor-view equivalent of checking blob ranges in Chapter 32. A
shape `[2, 3]` paired with five values must fail before a writer records
misleading evidence.

## Synthetic diagnostic and evidence

The temporary `--check-trace-filter` diagnostic offers three tiny typed views.
It proves wildcard selection, exact layer/name selection, valid zero matches,
and fail-closed malformed filters without loading the 18.97 GB model. It is a
boundary test, not a model trace.

[`tests/test_diagnostic_trace.py`](../tests/test_diagnostic_trace.py) also reads
the normal binary bytes and requires the diagnostic command and representative
tap strings to be absent. This guards the compile-time isolation claim directly.

## Proof boundary

**Measured:** diagnostic and normal objects build separately; valid exact and
wildcard filters select typed synthetic tensor views; invalid layer/name filters
fail; representative trace command/name strings are absent from the normal
binary.

**Proposed:** TRC-002 will offer actual scalar tensors at each registered point,
copy selected values into the Chapter 32 bundle, and compare them with pinned
authority traces. No model-semantic claim follows from the synthetic views.
