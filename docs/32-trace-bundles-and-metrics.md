# 32. Trace bundles and numeric comparison

[Index](README.md) · Implementation tasks: TRC-001 and EDU-018 in
[`implementation_ledger.md`](../implementation_ledger.md)

A final answer can be wrong even when the program runs, every array has the
expected size, and every number is finite. To find the first point where two
implementations disagree, Quartz needs evidence from *inside* the calculation.
That evidence is a diagnostic trace.

TRC-001 defines how Quartz stores and compares traces. It does not yet capture
the real 64-layer runtime; that wiring is TRC-002. Separating the file contract
from the model code lets us test malformed files, checksums, and comparison
rules with tiny arrays before a trace costs gigabytes.

## What a trace and a tap are

A **trace** is a recorded execution. A **tap** is one named observation point in
that execution, like attaching a voltmeter at a known point in a circuit.
Examples for the model are:

```text
embedding
layer.0.input_norm
layer.0.gdn.recurrent_output
layer.3.attention.rope_query
layer.63.residual
final_norm
logits
```

If Quartz and Transformers have equal embeddings but different layer-0
normalized input, the fault is already localized to that normalization step.
Comparing only the final chosen token would lose that information.

Tracing every temporary on every normal request would consume substantial time
and storage and could prevent CUDA fusion or graph capture. The approved design
therefore compiles expensive tensor tracing only into diagnostic builds.
TRC-002 will add layer and tap filters so a developer records only the evidence
needed for one investigation.

## The two files in a bundle

Each v1 bundle is a directory containing:

```text
manifest.json
tensors.f32le.bin
```

The **manifest** is readable JSON. It says what was run and how to interpret the
binary data. The **blob** is a compact sequence of tensor values. JSON is useful
for names and metadata, but encoding millions of numbers as decimal text would
be slow, large, and vulnerable to decimal round-trip differences.

[`pins/trace_contract.json`](../pins/trace_contract.json) freezes the schema name
`qw38.trace`, version `1`, filenames, data type, layout, and comparison rule.
[`tools/qw38_trace.py`](../tools/qw38_trace.py) is the typed writer, reader, and
comparator. A reader rejects unknown versions rather than guessing.

## FP32 and little endian

**FP32** means each value uses the IEEE-style 32-bit floating-point storage used
by the scalar taps. Each element occupies exactly four bytes. A tensor with
shape `[2, 3]` has six values and therefore occupies 24 bytes.

A multi-byte number has to specify byte order. **Little endian** stores the
least-significant byte first. For example, FP32 `1.0` has bit pattern
`0x3f800000`, stored as these four bytes:

```text
00 00 80 3f
```

Quartz calls the dtype `f32-le`: 32-bit float, little endian. The writer first
rounds Python numbers to exactly these FP32 bytes, then computes summaries from
the values read back from those bytes. The summary consequently describes the
file, not a higher-precision temporary used to create it.

## Shape, offset, and length

A tensor record contains its:

- unique name and semantic role;
- optional physical layer number;
- shape and `f32-le` dtype;
- byte offset and byte length in the blob;
- SHA-256 checksum; and
- numeric summary.

The **shape** describes the logical axes. `[2, 3]` means two rows of three
values. Its dimension product must equal the number of stored elements.

The **offset** is the number of bytes from the beginning of the blob to this
tensor. The first tensor begins at zero. V1 requires every following tensor to
begin exactly where the previous one ends. Overlaps, gaps, out-of-bounds ranges,
and unclaimed trailing bytes are rejected. This intentionally simple layout
prevents two readers from interpreting the same bytes differently.

## Checksums are identity, not correctness

SHA-256 turns any byte sequence into a 64-character hexadecimal digest. A
one-bit change almost certainly changes the digest. Quartz records a checksum
for:

- the original prompt bytes;
- the complete tensor blob;
- every individual tensor range; and
- each named session-state component before and after execution.

A matching checksum proves that the compared bytes are the bytes named by the
manifest. It does **not** prove that the mathematical result is correct. Two
implementations can deterministically produce the same wrong tensor, and a
correct tensor can have a different checksum after an allowed floating-point
rounding change. Semantic agreement is decided by the numeric metrics below.

## Pinning who produced the evidence

Both `model` and `tool` have exactly three identity fields:

```text
name       human-readable artifact or executable name
revision   source/model revision that defines its behavior
sha256     exact content hash
```

Without these fields, a trace named “Transformers output” could silently come
from a later library version or different weights. The identity does not grant
authority by itself; it makes the authority claim auditable against the pins.

## Prompt bytes, tokens, and positions

The manifest retains three views of model input:

1. exact prompt bytes, base64-encoded so arbitrary UTF-8 bytes survive JSON;
2. exact rendered token IDs; and
3. the position assigned to every token.

Base64 is a reversible text spelling of bytes. It is not encryption. Quartz
stores the original byte count and SHA-256 so truncation or modification fails
validation.

All three input views matter. Equal prompt text can tokenize differently if the
tokenizer or template changed. Equal token IDs can still execute differently if
their positions changed, because RoPE uses position. TRC-001 requires token and
position arrays to have equal length and contain non-negative integers.

## Session frontiers before and after

The **frontier** is the number of committed token positions in a session. A
trace records a snapshot before and after execution:

```text
before frontier: 7
input positions: 7, 8
after frontier: 9
```

Each snapshot also maps named persistent-state components to SHA-256 checksums.
The names can identify GDN recurrent matrices, convolution rings, attention K/V
rows, and later sampler state. This makes unexpected mutation visible even when
the complete state tensors are filtered out of a small trace.

The trace reader validates snapshot structure and checksums but does not assume
that every diagnostic advances the frontier. A failed or read-only diagnostic
may legitimately have equal before/after frontiers. Session APIs will enforce
their stronger atomic commit rules in SES-002.

## Tensor summaries

Every tensor record includes:

```text
element count and finite count
NaN, +infinity, and -infinity counts
minimum and maximum finite values
mean of finite values
root-mean-square magnitude of finite values
```

The reader recomputes this summary from the blob and requires exact agreement
with the manifest. A summary helps a person spot an exploded activation or an
all-zero output without loading the full array. It cannot replace element-wise
comparison: arrays `[1, 3]` and `[3, 1]` have identical summaries but different
meaning.

## Absolute error

For expected value `e` and actual value `a`:

```text
absolute error = |a - e|
```

If `e = 2.0` and `a = 2.001`, the absolute error is `0.001`. Absolute error is
easy to interpret near zero, but a fixed `0.001` means something very different
for a value near one million.

Quartz reports the maximum absolute error and its first index. An absolute
tolerance sets the small fixed difference a comparison permits.

## Relative error

Relative error scales the difference by the expected magnitude:

```text
relative error = |a - e| / max(|e|, relative_floor)
```

For `e = 1000` and `a = 1000.1`, the absolute error is `0.1`, but the relative
error is only `0.0001`, or 0.01%. The `relative_floor` prevents division by zero
and prevents tiny expected values from producing meaningless enormous ratios.

A finite element fails when **both** its absolute error exceeds the absolute
tolerance **and** its relative error exceeds the relative tolerance. This common
“absolute or relative closeness” rule lets small values use the absolute gate
and large values use the scaled gate. Tolerances must be finite and non-negative
and are frozen during the scalar oracle milestone; an optimized path may not
loosen the gate that admits it.

## RMS error

Root mean square error summarizes typical error magnitude:

```text
RMS = sqrt((d0² + d1² + ... + dn²) / n)
```

Squaring makes large differences count more heavily, averaging avoids dependence
on tensor length, and the square root returns to the original unit. Quartz uses
all finite pairs. Non-finite values are counted and fail separately.

RMS can hide one severe outlier among millions of good elements, which is why
the maximum and first-failing index remain mandatory.

## Cosine similarity

Cosine similarity compares vector direction rather than raw scale:

```text
cosine = dot(expected, actual) / (length(expected) * length(actual))
```

Identical nonzero vectors have cosine 1. A result near 1 can show that a tensor
has the right overall pattern even when magnitudes drift. It cannot admit a
result alone: multiplying every value by two preserves cosine but is numerically
wrong. Quartz reports no cosine (`null`) if either vector has zero norm or any
input is NaN or infinite.

## NaN, infinity, and the first failure

**NaN** means “not a number,” often produced by an invalid operation. Positive
or negative **infinity** usually means overflow or division by zero. Any NaN or
infinity fails numeric admission, even if both implementations contain it at the
same index. Treating `NaN == NaN` as success could allow two equally broken paths
through a gate.

The **first-failing index** is the earliest flat element that violates the
non-finite or tolerance rule. It is stable and immediately actionable: shape and
layout information let a developer map that flat index back to a head, row, or
channel. `null` means every element passed.

## Top-logit differences and near ties

Logits are one score per vocabulary token. Greedy decoding selects the highest
score. Two nearly equal top scores can swap order after a tiny permitted numeric
difference, so Quartz separately reports:

- expected and actual top-k token IDs;
- how many token IDs the two sets share;
- whether their exact order matches; and
- the largest score difference among their union.

Ties use lower token ID first, making reports deterministic. This report explains
a possible greedy near-tie; it does not waive tensor or logit gates. Any approved
greedy mismatch needs a stored fixture demonstrating the genuine near-tie.

## Fixture equality versus numeric equality

A **fixture** is a small, checked-in expected result. Exact fixture equality is
appropriate for identities, token IDs, positions, shapes, offsets, hashes, and
other discrete facts: one different bit means the contract changed.

Floating-point computations can legally differ by a small amount when operation
order or hardware changes. Those tensors use the frozen numeric report instead:
absolute, relative, RMS, cosine, non-finite counts, and first failure. Calling
both checks simply “equality” would hide this important distinction.

## Validation and negative evidence

[`tests/test_trace.py`](../tests/test_trace.py) demonstrates:

- deterministic writing and exact little-endian bytes;
- complete model, tool, prompt, token, position, tensor, and session metadata;
- writer rejection of invalid shapes, hashes, and token/position lengths;
- reader rejection of changed versions, dtypes, ranges, lengths, and blob bits;
- all required numeric metrics and the finite tolerance rule;
- mandatory failure on NaN/infinity; and
- top-logit order changes caused by a small near-tie.

The initial focused run also failed because pytest did not place the repository
root on its import path. The explicit `pythonpath = ["."]` test setting resolves
that packaging boundary. Two first-draft assertions incorrectly treated decimal
errors as exact binary ties; they were corrected to assert the actual FP64 metric
ordering. Both negative results remain recorded in the implementation ledger.

## Proof boundary

**Measured:** the typed writer produces deterministic v1 files; the reader
recomputes all byte ranges, checksums, and summaries; corruption and malformed
metadata fail closed; the comparator reports every required metric with explicit
non-finite and top-logit behavior.

**Proposed:** real embeddings, GDN stages/state, attention Q/K/RoPE/KV/output,
FFN stages, residuals, final norm, and logits will use this container in TRC-002.

TRC-001 proves that Quartz can preserve and compare diagnostic evidence. It does
not prove that the current full scalar output is semantically correct. That claim
requires real taps from pinned Quartz, Transformers, and llama.cpp executions in
TRC-002 and ORA-001.
