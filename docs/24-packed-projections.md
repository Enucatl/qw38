# 24. Packed projections: where Q, K, V, and gates live

[Index](README.md) · Implementation tasks: CPU-009 and EDU-010 in
[`implementation_ledger.md`](../implementation_ledger.md)

A linear projection can produce several logical results in one matrix
multiplication. Those results are **packed** into one output vector. Packing
saves launches and weight reads, but the runtime must use the model's exact
index layout when it separates the results again.

Quartz has two packed layouts, and they are intentionally handled by two
functions in [`src/projection.cpp`](../src/projection.cpp). The authority hashes,
production ranges, and shapes are frozen in
[`pins/projection_layout_contract.json`](../pins/projection_layout_contract.json).

## Packing and slicing

Imagine one projection that produces a 2-value query and a 2-value key. Running
two matrices would return two vectors. A packed matrix stacks their output rows
and returns one 4-value vector:

```text
packed = [q0, q1, k0, k1]
query  = packed[0:2]
key    = packed[2:4]
```

The bracket notation `[start:end]` is a **slice**: start is included and end is
excluded. Slicing changes the interpretation of positions; it does not perform
new learned arithmetic.

## GDN: three global contiguous ranges

The GDN packed QKV projection produces 10,240 values:

```text
Q = packed[0:2048]       16 heads × 128 lanes
K = packed[2048:4096]    16 heads × 128 lanes
V = packed[4096:10240]   48 heads × 128 lanes
```

These are three ranges across the whole vector. In the small test, each range
has four values:

```text
packed = [0,1,2,3, 10,11,12,13, 20,21,22,23]
Q      = [0,1,2,3]
K      = [10,11,12,13]
V      = [20,21,22,23]
```

The width-four depthwise causal convolution operates on the entire packed
10,240-channel result first. Quartz splits the convolved result afterward. This
ordering matters because every packed channel owns its own convolution history.

Chapter 22 described a second boundary: the GGUF converter tiles value heads.
Slicing finds the V segment but does not undo that permutation. After slicing,
the scheduler converts V from GGUF tiled order to the grouped order used by the
readable recurrence. Q and K are not value-associated and need no permutation.

## Attention: two halves inside every head

The attention query projection produces 12,288 values. It is best viewed as
shape `[24 heads, 2 roles, 256 lanes]`. Within each head, the first 256 lanes are
query and the next 256 are output gate:

```text
head 0: [Q0 lanes..., gate0 lanes...]
head 1: [Q1 lanes..., gate1 lanes...]
...
```

This is not one global query half followed by one global gate half. A small
three-head, two-lane example makes the distinction visible:

```text
packed = [0,1, 10,11, 100,101, 110,111, 200,201, 210,211]
query  = [0,1, 100,101, 200,201]
gate   = [10,11, 110,111, 210,211]
```

A global half split would incorrectly place head 1's query in the gate output
and head 1's gate in the query output. `split_attention_query_gate` therefore
walks one head at a time.

The split returns raw gate values. The attention core applies `sigmoid` later,
at the semantic point where the gate scales the attended output. Splitting must
not apply activation functions or RoPE.

## Counts, overflow, and aliases

Both functions calculate expected counts with checked multiplication. The GDN
input must equal `2 × key_heads × key_width + value_heads × value_width`.
Attention input must equal `2 × query_heads × head_width`. Every output count
must match its logical role exactly.

An **alias** occurs when input and output refer to overlapping memory. These
splitters reject all overlap between the packed input and outputs and between
the outputs themselves. Without that rule, writing an early query value could
overwrite a packed gate value that has not yet been read. Separate workspaces
make the operation order obvious and preserve the packed diagnostic tap.

The overlap check converts addresses to integer ranges only for comparison and
checks byte-count overflow first. It does not read outside any supplied range.

## Tests and proof boundary

[`tests/test_projection.py`](../tests/test_projection.py) freezes both small
worked examples above as exact little-endian FP32 values. It also proves that an
aliased attention output and a short packed count return explicit errors.

**External:** the pinned Transformers implementation defines the semantic
shapes and splits; the pinned llama.cpp converter defines the GGUF value-head
reordering that follows the GDN slice.

**Measured:** both synthetic layouts produce the exact expected lanes, and
invalid counts and aliases fail closed in the restricted native build.

CPU-009 proves only index interpretation. It does not prove that a real matrix
projection produced correct values, nor does it execute convolution, attention,
recurrence, residuals, or logits. Those remain CPU-004 work and will use these
small, explicit boundaries rather than repeating offset arithmetic.
