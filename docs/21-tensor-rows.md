# 21. GGUF dimensions, tensor rows, and matrix-vector multiplication

[Index](README.md) · Implementation tasks: CPU-006 and EDU-007 in
[`implementation_ledger.md`](../implementation_ledger.md)

The quantization chapters explain how to decode one block. A model projection
contains thousands or millions of those blocks arranged as a matrix. This
chapter explains how [`src/tensor.cpp`](../src/tensor.cpp) turns one admitted
GGUF tensor payload into checked rows without accidentally transposing it.

## A vector, a matrix, and an output

A **vector** is an ordered list of numbers. A **matrix** is a rectangular table
of numbers. Multiplying a matrix by an input vector takes one dot product per
matrix row:

```text
matrix = [ 1  2  3 ]     input = [ 2 ]
         [ 4  5  6 ]             [ 0 ]
                                  [-1 ]

output row 0 = 1×2 + 2×0 + 3×(-1) = -1
output row 1 = 4×2 + 5×0 + 6×(-1) =  2
output = [-1, 2]
```

The input length must equal the number of **columns**. The output length equals
the number of **rows**. A learned projection is exactly this operation, although
its rows may be quantized and much larger.

## Why GGUF shapes look reversed

Many Python libraries display a linear layer's weight as
`[output_features, input_features]`. GGML/GGUF records the fastest-changing
physical dimension first. For a two-dimensional matrix:

- GGUF dimension 0 is the number of contiguous input columns in one row;
- GGUF dimension 1 is the number of output rows.

For example, `blk.3.attn_q.weight` has GGUF shape `[5120, 12288]`. Its physical
meaning is 12,288 output rows, each containing 5,120 input weights:

```text
one 5,120-value activation
          │
          ▼
12,288 rows × dot product
          │
          ▼
12,288 projected values
```

Reading the bracketed pair as `[rows, columns]` would instead expect a
12,288-value input and produce 5,120 outputs. The bytes would still be in range,
so this error can look plausible while defining the wrong model.

**External format authority:** the pinned llama.cpp `ggml_tensor` definition
states that `ne[0]` is the first element count and that the next byte stride is
formed from `ne[0]` divided by the format's block size. The exact revision,
source hash, stride evidence, and local interpretation are frozen in
[`pins/tensor_layout_contract.json`](../pins/tensor_layout_contract.json).

## From a column count to row bytes

Quantized rows must end on a complete format block. Row storage is:

```text
blocks_per_row = columns / values_per_block
row_bytes = blocks_per_row × bytes_per_block
```

The admitted examples make the calculation concrete:

| Tensor | Format | Columns | Rows | Bytes per row |
|---|---|---:|---:|---:|
| `token_embd.weight` | Q4_K | 5,120 | 248,320 | `5120/256 × 144 = 2,880` |
| `output.weight` | Q6_K | 5,120 | 248,320 | `5120/256 × 210 = 4,200` |
| `blk.3.attn_q.weight` | Q8_0 | 5,120 | 12,288 | `5120/32 × 34 = 5,440` |
| `blk.0.ssm_conv1d.weight` | F32 | 4 | 10,240 | `4 × 4 = 16` |

The full payload must equal `row_bytes × rows`. A Q4_K row with 5,119 columns is
invalid because it cannot contain an integer number of 256-value blocks. A
payload that ends after 143 of a required 144 bytes is a partial block, not a
short final row.

## The checked TensorView

[`TensorView`](../src/tensor.h) is a non-owning description of already validated
mapped bytes. It records the payload pointer, storage size, GGML type, columns,
rows, and bytes per row. It does not allocate or copy the 19 GB model.

Creating or binding a view proves all of these conditions before arithmetic:

- the format is one of the admitted F32, Q8_0, Q4_K, or Q6_K types;
- dimensions and pointers are nonzero;
- the column count is block-aligned;
- multiplication used to calculate row and storage sizes cannot overflow;
- storage contains exactly the declared complete rows;
- the tensor has exactly two dimensions; and
- its absolute byte range stays inside the read-only model mapping.

`bind_tensor_view` also looks up the exact admitted tensor name. Product code
therefore does not recompute offsets or cast arbitrary file bytes at every layer.

## Decoding a row versus taking its dot product

`tensor_row_decode` expands a selected row to FP32. This is needed for token
embeddings and diagnostic taps. It walks blocks from left to right and places
each decoded segment at the matching column offset.

`tensor_row_dot` avoids storing a fully expanded row. It decodes or multiplies
one block at a time, computes an FP32 partial dot product, and adds block totals
from left to right. `tensor_matvec` calls that row operation for every output
row. The scalar order is deliberately fixed; optimized CUDA reductions may use
a different tree but must pass frozen numeric comparisons.

F32 rows are read explicitly as little-endian 32-bit patterns. Quantized rows
delegate to the already admitted Q8_0, Q4_K, and Q6_K primitives, so there is one
definition of each block format rather than a second decoder hidden in matvec.

## Synthetic and admitted fixtures

Synthetic tests use a visible 3-row × 5-column F32 matrix and two-block matrices
for every quantized format. The asymmetric rows prove that row 0 and row 1 are
not transposed or accidentally sharing the same activation segment. Tests also
reject partial storage and a 31-column Q8_0 row.

[`fixtures/tensor_rows.json`](../fixtures/tensor_rows.json) freezes four rows from
the exact admitted model: token embedding Q4_K, output Q6_K, attention projection
Q8_0, and convolution F32. Each entry records:

- model identity and tensor name;
- row index, format, columns, rows, and physical row bytes;
- SHA-256 of the selected stored row;
- the deterministic activation rule; and
- the expected little-endian FP32 dot-product bytes.

The generator independently maps those row bytes and uses the readable Python
block equations. Pytest first checks the stored-row hash and regenerated result,
then asks the native diagnostic to bind the named tensor from the GGUF and
requires exact dot bytes.

Two failed approaches remain in the implementation ledger. The first test run
could not import the fixture helper because pytest lacked the repository root in
its module path. After that was fixed, the admitted Q4 row appeared to disagree
because the Python oracle decoded only its first block and `zip` silently ignored
the remaining activation. Walking all 20 blocks produced the native result
without changing the native code. Preserving these failures matters: a test
oracle can be wrong even when it is written in a high-level language.

## Proof boundary

**Measured:** mixed-format synthetic matrices and four hashed admitted rows match
exactly. The view rejects incomplete rows, block misalignment, bad output sizes,
unknown names, unsupported ranks, and mapping overrun through explicit status
returns.

CPU-006 proves physical row binding and scalar matvec mechanics. It does not yet
bind all 851 tensors into typed layer structures, run projections, allocate
session state, execute 64 layers, normalize final activations, or compute all
248,320 logits. Those operations remain CPU-004 and will reuse this boundary
instead of introducing new file-layout arithmetic.
