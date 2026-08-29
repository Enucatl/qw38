# 23. Typed model weights: turning 851 tensors into one safe model

[Index](README.md) · Implementation tasks: CPU-008 and EDU-009 in
[`implementation_ledger.md`](../implementation_ledger.md)

The GGUF inventory is a list of names and byte ranges. A forward pass needs a
more useful answer: “give me layer 17's input norm” or “give me layer 31's key
projection.” [`src/weights.cpp`](../src/weights.cpp) performs that one-time
translation and refuses to publish a model if any expected field is missing or
incompatible.

## Inventory versus typed weights

An **inventory** resembles a warehouse manifest:

```text
name                         shape          format
blk.3.attn_k.weight          [5120, 1024]   Q8_0
blk.3.attn_k_norm.weight     [256]          F32
...
```

This is ideal for validation and diagnostics, but repeatedly searching strings
during every token would be slow and error-prone. A **typed weight structure**
has named C++ fields instead:

```text
model.layers[3].attention.key
model.layers[3].attention.key_norm
```

The compiler can now distinguish a matrix view from a vector view and common
weights from GDN- or attention-specific weights. The forward scheduler will
consume these fields; it will not construct filenames or interpret GGUF offsets.

“Typed” here does not mean a generic tensor type system. Quartz remains narrow:
the structures in [`src/weights.h`](../src/weights.h) describe only the one
admitted Qwen3.8 model.

## Views do not own the 19 GB model

A **view** is a checked description of bytes owned elsewhere. `MappedFile` owns
the operating-system mapping. `TensorView` points to a matrix payload and stores
its columns, rows, format, row bytes, and total bytes. The new `VectorView`
points to a one-dimensional F32 payload and stores its element count and bytes.

Neither view copies weights. If five structures refer to five portions of the
mapped file, the model still exists once in virtual memory:

```text
MappedFile owns model mapping
   ├── TensorView -> token embedding byte range
   ├── VectorView -> final norm byte range
   └── LayerWeights[0..63] -> their exact byte ranges
```

The mapping must outlive every view. Engine ownership will enforce that
lifetime when these structures are integrated into `Engine`; this gate proves
the binding object itself.

`VectorView` admits only nonempty, complete F32 storage. Its byte rule is
`count × 4 == storage_bytes`, with checked multiplication. `vector_decode`
reads each 32-bit value explicitly as little-endian bytes, so it does not depend
on an unaligned pointer cast or silently accept a partial final value.

## The common layer and its two variants

All 64 layers have five common tensors:

- input RMSNorm;
- FFN gate projection;
- FFN up projection;
- FFN down projection; and
- post-attention/FFN RMSNorm.

A GDN layer then adds nine tensors: packed QKV, value gate, alpha, beta,
four-tap depthwise convolution, folded A, time bias, GDN norm, and output
projection. Five common plus nine GDN-specific fields gives 14 tensors.

An attention layer adds six tensors: query/gate, key, value, query norm, key
norm, and output projection. Five common plus six attention-specific fields
gives 11 tensors.

The official schedule repeats three GDN layers followed by one attention layer.
For a zero-based layer number, `layer % 4 == 3` identifies attention layers:
3, 7, 11, and so on through 63. This produces 48 GDN layers and 16 attention
layers.

The complete count is a useful invariant:

```text
3 global tensors
+ 48 GDN layers × 14 tensors
+ 16 attention layers × 11 tensors
= 3 + 672 + 176
= 851 tensors
```

If binding consumes 850 or 852, it is not the admitted model even if individual
pointers appear plausible.

## What “exact binding” checks

Each field is admitted only when all of these agree:

- exact tensor name, including layer number;
- semantic role assigned during artifact validation;
- rank: one dimension for a vector or two for a matrix;
- every dimension in GGUF physical order;
- exact storage format: F32, Q8_0, Q4_K, or Q6_K as specified;
- complete row or vector byte size; and
- absolute payload range inside the read-only mapping.

For example, an attention output projection must be named
`blk.N.attn_output.weight`, have role `attention_output`, GGUF shape
`[6144, 5120]`, and use Q6_K. A Q4_K matrix with the same shape is rejected.
A correctly shaped key projection in a GDN layer is also rejected because that
variant must contain the packed GDN fields instead.

This schema check intentionally overlaps artifact validation. The loader proves
the file is admissible; typed binding proves every scheduler field receives the
right admitted object. The second check is close to the code that will use the
field, which makes future edits fail at model open rather than much later during
generation.

## Publishing only a complete model

`bind_model_weights` fills a local candidate. It assigns that candidate to the
caller's output only after all 851 bindings and the final count succeed. A
missing tensor can leave the temporary candidate partly filled, but it cannot
publish a partly valid `ModelWeights` object.

This is **fail closed** behavior: uncertain input is rejected instead of guessed.
The returned `Status` explains the category—typed matrix/vector mismatch or
mapped range failure—and no exception-based recovery is required.

## Tests and the preserved failed expectation

[`tests/test_weights.py`](../tests/test_weights.py) binds the exact pinned model
and requires:

- 851 bound tensors;
- 48 GDN and 16 attention layers;
- 5,120 embedding columns and 248,320 token rows;
- 5,120 final-norm values and 248,320 logit rows; and
- exact first/last decoded final-norm FP32 bytes.

Four copies of the parsed metadata are deliberately corrupted in memory: one
name, one semantic role, one shape, and one range. Every case must return an
error instead of producing weights.

The first endpoint expectation was guessed before inspecting the payload and
failed: expected `0000903f00008d3f`, actual `0000fb3f0000f13f`. That is not a
native decoder defect. An independent read matched the inventory's frozen
SHA-256 and the actual endpoint bytes, so the fixture was corrected without
changing C++. The failure remains in the ledger because an expected value is
not evidence merely because it was written in a test.

## Evidence and proof boundary

**Measured:** all 851 ranges in the installed pinned GGUF bind into the typed
structure, the layer and global counts match the arithmetic above, the final
norm vector decodes to independently checked endpoint bytes, and four corrupted
schemas fail closed.

CPU-008 does not yet execute a projection. It also does not allocate activations,
GDN state, convolution rings, KV cache, or logits. CPU-004 will use these views
to implement the scalar 64-layer schedule. CUDA weight upload and repacking are
later gates and may use different physical storage while preserving this typed
semantic boundary.
