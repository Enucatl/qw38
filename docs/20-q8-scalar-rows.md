# 20. Q8_0 blocks and scalar matrix rows

[Index](README.md) · Implementation tasks: CPU-005 and EDU-006 in
[`implementation_ledger.md`](../implementation_ledger.md)

CPU-001 implemented the Q4_K and Q6_K formats named by the original scalar
milestone. Preparing the complete real-weight scheduler exposed another admitted
format: 288 tensors in [`pins/tensor_inventory.json`](../pins/tensor_inventory.json)
use Q8_0. CPU-005 was added to the ledger before implementation so this required
work could not silently appear inside CPU-004.

## Why a “Q4” model contains other formats

`Q4_K_M` describes the artifact's overall quantization recipe, not a promise that
every tensor uses four-bit values. A converter can keep some tensors or
projections at higher precision when their accuracy is more sensitive or their
size tradeoff is worthwhile. The admitted file contains Q4_K, Q6_K, Q8_0, and
FP32 tensors, and Quartz fails closed if that exact inventory changes.

This is similar to a compressed image that stores colour and metadata in
different representations: the filename names the package, while each section
still has its own format. A runtime must inspect each tensor's declared type; it
cannot choose a decoder from the model filename alone.

## The Q8_0 block

One Q8_0 block stores 32 weights in 34 bytes:

| Byte range | Size | Meaning |
|---|---:|---|
| 0–1 | 2 | little-endian FP16 scale `d` |
| 2–33 | 32 | 32 signed 8-bit integers |

An 8-bit signed integer ranges from -128 through 127. Its highest bit carries
the sign according to two's-complement encoding. For example, byte `0x7f` means
127, byte `0x00` means zero, byte `0xff` means -1, and byte `0x80` means -128.
Quartz converts bytes above 127 by subtracting 256 instead of relying on a
platform-dependent narrowing cast.

Decoding is one equation:

```text
weight[i] = FP16_to_FP32(d) × signed_q[i]
```

If `d = 0.25` and a stored byte represents `-32`, the decoded FP32 weight is
`-8`. All 32 integers share the same scale. There are no packed nibbles,
sub-block minima, or split high bits as in Q4_K and Q6_K.

Including its scale, Q8_0 uses `34 × 8 / 32 = 8.5` bits per weight. It is larger
than Q4_K's 4.5 and Q6_K's 6.5625 bits per weight, but offers a denser set of
integer levels and a much simpler decoder.

## From blocks to a matrix row

A learned projection is a matrix. Multiplying one matrix row by an activation
vector is a dot product. If a Q8_0 row has 5,120 weights, it contains:

```text
5,120 / 32 = 160 blocks
160 × 34 = 5,440 stored bytes
```

A readable scalar row operation walks those 160 blocks in order, decodes each
32-weight segment, multiplies it by the matching 32 activation values, and adds
the products to an FP32 total. The block position determines which activation
segment it corresponds to; reordering blocks changes the row.

[`decode_q8_0`](../src/quant.cpp) and [`dot_q8_0`](../src/quant.cpp) implement the
single-block primitive. CPU-005 does not yet bind arbitrary GGUF tensor rows or
run whole projections. CPU-004 will combine admitted tensor offsets, declared
matrix orientation, Q4_K/Q6_K/Q8_0/F32 row handling, and the layer scheduler.

FP32 tensors need no quantized integer reconstruction: their four-byte values
are already stored as FP32. They still require exact byte order, dimensions,
orientation, and bounds checks when used as matrix rows.

## Fixtures and equality

The existing quant fixture now includes two Q8_0 cases. The signed-extremes case
contains -128, 127, negative and positive interior values, and zero under an
FP16 scale of 0.25. The zero case checks the neutral boundary. For each case the
fixture stores all decoded FP32 bytes and a deterministic FP32 dot product.

**External format authority:** llama.cpp revision
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` defines the 34-byte structure and
dequantization equation under the MIT license. The exact fields are added to
[`pins/quant_contract.json`](../pins/quant_contract.json).

**Measured:** the native decoder and dot product match every frozen Q8_0 FP32
bit. The focused adapter in
[`tools/llama_quant_oracle.c`](../tools/llama_quant_oracle.c) was independently
compiled with the pinned upstream `ggml-quants.c`; upstream and fixture decoded
bytes match for both Q8_0 cases. Wrong block sizes and unknown type names remain
fail-closed diagnostic cases.

## What this gate does not prove

CPU-005 completes the missing packed-block arithmetic dependency. It does not
prove matrix orientation, real tensor row traversal, a complete projection,
64-layer logits, CUDA performance, or model quality. Those are deliberately not
folded into a small format decoder: CPU-004 and later CUDA/oracle gates retain
their own evidence requirements.
