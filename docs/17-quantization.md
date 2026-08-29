# 17. Quantization, packed weights, and scalar arithmetic

[Index](README.md) · Implementation tasks: CPU-001 and EDU-003 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter assumes no background in numeric formats. Its immediate purpose is
to make [`src/quant.cpp`](../src/quant.cpp),
[`pins/quant_contract.json`](../pins/quant_contract.json), and
[`fixtures/quant_authority.json`](../fixtures/quant_authority.json) readable.
It does not claim that the scalar code is fast. The scalar path is deliberately
simple so that later CUDA kernels have an implementation they can be checked
against.

## From real numbers to stored bits

A model weight is conceptually a real number such as `-0.375`. A computer must
store a finite bit pattern instead. Quartz uses several representations:

- **FP32** uses 32 bits, or four bytes, for one floating-point number. It has a
  sign, an exponent that sets the scale, and a fraction that carries significant
  digits.
- **FP16**, also called IEEE 754 binary16 or “half,” uses 16 bits, or two bytes.
  It has less range and precision than FP32.
- **Q4_K** and **Q6_K** store small integers plus shared scale information. They
  use far fewer bits per weight than FP32.

A **bit** is one binary digit, zero or one. Eight bits make a **byte**. Four bits
make a **nibble**, which can represent an unsigned integer from 0 through 15.
“Little-endian” means that the least-significant byte of a multi-byte number is
stored first. The admitted GGUF and Quartz fixtures use this byte order.

Quantization is the act of replacing many high-precision numbers with a compact
set of integers and enough scale information to approximately reconstruct them.
It is like recording a length as an integer number of millimetres rather than an
arbitrary real number of metres: the unit supplies the scale and the integer is
cheap to store. The reconstruction is called **dequantization** or **decoding**.

Quantization is usually lossy: the reconstructed value need not equal the
original training value. CPU-001 does not decide whether the model-wide quality
loss is acceptable. It establishes the narrower fact that Quartz interprets the
already admitted GGUF bytes exactly as the pinned format defines them. Model
quality remains a later QLT-001 gate.

## Why weights are stored in blocks

Storing one scale beside every small integer would consume much of the saved
space. Q4_K and Q6_K therefore group 256 weights into one **block**. Values in a
smaller sub-block share scale metadata. A block is both a compression unit and a
natural unit of work for a decoder or matrix kernel.

**External contract:** Quartz pins the block structures and decoding equations
from llama.cpp revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` under its
MIT license. The exact source files and field offsets are recorded in
[`pins/quant_contract.json`](../pins/quant_contract.json); the dependency is also
listed in [`sources.md`](sources.md). Quartz does not infer these layouts from a
format nickname.

## Q4_K: four-bit values with scales and minima

One Q4_K block represents 256 weights in 144 bytes:

| Byte range | Size | Meaning |
|---|---:|---|
| 0–1 | 2 | FP16 super-block scale `d` |
| 2–3 | 2 | FP16 super-block minimum scale `dmin` |
| 4–15 | 12 | eight 6-bit scales and eight 6-bit minima |
| 16–143 | 128 | 256 unsigned four-bit values |

Two four-bit values fit in each byte, so 128 bytes hold 256 values. The scale
section contains sixteen 6-bit numbers: `16 × 6 = 96` bits, exactly 12 bytes.
Because those 6-bit fields cross ordinary byte boundaries, decoding requires
masks and shifts. [`q4_scale_min`](../src/quant.cpp) makes that bit unpacking a
named operation instead of hiding it in an optimized kernel.

For one 32-value sub-block, reconstruction is:

```text
weight = d × subblock_scale × q − dmin × subblock_min
```

Suppose `d = 0.5`, `subblock_scale = 3`, `q = 7`, `dmin = 0.25`, and
`subblock_min = 2`. Then:

```text
weight = 0.5 × 3 × 7 − 0.25 × 2
       = 10.5 − 0.5
       = 10.0
```

The minimum term lets a sub-block represent a shifted range instead of forcing
all reconstructed values to be centred on zero. The average storage is
`144 × 8 / 256 = 4.5` bits per weight, including scale metadata.

## Q6_K: signed six-bit values with scales

One Q6_K block represents 256 weights in 210 bytes:

| Byte range | Size | Meaning |
|---|---:|---|
| 0–127 | 128 | low four bits of every quantized value |
| 128–191 | 64 | high two bits of every quantized value |
| 192–207 | 16 | signed 8-bit sub-block scales |
| 208–209 | 2 | FP16 super-block scale `d` |

A six-bit value needs more than one nibble. Q6_K keeps every value's low four
bits in one region and packs its high two bits in another. The decoder joins
them with bitwise OR, producing an unsigned value from 0 through 63, then
subtracts 32 to obtain a signed integer from -32 through 31.

Reconstruction is:

```text
signed_q = unsigned_six_bit_q − 32
weight   = d × subblock_scale × signed_q
```

If `d = 0.25`, the sub-block scale is `-4`, and the packed value is `37`, then
`signed_q = 5` and the decoded weight is `0.25 × -4 × 5 = -5`. Signed scales are
important: interpreting the byte for `-4` as the unsigned number `252` would
produce a completely different value. Quartz converts the byte explicitly in
[`decode_q6_k`](../src/quant.cpp), avoiding a platform-dependent narrowing cast.

The average storage is `210 × 8 / 256 = 6.5625` bits per weight, including
metadata. Q6_K uses more space than Q4_K but can represent more integer levels.

## What a dot product does

Neural-network matrix multiplication is built from **dot products**. Given two
equal-length lists, multiply corresponding entries and add the products:

```text
weights:     [ 2, -1, 0.5 ]
activation:  [ 3,  4,   2 ]
products:    [ 6, -4,   1 ]
dot product: 6 + -4 + 1 = 3
```

Here an **activation** is a temporary number produced while running the model,
as distinct from a stored weight. The scalar functions
[`dot_q4_k`](../src/quant.cpp) and [`dot_q6_k`](../src/quant.cpp) decode one
256-weight block, multiply it by 256 FP32 activations, and accumulate from index
0 to 255 in FP32.

Order matters in floating-point arithmetic. Its finite precision means that
`(a + b) + c` can round differently from `a + (b + c)`. A fused multiply-add can
also round once where separate multiplication and addition round twice. The
scalar oracle is compiled with `-ffp-contract=off`, freezing separate FP32
multiply and add operations. Optimized CUDA code may organize work differently,
but it must pass the frozen comparison gate rather than silently redefine the
reference.

## Fixtures, authorities, and equality

A **fixture** is checked-in test input with an expected output. The generator
[`tools/generate_quant_fixtures.py`](../tools/generate_quant_fixtures.py) builds
four deterministic blocks: packed Q4 boundary patterns, all-zero Q4, signed Q6
extremes, and zero-scale Q6. It uses a small, direct implementation of the pinned
equations and records:

- every input byte;
- all 256 decoded FP32 values as little-endian bytes;
- one deterministic 256-element activation rule; and
- the final FP32 dot-product bytes.

The generator is readable evidence, not an additional production decoder. Its
revision attribution and generated values are frozen in
[`fixtures/quant_authority.json`](../fixtures/quant_authority.json).

**Measured:** [`tests/test_quant.py`](../tests/test_quant.py) invokes the native
diagnostic and requires every decoded FP32 byte and every dot-product byte to be
identical to the fixture. This is **fixture equality**: for these chosen inputs,
actual serialized output equals expected serialized output with no tolerance.
It proves the exercised block interpretation and arithmetic order. It does not
prove that every possible block is correct, that CUDA is correct, or that the
quantized model has acceptable language quality.

## Why numeric reports also use metrics

Later optimized paths may legitimately differ in their lowest bits because
parallel addition changes rounding order. A useful report therefore contains
several views of the error:

- **absolute error** is `|actual − expected|`;
- **relative error** divides absolute error by the expected magnitude, with an
  explicit rule near zero;
- **RMS error** squares all errors, averages them, then takes the square root;
- **cosine similarity** compares the direction of two whole vectors;
- **NaN/Inf counts** detect invalid or unbounded results;
- **first failing index** locates the earliest value outside its tolerance; and
- **top-logit differences** compare the output scores that can change token
  selection.

For CPU-001, exact fixture equality gives zero absolute and RMS error and no
NaN/Inf values. Cosine similarity is reported with a `1e-15` absolute tolerance:
even a vector compared with itself can produce a value infinitesimally below
one because the square roots used to compute the metric also round. This test
harness failure and its diagnosis are preserved in the CPU-001 ledger log. The
tolerance applies only to the derived cosine metric; decoded and dot-product
bytes remain exact.

TRC-001 will implement the reusable typed metric reporter before scalar/CUDA
tensor comparisons. CPU-001 deliberately does not pretend that its focused
pytest calculation is that future trace system.

## Failure modes and verification boundary

The native entry points reject null pointers, a block of the wrong byte length,
and an output or activation whose count is not exactly 256. The diagnostic also
rejects unknown quantization names and malformed hexadecimal input. These checks
prevent a short buffer from being interpreted as a complete packed block.

Common implementation errors include reversing nibble order, losing the high
two Q6 bits, treating a signed scale as unsigned, unpacking the Q4 6-bit fields
in the wrong order, reading FP16 in the wrong byte order, and changing reduction
order without measuring the numeric effect. The patterned fixtures are designed
to make these errors visible; the zero fixtures cover the neutral boundary.

CPU-001 admits only the readable host primitive. It does not complete model
layers, CUDA kernels, model-quality evaluation, or performance claims. Those
remain separately named tasks in the implementation ledger.
