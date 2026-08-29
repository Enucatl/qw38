# 5. Weight conversion, quantization, and 32 GiB fit

[Previous](04-numerics.md) · [Index](README.md) · [Next](06-system-optimization.md)

## Why this matters

BF16 language weights alone are about 54 GB decimal. A 5090 build therefore
requires quantization, but quantizing recurrence-sensitive tensors before the
high-precision path passes makes errors hard to attribute.

## Inventory before conversion

Model conversion is a translation between two representations of the same
numbers. A **tensor inventory** says what every source array means; a **repack**
changes its physical order so a kernel can read it efficiently; **quantization**
replaces exact values with an approximation plus scales. Keeping these three
operations separate makes it possible to identify whether a mismatch came from
the model binding, layout, or precision.

Emit one manifest row per checkpoint tensor: canonical role, source name, shape,
dtype, elements, source bytes, destination format/block size, payload bytes,
alignment, checksum, and transpose/repack. Reject missing, duplicate, extra
mandatory, or shape-incompatible tensors. The manifest, not filename folklore,
is the input to `ModelSpec` validation.

For block format `(N values, B bytes)`, storage is
`ceil(elements/N)*B`, rounded for tensor alignment. Record device copies and
derived layouts separately. Conversion must round-trip a sample block through a
scalar dequantizer and compare each converted tensor's checksum.

For example, imagine a format storing 32 values in 18 bytes: 16 bytes of packed
codes plus a two-byte scale. A 1,000-element tensor needs `ceil(1000/32)=32`
blocks, or 576 bytes before alignment—not `1000*4/8=500` bytes. If tensors begin
on 256-byte boundaries, the next tensor may start later still. This is why the
converter computes exact payload offsets rather than multiplying a marketing
bit rate by the parameter count.

A **transpose** changes logical axes: `[out,in]` becomes `[in,out]`. A repack
may instead preserve the logical axes while arranging blocks, columns, or tiles
in the order a CUDA kernel expects. The manifest must distinguish them so the
reference loader can reconstruct the original logical tensor.

## Quantization policy

Bring up BF16/FP16 weights with FP32 sensitive state first. A quantized block
usually stores a small group of integer codes and one or more scale values. The
decoder reconstructs approximate floating-point values while doing the dot
product. Thus “4-bit” describes the codes, not necessarily four bits per tensor
element after scales, padding, and headers are included. Then calibrate on a
representative corpus and change one tensor class at a time. A reasonable
**Proposed** order is FFN gate/up, FFN down, mixer projections, embeddings/LM
head, and finally recurrence-sensitive GDN paths only if evidence permits.
Keep norms, biases, `A_log`, `dt_bias`, and recurrent state high precision at v1.
q27's published per-tensor recipes are useful external evidence that sensitivity
is non-uniform, not proof that its choices transfer to this converter.

**Calibration** runs representative text through the high-precision model and
records which weights or channels see important activations. The converter can
spend more precision where an error is amplified. GDN deserves special caution:
a small recurrent-state error is fed into the next token and can accumulate over
a long prompt, so a short local dot-product test does not establish long-context
quality.

## Reproducible memory ledger

| Allocation | Formula / source | 32K scenario |
|---|---|---:|
| quantized resident weights | converter manifest | measured at load |
| repacked/derived weights | runtime allocation log | measured at load |
| GDN recurrent matrices | `48*48*128*128*4` | 144 MiB/session |
| GDN convolution rings | `48*10240*4*4` | 7.5 MiB/session |
| full-attention BF16 KV | `65536*context` | 2 GiB/session |
| logits | `248320*4` | 0.95 MiB/session |
| activation/workspace | maximum live plan | measured |
| CUDA libraries + graphs | allocation delta | measured |
| fragmentation/reserve | explicit policy | never zero |

“Resident weights” are the tensors kept on the GPU for the engine lifetime.
“Repacked weights” are additional optimized copies; they count even if the
original remains mapped. “Workspace” is reusable temporary memory sized for the
largest live operation. CUDA libraries and graph instantiation may allocate
memory outside the engine's own allocator. **Reserve** is intentionally unused
headroom for transient allocations and fragmentation.

The 4-bit arithmetic lower bound is `27e9*0.5 = 13.5 GB` decimal (12.6 GiB),
not an artifact size or fit proof. The allocation log must show current, peak,
and reserved bytes by owner and demonstrate headroom after graph instantiation.
At native 256K, BF16 KV alone is 16 GiB, so a 32K fit does not prove native
context fit. Larger context requires a measured lower-precision KV policy or a
smaller weight/workspace plan and its own quality gate.

GB and GiB differ: storage vendors often use `1 GB = 10^9` bytes, while
`1 GiB = 2^30` bytes. The RTX 5090's advertised 32 GB memory is normally exposed
to software in binary units close to 32 GiB, less whatever the driver and other
processes reserve. Always budget from the runtime's reported free bytes.

## DwarfStar transfer boundary

Reuse exact-name binding, block-size arithmetic, calibration, and allocation
guards. Adapt the quant policy to dense Qwen tensors. Reject routed-expert
formats and SSD expert streaming: every dense FFN weight is used every token.

## Failures and exercise

Do not omit alignment/scales, count mmap file bytes as VRAM, overwrite the only
high-precision oracle, or claim quality from perplexity alone. Exercise: fill
the ledger from an actual artifact and context 32K. Expected: logged allocation
totals reproduce the sum within allocator accounting, the process completes a
fixture, and free reserve remains after graphs; otherwise the fit gate fails.
