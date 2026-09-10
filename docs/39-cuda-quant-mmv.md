# 39. The first CUDA matrix-vector kernel

[Index](README.md) · Implementation tasks: CUD-001 and EDU-025 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter explains the first piece of Quartz model arithmetic that runs on
the RTX 5090. It is deliberately a small, inspectable boundary: one packed
weight matrix, one activation vector, and one FP32 output value per matrix row.
It is not yet a complete model layer or an optimized throughput claim.

## Matrix-vector multiplication

A matrix is a rectangular table. Each row in a neural-network weight matrix
describes one output feature. A vector is one list of current activation values.
Matrix-vector multiplication, abbreviated **MMV**, takes the dot product of each
matrix row with that vector:

```text
row 0 · activation -> output 0
row 1 · activation -> output 1
row 2 · activation -> output 2
```

Quartz stores the rows as Q4_K or Q6_K blocks rather than ordinary FP32 values.
[Chapter 17](17-quantization.md) explains every packed field and decoding
equation. [`cuda/quant_mmv.cu`](../cuda/quant_mmv.cu) reads those bytes directly;
it does not expand the whole matrix into a second, much larger FP32 allocation.

## Why the activation is quantized temporarily

Model activations are stored as **BF16**, a two-byte floating-point format. The
kernel first converts each group of 32 BF16 values into one temporary Q8 block:

```text
scale = largest absolute value / 127
q[i]  = round(BF16 value[i] / scale)
```

Each `q[i]` is a signed eight-bit integer. Multiplying a staged integer by its
scale approximately reconstructs the BF16 value. “Transient” means these Q8
blocks are scratch data for this operation, not persistent model or session
state. [`Q8Block`](../cuda/quant_mmv.h) deliberately uses an FP32 scale and 32
signed bytes, so its local layout is obvious. The exact rule is pinned in
[`pins/cuda_quant_contract.json`](../pins/cuda_quant_contract.json).

This staging loses a small amount of precision. Its benefit is that later tuned
kernels can use integer dot-product instructions while keeping packed weights
resident. CUD-001 establishes the semantics first; profiler-led instruction
selection and row buckets remain later optimization work.

## Threads, warps, and rows

A CUDA **thread** is one GPU worker. Thirty-two neighboring threads form a
**warp** and execute together. Quartz assigns one warp to one output row. Each
lane handles columns `lane`, `lane + 32`, `lane + 64`, and so on. The lanes then
combine their partial FP32 sums in a fixed five-step tree.

Eight warps form the 256-thread block. Bounds checks allow row counts that are
not multiples of eight: the `17`-row and `257`-row fixtures deliberately exercise
both a partly occupied final block and multiple blocks. Column counts must be a
multiple of the 256-value Q4_K/Q6_K block size and invalid pointers, zero sizes,
or incompatible columns fail before launch.

## Packed blockwise Q4_K/Q6_K loads

The first kernel decoded each column independently: lane `lane` walked
`lane`, `lane + 32`, `lane + 64`, and so on, and each visit re-read that
column's packed scales and nibbles from the same 256-value weight block.
Production decode Q4_K/Q6_K MMV now has a second path that still owns one
warp per row and still uses the same column order, but loads each block's
fields and scales **once**, then consumes the lane's eight values as
`lane + i*32` for `i = 0..7`. Products stay FP32 `__fmul_rn` /
`__fadd_rn`. The five-step warp tree is unchanged. Transient Q8 staging
(`quantize_bf16_q8`) and warp-count dispatch (`selected_mmv_warps`) stay.
Q8_0 MMV stays on the per-column decoder. Integer-dot (DP4A) reassociation
is out of scope.

The production pin is `kSelectedMmvLoadPath` in `{elementwise, packed}`.
**Keep:** A/B winner `packed` with byte-equal outputs versus the
per-column kernel and a strictly lower decode-occurrence-weighted
CUDA-event mean; production pin `packed`; live D2048 tok/s strictly
exceeds the frozen then-current accepted denominator; the cross-workload
guard held (D128/D2048 throughput and both p95 flavors within 5%; P
retain ≥95%). `reverted` is false. This increment does not own the 2K
llama.cpp parity gate. Quartz ≥ llama.cpp is informational. Live
exclusive A/B milliseconds and P / D128 / D2048 tok/s stay in the
report; this chapter does not replace them:
[`evidence/optimization/opt034-packed-mmv/REPORT.md`](../evidence/optimization/opt034-packed-mmv/REPORT.md).

## Integer-dot decode MMV admissibility study (OPT-042)

**Measured diagnostic, RTX 5090:** a study-only Q4_K `__dp4a` integer-dot
candidate on unchanged FP32-scale `Q8Block` staging was A/B'd against
production packed FP32 MMV on CUD-001 probe shapes, synthetic production FFN
gate/up (`17408×5120`) and down (`5120×17408`) shapes, and real layer-0/3/63
gate/up/down with prefix-2048 `post_prefix_hidden` activations. Complete
(stage+MMV) and prequant (MMV-only) views used 3 warm + 30 alternating pairs.
Synthetic weighted complete means: packed **0.0666723549** ms, integer
**0.0607566237** ms. Real weighted complete means: packed **0.0675881952** ms,
integer **0.061906416** ms. Integer occupancy warps 4/8/16 → 12/6/3 (all ≥
1); `staging_equal` true; candidate versus host-integer association is
byte-equal on every case. **Reject (`numeric_reject`):** synthetic production
shapes `q4k_gate_up` and `q4k_down` miss the frozen CUD-001 envelope (`3e-4`
max abs, `2e-4` RMS) for **both** packed and integer (e.g. gate/up packed max
abs **3.35693359e-4**, down packed max abs **2.31933594e-3**); probes and all
real gate/up/down pass. Production pin stays `packed`; `legal_mmv_load_path`
stays `{elementwise, packed}`. Integer-dot is not installed. This increment
**claims no performance improvement** and does not publish a successor oracle.
Accepted keep denominators remain unchanged. Live numbers stay in the report;
this chapter does not replace them:
[`evidence/optimization/opt042-mmv-integer-study/REPORT.md`](../evidence/optimization/opt042-mmv-integer-study/REPORT.md).

## What is compared

[`cuda/quant_mmv_test.cu`](../cuda/quant_mmv_test.cu) builds deterministic packed
matrices and BF16 activations. The host scalar decoder reconstructs every weight
using the admitted CPU-001 code. It then uses the same staged Q8 values to form a
readable sequential answer. The GPU result uses a parallel reduction, so exact
FP32 bytes are not required: addition in another order rounds differently.

The pre-optimization gates are:

- every transient Q8 scale and integer equals the host staging exactly;
- maximum absolute output error is at most `3e-4`;
- RMS output error is at most `2e-4`; and
- no CUDA call or result may fail.

Relative error is still printed, but is report-only. The first draft used it as
a gate and rejected a Q6 result whose absolute error was only `4.58e-5`, because
division by a small expected output amplified the ratio. That failed approach
is retained in the ledger. This is the same near-zero-denominator problem
explained in [Chapter 38](38-scalar-authority-tolerances.md).

## Measured evidence and boundary

**Measured local:** on the RTX 5090 with CUDA 13.0.2, four Q4_K/Q6_K cases at
`17x256` and `257x512` passed exact staging and both numeric gates. Each case ran
three warm-ups and 30 CUDA-event-timed samples. The raw summaries are frozen in
[`fixtures/cuda_quant_mmv.json`](../fixtures/cuda_quant_mmv.json). The observed
means around 0.006–0.008 ms are diagnostic launch measurements, not production
model throughput or a baseline comparison.

**External/adapted:** the packed weight layouts and the idea of transient Q8_1
activation staging come from the pinned MIT-licensed llama.cpp files recorded in
[`sources.md`](sources.md). Quartz keeps a focused local layout and readable
decoder instead of importing llama.cpp's generic dispatch system.

CUD-001 proves a real SM120 device can stage BF16 activations and multiply Q4_K
and Q6_K rows within frozen primitive limits. It does not yet prove prompt MMQ,
full production dimensions, resident model weights, diagnostic model taps,
GDN/attention layers, 128K memory fit, end-to-end quality, or speed. Those remain
separate ledger gates.
