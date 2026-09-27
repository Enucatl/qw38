# Pinned Q4_K reference

`cuda/attention.cu` uses the eight-column KQ/PV MMA fragment orientation,
separate asynchronous K/V staging, grouped-query reuse, bounded split
scheduling and cooperative merge patterns from `fattn-mma-f16.cuh`,
`fattn-common.cuh` and `fattn.cu` at the same pinned revision below.
The implementation uses installed CuTe BF16/FP32 MMA and asynchronous-copy
primitives, FP32 softmax/statistics and two BF16 probability components;
it does not adopt upstream FP16 accumulation. It specializes these patterns
to D256, Q32/K64 prefill and six-head grouped decode with a fixed split rule.

`cuda/q4k_q8.cu` adapts the integer MMQ/MMVQ structure from the same full
revision `e6ab7c1a41054a888ada952eab4c886444c2f5ad`: `mmq-config-blackwell.cuh`
falls through to `mmq-config-ampere.cuh` for Q4_K. We select I128/J32/K256,
256 threads, stage nibbles as INT8, use the `mma.cuh` m16n8k32 instruction,
and apply each K32 affine correction as in `mmq-vec-dot.cuh` and `vecdotq.cuh`.
The existing QW38 physical nibble order is retained. Activation scales,
weight scale products and corrections remain FP32, and activation sums are
exact INT32, as required by policy 1030. DP4A and MMA accumulators reset every
K32 group (absolute bound 60960); no scale factor enters an integer dot.

Source: [llama.cpp commit e6ab7c1a4](https://github.com/ggml-org/llama.cpp/tree/e6ab7c1a4).
Copyright (c) 2023-2026 The ggml authors; MIT license in `LICENSE`.

`reference.hpp` extracts the unmodified function bodies of `nearest_int`,
`make_qkx2_quants`, `get_scale_min_k4`, `quantize_row_q4_K_ref`, and
`dequantize_row_q4_K` from `ggml/src/ggml-quants.c`, and the portable half
conversion helpers from `ggml/src/ggml-impl.h`. Standalone includes, namespace,
block declarations and macros replace unrelated ggml build dependencies.
This header is a test-only oracle, independent of the production logical-code
adapter in `src/compiler/quantization/q4k.cpp`. The production fitting helpers
are a separately compiled port of these same pinned helpers.

Both implementations must be compiled with `-ffp-contract=off` and without
fast-math. The reference uses its built-in magnitude-derived fitting weights;
it receives no external importance matrix. This does not establish the
calibration provenance of the external comparator artifact.

Pinned upstream source SHA-256 (downloaded and verified before extraction):

- `ggml/src/ggml-quants.c`: `7878680cc60493f98469a116e9b14af8b84789292ccf891c249230b2fa3d157f`
- `ggml/src/ggml-impl.h`: `43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3`

The compiler rejects nonfinite inputs and values outside the reference fitter's
finite arithmetic domain. In particular, extremely small finite BF16 values
can overflow the fitting or scale-compression reciprocals before FP16 rounding;
those return `Unrepresentable` instead of invoking the reference's integer
rounding helper with NaN/Inf. Valid finite fitting results that round to zero
or to FP16 subnormal scales preserve the pinned representation exactly.
