# Pinned Q4_K reference

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
