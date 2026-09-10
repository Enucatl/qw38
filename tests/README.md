# CUDA test tiers

The shared quantization/scheduler CUDA pytest gates are opt-in with
`QW38_RUN_CUDA_TESTS=1`. Those two gates require an explicit tier so an
implementation loop cannot silently launch the expensive acceptance suite.
Other task-specific CUDA diagnostics retain their existing protocols.
The quantization and scheduler binaries also fail closed when run directly
without `QW38_CUDA_TEST_TIER`.

Select a tier with `QW38_CUDA_TEST_TIER`:

- `smoke`: one measured sample, no large MMA/Q8 admission sweeps.
- `correctness`: small MMA coverage with three measured samples; no large
  tuning or admission sweeps.
- `acceptance`: the historical three-warmup/30-sample protocol and all
  performance/admission checks. Use this before recording performance claims.

For example:

```sh
QW38_RUN_CUDA_TESTS=1 QW38_CUDA_TEST_TIER=smoke uv run pytest \
  tests/test_cuda_quant_mmv.py tests/test_cuda_scheduler_primitives.py
```

The two modules share one cached Docker build and one execution of each CUDA
binary per pytest process and tier, preventing the quantization binary from
being rerun by the scheduler test.
