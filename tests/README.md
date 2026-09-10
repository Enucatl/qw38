# CUDA test tiers

The canonical tier, complexity, sampling, reuse, and telemetry contract is
[`tasks/TASK-TESTING-STRATEGY.md`](../tasks/TASK-TESTING-STRATEGY.md), as
required by [`tasks/PERFORMANCE-RECOVERY-2026-09-10.md`](../tasks/PERFORMANCE-RECOVERY-2026-09-10.md).
It applies to shared and task-specific CUDA pytest/native diagnostics. GPU
tests are opt-in with `QW38_RUN_CUDA_TESTS=1` and must fail closed unless
`QW38_CUDA_TEST_TIER` is explicitly set.

Capture stdout as the run evidence when investigating a new bottleneck.

For example:

```sh
QW38_RUN_CUDA_TESTS=1 QW38_CUDA_TEST_TIER=smoke uv run pytest \
  tests/test_cuda_quant_mmv.py tests/test_cuda_scheduler_primitives.py
```

The two modules share one cached Docker build and one execution of each CUDA
binary per pytest process and tier, preventing the quantization binary from
being rerun by the scheduler test.
