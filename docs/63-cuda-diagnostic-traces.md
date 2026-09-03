# 63. CUDA diagnostic traces

[Index](README.md) · Implementation task: TRC-004 in
[`implementation_ledger.md`](../implementation_ledger.md)

This chapter explains how the CUDA scheduler exposes a small, stable set of
diagnostic tensors without putting tracing into the release execution path. A
trace is an inspection aid: it is compiled with `QW38_DIAGNOSTIC_TRACE`, uses
the backend-neutral sink from [Chapter 33](33-diagnostic-trace-isolation.md),
and is compared with the scalar and oracle evidence frozen in
[Chapter 38](38-scalar-authority-tolerances.md).

## The five visible boundaries

The scheduler offers exactly five filters for a one-token diagnostic run at
token 42. Three are complete residual vectors at physical layers 0, 3, and 63;
the final normalization and vocabulary logits are global boundaries. The
global layer is represented as `64` in the CUDA contract.

| Filter | Shape | Meaning |
|---|---:|---|
| `0 layer_residual` | `[5120]` | layer-0 output before the residual buffer swap |
| `3 layer_residual` | `[5120]` | layer-3 output before the residual buffer swap |
| `63 layer_residual` | `[5120]` | final decoder-layer output |
| `global final_norm` | `[5120]` | final FP32 RMS-normalized hidden vector |
| `global logits` | `[248320]` | complete FP32 vocabulary row before frontier advance |

These names and dimensions are not CUDA-specific guesses. They are linked to
the scalar contract in [`pins/scalar_trace_contract.json`](../pins/scalar_trace_contract.json).
The CUDA subset and its build/isolation rule are frozen in
[`pins/cuda_trace_contract.json`](../pins/cuda_trace_contract.json).

## Exact filters and atomic failure behavior

The traced entry point validates the filter and sink before launching GPU work.
Wildcards, unknown names, a layer tap at the global scope, a global tap at a
physical layer, and null sinks are rejected. A successful filter produces one
typed full-tensor view. The sink copies that view while the existing host
candidate buffer is available; a copy or sink failure returns an error before
candidate state is published or the frontier is incremented.

Tracing uses the retained unfused, non-graph scheduler path. This makes the
boundary easy to attribute and avoids changing graph captures or production
allocations. Ordinary `execute_token` still calls the same scheduler without a
trace filter or sink. The normal CUDA object therefore contains neither the
diagnostic entry point nor the stable tap literals.

## Measured token-42 evidence

The authenticated result is [`fixtures/cuda_trace.json`](../fixtures/cuda_trace.json),
captured in the pinned CUDA 13.0.2 / SM120 environment on an RTX 5090. Every
row below has 5,120 elements except logits, which has 248,320. The values are
full-tensor comparisons against the existing scalar/oracle admissions; they
are not spot checks.

| Filter | Max absolute | RMS | Cosine |
|---|---:|---:|---:|
| layer 0 residual | 0.00811195374 | 0.000325656829 | 0.999999276 |
| layer 3 residual | 0.037525177 | 0.000961786042 | 0.999998908 |
| layer 63 residual | 0.755096436 | 0.112812462 | 0.999936634 |
| final norm | 0.12295723 | 0.0220696393 | 0.999932363 |
| logits | 0.13822341 | 0.0272042993 | 0.999871596 |

All five comparisons passed the immutable `llama_vs_quartz` maximum-absolute,
RMS, and cosine gates. The greedy logit index was `3649` for both scalar and
CUDA. Invalid-filter rejection and an injected sink failure both passed, with
the failure-path frontier remaining zero.

## Build and verification

The diagnostic scheduler object is compiled separately from the normal object:

```sh
python -m json.tool pins/cuda_trace_contract.json >/dev/null
python -m json.tool fixtures/cuda_trace.json >/dev/null
uv run pytest -q tests/test_diagnostic_trace.py tests/test_cuda_trace.py tests/test_cuda_full_scheduler.py
```

The device gate uses the pinned image and model. It builds
`build/qw38-cuda-full-scheduler-test`, runs the native sink/filter capture, and
then runs the opt-in Python CUDA tests. The normal-object audit uses `nm -C`
and `strings` to check that trace symbols and literals are absent. Host systems
without `nvcc` should use the container; no host fallback is evidence for this
CUDA boundary.

## What this proves—and what it does not

Measured here: five stable, typed, full-tensor CUDA boundaries; exact filter
validation; sink/failure atomicity; scalar-pinned shapes and names; unchanged
frozen comparison gates; greedy equality; and diagnostic-only object isolation.

This does not expose private GDN, attention, or FFN scratch; it does not make
the BF16 embedding row satisfy the scalar zero-error contract; and it does not
trace prompt chunks. It also does not add a user-facing trace command or v1
bundle driver—those remain outside this increment and belong to `EVAL-001`.
