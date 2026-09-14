# OPT-133 — Nsight Systems decode activity

Diagnostics only. `claims_throughput=false`. Shipping selector remains `decode_segments8`.

Measured at `2026-09-14T11:38:35Z`. Image `qw38-cuda:13.0.2`. nsys_available=`True`.

## Answers

1. **Proven GPU idle** (Nsight hardware inactivity inside the bounded 12-token window, compared to OPT-125 unobserved): mean explained-by-idle `unmeasured` ms, fraction of OPT-125 unobserved `unmeasured`. Pinned nsys 2022.4.2 ships CUPTI 12.0 and cannot emit `gputrace`/`cudaapisum` against CUDA 13.0.2, so hardware idle is unmeasured rather than proven 0 ms.
2. **Host/API submission** visible to Nsight but not CUDA-event leaves: mean explained-by-API `unmeasured` ms, fraction `unmeasured`. NVTX push/pop ranges (`qw38.ffn`, `qw38.graph_launch`, mixer/attention) are present in `.qdrep` traces; CUDA runtime API sum is unavailable for the same CUPTI mismatch.
3. **Unresolved** after both instruments: mean `126.4` ms, fraction `1`. This remainder is not relabeled GPU idle.
4. **nsys wrapper overhead** on the bounded window: mean `-0.3342` ms.

## Reconciliation

| Window | OPT-125 unobserved_ms | nsys gpu_idle_ms | nsys cuda_api_ms | unresolved_ms |
| --- | --- | --- | --- | --- |
| d128-early | 126.3 | unmeasured | unmeasured | 126.3 |
| d128-middle | 125.7 | unmeasured | unmeasured | 125.7 |
| d128-late | 126 | unmeasured | unmeasured | 126 |
| d2048-early | 126.8 | unmeasured | unmeasured | 126.8 |
| d2048-middle | 126.6 | unmeasured | unmeasured | 126.6 |
| d2048-late | 126.7 | unmeasured | unmeasured | 126.7 |

OPT-125 windows were 4 tokens; values are scaled to the 12-token nsys window. `classify_disjoint_intervals(..., profiler="nsight_systems")` runs only when `gpu_idle_ms` comes from a parsed gputrace, and proven inactive is overwritten with that idle — not the CUDA-event stub relabel.

## Overhead

| Window | baseline_ms | nsys_ms | nsys_overhead_ms | ratio |
| --- | --- | --- | --- | --- |
| d128-early | 195.4 | 195.8 | 0.3773 | 1.002 |
| d128-middle | 200.2 | 198.8 | -1.353 | 0.9932 |
| d128-late | 202.6 | 201.2 | -1.373 | 0.9932 |
| d2048-early | 226 | 225.7 | -0.2086 | 0.9991 |
| d2048-middle | 243.4 | 243.7 | 0.2851 | 1.001 |
| d2048-late | 244.2 | 244.4 | 0.2664 | 1.001 |

## Raw traces

- `d128-early`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-128-early.qdrep`
- `d128-middle`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-128-middle.qdrep`
- `d128-late`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-128-late.qdrep`
- `d2048-early`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-2048-early.qdrep`
- `d2048-middle`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-2048-middle.qdrep`
- `d2048-late`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-2048-late.qdrep`

GPU residents at preflight:

```json
[]
```
