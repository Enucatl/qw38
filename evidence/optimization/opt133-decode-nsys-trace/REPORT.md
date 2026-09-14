# OPT-133 — Nsight Systems decode activity

Diagnostics only. `claims_throughput=false`. Shipping selector remains `decode_segments8`.

Measured at `2026-09-14T12:01:33Z`. Image `qw38-cuda:13.0.2`. nsys_available=`True`. nsys=`/usr/local/cuda/bin/nsys` version=`NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0`.

## Answers

1. **Proven GPU idle** (Nsight hardware inactivity inside the bounded 12-token window, compared to OPT-125 unobserved): mean explained-by-idle `126.4` ms, fraction of OPT-125 unobserved `1`. Hardware GPU trace parsed from `gputrace`.
2. **Host/API submission** visible to Nsight but not CUDA-event leaves: mean explained-by-API `0` ms, fraction `0`. NVTX push/pop ranges (`qw38.ffn`, `qw38.graph_launch`, mixer/attention) are retained in the trace when present.
3. **Unresolved** after both instruments: mean `0` ms, fraction `0`. This remainder is not relabeled GPU idle.
4. **nsys wrapper overhead** on the bounded window: mean `0.1826` ms.

## Reconciliation

| Window | OPT-125 unobserved_ms | nsys gpu_idle_ms | nsys cuda_api_ms | unresolved_ms |
| --- | --- | --- | --- | --- |
| d128-early | 126.3 | 188.5 | 196.1 | 0 |
| d128-middle | 125.7 | 191.7 | 199.3 | 0 |
| d128-late | 126 | 193.8 | 201.5 | 0 |
| d2048-early | 126.8 | 218.8 | 226.4 | 0 |
| d2048-middle | 126.6 | 235.9 | 243.7 | 0 |
| d2048-late | 126.7 | 236.7 | 244.5 | 0 |

OPT-125 windows were 4 tokens; values are scaled to the 12-token nsys window. `classify_disjoint_intervals(..., profiler="nsight_systems")` runs only when `gpu_idle_ms` comes from a parsed gputrace, and proven inactive is overwritten with that idle — not the CUDA-event stub relabel.

## Overhead

| Window | baseline_ms | nsys_ms | nsys_overhead_ms | ratio |
| --- | --- | --- | --- | --- |
| d128-early | 195.7 | 197 | 1.262 | 1.006 |
| d128-middle | 201.1 | 200 | -1.077 | 0.9946 |
| d128-late | 203.5 | 202.2 | -1.33 | 0.9935 |
| d2048-early | 226.5 | 227.5 | 0.9196 | 1.004 |
| d2048-middle | 243.7 | 244.4 | 0.6087 | 1.002 |
| d2048-late | 244.5 | 245.2 | 0.7119 | 1.003 |

## Raw traces

- `d128-early`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-128-early.nsys-rep`
- `d128-middle`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-128-middle.nsys-rep`
- `d128-late`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-128-late.nsys-rep`
- `d2048-early`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-2048-early.nsys-rep`
- `d2048-middle`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-2048-middle.nsys-rep`
- `d2048-late`: `evidence/optimization/opt133-decode-nsys-trace/traces/trace-2048-late.nsys-rep`

GPU residents at preflight:

```json
[]
```
