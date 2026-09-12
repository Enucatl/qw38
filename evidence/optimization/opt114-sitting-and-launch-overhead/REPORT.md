# OPT-114 — Fresh sitting calibration and decode launch overhead

Status: **measured**. A/A verdict
`repeatable`. Graph opportunity
`graph_ab_retain_ffn_only`.

fresh same-binary Quartz A/A is the control; OPT-098 arrays are historical calibration only; 3 warmups plus 10 interleaved AB/BA pairs at P4096, D128, and D2048; host launch/submit gaps are leaf CUDA-event gaps, never other_idle; other_idle is a legacy diagnostic and is not the launch budget; decode_segments8 is tested only when removable overhead is at least 0.50 ms/token at both D128 and D2048; no production selector change; no new profiling framework; do not raise the 1.5 GiB reserve cap without an allocation-backed reason; measurement_unstable stops post-106 keep/reject claims until a stable sitting exists.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- source: `7409d8640520d139cb2b59fd6b09b5bb84fbcf14` (dirty)
- nvccflags: `-O2 --fmad=false`
- execution_graphs: `ffn_only`
- hardware_executed: True
- gpu_blocker: None

Selectors frozen from the current binary (OPT-106 selected / post-098 combination).
This identity is the control for every A/A and graph run in this task.

## A/A same-binary Quartz control

3 warmups + 10 interleaved AB/BA pairs. OPT-098 arrays are **not** used for
this interval or for keep/reject.

| Workload | A tok/s | B tok/s | geo ratio | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| p4096 | 2981.0894 | 2980.2403 | 0.9997 | 0.9989 .. 1.0005 | repeatable |
| d128 | 53.4602 | 53.4634 | 1.0001 | 0.9994 .. 1.0007 | repeatable |
| d2048 | 48.3830 | 47.9853 | 0.9915 | 0.9588 .. 1.0253 | repeatable |

Combined A/A verdict: **repeatable**.
A `measurement_unstable` sitting blocks later keep/reject claims.

Historical OPT-098 vs OPT-106 (calibration evidence, not an optimization):
P4096 3046.23218 vs 3036.84448;
D128 37.2903061 vs 53.5021248;
D2048 35.4885025 vs 49.400589.

## Launch / submit overhead

Method: leaf CUDA-event gaps between kernels (host launch delay appears as GPU
idle on a serial stream), plus named host_submission_waits when present.
`other_idle` is reported only as a legacy diagnostic.

| Prefix | removable ms/token | GPU idle ms | kernel ms | legacy other_idle ms |
|---|---:|---:|---:|---:|
| D128 | 10.9928 | 10.9928 | 8.1174 | 0.3208 |
| D2048 | 10.9674 | 10.9674 | 9.7486 | 0.3288 |

Threshold 0.5 ms/token at both prefixes.

## Graph opportunity

verdict=`graph_ab_retain_ffn_only`; eligible=True;
reason=unhidden_idle_at_both_prefixes. Production selector remains `ffn_only`.
production_kept=True.
graph_ab=graph_ab_retain_ffn_only;
capture_error_present=True.
The existing `decode_segments8` path was attempted because removable launch
overhead exceeded 0.5 ms/token at both prefixes. Shipping stays
`ffn_only`; this task does not change the production selector.

## 128K memory reconcile

OPT-106 live sitting recorded explicit_bytes=29571258208 versus inventory 29571252064 (delta 6144 bytes). arithmetic=false so memory_fit.ok is false while reserve_ok remains true: free_bytes 3521118208 still exceed the 1.5 GiB reserve (1610612736). The discrepancy is allocator/graph-object accounting, not a capacity shortage. Do not raise the reserve cap.

inventory explicit=29571252064;
OPT-106 explicit=29571258208;
delta=6144;
reserve_ok=True; reserve_cap_increased=false.
