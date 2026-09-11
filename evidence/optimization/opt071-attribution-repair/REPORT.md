# OPT-071 — Repair full-engine attribution and production replay

This increment **claims no performance improvement** and makes **no tok/s
speedup claim**. It repairs OPT-060 timing windows and OPT-061 capture replay
so later ranking can use honest family gaps. There is **no throughput** gate.

## Proof limits

- Eager diagnostic CUDA events are **not** production CUDA-graph timings.
- Prefix, warmup, setup, and graph-creation time are excluded from decode
  latency. Dropped event-pool records fail acceptance.
- Fused members are counted; enclosing FFN time is not added to its leaves.
- Concurrent intervals are summed as work; union uses one sequence epoch.
- Unexplained measured-window share above 5% or invalid call counts stop
  gap attribution rather than inventing overlap.
- Nsight is optional. Event conservation and capture identity are required.
