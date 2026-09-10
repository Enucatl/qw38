# OPT-051 — Pipeline prompt attention with register softmax

## Claim labels and proof limits

Live exclusive-RTX-5090 staged A/B of F16 operands, register softmax,
cp.async KV staging, nbatch 64, and six-head GQA against the OPT-050
hoisted dual-F16 stream-K baseline. Complete cost includes prepare +
attention + combine. Dual-F16 like-arithmetic must match off; F16 uses
OPT-044 budgets. Keep requires lower complete attention time, improved P
versus the OPT-050 keep, 95% D128/D2048 floors, and 105% p95 ceilings
versus OPT-045. Does not substitute for the 2K llama.cpp parity gate.
Nsight is not used. Async overlap is evidenced by compiled cp.async
instructions plus occupancy, not source presence alone.

Proof limit: OPT-044 production-numerics budgets; like-arithmetic off control remains OPT-050 hoisted dual-F16; prepare+attention+combine complete cost; staged F16 / register-softmax / async / nbatch / GQA admission; tail/prefix/nonzero-start correctness; all-masked and empty KV partitions finite; committed isolation and graph/eager equality; cp.async instruction evidence for async candidates; 95% throughput floors versus OPT-050 keep; 105% p95 ceilings versus OPT-045; does not substitute for the 2K llama.cpp parity gate

## Decision

**keep**. Production pin `f16_async`.
A/B winner `f16_async`. Keep sitting ran. Quartz P 2489.33008,
D128 37.6655884, D2048 35.7582932 tok/s versus OPT-050 keep P
2221.82642, D128 37.4580574, D2048 35.7343941.

Staged admission: F16 vs dual (f16 29.09 ms, dual_reg 28.53 ms vs off
29.40 ms), then register-softmax (`f16_reg` 25.07 ms), then async
(`f16_async` 16.03 ms). `dual_async` occupancy 0, omitted. nbatch64
24.91 ms and gqa6 19.82 ms are slower than f16_async. SM120 SASS is
`LDGSTS.E.BYPASS.128`. Production fused commit uses stream sync instead
of recording a disable-timing event on the pipeline compute stream
(`cuEventDestroy` SIGSEGV after those kernels).

## Quality

Like-arithmetic dual paths match OPT-050 off within byte/zero envelopes.
F16 paths use OPT-044 budgets (max_abs 5.46e-5). Graph replay equals
eager. Committed KV stays isolated. Decode `warp_query` is unchanged.
60 token/start cases and layers 3/31/63 stay inside OPT-044.

## Throughput

Baseline is the OPT-050 keep sitting. Quartz P +267.50 tok/s (1.120×),
D128 +0.21 tok/s (1.006×), D2048 +0.02 tok/s (1.001×). D128 p95
26.82 ms and D2048 p95 28.10 ms stay inside 105% of OPT-045.
On reject, speedup is 0 and production stays off.
