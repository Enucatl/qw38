# OPT-050 — Prepare prompt Q once outside attention

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of upstream prompt query RMS/RoPE/dual-F16
preparation against in-kernel per-CTA repetition. Complete cost includes
prepare + attention + combine. Like-arithmetic movement must match
in-kernel Q/output; FMA uses OPT-044 budgets. Keep requires lower complete
attention time, improved P versus the OPT-049 keep, 95% D128/D2048 floors,
and 105% p95 ceilings versus OPT-045. Does not substitute for the 2K
llama.cpp parity gate. Nsight is not used.

Proof limit: OPT-044 production-numerics budgets; like-arithmetic hoisted Q/output match in_kernel; prepare+attention+combine complete cost; tail/prefix/nonzero-start correctness; committed isolation and graph/eager equality; no extra persistent allocation; 95% throughput floors versus OPT-049 keep; 105% p95 ceilings versus OPT-045; does not substitute for the 2K llama.cpp parity gate

## Decision

**keep**. Production pin `hoisted`.
A/B winner `hoisted` (29.3651066 ms vs in_kernel 34.5931778 ms).
`hoisted_fma` was eligible (29.3650455 ms, max_abs 1.71e-7) but within 1% of
like-arithmetic hoisted, so arithmetic-preserving movement was installed.
Keep sitting ran. Quartz P 2221.82642, D128 37.4580574, D2048 35.7343941
tok/s versus OPT-049 keep P 2130.79614, D128 37.2543182, D2048 35.4072151.

## Quality

Hoisted prepared Q and attention outputs are byte-equal to in-kernel
(max_abs 0) on the 4096-token probe, 60 token/start cases, and real
layer 3/31/63 query-norm captures. Graph replay equals eager. Committed
KV stays isolated. Decode `warp_query` is unchanged.

## Throughput

Baseline is the OPT-049 keep sitting. Quartz P +91.03 tok/s (1.043×),
D128 +0.20 tok/s (1.005×), D2048 +0.33 tok/s (1.009×). D2048 p95
28.13 ms and D128 p95 26.90 ms stay inside 105% of OPT-045.
Prepared Q aliases `prompt_projection_a_` (100,663,296 bytes at 4096
tokens); no extra persistent allocation.
