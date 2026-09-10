# OPT-049 — Share decode FFN staging and fuse gate/up

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of decode FFN shared Q8 staging and a
two-pointer paired Q4_K gate/up kernel with the admitted SwiGLU
(`gate / (1 + expf(-gate)) * up`) and a single BF16 rounding before the
down projection. Separate-leg packed Q4_K remains the control. Down,
residual, and RMSNorm stay on already accepted kernels. Complete cost
includes norm, stage, gate/up/SwiGLU, down, and residual-add-norm.
Keep requires **OPT-044 production-numerics budgets**, **graph/eager
equality**, **lower complete FFN time**, **95% P/D128 floors versus the
OPT-048 keep**, and a **strict D2048 improvement** versus OPT-048
**34.3371162 tok/s**. Decode p95 stays inside 105% of OPT-045. Does
not substitute for the 2K llama.cpp parity gate. Nsight is not used.

Proof limit: OPT-044 production-numerics budgets; separate-leg packed Q4_K control retained; shared staging of normalized FFN input; two-pointer paired Q4_K kernel; admitted SwiGLU and single BF16 rounding; down/residual/norm stay on accepted kernels; graph/eager equality within each path; complete FFN cost includes stage/SwiGLU/down/norm; 95% throughput floors versus OPT-048 keep; 105% p95 ceilings versus OPT-045; does not substitute for the 2K llama.cpp parity gate

## Decision

**keep**. Production pin `paired_staged`.
A/B winner `paired_staged` (0.239974757 ms vs separate 0.26093938 ms).
Keep sitting ran. Quartz P 2130.79614, D128 37.2543182, D2048
35.4072151 tok/s versus OPT-048 keep P 2129.85938, D128 35.9651642,
D2048 34.3371162. OPT-046 packed Q4, OPT-047 DP4A Q8, and OPT-048
integer Q6 remain.

## Quality

Shared-stage and paired-staged match the separate-leg control on real
layer activations (max_abs 0). Paired-only BF16 activation loads differ
from Q8 staging (~0.02–0.03 max_abs) and were not installed. Graph
replay equals eager within each path.

## Throughput

Baseline is the OPT-048 keep sitting. Quartz P +0.94 tok/s (1.0004×),
D128 +1.29 tok/s (1.036×), D2048 +1.07 tok/s (1.031×). D2048 p95
28.37 ms and D128 p95 27.08 ms stay inside 105% of OPT-045.
