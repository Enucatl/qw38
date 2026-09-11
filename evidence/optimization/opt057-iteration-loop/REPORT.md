# OPT-057 — Bounded optimization feedback and reliable CUDA builds

## Claim labels and proof limits

This increment delivers a **bounded iteration runner**, **CUDA depfiles and flag
stamps**, a **screen** tier, and a **short engine probe**. It makes **no throughput claim**.
Proof boundary:

- no throughput claim
- 300-second aggregate feedback deadline
- header and flag invalidation
- missing tier fails before GPU work
- over-budget subprocess teardown
- screen is not a historical P/D oracle
- historical evidence remains immutable

Short p95 figures from screen are diagnostic. Historical P/D oracles remain
release-only. Production pins are unchanged.

## Protocol

| Field | Frozen value |
|---|---|
| Runner | `tools/run_optimization_task.py` |
| Contract | `pins/opt057_iteration_contract.json` |
| Probe | `build/qw38-cuda-optimization-engine-probe` |
| Image | `qw38-cuda:13.0.2` with `-w /workspace` |
| Deadline | 300 s monotonic aggregate, including compile and teardown |
| Feedback | build once, then smoke → correctness → screen |
| Acceptance | one cold setup and two warm repetitions; native counts are one tiny case plus two tokens × two execution modes |
| Screen | at most one control/candidate pair: P4096 or prefix 2048 + 32 tokens |
| Release | historical `prefill_4k` / `decode` oracles only |

## Measured sitting

Device work used the already-built `qw38-cuda:13.0.2` image. First compile of
`build/qw38-cuda-optimization-engine-probe` took **48.298 s**. Whole feedback
(compile + smoke + correctness + screen) finished in **57.143 s**, inside the
300-second budget.

Acceptance (objects already present) reported three durations:

| Repetition | Duration (s) | Recompiled |
|---|---:|---|
| cold | 3.623 | no (cache hit; first compile was the prior feedback sitting) |
| warm1 | 3.532 | no |
| warm2 | 3.544 | no |

Native counts: one tiny 17×256 projection (1 warmup, 1 sample, no model) plus
two predetermined tokens × graph/eager. Screen used prefix 2048 + 32 tokens
once each for graph and eager; its tok/s values are diagnostic only.

Host pytest with fake clocks demonstrated header/flag invalidation, missing
tier failure before the probe, over-budget process-group teardown, screen vs
release separation, and unchanged OPT-056 historical bytes.
