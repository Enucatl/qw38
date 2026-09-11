# Kernel-parity and model-quality reset after OPT-080

Status: proposed implementation batch OPT-081 through OPT-088. Source review
at Quartz HEAD after OPT-080, 2026-09-11. This document is task design, not
kernel implementation, a new GPU sitting, a production keep, or a tok/s
claim. Authority remains llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`, target
hardware RTX 5090. Authenticate the artifact rather than infer identity from
its historical `Qwen3.8-27B-Q4_K_M.gguf` filename.

Read [the shared testing strategy](TASK-TESTING-STRATEGY.md),
[the preceding batch protocol](PERFORMANCE-RECOVERY-POST-069.md), and
[the post-056 protocol](PERFORMANCE-RECOVERY-2026-09-11.md).
This batch does not lower OPT-056 or OPT-016. OPT-080 remains the historical
final gate for the previous validation policy; OPT-088 is the successor gate
only for the new production combination.

## Why this batch exists

OPT-080 quality-blocked the previous combination before long P/D/2K release.
The active kernel-admission bottleneck was not a missing faster kernel. It was
OPT-059 v2 / OPT-074 production-family admission:

`held_out_exceeds_frozen_calibration_ceiling`

That rule compared llama GPU projections to independent FP64 (and then Quartz
to a llama-vs-FP64 envelope) at production K. Q4 gate/up, Q4 down, and both
Q8 mixer families stayed `unadmitted`. OPT-070 therefore labeled installed
Q8 `r2_w2` and MMQ `fma_async_x` inconclusive. OPT-075/076 retained packed Q4
despite measured complete-FFN savings (`integer_q8_paired` 16.43→11.77 ms;
`late_w4` 16.42→9.82 ms) because OPT-074 coverage was missing.

That is the wrong question for kernel admission. This batch adopts the `../ds4`
separation fully and adapts it to Quartz/Qwen3.8:

1. Kernel parity asks whether a CUDA kernel implements the intended quantized
   operation versus an independent CPU/dequant reference at the **same
   quantization**.
2. Model quality asks whether accepted numerical approximations preserve
   model behavior. Pinned llama.cpp remains the official inspectable local
   scorer/control. This is not a same-model comparison with ds4.
3. Performance independently decides whether an approved candidate is worth
   keeping.

These contracts stay separate in reports, fixtures, and ledger verdicts. A
legal result is:

`kernel_parity_pass=true`, `model_quality_pass=false`,
`performance_pass=true`, `production_kept=false`

or the corresponding all-true keep. Do not relabel one category as another.

Do **not** extend OPT-059/074 with another production-error envelope. Do **not**
replace it with `Quartz vs llama GPU <= new tolerance`. Preserve every
OPT-059/074 and OPT-070–080 report as historical evidence. Do not rewrite
historical verdicts or reinterpret old fixtures in place.

## Active validation hierarchy

After this batch, admission is:

1. Structural correctness (hard invariants; fail closed).
2. Quantized-kernel parity versus independent CPU/dequant reference.
3. Exact equivalence for same-math paths.
4. Full-model quality (regression from the shipping Quartz baseline, with
   pinned llama as inspectable external control).
5. Performance (component/E2E methodology unchanged).
6. Release outcome (OPT-088; historical OPT-056/016 conditions stay visible).

Required policy statements:

> Kernel admission proves implementation of the intended quantized operation.
> It does not prove full-model quality and does not depend on llama GPU versus
> FP64 consistency.

> A kernel does not need to reproduce llama.cpp GPU's layer-specific
> floating-point projection error. It must implement the intended quantized
> operation within the approved kernel-parity envelope and must not regress
> full-model quality.

> Passing kernel parity does not establish model quality. Passing model
> quality does not establish performance. Passing performance does not excuse
> kernel or quality failure.

`opt074_coverage_unadmitted` / `held_out_exceeds_frozen_calibration_ceiling`
must not be a prerequisite for future optimization keeps. OPT-074 remains a
historical production-numerics experiment.

## Kernel parity contract (OPT-081, consumed by OPT-082)

Canonical versioned contract: `pins/kernel_parity_v1_contract.json`.
Canonical checkers: `cuda/kernel_parity.cuh` and `tools/kernel_parity.py`.
Do not keep a second tolerance implementation for new work. Existing prompt
MMQ already uses ds4 Q4_K/Q6_K association in
[`pins/cuda_mmq_contract.json`](../pins/cuda_mmq_contract.json); fold that
rule into the canonical helper rather than inventing a tighter one.

### Reference operation

GPU candidate versus independent CPU decode of the **same quantized weights**
and the **same staged/input activations**, evaluated in high-precision host
arithmetic. The reference is not original unquantized FP64 model weights, not
llama GPU, and not a shallow-layer production budget.

### `quantized_operation_association`

An element fails only when **both** hold:

`abs_error > abs_tolerance` AND `relative_error > relative_tolerance`

Passing either test is sufficient. `abs_tolerance = abs_scale * sqrt(K)` with
`K` = weight columns.

Baseline ds4-style family envelope, encoded as Quartz `kernel_parity_v1`:

| Family | Applies in Quartz? | abs_scale | rel_tol |
|---|---|---:|---:|
| Q8_0 weight GEMM / MMV / MMQ | yes (mixer Q8_0) | 0.05 | 0.05 |
| Q4_K MMV / MMVQ / MMQ | yes (FFN) | 0.20 | 0.05 |
| Q6_K MMV / MMQ | yes (vocabulary and any Q6_K MMQ) | 0.20 | 0.05 |
| Q2_K / IQ2-style | no (not in the pinned Q4_K_M GGUF) | n/a | n/a |

Q6_K uses the Q4_K K-quant envelope already frozen for prompt MMQ. Do not
silently tighten any of these toward CUD-001 `3e-4`. Do not derive them from
llama GPU or FP64 production-layer measurements.

Diagnostics (max abs, max rel, failing count, RMS, optional cosine) are
recorded. Admission semantics are the explicit parity rule plus zero
non-finites.

### `same_math_equivalence`

Use this class when two paths are supposed to perform the same staged
arithmetic: fused versus unfused reconstruction of the same staged
representation; paired gate/up versus the same kernels separately; two
consumers of the same staged Q8 buffer; packing/unpacking of exact integers;
selector or graph/eager variants that must not change arithmetic.

Require bit identity where reasonable; otherwise an explicitly tiny
deterministic tolerance justified by the exact arithmetic difference. The
broad Q4 `0.20*sqrt(K) OR 5%` envelope must not hide a same-path defect.
Reports must name which contract applies.

### Hard invariants (never waived by a tolerance)

NaN/Inf; OOB access; wrong shape/K/M/N/dtype/layout/signedness; stale
staging; incorrect buffer lifetime or aliasing; incorrect Q8_1 field
interpretation; integer overflow; wrong exact-sum semantics; wrong scale/min
decode; graph using the wrong selector; eager/captured mismatch; fallback
path measured instead of the requested kernel; causal-state corruption;
reset/replay mismatch; checkpoint/cancellation failure; incomplete
vocabulary/logits; model/weight/input identity mismatch.

Continue distinguishing Quartz Q8 staging that stores integer sums,
llama-style Q8 staging, FP32 staging, and exact/recomputed integer sums. A
tolerance may never turn one typed staging representation into another.

## Quality contract (OPT-083, frozen by OPT-084)

Port the **testing strategy** of `../ds4/gguf-tools/quality-testing/`, not
ds4's DeepSeek/GLM datasets or model architecture. Official local
scorer/control: pinned llama.cpp. Quartz and llama must share tokenizer,
GGUF identity, prompts, continuations, target tokens, context construction,
and masking/scoring convention.

Inspectable evidence at minimum: per-example NLL, aggregate NLL, token
counts, per-token or sufficiently detailed log probabilities, continuation
text/token IDs, llama score, Quartz score, delta, failures/nonfinites,
model/runner identities. Where feasible, logits/top candidates at selected
divergence points.

`--quality` must disable performance shortcuts that would make the run fail
to represent the production arithmetic configuration under test. Document
the exact delta. Record effective selectors and dispatch. Do not silently
exercise a different kernel/layout/precision than the candidate intended for
production.

Optional remote OpenRouter scorer for `qwen/qwen3.8-27b`: interface only,
default off, no network from ordinary tests, credentials only via
environment, absence of credentials must not fail local tests, outputs
labeled external and non-authoritative. Do not invoke it in this batch.

Quality acceptance is frozen in OPT-084 **before** OPT-085/086/087 keep
decisions. Candidate quality is primarily **regression from the shipping
Quartz baseline**. Absolute task/model quality and Quartz-versus-llama
divergence stay separately visible. A known baseline defect does not
automatically reject a kernel that does not worsen it. Kernel parity pass
plus a meaningful quality regression must not ship.

Do not invent a single logit/projection threshold to replace OPT-074.

Retain or adapt: teacher-forced NLL; PPL ratio; recurrence incremental NLL;
deterministic continuation tests; state/continuation consistency; OPT-073
task-quality diagnostics where still useful.

ds4 datasets that cannot apply directly (DeepSeek V4 Flash/PRO, GLM 5.2
official continuations) are documented as not ported. This is a methodology
port, not a same-model comparison with ds4.

## Performance semantics

Unchanged. Kernel parity is not a speed claim. Quality pass is not a speed
claim. Only measured performance determines whether a valid candidate is
retained. Reuse repaired OPT-071 replay, AB/BA pairing, fixed sample counts,
confidence intervals, no sampling until significance, graph capture per
configuration, actual selector/dispatch proof, identical copy/commit/state
work. Existing performance evidence may rank candidates when identity remains
valid; a final production keep requires the owning task's specified fresh
confirmation.

## Shipping freeze at the start of this batch

Do not install new arithmetic candidates until OPT-084 freezes the quality
baseline on this combination:

- Q4 decode: `packed` / `paired_staged`
- Q8 decode: `r2_w2` (installed; OPT-070 inconclusive under the old policy)
- Prompt MMQ: `fma_async_x` tile `i128_j128` (installed; same)
- Prompt attention: OPT-079 `kv_once`
- Decode GDN: sequential
- Decode attention: `warp_query`
- Prompt-pair: `off`
- NVCCFLAGS: `-O2 --fmad=false`

## OPT-070–080 successor dispositions

Preserve all original reports. Historical outcomes stay as written.

| ID | Historical outcome | Successor |
|---|---|---|
| OPT-070 | inconclusive due OPT-074 coverage | OPT-086 formal Q8/MMQ decisions |
| OPT-071 | retain instrumentation/replay repairs | reused, not reopened |
| OPT-072 | retain historical rejection inventory | eligible leftovers feed OPT-087 |
| OPT-073 | retain historical quality findings | migrate useful PPL/recurrence/dual-verdict mechanisms into OPT-083/084 |
| OPT-074 | historical production GPU/FP64 characterization | no longer active kernel-admission authority |
| OPT-075 | retain-packed due missing admission | reopened by OPT-085 |
| OPT-076 | retain-packed due missing admission | reopened by OPT-085 |
| OPT-077 | `retain_sequential` / component interval not positive | do not reopen (performance, not numerical-policy) |
| OPT-078 | `performance_rejected` / negative complete saving | do not reopen (performance, not numerical-policy) |
| OPT-079 | keep `kv_once` | retain unless OPT-088 full-model quality exposes an interaction |
| OPT-080 | historical failed batch/release gate | superseded only by OPT-088 for the new combination |

## Dependency order

```
OPT-081 parity policy → OPT-082 parity suite
OPT-083 quality framework → OPT-084 shipping quality baseline   (independent of 081/082)
then OPT-085 Q4 + OPT-086 Q8/MMQ + OPT-087 selective historical
then OPT-088 combined production gate
```

First eligible pending task in ledger row order: **OPT-081**. OPT-083 is
also eligible once OPT-080 is done; do not start OPT-085/086/087 final keep
decisions until both OPT-082 and OPT-084 are frozen.

No unrelated optimization experiments in this batch. The purpose is to
remove the broken admission bottleneck, establish the ds4-style framework,
and resolve already-promising work before searching for new kernels.

## Common implementation acceptance

Each new task creates `pins/optNNN_iteration_contract.json`, a focused task
contract, host evidence tests, and one owned fixture/report with raw
sidecars. Commands in the dossiers are interfaces to implement, not commands
available before task execution. Reuse OPT-057. Ordinary pytest stays
read-only and GPU-free. CUDA parity and quality GPU work run through the
established bounded GPU task mechanism.

- Feedback: incremental build and all children within 300 seconds. A named
  phase selects a predeclared bounded workload.
- Kernel-parity GPU work uses small/medium bounded shapes. Production-shape
  performance remains a separate test.
- Quality GPU work uses identity-cached model/authority loads. Full quality
  runs once per surviving configuration, not per edit.
- Component/E2E performance rules from POST-069 remain for OPT-085–088.
- Record independent `kernel_parity_pass`, `model_quality_pass`,
  `performance_pass`, `production_kept`. Missing evidence is incomplete, not
  pass.

Keep power/clock policy unchanged. No GPU concurrency. `plan.md` unchanged.
No tok/s claim from OPT-081–084. OPT-085–088 may claim a keep only with
measured performance after kernel parity and quality pass.
