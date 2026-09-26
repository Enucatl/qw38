# TASK-033 — Attention probability reuse and tensor-core PV

## Status

DONE

## Milestone and dependency

M12 — Attention with less repeated work. Depends on TASK-030.
[FAST-01](../post-task030-plan.md) governs this and subsequent tasks.

## Delivered behavior and first implementation

Use the TASK-030 development attention path. In decode, compute each FP32
softmax probability once per key and reuse it for the denominator and all
256 value coordinates. Cooperatively rescale the running numerator. Preserve
the existing segment size, causal masking, fixed merge order and BF16 KV ABI.

In prefill, keep Q32/K64 and the existing BF16 tensor-core QK path. Replace
scalar PV with tensor-core BF16 P×V and FP32 accumulation. Maxima, exponentials,
normalizers, rescaling and accumulation remain FP32; rounding the local P tile
to BF16 is explicitly authorized as a new precision experiment. Scores and P
remain within the CTA; no global attention matrix or duplicate persistent KV.
Calculate P once, then convert once for its consumer. Do not share invalid
causal tails or softmax statistics between different query rows.

Likely files: `cuda/attention.cu`, `cuda/attention.hpp`,
`src/runtime/prefill_attention.cpp`, existing attention tests/benchmarks.
Reuse existing WMMA machinery. Do not add an attention framework or first
attempt a multi-head fusion: current register/shared-memory occupancy is tight.

Inspect saved TASK-030 C92 outputs/margins before changing arithmetic. If
necessary, run one diagnostic old/new attention case under identical inputs
to investigate the lost answer. Do not infer the root cause merely from the
same artifact, tune to that answer, or make restoring it the acceptance gate.
Performance development can proceed with the documented quality limitation.

## Small checks and completion

- Run affected independent attention numerical/causal tests once, covering
  query/key tails, populated prefix, segment boundary, nonempty-session handoff
  and same-schedule replay. Retain existing tolerances; check P rounding against
  the independent reference, not just the already changed candidate.
- One bounded prefill operation and one decode operation at a fixed 4096-key
  populated prefix (a 32K operation only if needed to decide the long-context schedule), plus the
  FAST-01 short integrated request once. Measure complete attention including
  merge/preparation; no full-model 32K performance run is required here.
- If BF16 P violates numerical/development quality gates, retain the FP32 PV
  control and attempt two BF16 components of P with two MMA contributions only
  if the failure demonstrates precision is the issue. This is one targeted
  fallback, not a precision sweep. If neither qualifies, record the remaining
  prefill bottleneck and retain the independently useful decode reuse change.

Complete when selected paths are wired into real prefill/decode, the affected
checks pass, one-run timings/resource use and retained fallback are recorded,
and TASK-034 can use the resulting development runtime. An inconclusive or
failed full quality gate cannot become promotion; TASK-036 owns that decision.

## Completion Report

**Result:** DONE. Decode reuses each FP32 softmax probability for its
normalizer and all value coordinates. Prefill uses Q32/K64 tensor-core QK and
two BF16 components for P×V with FP32 accumulation and softmax statistics.
The main-thread implementation and evidence collection were performed by
GPT-6 Codex; GPT-6 Luna handled this documentation and delivery.
The single-BF16 P attempt failed the existing artifact mixer bound
(maximum difference `0.0375192` against `0.02 + 0.002*abs(reference)`); the
contract-authorized two-component fallback passed without changing tolerances.
Its artifact mixer maximum was `0.000477791`, and final-layer maximum was
`0.00292161`. Independent tests also covered tails, populated prefixes,
segment boundaries, handoff, same-schedule replay, and state snapshot/reset/
poison behavior. No global workspace or BF16 KV ABI change was made.

**Exact evidence commands and results** (all from
`.cache/evaluation/qw38-language-v2/task033-support/`):

- `bash .cache/evaluation/qw38-language-v2/task033-support/baseline.sh` —
  built the baseline attention benchmark and decoder;
  its first attempt stopped at a build include error before execution. The
  corrected baseline completed both operation measurements and the short
  integrated request, recorded in `baseline.log`.
- `bash .cache/evaluation/qw38-language-v2/task033-support/validate.sh` — six
  attention CTests ran. Five passed; the artifact
  mixer test exposed the single-BF16 P failure above, so this attempt did not
  proceed to timing or integration. The four unchanged decode/unit/reference/
  integration tests passed and were reused after the fallback.
- `bash .cache/evaluation/qw38-language-v2/task033-support/validate-two-component.sh`
  — rebuilt the fallback and ran
  `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R "^attention_(prefill|prefill_artifact)$"`;
  both passed. `uv run --python 3.12 python .cache/evaluation/qw38-language-v2/task033-support/check-development.py`
  reported 128 targets, control NLL
  `1.5697900020025506`, candidate NLL `1.57088731675679`, delta
  `0.0010973147542394646` within the predeclared `0.03` bound. The frozen
  validation.window-000 development screen is not a promotion result.
- `bash .cache/evaluation/qw38-language-v2/task033-support/repair-r33-01.sh` —
  verified the benchmark upload lifetime fix, which synchronizes before
  temporary host input vectors leave scope. Corrected one-run,
  zero-warmup complete attention timings at populated prefix 4096 were prefill
  32 rows `1.85731 → 1.34397 ms` and decode 1 row `0.161376 → 0.137760 ms`;
  intervals include preparation, scan and decode merge. Resource use was
  baseline `102` registers / `57728` shared bytes / `0` local bytes / one
  block per SM, candidate `160` / `65920` / `0` / one block per SM. This adds
  8192 shared bytes and no global scratch. Earlier operation timings in
  `baseline.log` and `operations.log` are superseded by the corrected logs.

The FAST-01 short integrated request was run once per arm using
`build/pinned-release/src/qw38-decode --artifact .cache/candidates/candidate-v2-q4k-rope-fixed.qw38 --tokens-file .cache/evaluation/qw38-language-v2/task033-support/prompt-256.u32le --generate 8 --profile`.
At 256 prompt tokens, candidate prompt ingestion was `273.231 ms` versus
`274.737 ms`; seven continuation steps were `206.481 ms` versus `211.114 ms`;
model load was `3155.06 ms` versus `3160.74 ms`. Both readouts were finite.
The first six generated IDs matched and the final two differed, so this does
not establish quality parity.

Saved TASK-026/TASK-030 C92 diagnostics were inspected before changing
arithmetic. The generated streams share 258 IDs before diverging, while saved
teacher logits cover only 24 positions; no cause is inferred and no
answer-specific tuning was done. C92 remains inconclusive. No full-54, full-216
or 32K run was required or performed; TASK-036 owns final quality and
promotion.

**Environment and identities:** RTX 5090, driver `590.48.01`, 32607 MiB,
400 W, pinned CUDA 13.4.1 image
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
Artifact `.cache/candidates/candidate-v2-q4k-rope-fixed.qw38` has manifest
  digest `41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`;
  no payload digests were computed, per the manifest-only policy.
  `candidate-pass1.sha256` identifies binaries
used for numerical, development and integrated evidence. The benchmark repair
rebuild produced the binaries in `candidate.sha256`; inference source hashes
are unchanged. The two reviewed evidence sets are identified by
`reviewed-evidence.sha256` and `reviewed-evidence-pass2.sha256` in the same
support directory.

**Independent review:** GPT-6 Astra high returned `PASS` on review 2 of 2,
with no remaining code findings, acceptance/evidence gaps, or requests.
TASK-036 retains final quality and promotion responsibility.

The Q1/Q4 scalar FP32 PV diagnostic controls were retained; the prior Q32
scalar PV control is preserved at base commit `c140a36`.
