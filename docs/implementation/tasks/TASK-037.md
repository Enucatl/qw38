# TASK-037 — Integer Q4_K MLP consumers for prefill and decode

## Status

DONE

## Milestone and dependency

M16 — Quantized MLP arithmetic in both phases. Depends on TASK-036.
[FAST-02](../task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
governs architecture, development checks and final validation ownership.
Start from the TASK-036 development runtime and FP8 mixer artifact; this is
not production promotion. TASK-026/027 remains the rollback control.

## Delivered behavior

All 64 primary-language MLPs use existing Q4_K weights with GPU-produced
signed Q8 activations: integer tensor-core matrix multiplication for M>1 and
DP4A matrix-vector multiplication for M=1. Gate/up share one pack; down uses
a fresh pack of the SwiGLU result. Eliminate global BF16 weight expansion and
weight-by-weight floating reconstruction on this selected MLP path. Keep the
working FP8 mixer consumers, Q8 head, embeddings and small families unchanged.

Adapt the Q4_K block arithmetic and MMQ/MMVQ structure from llama.cpp revision
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`, using the existing third-party
attribution pattern. Reuse the pinned CUTLASS/CuTe dependency where useful;
do not link the llama runtime, import a generic tensor system, or repeat
TASK-034's scalar-unpack WMMA design. Use the pinned implementation's SM120
tile/dispatch pattern as the first schedule; select a concrete configuration
in code, not an autotuner or tile sweep. Submit full output width to each
consumer; kernel-internal tiling is expected.

## Numerical and artifact contract

This is a new execution precision policy, not bitwise preservation of the
old decoded-BF16 contraction. Reserve `Fp8MixerQ8MlpV1 = 0x0406` (1030),
named `precision_fp8_mixer_q8_mlp_v1`, for unchanged TASK-035 FP8 mixers plus
this MLP arithmetic. Add a compiler selection and schema/runtime/report
admission together. Existing policy IDs keep their existing behavior. Do not
silently execute policy 1029 with the new MLP rounding.

Use this activation recipe for each row and contiguous K32 group:

- Evaluate RMS or SwiGLU in FP32, then apply the existing BF16 RNE producer
  rounding locally. Compute FP32 `amax` from those rounded values only.
- For a nonzero group use FP32 `a = amax / 127`; for an all-zero group use
  `a = 1` and zero codes. Encode `z = clamp(RNE(x / a), -127, 127)` as signed
  INT8. Store FP32 `a` and exact INT32 `sum(z)` for the consumer.
- Zero-pad tails without influencing amax; no scale crosses token rows.
  Reject nonfinite producers or invalid scales through the existing typed
  failure/poison path. They must not silently encode as finite zero values.

Q4_K weight codes, superblock FP16 d/dmin and six-bit subgroup scale/minimum
codes retain their existing logical interpretation. For one K32 subgroup,
with `w_i = d*s*q_i - dmin*m`, compute the contribution as
`a * (d*s*sum(q_i*z_i) - dmin*m*sum(z_i))`. Integer dot products and sums
are exact INT32; convert scales and apply affine corrections and accumulation
in FP32. A raw K32 dot is bounded by `32*15*127 = 60960` in magnitude,
and its activation sum by `32*127 = 4064`. Bound any larger or transformed
integer accumulation interval before implementation;
do not accumulate an unbounded K dimension into one INT32 value. Do not round
each reconstructed weight to BF16. FP32 SwiGLU, residual addition and final
outputs remain unchanged. The reference must implement this equation
independently of the packed GPU decoder.

Keep the existing Q4_K physical view first; decode and prefill share it. If
consumer compatibility requires a lossless reordering, prepare it offline,
version the stored layout and replace the resident view. Do not requantize
weights, change their logical quantizer, or retain a duplicate fallback view.
An identity-preserving streaming conversion of the existing artifact may
change policy metadata/layout without recomputing unchanged FP8 weights;
validate the resulting metadata and decoded code/scale equality without
payload digests. New compiler output must reproduce the same contract.

## Producer, workspace and integration

The complete chain is FP32 residual → RMS → local BF16 rounding → one Q8
pack → gate/up → FP32 SwiGLU → local BF16 rounding → new Q8 pack → down →
FP32 residual. Fuse the two packs with their producers where straightforward.
Do not redo RMS, amax or packing inside output-row loops. A pack lives in
session-owned bounded workspace until its last sibling consumer completes;
scratch reuse or a changed residual requires a new pack. No persistent
activation cache, per-token allocations or unused BF16 materialization.
Bound any FP32 gate/up slabs explicitly. Keep chunk capacity 256.

Likely code: `cuda/{activation,prefill,decode_mmv}.*`,
`src/runtime/{mlp,prefill,prefill_gdn,prefill_attention,model}.*`, compiler and
format policy/layout admission, and existing evaluation/profile labels.
Preserve public validation, session stream, ownership, poison and recovery
contracts. Schema negative cases must reject mixed or mislabeled policies
before mutation; the real Session-backed binders must exercise the new path.

## Small checks and acceptance

1. Extend existing projection/MLP tests with an independent pack/affine-dot
   reference: asymmetric q/min/scales, negative activation sums, zero groups,
   halfway rounding, saturation, subnormal weight scales, nonfinite rejection,
   padding/tails, M=1 and M>1, and integer overflow bounds. Codes/sums are exact;
   FP32 contractions use `abs(error) <= 0.02 + 0.002*abs(reference)` against
   the new-policy reference. Keep stricter existing exact checks where
   applicable. Differences against old BF16 operands are diagnostics, not
   failures of the new equation or permission to relax quality gates.
2. Exercise one-pack gate/up reuse, a fresh down pack, workspace reuse across
   layers/chunks, SwiGLU/residual epilogues, policy admission and prefill/decode
   handoff. Verify no global weight-unpack calls on selected MLP dispatch.
3. Once per arm, measure one complete real-shape MLP at M=1 and M=256,
   including RMS, packing, gate/up, SwiGLU, down and residual. Compare with the
   TASK-036 implementation using identical nonzero inputs and weight values;
   label library first-use costs. Kernel-only wins are insufficient.
4. Run the frozen eight-window TASK-035 precision development screen using
   its unchanged inputs/reference and aggregate NLL delta limit +0.03, then
   the FAST-02 short integrated check. Do not fit to C92/P100 answers.
   Record actual workspace and projected final capacity with the 2 GiB reserve.

Completion requires both integer consumers integrated, passing correctness,
development quality and state checks, lower observed complete MLP cost at
M=256, and no regression in the matched integrated prefill or decode phases.
The user-directed retention decision dated 2026-09-27 supersedes the original
M=1 complete-cost win requirement: retain policy 1030 and defer the observed
M=1 MLP regression as decode optimization work. This is an explicit acceptance
change; the M=1 regression remains visible and is not counted as a pass. Use
FAST-02's one-run comparison policy; these development observations are not
statistical claims or full-model promotion. TASK-040 owns final quality,
capacity and performance acceptance.

## Blocked execution record — 2026-09-27

GPT-6 Codex implemented the candidate in the main thread against `2ac6c44`.
Independent GPT-6 Astra high review used both permitted passes: the first
requested changes; the second returned `BLOCKED`. Both code findings were
resolved, with no further material code findings. The required complete M=1
MLP cost win remains unmet. This is an observed performance failure; its
root cause is not established by the aggregate measurements.

The final candidate's once-per-arm comparison, including RMS, packing,
gate/up, SwiGLU, down, residual and first-use costs, was:

| Rows | Parent GPU ms | Candidate GPU ms | Parent host ms | Candidate host ms | Required win |
|---|---:|---:|---:|---:|---|
| 1 | 0.794336020947 | 1.61561596394 | 0.794248 | 1.614748 | FAIL |
| 256 | 79.9333724976 | 2.82448005676 | 79.935111 | 2.823906 | PASS |

Both arms used identical nonzero inputs and unchanged Q4_K weight values,
with one execution, zero warmups and zero hot allocations. These refreshed
measurements supersede the earlier favorable MLP observations. Successful
command exits do not establish the comparative performance gate. No timing
selection or unchanged rerun was used to override the failure.

Implemented work retained for review includes policy 1030, streaming artifact
conversion with byte equality checks, local BF16-RNE Q8 producers, K32-bounded
DP4A/MMQ consumers, shared gate/up packs, fresh down packs, Session-owned
41,222,148-byte workspace, schema/runtime/report admission and independent
arithmetic and real Session checks. No payload or scale digests were computed.

Review finding R37-01 was repaired with submission tracking at the shared CUDA
MLP boundary: prelaunch validation failures preserve Session health, device
state and host metadata; failures after submission poison. Production-caller
tests cover rejection followed by valid continuation and deferred failure
followed by reset and continuation. R37-02 was repaired by translating
`bad_alloc` and `length_error` in the conversion API through the existing typed
allocation-error mapping.

Exact final repair command:
`bash .cache/evaluation/qw38-language-v2/task037-support/repair-checks.sh`.
The same script was launched from `.cache/task037/repair-checks.sh`; the retained
copy contains the pinned-container build, focused CTest invocation and all four
benchmark commands. The build passed; `q4k_q8` and `q8_mlp_integration` passed.
The comparison failed acceptance as recorded above. `git diff --check` passed.

Earlier supporting checks passed: five focused arithmetic/schema/reader/MLP/
prefill tests, real Session integration, full-language state/replay/recovery
integration and 13 report tests. The frozen eight-window screen measured NLL
delta +0.0017427048138420886 against the +0.03 limit. Matched short-request
prefill measured 279.477598 → 265.306956 ms and decode 220.447424 → 169.278664 ms.
These observations remain supporting evidence because the repairs changed
neither arithmetic nor the successful submission sequence; they cannot satisfy
the failed M=1 criterion. Projected final-capacity free memory was 8,755,609,600
bytes, above the 2 GiB reserve; this is a projection, not final capacity proof.

Evidence is preserved under
`.cache/evaluation/qw38-language-v2/task037-support/`: `evidence-notes.md`,
`wake-ea6ba1321aae.log`, `repair-checks.sh`, `repair-tests.log`,
`repaired-{parent,candidate}-mlp-{1,256}.log`, `review-candidate.diff`,
`candidate-files.json`, `binary-identities.json`, `astra-review-2.md`,
`development-report.json`, `capacity-projection.json` and earlier command logs.
Hardware was NVIDIA GeForce RTX 5090, driver 590.48.01, with pinned CUDA 13.4.1
and GCC 14.2; full container and hardware identities are in `hardware.txt`.

The second Astra review returned `BLOCKED` solely because the original M=1
performance requirement failed; it recorded no remaining material code
findings after the first-review repairs. The user then explicitly accepted the
candidate under the revised criterion above. This historical review result is
not relabeled as PASS. TASK-038–040 remain subject to their existing contracts.

## Completion report — 2026-09-27

Retained policy 1030 for prefill and decode under the user-directed decision in
the implementation ledger. Correctness/arithmetic, Session state/recovery,
policy admission and development quality checks passed. The frozen eight-window
screen NLL delta was +0.0017427048 against the +0.03 limit. The matched short
request measured prefill 279.477598 → 265.306956 ms, eight-token decode
220.447424 → 169.278664 ms, and total 499.925053 → 434.585710 ms. The refreshed
complete MLP comparison measured M=256 at 79.933372 → 2.824480 ms and M=1 at
0.794336 → 1.615616 ms. The latter is deferred work, not a passed gate.

The focused build, `q4k_q8` and `q8_mlp_integration` tests, full-language
state/replay/recovery integration, and report tests are recorded in the blocked
execution evidence above and `.cache/evaluation/qw38-language-v2/task037-support/`.
The second Astra review found no remaining material code issue but blocked on
the superseded M=1 criterion. Final quality, measured 32K capacity, replay and
PERF-01 acceptance remain TASK-040. The 8,755,609,600-byte final-capacity value
is only a projection; no production promotion is claimed.

Deferred follow-up: use TASK-040's measured populated-decode costs to determine
whether the Q4_K×Q8 M=1 MLP kernel warrants a targeted optimization. TASK-038's
attention schedule and TASK-039's submission/graph work do not directly change
that kernel.
