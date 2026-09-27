# TASK-046 — Reduce vocabulary-head and normalization overhead

## Status

DONE

## Milestone and dependency

M25 — Resolve two bounded secondary decode opportunities. Depends on TASK-045.
[FAST-04](../task_ledger.md#optimization-amendment--fast-04-2026-09-27)
governs shared semantics and measurement. Use TASK-045's retained development
candidate, not a mixture of earlier binaries.

## Evidence and scope

TASK-043 records 178.652 ms for the vocabulary head and 229.650 ms for hidden
RMS kernel variants in 128-token 4K decode (7.1% and 9.1% of elapsed time).
The RMS family includes fused Q8 producer packing; preserve that work when
comparing costs. Comparator head time is 87.064 ms, but kernel categories
with overlapping activity are not additive explanations of the wall-time gap.

## Implementation guide

1. Head: test the existing `DirectInput` specialization of
   `cuda/decode_mmv.cu` for the production Q8 vocabulary shape, using direct
   BF16 input reads instead of staging the complete input for every eight
   output rows. Keep other Q8 callers on their existing selection initially.
   Inspect aligned loads and cached input reuse if that hypothesis fails;
   any replacement must stay a small shape-specific change with evidence.
   Preserve the Q8 decoder's per-weight BF16 reconstruction rounding and
   FP32 FMA. Quantizing head activations for DP4A is not authorized.
2. RMS: inspect `hidden_rms_kernel` in `cuda/activation.cu`, including plain
   BF16 output and fused Q8/FP8 packs. First retain each thread's loaded
   residual values for the max, sum and output passes; preserve reduction
   order, robust scaling and rounding. Check the resulting registers and
   spills rather than assuming fewer source loads are faster. Consider a
   shared reciprocal or reduction rewrite only if measurements justify it
   and independent tests cover the changed arithmetic.
3. Evaluate the two changes independently in their component drivers, then
   integrate only winners. A rejected component may retain its old path if
   its concrete first hypothesis was implemented, measured and explained.
   At least one component must deliver a real retained improvement; rejecting
   both is BLOCKED, not successful optimization. Avoid expanding this bounded
   task into fusion, GPU sampling, a new output API or a normalization framework.

Likely callers are `src/runtime/language_model.cpp`, MLP/mixer normalization,
prefill readout and `PrefillEngine::head_generation`. Inspect all consumers
of a changed helper. Shared changes must retain successful prefill behavior.

## Numerical and lifetime contract

Keep FP32 residuals and normalization, `1+gamma`, epsilon behavior and existing
BF16 RNE boundaries. Preserve overflow-safe normalization for extreme finite
inputs, zeros/subnormals and existing nonfinite detection. Multiplying by a
reciprocal may differ from division; validate it rather than declaring bitwise
equivalence. Do not replace robust RMS with an unguarded sum of squares.
Q8/FP8 pack scales, rounding and failure propagation remain policy 1030.

Keep all vocabulary logits available, finite validation and deterministic
argmax tie behavior, typed failure/poison handling and completion before host
commit. No narrowed head, omitted logits, extra resident weights, altered
artifact, hot allocation or changed persistent state.

## Verification and completion

- Run affected Release activation/reference/integration, head/MMV and pack
  checks. Use existing overflow-threshold and asymmetric reference fixtures;
  exercise actual full vocabulary geometry, direct/staged input extents,
  zero/tail cases, BF16 rounding and the real generation/readout caller.
  Apply existing declared tolerances; preserve exact checks where applicable.
- Extend the existing decode-MMV benchmark minimally for the full head,
  and an existing activation/MLP driver for plain RMS and fused Q8 packing
  at M=1 plus M512 shared-path coverage. One execution per arm per affected
  case, identical operands, zero warmups; include stores, packing and
  completion. Correctness tests do not become timing benchmarks. Retained
  components must improve their targeted M=1 complete cost and must not
  regress affected M512 complete cost; a rejected experiment is recorded.
- Run the FAST-04 matched 4096+8 check once per arm: decode improves and
  prefill does not regress. Include head/readout and first use. Run affected
  full-model graph/eager, replay and error-recovery integration checks.
- If arithmetic changes outputs or reduction order, run the unchanged
  eight-window TASK-035 NLL screen with aggregate delta <= +0.03; otherwise
  authenticate the reused screen with affected exact-output checks. Record
  registers/shared/local bytes, workspace and projected reserve.

DONE requires all affected correctness/state/development checks, at least
one retained component win and the integrated improvement above. Document
the measured decision for both head and RMS, including rejected hypotheses.
Do not label their possible gains as established before measurement.
TASK-048 owns combined quality, capacity and final performance acceptance.

## Completion Report — 2026-09-27

TASK-046 is complete for development. The main-thread implementation used
Codex/GPT-6; final documentation and delivery are handled by Luna/gpt-6-luna
(medium). The retained change selects the existing `DirectInput`
specialization for the unpaired Q8 vocabulary-head shape 248320x5120 in
`cuda/decode_mmv.cu`; other Q8 shapes retain their prior selection. The RMS
loaded-value reuse experiment passed correctness but failed the cost gates, so
it was rejected and `cuda/activation.cu` remains identical to the TASK-045
parent. No arithmetic, reduction order, precision, artifact, state, workspace,
or output behavior changed. TASK-043 remains production and TASK-048 promotion
gates remain unchanged.

Independent review by Astra (gpt-6-astra, high) passed on the first pass with
no findings. The preserved review is
`.cache/evaluation/qw38-language-v2/task046-support/astra-review-pass1.md`,
SHA-256 `d90d488efe596253003a0566dd7dbb4623868a149b77be5db0ddefe6bccb08d0`.
The reviewed parent was `8c5823bbc3f15a410324c16724202f60377fec3f`.
The main thread confirmed the complete reviewed diff and evidence were
unchanged. The only final changed code/benchmark/test files are
`benchmarks/decode_mmv_bench.cpp`, `benchmarks/mlp_bench.cpp`,
`cuda/decode_mmv.cu`, and `tests/activation_reference_test.cpp`.

### Measured decisions

The full head complete operation improved **1.590542 → 1.540266 ms**
(**3.1609%**). Head registers changed **34 → 39**; stack/local storage is
zero and dynamic application shared memory per CTA changed **10,240 → 0
bytes**. The cuobjdump resource report includes 1024 driver shared bytes.
The selected DirectInput path adds no session storage or hot allocation.

RMS loaded-value retention changed plain/Q8/FP8/NVFP4 register counts
**30/31/34/40 → 40/40/40/40**, with zero stack/local bytes and unchanged
static shared allocations. Plain RMS M=1 improved **0.350135 → 0.340367 ms**
(2.7898%), but M=512 regressed **0.728795 → 0.733894 ms** (0.6996%). Fused
Q8 RMS M=1 regressed **0.413305 → 0.444956 ms** (7.6580%) and M=512
regressed **0.622563 → 0.688629 ms** (10.6119%). Therefore the RMS family was
rejected; the cause of its cold complete-cost regressions is not established.
Both head and RMS cases had one execution per arm, zero warmups, with stores,
readout and completion included. No statistical claim is made.

### Correctness, integrated request and resources

`experiment-tests.log` records passing activation reference/integration,
decode-MMV unit/reference/integration (including full vocabulary), prefill
projection, NVFP4, FP8, Q8 MLP integration, and language-model integration.
The independent asymmetric Q8/BF16 oracles, boundary and rounding cases, and
new RMS edge fixtures passed. The final candidate reran the changed activation
reference fixture and real model integration after rejecting RMS. The latter
covers graph/eager equality, reset/restore, interleave, movement, injected
failure/poison and uncommitted state, prefill boundaries 255/256/257/511/512/
513, capacity 32896 first decode, and maximum graph upload. Final logits match
the parent byte-for-byte. All nine request output IDs agree, final position is
4104 and logits are finite.

For the matched authenticated 4096+8 request, parent → candidate prefill was
**1571.873963 → 1559.276586 ms** (0.8014% faster), first decode
**19.813492 → 19.565690 ms**, remaining seven **113.048710 → 112.283266 ms**,
and whole eight-step decode **132.862202 → 131.848956 ms** (0.7626% faster).
Complete inference was **1704.736195 → 1691.125612 ms** (0.7984% faster).
Cold setup was **5432.693034 → 5421.004844 ms**; CUDA initialization was
**119.818415 → 109.595840 ms**. These are single observations. The reused
TASK-045 frozen eight-window TASK-035 NLL screen covers 1024 target tokens;
its authenticated candidate delta is `+0.00045418668087737757` against the
`+0.03` limit. No arithmetic or output changes remain.

The unchanged policy-1030 artifact manifest is
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
Model bytes (**21,013,686,400**), maximum chunk arena (**513,802,240**), Q8
workspace (**82,444,292**), and hot-allocation behavior are unchanged. Sampled
free memory was **8,331,984,896 bytes**, more than the 2 GiB reserve; this is a
sample, not a continuous peak claim. No checkpoint or `.qw38` payload/scale
digest was computed.

### Exact acceptance commands and preserved evidence

The complete command lines, logs, binary identities and hardware/toolchain
records are preserved in
`.cache/evaluation/qw38-language-v2/task046-support/`:

- `bash .cache/evaluation/qw38-language-v2/task046-support/parent.sh` — exit
  0; parent measurements, resource records and matched request completed.
- `bash .cache/evaluation/qw38-language-v2/task046-support/components.sh` —
  exit 0; both hypotheses were built, 10 affected Release tests passed, and
  each component was measured once. Component outputs were byte-identical.
- `bash .cache/evaluation/qw38-language-v2/task046-support/candidate.sh` —
  exit 0; final candidate build, final activation-reference and model
  integration tests, and matched request completed.
- `bash .cache/evaluation/qw38-language-v2/task046-support/authenticate.sh`
  — exit 0; source/binary/report/frozen-input identity and exact-output
  checks passed, integrated timing observations were present, and diff
  whitespace validation passed.

Measurements used the pinned image
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`,
CUDA 13.4.1 / nvcc V13.4.59, GCC 14.2, CMake 3.28.3, RTX 5090 sm_120,
driver 590.48.01, and 400 W configured power limit. The retained support
scripts require `mkdir -p .cache/task046` and matching parent/candidate
revisions to reproduce parent binaries. The main thread removed the eligible `.cache/task046/` directory (31 MB)
after confirming no live process used it; `test ! -e .cache/task046` passed.
All preserved support evidence checksums remain unchanged. Shared support
evidence remains retained. No downstream task was activated.
