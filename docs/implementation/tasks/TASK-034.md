# TASK-034 — Compact projection consumers without global weight expansion

## Status

DONE

## Milestone and dependency

M13 — Efficient compact projections in both phases. Depends on TASK-033.
[FAST-01](../post-task030-plan.md) is the representation and validation contract.

## Delivered behavior and chosen implementation

1. Route the one-row Q8 prefill head through the existing decode GEMV consumer.
   Reuse a requested final-position readout as the generation readout instead
   of normalizing/projecting the identical row twice. Keep full FP32 evaluator
   logits, finite checks, greedy tie behavior and complete-token commit.
2. Repair the implemented TASK-029 NVFP4 gate/up path before proposing another
   FP4 format. Submit the full output width to CUTLASS instead of invoking it
   per 512-row output tile. Prepare reusable descriptors/workspace at bind time.
   First try native M=1 with GPU packing once and reuse across gate/up; compare
   its complete cost against Q4_K paired decode before selecting the artifact.
   This introduces FP4 activation rounding in decode compared with TASK-029's
   BF16-input GEMV; validate that changed arithmetic and its phase handoff.
   Start with the existing 128×128×128 native tile, no tile contest. The FP32
   gate/up slabs at M=256 cost about 34 MiB together; bound and reuse them,
   then fuse SwiGLU/narrow output when straightforward. Allocate one reusable
   slab pair per session; no per-layer copies or activation validity across requests.
3. Implement the retained Q4_K down prefill consumer with tile-local unpack
   feeding BF16 tensor cores. Reuse Q4_K reconstruction including its affine
   min/scale terms and BF16 rounding from the existing decoder. No full or
   bounded global BF16 weight tile on this selected path. This is a custom
   mainloop, explicitly part of this task; reuse CUDA/CUTLASS tile patterns.
   Retain FP32 accumulation/residual add. If NVFP4's complete path loses or
   fails quality, reuse this consumer for Q4_K gate/up with paired SwiGLU.

Static weights retain one resident view shared by phases. Use the TASK-029
recipe unchanged for NVFP4; Q4_K code/scale reordering may be lossless, with a
versioned layout if needed. A format switch reloads a separate artifact;
there is no resident Q4_K backup beside NVFP4. No new FP4 calibration or down
requantization. RMS/pack is outside output tiling and runs once per input.
M=1 may retain BF16-input same-view GEMV where that is faster; skip unused packs.

Likely files: `cuda/{prefill,nvfp4,decode_mmv}.*`,
`src/runtime/{prefill,language_model,mlp}.*`, associated format/layout code
only if the selected compact consumer requires it. Existing bounds, session
stream, poison/recovery and tensor identity contracts remain binding.

## Short decision loop and completion

Use one real gate/up shape at M=1 and M=256, one down shape at M=256,
and one real head at M=1; include packing, contraction and epilogue. Validate
independent reconstruction/contractions, tails and residual/SwiGLU behavior
once. Extend existing checks for a reused final readout and workspace reuse;
do not introduce a benchmark framework. Use the existing precision development
screen for selected NVFP4 arithmetic and one FAST-01 short integrated request.

If native M=1 loses, retain Q4_K for both phases and implement the specified
compact Q4_K path; do not make a 4× slower decode the price of native prefill.
If tile-local unpack does not improve total cost, retain bounded unpack/cuBLAS
for affected shapes and document that unresolved traffic. Only investigate
another schedule when the result identifies a concrete cause. Selected paths
must pass numerical/session checks and improve their affected one-run cost
without an observed integrated regression; ambiguous wins do not justify
extra complexity. Record any retained slow fallback honestly.

Complete with working production callers, one selected artifact/dispatch,
actual workspace budget and short results. No full quality rerun or long
request matrix here; TASK-036 owns promotion and matched final performance.

## Completion Report

Selected the existing Q4_K/Q8 artifact (`candidate-v2-q4k-rope-fixed.qw38`,
artifact manifest `41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`)
for both phases. The one-row Q8 head uses decode GEMV, and the requested
final-position readout supplies generation logits without a second projection.
Full FP32 evaluation logits, finite checks, greedy ties, complete-token commit,
and session failure/recovery behavior remain covered by production integration
checks. GPT-6 Codex implemented the task and collected evidence in the main
thread; GPT-6 Luna prepared this completion report and delivery bookkeeping.

The full-N NVFP4 repair and M=1 experiment passed independent encoding,
reconstruction, contraction, tail and SwiGLU checks. Native M=1 was rejected at
8.27418 ms versus 0.16144 ms for paired Q4_K decode; native M=256 measured
0.274528 ms. Q4_K remains selected, so this task makes no selected-NVFP4
precision-screen or quality-promotion claim.

The custom local-unpack Q4_K tensor-core consumer passed asymmetric-weight,
tail, residual, SwiGLU, workspace-reuse and artifact-reference checks. Although
isolated M=256 gate/up and down measured 9.10621 ms and 3.13184 ms, respectively,
the integrated prompt regressed from 273.404 ms to 910.958 ms. It remains
available only through explicit diagnostic dispatch. Production retains the
bounded-unpack/cuBLAS fallback, leaving repeated global BF16 expansion as
`FOLLOW_UP_REQUIRED` for compact-consumer cost evaluation.

On the selected production path, the same frozen 256-token prompt and all eight
generated IDs matched the baseline. Prompt ingestion measured 271.235 ms and
seven continuation steps 206.543 ms, compared with 273.404 ms and 208.573 ms at
baseline. These single observations do not establish a statistical speedup.
The isolated head measurement fell from 48.7715 ms to 1.13766 ms including
first-use cost; that difference is not attributed as integrated savings.

At capacity 256, engine workspace is 69,206,016 bytes: bounded weight tile
17,825,792; full-N accumulator pair 35,651,584; normalized BF16 2,621,440;
SwiGLU 8,912,896; library workspace 4,194,304. One accumulator pair is reused
per session across all layers, including the selected Q4 fallback. Native
NVFP4 packed input/scales alias the bounded weight tile. Workspace is up
34,603,008 bytes from baseline. Projecting that delta from TASK-030's 32K
free-memory sample gives 8,282,636,288 bytes; this is not fresh capacity
evidence. C92 remains inconclusive, no quality promotion is claimed, and the
overall fast-engine goal remains unmet. Final quality/performance acceptance
remains with TASK-036. No full suite or 32K rerun was done here, and TASK-035
is not activated.

Checks ran on NVIDIA GeForce RTX 5090, driver 590.48.01, using pinned image
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
The exact commands and logs are in
`.cache/evaluation/qw38-language-v2/task034-support/`:

- `bash .cache/evaluation/qw38-language-v2/task034-support/baseline.sh` exited
  0: built benchmark/decode targets; captured baseline
  MLP/head timings and integrated request.
- `bash .cache/evaluation/qw38-language-v2/task034-support/first-check.sh`
  exited 0: `git diff --check`, focused build, and
  `QW38_AUTHORITY_CHECKPOINT=/workspace/.cache/authorities/qwen3.8-27b-transformers ctest --test-dir build/pinned-release --output-on-failure -V -R "^(prefill_projection|nvfp4)$"`
  passed; measured Q4_K and NVFP4 complete consumers.
- `bash .cache/evaluation/qw38-language-v2/task034-support/integration.sh`
  exited 0: the integrated rejected local-Q4 request completed, and
  `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R "^(prefill_artifact_integration|language_model_integration)$"`
  passed; kernel resources were recorded.
- `bash .cache/evaluation/qw38-language-v2/task034-support/fallback.sh` exited
  0: `git diff --check`, selected-path focused build,
  `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R "^(prefill_projection|prefill_artifact_integration|language_model_integration)$"`
  passed, the integrated selected-path request completed, and explicit local-Q4
  diagnostic selector smoke passed.

Final source and executable identities are recorded in `final-source.sha256`
and `final-binaries.sha256`; the selected artifact and hardware details are in
`artifact-stat.txt` and `hardware.txt`. Independent GPT-6 Astra high review
passed in one pass with no findings, acceptance gaps or evidence requests
(`astra-review.md`).
No task-specific `.cache/task*` directories existed for cleanup; the nested
evaluation evidence directory was preserved.
