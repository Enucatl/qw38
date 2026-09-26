# TASK-028 — Reuse-oriented long-context attention

## Status

DONE

## Milestone and dependency

M10 — Fast attention and native projection integration.
Depends on [TASK-027](TASK-027.md).

## Authority and delivered behavior

[DELIVERY-01](../task_ledger.md#delivery-amendment--delivery-01-2026-09-26)
replaces this task's former FP4 experiment. Preserve the accepted weights,
state ABI, FP32 arithmetic and BF16 KV/history. Deliver a production causal
prefill attention path that reuses K/V across query rows and a cooperative
decode scan. TASK-027 already establishes the bottleneck; no baseline rerun
or algorithm bake-off is needed. This is development integration; final quality
promotion belongs to [TASK-030](TASK-030.md).

## First implementation and code areas

Use the existing CUDA online-softmax structure and cache API in
`cuda/attention.cu/.hpp`, `src/runtime/prefill_attention.cpp` and
`src/runtime/attention.cpp`. Start with a 32-query × 64-key tile per query
head, BF16 QK tensor-core products with FP32 accumulators using the pinned
CUDA/CUTLASS primitives, and cooperative FP32 softmax and probability-times-V
accumulation. Keep softmax probabilities FP32; narrowing them for a faster PV
MMA is not part of this first implementation. Retain local running maximum,
sum and numerator rather than a global score matrix. K/V staging is shared
across query rows; map the six query heads to each KV head without permanent
GQA replication. Do not wrap an external inference framework or introduce a
new attention library. The existing scalar Q1/Q4 kernels remain diagnostic
controls, not the intended optimization.

For M=1, retain the current 256-key split and deterministic merge, replacing
one thread's serial 256-element QK dot with a warp-cooperative reduction.
Reuse existing segmented partial storage and gate/output epilogue. This is a
separate decode consumer: do not pad one query to a prefill GEMM and assume
it is faster. Leave current 256-token model chunks and GDN recurrence alone.

Likely additional touch points are runtime workspace sizing if needed,
`tests/*attention*`, `benchmarks/attention_prefill_bench.cpp`,
`benchmarks/attention_bench.cpp` and the existing request benchmark. Reuse
available tensor-core primitives; select the tile from resource limits, not a
sweep. If the initial tile spills or exceeds shared memory, halve the query
tile once and record the reason. A further change needs a concrete failure or
conversion-inclusive result, not proof that the first tile is optimal.

## Representation and lifetime contract

Keep 24 query heads, four KV heads, width 256, scale 1/16, the existing cache
strides and BF16 Q/K/V. FP32 dot accumulators, softmax, output accumulation and
sigmoid gating retain existing rounding to BF16 at attention output. New
reduction order may change floating-point results; it does not change equations.
Causality uses absolute `first_position + row`, including nonempty prefixes,
masked query/key tails and separate populated length versus cache capacity.

Device KV persists across calls; Q/g/y and decode partials occupy bounded
reusable device workspace. Local Q/K/V, scores and accumulators live only in
registers/shared memory for the owning kernel. Never allocate T×T scores,
replicate persistent KV six times, pin cache contents, or rely on values living
in shared memory across launches. Preserve layer/session commit and poison,
reset, restore and stream ownership contracts.

## Smallest useful validation

- Once: affected attention numerical/contract tests against independent FP32
  references, covering causal tails around the new tiles, nonempty prefixes,
  absolute positions, capacity rejection and short prefill-to-decode handoff.
  Test identical-schedule replay and changed decode segmentation/reduction.
  Reuse existing tolerances; investigate failures rather than widening them.
- Once: production request-32768 with 128 generated tokens, using TASK-027
  frozen inputs and timing boundaries, capturing its prompt phase, attention
  timing and peak workspace. Also run populated decode-32768 once to check the
  changed decode path. Compare with saved TASK-027 QW38 observations; no new
  llama.cpp run. Label profiling/first-use differences and do not claim a
  controlled speedup if instrumentation differs.
- One short existing development prompt plus continuation verifies finite
  outputs and the intended dispatch. This smoke is not core-54 acceptance.

## Completion and fallback

Complete when the new prefill path runs through the full model, correctness
and session checks pass, the selected long request demonstrates reduced
attention/prefill cost, and changed decode cost and bounded memory are recorded.
Keep the old decode scan if its cooperative replacement regresses. A failed
prefill candidate requires a targeted repair or a documented concrete blocker;
a keep-only report of the existing 177.87 s scan does not deliver this task.
No numerical failure, unsupported required operation or capacity failure can
be labeled success. No speed-parity or global-optimum proof is required.

## Completion report

**Outcome:** Complete; development integration passed and is handed to TASK-029.
TASK-030 retains final quality promotion. No downstream task was activated.

**Changed paths:** `cuda/attention.cu`, `cuda/attention.hpp`,
`tests/attention_prefill_test.cpp`, `benchmarks/attention_prefill_bench.cpp`,
this task file, and `docs/implementation/task_ledger.md`. The production path
uses Q32/K64 BF16 WMMA QK with FP32 accumulators, online FP32 softmax and PV,
shared K/V reuse across query rows, and cooperative decode QK reduction. It
retains the existing global workspace, BF16 output/gating semantics, cache and
session contracts. The benchmark metadata correction resolved TASK028-R1:
the complete-layer schedule reports Q32 and corresponding traffic; the Q4
control reports K32. No production behavior changed for R1.

**Acceptance and exact commands:**

- Numerical, tile-tail, prefix/absolute-position, capacity, replay, session,
  and prefill-to-decode checks: in the pinned Release container, ran
  `cmake --build build/pinned-release --target qw38_attention_prefill_test qw38_attention_prefill_artifact_test qw38_attention_core_unit_test qw38_attention_unit_test qw38_attention_reference_test qw38_attention_integration_test qw38_bench_request qw38-evaluate -j4`
  followed by
  `ctest --test-dir build/pinned-release --output-on-failure -V -R '^attention_(prefill|prefill_artifact|core_unit|unit|reference|integration)$'`.
  Result: all six named tests passed (11.69 seconds total), with existing
  tolerances. The artifact test exercised layer binding, chunk continuation,
  snapshot restore, poison/reset, output/state bytes, and capacity rejection.
- Resource and code-generation check: the prefill test reported Q32/K64,
  102 registers, 57,728 shared bytes, zero local bytes, and one block per SM;
  `toolchain-sass.log` confirms native `HMMA.16816.F32.BF16`. No tile fallback
  was needed.
- Development prompt and production request/decode captures: ran
  `bash .cache/evaluation/qw38-language-v2/task028-support/run.sh` (exit 0).
  It executed one 384-token development prompt with 24 greedy continuation
  tokens, then one request-32768 run with 128 generated tokens and one
  populated decode-32768 run with 128 frozen inputs; each workload had zero
  warmups. The smoke completed with finite readout. Both production output
  sequences matched all 128 baseline IDs.
- Profile analysis: `uv run --python 3.12 python .cache/evaluation/qw38-language-v2/task028-support/analyze.py` completed
  successfully. It reports request attention scan 27,507.820127 ms versus
  177,872.024551 ms at TASK-027, and decode scan 1,225.895709 ms versus
  1,910.278340 ms. Prompt time was 56,703.638446 ms versus 208,460.004337 ms;
  request total was 61,997.933318 ms versus 214,376.765282 ms; populated decode
  total was 5,309.222950 ms versus 5,969.238395 ms.
- TASK028-R1 requested metadata rebuild: ran
  `docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned cmake --build build/pinned-release --target qw38_bench_attention_prefill -j4`.
  Result: target built successfully. `git diff --check` passed.

**Identities and measurement conditions:** The frozen artifact was
`.cache/candidates/candidate-v2-q4k-rope-fixed.qw38` (21,462,978,185 bytes;
manifest identity `41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`;
payload not hashed). Inputs, source, binaries, policies, and smoke prompt
hashes are in `.cache/evaluation/qw38-language-v2/task028-support/identities.sha256`;
the base source commit is `c62e905a10e29d95332cddcc0bb656ca69342571`, and the
request binary build ID is `d33a3fee0cc3aa123bc759f959bb40c32b85fc0b`.
The reviewed complete diff (`candidate-pass2.patch`) has SHA-256
`092f7b6c07e8aaef2db6407bfe76b934fc0e37a2c529510ddb74244570923cc4`.
Reviewed production/test source hashes are recorded in
`.cache/evaluation/qw38-language-v2/task028-support/reviewed-code.json`; the
benchmark metadata source hash is in `repair-identity.json` in the same
directory.
Build and capture used the pinned CUDA 13.4.1 image
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`,
NVCC 13.4.59, GCC 14.2.0, CMake 3.28.3, Nsight Systems 2026.3.0, and an
NVIDIA GeForce RTX 5090 (GPU-e51ee570-3143-784d-789d-e3054637ad0b), driver
590.48.01, 32,607 MiB, driver-managed clocks, 400 W power limit. The same
TASK-027 adapter, frozen tokens, image, profiler settings, capacity, and timing
boundaries were used for comparison. First-use and instrumentation costs are
included. No warmups or repeated-run statistics were used; no baseline or
llama.cpp rerun was made.

Peak tracked device allocation was 24,196,583,696 bytes in both captures;
free memory after run was 8,317,239,296 bytes, above the 2 GiB reserve.
Persistent and scratch workspace geometry remained unchanged. This is a
single-run development comparison, not a quality promotion or a repeated-run
performance claim. No 216-case suite was launched. The remaining development
performance gap and quality promotion belong to TASK-030. Exact commands and
raw results are preserved in `build-tests.log`, `capture.log`, and
`benchmark-build.log`; processed observations are in `summary.json` and
`profiles.json`. The runnable capture and analysis commands are preserved in
`run.sh` and `analyze.py`, all under
`.cache/evaluation/qw38-language-v2/task028-support/`.

**Review and delivery roles:** Main-thread implementation by GPT-6 Codex.
Independent gpt-6-astra high review: pass 1 found only TASK028-R1 (benchmark labels and
traffic estimates); the metadata was corrected and the requested target
rebuilt. Pass 2: PASS, no findings or evidence gaps. Review outputs are in
`.cache/evaluation/qw38-language-v2/task028-support/astra-review-pass1.txt`
and `astra-review-pass2.txt`. Reviewed source, tests, binaries, inputs and
capture evidence remained unchanged after review. Documentation and delivery
bookkeeping were completed by gpt-6-luna medium. No blockers or follow-up items
were identified. Completed-task cache cleanup found no matching
`.cache/taskNNN` or `.cache/taskNNN-*` directories for TASK-001 through
TASK-028; no removals were needed. The scan is recorded in
`.cache/evaluation/qw38-language-v2/task028-support/cache-cleanup.txt`.
