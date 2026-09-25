# TASK-022 — Candidate decode and core quality gate

## Status

BLOCKED

The selected TASK-021 artifact failed the language-v1 EVAL-01 core gate. The
language-v1 L12 scorer also required the frozen llama.cpp result of 10/12 to
become 12/12. The 2026-09-25 language-v2 policy amendment clarifies the L06
prompt and ignores capitalization for alphabetic L12 answers; the old result
remains historical evidence, not a language-v2 pass. The selected artifact
still fails NLL and C92. Do not use it for TASK-023 production-prefill
integration or mark this task complete. A corrected artifact needs the complete
54-case language-v2 gate before this task can complete.

The separately versioned [Q4_K MLP candidate](../q4k-candidate.md) was compiled
from original BF16 and verified with the retained Quartz runtime precision.
Its original eight-window development screen improved NLL by only 0.005765
nats/token over Q4G64 and remained +0.139444 above the comparator. A follow-up
found incorrect frozen RoPE frequencies in that artifact. The corrected
compiler-patch-2 artifact scores +0.001487 above the comparator on the same
screen and clears its frozen +0.03 promotion bound. The complete language-v2
gate has not run; the task remains blocked.

The [follow-up numerical attribution](../task022-numerical-attribution.md)
finds the first sharp residual difference at layer 3 full attention on
populated-context tokens, before that layer's MLP. The 353 GGUF F32 controls
are byte-identical across the BF16-derived source, selected proxy, and external
comparator, so an additional FP32 snapshot is not needed to inspect those
payloads. Matched layer-3 traces isolate the RoPE correction and remove the
first sharp attention jump; this diagnostic does not establish a complete
EVAL-01 pass.

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Integrate the selected compact weights into full-model decode and establish the complete core quality gate before production-prefill integration.

## Depends on

- [TASK-021](TASK-021.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — scoring
  criteria and measurement definitions.
- [54-case core amendment](../../architecture/evaluation-policy-core-54.md) —
  routine coverage and manual-only full-suite execution.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| Q-01/Q-02, projection part of P-02, L-01 | Consume TASK-021 candidate with explicit activation/dispatch identity | OVERALL-01 selection |
| P-01, S-01/S-02, G-01/G-02 | Retain model equations, FP32 residual/state arithmetic and correct session semantics | Retained |
| EVAL-01 | Complete 54-case core acceptance formerly owned by TASK-018 | Policy, coverage and ownership remapped |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

The TASK-021 candidate artifact is independently validated. Real-shape kernel evidence and old V0 semantic/development controls exist; the candidate has no accepted routine core quality pass.

## Scope

Integrate the selected weights into all primary-language decode layers and the
head. Compare native small-M W4A4 with same-weight W4A16 GEMV and retain the
measured choice per family/shape, with bounded scratch and no hot-path weight
repacking. Preserve FP32 residual/state arithmetic, complete-token commits,
poison/reset/restore semantics and FP32 output logits. Record activation policy
as part of dispatch identity; a kernel switch can change numerical behavior.

Run component and source-semantic regressions, then the complete 54-case EVAL-01 core
(15 sampled P100, 15 sampled C92, all L12 and 512/4096 retrieval, NLL/slices,
capability, selected P100 adjudication,
same-schedule replay and declared continuation boundaries). Regenerate invalid
reference evidence; historical partial outputs remain diagnostic only.
**Exit:** a passing candidate decode core and measured populated-decode costs.
If quality fails, diagnose and amend the candidate with the unchanged gate;
do not push an unaccepted quantizer into production-prefill integration.

## Out of scope

Production prefill, 32768 extension acceptance (TASK-026), incomplete-arm reuse, threshold relaxation, or silently broadening precision to hide a quality failure.

## Required interfaces and data representation

Typed decode plans bind family/shape, quantizer/layout, activation scaling, kernel/fallback, scratch and epilogue. Candidate evaluation records retain language-only mode, precision/schedule and all fixture/target/mask/reference identities. Reuse existing evaluation tooling where its contract remains valid.

## Required semantics and constraints

Preserve all 64 layers, FP32 logits, complete-token commits and poison/reset/restore behavior. Compare native small-M W4A4 against same-weight W4A16 GEMV with conversion and dispatch costs included. No hot-path full-weight repacking. Count teacher targets once, reset per document and do not infer full-vocabulary KL from top-k output.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Select dispatch from measured family/shape costs. EVAL-01 aggregate NLL delta <= +0.03 nats/token, declared slice delta <= +0.06, capability/uncertainty and generated-text rules remain binding; do not redefine those gates here.

## Expected files/modules

Decode CUDA consumers/wrappers, runtime layer/model binders and scratch planning, candidate evaluation/report integration and continuation tests.

## Tests required

### Unit and contract checks

Descriptor/policy compatibility, scaling/padding/epilogues, deterministic argmax, scratch bounds and failure behavior; target/mask/count/provenance checks where tooling changes.

### Reference and numerical checks

Decoded-operand contraction checks and retained TASK-017 source-semantic regressions. Separate quantization error from implementation error; BF16 remains a diagnostic component control.

### Integration checks

All 54 paired core cases including 512/4096 retrieval, NLL/slices, capability, resolved 15-case P100 review and required comparator replay. Same-schedule reset/snapshot/interleave at lengths 1, 3, 4, 63, 64, 65, 255, 256, 257 plus late-failure recovery. Invalid or partial source attempts cannot supply accepted core coverage. The optional 216-case suite is human-initiated interactive work only; agents must never launch it.

## Benchmark required

Populated decode and family-level native/GEMV comparison with input scaling, launch, synchronization and readout costs. Label these development measurements; the matched whole-request baseline is TASK-027.

## Acceptance criteria

- [ ] All primary-language layers and the head consume the selected artifact with explicit dispatch/activation policy and bounded scratch.
- [ ] Numerical, source-semantic and session failure/recovery regressions pass.
- [ ] Complete authenticated 54-case EVAL-01 core evidence passes, including NLL/slices, capability, retrieval, required replays and resolved selected-P100 adjudication.
- [ ] Same-schedule continuation/reset/snapshot and interleave checks pass at every required boundary.
- [ ] Measured decode costs and selected fallbacks are reported; quality failures preserve the baseline and require unchanged-gate retesting.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

BLOCKED — the integrated Q4G64 MLP / Q8G32 attention, GDN and head artifact
completed all 216 paired core cases, but it failed aggregate NLL, six of seven
NLL slices, and C92 capability. Retrieval at 512 and 4096 tokens and the
same-schedule state replay passed. P100 review is still pending because the
numerical and capability gates already reject this artifact. No quality pass,
production-prefill promotion, Astra review, Luna delivery, commit or push is
claimed.

### Changes made

Integrated candidate Q4G64/Q8G32 dispatch across all 64 layers and the head,
bound BF16 activation policy in decode descriptors and candidate identity,
and extended evaluation tooling to keep the selected artifact and binary
identities. The candidate artifact manifest digest is
`94c9ed5c9260ebde73b0eb9316b6ae7726ff82fe030f2d4caa72760deec79dd1`;
the evaluation binary SHA-256 is
`6e9d0e6ef842d73a34b86e37549229eddb1eba575677d0592bd3271a3b248829`.
The fixture-policy rebind is
[`task022-eval-policy-rebind.json`](../task022-eval-policy-rebind.json);
it preserves the frozen prompts, teacher IDs/probabilities, masks, scoring,
metrics and thresholds while recording the OVERALL-01 policy change.

### Tests run

The pinned Release build and four affected CTest targets passed in
`.codex-wake-run/9ceccd38804c.log` using
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace
-w /workspace qw38-dev:cuda13.4.1-pinned bash -lc 'cmake --build
build/pinned-release --target qw38_decode_mmv_unit_test
qw38_decode_mmv_integration_test qw38_language_model_plan_test
qw38_language_model_integration_test qw38_evaluate qw38_state_replay
qw38_decode -j4 && ctest --test-dir build/pinned-release
--output-on-failure -R
"^(decode_mmv_unit|decode_mmv_integration|language_model_plan|language_model_integration)$"'`.
`uv run --python 3.12 --with pytest --with numpy --with transformers python -m
pytest -q tests/test_task018_scoring.py` passed 13/13; Ruff, Python
compilation, shell syntax and `git diff --check` passed.

The two-token full-model decode smoke passed in
`.codex-wake-run/942d85cb871e.log` using `docker run --gpus all --rm -u
"$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace
qw38-dev:cuda13.4.1-pinned build/pinned-release/src/qw38-decode
--artifact .cache/task021/candidate.qw38 --tokens 1,2 --generate 2
--profile`. The replay command in `.codex-wake-run/b4ab7f01d146.log`
passed reset/restore at lengths
1, 3, 4, 63, 64, 65, 255, 256 and 257, plus A/reset/B/reset/A interleave;
raw results are `.cache/task022/state-replay.json`. Its exact command is the
first line of the cited log.

Fresh teacher capture: `bash scripts/run_task018_llama_eval.sh`,
`.codex-wake-run/e49ae77de4cd.log`, 216/216 cases and four identical
request replays. Candidate: `QW38_EVAL_RUN_PREFIX=candidate
QW38_EVAL_BINARY=build/pinned-release/src/qw38-evaluate
QW38_EVAL_ARTIFACT=.cache/task021/candidate.qw38
QW38_STATE_REPLAY_BINARY=build/pinned-release/src/qw38-state-replay bash
scripts/run_task018_v0_eval.sh`, `.codex-wake-run/722739ae3bda.log`,
216/216 cases and state replay PASS. The paired report command was
`QW38_EVAL_POLICY_REBIND=docs/implementation/task022-eval-policy-rebind.json
bash scripts/run_task018_pair_report.sh
.cache/evaluation/qw38-language-v1/runs/llama-20260924T231919Z-512992
.cache/evaluation/qw38-language-v1/runs/candidate-20260925T000444Z-521240`;
it exited zero and reported `FAIL` in `.codex-wake-run/2ef846d6476e.log`.
Raw runs and the paired report are under
`.cache/evaluation/qw38-language-v1/{runs,paired}/` with those run IDs.

The paired report's 4,467 teacher targets have candidate-minus-llama NLL
`+0.144261` nats/token versus the `+0.03` limit. P100 English and Italian
are `+0.098648` and `+0.092298` versus `+0.06`; C92 AIME, COMPSEC, GPQA and
SuperGPQA are `+0.140345`, `+0.280137`, `+0.211766` and `+0.170719` versus
`+0.06`. P100 code is `+0.051459` and passes that slice limit. C92 exact
accuracy is 9/92 candidate versus 41/92 llama; L12 is 9/12 candidate versus
10/12 llama. R-512 and R-4096 are 6/6 in both arms.

There is also an independent acceptance conflict in the fixed-answer scorer:
`summarize_fixed_answers` in `scripts/task018_pair_report.py` returns `PASS`
for L12 only when **both** arms answer all 12 cases correctly. The frozen
fresh llama.cpp run answers L03 as `Green` versus the exact key `green`, and
L06 as `sugar` versus `dog`, so its L12 result is 10/12. No candidate change
can make this paired scorer report L12 `PASS`. EVAL-01 requires both arms to
be graded against fixed keys but does not state that the comparator must score
12/12 for candidate acceptance. Resolving that gate interpretation requires
an explicit policy decision; the frozen baseline output and keys must remain
unchanged.

After this blocked run, the user selected the language-v2 L06 prompt
`Translate the Italian word 'cane' into English. Reply with one word.` and
case-insensitive grading for alphabetic L12 answers. Isolated diagnostics on
the unchanged artifacts returned `dog` for llama.cpp and `Dog` for the Quartz
candidate on the ASCII-quote prompt; the latter is correct under the revised
grader. These diagnostics do not replace a complete language-v2 paired gate.
Evidence: `.cache/task022/l06-probe.json` and
`.cache/task022/l06-quartz-probe/summary.json`.

For diagnosis, the corrected source-logit command in
`.codex-wake-run/21e5524ca34d.log` evaluated the selected candidate on the
archived TASK-017 token IDs 1 and 2. Its full-vocabulary logits are compared
in `.cache/task022/oracle-probe/summary.json`: candidate mean absolute error
against the source is 0.189197/0.180915 at positions 0/1, versus
0.264380/0.241433 for prior V0 and 0.024222/0.026764 for the BF16 engine
control. The candidate restores the source argmax at both positions, whereas
prior V0 differed at position 1. The historical V0 evaluation used an invalid
earlier teacher capture, so its case logits are not used for aligned quality
scoring. The source-logit control identifies a substantial remaining
quantization gap, not a passing quality result or proof that every runtime
path is free of defects.

### Benchmark results

The two-token full-model smoke measured populated decode at 25.0421 ms/step;
this is a narrow development timing, not the required family/native-versus-GEMV
benchmark or matched whole-request baseline. Those costs remain unmeasured
because the selected artifact was rejected before dispatch promotion.

### Architecture blocker

These are task execution and quality-gate contract blockers, not conflicts
with a locked Architecture V0 model decision. The selected artifact fails
EVAL-01 by a wide margin on an
authenticated complete pair; the existing Q4 fallback is lower precision for
the sensitive families and the prior all-Q4 V0 control is worse on the
source-logit probe. TASK-020's proxy winner was
explicitly not bit-identical to this QW38 artifact. At the time of this blocked
report, no alternate QW38 policy had a screened artifact and complete passing
core gate. The later RoPE-corrected Q4_K artifact passes the frozen development
screen but still requires source/component regressions and the then-required
full 216-case gate (now superseded by the 54-case routine gate above). Do not relax thresholds or reuse
the rejected report as acceptance evidence. The language-v2 L12 amendment
requires new fixture and paired-run identities before it can supply acceptance
evidence; the language-v1 L12 conflict remains part of the historical report.

### Follow-up observations

P100 adjudication, candidate-specific native/GEMV family cost comparisons,
and late-failure recovery beyond the existing continuation replay remain
unproven. TASK-023–026 remain inactive until TASK-022 has a quality-passing
candidate and the remaining acceptance evidence.
