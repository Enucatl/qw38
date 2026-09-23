# TASK-018 — Behavioral correctness baseline

## Status
IN_PROGRESS
## Milestone
M7 — Behavioral correctness baseline
## Purpose
Freeze and implement the language-only correctness evidence needed before performance experiments can change selected hypotheses.
## Depends on
- TASK-017
## Normative references
- `docs/architecture/architecture-v0.md` — Behavior and performance validation
- `docs/architecture/evaluation-policy-v0.md` — EVAL-01; selected suite and scoring authority
- `docs/architecture/quantization-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01–Q-03, P-01–P-02, S-01–S-02 | V0 runtime and precision behavior | LOCKED |
| EVAL-01 | Frozen qw38-language-v1 inputs, scoring, budgets and coverage | POLICY |
## Starting point
Complete V0 decode and BF16 diagnostic control exist; production prefill does not.
## Scope
Materialize EVAL-01's selected P100, C92, L12 and retrieval fixtures; capture up to 24-token teacher continuations and available top-k target log probabilities from the local Q4_K_M GGUF through CUDA llama.cpp, then freeze hashed tokenized inputs/loss masks and scoring rules before comparison. Implement aligned teacher-forced V0 target-token NLL, deterministic greedy text generation, fixed-key graders, qualitative review records, llama.cpp fresh-request replay, V0 reset/continuation/snapshot tests, and the mandatory 512/4096 retrieval cases. Freeze 32768 retrieval inputs now for execution in TASK-022. Run the complete llama.cpp and V0 arms on identical frozen inputs and targets, counting each target once and resetting at document boundaries. Compare generated capability answers against fixed keys. Do not infer full-vocabulary KL from llama.cpp REST top-k output. BF16 may receive only an optional elementary tensor/load sanity check; it is not a full evaluation arm. Clearly record untested coverage; implementers do not select a smaller passing subset.
## Out of scope
Performance tuning, threshold relaxation, calibration on evaluation data, MTP metrics, intermediate equality as the quality verdict, llama/GGUF as numerical authority.
## Required interfaces
Evaluation CLI/config produces machine-readable results bound to artifact, binary, input, mask, tokenizer, and policy hashes; language-only labels are mandatory.
## Required semantics
EVAL-01 selects mean NLL delta ≤+0.03 nats/token, each declared slice ≤+0.06, and the precise two-percentage-point capability regression screen with paired uncertainty. Its basic correctness, generated-text review and continuation gates also apply. INCONCLUSIVE, INVALID and FAIL results all block completion with distinct reasons. Greedy strings across models are inspected and graded, not required token-identical.
## Data representation
FP32 logits/log-softmax; immutable hashed token IDs/masks; results include counts by domain/context and uncertainty method.
## Implementation constraints
The Q4_K_M llama.cpp teacher is the external behavior comparator; fixed keys remain capability authority. Its REST probability output is limited to the explicitly recorded top-k and is not a full-vocabulary distribution. BF16 is not a full evaluation control. Failures block quality acceptance and are not silently repaired by wider precision.
## Tuning defaults
Validation thresholds above are policy defaults, not architecture.
## Expected files/modules
Evaluation tooling/harness, frozen manifests (not copyrighted corpus payload if licensing forbids), correctness integration tests.
## Tests required
### Unit tests
NLL/KL math, masks/counting, hash binding, reset boundaries, deterministic argmax.
### Reference/numerical tests
Hand-computed logits, target alignment and mask fixtures scored against frozen teacher targets.
### Integration tests
Run the complete EVAL-01 core; repeated-decode continuation and save/restore at its declared boundaries; mandatory 512/4096 retrieval. Partial runs are diagnostic evidence only and cannot complete this task.
## Benchmark required
No throughput benchmark.
## Acceptance criteria
- [ ] EVAL-01 fixtures, Q4_K_M teacher continuations and suite/scoring/input identities are frozen before llama.cpp/V0 comparison.
- [ ] llama.cpp and V0 results report available target NLL, top-k availability, generation, capability, continuation, and coverage; no truncated output is labeled full-vocabulary KL.
- [ ] Language-only quality gate passes or task is BLOCKED with failing slices preserved.
- [ ] MTP is excluded and no thresholds are silently changed.
- [ ] Every mandatory core case is present; qualitative reviews are resolved; 32768 inputs are frozen and their execution is explicitly assigned to TASK-022.
## Architecture blocker rule
If locked V0 cannot produce correct behavior, report full `ARCHITECTURE_BLOCKER`; a quality failure alone does not authorize EXP-A/B/C/E early.

## Quality failure recovery
Preserve the failing baseline and frozen evaluation suite. First distinguish
source/semantic implementation bugs from a failure of the selected quantization
or activation-transport hypothesis. Fix implementation bugs under the existing
contract. For a hypothesis failure, propose a narrow architecture amendment that
names the decision, failing evidence, diagnostic comparison, affected weight
families, artifact identity, and memory impact. Record the decision authority
and explicit acceptance in the ledger before changing a locked policy or task
order. Re-run the unchanged quality gate before accepting a replacement baseline
and resuming the ledger. Do not relax thresholds or run an experiment early.
## Completion report
### Result
IN_PROGRESS — P10 speed diagnostic complete; full EVAL-01 acceptance remains pending.
### Changes made
Updated EVAL-01 to make Q4_K_M llama.cpp versus V0 the full comparison pair,
with BF16 restricted to optional elementary sanity. Added fixture
materialization, Q4_K_M teacher capture/validation, a top-20 probability-only
replay bound to the frozen teacher IDs, fixed-key graders, paired metric and
report drivers, a full-core V0 evaluator, and a V0 reset/snapshot replay driver.
The paired report keeps teacher full-vocabulary KL unavailable and reports
V0-only fixed-answer NLL for L12/R.

### Independent review and repair record
The independent review returned `CHANGES_REQUIRED`. The single authorized
repair round addressed all seven findings:

1. **HIGH — prohibited whole-artifact digest.** The V0 wrapper hashed and
   propagated a SHA-256 of the complete `.qw38` payload. Removed that digest;
   the run records artifact byte size and embedded schema/quantization identity
   instead.
2. **HIGH — weak lineage and duplicate coverage checks.** The paired reporter
   accepted rows without proving they came from the same frozen inputs and
   collapsed duplicate llama rows. It now binds fixture, policy, grader,
   metrics, teacher, core TSV, target and mask identities; rejects duplicate or
   incomplete cases; requires all 216 unique cases in each arm; and validates
   output, stop/count, and replay identities against the manifests.
3. **HIGH — unreachable PASS and stale manifest status.** The reporter always
   left qualitative review pending and hardcoded the report manifest as
   `INCONCLUSIVE`. Added validated, output-hash-bound `--reviews` adjudications
   for all 100 P100 cases; absent reviews remain `PENDING`, and the manifest
   now records the computed summary status.
4. **MEDIUM — missing C92 paired outcomes.** Added per-slice llama-pass/V0-fail
   and llama-fail/V0-pass counts and case IDs, plus C92 micro-accuracy.
5. **MEDIUM — incomplete teacher probability disclosure.** Added requested
   and effective `n_probs`, selected target logprobs, and explicit top-20-only
   scope to case and summary output. No full-vocabulary KL is claimed.
6. **MEDIUM — missing capability and retrieval coverage detail.** Added L12/R
   per-arm fixed-key grade summaries, R512/R4096 horizon coverage, and an
   explicit deferred R32768 status/reason; teacher NLL remains unavailable for
   L12/R.
7. **LOW — combined reset/snapshot state.** Split replay reset and snapshot
   outcomes into independent fields so one result cannot mask the other.

Repair files: `scripts/run_task018_v0_eval.sh`,
`scripts/run_task018_pair_report.sh`, `scripts/task018_pair_report.py`, and
`src/runtime/state_replay.cpp`.

Post-repair refresh: Ruff passed across TASK-018 scripts/tests; Python
byte-compilation and `bash -n` passed for the harnesses; fixture validation
passed at 222/222 fixtures, 192/192 teacher references, and 192/192 probability
rows; scoring tests passed (9/9); `git diff --check` passed; and the pinned
CUDA build reported no work for `qw38-evaluate` and `qw38-state-replay`.
Post-repair state replay binary SHA-256 is
`e8041669eea015e83efe3ee47dd16530dc292c063a8f1a25bea3aae9b64e8381`; evaluator
binary SHA-256 remains `dea24d9609d644f5309334823ab5112a0c6e7fe8f29dcb96728a2746059a426c`.
The second review returned `CHANGES_REQUIRED`. The authoritative cache-default
llama attempt reached 216/216 generated outputs but exited 1 during
fresh-request replay; it is diagnostic only and is not accepted as completed
arm coverage. No V0 full arm was started.
### Tests run
Exact commands and outcomes:

- `bash scripts/run_task018_llama_capture.sh` — exit 0; attempt
  `llama-20260923T153025Z-1465265`, 116 s wrapper duration, 192/192 P100/C92
  references captured. The complete capture log, server startup log, server
  log, teacher identity and result are under
  `.cache/evaluation/qw38-language-v1/source_capture_job/attempts/llama-20260923T153025Z-1465265/`.
  The container image was
  `ghcr.io/ggml-org/llama.cpp:full-cuda13@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6`;
  server binary SHA-256 was
  `ca7289d8434a76b514699eb39258779d0c0e556b1d270d356690e9a7dd16d501`.
  Startup recorded architecture `qwen35`, GGUF name `Qwen3.8-27B`, `Q4_K_M`,
  tokenizer vocabulary 248320, matching chat-template SHA-256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`, and
  CUDA offload of 65/65 layers. The local GGUF is 18,973,870,432 bytes; its
  payload was not hashed. Teacher manifest SHA-256:
  `e4a56d51a77597cffbd27fbbb4851c169afe3b384abb3245c6eb7c46b6259575`.
- `uv run --script scripts/task018_validate_fixtures.py --fixtures .cache/evaluation/qw38-language-v1 --require-source-refs`
  — PASS; 222 unique cases (P100 100, C92 92, L12 12, R 18), all declared
  retrieval horizons (512/4096/32768, six each), and exactly 192/192 unique
  reference targets and masks validated. The validator accepts llama.cpp's
  `limit` stop only at the exact 24-token cap; all captured cases stopped at
  that cap.
- `bash scripts/run_task018_llama_probabilities.sh` — PASS; cache-default
  attempt `probabilities-20260923T155850Z-1474314`, 115 s, 192/192 cases and
  4,467 aligned teacher targets. All 4,467 targets appeared in returned top-20
  rows; the replay reproduced every frozen teacher ID and stop. Sidecar SHA-256:
  `1781535511d82d66642cacd0d4386d75d5d6d8039c44827b7f094ec705a0627c`.
  It contains selected target log probabilities and top-20 rows, not a full
  teacher vocabulary distribution. A cache-disabled diagnostic attempt diverged
  at compsec-089 and is excluded; exact-default retry matched the frozen IDs.
- `uv run --python 3.12 --with pytest --with numpy==2.5.3 python -m pytest -q tests/test_task018_scoring.py`
  — PASS, 9 tests.
- `uv run --python 3.12 --with ruff ruff check scripts/task018_materialize.py scripts/task018_capture_llama.py scripts/task018_validate_fixtures.py scripts/task018_scoring.py scripts/task018_metrics.py tests/test_task018_scoring.py`
  — PASS.
- `docker run --rm -v /home/user/qw38:/repo -w /repo qw38-dev:cuda13.4.1-pinned cmake -S . -B .cache/task018-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF`
  — PASS; GNU 14.2.0, CUDA 13.4.59, native `120-real`.
- `docker run --rm -v /home/user/qw38:/repo -w /repo qw38-dev:cuda13.4.1-pinned cmake --build .cache/task018-build --target qw38-evaluate qw38-state-replay -j 4`
  — PASS. Latest evaluator binary SHA-256:
  `dea24d9609d644f5309334823ab5112a0c6e7fe8f29dcb96728a2746059a426c`;
  state replay binary SHA-256:
  `e8041669eea015e83efe3ee47dd16530dc292c063a8f1a25bea3aae9b64e8381`;
  development image ID `sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
- V0 infrastructure smoke:
  `docker run --rm --gpus all -v /home/user/qw38:/repo -w /repo qw38-dev:cuda13.4.1-pinned .cache/task018-build/src/qw38-evaluate --artifact build/pinned-debug/qwen-v0.qw38 --cases .cache/task018-smoke.tsv --output .cache/task018-v0-smoke --eos-ids 248044,248046,248063,248064,248065`
  — PASS for diagnostic case `case_000`: 26 prompt tokens, 24 teacher target
  tokens, 24 generated tokens, cap stop; wrote FP32 candidate logits and token
  IDs under `.cache/task018-v0-smoke/`. The input artifact was 17,095,659,090
  bytes. The full evaluator now extracts the artifact's embedded manifest and
  quantization identity into each run; the older one-case smoke predates that
  identity record and remains diagnostic only.
- `uv run --script scripts/task018_prepare_core.py --fixtures .cache/evaluation/qw38-language-v1 --output .cache/task018-prepare-check/core.tsv`
  — PASS; generated 216 exact core cases (P100 100, C92 92, L12 12, R512/R4096
  12) and hashed tokenizer-encoded fixed-answer targets for L12.
- `bash scripts/run_task018_llama_eval.sh` — RUNNING at report update; default
  cache settings match teacher capture. Attempt directory
  `.cache/evaluation/qw38-language-v1/runs/llama-20260923T160709Z-1476101/`,
  container `qw38-task018-eval-1476111`, durable `run.log` and server logs. It
  had produced 155/216 case outputs at 16:35 UTC (P100 100/100, C92 55/92); the cache-disabled
  run in `llama-20260923T160557Z-1475336/` produced partial diagnostic output
  for case_000 through case_006 and is excluded from authoritative coverage.
- Host `cmake` invocation was unavailable (`cmake: command not found`); the
  pinned-container configure/build commands above succeeded.
### Benchmark results
No performance benchmark is required by TASK-018. The cache-default llama.cpp
attempt has 216/216 generated cases, but its coordinator-directed stop
interrupted fresh-request replay, so it is not a complete arm. The V0 arm has
not started and no paired quality gate has a result. The teacher top-20 capture is complete at 4,467/4,467
target IDs. Full-vocabulary KL from the llama.cpp REST endpoint is unavailable
and remains `null`; no threshold is claimed as passed or failed.
### Architecture blocker
None observed. The one-case V0 smoke is diagnostic only and does not establish
an architecture blocker or candidate quality failure.
### Follow-up observations
Fixture coverage is 222/222; Q4_K_M teacher-reference and top-20 coverage is
192/192 P100/C92 cases, with 4,467 aligned target positions. The authoritative
cache-default llama attempt wrote 216/216 generated outputs, then was
coordinator-stopped during fresh-request replay and exited 1. These partial
outputs do not count as complete arm coverage. The V0 infrastructure smoke
covers one diagnostic case and does not count toward suite coverage.
R32768 inputs remain frozen at 6/6 for TASK-022. Remaining evidence is the
authoritative llama.cpp run/replays, full V0 core plus state replay, paired
report and human P100 review; full-vocabulary KL remains unavailable.

### Second review and blocker
The same Sol reviewer returned `CHANGES_REQUIRED` on the revised candidate
(HEAD `3d6601b9`). The single authorized repair round resolved the seven
findings above, but the second review found remaining report-contract findings:

1. Invalid fixture or identity validation throws an exception and leaves only
   an exit code. Emit a durable machine-readable `INVALID` result and reason,
   without scoring partial data.
2. The pair summary/manifest/report omit `mode: language-only`,
   `mtp_enabled: false`, comparator/candidate roles, schedule, precision, and
   Markdown links to the raw generated text.
3. Required focused target-alignment, mask, hash-binding, and replay
   failure-path evidence is not recorded.

Full evidence is also incomplete: the coordinator-directed stop ended the
llama attempt with exit 1 when its server disconnected during fresh-request
replay, after it logged 216/216 generated cases;
its outputs and failure log are preserved in
`.cache/evaluation/qw38-language-v1/runs/llama-20260923T160709Z-1476101/`, but
the attempt is not accepted as complete coverage. No full V0 run, V0 state
replay, paired report, or resolved 100-case P100 adjudication exists.

Per `run-implementation-task-codex`, findings remaining after the single repair
round require `BLOCKED`. Do not treat the generated outputs as a complete
llama arm. The coordinator resumed TASK-018 to complete the V0 core/state
replay and top-20 versus top-20 distribution comparison, with llama.cpp
full-vocabulary KL remaining unavailable. See the resumption report below.

### Coordinator-directed resumed work — 2026-09-23

Addressed the remaining report-contract findings in the current candidate:

- Pair-report validation failures atomically produce `result.json` with
  `status: INVALID`, a reason and `scoring_performed: false`; successful
  summary/manifest/report output includes language-only mode, MTP disabled,
  comparator/candidate roles, schedule, precision and relative links to each
  raw generated text.
- Added focused negative checks for teacher-target alignment, loss-mask
  contents, hash binding and fresh-request replay mismatch, plus durable
  invalid-report output. The focused scoring/report tests passed 10/10;
  Ruff, Python compilation, shell syntax and `git diff --check` passed.
- Added an efficient llama replay-completion path. It reused the preserved
  216 generated llama outputs from the prior attempt and repeated the four
  required fresh-request probes; all four matched. Run manifest SHA-256:
  `318c44a32e623b8817a5f0a30a993a518f3c32012e1f08b280a1616c52f8c1f7`.
  The original full attempt and its exit-1 replay failure remain preserved.
- The full V0 run was coordinator-interrupted at 124/216 after 2,796 seconds
  (17:04:23–17:50:59 UTC), exit 137. Its partial result reports 124 generation
  outputs and 124 candidate-logit files. `interruption.json`, original
  `result.json`, logs and outputs are retained under
  `.cache/evaluation/qw38-language-v1/runs/v0-20260923T170423Z-1492403/`.
  This is incomplete diagnostic evidence, not a V0 quality failure.
- Independently ran the V0 reset/snapshot replay: PASS for lengths
  1, 3, 4, 63, 64, 65, 255, 256 and 257; all reset and snapshot checks were
  true, and the `A,reset,B,reset,A` interleave matched. Result SHA-256:
  `ff1033ca1899f9b7cb6df5b040c9940a642e457f84baaf64881892c0b49dd22f`.
- Ran a prompt-content-selected P10 speed-only diagnostic on case_080,
  case_082–084, case_087 and case_094–098, preserving each frozen 256-token
  cap. V0 completed 10/10 in 47 seconds including artifact load, generation and
  teacher-target logit scoring (755 generated tokens; 16.06 tokens/s end to
  end). A fresh Q4_K_M llama.cpp run completed the same 10 prompts in 12.307
  seconds of requests (0.813 requests/s; 54.85 generated tokens/s), with 24.222
  seconds startup-to-ready and 36.529 seconds launch-to-final-request, excluding
  server shutdown. The V0 and
  llama wall-time scopes differ because V0 also scored the frozen targets.
  Combined diagnostic report:
  `.cache/task018-p10-selected/report.md` and
  `.cache/task018-p10-selected/paired-speed-summary.json`.
- Reran the same P10 on V0 without target/mask inputs to isolate generation
  from teacher-logit scoring. It completed 10/10 with the same 755 generated
  tokens in 44.565 seconds launch-to-exit (16.94 tokens/s), including
  container/model load. The input identity is bound to the P10 selection and
  frozen prompt hashes; no teacher logits were written. Evidence:
  `.cache/task018-p10-generation-only/run-summary.json`, `run.log`,
  `cases.jsonl`, and `input-identity.json`. This remains a diagnostic, not a
  performance-parity or quality result.

The subsequent review found that the replay-completion path could not prove
per-case provenance for the failed source attempt. That reuse path was removed;
the preserved replay result is diagnostic only and does not establish a complete
llama arm. Report repairs now gate L12/R fixed answers, bind the pair-report
driver hash, and distinguish interrupted V0 log progress from durable JSONL
rows. The interrupted V0 run logged 124 completed cases, but only 66 case
JSONL rows survived intact; 124 generated and candidate-logit files remain.

The full V0 arm, paired top-20/NLL report, capability grading and 100 P100
qualitative adjudications remain incomplete. The P10 timing is not a quality
gate or substitute for the required 216-case core. Full-vocabulary KL remains
unavailable. Fresh independent Sol review passed for the repair round and
confirmed the code fixes and focused evidence; it also confirmed that full
TASK-018 acceptance remains incomplete because the full arms and P100
adjudications were stopped.
