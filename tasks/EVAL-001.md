# EVAL-001 — Implement `qw38-eval` logits/traces/checkpoints harness

## Control

- Primary ID: `EVAL-001`
- Coupled IDs: `none`
- Dependencies: `ORA-001`, `SES-003`; both were `done` at planning inspection
- Status: `blocked`
- Ledger acceptance: Focused native diagnostics are driven by typed pytest helpers

## Goal and boundaries

Turn the existing collection of low-level `qw38-eval` switches into a stable,
machine-readable CUDA evaluation harness. A caller supplies an exact non-empty
token sequence and selects logits, trace, or checkpoint verification; the
harness retains authenticated outputs that typed Python helpers can validate and
that QLT-001 can later consume without parsing ad-hoc terminal text.

- Constraints: preserve all existing low-level host/scalar diagnostic switches;
  use only the public `Engine`/`Session` boundary for normal logits and
  checkpoint modes; keep CUDA tensor tracing in a separately compiled
  `QW38_DIAGNOSTIC_TRACE` binary and force its already-admitted unfused/non-graph
  path; accept only the pinned model; use raw token IDs so evaluation inputs are
  authority-independent; emit complete FP32 logit/tensor rows, exact token and
  frontier metadata, SHA-256 identities, and atomic output publication; reject
  malformed, out-of-vocabulary, empty, duplicate-filter, existing-output, and
  unsupported-build requests before publishing evidence; retain C++17,
  no-exceptions/no-RTTI, the pinned CUDA 13.0.2 SM120 build, and one live GPU
  session at a time.
- Non-goals: choosing held-out prompts or quality thresholds; calculating NLL,
  recurrence, retrieval, or task scores; running the QLT-001 suite; comparing
  baseline engines; changing inference arithmetic, trace names/shapes,
  checkpoint format, immutable tolerances, public `Engine`/`Session` APIs,
  allocation budgets, or `plan.md`; supporting text/template input, sampling,
  vision, MTP, non-CUDA production, or concurrent evaluation cases.
- Plan impact: `none`
- Affected interfaces: new high-level `qw38-eval` CLI modes and versioned result
  schema; new `build/cuda/qw38-eval` and
  `build/cuda/qw38-eval-diagnostic` products; typed Python request/result helpers.
  Existing low-level switches remain compatible.

## Repository evidence

- `plan.md:34` — V1 explicitly assigns quality, logits, traces, and checkpoint
  verification to `qw38-eval`.
- `plan.md:58` — runtime logits are complete FP32 values.
- `plan.md:72` — normal product code is bounded by move-only `Engine` and
  `Session` owners with explicit `Status` results.
- `plan.md:76` — `Session::sync` owns exact token-prefix execution, while
  `Session::logits`, `save`, and `restore` expose the required product behavior.
- `plan.md:94` — trace evidence is a JSON manifest plus little-endian tensor
  blobs with participant identities, prompt/tokens/positions, checksums,
  summaries, and before/after session state.
- `plan.md:104` — expensive tracing belongs only in diagnostic builds and uses
  layer/tap filters with fusion or graphs disabled as needed.
- `plan.md:117` — token histories, state frontiers, and checkpoint metadata are
  exact correctness boundaries.
- `plan.md:126` — pytest must use typed Python helpers to invoke focused native
  diagnostics.
- `implementation_ledger.md:51` — ORA-001 is done and provides immutable
  authority inputs for future evaluation.
- `implementation_ledger.md:63` — SES-003 is done and supplies atomic
  checkpoint save/restore with exact resumed continuation.
- `implementation_ledger.md:78` — EVAL-001 is the pending harness increment;
  QLT-001, rather than this task, owns admitted quality thresholds and the 128K
  retrieval fixture.
- `include/qw38/engine.h:80` — the public session API already exposes sync,
  logits, eval, save, restore, and committed token retrieval.
- `cuda/full_scheduler.h:348` — the admitted CUDA trace entry point exists only
  under `QW38_DIAGNOSTIC_TRACE` and accepts one exact typed filter and sink.
- `tools/qw38_trace.py:31` — typed, immutable Python records and strict v1 trace
  readers/comparators already exist and should be reused rather than duplicated.
- `src/eval.cpp:2464` — the current executable is a switch dispatcher for many
  low-level diagnostics and has no coherent logits/checkpoint/production CUDA
  harness contract.
- `Makefile:57` — only a host `qw38-eval` product is currently linked; the CUDA
  product list omits eval.

## Implementation decisions

Keep the existing diagnostic entry points intact and add one high-level command
shape:

```text
qw38-eval MODEL --mode logits|checkpoint --tokens CSV --output DIR
  --source-revision REVISION --source-state clean|dirty [--continuation CSV]
qw38-eval-diagnostic MODEL --mode trace --tokens CSV --output DIR
  --source-revision REVISION --source-state clean|dirty
  --trace-filter LAYER:TAP [--trace-filter LAYER:TAP ...]
```

`CSV` is a comma-separated list of unsigned decimal IDs in `[0, 248320)` with
no whitespace or empty fields. All modes require a non-empty initial sequence;
checkpoint mode additionally requires a non-empty continuation; trace mode
requires one or more distinct filters from the five CUDA-visible filters frozen
by `pins/cuda_trace_contract.json`, with `global` denoting the non-layer scope.
Every high-level run requires a non-empty tested source revision and an explicit
`clean` or `dirty` source state; the record retains both and hashes the invoked
binary as its tool artifact identity.
`--help` exits 0, no arguments or usage errors exit 2, runtime/evidence failures
exit 1, and success exits 0. Normal builds reject trace mode explicitly;
diagnostic builds support all three. Parse and validate the complete request
before opening the model or creating output.

For logits mode, open the pinned model, create one session, sync the token
sequence, retrieve the complete 248,320-value FP32 row, and atomically publish a
new output directory containing `result.json` and `logits.f32le.bin`. The exact
`qw38.eval-result.v1` record includes mode/status, model/tool identities, input
tokens and positions, committed frontier, dtype/shape/byte count/SHA-256/numeric
summary, greedy token with lower-ID tie breaking, and explicit CUDA/backend
identity. It carries no quality verdict.

For checkpoint mode, create and sync an uninterrupted session, save its prefix
checkpoint inside a private temporary directory, evaluate the continuation,
and retain its final complete logit row and committed tokens. Destroy that
session before allocating a fresh restore session; restore the saved prefix,
evaluate the identical continuation, and require byte-exact tokens and FP32
logits. Atomically publish `result.json`, `checkpoint.qw38`, and
`continuation_logits.f32le.bin`. The record contains prefix/continuation tokens,
pre/post frontiers, checkpoint byte count and SHA-256, restored token equality,
continued-logit equality and digest, and pass/fail. A mismatch or any failed
save/restore/eval leaves no final output directory.

For trace mode, use the already-admitted diagnostic CUDA scheduler, not the
public release API. For each requested filter, build a fresh zero-state
diagnostic session, execute all tokens before the final token through the normal
unfused path, and execute the final token through `execute_token_traced`. Fresh
sessions ensure every filtered tensor describes the identical logical run.
Collect exactly one full tensor per filter, reject sink/count/name/shape
mismatches, and atomically publish the existing `qw38.trace` v1
`manifest.json`/`tensors.f32le.bin` layout. Record the raw token IDs and
positions in the existing `prompt` record with empty prompt bytes (the frozen
v1 schema has no separate input-label field), model/tool identity, exact
before/after frontiers, empty state-digest maps because the CUDA scheduler
exposes no state hashing without new device copies, and the exact scalar-pinned
tap roles/shapes. Do not change
`tools/qw38_trace.py`, `pins/trace_contract.json`,
`pins/scalar_trace_contract.json`, or frozen tolerances unless implementation
finds a pre-existing contract defect, in which case stop as discovered work.

Compile `src/eval.cpp` with `QW38_CUDA_RUNTIME` for
`build/cuda/qw38-eval`, and with both `QW38_CUDA_RUNTIME` and
`QW38_DIAGNOSTIC_TRACE` for `build/cuda/qw38-eval-diagnostic`; link the latter
to the existing diagnostic scheduler object and backend-neutral diagnostic
trace object, never to the ordinary scheduler object. Add both to the
applicable CUDA build goals. The ordinary CUDA eval and scheduler objects must
not contain the traced entry point, trace helper symbols, or admitted tap
literals. The final executable may legitimately contain `final_norm` from the
pinned model tensor-role metadata in `src/model.cpp`; executable-wide string
absence is therefore not a valid isolation test.

Add `tools/qw38_eval.py` with annotated frozen dataclasses for the three request
types and validated result types, one subprocess runner that never invokes a
shell, strict exact-key/schema/range/hash/blob validation, and integration with
`read_trace_bundle` for trace results. It must distinguish process failure from
malformed evidence and expose complete logits as immutable tuples. Pytest calls
the native binaries only through these helpers.

Freeze the command/result contract in `pins/eval_contract.json` and retain one
RTX 5090 smoke result for each mode plus malformed/no-publication cases in
`fixtures/eval_harness.json`. The smoke proves harness wiring only and is not
QLT-001 quality evidence. Document how raw tokens flow through each mode, what
is exact versus numeric, diagnostic isolation, atomic publication, failure
behavior, and the QLT-001 proof boundary.

- Exact anticipated changed files: `tasks/EVAL-001.md`, `Makefile`,
  `src/eval.cpp`, `tools/qw38_eval.py`, `tests/test_eval.py`,
  `tests/test_build.py`, `pins/eval_contract.json`,
  `fixtures/eval_harness.json`, `docs/64-eval-harness.md`, `README.md`,
  `docs/README.md`, `docs/sources.md`, and `implementation_ledger.md`.
- Invariants: low-level eval commands and host build tests remain compatible;
  release eval/scheduler objects contain no trace sink/entry point/tap literals;
  linked model metadata is not misclassified as trace instrumentation; each mode uses
  the pinned artifact and exact raw tokens; full logits/tensors are retained as
  little-endian FP32 with verified hashes; checkpoint comparison uses only one
  live GPU session at a time; failures never publish a partial final directory;
  inference math, checkpoint bytes, trace contracts, and frozen tolerances are
  unchanged.
- Rejected alternatives: a Python-only harness was rejected because the ledger
  assigns the product interface to `qw38-eval`; JSON request files were rejected
  because they would add a second native JSON/config surface when typed helpers
  can construct a narrow exact CLI; text prompts were rejected because raw
  tokens are the cross-authority comparison boundary; exposing trace methods in
  the public `Engine`/`Session` API was rejected because tracing is diagnostic
  internal behavior; keeping two simultaneous full sessions for checkpoint
  comparison was rejected because it conflicts with the one-session memory
  boundary; calculating quality scores here was rejected as QLT-001 scope.
- Discovered ledger work: `none`
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions: typed Python requests drive all three native modes;
  logits mode retains one complete finite 248,320-value FP32 row with matching
  byte count/hash/frontier and deterministic greedy token; trace mode accepts
  each of the five frozen exact CUDA filters, emits one correctly named/shaped
  full tensor per filter through a valid v1 trace bundle, and is unavailable
  from the normal binary/object; checkpoint mode publishes an authenticated
  prefix checkpoint and proves byte-exact tokens and complete continuation
  logits after restore using sequential sessions; malformed options/tokens,
  unsupported filters/builds, existing destinations, corrupt restored data, and
  injected output failures return the documented status and publish no partial
  result; existing low-level eval behavior and all repository tests pass;
  fixture/documentation claims remain explicitly harness-only.
- Tests/fixtures to add or change: add `tests/test_eval.py` for typed request and
  exact-key parser unit cases, fake-native subprocess outcomes, malformed blobs
  and hashes, contract/fixture/document linkage, atomic failure behavior, CUDA
  product smoke for logits/checkpoint, diagnostic trace smoke for all five
  filters, and release trace exclusion; update `tests/test_build.py` for the
  eval no-argument/help exit contract; add `pins/eval_contract.json` and
  `fixtures/eval_harness.json`.
- Focused commands: `python -m json.tool pins/eval_contract.json >/dev/null`;
  `python -m json.tool fixtures/eval_harness.json >/dev/null`;
  `uv run pytest -q tests/test_eval.py tests/test_build.py tests/test_trace.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py`;
  `make diagnostic`.
- Repository-wide commands: `uv run ruff format .`; `uv run ruff check .`;
  `make clean`; `make -j2`; `make diagnostic`; `uv run pytest -q`;
  `git diff --check`.
- Native/CUDA/hardware gates: `docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$(pwd):/workspace" qw38-cuda:13.0.2 make clean all diagnostic cuda-products cuda-native`;
  `QW38_RUN_CUDA_TESTS=1 uv run pytest -q tests/test_eval.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` on the repository RTX 5090;
  inspect the release eval/scheduler objects with the exact object-scoped
  commands in the diagnostic recovery plan below; do not use an
  executable-wide `final_norm` string check.
- Documentation/evidence updates: add `docs/64-eval-harness.md`; update
  `README.md`, `docs/README.md`, and `docs/sources.md`; add
  `pins/eval_contract.json` and `fixtures/eval_harness.json`; update only
  EVAL-001 evidence/status and its chronological start/result entries in
  `implementation_ledger.md`.
- Definition of done: all three modes are callable through strict typed Python
  helpers, their full authenticated outputs and exact metadata validate, the
  checkpoint and diagnostic-isolation guarantees pass on the pinned RTX 5090,
  all focused and repository-wide gates pass, and documentation clearly leaves
  quality admission to QLT-001.

## Run record

### Planning

- Agent/model: planning agent, `gpt-5.6-sol`, medium reasoning
- UTC/time/tokens/cost: 2026-09-03T11:14:54Z; 226.273 s;
  1,309,711 total tokens (1,298,998 input, 1,198,592 cached input,
  10,713 output, 4,215 reasoning output); cost unavailable
- Outcome: decision-complete dossier created at `tasks/EVAL-001.md`; no coupled
  IDs; no implementation, plan, ledger-status, or commit changes made.

### Implementation

- Agent/model: implementation agent, `gpt-5.6-luna`, medium reasoning
- Changes: added strict typed request parsing, subprocess runner, and logits
  evidence reader in `tools/qw38_eval.py`; added focused parser/reader tests in
  `tests/test_eval.py`; added initial `pins/eval_contract.json` and
  `fixtures/eval_harness.json`; added high-level request validation and logits
  publication path to `src/eval.cpp`; added CUDA eval product build rules to
  `Makefile`.
- Commands: `uv run ruff format tools/qw38_eval.py` (passed); `uv run ruff check
  tools/qw38_eval.py` (passed); `make -j2 build/qw38-eval` (passed);
  `uv run pytest -q tests/test_eval.py` (3 passed); `uv run pytest -q
  tests/test_eval.py tests/test_build.py` (16 passed, 1 failed: pre-existing
  SHA-256 evidence hash for `src/eval.cpp` no longer matches after the required
  source change).
- UTC/time/tokens/cost: 195.595 s; 1,081,545 total tokens (1,072,863 input,
  971,008 cached input, 8,682 output, 1,330 reasoning output); cost unavailable
- Additional checks: `python -m json.tool pins/eval_contract.json >/dev/null`
  and `python -m json.tool fixtures/eval_harness.json >/dev/null` (passed);
  `build/qw38-eval --help` (exit 0); malformed token invocation (exit 2);
  `git diff --check` (passed).
- Scope issue: the implementation currently provides the logits path and
  request/evidence scaffolding only; checkpoint publication and diagnostic CUDA
  trace capture remain incomplete. The focused combined test also requires the
  existing SHA-256 source digest to be refreshed by the evidence/documentation
  stage. No additional task was inferred or added.

### Documentation

- Agent/model: documentation/evidence agent, `gpt-5.6-luna`, medium reasoning
- Changes and evidence: added [`docs/64-eval-harness.md`](../docs/64-eval-harness.md),
  updated [`README.md`](../README.md), [`docs/README.md`](../docs/README.md),
  and [`docs/sources.md`](../docs/sources.md); added explicit partial-status and
  linkage fields to [`fixtures/eval_harness.json`](../fixtures/eval_harness.json);
  added local-source identities to [`pins/eval_contract.json`](../pins/eval_contract.json);
  refreshed the stale `src/eval.cpp` identity in
  [`pins/sha256_acceleration_contract.json`](../pins/sha256_acceleration_contract.json);
  added EVAL-001 evidence and this record to
  [`implementation_ledger.md`](../implementation_ledger.md).
- Limitations: only request validation and logits evidence reading are present;
  checkpoint publication, CUDA trace capture, RTX 5090 smokes, and QLT-001
  quality acceptance remain unimplemented/unverified. Documentation makes no
  checkpoint, trace, or quality claim.
- UTC/time/tokens/cost: 148.487 s; 874,981 total tokens (867,933 input,
  798,720 cached input, 7,048 output, 908 reasoning output); cost unavailable

### Verification

- Attempt: 1
- Agent/model: independent integration verifier, `gpt-5.6-luna`, medium reasoning
- Diff review: performed against this dossier, `plan.md`, `implementation_ledger.md`, and the complete worktree diff. The implementation is explicitly logits/request-scaffolding only; checkpoint publication and CUDA trace capture are absent, so the all-three-modes and definition-of-done conditions are not met.
- Commands and exact outcomes:
  - `python -m json.tool pins/eval_contract.json >/dev/null && python -m json.tool fixtures/eval_harness.json >/dev/null` — passed (exit 0).
  - `uv run ruff format .` — passed (exit 0); `71 files left unchanged`.
  - `uv run ruff check .` — passed (exit 0); `All checks passed!`.
  - `uv run pytest -q tests/test_eval.py tests/test_build.py tests/test_trace.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` — failed (35 passed, 1 failed, 1 skipped). Failure: `tests/test_cuda_checkpoint.py::test_checkpoint_contract_fixture_and_handbook_are_connected`, stale expected SHA-256 for `src/eval.cpp` (`e0dbdc...` expected, `91710a...` actual).
  - `make diagnostic` — passed for the host diagnostic executable (exit 0).
  - `make clean` — passed (exit 0).
  - `make -j2` — passed (exit 0), including host `build/qw38-eval`.
  - `make diagnostic` after the clean build — passed (exit 0) for host products.
  - `uv run pytest -q` — failed (145 passed, 12 failed, 19 skipped). The 12 failures are stale local-source SHA-256 expectations in benchmark, CLI, CUDA atomic-eval, CUDA checkpoint, CUDA dispatch, CUDA fusion, CUDA graph, CUDA memory-fit, CUDA prefix-sync, CUDA prompt-scheduler, server, and chat-completions contracts; all resolve to changed `src/eval.cpp` (and the benchmark/CLI source identity) rather than runtime assertion failures.
  - `git diff --check` — passed (exit 0).
  - `build/qw38-eval --help` — exit 0; usage printed. `build/qw38-eval` with no arguments — exit 2. Malformed `--tokens '1,,2'` invocation — exit 2 with `invalid_argument: malformed evaluation request`.
  - `build/qw38-eval-diagnostic --help` — exit 0; usage printed.
  - `docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$(pwd):/workspace" qw38-cuda:13.0.2 make clean all diagnostic cuda-products cuda-native` — failed (exit 2) while linking `build/cuda/qw38-eval-diagnostic`; undefined references include `validate_trace_filter`, `emit_trace_tensor`, `execute_scalar_token_traced`, and `trace_filter_matches`. The diagnostic trace object is not included by the new CUDA link rule.
  - `QW38_RUN_CUDA_TESTS=1 uv run pytest -q tests/test_eval.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` — failed (1 failed, 7 passed). Same stale `src/eval.cpp` checkpoint-contract hash failure.
  - `nvidia-smi -L` — passed; repository GPU detected as NVIDIA GeForce RTX 5090.
  - `docker image inspect qw38-cuda:13.0.2 >/dev/null` — passed (exit 0).
  - `nm -C build/cuda/qw38-eval | rg 'execute_token_traced|trace_filter|emit_trace|layer_residual|final_norm'` — no matching symbols.
  - `strings build/cuda/qw38-eval | rg 'execute_token_traced|layer_residual|final_norm'` — failed isolation check: emitted `final_norm`, `final_normalized_f32_le_hex`, and `final_norm_values=` literals. (The release binary was produced by the preceding CUDA build before the diagnostic link failure.)
- Formatting changed files: none; Ruff reported all 71 files unchanged. No semantic fixes, ledger status changes, or `plan.md` changes were made.
- Failed acceptance items:
  - typed helpers driving all three modes; checkpoint mode is not implemented;
  - trace mode for all five filters and valid v1 trace publication; diagnostic CUDA product does not link;
  - authenticated checkpoint/restore equality and failure/no-publication cases;
  - normal release binary/object trace-literal isolation (`final_norm` remains in strings);
  - all focused and repository-wide tests (stale hash-contract failures).
- Verdict: **FAIL**. Verification cannot authorize delivery. Recovery requires implementing the dossier's checkpoint/trace paths, fixing the diagnostic CUDA link inputs and release trace-literal isolation, and synchronizing every affected source-hash contract before rerunning this attempt.
- UTC/time/tokens/cost: 306.062 s; 759,507 total tokens (754,892 input,
  700,416 cached input, 4,615 output, 759 reasoning output); cost unavailable.

### Retries and escalation

#### Repair retry 1

- Agent/model: implementation repair agent
- Findings repaired: added sequential public `Session::save`/`restore` checkpoint
  execution with byte-authenticated checkpoint and continuation-logit evidence;
  added strict typed checkpoint result validation and frozen trace-filter
  validation; added the missing diagnostic CUDA `full_scheduler` trace object
  link input; removed newly introduced release trace-name literals; and
  synchronized every affected local-source SHA-256 contract.
- Commands/outcomes: `uv run ruff format tools/qw38_eval.py` (passed);
  `uv run ruff check tools/qw38_eval.py` (passed); `make -j2 build/qw38-eval
  diagnostic` (passed); focused pytest
  (`tests/test_eval.py tests/test_build.py tests/test_trace.py
  tests/test_cuda_trace.py tests/test_cuda_checkpoint.py`) (36 passed, 1
  skipped); release `nm` trace-symbol check (no matches).
- Remaining verification: pinned CUDA build/hardware trace smoke and full
  repository suite are delegated to the independent verifier. The release
  binary still contains the pre-existing model tensor metadata string
  `final_norm`; no architecture change was made.
- UTC/time/tokens/cost: 167.628 s; 1,199,451 total tokens (1,192,201 input,
  1,129,216 cached input, 7,250 output, 1,803 reasoning output); cost unavailable.

### Final outcome

- Status: `blocked`
- Acceptance evidence: host formatting/build gates pass and the full host suite
  reached 157 passed/19 skipped on attempt 2, but final CUDA acceptance failed
  before hardware execution; see verification attempts 1–3 above.
- Commit: not created
- Push: not attempted
- First-pass acceptance: no
- Total elapsed/tokens/cost: 2,346.606 s and 12,179,455 tokens across planning,
  implementation, documentation, three verification attempts, two repairs, and
  one diagnostic; cost unavailable
- Remaining risk or recovery condition: repair the CUDA release compile rule to
  include `cuda/`, refresh every affected `src/eval.cpp` source-hash contract,
  then rerun all focused/repository gates, build both CUDA eval products, pass
  object-scoped trace isolation, execute logits/checkpoint/five-filter trace and
  negative-publication smokes on the RTX 5090, and promote
  `fixtures/eval_harness.json` from partial only after typed validation passes.

### Verification

- Attempt: 2 (fresh independent integration verification after repair retry 1)
- Agent/model: independent integration verifier, `gpt-5.6-luna`, medium reasoning
- Diff and acceptance review: the repaired host checkpoint path and typed checkpoint
  reader are present, but `run_high_level` still returns
  `unsupported_build` for every trace request, including the diagnostic build;
  no diagnostic trace bundle publication path is implemented. The record schemas
  also omit required model/tool identity, numeric summary, greedy-token, frontier,
  and checkpoint frontier metadata. `fixtures/eval_harness.json` explicitly remains
  `partial-logits-only` with all three smokes `smoke-not-run`, so it is not evidence
  of the all-mode definition of done.
- Commands and exact outcomes:
  - `python -m json.tool pins/eval_contract.json >/dev/null && python -m json.tool fixtures/eval_harness.json >/dev/null` — passed (exit 0).
  - `uv run ruff format .` — passed (exit 0); `71 files left unchanged`; no formatting changes.
  - `uv run ruff check .` — passed (exit 0); `All checks passed!`.
  - `git diff --check` — passed (exit 0).
  - `uv run pytest -q tests/test_eval.py tests/test_build.py tests/test_trace.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` — passed (36 passed, 1 skipped).
  - `make clean && make -j2 && make diagnostic` — passed (exit 0); host release, host diagnostic, and all host products built.
  - `uv run pytest -q` — passed (157 passed, 19 skipped in 158.22s).
  - `docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$(pwd):/workspace" qw38-cuda:13.0.2 make clean all diagnostic cuda-products cuda-native` — failed (exit 2) while linking `build/cuda/qw38-eval-diagnostic`; undefined references to `validate_trace_filter`, `emit_trace_tensor`, `execute_scalar_token_traced`, and `trace_filter_matches`. The CUDA diagnostic link still omits `build/diagnostic/diagnostic_trace.o` (or equivalent diagnostic trace object).
  - `QW38_RUN_CUDA_TESTS=1 uv run pytest -q tests/test_eval.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` — passed (8 passed; CUDA runtime smoke cases were unavailable/skipped by fixture/model conditions).
  - `nm -C build/qw38-eval | rg 'execute_token_traced|trace_filter|emit_trace|layer_residual|final_norm'` — no symbols matched (exit 0 via `|| true`).
  - `strings build/qw38-eval | rg 'execute_token_traced|layer_residual|final_norm'` — failed isolation gate: output contains `final_norm`.
  - `strings build/cuda/qw38-eval | rg 'execute_token_traced|layer_residual|final_norm'` — failed isolation gate: output contains `final_norm`.
- Formatting changes: none.
- Failed acceptance items:
  - diagnostic trace mode does not capture/publish valid `qw38.trace` v1 bundles or accept the five frozen filters;
  - CUDA diagnostic product does not link, so pinned CUDA trace build/smoke and full CUDA product gate fail;
  - required release trace-literal isolation fails because `final_norm` remains in release strings (host and CUDA);
  - all-mode authenticated schema requirements (identity, summaries, greedy token, frontiers, trace metadata) are not met;
  - RTX 5090 smokes and malformed/no-publication fixture evidence remain unrun/partial.
- Verdict: **FAIL**. Focused and repository-wide host tests pass, but the CUDA
  diagnostic build failure, absent trace implementation/evidence, and explicit
  release `final_norm` isolation failure prevent delivery authorization.
- UTC/time/tokens/cost: 293.572 s; 1,572,505 total tokens (1,568,402 input,
  1,497,344 cached input, 4,103 output, 635 reasoning output); cost unavailable.

### Architectural diagnosis after verification attempt 2

- Agent/role: architectural/diagnostic agent after two failed verification
  attempts.
- UTC/time/tokens/cost: 2026-09-03T11:39:54Z; 296.344 s; 1,746,310 total
  tokens (1,732,560 input, 1,634,816 cached input, 13,750 output,
  5,022 reasoning output); cost unavailable.
- Scope: inspected this complete dossier, the worktree diff, `plan.md`,
  `implementation_ledger.md`, the Make build graph, public engine boundary,
  internal scalar/CUDA trace APIs, frozen trace reader/schema, CUDA hardware
  tests, and both verifier records. No semantic source, plan, ledger-status,
  commit, or push change was made.
- Commands used:
  - `sed -n '1,520p' tasks/EVAL-001.md`; `git status --short`; `git diff`;
    `git diff --stat`.
  - `sed`/`rg` inspection of `plan.md`, `implementation_ledger.md`, `Makefile`,
    `src/eval.cpp`, `src/engine.cpp`, `include/qw38/engine.h`,
    `src/diagnostic_trace.{h,cpp}`, `cuda/full_scheduler.{h,cu}`, native CUDA
    tests, `tools/qw38_{eval,trace}.py`, `tests/test_eval.py`, and the eval/trace
    pins and fixtures.
  - No build or test was rerun: the two fresh verifier records already establish
    the failures, and this attempt is diagnostic-only.

#### Root causes

1. The first implementation stopped after request parsing, a partial logits
   writer, and partial Python readers. Repair retry 1 added checkpoint mechanics
   but did not implement the trace branch and did not complete the schemas. The
   tests mirrored that partial implementation, so green focused/host suites did
   not cover the dossier's all-mode acceptance conditions. This is a scope and
   acceptance-coverage failure, not an inference-runtime defect.
2. The CUDA diagnostic link substituted `full_scheduler.trace.cuda.o` for the
   release scheduler but continued to link the ordinary host library objects.
   It therefore omitted both owners needed by diagnostic `src/eval.cpp`:
   `build/diagnostic/diagnostic_trace.o` defines the backend-neutral filter/sink
   helpers, and `build/diagnostic/scalar_runtime.o` defines
   `execute_scalar_token_traced`. The latter reference is retained because the
   same eval object preserves legacy scalar diagnostic entry points. Adding the
   traced CUDA scheduler object alone could never satisfy either family. The
   correct graph substitutes diagnostic host library objects for ordinary host
   library objects while retaining the CUDA-compiled engine object.
3. The planned executable-wide literal check was invalid. `src/model.cpp`
   intentionally contains `final_norm` as the semantic role for the pinned
   `output_norm.weight`; every release executable linking `model.o` may therefore
   contain that string without containing a trace tap. The isolation boundary is
   the release compilation units that could carry instrumentation
   (`eval.cuda.o` and `full_scheduler.cuda.o`) plus absence of trace symbols from
   the final release binary. Checking those objects distinguishes forbidden tap
   literals from legitimate model metadata and preserves the original
   diagnostic-only intent.
4. The original trace prose requested an explicit `raw_token_ids` label and
   state digests, but frozen `qw38.trace` v1 has neither an input-label field nor
   a requirement that `state_sha256` be non-empty. The CUDA scheduler exposes
   frontiers and host token copies but no state hashing API; producing state
   digests would require new device copies/API work expressly excluded by this
   task. The correct existing-schema representation is empty prompt bytes,
   exact `prompt.token_ids`/`positions`, before/after frontiers, and empty state
   digest maps.
5. A multi-filter bundle cannot use the raw tap name as every tensor `name`:
   three `layer_residual` captures would violate the v1 reader's unique-name
   rule. The stable tap stays in `role`, `layer` stays numeric, and the unique
   tensor names are `layer.0.layer_residual`, `layer.3.layer_residual`,
   `layer.63.layer_residual`, `global.final_norm`, and `global.logits`.

#### Final repair plan (authoritative for the next implementation retry)

1. **Finish request validation before model/output work (`src/eval.cpp`).**
   Reject duplicate single-value options, options illegal for the selected mode,
   missing/non-empty requirements, duplicate or non-pinned trace filters, an
   existing destination, and trace on a non-diagnostic or non-CUDA build. Do not
   create a temporary directory until all request validation succeeds. Keep
   legacy option dispatch unchanged. Add one JSON-string escaping helper and use
   it for every caller-, path-, device-, or revision-derived string.

2. **Use one explicit common result contract (`src/eval.cpp`,
   `pins/eval_contract.json`, `tools/qw38_eval.py`).** The logits and checkpoint
   `result.json` top level must have exactly `schema`, `version`, `mode`,
   `status`, `model`, `tool`, `runtime`, plus the mode fields below:
   - `model`: exactly `name`, `revision`, `sha256`, `byte_count`; values are the
     pinned model name, revision `0669b98607d47046c7c2b3f801011d54a08cfccf`,
     SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`,
     and 18,973,870,432 bytes.
   - `tool`: exactly `name`, `revision`, `source_state`, `sha256`; hash the
     resolved `/proc/self/exe`, use the requested source revision/state, and use
     `qw38-eval` as the release product name.
   - `runtime`: exactly `backend`, `cuda_target`, `cuda_runtime_version`,
     `cuda_driver_version`, `device_name`, `compute_capability`; CUDA builds must
     query the CUDA runtime/driver and active-device properties and record
     `backend="cuda"`, `cuda_target="sm_120"`, and capability `12.0`. A query
     failure is a run failure with no publication.
   - Each blob record is exactly `file`, `dtype`, `shape`, `byte_count`,
     `sha256`, `summary`; `summary` is exactly `count`, `finite_count`,
     `nan_count`, `positive_infinity_count`, `negative_infinity_count`,
     `minimum`, `maximum`, `mean`, `root_mean_square`. Native output requires all
     values finite; the Python reader recomputes counts/min/max and checks
     mean/RMS to a documented floating serialization tolerance.
   - Logits mode adds exactly `tokens`, `positions`, `frontier`, `logits`, and
     `greedy_token`. Positions are `[0..len(tokens)-1]`, frontier is token count,
     logits is `logits.f32le.bin` with shape `[248320]`, and greedy selection uses
     maximum value then lower token ID.
   - Checkpoint mode adds exactly `prefix_tokens`, `continuation_tokens`,
     `prefix_positions`, `continuation_positions`, `frontiers`, `checkpoint`,
     `continuation_logits`, `greedy_token`, and `equality`. `frontiers` is exactly
     `prefix`, `uninterrupted_final`, `restored_prefix`, `restored_final`;
     `checkpoint` is exactly `file`, `byte_count`, `sha256`; `equality` is
     exactly `tokens` and `logits`, both true. The continuation-logit blob has
     shape `[248320]`. Compare logits by bytes (`memcmp`/blob bytes), not
     element-wise `vector<float>::operator==`, so NaN payloads and signed zero
     cannot evade the exact claim (non-finite values still fail separately).
   - Update the frozen Python result dataclasses/readers and fake-native tests to
     enforce every exact key, identity, range, file name, size, digest, frontier,
     position, summary, greedy, and equality relationship. `run_native` must
     return the corresponding typed result after a successful subprocess, not
     merely `CompletedProcess`; process failure remains `EvalProcessError` and
     malformed evidence remains `EvalError`.

3. **Complete sequential checkpoint execution (`src/eval.cpp`).** Continue to
   use only public `Engine`/`Session`. Sync prefix, record prefix frontier from
   `tokens()`, save the checkpoint into the private temporary result directory,
   execute continuation, copy tokens/logits, destroy the first session, create a
   second session, restore, verify restored prefix tokens/frontier, execute the
   same continuation, and byte-compare complete tokens and logits. Never hold
   two sessions concurrently. Remove the no-op append-open of the checkpoint.

4. **Implement diagnostic CUDA trace directly on existing internal APIs
   (`src/eval.cpp`).** Compile this code only when both
   `QW38_CUDA_RUNTIME` and `QW38_DIAGNOSTIC_TRACE` are defined and include
   `cuda/full_scheduler.h` through `-Icuda`. Authenticate the pinned file exactly
   as `Engine::open` does, then use `inspect_gguf`, `validate_qwen38_contract`,
   `MappedFile::open`, `bind_model_weights`, `ResidentModel::upload`,
   `SchedulerSession::create(tokens.size())`, and
   `SchedulerWorkspace::create(tokens.size())`. Do not expose or alter
   `Engine::Impl`/`Session::Impl` and do not add a public trace API.
   For each requested filter in request order, create/reset a zero-frontier
   scheduler session, run every prefix token with `execute_token(...,
   PointwisePath::kUnfused, nullptr)`, then run the final token with
   `execute_token_traced`. The sink copies its borrowed view immediately and
   rejects a second callback. Require the captured raw name/layer/shape to match
   the five frozen filters exactly, final frontier to equal token count, copied
   committed tokens to equal the request, and complete final logits to be
   byte-identical across the five fresh runs. No graphs and no fused pointwise
   path participate.

5. **Publish a native `qw38.trace` v1 bundle without changing the frozen trace
   contract (`src/eval.cpp`, `tools/qw38_eval.py`).** Write one tensor per filter
   in request order. Use the unique names and raw tap roles listed in root cause
   5, layers `0`, `3`, `63`, `null`, `null`, and shapes `[5120]`, `[5120]`,
   `[5120]`, `[5120]`, `[248320]`. The manifest model identity uses the pinned
   name/revision/hash; tool identity uses name `qw38-eval-diagnostic`, requested
   revision, and `/proc/self/exe` hash. `prompt` contains canonical empty base64,
   zero bytes and its SHA-256, plus exact raw token IDs and positions. Session
   before/after records use frontiers 0/token-count and `{}` state hashes. The
   blob is contiguous little-endian FP32 with per-tensor and whole-blob hashes
   and summaries. Validate the completed temporary bundle with the unchanged
   `read_trace_bundle` through the typed Python result path.

6. **Repair publication for every mode (`src/eval.cpp`).** Use a same-parent
   uniquely named temporary directory, truncating binary/text writes, checked
   flush/close, and Linux `renameat2(..., RENAME_NOREPLACE)` (or an equivalent
   no-replace atomic primitive) to commit only an absent destination. On every
   parse/runtime/sink/write/hash/rename failure, remove only that exact temporary
   directory and leave the requested destination absent. Unit tests may force a
   native write failure with an invalid/unwritable parent; no new production
   failure-injection option is needed.

7. **Fix the Make object graph exactly (`Makefile`).** Keep release
   `build/cuda/qw38-eval` linked to `build/eval.cuda.o` and
   `build/full_scheduler.cuda.o`. Build `build/eval.trace.cuda.o` with
   `-DQW38_CUDA_RUNTIME -DQW38_DIAGNOSTIC_TRACE -Icuda` and dependencies on
   `src/diagnostic_trace.h` and `cuda/full_scheduler.h`. Link
   `build/cuda/qw38-eval-diagnostic` with a new object set equivalent to
   `$(filter-out $(DIAGNOSTIC_DIR)/engine.o,$(DIAGNOSTIC_LIB_OBJECTS))
   $(CUDA_BUILD_DIR)/engine.o $(THIRD_PARTY_OBJECTS)`, then
   `build/eval.trace.cuda.o`, `build/full_scheduler.trace.cuda.o`, checkpoint,
   and the same CUDA primitive objects as the release product. This supplies
   both diagnostic trace and scalar-runtime definitions without duplicate
   ordinary/diagnostic host symbols and keeps the CUDA-enabled engine. Do not
   link ordinary `$(LIB_OBJECTS)` or `build/full_scheduler.cuda.o` into the
   diagnostic product.

8. **Replace the invalid isolation test (`tests/test_eval.py`).** In the pinned
   CUDA container, build both eval products and assert:
   - `nm -C --defined-only build/eval.cuda.o build/full_scheduler.cuda.o` and
     `nm -C build/cuda/qw38-eval` have no exact trace API/helper symbols
     (`execute_token_traced`, `validate_trace_filter`, `trace_filter_matches`,
     `emit_trace_tensor`, or `TraceSink`).
   - `strings -a build/eval.cuda.o build/full_scheduler.cuda.o | rg -x
     'layer_residual|final_norm'` has no matches. Scope the string check to these
     two release instrumentation-bearing objects. Do not treat the exact
     `logits` string as forbidden in `eval.cuda.o`: it is the ordinary product
     mode name as well as a trace role, so trace-symbol absence is the meaningful
     isolation proof for that overlap.
   - The corresponding diagnostic objects do contain `execute_token_traced`,
     trace helpers, and the exact admitted tap literals.
   - `strings -a build/model.o | rg -x 'final_norm'` does match, documenting why
     the linked release executable is allowed to contain legitimate tensor-role
     metadata. Do not mutate runtime strings to game this test.

9. **Add real acceptance and hardware evidence (`tests/test_eval.py`,
   `fixtures/eval_harness.json`).** Under `QW38_RUN_CUDA_TESTS=1`, require the
   pinned GGUF (skip only when it is genuinely absent), run the CUDA release
   binary for logits and checkpoint and the CUDA diagnostic binary once with all
   five filters, then validate outputs only through typed helpers. Run malformed,
   duplicate-filter, normal-build trace, existing-destination, and publication
   failure cases and assert no new final output. The fixture must change from
   `partial-logits-only`/`smoke-not-run` to `complete` only after an actual RTX
   5090 run and retain: GPU name, compute capability, CUDA toolkit/runtime/driver,
   source revision/state, model and both binary hashes, exact command cases,
   per-mode artifact byte counts/hashes/frontiers/greedy IDs, all five trace
   tensor identities, negative-case exit/no-publication results, and an explicit
   `harness_wiring_only=true`/`quality_admission=false` boundary. Large generated
   checkpoint/logit/trace blobs need not be committed; their authenticated
   summaries and hashes are the retained fixture evidence.

10. **Synchronize evidence only after semantic completion.** Update
    `pins/eval_contract.json` to encode the exact schemas and object-isolation
    boundary above; update the handbook and indexes/sources to match actual
    behavior; refresh all local-source hash contracts affected by final source
    bytes once, at the end; then run the focused, repository-wide, CUDA build,
    hardware, isolation, JSON, Ruff, and `git diff --check` gates already listed.
    Mark EVAL-001 done only after the independent verifier sees the completed
    RTX 5090 fixture and all gates pass.

- Exact semantic files remaining: `Makefile`, `src/eval.cpp`,
  `tools/qw38_eval.py`, `tests/test_eval.py`, `pins/eval_contract.json`, and
  `fixtures/eval_harness.json`; then the already-listed documentation/index,
  local-source hash contracts, and EVAL-001-only ledger evidence/status. No new
  source file is required.
- Unresolved decisions: `none`.
- Architecture impact: `none`; the repair uses the admitted public engine
  boundary for product modes and already-existing diagnostic scheduler boundary
  for tracing.
- Discovered/new ledger work: `none`; all missing work is required acceptance
  scope of EVAL-001, not a new task.

#### Repair retry 2 (final implementation repair agent)

- Changes: corrected the CUDA diagnostic eval object graph to use diagnostic
  host objects plus the CUDA engine and traced scheduler; added diagnostic CUDA
  trace capture/publication through the existing scheduler APIs; tightened
  duplicate option/filter validation and removed checkpoint append-open; expanded
  typed Python result readers to require model/tool/runtime identities, exact
  blob records, summaries, positions, frontiers, greedy IDs, and checkpoint
  equality; synchronized the accelerated-eval source hash.
- Commands/outcomes: `make -j2 build/qw38-eval diagnostic` — passed; `uv run
  ruff format tools/qw38_eval.py` — passed; `uv run ruff check .` — passed;
  `make -n build/cuda/qw38-eval-diagnostic` — shows the required diagnostic
  object set, traced eval object, and traced scheduler; focused pytest — 16
  passed, 1 failed only until the final source hash was refreshed, then pending
  rerun.
- Hardware/evidence: no CUDA container or RTX 5090 run was available in this
  repair turn; fixture remains unpromoted until that required independent gate.
- Remaining risks: native CUDA compile must validate the new trace branch and
  complete v1 manifest against `read_trace_bundle`; checkpoint native output
  still requires final independent schema/hardware validation; fixture and
  documentation evidence must be completed only after a real RTX 5090 run.
- UTC/time/tokens/cost: 315.717 s; 2,270,410 total tokens (2,256,195 input,
  2,171,136 cached input, 14,215 output, 1,480 reasoning output); cost unavailable.

### Verification

- Attempt: 3 (fresh independent final verification after repair retry 2)
- Agent/model: independent final verifier, Codex; UTC 2026-09-03
- Dossier/plan/ledger/diff review: performed before execution. No semantic
  source changes, plan changes, status changes, commits, or delivery actions
  were made. `uv run ruff format .` reported `71 files left unchanged`.
- Commands and exact outcomes:
  - `python -m json.tool pins/eval_contract.json >/dev/null && python -m json.tool fixtures/eval_harness.json >/dev/null` — passed (exit 0).
  - `uv run ruff format .` — passed (exit 0); no formatting changes.
  - `uv run ruff check .` — passed (exit 0); `All checks passed!`.
  - `git diff --check` — passed (exit 0).
  - `uv run pytest -q tests/test_eval.py tests/test_build.py tests/test_trace.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` — failed (35 passed, 1 failed, 1 skipped). `tests/test_cuda_checkpoint.py::test_checkpoint_contract_fixture_and_handbook_are_connected` found the amended `src/eval.cpp` digest `5a252fe2ca798af9fcfc84d1b624c4413b22516640c171e366bbb0b87e2dd2d0` but contract value `91710a551fd8d382b93b29cc7da29d38c76bb9819cc70bd0725c899506a09179`.
  - `make clean` — passed (exit 0); `make -j2` — passed (exit 0); `make diagnostic` — passed (exit 0, nothing to do).
  - `uv run pytest -q` — failed (145 passed, 19 skipped, 12 failed). Every failure was a stale local-source SHA-256 expectation for the amended `src/eval.cpp` in existing benchmark, CLI, CUDA atomic-eval, CUDA checkpoint, CUDA dispatch, CUDA fusion, CUDA graph, CUDA memory-fit, CUDA prefix-sync, CUDA prompt-scheduler, server, and chat-completions contracts.
  - `docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$(pwd):/workspace" qw38-cuda:13.0.2 make clean all diagnostic cuda-products cuda-native` — failed (exit 2). CUDA 13.0.2 compilation stops at `src/eval.cpp:38:10: fatal error: full_scheduler.h: No such file or directory` while building `build/eval.cuda.o`; the release CUDA rule at `Makefile:88` omits `-Icuda`. No CUDA eval products were produced by this gate.
  - `QW38_RUN_CUDA_TESTS=1 uv run pytest -q tests/test_eval.py tests/test_cuda_trace.py tests/test_cuda_checkpoint.py` — failed (1 failed, 7 passed); the same stale `src/eval.cpp` hash failure. No hardware mode smoke ran.
  - `nvidia-smi -L` — passed; repository GPU detected as NVIDIA GeForce RTX 5090.
  - Native CLI checks: `./build/qw38-eval --help` — passed (exit 0); `./build/qw38-eval` — exit 2; normal-build trace request — exit 2 (`invalid_argument: malformed evaluation request`); normal-build logits request — exit 1 (`session creation has not passed its delivery gate`).
  - Object-scoped isolation commands — release objects/binary were absent after the failed CUDA build, so release checks failed to execute (`nm`/`strings`: `No such file or directory`). Diagnostic object checks found `validate_trace_filter`, `trace_filter_matches`, and `emit_trace_tensor`, and admitted `layer_residual`/`final_norm`; `strings -a build/model.o | rg -x 'final_norm'` matched as expected. The required `build/full_scheduler.trace.cuda.o` was absent.
- Formatting/generated evidence changes: none. `fixtures/eval_harness.json` remains
  `partial-logits-only` with all three modes `smoke-not-run`; no evidence was
  generated because the CUDA build failed before hardware execution.
- Failed acceptance items: focused and repository-wide suites are not green due
  stale source-hash contracts; the pinned CUDA container build fails; RTX 5090
  logits/checkpoint/trace smokes and all-mode schema/bundle validation did not
  run; negative publication and complete object-scoped release isolation gates
  cannot pass without the missing CUDA products; fixture evidence is not
  promoted; checkpoint/trace delivery remains unverified.
- Verdict: **FAIL**. EVAL-001 is not delivery-ready and cannot be authorized.
- UTC/time/tokens/cost: 396.928 s; 1,365,035 total tokens (1,359,957 input,
  1,284,864 cached input, 5,078 output, 944 reasoning output); cost unavailable.
