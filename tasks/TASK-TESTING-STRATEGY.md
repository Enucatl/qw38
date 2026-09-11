# Task testing strategy

Reviewed 2026-09-11 for OPT-057–069. The five-minute goal is time to a useful
implementation decision on a prepared machine, including incremental build,
not a promise that full release validation fits five minutes. OPT-057 implements
the missing support. Historical evidence and frozen acceptance protocols retain
their original workload/sample counts. New batch acceptance is specified in
[the batch protocol](PERFORMANCE-RECOVERY-2026-09-11.md).

This is the shared test-authoring contract for every recovery task that adds or
changes a pytest test, native diagnostic, CUDA A/B binary, or end-to-end oracle.
The shared performance protocol references this file, so task dossiers inherit
the contract through that protocol. They should describe only task-specific
cases and must not copy a second generic tier policy into each `tasks/OPT-*.md`
file.

## Required tiers

Every GPU test must consume `QW38_CUDA_TEST_TIER` and fail closed with an
actionable error when it is missing or invalid. The tiers are workload classes,
not informal labels:

- `smoke` is fixed-size compile/link and small or boundary coverage. Its work
  is O(1) in model dimensions: no full model load, production-sized matrix,
  candidate sweep, long prompt, or P/D oracle. Use at most one warmup and one
  measured sample per selected probe.
- `correctness` covers all required numerical/layout candidates on bounded
  probes. Its work is O(C × S × R), where C is the explicitly listed candidate
  count, S is the fixed small/boundary case count, and R is at most three
  measured repetitions. Large matrices use deterministic sampled rows (at most
  32 output rows and 32 prompt rows) rather than a full host reference. It
  never launches model-scale P/D performance oracles.
- `screen` is a new, explicitly implemented tier for bounded production-shaped
  component measurements and short engine feedback. It may load the model once,
  use actual production weights, and time at most four listed candidates with
  one warmup and three paired samples on at most three listed shape groups.
  Host reference work is sampled as in correctness. A dossier may narrow this
  further. Optional engine screening is one control/candidate pair, one prefix,
  at most 32 decode tokens or one P4096 prompt. Results are `screened_in` or
  `screened_out`; neither is a production keep or a throughput claim. Existing
  binaries must reject this tier until they actually implement its workload.
  Instrumentation/quality-only dossiers may instead enumerate a fixed set of
  diagnostic modes or functional cases, as OPT-058/060 do. Those runs still
  share the 300-second phase deadline and cannot make performance admission
  claims; they are not an exception for larger optimization candidate sweeps.
- `acceptance` is the only tier allowed to make a performance keep/reject
  decision. Its work is explicitly enumerated as O(C_survivors × S_production
  × (W + N)) for component A/B, plus conditional P/D work for the selected
  survivor. Full historical sample counts are retained here: three warmups and
  30 measured samples where the frozen protocol requires them. A candidate
  that fails screening must not receive an expensive P/D sitting.

For OPT-057–069, the versioned batch protocol permits 10 paired component rounds
and a short target non-regression gate, then one combined release sitting.
Do not feed 10-sample results to historical validators requiring 30. The existing
`cuda/decode_oracle_test.cu` hard-codes 33 runs and is a release workload, even if
its caller sets the environment to smoke. An environment variable is not proof
that a binary respects a tier. Release workloads require an explicit runner
mode; implement `--mode release` separately from the new tier selection.

An optional `deep` or `release` run may use full references or additional
replicates, but it must be a separately invoked command and never a default or
an unspecified-tier fallback. Its larger complexity, purpose, and proof limit
must be recorded beside the evidence.

## Complexity rules for test authors

1. Before implementation, enumerate every loop multiplier: candidates, shapes,
   rows, prompt rows, prefixes, warmups, samples, replicates, and model tokens.
   The product must be visible in the task's test plan. Do not hide a Cartesian
   sweep behind a helper or a default argument.
2. Keep implementation-loop complexity bounded by the tier. If a proposed test
   does not fit its tier, reduce or split the workload, promote only the
   necessary survivor to acceptance, or make it an explicit deep/release run.
   A fast GPU on one sitting does not justify a needlessly superlinear host
   reference or duplicated model execution.
3. Reuse invariant work: one cached Docker build per pytest process, one compile
   per source/flag set, one decoded weight row for multiple activation
   references, and one binary for argument-only prefix variants. Record cache
   hits/misses in phase telemetry.
4. Use deterministic sampling for large host references: first/last/tail rows
   plus a contract-seeded pseudo-random set. Keep small and boundary cases
   complete, report sampled/full point counts, and document that sampling does
   not prove every production row. Preserve one optional deep full-reference
   run for release validation when that proof is required.
5. Emit machine-readable phase telemetry (epoch start/end, elapsed duration,
   counts, and success/failure) for build, compilation, model load, host
   reference, A/B, P, each decode prefix, and evidence writing. Watchdogs may
   terminate a hung subprocess, but fixed seconds are not a substitute for
   declaring and reviewing workload complexity.
6. Only `acceptance` evidence may support a throughput claim. Smoke and
   correctness results are feedback gates; they must be labeled as such in
   stdout, fixtures, and reports.

## Five-minute implementation loop

Budget on a prepared image/model/capture cache: incremental build <=60 seconds,
smoke plus numerical probes <=60 seconds, production screening <=120 seconds,
report/decision <=15 seconds, 45 seconds spare. These are targets to measure,
not silently fabricated pass conditions. Use one 300-second parent deadline
for the whole feedback command, including compilation and subprocess teardown.
Report `budget_exhausted` with the slow phase and partial results if exceeded;
do not relabel timeout as numerical rejection or launch the same work again.
Terminate the owned container/process group, not only its parent CLI.

Prepare a cold image, compiler cache and model/capture authentication once,
outside candidate iteration. Report cold setup time separately and include any
cache misses in the actual iteration total. Authentication cache keys bind the
verified artifact to its identity; never skip verification on a changed file.
Model-scale timing excludes load/hash/capture time but telemetry includes them.

Each new task dossier must give concrete candidate and shape lists, reference
point counts, one build target, the feedback command, conditional acceptance,
and a stop rule. “Run the production suite” is insufficient. Build only changed
translation units with transitive header dependencies and flag invalidation.
Do not call `make clean`, rebuild Docker, compile once per pytest function, or
repeat full model loading for each sampled row. CUDA workloads run serially on
the one GPU; independent host reads/build preparation may be parallel.

Reference recipe: generate fixed small weights with **finite** FP16 scale bits,
and distinct deterministic activations. Keep small edge cases complete. For
production K, select at most 16 output rows and 8 activation rows by default
(absolute maximum 32×32), decode each selected weight row once, and reuse it
across vectors. Compute FP64 host dot products over full K. The reference must
not call the candidate's unpack/scale helper. Test original BF16 and the exact
staged values separately. Report exact counts of decoded rows and dot products.
Do not allocate or fill an entire host reference output merely to sample it.
GPU execution at full output size is permitted in screen/acceptance and must
retain full finite checks plus deterministic sampled numerical comparisons.

Correctness should check behavior: unpacking, numeric outputs, dispatch actually
executed, boundaries, rollback and stale-cache detection. A collection of string
searches for kernel names or selectors does not establish those properties.
Test historical pins against their historical records; test current selections
against the current admission record. Read-only evidence validation must never
run a GPU sitting or overwrite evidence as a side effect of normal pytest.

## Acceptance without repeated long sittings

Fail layout/nonfinite/reference checks first; then screen; then at most one
survivor receives complete component acceptance. Reuse deterministic real
captures, token IDs, pinned llama reference outputs and already-built binaries.
Cache numerical outputs by model, tokens, staging semantics, reference revision
and flags. Cache timing only within a controlled sitting, never across a driver,
power, kernel, graph or workload change.

Keep timing regimes explicit: hot repeated matrix, rotating production weights,
instrumented full engine, and uninstrumented full engine. Do not subtract unlike
regimes or double-count fused leaves. Restore state outside the timed interval,
then run identical tokens and graph choices. Alternate whole-run A/B order.
Count independently restored rounds as statistical observations; correlated
tokens within a run are not independent replicates. Report both estimates and
uncertainty. Do not increase sample counts repeatedly until a desired answer.

New arithmetic needs production quality before a default changes. A small
teacher-forced screening span is a fast alarm, not a substitute for the two
1024-target quality spans and recurrence checks. Exact-output changes can reuse
quality evidence only for the same admitted arithmetic and verified outputs;
the combined configuration is checked again in OPT-069. Long-context and
128K allocation tests are conditional on state/storage/graph changes during
iteration and are included at the batch release boundary.

## Review checklist

- Missing tier fails before GPU work.
- The smoke command is demonstrably small and does not load production data.
- Correctness has bounded candidate/case/sample products and no P/D.
- Acceptance runs full samples only after screening and gates P/D on a winner.
- Screen output cannot be promoted to acceptance or release by editing a label.
- A 300-second feedback deadline covers all child work; cold setup is explicit.
- Historical oracle calls occur only in explicit release mode.
- Read-only evidence tests never modify a previous task's measurements.
- Large host references are sampled and reuse decoded invariant data.
- Build and binary compilation are not duplicated for argument-only runs.
- Evidence contains phase telemetry and an explicit proof boundary.
