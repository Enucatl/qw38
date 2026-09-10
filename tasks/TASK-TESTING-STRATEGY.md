# Task testing strategy

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
- `acceptance` is the only tier allowed to make a performance keep/reject
  decision. Its work is explicitly enumerated as O(C_survivors × S_production
  × (W + N)) for component A/B, plus conditional P/D work for the selected
  survivor. Full historical sample counts are retained here: three warmups and
  30 measured samples where the frozen protocol requires them. A candidate
  that fails screening must not receive an expensive P/D sitting.

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

## Review checklist

- Missing tier fails before GPU work.
- The smoke command is demonstrably small and does not load production data.
- Correctness has bounded candidate/case/sample products and no P/D.
- Acceptance runs full samples only after screening and gates P/D on a winner.
- Large host references are sampled and reuse decoded invariant data.
- Build and binary compilation are not duplicated for argument-only runs.
- Evidence contains phase telemetry and an explicit proof boundary.
