# EVAL-002 — Tier and instrument task-specific CUDA oracles

Status: pending. Dependencies: EVAL-001, BEN-001. Coupled IDs: none.

## Objective

Make every task-specific CUDA pytest/native oracle obey the same explicit
execution contract as the shared quantization and scheduler gates. An agent
must be able to run a fast implementation loop without accidentally launching
a model-scale acceptance sitting, while the final acceptance protocol remains
unchanged and auditable.

## Design

1. Require `QW38_CUDA_TEST_TIER` for every GPU oracle. Accept exactly:
   - `smoke`: compile/link checks and small or boundary probes; one warmup and
     at most three measured samples; no model-scale P/D sitting.
   - `correctness`: complete numerical/structural candidate coverage with
     reduced sampling; no end-to-end P/D performance sitting.
   - `acceptance`: the frozen task protocol, including its full sample counts,
     quality envelopes, and conditional end-to-end guards.
   Missing or invalid values fail closed with the required tier in the error;
   they must not silently select an expensive default.
2. Keep the tier semantics in `tests/README.md`,
   `tasks/PERFORMANCE-RECOVERY-2026-09-10.md`, and each task dossier. Remove
   the exception that lets task-specific diagnostics retain an implicit,
   un-tiered protocol. A task may explicitly declare itself release-only, but
   that declaration must still fail closed unless selected deliberately.
3. Add machine-readable phase telemetry to task-specific runners and native
   diagnostics: epoch start/end, elapsed milliseconds, and success/failure for
   build, compilation, model load/upload, host references, A/B measurement,
   prefill, each decode prefix, and evidence serialization. Do not include
   instrumentation in kernel throughput samples.
4. Cache the Docker build and compiled binaries within one pytest process when
   inputs are unchanged. A single binary must be reused for multiple decode
   prefixes or other argument-only variants.
5. Preserve the existing acceptance contracts, numerical budgets, p95 rules,
   provenance, and rejection behavior. Tiered runs may not produce performance
   claims unless the acceptance tier has passed.

### Mandatory paragraph for every subsequent task dossier

Every new task that adds or changes a test must include a **Test execution
contract** paragraph in its own `tasks/TASK-ID.md`. The paragraph must name the
fast implementation-loop command (`smoke`), the reduced numerical command
(`correctness`), and the final evidence command (`acceptance`); state which
phases, sample counts, model loads, and P/D oracles each tier runs; require
`QW38_CUDA_TEST_TIER` and fail closed when it is missing; identify cached build
or binary reuse; and state that only acceptance-tier results may support a
performance claim. If a task has an optional deep or release-only run, it must
be explicitly labeled and never selected by a default or an unspecified tier.

## Acceptance

- Omitting `QW38_CUDA_TEST_TIER` from every task-specific GPU entry point exits
  nonzero before GPU work and prints an actionable error.
- Smoke and correctness runs demonstrably skip model-scale acceptance phases;
  acceptance runs execute the historical protocol without reduced sample
  counts.
- Repeated prefixes reuse one compiled oracle binary and one cached build per
  process where source and flags match.
- JSON/stdout evidence contains the phase timing schema, UTC/epoch identity,
  and failure phase; a timing report is retained beside the task evidence.
- Existing shared-gate and task-specific contract tests pass, and no
  production CUDA dispatch or numerical envelope changes are introduced.
- The repository contains the canonical paragraph above, and every subsequent
  task dossier that adds or changes tests repeats the paragraph (with its
  task-specific phase details) before implementation begins.

## Proof boundary

Fast tiers establish build and correctness feedback only. They are not
throughput evidence. Acceptance remains the sole tier allowed to record a
performance keep/reject decision; optional deep/release reruns are separate
evidence and must be labeled as such.
