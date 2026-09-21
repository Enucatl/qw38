# TASK-023 — Reproducible V0 performance baseline

## Status
TODO
## Milestone
M9 — End-to-end performance baseline
## Purpose
Create identity-matched reproducible measurements against which every architecture experiment is judged.
## Depends on
- TASK-022
## Normative references
- `docs/architecture/architecture-v0.md` — Behavior and performance validation
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| I-01–I-05 | Reference environment/hardware | LOCKED |
| T-01–T-03 | Baseline geometries being measured | TUNING |
| Initial validation policy | Measurement windows and 5% benefit heuristic | TUNING |
## Starting point
Behavior-accepted complete decode and production prefill exist.
## Scope
Implement/run separate benchmark harnesses for batch-one decode at populated lengths 512/4096/32768, prefill 256/4096/32768, and prefill+128-token request; measure memory footprint/capacity and selected region/kernel traffic, spills, occupancy, and timings. Exercise maximum feasible context separately. Capture container image/digest, source revision/dirty state, binary/artifact/config/tokenizer hashes, GPU, driver/runtime/toolchain, clocks/power conditions, context capacity/populated length, graph mode, output policy, warmups/repetitions, state restore, uncertainty, median and p99. Separate setup/upload/warmup/restore and TTFT from steady state.
## Out of scope
Architecture changes, tuning in response to results, parent+child time double counting, stale binaries, incomparable llama claims.
## Required interfaces
Benchmark CLI emits machine-readable raw samples and summary with complete identity; correctness tests remain separate.
## Required semantics
Warm up five runs; at least 20 timed repetitions from identical restored input state. Report missing coverage, never as zero. End-to-end identities and kernel regions are disjoint/accounted.
## Data representation
Versioned result records with units, timing boundaries, raw repetitions, memory categories, hashes, and profiler command/config.
## Implementation constraints
Rebuild current source; prove artifact/binary identity. Profiling perturbation runs are distinct from timing runs.
## Tuning defaults
Declared cases/repetitions above.
## Expected files/modules
Benchmark harness/CLI, result schema and baseline records/documentation.
## Tests required
### Unit tests
Statistics, identity completeness, state restoration, disjoint accounting, unit conversions.
### Reference/numerical tests
Behavior hash/logit smoke before and after timing.
### Integration tests
Run every feasible declared case; explicitly record capacity-limited omissions.
## Benchmark required
Yes: all declared cases and selected kernel metrics.
## Acceptance criteria
- [ ] Every result is bound to complete environment/source/binary/artifact identity.
- [ ] Declared windows use ≥5 warmups and ≥20 restored repetitions with median/p99/uncertainty.
- [ ] Decode, prefill, request, TTFT, setup, and memory identities are not conflated.
- [ ] Profiles explain coverage without double counting.
- [ ] Behavioral gate still passes.
## Architecture blocker rule
Measurement difficulty is not architectural. If a locked contract prevents measurable correct execution, report the full blocker; do not tune/redesign here.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
All identities, raw-record paths, summaries, omissions.
### Architecture blocker
None/full report.
### Follow-up observations
Measured bottlenecks only; no redesign.
