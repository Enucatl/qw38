# TASK-015 — Segmented online decode attention

## Status
DONE
## Milestone
M5 — Attention execution
## Purpose
Implement bounded-memory causal GQA decode attention with deterministic segment merging and complete mixer output.
## Depends on
- TASK-014
## Normative references
- `docs/architecture/architecture-v0.md` — Full-attention sequence; initial attention ownership
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| P-01–P-02, G-01, M-01 | FP32 attention math, BF16 outputs/cache, online materialization | LOCKED |
| T-03 | 128 threads, 256-key segment, 32-key subtile | TUNING |
## Starting point
Prepared Q/g and correctly appended K/V exist; packed Q4 output projection exists.
## Scope
Implement an independent online-attention reference, per-query-head/per-256-key-segment FP32 online softmax scan, FP32 partial max/sum/256-value numerator, fixed-order merge, normalization, FP32 sigmoid(g), BF16 gated output, and Q4 output projection with direct FP32 residual add.
## Out of scope
Quadratic scores/probabilities, nondeterministic/atomic merge, prefill attention, cache paging, fused projection/preparation/scan, alternate GQA storage.
## Required interfaces
Attention-core plan binds Q/g/cache, populated length, partial workspace, output projection/residual; explicit scan and merge launch wrappers.
## Required semantics
Scores/softmax/value accumulation FP32 with causal range `[0,populated)`. Standard max-rescaled segment and merge equations; segments merge increasing index; empty/tail keys mask; six Q heads address each KV head; sigmoid gate precedes BF16 store.
## Data representation
Partial per segment/query head: FP32 max, sum, numerator `[256]`; shared staging reuses at most one K or V subtile, never both/full segment; no score vector in global memory.
## Implementation constraints
Initial scan and merge remain separate; one ordered stream; deterministic results for fixed inputs.
## Tuning defaults
128 threads; 256 keys/segment; 32-key subtiles.
## Expected files/modules
Reference online attention, CUDA scan/merge, full attention mixer executor/tests.
## Tests required
### Unit tests
Lengths 1,31,32,255,256,257; tail/empty segments; extreme logits; GQA mapping; causal exclusion; gate extremes.
### Reference/numerical tests
Segmented versus unsegmented stable FP32 reference and full mixer residual comparison.
### Integration tests
Repeated cache append+attention; snapshot/restore; deterministic repeated runs at several lengths.
## Benchmark required
Diagnostic timings at populated lengths 512,4096,32768 where memory permits.
## Acceptance criteria
- [ ] Online result meets reference tolerance across boundaries.
- [ ] Merge order is fixed and no quadratic global scores exist.
- [ ] Full attention mixer residual and cache continuation pass.
- [ ] Timings are diagnostic and do not alter architecture.
## Architecture blocker rule
On locked conflict stop with full blocker report; tuning limits do not authorize semantic/layout changes.
## Completion report
### Result
DONE. Independent verification PASS (fresh Debug ctest 38/38; fresh Release
attention ctest 4/4; RTX 5090 `sm_120` diagnostic benchmark at populated
lengths 512, 4096, and 32768 with 5 warmups and 20 repetitions).
### Changes made
- Corrected `tests/attention_core_unit_test.cpp` gate oracle to evaluate FP32
  sigmoid at the BF16 `+80/-80` gate inputs, preserving nonzero negative-tail
  coverage instead of hardcoding sigmoid(-80) to zero.
- Added a two-segment (`populated=257`) CUDA core scenario with nonuniform
  logits spanning `[-120,120]`, independent FP64-stabilized host softmax
  comparison, finite per-segment partial checks, and BF16 output/gate checks.
  This specifically exercises scan rescaling and increasing-index merge
  rescaling.
- Updated diagnostic `benchmarks/attention_bench.cpp` to use 5 warmups and 20
  measured repetitions per length, reporting median, p99, and standard-error
  uncertainty. It explicitly labels itself diagnostic-only.
### Tests run
Hardware: NVIDIA GeForce RTX 5090 (32,607 MiB), driver 590.48.01, `sm_120`.
Source revision:
`90a2816fb3dad0933395f5c19d44eb81fcfa5eb9`.
Artifact: Docker image tag `qw38-dev:cuda13.4.1`, immutable image digest
`sha256:be0903b40ab2e14ec1b5ba285655219cc9e521cd1455b070a6dfdb68ed4e4bfb`;
CUDA 13.4.1 (`nvcc` 13.4.59), CMake 3.28.3, GCC 14.2.0.

Command:
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 bash -lc 'cmake --build build/debug --target qw38_attention_core_unit_test qw38_bench_attention -j2'`
Result: build passed. Final artifacts after the oracle-guard rebuild have
core-unit binary SHA-256
`3e410a04c1948d7953bcd02d01f943680c018beb1da7b3e0f86f110f7b4c3865`;
benchmark binary SHA-256
`4e15c66b62f8fc72c0e0b480f1cd607607137fcc95091755db58ea1701e2c7d8`.

Final oracle-guard rebuild/run:
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 bash -lc 'cmake --build build/debug --target qw38_attention_core_unit_test -j2 && ./build/debug/tests/qw38_attention_core_unit_test'`
Result: build passed; `attention core unit ok`.

Command:
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 bash -lc './build/debug/tests/qw38_attention_core_unit_test'`
Result: `attention core unit ok`.

Command:
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 bash -lc 'ctest --test-dir build/debug -R "attention_(unit|core_unit|reference|integration)$" --output-on-failure'`
Result: 4/4 passed (`attention_unit` 2.84 s, `attention_core_unit` 0.21 s,
`attention_reference` 13.04 s, `attention_integration` 7.26 s; total 23.36 s).

Independent verification:
- Fresh Debug `ctest --test-dir build/debug --output-on-failure`: 38/38 passed.
- Fresh Release attention ctest: 4/4 passed.
- The verifier also reran the diagnostic benchmark on NVIDIA GeForce RTX 5090
  `sm_120` at populated lengths 512, 4096, and 32768 with 5 warmups and 20
  measured repetitions, reporting median, p99, and standard error.
### Benchmark results
Command:
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 bash -lc './build/debug/benchmarks/qw38_bench_attention'`

Diagnostic identity: `sm_120`, 128 threads, 256-key segments, 32-key
subtiles, 128 merge threads; 5 warmups and 20 repetitions; timings include
the partial zero plus scan/merge launch sequence. Output (ms):

| populated | segments | median | p99 | uncertainty (standard error) |
|---:|---:|---:|---:|---:|
| 512 | 2 | 0.167344 | 0.180444 | 0.000988623 |
| 4096 | 16 | 0.206640 | 0.214420 | 0.000844191 |
| 32768 | 128 | 1.017200 | 1.033520 | 0.00189285 |
### Architecture blocker
None.
### Follow-up observations
None.
