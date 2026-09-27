# Candidate telemetry

Production attribution is opt-in through `QW38_PROFILE=1`. The markers cover
the actual internal enqueue path, including captured graph nodes. They do not
add stream synchronization or alter device arithmetic.

After building `qw38_bench_request` in the pinned environment, capture a fresh
candidate with the existing driver:

```bash
uv run scripts/task027_benchmark.py telemetry --depth 4096 \
  --output .cache/evaluation/qw38-language-v2/my-telemetry-4096 \
  --check-overhead
```

The command uses the pinned CUDA container and frozen prompt plus eight fixed
continuation inputs. Depths 256 and 32768 are also supported. `--artifact`
selects the candidate artifact. `--baseline PRIOR/telemetry.json` adds matched
host/family deltas, keeping kernel sums and GPU unions separate. Artifact and
output paths must be inside the repository's container mount.
A fresh output directory is required. The
main agent launches this long command through the installed wake-run skill.

Outputs include the exact command and source/binary/input identity manifest,
candidate diff, hardware snapshots, benchmark JSONL, final FP32 logits,
Nsight report and SQLite export, `telemetry.json`, and `summary.txt`.
Artifact identity is its manifest; tensor payloads are not hashed.
The human summary stays compact. Detailed JSON retains exact layer, prefix,
shape and dispatch records and reached 317 MB for the verified 32K capture;
budget storage for it and the raw trace. This is diagnostic output only.

`--check-overhead` adds one execution without profiling and requires identical
output IDs and final logits. Its latency difference includes both the profiler
and marker overhead plus run variation. It is a single observed difference,
not a calibrated correction. First-use work remains inside request timing.

Existing captures can be reprocessed without launching inference:

```bash
uv run scripts/task027_profiles.py --telemetry \
  .cache/evaluation/qw38-language-v2/my-telemetry-4096/development-4096-qw38.sqlite \
  --summary > report.json
```

Historical captures without the new operator markers fail strict attribution.
Their original category and decode-attribution reporting commands still work.
New reports identify operators from launch correlation and explicit graph-node
creation/clone IDs, never kernel-name or timing-order guesses. Capture supplies
static operator/layer/shape identity; actual graph launch supplies the replay
token and bucket. Missing ownership of an inference kernel/copy/memset is an
error. Keep raw traces when a reporting check fails; repair/reprocess the report
before repeating valid inference.

## Reading the report

- `benchmark_records`: original synchronized host measurements and identity.
- `windows`: setup, prefill/chunk/decode and host-stage walls, clipped GPU/API
  interval unions, and time outside recorded activity. Nested windows overlap.
- `operators`: exclusive ownership, shapes/roles/layers/prefixes, actual kernels,
  resource use, copies, sums and unions, and CUDA APIs. Fused work has one label.
- `family_costs`: family totals separated by phase and graph build/replay.
- `inference_totals` and `measured_totals`: causal inference work and work within
  the benchmark window respectively; `totals` also includes process setup.
- `graph_mapping`: capture/clone/replay counts; `unmapped_inference` must be empty.
- `memory`: tracked process allocation peak, distinct from sampled free memory
  in the benchmark records.

Kernel durations can overlap. Their sum is work duration, not elapsed request
time. GPU activity union measures occupied timeline coverage. CPU waits overlap
GPU work and must not be added to it. Host time outside CUDA APIs is measured
but does not identify disk, page faults or OS scheduling as its cause.

## Component boundaries

Component drivers emit `timing_schema=2` and `gpu_event_interval_ms`. The event
interval includes any GPU idle gaps while the host submits work. Pure kernel
duration comes from trace activities. Diagnostic D2H output copies are reported
separately from resident component cost.

The old attention driver is explicitly core-only. The artifact-backed M1
boundary now runs preparation/append, scan, merge/gating, real output projection
and nonzero residual addition:

```bash
build/pinned-release/benchmarks/qw38_bench_attention_prefill --decode \
  .cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 4096 output.f32
```

This starts with prepared projection outputs, matching the existing prefill
attention component boundary. It is not the full input-projection layer.
Whole-layer costs come from the production request trace. Historical scripts
that parse old `gpu_ms`/`complete_gpu_ms` fields must explicitly adopt schema 2.

## Verification

Verified on 2026-09-27 with the current TASK-047 policy-1030 candidate, artifact
manifest `6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`,
on RTX 5090 in the pinned build container. Evidence is under
`.cache/evaluation/qw38-language-v2/telemetry-support/`; `verify.sh` records
the coordinated commands. Capture manifests preserve the exact dirty source
and binary identities. Reports reprocessed after parser changes also record
`report_parser_sha256`; the original capture identities remain unchanged.

- Release build passed for the request driver, five changed component drivers,
  and the language-model/format integration targets (`build.log`).
- All 15 focused Python tests pass, including graph clone/recapture identity,
  overlap/clipping, strict missing-ownership rejection, setup API accounting,
  baseline matching, and profiling output/failure validation. Ruff passes.
- Language-model integration and format-reader tests passed (`integration.log`).
  Integration includes bitwise eager/graph checks around 255/256/257 and
  511/512/513, graph capture failures, movement, restore and replay.
- Live captures at 256, 4096 and 32768 prompt tokens plus eight inputs have zero
  unmapped inference activities (`capture-*/telemetry.json`). Each maps 884
  captured operator nodes through explicit clone links, with 7,072 graph
  activities across eight launches. Operator/family totals reconcile with
  independent SQLite sums and interval unions; offline self-comparison gives
  zero deltas. All 64 layers retain the current token position during replay.
- Profiling on/off at 4096 gives identical output IDs and byte-identical final
  logits. Those logits also match the saved TASK-047 pre-telemetry output.
  The observed request difference was +71.572 ms: 1777.058 ms profiled versus
  1705.487 ms unprofiled (+4.20%), including profiler effects and run variation.
- The repaired M1 attention component ran at visible lengths 4096 and 32768
  with finite outputs and zero hot allocations (`attention-*.log`). Event
  intervals were 1.075 and 1.285 ms; diagnostic copies were separate.

The current 4K trace already identifies concrete priorities: prefill Q4 MLP
projection takes 1009.527 ms (63.8% of prefill kernel time), followed by GDN
recurrence at 205.215 ms (13.0%). Across seven decode replays, Q4 projection
takes 45.414 ms (40.0% of replay kernel time); `fp8_projection` and `fp8_gemv`
together take 36.474 ms (32.1%). These are attributed observations with profiling
enabled, not an unprofiled speed comparison or an arithmetic optimization.
