# OPT-060 — Matched full-engine family attribution

This increment **claims no performance improvement** and makes **no tok/s
speedup claim**. It records enclosing family execution on Quartz and a private
instrumented build of pinned llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
Numbers below are ranking instruments and overhead diagnostics, not a keep/reject
throughput gate.

## Proof limits

- Eager diagnostic CUDA events are **not** production CUDA-graph timings.
- Graph-enabled per-kernel attribution was not available on the graph replay
  path. Uninstrumented graph wall totals are retained beside labeled
  `eager_diagnostic` event tables. Paired production kernel deltas are not
  invented.
- Fused gate/up/GLU (and graph FFN) charge **once** on the enclosing interval;
  member IDs are counted but not summed as extra milliseconds when the
  aggregator sees `attribution_role=enclosing`.
- Concurrent streams: summed GPU durations are **work**, not wall. Interval
  union / critical path is reported separately. Overlap is expected when
  streams actually overlap. Quartz decode/prefill records in this sitting used
  a single recorded stream id; decode union across tokens is also compressed
  because each `execute_token` records a fresh epoch origin. Do not treat that
  cross-token union as production multi-stream overlap.
- Isolated OPT-043 `component_gap.cpp` ggml-op timings are **not** this sitting.
- A matched-token diagnostic is distinct from `llama-bench` if that tool uses a
  different input policy. This sitting uses `(42 + index * 997) % 248320`.
- Nsight Systems / Compute are optional. OPT-061 owns targeted counters. Events
  plus actual dispatch are the required fallback. This task does not install
  host packages or run a full Nsight Compute sweep.

## Record

Each operation record carries engine, phase, sequence/token position, layer,
role, tensor name/type/M/N/K/strides, stream identity, launch family, fused
member IDs/count, start/end event IDs, graph mode, and complete-work
attribution. Activation quantization, final epilogues, copies, graph launches,
and CPU/unknown gaps are in-scope families.

## Builds

- Quartz: `build/qw38-cuda-opt060-engine-attribution-test` with pooled
  `GpuPhaseRecorder` events (create once, resolve after the run, no per-kernel
  synchronize/printf). Compile-time NVTX ranges carry operation IDs under
  `QW38_DIAGNOSTIC_TRACE`. Prefill `sync_tokens` preserves `families` across the
  attribution zeroing step.
- llama: private worktree `.cache/authorities/llama.cpp-opt060` plus
  `tools/llama_authority/patches/opt060-engine-attribution.patch`. Production
  `.cache/authorities/llama.cpp`, `llama-build`, and `qw38-llama-authority`
  remain untouched. Helper:
  `tools/llama_authority/build_opt060_instrumented.sh`. Provenance:
  `.cache/authorities/opt060-instrumented-provenance.json` (patch SHA
  `e6321fe8184df449dff5d35a27c27e6089207b7a4b91490c8d9665bd14f7afb0`, overlay SHA
  `a74057dd481824e909ae40039dd44a4d6d8c2a18bd2b5f19c31b4bb12e2f4234`, image
  `qw38-llama-authority:cuda-13.0.2`).
- Diagnostic eager llama sets `QW38_OPT060_EAGER` / `GGML_CUDA_DISABLE_GRAPHS`
  while leaving fusion enabled. Launch family is the actually selected
  `mmvq` / `mmvq_fused_glu` / `mmq` path. Unperturbed llama runs with
  attribution env unset so the wall is not event-instrumented.
- Engines are exclusive on the 32 GiB GPU: Quartz unloads and
  `cudaDeviceReset`s before llama is spawned. Session capacity for this probe
  is 8192, enough for P4096 / D2048+16.

## Host protocol sitting

Synthetic overlapping intervals in
`fixtures/opt060_engine_attribution.json`: fused FFN 4.0 ms on stream 0 and a
2.0 ms copy on stream 1 overlapping `[1,3]`. Summed work 6.0 ms, union 4.0 ms,
overlap 2.0 ms. The fused gate member is **not** added again.

Live sitting, pinned RTX 5090, `models/Qwen3.8-27B-Q4_K_M.gguf`, matched batch
(`decode` n_batch=2048 n_ubatch=1; `prefill` n_batch=n_ubatch=4096). Token IDs
match Quartz. Llama decode wall is the 16 output tokens after an untimed 2048
prefix, matching Quartz `execute_token` timing. Prefill walls cover the full
4096-token prompt.

## Uninstrumented walls and profiling overhead

These are not throughput claims.

| Phase | Engine | Unperturbed graph wall (ms) | Eager diagnostic wall (ms) | Overhead (ms) |
| --- | --- | ---: | ---: | ---: |
| D2048+16 | Quartz | 429.192 | 469.488 | 40.296 |
| D2048+16 | llama | 241.543 | 239.994 | −1.549 |
| P4096 | Quartz | 1403.806 | 1430.294 | 26.489 |
| P4096 | llama | 1252.058 | 1251.530 | −0.528 |

Llama diagnostic walls are slightly below unperturbed walls (noise / graph
warmup). That is **not** a speedup claim. Quartz diagnostic overhead is the
event-record path versus CUDA-graph replay.

## D2048 enclosing family ranking (eager diagnostic)

Quartz: 17312 records, pool overflow false. Category recorders emit enclosing
roles; leaf recorders emit members. `ffn_gate_up_glu` is one enclosing fused
interval (`ffn_gate,ffn_up,ffn_glu`, fused_member_count=3). `ffn_mmv` is the
OPT-038 category around the same FFN and is **not** added to `ffn_gate_up_glu`.
Selected decode launch families include `mmv`, `gdn`, `attention`, `row_decode`,
`rms_norm`, `copy`.

llama: 8192 records, **pool overflow true**. Dispatch actually selected
`mmvq` / `mmvq_fused_glu` (1693+846 launches) plus unlabeled `kernel`. Truncated
tables cannot be compared call-for-call with Quartz's 1024 fused FFN intervals.

| Role | Quartz enclosing ms | Quartz calls | llama ms | llama calls | Quartz−llama ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ffn_mmv | 261.509 | 1024 | — | 0 | ranking only |
| ffn_gate_up_glu | 136.629 | 1024 | 25.592 | 376 | truncated llama |
| mixer_mmv | 114.088 | 2048 | MUL_MAT 53.219 | 2069 | unmatched names |
| gdn_core | 37.660 | 784 | GATED_DELTA_NET 1.484 | 282 | unmatched names |
| attention_core | 35.994 | 256 | FLASH_ATTN_EXT 3.314 | 94 | unmatched names |
| gdn_conv_qk_norm_recurrence | 27.745 | 768 | SSM_CONV 1.634 | 283 | unmatched names |
| logits | 12.817 | 16 | (in MUL_MAT) | — | unmatched names |
| embedding | 0.245 | 16 | missing | 0 | missing on llama dump |
| state_commit | 0.256 | 16 | copy 1.727 | 283 | |

Role vocabularies do not match: Quartz uses scheduler leaves/categories;
llama uses ggml op names when `infer_role` does not map the tensor
(`MUL_MAT`, `RMS_NORM`, `FLASH_ATTN_EXT`, `GATED_DELTA_NET`). Unmatched roles
are listed in the raw JSON. Missing required names on this dump: Quartz
`activation_quant` (recorded as `activation_quant_ffn` /
`activation_quant_mixer`), `copy` (recorded as `state_copies` / `d2h`),
`cpu_unknown_gap`. llama dump missing `embedding`, `ffn_down`,
`logits_projection`, `activation_quant`, `cpu_unknown_gap` (overflow plus
naming).

Quartz decode summed chargeable work 915.4 ms versus interval union 29.1 ms.
That union is **not** concurrent-stream proof: all records used stream index 0
and per-token epoch origins overlap in timestamp space.

## P4096 enclosing family ranking (eager diagnostic)

Quartz: 1193 records, overflow false. Prefill does **not** fuse gate/up/GLU
into one family; members `ffn_gate` / `ffn_up` / `ffn_glu` remain separate
under category `ffn_mmq` (launch family `mmq`). llama: 3462 records, overflow
false. Selected llama prefill launches: `mmq` 992, `mmvq` 1, `kernel` 2469.
Gate/up are not fused (`ffn_gate` 128, `ffn_up` 128, `GLU` 128).

| Role | Quartz enclosing ms | Quartz calls | llama role / ms | llama calls |
| --- | ---: | ---: | --- | ---: |
| ffn_mmq | 663.845 | 64 | MUL_MAT 896.534 + ffn_up 391.691 + ffn_gate 387.933 | 706+128+128 |
| mixer_mmq | 309.094 | 128 | (in MUL_MAT / CONCAT) | — |
| attention_core | 255.335 | 16 | FLASH_ATTN_EXT 77.422 | 32 |
| gdn_core | 199.160 | 49 | GATED_DELTA_NET 370.083 | 96 |
| gdn_conv_qk_norm_recurrence | 198.795 | 48 | SSM_CONV 21.300 | 96 |
| logits | 0.698 | 1 | (in MUL_MAT) | — |
| embedding | 0.141 | 1 | missing | 0 |
| commit_sync | 0.515 | 1 | copy present | 96 |

Quartz prefill chargeable work 2856.7 ms, union 1428.9 ms (enclosing categories
plus remaining members; stream index 0). llama work 2444.3 ms, union 2444.3 ms
(no material overlap on the recorded stream). Inclusion: fused GDN recurrence
is one enclosing Quartz interval with members `gdn_conv,gdn_qk_norm,gdn_recurrence`.
llama `GATED_DELTA_NET` fused_member_count=4 is charged once.

## Nsight

Tried once per native screen binary via `command -v nsys` and `command -v ncu`
inside `qw38-cuda:13.0.2`. `nsys` was **not found**; no 16-token decode or
prefill timeline was captured. `ncu` was present; no full Nsight Compute sweep
was run (OPT-061). Fallback is CUDA events plus actual mmvq/mmq/mmv dispatch.

## Acceptance stability

Three repetitions per listed workload/mode completed. Feedback remains at
300 s; acceptance uses a separate 7200 s aggregate deadline in
`pins/opt060_iteration_contract.json` (repetition count unchanged). Numbers
are a stability snapshot, **not** a throughput claim.

`--mode acceptance` passed in 408.46 s / 7200 s:
`build/optimization-runs/OPT-060/acceptance/20260911T074042Z-3e10dc40`.
Smoke, correctness, three decode reps (2 engines × 2 modes), and three P4096
reps all finished. Engine order stayed alternated with exclusive GPU turns.

D2048+16 Quartz unperturbed ms: 429.375, 447.375, 446.539; diagnostic
469.608, 482.206, 481.379; overhead 40.232, 34.831, 34.840; 17312 records,
overflow false. llama unperturbed wall ms: 239.914, 242.375, 242.547;
eager_diagnostic 239.173, 239.919, 240.100; decode diagnostic
`pool_overflow=true` at 8192 on every rep (labeled, not silent).

P4096 Quartz unperturbed ms: 1461.643, 1449.420, 1448.401; diagnostic
1490.791, 1478.588, 1476.852; overhead 29.148, 29.168, 28.451; 1193 records.
llama unperturbed 1313.416, 1306.603, 1301.885; eager_diagnostic 1304.299,
1300.140, 1293.428; 3462 records, overflow false.

A prior 300 s acceptance sitting
(`build/optimization-runs/OPT-060/acceptance/20260911T072140Z-8ec04772`)
hit `budget_exhausted` mid decode-rep-2; that partial run is superseded.

## Raw records

- `evidence/optimization/opt060-engine-attribution/quartz-decode-rep{0,1,2}-records.json`
- `evidence/optimization/opt060-engine-attribution/quartz-prefill-rep{0,1,2}-records.json`
- `evidence/optimization/opt060-engine-attribution/llama-decode-{unperturbed,eager_diagnostic}-rep{0,1,2}.log`
- `evidence/optimization/opt060-engine-attribution/llama-prefill-{unperturbed,eager_diagnostic}-rep{0,1,2}.log`
- Feedback runs:
  `build/optimization-runs/OPT-060/feedback/20260911T071927Z-ed9ae184` (decode),
  `build/optimization-runs/OPT-060/feedback/20260911T071644Z-e0c07f3c` (prefill)
- Acceptance (complete):
  `build/optimization-runs/OPT-060/acceptance/20260911T074042Z-3e10dc40`
- Acceptance (superseded `budget_exhausted`):
  `build/optimization-runs/OPT-060/acceptance/20260911T072140Z-8ec04772`
