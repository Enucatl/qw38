# TASK-044 decode and startup attribution

The original **8.017558 ms first-decode regression remains UNRESOLVED** in the
saved evidence and the bounded new captures. TASK-042 measured
21.262324 → 29.279882 ms at position 4096.
Its later diagnostic traces measured 25.048150 → 25.498606 ms. These are
different executions; their CUDA timelines cannot explain an event that was
not traced. Graph construction is a measured first-use cost in both arms,
but the evidence does not establish it as the cause of the original difference.

TASK-042's accepted regression, TASK-043's output-bound P100 exception, and
the remaining speed gaps retain their recorded meanings. This investigation
does not change the production decision or implement a latency optimization.

## Workload and historical evidence

The paired workload is the frozen TASK-027 prompt at 4096 tokens, followed by
eight teacher-forced continuation inputs, capacity 4224, graph mode, one
execution and zero warmups per arm. It finishes at position 4104. The parent
uses TASK-041's M256 schedule; the candidate uses TASK-042's M512 schedule.
Their arithmetic schedules differ during prefill; parent/candidate logits
are therefore a cross-schedule diagnostic. Instrumented-versus-original
comparisons within each arm must remain byte-identical.

All times below are milliseconds. Setup excludes the separately timed CUDA
initialization. Prefill includes its first-use allocations/library work and
prompt readout. Decode includes first graph use, readback, completion and
the benchmark's existing completion wait. The remaining-seven column sums
the seven recorded step timers; the eight-step phase includes small loop
overhead outside those timers.

| Execution | CUDA init | Cold setup | Load/upload within setup | Prefill | First decode | Remaining seven | Eight-step phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TASK-042 original parent | 114.095496 | 5530.368926 | 5525.002046 | 1916.225098 | 21.262324 | 128.790766 | 150.054052 |
| TASK-042 original candidate | 117.223524 | 20233.325919 | 20209.652521 | 1553.510184 | 29.279882 | 127.043607 | 156.324259 |
| TASK-042 diagnostic parent | 330.390915 | 6651.996626 | 6631.102721 | 1969.980701 | 25.048150 | 133.216947 | 158.266690 |
| TASK-042 diagnostic candidate | 278.204904 | 5502.649727 | 5489.293325 | 1592.712052 | 25.498606 | 133.220414 | 158.720723 |

Original candidate-minus-parent differences are +8.017558 ms for the first
step, -1.747159 ms for the remaining seven and +6.270207 ms for the phase
(+4.1786%). Most of the original 14.703 s setup difference falls inside the
existing load/upload timer. That timer does not distinguish host file access,
mapping/page faults, allocation or upload; elapsed load time cannot identify
cache state. The diagnostic setup difference reverses direction. No statistical
or causal conclusion follows from these single observations.

Raw evidence is under `.cache/evaluation/qw38-language-v2/task042-support/`:
`parent-integrated.jsonl`, `candidate-integrated.jsonl`, their saved logits,
`diagnostic-{parent,candidate}.{jsonl,sqlite,nsys-rep}`, `trace-analysis.json`,
`baseline.sh`, `measure.sh`, `diagnose-first-use.sh`, and their identity records.
The old `.cache/task042` executable paths in those scripts are historical.
Authenticated parent executables survive in `parent-binaries/`.

## Saved first-decode timeline

The analyzer recomputes these costs from the raw SQLite tables. The original
TASK-042 `trace-analysis.json` independently fixes the range duration, kernel
count/sum, and first/last-kernel offsets used for the focused reporting check.

| TASK-042 diagnostic, first step | Parent | Candidate |
| --- | ---: | ---: |
| Host step timer | 25.048150 | 25.498606 |
| `decode_token` NVTX range | 25.037770 | 25.486784 |
| GPU kernels: count / sum | 883 / 18.330960 | 883 / 18.518773 |
| GPU kernels + copies + memset, interval union | 18.351984 | 18.538613 |
| NVTX start to first kernel | 6.390148 | 6.650210 |
| Last kernel to NVTX end | 0.135286 | 0.132937 |
| CUDA API time on calling thread, interval union | 23.971736 | 24.397991 |
| Host range outside recorded CUDA APIs | 1.066034 | 1.088793 |
| Host range without recorded GPU activity | 6.685786 | 6.948171 |
| Graph instantiation API | 3.657109 | 3.886116 |
| 883 capture-time launch APIs, sum | 1.108075 | 1.123119 |
| First graph launch API | 0.361657 | 0.382457 |
| Stream synchronization API | 18.531606 | 18.712890 |
| One control-buffer allocation API | 0.011863 | 0.012263 |
| Source graph destruction API | 0.098287 | 0.103186 |
| Three asynchronous-copy APIs, sum | 0.016922 | 0.020349 |

Every first-step GPU kernel's CUPTI correlation ID resolves to
`cudaGraphLaunch_v10000`. The 883 capture-time launch calls record the graph;
they are not another GPU execution of those kernels. Each token transfers
216 control bytes host-to-device, then 993280 FP32-logit bytes and four error
status bytes device-to-host. First-step transfer GPU sums are 0.020736 and
0.019552 ms; the range also includes a 0.000288 ms GPU memset in each arm.
No `cudaGraphUpload` call or `cuLibraryLoadData` call occurs within these saved
first-decode ranges. The driver may perform internal preparation during first
launch; the trace does not separately name or time that work.

The stream wait overlaps GPU execution. CPU API sums can also contain nested
calls, and GPU intervals may overlap one another. Only unions measure covered
wall-clock intervals. Neither the API sum nor synchronization time is added
to kernels or host latency. Time outside recorded APIs is unclassified host
time, which can include computation, scheduling and instrumentation. Time
without recorded GPU activity also includes graph construction and launch
latency; it is not an additional elapsed cost or proof of CPU scheduling delay.

The host step exceeds its inner NVTX range by 0.010380 / 0.011822 ms in these
traces. That boundary difference includes the benchmark wrapper and its
existing completion wait. For the remaining seven, kernel sums are
129.705589 / 129.765648 ms, while host-step sums are
133.216947 / 133.220414 ms. The recurring work is dominated by GPU kernels in
these observed traces. This does not explain the untraced original outlier.

## Reuse of TASK-043 captures

The final candidate uses the same inference implementation and artifact.
The saved request-4096 trace generates 127 continuation tokens; the saved
decode-4096 trace consumes 128 fixed inputs after an untimed populated-context
setup. Their first eight steps are supporting visibility evidence, not the
original paired 4096+8 acceptance workload.

| TASK-043 capture | CUDA init | Cold setup | Prefill or populated setup | First decode | Next seven, sum | First graph instantiate | First kernel sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| request-4096 | 280.834331 | 5441.096095 | 1618.472 | 25.652181 | 135.308400 | 3.698430 | 18.819138 |
| decode-4096 | 279.722841 | 5442.500919 | 1627.083 | 25.892824 | 136.317871 | 3.848012 | 18.942843 |

Each first decode again has 883 kernels correlated to one graph launch.
These captures show comparable first-use structure, with first decode near
25.6–25.9 ms under profiling. They have no TASK-041 parent arm and do not
establish causality for the original 8 ms difference. Evidence paths are
`task043-support/{request,decode}-4096-qw38.{jsonl,sqlite,nsys-rep}` under the
same evaluation root. `task042-source-reuse.json` records the unchanged
candidate inference sources used by TASK-043.

## Missing boundaries and bounded instrumentation

Saved CUDA API events cover setup, but the existing NVTX ranges begin with
`perf01_measured` and individual `decode_token` calls. QW38's setup JSON has
no separate runtime-create, session-create or plan-bind timers, and token
ranges omit position and graph state. Those are concrete visibility gaps;
the bounded additional captures are justified by them. An additional capture
cannot recover the missing timeline of the original unprofiled outlier.

The request benchmark now enables its existing opt-in profiling before
setup and adds `cuda_initialization`, `runtime_create`, `model_load_upload`,
`session_create`, `plan_bind` and `initial_prefill` ranges. JSON adds the
runtime-create, session-create and plan-bind durations. Model-load timing
retains its existing completion wait. Session-create and plan-bind timers
measure their existing calls without adding a completion wait. The enclosing
cold-setup timer still includes the existing final completion wait.

The runtime retains the existing outer `decode_token` range and adds an
opt-in inner range with token position, selected graph bucket and entry state
(`build`, `replay` or `eager`). The entry-state label describes the planned
path; successful CUDA API correlation and the benchmark's existing fallback
rejection establish the executed path. No inference work or preparation is
moved across an existing timing boundary and no synchronization is added.

The extended `scripts/task027_profiles.py --decode-attribution PATH...`
reads each SQLite/JSONL pair, reports setup markers and the first eight token
ranges, filters CPU APIs by the range's calling thread, and correlates GPU
kernels/copies to their submitting APIs. GPU activity is a window diagnostic;
it is not causally charged to a setup stage merely because it overlaps that
stage. Boundary-crossing durations are clipped; transfer byte counts retain
the full size of intersecting transfers. This fixed single-process benchmark
does not contain concurrent inference sessions.

## Cost classification and remaining observable

| Cost | Lifetime / accounting |
| --- | --- |
| CUDA initialization | Process/context preparation, separately reported |
| Runtime, model load/upload | Runtime/model preparation, inside cold setup |
| Session storage and plan binding | Session preparation, inside cold setup |
| Lazy prefill buffers and libraries | First prefill use, inside prefill |
| Control-buffer allocation | First graph preparation for the plan |
| Capture, instantiate, source-graph destruction | New graph bucket or invalidated executable; inside the triggering decode |
| Implicit executable upload / driver first-launch preparation | Not separately observable from the recorded first graph launch |
| Input control copy, graph replay, kernels, logits/error readback, sync, host scan and commit | Recurring token work |

For positions 4096–4103 the incoming token makes populated length
4097–4104. The source selects `min(4224, bit_ceil(populated)) = 4224`.
The first token builds that capacity-capped bucket; the next seven reuse it.
This is the existing graph selection rule, independent of the agreed future
small/large scheduling cutoff at 4096.

**FOLLOW_UP_REQUIRED:** a CUDA/API/host-scheduling timeline of a first decode
that actually exhibits the original excess latency is missing. The saved
traces disable CPU context-switch collection and do not distinguish OS wait,
page faults, instrumentation or uninstrumented driver work in the uncovered
host intervals. The smallest next discriminating measurement is a separately
authorized capture of that slow event with these phase/token markers plus
CPU scheduling and host file/page-fault visibility, correlated to CUDA APIs
and GPU activity. If the slow event does not occur, its original cause remains
unresolved. No repair is supported solely by equal kernel counts or elapsed
load-time differences.

## New instrumented observations

Exactly one new instrumented 4096+8 execution completed per arm. No workload
was repeated. The first launch script built both binaries and passed the
candidate integration check, then failed on an unset container shell variable
before launching either diagnostic workload. The corrected capture script used
those completed binaries, guarded against overwriting an existing capture,
and completed both runs. These observations are diagnostics and do not replace
TASK-042's original measurements.

| TASK-044 host timing | Parent | Candidate |
| --- | ---: | ---: |
| CUDA initialization | 322.460757 | 293.828415 |
| Cold setup | 5550.695654 | 5477.000855 |
| Runtime creation within setup | 0.114848 | 0.094008 |
| Model load/upload within setup | 5545.450188 | 5471.383782 |
| Session creation within setup | 0.919037 | 1.095281 |
| Plan binding within setup | 4.085001 | 4.326209 |
| Setup remainder, including final wait/reporting | 0.126580 | 0.101575 |
| Prefill / first prompt readout | 1972.165677 | 1601.870664 |
| First decode, position 4096 | 25.886605 | 25.543292 |
| Remaining seven steps | 133.510210 | 134.245496 |
| Eight-step phase | 159.398318 | 159.790272 |
| Measured request, prefill plus decode | 2131.564025 | 1761.660986 |
| Sum of CUDA initialization, cold setup and measured request | 8004.720436 | 7532.490256 |

The final row combines nonoverlapping benchmark timers, but excludes input
file reading and process teardown; it is not a whole-process elapsed-time
measurement. The candidate's first decode is 0.343313 ms lower in this new
pair, while its remaining seven are 0.735286 ms higher. The original first-step
regression was not reproduced. Neither this pair nor the earlier diagnostic
pair supports a causal claim or statistical speed estimate.

New setup ranges expose these costs:

| NVTX setup window | Parent / candidate wall ms | Parent / candidate CUDA API union ms | Parent / candidate outside-API ms |
| --- | ---: | ---: | ---: |
| CUDA initialization | 322.332132 / 293.720379 | 121.978289 / 108.400952 | 200.353843 / 185.319427 |
| Model load/upload | 5545.465889 / 5471.397960 | 1709.615093 / 1696.785236 | 3835.850796 / 3774.612724 |
| Session creation | 0.925609 / 1.101494 | 0.870100 / 1.044444 | 0.055509 / 0.057050 |
| Plan binding | 4.089940 / 4.331138 | 0.578009 / 0.883920 | 3.511931 / 3.447218 |

Model loading performs 1276 allocations and 1276 H2D copies, totaling
21,013,686,400 bytes in each arm. Their GPU transfer sums/unions are
1615.460304 / 1603.726859 ms. Most of the remaining model-load wall time lies
outside traced CUDA APIs; host mapping/validation/file access and scheduling
are not individually timed. It cannot be assigned to cache misses or disk I/O.
Session creation performs seven allocations and six memsets, with an existing
stream wait. Plan binding has one pinned-host and one device allocation.

Initial prefill contains nine device-allocation calls and 13
`cuLibraryLoadData` calls in each arm. The library-loading API sums are
30.513385 / 30.527594 ms, inside the measured prefill phase. Prefill GPU kernel
sums are 1912.326079 / 1545.791137 ms; GPU activity unions are
1911.909727 / 1545.549633 ms. Overlap explains why a sum can exceed its union.
First-use prefill preparation is not reclassified as model loading.

| TASK-044 first-decode attribution | Parent | Candidate |
| --- | ---: | ---: |
| Outer NVTX duration | 25.878069 | 25.535688 |
| First kernel offset from NVTX start | 7.037945 | 6.749516 |
| 883 kernels, sum/union | 18.509460 | 18.470583 |
| GPU activity union | 18.532276 | 18.490359 |
| Host outside CUDA APIs | 1.152384 | 1.132098 |
| Window without recorded GPU activity | 7.345793 | 7.045329 |
| Graph instantiation API | 4.006954 | 3.890382 |
| First graph launch API | 0.377758 | 0.358701 |
| Stream synchronization API | 18.705140 | 18.669562 |
| Control allocation API | 0.013807 | 0.012864 |

Both traces explicitly label position 4096 as `bucket=4224 graph=build`, then
positions 4097–4103 as `bucket=4224 graph=replay`. Each of the eight tokens
has 883 kernels correlated to one successful graph launch, one 216-byte H2D
control transfer, and two D2H transfers totaling 993284 bytes. Only the first
has graph instantiation and control allocation; no module/library-load or
explicit graph-upload call appears during these decode steps. The remaining
seven have kernel sums 129.905365 / 130.754273 ms, GPU activity unions
130.072853 / 130.892865 ms, and outside-API host totals
0.822007 / 0.812273 ms. All per-step API costs remain in
`task044-support/instrumented-attribution.json`.

These runs include Nsight's CUDA/NVTX interception and graph-node tracing.
New marker calls and string formatting also have overhead. The initialization
timer includes its marker entry/exit, while named setup timers sit inside
their marker ranges; range durations therefore differ slightly from JSON
timers. In particular, the large initialization interval outside recorded
CUDA APIs cannot be identified as pure driver initialization or profiler cost
without another observable. No independently measured profiler-overhead
correction exists, and none is subtracted. Cross-run timing differences are
not an estimate of instrumentation overhead.

Hardware snapshots bracket each workload, with no clock locking or page-cache
eviction. The required integration check preceded the captures. These facts
do not establish any cache state at inference entry.

| Snapshot, RTX 5090 | GPU / memory MHz | Power W | Temperature C | State |
| --- | ---: | ---: | ---: | --- |
| Parent before | 240 / 405 | 29.94 | 47 | P8 |
| Parent after | 2970 / 14001 | 104.55 | 51 | P0 |
| Candidate before | 2970 / 14001 | 98.13 | 51 | P0 |
| Candidate after | 2962 / 14001 | 174.48 | 53 | P0 |

The parent ran before the candidate, as in the original diagnostic command.
Snapshots are not continuous telemetry and cannot attribute an individual
token's latency to clocks, temperature or power.

## Identities, exact commands and verification

Parent inference source: `56cd53bda200bfa76ab5c279e58616e27b112945`.
The original TASK-042 baseline already applied the request benchmark adapter
in `task042-support/baseline-benchmark.diff` (4096 development-mode support,
chunk reporting and final-logit output). The rebuilt parent uses that same
adapter plus TASK-044 markers; its runtime changes are solely the markers.
Candidate inference source: `9648e697035cbadf8fcb0de3bf7752153b491b48`,
unchanged for these files at task admission
`959e045aa9577b229e7c1f13d2f78ce37f2bd7be`, plus TASK-044 instrumentation.
The separate parent checkout and candidate use matching Release compiler
commands after source-directory normalization: GCC 14, C++23, `-O3 -DNDEBUG`,
native `compute_120`/`sm_120`, and pinned CUTLASS
`098de2a652cf8f00fd70b2df54051c7eccbb855a`.

| Executable identity | SHA-256 |
| --- | --- |
| Original TASK-042 parent request | `7c6f093eb4b0d582a64208a27d83d1da6eaae354ec2c0408305d6049bd7833f8` |
| Original TASK-042 candidate request, historical recorded identity | `3d1677868d423b9b408cdc10963cb1e706335c14e3185914a8ae3106843b6825` |
| TASK-044 instrumented parent | `d123009feaf7897970717b0f511b0181f9ce98c90483416225b3ac4e68b04103` |
| TASK-044 instrumented candidate | `274682eb92647cfe477075ee58b227ea24afadeba0ce7913771adbf12b440a70` |

Both instrumented executables are preserved in `task044-support/` as
`{parent,candidate}-request-instrumented`. Their shared instrumented benchmark
source SHA-256 is
`7c15ed79ea19b8a13b69515ac7de482f61aebaa833a2d3eeb175cdd1eb30a4f3`;
instrumented `language_model.cpp` is
`e9a5b6d2dd0c6847fc0589c754197d4d5386a63892788729847f2859de78fa95`.
The schedules differ through the other parent/candidate runtime/CUDA sources,
not through the shared benchmark or decode-token marker source.

Both arms load `.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38`,
policy 1030, with manifest-only SHA-256
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
Frozen input `.cache/evaluation/qw38-language-v2/task027-support/tokens-4096.u32le`
has SHA-256
`fa28a473299ca58495ee860d08f8c532987f3df7de95563d0f575275817ba923`.
No artifact or checkpoint tensor payload/scale hashes were computed.

The immutable container is
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`:
CUDA 13.4.1, nvcc 13.4.59, GCC 14.2.0, CMake 3.28.3, Nsight Systems
2026.3.0.0. GPU UUID is `GPU-e51ee570-3143-784d-789d-e3054637ad0b`, driver
590.48.01, power limit 400 W. Full snapshots and tool versions are preserved.

Exact command records, diffs, binaries and logs are under
`.cache/evaluation/qw38-language-v2/task044-support/`. In the commands below,
`E` abbreviates that directory and `V` is
`.cache/evaluation/qw38-language-v2`; these abbreviations do not alter paths.

```bash
python3 /home/user/.agents/skills/wake-run/scripts/wake_run.py --command 'bash .cache/evaluation/qw38-language-v2/task044-support/run-instrumented.sh'
python3 /home/user/.agents/skills/wake-run/scripts/wake_run.py --command 'bash .cache/evaluation/qw38-language-v2/task044-support/capture.sh'
```

The first command's parent/candidate builds passed. Its affected check was
`QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R "^language_model_integration$"`:
**1/1 PASS**, 20.07 s. It covers graph/eager equality, bucket boundaries,
prefill handoff/replay/restore, poisoning/failure recovery, movement/interleave
and first-decode/maximum-graph-upload capacity. The script then exited 1 on
the unset variable, as recorded in wake log `f77e5a70bcac.log`; no diagnostic
workload started. The second command exited 0, wake log `d8a58e22f279.log`.
Its exact capture invocation inside the pinned container was:

```bash
QW38_PROFILE=1 /opt/nvidia/nsight-compute/2026.3.0/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  -o "$out/instrumented-$arm" "$out/$arm-request-instrumented" "$artifact" "$input" development 4096 \
  "$out/instrumented-$arm.jsonl"
```

Here `out` is `E`, `arm` takes `parent` then `candidate`, and artifact/input
are the exact paths above. `capture.sh` preserves all container, export,
hardware, copy and identity commands. It invokes
`uv run python E/verify-outputs.py`: **PASS for both arms**, checking frozen
setup identities, matching output IDs, final position 4104, finite and
byte-identical final logits, eight steps, one run and zero warmups.
The common output IDs are `[321,279,3578,3175,1105,13,814,20139,417]`.

Reporting commands:

```bash
E=.cache/evaluation/qw38-language-v2/task044-support
V=.cache/evaluation/qw38-language-v2
uv run --with ruff ruff format scripts/task027_profiles.py tests/test_task027_profiles.py
uv run --with pytest pytest -q tests/test_task027_profiles.py
uv run python scripts/task027_profiles.py --decode-attribution "$V/task042-support/diagnostic-parent.sqlite" "$V/task042-support/diagnostic-candidate.sqlite" "$V/task043-support/request-4096-qw38.sqlite" "$V/task043-support/decode-4096-qw38.sqlite" > "$E/saved-attribution.json"
uv run python scripts/task027_profiles.py --decode-attribution "$E/instrumented-parent.sqlite" "$E/instrumented-candidate.sqlite" > "$E/instrumented-attribution.json"
```

The focused reporting tests passed **2/2**, including overlapping/nested API
and GPU intervals, clipping at a boundary, excluding another host thread,
and graph/copy correlation. The saved-trace check matched all five known
TASK-042 first-decode metrics in each arm, including the 883 kernels and
their sums, and resolved all those kernels to the graph launch. Raw SQLite,
JSONL and generated analysis preserve both host and device time scales.
No full quality rerun, comparator rerun or optimization sweep was required.

`uv run python .cache/evaluation/qw38-language-v2/task044-support/verify-source-graph.py`
also passed: captured sources/input/binaries unchanged, original parent
binary authenticated, 37 normalized compiler commands equal, six startup
ranges per arm, correct token positions/bucket/build/replay and API/transfer
correlation. `git diff --check` passed. The original focused-test and
known-trace check transcript is preserved as `reporting-check-transcript.md`.
