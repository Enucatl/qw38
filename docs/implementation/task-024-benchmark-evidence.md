# TASK-024 GDN prefill benchmark evidence

Development measurement on 2026-09-25 UTC. CandidateV2 artifact
`candidate-v2-q4k-rope-fixed.qw38`, accepted manifest SHA-256
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`
(from TASK-023). Source base `e0be8fc4617e3ac9e67d650b9cbea016767ad34f`
plus TASK-024 candidate patch. Binary SHA-256
`71e5f789a05e43f3abe52c2c37227c2cdf2a7de699ad851c156c40929e0dda6f`.
Pinned image ID
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`;
CUDA 13.4.1, `sm_120`, RTX 5090 32607 MiB, driver 590.48.01,
400 W power limit. Input rows are 16 real candidate artifact embeddings repeated to
the tested token count. Each sample resets the session state before timing.
`state_read_write_bytes` is modeled FP32 persistent-state traffic at launch
boundaries, not a profiler measure of DRAM transactions. The first sample in
each group is warmup; four later repetitions support the reported medians.

Final command (from repository root; the last command writes the raw CSV):

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 -v "$PWD:/workspace" -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc "ctest --test-dir build/pinned-release --output-on-failure -R ^\\(gdn_unit\\|gdn_reference\\|gdn_integration\\|gdn_prefill\\|gdn_prefill_artifact\\|prefill_projection\\)$ && build/pinned-release/tests/qw38_gdn_prefill_artifact_test && build/pinned-release/benchmarks/qw38_bench_gdn_prefill /workspace/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 > .cache/task024/bench-final.csv"
```

Historical command exit code 0. Focused build and six-test pass: `.codex-wake-run/211641734b09.log` (build; artifact test initially failed on bitwise BF16 history comparison) and `.codex-wake-run/e1cb155179b1.log` (six-test pass plus direct artifact diagnostic and benchmark). R24-04 subsequently invalidated that run's continuation evidence because its fifth FP32 input row was outside the allocation. The repaired test was rebuilt and passed on 2026-09-25; [refreshed source/binary identities, command and numerical results](tasks/TASK-024.md#r24-04-repair-evidence-2026-09-25) supersede the old continuation result. The BF16 history tolerance remains 0.02 absolute plus 0.006 relative; observed max 0.00195312. The corrected test checks a nonzero GDN layer, four-token prefill followed by decode against five-token decode for output, state, history and metadata, plus injected deferred stream failure and reset recovery. Benchmark code and measurements are unchanged. Raw context and samples follow, without the container startup banner. This document is the durable copy of the raw CSV; `.cache/task024/bench-final.csv` is its disposable task-cache copy:

```csv
context,gpu,NVIDIA GeForce RTX 5090,sm,120,artifact_device_bytes,21462812800,workspace_bytes,64684032,free_bytes,10498342912,total_bytes,33664794624,recurrence_registers,40,recurrence_shared_bytes,1024,recurrence_local_bytes,0,recurrence_occupancy_blocks_per_sm,12
sample,kind,m,interval,rep,gpu_ms,host_ms,state_read_write_bytes,workspace_bytes
sample,complete_layer,64,1,0,49.695,49.6865,402653184,64684032
sample,complete_layer,64,1,1,2.35366,2.35062,402653184,64684032
sample,complete_layer,64,1,2,2.34928,2.34624,402653184,64684032
sample,complete_layer,64,1,3,2.34611,2.34299,402653184,64684032
sample,complete_layer,64,1,4,2.34512,2.34236,402653184,64684032
sample,recurrence,64,1,0,0.190752,0.194509,402653184,30081024
sample,recurrence,64,1,1,0.19776,0.201703,402653184,30081024
sample,recurrence,64,1,2,0.189568,0.193316,402653184,30081024
sample,recurrence,64,1,3,0.194496,0.197135,402653184,30081024
sample,recurrence,64,1,4,0.186176,0.190182,402653184,30081024
sample,complete_layer,64,64,0,2.25043,2.24669,6291456,64684032
sample,complete_layer,64,64,1,2.26227,2.25895,6291456,64684032
sample,complete_layer,64,64,2,2.25466,2.25181,6291456,64684032
sample,complete_layer,64,64,3,2.25056,2.24766,6291456,64684032
sample,complete_layer,64,64,4,2.25283,2.24879,6291456,64684032
sample,recurrence,64,64,0,0.071584,0.075924,6291456,30081024
sample,recurrence,64,64,1,0.072384,0.074933,6291456,30081024
sample,recurrence,64,64,2,0.064832,0.06858,6291456,30081024
sample,recurrence,64,64,3,0.065376,0.069112,6291456,30081024
sample,recurrence,64,64,4,0.0728,0.075435,6291456,30081024
sample,complete_layer,128,1,0,3.74384,3.74262,805306368,64684032
sample,complete_layer,128,1,1,2.82746,2.82336,805306368,64684032
sample,complete_layer,128,1,2,2.83043,2.82541,805306368,64684032
sample,complete_layer,128,1,3,2.82608,2.82361,805306368,64684032
sample,complete_layer,128,1,4,2.82739,2.82445,805306368,64684032
sample,recurrence,128,1,0,0.367776,0.371816,805306368,30081024
sample,recurrence,128,1,1,0.367264,0.370455,805306368,30081024
sample,recurrence,128,1,2,0.366624,0.369892,805306368,30081024
sample,recurrence,128,1,3,0.365824,0.369082,805306368,30081024
sample,recurrence,128,1,4,0.368064,0.371697,805306368,30081024
sample,complete_layer,128,64,0,2.6352,2.63149,12582912,64684032
sample,complete_layer,128,64,1,2.63843,2.63416,12582912,64684032
sample,complete_layer,128,64,2,2.63606,2.63303,12582912,64684032
sample,complete_layer,128,64,3,2.64003,2.63699,12582912,64684032
sample,complete_layer,128,64,4,2.64323,2.63918,12582912,64684032
sample,recurrence,128,64,0,0.129856,0.133544,12582912,30081024
sample,recurrence,128,64,1,0.135136,0.138122,12582912,30081024
sample,recurrence,128,64,2,0.129152,0.132913,12582912,30081024
sample,recurrence,128,64,3,0.135904,0.138514,12582912,30081024
sample,recurrence,128,64,4,0.129184,0.132382,12582912,30081024
sample,complete_layer,128,128,0,2.6328,2.62986,6291456,64684032
sample,complete_layer,128,128,1,2.6375,2.63438,6291456,64684032
sample,complete_layer,128,128,2,2.63699,2.63414,6291456,64684032
sample,complete_layer,128,128,3,2.6401,2.63726,6291456,64684032
sample,complete_layer,128,128,4,2.64058,2.63748,6291456,64684032
sample,recurrence,128,128,0,0.13184,0.135627,6291456,30081024
sample,recurrence,128,128,1,0.127104,0.130699,6291456,30081024
sample,recurrence,128,128,2,0.13552,0.138623,6291456,30081024
sample,recurrence,128,128,3,0.127872,0.13224,6291456,30081024
sample,recurrence,128,128,4,0.133056,0.135087,6291456,30081024
sample,complete_layer,256,1,0,5.51414,5.5111,1610612736,64684032
sample,complete_layer,256,1,1,3.89325,3.89022,1610612736,64684032
sample,complete_layer,256,1,2,3.89203,3.88971,1610612736,64684032
sample,complete_layer,256,1,3,3.89318,3.89031,1610612736,64684032
sample,complete_layer,256,1,4,3.89005,3.88709,1610612736,64684032
sample,recurrence,256,1,0,0.7192,0.723215,1610612736,30081024
sample,recurrence,256,1,1,0.71392,0.717974,1610612736,30081024
sample,recurrence,256,1,2,0.716128,0.720197,1610612736,30081024
sample,recurrence,256,1,3,0.715648,0.719267,1610612736,30081024
sample,recurrence,256,1,4,0.720128,0.723415,1610612736,30081024
sample,complete_layer,256,64,0,3.5241,3.52062,25165824,64684032
sample,complete_layer,256,64,1,3.52198,3.51711,25165824,64684032
sample,complete_layer,256,64,2,3.51347,3.51048,25165824,64684032
sample,complete_layer,256,64,3,3.52253,3.51956,25165824,64684032
sample,complete_layer,256,64,4,3.516,3.51299,25165824,64684032
sample,recurrence,256,64,0,0.263264,0.265855,25165824,30081024
sample,recurrence,256,64,1,0.250496,0.254724,25165824,30081024
sample,recurrence,256,64,2,0.257728,0.260746,25165824,30081024
sample,recurrence,256,64,3,0.258944,0.260615,25165824,30081024
sample,recurrence,256,64,4,0.252256,0.256217,25165824,30081024
sample,complete_layer,256,128,0,3.51466,3.51092,12582912,64684032
sample,complete_layer,256,128,1,3.51178,3.50876,12582912,64684032
sample,complete_layer,256,128,2,3.51613,3.51202,12582912,64684032
sample,complete_layer,256,128,3,3.51888,3.51603,12582912,64684032
sample,complete_layer,256,128,4,3.52275,3.51973,12582912,64684032
sample,recurrence,256,128,0,0.258496,0.262479,12582912,30081024
sample,recurrence,256,128,1,0.256736,0.259853,12582912,30081024
sample,recurrence,256,128,2,0.25808,0.260706,12582912,30081024
sample,recurrence,256,128,3,0.249824,0.253903,12582912,30081024
sample,recurrence,256,128,4,0.2576,0.261697,12582912,30081024
sample,complete_layer,256,256,0,3.51366,3.5092,6291456,64684032
sample,complete_layer,256,256,1,3.51424,3.51128,6291456,64684032
sample,complete_layer,256,256,2,3.51424,3.51033,6291456,64684032
sample,complete_layer,256,256,3,3.52406,3.52022,6291456,64684032
sample,complete_layer,256,256,4,3.51677,3.5127,6291456,64684032
sample,recurrence,256,256,0,0.260512,0.264613,6291456,30081024
sample,recurrence,256,256,1,0.258016,0.261336,6291456,30081024
sample,recurrence,256,256,2,0.25328,0.257449,6291456,30081024
sample,recurrence,256,256,3,0.258432,0.262009,6291456,30081024
sample,recurrence,256,256,4,0.260576,0.26338,6291456,30081024
```
