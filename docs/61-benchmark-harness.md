# The benchmark harness

[Index](README.md) · Implementation tasks: BEN-001, OPT-032, OPT-038, OPT-056, EDU-046, and SCH-002 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contracts:
[`pins/benchmark_contract.json`](../pins/benchmark_contract.json),
[`pins/opt032_decode_oracle_contract.json`](../pins/opt032_decode_oracle_contract.json),
[`pins/opt038_post_ladder_gap_contract.json`](../pins/opt038_post_ladder_gap_contract.json)
· Evidence:
[`fixtures/benchmark_harness.json`](../fixtures/benchmark_harness.json),
[`evidence/benchmark`](../evidence/benchmark),
[`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json),
[`fixtures/opt038_post_ladder_gap.json`](../fixtures/opt038_post_ladder_gap.json), and
[`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md),
[`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](../evidence/optimization/opt038-post-ladder-gap/REPORT.md)

## What a benchmark is

A test usually asks a yes-or-no question: did two token histories remain equal,
or did a malformed request fail? A **benchmark** measures how long and how much
memory a successful operation used. A profiler goes deeper and attributes time
to kernels or gaps. A comparison experiment runs the same controlled benchmark
against other engines. `qw38-bench` creates trustworthy Quartz measurements; it
does not, by itself, prove that Quartz is faster than another runtime.

One invocation measures one named **workload** and writes one JSON result. A
**run** is one complete execution of that workload. A **warm-up** is a run whose
result is retained but excluded from the summary; it lets lazy initialization,
GPU clocks, and caches settle. A **sample** is a retained run included in the
summary. Release-admission mode requires at least three warm-ups and 30 samples.
The smaller `--smoke` mode answers only “does this measurement path work?” and
is never admission evidence.

## Prefill, decode, and the latency names

**Prefill** processes the rendered prompt tokens and prepares all state needed
for continuation. Prompt throughput is

`evaluated prompt tokens / prefill seconds`.

**Decode** first prefills, then repeatedly samples one token and evaluates it to
produce the next logits. Output throughput is

`generated tokens / generation seconds`.

**Time to first token (TTFT)** is the interval from the beginning of prompt
processing through selection of the first output token. It therefore includes
prefill. **Inter-token latency (ITL)** is the time between later output tokens.
Lower TTFT and ITL are better; higher tokens per second is better. Quartz keeps
per-token wall times and generated token IDs, so a summary can be audited.

The release matrix labels prompt contexts as 128, 2K, 8K, 32K, or 128K tokens.
The actual rendered token count must equal `--expected-prompt-tokens`; a label is
not accepted as proof. A 131,072-position session that generates 256 tokens can
start with at most 130,815 prompt tokens because the evaluated continuation also
needs positions. This boundary is explicit rather than silently truncating.

## Why repeated samples become p50 and p95

Measurements vary because clocks, temperature, operating-system scheduling, and
other effects vary. Quartz sorts the sample values. **p50**, the median, is the
middle of the distribution. **p95** is a tail value: roughly 95 percent of
samples are no worse. With sorted values `[1, 2, 3, 4, 5]`, Quartz's linear
percentile rule gives p50 = 3 and p95 = 4.8. Thirty samples are still a modest
minimum, so the later comparative gate also uses paired bootstrap confidence
intervals; BEN-001 does not implement that comparison.

Every warm-up and sample stays in the raw arrays (`warmups` and `samples` in
the file), even though readers will usually begin with `summary`. A failed run
also produces a result with its explicit error. Keeping failures prevents an
out-of-memory or throttled trial from disappearing merely because it was
inconvenient.

## Cache policy and agent reuse

The primary comparison uses cache policy `disabled`. Before every run, Quartz
synchronizes the session to an empty token history, then evaluates the complete
prompt. This prevents an engine with a warm prefix from being compared with one
doing cold work.

The separately labelled `agent-reuse` workload models a continuing conversation.
Its next prompt begins with the exact committed token history and appends a new
turn. The raw record separates `reused_prefix_tokens` from
`evaluated_prompt_tokens`: reused tokens cost no model execution, while the
suffix does. Agent reuse is useful product behavior but cannot be mixed into the
primary cold-prefill numbers.

## Timing without changing the primary measurement

CUDA work is asynchronous, so
[`51-runtime-timing-and-nvtx.md`](51-runtime-timing-and-nvtx.md) explains why
synchronized CUDA events are needed. Recording events around every model stage
also adds work and can change the timing. The harness therefore uses ordinary
wall times for its throughput samples and performs one separate
`component_probe`. That probe first prefills without category attribution, then
attributes one decode `sample`/`eval`. It exposes embedding, GDN, attention, FFN,
logits, sampling, graph-launch, state-commit, idle-gap, loading, queueing, and
persistence categories. It is marked `perturbs_execution: true` and
`used_for_throughput_summary: false`.

That decode-token probe is not a 2K prefill breakdown. The historical
eight-category prefill report and the later nine-category mixer versus core
split are separate opt-in `PrefillAttribution` records collected on production
`sync_tokens`, retained in
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json)
and [`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json).
Neither report is a BEN-001 `qw38-bench` result. Default `qw38-bench`
throughput samples stay unattributed wall times. Chapter 51 owns the live 2K
categories, remainder other/idle, and the no-Nsight proof boundary.

The 2026-09-08 scaling `llama-bench` JSON
[`evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json`](../evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json)
is a pinned same-GGUF citation. OPT-015 copies its first object for exact-2048
llama.cpp 2K (**Measured** mean 3114.049476 tok/s, `n_ubatch` 512,
`flash_attn` -1, `build_commit` `cc83d7b`). That file is not a BEN-001
`qw38-bench` result, not a new GPU sample in this increment, and not the later
2K throughput gate. CMP-002/CMP-003 still own the 30-sample comparative matrix.

OPT-021 is a frozen exclusive-RTX-5090 **4K keep/reject oracle**, not a
BEN-001 `qw38-bench` result and not a change to this harness or its JSON
schema. It compares three cold exact-4096 production `sync_tokens` walls
(attribution null, graphs created) to same-sitting pinned llama.cpp
`llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99`. That yardstick is not the
OPT-016 exact-2048 2K parity gate, and it is not the 2026-09-08 scaling
citation above (2K/8K/32K objects). Exact 4096 is the production FFN
graph-replay length; exact 2048 is not. A 2026-09-09 scout sitting under
`/tmp/oracle4k/` is contract transparency only and is **not** the retained
fixture. Live same-sitting numbers stay in
[`evidence/optimization/opt021-4k-oracle/REPORT.md`](../evidence/optimization/opt021-4k-oracle/REPORT.md).

OPT-032 is a frozen exclusive-RTX-5090 **P, D128, and D2048 oracle sitting**,
not a BEN-001 `qw38-bench` result and not a change to this harness or its JSON
schema. P reuses the 4K keep/reject protocol (exact 4096, attribution null,
graphs created, three cold replicates versus same-sitting `llama-bench -p 4096
-n 0 --no-warmup -r 3 -ngl 99`). D128 and D2048 are operational decode
steering oracles: prefix exactly 128 or 2048 committed tokens, then 256
predetermined one-token evaluations (no sampling), 3 warm-ups and 30 measured
runs, host `steady_clock` around each full production `execute_token`. The
matched llama.cpp denominator is `qw38-llama-decode-oracle MODEL.gguf PREFIX`
through pinned `llama.h` and `llama_time_us()` (`PREFIX` 128 or 2048). Random-
token `llama-bench -p 0 -n 256 -d 128` and `-d 2048` in the same sitting are
informational only; they are not the keep/reject denominator. BEN-001
`--smoke` decode (short chat-rendered prompts, two generated tokens, composite
`component_probe`) is inadmissible as D128/D2048. Public `RuntimeTimings` /
`component_probe` stay composite. This sitting **claims no performance
improvement**, does not substitute for the 2K llama.cpp parity gate, and does
not require Quartz ≥ llama.cpp. Live tok/s stay in
[`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md)
and [`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json).

OPT-038 is a frozen exclusive-RTX-5090 **post-ladder P, D128, and D2048 refresh**,
not a BEN-001 `qw38-bench` result and not a change to this harness or its JSON
schema. It reuses the OPT-021 P and OPT-032 D128/D2048 oracle executables
unchanged (`task` stays `OPT-021` / `OPT-032` on those binaries) and adds
separate OPT-038 attribution diagnostics with independent raw host-wall fields.
Live measurements are retained from one same-sitting pass; accepted keep
denominators remain the frozen OPT-034 copies (P **1869.84412**, D128
**25.3816128**, D2048 **20.169548** tok/s). This increment **claims no
performance improvement**, does not publish a successor oracle, does not
substitute for the 2K llama.cpp parity gate, and does not require Quartz ≥
llama.cpp. Random-token `llama-bench` decode stays informational. Recorded
`next_task_order` and matched-component experiment specifications steer later
transferable work; they are not keep/reject gates. Live tok/s, gaps, raw-wall
accounting, and the order stay in the report; this chapter does not replace
them:
[`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](../evidence/optimization/opt038-post-ladder-gap/REPORT.md)
and
[`fixtures/opt038_post_ladder_gap.json`](../fixtures/opt038_post_ladder_gap.json).

An unavailable measurement is JSON `null`, not zero. Zero would mean an event
was measured and took no representable time. Queue time is `null` because this
single-process harness has no HTTP queue. Persistence is `null` when no save or
restore occurs.

## Telemetry and its limits

**Telemetry** is contextual machine data collected alongside timings. Quartz
records GPU identity, driver and CUDA versions, power, temperature, clocks,
performance state, device memory, process resident memory (**RSS**), build flags,
container identity, kernel/host identity, source revision, and clean/dirty source
state. Prompt bytes are authenticated by SHA-256 and the complete rendered token
IDs are stored, so two runs can establish that they used the same input.

The NVIDIA readings are device-wide snapshots taken before and after a run, not
continuous samples. A short power or temperature peak can be missed. “Peak
device memory” is the largest observed snapshot. “Reserved device memory” is
currently the same `nvidia-smi` observation, not an allocator-internal reserved
byte counter. Those limitations are visible in the evidence rather than hidden
behind false precision.

Release mode also requires an explicit source revision and `clean` source state.
The container intentionally does not need a Git client: the caller passes the
identity it obtained outside the container. Smoke results may say `dirty`, but
then `admission_eligible` is false.

## Atomic result publication

The harness writes a temporary file, flushes it to stable storage, renames it to
the requested output path, and flushes the parent directory. This **atomic**
publication means readers see either the older complete result or the newer
complete result, never a half-written JSON document. Validation errors that
occur before a meaningful run configuration exists do not create a result;
runtime failures after validation are retained as failed results.

## Current measured result and proof boundary

**Measured smoke, RTX 5090:** the first BEN-001 smoke executions successfully
recorded prefill, decode, agent reuse, component attribution, and telemetry.
They observed about 16.2 prompt tokens/s. Inspection showed that, at BEN-001
completion, `Session::sync` invoked a complete one-token scheduler execution for
each prompt token. The already-admitted MMQ and chunked GDN/attention primitives
were not connected to the full scheduler. Task **SCH-002** recorded that missing
integration before the later optimization; chapter 62 documents its resolution
without rewriting this historical benchmark evidence.

This result proves that BEN-001 can execute workloads, enforce release
minimums, preserve raw successes and failures, summarize samples, distinguish
cache policies, and publish a self-describing result. It does not prove release
throughput, the full workload matrix, stable thermal conditions, comparative
speed, or statistical superiority. The checked-in smoke runs are explicitly
not admission eligible. CMP-002 and CMP-003 own the controlled comparison and
confidence gates after the remaining quality work is complete. OPT-015 cites
the scaling `llama-bench` 2K object as an independent same-GGUF denominator; it
does not change this harness, its JSON schema, or admit 2K llama.cpp parity.
OPT-021 likewise leaves this harness unchanged: the 4K keep/reject oracle is a
separate diagnostic sitting, not a BEN-001 workload, not the 2K parity gate,
and not an admission of Quartz ≥ llama.cpp. OPT-032 likewise leaves this
harness unchanged: the P/D128/D2048 oracles and the pinned `llama.h` decode
driver are separate diagnostics; random llama-bench decode is informational;
exclusive decode categories are not `component_probe` fields. OPT-038 likewise
leaves this harness unchanged: the post-ladder refresh reuses those oracle
protocols, adds independent raw-wall attribution diagnostics, and records a
host-recomputed next-task order without changing accepted keep denominators or
public `component_probe` fields. OPT-056 likewise leaves this harness unchanged:
it reuses the OPT-021 P and OPT-032 D128/D2048 protocols plus Session TTFT as a
secondary metric only. The same-sitting exclusive RTX 5090 outcome gate versus
pinned llama.cpp did not pass (P 2808.50 vs 3263.52, D128 37.48 vs 68.93, D2048
35.72 vs 67.34 tok/s). Live numbers stay in
[`evidence/optimization/opt056-performance-gate/REPORT.md`](../evidence/optimization/opt056-performance-gate/REPORT.md).

## Reproduce a smoke safely

Build the CUDA product, then run one short non-admission decode sample:

```sh
make cuda-build
revision="$(git rev-parse HEAD)"
state=clean
test -z "$(git status --porcelain)" || state=dirty
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" qw38-cuda:13.0.2 \
  ./build/cuda/qw38-bench models/Qwen3.8-27B-Q4_K_M.gguf \
  --workload decode --prompt "Quartz benchmark prompt." \
  --context-label smoke --output-tokens 2 --warmups 1 --samples 2 \
  --cache-policy disabled --source-revision "$revision" \
  --source-state "$state" --smoke \
  --output evidence/benchmark/local-decode-smoke.json
```

Expected result: exit status zero, `status: "success"`, raw warm-up/sample
arrays, a separate component probe, and `admission_eligible: false`. Do not cite
that file as a release or comparison measurement.
