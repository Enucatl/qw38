# The benchmark harness

[Index](README.md) · Implementation tasks: BEN-001, EDU-046, and SCH-002 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contract:
[`pins/benchmark_contract.json`](../pins/benchmark_contract.json) · Evidence:
[`fixtures/benchmark_harness.json`](../fixtures/benchmark_harness.json) and
[`evidence/benchmark`](../evidence/benchmark)

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
`component_probe`. That probe exposes embedding, GDN, attention, FFN, logits,
sampling, graph-launch, state-commit, idle-gap, loading, queueing, and persistence
categories. It is marked `perturbs_execution: true` and
`used_for_throughput_summary: false`.

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
confidence gates after the remaining quality work is complete.

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
