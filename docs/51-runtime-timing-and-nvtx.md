# Synchronized runtime timing and NVTX attribution

[Index](README.md) · Implementation tasks: OPT-001, OPT-014, and EDU-037 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contracts:
[`pins/cuda_timing_contract.json`](../pins/cuda_timing_contract.json),
[`pins/cuda_prefill_attribution_contract.json`](../pins/cuda_prefill_attribution_contract.json)
· Evidence:
[`cuda/timing_test.cu`](../cuda/timing_test.cu),
[`cuda/prefill_attribution_test.cu`](../cuda/prefill_attribution_test.cu),
[`tests/test_cuda_timing.py`](../tests/test_cuda_timing.py),
[`tests/test_cuda_prefill_attribution.py`](../tests/test_cuda_prefill_attribution.py),
[`fixtures/cuda_timing.json`](../fixtures/cuda_timing.json), and
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json)

## Why a normal stopwatch is misleading

A CUDA launch is **asynchronous**: the CPU asks the GPU to run a kernel and can
continue before that kernel finishes. A CPU stopwatch around only the launch
therefore measures how quickly the request was queued, not how long the GPU did
the work. **Synchronization** means waiting until a known GPU marker completes.
Only then is the measured interval finished.

Quartz uses two clocks for two different jobs:

- A **CUDA event** is a timestamp placed into the GPU stream. Two synchronized
  events measure elapsed GPU-stream time around embedding, mixers, FFNs, logits,
  and device-state commit.
- A monotonic **CPU clock** measures host-only operations such as scanning logits
  for greedy sampling and saving a checkpoint. “Monotonic” means the clock only
  advances, so a wall-clock correction cannot produce a negative interval.

These numbers must not be casually added across different requests. BEN-001 now
supplies warm-ups, repeated samples, percentile reporting, and controlled
machine identity as explained in
[`61-benchmark-harness.md`](61-benchmark-harness.md). That harness keeps
throughput samples as ordinary wall times and uses a separate decode-token
`component_probe`; it is not the 2K prefill attribution report. Controlled
cross-runtime comparison remains CMP-002/CMP-003 work.

## What NVTX contributes

NVTX, the NVIDIA Tools Extension API, adds named ranges such as `qw38.gdn` and
`qw38.ffn` to a profiler timeline. A range is a label, not a stopwatch and not a
kernel. It lets a human connect CUDA launches and gaps back to an engine stage.
The implementation places ranges around loading, the complete token, embedding,
each GDN or attention mixer, each FFN, logits, state commit, sampling, and
checkpoint save/restore.

CUDA events answer “how long was this stream interval?” NVTX answers “which
logical engine stage scheduled this work?” Prompt work records those events on
the fused prompt compute stream, not the default stream used by decode. Nsight
Systems is the later timeline tool, while Nsight Compute inspects selected
kernel hardware counters. Live prefill attribution is produced from the timed
`sync_tokens` run itself without requiring a separate Nsight capture.

## The attribution record

`RuntimeTimings` in [`cuda/full_scheduler.h`](../cuda/full_scheduler.h) gives each
value both milliseconds and a `measured` flag. The flag matters: `0 ms` means a
real measurement rounded to zero, while **unavailable** means no such runtime
boundary exists yet.

| Category | V1 ownership at OPT-001 |
|---|---|
| loading | Copy the admitted GGUF bytes into resident GPU storage |
| embedding | Decode one token row and convert it to the FP32 residual |
| GDN | All 48 GDN mixer intervals, accumulated |
| attention | All 16 attention mixer intervals, accumulated |
| FFN | All 64 feed-forward intervals, accumulated |
| logits | Final norm, vocabulary projection, trace copy, and candidate host copies |
| sampling | Host greedy scan of committed logits |
| persistence | Atomic checkpoint save or restore wall time |
| state commit | Copy 16 candidate K/V rows into committed cache storage |
| idle gaps | Token stream span not covered by the named GPU categories |
| graph launch | Unavailable until OPT-003 implements graphs |
| queueing | Unavailable in this scheduler-local record; SRV-001 measures the HTTP gate separately |

The token total starts before embedding and ends after device commit. `idle gaps`
is the non-negative remainder after subtracting embedding, GDN, attention, FFN,
logits, and commit. It includes gaps between recorded stream work; it is not a
claim that the whole GPU was idle.

## Live 2K prefill attribution

`PrefillAttribution` sits beside `RuntimeTimings` and does not rewrite decode
field ownership. It is opt-in: `execute_prompt_chunk` and `sync_tokens` take an
optional trailing pointer that defaults to null, and a null pointer creates no
extra CUDA events on the prompt path. When the pointer is set, the same
production 2048-token `sync_tokens` call is the timed run. Categories come from
synchronized CUDA events recorded on the prompt compute stream, plus a host
monotonic clock around graph launch and the whole evaluated envelope.

| Category | 2K prefill ownership |
|---|---|
| embedding | Host-to-device token copy plus batched embedding decode/widen into the FP32 residual |
| GDN | Mixer RMSNorm when the layer is GDN, GDN MMQ projections, gate prep, parallel or sequential scan, gated output, and GDN output projection |
| attention | Mixer RMSNorm when the layer is attention, Q/K/V MMQ, split, tiled chunk attention, and attention output projection |
| FFN/MMQ | Fused residual-add-norm, SwiGLU, the three FFN MMQ projections, last-layer residual add, or the ordinary fused prompt-FFN helper that replaces them |
| logits | Final RMSNorm of the last residual row and vocabulary projection on the compute stream |
| commit/sync | Compute-done event, KV scatter, and compute-stream join through `cudaStreamSynchronize` on the fused path |
| graph | Host monotonic time around each `cudaGraphLaunch` of a prompt FFN executable. At 2048 tokens this is measured `0` with zero launches, not unavailable: a 2048-row chunk does not replay the 4096-row prompt FFN graphs |
| other/idle | Non-negative remainder `max(0, wall − exclusive named work)`. Overlapped copy-stream D2H that extends wall time lands here; it is not a ninth GPU interval |

`wall` is the host steady clock around the evaluated `sync_tokens` work, not
model upload or graph create. The eight named categories reconstruct that wall
by construction of `other/idle`. Contract and tests use `rel_tol = 1e-4` and
`abs_tol_ms = 0.05`. If exclusive GPU-plus-graph time exceeds wall beyond that
tolerance, the diagnostic fails closed rather than clamping a negative remainder.
`0 ms` with a successful measured flag means empty work, which is the expected
graph result at 2048 tokens.

Prompt NVTX ranges are labels, not stopwatches. The path always emits
`qw38.prefill_chunk`, `qw38.embedding`, `qw38.gdn` or `qw38.attention`,
`qw38.ffn`, `qw38.logits`, and `qw38.state_commit`, plus nested
`qw38.graph_launch` only when a prompt graph actually launches.

## Measured example and profiler limits

**Measured, RTX 5090:** one position-1 diagnostic token took 60.594017 ms. Its
largest category was FFN at 39.8940468 ms, followed by GDN at 13.7372789 ms,
attention at 4.34175968 ms, logits at 2.43088007 ms, and smaller embedding,
commit, and gap intervals. The attributed sum equals the token span. This single
sample identifies where to investigate; it is not a throughput benchmark.

**Measured, RTX 5090, 2048-token cold prefill:** the retained live report in
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json)
was produced by the timed diagnostic at 2026-09-08T10:46:11Z. One empty-session
2048-token production `sync_tokens` (capacity 131072, one 2048-row chunk) took
41963.8828 ms host wall (~48.80 tok/s). Exclusive compute-stream categories were
FFN/MMQ 32294.9512 ms, GDN 5300.55371 ms, attention 4364.3335 ms, logits
2.86684799 ms, commit/sync 0.576767981 ms, and embedding 0.076063998 ms. Graph
was measured 0 ms with `prompt_graph_launches == 0`. The other/idle remainder was
0.5234375 ms. The eight categories reconstruct wall within the documented
tolerance. Nsight Systems and Nsight Compute were `not_used`. This is
instrumentation of that timed run, not a throughput gate and not llama.cpp
parity.

**Measured negative result:** the pinned CUDA 13.0.2 image contains Nsight
Compute 2025.3.1, but the host denied performance-counter access with
`ERR_NVGPUCTRPERM`. Nsight Systems (`nsys`) is not installed in that image.
OPT-002 must retain these facts, obtain an admitted profiling environment, and
use profiler evidence before accepting a fusion. CUDA-event attribution still
works because it does not require privileged hardware counters. OPT-014's 2K
report is admitted without requiring a separate Nsight capture.

## Measurement overhead and failure rules

Detailed attribution creates and records many event pairs. Those markers
**perturb** the execution being measured, so the detailed record is opt-in. The
ordinary scheduler retains only its existing whole-compute timing. Optimization
benchmarks must quantify instrumentation overhead and use an appropriate mode.

If creating, recording, synchronizing, or reading an event fails, the diagnostic
returns an explicit status. A missing decode graph measurement or a queue value
outside this scheduler boundary is represented as unavailable, never fabricated
as zero. Prefill graph-at-2048 is measured empty work, not unavailable. Failed
prefill event operations return an error rather than a partial success report.
NVTX ranges are balanced on successful and error paths so a profiler
timeline does not accidentally swallow later work.

## Proof boundary

OPT-001 proves decode category exposure, synchronized ordering, NVTX range
placement, sampling/persistence timing, and explicit unavailable values on this
runtime. OPT-014 records live 2K prefill attribution from stream-aware CUDA
events plus a host-wall other/idle remainder, with graph-at-2048 measured zero,
without requiring a separate Nsight capture. Neither increment proves a fusion
is beneficial, provides an Nsight report, establishes p50/p95 request latency,
or passes the comparative speed gate. SRV-002 now exposes separately measured
queue depth/delay on Chat Completions responses. BEN-001 retains repeated engine
samples and keeps its unavailable queue field explicit; server queue experiments
and cross-runtime statistical comparison remain CMP work.
