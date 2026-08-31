# Synchronized runtime timing and NVTX attribution

[Index](README.md) · Implementation tasks: OPT-001 and EDU-037 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`cuda/timing_test.cu`](../cuda/timing_test.cu),
[`tests/test_cuda_timing.py`](../tests/test_cuda_timing.py), and
[`fixtures/cuda_timing.json`](../fixtures/cuda_timing.json)

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

These numbers must not be casually added across different requests. They become
comparable benchmark statistics only after BEN-001 supplies warm-ups, repeated
samples, percentile reporting, and controlled machine conditions.

## What NVTX contributes

NVTX, the NVIDIA Tools Extension API, adds named ranges such as `qw38.gdn` and
`qw38.ffn` to a profiler timeline. A range is a label, not a stopwatch and not a
kernel. It lets a human connect CUDA launches and gaps back to an engine stage.
The implementation places ranges around loading, the complete token, embedding,
each GDN or attention mixer, each FFN, logits, state commit, sampling, and
checkpoint save/restore.

CUDA events answer “how long was this stream interval?” NVTX answers “which
logical engine stage scheduled this work?” Nsight Systems is the intended tool
for the complete CPU/GPU timeline, while Nsight Compute inspects selected kernel
hardware counters. They solve different problems.

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
| queueing | Unavailable until SRV-001 implements the single-flight server queue |

The token total starts before embedding and ends after device commit. `idle gaps`
is the non-negative remainder after subtracting embedding, GDN, attention, FFN,
logits, and commit. It includes gaps between recorded stream work; it is not a
claim that the whole GPU was idle.

## Measured example and profiler limits

**Measured, RTX 5090:** one position-1 diagnostic token took 60.594017 ms. Its
largest category was FFN at 39.8940468 ms, followed by GDN at 13.7372789 ms,
attention at 4.34175968 ms, logits at 2.43088007 ms, and smaller embedding,
commit, and gap intervals. The attributed sum equals the token span. This single
sample identifies where to investigate; it is not a throughput benchmark.

**Measured negative result:** the pinned CUDA 13.0.2 image contains Nsight
Compute 2025.3.1, but the host denied performance-counter access with
`ERR_NVGPUCTRPERM`. Nsight Systems (`nsys`) is not installed in that image.
OPT-002 must retain these facts, obtain an admitted profiling environment, and
use profiler evidence before accepting a fusion. CUDA-event attribution still
works because it does not require privileged hardware counters.

## Measurement overhead and failure rules

Detailed attribution creates and records many event pairs. Those markers
**perturb** the execution being measured, so the detailed record is opt-in. The
ordinary scheduler retains only its existing whole-compute timing. Optimization
benchmarks must quantify instrumentation overhead and use an appropriate mode.

If creating, recording, synchronizing, or reading an event fails, the diagnostic
returns an explicit status. A missing graph or server queue is represented as
unavailable, never fabricated as zero. NVTX ranges are balanced on successful
and error paths so a profiler timeline does not accidentally swallow later work.

## Proof boundary

OPT-001 proves category exposure, synchronized ordering, NVTX range placement,
sampling/persistence timing, and explicit unavailable values on this runtime.
It does not prove a fusion is beneficial, provide an Nsight report, measure CUDA
graphs or server queueing, establish p50/p95 latency, or pass the comparative
speed gate. Those belong to OPT-002 through OPT-004, SRV-001, BEN-001, and CMP.
