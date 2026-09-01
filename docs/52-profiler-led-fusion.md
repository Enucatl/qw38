# Profiler-led pointwise fusion

[Index](README.md) · Implementation tasks: OPT-002 and EDU-038 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`cuda/full_scheduler.cu`](../cuda/full_scheduler.cu),
[`cuda/fusion_test.cu`](../cuda/fusion_test.cu),
[`tests/test_cuda_fusion.py`](../tests/test_cuda_fusion.py), and
[`fixtures/cuda_fusion.json`](../fixtures/cuda_fusion.json). The targeted raw
profiler summary is
[`evidence/profiling/opt002-nsight-compute.txt`](../evidence/profiling/opt002-nsight-compute.txt).

## What fusion means

A **kernel launch** is the CPU asking the GPU to execute one CUDA function. Even
small work pays launch and scheduling overhead. **Fusion** combines adjacent
operations into one kernel so an intermediate value can remain close to where it
was produced and one or more launches disappear.

Fusion is not automatically faster. A combined kernel can use more registers,
reduce occupancy, duplicate work, or serialize work that was parallel. It can
also change floating-point grouping. Quartz therefore keeps an **unfused** path:
the readable sequence that was already admitted. The **fused** production path
must compare with it at a visible boundary and win an A/B measurement.

## How the candidate was selected

OPT-001 showed that FFNs dominate one-token time, but most FFN time is matrix
multiplication. **Measured:** Nsight Compute classified a representative Q8 MMV
as balanced at 61.37% compute/memory throughput and 42.14% DRAM throughput. Its
74.98 µs profile did not justify rewriting the projection in this task, so that
candidate was rejected.

The separate 5,120-element input RMSNorm was different. Its grid contained one
block on a GPU with 170 SMs. Under Nsight Compute replay it reported 0.02%
compute throughput, 0.21% memory throughput, and 16.66% achieved occupancy on
the active SM. This does not mean the arithmetic is unnecessary; it means a
separate tiny launch is a plausible boundary to remove.

The host driver has `RmProfilingAdminOnly: 1`. Ordinary container execution and
container root both produced `ERR_NVGPUCTRPERM`. Adding the container capability
`SYS_ADMIN` admitted Nsight Compute 2025.3.1 without changing the host setting.
This privilege is needed only for profiling, not normal execution. Nsight
Systems is still absent from the pinned image, so no Systems timeline is claimed.

## The admitted boundary

Each decoder layer ends its FFN by adding a correction to the FP32 residual. The
next layer immediately normalizes that residual into BF16. The fused kernel:

1. performs the residual additions in parallel and writes the ordinary FP32
   layer residual;
2. computes the sum of squares in the same ordered FP32 loop as the unfused
   RMSNorm;
3. writes the next layer's scaled BF16 normalized input.

The next layer can consume that BF16 buffer directly. Layer zero still needs a
standalone input norm, and layer 63 still needs a standalone final residual add
because there is no next decoder layer. The change therefore removes **63**
launches per token, not 64. The residual remains visible for traces, checkpoint
state, cancellation semantics, and final output.

`PointwisePath::kUnfused` retains the old two-kernel sequence. Production uses
`PointwisePath::kFused` by default. This is a diagnostic switch, not a promise
of two separately supported production backends.

## A failed fusion that remains evidence

The first version assigned every residual addition and store to thread zero so
the complete operation used one ordered loop. It was **bit-exact**, but it
destroyed useful parallelism: 30 samples averaged 78.0371094 ms fused versus
63.2383232 ms unfused, a 0.81036222× result. That version was rejected.

The admitted version restores parallel residual writes while leaving only the
already-serial norm sum on thread zero. The residual additions and norm summation
order match the old kernels, so logits, final hidden values, selected trace taps,
complete session state, and greedy output are bit-exact.

## Paired A/B measurement

The diagnostic runs three warm-ups and 30 measured tokens for both paths. At
each position it executes both paths with identical token/state inputs and
alternates which path runs first. Alternation reduces a simple ordering bias from
temperature or clock drift. It records all **raw samples**, not only an average.

**Measured:** the retained run averaged 64.8529892 ms fused and 66.3453064 ms
unfused, a 1.02301073× speedup. Two additional independent runs measured
1.02286899× and 1.02247679× after an initial 1.02321708× run. The fused kernel's
replay profile was 45.54 µs; the separate norm replay profile was 159.39 µs.
Replay-instrumented kernel durations are not ordinary token latency, so the
paired CUDA-event samples—not subtraction of profiler durations—admit the win.

## Failure rules

- A faster path is rejected if any visible correctness boundary fails.
- A correct path is rejected if its measured mean is not lower.
- Profiler suggestions are hypotheses. The balanced-MMV and serialized-fusion
  negative results demonstrate that warnings and large categories are not
  automatic implementation instructions.
- Frozen scalar/oracle tolerances are never loosened to admit a fusion. This
  fusion is stronger: its retained boundaries are bit-exact.

## Proof boundary

OPT-002 proves one profiler-justified pointwise fusion on the pinned RTX 5090,
its retained unfused reference, exact boundary equality, and a repeated local
A/B improvement. It does not prove product throughput, prefill performance,
statistical comparative gates, a complete Nsight Systems timeline, or graph
speed. CUDA graphs remain OPT-003; dispatch tuning and formal product benchmarks
remain OPT-004 and BEN/CMP.
