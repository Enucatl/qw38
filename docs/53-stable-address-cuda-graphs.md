# Stable-address CUDA graphs

[Index](README.md) · Implementation tasks: OPT-003 and EDU-039 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`cuda/full_scheduler.cu`](../cuda/full_scheduler.cu),
[`cuda/graph_test.cu`](../cuda/graph_test.cu),
[`tests/test_cuda_graph.py`](../tests/test_cuda_graph.py), and
[`fixtures/cuda_graph.json`](../fixtures/cuda_graph.json)

## The launch problem

A normal CUDA token asks the driver to launch hundreds of kernels in a fixed
order. Most kernels do useful GPU work, but the CPU and driver must submit each
one. A CUDA graph stores a dependency schedule so it can later submit that
schedule with one `cudaGraphLaunch` call.

There are four distinct steps:

1. **Capture** records CUDA operations issued to a capture stream.
2. **Instantiate** validates the recorded graph and creates a replayable graph
   executable.
3. **Upload** prepares that executable on the target GPU before timed work.
4. **Replay** launches the already prepared executable on a CUDA stream.

These steps do not make arithmetic faster. They reduce repeated submission work.

## What “stable address” means

Captured kernel arguments contain pointers. If a graph records “read weights at
address A and write scratch at address B,” replay must still find the same
objects at A and B. A **stable address** is a pointer whose allocation and role
remain unchanged throughout graph lifetime.

`SchedulerGraphs` binds itself to one exact `ResidentModel` and
`SchedulerWorkspace`. It owns 64 graph definitions and 64 executable instances.
Replay with another model/workspace object fails before any kernel runs. The
graph owner must also be destroyed before its bound model or workspace.

## Why V1 graphs the FFNs, not the whole token

The accepted OPT-002 FFN branch has stable inputs and outputs:

- one layer's fixed norm and projection weights;
- fixed residual, normalization, projection, activation, and correction buffers;
- the fixed fused residual/next-layer-normalization operation.

Quartz captures one graph for each of the **64** decoder-layer FFNs. Each graph
contains input norm, gate/up projections, SwiGLU, down projection, and the fused
residual boundary. The final layer graph ends with residual add because there is
no next decoder input norm.

A whole-token graph is not yet honest. Attention uses a **dynamic** position for
RoPE, causal loop extent, and K/V destination. GDN committed/candidate owners
swap only after atomic success. The embedding row changes with the input token,
and host output copies plus cancellation polling cross the transaction boundary.
Capturing those values once would replay stale token/session information.

Those operations remain ordinary launches around the FFN graph launches. This
is why the implementation and evidence say “FFN graphs,” never “whole-token
graph.” A later design could add device-side control blocks and graph parameter
updates, but that work is unavailable in V1 until separately admitted.

## Graph creation and memory

Graph creation uses a non-blocking capture stream. After each capture is
instantiated, `cudaGraphUpload` prepares the executable. CUDA free-memory is
measured before capture and after all uploads with the objects still alive.

**Measured:** 64 FFN graph executables use 6,291,456 bytes, exactly **6 MiB**, on
the pinned RTX 5090/CUDA 13.0.2 environment. This is device/driver graph memory;
the process RSS is reported separately by the final memory ledger.

Move-only ownership prevents accidental double destruction. Partial capture,
instantiate, or upload failure destroys every graph created so far and returns
an explicit status. No half-created graph set becomes replayable.

## Replay and attribution

Within a layer, the GDN or attention mixer runs first on the default stream.
`cudaGraphLaunch` then queues that layer's FFN graph on the same stream, so CUDA
stream ordering preserves the decoder equation. The existing FFN CUDA events
measure GPU execution. `RuntimeTimings.graph_launch` separately accumulates the
CPU time spent submitting the 64 graph launches; in the diagnostic detailed run
that value was 0.149828002 ms.

The ordinary accepted fused path remains compiled. It launches the same FFN
kernels individually and is the graph equivalence oracle.

## Equality and paired measurements

The diagnostic runs both paths through the same 33-token history, alternating
which path runs first. It compares complete host logits, final hidden values,
selected trace taps, committed GDN/KV state, tokens/frontier, and greedy output.
Every boundary was **byte-exact**.

After three warm-ups, 30 paired samples averaged 65.5711594 ms for graph replay
and 65.9345779 ms for ordinary launches, a 1.0055424× local improvement. Two
replicates measured 1.00516605× and 1.00516462×. All raw samples remain in the
fixture. This small result is retained rather than rounded into a larger claim.

## Failure modes

- Replaying against different object addresses is rejected before mutation.
- Graph creation on an uninitialized model/workspace is rejected.
- The graph path accepts only the admitted fused pointwise path.
- Capture/instantiate/upload/launch errors return explicit status.
- Graph timing is unavailable on ordinary execution; unavailable is not zero.
- Destroying bound allocations before graph ownership is a caller lifetime bug;
  the engine owner must enforce graph-first destruction.

## Proof boundary

OPT-003 proves stable capture, instantiation, upload, replay, ownership checks,
graph-launch attribution, graph/non-graph byte equality, measured graph memory,
and a modest local A/B improvement for the FFN boundary. It does not prove a
whole-token graph, dynamic attention capture, product throughput, graph prefill,
or comparative speed. OPT-004 tunes dispatch; BEN/CMP own formal benchmarks.
