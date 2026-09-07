# Stable-address CUDA graphs

[Index](README.md) · Implementation tasks: OPT-003, OPT-012, and EDU-039 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`cuda/full_scheduler.cu`](../cuda/full_scheduler.cu),
[`cuda/graph_test.cu`](../cuda/graph_test.cu),
[`cuda/prompt_graph_test.cu`](../cuda/prompt_graph_test.cu),
[`tests/test_cuda_graph.py`](../tests/test_cuda_graph.py),
[`tests/test_cuda_prompt_graph.py`](../tests/test_cuda_prompt_graph.py),
[`fixtures/cuda_graph.json`](../fixtures/cuda_graph.json), and
[`fixtures/cuda_prompt_graph.json`](../fixtures/cuda_prompt_graph.json)

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
`SchedulerWorkspace`. Decode always owns 64 graph definitions and 64 executable
instances. Production workspaces with `prompt_chunk_rows_ == 4096` add a second
64 prompt FFN graphs, for 128 executables total. Replay with another
model/workspace object fails before any kernel runs. The graph owner must also
be destroyed before its bound model or workspace.

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

Prompt graphs follow the same exclusion list. They record prompt-scratch and
weight pointers (`prompt_mixer_output_`, `prompt_normalized_`, prompt residuals,
and layer norms) instead of decode-scratch addresses. Mixer GDN/attention,
embedding token rows, attention position or causal extent, KV destination, host
output copies, and atomic commit stay ordinary launches around those FFN graph
launches.

## Prompt FFN graphs at 4,096 rows

OPT-012 captures the fused prompt FFN helper, not a whole prompt chunk. Each of
the 64 prompt graphs records mixer residual-plus-FFN-norm, gate/up MMQ, SwiGLU,
down MMQ, and FFN residual-plus-next-input-norm (layer 63 ends with a standalone
residual add). Mixer GDN/attention must finish writing `prompt_mixer_output_`
before replay, matching decode writing `mixer_output_` before an FFN graph
launch.

Capture is gated on `prompt_chunk_rows_ == 4096`. A capacity-33 decode diagnostic
therefore still owns only the original 64 decode graphs. Capacity-65 scheduler
workspaces likewise skip the prompt set. Replay requires `token_count == 4096`
on that same bound graph object. Tails with `2 <= token_count < 4096` keep
ordinary fused FFN launches. One-row remainders never enter the prompt chunk
path; they replay the existing decode FFN graphs.

Mismatch is fail-closed: graphs bound to another workspace, or graphs combined
with the unfused serial path, are rejected before any publication. Cancelling a
4,096-row graph chunk still stream-synchronizes after the finished layer,
including after a graph replay, then polls; a stop at poll 8 publishes nothing.

**Measured, RTX 5090:** a capacity-4096 session owns 128 graphs
(`decode_graph_count = 64`, `prompt_graph_count = 64`, `prompt_graph_rows =
4096`) and a combined CUDA free-memory delta of 16,777,216 bytes. The captured
prompt FFN helper contains nine kernel nodes, including a row-wise
residual-add-norm grid of `4096 × 256`. A 4,096-row fused graph chunk and a
4,096-row ordinary fused chunk were **byte-exact** in committed GDN/KV, tokens,
frontier, last hidden, and logits. CUDA-event times were 89900.2422 ms (graph)
and 90766.3672 ms (ordinary); there is no speedup gate. A 64-row fused fallback
on the same 4096-bound object was also byte-exact. Successful graph counters
show 64 prompt graph launches; the paired ordinary fused chunk shows zero prompt
graph launches, 127 fused residual-add-norm launches, and one last residual add.

## Graph creation and memory

Graph creation uses a non-blocking capture stream. After each capture is
instantiated, `cudaGraphUpload` prepares the executable. CUDA free-memory is
measured before capture and after all uploads with the objects still alive.

**Measured:** 64 FFN graph executables use 6,291,456 bytes, exactly **6 MiB**, on
the pinned RTX 5090/CUDA 13.0.2 environment. This is the historical decode-only
OPT-003 figure; capacity-33 diagnostics still report those 64 graphs. Prompt
graphs are additional CUDA graph objects, not extra `cudaMalloc` scratch. The
combined 128-executable owner is measured in [Chapter 54](54-post-graph-128k-memory.md)
and [`fixtures/cuda_prompt_graph.json`](../fixtures/cuda_prompt_graph.json).
This is device/driver graph memory; the process RSS is reported separately by
the final memory ledger.

Move-only ownership prevents accidental double destruction. Partial capture,
instantiate, or upload failure destroys every graph created so far and returns
an explicit status. No half-created graph set becomes replayable.

## Replay and attribution

Within a decode layer, the GDN or attention mixer runs first on the default
stream. `cudaGraphLaunch` then queues that layer's FFN graph on the same stream,
so CUDA stream ordering preserves the decoder equation. The existing FFN CUDA
events measure GPU execution. `RuntimeTimings.graph_launch` separately
accumulates the CPU time spent submitting the 64 graph launches; in the
diagnostic detailed run that value was 0.149828002 ms.

Prompt FFN replay uses the same stream-order rule on `prompt_compute_stream_`:
mixer writes finish, then that layer's prompt executable launches. Capture used
a non-blocking capture stream; instantiated graphs may launch on a different
stream than the capture stream.

The ordinary accepted fused path remains compiled. Decode launches the same FFN
kernels individually. Prompt equality is graph versus ordinary fused FFN, not a
new unfused gate. Those ordinary fused launches are the graph equivalence
oracle.

## Equality and paired measurements

The decode diagnostic runs both paths through the same 33-token history,
alternating which path runs first. It compares complete host logits, final
hidden values, selected trace taps, committed GDN/KV state, tokens/frontier, and
greedy output. Every boundary was **byte-exact**. The prompt diagnostic compares
one 4,096-row fused graph chunk against one 4,096-row ordinary fused chunk on
separate capacity-4096 sessions; committed state, last hidden, logits, and
frontier 4096 were likewise **byte-exact**.

After three warm-ups, 30 paired samples averaged 65.5711594 ms for graph replay
and 65.9345779 ms for ordinary launches, a 1.0055424× local improvement. Two
replicates measured 1.00516605× and 1.00516462×. All raw samples remain in the
fixture. This small result is retained rather than rounded into a larger claim.

## Failure modes

- Replaying against different object addresses is rejected before mutation.
- Graph creation on an uninitialized model/workspace is rejected.
- The graph path accepts only the admitted fused pointwise path. Prompt graphs
  plus the unfused serial path fail closed before publication.
- Capture/instantiate/upload/launch errors return explicit status. Partial
  failure of either the decode or prompt loop destroys every graph created so
  far; no half-created set is replayable.
- Graph timing is unavailable on ordinary execution; unavailable is not zero.
- Destroying bound allocations before graph ownership is a caller lifetime bug;
  the engine owner must enforce graph-first destruction.

## Proof boundary

OPT-003 proves stable capture, instantiation, upload, replay, ownership checks,
graph-launch attribution, graph/non-graph byte equality, measured graph memory,
and a modest local A/B improvement for the FFN boundary. It does not prove a
whole-token graph, dynamic attention capture, product throughput, graph prefill,
or comparative speed. OPT-004 tunes dispatch; BEN/CMP own formal benchmarks.

OPT-012 proves the second 64-graph prompt FFN set on production 4,096-row
workspaces: capture gated on `prompt_chunk_rows_ == 4096`, replay only at
`token_count == 4096`, fail-closed mismatch and unfused use, graph-versus-ordinary
fused byte equality, 64-row ordinary fallback, poll-8 cancellation, and the
128-graph memory owner. It does not prove a whole-chunk graph, CUDA graph
parameter updates, Nsight Systems overlap, end-to-end prefill/decode speedup, or
128K quality. Equality is graph versus ordinary fused FFN, not a new unfused
gate. QLT-001 remains blocked.
