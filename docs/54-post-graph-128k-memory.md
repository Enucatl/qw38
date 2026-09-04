# Final post-graph 128K memory ledger

[Index](README.md) · Implementation tasks: MEM-001, MEM-002, and EDU-036 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`cuda/memory_fit_test.cu`](../cuda/memory_fit_test.cu),
[`tests/test_cuda_memory_fit.py`](../tests/test_cuda_memory_fit.py), and
[`fixtures/cuda_memory_fit_post_graph.json`](../fixtures/cuda_memory_fit_post_graph.json)

## What changed after the provisional gate

[Chapter 50](50-pre-graph-128k-memory.md) proved the model, complete 131,072-token
session, and workspace fit together, but honestly kept MEM-001 in progress
because CUDA graphs did not exist. OPT-003 now creates and uploads all 64 FFN
graph executables before the final free-memory reading. This chapter closes that
specific remaining condition; it does not rewrite the earlier measurement.

## Simultaneous owners

### GiB versus GB

Memory vendors and operating tools do not always use the same unit. One GB is
1,000,000,000 bytes, while one GiB is 1,073,741,824 bytes (2^30). Quartz records
raw bytes so the result is unambiguous, then uses GiB only when describing the
binary-sized 8 GiB KV cache and 1.5 GiB reserve.

All byte values below were live at the same time on the 32 GiB RTX 5090:

| Owner | Bytes | Meaning |
|---|---:|---|
| CUDA runtime context | 531,890,176 | Device memory used before Quartz owners |
| Resident model (GGUF) | 18,973,870,432 | One admitted model copy |
| GDN state | 158,859,264 | 48 FP32 convolution/recurrent session states |
| Attention K/V | 8,589,934,592 | 16 BF16 K and V caches at 131,072 rows |
| Complete session | 8,748,793,856 | GDN plus attention state |
| Atomic workspace | 198,882,816 | Decode candidates plus fixed 64-row prompt scratch, scores, projections, logits, taps |
| 64 uploaded graphs | 6,291,456 | Measured graph executable allocation |
| Explicit Quartz total | 27,927,838,560 | Model + session + workspace + graphs |
| Allocator delta | 12,517,536 | CUDA delta not assigned to explicit owners |

The graph owner reports its own before/after CUDA free-memory delta, and the
outer ledger observes the same 6 MiB change. This cross-check prevents a graph
object from being counted as a zero-byte host abstraction.

Before graph creation, 5,205,393,408 bytes were free. After uploading the graph
executables, 5,199,101,952 bytes were free; their difference is the measured
6,291,456-byte graph allocation.

## Reserve result

The device exposes 33,671,348,224 total bytes. With every owner and graph alive,
5,199,101,952 bytes remained free. The required reserve is 1,610,612,736 bytes
(1.5 GiB), leaving a margin of **3,588,489,216 bytes**.

Process resident host memory measured 19,114,557,440 bytes. Host RSS is not VRAM
and is not added to the GPU total, but it remains visible for operational sizing.

The arithmetic gate verifies session capacity, the independent 8 GiB KV formula,
all explicit owner counters, 64 graph objects, the equal inner/outer graph
allocation delta, allocator reconciliation, and the reserve comparison.

## Admission and proof boundary

**Measured:** MEM-001 and the prompt-scratch revalidation MEM-002 are admitted
on the pinned artifact, CUDA 13.0.2, and RTX 5090: 131,072-token capacity plus
all current graph/workspace/model/session
owners leaves more than the required reserve after graph creation.

This is physical allocation evidence. It does not execute a 131,072-token model
prefill, prove retrieval quality, measure long-context speed, or include future
server buffers not in the engine allocation contract. QLT-001 owns 128K retrieval
quality, while BEN/CMP own performance. New persistent device owners require a
new ledger measurement; this result cannot silently cover them.
