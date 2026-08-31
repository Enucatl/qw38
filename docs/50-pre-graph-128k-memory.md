# 50. The pre-graph 128K memory ledger

[Index](README.md) · Implementation tasks: MEM-001 and EDU-036 in
[`implementation_ledger.md`](../implementation_ledger.md)

Supporting a 131,072-token context is partly a semantics problem and partly a
physical question: can every allocation exist at the same time on the 32 GiB
RTX 5090? This chapter records the first complete physical allocation. It is a
pre-graph milestone, so MEM-001 remains **in progress** until OPT-003 creates
CUDA graphs and the same measurement is repeated.

## GiB versus GB

Memory labels are easy to misread. Decimal **GB** uses one billion bytes.
Binary **GiB** uses `1,073,741,824` bytes. The required 1.5 GiB reserve is
therefore exactly `1,610,612,736` bytes. CUDA reports raw bytes; this ledger
keeps raw values authoritative and uses GiB only as a readable description.

The GPU reports `33,671,348,224` usable bytes, about 31.36 GiB. A marketed
“32 GB” or “32 GiB” label is not precise enough for an admission test.

## Exact owners

The diagnostic in [`memory_fit_test.cu`](../cuda/memory_fit_test.cu) creates a
CUDA context, uploads the **resident model**, constructs a capacity-131,072
session, and constructs the largest current workspace without releasing any
earlier owner.

| Owner | Bytes | Meaning |
|---|---:|---|
| CUDA runtime context | 531,890,176 | Memory already unavailable before Quartz allocations |
| Resident model | 18,973,870,432 | One exact canonical GGUF device copy |
| GDN state | 158,859,264 | 48 convolution rings plus 48 FP32 recurrent matrices |
| Attention KV | 8,589,934,592 | 16 layers × 131,072 rows × 1,024 BF16 values × key/value |
| Session total | 8,748,793,856 | GDN plus the complete **8 GiB** KV allocation |
| Workspace | 172,963,328 | Atomic GDN candidate, candidate KV rows, scores, projections, residuals, logits, taps |
| Allocator delta | 2,785,440 | Measured CUDA delta not represented by exact requested owner sizes |
| Graphs | unavailable | OPT-003 has not created graph objects yet |

The exact Quartz-owned sum is `27,895,627,616` bytes. CUDA's free-memory change
is `27,898,413,056` bytes. Their difference is the **allocator delta**: padding,
rounding, or bookkeeping visible to the measurement but not returned by the
owners' requested-byte counters. Recording both prevents a tidy source-level
sum from hiding real device consumption.

The process's measured host RSS is `19,106,787,328` bytes. Much of that is the
memory-mapped model being faulted into host pages for upload. Host memory is not
part of the GPU reserve, but it is operational evidence and must not disappear
from deployment planning.

## What physically passed

After every current allocation, CUDA reports **5,241,044,992 bytes free**. That
exceeds the **1.5 GiB** requirement by `3,630,432,256` bytes. Session capacity
is exactly 131,072, and the session's requested bytes equal the independently
calculated GDN plus KV sizes. Allocation and zero-initialization really ran on
the RTX 5090; this is not spreadsheet-only arithmetic.

This complements Chapter 44, which executed attention at final legal position
131,071 for one layer. It does not execute the complete 27B model at a filled
128K history; quality/retrieval and performance gates own those separate claims.

## Why MEM-001 is not done

The approved plan requires at least 1.5 GiB free **after graph creation**. CUDA
graphs can allocate executable metadata or pin addresses and lifetimes. Today
there is no production graph, so “graph bytes = zero” would confuse absence with
a measurement. The fixture instead stores `null`, the diagnostic prints
`pending_OPT-003`, and the implementation ledger keeps MEM-001 in progress.

When OPT-003 is complete, this same diagnostic must instantiate the admitted
stable-address graphs, add their measured category, and recheck the reserve.
The current 3.63-billion-byte margin is encouraging but is not permission to
skip that gate.

The authenticated measurement contract is
[`cuda_memory_fit_contract.json`](../pins/cuda_memory_fit_contract.json), and
the raw retained ledger is
[`cuda_memory_fit_pre_graph.json`](../fixtures/cuda_memory_fit_pre_graph.json).
The proof boundary is simultaneous pre-graph allocation and reserve on this
device/toolkit/model. It excludes post-graph MEM-001 admission, a full 128K
forward pass, 128K checkpoint timing, long-context retrieval quality, tuned
throughput, and behavior with another large GPU process present.
