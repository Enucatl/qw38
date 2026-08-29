# 13. V1 allocation ledger

[Index](README.md)

This is the pre-implementation device-memory budget for BLD-003 and MEM-001.
Every value is **Proposed** or **Estimated** until the CUDA allocator emits the
corresponding measured record. The runtime must count actual allocator sizes,
not substitute this table for measurement.

| Allocation | Formula or source | Budget | Lifetime |
|---|---|---:|---|
| pinned Q4_K_M GGUF | pinned artifact byte size | 17.670 GiB | engine |
| attention KV | `65536 * 131072` | 8.000 GiB | session |
| GDN recurrence | `48 * 48 * 128 * 128 * 4` | 144.000 MiB | session |
| GDN convolution rings | `48 * 10240 * 4 * 4` | 7.500 MiB | session |
| FP32 logits | `248320 * 4` | 0.947 MiB | session |
| mandatory free reserve | delivery-gate minimum | 1.500 GiB | always free |
| repacks | not yet measured | unallocated | engine |
| activation workspaces | not yet measured | unallocated | eval |
| graph pools | not yet measured | unallocated | engine/session |
| CUDA allocator/driver overhead | not yet measured | unallocated | process |

The known fixed allocations plus reserve total approximately 27.32 GiB, leaving
about 4.52 GiB against the card's reported 32,607 MiB for repacks, workspaces,
graphs, and allocator overhead. That remainder is not proof of fit: the GGUF
file size is not necessarily the resident CUDA layout, and alignment or repacks
can increase it. Conversely, memory mapping does not imply that all file bytes
are device-resident. BLD-003 remains open until code declares each allocation;
MEM-001 remains open until a post-graph 131,072-token run measures at least 1.5
GiB free.

Failure is closed: allocation categories may not draw from the mandatory reserve,
and an unknown or unlogged persistent allocation invalidates the fit gate.

Evidence: model dimensions in [`pins/model_contract.json`](../pins/model_contract.json),
artifact bytes in [`pins/artifacts.lock.json`](../pins/artifacts.lock.json), and
ledger tasks BLD-003/MEM-001 in
[`implementation_ledger.md`](../implementation_ledger.md).
