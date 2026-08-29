# 7. Hybrid session state

[Previous](06-system-optimization.md) · [Index](README.md) · [Next](08-rtx-5090.md)

## Why this matters

Prefix reuse, batching, checkpointing, and CUDA Graphs all depend on precise
state ownership. Qwen's state is both fixed recurrent data and growing KV.

## Session layout and transitions

An engine owns data shared by all requests (weights and compiled plans). A
session owns data that changes when one conversation advances. A checkpoint is a
serialized snapshot of the latter at a particular committed position. The
distinction matters because sharing a mutable recurrent matrix between two
sessions silently makes one conversation affect another.

```text
SessionState
  gdn[48]: recurrent FP32[48,128,128], conv ring[10240,4], ring frontier
  attention[16]: K and V[positions,4,256], logical length/capacity
  position_frontier: committed text and future mRoPE coordinates
  committed_input_identity: token IDs plus template/tokenizer identity
  optional_mtp: absent in v1
```

`begin_step` obtains writable state, kernels produce logits and next state, and
`commit_step` advances every layer and the input frontier atomically. On error,
none advances. Prefill chunk boundaries are implementation details and may not
change the final state.

The **frontier** is the number of positions whose effects are present in every
state component. If the frontier is 100, every GDN matrix and ring and every KV
cache must describe exactly positions `0..99`. A state where attention reached
100 but a GDN layer reached only 99 has no valid public meaning, even if all its
buffers contain individually well-formed data.

Checkpoint headers include magic/version, endianness, model/config/weights,
tokenizer/template and quant hashes, context/capacity, committed positions,
per-section shapes/dtypes/lengths/checksums, and feature flags. Write to a new
payload and publish atomically. Restore validates everything before mutation.

`magic` is a fixed byte sequence identifying the file type; `version` selects
the binary layout; endianness says how multibyte numbers are encoded. Section
lengths allow bounds checking before reads, and checksums detect corruption.
Compatibility hashes detect a different but structurally similar model, which
shape checks alone cannot catch.

Prefix reuse compares rendered token IDs and position metadata. An exact saved
prefix can restore directly; a shorter common prefix needs a checkpoint at or
before it plus replay. Arbitrary rollback is not obtained by truncating KV:
GDN recurrence is not invertible. Maintain deliberate checkpoint intervals.

Consider checkpoints every 1,024 positions. To fork at position 2,500, restore
the checkpoint at 2,048 and replay 452 IDs. More frequent checkpoints reduce
replay but consume more storage and write bandwidth. This is a policy tradeoff;
the state format should support it without pretending arbitrary rollback is free.

## Batching and CUDA Graph constraints

Batch only ready work with compatible kernel shapes. Per-session state remains
disjoint; weight reads may be shared. Record queue delay separately from kernel
time. Capacity planning multiplies the full session ledger, not only KV.

**Batching** places rows from different sessions into one launch so they share a
weight sweep. It can improve total tokens per second while increasing the time
one request waits for compatible peers. That waiting is queue delay, not model
compute, and both must be reported. A batch descriptor maps each row back to its
session's state and position.

Graph capture requires stable addresses and allocation-free replay. A graph key
contains batch/row bucket, kernel and quant policy, KV layout/capacity class,
workspace addresses or generations, feature set, and any control path changing
topology. Logical lengths may be parameters only if every captured kernel reads
them safely. Instantiate graphs before declaring the VRAM fit.

A **CUDA Graph** records a launch sequence so it can be replayed with much less
CPU scheduling overhead. It does not record model semantics or make a changing
pointer safe. Stable addresses mean the graph's buffers remain allocated at the
same locations; a graph key selects a separate recorded sequence whenever
shape or control flow changes materially.

## DwarfStar transfer boundary

Reuse engine/session lifetimes, token-prefix comparison, versioned checkpoint
validation, allocation guards, and graph discipline. Adapt the payload and keys.
Discard compressed row counts/windows and expert identities.

## Failure modes and exercise

Failures include KV-only saves, physical rather than logical ring order,
non-atomic commits, restoring across a quant/template mismatch, sharing mutable
state between batch slots, or capturing allocator calls.

Save/restore after positions 1, 3, 4, 5, 32K, and arbitrary chunk boundaries.
Expected: continuation logits and every state component equal uninterrupted
execution. Prefix forks share immutable weights but never mutable buffers.
