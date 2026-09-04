# 49. Versioned atomic CUDA checkpoints

[Index](README.md) · Implementation tasks: SES-003 and EDU-035 in
[`implementation_ledger.md`](../implementation_ledger.md)

A session disappears when its process exits unless its state is written to
disk. A **checkpoint** is a complete, self-identifying snapshot from which the
same conversation can continue. It is not merely the token list: GDN recurrence
compresses the whole past, attention needs its earlier key/value rows, and
future sampling needs the same random-generator state.

## What the file contains

Checkpoint v1 begins with the eight bytes `QW38CKP1`. Every integer and tensor
uses **little-endian** byte order, meaning the least-significant byte comes
first. That matches the supported Linux/x86-64 host and makes the on-disk
representation explicit rather than compiler-dependent.

The 248-byte header records:

- format version and header size;
- the exact pinned model SHA-256;
- a **compatibility hash** covering the state layout and numeric types;
- saved capacity and committed frontier;
- sampler temperature, top-p, top-k, seed, and RNG state; and
- the byte length of every following section.

The payload then stores token IDs, all 48 FP32 GDN convolution rings, all 48
FP32 recurrent matrices, 16 BF16 key-cache prefixes, 16 BF16 value-cache
prefixes, the last 248,320 FP32 logits, and the last 5,120-value hidden vector.
Only **committed KV rows** below the frontier belong to the conversation. Empty
capacity is omitted, so checkpoint KV storage grows by exactly 65,536 bytes per
token rather than always consuming 8 GiB.

A final 64 ASCII characters hold the SHA-256 **payload digest** over the header
and every payload byte. A digest is a compact fingerprint: changing even one
tested byte changes the expected fingerprint. It detects damage; it is not
encryption and does not hide prompts or model state.

## Atomic and durable save publication

[`save_checkpoint`](../cuda/checkpoint.cu) never writes through the visible
destination. It writes an adjacent **temporary file** ending in `.tmp`, closes
it, computes and appends the digest, and calls `fsync` so the file contents
reach the operating system's durable-storage boundary. It then performs a
same-directory `rename`, which atomically changes the visible name on Linux,
and `fsync`s the parent directory so that name change is durable too.

Therefore readers see either the previous complete checkpoint or the new
complete checkpoint, never a half-written destination. The diagnostic also
checks that no temporary file remains after success. Device or filesystem
hardware may still violate durability guarantees; `fsync` is the standard
Linux contract available to the process, not a claim about every storage
controller.

## Restore validates before mutation

[`restore_checkpoint`](../cuda/checkpoint.cu) first checks magic, version,
model and layout identities, capacity/frontier bounds, sampler ranges, every
section size, exact total file size, the payload digest, and every token ID.
Only after all these checks pass does it copy state toward the GPU.

GDN bytes load into the existing SES-002 candidate workspace and become active
through pointer swaps. KV data is copied only for the saved prefix, outputs and
sampler fields are restored, and the frontier is published last. A corrupt
payload and a deliberately changed compatibility hash both return
`incompatible_artifact` without changing the already valid target session.

Validation and application use the same open file descriptor for framing data,
while the digest helper performs a bounded read of the same named file. V1
assumes applications do not concurrently replace a checkpoint being restored;
the single-flight product boundary enforces that ownership. A CUDA device-loss
failure during application can require discarding the session/context, just as
in ordinary evaluation.

## Exact continuation evidence

**Measured.** The retained two-token file is 160,004,416 bytes. Its largest section is the
150,994,944-byte recurrent state; the GDN convolution section is 7,864,320
bytes. At frontier two, keys and values each need 65,536 bytes. Tokens need
eight bytes, logits 993,280, hidden state 20,480, and framing/digest 312.

The diagnostic saves tokens `[42, 3649]` with non-default sampler fields,
restores into a fresh session, and compares every logical byte. It then evaluates
token `1277` in both uninterrupted and restored sessions. Both reach frontier
three with byte-exact state and outputs. This is **exact continuation**: not just
the same sampled word, but the same state that will govern later words.

The authenticated format is
[`cuda_checkpoint_contract.json`](../pins/cuda_checkpoint_contract.json), and
the retained result is
[`cuda_checkpoint.json`](../fixtures/cuda_checkpoint.json). The proof boundary
covers capacity-three save/restore, atomic publication, corrupt/incompatible
rejection, sampler persistence, and one exact continuation. It does not yet
measure 128K checkpoint time or disk space, inject power loss, wire the public
host `Engine`, provide cross-version migration, encrypt session data, or prove
the MEM-001 128K GPU-reserve gate.
