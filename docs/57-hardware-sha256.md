# Hardware-accelerated model authentication

This chapter explains why Quartz reads the complete model at startup, how the
same SHA-256 calculation became faster, and why filesystem checksums solve a
different problem. It corresponds to MDL-003 and EDU-042 in the
[implementation ledger](../implementation_ledger.md).

## What a cryptographic hash proves

A **cryptographic hash** turns any number of input bytes into a fixed-size
fingerprint. SHA-256 produces 256 bits, usually displayed as 64 hexadecimal
characters. Changing even one input bit is intended to produce an unrelated
fingerprint, and finding different bytes with the same fingerprint should be
computationally infeasible.

Quartz pins this whole-file SHA-256 for the one admitted model:

```text
31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
```

`Engine::open` checks the 18,973,870,432-byte size, validates the GGUF model
contract, calculates that digest over every byte, and compares it with the
compiled pin. Shape validation can identify a model with the right structure;
the hash establishes that its complete byte content is the admitted artifact.

## Why the original implementation was slow

[`src/sha256.cpp`](../src/sha256.cpp) originally contained only an inspectable,
portable implementation. **Portable** means it uses ordinary C++ integer
operations that work across supported CPUs. It expanded every 64-byte SHA-256
block and executed all 64 compression rounds with scalar operations. Its update
loop also copied input one byte at a time.

That code was correct, but full-model authentication took 56.603 seconds on the
local AMD Ryzen 9 9900X. The CPU advertises the x86 SHA extension, exposed on
Linux as `sha_ni`. These instructions perform several SHA round operations more
efficiently than the portable sequence.

## Runtime dispatch and OpenSSL EVP

**Runtime dispatch** means choosing an implementation after the program starts,
based on what the current machine and installed provider support. Quartz loads
the pinned CUDA container's `libcrypto.so.3` and resolves the OpenSSL EVP digest
interface. EVP is OpenSSL's high-level interface for initializing a digest,
feeding it multiple input chunks, and finalizing the result. OpenSSL recommends
this interface over its deprecated low-level SHA functions.

The provider performs its own CPU capability dispatch. On this machine it can
select its SHA-NI path. Quartz continues to stream the file in bounded chunks;
it does not place a second 18.97 GB copy in memory. Every EVP return value is
checked. A provider error becomes an explicit internal error rather than an
accepted model.

Run the diagnostic to see the selected Quartz boundary:

```bash
build/qw38-eval --sha256-backend
# openssl-evp
```

The production container includes OpenSSL 3.0.13 through its immutable base
image. Quartz dynamically resolves the ABI, so CPU-only compilation does not
require OpenSSL development headers.

On Darwin/x86_64 host builds, the same diagnostic may report `commoncrypto`
(Apple Common Crypto) or `openssl-evp` if Homebrew `libcrypto` is present.
`QW38_SHA256_FORCE_PORTABLE=1` still selects the portable fallback. That Darwin
path authenticates the additive Qwen3.5-2B laptop pin; it does not replace the
Linux container measurement in this chapter.

## Portable fallback

The self-contained implementation remains a **portable fallback**. It is used
when `libcrypto.so.3` or a required EVP symbol is unavailable, or when a test
sets `QW38_SHA256_FORCE_PORTABLE=1`. Its input loop now copies whole blocks
instead of individual bytes, but it deliberately retains the readable scalar
compression rounds.

The fallback is not a weaker verification mode. Both paths calculate the same
SHA-256 over the same whole-file byte stream and compare it with the same pin;
only elapsed time differs. Tests feed identical binary fixtures through both
paths and require exact hexadecimal digest equality.

## ZFS integrity versus artifact identity

ZFS stores a checksum for each block in that block's metadata. On a read, ZFS
uses it to detect corruption and can repair a bad copy when redundancy is
available. This is valuable storage integrity, but it is not the model's
whole-file identity: writing a different valid file creates new blocks with new
valid checksums.

An immutable, previously authenticated ZFS snapshot could support a separate
metadata receipt keyed to its stable snapshot identity. MDL-003 does not add
that policy. It still reads all model bytes, whether they come from ZFS ARC,
the Linux page cache, or physical storage.

## Warm cache and cold cache

A **warm cache** means the operating system or ZFS ARC already has much of the
file in memory. Hashing is then mostly CPU work. A **cold cache** means blocks
must be fetched from storage, so elapsed time cannot beat the device's read
bandwidth. For example, reading 18.97 GB at 2 GB/s already takes roughly 9.5
seconds before considering hash work.

The retained measurement is warm-cache and is therefore an implementation
comparison, not a ZFS or SSD benchmark:

| Path | Elapsed | User CPU | System CPU | Digest |
|---|---:|---:|---:|---|
| Original portable Quartz | 56.603 s | not retained | not retained | pinned |
| System `sha256sum` control | 8.56 s | 7.82 s | 0.73 s | pinned |
| Quartz OpenSSL EVP | 8.47 s | 7.81 s | 0.65 s | pinned |

**Measured:** Quartz's accelerated run was about 6.68 times faster than the
retained original run and produced the exact pinned digest. Raw structured
evidence is in
[`fixtures/sha256_acceleration.json`](../fixtures/sha256_acceleration.json).

## Failure modes

- Missing provider or symbols: use the portable fallback and remain exact.
- EVP initialization failure: release its context and use the portable fallback.
- EVP update/final failure after hashing starts: fail closed; do not silently
  restart and mask a provider failure.
- File read failure or truncation: return an I/O error.
- Digest mismatch: reject the artifact before CUDA upload.
- Forced fallback: change performance only, never expected identity.

## Sources and proof boundary

OpenSSL documents `EVP_DigestInit_ex`, `EVP_DigestUpdate`, and
`EVP_DigestFinal_ex` as the preferred streaming digest boundary and specifies
that they return one on success. Its x86 capability documentation assigns a
capability bit to the SHA extension. OpenZFS documents that its checksums are
calculated per block and stored in block pointers. Links and provenance are in
the [source ledger](sources.md).

**Proof boundary:** the evidence establishes exact digest equality, provider
selection, explicit portable selection, and one warm-cache full-model timing on
this CPU. It does not prove a maximum startup time on a cold cache, every
OpenSSL/CPU combination, ZFS snapshot receipts, or that model upload and session
allocation are included in the 8.47-second hash measurement.
