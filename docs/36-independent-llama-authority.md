# 36. Independent same-GGUF llama.cpp authority

[Index](README.md) · Implementation tasks: ORA-002 and EDU-022 in
[`implementation_ledger.md`](../implementation_ledger.md)

Quartz can produce stable logits and still be wrong. Comparing a Quartz run
with another Quartz run proves repeatability, not model semantics. ORA-002 adds
an independently implemented runtime—pinned llama.cpp—between the local scalar
engine and the primary Transformers authority.

This chapter explains what each authority can prove, how both programs receive
the same model and token history, and why this gate reports numeric differences
without choosing acceptance tolerances yet.

## Three authorities, three jobs

An **authority** is evidence we trust for a specific question. It is not a claim
that one program is infallible.

| Participant | Question it answers | What it cannot prove alone |
|---|---|---|
| Transformers eager execution | Does the computation follow the official Qwen implementation? | That the quantized GGUF was converted or decoded correctly |
| Pinned llama.cpp | Does a mature, independent runtime interpret this exact GGUF the same way? | That both GGUF runtimes match the original official checkpoint |
| Quartz scalar CPU | Can we inspect every local operation and state mutation? | Correctness, when compared only with itself |

Transformers is the **primary semantic authority** because Qwen's declared
`qwen3_5` implementation lives there. llama.cpp is the **independent same-GGUF
oracle** because its graph, quantized kernels, state owner, and scheduler were
written separately. Quartz's scalar path is the **local inspectable oracle**
used to diagnose a mismatch.

The intended chain is:

```text
official checkpoint + Transformers
              compare semantics
exact pinned Q4_K_M GGUF + llama.cpp
              compare GGUF execution
exact pinned Q4_K_M GGUF + Quartz scalar
```

A match at the bottom does not remove the top comparison. Two GGUF readers can
agree on the same conversion error.

## Exact source and build identity

[`pins/llama_authority_contract.json`](../pins/llama_authority_contract.json)
records the llama.cpp repository revision, model filename, byte count, SHA-256,
CUDA/CMake choices, diagnostic tokens, and expected binaries. The source
checkout and build directory live under ignored `.cache/authorities/`; generated
objects are evidence products, not source files.

[`tools/build_llama_authority.sh`](../tools/build_llama_authority.sh) fails if
the checkout is dirty, fetches the exact commit, configures CUDA architecture
120, and builds inside the digest-pinned CUDA 13.0.2 image defined by
[`docker/llama-authority.Dockerfile`](../docker/llama-authority.Dockerfile).
`GGML_NATIVE=OFF` prevents the container's build CPU from silently selecting
host-specific instructions. The build produces:

- `llama-eval-callback`, for later graph-tensor taps; and
- `qw38-llama-token-oracle`, the small repository-owned raw-token adapter.

The adapter uses llama.cpp's public API; it does not copy its model graph. It
loads all model layers on the GPU, checks token bounds, decodes one position at
a time, and writes every returned logit as little-endian FP32.

The much broader `llama-cli` target is intentionally outside this correctness
harness: at the pinned revision it also builds server, UI, and multimodal code
and downloads UI assets. The comparative-benchmark gate will own a separately
controlled upstream server build. Pulling that unrelated, moving asset into a
scalar authority build would make this gate neither narrow nor reproducible.

## Why this test starts with raw token IDs

Text normally follows this pipeline:

```text
messages -> chat template -> UTF-8 bytes -> tokenizer -> token IDs -> model
```

If two complete text runs disagree, the error might be in chat rendering,
Unicode normalization, token splitting, byte mapping, BPE merging, or the model
itself. This first low-level comparison deliberately starts after all text work:

```text
position 0: token 42
position 1: token 3649
```

Both runtimes therefore receive identical integers and positions. Token 3649 is
also the greedy result expected after token 42, so the second row tests a real
continuation edge rather than an unrelated token.

Bypassing the template here does **not** prove templates match. TOK-001 and
TOK-002 own exact native tokenizer/template fixtures; ORA-004 will join rendered
prompt bytes, IDs, tensor taps, and all three authorities in one admission
bundle. Keeping these gates separate makes a failure easier to locate.

ORA-002 also performs a separate identity-only check before model execution. It
renders the existing `user_no_thinking` case with Quartz, gives the exact 74
bytes to llama.cpp's vocabulary-only tokenizer, and requires this 13-ID sequence
from both implementations:

```text
248045, 846, 198, 9419, 248046, 198, 248045,
74455, 198, 248068, 271, 248069, 271
```

This proves both runtimes see the same already-rendered prompt bytes as the same
tokens. It does not ask llama.cpp to choose chat policy, and it does not feed
that longer prompt through the two-token low-level diagnostic.

## Complete logit rows

A **logit** is one unnormalized score for one possible next token. Qwen3.8 has
248,320 vocabulary entries, so each input position produces a row of 248,320
FP32 values:

```text
row 0 = scores after token 42
row 1 = scores after tokens [42, 3649]
```

Choosing only the largest score would hide most numeric errors. Both adapters
therefore retain all 496,640 values in ignored raw evidence files. The committed
fixture stores their SHA-256 hashes plus per-row comparison summaries, including
maximum absolute and relative error, RMS error, cosine similarity, non-finite
counts, first different index, top-ten token overlap/order, and both greedy IDs.

[`tools/run_llama_authority.sh`](../tools/run_llama_authority.sh) first asks
Quartz to verify the model hash, captures Quartz's two rows, runs the pinned
llama adapter in its CUDA container, and invokes
[`tools/compare_llama_authority.py`](../tools/compare_llama_authority.py). Raw
megabyte-scale rows and verbose runtime logs remain under
`.cache/authorities/llama-evidence`; the small, reviewable result is frozen in
the repository.

## Reporting is not admission

Different correct execution paths can round FP32 calculations differently.
For example, a GPU kernel may combine multiplication and addition, while the
Quartz scalar compiler is explicitly told not to contract them. Exact byte
equality is therefore not the default rule across runtimes.

ORA-002 uses zero tolerances only to locate and report the first difference. Its
fixture says `reporting_only_tolerances_not_frozen`: a false `passed` field at
zero tolerance is information, not a failed release gate. ORA-004 will freeze a
justified absolute and relative tolerance for every visible tap after
Transformers evidence exists. Later optimizations may meet those limits; they
may not loosen them.

Greedy equality remains independently visible. If the top two scores are nearly
tied, a small legal rounding difference may choose a different token. Such an
exception is allowed only when a stored fixture demonstrates the actual
near-tie; it cannot be asserted after seeing an inconvenient result.

## Failure boundaries

The harness fails rather than comparing when:

- model hash verification fails;
- either runtime receives different token IDs or vocabulary width;
- either raw file has a missing or extra byte;
- the pinned llama checkout is dirty or at another revision;
- CUDA/model/context initialization or either decode fails; or
- field output is duplicated or malformed.

Failed builds, OOMs, unsupported graphs, and numeric disagreements belong in
the chronological implementation ledger. They are evidence about the system,
not output to delete.

## Proof boundary

**Measured:** the 74 rendered bytes tokenize to the same 13 IDs in Quartz and
llama.cpp. Both raw-token executions choose 3649 then 1277. The two complete
logit rows have cosine similarities 0.999847 and 0.999853 and RMS differences
0.02946 and 0.03335. Their exact hashes, non-finite counts, extrema, first
different indices, and top-ten comparisons are recorded in
[`fixtures/llama_scalar_authority.json`](../fixtures/llama_scalar_authority.json)
and the dated ledger entry.

**Proposed:** ORA-003 must still establish feasible pinned Transformers
eager/offloaded execution. ORA-004 must then capture attributed taps and freeze
tolerances before ORA-001 can admit the scalar model semantically. This chapter
does not claim V1 quality, 128K capacity, or production CUDA performance.
