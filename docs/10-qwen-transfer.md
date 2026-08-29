# 10. Deferred extensions: MTP, then vision

[Previous](09-engine-comparison.md) · [Index](README.md) · [Next](11-glossary-worksheets.md)

## Why this matters

V1 is correct text inference. Its interfaces should admit later embeddings and
positions, but speculative decoding and vision must not complicate the first
runnable path or silently become requirements.

## MTP as a separate state machine

**Speculative decoding** separates a cheap proposer from the trusted target.
The proposer guesses several next tokens; the target evaluates them and decides
how many are valid. “Acceptance” means committing the target's state for a
prefix of those guesses, not merely displaying their text. The config exposes
one MTP hidden layer. Implement it only after the target text model passes all
gates. MTP proposes a sequence; the target model verifies it; only the accepted
prefix commits. Keep draft/MTP KV and transient hidden state separate from
target `SessionState` until acceptance.

Without speculation, the target reads its large weights once per generated
token. If a cheap proposer guesses several tokens and the target verifies them
as a batch, one target weight sweep can advance several positions. The speedup
depends on proposal cost, verification cost, and acceptance length. Incorrect
guesses are not incorrect output—the target rejects them—but low acceptance can
make speculation slower than ordinary decode.

```text
snapshot target frontier
draft candidates c0..cn
target verifies candidates in one or more rows
accept k: commit target state through c[k-1], commit matching MTP state
reject suffix: discard its target/MTP speculative state
```

If candidates `[A,B,C,D,E]` are proposed and only `[A,B]` pass, the public
session advances by two positions. State computed for `C,D,E` must be discarded
or rolled back, and the next ordinary target token is sampled from the correct
verification boundary. This is why speculative state cannot alias committed
state casually.

Test full acceptance, partial acceptance, and zero acceptance, including stop
tokens and stochastic sampling semantics. Each result must equal a non-speculative
target run under the selected sampler. DwarfStar's DSpark supplies useful
commit/rollback ideas, but its tensors and proposal rules are not Qwen MTP.

## Vision boundary and mRoPE

`SequenceInput` already accepts embeddings `[T,5120]` and position metadata, so
the text core does not need to know whether a row came from token lookup or the
visual projector. A separate official-compatible processor later performs
image/video token placement and produces temporal, height, and width positions.

An image is not passed directly to the language layers. The processor resizes
and normalizes it, divides it into spatial and temporal patches, and records the
patch grid. The vision encoder turns patches into vectors; the merger reduces
groups of patch vectors and projects them from width 1,152 to the text width
5,120. Those vectors are spliced among ordinary token embeddings at positions
defined by special boundary tokens.

The official vision contract is 27 layers, width 1,152, 16 heads, FFN 4,304,
patch 16, temporal patch 2, spatial merge 2, and projection to 5,120. The text
RoPE uses three mRoPE sections `[11,11,10]`, interleaved into the 64 rotary
dimensions. Image/video boundary IDs and processor ordering come from pinned
processor files, never handwritten assumptions.

Plain RoPE has one sequence coordinate. Multimodal RoPE supplies coordinates
for time, height, and width so two image patches can differ spatially even when
they are adjacent in the flattened sequence. The `[11,11,10]` sections describe
how rotary frequency pairs are assigned among those axes; “interleaved” defines
their ordering. The text core consumes the resulting cosines/sines and need not
understand image geometry.

```mermaid
flowchart LR
  T[token IDs] --> E[token embeddings]
  V[image/video] --> P[official processor + vision encoder]
  P --> X[visual embeddings + mRoPE positions]
  E --> S[SequenceInput rows + positions]
  X --> S
  S --> C[unchanged 64-layer text core]
```

Vision acceptance covers text-only regression, one image and one video, visual
row placement, position tensors, embedding splice, logits versus Transformers,
and save/restore with multimodal position metadata. Vision tokens consume the
same attention KV budget as text positions.

## Transfer boundary and failure modes

Reuse DwarfStar's speculative transactional discipline and session versioning.
Adapt both to Qwen MTP and multimodal positions. Reject DSpark weights/semantics
and any assumption that plain text tokenization represents visual input.

Failures include committing all proposals before verification, sharing MTP and
target KV, rolling back KV but not GDN state, flattening mRoPE to one position,
or embedding vision logic inside CUDA text kernels.

## Exercise and expected result

Draw state after partial acceptance of five proposals and after a prefix with
one image. Expected: target and MTP frontiers commit only the accepted prefix;
the multimodal checkpoint contains reproducible visual input/embedding identity,
mRoPE positions, all GDN state, all full-attention KV, and version hashes.
