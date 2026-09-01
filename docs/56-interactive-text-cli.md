# The interactive text CLI

This chapter explains how typed terminal text becomes a model answer, why the
production executable is CUDA-only, and what a successful CLI test does and
does not prove. It corresponds to CLI-001 and EDU-041 in the
[implementation ledger](../implementation_ledger.md).

## What the CLI is

A **command-line interface (CLI)** is a program used from a terminal. Quartz's
CLI is `qw38`. It is deliberately thin: it parses human-facing options and
calls the public [`Engine` and `Session`](../include/qw38/engine.h) operations.
It never receives a tensor pointer, chooses a CUDA kernel, or knows how GDN and
attention state are laid out.

There are two binaries with the same command-line source:

- `build/qw38` is the ordinary C++17 validation build. It can inspect and
  authenticate the model, tokenize, render templates, and fail closed, but it
  cannot create a GPU session.
- `build/cuda/qw38` is the production CUDA binary for `sm_120`. It uploads the
  admitted model and owns the complete 131,072-token session.

This split lets CPU-only contributors compile and test the public boundary
without pretending that host execution is a production backend.

## From terminal text to a generated token

Suppose the user types `Hello`. The following stages happen in order:

1. [`src/cli.cpp`](../src/cli.cpp) wraps the text in a user message.
2. [`src/template.cpp`](../src/template.cpp) applies the Qwen chat template.
   A **chat template** is punctuation for the model: special markers identify
   where a user message ends and where an assistant answer begins.
3. [`src/tokenizer.cpp`](../src/tokenizer.cpp) **encodes** the rendered bytes.
   Encoding converts text into integer **token IDs**, the model's input
   alphabet. Chapter 15 explains NFC normalization, Unicode-aware splitting,
   GPT-2 byte mapping, BPE, and exact fixtures from first principles.
4. `Session::sync` compares the requested token IDs with the committed token
   history. Their longest identical beginning is the **common prefix**. Quartz
   reuses its existing state and evaluates only the suffix after that prefix.
5. `Session::sample` chooses an ID from the current logits. **Logits** are one
   unnormalized score per vocabulary token. Temperature zero chooses the
   largest score; temperature, top-k, and top-p can instead make a seeded,
   probabilistic choice.
6. `Session::eval` evaluates the chosen ID through all 64 layers. Only a
   successful evaluation commits the new GDN state, KV row, position, token,
   output logits, and pending sampler RNG state. This is the **atomic** boundary:
   on failure the previous session remains the visible state.
7. The CLI appends the committed ID to its history and asks `Engine::decode` to
   turn generated token IDs back into display bytes. Decoding reverses the
   tokenizer's GPT-2 byte mapping and omits control tokens from visible output.

The deliberate `sample -> eval -> publish` order matters. Sampling does not
secretly evaluate a token, and a sampled stochastic RNG step is not persisted
unless evaluation commits that same token.

## Engine and session ownership

`Engine::open` authenticates the exact GGUF, memory-maps it, builds the
tokenizer, binds all typed weights, and uploads one immutable resident CUDA
model. `Engine::create_session` allocates the mutable timeline: recurrent GDN
matrices, convolution rings, attention KV rows, token history, scratch space,
and 64 uploaded FFN graphs.

The public classes are move-only, so one owner is responsible for releasing
each resource. A session keeps shared ownership of the immutable resident CUDA
model. It therefore cannot be left with a dangling model pointer if the
original `Engine` handle is moved or destroyed. The session timeline itself is
not shared and V1 still permits only one active GPU request.

## First turn and later turns

The first turn renders the system instruction, the first user message, and the
assistant generation prompt as one template. A later turn does not rebuild the
whole transcript. The checkpoint already contains every committed earlier
token and state row, so [`render_user_turn`](../src/template.cpp) emits only:

```text
<|im_start|>user
Next question<|im_end|>
<|im_start|>assistant
<think>
```

With reasoning disabled, the last line is instead an already closed, empty
thinking block. Exact byte tests cover both suffixes and reject an empty turn.
Appending a suffix avoids duplicating old messages after checkpoint restore.

## Stops and visible reasoning

`<|im_end|>` is the model's **stop token**: it marks the end of the assistant
message. The CLI also accepts repeatable `--stop TEXT` strings and a maximum
token count. A custom stop string is removed from visible output. If generation
ends at a text stop or the token limit, the CLI commits `<|im_end|>` itself so
the stored transcript still has a valid next-turn boundary.

With `--reasoning off`, Qwen's required empty thinking block is present in the
model input but not printed. With `low`, `medium`, or `xhigh`, generated text
before `</think>` is shown as `reasoning>` and later text as `assistant>`.
Output is decoded as generation proceeds but printed once the turn ends; V1's
CLI is interactive across turns, not token-streaming within a turn.

## Checkpoints and commands

A **checkpoint** is an atomic disk snapshot of the session. It contains the
token history, frontier, every GDN matrix and convolution ring, committed KV
rows, last hidden/logit outputs, sampler fields, and compatibility hashes.
Chapter 49 explains its byte format and durable temporary-file/rename protocol.

Interactive commands are:

- `/save PATH` writes the current committed state.
- `/load PATH` validates and restores a state, then restores its token history
  into the CLI so the next user turn is appended correctly.
- `/reset` synchronizes to an empty prefix and clears the visible history.
- `/help` prints options, and `/quit` exits.

`--save PATH` automatically saves after every completed turn. `--load PATH`
restores before the first new turn.

## Build and run

Build inside the pinned CUDA 13.0.2 environment:

```bash
make cuda-build
```

Then run an interactive session in that environment:

```bash
./build/cuda/qw38 models/Qwen3.8-27B-Q4_K_M.gguf \
  --reasoning off --temperature 0 --save conversation.qw38
```

For one non-interactive turn:

```bash
./build/cuda/qw38 models/Qwen3.8-27B-Q4_K_M.gguf \
  --prompt "Reply with exactly: hello" --reasoning off \
  --temperature 0 --max-tokens 8
```

The literal Quartz brand line is printed at startup. Unknown options and
invalid sampler values fail before inference.

## Can Codex call this engine?

Not through the CLI alone. Codex and other OpenAI-compatible clients expect an
HTTP service with endpoints, JSON request/response objects, streaming events,
tool-call semantics, cancellation, and protocol errors. Those belong to
SRV-001 through SRV-003. The CLI proves that a person or shell script can run
the engine now; `qw38-server` must pass its own gate before Codex can select
Quartz as an OpenAI-compatible provider.

## Evidence and proof boundary

**Measured:** on the pinned local RTX 5090, the CUDA product opened the exact
Q4_K_M artifact and greedily answered the eight-token-limit smoke prompt with
`assistant> hello`. The automated CUDA smoke also saves a 161,118,596-byte
checkpoint at that prompt frontier, verifies its `QW38CKP1` header, and restores
it in the same process. Checkpoint size grows with the number of committed KV
rows; it is not a format constant.
The exact command, environment, and results are retained in
[`fixtures/cli_smoke.json`](../fixtures/cli_smoke.json).

**Measured:** the original portable full-file SHA-256 took 56.603 seconds before
the first prompt. MDL-003 now uses the same pinned SHA-256 through an
accelerated provider and measured 8.47 seconds with a warm cache. Chapter 57
explains the dispatch, portable fallback, and cold-storage limit; no metadata
shortcut is implied.

**Proof boundary:** this gate demonstrates batch-1 text generation, public
runtime ownership, terminal turns, stops, sampling controls, and checkpoint
save/restore. It does not prove HTTP compatibility, concurrent sessions,
token-by-token terminal streaming, tool execution, cancellation, quality
scores, or the comparative speed release gate.
