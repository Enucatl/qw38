# Responses objects and exact continuation

**Tasks:** SRV-003, API-003, SES-004, EDU-045  
**Evidence:** [`responses_api.cpp`](../src/responses_api.cpp),
[`response_store.cpp`](../src/response_store.cpp),
[`server.cpp`](../src/server.cpp),
[`responses_contract.json`](../pins/responses_contract.json), and
[`responses.json`](../fixtures/responses.json)

This chapter starts from zero: a *request* is JSON sent by a client, a
*response* is JSON returned by the server, and an API is the agreed meaning of
those fields. The Responses API is a second public envelope around Quartz's one
text engine. It does not create another model implementation.

## From an input item to model tokens

The `input` field may be a plain string:

```json
{"model":"qwen3.8-27b-q4_k_m","input":"Say hello"}
```

Quartz treats that string as a user message. An array can carry typed *items*.
An item is a JSON object whose `type` tells us what the rest means:

- `message` carries text with a `system`, `developer`, `user`, or `assistant`
  role;
- `function_call` records an earlier assistant request to run a named function;
- `function_call_output` returns the text produced by that function.

“Typed” does not mean a C++ object crossed the network. It means the JSON has a
discriminator such as `"type":"message"`, which the strict parser checks before
mapping it to a C++ `ChatMessage`. Unknown fields are errors. Image, audio, and
file items are explicit text-only-v1 errors.

`instructions` becomes a leading developer message. Responses-style function
definitions put `name`, `description`, and `parameters` directly beside
`"type":"function"`; Chat Completions nests them below `function`. The adapter
in `responses_api.cpp` converts the former to the latter and then calls the same
validated `parse_chat_request`. Temperature, top-p, top-k, seed, stop strings,
maximum output tokens, reasoning effort, tool choice, and streaming therefore
have one internal meaning in both APIs.

The shared generation path is:

```text
Responses JSON -> strict typed items -> ChatRequest -> template bytes
               -> tokenizer token IDs -> Session::sync -> logits/sample/eval
               -> AssistantOutput -> Responses output items
```

This reuse matters: fixing tool parsing or a stop condition once fixes both
public envelopes instead of letting them drift.

## Response objects and output items

A successful non-streaming call returns an object with an ID such as
`resp_qw38_...`, status, model, output array, storage decision, and token usage.
The ID is a lookup key, not the conversation itself. Output is again typed:

- a `reasoning` item contains reasoning summary text emitted by this local text
  model;
- a `message` item contains assistant `output_text`;
- a `function_call` item contains `call_id`, function name, and JSON argument
  text.

The model can stop normally, call a tool, meet a caller stop string, or reach
`max_output_tokens`. The last case produces `status:"incomplete"` with reason
`max_output_tokens`; other successful cases are `completed`. Usage counts the
full input prefix and sampled output tokens. The reported `cached_tokens` is
currently zero because Quartz has not yet exposed an audited per-request reuse
counter, even though the session can reuse a prefix internally.

Structured output is different from ordinary JSON function arguments.
Structured output promises that the assistant's final prose obeys a requested
schema. V1 does not implement that constrained decoder, so any format other
than plain `text` fails instead of pretending the promise was met.

## What `previous_response_id` means

Suppose response B names response A as `previous_response_id`. The new input is
not used to reconstruct A from its visible prose. Reconstruction would be
unsafe: whitespace trimming, `<think>` boundaries, hidden tool XML, stop text,
and byte-pair token boundaries can all make the new token sequence differ from
what the model actually evaluated.

After a successful stored response, `Session::tokens` copies the exact committed
token IDs. A versioned record stores those integers plus the compatible function
schemas. For the next call, Quartz renders only one permitted suffix:

- one new user message; or
- one or more function-call outputs, grouped as Qwen tool responses.

It appends those suffix tokens to the saved IDs. `Session::sync` finds the exact
common prefix. In the same process, its GPU state is reused. After a server
restart, the same IDs are loaded and deterministically replayed to rebuild GPU
state. This is *exact prefix reuse*: integer equality, in order, from token zero
through the stored frontier.

Changing instructions during continuation is rejected because old instruction
tokens are already inside the immutable prefix. Function definitions must be
omitted (the stored definitions are inherited) or byte-canonical-equal to the
stored schemas. A continued request currently uses automatic tool choice; a new
forced-choice instruction would also require changing the old prefix.

## Atomic publication on disk

“Atomic” means readers see either the old complete state or the new complete
state, never a half-written JSON file. `response_store.cpp` performs four steps:

1. write a private `0600` temporary file in the response directory;
2. call `fsync` so its bytes reach the filesystem's durability boundary;
3. rename it to `<response-id>.json`, which publishes one directory entry;
4. `fsync` the directory so that entry is durable.

The record is limited to 16 MiB and validates its schema, model ID, JSON shape,
token range, regular-file type, and safe response-ID alphabet before use. A
malformed or incompatible record cannot mutate the session. This record is
small—roughly decimal token IDs—not an 8 GiB KV-cache copy. The separate binary
checkpoint format in Chapter 49 remains the complete state snapshot mechanism.

`store` defaults to true. With `store:false`, the response is returned but no
continuation record is published, so using that ID later returns a clear error.
Cancelled requests, generation errors, and validation failures also publish
nothing. V1 records have no time-based expiry and are never silently deleted;
the operator owns cleanup of `checkpoints/responses`.

## Streaming events

Server-Sent Events (SSE) is an HTTP response made of named event frames. Each
frame has an `event:` line and a `data:` line containing JSON. Quartz starts
with `response.created` and `response.in_progress`. As output becomes visible it
emits item/part additions, reasoning or text deltas, matching done events, then
`response.completed` or `response.incomplete` containing the final response.
Function arguments use `response.function_call_arguments.delta` and `.done`.

Every event has a monotonically increasing `sequence_number`. “Delta” means only
the newly visible bytes, so concatenating text deltas yields the final output
text. UTF-8 and caller stop boundaries use the same guarded implementation as
Chat Completions. If the client disconnects, cancellation reaches queued work or
the active CUDA evaluation and no continuation record is published.

## Failure examples

- Unknown response ID: the record was never stored, was removed, or belongs to
  another response directory.
- `store:false` then continuation: intentionally unavailable; an ID alone does
  not contain state.
- Corrupt JSON or an out-of-vocabulary token: incompatible record, rejected
  before `Session::sync`.
- New instructions or mixed user-plus-tool suffix: cannot preserve the exact
  prefix contract.
- `parallel_tool_calls:true`: unsupported because v1 admits one generated
  function call path at a time.
- Image/audio/file or JSON-schema output: outside the text-only v1 decoder.

## Evidence and proof boundary

**Measured (host):** the native diagnostic maps text, tools, reasoning, and
function outputs; rejects media, structured output, and instruction replacement;
and round-trips, misses, and corrupts an atomic record. Template tests compare
the exact bytes for grouped tool-result suffixes.

**Measured (RTX 5090):** a real non-streaming call returned exact requested
text; a continued call loaded the first record and the second record retained
every first-prefix token; `store:false` created no file and its ID could not be
continued; and a stream produced consecutive sequence numbers, ordered lifecycle
events, and deltas equal to the final text. The server was then stopped, a fresh
process loaded the original ID from disk, generated `restarted`, and published a
new record with the original token array as an exact prefix. The combined smoke
passed in 61.30 seconds.

**Proof limit:** these tests do not prove broad client-SDK compatibility or model
quality. Authentication and multi-tenant access control remain outside the
approved local-loopback v1 boundary.
