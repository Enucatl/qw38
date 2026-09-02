# 48. Atomic CUDA evaluation and separate sampling

[Index](README.md) · Implementation tasks: SES-002 and EDU-034 in
[`implementation_ledger.md`](../implementation_ledger.md)

Evaluation changes a conversation. If it fails halfway through 64 layers, a
session must not become a mixture of old and new state. SES-002 makes one CUDA
token a **transaction**: either all visible state advances together, or none of
it does.

## Committed state and candidate state

**Committed state** is the last complete token history callers may observe or
save. It includes the frontier, token IDs, 48 GDN states, the attention rows
below the frontier, and the latest outputs. **Candidate state** is temporary
work for a token that has not succeeded yet.

The distinction is like editing a document in a private draft. An error may
destroy the draft, but readers continue seeing the last published version.

For every GDN layer, [`execute_token`](../cuda/full_scheduler.cu) reads the
committed recurrence and writes a corresponding slot in a complete alternate
buffer. The buffer holds candidates for all 48 layers, so an early layer never
overwrites committed memory while later layers are still capable of failing.
Logits and the hidden vector first copy into workspace-owned host staging, so
caller output buffers also retain their previous contents on failure. After
success, two host **pointer swaps** make the complete candidate
convolution and recurrence buffers current. Swapping addresses publishes all
GDN layers as one host operation; it does not copy roughly 156 MB.

Attention needs only one candidate key row and value row for each of its 16
layers. Older rows are read-only. After all logits and the hidden vector copy
successfully, the candidate rows are copied to the unused position at the
frontier and synchronized. Rows at or above the frontier are not logical KV
state and are never read or checkpointed. Then GDN pointers, token/output host
data, and finally the frontier are published. **Frontier last** means no caller
can interpret candidate bytes as a committed token before every prerequisite
has succeeded.

Duplicating the entire 128K KV cache would cost another 8 GiB and threaten the
32 GiB product budget, so this row-staging design is intentional.

## Cancellation and errors

An optional status poll runs after a synchronized layer boundary. Returning OK
continues. Returning `cancelled`, `internal`, or any other non-OK status stops
before publication and returns that exact status to the caller. Synchronizing
before the poll also surfaces asynchronous CUDA failures while state is still
candidate-only.

The diagnostic cancels after eight completed layers and injects an internal
error after 31. In both cases, frontier remains one, caller outputs remain
unchanged, and committed state is byte-exact with a reference that never began
the second token. An invalid token ID is a separate preflight error: it is
rejected before any kernel runs.

The test's “injected error” is deterministic evidence for the publication
protocol, not a claim that it reproduces every hardware failure. A fatal device
loss can require destroying the CUDA context and session rather than continuing
it.

## Why sampling is separate and read-only

Evaluation produces logits: one FP32 score for each vocabulary token. Sampling
chooses a token from those scores. Combining choice with evaluation would make
it unclear whether merely asking for another candidate mutated the model.

[`greedy_sample`](../cuda/full_scheduler.cu) is deliberately **read-only**. It
returns the index of the greatest committed logit and changes no GDN value, KV
row, token, output, or frontier. The fixture obtains token `1277` at frontier
two, then proves the entire committed state is unchanged. Temperature, top-k,
top-p, seeded randomness, and sampler-field persistence remain product-sampler
work; SES-002 establishes the ownership boundary with deterministic greedy
sampling.

## Memory and measured evidence

The candidate is reusable per workspace, not accumulated per token. At capacity
three, the complete workspace is now **186.30 MB**, including the fixed 64-row
prompt scratch added by SCH-002. The one-token atomic candidate remains reusable
and is not duplicated per token; its earlier increase was about 155.6 MB over
the scratch-only workspace because it contains the
all-layer GDN transaction buffer and 16 candidate KV rows. It remains a named
input to MEM-001 and its SCH-002 revalidation MEM-002 rather than an unrecorded
allocation.

**Measured local:** cancellation, injected error, successful exact commit,
separate greedy sampling, and invalid-token preflight passed on the RTX 5090
with CUDA 13.0.2. The successful second-token state exactly matched ordinary
fresh execution, and the unchanged scheduler continued to meet its frozen
scalar logits, tap, and greedy gates.

The authenticated boundary is
[`cuda_atomic_eval_contract.json`](../pins/cuda_atomic_eval_contract.json), and
the retained cases are in
[`cuda_atomic_eval.json`](../fixtures/cuda_atomic_eval.json). The proof boundary
is one-token CUDA evaluation. Multi-token request rollback, stochastic sampler
policy, checkpoint persistence, public host `Engine` integration, 128K fit,
graphs, active-request CUDA cancellation, and throughput remain later gates.
SRV-001 separately proves cancellation while a request is waiting in the
single-flight server queue; it does not yet connect an HTTP generation request
to this one-token transaction.
