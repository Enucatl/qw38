# The single-flight HTTP server core

This chapter explains how `qw38-server` becomes reachable over a network, what
its first two routes mean, and how Quartz guarantees that only one future model
request owns the GPU at a time. It corresponds to SRV-001 and EDU-043 in the
[implementation ledger](../implementation_ledger.md).

## From a process to a network service

A terminal program reads from its own standard input. A server instead opens a
**TCP socket**: an operating-system endpoint identified by an IP address and a
port number. Clients connect to that endpoint and exchange byte streams.

Quartz defaults to `127.0.0.1:8080`. `127.0.0.1` is the IPv4 **loopback**
address, which means only programs on the same machine can connect. This is a
safer default than exposing an unfinished, unauthenticated service to the local
network. `--host` can select another literal IPv4 address explicitly, and
`--port 0` asks the operating system to choose a free port for tests.

[`src/server.cpp`](../src/server.cpp) binds the address first, then authenticates
and uploads the model and creates one complete 131,072-token session. Only after
those operations succeed does it start listening and print `listening=...`.
A client therefore cannot receive a successful readiness response from a
half-initialized process.

## What HTTP adds

HTTP gives structure to the TCP bytes. A minimal **HTTP request** looks like:

```text
GET /health HTTP/1.1\r\n
Host: 127.0.0.1\r\n
\r\n
```

The first line contains a method, target, and protocol version. Each following
header is a `name: value` line. The empty line ends the header section. `\r\n`
means carriage return followed by line feed: two exact bytes, not visible text.

An **HTTP response** has a status line, headers, an empty line, and a body:

```text
HTTP/1.1 200 OK\r\n
Content-Type: application/json\r\n
Content-Length: 123\r\n
Connection: close\r\n
\r\n
{"status":"ok"}
```

JSON represents objects with braces, names in quotes, and values such as text,
numbers, booleans, arrays, or nested objects. `Content-Length` is the exact
number of body bytes. V1 closes the connection after one request, which keeps
ownership and parsing easy to audit.

## A deliberately bounded parser

The local parser accepts HTTP/1.0 or HTTP/1.1 request lines and requires every
header line to contain a colon. It reads at most 65,536 header bytes and applies
a five-second receive timeout. Responses use `MSG_NOSIGNAL`, so a client that
disconnects during a write cannot terminate the server with `SIGPIPE`.

Malformed input receives status 400 and a JSON error. An unsupported method on
a known or unknown route receives 405 plus `Allow: GET`; an unknown GET path
receives 404. The parser does not yet accept JSON request bodies or chunked
request transfer encoding because SRV-001 has no generation route.

## Control plane versus data plane

The first routes are the **control plane**: they describe service state but do
not run the model.

### `GET /health`

This returns 200 only after the exact model and one GPU session exist. Its body
reports readiness, model ID, one-session ownership, whether the single-flight
gate is active, and the current number of waiting requests. It is suitable for
local startup probes.

### `GET /v1/models`

This returns the OpenAI-style list envelope and the one admitted model ID,
`qwen3.8-27b-q4_k_m`. Reporting a model does not imply that every OpenAI
endpoint is implemented. Chat Completions and Responses are the later SRV-002
and SRV-003 gates.

The **data plane** will contain requests that tokenize, synchronize a session,
run CUDA, and generate output. SRV-001 builds its ownership mechanism but does
not expose a placeholder generation route.

## Why single-flight exists

The RTX 5090 has one resident model and Quartz V1 permits **one GPU session**.
If two request threads mutated that session simultaneously, their token
histories, GDN recurrence, and KV rows could interleave. A mutex alone could
prevent overlap, but it would not make order, cancellation, or wait time
observable.

[`SingleFlightGate`](../src/server_core.h) instead uses a **FIFO** queue: first
in, first out. Each arriving request receives a monotonically increasing ticket
and records its **queue depth**, the number of active or earlier waiting owners.
Only the ticket at the front can become active. The measured **queue delay** is
the monotonic time from arrival until ownership; it is separate from model
execution time.

At most one ticket is active. Releasing it wakes the next waiter. The native
diagnostic holds the first owner for at least 30 ms, proves the second request
waits with arrival depth one, and records a maximum active count of one.

## Cancellation and shutdown

A queued request carries an atomic cancellation flag. The queue checks it while
waiting. On **cancellation**, it removes that ticket, wakes the other waiters,
and returns an explicit `cancelled` status; it never becomes the GPU owner.
This is the mechanism a later connection handler will set when its client
disconnects.

`SIGINT` and `SIGTERM` initiate **graceful shutdown**. The listener closes so no
new connections arrive, the gate wakes every queued waiter with cancellation,
and the server waits for its small control-plane connection handlers to finish.
The session and resident model are then released by their ordinary owners.

Active CUDA generation cancellation is not claimed here. SRV-002 must connect
client disconnect detection to the scheduler's existing cancellation control
while preserving atomic session state.

## Run the admitted control plane

Build the CUDA products, then use host networking so the loopback listener is
reachable outside the container:

```bash
make cuda-build
docker run --rm --network host --gpus all \
  --user "$(id -u):$(id -g)" -v "$PWD:/workspace" \
  qw38-cuda:13.0.2 \
  ./build/cuda/qw38-server models/Qwen3.8-27B-Q4_K_M.gguf
```

From another terminal:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

## Evidence and proof boundary

**Measured:** the native concurrency diagnostic admitted exactly one active
owner, a FIFO waiter at depth one with more than 30 ms queue delay, removal of a
cancelled waiter, and shutdown wake-up. The real CUDA smoke authenticated the
pinned model, allocated one session (27,110 MiB observed process GPU memory),
served both public routes plus 400/404/405 cases, and exited zero after SIGTERM.
The successful pytest case took 11.56 seconds. Structured evidence is retained
in [`fixtures/server_core.json`](../fixtures/server_core.json).

The first real harness timed out even though the server was listening because a
buffered text reader hid the readiness line from `select`; the corrected test
uses raw pipe reads. That negative remains in the fixture and ledger.

**Proof boundary:** this gate proves the socket lifecycle, bounded header
parser, exact control routes, one-session allocation, FIFO exclusion, queue
timing, queued cancellation, and graceful shutdown. It does not prove Chat
Completions, Responses, request JSON, streaming, tools, authentication, active
CUDA cancellation, TLS, persistent HTTP connections, remote deployment safety,
or concurrent GPU sessions.
