# Quartz Watch 38

**Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.**

Quartz Watch 38 (`qw38`) is a deliberately narrow, work-in-progress inference
engine for the pinned Qwen3.8-27B Q4_K_M artifact on one RTX 5090. The approved
scope is in [plan.md](plan.md), and implementation claims and evidence are in
[implementation_ledger.md](implementation_ledger.md).

The CUDA text CLI plus OpenAI Chat Completions and Responses endpoints are usable
now. Comparative benchmarks and the release quality gate remain under
construction; unfinished operations continue to fail closed.

## Chat with Quartz now

Prerequisites are Linux/x86-64, Docker with NVIDIA GPU access, one RTX 5090,
and the pinned model at `models/Qwen3.8-27B-Q4_K_M.gguf`. The build never
downloads the model.

Build the pinned CUDA 13.0.2 image and all SM120 products:

```sh
make cuda-build
```

Create a durable checkpoint directory and start an interactive terminal:

```sh
mkdir -p checkpoints
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  qw38-cuda:13.0.2 \
  ./build/cuda/qw38 models/Qwen3.8-27B-Q4_K_M.gguf \
  --reasoning off --temperature 0 --save checkpoints/chat.qw38
```

Type a message at `user>`. Inside the CLI, `/help` lists commands; `/save
PATH`, `/load PATH`, `/reset`, and `/quit` manage the conversation. Resume a
saved session and append another user turn with:

```sh
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  qw38-cuda:13.0.2 \
  ./build/cuda/qw38 models/Qwen3.8-27B-Q4_K_M.gguf \
  --load checkpoints/chat.qw38 --save checkpoints/chat.qw38 \
  --reasoning off --temperature 0
```

For a single shell-driven turn, add `--prompt "your question"`. Use
`--reasoning low`, `medium`, or `xhigh` to show a reasoning section. Run
`./build/cuda/qw38 --help` inside the same container for all sampling, stop,
and generation-limit options.

The complete beginner explanation is
[The interactive text CLI](docs/56-interactive-text-cli.md). Model
authentication still reads the complete artifact but now uses hardware-
accelerated SHA-256 where available; see
[Hardware-accelerated model authentication](docs/57-hardware-sha256.md).

## Start the OpenAI-compatible server

The CUDA server exposes `GET /health`, `GET /v1/models`,
`POST /v1/chat/completions`, and `POST /v1/responses`. Start it with host
networking so its loopback-only default is reachable from the host:

```sh
docker run --rm --network host --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  qw38-cuda:13.0.2 \
  ./build/cuda/qw38-server models/Qwen3.8-27B-Q4_K_M.gguf
```

Then inspect readiness and the admitted model from another terminal:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Generate a deterministic response:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-q4_k_m","messages":[{"role":"user","content":"Reply with exactly: hello"}],"reasoning_effort":"none","temperature":0}'
```

Add `"stream":true` and use `curl -N` for token streaming. Chat Completions
supports text roles, reasoning, function tools/results and choice, ordinary
sampling controls, stops, usage, FIFO queueing, and disconnect cancellation.
The Responses equivalent stores an exact continuation record by default:

```sh
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-q4_k_m","input":"Reply with exactly: hello","reasoning":{"effort":"none"},"temperature":0}'
```

Pass the returned `id` as `previous_response_id` with a new `input` to continue
the exact token prefix. Add `"store":false` when no later continuation is
needed. Records live in `checkpoints/responses`; override that location with
`--response-dir DIR`. Add `"stream":true` and use `curl -N` for named Responses
events. See the [server-core chapter](docs/58-http-server-core.md),
[Chat Completions chapter](docs/59-chat-completions.md), and
[Responses/continuation chapter](docs/60-responses-and-continuation.md).

The beginner-oriented [implementation handbook](docs/README.md) explains each
admitted concept and links it to code, fixtures, failures, and evidence. The
latest chapter covers [GGUF parameter conversion](docs/22-gguf-conversion.md),
including folded decay, RMSNorm scales, squeezed shapes, and GDN head order.
The following [typed-weight chapter](docs/23-typed-model-weights.md) explains how
all 851 admitted ranges become safe global and per-layer fields.
The [packed-projection chapter](docs/24-packed-projections.md) then fixes the
different GDN and attention slicing rules used by scalar execution.
The [real-mixer chapter](docs/25-real-mixer-projections.md) executes complete
typed layer-0 and layer-3 projections and records their exact workspaces and
independently decoded evidence taps.
The [real GDN layer chapter](docs/26-real-gdn-layer.md) follows one layer-0 token
through normalization, both persistent states, gated output, and residual.
The [real FFN chapter](docs/27-real-ffn-layer.md) then explains the complete
layer-0 SwiGLU branch, including its three Q4_K projections, temporary memory,
numeric evidence, and remaining scheduler boundary.
The [real attention chapter](docs/28-real-attention-layer.md) follows two
layer-3 positions through partial RoPE, grouped causal lookup, KV mutation,
output gating, projection, and residual.
The [complete-layer chapter](docs/29-complete-decoder-layer.md) joins each mixer
variant to its SwiGLU branch and explains both residuals, validation order,
memory lifetimes, and the remaining full-scheduler boundary.
The [embedding and logits chapter](docs/30-embeddings-and-logits.md) explains how
token IDs become hidden vectors and how final normalized state becomes all
248,320 vocabulary scores.
The [full scalar token chapter](docs/31-full-scalar-token.md) follows one real
token through all 48 GDN and 16 attention layers, final logits, state ownership,
scratch reuse, and the remaining oracle boundary.
The [trace bundle chapter](docs/32-trace-bundles-and-metrics.md) explains how
diagnostic taps are stored, authenticated, and compared using absolute,
relative, RMS, cosine, non-finite, first-failure, and top-logit evidence.
The [diagnostic isolation chapter](docs/33-diagnostic-trace-isolation.md) then
defines stable tap names and exact filters while proving trace machinery is
absent from the normal binary.
The [real scalar trace chapter](docs/34-real-scalar-traces.md) maps every tap to
its execution moment and shape, then follows one filtered real tensor through a
model-identified, state-aware trace-v1 bundle.
The [scalar chunk chapter](docs/35-scalar-token-chunks.md) explains multi-token
positions, whole-request preflight, vocabulary-logit row layout, and exact
equivalence with repeated one-token execution.
The [independent-authority chapter](docs/36-independent-llama-authority.md)
explains why Quartz is checked against both llama.cpp and Transformers, how the
same GGUF and raw token history are enforced, and why numeric reporting precedes
the tolerance freeze.
The [Transformers authority chapter](docs/37-transformers-authority.md) then
explains original Safetensors checkpoints, eager execution, GPU/CPU/disk
offload, diagnostic hooks and taps, and exactly what the official-checkpoint
comparison does and does not prove.
The [scalar tolerance chapter](docs/38-scalar-authority-tolerances.md) aligns all
three authorities, explains exact layout normalization and numeric error
metrics, and freezes the immutable scalar gates CUDA implementations must pass.
The [CUDA MMV chapter](docs/39-cuda-quant-mmv.md) then explains the first SM120
kernel, transient BF16-to-Q8 staging, warp-per-row ownership, and the measured
scalar-versus-device admission boundary.
The [CUDA prompt MMQ chapter](docs/40-cuda-prompt-mmq.md) extends that boundary
to arbitrary prompt rows with explicit two-dimensional tiles, packed-weight
reuse, token-major output, and tail handling.
The [CUDA GDN step chapter](docs/41-cuda-gdn-step.md) adds the first stateful GPU
model core and explains committed versus candidate recurrence, cancellation,
and frontier-last publication.
The [chunked CUDA GDN chapter](docs/42-cuda-gdn-chunks.md) carries that state
through arbitrary prompt chunks using continuous 64-token windows and proves
byte-exact equivalence with repeated one-token GPU execution.
The [CUDA attention decode chapter](docs/43-cuda-attention-decode.md) explains
grouped query-to-KV sharing, partial RoPE, BF16 cache rows, causal reads, stable
softmax, and candidate-row commit on the GPU.
The [CUDA attention prefill chapter](docs/44-cuda-attention-prefill.md) extends
that state through atomic prompt chunks with linear score storage and executes
the final legal position in a real 131,072-row production cache.
The [CUDA scheduler-prerequisite chapter](docs/45-cuda-scheduler-primitives.md)
then explains resident Q8_0 weights versus temporary Q8 activations, embedding
row lookup, BF16 pointwise operations, and the packed-layout conversions needed
before the complete 64-layer scheduler can be assembled.
The [complete CUDA token chapter](docs/46-cuda-full-scheduler.md) follows two
real tokens through resident model upload, all 48 GDN and 16 attention layers,
state continuation, selected trace boundaries, and every vocabulary logit.
The [exact prefix-sync chapter](docs/47-cuda-prefix-sync.md) explains token
histories and common prefixes, then proves append/no-op reuse and deterministic
replay for divergent or shortened requests with byte-exact state and outputs.
The [atomic evaluation chapter](docs/48-atomic-eval-and-sampling.md) explains
candidate state, failure/cancellation preservation, frontier-last commit, and
why sampling committed logits must not mutate the session.
The [checkpoint chapter](docs/49-cuda-checkpoints.md) explains the complete
versioned disk format, atomic durable publication, corruption and compatibility
checks, sampler persistence, and byte-exact resumed continuation.
The [pre-graph 128K memory chapter](docs/50-pre-graph-128k-memory.md) records the
first simultaneous full-capacity allocation and remaining reserve while keeping
the final post-graph MEM-001 admission explicitly open.
Chapters 51–55 cover timing, fusion, CUDA graphs, the final admitted 128K
allocation, and offline RTX 5090 dispatch tuning. Chapters 56–58 cover the
working interactive CLI, accelerated model authentication, the HTTP control
plane, and Chat Completions data plane. Chapters 59–61 cover Chat Completions,
Responses continuation, and the benchmark harness.

## Measure a local smoke workload

`qw38-bench` records raw runs, summaries, environment identity, telemetry, and
failures in an atomic JSON result. This short example validates the harness; its
`--smoke` result is deliberately not release-admission evidence:

```sh
revision="$(git rev-parse HEAD)"
state=clean
test -z "$(git status --porcelain)" || state=dirty
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" qw38-cuda:13.0.2 \
  ./build/cuda/qw38-bench models/Qwen3.8-27B-Q4_K_M.gguf \
  --workload decode --prompt "Quartz benchmark prompt." \
  --context-label smoke --output-tokens 2 --warmups 1 --samples 2 \
  --cache-policy disabled --source-revision "$revision" \
  --source-state "$state" --smoke \
  --output evidence/benchmark/local-decode-smoke.json
```

Release runs omit `--smoke`, require a clean source revision, an exact expected
prompt-token count, at least three warm-ups, and at least 30 samples. See
[The benchmark harness](docs/61-benchmark-harness.md) for every term, metric,
limitation, and the BEN-001 measurement that led to SCH-002.

SCH-002 now connects 64-row prompt MMQ and recurrent/attention chunks while
preserving byte-exact one-token results. The focused 17-token smoke is faster,
but it is not a release comparison; see
[Chunked full-model CUDA prefill](docs/62-cuda-full-prefill.md).

## Build

```sh
make
make test
```

The host-only build establishes and tests the public boundary. CUDA compilation
uses the immutable CUDA 13.0 container:

```sh
make cuda-image
make cuda-build
```

Run `build/qw38-eval --build-info` to inspect the compiled target and artifact
pin. No model is downloaded by the build.
