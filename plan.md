# Quartz Watch 38 v1 — Swiss-Made Qwen3.8 Engine

## First Artifacts and Working Protocol

Before any source code, build files, containers, or other documentation, create
this file and then `implementation_ledger.md` at the repository root.

- This plan is the approved implementation baseline. Change it only after an
  approved material scope or architecture revision, recorded as a dated change
  note without silently rewriting earlier decisions.
- `implementation_ledger.md` is the living operational source of truth. It must
  give each task a stable ID, description, dependencies, status, acceptance
  condition, and evidence links. Valid statuses are `pending`, `in_progress`,
  `blocked`, `done`, and `superseded`.
- The ledger keeps a chronological UTC log of work, commands, tests,
  measurements, errors, diagnoses, decisions, negative results, and resulting
  tasks. Failed approaches are retained with resolutions and follow-up IDs.
- Discovered work is entered before implementation, with its reason and affected
  gate. Update the ledger at task start/end, after discoveries, and before every
  commit. A task is done only after its acceptance evidence is recorded.

## Product and Architecture

Build a narrow inference system for Qwen3.8-27B on one RTX 5090, following
DwarfStar's philosophy: specialize aggressively, keep the code understandable,
develop correctness and speed together, and avoid a generic tensor framework.

V1 delivers:

- Restricted C++17 host runtime and CUDA C++ kernels for Linux/x86-64 `sm_120`.
- `qw38` interactive text CLI.
- `qw38-server` implementing OpenAI Chat Completions and Responses.
- `qw38-bench` for component and end-to-end performance.
- `qw38-eval` for quality, logits, traces, and checkpoint verification.
- One pinned Qwen3.8-27B Q4_K_M GGUF artifact.
- A guaranteed 131,072-token context on the 32 GiB RTX 5090.
- Atomic one-session disk persistence and exact token-prefix reuse.
- The existing educational handbook, augmented by code-linked notes and the
  implementation ledger.

V1 excludes vision, MTP/speculative decoding, continuous batching, concurrent
GPU sessions, Anthropic APIs, a coding-agent executable, and non-CUDA production
backends.

Use the literal brand line: **“Quartz Watch 38 — Swiss-made inference for
Qwen3.8-27B. It ticks fast.”**

### Model and execution design

- Validate the exact official 64-layer `qwen3_5` contract: 48 GDN layers, 16
  full-attention layers, 5,120 residual width, FP32 recurrence, grouped-query
  attention, and partial RoPE.
- Accept only the pinned GGUF identity. Inventory and hash every tensor, shape,
  dtype, byte range, and semantic role; fail closed on incompatible artifacts.
- Memory-map the canonical file, prepare resident CUDA weights once, and allow
  disposable SM120 repack caches keyed by model, quantization, architecture, and
  layout version.
- Use BF16 activation storage, FP32 sensitive accumulation and GDN recurrence,
  FP32 logits, and two-byte KV storage.
- The 128K fit gate includes weights, 8 GiB KV, GDN state and convolution rings,
  workspaces, repacks, graphs, allocator overhead, and at least 1.5 GiB free
  reserve after graph creation.
- Use a simple Makefile and pinned CUDA 13.0 Docker environment. Avoid
  exceptions, RTTI, inheritance-heavy backends, operator registries, and
  speculative abstractions.
- Adapt focused MIT-licensed code or techniques from DwarfStar and llama.cpp
  with explicit file-level provenance. Do not fork DwarfStar or translate its
  model-specific machinery.

### Public engine boundary

Provide move-only `Engine` and `Session` owners with explicit `Status` returns:

- `Engine::open`: validate and prepare the model.
- `Engine::create_session`: allocate an independent hybrid timeline.
- `Session::sync`: reuse or restore a common token prefix and prefill its suffix.
- `Session::logits`, `sample`, and `eval`: keep sampling separate from atomic
  state commit.
- `Session::save` and `restore`: persist every GDN matrix, convolution ring, KV
  row, token, position, sampler field, and compatibility hash.

CLI and server code operate on tokens, logits, and sessions; they never
manipulate tensors or device layouts.

The single-flight server supplies `/health`, `/v1/models`,
`/v1/chat/completions`, and `/v1/responses`. Support streaming, reasoning,
system/developer/user/assistant/tool roles, function calls and results, tool
choice, `previous_response_id`, ordinary sampling controls, cancellation, and
stop conditions. Explicitly reject image, audio, file, structured-output, and
unsupported protocol features.

## Correctness, Tracing, Testing, and Education

Create a versioned diagnostic trace bundle containing:

- A JSON manifest and little-endian tensor blobs.
- Model/tool revisions and hashes.
- Prompt bytes, rendered token IDs, and positions.
- Tensor names, shapes, dtypes, checksums, and numeric summaries.
- Session state before and after execution.
- Stable taps for embeddings, norms, GDN stages/state, attention Q/K/RoPE/KV/
  output, FFN stages, residuals, final norm, and logits.

Expensive tensor tracing is compiled only into diagnostic builds. Trace mode
supports layer/tap filters and disables graphs or fusion where necessary.
Optimized fused paths compare against retained unfused implementations at their
visible boundaries.

Use Transformers eager/offloaded execution as the primary semantic authority, a
straightforward Quartz scalar CPU path as the inspectable local oracle, pinned
llama.cpp debug traces as an independent same-GGUF oracle, and pinned vLLM
operator scripts for attributed component comparisons rather than final semantic
authority.

Correctness policy:

- Artifact hashes, tokenization, templates, layer schedules, state frontiers,
  quant block decoding, and checkpoint metadata are exact.
- Numeric fixtures report absolute, relative, RMS, cosine, NaN/Inf,
  first-failing-index, and top-logit differences.
- Freeze per-tap tolerances during the scalar milestone, before CUDA
  optimization. An optimization cannot loosen the tolerance used to admit it.
- Require greedy continuation equality unless a stored fixture demonstrates a
  genuine near-tie; tensor and logit gates remain mandatory.

All automated tests run through pytest with typed Python helpers invoking focused
native diagnostics. Cover:

- GGUF validation and malformed input.
- Tokenizer, template, roles, reasoning, and tool rendering.
- Q4_K/Q6_K decoding and dot products.
- GDN convolution warm-up, recurrence order, state mutation, head mapping, and
  chunk boundaries.
- Attention layers 3/7/63, partial RoPE, grouped KV, causality, and 128K capacity
  boundaries.
- Token-wise decode versus arbitrary chunked prefill.
- Scalar versus CUDA, fused versus unfused, and graph versus non-graph.
- Atomic failure behavior and uninterrupted versus save/restore continuation.
- API streaming, tools, cancellation, queueing, and response continuation.
- Held-out NLL, greedy continuations, long-recurrence drift, 128K retrieval, and
  task fixtures.

Documentation is a gate, not cleanup. Update relevant handbook material beside
each implemented concept; link it to code and fixtures; record actual layouts,
invariants, failure modes, measurements, and negative results; keep provenance
current; label claims measured, external, estimated, or proposed; and reference
the corresponding ledger task and evidence from each substantive update.

## CUDA Optimization and Measurement

Implement in this order:

1. Q4_K/Q6_K decode MMV with transient row quantization and FP32 accumulation.
2. Quantized tiled MMQ for prompt rows.
3. Exact one-token GDN execution and state commit.
4. Chunked GDN prefill with internal 64-token scans.
5. Grouped-query attention with partial RoPE and memory-bounded causal prefill.
6. Full 64-layer scheduler and logits.
7. Profiler-justified norm, residual, gate, and state fusion.
8. Stable-address CUDA graphs.
9. Offline RTX 5090 tuning of row buckets and chunk sizes, with the selected
   dispatch table and evidence checked in.

Instrumentation includes synchronized CUDA-event timings, NVTX ranges, Nsight
Systems timelines, and targeted Nsight Compute reports. Attribute time to
loading, embedding, GDN, attention, FFN, logits, sampling, graph launch,
queueing, persistence, and idle gaps.

Use pinned Docker profiles for Quartz, Transformers, llama.cpp, Ollama, vLLM,
and profiling. Record image digests, source revisions, model hashes, compiler
flags, driver/toolkit versions, clocks, power, temperature, and raw samples.
Enforce one large GPU process at a time.

### Comparison protocol

- Quartz, llama.cpp, and Ollama use the identical Q4_K_M GGUF.
- Attempt the same GGUF with vLLM first. If its experimental GGUF plugin cannot
  run Qwen3.8 correctly, select a pinned viable vLLM quant whose held-out NLL is
  within 1% of Quartz and whose task and long-context scores are not worse.
- Keep the vLLM artifact difference visible in every result.
- Use vLLM operator diagnostics for subcomponent attribution and its unmodified
  server for end-to-end results.
- Report Ollama independently, but use llama.cpp rather than counting both
  related runtimes separately in the hard gate.

Benchmark cold prefill at 128, 2K, and 8K tokens; and batch-1 generation of 256
tokens at 128, 2K, 8K, 32K, and 128K context. Primary comparisons disable prefix
reuse, with a separately labeled agent-style workload enabling it. Use at least
three warm-ups and 30 measured samples per case. Report load time, TTFT, prompt
tokens/s, p50/p95 ITL, output tokens/s, peak/reserved VRAM, host memory, queue
time, power, clocks, and temperature.

Against both eligible llama.cpp and vLLM baselines, v1 requires:

- At least 5% higher geometric-mean prefill throughput, with the paired 95%
  bootstrap confidence lower bound above 1.05.
- At least 5% better geometric-mean batch-1 decode performance under the same
  rule.
- No individual workload more than 5% slower by point estimate.
- Matching inputs, template, sampling, context, cache policy, and admitted
  quality.
- Preservation of failed, OOM, incorrect, and throttled runs in the ledger.

## Delivery Gates

1. Create and populate `plan.md`.
2. Create and populate `implementation_ledger.md`.
3. Pin all models, tools, containers, revisions, and expected hashes.
4. Establish branding, Makefile/Docker builds, public API skeleton, and
   allocation ledger.
5. Complete artifact loading, tokenizer/template, scalar execution, trace
   format, and frozen oracle fixtures.
6. Pass dense CUDA quantization and primitive gates.
7. Pass GDN, attention, hybrid scheduling, logits, and chunked-prefill gates.
8. Demonstrate atomic session behavior, 128K memory fit, prefix reuse, and disk
   continuation.
9. Complete profiler-led kernels, fusion, graphs, and dispatch tuning.
10. Complete CLI, OpenAI server, benchmark/eval tools, API tests, and quality
    suite.
11. Pass the comparative speed gate.
12. Publish a reproducible release evidence bundle and reconcile every
    documentation claim with measured evidence.

## Assumptions

- Existing untracked `docs/` content is preserved and evolved.
- Swiss provenance supports a literal “Swiss-made” statement.
- The local RTX 5090 is the sole production target for v1.
- The upstream multimodal wrapper is used only for its text model in v1.
- Newly discovered work is added to the ledger before execution and cannot
  silently expand the release boundary.

## Approved Change Notes

### 2026-09-02 — Additive Darwin/x86_64 CPU laptop track for Qwen3.5-2B

This is an additive track. It does not replace v1 CUDA, loosen any RTX 5090
gate, or change the pinned Qwen3.8-27B Q4_K_M identity.

V1 remains Linux/x86-64 `sm_120` on one RTX 5090. In addition, Quartz may admit
one second pinned dense `qwen35` artifact — official Qwen3.5-2B Q4_K_M — on the
host scalar runtime for Darwin/x86_64 laptops without a GPU. The first measured
target is a 16 GiB Intel Core i7 MacBook (AVX2, no Metal/OpenCL). Default host
context is 4096 tokens. Tied embeddings are required. CUDA kernels stay
27B-specialized.

Still excluded from this track: Linux/Windows CPU product builds, Apple Silicon,
AVX-512, BLAS, Iris/Metal, 4B/9B/27B-on-CPU, vision, MTP, and 128K laptop
context. A later change note is required before generalizing to any CPU laptop.

### 2026-09-04 — Darwin/x86_64 AVX2 class (not one SKU)

This is an additive clarification of the Darwin Qwen3.5-2B host path. It does
not replace v1 CUDA, admit Apple Silicon, or change the pinned 2B GGUF.

The supported class is Intel MacBook + AVX2, not the measured i7-8569U SKU.
Host matvec workers are `hw.logicalcpu` clamped to `[1, 8]` so this 4c/8t
laptop stays at eight workers. Darwin/x86_64 builds fail closed without AVX2
at compile time; host `Engine::open` fails closed without AVX2 at runtime.
Throughput numbers remain measured on this SKU, not a portable SLO.

Apple Silicon, NEON, Metal, raising the eight-worker cap, 27B-on-CPU, and
Linux/Windows CPU product remain excluded.

### 2026-09-02 — Additive 2B load/forward numeric gates

This is an additive numeric-proof track for the Darwin Qwen3.5-2B host path. It
does not change v1 CUDA, the RTX 5090 gates, or frozen 27B tap hex / llama
fixtures.

Quartz may freeze skip-if-missing 2B checks for GGUF bind, tokenizer IDs, one
GDN layer, one attention layer, one FFN, one scalar token, AVX2-vs-scalar
matvec, Q5_K decode, and a Darwin CPU llama.cpp same-GGUF logit comparison.
Transformers-on-HF remains out of scope (it cannot catch GGUF load bugs).
CUDA tasks are unchanged.
