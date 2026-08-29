# Quartz Watch 38 Implementation Ledger

This is the living operational source of truth for the v1 implementation. The
approved scope and architecture are in [`plan.md`](plan.md). Valid task statuses
are `pending`, `in_progress`, `blocked`, `done`, and `superseded`. Evidence links
are repository-relative unless stated otherwise.

## Gates and Tasks

| ID | Description | Dependencies | Status | Acceptance condition | Evidence |
|---|---|---|---|---|---|
| ART-001 | Save the approved implementation baseline | — | done | Complete approved plan exists at repository root | [`plan.md`](plan.md); log 2026-08-29T00:00:00Z |
| ART-002 | Create and populate the implementation ledger | ART-001 | done | All known gates have stable tasks, dependencies, acceptance conditions, and a UTC log | This file; log 2026-08-29T00:01:00Z |
| PIN-001 | Pin production GGUF identity and expected SHA-256 | ART-002 | done | Model URL/revision, size, identity metadata, and verified SHA-256 are checked in | [`pins/artifacts.lock.json`](pins/artifacts.lock.json); local hash log 2026-08-29T10:00:59Z |
| PIN-002 | Pin external semantic and performance authorities | ART-002 | done | Transformers, llama.cpp, vLLM, Ollama, and DwarfStar revisions and licenses are recorded | [`pins/artifacts.lock.json`](pins/artifacts.lock.json); log 2026-08-29T09:55:00Z |
| PIN-003 | Pin build, runtime, and profiling containers | PIN-002 | done | Dockerfiles/lock data use immutable bases or recorded image digests and CUDA 13.0 | [`docker/cuda.Dockerfile`](docker/cuda.Dockerfile); container build log 2026-08-29T09:55:00Z |
| ENV-001 | Capture and validate the local production-toolchain prerequisites | ART-002 | done | GPU/driver/toolkit/container availability is recorded; unavailable prerequisites have explicit follow-up | CUDA probe log 2026-08-29T09:55:00Z |
| BLD-001 | Establish brand, repository layout, and C++17 build | ART-002 | done | Literal brand appears in user-facing tools; Makefile builds restricted host targets | [`Makefile`](Makefile), [`include/qw38/engine.h`](include/qw38/engine.h), pytest log 2026-08-29T09:52:00Z |
| BLD-002 | Add pinned CUDA 13.0 SM120 build path | PIN-003, BLD-001 | in_progress | Diagnostic and release CUDA builds target `sm_120` with recorded flags | — |
| BLD-003 | Define and enforce the device allocation ledger | PIN-001 | in_progress | All persistent/transient allocations and 128K budgets are enumerated and checked | — |
| API-001 | Implement explicit `Status` and move-only Engine/Session boundary | BLD-001 | done | Public header compiles without exceptions/RTTI and exposes the approved operations | [`include/qw38/engine.h`](include/qw38/engine.h), build log 2026-08-29T09:52:00Z |
| MDL-001 | Parse, mmap, inventory, and fail-closed validate GGUF | PIN-001, API-001 | done | Exact tensor metadata/ranges/roles are checked; malformed fixtures pass pytest | [`pins/tensor_inventory.json`](pins/tensor_inventory.json), [`src/model.cpp`](src/model.cpp); log 2026-08-29T10:35:51Z |
| MDL-002 | Validate official 64-layer hybrid model contract | MDL-001 | done | 48 GDN/16 attention schedule, width, GQA, partial RoPE, and dtypes match authority | [`pins/model_contract.json`](pins/model_contract.json), [`docs/14-artifact-validation.md`](docs/14-artifact-validation.md); log 2026-08-29T10:35:51Z |
| TOK-001 | Implement pinned tokenizer | MDL-001, PIN-002 | done | Token IDs match frozen authority fixtures byte-for-byte | [`src/tokenizer.cpp`](src/tokenizer.cpp), [`fixtures/tokenizer_authority.json`](fixtures/tokenizer_authority.json); log 2026-08-29T11:02:00Z |
| TOK-002 | Implement chat/reasoning/tool template | TOK-001 | done | All supported roles, reasoning, tool calls/results, and rejection cases match fixtures | [`src/template.cpp`](src/template.cpp), [`fixtures/template_authority.json`](fixtures/template_authority.json); log 2026-08-29T11:39:57Z |
| CPU-001 | Implement Q4_K/Q6_K scalar decoding and dot products | MDL-001 | done | Numeric fixtures meet frozen metrics and exact structural checks | [`src/quant.cpp`](src/quant.cpp); [`fixtures/quant_authority.json`](fixtures/quant_authority.json); [`tests/test_quant.py`](tests/test_quant.py); log 2026-08-29T12:05:00Z |
| CPU-005 | Implement discovered Q8_0 scalar decoding and dot products | CPU-001, MDL-001 | done | Exact 34-byte/32-value layout, signed values, FP16 scale, malformed sizes, and frozen decode/dot fixtures pass | [`src/quant.cpp`](src/quant.cpp); [`fixtures/quant_authority.json`](fixtures/quant_authority.json); [`tests/test_quant.py`](tests/test_quant.py); log 2026-08-29T13:07:00Z |
| CPU-006 | Bind admitted tensor rows and implement mixed-format scalar matvec | CPU-001, CPU-005, MDL-001 | done | GGUF dimension order, row bounds/bytes, F32/Q8_0/Q4_K/Q6_K dots, malformed views, synthetic matrices, and admitted artifact rows pass | [`src/tensor.cpp`](src/tensor.cpp); [`fixtures/tensor_rows.json`](fixtures/tensor_rows.json); [`tests/test_tensor.py`](tests/test_tensor.py); log 2026-08-29T13:28:00Z |
| CPU-007 | Implement pinned GGUF-to-semantic parameter and GDN head-layout transforms | CPU-002, CPU-006, PIN-002 | done | Folded A, RMSNorm convention, squeezed convolution, grouped/tiled head permutations, round trips, and admitted parameter fixtures pass | [`src/conversion.cpp`](src/conversion.cpp); [`pins/gguf_conversion_contract.json`](pins/gguf_conversion_contract.json); [`fixtures/gguf_conversion.json`](fixtures/gguf_conversion.json); [`tests/test_conversion.py`](tests/test_conversion.py); log 2026-08-29T17:29:00Z |
| CPU-008 | Bind every admitted global and layer tensor into typed scalar weight structures | CPU-006, CPU-007, MDL-002 | done | All 851 tensors bind by exact name, layer kind, shape, dtype, and mapped range; missing, swapped, or incompatible roles fail closed | [`src/weights.cpp`](src/weights.cpp); [`src/tensor.cpp`](src/tensor.cpp); [`tests/test_weights.py`](tests/test_weights.py); log 2026-08-29T17:36:00Z |
| CPU-009 | Implement exact packed GDN QKV and attention query/gate projection slicing | CPU-003, CPU-007, CPU-008 | done | GDN contiguous Q/K/V ranges, per-head attention query/gate halves, output counts, alias rejection, and frozen layout fixtures pass | [`src/projection.cpp`](src/projection.cpp); [`pins/projection_layout_contract.json`](pins/projection_layout_contract.json); [`tests/test_projection.py`](tests/test_projection.py); log 2026-08-29T17:40:00Z |
| CPU-010 | Execute typed real-artifact GDN and attention mixer projections with exact scalar workspaces | CPU-006, CPU-008, CPU-009 | done | Deterministic activation drives complete layer-0/layer-3 mixer projections; packed and split taps match independently decoded admitted rows and workspace guards fail closed | [`src/mixer.cpp`](src/mixer.cpp); [`fixtures/mixer_projections.json`](fixtures/mixer_projections.json); [`tests/test_mixer.py`](tests/test_mixer.py); log 2026-08-29T18:20:00Z |
| CPU-011 | Execute one complete real layer-0 GDN mixer update and residual | CPU-002, CPU-007, CPU-010 | done | GGUF-scale input norm, real projections, convolution ring, grouped recurrence, gated norm, output projection, residual, state taps, and malformed workspaces pass frozen evidence | [`src/mixer.cpp`](src/mixer.cpp); [`fixtures/real_gdn_step.json`](fixtures/real_gdn_step.json); [`tests/test_real_gdn.py`](tests/test_real_gdn.py); log 2026-08-29T18:33:00Z |
| CPU-012 | Execute one complete real Q4_K SwiGLU FFN branch and residual | CPU-003, CPU-008, CPU-011 | done | Direct-scale FFN norm, complete gate/up/down projections, SwiGLU taps, exact workspace, residual addition, and malformed workspace behavior pass frozen evidence | [`src/mixer.cpp`](src/mixer.cpp); [`fixtures/real_ffn_step.json`](fixtures/real_ffn_step.json); [`tests/test_real_ffn.py`](tests/test_real_ffn.py); log 2026-08-29T18:46:07Z |
| CPU-013 | Execute real layer-3 grouped-query attention steps, KV mutation, output projection, and residual | CPU-003, CPU-008, CPU-009, CPU-010 | done | Direct-scale norms, packed projection split, partial RoPE, two-position grouped causal attention, KV state, output gate/projection, residual, capacity, and malformed buffers pass frozen evidence | [`src/mixer.cpp`](src/mixer.cpp); [`fixtures/real_attention_step.json`](fixtures/real_attention_step.json); [`tests/test_real_attention.py`](tests/test_real_attention.py); log 2026-08-29T18:54:26Z |
| CPU-002 | Implement scalar GDN oracle | CPU-001, MDL-002 | done | Warm-up, recurrence, state, head mapping, and chunk-boundary fixtures pass | [`src/gdn.cpp`](src/gdn.cpp); [`fixtures/gdn_authority.json`](fixtures/gdn_authority.json); [`tests/test_gdn.py`](tests/test_gdn.py); log 2026-08-29T12:26:00Z |
| CPU-003 | Implement scalar attention and FFN oracle | CPU-001, MDL-002 | done | Layers 3/7/63, partial RoPE, grouped KV, causality, and FFN taps pass | [`src/attention.cpp`](src/attention.cpp); [`fixtures/attention_ffn_authority.json`](fixtures/attention_ffn_authority.json); [`tests/test_attention.py`](tests/test_attention.py); log 2026-08-29T12:54:00Z |
| CPU-004 | Implement full scalar 64-layer scheduler and logits | CPU-002, CPU-003, CPU-005, CPU-006, CPU-007, CPU-008, CPU-009, CPU-010, CPU-011, CPU-012, CPU-013 | pending | Token/chunk execution and logits match semantic-authority fixtures | — |
| TRC-001 | Define versioned trace bundle and typed comparison metrics | PIN-002 | pending | Manifest/blob schema, checksums, summaries, session frontiers, and metric reporter pass tests | — |
| TRC-002 | Add diagnostic-only stable scalar/CUDA taps | TRC-001, CPU-004 | pending | Required taps filter by layer/name and are absent from release builds | — |
| ORA-001 | Generate and freeze scalar/oracle fixtures and tolerances | TOK-002, CPU-004, TRC-002 | pending | Three authorities are attributed; tolerances and greedy tie exceptions are immutable inputs | — |
| CUD-001 | Implement CUDA Q4_K/Q6_K decode MMV | CPU-001, BLD-002, ORA-001 | pending | Scalar-vs-CUDA and focused primitive pytest gates pass | — |
| CUD-002 | Implement quantized tiled prompt MMQ | CUD-001 | pending | Arbitrary prompt-row fixtures pass frozen tolerances | — |
| GDN-001 | Implement exact one-token CUDA GDN and atomic state commit | CUD-001, CPU-002 | pending | State/taps match oracle and injected failures leave frontier unchanged | — |
| GDN-002 | Implement chunked GDN prefill with 64-token scans | GDN-001, CUD-002 | pending | Arbitrary chunks equal token-wise execution under frozen gates | — |
| ATN-001 | Implement grouped-query attention and partial RoPE | CUD-001, CPU-003 | pending | Decode, causality, KV grouping, and layers 3/7/63 pass | — |
| ATN-002 | Implement memory-bounded causal attention prefill | ATN-001, CUD-002 | pending | Chunked prompt fixtures and 131,072 capacity boundary pass | — |
| SCH-001 | Implement hybrid 64-layer CUDA scheduler and FP32 logits | GDN-002, ATN-002 | pending | Full traces/logits and greedy continuations meet frozen gates | — |
| SES-001 | Implement exact common-prefix sync/reuse | SCH-001 | pending | Reuse and full replay produce the same committed state and logits | — |
| SES-002 | Implement atomic eval/sample/commit semantics | SCH-001 | pending | Sampling is separate; cancellation/error cannot partially commit state | — |
| SES-003 | Implement atomic checkpoint save/restore | SES-001, SES-002 | pending | All state and compatibility hashes persist; resumed continuation is exact | — |
| MEM-001 | Demonstrate 131,072-token fit with 1.5 GiB reserve | BLD-003, SCH-001 | pending | Post-graph measured ledger includes 8 GiB KV and every named allocation on RTX 5090 | — |
| OPT-001 | Add synchronized timings, NVTX, and attribution | SCH-001 | pending | Component/end-to-end measurements expose every named time category | — |
| OPT-002 | Profile and implement justified fusions | OPT-001, ORA-001 | pending | Nsight evidence justifies each fusion; fused/unfused boundaries pass frozen gates | — |
| OPT-003 | Implement stable-address CUDA graphs | OPT-002 | pending | Graph/non-graph equivalence passes and graph allocations are in MEM-001 | — |
| OPT-004 | Tune row buckets/chunks and check in dispatch evidence | OPT-003 | pending | Offline RTX 5090 sweep selects a reproducible table from retained raw results | — |
| CLI-001 | Implement interactive `qw38` text CLI | TOK-002, SES-003 | pending | Interactive generation, reasoning, stops, sampling, and persistence pass smoke tests | — |
| SRV-001 | Implement single-flight HTTP server core and queue | API-001 | pending | Health/models endpoints, cancellation, queue timing, and one GPU session pass tests | — |
| SRV-002 | Implement Chat Completions API | TOK-002, SES-002, SRV-001 | pending | Supported roles/tools/streaming/sampling/stops pass; exclusions reject explicitly | — |
| SRV-003 | Implement Responses API and continuation | TOK-002, SES-003, SRV-001 | pending | Streaming/tools/`previous_response_id` and exclusions pass API fixtures | — |
| BEN-001 | Implement `qw38-bench` component/end-to-end harness | OPT-001 | pending | Warmups/samples, telemetry, raw samples, failures, and environment metadata are retained | — |
| EVAL-001 | Implement `qw38-eval` logits/traces/checkpoints harness | ORA-001, SES-003 | pending | Focused native diagnostics are driven by typed pytest helpers | — |
| QLT-001 | Pass held-out NLL, continuation, recurrence, retrieval, and task quality | EVAL-001, MEM-001 | pending | Admitted artifact passes every documented threshold and 128K retrieval fixture | — |
| CMP-001 | Pin and validate comparable baseline artifacts | PIN-001, PIN-002, QLT-001 | pending | llama/Ollama share GGUF; vLLM difference and <=1% NLL admission are explicit | — |
| CMP-002 | Run controlled 30-sample comparative matrix | BEN-001, OPT-004, CMP-001 | pending | All contexts/metrics/environment data and negative runs are retained | — |
| CMP-003 | Pass prefill/decode statistical speed gates | CMP-002 | pending | Paired bootstrap lower bounds exceed 1.05 and no workload is >5% slower | — |
| DOC-001 | Maintain code-linked handbook and provenance ledger | BLD-001 | pending | Each implemented concept has claim labels, invariants, failures, task IDs, and evidence | — |
| EDU-001 | Explain tokenizer concepts for readers with no prior background | TOK-001, DOC-001 | done | NFC, Unicode splitting, byte mapping, BPE, fixtures, and equality gates have worked examples linked to code/evidence | [`docs/15-tokenizer-authority.md`](docs/15-tokenizer-authority.md); tests; log 2026-08-29T11:02:00Z |
| EDU-002 | Explain chat-template concepts and policy ownership for beginners | TOK-002, DOC-001 | done | Roles, delimiters, reasoning, tools, results, mapping, and byte-equality gates have worked examples linked to code/evidence | [`docs/16-chat-template.md`](docs/16-chat-template.md); tests; log 2026-08-29T11:39:57Z |
| EDU-003 | Explain scalar quantization and numeric equality for beginners | CPU-001, DOC-001 | done | Bits/bytes, FP16/FP32, blocks, Q4_K/Q6_K packing, decoding, dot products, accumulation, fixtures, and numeric metrics have worked examples linked to code/evidence | [`docs/17-quantization.md`](docs/17-quantization.md); [`tests/test_quant.py`](tests/test_quant.py); log 2026-08-29T12:05:00Z |
| EDU-004 | Explain GDN recurrence and persistent state for beginners | CPU-002, DOC-001 | done | Projections, heads, convolution warm-up/rings, gates, delta-rule recurrence, mutation order, chunk equivalence, and FP32 state have worked examples linked to code/evidence | [`docs/18-gated-delta-network.md`](docs/18-gated-delta-network.md); [`tests/test_gdn.py`](tests/test_gdn.py); log 2026-08-29T12:26:00Z |
| EDU-005 | Explain grouped causal attention, partial RoPE, and SwiGLU for beginners | CPU-003, DOC-001 | done | Q/K/V, KV history, causality, grouped heads, RoPE pairs/positions, softmax, RMSNorm, gate/up/down FFN, fixtures, and numeric gates have worked examples linked to code/evidence | [`docs/19-attention-and-ffn.md`](docs/19-attention-and-ffn.md); [`tests/test_attention.py`](tests/test_attention.py); log 2026-08-29T12:54:00Z |
| EDU-006 | Explain Q8_0 and scalar matrix rows for beginners | CPU-005, DOC-001 | done | Signed bytes, per-block scale, row blocks, decode/dot use, format selection, fixtures, and full-scheduler dependency are code-linked and worked | [`docs/20-q8-scalar-rows.md`](docs/20-q8-scalar-rows.md); [`tests/test_quant.py`](tests/test_quant.py); log 2026-08-29T13:07:00Z |
| EDU-007 | Explain GGUF tensor dimensions, row orientation, and matvec for beginners | CPU-006, DOC-001 | done | Shape order, fastest dimension, rows/outputs, block alignment, mixed formats, bounds, synthetic/admitted fixtures, and scheduler use are code-linked and worked | [`docs/21-tensor-rows.md`](docs/21-tensor-rows.md); [`tests/test_tensor.py`](tests/test_tensor.py); log 2026-08-29T13:28:00Z |
| EDU-008 | Explain converter-owned parameter folding and GDN head reordering for beginners | CPU-007, DOC-001 | done | Source vs GGUF semantics, folded exponent, norm weight convention, squeeze, grouped/tiled indices, affected tensors, inverse transforms, and checkpoint ownership are code-linked and worked | [`docs/22-gguf-conversion.md`](docs/22-gguf-conversion.md); [`tests/test_conversion.py`](tests/test_conversion.py); log 2026-08-29T17:29:00Z |
| EDU-009 | Explain typed model weights and complete layer binding for beginners | CPU-008, DOC-001 | done | Views versus ownership, vectors/matrices, common and variant layer fields, exact-name/schema admission, counts, and scheduler boundary are code-linked and worked | [`docs/23-typed-model-weights.md`](docs/23-typed-model-weights.md); [`tests/test_weights.py`](tests/test_weights.py); log 2026-08-29T17:36:00Z |
| EDU-010 | Explain packed projection layouts and slicing for beginners | CPU-009, DOC-001 | done | Packing purpose, GDN contiguous segments, attention per-head halves, physical versus semantic order, aliasing, and downstream conversion are code-linked and worked | [`docs/24-packed-projections.md`](docs/24-packed-projections.md); [`tests/test_projection.py`](tests/test_projection.py); log 2026-08-29T17:40:00Z |
| EDU-011 | Explain real scalar projection execution and workspaces for beginners | CPU-010, DOC-001 | done | Activations, projection rows, typed views, packed/split workspace sizes, selected-tap evidence, cost, and remaining layer boundary are code-linked and worked | [`docs/25-real-mixer-projections.md`](docs/25-real-mixer-projections.md); [`tests/test_mixer.py`](tests/test_mixer.py); log 2026-08-29T18:20:00Z |
| EDU-012 | Explain a complete real GDN mixer layer update for beginners | CPU-011, DOC-001 | done | Input norm, convolution state, physical/semantic head order, recurrence state, gated norm, output projection, residual, atomicity boundary, and evidence taps are code-linked and worked | [`docs/26-real-gdn-layer.md`](docs/26-real-gdn-layer.md); [`tests/test_real_gdn.py`](tests/test_real_gdn.py); log 2026-08-29T18:33:00Z |
| EDU-013 | Explain a complete real SwiGLU FFN branch for beginners | CPU-012, DOC-001 | done | Post-mixer norm, gate/up/down roles, SiLU and elementwise product, intermediate width, Q4_K cost, workspace, residual, evidence limits, and layer boundary are code-linked and worked | [`docs/27-real-ffn-layer.md`](docs/27-real-ffn-layer.md); [`tests/test_real_ffn.py`](tests/test_real_ffn.py); log 2026-08-29T18:46:07Z |
| EDU-014 | Explain real grouped-query attention and KV mutation for beginners | CPU-013, DOC-001 | done | Query/key/value, head normalization, partial RoPE, grouped head mapping, causal scores/softmax, KV ownership, output gate/projection, residual, and atomicity are code-linked and worked | [`docs/28-real-attention-layer.md`](docs/28-real-attention-layer.md); [`tests/test_real_attention.py`](tests/test_real_attention.py); log 2026-08-29T18:54:26Z |
| REL-001 | Publish reproducible release evidence bundle | CMP-003, QLT-001, DOC-001 | pending | Builds, hashes, raw results, reports, and documentation claims reconcile | — |

## Delivery-Gate Mapping

| Gate | Tasks |
|---|---|
| 1. Approved plan | ART-001 |
| 2. Operational ledger | ART-002 |
| 3. Pins | PIN-001–PIN-003 |
| 4. Build/API/allocation skeleton | BLD-001–BLD-003, API-001 |
| 5. Loader/tokenizer/scalar/traces/fixtures | MDL-001–MDL-002, TOK-001–TOK-002, CPU-001–CPU-004, TRC-001–TRC-002, ORA-001 |
| 6. CUDA primitives | CUD-001–CUD-002 |
| 7. GDN/attention/scheduler | GDN-001–GDN-002, ATN-001–ATN-002, SCH-001 |
| 8. Sessions and 128K | SES-001–SES-003, MEM-001 |
| 9. Profiling/optimization | OPT-001–OPT-004 |
| 10. Product tools/API/quality | CLI-001, SRV-001–SRV-003, BEN-001, EVAL-001, QLT-001 |
| 11. Comparative speed | CMP-001–CMP-003 |
| 12. Documentation/release | DOC-001, REL-001 |

## Chronological UTC Log

### 2026-08-29T00:00:00Z — ART-001 done

- Created `plan.md` first, before source, build, container, or additional
  documentation artifacts.
- Recorded the complete approved product boundary, architecture, correctness
  policy, optimization order, comparison protocol, gates, and assumptions.
- Evidence: [`plan.md`](plan.md).

### 2026-08-29T00:01:00Z — ART-002 done

- Created this ledger second and populated all implementation groups named by the
  baseline: artifact pins, builds/containers, loading, tokenizer/template,
  scalar oracle, tracing, CUDA primitives, GDN, attention, scheduling, memory,
  checkpointing, APIs, quality, optimization, comparisons, documentation, and
  release evidence.
- No implementation task is marked done without acceptance evidence.

### 2026-08-29T09:48:32Z — Repository and environment inventory

- Began PIN-001, PIN-002, PIN-003, ENV-001, and BLD-001 before implementation.
- Repository inventory found the existing untracked `docs/` handbook, README,
  license, and gitignore; no source code, tests, build files, or containers exist.
- Host inspection measured an NVIDIA GeForce RTX 5090 with 32,607 MiB and driver
  590.48.01. Docker 29.7.2, `g++`, and `uv` are present; host `nvcc` is absent.
  CUDA compilation must therefore use the pinned development container.
- Hugging Face API inspection selected `ggml-org/Qwen3.8-27B-GGUF` revision
  `0669b98607d47046c7c2b3f801011d54a08cfccf`; its Q4_K_M file is 18,973,870,432
  bytes with LFS SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
- Recorded current upstream refs for Transformers, llama.cpp, vLLM, Ollama, and
  DwarfStar for a checked-in pin manifest. These are not yet accepted until the
  manifest and license/provenance records are committed to the tree.
- Docker registry inspection resolved CUDA 13.0.2 Ubuntu 24.04 linux/amd64 to
  `sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c`.

### 2026-08-29T09:52:00Z — Host skeleton verification

- Added the immutable artifact/tool pin manifest, CUDA development Dockerfile,
  restricted C++17 Makefile, move-only public Engine/Session API, explicit
  Status type, and fail-closed product executable stubs. Began BLD-002 and
  BLD-003 before extending the CUDA and allocation build checks.
- Command: `make -j2` — passed with `-Werror -fno-exceptions -fno-rtti`.
- Command: `uv run pytest -q` — 3 passed.
- Commands: `uv run ruff check .` and `uv run ruff format --check .` — passed.
- The stubs intentionally return `unimplemented`; this is a positive
  fail-closed boundary test, not evidence for inference, server, benchmark, or
  evaluation completion.

### 2026-08-29T09:55:00Z — CUDA build and device probe

- Command: `docker build -f docker/cuda.Dockerfile -t qw38-cuda:13.0.2 .` —
  passed from the pinned linux/amd64 digest.
- Command: containerized `make cuda-native` — CUDA 13.0.2 `nvcc` compiled the
  diagnostic with `-arch=sm_120`.
- Command: containerized `./build/qw38-cuda-probe` — passed on NVIDIA GeForce
  RTX 5090, compute capability 12.0; measured 33,671,348,224 total bytes and
  33,139,458,048 free bytes at probe time.
- Marked PIN-002, PIN-003, ENV-001, BLD-001, and API-001 done with the evidence
  linked above. BLD-002 remains in progress until both diagnostic and engine
  CUDA objects build; this probe does not admit inference kernels.
- Added the exact official text contract and an explicitly estimated allocation
  ledger. BLD-003 remains in progress because runtime enforcement and post-graph
  measurement do not exist.

### 2026-08-29T10:00:59Z — Production artifact identity verified

- Downloaded the canonical file to the git-ignored `models/` runtime directory
  from the immutable Hugging Face revision.
- Command: `stat -c 'bytes=%s'` — 18,973,870,432 bytes, exact match.
- Command: `sha256sum` —
  `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`,
  exact match. Marked PIN-001 done.
- Direct little-endian GGUF inspection found version 3, 851 tensors, and 39
  metadata entries. Core metadata reports architecture `qwen35`, 64 blocks,
  context 262,144, width 5,120, 24 query heads, 4 KV heads, and 64 rotary
  dimensions.
- Began MDL-001 and MDL-002 before implementing the native inventory and
  fail-closed contract validator. Raw inspection is discovery evidence only;
  these tasks remain in progress.

### 2026-08-29T10:06:58Z — Native validation and clean verification

- Implemented bounded little-endian GGUF v3 parsing for metadata and tensor
  descriptors, alignment and byte-range validation, selected exact Qwen contract
  checks, and a standalone in-process SHA-256 implementation.
- `Engine::open` now fails closed on byte size, malformed GGUF, exposed contract
  mismatch, tensor-count mismatch, and whole-file SHA-256 mismatch. It returns
  success for the locally verified canonical artifact; session creation remains
  explicitly unimplemented.
- Real artifact command: `qw38-eval --inspect-gguf` — passed with version 3,
  39 metadata entries, 851 tensors, and data offset 10,994,016.
- Real artifact command: `qw38-eval --sha256` — exact digest match in 56.603 s.
  This is a **Measured** load-time cost and remains an optimization candidate,
  not a reason to weaken identity validation.
- Real artifact command: `qw38-eval --verify-model` — passed in 56.707 s.
- Final clean commands: `uv run ruff format .`, `uv run ruff check .`,
  `make clean`, `make -j2`, and `uv run pytest -q` — all passed; 6 pytest tests.
- Final pinned-container `make cuda-native` and CUDA probe — passed with CUDA
  13.0.2, `-arch=sm_120`, compute capability 12.0, 33,671,348,224 total bytes,
  and 33,139,458,048 free bytes.
- MDL-001 remains in progress: the parser does not mmap the file, assign verified
  semantic roles, hash each tensor separately, or prepare CUDA-resident weights.
  MDL-002 remains in progress until tensor roles/shapes prove the full GDN and
  attention schedule rather than relying only on official config plus exposed
  GGUF metadata.

### 2026-08-29T10:29:05Z — Foundation commit boundary

- Reviewed the complete worktree before the requested first commit. The staged
  boundary will contain the approved plan and ledger, immutable pins, restricted
  host/container build, public API skeleton, native GGUF/SHA validation,
  allocation documentation, and tests.
- The 18,973,870,432-byte runtime model remains under the git-ignored `models/`
  directory and will not be committed or pushed.
- Reverification and the resulting commit/push identifiers are recorded in the
  follow-up log entry below before any subsequent implementation commit.

### 2026-08-29T10:30:00Z — Foundation committed and pushed

- Reverification before commit: `uv run ruff format .`, `uv run ruff check .`,
  clean `make -j2`, `uv run pytest -q`, and staged diff checks all passed.
- Commit `5bc6610` (`feat: establish verified engine foundation`) created with
  the approved foundation boundary and pushed to `origin/main` successfully.
- Began the next MDL-001/MDL-002 increment: mmap ownership, exact quantized
  tensor byte sizing, per-tensor checksums, and semantic role/shape validation.

### 2026-08-29T10:35:51Z — MDL-001 and MDL-002 accepted

- Added move-only POSIX mmap ownership retained by a successfully opened Engine.
  Mapping is private/read-only and its descriptor/mapping lifetimes are coupled.
- Replaced inferred byte spans with block-exact F32, Q8_0, Q4_K, and Q6_K
  storage calculations; added checked offset arithmetic and payload-bound proofs.
- Encoded and passed the complete semantic tensor schedule: 48 GDN layers × 14
  tensors, 16 attention layers × 11 tensors, and three global tensors. Every
  expected name, role, shape, and GGML type is exact and extra/duplicate tensors
  fail admission.
- Generated [`pins/tensor_inventory.json`](pins/tensor_inventory.json) after
  full artifact validation. The 851 payload SHA-256 values cover 18,962,876,416
  bytes; all names are unique, all roles are assigned, and observed inter-tensor
  padding is zero. Generation took 1m56.997s.
- Added code-linked artifact admission documentation with **Measured** and
  implementation-boundary labels.
- Commands: Ruff format/check, clean restricted C++17 build, 8 pytest tests,
  production `--check-contract`, diff check, and production `--verify-model` —
  all passed. Final full open took 56.485s.
- Marked MDL-001 and MDL-002 done. CUDA-resident weight preparation remains
  future CUDA/build work and is not implied by these host artifact gates.

### 2026-08-29T10:36:00Z — Model admission commit boundary

- Reviewed the increment for commit: mmap ownership, exact storage/range checks,
  full semantic contract, payload hashes, malformed tests, and handbook evidence.
- The runtime GGUF remains ignored. Formatter, build, tests, production contract,
  and production full-open evidence above satisfy the commit boundary.

### 2026-08-29T10:37:00Z — Model admission pushed; TOK-001 started

- Commit `73c5745` (`feat: validate exact model tensor contract`) created after
  Ruff and 8 pytest tests passed, then pushed to `origin/main` successfully.
- Began TOK-001 by resolving tokenizer assets from the already pinned official
  source revision. Template behavior remains TOK-002 and will not be mixed into
  tokenizer admission.

### 2026-08-29T10:42:00Z — Tokenizer authority boundary established

- Resolved `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and
  `merges.txt` at official revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; recorded exact byte sizes and
  locally verified SHA-256 values in the artifact lock.
- Extended native GGUF admission to parse rather than skip tokenizer model,
  token, token-type, and merge metadata. The production artifact reports GPT-2,
  248,320 tokens/token types, and 247,587 merges; contract admission passes.
- Pinned `tokenizers==0.22.1` as the fixture authority tool and added a typed,
  identity-checking fixture generator. Generated 12 cases retaining prompt
  bytes, IDs, token strings, and decoded bytes across Unicode and special-token
  boundaries.
- Commands: fixture generator, Ruff format/check, and pytest — passed; 9 tests.
- TOK-001 remains in progress. No C++ tokenization claim is made until the native
  NFC, Unicode splitting, byte mapping, BPE, and special-token path matches every
  frozen fixture.

### 2026-08-29T10:43:00Z — Tokenizer oracle commit boundary

- Reviewed pins, GGUF metadata admission, generated fixtures, generator, tests,
  dependency lock, and code-linked handbook note before commit.
- Runtime tokenizer assets remain ignored and will not be committed.

### 2026-08-29T10:44:00Z — Tokenizer oracle pushed; Unicode dependency decision

- Commit `0312cba` (`test: pin tokenizer authority fixtures`) created after a
  clean restricted build, Ruff, 9 pytest tests, and production contract check,
  then pushed to `origin/main` successfully.
- TOK-001 requires full NFC normalization and Unicode general categories for the
  pinned pre-tokenization expression. An ASCII approximation is rejected because
  it would invalidate the frozen NFD, multilingual, combining-mark, and emoji
  cases.
- Selected a pinned utf8proc submodule for NFC/category primitives under its MIT
  license. Quartz will retain local control of Qwen-specific splitting, GPT-2
  byte mapping, special-token handling, and BPE; no generic tokenizer backend is
  introduced. Exact revision and license evidence must be recorded before use.

### 2026-08-29T10:45:00Z — Unicode primitive pinned

- Added utf8proc `v2.11.0` at commit
  `d7bf128df773c2a1a7242eb80e51e91a769fc985` as a git submodule.
- Inspected and retained its MIT license and Unicode data notice. Recorded the
  exact revision, role, and narrow usage boundary in the artifact/source ledgers.
- This dependency pin does not complete TOK-001; native splitting, byte mapping,
  BPE, special tokens, and fixture equality remain to implement.

### 2026-08-29T10:53:18Z — Beginner documentation gap recorded

- User review identified that the tokenizer authority note names NFC
  normalization, Unicode splitting, GPT-2 byte mapping, BPE, and fixture equality
  without explaining them to a reader with no tokenizer background.
- Added EDU-001 before corrective work. It affects TOK-001 and DOC-001: native
  tokenizer code cannot be admitted until the handbook explains each stage with
  concrete bytes/tokens, invariants, failure modes, and evidence links.

### 2026-08-29T10:57:00Z — First native tokenizer comparison (negative result)

- Implemented the initial native NFC, Unicode-category split, GPT-2 byte map,
  special-token matcher, and ranked BPE path using the GGUF vocabulary/merges.
- First comparison matched 10 of 12 authority cases. `spaces` and `code` failed:
  native IDs were `[256,20028,262,19055,256,371,13868,256]` instead of
  `[220,6187,256,5956,220,26849,256]`, and the indented code boundary emitted
  `[257,671]` instead of `[262,460]`.
- Diagnosis: the `\s+(?!\S)` regex alternative backtracks before a following
  non-space, leaving the final ASCII space for the next word's optional prefix.
  The first implementation consumed the whole whitespace run. Added the exact
  backtracking rule; follow-up remains TOK-001 fixture equality.

### 2026-08-29T11:02:00Z — TOK-001 and EDU-001 accepted

- Corrected whitespace lookahead behavior. All 12 frozen authority cases now
  match exact native token IDs, including NFC/NFD, multilingual text, emoji,
  whitespace/indentation, CR/LF, contractions, and special markers.
- Ran 100 additional seeded randomized differential cases over ASCII, combining
  marks, CJK, Arabic, Indic, emoji/ZWJ, punctuation, and whitespace: zero ID
  differences against the pinned authority.
- Added invalid UTF-8 rejection and an integration pytest that exercises every
  authority fixture when the ignored production GGUF is installed.
- Integrated tokenizer construction into successful `Engine::open`; an Engine
  cannot exist with invalid vocabulary, merges, byte map, or special-token data.
- Expanded the tokenizer handbook from an evidence note into a beginner chapter:
  bytes/code points/tokens, NFC, ordered Unicode splitting, GPT-2 byte symbols,
  ranked BPE, special tokens, a real byte-to-ID example, fixture equality, and
  failure modes are explained and linked to code/evidence.
- Clean host build, Ruff, 12 pytest tests, pinned CUDA 13.0.2 container build,
  SM120 device probe, and full production `Engine::open` all passed. Full open
  with tokenizer construction took 58.077s.
- Marked TOK-001 and EDU-001 done. TOK-002 chat rendering remains separate.

### 2026-08-29T11:03:00Z — Native tokenizer commit boundary

- Reviewed native/tokenizer dependency code, oracle fixtures, randomized result,
  tests, beginner documentation, negative result, and final environment evidence
  before the requested commit and push.

### 2026-08-29T11:04:00Z — Native tokenizer pushed; TOK-002 started

- Commit `93efe69` (`feat: implement exact native tokenizer`) created after Ruff
  and 12 pytest tests passed, then pushed to `origin/main` successfully.
- Began TOK-002 by freezing the official text-template behavior before writing a
  native renderer. Roles, reasoning modes, tools, continuations, and rejection
  behavior stay within the v1 text boundary.

### 2026-08-29T11:06:00Z — Template generator strict-undefined negative result

- First authority render used Jinja `StrictUndefined` and failed on an ordinary
  user message because the official template probes optional `tool_calls` fields.
- Diagnosis: Transformers-compatible Jinja treats missing optional mapping fields
  as false/undefined rather than raising on access. Switched the fixture renderer
  to standard Jinja undefined behavior; explicit template `raise_exception`
  calls remain authoritative rejection evidence.

### 2026-08-29T11:07:00Z — Vision rejection ownership clarified

- The official template accepted a user image item and rendered vision markers;
  the initial fixture incorrectly expected an upstream error.
- Diagnosis: Qwen is multimodal, while Quartz v1 explicitly excludes vision.
  Moved this case to a separately labeled Quartz v1 policy-rejection set. Native
  rendering must reject it before emitting any vision marker; documentation must
  not attribute the rejection to the upstream template.

### 2026-08-29T11:10:00Z — Template diagnostic compile error

- First strict build of the native template diagnostic failed under `-Werror`
  because aggregate `Message` fixtures omitted later optional fields and triggered
  `-Wmissing-field-initializers`.
- Added a two-field Message constructor that deliberately default-initializes
  reasoning, tool calls, and policy flags. No warning suppression or relaxed
  compiler flag was introduced.

### 2026-08-29T11:39:57Z — TOK-002 and EDU-002 accepted

- Pinned the official Jinja chat-template text hash and Jinja 3.1.6 fixture tool.
  Frozen five official success paths, three upstream errors, one developer-role
  policy mapping, and one v1 vision policy rejection.
- Implemented a typed native renderer for leading system/developer instructions,
  user/assistant/tool roles, low/medium/xhigh reasoning, historical thinking,
  generation prompts, canonical tool definitions, function arguments/results,
  consecutive results, and explicit unsupported-content rejection.
- Native rendered UTF-8 bytes match every official/policy success fixture exactly;
  all rendered token ID sequences also match the pinned tokenizer authority.
- Added a beginner chat-template chapter explaining roles, delimiters, reasoning,
  tool execution/results, developer mapping, official-versus-Quartz ownership,
  a real structured-message-to-prompt example, byte equality, and failure modes.
- Commands: generator, Ruff, clean restricted host build, 16 pytest tests,
  pinned CUDA 13.0.2 full build, SM120 compilation, and RTX 5090 probe — passed.
- Marked TOK-002 and EDU-002 done. HTTP/API behavior remains SRV work.

### 2026-08-29T11:40:00Z — Chat-template commit boundary

- Reviewed source, authority/policy fixtures, generator, dependency lock, tests,
  beginner documentation, negative results, and container evidence before commit.

### 2026-08-29T11:42:00Z — Chat template pushed; CPU-001 started

- Commit `dc96299` (`feat: implement exact chat template`) created after Ruff and
  16 pytest tests passed, then pushed to `origin/main` successfully.
- Began CPU-001. Exact Q4_K/Q6_K packed layouts and fixture values must be pinned
  to the admitted GGML definitions before implementing scalar decoding or dot
  products; CUDA work remains downstream.

### 2026-08-29T11:55:00Z — CPU-001 cosine-metric test negative result

- Pinned the exact llama.cpp Q4_K/Q6_K block definitions and generated four
  deterministic authority fixtures before implementing the scalar decoder.
- The native decoder and dot product matched every frozen FP32 bit, but the
  first 19-test run had one failure in the Python metric reporter: it required
  computed cosine similarity to equal exactly `1.0`. For a bit-identical vector,
  `sum(x*x) / (sqrt(sum(x*x)) * sqrt(sum(x*x)))` rounded just below one because
  the square-root operations introduce floating-point rounding.
- Diagnosis: this was an invalid exact assertion on a derived metric, not a
  decoded-value or dot-product mismatch. The exact byte gates remain unchanged;
  follow-up CPU-001 work uses a declared `1e-15` absolute tolerance only for the
  reported cosine calculation.

### 2026-08-29T11:57:00Z — EDU-003 started

- Added EDU-003 before writing the quantization chapter. The user-facing
  documentation boundary now explicitly requires a no-prerequisites account of
  binary storage, floating-point scales, Q4_K/Q6_K block packing, scalar decode,
  dot products, FP32 accumulation, fixture provenance, equality, error metrics,
  and the cosine-rounding negative result. This elaborates CPU-001 without
  expanding the v1 product boundary.

### 2026-08-29T12:05:00Z — CPU-001 and EDU-003 accepted

- Pinned the exact 256-value Q4_K (144-byte) and Q6_K (210-byte) structures,
  field offsets, equations, upstream files, MIT license, and llama.cpp revision.
- Implemented bounded scalar decoders, explicit little-endian FP16 conversion,
  portable signed Q6 scale conversion, and separate FP32 block dot products.
  The scalar build disables multiply-add contraction to freeze its arithmetic
  order; CUDA arithmetic remains a later differential gate.
- Generated four deterministic fixtures covering Q4 packed 6-bit scale/minimum
  boundaries, Q4 zero, signed Q6 extremes, and Q6 zero scale. All 1,024 decoded
  FP32 values and four dot products match their frozen bytes exactly.
- Independently compiled [`tools/llama_quant_oracle.c`](tools/llama_quant_oracle.c)
  with the pinned upstream `ggml-quants.c`; upstream output matched all 1,024
  frozen decoded FP32 values exactly. This check used a fresh checkout at the
  pinned revision and did not link Quartz's decoder.
- Added malformed kind/hex/byte-size checks and explicit zero absolute/RMS,
  no-NaN/Inf, and cosine metric assertions. The previously recorded cosine
  harness error is resolved without weakening decoded-value or dot equality.
- Added the beginner chapter explaining binary storage, FP16/FP32, lossy
  quantization, Q4_K/Q6_K layouts and equations, dot products, rounding order,
  fixtures, exact equality, metrics, failure modes, and the proof boundary.
- Commands: fixture regeneration, Ruff format/check, clean restricted C++17
  build, 19 pytest tests, `git diff --check`, pinned-container full host build,
  SM120 compilation, and RTX 5090 probe — passed. The device probe measured
  33,671,348,224 total and 33,139,458,048 free bytes at this run.
- Marked CPU-001 and EDU-003 done. Scalar model layers begin at CPU-002; reusable
  trace metrics remain TRC-001 and CUDA quant kernels remain CUD-001.

### 2026-08-29T12:06:00Z — Scalar quantization commit boundary

- Reviewed the pin, fixture generator/output, upstream adapter, scalar source,
  strict build flag, diagnostics, tests, beginner documentation, provenance,
  and preserved negative result before the requested commit.

### 2026-08-29T12:08:00Z — Scalar quantization pushed; CPU-002 started

- Commit `290ff0f` (`feat: add scalar quantization oracle`) created after Ruff,
  19 pytest tests, restricted host build, upstream differential check, pinned
  CUDA container build, SM120 compilation, and RTX 5090 probe passed; pushed to
  `origin/main` successfully.
- Began CPU-002 and added EDU-004 before implementation. The official pinned
  Qwen3.5 authority must first fix the GDN projection packing, head mapping,
  causal convolution warm-up, gate equations, recurrence mutation order, and
  persistent-state shapes. The native oracle and beginner explanation will be
  derived from that frozen contract, not from a generic recurrent abstraction.

### 2026-08-29T12:18:00Z — CPU-002 authority-environment boundary

- Inspected and hashed the exact pinned Transformers Qwen3.5 implementation.
  The local Python environment does not contain PyTorch, so it cannot directly
  execute the decorated Transformers fallback as part of this focused host gate.
- The CPU-002 fixture generator is therefore labeled precisely as an explicit
  FP32 scalar transcription of the pinned `causal_conv1d_update`, `l2norm`, gate,
  and `torch_recurrent_gated_delta_rule` equations. It is not labeled as a
  Transformers eager trace. Direct eager/offloaded model traces remain required
  by ORA-001 and cannot be inferred from CPU-002 success.

### 2026-08-29T12:26:00Z — CPU-002 and EDU-004 accepted

- Pinned the official implementation revision, file SHA-256, used symbols,
  production shapes, normalization/gate equations, mutation order, convolution
  semantics, and FP32 state dtype in the GDN contract.
- Implemented a shape-bounded scalar GDN core: stable sigmoid/softplus gates,
  per-head FP32 L2 normalization, query scaling, exact three-to-one head reuse,
  decay-before-delta mutation, updated-state readout, four-wide zero-warmed
  depthwise causal convolution, and SiLU.
- Frozen `2e-6` maximum absolute, `2e-5` maximum relative, `1e-6` RMS, and
  `0.999999` minimum cosine tolerances before CUDA work. Measured maxima were
  `1.1920928955078125e-7` absolute, `2.703506447862878e-6` relative, and
  `3.825640424749008e-8` RMS; minimum measured cosine exceeded
  `0.999999999999994`.
- Whole, `[2,1,2]`, and token-wise recurrent schedules produced byte-identical
  gates, outputs, and final matrices. Whole, `[1,2,3]`, and token-wise
  convolution schedules likewise produced byte-identical outputs and histories.
  Invalid head ratios, buffer counts, components, and chunk names fail closed.
- Added a beginner chapter covering fixed-state memory, projections, heads and
  the three-to-one map, matrix sizes, normalization, gates, a worked delta-rule
  update, convolution warm-up/rings, mutation ownership, atomicity boundaries,
  chunks, fixtures, tolerances, authority labels, and remaining proof gaps.
- Commands: fixture regeneration, Ruff format/check, clean restricted C++17
  build, 24 pytest tests, `git diff --check`, pinned CUDA 13.0.2 full build,
  SM120 compilation, and RTX 5090 probe — passed. Probe values were
  33,671,348,224 total and 33,139,458,048 free bytes.
- Marked CPU-002 and EDU-004 done. Direct Transformers eager trace evidence is
  still ORA-001; full real-weight GDN projections/scheduling remain CPU-004.

### 2026-08-29T12:27:00Z — Scalar GDN commit boundary

- Reviewed the official pin/hash, scalar transcription and tolerances, native
  recurrence/convolution code, diagnostics, shape failures, tests, educational
  chapter, provenance, measured metrics, and explicit eager-trace limitation
  before committing.

### 2026-08-29T12:29:00Z — Scalar GDN pushed; CPU-003 started

- Commit `f0c5087` (`feat: add scalar GDN oracle`) created after the documented
  24-test and container gates, then pushed to `origin/main` successfully.
- Began CPU-003 and added EDU-005 before implementation. The next frozen contract
  must cover the official 24-query/4-KV grouped mapping, 256-wide heads, 64
  rotary dimensions, causal KV append/read order, stable softmax, and SwiGLU FFN
  order, with focused fixtures representing required attention layers 3/7/63.

### 2026-08-29T12:43:00Z — CPU-003 relative-metric failure

- The first 28-test run failed one CPU-003 attention metric. Native outputs met
  the frozen `3e-6` absolute and `1e-6` RMS limits, but a near-zero output had
  `1.151611987684236e-4` relative error, above the frozen `3e-5` limit. The
  difference is at the one-ULP/library-math scale, but the relative gate remains
  binding.
- No tolerance was loosened. CPU-003 remains in progress while the exact failing
  lane and arithmetic source are diagnosed; the failed run is retained here as
  required admission evidence.

### 2026-08-29T12:46:00Z — CPU-003 metric failure resolved

- Located the worst lane at attention layer fixture 63, flattened output index
  108: native `-0.00044878353946842253` versus fixture
  `-0.0004488352278713137`, an absolute difference of
  `5.168840289115906e-8` but relative error `1.151611987684236e-4`.
- Diagnosis: the fixture claimed explicit FP32 scalar transcendental operations
  but used Python's double-precision `math` functions followed by FP32 rounding.
  The native oracle and the pinned fallback operate on FP32 tensors. Corrected
  the generator to call `expf`, `sqrtf`, `powf`, `sinf`, and `cosf` directly and
  then regenerated expected data. No native code or frozen tolerance changed.
- The focused four-test CPU-003 suite then passed. The worst relative error is
  now `7.796529634717427e-6`, below the unchanged `3e-5` gate.

### 2026-08-29T12:54:00Z — CPU-003 and EDU-005 accepted

- Pinned the official source identity/symbols, 24-query/4-KV production shape,
  six-to-one grouped mapping, 256 head width, first-64-lane partial RoPE,
  10,000,000 theta, stable causal attention order, RMSNorm equation, query
  output gate, and 5,120→17,408→5,120 SwiGLU equation.
- Implemented per-head RMSNorm, partial half-rotation RoPE, causal KV append and
  bounded-frontier reads, grouped-query scoring, FP32 stable softmax, sigmoid
  output gating, and visible dense FFN gate/up/activated/down taps.
- Frozen `3e-6` maximum absolute, `3e-5` maximum relative, `1e-6` RMS, and
  `0.999999` minimum cosine gates. Across layers 3/7/63, worst measured absolute
  error was `2.384185791015625e-7`, relative error
  `7.796529634717427e-6`, and aggregate fixture RMS below `2.7e-8`.
- Future KV rows were initialized with ±1,000-scale sentinels; first-position
  outputs remained below magnitude two and matched the causal fixture. Rotated
  key caches, unrotated value caches, and all FFN taps passed. The earlier failed
  relative gate and FP64-generator diagnosis remain preserved above.
- Added a beginner chapter covering lookup intuition, heads and six-to-one GQA,
  query gating, RMSNorm, worked partial-RoPE pairing, KV causality, the 64 KiB per
  token/8 GiB estimate, stable softmax, SwiGLU projection order, fixtures,
  metrics, the failed gate, authority labels, and proof limits.
- Commands: fixture regeneration, Ruff format/check, clean restricted C++17
  build, 28 pytest tests, `git diff --check`, pinned CUDA 13.0.2 full build,
  SM120 compilation, and RTX 5090 probe — passed. Probe values were
  33,671,348,224 total and 33,139,458,048 free bytes.
- Marked CPU-003 and EDU-005 done. Direct eager traces remain ORA-001; real
  tensor projections, residual scheduling, and logits remain CPU-004.

### 2026-08-29T12:55:00Z — Scalar attention/FFN commit boundary

- Reviewed the official contract, scalar attention/RoPE/RMSNorm/FFN code,
  synthetic layer fixtures, frozen metrics, causal sentinels, tests, beginner
  chapter, provenance, failed run and resolution, and eager-trace limitation
  before committing.

### 2026-08-29T12:57:00Z — Scalar attention pushed; CPU-005 discovered

- Commit `c594040` (`feat: add scalar attention oracle`) created after the
  documented 28-test and container gates, then pushed to `origin/main`.
- Added CPU-005 before implementation and made it an explicit CPU-004 dependency.
  The exact admitted inventory contains Q8_0 attention/GDN projections, but the
  approved CPU-001 boundary named only Q4_K/Q6_K. A full real-weight scalar
  scheduler cannot interpret those rows without a Q8_0 primitive.
- Added EDU-006 so this discovered format and its role in matrix rows are
  explained rather than appearing as an unexplained implementation detour. This
  fills a required artifact format; it does not expand the v1 product boundary.

### 2026-08-29T13:07:00Z — CPU-005 and EDU-006 accepted

- Extended the pinned quant contract with upstream Q8_0's exact 34-byte block:
  one little-endian FP16 scale and 32 signed bytes representing 32 weights.
- Implemented portable signed-byte decoding and a 32-value FP32 block dot product.
  Added signed-extreme and zero fixtures; decoded values and dot products match
  every frozen FP32 bit. Existing Q4_K/Q6_K gates remain unchanged.
- Recompiled the focused llama adapter with pinned upstream `ggml-quants.c`.
  All six Q4_K/Q6_K/Q8_0 cases matched the upstream decoded bytes exactly.
- Added a beginner chapter explaining why a Q4_K_M artifact mixes formats,
  signed two's-complement bytes, Q8_0's equation and 8.5-bit effective size, a
  worked value, how 32-value blocks form a matrix row, the distinction from F32,
  fixtures, provenance, and the CPU-004 dependency/proof boundary.
- Commands: fixture regeneration, Ruff format/check, clean restricted C++17
  build, 28 pytest tests, upstream differential compile/run, `git diff --check`,
  pinned CUDA 13.0.2 full build, SM120 compilation, and RTX 5090 probe — passed.
  Probe values were 33,671,348,224 total and 33,139,458,048 free bytes.
- Marked CPU-005 and EDU-006 done. Arbitrary real tensor row binding, orientation,
  complete projections, the 64-layer schedule, and logits remain CPU-004.

### 2026-08-29T13:08:00Z — Q8_0 scalar commit boundary

- Reviewed the discovered-task rationale/dependency, upstream layout, portable
  decoder/dot code, expanded fixture generator, upstream adapter, tests,
  beginner documentation, provenance, and downstream boundary before committing.

### 2026-08-29T13:10:00Z — Q8_0 pushed; CPU-006 started

- Commit `9fd22ee` (`feat: add scalar Q8_0 oracle`) was pushed and independently
  confirmed current on `origin/main` by the user's subsequent `git push`.
- Added CPU-006 before implementation and made it an explicit CPU-004 dependency.
  Block decoders alone cannot execute a projection: the runtime must bind the
  exact mapped tensor payload, interpret GGUF's fastest-changing first dimension
  as row width, prove block-aligned row byte spans, and accumulate every output
  row without transposing the matrix.
- Added EDU-007 so shape notation, physical row order, mixed formats, bounds, and
  matrix-vector multiplication are explained before the full scheduler uses them.
  This is required scalar plumbing, not a release-boundary expansion.

### 2026-08-29T13:18:00Z — CPU-006 pytest collection failure

- The first focused `tests/test_tensor.py` run stopped during collection with
  `ModuleNotFoundError: No module named 'tools'`; no native diagnostic executed.
- Diagnosis: the pytest launcher did not place the repository root on Python's
  module search path, while the test reuses deterministic decoders from the
  checked-in quant fixture generator. The follow-up adds the explicit repository
  root before importing that helper; tensor code and expected arithmetic remain
  unchanged.

### 2026-08-29T13:20:00Z — CPU-006 admitted-row oracle failure

- After collection was fixed, three synthetic tests passed and the admitted
  token-embedding row failed: native dot bytes `37778a3e` versus Python oracle
  `dab46e3e`.
- Diagnosis: the reused fixture decoder intentionally accepts one quant block,
  but the admitted-row helper passed an entire 5,120-value row. It decoded only
  the first 256 weights, and Python `zip` then silently truncated the activation.
  The native implementation had processed all 20 Q4_K blocks as designed.
- The follow-up changes only the independent admitted-row helper to split the
  payload by the pinned block byte size and concatenate every decoded block.
  The failed values remain recorded; CPU-006 is not admitted yet.

### 2026-08-29T13:28:00Z — CPU-006 and EDU-007 accepted

- Pinned GGML's dimension/stride source identity and froze the interpretation:
  dimension 0 is contiguous input columns, dimension 1 is output rows, row bytes
  derive from complete format blocks, and matvec emits one dot per stored row.
- Implemented non-owning checked TensorView creation and model-name binding,
  little-endian F32 row reads, Q8_0/Q4_K/Q6_K row decoding, block-ordered FP32
  row dots, and mixed-format matvec. Offset, multiplication, mapping, rank, row,
  activation/output count, block alignment, and complete-storage checks return
  explicit Status failures.
- Synthetic evidence covers an asymmetric 3×5 F32 matrix and two-block,
  asymmetric two-row matrices for every quantized format. Partial Q4 storage, a
  31-column Q8 row, and a wrong activation count fail closed.
- Froze four admitted model rows with payload SHA-256 and exact dot bytes:
  `token_embd.weight` row 42 (Q4_K), `output.weight` row 17 (Q6_K),
  `blk.3.attn_q.weight` row 9 (Q8_0), and
  `blk.0.ssm_conv1d.weight` row 5 (F32). Independent mapped-byte decoding and
  native named binding agree exactly. Unknown names, rank-1 tensors, and a row at
  the exclusive upper bound fail.
- Added a beginner chapter explaining vectors/matrices, why GGUF shapes appear
  reversed relative to Python, worked row-byte arithmetic, TensorView ownership
  and bounds, decode versus dot, mixed formats, synthetic/admitted fixtures, both
  failed oracle approaches, and the CPU-004 proof boundary.
- Commands: fixture generation, Ruff format/check, clean restricted C++17 build,
  33 pytest tests, `git diff --check`, pinned CUDA 13.0.2 full build, SM120
  compilation, and RTX 5090 probe — passed. Probe values were 33,671,348,224
  total and 33,139,458,048 free bytes.
- Marked CPU-006 and EDU-007 done. Typed binding of all layer roles, complete
  real projections, session state, the 64-layer schedule, final norm, and logits
  remain CPU-004.

### 2026-08-29T13:29:00Z — Tensor-row commit boundary

- Reviewed the discovered dependency, pinned physical layout, checked view and
  arithmetic code, synthetic/admitted fixtures, two preserved test failures,
  malformed cases, beginner documentation, provenance, and clean verification
  before committing.

### 2026-08-29T13:31:00Z — Tensor rows pushed; CPU-007 discovered

- Commit `f92b4fd` (`feat: bind scalar tensor rows`) created after the documented
  33-test and container gates, then pushed to `origin/main` successfully.
- Began the next CPU-004 binding increment by auditing the pinned GGUF converter
  and llama.cpp Qwen3.5 graph. This discovered semantic conversions that cannot
  be represented as mere byte/shape binding, so CPU-007 and EDU-008 were added
  before implementation and CPU-004 now depends on them.
- The converter folds `A_log` to `-exp(A_log)`, squeezes depthwise convolution,
  adds one to ordinary Qwen RMSNorm weights, and reorders GDN value-associated
  rows/columns from Hugging Face grouped order to GGML tiled order. The native
  semantic core currently uses grouped head order; explicit reversible transforms
  are required rather than silently applying the wrong value-to-key mapping.

### 2026-08-29T17:29:00Z — CPU-007 and EDU-008 accepted

- Pinned the exact llama.cpp converter and Qwen3.5 graph file hashes and froze
  the converted meanings for folded decay, direct norm scales, squeezed
  convolution, and every value-associated grouped/tiled GDN tensor role.
- Added checked grouped-to-tiled and tiled-to-grouped FP32 transforms. A visible
  2-key, 3-replica, 2-lane fixture proves the exact index mapping and reproduces
  the grouped input bit-for-bit after a round trip.
- Split source `-exp(A_log)` gate construction from the GGUF folded-A route and
  split source-offset RMSNorm from GGUF direct-scale RMSNorm. Both pairs emit
  identical FP32 bytes in focused diagnostics. Nonfinite, zero, and positive
  folded decay parameters fail closed before outputs are changed.
- Froze exact payload hashes, endpoint bytes, ranges, shapes, and storage sizes
  for layer-0 decay, time bias, ordinary norm, GDN norm, and convolution. The
  admitted decay payload is entirely negative and the admitted convolution is
  the squeezed GGUF `[4, 10240]` shape.
- Added a beginner chapter explaining source checkpoints versus runtime GGUF,
  exponent folding, softplus/decay, norm offsets versus scales, squeezing,
  grouped/tiled indices with a worked example, every affected tensor role,
  fixture equality, engine/session ownership, and the remaining CPU-004 proof
  boundary. Updated the handbook index, root entry point, and provenance ledger.
- Commands: fixture regeneration, Ruff format/check, clean restricted C++17
  build, 38 pytest tests, `git diff --check`, pinned CUDA 13.0.2 full build,
  SM120 compilation, and RTX 5090 probe — passed. Probe values were
  33,671,348,224 total and 33,139,458,048 free bytes.
- Marked CPU-007 and EDU-008 done. Complete typed binding of all tensor roles,
  real projections, the 64-layer schedule, final norm, and logits remain CPU-004.

### 2026-08-29T17:30:00Z — GGUF conversion commit boundary

- Reviewed the source pins, conversion contract, explicit semantic APIs,
  reversible layout code, synthetic and admitted fixtures, malformed input,
  beginner documentation, evidence labels, and clean host/container verification
  before committing.

### 2026-08-29T17:34:00Z — Conversion pushed; CPU-008 started

- Commit `7f296f5` (`feat: define GGUF conversion boundary`) was pushed to
  `origin/main` successfully after the documented host, fixture, container, and
  device gates.
- Added CPU-008 before implementation and made it an explicit CPU-004
  dependency. The inventory proves 851 byte ranges, but a scheduler also needs
  compile-time named fields for 3 globals, 48 fourteen-tensor GDN layers, and 16
  eleven-tensor attention layers; unchecked string lookup inside execution would
  allow missing or variant-incompatible roles to surface too late.
- Added EDU-009 so non-owning vector/matrix views, common versus variant-specific
  weights, exact schema checks, and the remaining execution boundary are taught
  alongside the binding code.

### 2026-08-29T17:35:00Z — CPU-008 endpoint fixture mismatch

- The first focused typed-binding run bound all 851 tensors and passed four
  malformed-schema cases, but the final-norm endpoint assertion used guessed
  bytes `0000903f00008d3f`; the mapped vector decoder returned
  `0000fb3f0000f13f`.
- Diagnosis: the expected endpoint was entered before reading the admitted
  payload and had no authority. The follow-up freezes the observed bytes only
  after checking them independently against the already hashed
  `output_norm.weight` range in the tensor inventory. Native binding and decoding
  code are unchanged; this negative result remains part of the evidence trail.

### 2026-08-29T17:36:00Z — CPU-008 and EDU-009 accepted

- Added checked non-owning F32 VectorView creation, mapped name binding, and
  explicit little-endian decode alongside the already admitted matrix view.
- Added narrow typed structures for the three globals, five common layer fields,
  nine GDN fields, and six attention fields. Binding uses the official
  three-GDN/one-attention schedule, validates exact name, semantic role, rank,
  dimensions, dtype, complete storage, and mapped range, and publishes only a
  fully populated candidate.
- The pinned artifact binds exactly 851 tensors: 3 globals, 48 × 14 GDN tensors,
  and 16 × 11 attention tensors. Its embedding and output shapes, final-norm
  width, and independently verified final-norm endpoint bytes match the focused
  diagnostic.
- In-memory corruptions of one required name, semantic role, vector shape, and
  mapped offset all return errors. The initially guessed endpoint fixture and
  its independent hash-based resolution remain in the preceding log entry.
- Added a beginner chapter explaining inventory versus typed fields, views and
  ownership, vector/matrix distinction, common and variant layers, schedule and
  851-count arithmetic, exact schema admission, transactional publication,
  fail-closed behavior, the failed fixture, and the CPU-004 execution boundary.
- Commands: independent final-norm payload/hash read, Ruff format/check, clean
  restricted C++17 build, 43 pytest tests, `git diff --check`, pinned CUDA 13.0.2
  full build, SM120 compilation, and RTX 5090 probe — passed. Probe values were
  33,671,348,224 total and 33,139,458,048 free bytes.
- Marked CPU-008 and EDU-009 done. Activation workspaces, real projection
  execution, session state, the 64-layer schedule, final norm, and logits remain
  CPU-004.

### 2026-08-29T17:36:30Z — Typed-weight commit boundary

- Reviewed the discovered dependency, vector and matrix lifetime/range checks,
  exact global/common/variant schema, all 851 bindings, failure publication
  behavior, corrupted metadata cases, preserved fixture error, beginner
  documentation, evidence labels, and clean host/container verification before
  committing.

### 2026-08-29T17:39:00Z — Typed weights pushed; CPU-009 started

- Commit `30204e1` (`feat: bind typed model weights`) was pushed to `origin/main`
  successfully after all recorded acceptance gates; the worktree was clean.
- Added CPU-009 before implementation and made it a CPU-004 dependency. The GDN
  packed projection uses three global contiguous channel ranges, while the
  attention query/gate projection stores two contiguous 256-lane halves inside
  each of 24 heads. Treating both 12,288/10,240-value outputs with one generic
  “split in half” rule would silently mix heads or roles.
- Added EDU-010 so packing, slicing, per-head layout, alias restrictions, and the
  subsequent tiled-to-grouped GDN conversion are taught at the code boundary.

### 2026-08-29T17:40:00Z — CPU-009 and EDU-010 accepted

- Pinned the exact Transformers semantic source and llama.cpp converter hashes
  with production GDN ranges and attention `[24, 2, 256]` layout.
- Implemented checked GDN splitting into global contiguous Q/K/V ranges and
  attention splitting into query/gate halves inside each query head. Count
  arithmetic detects overflow, and packed/output or output/output byte-range
  aliases fail before copying.
- Frozen asymmetric examples prove GDN `Q|K|V` segmentation and per-head
  attention deinterleaving. A short packed input and an aliased output both
  return explicit errors.
- Added a beginner chapter explaining projections, packing, slice notation,
  both layouts with worked values, convolution-before-GDN-split ordering,
  tiled-to-grouped V conversion after the split, gate activation ownership,
  count validation, aliases, and the remaining real-projection proof boundary.
- Commands: Ruff format/check, clean restricted C++17 build, 46 pytest tests,
  `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation, and RTX
  5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-009 and EDU-010 done. Real typed projections, activation/state
  workspaces, complete layer execution, residuals, final norm, and logits remain
  CPU-004.

### 2026-08-29T17:40:30Z — Packed-layout commit boundary

- Reviewed source identities, production and synthetic layouts, checked count
  and address arithmetic, alias behavior, downstream conversion ownership,
  diagnostic fixtures, beginner documentation, and clean host/container
  verification before committing.

### 2026-08-29T17:45:00Z — Packed layouts pushed; CPU-010 started

- Commit `b267891` (`feat: split packed projections`) was pushed to `origin/main`
  successfully and the worktree was clean.
- Added CPU-010 before implementation and made it a CPU-004 dependency. Earlier
  gates proved individual mapped rows and synthetic split layouts, but had not
  driven a complete real mixer projection through the typed layer structures or
  fixed the exact activation and output workspace contract.
- Added EDU-011 so the difference between one projection row, a complete matvec,
  packed output storage, split storage, evidence taps, and scalar cost is
  explained alongside the executable boundary.

### 2026-08-29T18:20:00Z — CPU-010 and EDU-011 accepted

- Added exact pointer/count workspace contracts and real typed matvec composition
  for layer-0 GDN packed QKV, value gate, alpha and beta, plus layer-3 attention
  packed query/gate, split query/gate, key, and value projections.
- The deterministic 5,120-value activation drives 30,816 complete real matrix
  rows, representing 157,777,920 scalar weight/activation products. All 43,104
  output and split-workspace values replace NaN sentinels with finite results.
- Independently decoded selected rows cover every GDN Q/K/V packed boundary,
  gate/control endpoints, attention per-head query/gate boundaries, the last
  gate lane, and K/V endpoints. Each physical row hash and expected FP32 dot is
  frozen; native complete matvec taps match exactly.
- A 10,239-value GDN packed workspace returns an error before the first matvec.
  The focused complete diagnostic measured 0.29 seconds and 189,760 KiB maximum
  resident host memory; this is explicitly not an end-to-end or CUDA speed claim.
- Added a beginner chapter explaining activations, rows and matvec, typed
  execution, exact workspace tables, ownership and reuse, why GDN stays packed
  until convolution, real attention splitting, selected-tap evidence strategy,
  scalar cost, measurement scope, and the remaining full-layer boundary.
- Commands: fixture regeneration, timed focused diagnostic, Ruff format/check,
  clean restricted C++17 build, 49 pytest tests, `git diff --check`, pinned CUDA
  13.0.2 full build, SM120 compilation, and RTX 5090 probe — passed. Probe values
  were 33,671,348,224 total and 33,139,458,048 free bytes.
- Marked CPU-010 and EDU-011 done. Input normalization, GDN convolution and
  recurrence, attention KV execution, mixer output projection, residuals, FFN,
  final norm, and logits remain CPU-004.

### 2026-08-29T18:20:30Z — Real-mixer commit boundary

- Reviewed typed weight use, exact workspace guards, complete projection loops,
  packed/split ordering, independent row generator and hashes, real-output taps,
  finite sentinels, measured-cost labels, beginner documentation, proof boundary,
  and clean host/container verification before committing.

### 2026-08-29T18:28:00Z — Real projections pushed; CPU-011 started

- Commit `643fe5c` (`feat: execute real mixer projections`) was pushed to
  `origin/main` successfully and the worktree was clean.
- Added CPU-011 before implementation and made it a CPU-004 dependency. This is
  the first state-mutating real layer boundary: it must compose direct-scale
  input RMSNorm, packed projection, convolution-before-split, GGUF tiled to
  semantic grouped conversion, folded gates, FP32 recurrence, gated RMSNorm,
  output projection, and residual addition without losing physical-state order.
- Added EDU-012 so convolution versus recurrent state, layout transitions,
  gate/norm order, residual ownership, evidence taps, and the later session
  atomicity boundary are explained alongside the implementation.
- Re-verified the pinned Transformers source after the web fetch could not serve
  the commit URL. The exact hashed file defines variance-mean RMSNorm with
  epsilon `1e-6`, direct learned scale, then FP32 SiLU gate multiplication; it
  also confirms convolution precedes Q/K/V split and recurrence precedes gated
  norm/output projection.

### 2026-08-29T18:33:00Z — CPU-011 and EDU-012 accepted

- Extended the pinned GDN contract with the exact gated RMSNorm symbol and
  equation: per-head FP32 variance mean, reciprocal square root with `1e-6`,
  direct learned scale, then FP32 SiLU gate multiplication.
- Implemented prepared scalar parameter decoding for input norm, 10,240 × 4
  convolution weights, folded A, time bias, and recurrent norm. Added exact
  state/workspace contracts and a complete real layer-0 GDN mixer step through
  input normalization, projection, convolution, split, layout conversions,
  gates, recurrence, gated norm, output projection, and residual addition.
- The scalar path keeps convolution rings in physical packed/tiled order and
  recurrent matrices in semantic grouped order. Value-associated activations
  cross the explicit conversion boundary in each direction required by the
  stored GGUF matrices.
- An independent Python transcription maps only the pinned GGUF, decodes selected
  real rows, uses float libm equations, and matches native normalization,
  convolution, grouped heads 0/1/3, folded/update gates, recurrence, gated norm,
  convolution state, and recurrent state under frozen tolerances. Exact selected
  residual additions also pass.
- A short gated-tiled workspace is rejected before projection or state mutation;
  both persistent buffers remain exactly zero. Later failures after validated
  in-place mutation remain the explicit SES-002 atomic-staging boundary.
- Added a beginner chapter explaining the complete branch, parameter/workspace/
  state ownership and byte counts, direct norm scale, convolution-before-split,
  physical versus semantic state, recurrence order, gated norm, output and
  residual, evidence selection, and remaining oracle/atomicity limitations.
- Commands: pinned raw-source re-verification, fixture generation (2.57 s),
  timed real diagnostic (0.25 s; 149,920 KiB maximum RSS), Ruff format/check,
  JSON validation, clean restricted C++17 build, 52 pytest tests,
  `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation, and RTX
  5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-011 and EDU-012 done. The layer FFN, real attention state, multiple
  real tokens, 64-layer schedule, final norm, and logits remain CPU-004.

### 2026-08-29T18:33:30Z — Real-GDN commit boundary

- Reviewed source equations, prepared parameters, all workspace/state counts,
  physical/semantic layout transitions, mutation order, independent selected
  taps, residual arithmetic, pre-mutation failure behavior, documentation,
  evidence labels, and clean host/container verification before committing.

### 2026-08-29T18:41:00Z — Real GDN pushed; CPU-012 started

- Commit `1725fc5` (`feat: execute real GDN mixer step`) was pushed to
  `origin/main` successfully. The separate runtime-architecture artifacts were
  subsequently committed as `1ee916c` and the shared worktree returned clean.
- Added CPU-012 before implementation and made it a CPU-004 dependency. The
  existing FFN oracle used small dense synthetic matrices; the real path must
  prove direct-scale post-mixer normalization, all 17,408 Q4_K gate/up rows,
  elementwise SwiGLU, all 5,120 Q4_K down rows, exact temporary storage, and the
  second residual addition through typed GGUF views.
- Added EDU-013 so the two wide projections, nonlinear elementwise stage, down
  projection, scalar cost, evidence limits, and complete decoder-layer boundary
  are explained with the implementation.

### 2026-08-29T18:45:00Z — CPU-012 diagnostic compile failure

- The first native diagnostic build failed under restricted C++17 because two
  local `constexpr std::initializer_list` objects referred to compiler-created
  backing arrays that were not constant expressions. No diagnostic executed.
- The tap indices are runtime-only display metadata and require no compile-time
  evaluation. The follow-up changes those two locals to `const`; FFN arithmetic,
  workspace sizes, and expected evidence remain unchanged.

### 2026-08-29T18:46:07Z — CPU-012 and EDU-013 accepted

- Implemented prepared direct-scale post-mixer norm parameters and the complete
  real layer-0 FFN branch: 5,120-to-17,408 Q4_K gate/up projections, FP32
  `SiLU(gate) * up`, 17,408-to-5,120 Q4_K down projection, and FP32 residual
  addition. Every pointer and exact count is rejected before any write.
- Added an independently mapped GGUF fixture generator using the pinned Q4_K
  equations and float libm functions. Selected normalization, gate, up, and
  activated taps meet frozen absolute, relative, and RMS tolerances; exact
  SHA-256 hashes bind all selected physical gate/up rows to the admitted model.
- The native diagnostic executes all 62,464 workspace values, verifies they are
  finite, and proves exact FP32 residual addition at selected lanes. A one-value
  short activated buffer fails before either the gate or output is written.
- The complete scalar branch performs an estimated 267,386,880 weight products.
  Its 62,464-value FP32 workspace occupies 249,856 bytes (244 KiB), and its
  prepared 5,120-value norm occupies 20 KiB. The timed admitted-model diagnostic
  completed in 0.33 s with 172,960 KiB maximum RSS.
- Added a beginner chapter explaining feature vectors, feed-forward execution,
  the intermediate width, independent gate/up projections, SiLU, elementwise
  multiplication versus dot products, Q4_K row execution, down projection,
  residuals, workspace lifetime, fixture equality, and the remaining oracle and
  scheduler boundary. Updated the handbook index, root reading path, and source
  ledger.
- Commands: fixture regeneration, focused diagnostic and 4 focused tests, Ruff
  format/check, clean restricted C++17 build, 56 pytest tests,
  `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation, and RTX
  5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-012 and EDU-013 done. GDN-to-FFN layer composition, real attention
  state/output, multiple real tokens, all 64 layers, final norm, logits, and
  direct semantic trace admission remain CPU-004/TRC/ORA work.

### 2026-08-29T18:46:30Z — Real-FFN commit boundary

- Reviewed exact workspace validation, projection dimensions, FP32 SwiGLU and
  residual arithmetic, independent physical-row fixture evidence, generator
  reproducibility, finite full-path execution, documentation claims and labels,
  preserved compile failure, and clean host/container verification before
  committing.

### 2026-08-29T18:47:14Z — Real FFN pushed; CPU-013 started

- Commit `3db3f0a` (`feat: execute real SwiGLU FFN`) was pushed to `origin/main`
  successfully and the worktree was clean.
- Added CPU-013 before implementation and made it a CPU-004 dependency. The
  existing attention oracle proves the equation with synthetic tensors, while
  CPU-010 stops after real layer-3 Q/gate/K/V projection. The new boundary must
  compose direct-scale input/query/key norms, partial RoPE, two sequential KV
  appends, six-query-head grouping, causal softmax, sigmoid output gating, the
  Q6_K output projection, and residual addition through admitted typed views.
- Two positions are required because position zero alone makes RoPE an identity
  and causal softmax a single value. Capacity and malformed-buffer failures must
  be tested before persistent KV mutation; session-level rollback after later
  failures remains SES-002 work.
- Added EDU-014 so KV ownership, query-to-KV grouping, position-dependent RoPE,
  causal lookup, softmax, output gating, projection, residual, and the atomicity
  boundary are explained alongside the implementation.

### 2026-08-29T18:54:26Z — CPU-013 and EDU-014 accepted

- Added a direct-scale form of the retained scalar attention primitive so real
  converted GGUF query/key norm scales are not treated as source-checkpoint
  offsets. The original offset-form synthetic oracle and its fixtures remain
  unchanged and passing.
- Implemented prepared layer-3 input/query/key norm parameters and two real
  attention steps through input normalization, Q8_0 packed Q/gate/K/V
  projections, head-local split, per-head direct RMSNorm, 64-of-256 partial
  RoPE, grouped causal attention, sigmoid output gate, Q6_K output projection,
  and FP32 residual addition.
- The two-position FP32 state contains 4,096 KV values. Exact preflight checks
  cover pointer/count contracts, cache-size multiplication overflow, position
  capacity, score scratch, and output sizes before persistent KV mutation.
- An independent mapped-GGUF transcription matches exact FP32 taps for token-one
  normalization and raw Q/gate, both positions' rotated-normalized K and raw V,
  and two-position gated attention. Taps cover RoPE lanes 31/32/63/64, query
  heads 5/6 around the first six-to-one group boundary, and the final head 23;
  selected physical rows are bound by SHA-256.
- The full Q6_K output projection executes natively with finite buffers and
  selected output lanes obey exact token-one FP32 residual addition. Its complete
  semantic trace remains explicit TRC/ORA work rather than being promoted from
  a native self-check.
- Malformed attention-output storage and a position equal to capacity both fail
  before KV or final output changes. A hypothetical failure after the admitted
  KV append remains the explicit SES-002 transactional-staging boundary.
- Added a beginner chapter explaining Q/K/V, heads and lanes, direct norms,
  partial RoPE pairs, KV ownership and 8 GiB production arithmetic, GQA mapping,
  causal scoring, stable softmax, output gates, projection/residual, workspace,
  evidence, and atomicity. Updated the handbook index, root reading path, and
  source ledger.
- The independent generator completed in 4.49 s with 28,800 KiB maximum RSS;
  the two-token native diagnostic completed in 0.35 s with 127,520 KiB maximum
  RSS. Projection dimension arithmetic is 104,857,600 scalar weight products
  per token; explicit two-position workspace is 43,010 FP32 values.
- Commands: Ruff format/check, fixture generation and JSON validation, focused
  9-test attention suite, clean restricted C++17 build, 61 pytest tests,
  `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation, and RTX
  5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-013 and EDU-014 done. Full-layer composition, multiple real hybrid
  layers, embedding/final norm/logits, and direct trace admission remain
  CPU-004/TRC/ORA work.

### 2026-08-29T18:55:00Z — Real-attention commit boundary

- Reviewed source-offset versus converted direct-scale norm ownership, exact
  production shapes, partial-RoPE lanes, GQA boundaries, causal FP32 arithmetic,
  KV mutation order, overflow/count validation, physical-row hashes, output and
  residual proof limits, beginner documentation, and clean host/container
  verification before committing.

## Decisions and Negative Results

- **2026-08-29 / BLD-002:** Host `nvcc` is absent. Resolved for reproducibility
  by the pinned CUDA 13.0.2 container; the task stays in progress because only a
  device diagnostic, not engine CUDA kernels, targets SM120.
- **2026-08-29 / MDL-001:** Whole-file in-process SHA-256 takes 56.603 s. It is
  correct and fail-closed, but cache-keyed verification has not been designed or
  admitted. No shortcut was implemented.
- **2026-08-29 / release boundary:** CUDA primitives, scalar execution,
  tokenizer/template, sessions, HTTP APIs, quality evaluation, 128K fit,
  optimization, and comparative performance have no implementation evidence and
  remain pending. Product stubs exit nonzero so their presence cannot be mistaken
  for delivery.
- **2026-08-29 / TOK-001:** Initial native compilation failed because the C++
  include path omitted the pinned utf8proc directory. Added the explicit include
  path to the restricted Makefile; clean host and container builds then passed.
- **2026-08-29 / CPU-001:** Exact equality is appropriate for frozen decoded
  FP32 bytes, but not for a cosine value derived through square roots. The first
  metric test failed on this distinction; the cosine reporter now has a `1e-15`
  absolute tolerance while structural, decoded-value, and dot bits stay exact.
- **2026-08-29 / CPU-003:** The first attention fixture used double-precision
  Python transcendental functions despite claiming scalar FP32 operations and
  failed the frozen relative gate. Regenerating through float libm functions
  fixed the authority transcription; no native code or tolerance was changed.
- **2026-08-29 / CPU-006:** Initial test collection lacked the repository import
  root; after fixing that harness issue, the admitted-row oracle decoded only
  the first block of a multi-block Q4 row. Walking every block resolved the
  mismatch without changing native tensor code or expected arithmetic order.
