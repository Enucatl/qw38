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
| TOK-001 | Implement pinned tokenizer | MDL-001, PIN-002 | in_progress | Token IDs match frozen authority fixtures byte-for-byte | — |
| TOK-002 | Implement chat/reasoning/tool template | TOK-001 | pending | All supported roles, reasoning, tool calls/results, and rejection cases match fixtures | — |
| CPU-001 | Implement Q4_K/Q6_K scalar decoding and dot products | MDL-001 | pending | Numeric fixtures meet frozen metrics and exact structural checks | — |
| CPU-002 | Implement scalar GDN oracle | CPU-001, MDL-002 | pending | Warm-up, recurrence, state, head mapping, and chunk-boundary fixtures pass | — |
| CPU-003 | Implement scalar attention and FFN oracle | CPU-001, MDL-002 | pending | Layers 3/7/63, partial RoPE, grouped KV, causality, and FFN taps pass | — |
| CPU-004 | Implement full scalar 64-layer scheduler and logits | CPU-002, CPU-003 | pending | Token/chunk execution and logits match semantic-authority fixtures | — |
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
