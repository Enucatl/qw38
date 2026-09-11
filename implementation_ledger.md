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
| PIN-004 | Decouple source provenance from feature contracts | PIN-003 | done | Feature contracts omit routine local-source hashes; release provenance records Git, build/container, and selected evidence identity | [`tools/release_provenance.py`](tools/release_provenance.py); [`tasks/PIN-004.md`](tasks/PIN-004.md); verification 2026-09-04 |
| ENV-001 | Capture and validate the local production-toolchain prerequisites | ART-002 | done | GPU/driver/toolkit/container availability is recorded; unavailable prerequisites have explicit follow-up | CUDA probe log 2026-08-29T09:55:00Z |
| BLD-001 | Establish brand, repository layout, and C++17 build | ART-002 | done | Literal brand appears in user-facing tools; Makefile builds restricted host targets | [`Makefile`](Makefile), [`include/qw38/engine.h`](include/qw38/engine.h), pytest log 2026-08-29T09:52:00Z |
| BLD-002 | Add pinned CUDA 13.0 SM120 build path | PIN-003, BLD-001 | done | Diagnostic and release CUDA builds target `sm_120` with recorded flags | [`Makefile`](Makefile); [`pins/cuda_quant_contract.json`](pins/cuda_quant_contract.json); log 2026-08-31T06:05:47Z |
| BLD-003 | Define and enforce the device allocation ledger | PIN-001 | done | All persistent/transient allocations and 128K budgets are enumerated and checked | [`docs/13-allocation-ledger.md`](docs/13-allocation-ledger.md); [`docs/54-post-graph-128k-memory.md`](docs/54-post-graph-128k-memory.md); [`cuda/memory_fit_test.cu`](cuda/memory_fit_test.cu); [`tests/test_cuda_memory_fit.py`](tests/test_cuda_memory_fit.py); log 2026-09-03T12:50:48Z |
| BLD-004 | Reconcile diagnostic CUDA target linkage and workspace accounting | PIN-004, BLD-002, BLD-003, EVAL-001 | done | Diagnostic scheduler consumers link trace-enabled objects and prompt and memory byte accounting agree exactly | [`tasks/BLD-004.md`](tasks/BLD-004.md); clean CUDA verification 2026-09-04T12:25:31Z |
| API-001 | Implement explicit `Status` and move-only Engine/Session boundary | BLD-001 | done | Public header compiles without exceptions/RTTI and exposes the approved operations | [`include/qw38/engine.h`](include/qw38/engine.h), build log 2026-08-29T09:52:00Z |
| API-002 | Implement bounded JSON parsing and canonical serialization for the server | SRV-001 | done | Valid JSON, Unicode escapes, depth/body limits, canonical tool JSON, and malformed inputs pass native fixtures without a general JSON dependency | [`src/server_json.cpp`](src/server_json.cpp); [`src/server_api_test.cpp`](src/server_api_test.cpp); [`pins/chat_completions_contract.json`](pins/chat_completions_contract.json); log 2026-09-01T19:00:04Z |
| MDL-001 | Parse, mmap, inventory, and fail-closed validate GGUF | PIN-001, API-001 | done | Exact tensor metadata/ranges/roles are checked; malformed fixtures pass pytest | [`pins/tensor_inventory.json`](pins/tensor_inventory.json), [`src/model.cpp`](src/model.cpp); log 2026-08-29T10:35:51Z |
| MDL-003 | Accelerate full-file SHA-256 authentication with hardware dispatch | PIN-001, MDL-001 | done | The exact pinned digest is unchanged; accelerated and portable backends pass byte fixtures; the production container selects acceleration; full-model before/after timings and fallback behavior are retained | [`src/sha256.cpp`](src/sha256.cpp); [`fixtures/sha256_acceleration.json`](fixtures/sha256_acceleration.json); [`docs/57-hardware-sha256.md`](docs/57-hardware-sha256.md); log 2026-09-01T11:56:52Z |
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
| CPU-014 | Compose complete real GDN and attention decoder layers through their FFN branches | CPU-011, CPU-012, CPU-013 | done | Layer-0 GDN→FFN and layer-3 attention→FFN use the post-mixer residual, preserve exact branch order, meet frozen taps, and reject malformed FFN storage before persistent mixer-state mutation | [`src/scheduler.cpp`](src/scheduler.cpp); [`fixtures/real_layer_composition.json`](fixtures/real_layer_composition.json); [`tests/test_real_layer_composition.py`](tests/test_real_layer_composition.py); log 2026-08-30T06:31:18Z |
| CPU-015 | Execute real token embedding lookup, final RMSNorm, and complete FP32 vocabulary logits | CPU-006, CPU-008, CPU-014 | done | Valid token rows decode exactly, out-of-range IDs fail before writes, direct-scale final norm and all 248,320 Q6_K logits are finite, selected logits match independently decoded rows, and malformed workspaces fail closed | [`src/scheduler.cpp`](src/scheduler.cpp); [`fixtures/real_model_boundaries.json`](fixtures/real_model_boundaries.json); [`tests/test_real_model_boundaries.py`](tests/test_real_model_boundaries.py); log 2026-08-30T06:39:56Z |
| CPU-016 | Execute one real token through the exact 64-layer scalar schedule and complete logits | CPU-014, CPU-015 | done | Prepared parameters and independent state slots cover 48 GDN/16 attention layers; token embedding flows through layers 0–63, final norm, and all logits; stable boundary taps and state mutations are retained; malformed global storage fails before mutation | [`src/scalar_runtime.cpp`](src/scalar_runtime.cpp); [`fixtures/real_scalar_token.json`](fixtures/real_scalar_token.json); [`tests/test_real_scalar_token.py`](tests/test_real_scalar_token.py); log 2026-08-30T06:53:51Z |
| CPU-002 | Implement scalar GDN oracle | CPU-001, MDL-002 | done | Warm-up, recurrence, state, head mapping, and chunk-boundary fixtures pass | [`src/gdn.cpp`](src/gdn.cpp); [`fixtures/gdn_authority.json`](fixtures/gdn_authority.json); [`tests/test_gdn.py`](tests/test_gdn.py); log 2026-08-29T12:26:00Z |
| CPU-003 | Implement scalar attention and FFN oracle | CPU-001, MDL-002 | done | Layers 3/7/63, partial RoPE, grouped KV, causality, and FFN taps pass | [`src/attention.cpp`](src/attention.cpp); [`fixtures/attention_ffn_authority.json`](fixtures/attention_ffn_authority.json); [`tests/test_attention.py`](tests/test_attention.py); log 2026-08-29T12:54:00Z |
| CPU-004 | Implement full scalar 64-layer scheduler and logits | CPU-002, CPU-003, CPU-005, CPU-006, CPU-007, CPU-008, CPU-009, CPU-010, CPU-011, CPU-012, CPU-013, CPU-014, CPU-015, CPU-016 | done | Multi-token token-wise and arbitrary chunk execution have exact state/frontier/chunk equivalence and emit logits ready for oracle comparison | [`src/scalar_runtime.cpp`](src/scalar_runtime.cpp); [`fixtures/real_scalar_chunk.json`](fixtures/real_scalar_chunk.json); [`tests/test_real_scalar_chunk.py`](tests/test_real_scalar_chunk.py); log 2026-08-30T11:57:06Z |
| TRC-001 | Define versioned trace bundle and typed comparison metrics | PIN-002 | done | Manifest/blob schema, checksums, summaries, session frontiers, and metric reporter pass tests | [`pins/trace_contract.json`](pins/trace_contract.json); [`tools/qw38_trace.py`](tools/qw38_trace.py); [`tests/test_trace.py`](tests/test_trace.py); log 2026-08-30T07:13:12Z |
| TRC-003 | Add build-isolated backend-neutral trace sink and exact filters | TRC-001 | done | Diagnostic build accepts validated layer/name filters and emits typed views; release objects contain no trace API or tap names | [`src/diagnostic_trace.h`](src/diagnostic_trace.h); [`tests/test_diagnostic_trace.py`](tests/test_diagnostic_trace.py); log 2026-08-30T07:19:17Z |
| TRC-002 | Add diagnostic-only stable scalar taps | TRC-003, CPU-016 | done | Required scalar taps use the backend-neutral sink, emit through the v1 bundle, and match filtered native scalar evidence | [`pins/scalar_trace_contract.json`](pins/scalar_trace_contract.json); [`src/scalar_runtime.cpp`](src/scalar_runtime.cpp); [`tests/test_real_scalar_trace.py`](tests/test_real_scalar_trace.py); log 2026-08-30T07:32:39Z |
| TRC-004 | Add diagnostic-only stable CUDA taps | TRC-002, CUD-001 | done | CUDA visible boundaries use pinned scalar tap names/shapes and pass frozen scalar/oracle comparison gates | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`pins/cuda_trace_contract.json`](pins/cuda_trace_contract.json); [`fixtures/cuda_trace.json`](fixtures/cuda_trace.json); [`tests/test_cuda_trace.py`](tests/test_cuda_trace.py); [`docs/63-cuda-diagnostic-traces.md`](docs/63-cuda-diagnostic-traces.md); log 2026-09-03T10:27:00Z |
| ORA-002 | Build and validate pinned llama.cpp same-GGUF authority harness | PIN-001, PIN-002, CPU-004 | done | Exact revision builds reproducibly; identical tokens/template run on the pinned GGUF; logits/continuation metadata and source identity are retained | [`pins/llama_authority_contract.json`](pins/llama_authority_contract.json); [`fixtures/llama_scalar_authority.json`](fixtures/llama_scalar_authority.json); [`tests/test_llama_authority.py`](tests/test_llama_authority.py); log 2026-08-30T12:27:27Z |
| ORA-003 | Build pinned Transformers eager/offload semantic trace authority | PIN-002, TRC-002 | done | Exact source/model revisions execute within host/GPU limits and emit required taps, or an evidenced infeasibility creates an approved replacement task | [`pins/transformers_authority_contract.json`](pins/transformers_authority_contract.json); [`fixtures/transformers_scalar_authority.json`](fixtures/transformers_scalar_authority.json); [`tools/run_transformers_authority.py`](tools/run_transformers_authority.py); [`tests/test_transformers_authority.py`](tests/test_transformers_authority.py); log 2026-08-30T17:38:10Z |
| ORA-004 | Freeze three-authority scalar fixtures and per-tap tolerances | ORA-002, ORA-003, TRC-002, CPU-004 | done | Attributed bundles compare every required tap/logit; tolerances and genuine greedy near-ties are immutable | [`fixtures/scalar_authority_alignment.json`](fixtures/scalar_authority_alignment.json); [`pins/scalar_oracle_tolerances.json`](pins/scalar_oracle_tolerances.json); [`tests/test_scalar_authority_alignment.py`](tests/test_scalar_authority_alignment.py); log 2026-08-30T18:16:04Z |
| ORA-001 | Generate and freeze scalar/oracle fixtures and tolerances | ORA-002, ORA-003, ORA-004 | done | Three authorities are attributed; tolerances and greedy tie exceptions are immutable inputs | [`fixtures/scalar_authority_alignment.json`](fixtures/scalar_authority_alignment.json); [`pins/scalar_oracle_tolerances.json`](pins/scalar_oracle_tolerances.json); log 2026-08-30T18:16:04Z |
| CUD-001 | Implement CUDA Q4_K/Q6_K decode MMV | CPU-001, BLD-002, ORA-001 | done | Scalar-vs-CUDA and focused primitive pytest gates pass | [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu); [`fixtures/cuda_quant_mmv.json`](fixtures/cuda_quant_mmv.json); [`tests/test_cuda_quant_mmv.py`](tests/test_cuda_quant_mmv.py); log 2026-08-31T06:05:47Z |
| CUD-002 | Implement quantized tiled prompt MMQ | CUD-001 | done | Arbitrary prompt-row fixtures pass frozen tolerances | [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu); [`fixtures/cuda_quant_mmq.json`](fixtures/cuda_quant_mmq.json); [`tests/test_cuda_quant_mmv.py`](tests/test_cuda_quant_mmv.py); log 2026-08-31T06:16:39Z |
| CUD-003 | Complete real-scheduler CUDA projection and pointwise prerequisites | CUD-001, CPU-016 | done | Q8_0 MMV, quantized embedding-row decode, BF16 RMSNorm/residual/SwiGLU, packed attention split, GDN gate/layout preparation, and gated output meet scalar/device gates | [`cuda/scheduler_primitives.cu`](cuda/scheduler_primitives.cu); [`fixtures/cuda_scheduler_primitives.json`](fixtures/cuda_scheduler_primitives.json); [`tests/test_cuda_scheduler_primitives.py`](tests/test_cuda_scheduler_primitives.py); log 2026-08-31T11:26:17Z |
| GDN-001 | Implement exact one-token CUDA GDN and atomic state commit | CUD-001, CPU-002 | done | State/taps match oracle and injected failures leave frontier unchanged | [`cuda/gdn_step.cu`](cuda/gdn_step.cu); [`fixtures/cuda_gdn_step.json`](fixtures/cuda_gdn_step.json); [`tests/test_cuda_gdn.py`](tests/test_cuda_gdn.py); log 2026-08-31T06:29:12Z |
| GDN-002 | Implement chunked GDN prefill with 64-token scans | GDN-001, CUD-002 | done | Arbitrary chunks equal token-wise execution under frozen gates | [`cuda/gdn_step.cu`](cuda/gdn_step.cu); [`fixtures/cuda_gdn_chunk.json`](fixtures/cuda_gdn_chunk.json); [`tests/test_cuda_gdn_chunk.py`](tests/test_cuda_gdn_chunk.py); log 2026-08-31T06:42:29Z |
| ATN-001 | Implement grouped-query attention and partial RoPE | CUD-001, CPU-003 | done | Decode, causality, KV grouping, and layers 3/7/63 pass | [`cuda/attention_decode.cu`](cuda/attention_decode.cu); [`fixtures/cuda_attention_decode.json`](fixtures/cuda_attention_decode.json); [`tests/test_cuda_attention.py`](tests/test_cuda_attention.py); log 2026-08-31T06:57:49Z |
| ATN-002 | Implement memory-bounded causal attention prefill | ATN-001, CUD-002 | done | Chunked prompt fixtures and 131,072 capacity boundary pass | [`cuda/attention_decode.cu`](cuda/attention_decode.cu); [`fixtures/cuda_attention_prefill.json`](fixtures/cuda_attention_prefill.json); [`tests/test_cuda_attention_prefill.py`](tests/test_cuda_attention_prefill.py); log 2026-08-31T09:33:05Z |
| ATN-003 | Repair OPT-005 normalized-key head indexing regression | ATN-002 | done | Retained attention oracle passes after every KV head writes its own normalized-key range | [`cuda/attention_decode.cu`](cuda/attention_decode.cu); [`tests/test_cuda_attention.py`](tests/test_cuda_attention.py); clean CUDA verification 2026-09-04T12:25:31Z |
| SCH-001 | Implement hybrid 64-layer CUDA scheduler and FP32 logits | GDN-002, ATN-002, CUD-003 | done | Full traces/logits and greedy continuations meet frozen gates | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`fixtures/cuda_full_scheduler.json`](fixtures/cuda_full_scheduler.json); [`tests/test_cuda_full_scheduler.py`](tests/test_cuda_full_scheduler.py); log 2026-08-31T12:13:55Z |
| SCH-002 | Integrate chunked prompt execution into the full CUDA scheduler | SCH-001, GDN-002, ATN-002, CUD-002, BEN-001 | done | End-to-end prefill uses prompt-row MMQ and chunked GDN/attention paths, remains token-wise equivalent, and no longer dispatches one complete decode token at a time | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`fixtures/cuda_prompt_scheduler.json`](fixtures/cuda_prompt_scheduler.json); [`tests/test_cuda_prompt_scheduler.py`](tests/test_cuda_prompt_scheduler.py); log 2026-09-02T16:25:43Z |
| SES-001 | Implement exact common-prefix sync/reuse | SCH-001 | done | Reuse and full replay produce the same committed state and logits | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`fixtures/cuda_prefix_sync.json`](fixtures/cuda_prefix_sync.json); [`tests/test_cuda_prefix_sync.py`](tests/test_cuda_prefix_sync.py); log 2026-08-31T13:45:38Z |
| SES-002 | Implement atomic eval/sample/commit semantics | SCH-001 | done | Sampling is separate; cancellation/error cannot partially commit state | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`fixtures/cuda_atomic_eval.json`](fixtures/cuda_atomic_eval.json); [`tests/test_cuda_atomic_eval.py`](tests/test_cuda_atomic_eval.py); log 2026-08-31T17:55:34Z |
| SES-003 | Implement atomic checkpoint save/restore | SES-001, SES-002 | done | All state and compatibility hashes persist; resumed continuation is exact | [`cuda/checkpoint.cu`](cuda/checkpoint.cu); [`fixtures/cuda_checkpoint.json`](fixtures/cuda_checkpoint.json); [`tests/test_cuda_checkpoint.py`](tests/test_cuda_checkpoint.py); log 2026-08-31T19:09:54Z |
| SES-004 | Implement atomic Responses continuation records | SES-001, SES-003, API-003 | done | A completed stored response atomically records its exact committed token prefix and compatible tool contract; missing, malformed, incompatible, cancelled, and `store=false` responses cannot be continued | [`src/response_store.cpp`](src/response_store.cpp); [`fixtures/responses.json`](fixtures/responses.json); [`tests/test_server.py`](tests/test_server.py); log 2026-09-02T14:27:35Z |
| MEM-001 | Demonstrate 131,072-token fit with 1.5 GiB reserve | BLD-003, SCH-001 | done | Post-graph measured ledger includes 8 GiB KV and every named allocation on RTX 5090 | [`cuda/memory_fit_test.cu`](cuda/memory_fit_test.cu); [`fixtures/cuda_memory_fit_post_graph.json`](fixtures/cuda_memory_fit_post_graph.json); [`docs/54-post-graph-128k-memory.md`](docs/54-post-graph-128k-memory.md); [`tests/test_cuda_memory_fit.py`](tests/test_cuda_memory_fit.py); log 2026-09-01T07:08:29Z |
| MEM-002 | Revalidate 128K reserve after adding fixed prompt-chunk scratch | MEM-001, SCH-002 | done | Full session, decode workspace, prompt workspace, and uploaded graphs still leave at least 1.5 GiB free with allocator deltas reconciled | [`fixtures/cuda_memory_fit_post_graph.json`](fixtures/cuda_memory_fit_post_graph.json); [`tests/test_cuda_memory_fit.py`](tests/test_cuda_memory_fit.py); log 2026-09-02T16:25:43Z |
| OPT-001 | Add synchronized timings, NVTX, and attribution | SCH-001 | done | Component/end-to-end measurements expose every named time category | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/timing_test.cu`](cuda/timing_test.cu); [`fixtures/cuda_timing.json`](fixtures/cuda_timing.json); tests; log 2026-08-31T20:00:28Z |
| OPT-002 | Profile and implement justified fusions | OPT-001, ORA-001 | done | Nsight evidence justifies each fusion; fused/unfused boundaries pass frozen gates | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`fixtures/cuda_fusion.json`](fixtures/cuda_fusion.json); [`evidence/profiling/opt002-nsight-compute.txt`](evidence/profiling/opt002-nsight-compute.txt); tests; log 2026-09-01T05:44:16Z |
| OPT-003 | Implement stable-address CUDA graphs | OPT-002 | done | Graph/non-graph equivalence passes and graph allocations are in MEM-001 | [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`fixtures/cuda_graph.json`](fixtures/cuda_graph.json); [`tests/test_cuda_graph.py`](tests/test_cuda_graph.py); log 2026-09-01T07:08:29Z |
| OPT-004 | Tune row buckets/chunks and check in dispatch evidence | OPT-003 | done | Offline RTX 5090 sweep selects a reproducible table from retained raw results | [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu); [`fixtures/cuda_dispatch_tuning.json`](fixtures/cuda_dispatch_tuning.json); [`evidence/profiling/opt004-dispatch-sweep-raw.txt`](evidence/profiling/opt004-dispatch-sweep-raw.txt); [`tests/test_cuda_dispatch_tuning.py`](tests/test_cuda_dispatch_tuning.py); log 2026-09-01T07:30:51Z |
| OPT-005 | Implement exact tiled causal prompt attention with online softmax | OPT-004, SCH-002, ATN-002 | done | Multi-row attention tiles preserve full causal semantics and frozen outputs while eliminating per-token QK/softmax/value launches | [`tasks/OPT-005.md`](tasks/OPT-005.md); recovery accepted and delivered 2026-09-04T13:09:32Z |
| OPT-006 | Reuse tiled KV loads across grouped query heads | OPT-005 | done | Each shared KV head is loaded once per tile for its six query heads; exact GQA outputs pass and measured KV traffic falls | [`tasks/OPT-006.md`](tasks/OPT-006.md); [`cuda/gqa_attention_test.cu`](cuda/gqa_attention_test.cu); [`fixtures/cuda_gqa_attention.json`](fixtures/cuda_gqa_attention.json); log 2026-09-04T15:54:44Z |
| OPT-007 | Execute multiple prompt query rows per CUDA block | OPT-006 | done | Prompt attention maps query-row tiles to occupied blocks with bounded scratch and passes short/chunk boundary equivalence | [`tasks/OPT-007.md`](tasks/OPT-007.md); [`cuda/query_row_attention_test.cu`](cuda/query_row_attention_test.cu); [`tests/test_cuda_query_row_attention.py`](tests/test_cuda_query_row_attention.py); [`pins/cuda_query_row_attention_contract.json`](pins/cuda_query_row_attention_contract.json); [`fixtures/cuda_query_row_attention.json`](fixtures/cuda_query_row_attention.json); log 2026-09-04T17:43:02Z |
| OPT-008 | Make 4,096 tokens the default prompt chunk | OPT-007, MEM-002 | done | Default prefill chunks are 4,096 tokens with bounded fallback for tails/capacity; atomic commit, cancellation, and 128K reserve gates pass | [`tasks/OPT-008.md`](tasks/OPT-008.md); [`cuda/full_scheduler.h`](cuda/full_scheduler.h); [`cuda/prompt_scheduler_test.cu`](cuda/prompt_scheduler_test.cu); [`pins/cuda_prompt_scheduler_contract.json`](pins/cuda_prompt_scheduler_contract.json); [`fixtures/cuda_prompt_scheduler.json`](fixtures/cuda_prompt_scheduler.json); [`fixtures/cuda_memory_fit_post_graph.json`](fixtures/cuda_memory_fit_post_graph.json); log 2026-09-07T13:08:07Z |
| OPT-009 | Implement true batched Q8_0/Q4_K/Q6_K prompt MMQ | OPT-004, SCH-002 | done | Prompt projections reuse weight tiles across rows, select measured SM120 kernels, and preserve frozen numeric envelopes | [`tasks/OPT-009.md`](tasks/OPT-009.md); [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu); [`cuda/prompt_mmq_test.cu`](cuda/prompt_mmq_test.cu); [`pins/cuda_prompt_mmq_contract.json`](pins/cuda_prompt_mmq_contract.json); [`fixtures/cuda_prompt_mmq.json`](fixtures/cuda_prompt_mmq.json); [`evidence/profiling/opt009-mmq-tile-sweep-raw.txt`](evidence/profiling/opt009-mmq-tile-sweep-raw.txt); log 2026-09-07T15:11:00Z |
| OPT-010 | Tile KV layout for coalesced exact GQA access | OPT-007, SES-003 | done | KV storage supports coalesced tile loads without changing logical values, checkpoint compatibility, prefix reuse, or capacity | [`tasks/OPT-010.md`](tasks/OPT-010.md); [`cuda/attention_decode.cu`](cuda/attention_decode.cu); [`cuda/kv_tile_layout_test.cu`](cuda/kv_tile_layout_test.cu); [`pins/cuda_kv_tile_layout_contract.json`](pins/cuda_kv_tile_layout_contract.json); [`fixtures/cuda_kv_tile_layout.json`](fixtures/cuda_kv_tile_layout.json); log 2026-09-07T16:55:25Z |
| OPT-011 | Pipeline and fuse prompt execution and chunk commit | OPT-008, OPT-009, OPT-010 | done | Justified fusion/overlap removes avoidable copies, launches, and barriers while preserving cancellation and atomic publication | [`tasks/OPT-011.md`](tasks/OPT-011.md); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/prompt_pipeline_test.cu`](cuda/prompt_pipeline_test.cu); [`pins/cuda_prompt_pipeline_contract.json`](pins/cuda_prompt_pipeline_contract.json); [`fixtures/cuda_prompt_pipeline.json`](fixtures/cuda_prompt_pipeline.json); log 2026-09-07T18:12:37Z |
| OPT-012 | Add stable-address prompt CUDA graphs | OPT-003, OPT-011 | done | Common 4,096-token prompt paths replay from stable addresses with graph/non-graph equality and reconciled memory | [`tasks/OPT-012.md`](tasks/OPT-012.md); verification 2026-09-07T20:27:06Z |
| OPT-013 | Implement associative block-parallel GDN prompt scan | GDN-002, OPT-011 | done | Parallel GDN scan preserves recurrence/frontier tolerances across chunk boundaries and demonstrates measured prompt speedup | [`tasks/OPT-013.md`](tasks/OPT-013.md); [`pins/cuda_gdn_scan_contract.json`](pins/cuda_gdn_scan_contract.json); [`fixtures/cuda_gdn_scan.json`](fixtures/cuda_gdn_scan.json); verification 2026-09-08T08:58:06Z |
| OPT-014 | Instrument attributed 2K prefill time breakdown | BEN-001, OPT-013 | done | A cold 2K Quartz prefill emits a retained live attribution report whose named categories (at least embedding, GDN, attention, FFN/MMQ, logits, commit/sync, graph, other/idle) sum to the measured prefill wall time within a documented tolerance; the report is produced from the timed run itself without requiring a separate Nsight capture | [`tasks/OPT-014.md`](tasks/OPT-014.md); [`pins/cuda_prefill_attribution_contract.json`](pins/cuda_prefill_attribution_contract.json); [`fixtures/cuda_prefill_attribution.json`](fixtures/cuda_prefill_attribution.json); verification 2026-09-08T11:06:41Z |
| OPT-015 | Compare Quartz 2K prefill to llama.cpp; mine ds4 for techniques | OPT-013, PIN-002 | done | Checked-in report explains what pushes pinned same-GGUF llama.cpp into thousands of tok/s at 2K, uses `../ds4` only as MIT-licensed technique inspiration (ds4 cannot run this Qwen GGUF; no ds4 same-model baseline), separates transferable methods under `plan.md` provenance from non-transferable ds4 model policies such as sparse/compressed attention, maps each major Quartz 2K time sink to a faster path, and proposes a ranked recovery sequence | [`tasks/OPT-015.md`](tasks/OPT-015.md); [`evidence/optimization/opt015-2k-recovery/REPORT.md`](evidence/optimization/opt015-2k-recovery/REPORT.md); [`pins/opt015_recovery_contract.json`](pins/opt015_recovery_contract.json); [`fixtures/opt015_recovery.json`](fixtures/opt015_recovery.json); verification 2026-09-08T11:39:48Z |
| OPT-016 | Reach 2K prefill throughput at or above pinned llama.cpp | OPT-014, OPT-015 | blocked | On the same GGUF and RTX 5090, cold 2K Quartz prefill tok/s is ≥ pinned llama.cpp `llama-bench` 2K under a frozen protocol; until this passes, do not claim 8K/32K/128K throughput or QLT 128K as speed gates; OPT-021+ may use a 4K steering oracle without substituting for this gate | [`tasks/OPT-016.md`](tasks/OPT-016.md); blocked 2026-09-08T13:09:00Z after Ranks 1–3; recovery via `OPT-017`–`OPT-019` done, then `OPT-021`–`OPT-026` first 4K idea ladder and `OPT-027`–`OPT-031` second ladder before re-pass of the frozen 2K gate |
| OPT-017 | Admit production mixer Q8_0 MMA MMQ without loosening OPT-009 | OPT-009, OPT-015 | done | Production mixer Q8_0 prompt MMQ uses an MMA path while `launch_q8_mmq_bf16_reference` (or the existing OPT-009 byte-exact kernel) remains the visible unloosened reference; CUD-002/OPT-009 envelopes stay unloosened; a live exact-2048 re-attribution and unperturbed 2K remasurement are checked in (OPT-016 remains the parity gate owner) | [`tasks/OPT-017.md`](tasks/OPT-017.md); [`pins/opt017_mixer_mma_contract.json`](pins/opt017_mixer_mma_contract.json); [`fixtures/opt017_mixer_mma.json`](fixtures/opt017_mixer_mma.json); [`evidence/optimization/opt017-mixer-q8-mma/REPORT.md`](evidence/optimization/opt017-mixer-q8-mma/REPORT.md); verification 2026-09-08T14:31:30Z |
| OPT-018 | Close Q4_K/Q6_K MMA MMQ quality gap versus pinned llama.cpp | OPT-009, OPT-015 | done | Production Q4_K/Q6_K prompt MMA MMQ is brought to llama-competitive 2K FFN time on the same GGUF and RTX 5090 under unloosened CUD-002 envelopes, with retained variant reference, measured before/after FFN category time, and file-level llama.cpp/`plan.md` provenance; does not substitute for the OPT-016 end-to-end gate | [`tasks/OPT-018.md`](tasks/OPT-018.md); [`pins/opt018_ffn_mma_contract.json`](pins/opt018_ffn_mma_contract.json); [`fixtures/opt018_ffn_mma.json`](fixtures/opt018_ffn_mma.json); [`evidence/optimization/opt018-ffn-mma-quality/REPORT.md`](evidence/optimization/opt018-ffn-mma-quality/REPORT.md); verification 2026-09-08T18:55:28Z |
| OPT-019 | Close residual GDN and attention core time after mixer Q8_0 MMA | OPT-017, OPT-015 | done | After mixer projections are on MMA, remaining non-projection GDN scan/recurrence and causal attention core paths are brought down under frozen GDN/attention envelopes with retained references; live 2K re-attribution shows the residual sinks addressed; does not substitute for the OPT-016 end-to-end gate | [`tasks/OPT-019.md`](tasks/OPT-019.md); [`pins/opt019_core_recovery_contract.json`](pins/opt019_core_recovery_contract.json); [`fixtures/opt019_core_recovery.json`](fixtures/opt019_core_recovery.json); [`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](evidence/optimization/opt019-gdn-attention-core/REPORT.md); verification 2026-09-08T20:54:21Z |
| OPT-020 | Split 2K attribution into mixer MMQ versus core GDN/attention | OPT-014 | done | Cold exact-2048 attribution exposes separate mixer-projection MMQ time versus GDN-core and attention-core time (plus existing FFN/logits/commit/graph/idle), reconstructs wall within the OPT-014 tolerance, and is retained for OPT-017–OPT-019 steering; not a throughput gate | [`tasks/OPT-020.md`](tasks/OPT-020.md); [`pins/opt020_prefill_split_contract.json`](pins/opt020_prefill_split_contract.json); [`fixtures/opt020_prefill_split.json`](fixtures/opt020_prefill_split.json); verification 2026-09-08T21:29:30Z |
| OPT-021 | Pin cold 4K prefill oracle protocol and baseline | OPT-020 | done | Frozen protocol and checked-in report compare cold exact-4096 Quartz `sync_tokens` (3 replicates, attribution null, graphs created, production path) to same-sitting llama.cpp `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` on the pinned GGUF and RTX 5090; baseline Quartz/llama tok/s are retained as the OPT-022+ keep/reject oracle; does not substitute for OPT-016 | [`tasks/OPT-021.md`](tasks/OPT-021.md); [`pins/opt021_oracle_contract.json`](pins/opt021_oracle_contract.json); [`fixtures/opt021_oracle.json`](fixtures/opt021_oracle.json); [`evidence/optimization/opt021-4k-oracle/REPORT.md`](evidence/optimization/opt021-4k-oracle/REPORT.md); verification 2026-09-09T01:11:06Z |
| OPT-022 | Mixer Q8_0 quality MMQ with shared residual Y | OPT-017, OPT-018, OPT-021 | done | Production mixer Q8_0 prompt MMQ uses a llama/ds4-style quality stack (D4 `quantize_mmq_q8_1`, packed load-tiles, `MMQ_ITER_K=256`, shared residual Y across mixer GEMMs that share the same activation) under the plan.md Q8 association rule while OPT-009 byte-exact reference stays visible; cold exact-4096 mean tok/s strictly beats the OPT-021 Quartz baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-022.md`](tasks/OPT-022.md); [`pins/opt022_mixer_q8_quality_contract.json`](pins/opt022_mixer_q8_quality_contract.json); [`fixtures/opt022_mixer_q8_quality.json`](fixtures/opt022_mixer_q8_quality.json); [`evidence/optimization/opt022-mixer-q8-quality/REPORT.md`](evidence/optimization/opt022-mixer-q8-quality/REPORT.md); verification 2026-09-09T03:03:00Z |
| OPT-023 | Skinny-M mixer dispatch for small output rows | OPT-022 | done | Mixer projections with small `output_rows` (at least GDN α/β) dispatch through MMV or small-tile paths instead of J=128 MMA when that wins a paired CUDA-event A/B; cold exact-4096 mean tok/s strictly beats the post-OPT-022 oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-023.md`](tasks/OPT-023.md); [`pins/opt023_skinny_mixer_contract.json`](pins/opt023_skinny_mixer_contract.json); [`fixtures/opt023_skinny_mixer.json`](fixtures/opt023_skinny_mixer.json); [`evidence/optimization/opt023-skinny-mixer/REPORT.md`](evidence/optimization/opt023-skinny-mixer/REPORT.md); verification 2026-09-09T03:57:39Z |
| OPT-024 | Blackwell-aligned Q8_0 D2R for large mixer GEMMs | OPT-022 | done | Large mixer Q8_0 GEMMs use a ds4-inspired aligned SoA D2R / int8 MMA path (technique inspiration, not wholesale `../ds4/cuda/mmq` vendoring) under the plan.md Q8 association rule when it beats quality MMQ on those shapes; cold exact-4096 mean tok/s strictly beats the post-OPT-022 oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-024.md`](tasks/OPT-024.md); [`pins/opt024_mixer_q8_d2r_contract.json`](pins/opt024_mixer_q8_d2r_contract.json); [`fixtures/opt024_mixer_q8_d2r.json`](fixtures/opt024_mixer_q8_d2r.json); [`evidence/optimization/opt024-mixer-q8-d2r/REPORT.md`](evidence/optimization/opt024-mixer-q8-d2r/REPORT.md); [`evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md`](evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md); verification 2026-09-09T04:37:09Z |
| OPT-025 | Dense FFN shared-Y and SwiGLU-into-down Q8 | OPT-018, OPT-021 | done | Dense FFN reuses one Q8_1 of the FFN input for gate and up and writes down-leg Y from SwiGLU without an extra BF16 mid materialize where a paired A/B wins; cold exact-4096 mean tok/s strictly beats the then-current oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-025.md`](tasks/OPT-025.md); [`pins/opt025_ffn_shared_y_contract.json`](pins/opt025_ffn_shared_y_contract.json); [`fixtures/opt025_ffn_shared_y.json`](fixtures/opt025_ffn_shared_y.json); [`cuda/prefill_4k_ffn_test.cu`](cuda/prefill_4k_ffn_test.cu); [`evidence/optimization/opt025-ffn-shared-y/REPORT.md`](evidence/optimization/opt025-ffn-shared-y/REPORT.md); verification 2026-09-09T05:25:00Z |
| OPT-026 | Attention fattn stream-K occupancy | OPT-019, OPT-021 | done | Production fattn-mma prompt attention enables Ada+ stream-K (or equivalent occupancy fixup) under frozen OPT-019 envelopes when a paired A/B wins; cold exact-4096 mean tok/s strictly beats the then-current oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-026.md`](tasks/OPT-026.md); [`pins/opt026_fattn_streamk_contract.json`](pins/opt026_fattn_streamk_contract.json); [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json); [`cuda/prefill_4k_fattn_test.cu`](cuda/prefill_4k_fattn_test.cu); [`evidence/optimization/opt026-fattn-streamk/REPORT.md`](evidence/optimization/opt026-fattn-streamk/REPORT.md); verification 2026-09-09T05:54:24Z |
| OPT-027 | Persistent Ada+ fattn stream-K | OPT-026 | done | Replace OPT-026 `grid.z=2` KV bipartition with llama.cpp Ada+ persistent stream-K (`nsm × occupancy` linearized tiles + efficiency rounding) under frozen OPT-005 envelopes when a paired A/B wins; cold exact-4096 mean tok/s strictly beats the then-current oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-027.md`](tasks/OPT-027.md); [`pins/opt027_persistent_fattn_contract.json`](pins/opt027_persistent_fattn_contract.json); [`fixtures/opt027_persistent_fattn.json`](fixtures/opt027_persistent_fattn.json); [`cuda/prefill_4k_persistent_fattn_test.cu`](cuda/prefill_4k_persistent_fattn_test.cu); [`evidence/optimization/opt027-persistent-fattn/REPORT.md`](evidence/optimization/opt027-persistent-fattn/REPORT.md); [`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](evidence/optimization/opt027-persistent-fattn/REJECTION.md); verification 2026-09-09T07:05:46Z |
| OPT-028 | Q4_K/Q6_K MMQ stream-K | OPT-018, OPT-021 | done | Production quality MMA MMQ uses llama.cpp-style stream-K tile decomposition plus optional fixup under frozen Q4_K/Q6_K (and Q8 association where touched) envelopes when a paired A/B wins; recapture 4096 FFN graphs if nodes change; cold exact-4096 mean tok/s strictly beats the then-current oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-028.md`](tasks/OPT-028.md); [`pins/opt028_mmq_streamk_contract.json`](pins/opt028_mmq_streamk_contract.json); [`fixtures/opt028_mmq_streamk.json`](fixtures/opt028_mmq_streamk.json); [`cuda/prefill_4k_mmq_streamk_test.cu`](cuda/prefill_4k_mmq_streamk_test.cu); [`evidence/optimization/opt028-mmq-streamk/REPORT.md`](evidence/optimization/opt028-mmq-streamk/REPORT.md); [`evidence/optimization/opt028-mmq-streamk/REJECTION.md`](evidence/optimization/opt028-mmq-streamk/REJECTION.md); verification 2026-09-09T10:06:27Z |
| OPT-029 | Fuse GDN conv and gated output | OPT-019, OPT-021 | done | Collapse tiled causal conv and/or gated-output into the warp-column fused GDN token loop under frozen GDN-002 envelopes when a paired A/B wins; cold exact-4096 mean tok/s strictly beats the then-current oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-029.md`](tasks/OPT-029.md); [`pins/opt029_gdn_fuse_contract.json`](pins/opt029_gdn_fuse_contract.json); [`fixtures/opt029_gdn_fuse.json`](fixtures/opt029_gdn_fuse.json); [`cuda/prefill_4k_gdn_fuse_test.cu`](cuda/prefill_4k_gdn_fuse_test.cu); [`evidence/optimization/opt029-gdn-fuse/REPORT.md`](evidence/optimization/opt029-gdn-fuse/REPORT.md); [`evidence/optimization/opt029-gdn-fuse/REJECTION.md`](evidence/optimization/opt029-gdn-fuse/REJECTION.md); verification 2026-09-09T10:50:03Z |
| OPT-030 | Hopper/Blackwell PDL prompt launches | OPT-021 | done | Successive prompt kernels use programmatic dependent launch (PDL) serialization on sm_120 where a paired A/B wins versus current stream launches; arithmetic and graph-vs-fused byte equality stay; cold exact-4096 mean tok/s strictly beats the then-current oracle baseline or the change is reverted with a retained rejection; does not substitute for OPT-016 | [`tasks/OPT-030.md`](tasks/OPT-030.md); [`pins/opt030_pdl_launches_contract.json`](pins/opt030_pdl_launches_contract.json); [`fixtures/opt030_pdl_launches.json`](fixtures/opt030_pdl_launches.json); [`cuda/prefill_4k_pdl_test.cu`](cuda/prefill_4k_pdl_test.cu); [`evidence/optimization/opt030-pdl-launches/REPORT.md`](evidence/optimization/opt030-pdl-launches/REPORT.md); [`evidence/optimization/opt030-pdl-launches/REJECTION.md`](evidence/optimization/opt030-pdl-launches/REJECTION.md); verification 2026-09-09T11:34:14Z |
| OPT-031 | Mixer and GDN 4096 prompt graphs | OPT-012, OPT-021 | superseded | Remaining prompt graph work is included in OPT-055 after measured compute recovery and physical-batch selection; no implementation or speedup claimed for this row | [Recovery design](tasks/PERFORMANCE-RECOVERY-2026-09-10.md); [OPT-055](tasks/OPT-055.md); design 2026-09-10 |
| OPT-032 | Freeze decode oracle and refresh sink attribution | BEN-001, OPT-020, OPT-026 | done | Extend diagnostics with fresh P, D128, and D2048 baselines, exclusive decode categories, and matched pinned llama.cpp measurements; accept complete reproducible evidence and a recorded next-task order; this task claims no performance improvement | [`tasks/OPT-032.md`](tasks/OPT-032.md); [`pins/opt032_decode_oracle_contract.json`](pins/opt032_decode_oracle_contract.json); [`fixtures/opt032_decode_oracle.json`](fixtures/opt032_decode_oracle.json); [`evidence/optimization/opt032-decode-oracle/REPORT.md`](evidence/optimization/opt032-decode-oracle/REPORT.md); verification 2026-09-09T16:08:20Z |
| OPT-033 | Retain attention value sums in registers | OPT-032, OPT-026 | done | A/B full 4096-row attention including combine; keep register-resident value accumulation only with byte-equal output, lower component time, improved P, and the cross-workload guard; otherwise reject | [`tasks/OPT-033.md`](tasks/OPT-033.md); [`pins/opt033_register_vkq_contract.json`](pins/opt033_register_vkq_contract.json); [`fixtures/opt033_register_vkq.json`](fixtures/opt033_register_vkq.json); [`cuda/fattn_register_vkq_ab_test.cu`](cuda/fattn_register_vkq_ab_test.cu); [`evidence/optimization/opt033-register-vkq/REPORT.md`](evidence/optimization/opt033-register-vkq/REPORT.md); verification 2026-09-09T18:39:00Z |
| OPT-034 | Packed blockwise Q4_K/Q6_K MMV loads | OPT-032, CUD-001 | done | A/B production batch-1 projection shapes with unchanged packed staging and FP32 lane/reduction order; keep only with byte equality, lower weighted MMV time, improved D2048, and the cross-workload guard; otherwise reject | [`tasks/OPT-034.md`](tasks/OPT-034.md); [`pins/opt034_packed_mmv_contract.json`](pins/opt034_packed_mmv_contract.json); [`fixtures/opt034_packed_mmv.json`](fixtures/opt034_packed_mmv.json); [`cuda/packed_mmv_ab_test.cu`](cuda/packed_mmv_ab_test.cu); [`evidence/optimization/opt034-packed-mmv/REPORT.md`](evidence/optimization/opt034-packed-mmv/REPORT.md); verification 2026-09-09T21:41:35Z |
| OPT-035 | MMA attention probability times V | OPT-033 | done | A/B full 4096-row attention using dual-F16 probability×V MMA; retain frozen attention envelopes, lower component time, improved P, and the cross-workload guard; otherwise reject | [`tasks/OPT-035.md`](tasks/OPT-035.md); [`pins/opt035_pv_mma_contract.json`](pins/opt035_pv_mma_contract.json); [`fixtures/opt035_pv_mma.json`](fixtures/opt035_pv_mma.json); [`cuda/fattn_pv_mma_ab_test.cu`](cuda/fattn_pv_mma_ab_test.cu); [`evidence/optimization/opt035-pv-mma/REPORT.md`](evidence/optimization/opt035-pv-mma/REPORT.md); verification 2026-09-09T19:58:00Z |
| OPT-036 | Partition KV for vector decode attention | OPT-032, OPT-026 | done | A/B one-token vector attention at 1/4/8/16 KV partitions for D128/D2048 with deterministic partial-statistic merge; keep only with frozen attention envelopes, lower component time, improved D2048, and the cross-workload guard; otherwise reject | [`tasks/OPT-036.md`](tasks/OPT-036.md); [`pins/opt036_decode_kv_partition_contract.json`](pins/opt036_decode_kv_partition_contract.json); [`fixtures/opt036_decode_kv_partition.json`](fixtures/opt036_decode_kv_partition.json); [`cuda/decode_kv_partition_ab_test.cu`](cuda/decode_kv_partition_ab_test.cu); [`evidence/optimization/opt036-decode-kv-partition/REPORT.md`](evidence/optimization/opt036-decode-kv-partition/REPORT.md); verification 2026-09-09T17:32:00Z |
| OPT-037 | Select 4K FFN tiles per projection | OPT-032, OPT-025 | done | Sweep quality-MMQ I={64,128}, J={32,64,128} independently for 4096-row gate/up/down, preserve shared-Y and recapture graphs; keep only with admitted component wins, improved P, frozen MMQ envelopes, and the cross-workload guard; otherwise reject | [`tasks/OPT-037.md`](tasks/OPT-037.md); [`pins/opt037_ffn_tile_contract.json`](pins/opt037_ffn_tile_contract.json); [`fixtures/opt037_ffn_tiles.json`](fixtures/opt037_ffn_tiles.json); [`cuda/ffn_tile_ab_test.cu`](cuda/ffn_tile_ab_test.cu); [`tests/test_opt037_ffn_tiles.py`](tests/test_opt037_ffn_tiles.py); [`evidence/optimization/opt037-ffn-tiles/REPORT.md`](evidence/optimization/opt037-ffn-tiles/REPORT.md); [`evidence/optimization/opt037-ffn-tiles/REJECTION.md`](evidence/optimization/opt037-ffn-tiles/REJECTION.md); verification 2026-09-09T23:03:58Z |
| OPT-038 | Refresh post-ladder attribution and source gap map | OPT-032, OPT-034, OPT-035, OPT-036, OPT-037 | done | Retain fresh frozen-protocol P/D128/D2048 measurements, current subsystem breakdowns, independent raw host-wall accounting, pinned llama.cpp comparisons and matched-component experiment specifications, and a reproducible next-task order; `claims_performance_improvement: false`; accepted keep denominators remain unchanged | [`tasks/OPT-038.md`](tasks/OPT-038.md); [`pins/opt038_post_ladder_gap_contract.json`](pins/opt038_post_ladder_gap_contract.json); [`fixtures/opt038_post_ladder_gap.json`](fixtures/opt038_post_ladder_gap.json); [`cuda/opt038_prefill_attribution_test.cu`](cuda/opt038_prefill_attribution_test.cu); [`cuda/opt038_decode_attribution_test.cu`](cuda/opt038_decode_attribution_test.cu); [`tests/test_opt038_post_ladder_gap.py`](tests/test_opt038_post_ladder_gap.py); [`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](evidence/optimization/opt038-post-ladder-gap/REPORT.md); [`evidence/optimization/opt038-post-ladder-gap/COMPONENT-PROTOCOL.md`](evidence/optimization/opt038-post-ladder-gap/COMPONENT-PROTOCOL.md); verification 2026-09-10T08:17:30Z |
| OPT-039 | Warp-owned vector decode attention | OPT-038, OPT-036 | done | A/B warp-owned query-head attention against accepted 16-partition decode attention; keep only with frozen attention envelopes, exact candidate KV and state isolation, lower D2048 component time, improved D2048 versus the then-current keep oracle, P/D128/D2048 throughput floors of 95%, and both decode p95 measures at most 105%; otherwise reject | [`tasks/OPT-039.md`](tasks/OPT-039.md); [`pins/opt039_decode_warp_contract.json`](pins/opt039_decode_warp_contract.json); [`fixtures/opt039_decode_warp.json`](fixtures/opt039_decode_warp.json); [`cuda/opt039_decode_warp_ab_test.cu`](cuda/opt039_decode_warp_ab_test.cu); [`evidence/optimization/opt039-decode-warp/REPORT.md`](evidence/optimization/opt039-decode-warp/REPORT.md); verification 2026-09-10T09:13:29Z |
| OPT-040 | Hoist prompt GDN inverse normalization | OPT-038, OPT-019 | done | A/B shared per-token/key-head inverse norms against repeated warp-column normalization; keep only with byte-equal quality outputs/state, frozen sequential GDN gates, lower complete GDN component time, improved P versus the then-current keep oracle, D128/D2048 throughput at least 95%, and both decode p95 measures at most 105%; otherwise reject | [`tasks/OPT-040.md`](tasks/OPT-040.md); [`pins/opt040_gdn_shared_inverse_contract.json`](pins/opt040_gdn_shared_inverse_contract.json); [`fixtures/opt040_gdn_shared_inverse.json`](fixtures/opt040_gdn_shared_inverse.json); [`cuda/opt040_gdn_shared_inverse_ab_test.cu`](cuda/opt040_gdn_shared_inverse_ab_test.cu); [`evidence/optimization/opt040-gdn-shared-inverse/REPORT.md`](evidence/optimization/opt040-gdn-shared-inverse/REPORT.md); verification 2026-09-10T10:08:22Z |
| OPT-041 | Give prompt QK microtiles warp ownership | OPT-038, OPT-035 | done | A/B warp-owned prompt QK microtiles preserving existing virtual-warp partial order against shared cparts reduction; keep only with byte-equal quality attention outputs, frozen attention gates, lower complete component time, improved P versus the then-current keep oracle, D128/D2048 throughput at least 95%, and both decode p95 measures at most 105%; otherwise reject | [`tasks/OPT-041.md`](tasks/OPT-041.md); [`pins/opt041_fattn_warp_qk_contract.json`](pins/opt041_fattn_warp_qk_contract.json); [`fixtures/opt041_fattn_warp_qk.json`](fixtures/opt041_fattn_warp_qk.json); [`cuda/opt041_fattn_warp_qk_ab_test.cu`](cuda/opt041_fattn_warp_qk_ab_test.cu); [`evidence/optimization/opt041-fattn-warp-qk/REPORT.md`](evidence/optimization/opt041-fattn-warp-qk/REPORT.md); verification 2026-09-10T11:35:00Z |
| OPT-042 | Study integer decode MMV admissibility | OPT-038, OPT-034 | done | Retain paired FP32-packed versus Q4_K integer-dot diagnostic timings and complete numeric results using unchanged FP32-scale Q8 staging on synthetic and real FFN inputs, report fixed CUD-001 envelope eligibility and a promotion/rejection recommendation without changing production dispatch or keep denominators; `claims_performance_improvement: false` | [`tasks/OPT-042.md`](tasks/OPT-042.md); [`pins/opt042_mmv_integer_study_contract.json`](pins/opt042_mmv_integer_study_contract.json); [`fixtures/opt042_mmv_integer_study.json`](fixtures/opt042_mmv_integer_study.json); [`cuda/opt042_mmv_integer_study.cu`](cuda/opt042_mmv_integer_study.cu); [`tests/test_opt042_mmv_integer_study.py`](tests/test_opt042_mmv_integer_study.py); [`evidence/optimization/opt042-mmv-integer-study/REPORT.md`](evidence/optimization/opt042-mmv-integer-study/REPORT.md); verification 2026-09-10T12:09:20Z |
| OPT-043 | Measure post-042 kernel gaps against pinned llama.cpp | OPT-041, OPT-042 | done | Fresh unperturbed P/D controls, exclusive leaf accounting, actual layer captures, matched complete components and dispatch evidence cover every major family; no production change; `claims_performance_improvement: false`; accepted keep denominators remain historical | [`tasks/OPT-043.md`](tasks/OPT-043.md); [`pins/opt043_component_gap_contract.json`](pins/opt043_component_gap_contract.json); [`fixtures/opt043_component_gap.json`](fixtures/opt043_component_gap.json); [`cuda/opt043_prefill_attribution_test.cu`](cuda/opt043_prefill_attribution_test.cu); [`cuda/opt043_decode_attribution_test.cu`](cuda/opt043_decode_attribution_test.cu); [`cuda/opt043_activation_capture_test.cu`](cuda/opt043_activation_capture_test.cu); [`tests/test_opt043_component_gap.py`](tests/test_opt043_component_gap.py); [`evidence/optimization/opt043-component-gap/REPORT.md`](evidence/optimization/opt043-component-gap/REPORT.md); verification 2026-09-10T14:23:30Z |
| OPT-044 | Admit documented production arithmetic and quality budgets | OPT-043 | done | Freeze independent llama-calibrated primitive budgets and held-out model-quality gates before tuning; retain strict reference contracts and exact structural/session invariants; document user-authorized accuracy compromises; production selector remains strict; `claims_performance_improvement: false` | [`tasks/OPT-044.md`](tasks/OPT-044.md); [`pins/production_numerics_contract.json`](pins/production_numerics_contract.json); [`fixtures/opt044_production_numerics.json`](fixtures/opt044_production_numerics.json); [`tests/test_production_numerics.py`](tests/test_production_numerics.py); [`evidence/optimization/opt044-production-numerics/REPORT.md`](evidence/optimization/opt044-production-numerics/REPORT.md); verification 2026-09-10T16:25:35Z |
| OPT-045 | Parallelize RMSNorm and admitted fused arithmetic | OPT-044 | done | Cooperative residual/head normalization removes serial reductions with production quality, complete component and end-to-end wins, or retained measured rejection | [`tasks/OPT-045.md`](tasks/OPT-045.md); [`pins/opt045_parallel_norm_contract.json`](pins/opt045_parallel_norm_contract.json); [`fixtures/opt045_parallel_norm.json`](fixtures/opt045_parallel_norm.json); [`cuda/opt045_parallel_norm_ab_test.cu`](cuda/opt045_parallel_norm_ab_test.cu); [`cuda/rms_norm.cuh`](cuda/rms_norm.cuh); [`evidence/optimization/opt045-parallel-norm/REPORT.md`](evidence/optimization/opt045-parallel-norm/REPORT.md); verification 2026-09-10T17:35:00Z |
| OPT-046 | Implement cooperative packed Q4_K decode dots | OPT-044, OPT-045 | done | Packed integer dots, cooperative K reduction and admitted staging improve complete real FFN and D2048 under production quality and cross-workload guards, or retained rejection | [`tasks/OPT-046.md`](tasks/OPT-046.md); [`pins/opt046_q4_decode_contract.json`](pins/opt046_q4_decode_contract.json); [`fixtures/opt046_q4_decode.json`](fixtures/opt046_q4_decode.json); [`cuda/q4k_decode_dots.cu`](cuda/q4k_decode_dots.cu); [`cuda/q4k_decode_path.cuh`](cuda/q4k_decode_path.cuh); [`cuda/opt046_q4_decode_ab_test.cu`](cuda/opt046_q4_decode_ab_test.cu); [`tests/test_opt046_q4_decode.py`](tests/test_opt046_q4_decode.py); [`evidence/optimization/opt046-q4-decode/REPORT.md`](evidence/optimization/opt046-q4-decode/REPORT.md); verification 2026-09-10T19:01:30Z |
| OPT-047 | Accelerate Q8_0 mixer decode projections | OPT-044, OPT-046 | done | Shared activation staging and role-specific packed dots improve the complete mixer group and D2048; new activation approximation is documented and quality-tested, or retained rejection | [`tasks/OPT-047.md`](tasks/OPT-047.md); [`pins/opt047_q8_decode_contract.json`](pins/opt047_q8_decode_contract.json); [`fixtures/opt047_q8_decode.json`](fixtures/opt047_q8_decode.json); [`cuda/q8_decode_dots.cu`](cuda/q8_decode_dots.cu); [`cuda/q8_decode_path.cuh`](cuda/q8_decode_path.cuh); [`cuda/opt047_q8_decode_ab_test.cu`](cuda/opt047_q8_decode_ab_test.cu); [`tests/test_opt047_q8_decode.py`](tests/test_opt047_q8_decode.py); [`evidence/optimization/opt047-q8-decode/REPORT.md`](evidence/optimization/opt047-q8-decode/REPORT.md); verification 2026-09-10T19:42:58Z |
| OPT-048 | Accelerate full Q6_K vocabulary projection | OPT-044, OPT-046 | done | Full 248320-row logits use admitted packed integer dots with complete output quality and end-to-end decode win, or retained rejection; no vocabulary pruning | [`tasks/OPT-048.md`](tasks/OPT-048.md); [`pins/opt048_q6_logits_contract.json`](pins/opt048_q6_logits_contract.json); [`fixtures/opt048_q6_logits.json`](fixtures/opt048_q6_logits.json); [`cuda/q6k_decode_dots.cu`](cuda/q6k_decode_dots.cu); [`cuda/q6k_decode_path.cuh`](cuda/q6k_decode_path.cuh); [`cuda/opt048_q6_logits_ab_test.cu`](cuda/opt048_q6_logits_ab_test.cu); [`tests/test_opt048_q6_logits.py`](tests/test_opt048_q6_logits.py); [`evidence/optimization/opt048-q6-logits/REPORT.md`](evidence/optimization/opt048-q6-logits/REPORT.md); verification 2026-09-10T20:27:00Z |
| OPT-049 | Share decode FFN staging and fuse gate/up | OPT-046, OPT-047, OPT-048 | done | Complete FFN including staging/SwiGLU/down wins with admitted BF16 rounding and graph/eager equivalence, production quality and P/D guards, or retained rejection | [`tasks/OPT-049.md`](tasks/OPT-049.md); [`pins/opt049_decode_ffn_fusion_contract.json`](pins/opt049_decode_ffn_fusion_contract.json); [`fixtures/opt049_decode_ffn_fusion.json`](fixtures/opt049_decode_ffn_fusion.json); [`cuda/ffn_decode_path.cuh`](cuda/ffn_decode_path.cuh); [`cuda/opt049_decode_ffn_fusion_ab_test.cu`](cuda/opt049_decode_ffn_fusion_ab_test.cu); [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`tests/test_opt049_decode_ffn_fusion.py`](tests/test_opt049_decode_ffn_fusion.py); [`evidence/optimization/opt049-decode-ffn-fusion/REPORT.md`](evidence/optimization/opt049-decode-ffn-fusion/REPORT.md); verification 2026-09-10T21:05:17Z |
| OPT-050 | Prepare prompt Q normalization and RoPE once | OPT-044, OPT-045 | done | Prepared queries are reused across attention KV partitions; preparation-inclusive P improves with tail/prefix correctness, quality and memory guards, or retained rejection | [`tasks/OPT-050.md`](tasks/OPT-050.md); [`pins/opt050_attention_query_prepare_contract.json`](pins/opt050_attention_query_prepare_contract.json); [`fixtures/opt050_attention_query_prepare.json`](fixtures/opt050_attention_query_prepare.json); [`cuda/attention_decode.cu`](cuda/attention_decode.cu); [`cuda/fattn_mma_f16.cuh`](cuda/fattn_mma_f16.cuh); [`cuda/opt050_attention_query_prepare_ab_test.cu`](cuda/opt050_attention_query_prepare_ab_test.cu); [`tests/test_opt050_attention_query_prepare.py`](tests/test_opt050_attention_query_prepare.py); [`evidence/optimization/opt050-attention-query-prepare/REPORT.md`](evidence/optimization/opt050-attention-query-prepare/REPORT.md); verification 2026-09-10T21:40:39Z |
| OPT-051 | Pipeline prompt attention with register softmax | OPT-044, OPT-050 | done | Staged admission of F16 operands, register reductions and tiled asynchronous loads reduces complete attention and P under documented quality and P/D guards, or retained rejection | [`tasks/OPT-051.md`](tasks/OPT-051.md); [`pins/opt051_attention_pipeline_contract.json`](pins/opt051_attention_pipeline_contract.json); [`fixtures/opt051_attention_pipeline.json`](fixtures/opt051_attention_pipeline.json); [`cuda/fattn_mma_f16_pipeline.cuh`](cuda/fattn_mma_f16_pipeline.cuh); [`cuda/fattn_mma_f16.cuh`](cuda/fattn_mma_f16.cuh); [`cuda/opt051_attention_pipeline_ab_test.cu`](cuda/opt051_attention_pipeline_ab_test.cu); [`tests/test_opt051_attention_pipeline.py`](tests/test_opt051_attention_pipeline.py); [`evidence/optimization/opt051-attention-pipeline/REPORT.md`](evidence/optimization/opt051-attention-pipeline/REPORT.md); verification 2026-09-10T22:50:00Z |
| OPT-052 | Remove redundant GDN arithmetic and state traffic | OPT-044, OPT-045 | done | Hoisted scaled Q/K and decay plus admitted FMA improve complete GDN and P; optional conversion-inclusive state tiling wins where measured; recurrent quality and atomic state stay valid, or retained rejection | [`tasks/OPT-052.md`](tasks/OPT-052.md); [`pins/opt052_gdn_arithmetic_contract.json`](pins/opt052_gdn_arithmetic_contract.json); [`fixtures/opt052_gdn_arithmetic.json`](fixtures/opt052_gdn_arithmetic.json); [`cuda/gdn_fused_quality.cuh`](cuda/gdn_fused_quality.cuh); [`cuda/opt052_gdn_arithmetic_ab_test.cu`](cuda/opt052_gdn_arithmetic_ab_test.cu); [`tests/test_opt052_gdn_arithmetic.py`](tests/test_opt052_gdn_arithmetic.py); [`evidence/optimization/opt052-gdn-arithmetic/REPORT.md`](evidence/optimization/opt052-gdn-arithmetic/REPORT.md); verification 2026-09-10T23:30:00Z |
| OPT-053 | Optimize MMQ scaling and tile staging | OPT-044, OPT-045 | done | Existing quality MMA gains measured scaling/staging efficiency and full P speed under quality and memory guards, or a measured no-change result; no unchanged rejected tile/stream-K rerun | [`tasks/OPT-053.md`](tasks/OPT-053.md); [`pins/opt053_mmq_pipeline_contract.json`](pins/opt053_mmq_pipeline_contract.json); [`fixtures/opt053_mmq_pipeline.json`](fixtures/opt053_mmq_pipeline.json); [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh); [`cuda/opt053_mmq_pipeline_ab_test.cu`](cuda/opt053_mmq_pipeline_ab_test.cu); [`tests/test_opt053_mmq_pipeline.py`](tests/test_opt053_mmq_pipeline.py); [`evidence/optimization/opt053-mmq-pipeline/REPORT.md`](evidence/optimization/opt053-mmq-pipeline/REPORT.md); verification 2026-09-11T00:05:20Z |
| OPT-054 | Tune internal prefill batches within atomic 4K | OPT-049, OPT-051, OPT-052, OPT-053 | done | Compare 512/1024/2048/4096 physical batches while preserving one atomic 4096-token operation; keep only a complete P win with quality, cancellation, graph and memory checks | [`tasks/OPT-054.md`](tasks/OPT-054.md); [`pins/opt054_prefill_microbatch_contract.json`](pins/opt054_prefill_microbatch_contract.json); [`fixtures/opt054_prefill_microbatch.json`](fixtures/opt054_prefill_microbatch.json); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/opt054_prefill_microbatch_ab_test.cu`](cuda/opt054_prefill_microbatch_ab_test.cu); [`tests/test_opt054_prefill_microbatch.py`](tests/test_opt054_prefill_microbatch.py); [`evidence/optimization/opt054-prefill-microbatch/REPORT.md`](evidence/optimization/opt054-prefill-microbatch/REPORT.md); verification 2026-09-11T01:03:00Z |
| OPT-055 | Capture measured remaining decode and prompt launch gaps | OPT-054 | done | Supersedes OPT-031; broader stable graphs save measured wall time with bounded cancellation, correct dynamic positions and post-graph memory reserve, or measured no-change result | [`tasks/OPT-055.md`](tasks/OPT-055.md); [`pins/opt055_execution_graphs_contract.json`](pins/opt055_execution_graphs_contract.json); [`fixtures/opt055_execution_graphs.json`](fixtures/opt055_execution_graphs.json); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/opt055_execution_graphs_ab_test.cu`](cuda/opt055_execution_graphs_ab_test.cu); [`tests/test_opt055_execution_graphs.py`](tests/test_opt055_execution_graphs.py); [`evidence/optimization/opt055-execution-graphs/REPORT.md`](evidence/optimization/opt055-execution-graphs/REPORT.md); verification 2026-09-11T01:30:00Z |
| OPT-056 | Exceed pinned llama.cpp prefill and decode with quality | OPT-045, OPT-046, OPT-047, OPT-048, OPT-049, OPT-050, OPT-051, OPT-052, OPT-053, OPT-054, OPT-055 | blocked | Same-sitting P/D128/D2048 throughput exceeds llama by at least 5%, decode p95 is no worse, combined production quality passes, and original OPT-016 2K parity passes; candidate-task completion alone is insufficient | [`tasks/OPT-056.md`](tasks/OPT-056.md); [`pins/opt056_performance_gate_contract.json`](pins/opt056_performance_gate_contract.json); [`fixtures/opt056_performance_gate.json`](fixtures/opt056_performance_gate.json); [`tests/test_opt056_performance_gate.py`](tests/test_opt056_performance_gate.py); [`evidence/optimization/opt056-performance-gate/REPORT.md`](evidence/optimization/opt056-performance-gate/REPORT.md); blocked 2026-09-11T02:15:00Z measured gate unpassed P 2808.50 vs llama 3263.52 (+618 tok/s to 5% bar), D128 37.48 vs 68.93, D2048 35.72 vs 67.34, decode p95 worse; recovery: close remaining P/D gaps vs llama before re-pass |

### Post-056 implementation batch

All rows below are proposed implementation work, not delivered speedups. Use
the [source analysis and batch protocol](tasks/PERFORMANCE-RECOVERY-2026-09-11.md)
and [testing strategy](testing-strategy.md). The first eligible task is OPT-061.
Existing OPT-056 and OPT-016 gates remain blocked on their original conditions.

| ID | Description | Dependencies | Status | Acceptance condition | Evidence |
|---|---|---|---|---|---|
| OPT-057 | Bound optimization feedback and fix incremental CUDA builds | OPT-055 | done | Reliable header/flag invalidation, explicit tier/workload limits, isolated evidence and measured warm feedback within 300 seconds; no throughput claim | [`tasks/OPT-057.md`](tasks/OPT-057.md); [`pins/opt057_iteration_contract.json`](pins/opt057_iteration_contract.json); [`fixtures/opt057_iteration_loop.json`](fixtures/opt057_iteration_loop.json); [`tools/run_optimization_task.py`](tools/run_optimization_task.py); [`cuda/optimization_engine_probe.cu`](cuda/optimization_engine_probe.cu); [`tests/test_optimization_loop.py`](tests/test_optimization_loop.py); [`evidence/optimization/opt057-iteration-loop/REPORT.md`](evidence/optimization/opt057-iteration-loop/REPORT.md); verification 2026-09-11T05:26:00Z |
| OPT-058 | Establish finite scheduler and valid functional/held-out quality baselines | OPT-057 | done | Current eager/graph/trace nonfinites resolved or honestly blocked; both engines evaluated on valid functional prompts; missing held-out llama reference frozen and required | [`tasks/OPT-058.md`](tasks/OPT-058.md); [`pins/opt058_quality_baseline_contract.json`](pins/opt058_quality_baseline_contract.json); [`pins/opt058_iteration_contract.json`](pins/opt058_iteration_contract.json); [`pins/production_quality_v2_inputs.json`](pins/production_quality_v2_inputs.json); [`pins/production_quality_v2_llama_reference.json`](pins/production_quality_v2_llama_reference.json); [`fixtures/opt058_quality_baseline.json`](fixtures/opt058_quality_baseline.json); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/opt058_quality_baseline_test.cu`](cuda/opt058_quality_baseline_test.cu); [`tests/test_opt058_quality_baseline.py`](tests/test_opt058_quality_baseline.py); [`evidence/optimization/opt058-quality-baseline/REPORT.md`](evidence/optimization/opt058-quality-baseline/REPORT.md); verification 2026-09-11T06:04:00Z |
| OPT-059 | Calibrate GPU numerical error and wire per-family production admission | OPT-058 | done | Actual pinned llama GPU and independent FP64 calibration, held-out v2 budgets, staging semantics and strict/current test separation; no unvalidated default | [`tasks/OPT-059.md`](tasks/OPT-059.md); [`pins/production_numerics_v2_contract.json`](pins/production_numerics_v2_contract.json); [`pins/opt059_admission_manifest.json`](pins/opt059_admission_manifest.json); [`pins/opt059_iteration_contract.json`](pins/opt059_iteration_contract.json); [`fixtures/opt059_gpu_numerics.json`](fixtures/opt059_gpu_numerics.json); [`cuda/opt059_numerics_test.cu`](cuda/opt059_numerics_test.cu); [`tools/llama_authority/projection_export.cpp`](tools/llama_authority/projection_export.cpp); [`tools/production_numerics_v2.py`](tools/production_numerics_v2.py); [`tests/test_opt059_numerics.py`](tests/test_opt059_numerics.py); [`evidence/optimization/opt059-gpu-numerics/REPORT.md`](evidence/optimization/opt059-gpu-numerics/REPORT.md); verification 2026-09-11T06:37:10Z |
| OPT-060 | Instrument matched full-engine Quartz and pinned llama family execution | OPT-057 | done | Real dispatch/fusion/stream records and matched P/D family attribution with overhead/overlap limits; reproducible private authority patch; no speedup required | [`tasks/OPT-060.md`](tasks/OPT-060.md); [`pins/opt060_engine_attribution_contract.json`](pins/opt060_engine_attribution_contract.json); [`pins/opt060_iteration_contract.json`](pins/opt060_iteration_contract.json); [`fixtures/opt060_engine_attribution.json`](fixtures/opt060_engine_attribution.json); [`cuda/engine_attribution.h`](cuda/engine_attribution.h); [`cuda/opt060_engine_attribution_test.cu`](cuda/opt060_engine_attribution_test.cu); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`tools/llama_authority/patches/opt060-engine-attribution.patch`](tools/llama_authority/patches/opt060-engine-attribution.patch); [`tools/llama_authority/build_opt060_instrumented.sh`](tools/llama_authority/build_opt060_instrumented.sh); [`tools/llama_authority/engine_attribution.cpp`](tools/llama_authority/engine_attribution.cpp); [`tools/opt060_engine_attribution.py`](tools/opt060_engine_attribution.py); [`tests/test_opt060_engine_attribution.py`](tests/test_opt060_engine_attribution.py); [`evidence/optimization/opt060-engine-attribution/REPORT.md`](evidence/optimization/opt060-engine-attribution/REPORT.md); verification 2026-09-11T07:55:00Z |
| OPT-061 | Build real-input streaming replay and hardware bottleneck evidence | OPT-060 | done | Reusable captured FFN/mixer inputs, hot versus rotating weights, correct call counts and complete costs; counters or explicitly limited event/resource fallback | [`tasks/OPT-061.md`](tasks/OPT-061.md); [`pins/opt061_component_replay_contract.json`](pins/opt061_component_replay_contract.json); [`pins/opt061_iteration_contract.json`](pins/opt061_iteration_contract.json); [`fixtures/opt061_component_replay.json`](fixtures/opt061_component_replay.json); [`cuda/optimization_component_replay.h`](cuda/optimization_component_replay.h); [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`tools/opt061_component_replay.py`](tools/opt061_component_replay.py); [`tests/test_opt061_component_replay.py`](tests/test_opt061_component_replay.py); [`evidence/optimization/opt061-component-replay/REPORT.md`](evidence/optimization/opt061-component-replay/REPORT.md); verification 2026-09-11T08:21:30Z |
| OPT-062 | Revalidate cooperative Q4 and dispatch all decode FFN projections | OPT-059, OPT-061 | done | Gate/up/down actually use admitted integer variants with shared staging, graph coverage, complete FFN benefit and batch quality/non-regression, or retained rejection | [`tasks/OPT-062.md`](tasks/OPT-062.md); [`pins/opt062_q4_admission_contract.json`](pins/opt062_q4_admission_contract.json); [`pins/opt062_iteration_contract.json`](pins/opt062_iteration_contract.json); [`fixtures/opt062_q4_admission.json`](fixtures/opt062_q4_admission.json); [`cuda/opt062_q4_admission_test.cu`](cuda/opt062_q4_admission_test.cu); [`cuda/q4k_decode_dots.cu`](cuda/q4k_decode_dots.cu); [`cuda/q4k_decode_path.cuh`](cuda/q4k_decode_path.cuh); [`cuda/ffn_decode_path.cuh`](cuda/ffn_decode_path.cuh); [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`tests/test_opt062_q4_admission.py`](tests/test_opt062_q4_admission.py); [`evidence/optimization/opt062-q4-admission/REPORT.md`](evidence/optimization/opt062-q4-admission/REPORT.md); verification 2026-09-11T08:49:10Z |
| OPT-063 | Fuse admitted integer gate/up dots and SwiGLU | OPT-062 | done | Shared-input paired Q4 improves complete FFN while preserving admitted arithmetic and output boundaries, or documented no-change/no-go | [`tasks/OPT-063.md`](tasks/OPT-063.md); [`pins/opt063_integer_ffn_contract.json`](pins/opt063_integer_ffn_contract.json); [`pins/opt063_iteration_contract.json`](pins/opt063_iteration_contract.json); [`fixtures/opt063_integer_ffn.json`](fixtures/opt063_integer_ffn.json); [`cuda/opt063_integer_ffn_test.cu`](cuda/opt063_integer_ffn_test.cu); [`cuda/q4k_decode_dots.cu`](cuda/q4k_decode_dots.cu); [`cuda/q4k_decode_dots.cuh`](cuda/q4k_decode_dots.cuh); [`cuda/q4k_decode_path.cuh`](cuda/q4k_decode_path.cuh); [`cuda/ffn_decode_path.cuh`](cuda/ffn_decode_path.cuh); [`cuda/quant_mmv.h`](cuda/quant_mmv.h); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu); [`tests/test_opt063_integer_ffn.py`](tests/test_opt063_integer_ffn.py); [`evidence/optimization/opt063-integer-ffn/REPORT.md`](evidence/optimization/opt063-integer-ffn/REPORT.md); verification 2026-09-11T09:10:28Z |
| OPT-064 | Improve Q8 decode row grouping and reduction cost | OPT-059, OPT-061 | done | Bounded four-layout study improves frequency-weighted rotating mixer time with legal alignment, staging reuse and quality, or measured rejection | [`tasks/OPT-064.md`](tasks/OPT-064.md); [`pins/opt064_q8_rows_contract.json`](pins/opt064_q8_rows_contract.json); [`pins/opt064_iteration_contract.json`](pins/opt064_iteration_contract.json); [`fixtures/opt064_q8_rows.json`](fixtures/opt064_q8_rows.json); [`cuda/opt064_q8_rows_test.cu`](cuda/opt064_q8_rows_test.cu); [`cuda/q8_decode_dots.cu`](cuda/q8_decode_dots.cu); [`cuda/q8_decode_dots.cuh`](cuda/q8_decode_dots.cuh); [`cuda/q8_decode_path.cuh`](cuda/q8_decode_path.cuh); [`cuda/quant_mmv.h`](cuda/quant_mmv.h); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu); [`tests/test_opt064_q8_rows.py`](tests/test_opt064_q8_rows.py); [`evidence/optimization/opt064-q8-rows/REPORT.md`](evidence/optimization/opt064-q8-rows/REPORT.md); verification 2026-09-11T09:47:34Z |
| OPT-065 | Retune MMQ tile resources with the current pipeline active | OPT-059, OPT-061 | done | Four Q4 tiles compared with FMA/async Y preserved, actual dispatch/resource evidence, complete FFN/P benefit and tail correctness, or retained 128x128 | [`tasks/OPT-065.md`](tasks/OPT-065.md); [`pins/opt065_mmq_tiles_contract.json`](pins/opt065_mmq_tiles_contract.json); [`pins/opt065_iteration_contract.json`](pins/opt065_iteration_contract.json); [`fixtures/opt065_mmq_tiles.json`](fixtures/opt065_mmq_tiles.json); [`cuda/opt065_mmq_tiles_test.cu`](cuda/opt065_mmq_tiles_test.cu); [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh); [`cuda/quant_mmv.h`](cuda/quant_mmv.h); [`tests/test_opt065_mmq_tiles.py`](tests/test_opt065_mmq_tiles.py); [`evidence/optimization/opt065-mmq-tiles/REPORT.md`](evidence/optimization/opt065-mmq-tiles/REPORT.md); verification 2026-09-11T10:36:00Z |
| OPT-066 | Pipeline packed Q4 prompt weight fetches | OPT-065 | done | Bounded X-prefetch variant preserves stage lifetimes and arithmetic and improves complete P work, or measured resource/performance no-go | [`tasks/OPT-066.md`](tasks/OPT-066.md); [`pins/opt066_mmq_x_pipeline_contract.json`](pins/opt066_mmq_x_pipeline_contract.json); [`pins/opt066_iteration_contract.json`](pins/opt066_iteration_contract.json); [`fixtures/opt066_mmq_x_pipeline.json`](fixtures/opt066_mmq_x_pipeline.json); [`cuda/opt066_mmq_x_pipeline_test.cu`](cuda/opt066_mmq_x_pipeline_test.cu); [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh); [`cuda/quant_mmv.h`](cuda/quant_mmv.h); [`tests/test_opt066_mmq_x_pipeline.py`](tests/test_opt066_mmq_x_pipeline.py); [`evidence/optimization/opt066-mmq-x-pipeline/REPORT.md`](evidence/optimization/opt066-mmq-x-pipeline/REPORT.md); verification 2026-09-11T10:57:28Z |
| OPT-067 | Pair prompt FFN gate/up tiles and BF16 SwiGLU output | OPT-065 | done | Conditional two-candidate study proves complete FFN benefit, resource fit and output/staging quality, or explicit no-go/rejection | [`tasks/OPT-067.md`](tasks/OPT-067.md); [`pins/opt067_prompt_pair_contract.json`](pins/opt067_prompt_pair_contract.json); [`pins/opt067_iteration_contract.json`](pins/opt067_iteration_contract.json); [`fixtures/opt067_prompt_pair.json`](fixtures/opt067_prompt_pair.json); [`cuda/opt067_prompt_pair_test.cu`](cuda/opt067_prompt_pair_test.cu); [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh); [`cuda/quant_mmv.h`](cuda/quant_mmv.h); [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu); [`tests/test_opt067_prompt_pair.py`](tests/test_opt067_prompt_pair.py); [`evidence/optimization/opt067-prompt-pair/REPORT.md`](evidence/optimization/opt067-prompt-pair/REPORT.md); verification 2026-09-11T11:23:26Z |
| OPT-068 | Measure scoped optimized compilation for one projection family | OPT-059, OPT-061 | pending | Three isolated O2/O3/FMA builds yield a quality-admitted complete win with strict/host flags intact, or retained flags | [Implementation guide](tasks/OPT-068.md); proposed 2026-09-11 |
| OPT-069 | Validate combined batch and original llama outcome gates | OPT-058, OPT-059, OPT-062, OPT-063, OPT-064, OPT-065, OPT-066, OPT-067, OPT-068 | pending | Complete combined quality, original P/D/2K protocols and state/memory evidence; separately report improvement, parity and original +5% outcome without relabeling failed gates | [Implementation guide](tasks/OPT-069.md); proposed 2026-09-11 |

### Post-042 recovery execution order (historical batch)

The [2026-09-10 design](tasks/PERFORMANCE-RECOVERY-2026-09-10.md) compares the
admitted Quartz, pinned llama.cpp and ds4 paths. It is source analysis and task
design, not new performance evidence. **OPT-056 outcome gate measured unpassed**
(2026-09-11); recovery requires closing remaining P/D throughput and decode-p95
gaps versus llama before re-pass. The 2026-09-10 ladder is exhausted; the new
OPT-057–069 batch above supplies the current recovery work.
Dependencies permit independent work but do not authorize subagents. The user
accepts documented llama.cpp/ds4-like accuracy compromises; strict reference
arithmetic and exact structural/transaction guarantees remain separately tested.
Old pending OPT-031 is superseded, not delivered. OPT-016 remains blocked until
its original 2K acceptance actually passes. Prior chronological “next task”
statements below are historical, not the current execution order.

### 2026-09-04T13:09:32Z — OPT-005 delivered

- Fresh recovery attempt 1 independently passed: focused contract/validator,
  pinned native diagnostic, focused integration, ordinary pytest (`170 passed,
  20 skipped`), and CUDA-enabled pytest (`190 passed`) all passed.
- Acceptance evidence is the regenerated fixture and matching documentation;
  the component-only timing boundary and remaining end-to-end risk are recorded
  in [`tasks/OPT-005.md`](tasks/OPT-005.md).
- Marked OPT-005 `done`; delivery is limited to the already-verified recovery
  allowlist plus workflow bookkeeping. `plan.md` and semantic implementation
  evidence are unchanged by delivery.
- Final checks: `uv run pytest -q tests/test_documentation.py` and
  `git diff --check` passed. Commit and push follow this bookkeeping entry.
| CLI-001 | Implement interactive `qw38` text CLI | TOK-002, SES-003 | done | Interactive generation, reasoning, stops, sampling, and persistence pass smoke tests | [`src/cli.cpp`](src/cli.cpp); [`src/engine.cpp`](src/engine.cpp); [`fixtures/cli_smoke.json`](fixtures/cli_smoke.json); [`tests/test_cli.py`](tests/test_cli.py); log 2026-09-01T11:56:52Z |
| SRV-001 | Implement single-flight HTTP server core and queue | API-001 | done | Health/models endpoints, cancellation, queue timing, and one GPU session pass tests | [`src/server.cpp`](src/server.cpp); [`src/server_core.cpp`](src/server_core.cpp); [`fixtures/server_core.json`](fixtures/server_core.json); [`tests/test_server.py`](tests/test_server.py); log 2026-09-01T18:19:42Z |
| SRV-002 | Implement Chat Completions API | TOK-002, SES-002, SRV-001 | done | Supported roles/tools/streaming/sampling/stops pass; exclusions reject explicitly | [`src/server.cpp`](src/server.cpp); [`src/server_generation.cpp`](src/server_generation.cpp); [`fixtures/chat_completions.json`](fixtures/chat_completions.json); [`tests/test_server.py`](tests/test_server.py); log 2026-09-01T19:00:04Z |
| SRV-003 | Implement Responses API and continuation | TOK-002, SES-003, SRV-001 | done | Streaming/tools/`previous_response_id` and exclusions pass API fixtures | [`src/server.cpp`](src/server.cpp); [`fixtures/responses.json`](fixtures/responses.json); [`tests/test_server.py`](tests/test_server.py); log 2026-09-02T14:27:35Z |
| API-003 | Implement strict Responses request and event mapping | API-002, SRV-002 | done | Text inputs, reasoning, function calls/results, sampling, response objects, ordered SSE events, and explicit exclusions pass native fixtures | [`src/responses_api.cpp`](src/responses_api.cpp); [`src/responses_api_test.cpp`](src/responses_api_test.cpp); [`pins/responses_contract.json`](pins/responses_contract.json); log 2026-09-02T14:27:35Z |
| BEN-001 | Implement `qw38-bench` component/end-to-end harness | OPT-001 | done | Warmups/samples, telemetry, raw samples, failures, and environment metadata are retained | `pins/benchmark_contract.json`; `fixtures/benchmark_harness.json`; `evidence/benchmark/`; `tests/test_benchmark.py`; log 2026-09-02T15:24:00Z |
| BEN-002 | Preserve the product-wide no-argument usage exit contract in `qw38-bench` | BEN-001, BLD-001 | done | Invoking the benchmark with no arguments prints usage and returns exit code 2 without creating output | `tests/test_build.py`; log 2026-09-02T15:24:00Z |
| EVAL-001 | Implement `qw38-eval` logits/traces/checkpoints harness | ORA-001, SES-003 | done | Focused native diagnostics are driven by typed pytest helpers | [`tasks/EVAL-001.md`](tasks/EVAL-001.md); reopened 2026-09-03T14:00:00Z after build/hash repair; recovery readmitted 2026-09-03T15:44:53Z; completed 2026-09-03T17:01:54Z |
| EVAL-002 | Tier and instrument task-specific CUDA oracles | EVAL-001, BEN-001 | pending | Every GPU oracle requires the shared task-testing strategy, fails closed when the tier is omitted, reuses builds/binaries where safe, emits phase timestamps without weakening final evidence, and bounds workload complexity | [`tasks/EVAL-002.md`](tasks/EVAL-002.md); [`tasks/TASK-TESTING-STRATEGY.md`](tasks/TASK-TESTING-STRATEGY.md) |
| EVAL-003 | Make OPT-046 Q4 decode evidence representative and sampled | OPT-046, EVAL-002 | pending | Large host references use deterministic documented sampling with decoded-row reuse, production captures supplement probes, final survivor A/B and conditional P/D gates remain valid, redundant oracle compilation is removed, and the shared task-testing strategy is honored | [`tasks/EVAL-003.md`](tasks/EVAL-003.md); [`tasks/TASK-TESTING-STRATEGY.md`](tasks/TASK-TESTING-STRATEGY.md) |
| QLT-001 | Pass held-out NLL, continuation, recurrence, retrieval, and task quality | EVAL-001, MEM-001, OPT-012, OPT-013, OPT-016 | blocked | Admitted artifact passes every documented threshold and 128K retrieval fixture | [`tasks/QLT-001.md`](tasks/QLT-001.md); blocked 2026-09-08T10:22:44Z pending 2K llama.cpp prefill parity (`OPT-016`); OPT-016 recovery via `OPT-017`–`OPT-019` then gate re-pass; prior scaling report [`evidence/quality/scaling-2026-09-08/REPORT.md`](evidence/quality/scaling-2026-09-08/REPORT.md) |
| CMP-001 | Pin and validate comparable baseline artifacts | PIN-001, PIN-002, QLT-001 | pending | llama/Ollama share GGUF; vLLM difference and <=1% NLL admission are explicit | — |
| CMP-002 | Run controlled 30-sample comparative matrix | BEN-001, OPT-004, CMP-001 | pending | All contexts/metrics/environment data and negative runs are retained | — |
| CMP-003 | Pass prefill/decode statistical speed gates | CMP-002 | pending | Paired bootstrap lower bounds exceed 1.05 and no workload is >5% slower | — |
| DOC-001 | Maintain code-linked handbook and provenance ledger | BLD-001 | done | Each implemented concept has claim labels, invariants, failures, task IDs, and evidence | [`docs/65-documentation-audit.md`](docs/65-documentation-audit.md); [`tests/test_documentation.py`](tests/test_documentation.py); log 2026-09-03T13:21:51Z |
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
| EDU-015 | Explain complete decoder-layer composition for beginners | CPU-014, DOC-001 | done | Mixer→residual→post-mixer norm→SwiGLU→residual order, layer variants, buffer reuse, preflight validation, state mutation, and scheduler boundary are code-linked and worked | [`docs/29-complete-decoder-layer.md`](docs/29-complete-decoder-layer.md); [`tests/test_real_layer_composition.py`](tests/test_real_layer_composition.py); log 2026-08-30T06:31:18Z |
| EDU-016 | Explain embeddings, final normalization, logits, and token choice for beginners | CPU-015, DOC-001 | done | Token IDs versus embeddings, row lookup, hidden vectors, final RMSNorm, vocabulary projection, logits versus probabilities, argmax, workspace/cost, exact bounds, and evidence limits are code-linked and worked | [`docs/30-embeddings-and-logits.md`](docs/30-embeddings-and-logits.md); [`tests/test_real_model_boundaries.py`](tests/test_real_model_boundaries.py); log 2026-08-30T06:39:56Z |
| EDU-017 | Explain the full hybrid layer schedule and scalar runtime ownership for beginners | CPU-016, DOC-001 | done | Layer order, slot mapping, prepared parameters, per-session state, shared scratch, ping-pong residuals, one-token execution, final logits, state frontier, structural fixtures, and oracle limits are code-linked and worked | [`docs/31-full-scalar-token.md`](docs/31-full-scalar-token.md); [`tests/test_real_scalar_token.py`](tests/test_real_scalar_token.py); log 2026-08-30T06:53:51Z |
| EDU-018 | Explain trace bundles and numeric comparison metrics for beginners | TRC-001, DOC-001 | done | Taps, manifests, little-endian blobs, shapes, checksums, frontiers, absolute/relative/RMS/cosine errors, non-finite values, first failures, top logits, and evidence limits are code-linked and worked | [`docs/32-trace-bundles-and-metrics.md`](docs/32-trace-bundles-and-metrics.md); [`tests/test_trace.py`](tests/test_trace.py); log 2026-08-30T07:13:12Z |
| EDU-019 | Explain diagnostic build isolation and stable runtime taps for beginners | TRC-003, DOC-001 | done | Compile-time isolation, filters, stable tap names/shapes, capture timing, backend-neutral sinks, cost, and oracle limits are code-linked and worked | [`docs/33-diagnostic-trace-isolation.md`](docs/33-diagnostic-trace-isolation.md); [`tests/test_diagnostic_trace.py`](tests/test_diagnostic_trace.py); log 2026-08-30T07:19:17Z |
| EDU-020 | Explain real scalar tap timing and v1 bundle capture for beginners | TRC-002, DOC-001 | done | Each real tap's semantic timing, shape, state scope, filter/copy behavior, bundle path, evidence, and authority limit are code-linked and worked | [`docs/34-real-scalar-traces.md`](docs/34-real-scalar-traces.md); [`tests/test_real_scalar_trace.py`](tests/test_real_scalar_trace.py); log 2026-08-30T07:32:39Z |
| EDU-021 | Explain multi-token scalar chunks and exact equivalence for beginners | CPU-004, DOC-001 | done | Chunk preflight, token/position order, logits layout, repeated-token equivalence, state/frontier equality, failure behavior, cost, and oracle limits are code-linked and worked | [`docs/35-scalar-token-chunks.md`](docs/35-scalar-token-chunks.md); [`tests/test_real_scalar_chunk.py`](tests/test_real_scalar_chunk.py); log 2026-08-30T11:57:06Z |
| EDU-022 | Explain independent authority hierarchy and same-GGUF llama comparison for beginners | ORA-002, DOC-001 | done | Primary versus independent versus native authority, artifact/template/token identity, build pins, logits/continuation limits, and failure evidence are code-linked and worked | [`docs/36-independent-llama-authority.md`](docs/36-independent-llama-authority.md); [`tests/test_llama_authority.py`](tests/test_llama_authority.py); log 2026-08-30T12:27:27Z |
| EDU-023 | Explain official-checkpoint Transformers eager/offload authority for beginners | ORA-003, DOC-001 | done | Original checkpoint versus GGUF roles, Safetensors shards, eager execution, CPU/GPU/disk offload, hooks/taps, memory limits, and semantic proof boundaries are code-linked and worked | [`docs/37-transformers-authority.md`](docs/37-transformers-authority.md); [`tests/test_transformers_authority.py`](tests/test_transformers_authority.py); log 2026-08-30T17:38:10Z |
| EDU-024 | Explain three-authority tap alignment and tolerance freezing for beginners | ORA-004, DOC-001 | done | Comparable versus runtime-private boundaries, layout normalization, error distributions, tolerance selection, near-ties, and immutable admission are code-linked and worked | [`docs/38-scalar-authority-tolerances.md`](docs/38-scalar-authority-tolerances.md); [`tests/test_scalar_authority_alignment.py`](tests/test_scalar_authority_alignment.py); log 2026-08-30T18:16:04Z |
| EDU-025 | Explain CUDA decode MMV and transient activation quantization for beginners | CUD-001, DOC-001 | done | Thread/warp ownership, BF16-to-Q8 staging, packed-weight decoding, FP32 reduction, launch validation, numeric gates, timing, and proof limits are code-linked and worked | [`docs/39-cuda-quant-mmv.md`](docs/39-cuda-quant-mmv.md); [`tests/test_cuda_quant_mmv.py`](tests/test_cuda_quant_mmv.py); log 2026-08-31T06:05:47Z |
| EDU-026 | Explain tiled CUDA prompt MMQ for beginners | CUD-002, DOC-001 | done | Prompt rows, output layout, two-dimensional tiles, weight reuse, tail handling, scalar equivalence, numeric gates, timing, and proof limits are code-linked and worked | [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md); [`tests/test_cuda_quant_mmv.py`](tests/test_cuda_quant_mmv.py); log 2026-08-31T06:16:39Z |
| EDU-027 | Explain one-token CUDA GDN staging and atomic commit for beginners | GDN-001, DOC-001 | done | Convolution candidate state, normalized head reuse, recurrent mutation order, prepare/commit ownership, failure injection, numeric gates, timing, and proof limits are code-linked and worked | [`docs/41-cuda-gdn-step.md`](docs/41-cuda-gdn-step.md); [`tests/test_cuda_gdn.py`](tests/test_cuda_gdn.py); log 2026-08-31T06:29:12Z |
| EDU-028 | Explain chunked CUDA GDN prefill and 64-token windows for beginners | GDN-002, DOC-001 | done | Prefill chunks, strict recurrence order, internal windows, candidate continuity, output layout, chunk-vs-token equivalence, cancellation, timing, and proof limits are code-linked and worked | [`docs/42-cuda-gdn-chunks.md`](docs/42-cuda-gdn-chunks.md); [`tests/test_cuda_gdn_chunk.py`](tests/test_cuda_gdn_chunk.py); log 2026-08-31T06:42:29Z |
| EDU-029 | Explain CUDA grouped-query decode attention and partial RoPE for beginners | ATN-001, DOC-001 | done | Query/KV head sharing, normalization, partial RoPE, two-byte KV rows, causal reads, stable softmax, candidate/commit ownership, timing, numeric gates, and proof limits are code-linked and worked | [`docs/43-cuda-attention-decode.md`](docs/43-cuda-attention-decode.md); [`tests/test_cuda_attention.py`](tests/test_cuda_attention.py); log 2026-08-31T06:57:49Z |
| EDU-030 | Explain memory-bounded CUDA attention prefill and the 128K boundary for beginners | ATN-002, DOC-001 | done | Chunk causality, candidate-row continuity, linear score workspace, whole-chunk commit/cancellation, token-wise equivalence, 131,072 sizing/execution, timing, and proof limits are code-linked and worked | [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md); [`tests/test_cuda_attention_prefill.py`](tests/test_cuda_attention_prefill.py); log 2026-08-31T09:33:05Z |
| EDU-031 | Explain the CUDA scheduler prerequisite primitives for beginners | CUD-003, DOC-001 | done | Q8_0 resident weights versus transient Q8 activations, embedding row decode, BF16 pointwise storage, packed layouts, GDN grouped/tiled mapping, numeric gates, timing, and proof limits are code-linked and worked | [`docs/45-cuda-scheduler-primitives.md`](docs/45-cuda-scheduler-primitives.md); [`tests/test_cuda_scheduler_primitives.py`](tests/test_cuda_scheduler_primitives.py); log 2026-08-31T11:26:17Z |
| EDU-032 | Explain resident CUDA model execution and the 64-layer hybrid schedule for beginners | SCH-001, DOC-001 | done | Upload ownership, layer alternation, scratch reuse, BF16/FP32 boundaries, state offsets, full logits, numeric/greedy gates, timing, and proof limits are code-linked and worked | [`docs/46-cuda-full-scheduler.md`](docs/46-cuda-full-scheduler.md); [`tests/test_cuda_full_scheduler.py`](tests/test_cuda_full_scheduler.py); log 2026-08-31T12:13:55Z |
| EDU-033 | Explain exact CUDA prefix synchronization and replay for beginners | SES-001, DOC-001 | done | Tokens, common prefixes, append/no-op reuse, divergent/shortened replay, exact equality, memory rationale, preflight failure behavior, and proof limits are code-linked and worked | [`docs/47-cuda-prefix-sync.md`](docs/47-cuda-prefix-sync.md); [`tests/test_cuda_prefix_sync.py`](tests/test_cuda_prefix_sync.py); log 2026-08-31T13:45:38Z |
| EDU-034 | Explain atomic CUDA evaluation, commit, cancellation, errors, and separate sampling for beginners | SES-002, DOC-001 | done | Candidate versus committed state, pointer publication, frontier-last visibility, polling, injected failure, sampling purity, memory cost, and proof limits are code-linked and worked | [`docs/48-atomic-eval-and-sampling.md`](docs/48-atomic-eval-and-sampling.md); [`tests/test_cuda_atomic_eval.py`](tests/test_cuda_atomic_eval.py); log 2026-08-31T17:55:34Z |
| EDU-035 | Explain versioned atomic CUDA checkpoint save/restore for beginners | SES-003, DOC-001 | done | File framing, compatibility and payload hashes, logical state sections, atomic rename, validation-before-mutation, sampler persistence, exact continuation, corruption, and proof limits are code-linked and worked | [`docs/49-cuda-checkpoints.md`](docs/49-cuda-checkpoints.md); [`tests/test_cuda_checkpoint.py`](tests/test_cuda_checkpoint.py); log 2026-08-31T19:09:54Z |
| EDU-036 | Explain the complete pre-graph 128K GPU allocation ledger and remaining graph gate for beginners | MEM-001, DOC-001 | done | GiB versus GB, resident/session/workspace categories, KV arithmetic, runtime/allocator deltas, reserve calculation, physical allocation, and why post-graph admission remains open are code-linked and worked | [`docs/50-pre-graph-128k-memory.md`](docs/50-pre-graph-128k-memory.md); [`tests/test_cuda_memory_fit.py`](tests/test_cuda_memory_fit.py); log 2026-08-31T19:17:33Z |
| EDU-037 | Explain synchronized GPU timing, NVTX ranges, attribution categories, and profiler evidence for beginners | OPT-001, DOC-001 | done | CPU clocks versus CUDA events, asynchronous work, synchronization, ranges, category sums, unavailable boundaries, perturbation, and Nsight proof limits are code-linked and worked | [`docs/51-runtime-timing-and-nvtx.md`](docs/51-runtime-timing-and-nvtx.md); [`tests/test_cuda_timing.py`](tests/test_cuda_timing.py); log 2026-08-31T20:00:28Z |
| EDU-038 | Explain profiler-led fusion and fused/unfused admission for beginners | OPT-002, DOC-001 | done | Kernel launch cost, utilization, fusion boundary, retained reference path, exact/numeric comparison, A/B samples, rejected candidates, and proof limits are code-linked and worked | [`docs/52-profiler-led-fusion.md`](docs/52-profiler-led-fusion.md); [`tests/test_cuda_fusion.py`](tests/test_cuda_fusion.py); log 2026-09-01T05:44:16Z |
| EDU-039 | Explain stable-address CUDA graph capture, replay, ownership, and dynamic-boundary limits for beginners | OPT-003, DOC-001 | done | Capture/instantiate/replay, stable pointers, FFN graph scope, dynamic attention exclusion, retained non-graph path, equivalence, memory ownership, and proof limits are code-linked and worked | [`docs/53-stable-address-cuda-graphs.md`](docs/53-stable-address-cuda-graphs.md); [`tests/test_cuda_graph.py`](tests/test_cuda_graph.py); log 2026-09-01T07:08:29Z |
| EDU-040 | Explain offline CUDA dispatch tuning, row buckets, prompt tiles, and selection limits for beginners | OPT-004, DOC-001 | done | Candidate launch shapes, row buckets, prompt tiles/chunks, warmups, samples, selection rule, retained negatives, checked-in table, and proof limits are code-linked and worked | [`docs/55-offline-dispatch-tuning.md`](docs/55-offline-dispatch-tuning.md); [`tests/test_cuda_dispatch_tuning.py`](tests/test_cuda_dispatch_tuning.py); log 2026-09-01T07:30:51Z |
| EDU-041 | Explain the interactive CLI, public runtime ownership, generation loop, sampling, stops, and persistence for beginners | CLI-001, DOC-001 | done | Terminal input, chat rendering, encode/decode, sync/eval/sample separation, reasoning display, stop tokens, checkpoint commands, CUDA-only production build, and proof limits are code-linked and worked | [`docs/56-interactive-text-cli.md`](docs/56-interactive-text-cli.md); [`README.md`](README.md); log 2026-09-01T11:56:52Z |
| EDU-042 | Explain artifact hashing, SHA-NI dispatch, fallback, and cold/warm storage limits for beginners | MDL-003, DOC-001 | done | Whole-file identity versus ZFS block integrity, CPU instructions, runtime dispatch, exact digest equality, cache/storage limits, measurements, and proof boundaries are code-linked and worked | [`docs/57-hardware-sha256.md`](docs/57-hardware-sha256.md); [`fixtures/sha256_acceleration.json`](fixtures/sha256_acceleration.json); log 2026-09-01T11:56:52Z |
| EDU-043 | Explain the HTTP listener, routes, single-flight queue, cancellation, and server lifecycle for beginners | SRV-001, DOC-001 | done | Sockets, HTTP requests/responses, loopback binding, health/models payloads, queue tickets/timing, cancellation, one-session ownership, shutdown, exclusions, and proof limits are code-linked and worked | [`docs/58-http-server-core.md`](docs/58-http-server-core.md); [`pins/server_core_contract.json`](pins/server_core_contract.json); [`fixtures/server_core.json`](fixtures/server_core.json); log 2026-09-01T18:19:42Z |
| EDU-044 | Explain JSON, Chat Completions, SSE streaming, tools, stops, usage, queueing, and cancellation for beginners | SRV-002, API-002, DOC-001 | done | Request/response fields, validation, token generation, reasoning/content separation, function calls, stream events, stop behavior, session reuse, and proof limits are code-linked and worked | [`docs/59-chat-completions.md`](docs/59-chat-completions.md); [`pins/chat_completions_contract.json`](pins/chat_completions_contract.json); [`fixtures/chat_completions.json`](fixtures/chat_completions.json); log 2026-09-01T19:00:04Z |
| EDU-045 | Explain Responses objects, typed items, events, storage, and exact continuation for beginners | SRV-003, SES-004, API-003, DOC-001 | done | A reader new to APIs can follow input items through tokens and output items, distinguish IDs from state, understand atomic records and exact prefixes, and identify every supported/rejected boundary | [`docs/60-responses-and-continuation.md`](docs/60-responses-and-continuation.md); [`pins/responses_contract.json`](pins/responses_contract.json); [`fixtures/responses.json`](fixtures/responses.json); log 2026-09-02T14:27:35Z |
| EDU-046 | Explain benchmark workloads, samples, percentiles, throughput, telemetry, and evidence limits for beginners | BEN-001, OPT-001, DOC-001 | done | A new reader can distinguish warm-ups/runs/tokens, prefill/decode/TTFT/ITL, p50/p95, raw versus summary data, process/device telemetry, unavailable values, and benchmark versus comparison claims | `docs/61-benchmark-harness.md`; `docs/README.md`; log 2026-09-02T15:24:00Z |
| EDU-047 | Explain layer-major chunked prefill, scratch reuse, atomic chunk commit, and its decode/reference boundary | SCH-002, MEM-002, DOC-001 | done | A new reader can follow 64 prompt rows through embeddings, MMQ, GDN/attention state, FFN, final logits, commit, and equivalence/performance evidence | [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md); log 2026-09-02T16:25:43Z |
| REL-001 | Publish reproducible release evidence bundle | CMP-003, QLT-001, DOC-001 | pending | Builds, hashes, raw results, reports, and documentation claims reconcile | — |

## Delivery-Gate Mapping

| Gate | Tasks |
|---|---|
| 1. Approved plan | ART-001 |
| 2. Operational ledger | ART-002 |
| 3. Pins | PIN-001–PIN-003 |
| 4. Build/API/allocation skeleton | BLD-001–BLD-003, API-001 |
| 5. Loader/tokenizer/scalar/traces/fixtures | MDL-001–MDL-002, TOK-001–TOK-002, CPU-001–CPU-004, TRC-001–TRC-002, ORA-001 |
| 6. CUDA primitives | CUD-001–CUD-003 |
| 7. GDN/attention/scheduler | GDN-001–GDN-002, ATN-001–ATN-002, SCH-001 |
| 8. Sessions and 128K | SES-001–SES-003, MEM-001 |
| 9. Profiling/optimization | OPT-001–OPT-069 (OPT-031 superseded by OPT-055; OPT-057–069 proposed) |
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

### 2026-08-30T06:25:03Z — CPU-014 layer composition started

- Commit `f5837e7` (`feat: execute real attention steps`) is present on both
  `main` and `origin/main`; the worktree was clean before this task began.
- Added CPU-014 before implementation and made it a CPU-004 dependency. The
  admitted mixer and FFN branches currently run only as separate diagnostics;
  the new boundary must prove that the mixer residual—not the original layer
  input—feeds post-mixer RMSNorm and SwiGLU for both layer kinds.
- The scheduler-facing layer wrapper will preflight the complete FFN storage
  contract before invoking a state-mutating GDN or attention mixer. This covers
  deterministic caller errors without claiming rollback for a hypothetical
  failure after validated persistent-state mutation; SES-002 remains responsible
  for transactional session commit.
- Added EDU-015 so branch order, the two residual additions, GDN versus attention
  state, temporary buffer reuse, validation order, and the remaining full-model
  boundary are explained alongside the implementation.

### 2026-08-30T06:31:18Z — CPU-014 and EDU-015 accepted

- Added explicit scalar scheduler-layer types for GDN and attention parameters,
  state, mixer workspace, the 5,120-value post-mixer handoff, shared FFN
  workspace, and final output. The layer entry points preserve the pinned
  pre-norm mixer → first residual → post-mixer pre-norm SwiGLU → second residual
  order without a generic operator registry or backend inheritance.
- Separated FFN structural validation from arithmetic. Both layer wrappers
  preflight the complete FFN and final-output contract before invoking a
  state-mutating mixer. One-value-short FFN activation storage is rejected with
  GDN convolution/recurrent state or attention KV and final output untouched.
- Layer-0 and two-position layer-3 diagnostics execute every production-sized
  mixer and FFN value. Their post-mixer selected lanes are bit-identical to the
  separately admitted mixer diagnostics; final selected lanes equal exact FP32
  post-mixer plus FFN-correction additions.
- Frozen post-mixer, FFN norm/gate/up/SwiGLU/correction, and layer-output taps are
  stored as an explicitly labeled native composition regression. It proves the
  wrapper handoff and order but does not claim to replace the pending direct
  Transformers full-layer trace.
- Added a beginner chapter explaining a decoder layer, pre-norm, both residual
  branches, why FFN must consume the first residual, GDN versus attention state,
  parameter/state/workspace/output lifetimes, exact scalar memory totals,
  validation order, a four-value worked analogy, evidence limits, and the
  remaining full-model scheduler boundary.
- Measured real composed diagnostics were 0.65 s / 297,120 KiB maximum RSS for
  layer 0 and 0.97 s / 274,720 KiB for the two-position layer 3 run. Fixture
  capture completed in 1.26 s with 297,280 KiB maximum RSS.
- Commands: Ruff format/check, fixture regeneration and JSON validation,
  focused 20-test real-branch suite, clean restricted C++17 build, 69 pytest
  tests, `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation,
  and RTX 5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-014 and EDU-015 done. CPU-004 still requires cross-layer schedule
  ownership, embedding lookup, all 64 state slots, final norm, vocabulary logits,
  and token/chunk execution.

### 2026-08-30T06:31:45Z — Layer-composition commit boundary

- Reviewed both layer kinds, scheduler-visible types, exact branch and residual
  order, preflight-before-state behavior, full-workspace finite checks, native
  fixture labeling, memory arithmetic, beginner documentation, and clean
  host/container verification before committing.

### 2026-08-30T06:35:08Z — CPU-015 model boundaries started

- Commit `c8f0c02` (`feat: compose complete decoder layers`) is present on both
  `main` and `origin/main`; the worktree was clean before this task began.
- Added CPU-015 before implementation and made it a CPU-004 dependency. The
  typed model already binds the 248,320 × 5,120 Q4_K embedding table, 5,120
  final direct-scale norm, and 248,320 × 5,120 Q6_K output matrix, but no
  scheduler-facing operation currently executes them.
- The new boundary will decode exactly one admitted embedding row, reject token
  IDs outside `[0, 248320)` before writes, normalize a deterministic final
  hidden vector, compute every FP32 logit, and independently verify selected
  physical Q4_K/Q6_K rows. It will not claim that the deterministic vector is a
  real layer-63 result; cross-layer execution remains CPU-004.
- Added EDU-016 so token IDs, embeddings, hidden vectors, final normalization,
  logits, probabilities, greedy selection, vocabulary size, scalar cost,
  workspace, bounds, and the remaining model boundary are explained alongside
  implementation.

### 2026-08-30T06:39:56Z — CPU-015 and EDU-016 accepted

- Added scheduler-facing real model boundaries: exact-range Q4_K token embedding
  row decode, prepared 5,120-value direct final-norm scale, complete final
  RMSNorm, and the 248,320-row Q6_K FP32 vocabulary projection. Sampling remains
  separate from raw logits and neither boundary mutates session state.
- The diagnostic decodes endpoint IDs 0 and 248,319 plus interior ID 42, uses
  row 42 as an explicitly artificial final-hidden input, computes every native
  logit, verifies all outputs are finite, and reports the full-vector greedy
  index and exact count. It does not label those scores as a model continuation.
- An independent mapped-GGUF generator hashes and decodes all three physical
  embedding rows, applies float-libm direct-scale RMSNorm, hashes and decodes
  output rows 0/1/42/1000/248319, and matches native selected logits under frozen
  absolute, relative, and RMS limits.
- Token ID 248,320 is rejected before any embedding write. A one-value-short
  final normalized workspace is rejected before either normalization or logits
  are written. Exact vocabulary and residual widths are required.
- Added a beginner chapter explaining token IDs versus text, vocabulary bounds,
  embedding rows and hidden features, final RMSNorm, output rows, logits versus
  probabilities, greedy and sampled choices, Q4_K/Q6_K storage, exact scalar
  cost/workspace, independent evidence, and why this is not yet a continuation.
- The complete native boundary ran in 2.21 s with 1,045,280 KiB maximum RSS.
  Independent selected-row fixture generation ran in 0.04 s with 28,800 KiB
  maximum RSS. The full output performs exactly 1,271,398,400 scalar weight
  products and emits 970 KiB of FP32 logits.
- Commands: Ruff format/check, fixture generation and JSON validation, focused
  14-test boundary/tensor/weight suite, clean restricted C++17 build, 73 pytest
  tests, `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation,
  and RTX 5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-015 and EDU-016 done. CPU-004 still requires owning all layer
  parameters/state, iterating the exact 64-layer schedule, joining real embedding
  to layer 0 and layer 63 to final logits, and token/chunk execution evidence.

### 2026-08-30T06:40:15Z — Model-boundary commit boundary

- Reviewed vocabulary bounds, exact Q4_K/Q6_K typed views, direct final norm,
  all-logit finite execution, independent row hashes and selected dots,
  pre-write failures, scalar arithmetic and memory claims, beginner
  documentation, and clean host/container verification before committing.

### 2026-08-30T06:43:22Z — CPU-016 full scalar pass started

- Commit `27b0d11` (`feat: execute embeddings and logits`) is present on both
  `main` and `origin/main`; the worktree was clean before this task began.
- Added CPU-016 before implementation and made it a CPU-004 dependency. It will
  own prepared scalar parameters for every layer, allocate 48 independent GDN
  state slots and 16 independent attention KV slots, reuse one exact scratch
  arena sequentially, and connect token embedding through layers 0–63 to final
  norm and complete logits.
- CPU-016 is a structural one-token zero-state admission. Frozen native layer
  boundary/state/logit taps will detect schedule or ownership regressions, while
  the result remains explicitly ineligible as semantic continuation authority
  until TRC-001/TRC-002/ORA-001 compare it with pinned Transformers and the
  independent same-GGUF oracle.
- Added EDU-017 so the hybrid schedule, physical layer versus variant slot,
  engine-prepared data, session-owned state, shared scratch, ping-pong residuals,
  state frontier, runtime cost, evidence, and remaining oracle/token-chunk work
  are explained alongside implementation.

### 2026-08-30T06:53:51Z — CPU-016 and EDU-017 accepted

- Added move-only scalar runtime owners for engine-prepared parameters,
  session-persistent state, and reusable execution scratch. Preparation checks
  the exact physical schedule and builds direct physical-layer views over 48
  compact GDN and 16 compact attention slots without copying mapped matrices.
- Prepared storage contains 2,645,504 FP32 values (10.091796875 MiB). A
  capacity-one zero state contains 39,747,584 FP32 values (151.625 MiB), and the
  complete named scratch arena contains 204,161 FP32 values (about 797.504 KiB),
  excluding the caller-owned 970 KiB logits.
- Implemented token execution through embedding, exact layers 0–63, shared
  branch workspaces, ping-pong residual buffers, final direct norm, and every
  vocabulary logit. Successful token 42 completed 64 layers, mutated exactly 48
  GDN and 16 attention slots, produced finite state/hidden/logits, selected
  native greedy token 3,649, and advanced the frontier from zero to one only
  after logits completed.
- Global preflight validates exact parameter/state/workspace vector sizes,
  capacity/frontier, vocabulary/logit counts, and the complete layer-kind
  schedule before embedding or state mutation. Removing one shared FFN value
  leaves zero completed layers, frontier zero, all state unchanged, and all
  logits untouched.
- Frozen final-hidden, final-norm, selected-logit, early/middle/final GDN and
  attention state taps, counts, and greedy index are explicitly labeled a native
  structural zero-state regression. CPU-004 remains pending because no direct
  full-model semantic authority has admitted these continuation logits.
- Added a beginner chapter explaining the 3-GDN/1-attention schedule, physical
  layers versus compact slots, engine/session/scratch ownership, move-only
  pointers, exact memory totals, workspace reuse, ping-pong residuals, frontier,
  preflight, structural fixtures, timings, and oracle/atomicity limits.
- The first full host run took 36.21 s with 18,013,440 KiB maximum RSS; warm
  fixture capture took 22.81 s with the same maximum RSS. The clean full suite
  completed 75 tests in 38.27 s.
- Commands: Ruff format/check, fixture JSON validation, focused two-test
  full-token suite, clean restricted C++17 build, 75 pytest tests,
  `git diff --check`, pinned CUDA 13.0.2 full build, SM120 compilation, and RTX
  5090 probe — passed. Probe values were 33,671,348,224 total and
  33,139,458,048 free bytes.
- Marked CPU-016 and EDU-017 done. CPU-004 still requires direct full-model trace
  admission, multiple-token continuation equality, arbitrary chunks, and
  token-wise versus chunked execution.

### 2026-08-30T06:54:20Z — Full-scalar-token commit boundary

- Reviewed move-only pointer ownership, exact storage formulas, schedule and slot
  mapping, state isolation, shared workspace construction, 64 ping-pong handoffs,
  frontier order, full finite checks, global preflight, structural fixture
  labeling, beginner documentation, and clean host/container verification before
  committing.

### 2026-08-30T07:04:30Z — TRC-001 trace contract started

- Commit `584ad97` (`feat: execute full scalar token`) is present on both `main`
  and `origin/main`; the worktree was clean before this task began.
- Began TRC-001. The v1 format will use a versioned JSON manifest plus one
  deterministic little-endian FP32 blob, with exact model/tool identities,
  prompt bytes, token IDs, positions, tensor names/shapes/ranges/checksums and
  summaries, and session frontiers before and after execution.
- The typed comparator will report absolute, relative, RMS, cosine, NaN/Inf,
  first-failing-index, and top-logit differences. Its tolerance rule and
  non-finite behavior are part of the contract, not caller-specific convention.
- Added EDU-018 before implementation so a reader new to numerical inference can
  understand taps, binary layout, checksums, frontiers, every metric, and the
  difference between structural integrity and semantic agreement.
- Real scalar layer taps remain TRC-002 work. This task defines and validates the
  evidence container; it does not claim that the current native scalar output
  agrees with Transformers or llama.cpp.

### 2026-08-30T07:09:36Z — TRC-001 focused-test corrections

- The first focused pytest collection failed because the repository root was
  absent from pytest's import path, so `tools.qw38_trace` could not be imported.
  Adding only `tools/__init__.py` did not resolve collection. Added the explicit
  `pythonpath = ["."]` pytest setting; the helper then imported normally.
- The next focused run passed 13 cases and failed two assertions. One expected a
  different maximum-relative-error index for two decimal values whose binary
  errors were effectively tied; the other expected equal decimal logit deltas,
  although their FP64 representations differed. The implementation was stable;
  corrected the assertions to the actual deterministic floating-point ordering
  instead of adding an undocumented approximate tie rule.
- Tightened participant metadata to a typed `ArtifactIdentity` with exact name,
  revision, and SHA-256 fields. Also made every NaN or infinity an admission
  failure even when both arrays contain the same non-finite value; equal broken
  outputs must not pass a semantic gate.
- Commands: Ruff format/check and 15 focused trace tests — passed after the
  corrections. The earlier collection and assertion failures are preserved
  above rather than erased.

### 2026-08-30T07:13:12Z — TRC-001 and EDU-018 accepted

- Froze trace schema `qw38.trace` version 1: exact typed model/tool identities,
  base64 prompt bytes with count/hash, token IDs and positions, before/after
  session frontiers and named state hashes, a canonical JSON manifest, and one
  contiguous manifest-ordered little-endian FP32 blob.
- The writer rounds values to stored FP32 before summaries and emits deterministic
  files. The fail-closed reader validates exact versioned fields, canonical
  prompt encoding, dimensions, byte ranges, full/per-tensor hashes, complete
  blob coverage, and recomputed finite/non-finite summaries before returning
  values.
- Added typed absolute, relative, RMS, cosine, NaN/Inf, first-failing-index, and
  deterministic top-logit reports. The frozen finite gate fails only when both
  absolute and relative tolerances are exceeded; every non-finite input fails.
- Added 15 focused tests covering deterministic round trips, exact endian bytes,
  FP32 rounding, metadata/frontiers, invalid writer inputs, malformed manifests,
  blob corruption, tolerance behavior, non-finite values, and near-tie top-logit
  order. Added explicit repository-root pytest import configuration for typed
  diagnostic helpers.
- Added a beginner chapter defining taps, manifests versus blobs, FP32 and byte
  order, shapes/ranges, checksums versus correctness, pinned identities, prompt
  representations, frontiers, summaries, every metric, near ties, fixture versus
  numeric equality, negative evidence, and the remaining semantic proof boundary.
- Verification: Ruff format/check, JSON validation, focused 15-test trace suite,
  clean restricted C++17 build, all 90 pytest tests in 38.26 s, and
  `git diff --check` passed. The pinned CUDA 13.0.2 container rebuilt all host
  tools, compiled the SM120 probe, and ran it on the RTX 5090 with
  33,671,348,224 total and 33,139,458,048 free bytes.
- Marked TRC-001 and EDU-018 done. TRC-002 remains pending: no real runtime tap
  has yet been emitted through this format, so CPU-004 and semantic oracle
  admission remain open.

### 2026-08-30T07:13:40Z — Trace-contract commit boundary

- Reviewed schema/version fail-closed behavior, typed identities, prompt and
  state metadata, deterministic FP32 conversion, contiguous range arithmetic,
  whole/per-tensor checksums, summaries, tolerance/non-finite/top-logit rules,
  corruption tests, proof labels, and beginner documentation before committing.

### 2026-08-30T07:14:15Z — TRC-002 runtime taps started

- Commit `4ef411e` (`feat: define diagnostic trace bundles`) is present on both
  `main` and `origin/main`; the worktree was clean before this task began.
- Found an operational dependency cycle: CPU-004 required semantic-authority
  fixtures, TRC-002 depended on CPU-004, and ORA-001 depended on both. This did
  not reflect the approved plan's separation between scalar mechanics, trace
  plumbing, and later oracle admission.
- Corrected the ledger boundary before implementation. CPU-004 now owns
  multi-token/arbitrary-chunk scalar mechanics and exact internal equivalence;
  TRC-002 depends on the admitted one-token runtime CPU-016 and owns stable
  backend-neutral trace plumbing; ORA-001 remains the independent semantic
  authority gate. Product scope and execution architecture are unchanged.
- Added and began TRC-003 before implementation after inspecting the build and
  scalar runtime. It isolates the reusable sink/filter/build boundary from the
  much larger real 64-layer tap wiring owned by TRC-002; this keeps invalid
  filters and release-symbol absence independently testable.
- Began EDU-019 alongside TRC-003. Diagnostic tap code must be excluded at compile time
  from normal binaries, accept exact layer/name filters, expose stable semantic
  names and shapes through a backend-neutral sink, and feed the v1 bundle
  without making the C++ runtime depend on JSON or Python.

### 2026-08-30T07:19:17Z — TRC-003 and EDU-019 accepted

- Added a diagnostic-only C++ trace boundary with exact registered semantic tap
  names, layer/name filters, typed non-owning tensor views, one-to-three-
  dimensional shape validation, overflow/count checks, and an explicit-Status
  backend-neutral sink callback.
- Added a separate `make diagnostic` object tree and executable. The guarded
  header cannot be included without `QW38_DIAGNOSTIC_TRACE`; ordinary library
  sources do not link the trace implementation. The diagnostic evaluator's
  synthetic command exercises wildcard, exact, zero-match, and malformed-filter
  behavior without loading the production model.
- Added four tests that build the diagnostic target, exercise filter selection
  and failure behavior, and inspect the normal evaluator bytes. The normal
  binary contains neither `--check-trace-filter` nor representative attention
  and GDN tap strings.
- Added a beginner chapter explaining separate builds, compile-time versus
  runtime disabling, sinks and callbacks, borrowed views, exact filters, stable
  semantic naming, shape validation, CUDA transfer implications, and why
  synthetic views are not model evidence.
- Verification: Ruff format/check, 19 focused trace tests, clean restricted
  C++17 normal and diagnostic builds, all 94 pytest tests in 38.46 s, and
  `git diff --check` passed. The pinned CUDA 13.0.2 container rebuilt the normal
  tools, compiled SM120, and probed 33,671,348,224 total and 33,139,458,048 free
  device bytes.
- Marked TRC-003 and EDU-019 done. TRC-002 remains pending until the real scalar
  stages call this sink and selected tensors are written through the v1 bundle.

### 2026-08-30T07:19:30Z — Diagnostic-isolation commit boundary

- Reviewed dependency-cycle correction, macro/object isolation, tap-name
  registry coverage, filter validation, shape arithmetic, callback status
  propagation, release binary inspection, documentation proof labels, and clean
  host/container evidence before committing.

### 2026-08-30T07:21:07Z — TRC-002 real scalar taps started

- Commit `966d947` (`feat: isolate diagnostic trace taps`) is present on both
  `main` and `origin/main`; the worktree was clean before this task began.
- Began TRC-002 and added EDU-020 before implementation. A diagnostic-only
  scalar entry point will offer stable embedding, GDN, attention, FFN, residual,
  state, final-norm, and logits views immediately after the stage that owns each
  value, while the existing release entry point and object code remain unchanged.
- A focused evaluator capture will require one exact layer/name filter, copy the
  selected real tensor as little-endian FP32, and expose sufficient token,
  position, frontier, and state identity metadata for the typed Python helper to
  create and re-read a v1 bundle.
- The existing `real_scalar_token.json` remains explicitly native structural
  evidence. Matching a filtered tap against it proves trace placement and copy
  fidelity, not agreement with Transformers or llama.cpp; ORA-001 retains that
  semantic admission responsibility.

### 2026-08-30T07:26:00Z — TRC-002 tap corrections

- During review, the first attention tap wiring labeled raw projected query/key
  workspace as RoPE output. Inspection of `attention_decode_step_impl` showed
  normalized rotated Q/K lived in local arrays while the projection workspace
  stayed raw. Added distinct `attention.query`/`attention.key` taps and
  diagnostic-only buffers that copy the exact normalized partial-RoPE arrays
  used for score/cache computation; no mislabeled fixture was admitted.
- The first diagnostic rebuild after adding those guarded workspace fields
  failed under `-Werror=missing-field-initializers` in the older real-attention
  evaluator fixture. The first correction accidentally declared the buffers in
  the GDN fixture because both functions used the same `mixer_output` context;
  the next compile reported the names undeclared at the attention initializer.
  Moved the guarded buffers to the exact attention fixture scope and supplied
  all four fields. Normal and diagnostic builds then passed.
- Focused attention/scalar tests passed 11 cases in 34.47 s after the correction.
  These failed builds and the pre-fixture label diagnosis are retained as
  negative evidence rather than hidden.

### 2026-08-30T07:32:39Z — TRC-002 and EDU-020 accepted

- Added a diagnostic-only traced scalar entry point over the same internal
  64-layer loop used by release execution. It offers the embedding, input norms,
  GDN packed/convolution/Q/K/recurrence/state/output, raw and RoPE attention
  Q/K, V/KV/context/output, mixer/FFN/residual, final norm, and logits at stable
  post-stage lifetimes before shared workspace reuse.
- Froze 29 global/layer tap names, applicable layer kinds, and exact shapes in a
  versioned scalar trace contract. Source coverage tests require every pin to be
  registered and offered; runtime view validation checks each shape product.
- Added an exact-filter native evaluator capture with fail-closed output handling,
  token/position/frontier metadata, and before/after SHA-256 for all GDN and
  attention state slabs. Added a typed Python driver that first hashes the full
  GGUF, identifies the diagnostic executable, writes trace v1, and re-reads the
  completed bundle before success.
- The real integration test captured all 5,120 final-normalized values for token
  42. Indices 0, 1, 2,559, and 5,119 matched the existing native structural
  fixture exactly; model hash, token, position, shape, global-layer encoding,
  and frontier 0→1 also matched. A fake artifact failed before native execution.
- Added a beginner chapter explaining stage timing, shared-workspace copy
  lifetime, raw versus normalized/RoPE attention values, every important shape,
  exact filters, native/Python responsibilities, whole-model identity, state
  hash scope, failure atomicity, timings, and the remaining oracle boundary.
- Split future CUDA tap wiring into TRC-004, dependent on the first real CUDA
  kernel. Keeping it inside TRC-002 would recreate a dependency cycle by blocking
  ORA-001, whose frozen scalar tolerances must exist before CUD-001 admission.
- Verification: JSON validation, Ruff format/check, 22 focused trace tests in
  33.28 s, clean normal and diagnostic restricted C++17 builds, all 97 pytest
  tests in 71.63 s, and `git diff --check` passed. The pinned CUDA 13.0.2
  container rebuilt the normal tools, compiled SM120, and probed 33,671,348,224
  total and 33,139,458,048 free device bytes.
- Marked TRC-002 and EDU-020 done. This closes scalar trace transport only;
  ORA-001 still must obtain independent authority traces and freeze tolerances.

### 2026-08-30T07:33:00Z — Real-scalar-trace commit boundary

- Reviewed compile-time release exclusion, exact tap timing and shapes, raw/RoPE
  separation, state row addressing, callback failure propagation, single-match
  file behavior, full artifact/tool identity, bundle revalidation, source pin
  coverage, structural proof labels, and clean host/container evidence before
  committing.

### 2026-08-30T10:54:58Z — CPU-004 scalar chunks started

- Commit `c9c9c1c` (`feat: capture real scalar traces`) is present on both `main`
  and `origin/main`; the worktree was clean before this task began.
- Began CPU-004 and added EDU-021 before implementation. The chunk API will
  validate the complete token span, remaining session capacity, output stride,
  and multiplication bounds before calling the admitted one-token runtime, then
  retain one full FP32 vocabulary row per input token.
- Exact equivalence will compare a two-token chunk with two repeated one-token
  calls from independent zero states, including every logit, all four owning
  state slabs, frontier, and completed-layer count. Negative cases must leave
  frontier/state/logit sentinels untouched when whole-chunk preflight fails.
- This gate proves scalar scheduling and chunk-boundary invariance. The native
  continuation remains ineligible as semantic authority until ORA-001 compares
  traces/logits against the pinned external implementations.

### 2026-08-30T11:57:06Z — CPU-004 and EDU-021 accepted

- Added `execute_scalar_chunk`, which retains a complete FP32 vocabulary row per
  input token and calls the admitted one-token scheduler in strict token order at
  the session's current frontier. The API checks parameter/state/workspace
  owners, positive token count, remaining capacity, output multiplication and
  exact length, and every vocabulary ID before executing the first token.
- The real equivalence diagnostic ran `[42, 3649]` as one length-2 chunk and as
  two independent one-token calls from a second zero state. All 496,640 logits,
  every GDN convolution/recurrent value, every attention K/V value, and frontier
  matched exactly; both final workspaces reported 64 completed layers.
- The first row retained greedy token 3,649 and the second selected token 1,277.
  Five exact logits from each row plus stride/frontier/count evidence are frozen
  as native structural evidence, explicitly not an external semantic fixture.
- Invalid second token, capacity one for two inputs, and one-short logits storage
  all failed during whole-chunk preflight with frontier zero, zero state,
  `layers_completed = 0`, and every logit NaN sentinel untouched.
- Added a beginner chapter explaining token-wise versus chunk scheduling,
  history and positions, full preflight, row-major logits, exact state equality,
  completed-layer counter meaning, arbitrary partitions, measured cost, and the
  remaining semantic-authority boundary.
- The standalone four-token equivalence run took 87.08 s with 18,172,800 KiB
  maximum RSS. The four focused chunk tests passed in 87.48 s. Clean normal and
  diagnostic builds plus all 101 pytest tests passed in 157.94 s; JSON, Ruff,
  and `git diff --check` also passed.
- The pinned CUDA 13.0.2 container rebuilt all normal tools, compiled the SM120
  probe, and reported 33,671,348,224 total and 33,139,458,048 free device bytes.
- Marked CPU-004 and EDU-021 done. ORA-001 is now unblocked and is the remaining
  scalar semantic-admission gate before CUDA MMV implementation.

### 2026-08-30T11:57:20Z — Scalar-chunk commit boundary

- Reviewed whole-chunk preflight ordering and overflow arithmetic, row strides,
  token positions, state ownership, exact full-output/state comparisons,
  negative sentinel behavior, fixture authority labels, measured resource cost,
  and clean host/container evidence before committing.

### 2026-08-30T11:58:50Z — ORA-002 llama authority started

- Commit `6d4ed6d` (`feat: execute scalar token chunks`) is present on both
  `main` and `origin/main`; the worktree was clean before this task began.
- Split the broad ORA-001 gate into executable evidence increments before doing
  authority work: ORA-002 owns pinned same-GGUF llama.cpp, ORA-003 owns pinned
  Transformers eager/offload feasibility and taps, and ORA-004 freezes the
  resulting three-authority fixtures/tolerances. ORA-001 remains the umbrella
  admission gate and the product architecture is unchanged.
- Began ORA-002 and EDU-022. The machine has 248 GiB disk free, about 29 GiB
  available host RAM, and 32,607 MiB GPU memory. The pinned 18,973,870,432-byte
  GGUF is installed; PyTorch, Transformers, Accelerate, and Safetensors are not
  installed, and no pinned llama.cpp checkout or binary exists locally.
- The immediate work will fetch/build exact llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` in ignored authority cache, retain
  reproducible commands/configuration in the repository, and first establish
  exact token/template/model identity plus deterministic greedy continuation.
  Debug tensor taps remain ORA-004 work and cannot be substituted by native
  Quartz self-comparison.

### 2026-08-30T12:09:12Z — ORA-002 harness tests corrected

- Added the exact-build contract, pinned CUDA container, public-API raw-token
  adapter, full-logit comparison helper, focused tests, and beginner authority
  chapter while the upstream CUDA build continued.
- The first five focused tests had two documentation/metadata assertion errors:
  the artifact lock uses `sources.llama_cpp`, not `tools.llama_cpp`, and a test
  searched across a Markdown line break for the full phrase `independent
  same-GGUF oracle`. Neither error affected runtime code. Corrected the JSON key
  and asserted the stable unbroken phrase; the failed result is retained here
  rather than erased.
- The first pinned upstream build was intentionally stopped after 172 of 484
  Ninja edges because `-j 2` used only two of 20 host CPUs while more than 28
  GiB RAM remained available. No completed objects were removed. Increased the
  checked-in build limit to six jobs and resumed the same configured build
  incrementally; this is a build-time adjustment, not a source/configuration
  identity change.

### 2026-08-30T12:12:15Z — ORA-002 build scope corrected

- The resumed compile completed both requested upstream targets and the local
  adapter, then failed during host-side `llama-cli --version`: container-built
  `llama-cli` could not resolve `libllama-cli-impl.so` on the host. The relevant
  binaries and libraries had compiled successfully; the failure was in the
  verification environment.
- More importantly, the pinned revision's `llama-cli` dependency graph built
  unrelated server, UI, and multimodal targets and attempted a moving UI asset
  download. Its first bucket URL failed, then the unversioned `latest` URL
  succeeded. This network-dependent target is inadmissible in ORA-002 and the
  negative result is retained here.
- Narrowed ORA-002 to `llama-eval-callback` plus the repository-owned public-API
  raw-token adapter. The later comparative-baseline gate owns its own controlled
  upstream server build. Moved binary verification inside the pinned container,
  where its shared-library and CUDA runtime environment are defined.
- Audited the authority image's installed package versions and pinned all five
  packages literally in the Dockerfile/contract. The first image already
  resolved to those exact versions; a final rebuild will prove the checked-in
  pins rather than relying on an `apt` moving choice.

### 2026-08-30T12:16:03Z — ORA-002 first real run retained

- The fully pinned narrow build completed and container-side binary verification
  passed. Ninja reported and recovered from a premature `.ninja_log` end left
  by the intentional earlier termination; all required target edges then built
  successfully.
- Model verification, the two-token Quartz scalar execution, and the same
  two-token llama.cpp CUDA execution all succeeded. Both wrote exactly
  1,986,560 bytes (2 × 248,320 × 4) of logits and independently chose greedy
  tokens 3649 then 1277. llama.cpp reported 38.68 token/s for its two decode
  runs; this diagnostic timing is not a benchmark result.
- Final orchestration failed before comparison because invoking the Python file
  directly set its import root to `tools/`, so `from tools.qw38_trace` raised
  `ModuleNotFoundError`. The complete raw evidence was preserved. Corrected the
  checked-in harness to invoke the typed helper as module
  `python -m tools.compare_llama_authority`; no model/runtime result was changed.
- The first standalone comparison then rejected NVIDIA's stdout banner
  (`==========`) as an empty duplicate key before reading the adapter fields.
  Tightened the mixed-output parser to accept only lowercase identifier field
  names while retaining duplicate rejection, and added the banner as a focused
  regression test. Raw logits again remained unchanged.
- Added exact template identity to the harness using the existing
  `user_no_thinking` native rendering and llama.cpp's public tokenizer in
  vocabulary-only mode. Its first focused run incorrectly reported a read
  failure because `istreambuf_iterator` completion does not guarantee the
  stream's `eofbit` is set. Replaced that invalid success test with explicit
  open and `badbit` checks before any authority comparison.

### 2026-08-30T12:27:27Z — ORA-002 and EDU-022 accepted

- The corrected checked-in authority build completed from exact llama.cpp
  revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` with a clean detached
  checkout, pinned package versions, CUDA 13.0.2, requested architecture 120
  rewritten by upstream CMake to `120a`, `GGML_CUDA=ON`, `GGML_NATIVE=OFF`,
  tests off, and only the callback/local-adapter target boundary.
- The final `tools/run_llama_authority.sh` invocation completed with exit zero.
  Quartz and llama.cpp exactly matched the 74 rendered bytes and all 13 token
  IDs for `user_no_thinking`. Both complete 248,320-wide logit rows were finite
  and both runtimes chose greedy tokens 3649 then 1277.
- Position 0 reported maximum absolute error 0.17721319, RMS error 0.02946198,
  cosine 0.99984713, and 9/10 common top logits. Position 1 reported maximum
  absolute error 0.19039965, RMS error 0.03334594, cosine 0.99985330, and 10/10
  common top logits. Exact zero-tolerance equality failed at index 0 in both
  rows as expected for reporting-only cross-runtime evidence. No tolerance was
  admitted or loosened; ORA-003/ORA-004 remain mandatory.
- Frozen raw-row hashes are Quartz
  `1be136936bca8baea761464e16814ae01471f5e4e09d908efaf2432df834095b`
  and llama.cpp
  `03d747c8291b07f44ac47649c317a71fea633886f48e9dfbd4da6bb27ae74513`.
  The raw 1,986,560-byte files/logs remain in ignored evidence storage; their
  identities and complete metrics are committed in the small fixture.
- Added and tested an atomic Quartz full-logit dump diagnostic, strict raw-byte
  sizing, mixed container-output parsing, duplicate-field rejection, exact
  authority identity checks, and explicit reporting/admission separation.
- `uv run ruff format .` formatted the Python sources. Clean release and
  diagnostic builds passed, followed by all 108 pytest tests in 166.83 s.
  Ruff, every JSON parse, Markdown local-link validation, and `git diff --check`
  passed. The pinned CUDA build compiled all products plus the SM120 probe; the
  RTX 5090 reported compute capability 12.0, 33,671,348,224 total bytes, and
  33,139,458,048 free bytes.
- Marked ORA-002 and EDU-022 done. ORA-003—the pinned Transformers
  eager/offload feasibility and semantic trace—is the next task. ORA-001 and
  CUDA MMV remain blocked on the complete three-authority/tolerance gate.

### 2026-08-30T12:29:51Z — Llama-authority commit boundary

- Re-reviewed the exact source/model/container contract, narrow target graph,
  adapter ownership and cleanup, template/token identity, atomic native output,
  strict byte sizing, comparison direction, zero-tolerance label, proof limits,
  beginner chapter, source ledger, and every preserved negative result before
  commit. Made the runner independent of its caller's working directory.
- Final shell syntax, seven focused authority tests, Ruff, and
  `git diff --check` passed after the last script-only adjustment. The full
  108-test, clean-build, real authority, and CUDA evidence remains the accepted
  evidence recorded immediately above.

### 2026-08-30T12:30:42Z — ORA-003 Transformers authority started

- Commit `6ea9750` (`feat: add llama GGUF authority`) is present on both `main`
  and `origin/main`; the worktree was clean before this task began.
- Began ORA-003 and added EDU-023 before implementation. The primary authority
  must use official checkpoint revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and exact Transformers revision
  `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`; it cannot reuse GGUF or promote
  llama.cpp to official-checkpoint authority.
- Initial capacity audit: 247 GiB disk free, about 29 GiB host RAM available,
  no swap, and 32,111 MiB GPU memory free on the RTX 5090. No PyTorch,
  Transformers, Accelerate, or Safetensors environment exists yet. The next
  read-only step inventories official checkpoint shards/bytes and exact pinned
  source dependencies before selecting an eager CPU/GPU/disk offload map.
- The first checkpoint command incorrectly combined one `--include` option with
  additional positional glob arguments. `hf` warned that it ignored the include
  filter and downloaded only the ten requested metadata/template/tokenizer
  files, not any weight shard. Those files are required and valid, so they were
  retained. Corrected the next command to one Safetensors include filter; shard
  inventory remains 18 files and 55,563,006,776 bytes.

### 2026-08-30T12:36:58Z — ORA-003 eager feasibility and taps measured

- Downloaded and verified all 18 official Safetensors shards at exact checkpoint
  revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: 55,563,006,776 file bytes,
  55,562,855,904 indexed tensor bytes, and 1,199 tensors. The exact clean
  Transformers checkout is revision
  `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`; the hash-locked Python 3.12.3
  environment uses Torch 2.10.0+cu130 and the verified CUDA device reports
  compute capability 12.0.
- The first untapped real eager run succeeded with embeddings, final norm,
  output head, and layers 0–24 on GPU; layers 25–44 on CPU; layers 45–63 and
  the unused visual module on disk. It loaded in 23.03 seconds, executed two
  tokens in 45.18 seconds, peaked at 24,460,563,456 allocated and
  24,490,541,056 reserved GPU bytes, and selected tokens 3649 then 1277. The
  complete 1,986,560-byte logit blob hash was
  `9b64105a1c7262271c85054ef30cd116e0af4e85a497e6ae24c478007ed97947`.
- The upstream text wrapper loaded 1,184 of 1,199 checkpoint tensors. Its
  explicit unexpected-key policy ignored the 15 MTP tensors, consistent with
  V1's MTP exclusion; vision remains present but unused/offloaded. Accelerate
  directly reopens original Safetensors ranges for disk-mapped layers, so the
  offload directory remained empty rather than duplicating weights. Both facts
  are documented instead of being silently treated as missing work.
- Added observation-only module hooks and wrappers around the original upstream
  GDN recurrence, partial RoPE, and eager-attention functions. The captured run
  again chose tokens 3649 and 1277 with the identical full-logit hash, proving
  the taps did not change this diagnostic result. It emitted 238 finite named
  tensors totaling 20,160,352 canonical little-endian FP32 bytes with blob hash
  `1083ab56433026ac03128603dbed017391c98e3053b9169774e9654e8e85a031`.
  Representative evidence includes full `[1,48,128,128]` GDN states and an
  attention K/V history growing from one to two positions.
- Reporting-only official-versus-GGUF row metrics were recorded with zero
  tolerance, not admitted: Quartz cosine was 0.99557/0.99402 with RMS
  0.15852/0.21300; llama.cpp cosine was 0.99535/0.99375 with RMS
  0.16242/0.21763. All three selected the same greedy continuations. These are
  expected cross-artifact quantization differences; ORA-004 still owns
  first-failure diagnosis and immutable per-tap tolerances.
- The repository-wide validation command mistakenly requested nonexistent Make
  target `cuda-probe` after `make cuda-build` had already compiled the probe.
  Make correctly failed with “No rule to make target”; no build result was
  invalidated. The follow-up executes `build/qw38-cuda-probe` inside the pinned
  CUDA container, its actual runtime environment.

### 2026-08-30T17:38:10Z — ORA-003 and EDU-023 accepted

- Added a reproducible setup script, exact 18-shard/source/environment contract,
  fail-closed verifier, eager/offload runner, diagnostic tap collector, fixture
  freezer, complete 238-record checked-in evidence manifest, and five focused
  tests. The 20 MB raw tap blob and full logits remain in ignored local evidence
  storage and are authenticated by committed SHA-256 values.
- Added handbook Chapter 37 and linked it from the root and handbook indexes and
  source ledger. It explains original BF16 versus quantized GGUF, Safetensors
  and shards, eager execution, GPU/CPU/disk offload, hooks and taps, raw-token
  isolation, logits, greedy selection, fixture equality, MTP/vision exclusions,
  measured resource use, failure history, and the exact proof boundary. Every
  performance-looking value is labeled as a feasibility measurement, not a
  benchmark claim.
- `uv run ruff format .` and `uv run ruff check .` passed. Five focused tests
  passed, then clean normal and diagnostic builds and all 113 pytest tests
  passed in 185.49 seconds. JSON parsing, shell syntax, Markdown local links,
  and `git diff --check` passed.
- The pinned CUDA 13.0.2 image rebuilt, all host products and the SM120 probe
  compiled inside it, and the probe ran on the RTX 5090: compute capability
  12.0, 33,671,348,224 total bytes, and 33,139,458,048 free bytes.
- Marked ORA-003 and EDU-023 done. No numeric tolerance was frozen or loosened.
  ORA-004 is now the next task: align official, llama.cpp, and Quartz stable taps,
  diagnose each boundary, and freeze the immutable scalar admission tolerances.

### 2026-08-30T17:39:58Z — ORA-004 three-authority alignment started

- Commit `05dfeb3` (`feat: add Transformers semantic authority`) is present on
  both `main` and `origin/main`; the worktree was clean before this task began.
  Began ORA-004 and added EDU-024 before implementation.
- The initial alignment audit found that Quartz already emits all conceptual
  scalar boundaries, but its old one-filter-per-process capture wrapper cannot
  collect a complete two-token trace efficiently. The diagnostic sink supports
  wildcard filters; ORA-004 must add a multi-tensor writer rather than rerun the
  roughly 36-second scalar model once per tap.
- Pinned llama.cpp's public evaluation callback already exposes Qwen3.5 graph
  tensors such as `model.input_embed`, `attn_norm`, `attn_residual`,
  `attn_post_norm`, `ffn_out`, `post_ffn`, `result_norm`, and `result_output`,
  plus detailed GDN/attention nodes. ORA-002 deliberately deferred wiring these
  debug tensors; ORA-004 now owns a narrow callback adapter and exact name/shape
  inventory. No upstream source edit is required.
- Layout differences must be normalized explicitly before comparison. Examples:
  Transformers repeats each of 16 GDN Q/K heads three times for 48 value heads,
  while Quartz retains 16 unique heads; Transformers attention caches include
  the full history while Quartz's existing tap exposes the current row; and the
  upstream FFN activation hook observes SiLU(gate) before multiplication by the
  up branch while Quartz `ffn.activated` is the product. These are mapping tasks,
  not grounds to compare incompatible arrays or inflate a tolerance.
- The first callback-adapter compile failed under `-Werror` because
  `ggml_bf16_t` is a wrapper type and cannot be initialized from integer zero.
  Changed that temporary to value-initialization (`{}`); the failure occurred
  before linking or execution and is retained here.
- After that compile fix, linking exposed a formerly unused direct call to
  `ggml_backend_tensor_get`; the adapter previously inherited enough libraries
  for llama/logits only, but the backend symbol lives in `ggml-base` and the
  linker correctly rejected the missing direct dependency. Added `ggml-base`
  explicitly to the narrow adapter target rather than relying on transitive
  shared-library behavior.
- The first real llama.cpp callback inventory succeeded and retained 168
  selected tensors (5,010,944 canonical FP32 bytes) while choosing tokens 3649
  and 1277. It also showed that names such as `Vcur-3` can identify both a
  projected `[1024]` node and a reshaped `[256,4]` node, so the mapping key must
  include shape and cannot assume names are unique.
- The first aligned Transformers rerun added unique 16-head GDN Q/K views,
  convolution states, FFN products, and current attention cache rows: 272 finite
  taps and 21,626,720 bytes with unchanged full-logit hash
  `9b64105a1c7262271c85054ef30cd116e0af4e85a497e6ae24c478007ed97947`.
  Alignment then found four diagnostic-only Quartz registry omissions—grouped
  GDN value, log-decay, update beta, and gated-normalized output—and one missing
  official post-convolution view. Added these existing workspace/function views
  before comparing; no model arithmetic or production build path changed.
- While aligning persistent state, the audit found that Quartz's convolution
  buffer is physically and semantically channel-major (`[10240,4]`), but the
  diagnostic manifest had labeled the unchanged flat bytes as `[4,10240]`.
  Corrected the trace shape and contract; the convolution implementation and
  stored bytes were already channel-major, so no runtime arithmetic changed.
- The first reporting-only comparison stopped on an unequal tensor length, as
  required, but its generic metric error did not identify the boundary. Added
  position/layer/boundary and both lengths to this structural precondition so
  the mismatch can be diagnosed before any numeric metric is considered.
- The identified mismatch was the official first-token convolution function's
  four-position padded return versus Quartz's current-position output. The
  upstream model slices that return to the current sequence length immediately
  afterward. Added an explicitly derived `convolution_current` tap (last padded
  position for the warm-up call, direct output for cached updates) and retained
  the full upstream function result as separate evidence; no unequal arrays are
  compared.
- The first complete reporting pass produced 194 official/Quartz and 156
  llama.cpp/Quartz rows, but several same-GGUF cosine values were obviously
  incompatible with the near-identical enclosing residuals. Diagnosis found
  three mapping errors: official GDN value-associated channels use grouped head
  order while the GGUF projection/convolution storage is tiled; Quartz's
  `attention.query` is the raw split projection, not the normalized query; and
  Quartz's `attention.context` includes the sigmoid output gate while the mapped
  upstream tensors were pre-gate. Added the already documented 16-by-3 GDN
  permutation, derives raw query lanes from the packed official projection, and
  captures/maps post-gate attention context. No tolerance has been selected.
- The corrected report reduced every previously suspect GDN/context boundary to
  the expected neighborhood. Raw attention K still mapped to normalized K in
  both upstream authorities, and llama.cpp raw Q still mapped to `Qcur_normed`.
  Remapped official K to its projection, llama K to the 1,024-value pre-reshape
  node (shape disambiguates the repeated name), and llama Q to
  `Qcur_reshaped`. RoPE taps continue to use the normalized/rotated nodes.

### 2026-08-30T18:16:04Z — ORA-004, ORA-001, and EDU-024 accepted

- Added one-pass two-token Quartz bundle capture. The final diagnostic execution
  emitted 2,502 tensors and 383,393,792 bytes in 45.33 seconds on its first
  measured run, with maximum resident set 18,014,080 KiB. Its final raw hash is
  `96e14a3e29af2781a9a716ec913098f2b576d988e27ab9ff5d8c3ab548261b17`.
  The first attempted command was rejected before execution because it combined
  unconditional file removal with the run; switched to validated unique output
  paths and the checked-in reproducer now creates a fresh `mktemp` evidence
  directory without deleting previous evidence.
- Extended the public-API llama.cpp adapter with a selective evaluation callback
  and direct `ggml-base` dependency. It requests only named nodes at layers
  0/3/7/62/63 and global endpoints, disambiguates repeated names by shape,
  converts F32/F16/BF16 to canonical FP32, and emitted 180 tensors totaling
  5,305,856 bytes with hash
  `e950c76b04580d251ba2a9da5a0ba21cb73135202201f1ebd0696066ef0dc245`.
  It selected greedy tokens 3649 and 1277.
- The final Transformers eager capture emitted 286 finite selected taps totaling
  22,347,616 bytes with hash
  `99d47367f411786f4d5f483a0a927491e412eca119bf7d7dcf0805538b1ab164`.
  Observation-only additions exposed the current convolution position, unique
  GDN Q/K heads, persistent convolution states, FFN product, current KV rows,
  and post-gate attention context. Its complete logits remained byte-identical
  to ORA-003 and selected the same two greedy tokens.
- Structural comparison failures were resolved by semantic mapping rather than
  tolerance changes: padded versus current convolution output; grouped versus
  tiled 16-by-3 GDN value layouts; raw versus normalized attention Q/K; pre-gate
  versus post-gate attention context; and duplicate llama graph names at
  different shapes. Corrected the mislabeled native convolution state shape
  from `[4,10240]` to its actual channel-major `[10240,4]` without changing its
  bytes or arithmetic.
- The admitted fixture contains 194 official/Quartz rows and 156 independently
  visible llama.cpp/Quartz comparisons. All are finite. Same-GGUF global extrema
  are minimum cosine 0.99956792, maximum RMS 0.14392873, and maximum absolute
  error 1.68186188. Official-BF16 versus Q4 global extrema are minimum cosine
  0.98949384, maximum RMS 0.83560138, and maximum absolute error 15.82871628;
  the frozen gates remain per authority/layer/tap rather than using these broad
  global extrema.
- Froze 97 official/Quartz and 78 llama.cpp/Quartz tap identities before CUDA
  optimization. Maximum-absolute and RMS gates use 1.10 times the observed
  maximum rounded upward to two significant digits; cosine expands the observed
  distance from one by 1.10 then floors to six decimals. Non-finite counts must
  remain zero. Relative errors and first-failing indices remain reported, but
  maximum relative error is not gated because near-zero denominators make it
  unstable. Future paths may not regenerate or loosen these pins to admit
  themselves.
- All three authorities greedily selected 3649 then 1277. The smallest winner
  margin was the official position-0 margin of 0.0625. No greedy near-tie
  exception was required or stored; the immutable exception list is empty.
- Added a sequential one-large-GPU-process reproducer, six focused admission and
  provenance tests, and beginner Chapter 38 covering comparable/runtime-private
  boundaries, exact layout normalization, every metric, failed mappings,
  deterministic gate selection, greedy near-ties, immutability, and proof
  limits. Source/evidence and handbook/root indexes are reconciled.
- Reconciled the earlier scalar, trace, llama.cpp, and Transformers handbook
  chapters at the commit boundary. Their original gate-specific proof limits
  remain explicit, while dated or historical wording now points beginners to
  Chapter 38 instead of incorrectly presenting completed oracle work as future.
- At `2026-08-30T18:21:20Z`, the first focused commit-boundary pytest command
  named a nonexistent `tests/test_docs.py`; pytest exited 4 before running any
  test. No product result failed. Replaced that mistaken aggregate name with the
  actual authority documentation/provenance test modules and the repository's
  explicit Markdown-link check below.
- Clean normal and diagnostic builds passed, followed by all 119 pytest tests in
  166.24 seconds. The pinned CUDA 13.0.2 image rebuilt every host product and the
  SM120 probe; the RTX 5090 reported compute capability 12.0,
  33,671,348,224 total bytes, and 33,139,458,048 free bytes. Ruff, JSON, shell
  syntax, Markdown local links, and diff whitespace checks are the final commit
  boundary checks.
- Marked ORA-004, umbrella ORA-001, and EDU-024 done. The scalar path is now an
  admitted numeric oracle. CUD-001—the first CUDA Q4_K/Q6_K decode MMV—is the
  next implementation task and must use these frozen gates without loosening.

### 2026-08-30T18:23:10Z — CUD-001 CUDA decode MMV started

- Marked CUD-001 and discovered documentation task EDU-025 in progress before
  kernel edits. The admitted boundary is deliberately narrow: BF16 activation
  input, transient per-32-value signed Q8 staging, Q4_K/Q6_K packed weights,
  one FP32 output per matrix row, and CUDA-stream ordering on SM120.
- The first implementation will retain readable unpacking and an unfused
  scalar-equivalent diagnostic boundary. Tiled multi-prompt MMQ, model weight
  residency, fusion, and row-bucket tuning remain CUD-002 and later tasks.
- Inspected pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` for the MIT-licensed Q8_1 staging
  and Q4_K/Q6_K vector-dot technique. Quartz will implement a focused local
  layout and kernel rather than copy its generic dispatch machinery; exact
  source paths and the adapted boundary must be added to `docs/sources.md`.
- At `2026-08-30T18:25:58Z`, the first four-case device diagnostic stopped on
  Q6_K `17x256`: transient Q8 bytes/scales matched exactly and maximum absolute
  error was only `4.57763672e-05`, but the draft `2e-5` relative gate observed
  `2.84627913e-05`. Diagnosed the same unstable-small-denominator policy already
  established by ORA-004. Relative error remains reported; CUD-001 admission
  uses pre-optimization maximum-absolute and RMS limits calibrated across all
  boundary cases, plus exact Q8 staging and zero non-finite values.
- At `2026-08-31T05:59:05Z`, the first documentation-connection test compared
  the lowercase phrase `matrix-vector multiplication` against the chapter's
  sentence-initial capitalized spelling and failed before the GPU test (which
  correctly skipped without its opt-in variable). Made the vocabulary check
  case-insensitive; no handbook or implementation claim changed.
- At `2026-08-31T05:59:27Z`, the corrected check exposed a second test-only
  vocabulary mismatch: it required the literal word `proof`, while Chapter 39
  consistently calls the same concept an evidence `boundary`. Changed the test
  to require the chapter's actual beginner vocabulary.
- At `2026-08-31T06:03:55Z`, adding the host project's `-Wpedantic -Werror` pair
  to NVCC produced hundreds of errors for NVCC-generated `# <line>` directives
  and CUDA headers, not Quartz source. Retained `-Wall -Wextra -Werror`, disabled
  exceptions/RTTI and FMA contraction, and omitted only `-Wpedantic` from CUDA
  host compilation. The ordinary C++ build keeps its full original flags.

### 2026-08-31T06:05:47Z — CUD-001, BLD-002, and EDU-025 accepted

- Added a restricted C++17 SM120 primitive with explicit device-pointer and
  stream ownership. One kernel converts BF16 activations into local 32-value Q8
  scratch blocks; a second assigns one warp per output row, decodes Q4_K/Q6_K
  bytes in place, and reduces FP32 lane sums. Invalid pointers, zero dimensions,
  unknown kinds, and columns outside the 256-value block contract fail before
  either kernel launches.
- Froze the local source hashes, CUDA 13.0.2 flags, activation staging rule, and
  pre-optimization numeric gates in `pins/cuda_quant_contract.json`. The local
  Q8 layout is 36 bytes with an FP32 scale and does not claim binary compatibility
  with llama.cpp's FP16-scale `block_q8_1`.
- Four deterministic Q4_K/Q6_K cases cover 17/257 rows and 256/512 columns,
  including partially occupied final CUDA blocks. All transient Q8 scales and
  integers matched the host reference exactly. Global observed maximum absolute
  error was `0.000228881836`; global maximum RMS was `0.000148385821`; all values
  were finite and passed the frozen `3e-4` absolute and `2e-4` RMS gates.
- Each case ran three warm-ups and 30 synchronized CUDA-event samples. Observed
  mean diagnostic times were about 0.0062–0.0083 ms and are explicitly not
  throughput claims. The compact measured record is
  `fixtures/cuda_quant_mmv.json`.
- The opt-in focused pytest executed on the RTX 5090 and passed both structural
  and device tests. Clean host and diagnostic builds passed; the full ordinary
  suite passed 120 tests with the exclusive CUDA test skipped in 178.81 seconds.
  A clean pinned container rebuild compiled every host product, the SM120 probe,
  the CUDA primitive, and its diagnostic with the frozen flags; the subsequent
  opt-in GPU pytest passed 2 tests.
- Added beginner Chapter 39 and reconciled root/handbook/source indexes. Marked
  BLD-002, CUD-001, and EDU-025 done. CUD-002 quantized tiled prompt MMQ is the
  next delivery-plan task; TRC-004 CUDA taps depend on this completed primitive
  but become meaningful at a model-stage boundary rather than this standalone
  row result.

### 2026-08-31T06:08:02Z — CUD-002 tiled prompt MMQ started

- Marked CUD-002 and discovered documentation task EDU-026 in progress before
  source edits. The admitted boundary is a BF16 prompt matrix with any positive
  row count, shared packed Q4_K/Q6_K weights, token-major FP32 output, and the
  same transient Q8 semantics already frozen by CUD-001.
- The first tile assigns one warp to one output-weight row and four prompt rows.
  Each lane decodes a packed weight once and applies it to up to four staged
  activations, making weight reuse explicit while retaining the readable
  warp-tree reduction. Partial prompt and output tiles must be covered directly.
- CUD-002 does not choose final production tile buckets or claim peak prefill
  throughput. Those require full model shapes and profiler evidence in OPT-001
  and later tuning tasks.
- At `2026-08-31T06:10:17Z`, the first device run stopped at Q4_K
  `5x257x512`. Staging, finiteness, and layout were exact, but maximum absolute
  error `0.000427246094` and RMS `0.000204815471` narrowly exceeded the reused
  MMV ceilings. This is a new reduction boundary, so CUD-002 will freeze its own
  pre-optimization `5e-4` absolute and `2.5e-4` RMS ceilings after running every
  Q4_K/Q6_K prompt-tail case; the CUD-001 limits remain unchanged.
  The first mechanical threshold edit touched the earlier MMV return check;
  immediate source review caught and reversed it before compilation, then
  applied the new ceilings only to the MMQ result.

### 2026-08-31T06:16:39Z — CUD-002 and EDU-026 accepted

- Extended the admitted CUDA primitive with token-major BF16 prompt input and
  FP32 `[prompt_rows, output_rows]` output. Each 256-thread block owns eight
  output rows; each warp decodes one weight per lane and applies it to a
  four-prompt-row tile before fixed FP32 warp reductions.
- Added exact workspace sizing and fail-before-launch validation for null
  pointers, unknown quant kinds, zero dimensions, and columns outside the
  256-value packed-block contract. Tail predicates cover prompt rows and output
  rows independently without exposing padding as output.
- Frozen fixtures cover Q4_K and Q6_K at prompt counts 1, 3, 5, and 9; output
  counts 17 and 257; and column counts 256 and 512. Every transient Q8 scale and
  signed integer matched the host reference. All outputs were finite. Global
  maximum absolute error was `0.000427246094`, and global maximum RMS was
  `0.000204815471`, passing the frozen `5e-4` and `2.5e-4` MMQ gates without
  changing CUD-001's stricter MMV limits.
- Three warm-ups and 30 synchronized CUDA-event samples per case produced
  diagnostic means of roughly 0.0061–0.0124 ms. These small-shape measurements
  are retained for reproducibility and explicitly excluded from production
  prefill-performance claims.
- Added source-authenticated MMQ contract, compact evidence, pytest structural
  and opt-in GPU checks, beginner Chapter 40, and reconciled handbook/source
  indexes. Clean normal and diagnostic builds passed. The full ordinary suite
  passed 121 tests with one expected exclusive-GPU skip in 164.76 seconds. A
  clean pinned CUDA 13.0.2 rebuild passed, followed by all three opt-in RTX 5090
  tests.
- Marked CUD-002 and EDU-026 done. Delivery gate 6 now has admitted decode MMV
  and tiled prompt MMQ primitives. GDN-001, exact one-token CUDA GDN with atomic
  state commit, is the next implementation-plan task.

### 2026-08-31T06:19:46Z — GDN-001 one-token CUDA state staging started

- Marked GDN-001 and discovered documentation task EDU-027 in progress before
  source edits. The focused boundary begins after learned projection: one
  10,240-channel convolution input is split into Q/K/V after width-four causal
  convolution and SiLU, then precomputed decay/beta drive the exact recurrent
  mutation order at the production 16-key-head/48-value-head/128-lane shape.
- Atomicity is explicit rather than inferred from stream ordering. A prepare
  operation reads committed convolution/recurrent state and writes separate
  candidate buffers plus output. A distinct commit operation copies candidates
  only after the caller has synchronized, checked errors, and chosen to accept
  the token. Invalid preparation and deliberate cancellation must leave the
  committed bytes unchanged.
- Learned projections, gated RMSNorm, output projection, the FFN branch, and the
  64-layer transaction remain scheduler/session work. GDN-001 must not describe
  a focused arithmetic state commit as a complete request commit.

### 2026-08-31T06:29:12Z — GDN-001 and EDU-027 accepted

- Added production-shape one-token CUDA convolution and recurrent kernels. The
  convolution prepares a distinct width-four history and SiLU output. One
  128-thread recurrence block per value head preserves 16-to-48 head reuse,
  sequential FP32 key-lane accumulation, decay/prediction/delta/update/read
  order, and the exact 786,432-value production state layout.
- Added separate prepare and commit entry points. Prepare rejects aliased
  candidate/committed state before launch and never changes committed state or
  frontier. The one-block commit copies both state families, synchronizes all
  256 threads, and advances the frontier last. The diagnostic proves deliberate
  cancellation at frontier 41 and byte-exact acceptance at frontier 42.
- Small `2/6/8/8` and production `16/48/128/128` cases compare candidate
  convolution state, recurrent state, convolution output, and recurrent output
  with the admitted scalar functions. All outputs were finite. Global maximum
  absolute error was `3.7252903e-9`; global aggregate RMS was
  `2.7502714e-10`, passing frozen `5e-8` and `5e-9` limits.
- Three warm-ups and 30 synchronized CUDA-event samples produced mean prepare
  times of about `0.00612 ms` for the small shape and `0.02055 ms` for the
  production state core. The evidence explicitly excludes learned projections,
  gated RMSNorm, output projection, FFN, and request-level transaction time.
- Added source-authenticated contract/fixture, ordinary and opt-in pytest gates,
  beginner Chapter 41, and reconciled handbook/source indexes. Clean normal and
  diagnostic builds passed. The full suite passed 122 tests with two expected
  exclusive-GPU skips in 164.24 seconds. A clean pinned CUDA rebuild compiled
  all products, and the combined explicit RTX 5090 primitive suite passed five
  tests.
- Marked GDN-001 and EDU-027 done. GDN-002, arbitrary chunked GDN prefill with
  internal 64-token scans and token-wise equivalence, is the next plan task.

### 2026-08-31T06:32:01Z — GDN-002 chunked CUDA prefill started

- Marked GDN-002 and discovered documentation task EDU-028 in progress before
  source edits. The API accepts token-major post-projection channels and gates,
  emits token-major recurrent outputs, and divides work into internal windows of
  at most 64 tokens while preserving one-token mutation order.
- The whole chunk retains GDN-001's candidate-state transaction. Window zero
  reads committed state; later windows continue from the same candidate state;
  none advances the committed frontier. Cancellation discards the final
  candidate regardless of how many internal windows completed.
- Admission must compare one chunk against repeated one-token CUDA prepare and
  commit, including counts immediately below, at, and above 64. This increment
  is correctness-first sequential recurrence inside each head block, not yet a
  parallel associative scan or tuned prefill claim.

### 2026-08-31T06:42:29Z — GDN-002 chunked CUDA prefill admitted

- Added arbitrary-token chunk preparation in [`cuda/gdn_step.cu`](cuda/gdn_step.cu).
  External chunks are divided into at-most-64-token windows while convolution
  rings and recurrent matrices continue through one whole-chunk candidate. The
  committed state and frontier remain byte-identical until explicit commit.
- Added a native SM120 diagnostic covering 3, 64, 65, and 129 tokens plus a
  production-state 65-token case. Every chunk output and ending state was
  byte-identical to repeated one-token CUDA prepare/commit. Against the scalar
  oracle, the worst absolute error was `2.23517418e-8`, aggregate RMS stayed
  below `1.79e-9`, and all non-finite counts were zero.
- Three warm-ups and 30 synchronized CUDA-event samples measured about
  `0.100 ms` for the small 65-token state core and `0.500 ms` for the
  production-state 65-token core. These figures exclude projections, norms,
  output projection, FFN, attention, and scheduler work.
- Added the source-authenticated contract, retained fixture, ordinary and
  opt-in pytest gates, beginner Chapter 42, and source/index reconciliation.
  JSON parsing and diff-whitespace checks passed. Clean normal and diagnostic
  builds passed; the full suite passed 123 tests with three expected
  exclusive-GPU skips in 164.55 seconds. A clean pinned CUDA rebuild and the
  combined explicit quantization/GDN suite passed all seven tests.
- Marked GDN-002 and EDU-028 done. ATN-001, grouped-query CUDA attention with
  partial RoPE and focused layers 3, 7, and 63, is the next plan task.

### 2026-08-31T06:45:22Z — ATN-001 CUDA decode attention started

- Marked ATN-001 and discovered documentation task EDU-029 in progress before
  CUDA source edits. The minimal boundary consumes already projected FP32 Q/K/V
  and query gates, performs per-head normalization and first-64-lane RoPE, and
  maps each group of six production query heads to one KV head.
- The production cache must use two-byte BF16 rows. Prepare will stage only the
  current candidate K/V row, read committed history only through the declared
  position, and leave the committed frontier unchanged. Commit publishes the
  candidate row before advancing the frontier; cancellation discards it.
- Admission will cover the frozen semantic layers 3, 7, and 63, future-row
  sentinel causality, grouped-head boundaries, partial-RoPE lanes, scalar/device
  metrics, malformed inputs, and synchronized RTX 5090 timings. ATN-002 retains
  arbitrary causal prefill and the 131,072-token capacity boundary.

### 2026-08-31T06:51:07Z — ATN-001 two-byte KV tolerance diagnosis

- The first native device run passed CUDA-versus-BF16-aware reference metrics,
  candidate isolation, exact BF16 row conversion, commit order, and all finite
  checks. Adding a comparison with CPU-003's all-FP32 synthetic cache then failed
  its `3e-6` absolute/`1e-6` RMS transcription limits: layer 3 measured about
  `1.80e-5` absolute and `9.25e-6` RMS after the required BF16 key row rounding.
- This is a real storage-policy difference, not a CUDA arithmetic regression.
  V1 explicitly requires two-byte KV storage, while CPU-003's tight fixture
  verifies a scalar equation with FP32 cache rows. No CPU-003 tolerance is being
  changed. Device admission will additionally apply ORA-004's already-frozen
  actual-model `layer.3.attention.context` boundary (`0.051` absolute, `0.0016`
  RMS, `0.999424` cosine), which is stricter than the corresponding layer 7 and
  63 context gates. The failed all-FP32 synthetic admission remains recorded.

### 2026-08-31T06:57:49Z — ATN-001 CUDA decode attention admitted

- Added the post-projection single-token CUDA attention boundary with direct
  per-head RMSNorm scales, partial first-64-lane RoPE, 24-to-4 grouped-query
  head mapping, causal scaled scores, stable FP32 softmax, query output gating,
  and BF16 K/V cache rows.
- Prepare writes a distinct candidate row and leaves every committed cache byte
  plus the frontier unchanged. Commit replaces only the declared position's K/V
  bytes, synchronizes, and publishes the new frontier last. A future committed
  row contains large sentinels and is excluded by the causal position bound.
- Layers 3, 7, and 63 passed the inspectable fixture, and the exact production
  `24 × 4 × 256` shape passed. The production case measured `1.1920929e-7`
  maximum error and `7.15209136e-9` RMS against the BF16-aware reference. Its
  all-FP32 scalar comparison measured `2.05144286e-4` maximum error,
  `1.72939599e-5` RMS, and `0.999999981848` cosine, inside the frozen ORA-004
  layer-3 context gate.
- Three warm-ups and 30 synchronized CUDA-event samples averaged about
  `0.0328 ms` for the production-shape four-row attention core. This excludes
  learned projections, output projection, residual/FFN work, graph launch, and
  long-context traffic.
- Added the source-authenticated contract, retained device fixture, ordinary and
  opt-in pytest gates, beginner Chapter 43, and source/index reconciliation.
  Clean normal/diagnostic builds passed; the complete suite passed 124 tests
  with four expected exclusive-GPU skips in 164.38 seconds. A clean pinned CUDA
  rebuild compiled every product, and the combined explicit primitive suite
  passed all nine tests.
- Marked ATN-001 and EDU-029 done. ATN-002, arbitrary memory-bounded causal
  attention prefill and the 131,072-token capacity boundary, is the next task.

### 2026-08-31T09:17:26Z — ATN-002 causal attention prefill started

- Marked ATN-002 and discovered documentation task EDU-030 in progress before
  CUDA edits. The chunk API will accept token-major projected Q/K/V/gates,
  preserve earlier candidate K/V rows across the chunk, and emit token-major
  contexts in strict causal order.
- The implementation will reuse one current-token normalization workspace and
  one score slab sized `query_heads × (start_position + token_count)`. It will
  never allocate the quadratic `[chunk_tokens, context_tokens]` score matrix.
  This correctness-first launch sequence is memory-bounded but not yet a fused
  or tuned attention-prefill claim.
- Whole-chunk prepare must leave committed cache/frontier unchanged; later chunk
  tokens read prior candidate rows, and commit copies all candidate rows before
  publishing the final frontier. Admission includes chunk-versus-repeated-token
  equality, positions crossing arbitrary chunk boundaries, overflow rejection,
  and a real production-shape execution at position 131,071 with a 131,072-row
  BF16 cache. MEM-001 remains separate because it measures the complete model,
  graphs, allocator overhead, and required free reserve.

### 2026-08-31T09:33:05Z — ATN-002 causal attention prefill admitted

- Generalized the single-token boundary into arbitrary token-major chunks.
  Tokens launch in strict order, reuse current-token normalization buffers and
  one `query_heads × maximum_context` score slab, and read earlier rows from
  either committed prefix or the same provisional candidate chunk. No
  `chunk_tokens × context_tokens` score rectangle is allocated.
- Whole-chunk outputs, BF16 candidate rows, final committed caches, and frontier
  were byte-identical to repeated one-token CUDA prepare/commit for 3-token and
  9-token inspectable cases and a production-shape 9-token case. Prepare left
  committed cache/frontier unchanged; invalid zero-length, overflow, and
  out-of-capacity boundaries fail closed.
- Allocated a real production one-layer cache with 131,072 rows: 512 MiB for K/V
  plus a 12 MiB FP32 score slab. Position 131,071 executed on the RTX 5090,
  emitted finite output, preserved the prepare frontier, committed frontier
  131,072, and rejected position/chunk overflow. Device free memory after the
  diagnostic allocations was 32,585,809,920 of 33,671,348,224 bytes.
- Three warm-ups and 30 synchronized samples averaged about `0.369 ms` for the
  short-prefix production nine-token core. The correctness-first final-position
  call took about `871.98 ms`; this negative performance result is retained for
  later profiling/tuning and is not presented as release throughput.
- Added a source-authenticated contract, retained fixture, ordinary/opt-in
  pytest gates, beginner Chapter 44, and source/index reconciliation. Clean
  normal and diagnostic builds passed; the complete suite passed 125 tests with
  five expected exclusive-GPU skips in 164.50 seconds. A clean pinned CUDA build
  compiled every product, and all 11 explicit primitive tests passed together.
- Marked ATN-002 and EDU-030 done. SCH-001, the hybrid 64-layer CUDA scheduler
  and FP32 logits, is the next delivery-gate task. MEM-001 remains pending until
  complete-model post-graph allocation and reserve evidence exists.

### 2026-08-31T11:00:45Z — CUD-003 scheduler prerequisite discovered

- Read the typed real-artifact schema, scalar mixer/layer/token schedule, and all
  admitted CUDA boundaries before starting SCH-001. The real GDN and attention
  projection matrices are Q8_0, but CUD-001 intentionally admitted only Q4_K
  and Q6_K. A CUDA scheduler cannot execute the pinned artifact through the
  existing API, so CUD-003 was added before implementation and SCH-001 now
  depends on it.
- The same bounded prerequisite owns Q4_K embedding-row decode and the small
  visible operations between matrix products: BF16/direct-scale RMSNorm,
  residual publication, SwiGLU, per-head attention query/gate splitting, GDN
  gate conversion, grouped/tiled value mapping, and gated recurrent output.
  These are required semantic boundaries, not speculative fusion work.
- Admission will compare every operation with the retained scalar equations,
  preserve Q8_0 block decoding and transient activation bytes, cover production
  dimensions and layout boundary indices, and retain synchronized device timing.
  CUD-003 does not upload the complete model, schedule layers, allocate session
  state, produce logits, or claim end-to-end speed; those remain SCH-001.

### 2026-08-31T11:26:17Z — CUD-003 scheduler prerequisite admitted

- Added resident GGUF Q8_0 decoding to the existing MMV/MMQ boundary without
  changing its temporary FP32-scale Q8 activation layout. Two MMV shapes and a
  three-row MMQ case preserved temporary activation bytes and stayed below the
  frozen quantized projection gates; the largest Q8_0 error was
  `7.62939453e-5` absolute and `2.69527864e-5` RMS.
- Added exact BF16 embedding-row decode plus production-sized BF16 RMSNorm,
  residual add, SwiGLU, and FP32-to-BF16 publication. The three exercised
  pointwise operations matched BF16-aware references exactly and averaged
  about `0.0624 ms` together after three warm-ups across 30 synchronized
  CUDA-event samples. This is retained as component evidence, not scheduler
  throughput.
- Added exact packed per-head attention query/gate splitting, folded GDN gate
  preparation, grouped recurrent output gating, and the real artifact's tiled
  value-head entry into GDN. Production gate conversion measured
  `2.98023224e-8` maximum and `1.22614319e-8` RMS error; tiled and already
  admitted grouped recurrence executions produced byte-identical output and
  recurrent state.
- Re-authenticated the shared quant and GDN sources, then reran their existing
  one-token and chunk gates to ensure the new formats/layout entry did not
  regress prior behavior. Clean normal and diagnostic builds passed; the full
  suite passed 126 tests with six expected exclusive-GPU skips in 164.63
  seconds. A clean pinned CUDA build compiled every product, and all seven CUDA
  diagnostics passed together, including the real 131,071 attention position.
- Added the source-authenticated contract, retained fixture, ordinary and
  opt-in pytest gates, beginner Chapter 45, and source/index reconciliation.
  Marked CUD-003 and EDU-031 done. SCH-001, complete resident model upload,
  hybrid 64-layer scheduling, and FP32 logits, is now unblocked and next.

### 2026-08-31T11:36:47Z — SCH-001 hybrid CUDA scheduler started

- Marked SCH-001 and the newly discovered documentation task EDU-032 in
  progress before implementation. The scheduler will copy the exact admitted
  18,973,870,432-byte mapped GGUF into one device allocation, then remap all
  typed tensor views by checked byte offset. This keeps canonical row bytes and
  tensor padding intact while avoiding 851 independent allocations.
- A capacity-bounded CUDA session will own all 48 GDN states and 16 BF16
  attention caches. One reusable workspace will carry BF16 residuals and FFN
  activations, FP32 projection/recurrent/attention intermediates, transient Q8
  blocks, per-layer candidate state, and FP32 vocabulary logits.
- Decode admission starts with two zero-state tokens so every layer variant,
  recurrent/cache continuation, final norm, all 248,320 logits, and greedy
  choices are exercised. The diagnostic will compare full CUDA logit rows with
  the retained scalar implementation and report absolute, relative, RMS,
  cosine, non-finite, first-failing-index, and top-logit differences under the
  frozen pre-CUDA policy. Atomic request rollback, 128K simultaneous fit,
  tracing, graphs, and optimized prefill remain their existing later tasks.

### 2026-08-31T11:43:58Z — SCH-001 first full-logit negative result

- The first complete two-token device execution crossed all 64 layers and
  produced the required greedy continuation `3649, 1277` in about `64 ms` per
  token. It uploaded the exact 18,973,870,432-byte artifact in about `1.56 s`
  and left roughly 13.997 GB device memory free at capacity two.
- Full-row comparison did not pass the frozen same-GGUF logit envelope. Against
  the scalar path, token 0 measured `0.230289` maximum error, `0.0506235` RMS,
  and `0.999561` cosine. Direct comparison with pinned llama.cpp reduced token
  0 maximum error to `0.209277`, but RMS/cosine still failed; token 1 measured
  `0.233516` maximum, `0.0394132` RMS, and `0.999796` cosine. No non-finite
  values occurred. The harness and failed result are retained; no tolerance was
  changed.
- Diagnosis: every projection, including resident Q8_0 weights, used the
  transient activation quantizer. The approved CUD-001 design requires that
  staging for Q4_K/Q6_K, while Q8_0 was added later only to unblock the real
  scheduler. SCH-001 will test a direct BF16-input Q8_0 row dot so its already
  eight-bit resident weights do not incur a second eight-bit rounding. Q4_K and
  Q6_K continue to use the admitted transient-Q8 path.

### 2026-08-31T12:13:55Z — SCH-001 hybrid CUDA scheduler admitted

- Added one checked device copy of the canonical 18,973,870,432-byte GGUF and
  remapped all 851 typed views by their authenticated mapped-file offsets. The
  move-only resident model, capacity-bounded session, and reusable workspace
  own their allocations explicitly and release them without exceptions.
- Implemented the literal 64-layer schedule: 48 GDN and 16 attention mixers,
  each followed by its SwiGLU FFN, then final RMSNorm and the complete 248,320
  row FP32 output projection. Two real tokens advanced frontier 0 to 2 and
  produced exact scalar greedy continuation `3649, 1277`.
- Retained FP32 only for the numerically sensitive residual accumulator while
  keeping normalized projection inputs, mixer output-projection inputs, and
  SwiGLU activations in BF16. Q4_K/Q6_K retain transient-Q8 activation staging;
  Q8_0 resident weights consume BF16 directly. This resolved the earlier
  full-logit failure without changing a frozen tolerance.
- Both complete scalar-device logit rows passed: token 0 measured `0.138223`
  maximum, `0.0272043` RMS, and `0.999872` cosine; token 1 measured `0.161953`,
  `0.0321326`, and `0.999864`. All 5,120 values at layer 0, layer 3, layer 63,
  and final norm passed their immutable layer-specific absolute/RMS/cosine and
  finite-count gates at both positions.
- Preserved the independent same-GGUF result separately. Both llama.cpp greedy
  choices match and absolute/RMS limits pass, but token 1 cosine
  `0.999828237` is about `0.0000028` below the scalar-derived independent
  envelope. It remains a named negative result for quality evaluation; it was
  not used to widen or replace the scalar-device admission gate.
- The clean run measured about `1.58 s` for canonical upload and `60.7/60.3 ms`
  for diagnostic decode tokens. At capacity two, resident model, 158,990,336
  session bytes, and 4,769,472 workspace bytes left 13,996,654,592 device bytes
  free. These are correctness measurements, not tuned throughput or MEM-001.
- Added the source-authenticated contract, retained fixture, ordinary/opt-in
  pytest gates, beginner Chapter 46, and source/index reconciliation. Clean
  normal and diagnostic builds passed; the full suite passed 127 tests with
  seven expected exclusive-GPU skips in 164.14 seconds. A clean pinned CUDA
  build compiled every product, and all eight CUDA diagnostics passed together.
- Marked SCH-001 and EDU-032 done. SES-001, exact common-prefix synchronization
  and reuse over the admitted hybrid state, is the next task. Request-level
  atomicity remains SES-002, and simultaneous 128K allocation remains MEM-001.

### 2026-08-31T12:59:26Z — SES-001 exact prefix synchronization started

- Marked SES-001 and the newly discovered beginner-documentation task EDU-033
  in progress before source changes. Acceptance requires exact committed-state,
  hidden-vector, and full-logit equality between synchronization and execution
  from a fresh zero state.
- The admitted reuse boundary is deliberately narrow: an unchanged request is
  a no-op, and a request that appends to the complete committed token history
  evaluates only its suffix. A shorter or divergent request clears the session
  and deterministically replays from token zero. Keeping an approximately
  159 MB GDN snapshot at every token would make the 131,072-token product
  requirement impossible; SES-003 later adds explicit disk checkpoints instead.
- All requested token IDs and output sizes will be checked before reset or
  execution. This proves invalid-input preflight preservation; general atomic
  behavior under cancellation or CUDA failure remains the separate SES-002
  gate and is not silently claimed here.

### 2026-08-31T13:45:38Z — SES-001 exact prefix synchronization admitted

- Added an exact host token history to the move-only CUDA session and a
  synchronization result that reports common-prefix, reused-token,
  evaluated-token, and reset/replay counts. An unchanged request evaluates
  nothing; a pure append evaluates only its suffix; a shorter, divergent, or
  empty request clears all persistent CUDA state and replays the requested
  history from position zero.
- Kept the last 248,320 FP32 logits and 5,120-value FP32 hidden vector in
  session-owned host memory (about 1 MB). This discovered ownership requirement
  makes no-op output independent of scratch-workspace identity. Persistent GDN,
  attention, token, frontier, hidden, and logit state is compared byte-exactly;
  a small CUDA mismatch flag avoids copying the approximately 159 MB persistent
  state to the host merely to compare it.
- The capacity-three RTX 5090 diagnostic passed initial execution, append reuse,
  no-op reuse, divergent replay, shorter replay, empty reset, and whole-request
  invalid-token preflight. Append, divergent, and shorter results matched fresh
  execution byte for byte across persistent state, full logits, and the final
  hidden vector. Token ID `248320` was rejected before mutation and left state
  equal to the valid reference.
- Added the source-authenticated contract, retained fixture, focused native
  diagnostic, ordinary/opt-in pytest gates, beginner Chapter 47, and source and
  index reconciliation. `uv run ruff format .` completed; the full suite passed
  128 tests with eight expected exclusive-GPU skips in 166.50 seconds. A clean
  pinned CUDA build compiled all host products and nine CUDA diagnostics, and
  the complete opt-in CUDA test set passed 17 tests in 59.83 seconds. The
  prefix/full-scheduler regression alone passed four tests in 51.10 seconds.
- Marked SES-001 and EDU-033 done. SES-002, whole-request atomic
  eval/sample/commit behavior under cancellation and execution failure, is the
  next task. Checkpoint persistence remains SES-003 and 128K simultaneous fit
  remains MEM-001.

### 2026-08-31T17:40:47Z — SES-002 atomic CUDA evaluation started

- Marked SES-002 and newly discovered documentation task EDU-034 in progress
  before source changes. The admitted implementation boundary is the real CUDA
  scheduler; the host `Engine`/`Session` facade remains deliberately fail-closed
  until its later product integration gate.
- One token will prepare every GDN layer in a complete alternate state buffer
  and every attention layer in a candidate KV row. Only successful final output
  copies may swap the GDN buffers, publish the candidate KV rows, copy the token
  and outputs, and advance the frontier. A caller-supplied status poll between
  layers supplies both cancellation and deterministic injected-error evidence.
- Sampling will be a read-only operation over committed logits. The extra GDN
  transaction buffer is about 159 MB per active workspace, not per token;
  duplicating the 8 GiB 128K KV cache is rejected. Unpublished attention rows
  beyond the frontier are explicitly outside logical/checkpoint state, so exact
  comparison must inspect only committed rows.

### 2026-08-31T17:55:34Z — SES-002 atomic CUDA evaluation admitted

- Replaced layer-by-layer GDN mutation with 48 corresponding candidate slots in
  one reusable workspace allocation. Successful evaluation publishes the whole
  GDN transaction through two pointer swaps. All 16 attention layers likewise
  retain candidate key/value rows until output succeeds; committed cache state
  is now defined precisely as rows below the frontier. A final audit also moved
  device-to-host logits and hidden copies into workspace-owned host staging, so
  caller output buffers are published only after the state commit succeeds.
- Added a caller status poll at synchronized layer boundaries. Cancellation
  after layer eight and an injected internal error after layer 31 both returned
  their exact statuses with frontier one, unchanged caller outputs, and
  byte-exact committed state versus an untouched reference. Invalid token
  `248320` remained a preflight error.
  A successful retry advanced to frontier two and matched uninterrupted fresh
  execution exactly.
- Added read-only greedy sampling over session-owned committed FP32 logits. It
  selected token `1277` without changing the frontier or any compared state.
  Stochastic temperature/top-k/top-p sampling remains later product-sampler
  work and is not silently claimed by this correctness boundary.
- The reusable candidate increased the capacity-three workspace to 160,380,704
  bytes. A remeasured capacity-two scheduler used 160,380,608 workspace bytes
  and left 13,841,465,344 of 33,671,348,224 device bytes free. This is recorded
  MEM-001 input, not a 128K-fit result. A duplicate 8 GiB KV cache was avoided.
- Added the authenticated contract, retained fixture, focused native diagnostic,
  pytest gate, beginner Chapter 48, and reconciled prior scheduler/prefix claims.
  `uv run ruff format .` completed; the full suite passed 129 tests with nine
  expected exclusive-GPU skips in 166.20 seconds. A clean pinned build compiled
  every host product and ten CUDA diagnostics; the complete opt-in CUDA suite
  passed 19 tests in 64.01 seconds, while the atomic/prefix/full regression
  passed six tests in 59.20 seconds after the final host-output staging audit.
- Marked SES-002 and EDU-034 done. SES-003, atomic disk checkpoint save/restore
  of every logical state component and compatibility identity, is next.

### 2026-08-31T18:50:52Z — SES-003 atomic CUDA checkpointing started

- Marked SES-003 and newly discovered documentation task EDU-035 in progress
  before source changes. The versioned little-endian checkpoint will carry the
  pinned model SHA-256, a fixed state-layout compatibility hash, capacity and
  frontier, sampler configuration/RNG fields, exact token IDs, all GDN bytes,
  committed attention rows, and the last committed logits/hidden vector.
- Save will write and close an adjacent temporary file, authenticate its framed
  bytes, append the digest, and publish by same-directory rename. Restore will
  validate magic, version, identities, section arithmetic, token bounds, exact
  file size, and payload digest before changing logical session state.
- Acceptance will compare a restored session byte-for-byte with its source,
  then compare their next-token continuation. Corrupt and incompatible files
  must fail without changing an already valid target. The file stores only KV
  rows below the frontier; unwritten capacity is not conversation state.

### 2026-08-31T19:09:54Z — SES-003 atomic CUDA checkpointing admitted

- Added checkpoint-v1 little-endian framing with fixed `QW38CKP1` magic,
  version/header sizes, the pinned model SHA-256, a state-layout compatibility
  SHA-256, capacity/frontier, five sampler fields, and exact section sizes. The
  payload contains u32 tokens, every FP32 GDN convolution/recurrent byte,
  committed BF16 KV prefixes, and last committed FP32 logits/hidden state.
- Save streams device state through a bounded 1 MiB host buffer to an adjacent
  `.tmp`, closes it, authenticates header plus payload, appends the 64-character
  SHA-256 footer, syncs the file, renames it over the destination, and syncs the
  parent directory. The diagnostic observed the destination and no temporary
  file after success.
- Restore validates magic/version, model/layout identities, capacity/frontier,
  sampler bounds, section arithmetic, exact total size, payload digest, and all
  token IDs before changing logical state. GDN restores through the existing
  transaction candidate and frontier publishes last. Corrupt-payload and
  incompatible-layout files were rejected with the already valid target still
  byte-exact to its uninterrupted reference.
- The two-token capacity-three checkpoint measured 160,004,416 bytes: 248-byte
  header, 8 token bytes, 7,864,320 convolution bytes, 150,994,944 recurrent
  bytes, 65,536 bytes each for keys and values, 993,280 logits bytes, 20,480
  hidden bytes, and a 64-byte digest footer. Restored sampler fields matched;
  evaluating token `1277` in uninterrupted/restored sessions produced identical
  frontier-three state and outputs.
- Added a bounded SHA-256 file-prefix helper, authenticated contract, retained
  fixture, native diagnostic, pytest gate, beginner Chapter 49, and source/index
  reconciliation. `uv run ruff format .` completed; the full suite passed 130
  tests with ten expected exclusive-GPU skips in 168.27 seconds. A clean pinned
  build compiled every host product and eleven CUDA diagnostics; all 21 opt-in
  CUDA tests passed in 67.44 seconds, and the focused checkpoint/atomic/prefix/
  scheduler regression passed eight tests in 63.70 seconds.
- Marked SES-003 and EDU-035 done. MEM-001, simultaneous 131,072-token allocation
  with every workspace/graph/overhead category and 1.5 GiB reserve, is next.

### 2026-08-31T19:12:15Z — MEM-001 pre-graph 128K fit measurement started

- Marked MEM-001 and newly discovered documentation task EDU-036 in progress
  before source changes. The diagnostic will simultaneously allocate the exact
  resident GGUF, a capacity-131,072 session with all 16 BF16 K/V caches, and the
  current atomic-evaluation workspace, then reconcile explicit owner bytes with
  CUDA's before/after free-memory measurements and process host RSS.
- The final acceptance condition explicitly says post-graph. OPT-003 has not
  implemented graph objects, so this increment must report graph bytes as
  unavailable and keep MEM-001 open even if the pre-graph reserve exceeds
  1.5 GiB. It is a retained physical-capacity measurement, not a semantic
  substitute for the later post-graph rerun.

### 2026-08-31T19:17:33Z — MEM-001 pre-graph 128K allocation passed

- Simultaneously uploaded the 18,973,870,432-byte resident model, allocated a
  capacity-131,072 session containing 158,859,264 GDN bytes and the complete
  8,589,934,592-byte K/V cache, and allocated the 172,963,328-byte atomic
  workspace. Exact Quartz owners totalled 27,895,627,616 device bytes.
- CUDA measured a 27,898,413,056-byte free-memory delta from the post-context
  baseline. The 2,785,440-byte difference from explicit owners is retained as
  allocator delta; the context itself occupied 531,890,176 bytes before Quartz
  allocations. Process host RSS measured 19,106,787,328 bytes after model upload
  and allocations.
- With all current owners live, 5,241,044,992 of 33,671,348,224 device bytes
  remained free, exceeding the 1,610,612,736-byte (1.5 GiB) reserve by
  3,630,432,256 bytes. Session capacity and independent GDN/KV arithmetic were
  exact, and allocation/zero-initialization executed on the RTX 5090.
- Added the authenticated provisional contract, raw fixture, focused native
  diagnostic and pytest gate, beginner Chapter 50, and source/index updates.
  EDU-036 is done. MEM-001 deliberately remains in progress: graph bytes are
  recorded as unavailable, and its stated post-graph acceptance condition
  cannot be satisfied until OPT-003 creates and measures the admitted graphs.
- `uv run ruff format .` completed; the full suite passed 131 tests with eleven
  expected exclusive-GPU skips in 163.26 seconds. A clean pinned build compiled
  every host product and twelve CUDA diagnostics; all 23 opt-in CUDA tests
  passed in 70.32 seconds.
- OPT-001 synchronized timing/NVTX attribution is the next executable task in
  the optimization sequence. After OPT-002 and OPT-003, this exact allocation
  diagnostic must rerun with graph ownership before MEM-001 may become done.

### 2026-08-31T19:30:00Z — OPT-001 and EDU-037 started

- Began synchronized runtime attribution before fusion or CUDA graph work. The
  existing scheduler exposes model-upload time and one whole-token CUDA-event
  duration, but cannot distinguish embedding, GDN mixer, attention mixer, FFN,
  logits, or atomic commit work.
- Added EDU-037 before source changes so asynchronous GPU timing, explicit
  synchronization, NVTX timeline ranges, category ownership, measurement
  overhead, and unavailable future boundaries are explained for a reader with
  no profiler background.
- The implementation will preserve atomic execution and make detailed timing
  opt-in. Queueing and idle gaps belong to the pending single-flight server;
  graph launch belongs to OPT-003. They must appear explicitly as unavailable,
  never as measured zero-duration work.

### 2026-08-31T20:00:28Z — OPT-001 and EDU-037 accepted

- Added opt-in synchronized CUDA-event attribution for embedding, all 48 GDN
  mixers, all 16 attention mixers, all 64 FFNs, logits/output copies, state
  commit, the complete token span, and the measured unassigned gap. Resident
  upload retains its CUDA-event measurement; steady monotonic CPU clocks now
  measure greedy sampling and checkpoint save/restore.
- Resolved the start-time assumption that all idle time belonged to the future
  server: the current GPU-stream idle gap is measurable as the non-negative
  token-span remainder. Server queueing remains a distinct unavailable category.
- Added balanced NVTX v3 ranges for loading, complete token execution, every
  attributed GPU category, sampling, and checkpoint persistence. Detailed event
  pairs are created only when a `RuntimeTimings` record is requested because
  instrumentation perturbs the schedule. Graph launch and server queueing are
  explicit unavailable values pending OPT-003 and SRV-001, not measured zeros.
- The focused RTX 5090 sample measured a 60.594017 ms token stream span. FFN was
  39.8940468 ms, GDN 13.7372789 ms, attention 4.34175968 ms, logits 2.43088007
  ms, commit 0.0804480016 ms, embedding 0.0100480001 ms, and the remaining gap
  0.0995559692 ms; the attributed sum equalled the total. Greedy sampling chose
  token 1277 and checkpoint persistence measured 609.965942 ms. This one sample
  is diagnostic attribution, not a benchmark distribution or speed claim.
- Retained two profiling environment negatives. The pinned image does not
  contain `nsys`. Nsight Compute 2025.3.1 connected to the diagnostic, then the
  host denied hardware-counter access with `ERR_NVGPUCTRPERM`; the instrumented
  executable still completed, but its perturbed 107.092094 ms run is not used
  as optimization evidence. OPT-002 must establish admitted profiler access
  before accepting a fusion.
- Added the authenticated timing contract, raw fixture, focused native/pytest
  diagnostic, beginner Chapter 51, source provenance, and handbook navigation.
  During final review, fixed cleanup of an active event pair on cancellation or
  event-record failure; focused atomic/timing CUDA regressions then passed.
- `uv run ruff format .` reported 60 files unchanged. The ordinary suite passed
  132 tests with twelve expected exclusive-GPU skips in 163.36 seconds. A clean
  pinned CUDA build compiled all four host products and thirteen native
  diagnostics; the complete opt-in suite passed 144 tests in 234.12 seconds.
  Post-review focused contracts passed five tests with five expected skips, and
  the real timing/atomic pair passed four tests in 10.50 seconds.
- Marked OPT-001 and EDU-037 done. OPT-002 profiler-led fusion is next; OPT-003
  stable-address CUDA graphs and the final post-graph MEM-001 rerun remain
  explicitly queued after it.

### 2026-09-01T05:29:23Z — OPT-002 and EDU-038 started

- Began OPT-002 with the clean OPT-001 scheduler and added EDU-038 before fusion
  source changes. The chapter must explain why adjacent operations may be
  combined, what remains visible, how the unfused reference is retained, and
  why profiler output is evidence rather than an instruction to optimize every
  warning.
- Confirmed the host NVIDIA driver has `RmProfilingAdminOnly: 1`. Running Nsight
  Compute as ordinary container user and as container root both reproduced
  `ERR_NVGPUCTRPERM`; adding only the `SYS_ADMIN` container capability admitted
  performance counters without changing the host driver setting.
- Profiled one 5,120-row Q8 MMV with Nsight Compute 2025.3.1. It took 74.98 us,
  reached 61.37% compute/memory throughput and 42.14% DRAM throughput, and was
  described as balanced. This rejects blindly fusing or rewriting the dominant
  FFN matrix multiplication in this task.
- Profiled the 5,120-element FP32-to-BF16 RMSNorm boundary and retained
  `build/opt002-rms-baseline.ncu-rep` as working evidence. Its one-block grid
  used one of 170 SMs, reported 0.02% compute throughput, 0.21% memory
  throughput, 16.66% achieved occupancy on its active SM, and a 159.39 us
  replay-measured duration. This justifies testing fusion of the preceding FFN
  residual add with the next layer's input normalization, while retaining the
  current two-kernel path for A/B correctness and timing.
- The first fused implementation made thread 0 perform both all 5,120 residual
  additions/stores and the ordered norm sum. It was bit-exact at logits, hidden,
  taps, complete session state, and greedy output, but 30 alternating samples
  measured 78.0371094 ms fused versus 63.2383232 ms unfused (0.81036222x).
  Rejected that work assignment. Follow-up keeps parallel residual writes and
  only the already-serial ordered norm sum on thread 0, preserving arithmetic
  while still testing whether removal of the second launch pays off.

### 2026-09-01T05:44:16Z — OPT-002 and EDU-038 accepted

- Implemented one admitted fusion: each layer's final FFN residual add also
  prepares the next layer's BF16 input norm. Layer zero retains its standalone
  input norm and layer 63 retains its final residual add, removing 63 launches
  per token. `PointwisePath::kUnfused` preserves the old path for diagnostics;
  fused is the production default.
- The revised kernel parallelizes residual additions/stores, synchronizes its
  block, then performs the same ordered FP32 sum and BF16 conversion as the
  unfused norm. Across 33 repeated tokens, fused and unfused logits, final
  hidden values, selected CUDA trace taps, complete session state, and greedy
  output were byte-exact.
- Three independent 3-warmup/30-sample alternating A/B runs measured speedups
  of 1.02321708x, 1.02286899x, and 1.02247679x. The retained raw run measured
  64.8529892 ms fused versus 66.3453064 ms unfused (1.02301073x), with all 30
  paired samples stored. This is a local fusion result, not a product or
  comparative benchmark claim.
- Nsight Compute profiled the admitted fused kernel at 45.54 us under nine-pass
  replay versus 159.39 us for the separate input RMSNorm profile. The targeted
  report records commands, environment, metrics, and the rejected balanced Q8
  MMV candidate. Replay durations are not treated as ordinary token timings.
  Nsight Systems remains unavailable in this image; OPT-001 NVTX ranges remain
  ready for a later profiling container.
- Added the authenticated fusion contract, raw fixture, focused native/pytest
  A/B gate, targeted profiler report, beginner Chapter 52, provenance entry,
  and handbook navigation. The first static fixture test failed because Python
  double-precision averaging differed by about 1.2e-5 ms from the diagnostic's
  FP32 running sum; widened only that fixture-consistency rounding check from
  `1e-7` to `5e-7` relative tolerance. No model/fusion correctness tolerance
  changed.
- `uv run ruff format .` reformatted one Python file. The ordinary suite passed
  133 tests with thirteen expected exclusive-GPU skips in 161.61 seconds. A
  clean pinned build compiled all four host products and fourteen native CUDA
  diagnostics; the complete opt-in RTX 5090 suite passed 146 tests in 244.85
  seconds.
- Final review added fail-closed validation for an out-of-range diagnostic path
  selector and verified that it leaves the frontier at zero. Six affected
  contract tests passed with six expected GPU skips; the final real fusion gate
  then passed both tests in 10.18 seconds.
- Marked OPT-002 and EDU-038 done. OPT-003 stable-address CUDA graphs is now the
  next executable task, followed immediately by the final post-graph MEM-001
  reserve measurement.

### 2026-09-01T06:50:53Z — OPT-003 and EDU-039 started

- Began stable-address graph work and added EDU-039 before implementation. The
  current full token cannot be captured once and replayed unchanged: RoPE,
  causal attention loop bounds, and K/V commit destinations depend on the host
  session frontier, while GDN state owners swap after atomic commit.
- Selected the 64 per-layer FFN branches as the first honest graph boundary.
  Their layer weights and workspace residual/projection addresses remain stable
  for the lifetime of a resident model/workspace pair, and OPT-002 already made
  the next-layer norm boundary stable and exact. Attention, GDN state ownership,
  output copies, cancellation polls, and atomic commit remain outside capture.
- The graph owner will capture and instantiate each FFN once, fail closed if
  replay is attempted with different model/workspace addresses, retain ordinary
  fused execution for comparison/fallback, expose graph-launch attribution, and
  report its measured allocation while graph objects are alive. This scoped
  graph design must pass end-to-end state/logit equality; it is not presented as
  a whole-token graph.

### 2026-09-01T07:08:29Z — OPT-003, MEM-001, and EDU-039 accepted

- Added move-only `SchedulerGraphs` ownership for 64 per-layer FFN graph
  definitions and executable instances. Creation captures on a non-blocking
  stream, instantiates, uploads, and cleans up the complete set on partial
  failure. Replay requires the exact bound resident-model and workspace object
  addresses and the admitted fused pointwise path; a mismatch is rejected before
  session mutation. The ordinary fused launch path remains the equivalence
  oracle and fallback.
- Kept token-varying embedding lookup, attention position/causal/KV work, GDN
  pointer publication, host copies, cancellation polling, and atomic commit
  outside capture. This is an intentional 64-FFN graph boundary, not a claim of
  whole-token capture. Beginner Chapter 53 explains capture, instantiation,
  upload, replay, stable pointer lifetime, dynamic exclusions, timing, ownership,
  and proof limits.
- Across the same 33-token frontier, graph and ordinary execution produced
  byte-exact FP32 logits, final hidden values, selected trace taps, complete GDN
  and KV session state, token/frontier state, and greedy output. After three
  warm-ups, 30 alternating pairs averaged 65.5711594 ms graphed versus
  65.9345779 ms ordinary, a 1.0055424x improvement. Two additional replicates
  measured 1.00516605x and 1.00516462x. The diagnostic attributed 0.149828002 ms
  of CPU submission time to 64 graph launches. Raw samples and exact-equality
  results remain in `fixtures/cuda_graph.json`; this local A/B is not a product
  throughput claim.
- Measured all 64 uploaded graph executables at 6,291,456 device bytes. The final
  simultaneous 131,072-token allocation was 18,973,870,432 resident-model bytes,
  8,748,793,856 session bytes including the independent 8 GiB KV cache,
  172,963,328 workspace bytes, and 6,291,456 graph bytes. With CUDA runtime and
  allocator ownership reconciled, 5,234,753,536 bytes remained free against the
  1,610,612,736-byte requirement, a 3,624,140,800-byte margin. MEM-001 is now
  admitted; Chapter 54 preserves the provisional pre-graph reading and explains
  GiB/GB units, each owner, arithmetic, reserve, and proof boundary.
- Preserved two finalization negatives. The first static memory test still summed
  only model, session, and workspace, so it failed after graph bytes became an
  explicit owner; adding the graph term fixed the stale test equation without a
  runtime change. The next run found that the new beginner chapter described the
  concepts but omitted exact contract phrases for GiB versus GB, resident model,
  and GDN state; the prose and measured pre/post graph comparison were made
  explicit rather than weakening the documentation gate.
- `uv run ruff format .` reformatted two Python files. The ordinary suite passed
  134 tests with fourteen expected exclusive-GPU skips in 161.56 seconds. A clean
  pinned CUDA 13.0.2 build compiled all four host products and fifteen native
  SM120 diagnostics. The complete RTX 5090 suite passed all 148 tests in 252.83
  seconds; the focused real graph/final-memory pair passed four tests in 9.99
  seconds.
- Marked OPT-003, MEM-001, and EDU-039 done. OPT-004 offline RTX 5090 row-bucket
  and chunk-size tuning is the next executable task.

### 2026-09-01T07:10:55Z — OPT-004 and EDU-040 started

- Began the pinned RTX 5090 dispatch sweep and added EDU-040 before changing
  production kernels. The existing quantized decode MMV always launches 256
  threads (eight output-row warps) per block, and prompt MMQ always reuses each
  decoded weight across four prompt rows. Those constants are correct but have
  not been selected from retained measurements.
- Scoped the sweep to the production Q4_K/Q6_K/Q8_0 projection boundary: compare
  candidate output-row warp buckets and prompt-row tile sizes at actual model
  widths, retain raw warmup/sample results and losing candidates, and check in a
  small explicit dispatch table. Numeric order within each output remains
  unchanged. Full-token chunk scheduling is not silently added to this kernel
  task and remains a separate scheduler concern.

### 2026-09-01T07:30:51Z — OPT-004 and EDU-040 accepted

- Specialized the quantized MMV kernel at 4, 8, and 16 output-row warps per
  block and the prompt MMQ kernel at 1, 2, 4, and 8 prompt rows per weight reuse
  tile. Candidate launch APIs reject all other values. The selected production
  table is a small ordered conditional, not a runtime autotuner; one warp still
  owns and reduces each output in the frozen arithmetic order.
- Swept the eight exact Qwen output-row shapes (48, 1,024, 5,120, 6,144, 10,240,
  12,288, 17,408, and 248,320) and every power-of-two prompt chunk from 1 through
  64. Each candidate received three warm-ups and 30 CUDA-event samples in each
  of three independent runs. The lowest cross-replicate arithmetic mean selected
  MMV warps 4/8/16/8/8/4/8/4 at those shapes and prompt tiles 1/2/4/8/8/8/8.
- Retained all 156 candidate/replicate means, including losers, in
  `fixtures/cuda_dispatch_tuning.json` and all 1,560 individual samples from the
  admitted run in `evidence/profiling/opt004-dispatch-sweep-raw.txt`. The 64-row
  prompt case averaged about 11.06 ms at tile 1 versus 4.17 ms at tile 8. Close
  MMV results remain visible: the 17,408-row case averaged about 0.1567 ms at four
  warps and 0.1561 ms at eight, so this is not generalized into a portable speed
  claim.
- Resolved an evidence-design negative before admission. The first sweep covered
  prompt sizes 1, 4, 16, and 64, but the draft table inferred a two-row bucket
  without measuring it. Expanded the diagnostic to 1, 2, 4, 8, 16, 32, and 64,
  discarded the incomplete draft evidence, and reran all three replicates. Also
  corrected one manual full-scheduler invocation that omitted its required llama
  logits argument; the proper real-model gate subsequently passed.
- Zero-filled synthetic weights isolate launch scheduling while retaining real
  shapes, instruction paths, allocation sizes, and memory traffic. They are not
  used as numeric authority. The nonzero quant diagnostic passed its frozen
  Q4_K/Q6_K/Q8_0 gates, and the focused real tuning/quant/full-scheduler suite
  passed seven tests in 51.19 seconds. Chapter 55 explains launch shapes, warps,
  buckets, tiles, selection, reproducibility, and this proof boundary for a new
  reader.
- `uv run ruff format .` reformatted one Python file. The ordinary suite passed
  135 tests with fifteen expected exclusive-GPU skips in 161.61 seconds. A clean
  pinned CUDA 13.0.2 build compiled all four host products and sixteen native
  SM120 diagnostics. The complete RTX 5090 suite passed all 150 tests in 263.17
  seconds, including graph replay and the post-graph 128K reserve gate.
- The first staged diff audit rejected one extra blank line at the end of the raw
  sweep file. Removed that line, updated its authenticated SHA-256, and reran the
  focused contract checks; no measurement value or source code changed.
- Marked OPT-004 and EDU-040 done. The profiler/optimization delivery sequence
  is complete; CLI-001 is the next executable product task.

### 2026-09-01T11:18:35Z — CLI-001 and EDU-041 started

- Began the first usable product surface and added EDU-041 before implementation.
  The current `qw38` binary intentionally exits 2, while public `Engine::open`
  validates/maps/tokenizes the artifact but does not upload CUDA weights and all
  public `Session` operations return `kUnimplemented`. The accepted CUDA model,
  session, workspace, graphs, prefix sync, eval, greedy sampling, and checkpoint
  implementations currently live only behind native diagnostics.
- Scoped the work as two connected boundaries: compile a CUDA-enabled production
  `Engine`/`Session` implementation while preserving the host-only validation
  build, then keep the CLI limited to chat messages, tokens, logits, and session
  methods. Token decoding and chat rendering require narrow public engine helpers;
  no tensor pointer or device layout may enter `src/cli.cpp`.
- The CLI will support interactive user turns, generation limits, reasoning
  enable/disable and effort, temperature/top-p/top-k/seed, stop at the admitted
  chat terminator, and explicit save/restore commands. HTTP compatibility and
  Codex use remain SRV-001–SRV-003; a terminal executable alone is not presented
  as an OpenAI-compatible model provider.

### 2026-09-01T11:42:18Z — CLI-001 smoke harness negative

- The first automated CUDA CLI smoke exited successfully after opening the
  model but produced no answer. Diagnosis: the test passed input to
  `subprocess.run`, but its `docker run` command omitted `-i`, so Docker did not
  attach that input to the container and `qw38` observed immediate EOF at its
  first `user>` prompt. Added Docker's stdin flag to the harness; no runtime or
  expected model output changed. The failed 59.69-second run remains recorded
  here and the corrected test must pass before CLI-001 can be accepted.

### 2026-09-01T11:45:09Z — CLI-001 checkpoint-size negative

- The corrected stdin-attached smoke generated exact `assistant> hello`, saved
  a checkpoint with the expected magic, and executed `/load`, but its final
  assertion reused the 160,004,416-byte size of SES-003's two-token fixture.
  This CLI prompt has a longer committed frontier, so its checkpoint contains
  more committed attention KV rows and is 161,118,596 bytes. Updated the test
  and handbook to treat size as frontier-dependent, and made successful
  interactive `/save` and `/load` commands print their resolved path so restore
  success is directly assertable. No checkpoint layout changed.

### 2026-09-01T11:41:26Z — MDL-003 and EDU-042 started

- Added the startup-authentication task before changing hashing code. Quartz's
  portable SHA-256 path authenticates the complete 18,973,870,432-byte artifact
  on every `Engine::open` and previously measured 56.603 seconds. It copies
  input through a byte-at-a-time update loop and runs a scalar compression
  function despite this Ryzen 9 9900X exposing `sha_ni`.
- A warm-cache `/usr/bin/sha256sum` control produced the identical pinned digest
  `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
  in 8.56 seconds (7.82 user, 0.73 system), establishing that full SHA-256 can
  be kept while removing most CPU cost. The selected boundary is runtime
  dispatch to the pinned container's accelerated cryptographic provider with
  the self-contained portable implementation retained as an explicit fallback.
  BLAKE3 and metadata-only identity are not introduced by this task.

### 2026-09-01T11:56:52Z — CLI-001, MDL-003, EDU-041, and EDU-042 accepted

- Added the CUDA production `Engine`/`Session` implementation and linked
  `build/cuda/qw38` while preserving the host validation product. A session now
  owns the resident model safely beyond the originating engine handle and owns
  its independent 131,072-token state, workspace, and 64 FFN graphs. Public
  helpers expose only chat messages, token IDs, decoded text, logits, sampling,
  evaluation, and persistence; the CLI never handles tensor/device layouts.
- Added exact inverse GPT-2 byte decoding and an incremental user-turn template.
  The interactive and one-shot CLI supports reasoning modes, greedy or seeded
  temperature/top-p/top-k sampling, generation/custom/model stops, reset,
  atomic checkpoint save/load, and exact token-history continuation. Pending
  stochastic RNG state is committed only after evaluation of its sampled token
  succeeds. A real RTX 5090 smoke produced exact `assistant> hello`, wrote a
  frontier-dependent 161,118,596-byte `QW38CKP1` checkpoint, restored it in the
  same process, and completed in 13.03 seconds.
- Preserved two implementation negatives in addition to the smoke-harness and
  checkpoint-size negatives above. The first CUDA engine compilation rejected
  the host-only `unavailable` helper as unused under `-Werror`; isolating that
  helper behind the host build condition fixed the ownership boundary. Review
  also found that the first stochastic sampler draft advanced persisted RNG
  state before token evaluation and held a raw resident-model pointer; pending
  sampler commit and shared immutable model ownership fixed both issues before
  admission.
- Replaced the byte-at-a-time portable-only SHA-256 default with runtime
  resolution of OpenSSL 3 EVP from the pinned container's `libcrypto.so.3`.
  OpenSSL selects the CPU implementation, including SHA-NI on this Ryzen 9
  9900X. The exact full-model digest remained
  `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`;
  warm-cache time fell from 56.603 to 8.47 seconds (6.682x). The block-copy
  portable implementation remains selectable and exact; missing provider/init
  falls back, while failures after provider initialization fail closed.
- Added top-level copy-paste interactive/resume instructions and beginner
  Chapters 56–57 covering runtime ownership, templates/tokens, atomic
  sample/eval, stops, persistence, cryptographic hashes, SHA-NI/provider
  dispatch, ZFS block integrity versus whole-file identity, and warm/cold cache
  limits. Contracts and structured fixtures authenticate both gates.
- `uv run ruff format .` completed with 64 Python files unchanged on the final
  formatting pass. The ordinary suite passed 141 tests with sixteen expected
  opt-in GPU skips in 162.25 seconds. A clean pinned CUDA 13.0.2 build compiled
  all four host products, the CUDA CLI, and sixteen native SM120 diagnostics.
  The complete RTX 5090 suite passed all 157 tests in 263.85 seconds.
- Marked CLI-001, MDL-003, EDU-041, and EDU-042 done. SRV-001, the single-flight
  HTTP server core, is the next executable product task; the CLI is usable now
  but is not presented as a Codex/OpenAI-compatible provider.

### 2026-09-01T12:04:17Z — SRV-001 and EDU-043 started

- Audited the server boundary after CLI admission. `src/server.cpp` remains a
  three-line fail-closed stub, there is no HTTP/JSON dependency, and the Makefile
  links only a host server. The CUDA `Engine` now supplies the exact one-session
  owner required by the gate, but Chat Completions and Responses parsing remain
  explicitly owned by SRV-002 and SRV-003.
- Scoped SRV-001 to a narrow Linux socket listener with bounded HTTP/1.1 parsing,
  one request per connection, explicit JSON responses, loopback default binding,
  graceful process shutdown, and public `GET /health` plus `GET /v1/models`.
  A reusable FIFO single-flight gate will admit at most one future GPU handler,
  measure queue delay, remove cancelled waiters, and wake all waiters on
  shutdown. Focused native concurrency tests will exercise this gate without a
  model; a real CUDA smoke must prove that the production server authenticates
  the model, allocates exactly one session, and serves both routes.
- Request bodies, Chat Completions, Responses, streaming, tools, API keys, and
  request-to-CUDA cancellation are not silently pulled into the core gate. The
  queue cancellation contract covers a client that disconnects while waiting;
  active generation cancellation is completed with the inference routes in
  SRV-002. The existing untracked `checkpoints/` directory is preserved and
  will be ignored as runtime state rather than committed.

### 2026-09-01T12:13:07Z — SRV-001 readiness-harness negative

- The first real CUDA server smoke allocated 27,110 MiB and printed a valid
  ephemeral listener at `127.0.0.1:42209`, but the test timed out before making
  requests. Diagnosis: text-mode `readline()` had pulled the readiness line into
  Python's internal buffer while `select()` watched only the empty kernel pipe;
  the timeout path then blocked reading stderr from the still-running server.
  Stopped the specifically named test container, retained the 137.51-second
  failure, and changed readiness detection to raw `os.read` bytes with no
  blocking stderr read. No server behavior or expected route payload changed.

### 2026-09-01T18:19:42Z — SRV-001 and EDU-043 accepted

- Replaced the fail-closed server stub with a bounded Linux IPv4 HTTP/1.1
  listener. The production CUDA binary authenticates the pinned GGUF, prepares
  resident weights, creates exactly one session, and only then reports ready.
  It serves exact `GET /health` and `GET /v1/models` control-plane responses;
  malformed requests, unknown routes, and unsupported methods return explicit
  400, 404, and 405 JSON errors. The 65,536-byte header cap, five-second receive
  timeout, one request per connection, loopback default, and graceful
  SIGINT/SIGTERM shutdown keep this first protocol boundary deliberately small.
- Added a reusable FIFO single-flight gate with stable ticket order, monotonic
  microsecond queue delay, arrival-depth capture, one active owner, queued
  cancellation, and shutdown wake-up. The native concurrency diagnostic measured
  a second-arrival depth of one, at least 30,000 microseconds of wait, and a
  maximum of one active owner; it also proved that cancelled waiters are removed
  and shutdown wakes a blocked waiter. Generation routes do not exist yet, so
  active CUDA cancellation and request-to-gate integration remain SRV-002 work.
- The corrected live CUDA smoke passed in 11.56 seconds: it observed one session,
  exact health/model payloads, every error route, an ephemeral loopback port,
  27,110 MiB of GPU use, and process exit zero after SIGTERM. The earlier
  137.51-second readiness-harness failure remains preserved above and in
  [`fixtures/server_core.json`](fixtures/server_core.json).
- Added beginner Chapter 58 explaining TCP sockets, addresses and ports,
  loopback, HTTP messages, CRLF framing, JSON and `Content-Length`, bounded
  parsing, control versus data planes, FIFO queueing, queue depth/delay,
  cancellation, one-session ownership, and graceful shutdown. Reconciled older
  tokenizer, atomic-eval, timing, and CLI chapters so they distinguish the now
  admitted control plane from still-pending Chat Completions and Responses.
- Verification before this acceptance update: `uv run ruff format .` left 65
  Python files unchanged; focused build/server/timing tests passed 17 tests with
  one expected opt-in skip; the ordinary suite passed 143 tests with seventeen
  expected GPU skips in 173.73 seconds. A clean pinned CUDA 13.0.2 container build
  compiled four host products, the native queue diagnostic, both CUDA products,
  and sixteen SM120 diagnostics with warnings as errors. The complete exclusive
  RTX 5090 suite passed all 160 tests in 275.58 seconds.
- Marked SRV-001 and EDU-043 done. SRV-002, the Chat Completions request,
  generation, streaming, and cancellation boundary, is the next product task.

### 2026-09-01T18:24:29Z — SRV-002, API-002, and EDU-044 started

- Audited the admitted control plane, public Engine/Session API, official Qwen
  template fixtures, CLI generation loop, and CUDA atomic-evaluation controls.
  No JSON library is present by design. Added API-002 before implementation so
  the strict JSON grammar, Unicode handling, nesting/body bounds, and canonical
  tool serialization have their own acceptance evidence rather than being
  hidden inside the HTTP handler.
- Scoped SRV-002 to `POST /v1/chat/completions` with the exact pinned model,
  system/developer/user/assistant/tool history, text content parts, reasoning
  controls, function definitions/calls/results and tool choice, temperature,
  top-p, top-k, seed, token limit, string/array stops, non-streaming JSON, and
  `text/event-stream` output. One request acquires the existing FIFO gate and
  synchronizes the one session to the rendered prompt, allowing exact-prefix
  reuse while divergent histories reset and replay through the existing engine.
- Active cancellation must cover both queue wait and CUDA work. A connection
  watcher will turn peer disconnect into one atomic flag; the public session
  boundary will pass that flag into prompt synchronization and per-token
  `EvalControl` polling. Atomic CUDA evaluation ensures an interrupted token
  does not publish partial GDN/KV state. Previously committed response tokens
  are harmless because the next request synchronizes to its own exact prompt.
- Explicitly kept Responses and `previous_response_id` in SRV-003. V1 text Chat
  Completions will reject image/audio/file content, structured output, `n != 1`,
  unsupported penalties/logprobs, legacy function fields, chunked request
  bodies, and unknown protocol controls instead of silently pretending support.
  Authentication, TLS, persistent connections, continuous batching, and
  concurrent GPU sessions remain outside the approved v1 boundary.

### 2026-09-01T18:30:27Z — SRV-002 first host-build negative

- The first `make -j2 all` stopped while recompiling `src/cli.cpp`: extending
  public `ChatMessage` with tool-call history made its two existing three-field
  aggregate initializers trigger `-Werror=missing-field-initializers`. This is a
  compile-time compatibility diagnosis, not a runtime or numerical failure.
  Resolve it by explicitly initializing the empty tool-call vector at both CLI
  sites, then rerun the complete host build before evaluating the new parser.

### 2026-09-01T18:32:01Z — API-002 fixture-encoding negative

- The new native API diagnostic compiled but exited one with no label. Source
  inspection found that the patch transport had interpreted intended C++ `\xNN`
  byte escapes before writing the file, producing UTF-8 for mojibake characters
  instead of the expected emoji and deliberately incomplete sequence. Replace
  those fixture literals with explicit C++ byte construction and make every
  diagnostic failure print its stable case name. Parser behavior is unchanged.

### 2026-09-01T18:49:50Z — SRV-002 focused-contract negatives

- The focused 24-case contract run passed 19 cases with three expected CUDA
  skips and failed two documentation/source-authentication assertions. Chapter
  59 used the equally descriptive heading "One-session generation" while its
  beginner vocabulary test requires the literal term "single-session". The CLI
  contract also retained the old `src/cli.cpp` hash after the compile-time
  compatibility fix explicitly initialized two empty tool-call vectors. Change
  the heading and refresh only those authenticated hashes; no runtime behavior,
  tolerance, or expected model output changes.

### 2026-09-01T19:00:04Z — API-002, SRV-002, and EDU-044 accepted

- Added a strict original JSON parser with all six JSON kinds, exact number
  grammar, Unicode escapes and surrogate pairs, UTF-8 validation, duplicate-key
  rejection, sorted canonical objects, a 64-level nesting limit, and compact or
  template-compatible spaced serialization. HTTP POST bodies require a valid
  `Content-Length`, `application/json`, no transfer encoding, and at most 1 MiB.
- Added `POST /v1/chat/completions` for the exact pinned model. It validates
  system/developer/user/assistant/tool history and text parts; reasoning modes;
  temperature, top-p, top-k, seed and token limits; string/array stops; function
  definitions, historical calls/results and auto/none/required/named choice;
  ordinary JSON and SSE with optional usage. Unknown controls and image, audio,
  file, structured output, multiple choices, parallel calls, log probabilities,
  nonzero penalties, legacy functions, and chunked request bodies fail explicitly.
- Extended the public message shape only with tool-call records and canonical
  definitions, keeping HTTP objects outside the engine. The public session sync
  and eval overloads accept one atomic cancellation flag; CUDA prefix replay now
  forwards it into the same per-layer `EvalControl` used by atomic token
  execution. The original overloads and CLI behavior remain intact.
- Generation acquires the FIFO gate, renders/tokenizes the complete prompt,
  enforces the 131,072-token capacity, synchronizes exact prefixes, and keeps
  sample/eval commit ordering. It separates reasoning/content, parses declared
  Qwen XML calls into typed JSON arguments, enforces forced tool choice, withholds
  split stops, partial UTF-8 and internal tool XML, reports exact usage/finish
  reasons, and exposes queue depth/delay headers. A socket monitor cancels queued
  or active work; `/health` makes the accumulated cancellation count observable.
- **Measured RTX 5090:** exact non-streaming `hello` and streamed `stream`
  responses passed; SSE ended with `[DONE]` and usage; a canonical function-tool
  prompt traversed the public engine; a custom `!` stop returned only `hello`;
  the second concurrent request arrived at queue depth one with positive delay;
  an already-active request observed peer disconnect, incremented the cancellation
  counter, released the gate, and left the server healthy. The focused live case
  passed in 43.02 seconds and SIGTERM exited zero.
- Added beginner Chapter 59 from JSON value kinds, escapes and surrogate pairs
  through HTTP body framing, every admitted/rejected field, roles, Qwen template
  conversion, tool schemas and choices, prefix reuse, sampling/commit, SSE,
  partial characters/stops, cancellation, and proof limits. Reconciled the
  README, older handbook chapters, source ledger, and architecture view.
- Final verification: `uv run ruff format .` left 65 Python files unchanged;
  the focused affected-contract set passed 23 tests with five expected GPU
  skips. The full ordinary suite passed 145 tests with seventeen expected GPU
  skips in 161.44 seconds. A clean CUDA 13.0.2 build compiled four host products,
  two native server diagnostics, two CUDA products, and sixteen SM120 diagnostics
  with warnings as errors. The complete exclusive RTX 5090 suite passed all 162
  tests in 304.97 seconds.
- Marked API-002, SRV-002, and EDU-044 done. SRV-003—Responses objects,
  streaming, tools, continuation storage, and `previous_response_id`—is the next
  API product task. Authentication and remote-deployment concerns remain outside
  the approved local v1 boundary.

### 2026-09-01T19:24:00Z — SRV-003 continuation boundary started

- Began SRV-003 and added discovered tasks API-003, SES-004, and EDU-045 before
  source changes. Responses has a different public envelope from Chat
  Completions, but both will map into the same validated `ChatRequest` and
  generation loop so model semantics do not fork.
- Exact continuation cannot be derived safely from assistant text: trimming,
  reasoning delimiters, tool XML, and tokenizer boundaries can change a
  re-rendered prefix. SES-004 therefore stores the token IDs copied from the
  session only after generation commits. A later request appends a separately
  rendered user/tool-result suffix to those exact IDs.
- The continuation record is a compact, versioned JSON control record rather
  than a duplicate 8 GiB KV/checkpoint image. Atomic temporary-file write,
  `fsync`, rename, and directory `fsync` protect publication. In-process reuse
  retains the GPU prefix; a restart deterministically replays the stored tokens.
  Records have no silent expiry or automatic deletion in v1. `store=false`,
  cancellations, errors, malformed records, and incompatible model/schema
  versions deliberately publish no continuable response.
- API-003 will admit text messages, instructions, function declarations,
  function calls/results, ordinary sampling, stops, reasoning effort, and
  ordered Responses SSE events. Image, audio, file, structured-output, parallel
  tool-call, and unknown features remain explicit errors rather than ignored
  fields. EDU-045 owns the beginner explanation and code/fixture links.

### 2026-09-02T00:12:00Z — SRV-003 full-suite pin mismatch

- The first full ordinary suite completed with 147 passes, seventeen expected
  CUDA skips, and one failure in 161.20 seconds. The response implementation
  and functional tests passed; the failure was the expected source-identity
  mismatch for `src/eval.cpp` in the older MDL-003 contract after adding the
  new exact follow-up template diagnostic.
- Updated only that authenticated source digest after confirming a repository-
  wide contract audit found no other mismatch. This was a bookkeeping failure,
  not a SHA-256 implementation or runtime failure; the rerun remains required
  before admission.

### 2026-09-02T14:27:35Z — SRV-003, API-003, SES-004, and EDU-045 accepted

- Added the strict `POST /v1/responses` adapter. Plain text and typed message,
  function-call, and function-output items map to the shared `ChatRequest`;
  instructions, reasoning effort, ordinary sampling, stops, tools, tool choice,
  and streaming retain the same model semantics as Chat Completions. Media,
  files, structured output, parallel calls, background mode, unknown fields,
  and unsafe continuation mutations fail explicitly.
- Added Responses objects and ordered named SSE events with consecutive sequence
  numbers, reasoning/text deltas, function-argument events, typed output items,
  completed/incomplete status, usage, cancellation, and single-flight timing.
- Added exact incremental rendering for one user turn or grouped function
  outputs. Durable continuation records retain committed token IDs and compatible
  tool schemas, validate a safe ID/model/schema/token range, cap records at
  16 MiB, and publish with temporary write, file `fsync`, rename, and directory
  `fsync`. `store=false`, missing, corrupt, incompatible, cancelled, and failed
  responses cannot be continued. Records do not expire or delete themselves.
- **Measured host:** the native Responses diagnostic passed request/tool/
  reasoning mapping, function-output continuation, explicit exclusions, atomic
  record round-trip, missing lookup, and corrupt-record rejection. Six template
  tests passed, including byte-exact grouped tool-result suffixes and mixed-item
  rejection.
- **Measured RTX 5090:** the integrated server produced exact non-streaming text,
  continued from a stored response, preserved the first record's full token
  array as a prefix, rejected continuation of `store=false`, emitted an ordered
  SSE lifecycle whose deltas equalled final text, then stopped and successfully
  continued the original ID from a fresh server process. The focused case passed
  in 61.30 seconds.
- Added beginner Chapter 60 explaining APIs, typed items, shared mapping, tokens,
  response IDs versus state, output items, exact prefix equality, incremental
  suffixes, atomic disk publication, restart replay, storage ownership, SSE
  events, exclusions, failures, and proof limits. Updated README commands,
  handbook navigation/provenance, earlier API proof boundaries, and architecture
  status.
- A clean pinned CUDA 13.0.2 build compiled all host products, three native
  server diagnostics, both CUDA products, and all sixteen SM120 diagnostics with
  warnings as errors. After resolving and retaining the one stale-pin negative,
  the full ordinary suite passed 148 tests with seventeen expected GPU skips in
  159.35 seconds. The complete exclusive RTX 5090 suite passed all 165 tests in
  322.46 seconds.
- Marked SRV-003, API-003, SES-004, and EDU-045 done. The approved CLI plus both
  OpenAI text API envelopes are now implemented; BEN-001, the benchmark harness,
  is the next delivery task. Broader SDK compatibility, authentication, remote
  serving, structured output, and multimodal inputs remain outside this gate.

### 2026-09-02T14:35:00Z — BEN-001 and EDU-046 started

- Began replacing the fail-closed `qw38-bench` stub and added EDU-046 before
  source changes. The executable will run one explicit workload/case per
  invocation so orchestration can enforce one large GPU process and preserve a
  result even when another case fails.
- The admitted primary modes are prefill and batch-one decode. Callers provide
  exact prompt text, expected token count, output-token count, cache policy,
  at least three warm-ups, at least thirty measured runs, and an output path.
  Smaller counts remain available only under an explicit smoke flag and cannot
  be mistaken for release evidence.
- Every run will reset the session before primary measurement, retain raw wall
  timings and per-token latencies, and calculate prompt tokens/s, TTFT, p50/p95
  inter-token latency, and output tokens/s from those raw values. An explicitly
  labeled reuse mode is separate. No synthetic confidence interval or baseline
  comparison belongs to BEN-001; CMP-002/CMP-003 own those gates.
- Added a narrow public timing result for `sample`/`eval` so the product harness
  can retain the already-admitted CUDA loading, embedding, GDN, attention, FFN,
  logits, sampling, graph-launch, queueing, persistence, idle-gap, commit, and
  token-total categories without exposing tensors or device layouts. Missing
  boundaries serialize as JSON `null`.
- Environment evidence will include UTC time, source revision, pinned model
  identity, build flags, host/kernel, container image name, CUDA runtime/driver,
  GPU identity, clocks, power, temperature, device memory, process RSS, prompt
  bytes/hash/token IDs, requested cache policy, and raw successful or failed
  runs. `nvidia-smi` telemetry is observational and will fail closed if the
  production CUDA benchmark cannot obtain it.

### 2026-09-02T14:48:00Z — BEN-001 exposed missing scheduler prefill integration

- The first real harness smoke succeeded but measured only about 16 prompt
  tokens/s for a 17-token prompt. Inspection confirmed `sync_tokens` still calls
  the complete one-token scheduler in a loop. The separately admitted tiled MMQ,
  64-token GDN scan, and memory-bounded attention chunk primitives are not wired
  into a layer-major full-prompt schedule.
- Added SCH-002 before any attempted fix. Earlier GDN-002 and ATN-002 claims
  remain valid at their focused primitive boundaries, and SCH-001 remains the
  admitted one-token full scheduler; none is silently relabeled as optimized
  end-to-end prefill. This negative makes the eventual comparative prefill gate
  unattainable until SCH-002 is completed.

### 2026-09-02T15:14:00Z — BEN-002 product exit-code regression discovered

- The first repository-wide host verification for BEN-001 stopped in
  `tests/test_build.py`: invoking `qw38-bench` with no arguments returned 1,
  while the established product boundary requires usage exit code 2.
- Added BEN-002 before changing the executable. This is a compatibility fix
  inside the benchmark delivery gate, not a new product feature; the failed
  `uv run pytest -q` / focused `uv run pytest -x -q` results remain recorded.
- The first BEN-002 fix restored exit code 2 but its focused rerun exposed the
  second half of the same established contract: the brand line must remain on
  stdout. The run was `13 passed, 2 failed`; the source-hash failure is expected
  until the behavior fix is final and the BEN-001 contract is refreshed.
- After restoring the brand, the product test still asserted the superseded
  benchmark-stub message (`has not passed its delivery gate`). BEN-001 is the
  delivery of that gate, so BEN-002 also updates this assertion to the real
  harness usage contract while retaining the no-argument fail-closed behavior.

### 2026-09-02T15:24:00Z — BEN-001, BEN-002, and EDU-046 completed

- Replaced the benchmark stub with the CUDA `qw38-bench` product. One invocation
  runs one explicit prefill or fixed-length decode case through only the public
  Engine/Session boundary, retains every warm-up and measured sample, records
  generated tokens and per-token wall times, and publishes JSON with
  file-and-directory synchronization plus atomic rename.
- Release mode fails closed unless it receives at least three warm-ups, 30
  samples, an exact expected rendered-token count, a context label, an explicit
  source revision, and `source-state=clean`. `--smoke` is always marked
  `admission_eligible=false`; runtime failures are retained after valid argument
  parsing. Primary samples reset to an empty token history, while separately
  labelled `agent-reuse` samples report reused-prefix and evaluated-suffix token
  counts independently.
- Added public optional timing overloads without changing existing callers.
  Detailed CUDA event attribution runs as one separate component probe marked as
  perturbing execution and excluded from throughput summaries. Unavailable
  queue/persistence values remain JSON `null`, not fabricated zeroes.
- Raw RTX 5090 smoke results are retained and authenticated in
  `evidence/benchmark/`: cold prefill p50 was 16.1989587 prompt tokens/s; cold
  decode p50 was 16.1515444 prompt tokens/s and 15.8418326 output tokens/s; the
  second agent-reuse sample reused 20 exact prefix tokens. All three used the
  pinned model/container and are explicitly dirty-worktree, non-admission
  evidence. The separate focused CUDA benchmark test passed in 17.90 s.
- Preserved the measured negative result: end-to-end prompt execution remains a
  loop over complete one-token scheduler calls. SCH-002 remains pending to
  connect the already-admitted MMQ and chunked GDN/attention primitives; no
  release or comparative throughput claim is made from these smoke numbers.
- Added the beginner-facing benchmark chapter, README example, source/provenance
  entry, machine-readable contract, authenticated fixture, and tests. It defines
  workload/run/warm-up/sample, prefill/decode, TTFT/ITL, throughput, p50/p95,
  raw versus summary records, cache policy, telemetry limitations, null values,
  source identity, atomic publication, and the proof boundary.
- First full host command: `uv run pytest -q` — failed at the pre-existing
  no-argument product assertion because the new harness returned 1. BEN-002 was
  added before the fix. Its next two focused runs exposed and retained the brand
  and obsolete-stub-assertion portions of the same contract; both were resolved.
- Final commands: `uv run ruff format .` — 66 files unchanged;
  `uv run ruff check .` — passed; `uv run pytest -q` — 150 passed, 18 skipped in
  159.16 s; `QW38_RUN_CUDA_TESTS=1 uv run pytest -q` — 168 passed in 350.46 s.
  Marked BEN-001, BEN-002, and EDU-046 done. The next performance-critical
  implementation task is SCH-002, while EVAL-001 remains the next product-tool
  gate if delivery order is followed strictly.
- The smoke also showed why detailed attribution cannot run inside every timed
  output token: its many CUDA event pairs perturb product throughput. The harness
  now uses ordinary graph execution for warm-up/measured distributions and runs
  one separately labeled component probe with `perturbs_execution=true` and
  `used_for_throughput_summary=false`.

### 2026-09-02T15:31:00Z — SCH-002, MEM-002, and EDU-047 started

- Began SCH-002 after BEN-001 measured the production `Session::sync` path at
  about 16 prompt tokens/s and confirmed that it dispatched one complete decode
  token at a time. The semantic target remains the admitted SCH-001 token path;
  the optimization may not relax its numeric or state gates.
- Selected fixed 64-row prompt chunks. Each chunk will flow layer-major through
  token-row embedding lookup, per-row normalization, prompt MMQ projections,
  the already-admitted chunked GDN/attention recurrence, prompt-row FFNs, and a
  final-row logits projection. Decode and one-token evaluation retain the
  stable-address graph path.
- The chunk owns candidate GDN state and attention KV rows until every layer and
  the final output complete. Only then may it publish state, tokens, last hidden,
  logits, and frontier. Cancellation is polled between layers. A retained
  token-wise synchronization entry point will provide direct differential
  evidence rather than making the optimized path its own oracle.
- Added MEM-002 before allocating permanent 64-row scratch because the final
  128K reserve proof must include it. Added EDU-047 before documentation work;
  it will explain row-major versus layer-major execution, scratch aliases,
  candidate versus committed state, the 64-row choice, and proof limits.
- First compiled two-row integration run: `qw38-cuda-prefix-sync-test` reached
  the optimized fresh-prompt path but `append_vs_fresh` was not byte exact;
  no-op, divergent replay, shorter replay, and invalid-input preservation still
  passed. This negative is retained before diagnosis. No optimized path is
  admitted until output and persistent-state differences are isolated and the
  existing SES-001 exactness contract passes again.
- After the optimized path passed, the first focused documentation-contract run
  failed because `tests/test_cuda_memory_fit.py` still required the historical
  pre-prompt-scratch free-memory number `5,241,044,992`. MEM-002 replaces that
  simultaneous-owner measurement, so the assertion must follow the new measured
  pre-graph value `5,205,393,408`; the old MEM-001 result remains in ledger
  history rather than being presented as current.
- Full CUDA command `QW38_RUN_CUDA_TESTS=1 uv run pytest -q` completed with 169
  passed and one failed allocation assertion. All five atomic-eval behavior
  cases passed; the test still expected the pre-SCH-002 capacity-three workspace
  size. Direct rerun measured 186,300,192 bytes. This is the fixed prompt scratch
  added under MEM-002, so the atomic contract/fixture/docs and exact assertion
  are updated without changing any atomic semantic gate.
- The first focused rerun found the fixture-value assertion duplicated in the
  contract test; the live binary gate passed. Updated that second stale value as
  part of the same MEM-002 reconciliation. Its next rerun found the same old
  number in the handbook-term list; that final duplicate is also updated.

### 2026-09-02T16:25:43Z — SCH-002, MEM-002, and EDU-047 completed

- Implemented a fixed 64-row, layer-major full-model prompt path. Q4_K/Q6_K
  projections use the admitted prompt MMQ, GDN layers use tiled 64-row scans,
  attention layers read committed prior chunks plus causal candidate rows, and
  FFNs process all prompt rows before the next layer. Only the final prompt row
  computes output logits; one-token evaluation and CUDA graphs are unchanged.
- Added permanent prompt scratch to `SchedulerWorkspace`: token-major FP32
  residuals, BF16 projection inputs, transient Q8 rows, projections, mixer/GDN
  intermediates, and 16 layers of 64 candidate K/V rows. The workspace remains
  move-only, stable-addressed, counted, and released with the existing owners.
- Retained the first failed two-token differential. Diagnosis found Q8_0 model
  weights had been routed through generic MMQ, adding activation requantization
  absent from SCH-001. Replaced only that format with batched direct Q8_0×BF16
  arithmetic using the same warp accumulation order as decode. The existing
  prefix suite then restored byte-exact append/fresh state and outputs.
- Measured the 65-token `[64, 1]` boundary against repeated SCH-001 execution:
  optimized 1,417.114341 ms, token-wise 4,204.655779 ms, 2.967055× faster. Every
  committed GDN/KV byte, token/frontier, last hidden value, and logit was exact.
  A forced cancellation after a synchronized layer left frontier zero and the
  committed session byte-equal to empty. Focused prompt/prefix/memory tests
  passed together (`3 passed in 14.53s`).
- BEN-001's 17-token smoke was repeated through the product harness: p50 prefill
  fell to 415.068169 ms and rose to 40.9571794 prompt tokens/s. Its two samples,
  dirty source identity, and `admission_eligible=false` record are retained at
  `evidence/prefill/sch002-prefill-smoke.json`; this is not a release matrix or
  comparative claim.
- MEM-002 measured the simultaneous 131,072-token state, resident model,
  198,882,816-byte workspace, and 64 uploaded graphs. Explicit Quartz ownership
  is 27,927,838,560 bytes; 5,199,101,952 bytes remained free, leaving a
  3,588,489,216-byte margin over the unchanged 1.5 GiB requirement. The first
  run failed only the old exact workspace equation; the updated arithmetic run
  passed.
- The first full GPU suite retained a second stale allocation expectation in
  the atomic-eval test: 169 tests passed and one expected the old workspace
  bytes, while all five atomic behavior cases passed. The exact assertion,
  fixture, contract, and handbook were reconciled; two focused reruns exposed
  and removed duplicate stale values without changing semantic gates.
- Added the beginner chapter for token-major storage, layer-major execution,
  MMQ, Q8_0 arithmetic, causal state, scratch, cancellation, commit, memory, and
  proof limits. Refreshed historical BEN/MEM/SES contracts without erasing their
  earlier results.
- Final verification: `uv run ruff format .` — one file reformatted, then 67
  files unchanged; `uv run ruff check .` — passed; host `uv run pytest -q` —
  151 passed, 19 skipped in 159.27 s; final
  `QW38_RUN_CUDA_TESTS=1 uv run pytest -q` — 170 passed in 320.44 s. Marked
  SCH-002, MEM-002, and EDU-047 done.
- A final API review added whole-chunk token-ID validation before any prompt
  GPU work. The first validation command used the nonexistent phony target
  `make qw38-cuda-prompt-scheduler-test` and failed immediately. The actual
  Makefile target is `build/qw38-cuda-prompt-scheduler-test`, already invoked
  by the pytest gate; no build or test failure was concealed.
- Final-source revalidation at 2026-09-02T16:35:39Z: focused
  `QW38_RUN_CUDA_TESTS=1 uv run pytest -q tests/test_cuda_prompt_scheduler.py`
  passed 2 tests in 11.63 s; complete
  `QW38_RUN_CUDA_TESTS=1 uv run pytest -q` passed all 170 tests in 334.07 s.
  `ruff format`, `ruff check`, JSON parsing, `git diff --check`, and the complete
  `local_sources` SHA-256 audit also passed. These results cover the exact bytes
  being committed, including the final token-ID preflight.

### 2026-09-03T10:01:35Z — TRC-004 documentation started

- Began the documentation/evidence stage for TRC-004 after the CUDA
  implementation and token-42 fixture were present. The task remains
  `in_progress`; no coupled task or plan change was introduced.
- The documentation boundary is a beginner chapter covering the five exact
  CUDA filters, scalar-pinned shapes, full-tensor metrics, greedy equality,
  filter/sink failure atomicity, diagnostic-only build isolation, and explicit
  non-goals. README and handbook indexes and the provenance ledger will link
  the chapter, contract, fixture, and focused tests.

### 2026-09-03T10:05:00Z — TRC-004 documentation evidence recorded

- Added [`docs/63-cuda-diagnostic-traces.md`](docs/63-cuda-diagnostic-traces.md),
  updated [`README.md`](README.md) and [`docs/README.md`](docs/README.md), and
  added the TRC-004 provenance/evidence entry to [`docs/sources.md`](docs/sources.md).
- Reconciled the TRC-004 row with implementation, contract, fixture, focused
  test, and documentation evidence while preserving status `in_progress`.
- Documentation checks: `python -m json.tool pins/cuda_trace_contract.json
  >/dev/null` and `python -m json.tool fixtures/cuda_trace.json >/dev/null`
  passed; `uv run pytest -q tests/test_cuda_trace.py` passed (3 tests);
  `git diff --check` passed. CUDA device/build gates remain recorded by the
  implementation stage and are not rerun by this documentation stage.

### 2026-09-03T10:27:00Z — TRC-004 delivered

- Independent verification attempt 2 passed focused trace tests, the full
  pytest suite, clean native/diagnostic builds, pinned CUDA 13.0.2 builds,
  RTX 5090 token-42 comparison, provenance audit, frozen scalar-gate checks,
  and normal-object symbol/string isolation.
- Attempt 1's provenance and diagnostic-test linkage failures were repaired
  within scope; first-pass acceptance is `no`, with one repair retry.
- Marked only TRC-004 `done`; there are no coupled IDs. `plan.md`,
  `pins/scalar_trace_contract.json`, and
  `pins/scalar_oracle_tolerances.json` remained unchanged.

### 2026-09-03T11:40:00Z — EVAL-001 documentation/evidence recorded

- Added [`docs/64-eval-harness.md`](docs/64-eval-harness.md), linked the chapter
  from [`README.md`](README.md) and [`docs/README.md`](docs/README.md), and
  recorded local provenance in [`docs/sources.md`](docs/sources.md).
- Updated [`pins/eval_contract.json`](pins/eval_contract.json) with current
  local-source identities and expanded [`fixtures/eval_harness.json`](fixtures/eval_harness.json)
  with explicit partial-status and evidence links. Refreshed the existing
  SHA-256 contract identity for the changed `src/eval.cpp`.
- The implementation remains `in_progress`: typed request validation and the
  logits evidence reader exist, but checkpoint publication and diagnostic CUDA
  trace capture have not run or been accepted. No checkpoint, trace, or
  QLT-001 quality claim is made.
- Documentation/evidence checks: `python -m json.tool pins/eval_contract.json
  >/dev/null`, `python -m json.tool fixtures/eval_harness.json >/dev/null`, and
  `git diff --check` passed.

## Decisions and Negative Results

- **2026-08-29 / BLD-002:** Host `nvcc` is absent. Resolved for reproducibility
  by the pinned CUDA 13.0.2 container; the task stays in progress because only a
  device diagnostic, not engine CUDA kernels, targets SM120.
- **2026-08-29 / MDL-001:** Whole-file in-process SHA-256 originally took
  56.603 s. MDL-003 resolved the CPU bottleneck without a metadata shortcut:
  the unchanged full-file digest now uses OpenSSL EVP runtime dispatch with a
  portable fallback and measured 8.47 s warm-cache. Cold-cache storage cost and
  optional immutable-snapshot receipts remain distinct future concerns.
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

### 2026-09-03T11:55:08Z — EVAL-001 blocked

- The bounded run completed three independent verification attempts, two Luna
  repairs, and one Sol diagnostic without reaching CUDA acceptance.
- Final failure evidence: the CUDA 13.0.2 build stops because the release eval
  compile rule omits the `cuda/` include path; amended `src/eval.cpp` hashes are
  stale across source-integrity contracts; therefore RTX 5090 logits,
  checkpoint, trace, isolation, and negative-publication smokes did not run.
- Recovery: apply the dossier's remaining build/hash repairs, pass all host and
  CUDA gates, validate all three modes through typed helpers on the RTX 5090,
  and promote the retained fixture only after those checks succeed.
- No commit or push was created. Evidence: [`tasks/EVAL-001.md`](tasks/EVAL-001.md).

### 2026-09-03T14:00:00Z — EVAL-001 reopened for recovery

- Commit `6a5408b` repaired the recorded CUDA `-Icuda` compile omission and
  refreshed the affected `src/eval.cpp` integrity hashes.
- Reopened the task as `pending` for the remaining admitted scope: complete
  checkpoint and diagnostic trace publication, repair diagnostic scheduler
  object wiring and object-scoped isolation checks, run the RTX 5090 smokes,
  and reconcile the typed schemas/fixture evidence. `plan.md` and inference
  arithmetic remain out of scope.

### 2026-09-03T12:50:48Z — BLD-003 status reconciled

- Reconciled the stale task row with the existing implementation and evidence.
  The runtime reports model, session, workspace, and graph ownership; the
  post-graph 131,072-token diagnostic checks the explicit ledger arithmetic,
  allocator delta, graph count, and required free reserve.
- Marked BLD-003 done. Its dependent MEM-001 and MEM-002 gates were already
  admitted and remain the authoritative physical allocation evidence.

### 2026-09-03T12:52:46Z — DOC-001 started

- Selected DOC-001 as the first eligible pending task in ledger order; BLD-001
  was done, the worktree was clean, and `main` tracked `origin/main`.
- Planning fixed the boundary at handbook/provenance reconciliation and a
  deterministic documentation audit gate. No runtime, fixture, pin, or plan
  changes are in scope. Evidence: [`tasks/DOC-001.md`](tasks/DOC-001.md).

### 2026-09-03T13:21:51Z — DOC-001 completed

- Reconciled handbook chapters 49, 50, 63, and 64; added the task coverage and
  provenance audit in Chapter 65; and added the repository-local documentation
  coverage/link test. The four authorized README source digests were refreshed
  to authenticate the final README bytes.
- Acceptance evidence: verification attempt 3 passed focused documentation and
  contract tests (8 total), Ruff format/check, `make clean`, `make -j2`,
  `make diagnostic`, full pytest (161 passed, 19 skipped), and `git diff --check`.
  Historical MEM-001 and partial/blocked EVAL-001 boundaries remain explicit.
- First-pass acceptance: no; two independent verification attempts failed only
  on stale README digests, followed by one bounded repair and passing attempt 3.
  No remaining risk identified within DOC-001 scope.

### 2026-09-03T15:44:53Z — EVAL-001 recovery readmitted

- Explicit admission found exactly one EVAL-001 ledger row in `pending` state;
  its listed dependencies ORA-001 and SES-003 each occurred exactly once and
  were `done`. The worktree was clean, `main` tracked `origin/main`, and the
  recovery commit `6a5408b` was present at HEAD ancestor `de3823c`.
- Reconciled the reopened dossier against the approved eval, checkpoint, trace,
  diagnostic-isolation, quality-separation, and documentation boundaries in
  `plan.md`; no plan or architecture change is required. Current eval source,
  typed helper, and separate CUDA release/diagnostic build rules support the
  recorded recovery scope.
- Coupled IDs are `none`; QLT-001 remains a dependent quality gate. The dossier
  fixes the exact changed-file allowlist, nine implementation decisions,
  focused/CUDA/hardware/isolation/repository gates, and definition of done, with
  no unresolved decisions. Marked only EVAL-001 `in_progress`; no code,
  documentation, fixture, contract, `Makefile`, or `plan.md` content changed.

### 2026-09-03T17:01:54Z — EVAL-001 completed

- Verification attempt 5 passed the focused JSON/build/pytest gates, pinned
  CUDA 13.0.2 build and native targets, release-vs-diagnostic object
  isolation, containerized typed RTX 5090 tests, direct logits/checkpoint and
  five-filter trace smokes, negative no-publication cases, repository-wide
  formatting/lint/build/test gates, and `git diff --check`.
- Acceptance evidence retained in [`fixtures/eval_harness.json`](fixtures/eval_harness.json)
  and documented in [`docs/64-eval-harness.md`](docs/64-eval-harness.md):
  complete FP32 logits, sequential checkpoint equality, all five diagnostic
  trace filters, authenticated metadata, and failure/no-publication behavior.
  The fixture remains harness-only; quality admission remains QLT-001 scope.
- Confirmed the complete diff is within the dossier allowlist, `plan.md` is
  unchanged, and no generated model outputs are included. Marked only
  EVAL-001 `done`; coupled IDs remain `none`.

### 2026-09-04T06:09:51Z — QLT-001 blocked on prompt performance

- Stopped the unpublished 131,072-token Quartz retrieval run after more than
  eight hours at approximately 4.2 tokens/s. The completed pinned same-GGUF
  llama.cpp reference and shorter Quartz evidence were preserved; no 128K
  retrieval result or passing quality report was published.
- Source and bounded-profile diagnosis ranked untiled per-token full attention,
  missing grouped-query KV reuse, row-wise prompt GEMV, 64-token chunks, about
  8.99 million launches, repeated copies/barriers, and serial GDN scans as the
  recovery boundary. Full causal attention semantics remain mandatory; sparse
  attention is not authorized.
- Added OPT-005 through OPT-013 before implementation. They cover exact tiled
  online-softmax attention, GQA KV reuse, multi-row query blocks, a 4,096-token
  default prompt chunk, batched quantized MMQ, tiled KV layout, prompt
  pipelining/fusion, prompt graphs, and block-parallel GDN scan.
- Marked QLT-001 blocked. Recovery requires OPT-012 and OPT-013 done, which
  transitively requires the complete optimization chain, followed by bounded
  2K/8K/32K scaling evidence and a fresh 128K quality retry. Evidence:
  [`tasks/QLT-001.md`](tasks/QLT-001.md).

### 2026-09-04T06:59:41Z — BLD-004 discovered during OPT-005

- Independent OPT-005 verification exposed a pre-existing diagnostic build
  mismatch: prompt, prefix, atomic, checkpoint, memory-fit, and timing targets
  link the non-trace scheduler object while their accounting expects the
  81,920-byte diagnostic trace workspace.
- The same audit found historical Makefile-owning contract hashes and two
  eval/SHA source hashes that require coordinated evidence reconciliation.
  This work is outside OPT-005's tiled-attention boundary and was registered
  before resuming its bounded repair. Evidence and exact affected files are in
  [`tasks/OPT-005.md`](tasks/OPT-005.md).

### 2026-09-04T07:53:56Z — OPT-005 blocked after bounded verification

- Three independent verification attempts and two bounded repairs did not
  produce admissible final evidence. The latest native diagnostic exited zero
  but omitted computed semantic predicates claimed by its fixture, and the
  checked-in timing arrays were synthetic repetitions rather than independent
  samples. The retained attention oracle also failed in the final native pass.
- Repository-wide acceptance remains red: ordinary pytest had one stale
  eval/SHA hash failure, and the CUDA suite had that failure plus four
  diagnostic linkage/workspace failures tracked by BLD-004.
- No commit or push was made. Recovery requires completing BLD-004, generating
  one authentic OPT-005 diagnostic record with every claimed predicate and raw
  sample, regenerating its fixture, and passing fresh ordinary and CUDA suites.
  Full evidence and commands are retained in [`tasks/OPT-005.md`](tasks/OPT-005.md).

### 2026-09-04T11:06:40Z — ATN-003 discovered during BLD-004 verification

- Two clean BLD-004 verification passes reproduced the retained layer-3
  attention failure while every BLD-004 object-linkage and allocation check
  passed.
- Escalation traced the regression to blocked OPT-005 commit `d0511a0`:
  `stage_chunk_rows` writes `normalized_key[lane]` instead of the per-head
  `normalized_key[base + lane]` range in `cuda/attention_decode.cu`.
- Registered ATN-003 before any source repair. It is separate recovery work,
  not coupled BLD-004 documentation/evidence; BLD-004 still requires a clean
  repository-wide CUDA pass before delivery. Evidence: [`tasks/BLD-004.md`](tasks/BLD-004.md).

### 2026-09-04T11:27:32Z — BLD-004 and ATN-003 blocked after bounded verification

- BLD-004 object linkage, exact diagnostic workspace accounting, focused CUDA
  tests, pinned builds, and all native gates passed. The registered ATN-003
  one-line normalized-key repair also passed all four retained attention cases.
- Final ordinary and full CUDA pytest each failed only the documentation audit
  because active ATN-003 had no audit-table row. The bounded workflow exhausted
  its repair and escalation allowance; no commit or push was made.
- Recovery requires adding an accurate non-final ATN-003 row to
  `docs/65-documentation-audit.md`, then rerunning ordinary pytest and the full
  clean CUDA suite. Evidence and all commands are in [`tasks/BLD-004.md`](tasks/BLD-004.md).

### 2026-09-04T12:25:31Z — BLD-004 and ATN-003 completed after recovery

- Added the required non-final ATN-003 documentation-audit row, then confirmed
  the audit with `uv run pytest -q tests/test_documentation.py` (4 passed).
- `uv run pytest -q` passed with 169 tests and 20 skips. A clean pinned CUDA
  13.0.2 `sm_120` rebuild of `cuda-products cuda-native` passed, followed by
  the complete CUDA-enabled suite with 189 tests passed.
- The diagnostic targets now use layout-compatible trace engine, scheduler,
  and checkpoint objects with exact prompt and memory accounting. Every KV head
  writes its own normalized-key range and the retained attention oracle passes.
  Marked BLD-004 and ATN-003 done; no `plan.md` change was required.

### 2026-09-04T12:30:16Z — OPT-005 recovery reopened

- Rechecked the blocked recovery boundary after BLD-004 and ATN-003 completed:
  the diagnostic object graph, workspace accounting, retained attention oracle,
  ordinary suite, and clean full CUDA suite are now green.
- The pinned CUDA 13.0.2 image and exclusive RTX 5090 are available. Remaining
  OPT-005 work is local and decision-bounded: compute every claimed semantic
  predicate, retain independent raw timing samples, regenerate the fixture from
  one successful diagnostic, strengthen evidence validation, and rerun all gates.
- Marked OPT-005 `pending` for fresh workflow admission. No architecture or
  `plan.md` change is required.

### 2026-09-04T15:54:44Z — OPT-006 delivered

- Independent verification passed the grouped CUDA attention acceptance: exact
  grouped/per-query outputs, inherited causal/transaction/scratch/validation/
  capacity/two-node predicates, frozen untiled metrics, and executed device
  K/V request counters at 2K/8K/32K prefixes. Grouped requests and bytes are
  exactly one sixth of the retained per-query path.
- The schema-1 contract, pinned RTX 5090 fixture, shared fail-closed validator,
  and attention-prefill documentation record the evidence and explicitly limit
  it to executed BF16 global-load requests, excluding physical DRAM, latency,
  throughput, and end-to-end claims. Full ordinary and CUDA-enabled suites
  passed (`172 passed, 21 skipped` and `193 passed`).
- Marked OPT-006 `done`; delivery changes are limited to the verified task
  scope plus this ledger/audit bookkeeping. `plan.md` and `Makefile` remain
  unchanged.

### 2026-09-04T17:43:02Z — OPT-007 delivered

- Independent verification attempt 2 passed the exact two-row/one-row semantic,
  causal, transaction, launch-topology, bounded-resource, and 65-versus-64+1
  boundary acceptance on the pinned RTX 5090, including full captured grid and
  block tuples and fail-closed negative validator cases.
- The schema-1 contract, measured fixture, native diagnostic, Python validator,
  and prefill documentation provide the acceptance evidence. The proof remains
  component-only and makes no speedup, end-to-end, or complete-model-memory
  claim; OPT-008 and OPT-010 remain pending. `plan.md` and `Makefile` are
  unchanged.
- Marked OPT-007 `done`; the native diagnostic's repeated ignored cleanup calls
  are non-blocking test hygiene and do not affect production semantics or
  emitted evidence.

### 2026-09-04T17:59:04Z — OPT-008 started

- Admitted OPT-008 as the first pending ledger row whose dependencies are done;
  the worktree was clean and `main` tracked `origin/main`.
- Planning fixed the default outer prompt transaction at 4,096 rows with
  capacity-bounded scratch, a `[4096, 1]` tail, retained 64-row GDN scans,
  atomic cancellation, exact differential evidence, and a live 128K reserve
  measurement. No coupled task or `plan.md` change is required.
- Evidence: [`tasks/OPT-008.md`](tasks/OPT-008.md).

### 2026-09-07T13:08:07Z — OPT-008 delivered

- Independent verification attempt 1 passed the 4,096-row default, `[4096, 1]`
  tail, capacity-bounded fallback, exact 64-row reference equality, 4,096-row
  cancellation, diagnostic workspace totals, and live 128K owner/reserve gates
  on the pinned RTX 5090.
- Acceptance evidence: [`tasks/OPT-008.md`](tasks/OPT-008.md);
  [`cuda/full_scheduler.h`](cuda/full_scheduler.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/prompt_scheduler_test.cu`](cuda/prompt_scheduler_test.cu);
  [`pins/cuda_prompt_scheduler_contract.json`](pins/cuda_prompt_scheduler_contract.json);
  [`fixtures/cuda_prompt_scheduler.json`](fixtures/cuda_prompt_scheduler.json);
  [`fixtures/cuda_atomic_eval.json`](fixtures/cuda_atomic_eval.json);
  [`fixtures/cuda_memory_fit_post_graph.json`](fixtures/cuda_memory_fit_post_graph.json);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md);
  [`docs/54-post-graph-128k-memory.md`](docs/54-post-graph-128k-memory.md).
  The proof excludes comparative speed and 128K prefill execution. `plan.md`
  and `Makefile` are unchanged. OPT-009 through OPT-013 remain pending;
  QLT-001 remains blocked.
- Marked OPT-008 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-07T13:43:58Z — OPT-009 started

- Admitted OPT-009 as the first pending ledger row whose dependencies are done;
  the worktree was clean and `main` tracked `origin/main`.
- Planning keeps production Q8_0 on direct BF16 activations with a CUD-002-style
  prompt tile, extends Q4_K/Q6_K tiles through 64 with a new RTX 5090 sweep, and
  preserves frozen numeric envelopes. No coupled task or `plan.md` change is
  required.
- Evidence: [`tasks/OPT-009.md`](tasks/OPT-009.md).

### 2026-09-07T15:11:00Z — OPT-009 delivered

- Independent verification attempt 1 passed production weight-tile reuse,
  measured SM120 kind×bucket selection, Q8_0 memcmp equality to the retained
  row-wise kernel, frozen CUD-002 Q4_K/Q6_K envelopes, component timing
  predicates, and inherited scheduler/atomic/memory/prefix/graph gates on the
  pinned RTX 5090.
- Acceptance evidence: [`tasks/OPT-009.md`](tasks/OPT-009.md);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/prompt_mmq_test.cu`](cuda/prompt_mmq_test.cu);
  [`pins/cuda_prompt_mmq_contract.json`](pins/cuda_prompt_mmq_contract.json);
  [`fixtures/cuda_prompt_mmq.json`](fixtures/cuda_prompt_mmq.json);
  [`evidence/profiling/opt009-mmq-tile-sweep-raw.txt`](evidence/profiling/opt009-mmq-tile-sweep-raw.txt);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/55-offline-dispatch-tuning.md`](docs/55-offline-dispatch-tuning.md).
  The proof remains component-only and excludes end-to-end speed and 128K
  quality recovery. `plan.md` and `Makefile` are unchanged. OPT-010 through
  OPT-013 remain pending; QLT-001 remains blocked.
- Marked OPT-009 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-07T15:32:39Z — OPT-010 started

- Admitted OPT-010 as the first pending ledger row whose dependencies are done;
  the worktree was clean and `main` tracked `origin/main`.
- Planning stores committed K/V as head-major token-contiguous physical layout
  with logical SES-003 checkpoint pack/unpack, scatter commit, and coalesced
  32-row tile-span evidence. No coupled task or `plan.md` change is required.
- Evidence: [`tasks/OPT-010.md`](tasks/OPT-010.md).

### 2026-09-07T16:55:25Z — OPT-010 delivered

- Independent verification attempt 1 passed coalesced 32×256 committed-tile
  pointer spans, logical pack/unpack round-trip, production versus one-row
  exact GQA outputs, SES-003 checkpoint size/hash/continuation, prefix reuse,
  and the unchanged 8 GiB 16-layer KV capacity on the pinned RTX 5090.
- Acceptance evidence: [`tasks/OPT-010.md`](tasks/OPT-010.md);
  [`cuda/attention_decode.h`](cuda/attention_decode.h);
  [`cuda/attention_decode.cu`](cuda/attention_decode.cu);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/checkpoint.cu`](cuda/checkpoint.cu);
  [`cuda/kv_tile_layout_test.cu`](cuda/kv_tile_layout_test.cu);
  [`pins/cuda_kv_tile_layout_contract.json`](pins/cuda_kv_tile_layout_contract.json);
  [`fixtures/cuda_kv_tile_layout.json`](fixtures/cuda_kv_tile_layout.json);
  [`docs/43-cuda-attention-decode.md`](docs/43-cuda-attention-decode.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md);
  [`docs/49-cuda-checkpoints.md`](docs/49-cuda-checkpoints.md).
  The proof remains pointer-span / exact-value evidence and excludes Nsight
  DRAM, latency, and end-to-end recovery. `plan.md` and `Makefile` are
  unchanged. OPT-011 through OPT-013 remain pending; QLT-001 remains blocked.
- Marked OPT-010 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-07T18:12:37Z — OPT-011 delivered

- Independent verification attempt 1 passed batched embedding (`grid.y` =
  token count), 127 fused residual-add-norm launches plus one last FFN add,
  one all-layer multi-block scatter (`grid.y` = 16), two async D2H copies
  overlapping that scatter then two stream joins, fused/unfused 2/3/64/65-row
  byte equality, poll-8 cancellation with no publication, fused CUDA-event
  mean strictly below unfused, and inherited scheduler/atomic/prefix/checkpoint/
  graph/memory/MMQ/KV-layout gates on the pinned RTX 5090.
- Acceptance evidence: [`tasks/OPT-011.md`](tasks/OPT-011.md);
  [`cuda/full_scheduler.h`](cuda/full_scheduler.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`cuda/attention_decode.h`](cuda/attention_decode.h);
  [`cuda/attention_decode.cu`](cuda/attention_decode.cu);
  [`cuda/prompt_pipeline_test.cu`](cuda/prompt_pipeline_test.cu);
  [`pins/cuda_prompt_pipeline_contract.json`](pins/cuda_prompt_pipeline_contract.json);
  [`fixtures/cuda_prompt_pipeline.json`](fixtures/cuda_prompt_pipeline.json);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md);
  [`docs/52-profiler-led-fusion.md`](docs/52-profiler-led-fusion.md);
  [`docs/48-atomic-eval-and-sampling.md`](docs/48-atomic-eval-and-sampling.md).
  The proof remains component-only launch/barrier and CUDA-event A/B evidence
  and excludes a Nsight Systems overlap screenshot, end-to-end prefill/decode
  speedup, and 128K quality recovery. `plan.md` and `Makefile` are unchanged.
  OPT-012 and OPT-013 remain pending; QLT-001 remains blocked.
- Marked OPT-011 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-07T20:27:06Z — OPT-012 delivered

- Independent verification attempt 1 passed 64 decode plus 64 prompt FFN
  graphs at 4,096 rows, graph/ordinary fused byte equality, fail-closed
  mismatch/unfused, poll-8 cancellation with no publication, capacity-65
  prompt-graph skip, and reconciled 128-graph 128K memory with the 1.5 GiB
  reserve intact on the pinned RTX 5090.
- Acceptance evidence: [`tasks/OPT-012.md`](tasks/OPT-012.md);
  [`cuda/full_scheduler.h`](cuda/full_scheduler.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`src/engine.cpp`](src/engine.cpp);
  [`cuda/prompt_graph_test.cu`](cuda/prompt_graph_test.cu);
  [`pins/cuda_prompt_graph_contract.json`](pins/cuda_prompt_graph_contract.json);
  [`fixtures/cuda_prompt_graph.json`](fixtures/cuda_prompt_graph.json);
  [`fixtures/cuda_memory_fit_post_graph.json`](fixtures/cuda_memory_fit_post_graph.json);
  [`docs/53-stable-address-cuda-graphs.md`](docs/53-stable-address-cuda-graphs.md);
  [`docs/54-post-graph-128k-memory.md`](docs/54-post-graph-128k-memory.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  The proof remains component-only and excludes a whole-chunk graph, Nsight
  Systems, end-to-end prefill/decode speedup, and 128K quality recovery.
  `plan.md` and `Makefile` are unchanged. OPT-013 remains pending; QLT-001
  remains blocked.
- Marked OPT-012 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-08T08:58:06Z — OPT-013 delivered

- Independent verification attempt 2 passed sequential GDN-002
  chunk-versus-tokenwise byte equality, parallel-versus-sequential envelopes
  (max abs `5e-8`, RMS `5e-9`, zero non-finite) across window and
  4,096-versus-64-window boundaries, overlay `W_fit(4096)=22` with sequential
  fallback when one `(A, B)` pair cannot overlay, captured intra/prefix/
  from-state geometry, fail-closed isolation, and a component-only 4,096-token
  CUDA-event parallel mean strictly below sequential on the pinned RTX 5090.
- Acceptance evidence: [`tasks/OPT-013.md`](tasks/OPT-013.md);
  [`cuda/gdn_step.h`](cuda/gdn_step.h);
  [`cuda/gdn_step.cu`](cuda/gdn_step.cu);
  [`cuda/full_scheduler.h`](cuda/full_scheduler.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/prompt_scheduler_test.cu`](cuda/prompt_scheduler_test.cu);
  [`cuda/gdn_scan_test.cu`](cuda/gdn_scan_test.cu);
  [`pins/cuda_gdn_scan_contract.json`](pins/cuda_gdn_scan_contract.json);
  [`fixtures/cuda_gdn_scan.json`](fixtures/cuda_gdn_scan.json);
  [`docs/42-cuda-gdn-chunks.md`](docs/42-cuda-gdn-chunks.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md);
  [`docs/18-gated-delta-network.md`](docs/18-gated-delta-network.md).
  The proof remains component-only and excludes Nsight Systems, end-to-end
  prefill/decode speedup, and 128K quality recovery. `plan.md` and `Makefile`
  are unchanged. QLT-001 remains blocked.
- Marked OPT-013 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-08T09:21:04Z — QLT-001 readmitted after OPT chain

- OPT-012 and OPT-013 are `done`, satisfying the 2026-09-04T06:09:51Z recovery
  dependency on the prompt optimization chain.
- Marked QLT-001 `pending` (not `in_progress`). Coupled IDs remain `none`.
- User gate before any 131,072-token retrieval retry: publish a full 2K/8K/32K
  scaling report with Quartz wall-clock/runtime, plain pinned-llama.cpp
  comparison on the same GGUF/RTX 5090, and an explicit 128K completion-time
  forecast. Do not start the 128K quality run until that report is reviewed.
- Evidence dossier: [`tasks/QLT-001.md`](tasks/QLT-001.md).

### 2026-09-08T09:53:29Z — QLT-001 2K/8K/32K scaling report

- Published forecast-only prefill scaling evidence before any 128K quality
  retry: [`evidence/quality/scaling-2026-09-08/REPORT.md`](evidence/quality/scaling-2026-09-08/REPORT.md).
- Quartz cold prefill: ~49 / 40 / 25 tok/s at ~2K / 8K / 32K; pinned llama.cpp
  same-GGUF `llama-bench` ~3114 / 3027 / 2626 tok/s.
- Preferred 128K Quartz prefill forecast from quadratic fit: ~3.64 h
  (budget ~3.6–4.7 h). QLT-001 remains `pending` until the report is accepted
  and a later ledger run starts implementation.

### 2026-09-08T10:22:44Z — 2K llama.cpp parity gate admitted

- After the 2K/8K/32K scaling report showed Quartz still ~63–105× behind pinned
  llama.cpp, admitted OPT-014 through OPT-016 before further long-context or
  quality retries:
  - OPT-014: live attributed 2K prefill time breakdown report.
  - OPT-015: explain thousand-tok/s same-GGUF llama.cpp 2K prefill and mine
    `../ds4` only for transferable technique inspiration (ds4 cannot run this
    Qwen GGUF; not a same-model baseline; sparse/compressed attention remains
    non-transferable), then rank recoveries with `plan.md` provenance.
  - OPT-016: reach cold 2K Quartz prefill ≥ pinned llama.cpp on the same GGUF
    and RTX 5090; until then the performance yardstick is **2K only**.
- Marked QLT-001 `blocked` again. Recovery now requires OPT-016 done (and its
  deps), then the previously required bounded long-context/quality retry path.
  `plan.md` is unchanged: this is an operational recovery gate ahead of the
  existing CMP matrix and QLT 128K admission.
- Next eligible pending task for `/run-ledger-task-cursor` is OPT-014 (BEN-001
  and OPT-013 are done). OPT-015 is also eligible in parallel dependency terms
  after OPT-013/PIN-002; ledger row order still prefers OPT-014 first.

### 2026-09-08T10:35:03Z — OPT-014 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-014.md`](tasks/OPT-014.md).
- Coupled IDs: none. Plan impact: none.
- Marked OPT-014 `in_progress`.

### 2026-09-08T11:06:41Z — OPT-014 delivered

- Independent verification attempt 1 passed a cold empty-session 2048-token
  production `sync_tokens` on exclusive RTX 5090: eight measured CUDA-event
  plus remainder categories reconstruct host wall within `rel_tol=1e-4` /
  `abs_tol_ms=0.05`; `graph` is measured zero with `prompt_graph_launches == 0`;
  `chunk_count` is 1; `nsight_*` are `not_used`. The retained report is
  [`fixtures/cuda_prefill_attribution.json`](fixtures/cuda_prefill_attribution.json)
  from the timed diagnostic, not a Nsight capture. Null-pointer prompt
  execution creates no extra CUDA events. `plan.md`, `Makefile`, public Engine
  / CLI / server / `qw38-bench` schemas, and BEN-001 throughput JSON are
  unchanged. OPT-015 remains pending; OPT-016 remains pending on OPT-014 and
  OPT-015; QLT-001 remains blocked.
- Acceptance evidence: [`tasks/OPT-014.md`](tasks/OPT-014.md);
  [`cuda/full_scheduler.h`](cuda/full_scheduler.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/prefill_attribution_test.cu`](cuda/prefill_attribution_test.cu);
  [`pins/cuda_prefill_attribution_contract.json`](pins/cuda_prefill_attribution_contract.json);
  [`fixtures/cuda_prefill_attribution.json`](fixtures/cuda_prefill_attribution.json);
  [`docs/51-runtime-timing-and-nvtx.md`](docs/51-runtime-timing-and-nvtx.md);
  [`docs/61-benchmark-harness.md`](docs/61-benchmark-harness.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is live instrumentation, not a throughput gate or llama.cpp parity.
- Marked OPT-014 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-08T11:19:55Z — OPT-015 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-015.md`](tasks/OPT-015.md).
- Coupled IDs: none. Plan impact: none.
- Marked OPT-015 `in_progress`.

### 2026-09-08T11:41:40Z — OPT-015 delivered

- Independent verification attempt 1 passed: host-tested citation report
  explains thousand-tok/s same-GGUF llama.cpp 2K prefill from OPT-014
  exact-2048 attribution and the scaling `llama-bench` 2K JSON; `../ds4` is
  MIT technique inspiration only (`ds4 cannot run this Qwen GGUF`; no ds4
  same-model baseline); transferable MMA MMQ / fused GDN / causal MMA
  attention are separated from non-transferable sparse/compressed
  attention; every OPT-014 category is mapped; ranked recovery is
  Proposed, not OPT-016. `plan.md`, `Makefile`, and production kernels are
  unchanged. OPT-016 remains pending on OPT-014 and OPT-015; QLT-001
  remains blocked.
- Acceptance evidence: [`tasks/OPT-015.md`](tasks/OPT-015.md);
  [`evidence/optimization/opt015-2k-recovery/REPORT.md`](evidence/optimization/opt015-2k-recovery/REPORT.md);
  [`pins/opt015_recovery_contract.json`](pins/opt015_recovery_contract.json);
  [`fixtures/opt015_recovery.json`](fixtures/opt015_recovery.json);
  [`tests/test_opt015_recovery.py`](tests/test_opt015_recovery.py);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is a citation report, not a throughput gate or llama.cpp parity.
- Marked OPT-015 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping.

### 2026-09-08T13:09:00Z — OPT-016 blocked; OPT-017 discovered

- OPT-016 implementation landed ranked recoveries 1–3 (Q4_K/Q6_K MMA MMQ
  J=128, fused GDN token loop, causal MMA attention). Rank 4 skipped
  (`graph` 0 ms; `other_idle` ~0.005% of wall).
- Frozen 2K gate failed on exclusive RTX 5090 same sitting: Quartz mean
  208.758591 tok/s versus live llama.cpp `avg_ts` 3183.528255
  (`gate_passed=false`). Post-rank attribution: `gdn`+`attention` 73.7% of
  wall with mixer projections still on byte-exact `launch_q8_mmq_bf16`;
  `ffn_mmq` 26.3%.
- Per dossier stop rule, did not loosen Q8_0 byte equality or expand OPT-016
  into mixer MMA. Added discovered pending task OPT-017. Marked OPT-016
  `blocked`. Recovery: OPT-017 done, then re-pass the OPT-016 frozen 2K gate.
  QLT-001 remains blocked on OPT-016. `plan.md` unchanged. No delivery
  commit or push (acceptance not met). WIP remains in the worktree pending
  an explicit user commit decision before the next clean admission.
- Evidence: [`tasks/OPT-016.md`](tasks/OPT-016.md);
  [`fixtures/opt016_parity.json`](fixtures/opt016_parity.json);
  [`evidence/optimization/opt016-2k-parity/REPORT.md`](evidence/optimization/opt016-2k-parity/REPORT.md).

### 2026-09-08T13:28:15Z — OPT-018–OPT-020 admitted for 2K recovery ladder

- After Rank 1–3 WIP showed ~15× remaining gap (FFN MMA still ~2.55 s alone;
  `gdn`+`attention` still dominated by non-MMA mixer Q8_0), admitted follow-on
  pending tasks so the ledger covers the full recovery path without waiting for
  another discovery stop:
  - OPT-017 (amended): mixer Q8_0 MMA with unloosened OPT-009 reference plus
    live 2K remasurement/attribution; **not** the end-to-end parity gate.
  - OPT-018: llama-competitive Q4_K/Q6_K MMA MMQ quality under CUD-002.
  - OPT-019: residual GDN/attention core time after OPT-017 (depends on OPT-017).
  - OPT-020: split 2K attribution into mixer MMQ versus GDN/attention core.
- OPT-016 remains `blocked` and remains the sole ≥ llama.cpp 2K gate owner.
  Recovery is OPT-017–OPT-019 as needed, then re-pass the frozen gate. QLT-001
  still blocked on OPT-016. `plan.md` unchanged.
- Next eligible pending by ledger row order: OPT-017 (deps OPT-009, OPT-015
  done). OPT-018 and OPT-020 are also dependency-eligible; row order prefers
  OPT-017.

### 2026-09-08T14:35:43Z — OPT-017 delivered

- Independent verification attempt 1 after repair passed: production mixer
  Q8_0 prompt MMQ (`launch_q8_mmq_bf16`, `prompt_rows >= 8`) is MMA at J=128;
  `launch_q8_mmq_bf16_variant` versus `launch_q8_mmq_bf16_reference` remains
  byte-exact; CUD-002 pin numbers are unloosened; MMA versus
  `launch_quant_mmq_variant` `kQ8_0` meets max abs `5e-4`, RMS `2.5e-4`, and
  zero non-finites (`gpu_staged_*`); live exact-2048 OPT-014 attribution
  reconstructs wall within `rel_tol=1e-4` / `abs_tol_ms=0.05`; unperturbed
  3-replicate exact-2048 remasurement is checked in (Quartz mean 375.57019
  tok/s versus live llama.cpp 3208.29056). OPT-016 remains `blocked` and
  remains the parity gate owner (`owns_opt016_parity_gate=false`). `plan.md`
  and `Makefile` `NVCCFLAGS` are unchanged.
- Acceptance evidence: [`tasks/OPT-017.md`](tasks/OPT-017.md);
  [`pins/opt017_mixer_mma_contract.json`](pins/opt017_mixer_mma_contract.json);
  [`fixtures/opt017_mixer_mma.json`](fixtures/opt017_mixer_mma.json);
  [`evidence/optimization/opt017-mixer-q8-mma/REPORT.md`](evidence/optimization/opt017-mixer-q8-mma/REPORT.md);
  [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is mixer MMA admission plus live 2K remasurement/attribution, not an
  end-to-end 2K tok/s gate.
- Marked OPT-017 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. OPT-018 is next eligible pending by ledger
  row order.

### 2026-09-08T15:30:58Z — OPT-018 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-018.md`](tasks/OPT-018.md).
- Coupled IDs: none. Plan impact: none.
- Marked OPT-018 `in_progress`.

### 2026-09-08T16:41:28Z — OPT-018 blocked

- Implementation built Q4_K/Q6_K quality MMA (`MMQ_ITER_K=256`, packed
  load-tiles, Q8_1 MMQ Y, `dim3(32,8)`). Q4_K through `8×1024×5120` met
  unloosened CUD-002; required Q6_K `64×5120×6144` failed
  (`max_abs=0.000946044922`, `rms=0.000258449378` vs `5e-4` / `2.5e-4`).
- Diagnostic (cursor-grok-4.6-high): miss is K-linear INT8-MMA versus
  variant per-element FP association, not a leftover packed-tile bug.
  In-scope epilogue tweaks rejected; pins not loosened; no stream-K; no
  extra `cudaMalloc`; `plan.md` unchanged.
- Marked OPT-018 `blocked`. No verification, commit, or push.
- Recovery: measured Q6 quality kernel under locked geometry meeting CUD-002
  on `64×5120×6144` and `2048×5120×6144`, or approved `plan.md` change
  before any envelope change / Q6 shape drop / splitting Q6 off production
  quality MMA. Evidence: [`tasks/OPT-018.md`](tasks/OPT-018.md).

### 2026-09-08T16:58:43Z — OPT-018 re-admitted under ds4 association gate

- User authorized adopting the ds4 MMQ parity association gate for Q4_K/Q6_K
  MMA versus the scalar variant (element fails only when both
  `abs > 0.20*sqrt(K)` and `rel > 0.05`), matching
  `../ds4/cuda/mmq/test/test_mmq_parity.cu` Q4_K / `check_close`.
- `plan.md` Correctness policy gains an INT8 MMA association exception;
  CUD-002 fixed `5e-4` / `2.5e-4` remain mandatory and unloosened for
  `launch_quant_mmq_variant`.
- Pins/tests: `pins/opt018_ffn_mma_contract.json`,
  `cuda/quant_mmv_test.cu` `run_mma_case`, host OPT-018 fixture/test.
- Marked OPT-018 `in_progress` again to finish J-sweep, live FFN bar, and
  evidence under the new gate. Evidence: [`tasks/OPT-018.md`](tasks/OPT-018.md).

### 2026-09-08T17:15:19Z — OPT-018 option C: ds4 CPU-dequant MMQ admission

- User selected full ds4 parity (option C): Q4_K/Q6_K MMQ (variant and MMA)
  admitted against host CPU dequant-weight × BF16→float GEMM under
  `abs > 0.20*sqrt(K)` AND `rel > 0.05`; fixed `5e-4`/`2.5e-4` retired as the
  Q4_K/Q6_K MMQ admitting envelope (`plan.md` Correctness policy).
- Pins: `pins/cuda_mmq_contract.json`, `pins/cuda_prompt_mmq_contract.json`,
  `pins/opt018_ffn_mma_contract.json`. Legacy abs/rms keys retained only for
  Q8_0 / OPT-016–017 pin equality.
- Tests: `cuda/quant_mmv_test.cu`, `cuda/prompt_mmq_test.cu`.
- OPT-018 remains `in_progress` (live FFN evidence still pending).

### 2026-09-08T18:00:59Z — OPT-018 FFN-bar diagnostic (verdict A)

- Live exact-2048 attribution: `ffn_mmq` **9188.24902** ms vs OPT-017 before
  2524.67725 ms and llama.cpp 2K wall 637.874802 ms. Association gate
  (ds4 option C vs CPU dequant) passes (`mma_bad=0`). J=128. Component
  Q4 `2048×17408×5120` 46.099 ms; Q4 kernel 255 regs / 224 B local /
  occupancy 1.
- Independent compare of `cuda/quant_mmq_mma.cuh` quality kernel vs
  Rank-1 K=32 MMA vs llama.cpp `cc83d7b` `mmq.cuh` /
  `mmq-vec-dot.cuh` / `mmq-load-tiles.cuh` / `quantize.cu`: quality is
  Ampere-shaped but uses `double` accumulators, D4 Y plus inner-loop
  `quality_sum_i8x32` instead of Q4 DS4, and float2 X scales at stride 84
  instead of half2 stride 76. Occupancy 1 matches llama.cpp Ampere Q4_K
  and is not the stop. Stream-K remains rejected.
- Verdict **A** (fixable repair). Status stays `in_progress`. Locked
  repair plan in [`tasks/OPT-018.md`](tasks/OPT-018.md). No commit/push.
  OPT-016 remains blocked and is not this increment.

### 2026-09-08T18:55:28Z — OPT-018 delivered

- Independent verification after inner-loop repair and coordinator ruff
  F841 fix: production Q4_K/Q6_K prompt MMQ (`launch_quant_mmq`,
  `prompt_rows >= 8`) is quality MMA (`MMQ_ITER_K=256`, packed load-tiles,
  Q8_1 MMQ Y in existing workspace, J=128); `launch_quant_mmq_variant`
  remains the retained scalar reference; Q4_K/Q6_K MMA uses the ds4 option
  C association gate versus CPU dequant GEMM (`mma_bad=0`). Live exact-2048
  `ffn_mmq` **399.287018** ms is strictly below the OPT-017 before snapshot
  **2524.67725** ms and live llama.cpp 2K wall **636.184782** ms
  (`llama_competitive` true). Quartz mean 625.792114 tok/s does not claim
  the end-to-end 2K tok/s gate (`would_pass_opt016` false;
  `owns_opt016_parity_gate` false). OPT-016 remains `blocked`. Coupled IDs:
  none. Delivery re-check: `uv run ruff check .` passed; `uv run pytest -q
  tests/test_documentation.py tests/test_opt018_ffn_mma.py` passed (6
  passed, 1 skipped).
- Acceptance evidence: [`tasks/OPT-018.md`](tasks/OPT-018.md);
  [`pins/opt018_ffn_mma_contract.json`](pins/opt018_ffn_mma_contract.json);
  [`fixtures/opt018_ffn_mma.json`](fixtures/opt018_ffn_mma.json);
  [`evidence/optimization/opt018-ffn-mma-quality/REPORT.md`](evidence/optimization/opt018-ffn-mma-quality/REPORT.md);
  [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is llama-competitive 2K FFN MMA quality, not an end-to-end 2K tok/s
  gate.
- Marked OPT-018 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. OPT-016 stays `blocked`. OPT-019 is next
  eligible pending by ledger row order.

### 2026-09-08T20:54:21Z — OPT-019 delivered

- Independent verification attempt 2 passed after attempt 1 failed (live
  OPT-018 writer overwrote the locked before snapshot). Production prompt
  GDN recurrence (`kFusedTokenLoop`) is warp-column quality; sequential
  remains the unloosened GDN-002/OPT-013 reference; Rank-2 fused is
  retained for A/B. Production prompt attention (`token_count >= 16`) is
  fattn-mma quality (`ncols1=16`); tiled remains the unloosened OPT-005
  reference; Rank-3 MMA is retained for A/B. Live exact-2048 `gdn`
  **1212.50647** ms and `attention` **477.650085** ms meet the locked
  85%/85%/70% addressed rule versus the immutable before copy
  (`measurement_utc` 2026-09-08T18:33:22Z; `gdn` 1643.46082,
  `attention` 1148.47461, combined 2791.93543). Combined after
  **1690.156555** ms. Quartz mean 978.756592 tok/s does not claim the
  end-to-end 2K tok/s gate (`would_pass_opt016` false;
  `owns_opt016_parity_gate` false). Coupled IDs:   none. Delivery re-check: `uv run pytest -q tests/test_documentation.py
  tests/test_opt019_core_recovery.py` passed (6 passed, 1 skipped).
- Acceptance evidence: [`tasks/OPT-019.md`](tasks/OPT-019.md);
  [`pins/opt019_core_recovery_contract.json`](pins/opt019_core_recovery_contract.json);
  [`fixtures/opt019_core_recovery.json`](fixtures/opt019_core_recovery.json);
  [`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](evidence/optimization/opt019-gdn-attention-core/REPORT.md);
  [`cuda/gdn_fused_quality.cuh`](cuda/gdn_fused_quality.cuh);
  [`cuda/fattn_mma_f16.cuh`](cuda/fattn_mma_f16.cuh);
  [`docs/42-cuda-gdn-chunks.md`](docs/42-cuda-gdn-chunks.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md).
  Proof is residual GDN/attention core recovery after mixer MMA, not an
  end-to-end 2K tok/s gate.
- Marked OPT-019 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. OPT-016 stays `blocked`. OPT-020 remains
  the next eligible pending by ledger row order.

### 2026-09-08T21:01:34Z — OPT-020 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-020.md`](tasks/OPT-020.md).
- Coupled IDs: none. Plan impact `none`. Composite OPT-014 `gdn`/`attention`
  become exclusive `mixer_mmq`, `gdn_core`, and `attention_core`; historical
  OPT-014 contract/fixture stay frozen.
- Marked OPT-020 `in_progress`.

### 2026-09-08T21:29:30Z — OPT-020 delivered

- Independent verification attempt 1 passed. Cold exact-2048 production
  `sync_tokens` on exclusive RTX 5090 emits exclusive mixer-projection MMQ
  versus GDN-core and attention-core (plus existing FFN/logits/commit/graph/
  idle), reconstructs host wall within the frozen remainder tolerance
  (`rel_tol=1e-4`, `abs_tol_ms=0.05`), and is retained in
  [`fixtures/opt020_prefill_split.json`](fixtures/opt020_prefill_split.json)
  (`measurement_utc` 2026-09-08T21:21:53Z; `mixer_mmq` 1200.74377,
  `gdn_core` 212.581161, `attention_core` 274.870209, `wall_ms` 2089.4812).
  Historical eight-category contract/fixture stay frozen. Coupled IDs:
  none. Delivery re-check: `uv run pytest -q tests/test_documentation.py
  tests/test_opt020_prefill_split.py tests/test_cuda_prefill_attribution.py`
  passed.
- Acceptance evidence: [`tasks/OPT-020.md`](tasks/OPT-020.md);
  [`pins/opt020_prefill_split_contract.json`](pins/opt020_prefill_split_contract.json);
  [`fixtures/opt020_prefill_split.json`](fixtures/opt020_prefill_split.json);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`docs/51-runtime-timing-and-nvtx.md`](docs/51-runtime-timing-and-nvtx.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is mixer-versus-core instrumentation, not an end-to-end 2K tok/s
  gate.
- Marked OPT-020 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. OPT-016 stays `blocked`. No remaining
  pending task is dependency-eligible (CMP-001 waits on blocked QLT-001).

### 2026-09-09T00:43:36Z — OPT-021–OPT-026 admitted (4K oracle idea ladder)

- After OPT-017–020, scout cold exact-4096 remasurement on exclusive RTX 5090
  showed Quartz mean **965.204895** tok/s versus same-sitting llama.cpp
  `llama-bench -p 4096` **3182.476587** tok/s (~3.3×). OPT-020 split still
  names `mixer_mmq` as ~57% of 2K wall.
- Compared pinned llama.cpp `cc83d7b` and local `../ds4` for technique ideas
  (not wholesale copies). Dominant leftover: mixer Q8_0 still on Rank-1 fused
  MMA, not OPT-018 quality MMQ. User-authorized accuracy relaxation: plan.md
  now admits Q8_0 quality paths under ds4 Q8 association
  (`abs > 0.05√K` AND `rel > 0.05`) while OPT-009 byte-exact reference stays.
- Admitted pending tasks (row order = try order after OPT-021 baseline):
  - OPT-021: pin cold 4K prefill oracle protocol and baseline.
  - OPT-022: mixer Q8_0 quality MMQ + shared residual Y (keep iff 4K improves).
  - OPT-023: skinny-M mixer dispatch (keep iff 4K improves).
  - OPT-024: Blackwell-aligned Q8 D2R for large mixer GEMMs (keep iff improves).
  - OPT-025: dense FFN shared-Y / SwiGLU-into-down Q8 (keep iff improves).
  - OPT-026: fattn stream-K occupancy (keep iff improves).
- Each OPT-022–OPT-026 must revert and retain a rejection if the 4K oracle does
  not strictly improve. Stop condition for the ladder: Quartz cold 4K mean
  tok/s ≥ same-protocol llama.cpp. OPT-016 remains `blocked` and remains the
  sole ≥ llama.cpp **2K** gate owner. QLT-001 remains blocked on OPT-016.
- Next eligible pending by ledger row order: **OPT-021** (deps OPT-020 done).

### 2026-09-09T00:48:00Z — OPT-021 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-021.md`](tasks/OPT-021.md).
- Coupled IDs: none. Plan impact `none`. Frozen cold exact-4096 protocol
  mirrors OPT-016 measurement mechanics (`qw38-cuda:13.0.2` /
  `qw38-llama-authority:cuda-13.0.2`, 3 unperturbed replicates, graphs
  created, llama.cpp `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99`
  first in the same sitting). Scout 965.204895 / 3182.476587 tok/s are
  not the retained fixture. Keep/reject for OPT-022+ is strictly greater
  Quartz mean tok/s; Quartz ≥ llama.cpp is informational and is not this
  gate. OPT-016 remains the blocked 2K parity owner.
- Marked OPT-021 `in_progress`.

### 2026-09-09T01:11:06Z — OPT-021 delivered

- Independent verification attempt 1 passed. Frozen cold exact-4096
  protocol and same-sitting llama.cpp `llama-bench -p 4096 -n 0
  --no-warmup -r 3 -ngl 99` are retained as the OPT-022+ keep/reject
  oracle. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt021_oracle.json`](fixtures/opt021_oracle.json)
  (`measurement_utc` 2026-09-09T01:03:49Z; Quartz mean **967.267761**
  tok/s; llama.cpp `avg_ts` **3243.626016**; `quartz_meets_llama` false
  informational). Attribution null, graphs created, production path.
  Native test does not require Quartz ≥ llama.cpp. Scout `/tmp/oracle4k/`
  is not the fixture. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt021_oracle.py`
  passed.
- Acceptance evidence: [`tasks/OPT-021.md`](tasks/OPT-021.md);
  [`pins/opt021_oracle_contract.json`](pins/opt021_oracle_contract.json);
  [`fixtures/opt021_oracle.json`](fixtures/opt021_oracle.json);
  [`cuda/prefill_4k_oracle_test.cu`](cuda/prefill_4k_oracle_test.cu);
  [`evidence/optimization/opt021-4k-oracle/REPORT.md`](evidence/optimization/opt021-4k-oracle/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/61-benchmark-harness.md`](docs/61-benchmark-harness.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is a 4K keep/reject oracle, not the 2K llama.cpp parity gate.
- Marked OPT-021 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-022**.

### 2026-09-09T01:17:03Z — OPT-022 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-022.md`](tasks/OPT-022.md).
- Coupled IDs: none. Plan impact `none` (the plan.md Q8 association
  exception for OPT-022 and later is already committed). Production mixer
  Q8_0 quality MMA uses D4 `quantize_mmq_q8_1`, packed load-tiles,
  `MMQ_ITER_K=256`, and shared residual Y in existing `prompt_q8_`.
  OPT-009 tiled-versus-reference byte equality stays unloosened. Keep iff
  cold exact-4096 mean tok/s is strictly greater than the OPT-021 Quartz
  baseline **967.267761**; otherwise revert Rank-1 production mixer Q8 and
  retain rejection evidence. OPT-016 remains the blocked 2K parity owner.
- Marked OPT-022 `in_progress`.

### 2026-09-09T03:03:00Z — OPT-022 delivered

- Independent verification attempt 1 passed. Production mixer Q8_0
  prompt MMQ is the quality stack (D4 `quantize_mmq_q8_1`, packed
  load-tiles, `MMQ_ITER_K=256`, `dim3(32, 8)`) with shared residual Y in
  existing `prompt_q8_`. OPT-009 tiled-versus-reference stays byte-exact.
  Exclusive RTX 5090 sitting wrote
  [`fixtures/opt022_mixer_q8_quality.json`](fixtures/opt022_mixer_q8_quality.json)
  (`measurement_utc` 2026-09-09T02:42:58Z; Quartz mean **1680.80627**
  tok/s versus frozen OPT-021 **967.267761**; llama.cpp `avg_ts`
  **3227.546524**; `quartz_meets_llama` false informational; `reverted`
  false; `successor_oracle` true). Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt022_mixer_q8_quality.py`
  passed.
- Acceptance evidence: [`tasks/OPT-022.md`](tasks/OPT-022.md);
  [`pins/opt022_mixer_q8_quality_contract.json`](pins/opt022_mixer_q8_quality_contract.json);
  [`fixtures/opt022_mixer_q8_quality.json`](fixtures/opt022_mixer_q8_quality.json);
  [`cuda/prefill_4k_keep_test.cu`](cuda/prefill_4k_keep_test.cu);
  [`evidence/optimization/opt022-mixer-q8-quality/REPORT.md`](evidence/optimization/opt022-mixer-q8-quality/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is mixer Q8 quality plus shared residual Y and a 4K keep versus
  the frozen oracle baseline, not the 2K llama.cpp parity gate.
- Marked OPT-022 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-023**.

### 2026-09-09T03:07:00Z — OPT-023 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-023.md`](tasks/OPT-023.md).
- Coupled IDs: none. Plan impact `none`. Skinny predicate is mixer Q8_0
  `output_rows < 128` (GDN α/β at 48). Paired CUDA-event A/B on
  `4096×48×5120` compares `mma_i128_j128` (current Fallback baseline)
  against small-I quality MMA `mma_i32_j128` / `mma_i64_j128` (shared D4 Y)
  and prompt-MMV `mmv_tiled_j1` (`launch_q8_mmq_bf16_variant` tile 1 from
  BF16). Install skinny iff a skinny id is strictly faster. Keep iff cold
  exact-4096 mean tok/s is strictly greater than the post-OPT-022 successor
  oracle **1680.80627**; otherwise revert α/β to OPT-022 I=128 J=128 quality
  MMA and retain rejection evidence. OPT-016 remains the blocked 2K parity
  owner. Decode `q8_mmv_bf16` unchanged. Large mixer GEMMs unchanged.
- Marked OPT-023 `in_progress`.

### 2026-09-09T03:57:39Z — OPT-023 delivered

- Independent verification attempt 1 passed. Production mixer Q8_0
  projections with `output_rows < 128` (GDN α/β) dispatch `mma_i32_j128`
  quality MMA on the shared residual D4 Y. Large mixer Q8 remains I=128
  quality MMA with shared Y. OPT-009 tiled-versus-reference stays
  byte-exact. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt023_skinny_mixer.json`](fixtures/opt023_skinny_mixer.json)
  (`measurement_utc` 2026-09-09T03:49:04Z; Quartz mean **1687.86169**
  tok/s versus frozen OPT-022 **1680.80627**; llama.cpp `avg_ts`
  **3232.279094**; `quartz_meets_llama` false informational; `reverted`
  false; `successor_oracle` true; A/B winner `mma_i32_j128`). Native test
  does not require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt023_skinny_mixer.py`
  passed.
- Acceptance evidence: [`tasks/OPT-023.md`](tasks/OPT-023.md);
  [`pins/opt023_skinny_mixer_contract.json`](pins/opt023_skinny_mixer_contract.json);
  [`fixtures/opt023_skinny_mixer.json`](fixtures/opt023_skinny_mixer.json);
  [`cuda/prefill_4k_skinny_test.cu`](cuda/prefill_4k_skinny_test.cu);
  [`evidence/optimization/opt023-skinny-mixer/REPORT.md`](evidence/optimization/opt023-skinny-mixer/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is skinny-M mixer dispatch under the Q8 association rule and a
  4K keep versus the post-OPT-022 oracle baseline, not the 2K llama.cpp
  parity gate.
- Marked OPT-023 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-024**.

### 2026-09-09T04:05:00Z — OPT-024 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-024.md`](tasks/OPT-024.md).
- Coupled IDs: none. Plan impact `none`. Large-mixer predicate is mixer Q8_0
  `output_rows >= 128`. Paired CUDA-event A/B on five production large
  shapes compares `quality_mma` (I=128 J=128 GGUF baseline on prequantized
  D4 Y) against `d2r_soa` (aligned SoA + int8 MMA). Install D2R iff every
  shape is strictly faster. Keep iff cold exact-4096 mean tok/s is strictly
  greater than the current successor oracle **1687.86169** from
  `fixtures/opt023_skinny_mixer.json` (OPT-023 kept; live production
  baseline). Otherwise do not install / revert and retain rejection.
  A/B-lose reject is success. OPT-016 remains the blocked 2K parity owner.
  Skinny α/β stay `mma_i32_j128`. Do not vendor `../ds4/cuda/mmq/`.
- Marked OPT-024 `in_progress`.

### 2026-09-09T04:37:09Z — OPT-024 delivered

- Independent verification attempt 1 passed. Production large mixer Q8_0
  GEMMs remain I=128 / J=128 quality MMA on the shared residual D4 Y.
  Aligned-SoA D2R is not installed. Skinny α/β stay `mma_i32_j128`. Decode
  MMV stays on GGUF 34-byte blocks. OPT-009 tiled-versus-reference stays
  byte-exact. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt024_mixer_q8_d2r.json`](fixtures/opt024_mixer_q8_d2r.json)
  (`measurement_utc` 2026-09-09T04:20:55Z; Quartz mean **1684.9541** tok/s
  versus frozen OPT-023 **1687.86169**; llama.cpp `avg_ts` **3207.633195**;
  `quartz_meets_llama` false informational; `reverted` true;
  `successor_oracle` false; `production_d2r` false; A/B winner
  `quality_mma` `win=false`). Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt024_mixer_q8_d2r.py`
  passed after the audit row was finalized.
- Acceptance evidence: [`tasks/OPT-024.md`](tasks/OPT-024.md);
  [`pins/opt024_mixer_q8_d2r_contract.json`](pins/opt024_mixer_q8_d2r_contract.json);
  [`fixtures/opt024_mixer_q8_d2r.json`](fixtures/opt024_mixer_q8_d2r.json);
  [`cuda/prefill_4k_d2r_test.cu`](cuda/prefill_4k_d2r_test.cu);
  [`evidence/optimization/opt024-mixer-q8-d2r/REPORT.md`](evidence/optimization/opt024-mixer-q8-d2r/REPORT.md);
  [`evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md`](evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is aligned-SoA D2R for large mixer Q8_0 under the Q8 association
  rule when it beats quality MMQ, and a retained 4K reject versus the
  current successor-oracle baseline, not the 2K llama.cpp parity gate.
- Marked OPT-024 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-025**.

### 2026-09-09T04:40:00Z — OPT-025 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-025.md`](tasks/OPT-025.md).
- Coupled IDs: none. Plan impact `none`. Then-current oracle denominator is
  OPT-023 Quartz mean **1687.86169** (`fixtures/opt023_skinny_mixer.json`;
  OPT-024 rejected, `successor_oracle` false). A/B candidates: `baseline`,
  `shared_y`, `swiglu_q8`, `shared_y_swiglu_q8`. Keep iff cold exact-4096
  mean tok/s strictly beats **1687.86169**; else revert and retain rejection.
- Marked OPT-025 `in_progress`.

### 2026-09-09T05:25:00Z — OPT-025 delivered

- Independent verification attempt 1 passed. Production prompt FFN Q4_K
  shares one DS4 Q8_1 of the FFN-norm activation for gate and up and writes
  down-leg Y from SwiGLU without a global BF16 mid store
  (`kSelectedFfnPath = shared_y_swiglu_q8`). Mixer Q8 remains quality MMA
  plus skinny `mma_i32_j128`. Decode FFN stays MMV. OPT-009
  tiled-versus-reference stays byte-exact. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt025_ffn_shared_y.json`](fixtures/opt025_ffn_shared_y.json)
  (`measurement_utc` 2026-09-09T05:14:45Z; Quartz mean **1709.21912**
  tok/s versus frozen OPT-023 **1687.86169**; llama.cpp `avg_ts`
  **3266.276516**; `quartz_meets_llama` false informational; `reverted`
  false; `successor_oracle` true; A/B winner `shared_y_swiglu_q8`). Native
  test does not require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery
  re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt025_ffn_shared_y.py`.
- Acceptance evidence: [`tasks/OPT-025.md`](tasks/OPT-025.md);
  [`pins/opt025_ffn_shared_y_contract.json`](pins/opt025_ffn_shared_y_contract.json);
  [`fixtures/opt025_ffn_shared_y.json`](fixtures/opt025_ffn_shared_y.json);
  [`cuda/prefill_4k_ffn_test.cu`](cuda/prefill_4k_ffn_test.cu);
  [`evidence/optimization/opt025-ffn-shared-y/REPORT.md`](evidence/optimization/opt025-ffn-shared-y/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is dense FFN shared-Y and SwiGLU-into-down Q8 under the Q4_K
  association rule and a 4K keep versus the OPT-023 oracle baseline, not
  the 2K llama.cpp parity gate.
- Marked OPT-025 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-026**.

### 2026-09-09T05:30:00Z — OPT-026 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-026.md`](tasks/OPT-026.md).
- Coupled IDs: none. Plan impact `none`. Then-current oracle denominator
  OPT-025 Quartz mean **1709.21912**. A/B on production 4096-row fattn
  among `baseline`, `occ2`, `stream_k`; install iff an optimized id is
  strictly faster under frozen OPT-005. Keep iff cold exact-4096 mean
  tok/s > **1709.21912** else revert and retain rejection. Last 4K
  idea-ladder task (`ladder_exhausted` true either way). OPT-016 remains
  the blocked 2K parity owner.
- Marked OPT-026 `in_progress`.

### 2026-09-09T05:55:00Z — OPT-026 delivered

- Independent verification attempt 1 passed. Production prompt fattn uses
  Ada+ stream-K KV bipartition plus softmax combine
  (`kSelectedFattnPath = stream_k`) after a paired A/B win versus
  occupancy-1 whole-tile. Occupancy-2 whole-tile was eligible but slower
  than stream-K. Mixer Q8 remains quality MMA plus skinny `mma_i32_j128`.
  FFN remains `shared_y_swiglu_q8`. Decode attention stays one-token.
  Tiled attention remains the unloosened reference. Exclusive RTX 5090
  sitting wrote
  [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  (`measurement_utc` 2026-09-09T05:51:48Z; Quartz mean **1746.71973**
  tok/s versus frozen OPT-025 **1709.21912**; llama.cpp `avg_ts`
  **3253.993621**; `quartz_meets_llama` false informational; `reverted`
  false; `successor_oracle` true; A/B winner `stream_k`;
  `ladder_exhausted` true). Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt026_fattn_streamk.py`.
- Acceptance evidence: [`tasks/OPT-026.md`](tasks/OPT-026.md);
  [`pins/opt026_fattn_streamk_contract.json`](pins/opt026_fattn_streamk_contract.json);
  [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json);
  [`cuda/prefill_4k_fattn_test.cu`](cuda/prefill_4k_fattn_test.cu);
  [`evidence/optimization/opt026-fattn-streamk/REPORT.md`](evidence/optimization/opt026-fattn-streamk/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is fattn occupancy / stream-K under unloosened envelopes and a
  4K keep versus the OPT-025 oracle baseline, not the 2K llama.cpp
  parity gate. The 4K idea ladder is exhausted; Quartz 4K still does not
  beat llama.cpp.
- Marked OPT-026 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. No further OPT-022–OPT-026 tasks. Next pending by
  ledger row order remains CMP-001, still blocked on QLT-001.

### 2026-09-09T06:12:00Z — speedup-loop scout; OPT-027–OPT-031 admitted

- OPT-022–OPT-026 exhausted without Quartz 4K ≥ llama.cpp. Post-OPT-026
  exclusive RTX 5090 CUDA-event scout (rebuilt production objects):
  - Cold exact-2048: wall 997.08 ms / 2054.0 tok/s; shares `ffn_mmq` 39.2%,
    `attention_core` 24.7%, `gdn_core` 21.2%, `mixer_mmq` 14.6% (graphs=0).
  - Cold exact-4096: wall 2348.33 ms / 1744.2 tok/s; shares
    `attention_core` **36.9%**, `ffn_mmq` 31.4%, `gdn_core` 19.3%,
    `mixer_mmq` 12.2% (graphs=64). Oracle-length next pick is attention.
- Compared pinned llama.cpp `cc83d7b` (and `../ds4` inspiration only) for
  unused transferable techniques. Excluded sparse/compressed attention,
  MoE, mHC, DSpark, production cuBLAS/cuDNN, and rejected OPT-024 mixer
  Q8 D2R.
- Admitted pending second 4K idea ladder (row order = try order; keep iff
  cold exact-4096 mean tok/s strictly beats then-current successor oracle
  starting from OPT-026 **1746.71973**; else revert + rejection):
  - OPT-027: persistent Ada+ fattn stream-K (`attention_core`).
  - OPT-028: Q4_K/Q6_K MMQ stream-K (`ffn_mmq`).
  - OPT-029: fuse GDN conv + gated output (`gdn_core`).
  - OPT-030: Hopper/Blackwell PDL prompt launches (cross-cutting).
  - OPT-031: mixer + GDN 4096 prompt graphs (`mixer_mmq` / `gdn_core`).
- Evidence: [`speedup-plan.md`](speedup-plan.md);
  [`evidence/optimization/speedup-loop-post026/`](evidence/optimization/speedup-loop-post026/).
  OPT-016 remains `blocked`. QLT-001 remains blocked on OPT-016.
- Next eligible pending by ledger row order: **OPT-027** (deps OPT-026 done).

### 2026-09-09T07:10:00Z — OPT-027 delivered

- Independent verification attempt 1 passed. Production prompt fattn remains
  OPT-026 Ada+ stream-K KV bipartition (`kSelectedFattnPath = stream_k`,
  `grid.z=2`). Persistent linearized stream-K is not installed
  (`kSelectedPersistentFattnPath = off`). Mixer Q8 remains quality MMA plus
  skinny `mma_i32_j128`. FFN remains `shared_y_swiglu_q8`. Decode attention
  stays one-token. Tiled attention remains the unloosened reference.
  Exclusive RTX 5090 sitting wrote
  [`fixtures/opt027_persistent_fattn.json`](fixtures/opt027_persistent_fattn.json)
  (`measurement_utc` 2026-09-09T06:42:08Z; Quartz mean **1734.68005** tok/s
  versus frozen OPT-026 **1746.71973**; llama.cpp `avg_ts` **3193.705927**;
  `quartz_meets_llama` false informational; `reverted` true;
  `successor_oracle` false; `production_persistent_installed` false; A/B
  winner `stream_k` `win=false`; `ladder_exhausted` false). Throughput
  delta: baseline [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  **1746.71973** → measured post **1734.68005**; speedup `0`
  (reverted/not installed). Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt027_persistent_fattn.py`.
- Acceptance evidence: [`tasks/OPT-027.md`](tasks/OPT-027.md);
  [`pins/opt027_persistent_fattn_contract.json`](pins/opt027_persistent_fattn_contract.json);
  [`fixtures/opt027_persistent_fattn.json`](fixtures/opt027_persistent_fattn.json);
  [`cuda/prefill_4k_persistent_fattn_test.cu`](cuda/prefill_4k_persistent_fattn_test.cu);
  [`evidence/optimization/opt027-persistent-fattn/REPORT.md`](evidence/optimization/opt027-persistent-fattn/REPORT.md);
  [`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](evidence/optimization/opt027-persistent-fattn/REJECTION.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is persistent Ada+ fattn stream-K under unloosened envelopes when
  a paired A/B wins, and a retained 4K reject versus the then-current
  successor-oracle baseline, not the 2K llama.cpp parity gate. The second
  4K idea ladder is not exhausted.
- Marked OPT-027 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-028**.

### 2026-09-09T10:10:00Z — OPT-028 delivered

- Independent verification attempt 1 passed. Production Q4_K/Q6_K quality
  MMA remains 2D tiling (`kSelectedMmqStreamKPath = off`). Stream-K
  kernels remain non-production symbols. Mixer Q8 remains quality MMA plus
  skinny `mma_i32_j128`. FFN remains `shared_y_swiglu_q8`. Fattn remains
  OPT-026 `stream_k`. Decode FFN stays MMV. Exclusive RTX 5090 sitting
  wrote
  [`fixtures/opt028_mmq_streamk.json`](fixtures/opt028_mmq_streamk.json)
  (`measurement_utc` 2026-09-09T08:06:02Z; Quartz mean **1729.0459** tok/s
  versus frozen OPT-026 **1746.71973**; llama.cpp `avg_ts` **3231.577696**;
  `quartz_meets_llama` false informational; `reverted` true;
  `successor_oracle` false; `production_mmq_stream_k_installed` false; A/B
  winner `off` `win=false`; `ladder_exhausted` false). Throughput
  delta: baseline [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  **1746.71973** → measured post **1729.0459**; speedup `0`
  (reverted/not installed). Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt028_mmq_streamk.py`.
- Acceptance evidence: [`tasks/OPT-028.md`](tasks/OPT-028.md);
  [`pins/opt028_mmq_streamk_contract.json`](pins/opt028_mmq_streamk_contract.json);
  [`fixtures/opt028_mmq_streamk.json`](fixtures/opt028_mmq_streamk.json);
  [`cuda/prefill_4k_mmq_streamk_test.cu`](cuda/prefill_4k_mmq_streamk_test.cu);
  [`evidence/optimization/opt028-mmq-streamk/REPORT.md`](evidence/optimization/opt028-mmq-streamk/REPORT.md);
  [`evidence/optimization/opt028-mmq-streamk/REJECTION.md`](evidence/optimization/opt028-mmq-streamk/REJECTION.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is Q4_K/Q6_K MMQ stream-K under unloosened association when a
  paired A/B wins, and a retained 4K reject versus the then-current
  successor-oracle baseline, not the 2K llama.cpp parity gate. The second
  4K idea ladder is not exhausted.
- Marked OPT-028 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-029**.

### 2026-09-09T10:53:00Z — OPT-029 delivered

- Independent verification attempt 1 passed. Production prompt GDN remains
  split parallel conv + warp-column + `gdn_gated_output_rows`
  (`kSelectedGdnFusePath = off`). Fused kernels remain non-production
  symbols. Mixer Q8 remains quality MMA plus skinny `mma_i32_j128`. FFN
  remains `shared_y_swiglu_q8`. Fattn remains OPT-026 `stream_k`. Decode
  GDN stays split tiled prepare + `launch_gdn_gated_output`. Exclusive
  RTX 5090 sitting wrote
  [`fixtures/opt029_gdn_fuse.json`](fixtures/opt029_gdn_fuse.json)
  (`measurement_utc` 2026-09-09T10:38:37Z; Quartz mean **1725.36658** tok/s
  versus frozen OPT-026 **1746.71973**; llama.cpp `avg_ts` **3187.39006**;
  `quartz_meets_llama` false informational; `reverted` true;
  `successor_oracle` false; `production_gdn_fuse_installed` false; A/B
  winner `off` `win=false`; `ladder_exhausted` false). Throughput
  delta: baseline [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  **1746.71973** → measured post **1725.36658**; speedup `0`
  (reverted/not installed). Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt029_gdn_fuse.py`.
- Acceptance evidence: [`tasks/OPT-029.md`](tasks/OPT-029.md);
  [`pins/opt029_gdn_fuse_contract.json`](pins/opt029_gdn_fuse_contract.json);
  [`fixtures/opt029_gdn_fuse.json`](fixtures/opt029_gdn_fuse.json);
  [`cuda/prefill_4k_gdn_fuse_test.cu`](cuda/prefill_4k_gdn_fuse_test.cu);
  [`evidence/optimization/opt029-gdn-fuse/REPORT.md`](evidence/optimization/opt029-gdn-fuse/REPORT.md);
  [`evidence/optimization/opt029-gdn-fuse/REJECTION.md`](evidence/optimization/opt029-gdn-fuse/REJECTION.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/42-cuda-gdn-chunks.md`](docs/42-cuda-gdn-chunks.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is fused GDN conv and/or gated-output under unloosened GDN-002
  envelopes when a paired A/B wins, and a retained 4K reject versus the
  then-current successor-oracle baseline, not the 2K llama.cpp parity
  gate. The second 4K idea ladder is not exhausted.
- Marked OPT-029 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 stays
  `blocked`. Next eligible pending by ledger row order: **OPT-030**.

### 2026-09-09T00:00:00Z — speedup plan drafted

- Replaced [`speedup-plan.md`](speedup-plan.md) with the measurement-led
  post-OPT-026 ladder and its frozen P/D128/D2048 protocols.
- Added six independent pending rows, OPT-032 through OPT-037, with the
  supplied dependencies, acceptance conditions, and cross-workload guard.
- Planning only: no implementation, benchmark, task dossier, commit, or push
  was performed. Existing OPT-030 worktree state is preserved.

### 2026-09-09T12:23:50Z — OPT-032 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-032.md`](tasks/OPT-032.md).
- Coupled IDs: none. Plan impact `none`. This increment claims no
  performance improvement. P reuses the frozen OPT-021 protocol on a
  fresh exclusive sitting. D128/D2048 freeze batch-1, exact 128/2048
  prefix tokens, 256 predetermined one-token evaluations, 3 warm-ups
  and 30 measured runs, graphs enabled, prefix excluded from the timed
  region. Matched llama.cpp decode uses a new public-API driver;
  `llama-bench -p 0 -n 256 -d 2048` is informational. Exclusive decode
  categories are opt-in `DecodeAttribution` (`mixer_mmv`, `gdn_core`,
  `attention_core`, `ffn_mmv`); public `RuntimeTimings` stays composite.
  Next-task order is recorded from live P versus D2048 gaps and the
  D2048 MMV-versus-attention_core ranking. Historical 4K successor
  oracle remains [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  **1746.71973** tok/s until the live sitting publishes a fresh P without
  a speedup claim. OPT-030 left `in_progress` and unmutated. OPT-016
  remains the blocked 2K parity owner.
- Marked OPT-032 `in_progress`.

### 2026-09-09T16:12:00Z — OPT-030 delivered (reject)

- Independent verification attempt 1 had already passed; this entry closes
  deferred delivery bookkeeping after evidence landed in
  `e8cc8b7475c2849fe188d8164c83874b9e17bb32`. Production ungraphed prompt
  launches remain ordinary `<<<>>>` (`kSelectedPdlPath = off`). PDL wrapper
  and A/B diagnostic remain non-production. A/B winner was `pdl`
  (`win=true`; `off` 76.1488266 ms, `pdl_host` 76.1686554 ms, `pdl`
  76.0484467 ms; byte-equal), but exclusive RTX 5090 cold exact-4096 mean
  tok/s **1734.4137** did not strictly beat frozen OPT-026
  [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  **1746.71973**. Fixture
  [`fixtures/opt030_pdl_launches.json`](fixtures/opt030_pdl_launches.json)
  (`measurement_utc` 2026-09-09T11:34:14Z; `status` rejected; `reverted`
  true; `successor_oracle` false; `production_pdl_installed` false;
  llama.cpp `avg_ts` **3202.544049**; `ladder_exhausted` false).
  Throughput delta: baseline **1746.71973** → measured post **1734.4137**;
  speedup `0` (reverted/not installed). Coupled IDs: none. Delivery
  re-check: `uv run pytest -q tests/test_documentation.py tests/test_opt030_pdl_launches.py`.
- Acceptance evidence: [`tasks/OPT-030.md`](tasks/OPT-030.md);
  [`pins/opt030_pdl_launches_contract.json`](pins/opt030_pdl_launches_contract.json);
  [`fixtures/opt030_pdl_launches.json`](fixtures/opt030_pdl_launches.json);
  [`cuda/prefill_4k_pdl_test.cu`](cuda/prefill_4k_pdl_test.cu);
  [`evidence/optimization/opt030-pdl-launches/REPORT.md`](evidence/optimization/opt030-pdl-launches/REPORT.md);
  [`evidence/optimization/opt030-pdl-launches/REJECTION.md`](evidence/optimization/opt030-pdl-launches/REJECTION.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/53-stable-address-cuda-graphs.md`](docs/53-stable-address-cuda-graphs.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is PDL keep/reject versus the then-current successor-oracle
  baseline with retained rejection, not the 2K llama.cpp parity gate.
- Marked OPT-030 `done`; delivery is limited to this ledger/audit
  bookkeeping on top of the already-committed reject evidence. `plan.md`
  is unchanged. OPT-016 stays `blocked`. OPT-032 remains independently
  `in_progress`.

### 2026-09-09T16:28:45Z — OPT-032 delivered (measurement; no speedup)

- Independent verification attempt 1 passed. This increment installs no
  production kernel and claims no performance improvement
  (`claims_performance_improvement` false; speedup N/A). Exclusive RTX
  5090 sitting `measurement_utc` 2026-09-09T15:07:45Z wrote
  [`fixtures/opt032_decode_oracle.json`](fixtures/opt032_decode_oracle.json)
  with complete P, D128, and D2048 protocols, exclusive decode
  categories, matched pinned llama.cpp public-API measurements, and a
  host-recomputed `next_task_order`. Historical OPT-026 P
  [`fixtures/opt026_fattn_streamk.json`](fixtures/opt026_fattn_streamk.json)
  **1746.71973** tok/s → fresh P Quartz **1637.58594** (llama.cpp
  `avg_ts` **3139.909678**). D128 Quartz **11.8731956** vs llama.cpp
  **68.506172**. D2048 Quartz **3.70951414** vs llama.cpp **66.9333082**.
  `p_gap` 1.9174014635225802; `d2048_gap` 18.04368595829102;
  `decode_deficit_larger` true. `next_task_order`
  `["OPT-036", "OPT-033", "OPT-035", "OPT-034", "OPT-037"]`. Native test
  does not require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery
  re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt032_decode_oracle.py`.
- Acceptance evidence: [`tasks/OPT-032.md`](tasks/OPT-032.md);
  [`pins/opt032_decode_oracle_contract.json`](pins/opt032_decode_oracle_contract.json);
  [`fixtures/opt032_decode_oracle.json`](fixtures/opt032_decode_oracle.json);
  [`evidence/optimization/opt032-decode-oracle/REPORT.md`](evidence/optimization/opt032-decode-oracle/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/51-runtime-timing-and-nvtx.md`](docs/51-runtime-timing-and-nvtx.md);
  [`docs/61-benchmark-harness.md`](docs/61-benchmark-harness.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is frozen P/D128/D2048 oracles, exclusive decode attribution,
  matched llama.cpp measurements, and a recorded next-task order, not a
  throughput keep and not the 2K llama.cpp parity gate.
- Marked OPT-032 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. OPT-033–OPT-037 remain `pending`. Next eligible
  pending by ledger row order: **OPT-031**.

### 2026-09-09T16:35:00Z — OPT-036 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-036.md`](tasks/OPT-036.md).
- Coupled IDs: none. Plan impact `none`. Chosen sink is D2048 decode
  `attention_core` **201.241058 ms** from
  [`fixtures/opt032_decode_oracle.json`](fixtures/opt032_decode_oracle.json).
  Keep denominators are that fixture's P **1637.58594**, D128
  **11.8731956**, D2048 **3.70951414** (and the recorded p95s), not
  historical OPT-026 P **1746.71973**. A/B is 1/4/8/16 contiguous KV
  partitions at positions 128 and 2048, 3 warm + 30 alternating,
  deterministic ascending-part FP32 merge, independent selection including
  the current one-partition kernel. Keep only with frozen attention
  envelopes, strictly lower D2048 component time, improved D2048 tok/s,
  and the cross-workload guard; otherwise revert with retained evidence.
- Marked OPT-036 `in_progress`. OPT-033, OPT-034, OPT-035, and OPT-037
  remain `pending` and are not started. Non-final audit row inserted.
  No commit. `plan.md` is unchanged.

### 2026-09-09T17:35:00Z — OPT-036 delivered (KEEP)

- Independent verification attempt 1 passed. Production decode attention
  installs 16 KV partitions below 2048 and at or above 2048
  (`kSelectedDecodeKvPartsLow/High = 16/16`) after independent 1/4/8/16
  A/B wins at positions 128 and 2048 with deterministic ascending-part
  FP32 merge. Candidate `1` remains the tiled kernel. Prefill fattn
  stream-K is unchanged. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt036_decode_kv_partition.json`](fixtures/opt036_decode_kv_partition.json)
  (`measurement_utc` 2026-09-09T17:08:11Z; D2048 Quartz **3.70951414** →
  **13.5282431** tok/s; D128 Quartz **11.8731956** → **15.200716**; P
  Quartz **1637.58594** → **1644.04822**; D128 all-token p95 95.380661 →
  66.3502579 and run-mean p95 84.2481613 → 65.8343124; D2048 all-token
  p95 280.727844 → 74.4495544 and run-mean p95 269.61731 → 73.996994;
  A/B winners 16/16; D2048 component 12.2984858 → 0.804042637 ms;
  `reverted` false; `status` measured). Cross-workload guard held.
  Native test does not require Quartz ≥ llama.cpp. Coupled IDs: none.
  Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt036_decode_kv_partition.py`.
- Acceptance evidence: [`tasks/OPT-036.md`](tasks/OPT-036.md);
  [`pins/opt036_decode_kv_partition_contract.json`](pins/opt036_decode_kv_partition_contract.json);
  [`fixtures/opt036_decode_kv_partition.json`](fixtures/opt036_decode_kv_partition.json);
  [`cuda/decode_kv_partition_ab_test.cu`](cuda/decode_kv_partition_ab_test.cu);
  [`evidence/optimization/opt036-decode-kv-partition/REPORT.md`](evidence/optimization/opt036-decode-kv-partition/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/43-cuda-attention-decode.md`](docs/43-cuda-attention-decode.md).
  Proof is partitioned decode attention under frozen envelopes and a
  D2048 keep versus decode-oracle P/D128/D2048 denominators, not the 2K
  llama.cpp parity gate.
- Marked OPT-036 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. OPT-033, OPT-034, OPT-035, and OPT-037 remain
  `pending` and are not started. Next eligible pending by ledger row
  order: **OPT-031**.

### 2026-09-09T17:45:00Z — OPT-033 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-033.md`](tasks/OPT-033.md).
- Coupled IDs: none. Plan impact `none`. Chosen sink is prefill
  `attention_core`. Keep denominators are OPT-036 post-keep P
  **1644.04822**, D128 **15.200716**, D2048 **13.5282431** (and the
  recorded p95s), not historical OPT-026 P **1746.71973**. A/B is
  `global` versus `registers` at 4096 rows including combine, 3 warm +
  30 alternating, byte-equal output, frozen Ncols1=16, Ncols2=2, KV
  tile 32, dual-F16 Q, `grid.z=2`. Keep only with byte equality, strictly
  lower component time, improved P, and the cross-workload guard;
  otherwise revert with retained evidence. Spills or occupancy loss that
  erase the A/B win are a reject.
- Marked OPT-033 `in_progress`. OPT-034, OPT-035, and OPT-037 remain
  `pending` and are not started. Non-final audit row inserted.
  No commit. `plan.md` is unchanged.

### 2026-09-09T18:45:00Z — OPT-033 delivered (KEEP)

- Independent verification attempt 1 passed. Production prompt stream-K
  installs register-resident value accumulation
  (`kSelectedVkqAccum = "registers"`) after the 4096-row A/B including
  combine: `registers` is byte-equal versus `global` and strictly faster
  (global **53.8520393** ms → registers **44.8948784** ms). Exclusive
  RTX 5090 sitting wrote
  [`fixtures/opt033_register_vkq.json`](fixtures/opt033_register_vkq.json)
  (`measurement_utc` 2026-09-09T18:11:19Z; P Quartz **1644.04822** →
  **1745.10315** tok/s; D128 Quartz **15.200716** → **15.1528101**;
  D2048 Quartz **13.5282431** → **13.5596962**; D128 all-token p95
  66.3502579 → 66.5563431 and run-mean p95 65.8343124 → 66.0279617;
  D2048 all-token p95 74.4495544 → 74.2697372 and run-mean p95
  73.996994 → 73.807579; occupancy 1 on both; OPT-005 vs tiled
  max_abs 9.83476639e-07, rms 2.90231217e-08; `reverted` false;
  `status` measured). Cross-workload guard held. Native test does not
  require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt033_register_vkq.py`.
- Acceptance evidence: [`tasks/OPT-033.md`](tasks/OPT-033.md);
  [`pins/opt033_register_vkq_contract.json`](pins/opt033_register_vkq_contract.json);
  [`fixtures/opt033_register_vkq.json`](fixtures/opt033_register_vkq.json);
  [`cuda/fattn_register_vkq_ab_test.cu`](cuda/fattn_register_vkq_ab_test.cu);
  [`evidence/optimization/opt033-register-vkq/REPORT.md`](evidence/optimization/opt033-register-vkq/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is register-resident prefill attention value sums under frozen
  envelopes versus then-current accepted P/D128/D2048 denominators, not
  the 2K llama.cpp parity gate.
- Marked OPT-033 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. OPT-034, OPT-035, and OPT-037 remain `pending` and
  are not started. Next eligible pending by ledger row order:
  **OPT-031**.

### 2026-09-09T18:47:00Z — OPT-035 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-035.md`](tasks/OPT-035.md).
- Coupled IDs: none. Plan impact `none`. Chosen sink is prefill
  `attention_core`. Keep denominators are OPT-033 post-keep P
  **1745.10315**, D128 **15.1528101**, D2048 **13.5596962** (and the
  recorded p95s), not historical OPT-026 P **1746.71973**. A/B is
  `scalar` versus `mma` at 4096 rows including combine, 3 warm +
  30 alternating, dual-F16 probability × F16 V MMA with FP32 C,
  FP32 online max/denominator, frozen Ncols1=16, Ncols2=2, KV tile 32,
  dual-F16 Q, `grid.z=2`, register VKQ. Keep only with frozen
  envelopes versus tiled, strictly lower component time, improved P,
  and the cross-workload guard; otherwise revert with retained
  evidence. Byte equality versus scalar is not the keep predicate.
  Spills or occupancy loss that erase the A/B win are a reject.
- Marked OPT-035 `in_progress`. OPT-034 and OPT-037 remain `pending`
  and are not started. Non-final audit row inserted. No commit.
  `plan.md` is unchanged.

### 2026-09-09T20:00:00Z — OPT-035 delivered (KEEP)

- Independent verification attempt 1 passed. Production prompt stream-K
  installs dual-F16 probability×V MMA (`kSelectedPvPath = "mma"`) after
  the 4096-row A/B including combine: `mma` meets frozen OPT-005 envelopes
  versus tiled and is strictly faster (scalar **44.8905029** ms → mma
  **35.445816** ms). Exclusive RTX 5090 sitting wrote
  [`fixtures/opt035_pv_mma.json`](fixtures/opt035_pv_mma.json)
  (`measurement_utc` 2026-09-09T19:20:04Z; P Quartz **1745.10315** →
  **1865.21155** tok/s; D128 Quartz **15.1528101** → **15.0562878**;
  D2048 Quartz **13.5596962** → **13.5411425**; D128 all-token p95
  66.5563431 → 66.6085587 and run-mean p95 66.0279617 → 66.06633;
  D2048 all-token p95 74.2697372 → 74.366951 and run-mean p95
  73.807579 → 73.9000702; occupancy 1 on both; OPT-005 vs tiled
  max_abs 9.76026058e-07, rms 2.94235409e-08; `reverted` false;
  `status` measured). Cross-workload guard held. Native test does not
  require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py tests/test_opt035_pv_mma.py`.
- Acceptance evidence: [`tasks/OPT-035.md`](tasks/OPT-035.md);
  [`pins/opt035_pv_mma_contract.json`](pins/opt035_pv_mma_contract.json);
  [`fixtures/opt035_pv_mma.json`](fixtures/opt035_pv_mma.json);
  [`cuda/fattn_pv_mma_ab_test.cu`](cuda/fattn_pv_mma_ab_test.cu);
  [`evidence/optimization/opt035-pv-mma/REPORT.md`](evidence/optimization/opt035-pv-mma/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is dual-F16 probability×V MMA under frozen envelopes versus
  then-current accepted P/D128/D2048 denominators, not the 2K llama.cpp
  parity gate.
- Marked OPT-035 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. OPT-034 and OPT-037 remain `pending` and are not
  started. Next eligible pending by ledger row order: **OPT-031**.

### 2026-09-09T19:57:11Z — OPT-034 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-034.md`](tasks/OPT-034.md).
  Coupled IDs: none. Unresolved decisions: none. Plan impact: none.
- Then-current keep denominators copied from
  [`fixtures/opt035_pv_mma.json`](fixtures/opt035_pv_mma.json): Quartz P
  **1865.21155**, D128 **15.0562878**, D2048 **13.5411425**, D128
  all-token p95 **66.6085587** / run-mean p95 **66.06633**, D2048
  all-token p95 **74.366951** / run-mean p95 **73.9000702**. P retain
  floor is ≥95% of OPT-035 P. D2048 must strictly improve.
- Chosen sink: decode Q4_K/Q6_K MMV packed blockwise loads with
  unchanged Q8 staging, warp-count dispatch, and FP32 lane/reduction
  order. Byte equality versus the current elementwise kernel is a keep
  predicate. Q8_0 direct-BF16 and DP4A are out of scope. OPT-037 is not
  started.
- Marked OPT-034 `in_progress`. OPT-037 remains `pending`. Non-final
  audit row inserted. No commit. `plan.md` is unchanged.

### 2026-09-09T21:42:00Z — OPT-034 delivered (KEEP)

- Independent verification attempt 1 passed. Production decode Q4_K/Q6_K
  MMV installs packed blockwise field/scale loads
  (`kSelectedMmvLoadPath = "packed"`) after the production-shape A/B:
  `packed` is byte-equal and strictly faster (elementwise weighted mean
  **0.162414238** ms → packed **0.0734361857** ms). Exclusive RTX 5090
  sitting wrote
  [`fixtures/opt034_packed_mmv.json`](fixtures/opt034_packed_mmv.json)
  (`measurement_utc` 2026-09-09T20:44:11Z; D2048 Quartz **13.5411425** →
  **20.169548** tok/s; P Quartz **1865.21155** → **1869.84412**; D128
  Quartz **15.0562878** → **25.3816128**; D128 all-token p95 66.6085587 →
  39.9736366 and run-mean p95 66.06633 → 39.4155655; D2048 all-token p95
  74.366951 → 50.2872772 and run-mean p95 73.9000702 → 49.590683;
  `reverted` false; `status` measured). Cross-workload guard held. Native
  test does not require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery
  re-check: `uv run pytest -q tests/test_documentation.py
  tests/test_opt034_packed_mmv.py`.
- Acceptance evidence: [`tasks/OPT-034.md`](tasks/OPT-034.md);
  [`pins/opt034_packed_mmv_contract.json`](pins/opt034_packed_mmv_contract.json);
  [`fixtures/opt034_packed_mmv.json`](fixtures/opt034_packed_mmv.json);
  [`cuda/packed_mmv_ab_test.cu`](cuda/packed_mmv_ab_test.cu);
  [`evidence/optimization/opt034-packed-mmv/REPORT.md`](evidence/optimization/opt034-packed-mmv/REPORT.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/39-cuda-quant-mmv.md`](docs/39-cuda-quant-mmv.md).
  Proof is packed blockwise Q4_K/Q6_K MMV loads under byte equality
  versus then-current accepted P/D128/D2048 denominators, not the 2K
  llama.cpp parity gate.
- Marked OPT-034 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. OPT-037 remains `pending` and is not started. Next
  eligible pending by ledger row order: **OPT-031**.

### 2026-09-09T21:47:38Z — OPT-037 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-037.md`](tasks/OPT-037.md).
  Coupled IDs: none. Unresolved decisions: none. Plan impact: none.
- Then-current keep denominators copied from
  [`fixtures/opt034_packed_mmv.json`](fixtures/opt034_packed_mmv.json): Quartz P
  **1869.84412**, D128 **25.3816128**, D2048 **20.169548**, D128
  all-token p95 **39.9736366** / run-mean p95 **39.4155655**, D2048
  all-token p95 **50.2872772** / run-mean p95 **49.590683**. P must
  strictly improve. D2048 tok/s improvement is not required for this
  prefill keep.
- Chosen sink: 4096-row Q4_K FFN quality MMA I/J per projection (gate,
  up, down independently) with shared-Y preserved, 2D scheduling, and
  graph recapture if tiles change. Mixer Q8 and global
  `selected_mma_mmq_prompt_tile()` stay 128. Stream-K stays off.
- Marked OPT-037 `in_progress`. Non-final audit row inserted. No commit.
  `plan.md` is unchanged.

### 2026-09-09T23:03:58Z — OPT-037 delivered (reject)

- Independent verification attempt 1 passed. Production 4096-row Q4_K FFN
  quality MMA remains I=128 / J=128 on every projection
  (`kSelectedFfn{Gate,Up,Down}{QualityI,PromptTile}` = 128). Shared-Y /
  SwiGLU-into-Q8 stays. 2D scheduling stays. Mixer Q8 remains quality MMA
  plus skinny `mma_i32_j128`. Global `selected_mma_mmq_prompt_tile()`
  stays 128. Stream-K stays `off`. Packed MMV stays `packed`. Prompt FFN
  graphs were not recaptured. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt037_ffn_tiles.json`](fixtures/opt037_ffn_tiles.json)
  (`measurement_utc` 2026-09-09T22:23:11Z; `any_win=false`; gate/up/down
  winners `i128_j128`; `keep_sitting_skipped` true; `reverted` true;
  `status` rejected). Throughput delta: P **1869.84412** → **1869.84412**;
  D128 **25.3816128** → **25.3816128**; D2048 **20.169548** →
  **20.169548** (tok/s sitting skipped; keep denominators unchanged).
  Native test does not require Quartz ≥ llama.cpp. Coupled IDs: none.
  Delivery re-check: `uv run pytest -q tests/test_documentation.py
  tests/test_opt037_ffn_tiles.py`.
- Acceptance evidence: [`tasks/OPT-037.md`](tasks/OPT-037.md);
  [`pins/opt037_ffn_tile_contract.json`](pins/opt037_ffn_tile_contract.json);
  [`fixtures/opt037_ffn_tiles.json`](fixtures/opt037_ffn_tiles.json);
  [`cuda/ffn_tile_ab_test.cu`](cuda/ffn_tile_ab_test.cu);
  [`tests/test_opt037_ffn_tiles.py`](tests/test_opt037_ffn_tiles.py);
  [`evidence/optimization/opt037-ffn-tiles/REPORT.md`](evidence/optimization/opt037-ffn-tiles/REPORT.md);
  [`evidence/optimization/opt037-ffn-tiles/REJECTION.md`](evidence/optimization/opt037-ffn-tiles/REJECTION.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/40-cuda-prompt-mmq.md`](docs/40-cuda-prompt-mmq.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is 4K FFN I/J per projection under unloosened Q4_K association
  when a paired A/B wins, and a retained reject with production 128/128
  pins versus then-current accepted P/D128/D2048 denominators, not the 2K
  llama.cpp parity gate.
- Marked OPT-037 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-031**.

### 2026-09-10T06:12:20Z — speedup-loop scout; OPT-038–OPT-042 admitted

- Source-only post-OPT-037 planning; no CUDA implementation or benchmark was run.
- Current accepted denominators remain OPT-034: P 1869.84412, D128 25.3816128,
  D2048 20.169548 tok/s, with its recorded p95 values. OPT-037 rejected all
  candidates and did not move the denominator.
- OPT-038 records the required fresh attribution/source-gap study and explicitly
  claims no performance improvement. OPT-039 covers warp-owned decode attention;
  OPT-040 hoists prompt GDN inverse norms; OPT-041 gives prompt QK microtiles
  warp ownership; OPT-042 is a diagnostic-only integer decode-MMV feasibility
  study.
- Prompt FFN quality MMA, shared-Y, stream-K, and I/J ideas are exhausted or
  already installed; rejected OPT-024, OPT-027, OPT-028, OPT-029, OPT-030, and
  OPT-037 variants are not reopened. OPT-031 remains pending and is not superseded.
- Evidence: `evidence/optimization/speedup-loop-post037/`
- Recommended explicit next task: OPT-038. Unchanged row order leaves OPT-031 as
  the next eligible pending task for automatic ledger execution.
- OPT-016 remains the blocked exact-2K parity owner; `plan.md` is unchanged.

### 2026-09-10T07:06:45Z — OPT-038 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-038.md`](tasks/OPT-038.md).
  Coupled IDs: none. Unresolved decisions: none. Plan impact: none.
- Measurement-only refresh. Reuse frozen P/D128/D2048 oracle executables.
  New attribution diagnostics store independent raw host wall, GPU-event
  sum, graph-host interval, and adjusted reconstruction as separate fields
  without modifying `finish_decode_attribution`.
- Accepted keep denominators remain
  [`fixtures/opt034_packed_mmv.json`](fixtures/opt034_packed_mmv.json):
  Quartz P **1869.84412**, D128 **25.3816128**, D2048 **20.169548**,
  D128 all-token p95 **39.9736366** / run-mean p95 **39.4155655**,
  D2048 all-token p95 **50.2872772** / run-mean p95 **49.590683**.
  This increment does not publish a successor oracle.
- Next-task order is a host-recomputed function of live P-versus-D2048
  gaps and live exclusive milliseconds over the pending transferable
  ideas after this map. Mixer/GDN prompt graphs stay pending under the
  existing mixer/GDN graph task and are not inserted into that array.
- Marked OPT-038 `in_progress`. Non-final audit row inserted. No commit.
  `plan.md` is unchanged.

### 2026-09-10T08:18:15Z — OPT-038 delivered (measurement; no speedup)

- Independent verification attempt 1 passed. This increment installs no
  production kernel and claims no performance improvement
  (`claims_performance_improvement` false; speedup N/A). Exclusive RTX
  5090 sitting `measurement_utc` 2026-09-10T07:55:02Z wrote
  [`fixtures/opt038_post_ladder_gap.json`](fixtures/opt038_post_ladder_gap.json)
  with fresh frozen-protocol P, D128, and D2048 measurements, exclusive
  subsystem breakdowns, independent raw host-wall accounting, matched
  pinned llama.cpp public-API measurements, matched-component experiment
  specifications, and a host-recomputed `next_task_order`. Accepted keep
  denominators remain OPT-034: P **1869.84412**, D128 **25.3816128**,
  D2048 **20.169548** tok/s. Live sitting measurements: P Quartz
  **1872.63806** (llama.cpp `avg_ts` **3241.632261**); D128 Quartz
  **25.3562603** vs llama.cpp **68.8080723**; D2048 Quartz
  **20.1668205** vs llama.cpp **67.0727771**. `p_gap` 1.731;
  `d2048_gap` 3.326; `decode_deficit_larger` true. `next_task_order`
  `["OPT-042", "OPT-041", "OPT-040", "OPT-039"]`. Native test does not
  require Quartz ≥ llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py
  tests/test_opt038_post_ladder_gap.py`.
- Acceptance evidence: [`tasks/OPT-038.md`](tasks/OPT-038.md);
  [`pins/opt038_post_ladder_gap_contract.json`](pins/opt038_post_ladder_gap_contract.json);
  [`fixtures/opt038_post_ladder_gap.json`](fixtures/opt038_post_ladder_gap.json);
  [`cuda/opt038_prefill_attribution_test.cu`](cuda/opt038_prefill_attribution_test.cu);
  [`cuda/opt038_decode_attribution_test.cu`](cuda/opt038_decode_attribution_test.cu);
  [`tests/test_opt038_post_ladder_gap.py`](tests/test_opt038_post_ladder_gap.py);
  [`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](evidence/optimization/opt038-post-ladder-gap/REPORT.md);
  [`evidence/optimization/opt038-post-ladder-gap/COMPONENT-PROTOCOL.md`](evidence/optimization/opt038-post-ladder-gap/COMPONENT-PROTOCOL.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/43-cuda-attention-decode.md`](docs/43-cuda-attention-decode.md);
  [`docs/51-runtime-timing-and-nvtx.md`](docs/51-runtime-timing-and-nvtx.md);
  [`docs/61-benchmark-harness.md`](docs/61-benchmark-harness.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md).
  Proof is post-ladder P/D128/D2048 measurements, exclusive attribution,
  independent raw-wall fields, matched llama.cpp measurements,
  matched-component specs, and a recorded next-task order, not a throughput
  keep and not the 2K llama.cpp parity gate.
- Marked OPT-038 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-031**.

### 2026-09-10T08:22:55Z — OPT-039 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-039.md`](tasks/OPT-039.md).
  Coupled IDs: none. Unresolved decisions: none. Plan impact: none.
- Chosen sink is live OPT-038 D2048 decode `attention_core`
  **13.2069445 ms** (`fixtures/opt038_post_ladder_gap.json`). Keep
  denominators remain OPT-034: Quartz P **1869.84412**, D128
  **25.3816128**, D2048 **20.169548**, D128 all-token p95 **39.9736366**
  / run-mean p95 **39.4155655**, D2048 all-token p95 **50.2872772** /
  run-mean p95 **49.590683**.
- Two-way A/B `cta_group` versus `warp_query` against accepted 16/16
  partitions. Warp path is production-shape-only; merge, partition
  pins, and scheduler call site stay unchanged. D2048 must win
  component time to install; else reject.
- Marked OPT-039 `in_progress`. Non-final audit row inserted. No commit.
  `plan.md` is unchanged.

### 2026-09-10T09:14:15Z — OPT-039 delivered (KEEP)

- Independent verification attempt 1 passed. Production decode attention
  installs warp-owned one-token query-head vector attention
  (`kSelectedDecodeAttentionVec = "warp_query"`) at fixed 16/16 KV
  partitions after two-way A/B wins at positions 128 and 2048 with
  unchanged FP32 ascending-part merge. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt039_decode_warp.json`](fixtures/opt039_decode_warp.json)
  (`measurement_utc` 2026-09-10T08:57:52Z; D2048 Quartz **20.169548** →
  **25.3357754** tok/s; D128 Quartz **25.3816128** → **26.1887932**; P
  Quartz **1869.84412** → **1868.51721**; D128 all-token p95 39.9736366 →
  38.3246689 and run-mean p95 39.4155655 → 38.2041283; D2048 all-token
  p95 50.2872772 → 39.5857964 and run-mean p95 49.590683 → 39.4975739;
  D2048 component **0.802414954** → **0.0774026662** ms; A/B winners
  `warp_query`/`warp_query`; `reverted` false; `status` measured).
  Cross-workload guard held. Native test does not require Quartz ≥
  llama.cpp. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py
  tests/test_opt039_decode_warp_attention.py`.
- Acceptance evidence: [`tasks/OPT-039.md`](tasks/OPT-039.md);
  [`pins/opt039_decode_warp_contract.json`](pins/opt039_decode_warp_contract.json);
  [`fixtures/opt039_decode_warp.json`](fixtures/opt039_decode_warp.json);
  [`cuda/opt039_decode_warp_ab_test.cu`](cuda/opt039_decode_warp_ab_test.cu);
  [`cuda/attention_decode.cu`](cuda/attention_decode.cu);
  [`evidence/optimization/opt039-decode-warp/REPORT.md`](evidence/optimization/opt039-decode-warp/REPORT.md);
  [`docs/43-cuda-attention-decode.md`](docs/43-cuda-attention-decode.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md).
  Proof is warp-owned decode attention under frozen envelopes versus
  then-current accepted OPT-034 P/D128/D2048 denominators, not the 2K
  llama.cpp parity gate.
- Marked OPT-039 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-031**.

### 2026-09-10T10:09:04Z — OPT-040 delivered (KEEP)

- Independent verification attempt 1 passed. Production prompt GDN installs
  hoisted per-token/key-head Q/K inverse normalization
  (`kSelectedGdnInversePath = "shared"`) in existing `prompt_projected_bf16_`
  overlay scratch after two-way A/B wins at 4096 with byte-equal quality
  versus repeated warp-column L2. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt040_gdn_shared_inverse.json`](fixtures/opt040_gdn_shared_inverse.json)
  (`measurement_utc` 2026-09-10T09:47:19Z; P Quartz **1869.84412** →
  **2060.62183** tok/s; D128 Quartz **26.1887932** → **26.1689129**; D2048
  Quartz **25.3357754** → **25.2998886**; D128 all-token p95 38.3246689 →
  38.350193 and run-mean p95 38.2041283 → 38.2333679; D2048 all-token p95
  39.5857964 → 39.6363754 and run-mean p95 39.4975739 → 39.55233; 4096
  complete-GDN component **11.9163837** → **7.66518307** ms; A/B winner
  `shared`; `reverted` false; `status` measured). Cross-workload guard held.
  Decode sequential GDN and OPT-029 fuse `off` unchanged. Coupled IDs: none.
  Delivery re-check:
  `uv run pytest -q tests/test_documentation.py
  tests/test_opt040_gdn_shared_inverse.py`.
- Acceptance evidence: [`tasks/OPT-040.md`](tasks/OPT-040.md);
  [`pins/opt040_gdn_shared_inverse_contract.json`](pins/opt040_gdn_shared_inverse_contract.json);
  [`fixtures/opt040_gdn_shared_inverse.json`](fixtures/opt040_gdn_shared_inverse.json);
  [`cuda/opt040_gdn_shared_inverse_ab_test.cu`](cuda/opt040_gdn_shared_inverse_ab_test.cu);
  [`cuda/gdn_fused_quality.cuh`](cuda/gdn_fused_quality.cuh);
  [`cuda/gdn_step.cu`](cuda/gdn_step.cu);
  [`cuda/gdn_step.h`](cuda/gdn_step.h);
  [`evidence/optimization/opt040-gdn-shared-inverse/REPORT.md`](evidence/optimization/opt040-gdn-shared-inverse/REPORT.md);
  [`docs/42-cuda-gdn-chunks.md`](docs/42-cuda-gdn-chunks.md);
  [`docs/41-cuda-gdn-step.md`](docs/41-cuda-gdn-step.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md).
  Proof is hoisted prompt GDN inverse under frozen envelopes versus
  then-current accepted OPT-034 P and OPT-039 D128/D2048 denominators, not
  the 2K llama.cpp parity gate.
- Marked OPT-040 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-031**.

### 2026-09-10T10:16:34Z — OPT-041 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-041.md`](tasks/OPT-041.md).
  Coupled IDs: none. Unresolved decisions: none. Plan impact: none.
- Chosen sink is live OPT-038 4K prefill `attention_core`
  **579.519836 ms** (`fixtures/opt038_post_ladder_gap.json`). Keep
  denominators are the OPT-040 keep sitting: Quartz P **2060.62183**,
  D128 **26.1689129**, D2048 **25.2998886**, D128 all-token p95
  **38.350193** / run-mean p95 **38.2333679**, D2048 all-token p95
  **39.6363754** / run-mean p95 **39.55233**.
- Two-way A/B `cparts` versus `warp_microtile` on production stream-K
  (Ncols1=16, KV-tile-32, dual-F16 Q, register VKQ, P×V MMA, grid.z=2).
  Warp path preserves four virtual-warp K-partials and the current FP32
  fold; the `cparts` shared slab stays. P must win component time and
  strictly beat **2060.62183** to install; else reject.
- Marked OPT-041 `in_progress`. Non-final audit row inserted. No commit.
  `plan.md` is unchanged.

### 2026-09-10T11:35:57Z — OPT-041 delivered (KEEP)

- Independent verification attempt 1 passed. Production prompt attention
  installs warp-owned 16×8 QK microtiles (`kSelectedQKPath =
  "warp_microtile"`) on stream-K register-VKQ P×V MMA after two-way A/B
  wins at 4096 with byte-equal quality versus shared `cparts` reduction.
  Exclusive RTX 5090 sitting wrote
  [`fixtures/opt041_fattn_warp_qk.json`](fixtures/opt041_fattn_warp_qk.json)
  (`measurement_utc` 2026-09-10T11:10:31Z; P Quartz **2060.62183** →
  **2076.98315** tok/s; D128 Quartz **26.1689129** → **26.1599541**; D2048
  Quartz **25.2998886** → **25.2924843**; D128 all-token p95 38.350193 →
  38.3730087 and run-mean p95 38.2333679 → 38.2578201; D2048 all-token
  p95 39.6363754 → 39.6526222 and run-mean p95 39.55233 → 39.5695038;
  4096 complete-attention component **35.5102272** → **34.591423** ms;
  A/B winner `warp_microtile`; `reverted` false; `status` measured).
  Cross-workload guard held. Decode `warp_query`, OPT-040 shared GDN
  inverse, and P×V MMA unchanged. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_documentation.py
  tests/test_opt041_fattn_warp_qk.py`.
- Acceptance evidence: [`tasks/OPT-041.md`](tasks/OPT-041.md);
  [`pins/opt041_fattn_warp_qk_contract.json`](pins/opt041_fattn_warp_qk_contract.json);
  [`fixtures/opt041_fattn_warp_qk.json`](fixtures/opt041_fattn_warp_qk.json);
  [`cuda/opt041_fattn_warp_qk_ab_test.cu`](cuda/opt041_fattn_warp_qk_ab_test.cu);
  [`cuda/fattn_mma_f16.cuh`](cuda/fattn_mma_f16.cuh);
  [`cuda/attention_decode.cu`](cuda/attention_decode.cu);
  [`cuda/attention_decode.h`](cuda/attention_decode.h);
  [`evidence/optimization/opt041-fattn-warp-qk/REPORT.md`](evidence/optimization/opt041-fattn-warp-qk/REPORT.md);
  [`docs/44-cuda-attention-prefill.md`](docs/44-cuda-attention-prefill.md);
  [`docs/62-cuda-full-prefill.md`](docs/62-cuda-full-prefill.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md).
  Proof is warp-owned prompt QK microtiles under frozen envelopes versus
  then-current accepted OPT-040 P/D128/D2048 denominators, not the 2K
  llama.cpp parity gate.
- Marked OPT-041 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-042**.

### 2026-09-10T11:44:18Z — OPT-042 planning admitted

- Planning produced decision-complete dossier [`tasks/OPT-042.md`](tasks/OPT-042.md).
  Coupled IDs: none. Unresolved decisions: none. Plan impact: none.
- Diagnostic-only integer Q4_K decode-MMV feasibility study. Chosen sink
  is live OPT-038 D2048 decode `ffn_mmv` **20.2506523** ms. Candidate
  keeps Quartz FP32-scale `Q8Block` staging and pairs packed FP32 MMV
  against subgroup `__dp4a` dots. Complete (stage+MMV) weighted means on
  synthetic and real FFN inputs decide `eligible_and_faster` /
  `numeric_reject` / `performance_reject` under frozen decode-MMV
  envelope max abs `3e-4` / RMS `2e-4`. Production dispatch stays packed.
  This increment does not publish a successor oracle.
- Accepted keep denominators remain
  [`fixtures/opt041_fattn_warp_qk.json`](fixtures/opt041_fattn_warp_qk.json):
  Quartz P **2076.98315**, D128 **26.1599541**, D2048 **25.2924843**,
  D128 all-token p95 **38.3730087** / run-mean p95 **38.2578201**,
  D2048 all-token p95 **39.6526222** / run-mean p95 **39.5695038**.
- Marked OPT-042 `in_progress`. Non-final audit row inserted. No commit.
  `plan.md` is unchanged.

### 2026-09-10T12:09:52Z — OPT-042 delivered (numeric_reject)

- Independent verification attempt 1 passed. Diagnostic-only integer Q4_K
  decode-MMV admissibility study pairs packed FP32 MMV against subgroup
  `__dp4a` dots on unchanged FP32-scale `Q8Block` staging. Exclusive
  RTX 5090 sitting wrote
  [`fixtures/opt042_mmv_integer_study.json`](fixtures/opt042_mmv_integer_study.json)
  (`measurement_utc` 2026-09-10T12:04:18Z; complete weighted means
  synthetic packed **0.0666723549** ms vs integer **0.0607566237** ms;
  real packed **0.0675881952** ms vs integer **0.061906416** ms;
  `admissibility` `numeric_reject`; `promote_to_production_ab` false;
  `selected_mmv_load_path` `packed`; `claims_performance_improvement`
  false). Synthetic production shapes `q4k_gate_up` / `q4k_down` miss
  frozen CUD-001 max abs `3e-4` / RMS `2e-4` for **both** candidates, so
  the frozen rule rejects promotion even though integer complete means are
  lower. Production `kSelectedMmvLoadPath` stays `packed`; OPT-041 keep
  denominators unchanged. Coupled IDs: none. Delivery re-check:
  `uv run pytest -q tests/test_opt042_mmv_integer_study.py
  tests/test_documentation.py`.
- Acceptance evidence: [`tasks/OPT-042.md`](tasks/OPT-042.md);
  [`pins/opt042_mmv_integer_study_contract.json`](pins/opt042_mmv_integer_study_contract.json);
  [`fixtures/opt042_mmv_integer_study.json`](fixtures/opt042_mmv_integer_study.json);
  [`cuda/opt042_mmv_integer_study.cu`](cuda/opt042_mmv_integer_study.cu);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`tests/test_opt042_mmv_integer_study.py`](tests/test_opt042_mmv_integer_study.py);
  [`evidence/optimization/opt042-mmv-integer-study/REPORT.md`](evidence/optimization/opt042-mmv-integer-study/REPORT.md);
  [`docs/39-cuda-quant-mmv.md`](docs/39-cuda-quant-mmv.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md).
  Proof is paired packed-versus-integer diagnostic timings and CUD-001
  eligibility under the fixed decode-MMV envelope, not a throughput keep,
  not 2K parity, not a successor oracle. Tok/s delta **0** (baseline
  unchanged).
- Marked OPT-042 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-031**.

### 2026-09-10T12:33:01Z — Post-042 recovery design admitted

- User requested source comparison against sibling llama.cpp/ds4 and task
  design only, subsequently authorizing documented accuracy compromises
  comparable to those engines. No implementation, numeric-pin change, build
  change, new GPU timing, commit, or push occurred in this design pass.
- Added [source analysis and common validation protocol](tasks/PERFORMANCE-RECOVERY-2026-09-10.md)
  and pending dossiers OPT-043–OPT-056. Findings include strict FP32 decode
  dots, serial normalization, redundant attention preparation, dual-F16 work,
  repeated GDN preprocessing, and unmeasured physical-batch/dispatch differences.
  Existing measured evidence is distinguished from source findings and proposals.
- OPT-044 defines a future separately documented production-accuracy contract
  while retaining strict references. OPT-054 compares physical microbatch sizes
  without changing atomic 4K publication. OPT-055 subsumes old pending OPT-031,
  now `superseded`. OPT-056 requires actual speed and quality outcomes; it cannot
  pass merely because all candidate experiments finished.
- Current next recovery task is **OPT-043**. OPT-016 remains blocked under its
  unchanged original 2K gate. No claimed speedup from this documentation change.

### 2026-09-10T14:24:30Z — OPT-043 delivered (component gap)

- Independent verification attempt 1 passed. Measurement-only post-042
  component-gap diagnostic retains fresh unperturbed P/D128/D2048 controls,
  exclusive leaf intervals, activation captures, matched llama.cpp complete
  components, and dispatch evidence per major family. Exclusive RTX 5090
  sitting wrote
  [`fixtures/opt043_component_gap.json`](fixtures/opt043_component_gap.json)
  (`measurement_utc` 2026-09-10T14:02:32Z; P Quartz **2076.24** vs llama
  **3195.83** tok/s; D128 **26.15** vs **68.72** tok/s; D2048 **25.30** vs
  **67.05** tok/s; `claims_performance_improvement` false). Nsight Systems
  unavailable; Nsight Compute counters not collected; no bandwidth- or
  compute-bound claim. Production dispatch unchanged; accepted keep
  denominators remain historical OPT-041 copies. Successor decisions for
  OPT-044–056 recorded with recoverable bounds. Coupled IDs: none. Delivery
  re-check: `uv run pytest -q tests/test_opt043_component_gap.py
  tests/test_opt038_post_ladder_gap.py tests/test_documentation.py`.
- Acceptance evidence: [`tasks/OPT-043.md`](tasks/OPT-043.md);
  [`pins/opt043_component_gap_contract.json`](pins/opt043_component_gap_contract.json);
  [`fixtures/opt043_component_gap.json`](fixtures/opt043_component_gap.json);
  [`cuda/opt043_prefill_attribution_test.cu`](cuda/opt043_prefill_attribution_test.cu);
  [`cuda/opt043_decode_attribution_test.cu`](cuda/opt043_decode_attribution_test.cu);
  [`cuda/opt043_activation_capture_test.cu`](cuda/opt043_activation_capture_test.cu);
  [`tools/llama_authority/component_gap.cpp`](tools/llama_authority/component_gap.cpp);
  [`tests/test_opt043_component_gap.py`](tests/test_opt043_component_gap.py);
  [`evidence/optimization/opt043-component-gap/REPORT.md`](evidence/optimization/opt043-component-gap/REPORT.md);
  [`docs/65-documentation-audit.md`](docs/65-documentation-audit.md).
  Proof is measured component gap and ranked successor decisions, not a
  throughput keep, not 2K parity, not a successor oracle.
- Marked OPT-043 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-044**.

### 2026-09-10T16:26:01Z — OPT-044 delivered (production numerics policy)

- Independent verification attempt 1 passed. Policy task freezes separate
  strict-reference and optimized-production arithmetic roles with independent
  host FP64 / llama Q8_1 family/shape ceilings, a versioned 1.01
  production-optimization quality suite, held-out 1024 span hashes, and gate
  inventory across CUD/GDN/attention/graph/session/trace/oracle/QLT checks.
  Production selector remains `strict`; `optimized_admitted=false`; no
  unvalidated fast kernel; OPT-042 numeric reject and strict decode-MMV
  contracts retained. `claims_performance_improvement: false`. Coupled IDs:
  none. Delivery re-check: `uv run pytest -q tests/test_production_numerics.py
  tests/test_quality.py tests/test_documentation.py`.
- Acceptance evidence: [`tasks/OPT-044.md`](tasks/OPT-044.md);
  [`pins/production_numerics_contract.json`](pins/production_numerics_contract.json);
  [`fixtures/opt044_production_numerics.json`](fixtures/opt044_production_numerics.json);
  [`tests/test_production_numerics.py`](tests/test_production_numerics.py);
  [`tools/production_numerics.py`](tools/production_numerics.py);
  [`tools/freeze_production_numerics.py`](tools/freeze_production_numerics.py);
  [`src/opt044_production_numerics.cpp`](src/opt044_production_numerics.cpp);
  [`cuda/production_numerics.h`](cuda/production_numerics.h);
  [`evidence/optimization/opt044-production-numerics/REPORT.md`](evidence/optimization/opt044-production-numerics/REPORT.md);
  [`docs/04-numerics.md`](docs/04-numerics.md);
  [`docs/06-system-optimization.md`](docs/06-system-optimization.md);
  [`docs/65-documentation-audit.md`](docs/65-documentation-audit.md).
  Proof is frozen production-numerics policy and quality budgets, not a
  throughput keep, not 2K parity, not a successor oracle. Tok/s delta **0**
  (baseline unchanged).
- Marked OPT-044 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. User-authorized `plan.md` note added.
  OPT-016 stays `blocked`. Next eligible pending by ledger row order:
  **OPT-045**.

### 2026-09-10T17:36:03Z — OPT-045 delivered (KEEP)

- Independent verification attempt 1 passed. Production RMSNorm installs
  cooperative fmaf parallel normalization (`kSelectedRmsNormPath =
  "parallel_fma"`, residual threads **256**, GDN threads **32**) after
  five-candidate A/B wins prompt-norm and decode RMS component time under
  OPT-044 production-numerics budgets with retained serial references.
  Exclusive RTX 5090 sitting wrote
  [`fixtures/opt045_parallel_norm.json`](fixtures/opt045_parallel_norm.json)
  (`measurement_utc` 2026-09-10T16:47:28Z; P Quartz **2076.98315** →
  **2130.41089** tok/s; D128 **26.1599541** → **28.5522804**; D2048
  **25.2924843** → **27.5438766**; D128 all-token p95 38.3730087 →
  35.1905479 and run-mean p95 38.2578201 → 35.0761757; D2048 all-token
  p95 39.6526222 → 36.4285774 and run-mean p95 39.5695038 → 36.3359795;
  prompt-norm **0.341054976** → **0.197026104** ms; decode RMS
  **0.0578549355** → **0.00944320019** ms; A/B winner `parallel_fma_t256`;
  `reverted` false; `status` measured). Cross-workload guard held. MMV,
  attention QK, and OPT-040 shared GDN inverse unchanged. Coupled IDs: none.
  Delivery re-check: `uv run pytest -q tests/test_opt045_parallel_norm.py
  -k "not exclusive" tests/test_documentation.py`.
- Acceptance evidence: [`tasks/OPT-045.md`](tasks/OPT-045.md);
  [`pins/opt045_parallel_norm_contract.json`](pins/opt045_parallel_norm_contract.json);
  [`fixtures/opt045_parallel_norm.json`](fixtures/opt045_parallel_norm.json);
  [`cuda/opt045_parallel_norm_ab_test.cu`](cuda/opt045_parallel_norm_ab_test.cu);
  [`cuda/rms_norm.cuh`](cuda/rms_norm.cuh);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/scheduler_primitives.cu`](cuda/scheduler_primitives.cu);
  [`tests/test_opt045_parallel_norm.py`](tests/test_opt045_parallel_norm.py);
  [`evidence/optimization/opt045-parallel-norm/REPORT.md`](evidence/optimization/opt045-parallel-norm/REPORT.md);
  [`docs/65-documentation-audit.md`](docs/65-documentation-audit.md).
  Proof is cooperative parallel RMSNorm under frozen OPT-044 numerics versus
  then-current accepted OPT-041 P/D128/D2048 denominators, not the 2K
  llama.cpp parity gate.
- Marked OPT-045 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-046**.
  Tok/s delta P **2076.98** → **2130.41** (+53.4, 1.026×); D128 **26.16**
  → **28.55**; D2048 **25.29** → **27.54**.

### 2026-09-10T19:01:30Z — OPT-046 delivered (REJECT)

- Independent verification attempt 1 passed. Cooperative Q4_K integer-dot
  decode MMV (`q4k_coop_mmv`, DP4A packed products, K distributed across
  warps per row, Q8Block control and llama Q8_1 staging variants) was
  A/B'd against the retained packed FP32 path. Exclusive RTX 5090 sitting
  wrote
  [`fixtures/opt046_q4_decode.json`](fixtures/opt046_q4_decode.json)
  (`measurement_utc` 2026-09-10T18:02:50Z; A/B winner `integer_q8_w4`
  weighted complete **0.0191882669** ms vs packed **0.067196091** ms,
  ~3.50×; integer sitting P Quartz **2130.48462**, D128 **40.7888718**,
  D2048 **38.3443794** tok/s; `status` rejected; `reverted` true;
  `keep_sitting_skipped` false; `selected_q4_decode_path` packed;
  warps per row **4**). **Reject:** CUD-001 `q4_k_17x256` max_abs
  **0.000305175781** exceeds frozen **3e-4** envelope (packed
  **0.000244140625** passes); full-scheduler quality fails under the
  integer production pin. Production `kSelectedQ4DecodePath` stays
  `packed`. OPT-045 keep denominators unchanged. Coupled IDs: none.
  Delivery re-check: `uv run pytest -q tests/test_opt046_q4_decode.py
  tests/test_documentation.py -k "not sitting and not exclusive"`.
- Acceptance evidence: [`tasks/OPT-046.md`](tasks/OPT-046.md);
  [`pins/opt046_q4_decode_contract.json`](pins/opt046_q4_decode_contract.json);
  [`fixtures/opt046_q4_decode.json`](fixtures/opt046_q4_decode.json);
  [`cuda/q4k_decode_dots.cu`](cuda/q4k_decode_dots.cu);
  [`cuda/q4k_decode_dots.cuh`](cuda/q4k_decode_dots.cuh);
  [`cuda/q4k_decode_path.cuh`](cuda/q4k_decode_path.cuh);
  [`cuda/opt046_q4_decode_ab_test.cu`](cuda/opt046_q4_decode_ab_test.cu);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`tests/test_opt046_q4_decode.py`](tests/test_opt046_q4_decode.py);
  [`evidence/optimization/opt046-q4-decode/REPORT.md`](evidence/optimization/opt046-q4-decode/REPORT.md);
  [`evidence/optimization/opt046-q4-decode/REJECTION.md`](evidence/optimization/opt046-q4-decode/REJECTION.md);
  [`docs/65-documentation-audit.md`](docs/65-documentation-audit.md).
  Proof is cooperative integer-dot A/B with frozen OPT-044 numerics and
  OPT-045 cross-workload guards, not the 2K llama.cpp parity gate.
- Marked OPT-046 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-047**.
  Tok/s delta **0** (baseline unchanged: P **2130.41**, D128 **28.55**,
  D2048 **27.54** tok/s).

### 2026-09-10T19:42:58Z — OPT-047 delivered (KEEP)

- Independent verification attempt 1 passed. Cooperative DP4A Q8_0×Q8_1
  mixer decode (`q8_coop_mmv`, shared Q8_1 staging, K distributed across
  warps per row) was A/B'd against the retained direct_bf16 reference.
  Exclusive RTX 5090 sitting wrote
  [`fixtures/opt047_q8_decode.json`](fixtures/opt047_q8_decode.json)
  (`measurement_utc` 2026-09-10T19:22:46Z; A/B winner `dp4a_w4`
  weighted complete mixer group **5.44419813** ms vs direct_bf16
  **10.6800642** ms, ~1.96×; sitting P Quartz **2128.54175**, D128
  **33.8896103**, D2048 **32.390007** tok/s; `status` measured;
  `reverted` false; `keep_sitting_skipped` false;
  `selected_q8_decode_path` dp4a_q8_1; warps skinny/medium/wide **4/4/4**).
  **Keep:** OPT-044 Q8_1 activation approximation within documented
  budgets; OPT-045 cross-workload guards pass (P ≥ 95%, D128/D2048 p95
  ≤ 105%). Production `kSelectedQ8DecodePath` is `dp4a_q8_1`.
  OPT-046 packed Q4_K unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-047.md`](tasks/OPT-047.md);
  [`pins/opt047_q8_decode_contract.json`](pins/opt047_q8_decode_contract.json);
  [`fixtures/opt047_q8_decode.json`](fixtures/opt047_q8_decode.json);
  [`cuda/q8_decode_dots.cu`](cuda/q8_decode_dots.cu);
  [`cuda/q8_decode_dots.cuh`](cuda/q8_decode_dots.cuh);
  [`cuda/q8_decode_path.cuh`](cuda/q8_decode_path.cuh);
  [`cuda/opt047_q8_decode_ab_test.cu`](cuda/opt047_q8_decode_ab_test.cu);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`tests/test_opt047_q8_decode.py`](tests/test_opt047_q8_decode.py);
  [`evidence/optimization/opt047-q8-decode/REPORT.md`](evidence/optimization/opt047-q8-decode/REPORT.md);
  [`docs/65-documentation-audit.md`](docs/65-documentation-audit.md).
  Proof is cooperative DP4A A/B with frozen OPT-044 numerics and OPT-045
  cross-workload guards, not the 2K llama.cpp parity gate.
- Marked OPT-047 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-048**.
  Tok/s delta: P **2130.41** → **2128.54** (−1.87, 0.999×); D128
  **28.55** → **33.89** (+5.34, 1.19×); D2048 **27.54** → **32.39**
  (+4.85, 1.18×).

### 2026-09-10T20:27:00Z — OPT-048 delivered (KEEP)

- Independent verification attempt 1 passed. Cooperative DP4A Q6_K integer
  vocabulary dots (`q6k_coop_mmv`, llama-style `vec_dot_q6k_q8`, 210-byte
  unaligned word loads, K distributed across warps per row) were A/B'd against
  the retained packed FP32 `quant_mmv` reference on the full 248320×5120
  output matrix. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt048_q6_logits.json`](fixtures/opt048_q6_logits.json)
  (`measurement_utc` 2026-09-10T20:05:28Z; A/B winner `integer_q8_1_w2`
  weighted complete vocab **0.642994106** ms vs packed **1.56411517** ms,
  ~2.43×; sitting P Quartz **2129.85938**, D128 **35.9651642**, D2048
  **34.3371162** tok/s; `status` measured; `reverted` false;
  `keep_sitting_skipped` false; `selected_q6_decode_path` integer_q8_1;
  warps per row **2**; integer dispatch gated at `kQ6IntegerMinRows = 248320`).
  **Keep:** OPT-044 production-numerics budgets; OPT-047 keep P/D128 floors
  (P ≥ 95%, D128 ≥ 95%) and strict D2048 improvement versus **32.390007**
  tok/s; decode p95 inside 105% of OPT-045. Production
  `kSelectedQ6DecodePath` is `integer_q8_1`. OPT-046 packed Q4_K and OPT-047
  DP4A Q8 unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-048.md`](tasks/OPT-048.md);
  [`pins/opt048_q6_logits_contract.json`](pins/opt048_q6_logits_contract.json);
  [`fixtures/opt048_q6_logits.json`](fixtures/opt048_q6_logits.json);
  [`cuda/q6k_decode_dots.cu`](cuda/q6k_decode_dots.cu);
  [`cuda/q6k_decode_dots.cuh`](cuda/q6k_decode_dots.cuh);
  [`cuda/q6k_decode_path.cuh`](cuda/q6k_decode_path.cuh);
  [`cuda/opt048_q6_logits_ab_test.cu`](cuda/opt048_q6_logits_ab_test.cu);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`tests/test_opt048_q6_logits.py`](tests/test_opt048_q6_logits.py);
  [`evidence/optimization/opt048-q6-logits/REPORT.md`](evidence/optimization/opt048-q6-logits/REPORT.md).
  Proof is cooperative DP4A A/B with frozen OPT-044 numerics and OPT-047 keep
  denominators, not the 2K llama.cpp parity gate.
- Marked OPT-048 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-049**.
  Tok/s delta: P **2128.54** → **2129.86** (+1.32, 1.001×); D128
  **33.89** → **35.97** (+2.08, 1.06×); D2048 **32.39** → **34.34**
  (+1.95, 1.06×).

### 2026-09-10T21:05:17Z — OPT-049 delivered (KEEP)

- Independent verification attempt 1 passed. Shared Q8 staging and a
  two-pointer paired Q4_K gate/up/SwiGLU kernel with admitted BF16 rounding
  were A/B'd against the retained separate-leg packed Q4_K control on real
  layer activations. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt049_decode_ffn_fusion.json`](fixtures/opt049_decode_ffn_fusion.json)
  (`measurement_utc` 2026-09-10T20:52:08Z; A/B winner `paired_staged`
  weighted complete FFN **0.239974757** ms vs separate **0.26093938** ms,
  ~1.09×; sitting P Quartz **2130.79614**, D128 **37.2543182**, D2048
  **35.4072151** tok/s; `status` measured; `reverted` false;
  `keep_sitting_skipped` false; `selected_ffn_decode_path` paired_staged).
  **Keep:** OPT-044 production-numerics budgets; OPT-048 keep P/D128 floors
  (P ≥ 95%, D128 ≥ 95%) and strict D2048 improvement versus **34.3371162**
  tok/s; decode p95 inside 105% of OPT-045. Production
  `kSelectedFfnDecodePath` is `paired_staged`. OPT-046 packed Q4_K, OPT-047
  DP4A Q8, and OPT-048 integer Q6 unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-049.md`](tasks/OPT-049.md);
  [`pins/opt049_decode_ffn_fusion_contract.json`](pins/opt049_decode_ffn_fusion_contract.json);
  [`fixtures/opt049_decode_ffn_fusion.json`](fixtures/opt049_decode_ffn_fusion.json);
  [`cuda/ffn_decode_path.cuh`](cuda/ffn_decode_path.cuh);
  [`cuda/opt049_decode_ffn_fusion_ab_test.cu`](cuda/opt049_decode_ffn_fusion_ab_test.cu);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`tests/test_opt049_decode_ffn_fusion.py`](tests/test_opt049_decode_ffn_fusion.py);
  [`evidence/optimization/opt049-decode-ffn-fusion/REPORT.md`](evidence/optimization/opt049-decode-ffn-fusion/REPORT.md).
  Proof is shared-staging and paired gate/up A/B with frozen OPT-044 numerics
  and OPT-048 keep denominators, not the 2K llama.cpp parity gate.
- Marked OPT-049 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-050**.
  Tok/s delta: P **2129.86** → **2130.80** (+0.94, 1.0004×); D128
  **35.97** → **37.25** (+1.29, 1.036×); D2048 **34.34** → **35.41**
  (+1.07, 1.031×).

### 2026-09-10T21:40:39Z — OPT-050 delivered (KEEP)

- Independent verification attempt 1 passed. Upstream prompt query
  normalization and RoPE preparation (`hoisted`) were A/B'd against the
  retained in-kernel per-CTA preparation control on complete
  prepare+attention+combine. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt050_attention_query_prepare.json`](fixtures/opt050_attention_query_prepare.json)
  (`measurement_utc` 2026-09-10T21:27:34Z; A/B winner `hoisted`
  weighted complete attention **29.3651066** ms vs in_kernel
  **34.5931778** ms, ~1.18×; sitting P Quartz **2221.82642**, D128
  **37.4580574**, D2048 **35.7343941** tok/s; `status` measured;
  `reverted` false; `keep_sitting_skipped` false;
  `selected_query_prepare_path` hoisted). **Keep:** OPT-044
  production-numerics budgets; OPT-049 keep P floor (strict P improvement)
  and D128/D2048 ≥ 95% floors; decode p95 inside 105% of OPT-045.
  Production `kSelectedQueryPreparePath` is `hoisted`. Prepared Q aliases
  `prompt_projection_a_` (no extra persistent allocation). Decode
  `warp_query` unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-050.md`](tasks/OPT-050.md);
  [`pins/opt050_attention_query_prepare_contract.json`](pins/opt050_attention_query_prepare_contract.json);
  [`fixtures/opt050_attention_query_prepare.json`](fixtures/opt050_attention_query_prepare.json);
  [`cuda/attention_decode.cu`](cuda/attention_decode.cu);
  [`cuda/fattn_mma_f16.cuh`](cuda/fattn_mma_f16.cuh);
  [`cuda/opt050_attention_query_prepare_ab_test.cu`](cuda/opt050_attention_query_prepare_ab_test.cu);
  [`tests/test_opt050_attention_query_prepare.py`](tests/test_opt050_attention_query_prepare.py);
  [`evidence/optimization/opt050-attention-query-prepare/REPORT.md`](evidence/optimization/opt050-attention-query-prepare/REPORT.md).
  Proof is hoisted upstream Q prepare with frozen OPT-044 numerics and
  OPT-049 keep denominators, not the 2K llama.cpp parity gate.
- Marked OPT-050 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-051**.
  Tok/s delta: P **2130.80** → **2221.83** (+91.03, 1.043×); D128
  **37.25** → **37.46** (+0.20, 1.005×); D2048 **35.41** → **35.73**
  (+0.33, 1.009×).

### 2026-09-10T22:50:00Z — OPT-051 delivered (KEEP)

- Independent verification pass confirmed. Staged A/B of F16 operands,
  register softmax, cp.async KV staging, nbatch 64, and six-head GQA
  against the OPT-050 hoisted dual-F16 stream-K baseline on complete
  prepare+attention+combine. Exclusive RTX 5090 sitting wrote
  [`fixtures/opt051_attention_pipeline.json`](fixtures/opt051_attention_pipeline.json)
  (`measurement_utc` 2026-09-10T22:32:51Z; A/B winner `f16_async`
  weighted complete attention **16.0279236** ms vs off **29.4006214** ms,
  ~1.83×; sitting P Quartz **2489.33008**, D128 **37.6655884**, D2048
  **35.7582932** tok/s; `status` measured; `reverted` false;
  `keep_sitting_skipped` false; `selected_attention_pipeline_path`
  f16_async). **Keep:** OPT-044 production-numerics budgets;
  OPT-050 keep P floor (strict P improvement) and D128/D2048 ≥ 95%
  floors; decode p95 inside 105% of OPT-045. Production
  `kSelectedAttentionPipelinePath` is `f16_async`. Decode `warp_query`
  unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-051.md`](tasks/OPT-051.md);
  [`pins/opt051_attention_pipeline_contract.json`](pins/opt051_attention_pipeline_contract.json);
  [`fixtures/opt051_attention_pipeline.json`](fixtures/opt051_attention_pipeline.json);
  [`cuda/fattn_mma_f16_pipeline.cuh`](cuda/fattn_mma_f16_pipeline.cuh);
  [`cuda/fattn_mma_f16.cuh`](cuda/fattn_mma_f16.cuh);
  [`cuda/opt051_attention_pipeline_ab_test.cu`](cuda/opt051_attention_pipeline_ab_test.cu);
  [`tests/test_opt051_attention_pipeline.py`](tests/test_opt051_attention_pipeline.py);
  [`evidence/optimization/opt051-attention-pipeline/REPORT.md`](evidence/optimization/opt051-attention-pipeline/REPORT.md).
  Proof is pipelined prompt attention with frozen OPT-044 numerics and
  OPT-050 keep denominators, not the 2K llama.cpp parity gate.
- Marked OPT-051 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-052**.
  Tok/s delta: P **2221.83** → **2489.33** (+267.50, 1.120×); D128
  **37.46** → **37.67** (+0.21, 1.006×); D2048 **35.73** → **35.76**
  (+0.03, 1.001×).

### 2026-09-10T23:30:00Z — OPT-052 delivered (KEEP)

- Independent verification pass confirmed. Preprocessing sibling hoists
  L2-scaled Q/K and decay into `prompt_projected_bf16_`, then runs the
  existing register recurrence; conversion-inclusive column-major state
  tile won the complete conv+preprocessing+recurrence+gated A/B. Exclusive
  RTX 5090 sitting wrote
  [`fixtures/opt052_gdn_arithmetic.json`](fixtures/opt052_gdn_arithmetic.json)
  (`measurement_utc` 2026-09-10T23:07:40Z; A/B winner `transpose`
  weighted complete GDN **23.217959473333334** ms vs shared
  **26.516851046666666** ms, ~1.14×; sitting P Quartz **2692.64575**, D128
  **37.5680695**, D2048 **35.6793633** tok/s; `status` measured;
  `reverted` false; `keep_sitting_skipped` false;
  `selected_gdn_preproc_path` transpose). **Keep:** OPT-044
  production-numerics budgets; OPT-051 keep P floor (strict P improvement)
  and D128/D2048 ≥ 95% floors; decode p95 inside 105% of OPT-051.
  Production `kSelectedGdnPreprocPath` is `transpose`. Inverse `shared`,
  fuse `off`, decode sequential GDN unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-052.md`](tasks/OPT-052.md);
  [`pins/opt052_gdn_arithmetic_contract.json`](pins/opt052_gdn_arithmetic_contract.json);
  [`fixtures/opt052_gdn_arithmetic.json`](fixtures/opt052_gdn_arithmetic.json);
  [`cuda/gdn_fused_quality.cuh`](cuda/gdn_fused_quality.cuh);
  [`cuda/opt052_gdn_arithmetic_ab_test.cu`](cuda/opt052_gdn_arithmetic_ab_test.cu);
  [`tests/test_opt052_gdn_arithmetic.py`](tests/test_opt052_gdn_arithmetic.py);
  [`evidence/optimization/opt052-gdn-arithmetic/REPORT.md`](evidence/optimization/opt052-gdn-arithmetic/REPORT.md).
  Proof is hoisted GDN arithmetic with frozen OPT-044 numerics and OPT-051
  keep denominators, not the 2K llama.cpp parity gate.
- Marked OPT-052 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-053**.
  Tok/s delta: P **2489.33** → **2692.65** (+203.32, 1.082×); D128
  **37.67** → **37.57** (−0.10, 0.997×); D2048 **35.76** → **35.68**
  (−0.08, 0.998×).

### 2026-09-11T00:05:20Z — OPT-053 delivered (KEEP)

- Independent verification pass confirmed. Explicit FMA scale accumulation and
  two-stage packed-Y `cp.async` in `cuda/quant_mmq_mma.cuh::quality_mma_process_tile`.
  Exclusive RTX 5090 sitting wrote
  [`fixtures/opt053_mmq_pipeline.json`](fixtures/opt053_mmq_pipeline.json)
  (`measurement_utc` 2026-09-10T23:58:33Z; A/B winner `fma_async` complete FFN
  **9.68816757** ms vs off **11.2367001** ms, ~1.16×; sitting P Quartz
  **2895.42773**, D128 **37.5605927**, D2048 **35.7286987** tok/s; `status`
  measured; `reverted` false; `keep_sitting_skipped` false;
  `selected_mmq_pipeline_path` fma_async). **Keep:** OPT-044
  production-numerics budgets; OPT-052 keep P floor (strict P improvement)
  and D128/D2048 ≥ 95% floors; decode p95 inside 105% of OPT-052.
  Production `kSelectedMmqPipelinePath` is `fma_async`. Stream-K `off`, FFN
  tiles I=128/J=128 unchanged. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-053.md`](tasks/OPT-053.md);
  [`pins/opt053_mmq_pipeline_contract.json`](pins/opt053_mmq_pipeline_contract.json);
  [`fixtures/opt053_mmq_pipeline.json`](fixtures/opt053_mmq_pipeline.json);
  [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh);
  [`cuda/opt053_mmq_pipeline_ab_test.cu`](cuda/opt053_mmq_pipeline_ab_test.cu);
  [`tests/test_opt053_mmq_pipeline.py`](tests/test_opt053_mmq_pipeline.py);
  [`evidence/optimization/opt053-mmq-pipeline/REPORT.md`](evidence/optimization/opt053-mmq-pipeline/REPORT.md).
  Proof is FMA scale accumulation and packed-Y cp.async with frozen OPT-044
  numerics and OPT-052 keep denominators, not the 2K llama.cpp parity gate.
- Marked OPT-053 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-054**.
  Tok/s delta: P **2692.65** → **2895.43** (+202.78, 1.075×); D128
  **37.57** → **37.56** (−0.007, 1.000×); D2048 **35.68** → **35.73**
  (+0.049, 1.001×).

### 2026-09-11T01:03:00Z — OPT-054 keep 4096 microbatch

- **Keep:** internal prefill microbatch rows remain **4096** after graphs-off
  A/B (512 2476.38, 1024 2697.29, 2048 2822.63, 4096 2842.41 tok/s) and
  graphs-on 4096 shipping (2840.70 tok/s). No smaller candidate wins complete
  P. Cancellation frontier stays 0 after internal batches 1, 2, and the final
  batch. `keep_sitting_skipped`=true; P/D128/D2048 copied from OPT-053.
  Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-054.md`](tasks/OPT-054.md);
  [`pins/opt054_prefill_microbatch_contract.json`](pins/opt054_prefill_microbatch_contract.json);
  [`fixtures/opt054_prefill_microbatch.json`](fixtures/opt054_prefill_microbatch.json);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/opt054_prefill_microbatch_ab_test.cu`](cuda/opt054_prefill_microbatch_ab_test.cu);
  [`tests/test_opt054_prefill_microbatch.py`](tests/test_opt054_prefill_microbatch.py);
  [`evidence/optimization/opt054-prefill-microbatch/REPORT.md`](evidence/optimization/opt054-prefill-microbatch/REPORT.md).
  Proof is atomic 4096-token transaction with internal microbatch sweep and
  no early commit, not the 2K llama.cpp parity gate.
- Marked OPT-054 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-055**.
  Tok/s delta: P **2895.43** → **2895.43** (0, 1.000×); D128 **37.56** →
  **37.56** (0, 1.000×); D2048 **35.73** → **35.73** (0, 1.000×). Speedup
  **0** (measured no-change).

### 2026-09-11T01:30:00Z — OPT-055 measured no-change

- **No-change:** FFN-only decode/prompt graphs remain the shipping path after
  exclusive A/B of remaining `other_idle` on D128/D2048 and 4096-row prompt with
  poll versus null. `below_noise`=true; winner `ffn_only`; layer-segment
  mixer/core slots stay empty (0 decode-segment + 0 prompt-mixer graphs).
  D128 graphs idle 0.0855464935 ms / wall 25.5044994 ms; D2048 idle
  0.105142593 ms / wall 27.6636486 ms; prompt idle 0 ms / wall 1422.6759 ms.
  Graph/eager equal; poll-8 cancel frontier 0; 384 decode + 448 prompt FFN
  nodes; 16,777,216 graph bytes; `keep_sitting_skipped`=true; P/D copied from
  OPT-054. Coupled IDs: none.
- Acceptance evidence: [`tasks/OPT-055.md`](tasks/OPT-055.md);
  [`pins/opt055_execution_graphs_contract.json`](pins/opt055_execution_graphs_contract.json);
  [`fixtures/opt055_execution_graphs.json`](fixtures/opt055_execution_graphs.json);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/opt055_execution_graphs_ab_test.cu`](cuda/opt055_execution_graphs_ab_test.cu);
  [`tests/test_opt055_execution_graphs.py`](tests/test_opt055_execution_graphs.py);
  [`evidence/optimization/opt055-execution-graphs/REPORT.md`](evidence/optimization/opt055-execution-graphs/REPORT.md).
  Proof is remaining launch-gap measurement with graph/eager equality and
  cancellation before publication, not the 2K llama.cpp parity gate.
- Marked OPT-055 `done`; delivery is limited to the verified task scope
  plus this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016
  stays `blocked`. Next eligible pending by ledger row order: **OPT-056**.
  Tok/s delta: P **2895.43** → **2895.43** (0, 1.000×); D128 **37.56** →
  **37.56** (0, 1.000×); D2048 **35.73** → **35.73** (0, 1.000×). Speedup
  **0** (measured no-change).

### 2026-09-11T02:15:00Z — OPT-056 blocked (performance gate unpassed)

- Same-sitting exclusive RTX 5090 outcome gate measured on combined production
  paths after OPT-045–055. `gate.passed` is false. P Quartz **2808.50** vs llama
  **3263.52** tok/s (remaining **+618** tok/s to the 5% bar); D128 **37.48** vs
  **68.93**; D2048 **35.72** vs **67.34**; decode p95 worse than llama on both
  prefixes; production-optimization greedy tasks failed; OPT-016 2K **3012.70**
  vs **3169.57**. Candidate-task completion is not a pass.
- Acceptance evidence: [`tasks/OPT-056.md`](tasks/OPT-056.md);
  [`pins/opt056_performance_gate_contract.json`](pins/opt056_performance_gate_contract.json);
  [`fixtures/opt056_performance_gate.json`](fixtures/opt056_performance_gate.json);
  [`tests/test_opt056_performance_gate.py`](tests/test_opt056_performance_gate.py);
  [`evidence/optimization/opt056-performance-gate/REPORT.md`](evidence/optimization/opt056-performance-gate/REPORT.md).
  Proof is honest same-sitting P/D128/D2048 versus pinned llama.cpp with quality
  and OPT-016 evidence, not a substitute 2K gate.
- Marked OPT-056 `blocked`; recovery: close remaining P/D throughput and
  decode-p95 gaps versus llama before re-pass. `plan.md` is unchanged. OPT-016
  stays `blocked`. No pending recovery rows remain until new measured-bottleneck
  tasks are admitted.

### 2026-09-11T04:58:09Z — Post-056 recovery batch proposed

- Added OPT-057–069 and detailed implementation guides based on current Quartz,
  pinned llama.cpp and ds4 source plus retained OPT-044–056 evidence. The
  [design](tasks/PERFORMANCE-RECOVERY-2026-09-11.md) distinguishes observations,
  estimates, hypotheses and unmeasured candidates.
- Priority findings: current paired gate/up bypass cooperative Q4 dispatch;
  Q8_1 sum semantics differ from pinned llama; production admission remains a
  strict stub; missing functional/held-out authority and historical scheduler
  nonfinites need independent validation; pipelined MMQ tile/resource selection
  needs a new bounded comparison.
- Reviewed [testing policy](testing-strategy.md): first useful feedback targets
  300 seconds including incremental build, with explicit screening workloads,
  cached sampled references and one batch-level long release sitting. OPT-057
  implements the runner; proposed commands do not exist yet.
- Proposed versioned GPU-calibrated primitive error limits and incremental
  component acceptance with E2E non-regression; keep PPL<=1.01, zero nonfinites,
  functional correctness and original OPT-056/OPT-016 outcome conditions.
- Read-only hardware inventory observed a 400 W power cap; no hardware settings
  or runtime code changed. No new GPU speedup measurement or delivered candidate
  is claimed. First eligible implementation task: OPT-057.

### 2026-09-11T05:26:54Z — OPT-057 bounded iteration runner delivered

- **Infrastructure:** CUDA `-MMD -MP` depfiles and content-compared flag stamps;
  `screen` tier in `cuda/test_tier.h`; `tools/run_optimization_task.py` with
  feedback/acceptance/release modes; `cuda/optimization_engine_probe.cu` short
  engine probe; 300-second aggregate deadline with owned-child teardown. Coupled
  IDs: none. No throughput claim.
- Acceptance evidence: [`tasks/OPT-057.md`](tasks/OPT-057.md);
  [`pins/opt057_iteration_contract.json`](pins/opt057_iteration_contract.json);
  [`fixtures/opt057_iteration_loop.json`](fixtures/opt057_iteration_loop.json);
  [`tools/run_optimization_task.py`](tools/run_optimization_task.py);
  [`cuda/optimization_engine_probe.cu`](cuda/optimization_engine_probe.cu);
  [`tests/test_optimization_loop.py`](tests/test_optimization_loop.py);
  [`evidence/optimization/opt057-iteration-loop/REPORT.md`](evidence/optimization/opt057-iteration-loop/REPORT.md).
  Proof is bounded feedback/acceptance validation with reliable incremental builds,
  not a model speedup or 2K llama.cpp parity gate.
- Marked OPT-057 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-058**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T06:04:25Z — OPT-058 finite scheduler and quality baseline delivered

- **Scheduler:** Q8 decode staging invalidated after graph capture
  (`SchedulerWorkspace::invalidate_q8_decode_staging()`); eager/traced/FFN-graph
  logits finite and byte-identical for tokens `[42, 3649]`. Coupled IDs: none.
  No throughput claim.
- **Quality:** v2 chat-template inputs and held-out llama NLL reference frozen;
  relative PPL/recurrence scores within v2 bounds. Functional prerequisite
  **blocked** on `task_arithmetic` — pinned llama and Quartz both emit A (token
  32) vs expected B on both engines; exact defect in fixture `blocker` and
  REPORT. quality-v2 suite `fail` is expected under the blocked-prerequisite
  acceptance path.
- Acceptance evidence: [`tasks/OPT-058.md`](tasks/OPT-058.md);
  [`pins/opt058_quality_baseline_contract.json`](pins/opt058_quality_baseline_contract.json);
  [`pins/opt058_iteration_contract.json`](pins/opt058_iteration_contract.json);
  [`pins/production_quality_v2_inputs.json`](pins/production_quality_v2_inputs.json);
  [`pins/production_quality_v2_llama_reference.json`](pins/production_quality_v2_llama_reference.json);
  [`fixtures/opt058_quality_baseline.json`](fixtures/opt058_quality_baseline.json);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/opt058_quality_baseline_test.cu`](cuda/opt058_quality_baseline_test.cu);
  [`tests/test_opt058_quality_baseline.py`](tests/test_opt058_quality_baseline.py);
  [`evidence/optimization/opt058-quality-baseline/REPORT.md`](evidence/optimization/opt058-quality-baseline/REPORT.md).
  Proof is finite scheduler baseline, frozen v2 quality references, and honest
  functional blocker documentation — not a model speedup gate.
- Marked OPT-058 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-059**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T06:38:19Z — OPT-059 GPU numerics v2 policy delivered

- **v2 policy:** Frozen budgets in `pins/production_numerics_v2_contract.json`;
  explicit `kProductionAdmission[]` mapping; engine summary **mixed** (Q8/Q6
  optimized, Q4 **packed**/strict). v1 contract unchanged (`selected=strict`).
  Coupled IDs: none. No throughput claim.
- **GPU export:** `qw38-llama-projection-export` synthetic Q4 17×256 probe with
  sync; sidecar
  [`evidence/optimization/opt059-gpu-numerics/llama-gpu-export/`](evidence/optimization/opt059-gpu-numerics/llama-gpu-export/).
  Production-K GGUF rows **unadmitted** (`llama_gpu_error=null`); strict fallback
  per recipe §7.
- **OPT-046:** Independent verdict — CUD-001 3.05175781e-4 > 3e-4 → fail; **not
  installed**; `keep_strict_q4_dispatch=true`. Later Q4 work owns any keep.
- Acceptance evidence: [`tasks/OPT-059.md`](tasks/OPT-059.md);
  [`pins/production_numerics_v2_contract.json`](pins/production_numerics_v2_contract.json);
  [`pins/opt059_admission_manifest.json`](pins/opt059_admission_manifest.json);
  [`pins/opt059_iteration_contract.json`](pins/opt059_iteration_contract.json);
  [`fixtures/opt059_gpu_numerics.json`](fixtures/opt059_gpu_numerics.json);
  [`cuda/opt059_numerics_test.cu`](cuda/opt059_numerics_test.cu);
  [`tools/llama_authority/projection_export.cpp`](tools/llama_authority/projection_export.cpp);
  [`tools/production_numerics_v2.py`](tools/production_numerics_v2.py);
  [`tests/test_opt059_numerics.py`](tests/test_opt059_numerics.py);
  [`evidence/optimization/opt059-gpu-numerics/REPORT.md`](evidence/optimization/opt059-gpu-numerics/REPORT.md).
  Proof is measured v2 policy freeze, GPU probe path, and honest unadmitted
  production-K coverage — not a model speedup gate.
- Marked OPT-059 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-061**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T07:56:29Z — OPT-060 full-engine attribution delivered

- **Instrumentation:** Matched full-engine Quartz and private pinned-llama
  family attribution on P4096 and D2048+16 with exclusive GPU turns, pooled
  `GpuPhaseRecorder` events, private patch/worktree build, and stream-aware
  aggregation. Coupled IDs: none. No throughput claim.
- **Acceptance deadline:** Feedback and top-level runner remain 300 s;
  `modes.acceptance.aggregate_deadline_s` is 7200 in
  `pins/opt060_iteration_contract.json` so three repetitions per workload/mode
  can finish (~409 s measured). Repetition count unchanged.
- **Proof limits:** Eager diagnostic labeled; uninstrumented graph walls
  retained; llama decode event-pool overflow at 8192 detected and labeled
  (`pool_overflow=true`, truncated tables); `nsys` missing; role vocabularies
  unmatched (`ffn_mmv` vs `MUL_MAT`). Not isolated OPT-043 timings.
- Acceptance evidence: [`tasks/OPT-060.md`](tasks/OPT-060.md);
  [`pins/opt060_engine_attribution_contract.json`](pins/opt060_engine_attribution_contract.json);
  [`pins/opt060_iteration_contract.json`](pins/opt060_iteration_contract.json);
  [`fixtures/opt060_engine_attribution.json`](fixtures/opt060_engine_attribution.json);
  [`cuda/engine_attribution.h`](cuda/engine_attribution.h);
  [`cuda/opt060_engine_attribution_test.cu`](cuda/opt060_engine_attribution_test.cu);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`tools/llama_authority/patches/opt060-engine-attribution.patch`](tools/llama_authority/patches/opt060-engine-attribution.patch);
  [`tools/llama_authority/build_opt060_instrumented.sh`](tools/llama_authority/build_opt060_instrumented.sh);
  [`tools/llama_authority/engine_attribution.cpp`](tools/llama_authority/engine_attribution.cpp);
  [`tools/opt060_engine_attribution.py`](tools/opt060_engine_attribution.py);
  [`tests/test_opt060_engine_attribution.py`](tests/test_opt060_engine_attribution.py);
  [`evidence/optimization/opt060-engine-attribution/REPORT.md`](evidence/optimization/opt060-engine-attribution/REPORT.md).
  Proof is full-engine family records, overhead diagnostics, and honest
  truncation/overlap limits — not a model speedup gate.
- Marked OPT-060 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-061**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T08:22:00Z — OPT-061 component replay delivered

- **Replay:** Real-input streaming component replay for decode-ffn,
  decode-mixer projections, and prompt-ffn with hot versus rotating weight
  walks, complete-family CUDA-event times, and reusable capture bundles.
  Coupled IDs: none. No throughput claim.
- **Proof limits:** Hot cache diagnostic only; mixer excludes GDN/attention
  core; `ncu` present without `--set full`; `nsys` absent; streaming useful-byte
  rate is an estimate (`dram_counter_claim=false`). No production pin change.
- Acceptance evidence: [`tasks/OPT-061.md`](tasks/OPT-061.md);
  [`pins/opt061_component_replay_contract.json`](pins/opt061_component_replay_contract.json);
  [`pins/opt061_iteration_contract.json`](pins/opt061_iteration_contract.json);
  [`fixtures/opt061_component_replay.json`](fixtures/opt061_component_replay.json);
  [`cuda/optimization_component_replay.h`](cuda/optimization_component_replay.h);
  [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`tools/opt061_component_replay.py`](tools/opt061_component_replay.py);
  [`tests/test_opt061_component_replay.py`](tests/test_opt061_component_replay.py);
  [`evidence/optimization/opt061-component-replay/REPORT.md`](evidence/optimization/opt061-component-replay/REPORT.md).
  Proof is rotating replay, hardware/resource evidence, and honest
  microbench/full-engine disagreement — not a model speedup gate.
- Marked OPT-061 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-062**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T08:49:10Z — OPT-062 Q4 FFN wiring delivered (packed retained)

- Cooperative integer Q8Block reaches gate, up, and down on decode FFN with
  typed prequant dispatch, eager/graph dispatch records, and complete-FFN
  screen evidence. Half-scale Q8_1 skipped (OPT-059 `v2_admitted=false`).
  Coupled IDs: none. **Packed retained:** component screen winner is
  integer_q8 (~1.26× complete FFN: packed **15.7643099** ms vs integer
  **12.4760742** ms) but production stays `packed` / `paired_staged` because
  OPT-059 `integer_dp4a_q8` is testing-admitted only and this sitting did not
  run combined v2 NLL / functional answers. No tok/s claim
  (`claims_throughput=false`; component ms only).
- Acceptance evidence: [`tasks/OPT-062.md`](tasks/OPT-062.md);
  [`pins/opt062_q4_admission_contract.json`](pins/opt062_q4_admission_contract.json);
  [`pins/opt062_iteration_contract.json`](pins/opt062_iteration_contract.json);
  [`fixtures/opt062_q4_admission.json`](fixtures/opt062_q4_admission.json);
  [`cuda/opt062_q4_admission_test.cu`](cuda/opt062_q4_admission_test.cu);
  [`cuda/q4k_decode_dots.cu`](cuda/q4k_decode_dots.cu);
  [`cuda/q4k_decode_path.cuh`](cuda/q4k_decode_path.cuh);
  [`cuda/ffn_decode_path.cuh`](cuda/ffn_decode_path.cuh);
  [`cuda/quant_mmv.cu`](cuda/quant_mmv.cu);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu);
  [`tests/test_opt062_q4_admission.py`](tests/test_opt062_q4_admission.py);
  [`evidence/optimization/opt062-q4-admission/REPORT.md`](evidence/optimization/opt062-q4-admission/REPORT.md).
  Proof is typed Q8Block wiring on all three FFN legs, graph/eager dispatch
  records, and component complete-FFN screen — not a production install gate.
- Marked OPT-062 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-063**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T09:10:28Z — OPT-063 paired integer fusion delivered (packed retained)

- Cooperative 4-warp `q4k_coop_gate_up_swiglu_prequant_q8` fuses admitted
  integer gate/up dots and SwiGLU with separate GGUF pointers, shared Q8Block
  staging, graph/eager dispatch records, and complete-FFN screen evidence.
  Coupled IDs: none. **Packed retained:** screen winner is paired_integer
  (~1.09× vs unfused integer **12.3049278** ms → **11.2725649** ms; ~1.40× vs
  packed **15.8205442** ms) but production stays `packed` / `paired_staged`
  because OPT-062 did not install integer Q8Block (`v2_admitted=false`) and this
  sitting did not run combined v2 NLL / functional answers. No tok/s claim
  (`claims_throughput=false`; component ms only).
- Acceptance evidence: [`tasks/OPT-063.md`](tasks/OPT-063.md);
  [`pins/opt063_integer_ffn_contract.json`](pins/opt063_integer_ffn_contract.json);
  [`pins/opt063_iteration_contract.json`](pins/opt063_iteration_contract.json);
  [`fixtures/opt063_integer_ffn.json`](fixtures/opt063_integer_ffn.json);
  [`cuda/opt063_integer_ffn_test.cu`](cuda/opt063_integer_ffn_test.cu);
  [`cuda/q4k_decode_dots.cu`](cuda/q4k_decode_dots.cu);
  [`cuda/q4k_decode_dots.cuh`](cuda/q4k_decode_dots.cuh);
  [`cuda/q4k_decode_path.cuh`](cuda/q4k_decode_path.cuh);
  [`cuda/ffn_decode_path.cuh`](cuda/ffn_decode_path.cuh);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu);
  [`tests/test_opt063_integer_ffn.py`](tests/test_opt063_integer_ffn.py);
  [`evidence/optimization/opt063-integer-ffn/REPORT.md`](evidence/optimization/opt063-integer-ffn/REPORT.md).
  Proof is typed paired-integer selector wiring, trace unfused fallback, and
  component complete-FFN screen — not a production install gate.
- Marked OPT-063 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-064**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T09:47:34Z — OPT-064 Q8 row grouping delivered (r2_w2 installed)

- Four-layout bounded study selected **r2_w2** (2 rows/CTA, 2 warps/row) for
  skinny, medium, and wide Q8 decode mixer projections. OPT-047 historical
  warps remain 4/4/4. Coupled IDs: none. **r2_w2 installed for Q8 decode.**
  Rotating weighted complete mixer: control r1_w4 **3.34318942** ms → winner
  r2_w2 **2.67639479** ms (**0.667** ms/token saving). Hot weighted: control
  **2.47074127** ms vs r2_w2 **2.85320511** ms. 40 registers, 0 local bytes,
  occupancy 12. No tok/s claim (`claims_throughput=false`; component ms only).
- Acceptance evidence: [`tasks/OPT-064.md`](tasks/OPT-064.md);
  [`pins/opt064_q8_rows_contract.json`](pins/opt064_q8_rows_contract.json);
  [`pins/opt064_iteration_contract.json`](pins/opt064_iteration_contract.json);
  [`fixtures/opt064_q8_rows.json`](fixtures/opt064_q8_rows.json);
  [`cuda/opt064_q8_rows_test.cu`](cuda/opt064_q8_rows_test.cu);
  [`cuda/q8_decode_dots.cu`](cuda/q8_decode_dots.cu);
  [`cuda/q8_decode_dots.cuh`](cuda/q8_decode_dots.cuh);
  [`cuda/q8_decode_path.cuh`](cuda/q8_decode_path.cuh);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`cuda/optimization_component_replay.cu`](cuda/optimization_component_replay.cu);
  [`tests/test_opt064_q8_rows.py`](tests/test_opt064_q8_rows.py);
  [`evidence/optimization/opt064-q8-rows/REPORT.md`](evidence/optimization/opt064-q8-rows/REPORT.md).
  Proof is typed r2_w2 selector wiring, staging reuse/invalidate checks, and
  component complete-mixer screen — not combined E2E (OPT-069 owns that gate).
- Marked OPT-064 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-065**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T10:36:00Z — OPT-065 MMQ tile retune delivered (128×128 retained)

- Four Q4 tile candidates compared with FMA/async Y on aligned full tiles.
  Coupled IDs: none. **128×128 retained** for gate/up and down; pipeline
  extended to all four (I,J) pairs. Complete FFN control **9.649** ms; selected
  pair i128_j128 **9.629** ms (`keep=false`). Occupancy 1 on all candidates;
  i128_j128 spills 80 local bytes. No tok/s claim (`claims_throughput=false`;
  component ms only).
- Acceptance evidence: [`tasks/OPT-065.md`](tasks/OPT-065.md);
  [`pins/opt065_mmq_tiles_contract.json`](pins/opt065_mmq_tiles_contract.json);
  [`pins/opt065_iteration_contract.json`](pins/opt065_iteration_contract.json);
  [`fixtures/opt065_mmq_tiles.json`](fixtures/opt065_mmq_tiles.json);
  [`cuda/opt065_mmq_tiles_test.cu`](cuda/opt065_mmq_tiles_test.cu);
  [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`tests/test_opt065_mmq_tiles.py`](tests/test_opt065_mmq_tiles.py);
  [`evidence/optimization/opt065-mmq-tiles/REPORT.md`](evidence/optimization/opt065-mmq-tiles/REPORT.md).
  Proof is typed four-tile pipeline wiring, dispatch/resource evidence, and
  component complete-FFN screen — not combined E2E (OPT-069 owns that gate).
- Marked OPT-065 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-066**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T10:57:28Z — OPT-066 packed Q4 X-prefetch delivered (fma_async_x installed)

- Bounded raw-X prefetch on the accepted OPT-065 `i128_j128` tile with FMA/async
  Y retained. Coupled IDs: none. Single prefetched raw slot (18448 B) because a
  two-stage X ring exceeds SM120 opt-in (101376 B). Complete FFN control
  **9.360** ms; candidate **8.615** ms (`keep=true`). Production pin
  `kSelectedMmqAsyncX = true`; `fma_async_x` installed. Occupancy 1 on both;
  control spills 64 local bytes, candidate 0. No tok/s claim
  (`claims_throughput=false`; component ms only).
- Acceptance evidence: [`tasks/OPT-066.md`](tasks/OPT-066.md);
  [`pins/opt066_mmq_x_pipeline_contract.json`](pins/opt066_mmq_x_pipeline_contract.json);
  [`pins/opt066_iteration_contract.json`](pins/opt066_iteration_contract.json);
  [`fixtures/opt066_mmq_x_pipeline.json`](fixtures/opt066_mmq_x_pipeline.json);
  [`cuda/opt066_mmq_x_pipeline_test.cu`](cuda/opt066_mmq_x_pipeline_test.cu);
  [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`tests/test_opt066_mmq_x_pipeline.py`](tests/test_opt066_mmq_x_pipeline.py);
  [`evidence/optimization/opt066-mmq-x-pipeline/REPORT.md`](evidence/optimization/opt066-mmq-x-pipeline/REPORT.md).
  Proof is typed X-prefetch wiring, stage-lifetime/resource evidence, and
  component complete-FFN screen — not combined E2E (OPT-069 owns that gate).
- Marked OPT-066 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-067**. No tok/s
  claim (`claims_throughput=false`).

### 2026-09-11T11:23:26Z — OPT-067 paired prompt gate/up delivered (rejected, off retained)

- Paired I64/J64 gate/up + BF16 SwiGLU epilogue written after OPT-061 traffic
  estimate **35.97 ms/prompt** cleared the 5 ms go threshold. Coupled IDs: none.
  Complete FFN control **8.544** ms vs paired **11.330** ms (`keep=false`).
  Production pin `kSelectedFfnPromptPairPath = "off"`; separate gate/up retained.
  Occupancy 1; paired 247 regs / 0 local / 38144 shared; no spills. Graph/eager
  match. No tok/s claim (`claims_throughput=false`; component ms only).
- Acceptance evidence: [`tasks/OPT-067.md`](tasks/OPT-067.md);
  [`pins/opt067_prompt_pair_contract.json`](pins/opt067_prompt_pair_contract.json);
  [`pins/opt067_iteration_contract.json`](pins/opt067_iteration_contract.json);
  [`fixtures/opt067_prompt_pair.json`](fixtures/opt067_prompt_pair.json);
  [`cuda/opt067_prompt_pair_test.cu`](cuda/opt067_prompt_pair_test.cu);
  [`cuda/quant_mmq_mma.cuh`](cuda/quant_mmq_mma.cuh);
  [`cuda/quant_mmv.h`](cuda/quant_mmv.h);
  [`cuda/full_scheduler.cu`](cuda/full_scheduler.cu);
  [`tests/test_opt067_prompt_pair.py`](tests/test_opt067_prompt_pair.py);
  [`evidence/optimization/opt067-prompt-pair/REPORT.md`](evidence/optimization/opt067-prompt-pair/REPORT.md).
  Proof is paired-kernel wiring, resource/quality evidence, and component
  complete-FFN screen — not combined E2E (OPT-069 owns that gate).
- Marked OPT-067 `done`; delivery is limited to the verified task scope plus
  this ledger/audit bookkeeping. `plan.md` is unchanged. OPT-016 and OPT-056 stay
  `blocked`. Next eligible pending by ledger row order: **OPT-068**. No tok/s
  claim (`claims_throughput=false`).
