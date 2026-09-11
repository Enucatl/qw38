# Source and evidence ledger

[Index](README.md)

Pin exact revisions in the milestone-1 fixture; moving `main` links below are
discovery links, not reproducibility records.

## Semantic authorities

| Source | Use | Status |
|---|---|---|
| [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | dimensions, layer schedule, state dtype, positions, vision | **External primary** |
| [Qwen3.8-27B repository](https://huggingface.co/Qwen/Qwen3.8-27B) | checkpoint, tokenizer, processor, template, model card | **External primary** |
| [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | norm, packed layouts, recurrence, attention, residual order | **External primary** |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | independent trace after pinning a supporting revision | **External oracle** |

The Qwen3.8 config declares `Qwen3_5ForConditionalGeneration`; this is why the
official Transformers implementation path contains `qwen3_5`.

## DwarfStar anchors

Baseline: DwarfStar [`c1d4597`](https://github.com/antirez/ds4/tree/c1d4597a80e300b803dc642519718f2c999589da).
Relevant reusable evidence lives in `ds4_shape`, `model_open`,
`config_validate_model`, `weights_bind`, `ds4_session_create`,
`ds4_session_save_payload`, `forward_token_raw_swa_cpu`,
`prefill_layer_major_cpu`, CUDA MMV/MMQ dispatch, allocation guards, and graph
capture. These are **source-verified patterns**, not Qwen support or 5090 results.

DeepSeek compressed attention, sparse indexer, MoE expert streaming, mHC, and
DSpark are explicitly rejected as Qwen model semantics. OPT-015 inspects local
`../ds4` as MIT technique inspiration only (PIN-002 pin `c1d4597`, inspected
HEAD `c238077`); ds4 cannot run this Qwen GGUF, and there is no ds4 same-model
baseline.

## Focused implementation dependencies

- [utf8proc](https://github.com/JuliaStrings/utf8proc) is pinned as a submodule
  at `d7bf128df773c2a1a7242eb80e51e91a769fc985` (`v2.11.0`). Quartz uses only its
  NFC normalization and Unicode general-category primitives for TOK-001. Its MIT
  license and Unicode data notice are retained in the submodule. Qwen-specific
  splitting, byte mapping, special-token handling, and BPE remain local code.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` supplies the MIT-licensed GGML
  `block_q4_K`/`block_q6_K` layouts and scalar dequantization equations pinned in
  [`pins/quant_contract.json`](../pins/quant_contract.json). Quartz uses those
  definitions as the CPU-001 Q4_K/Q6_K and discovered CPU-005 Q8_0 format
  authority; its local scalar implementation and independently stored fixtures
  remain code-reviewed and differentially tested boundaries.
- [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
  revision `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c` is the Apache-2.0
  semantic authority for CPU-002's GDN convolution, L2 normalization, gate, head
  mapping, and recurrent mutation order. Its source SHA-256 and used symbols are
  frozen in [`pins/gdn_contract.json`](../pins/gdn_contract.json). The local
  fixture is transparently labeled a scalar transcription. At CPU-002, the
  direct eager trace remained ORA-001 work; ORA-004 later completed it.
- The same pinned Transformers file is CPU-003's authority for partial text
  RoPE, grouped KV repetition, eager causal attention, per-head RMSNorm, query
  output gating, and the SwiGLU MLP. Production dimensions, order, used symbols,
  and source identity are frozen separately in
  [`pins/attention_ffn_contract.json`](../pins/attention_ffn_contract.json).
- llama.cpp's pinned `ggml_tensor` element counts and byte-stride definition are
  CPU-006's physical row-layout authority. The source hash, dimension-0 row-width
  interpretation, format block sizes, and admitted fixture link are frozen in
  [`pins/tensor_layout_contract.json`](../pins/tensor_layout_contract.json).
- llama.cpp's pinned Qwen converter and Qwen3.5 graph are CPU-007's authority
  for folded `-exp(A_log)`, ordinary norm offset conversion, convolution
  squeezing, and value-head tiling. Exact revision and file hashes, the affected
  tensor roles, and the local transform meanings are frozen in
  [`pins/gguf_conversion_contract.json`](../pins/gguf_conversion_contract.json).
  Quartz implements only the small reversible layout boundary and two explicit
  scalar conventions; it does not copy the upstream model graph.
- The official Qwen model contract and the already pinned GGUF tensor inventory
  are CPU-008's authority for typed global, common-layer, GDN, and attention
  fields. [`src/weights.cpp`](../src/weights.cpp) deliberately repeats the exact
  scheduler-facing schema and binds only non-owning mapped views; this increment
  introduces no new external implementation dependency.
- The pinned Transformers Qwen3.5 source and llama.cpp Qwen converter jointly
  define CPU-009's packed projection boundary: semantic Q/K/V and per-head
  query/gate slices come from Transformers, while the post-slice GGUF value-head
  order comes from the converter. Their existing hashes and the exact production
  ranges are collected in
  [`pins/projection_layout_contract.json`](../pins/projection_layout_contract.json).
- CPU-010 introduces no new external dependency. It composes the already pinned
  typed GGUF weights, scalar quantization equations, tensor-row arithmetic, and
  packed-layout contract. The deterministic real-row evidence and physical row
  hashes are frozen in
  [`fixtures/mixer_projections.json`](../fixtures/mixer_projections.json).
- CPU-011 composes the same pinned sources through a state-mutating real GDN
  layer. The GDN contract now explicitly includes `Qwen3_5RMSNormGated`; selected
  real parameter/state taps are independently transcribed and frozen in
  [`fixtures/real_gdn_step.json`](../fixtures/real_gdn_step.json). The web reader
  could not serve the pinned commit URL during re-verification, so the exact
  already-hashed raw source was read directly; no moving branch was substituted.
- CPU-012 composes the pinned Transformers SwiGLU equation, typed GGUF views,
  direct-scale conversion contract, and Q4_K row decoder. Selected gate/up rows
  are independently decoded and physically hashed in
  [`fixtures/real_ffn_step.json`](../fixtures/real_ffn_step.json). At CPU-012,
  the complete down projection was native regression evidence; TRC-001,
  TRC-002, and ORA-001 later supplied direct semantic trace admission.
- CPU-013 composes the same pinned Transformers attention contract with the
  typed layer-3 Q8_0/Q6_K GGUF views and packed projection rules. The independent
  two-position transcription and exact selected physical-row hashes are frozen
  in [`fixtures/real_attention_step.json`](../fixtures/real_attention_step.json).
  At CPU-013, its full output projection was native regression evidence pending
  direct TRC-001/TRC-002/ORA-001 trace admission; those gates later completed.
- CPU-014 introduces no new external implementation dependency. It composes the
  already pinned and separately admitted real GDN, attention, and SwiGLU
  boundaries in Transformers decoder-layer order. The explicitly labeled native
  structural regression is frozen in
  [`fixtures/real_layer_composition.json`](../fixtures/real_layer_composition.json).
  It did not replace the then-pending independent full-layer trace authority.
- CPU-015 introduces no new external implementation dependency. It executes the
  already admitted typed embedding, final-norm, and output views using the
  pinned Q4_K/Q6_K scalar contracts. Independently decoded endpoint/interior
  rows, direct-scale normalization, and selected logits are frozen in
  [`fixtures/real_model_boundaries.json`](../fixtures/real_model_boundaries.json).
  The deterministic embedding-to-output diagnostic is a boundary proof, not a
  real 64-layer continuation authority.
- CPU-016 introduces no new external implementation dependency. It composes all
  already pinned and admitted scalar boundaries using the official 48-GDN/16-
  attention layer schedule. [`fixtures/real_scalar_token.json`](../fixtures/real_scalar_token.json)
  is explicitly a native structural zero-state regression. By itself it must not
  be cited as Transformers, llama.cpp, quality, or continuation agreement;
  TRC-001/TRC-002/ORA-001 later supplied that separate evidence.
- TRC-001 introduces no new external implementation dependency. The deliberately
  narrow JSON plus little-endian FP32 format and comparison semantics are local
  Quartz contracts frozen in [`pins/trace_contract.json`](../pins/trace_contract.json).
  They carry the already pinned model/tool identities but do not promote any
  producer to semantic authority. At TRC-001 completion, real cross-runtime
  evidence remained TRC-002/ORA-001 work.
- TRC-003 introduces no new external implementation dependency. Its callback,
  tensor-view validation, tap registry, filter semantics, and separate
  diagnostic-object build are local infrastructure. Synthetic filter output is
  structural evidence only; it is not a Transformers or llama.cpp trace.
- TRC-002's scalar increment introduces no new external implementation
  dependency. It exposes already implemented scalar stages through the local
  diagnostic sink and wraps them with the local trace-v1 writer. The filtered
  final-norm check uses the explicitly labeled native structural fixture. At
  TRC-002 completion, direct pinned Transformers and llama.cpp comparison
  remained ORA-001 work; CUDA tap wiring remains deferred to its kernel gates.
- CPU-004 introduces no new external implementation dependency. Its chunk loop
  invokes the already admitted one-token scalar scheduler without changing
  arithmetic order. [`fixtures/real_scalar_chunk.json`](../fixtures/real_scalar_chunk.json)
  is explicitly native structural equivalence evidence and is not a
  Transformers or llama.cpp continuation fixture.
- ORA-002 executes pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` as an independent same-GGUF
  authority. The reproducible build and raw-token adapter are frozen in
  [`pins/llama_authority_contract.json`](../pins/llama_authority_contract.json);
  the repository-owned adapter calls the public llama API and copies no model
  graph. Complete raw logit rows stay in ignored evidence storage, while their
  hashes and numeric summaries are retained in
  [`fixtures/llama_scalar_authority.json`](../fixtures/llama_scalar_authority.json).
  At ORA-002 completion, this comparison was not a substitute for the then-
  pending pinned Transformers authority or ORA-004 tolerance freeze.
- ORA-003 executes the official Qwen checkpoint revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` through Transformers revision
  `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`. Every Safetensors shard and
  support file is pinned in
  [`pins/transformers_authority_contract.json`](../pins/transformers_authority_contract.json),
  and the Python environment is hash-locked. Local hook infrastructure only
  observes original upstream operations. The raw official logit/tap blobs stay
  in ignored evidence storage; their identities, all 238 tap records, measured
  resource use, and reporting-only cross-artifact metrics are retained in
  [`fixtures/transformers_scalar_authority.json`](../fixtures/transformers_scalar_authority.json).
  At ORA-003 completion, per-tap admission tolerances remained ORA-004 work;
  the following entry records their completion.
- ORA-004 introduces no new model implementation source. It aligns the already
  pinned Transformers, llama.cpp, and Quartz diagnostics through explicit local
  mappings in [`tools/compare_scalar_authorities.py`](../tools/compare_scalar_authorities.py).
  The only data reorder is the previously documented pinned-GGUF 16-key-head by
  3-value-replica permutation; its inverse is tested. Raw blobs remain ignored,
  while [`fixtures/scalar_authority_alignment.json`](../fixtures/scalar_authority_alignment.json)
  retains their hashes and 350 attributed comparisons across 194 aligned rows.
  Immutable pre-CUDA gates
  and the deterministic calibration rule are frozen in
  [`pins/scalar_oracle_tolerances.json`](../pins/scalar_oracle_tolerances.json).
- CUD-001 adapts the pinned llama.cpp revision's MIT-licensed Q8_1 activation
  staging and Q4_K/Q6_K CUDA vector-dot technique. The consulted files and
  revision are frozen in
  [`pins/cuda_quant_contract.json`](../pins/cuda_quant_contract.json). Quartz
  retains the packed formats but uses a smaller local FP32-scale Q8 scratch
  layout, readable per-value unpacking, and warp-per-row scheduler; it does not
  copy llama.cpp's generic type dispatch or kernel framework. The scalar/device
  metrics and CUDA-event timing samples are frozen in
  [`fixtures/cuda_quant_mmv.json`](../fixtures/cuda_quant_mmv.json).
- CUD-002 introduces no new external implementation source. It extends the
  local CUD-001 decoder and transient Q8 contract with an `8 × 4` result tile;
  [`pins/cuda_mmq_contract.json`](../pins/cuda_mmq_contract.json) freezes its
  layout and limits. Arbitrary prompt-row and tail evidence is retained in
  [`fixtures/cuda_quant_mmq.json`](../fixtures/cuda_quant_mmq.json). Production
  MMQ tile selection is later OPT-009 work rather than an externally borrowed
  performance claim. CUD-002 remains the Q4_K/Q6_K numeric envelope.
- GDN-001 uses the already pinned Transformers Qwen3.5 GDN equations and local
  CPU-002 scalar implementation as its semantic references; it introduces no
  new external source. [`pins/cuda_gdn_contract.json`](../pins/cuda_gdn_contract.json)
  freezes the production shape, prepare/commit protocol, limits, and CUDA source
  identities. Small and production-shape device comparisons are retained in
  [`fixtures/cuda_gdn_step.json`](../fixtures/cuda_gdn_step.json).
- GDN-002 introduces no new external source. It repeats the already admitted
  GDN-001 mutation order inside bounded local windows and compares directly with
  repeated one-token CUDA commits. The window/candidate protocol and current
  source identities are frozen in
  [`pins/cuda_gdn_chunk_contract.json`](../pins/cuda_gdn_chunk_contract.json),
  with boundary and production-shape measurements retained in
  [`fixtures/cuda_gdn_chunk.json`](../fixtures/cuda_gdn_chunk.json).
- ATN-001 implements the attention ordering already pinned from the official
  Transformers Qwen3.5 source for CPU-003. The CUDA-specific two-byte cache,
  candidate-row protocol, frozen ORA-004 gate, and current source identities are
  retained in [`pins/cuda_attention_contract.json`](../pins/cuda_attention_contract.json).
  Layers 3/7/63 and production-shape device results are retained in
  [`fixtures/cuda_attention_decode.json`](../fixtures/cuda_attention_decode.json).
- ATN-002 introduces no new semantic source. It repeats ATN-001 in strict token
  order while selecting committed or candidate BF16 rows by causal position.
  The linear workspace, whole-chunk transaction, capacity arithmetic, and source
  identities are frozen in
  [`pins/cuda_attention_prefill_contract.json`](../pins/cuda_attention_prefill_contract.json).
  Chunk-equivalence and final-position RTX 5090 evidence are retained in
  [`fixtures/cuda_attention_prefill.json`](../fixtures/cuda_attention_prefill.json).
- OPT-005 introduces no new external implementation source. Its tiled online
  softmax, 32-row shared-memory tile, two-grid launch topology, and retained
  reference boundary are local derivations over admitted ATN-001/ATN-002
  arithmetic. The boundary and local source digests are frozen in
  [`pins/cuda_tiled_attention_contract.json`](../pins/cuda_tiled_attention_contract.json).
  The dedicated fixture is one measured pinned RTX 5090 component record
  (`CUDA 13.0.2`, `sm_120`) with 30 tiled and three retained-reference raw
  timing samples per prefix and recomputed speedups; it is not an end-to-end
  recovery measurement:
  [`fixtures/cuda_tiled_attention.json`](../fixtures/cuda_tiled_attention.json).
  No external kernel implementation was copied or adapted.
- OPT-006 introduces no new external implementation source. It is a local
  ownership change over the admitted ATN-001/ATN-002 arithmetic: a grouped
  `(KV head, query row)` block stages one 32-row BF16 K/V tile and reuses it for
  the six mapped query heads. The local contract freezes the production shape,
  semantic predicates, and counter formula in
  [`pins/cuda_gqa_attention_contract.json`](../pins/cuda_gqa_attention_contract.json).
  One pinned RTX 5090 fixture retains byte-exact grouped/per-query output,
  untiled-reference metrics, and executed K/V global-load-request counts at
  2K/8K/32K prefixes. Those counters are explicitly not physical DRAM
  transactions or end-to-end performance evidence:
  [`fixtures/cuda_gqa_attention.json`](../fixtures/cuda_gqa_attention.json).
  No external kernel implementation was copied or adapted.
- OPT-007 introduces no new external implementation source. It is a local
  two-query-row ownership change over admitted ATN-001/ATN-002 and OPT-006
  arithmetic: a `(KV head, query-row tile)` block stages each 32-row BF16 K/V
  tile once for its two rows and six mapped query heads, while per-row causal
  admission preserves the retained one-row operation order. The local
  schema-1 contract freezes the two-row shape, exact semantic predicates,
  launch topology, and resource limits in
  [`pins/cuda_query_row_attention_contract.json`](../pins/cuda_query_row_attention_contract.json).
  One pinned RTX 5090 fixture retains exact short/chunk-boundary and captured
  launch/occupancy evidence; it is component-only and explicitly not a
  speedup or end-to-end measurement:
  [`fixtures/cuda_query_row_attention.json`](../fixtures/cuda_query_row_attention.json).
  No external kernel implementation was copied or adapted.
- OPT-010 introduces no new external implementation source. It is a local
  committed-KV layout change over admitted ATN-001/ATN-002 and OPT-007
  arithmetic: device K/V become head-major token-contiguous so a full 32-row
  committed tile is one contiguous BF16 span per KV head, while candidate
  rows stay token-major and checkpoints keep the SES-003 logical token-major
  payload under the same magic, version, and layout hash. The schema-1
  contract freezes the physical index, tile-span predicates, exact GQA
  equality, and capacity product in
  [`pins/cuda_kv_tile_layout_contract.json`](../pins/cuda_kv_tile_layout_contract.json).
  One pinned RTX 5090 fixture retains bijection, pack round-trip, full-tile
  pointer spans, byte-exact production versus one-row outputs, and the
  explicit pointer-span / exact-value proof limit; it is not Nsight DRAM,
  latency, or end-to-end evidence:
  [`fixtures/cuda_kv_tile_layout.json`](../fixtures/cuda_kv_tile_layout.json).
  No external kernel implementation was copied or adapted.
- OPT-011 introduces no new external implementation source. It is a local
  prompt-orchestration change over admitted SCH-002/OPT-008 chunks, OPT-002
  residual-add-norm fusion, OPT-009 MMQ, and OPT-010 physical KV scatter: one
  batched embedding kernel that preserves the BF16 round-trip, a row-wise
  residual-add-norm on mixer and next-input residuals, one multi-block all-layer
  scatter, and two async last-row D2H copies overlapped with that scatter before
  host publication. `PromptPipelinePath::kUnfusedSerial` is the retained
  exactness reference. The schema-1 contract freezes launch/barrier counters,
  captured grids, fused-versus-unfused predicates, and the component-only proof
  limit in
  [`pins/cuda_prompt_pipeline_contract.json`](../pins/cuda_prompt_pipeline_contract.json).
  One pinned RTX 5090 fixture retains exact fused/unfused equality, poll-8
  cancellation with no scatter, executed counters, and paired CUDA-event
  samples; it is not a Nsight Systems overlap screenshot or an end-to-end
  speedup:
  [`fixtures/cuda_prompt_pipeline.json`](../fixtures/cuda_prompt_pipeline.json).
  The beginner explanation is
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  No external kernel implementation was copied or adapted.
- OPT-012 introduces no new external implementation source. It uses CUDA 13.0.2
  stream-capture, graph instantiate/upload, and graph launch APIs already used
  by OPT-003; no external graph implementation is copied. Production workspaces
  with `prompt_chunk_rows_ == 4096` capture a second 64-graph fused prompt FFN
  set from stable prompt-scratch and weight addresses. Capture is skipped when
  `prompt_chunk_rows_ != 4096`. Replay is `cudaGraphLaunch` on the prompt compute
  stream only when `token_count == 4096`; tails keep ordinary fused
  `execute_prompt_ffn`. Mixer GDN/attention, embedding, KV destination, host
  copies, and commit stay outside the graph. The schema-1 contract freezes the
  128-graph production counts, capacity-65 skip, kernel-node/grid predicates,
  fail-closed mismatch/unfused rules, graph-versus-ordinary fused equality, 64-row
  fallback, poll-8 cancellation, executed counters, and the component-only proof
  limit in
  [`pins/cuda_prompt_graph_contract.json`](../pins/cuda_prompt_graph_contract.json).
  One pinned RTX 5090 fixture retains those executed results and CUDA-event
  times; it is not a whole-chunk graph, Nsight Systems, end-to-end speedup, or
  128K quality claim:
  [`fixtures/cuda_prompt_graph.json`](../fixtures/cuda_prompt_graph.json).
  The 128K owner remeasure after 128 uploads is retained in
  [`fixtures/cuda_memory_fit_post_graph.json`](../fixtures/cuda_memory_fit_post_graph.json).
  The beginner explanation is
  [`docs/53-stable-address-cuda-graphs.md`](53-stable-address-cuda-graphs.md).
  No external kernel implementation was copied or adapted.
- OPT-013 introduces no new external implementation source. It is a local
  derivation of the already admitted GDN-002 sequential 64-token windows: each
  window's gated-delta map is stored as a dense affine operator `(A_w, B_w)`, a
  48-block `A S + B` prefix produces incoming window state, and a parallel
  from-state grid replays sequential window arithmetic. No Flash Linear
  Attention, Triton, or vendor GDN kernel is copied or adapted. Overlay scratch
  uses existing `prompt_projected_bf16_` (`W_fit(4096) = 22`); there is no extra
  session `cudaMalloc`. The schema-1 contract freezes envelopes, overlay
  `W_fit`, launch geometry, fail-closed rules, and the component-only proof
  limit in
  [`pins/cuda_gdn_scan_contract.json`](../pins/cuda_gdn_scan_contract.json).
  One pinned RTX 5090 fixture retains sequential byte-exact regression,
  parallel-versus-sequential envelopes, the 4,096-versus-64-window split,
  captured intra/prefix/from-state nodes, overlay arithmetic, and 30 paired
  CUDA-event samples; it is not Nsight Systems, end-to-end prefill/decode
  speedup, or 128K quality evidence:
  [`fixtures/cuda_gdn_scan.json`](../fixtures/cuda_gdn_scan.json).
  The beginner explanation is
  [`docs/42-cuda-gdn-chunks.md`](42-cuda-gdn-chunks.md).
- OPT-014 introduces no new external implementation source. CUDA events and
  NVTX ranges are already used by OPT-001; this increment records synchronized
  events on the prompt compute stream for an opt-in `PrefillAttribution` during
  production `sync_tokens`. No Nsight Systems or Nsight Compute capture is
  required or performed. The schema-1 contract freezes the eight summing
  categories, 2048-token geometry, remainder tolerance, measured-zero
  graph-at-2048, and `nsight: not_required` in
  [`pins/cuda_prefill_attribution_contract.json`](../pins/cuda_prefill_attribution_contract.json).
  One pinned RTX 5090 fixture retains the live cold 2048-token report from the
  timed diagnostic (`wall_ms` 41963.8828, `graph` 0, `nsight_*` `not_used`); it
  is instrumentation, not a throughput gate, Nsight capture, or llama.cpp
  parity claim:
  [`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json).
  The beginner explanation is
  [`docs/51-runtime-timing-and-nvtx.md`](51-runtime-timing-and-nvtx.md).
  The prefill proof boundary is
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  No external profiler workflow or kernel implementation was copied or adapted.
- OPT-015 introduces no new external implementation source beyond already-pinned
  authorities. It inspects llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT) as the same-GGUF 2K
  explanation—MMA MMQ, fused GDN token loop, MMA flash attention—and inspects
  local `../ds4` HEAD `c238077a87186381bf626cc531bccffe1fef79e7` as MIT
  technique inspiration (PIN-002 pin
  `c1d4597a80e300b803dc642519718f2c999589da`). ds4 cannot run this Qwen GGUF;
  there is no ds4 same-model baseline. Sparse/compressed attention, MoE
  streaming, mHC, DSpark, a DwarfStar fork, and a production cuBLAS dependency
  remain non-transferable. Copied timings are the OPT-014 fixture and the
  2026-09-08 scaling `llama-bench` 2K JSON; this increment does not re-run
  those GPU jobs. The schema-1 contract, claim index, and report are
  [`pins/opt015_recovery_contract.json`](../pins/opt015_recovery_contract.json),
  [`fixtures/opt015_recovery.json`](../fixtures/opt015_recovery.json), and
  [`evidence/optimization/opt015-2k-recovery/REPORT.md`](../evidence/optimization/opt015-2k-recovery/REPORT.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  The 2K comparison and ranked map are
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  Proof limit: not a throughput gate, not llama.cpp parity.
- OPT-017 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors) Q8_0 MMQ
  MMA from `mmq.cuh`, `mma.cuh`, `mmq-config-ampere.cuh`, `mmq-load-tiles.cuh`,
  and `mmq-vec-dot.cuh`. Geometry is 256 threads × 128 output rows × J in
  `{32, 64, 128}` with Q8_1 shared-memory staging and
  `mma.sync.aligned.m16n8k32`. ds4 `cuda/mmq` is the ggml-free launcher pattern
  only; this increment does not copy `../ds4/cuda/mmq/` and does not include
  ggml headers. Production mixer Q8_0 uses `launch_q8_mmq_mma` behind
  `launch_q8_mmq_bf16` when `prompt_rows >= 8`; it does not route Q8_0 through
  `launch_quant_mmq`. The OPT-009 tiled kernel remains the unloosened
  byte-exact reference versus `launch_q8_mmq_bf16_reference`. MMA numeric
  admission is frozen CUD-002 versus diagnostic `launch_quant_mmq_variant`
  `kQ8_0`. The schema-1 contract, measured fixture, and report are
  [`pins/opt017_mixer_mma_contract.json`](../pins/opt017_mixer_mma_contract.json),
  [`fixtures/opt017_mixer_mma.json`](../fixtures/opt017_mixer_mma.json), and
  [`evidence/optimization/opt017-mixer-q8-mma/REPORT.md`](../evidence/optimization/opt017-mixer-q8-mma/REPORT.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: envelopes unloosened; OPT-009 Q8_0 reference remains
  byte-exact; OPT-016 remains the parity gate owner; not an end-to-end 2K
  tok/s gate; no 8K/32K/128K throughput gate.
- OPT-018 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors) Q4_K/Q6_K
  MMQ MMA from `mmq.cuh`, `mma.cuh`, `mmq-config-ampere.cuh`,
  `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, and `quantize.cu`. Geometry is
  `MMQ_ITER_K=256`, packed load-tiles, Q8_1 MMQ Y in the existing prompt
  workspace, and `dim3(32, 8)`. Inner-loop repair uses float accumulators,
  Q4_K DS4 Y, half2 X at SRAM stride 76, and `__launch_bounds__(256, 1)`.
  ds4 `cuda/mmq` is the ggml-free launcher pattern; Q4_K/Q6_K numeric
  admission follows ds4 `cuda/mmq/test/test_mmq_parity.cu` (option C; CPU
  dequant GEMM; `abs > 0.20*sqrt(K)` and `rel > 0.05`). This increment does
  not copy `../ds4/cuda/mmq/` and does not include ggml headers. Production
  `launch_quant_mmq` uses quality MMA when `prompt_rows >= 8`. **Measured,
  RTX 5090:** live exact-2048 `ffn_mmq` 399.287018 ms versus before
  2524.67725 ms and llama.cpp 2K wall 636.184782 ms (`avg_ts` 3219.6604);
  Quartz mean 625.792114 tok/s; `llama_competitive` true;
  `would_pass_opt016` informational false. The schema-1 contract,
  measured fixture, and report are
  [`pins/opt018_ffn_mma_contract.json`](../pins/opt018_ffn_mma_contract.json),
  [`fixtures/opt018_ffn_mma.json`](../fixtures/opt018_ffn_mma.json), and
  [`evidence/optimization/opt018-ffn-mma-quality/REPORT.md`](../evidence/optimization/opt018-ffn-mma-quality/REPORT.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: Q4_K/Q6_K MMQ vs CPU dequant GEMM uses ds4 Q4_K parity
  association gate; fixed abs/rms retired for Q4_K/Q6_K MMQ admission;
  variant retained with exact Q8 staging; parity gate owner remains
  blocked; not an end-to-end 2K tok/s gate; no 8K/32K/128K throughput gate.
- OPT-019 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors) warp-column
  fused GDN from `gated_delta_net.cu` / `gated_delta_net.cuh` and Ampere
  fattn-mma-f16 from `fattn.cu`, `fattn-mma-f16.cuh`, and `mma.cuh`. Production
  GDN fused recurrence is grid `(value_heads, 1, value_width/4)` and block
  `(32, 4)` with `s_shard[4]` FP32 column state. Production prompt attention
  (`token_count >= 16`) uses `ncols2 = 2` and pinned `ncols1 = 16`. This
  increment does not vendor those llama.cpp files, does not copy `../ds4`, and
  does not include ggml headers. ds4 has no GDN analog and is mentioned only
  for multi-row online attention structure. Sequential `kSequentialWindows`
  remains the unloosened GDN-002/OPT-013 reference. Tiled
  `launch_attention_prepare_chunk_tiled` remains the unloosened OPT-005
  reference. Rank-2 fused GDN and Rank-3 warp-0 MMA stay for component A/B.
  **Measured, RTX 5090:** quality GDN versus sequential at 2048 tokens
  `max_abs=1.58324838e-08` / `rms=1.28841632e-10`; quality attention versus
  tiled `max_abs=3.16649675e-06` / `rms=1.31638146e-07`; ncols1 pin 16; live
  exact-2048 `gdn` 1205.38806 ms and `attention` 475.116028 ms versus locked
  befores 1643.46082 / 1148.47461 ms; Quartz mean 978.151855 tok/s versus
  llama.cpp 3197.246224 tok/s; `would_pass_opt016` informational false. The
  schema-1 contract, measured fixture, and report are
  [`pins/opt019_core_recovery_contract.json`](../pins/opt019_core_recovery_contract.json),
  [`fixtures/opt019_core_recovery.json`](../fixtures/opt019_core_recovery.json),
  and
  [`evidence/optimization/opt019-gdn-attention-core/REPORT.md`](../evidence/optimization/opt019-gdn-attention-core/REPORT.md).
  The beginner explanations are
  [`docs/42-cuda-gdn-chunks.md`](42-cuda-gdn-chunks.md) and
  [`docs/44-cuda-attention-prefill.md`](44-cuda-attention-prefill.md).
  Proof limit: envelopes unloosened; sequential GDN remains the reference;
  tiled attention remains the reference; parity gate owner remains blocked;
  not an end-to-end 2K tok/s gate; no 8K/32K/128K throughput gate; mixer
  versus core split is not this increment.
- OPT-020 introduces no new external implementation source. CUDA events and
  NVTX ranges are already used by OPT-001 and OPT-014; this increment splits
  mixer-projection MMQ out of the former composite GDN and attention buckets on
  the prompt compute stream. No Nsight Systems or Nsight Compute capture is
  required or performed. The schema-1 contract freezes the nine summing
  categories, 2048-token geometry, remainder tolerance copied from the historical
  eight-category contract, measured-zero graph-at-2048, and
  `nsight: not_required` in
  [`pins/opt020_prefill_split_contract.json`](../pins/opt020_prefill_split_contract.json).
  One pinned RTX 5090 fixture retains the live cold 2048-token split report
  from the timed diagnostic (`wall_ms` 2086.2561, `mixer_mmq` 1199.25122,
  `gdn_core` 212.156006, `attention_core` 273.986725, `graph` 0, `nsight_*`
  `not_used`); it is instrumentation, not a throughput gate, Nsight
  capture, or llama.cpp parity claim:
  [`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json).
  The beginner explanation is
  [`docs/51-runtime-timing-and-nvtx.md`](51-runtime-timing-and-nvtx.md).
  The prefill proof boundary is
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  The historical eight-category snapshot remains
  [`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json).
  No external profiler workflow or kernel implementation was copied or adapted.
- OPT-021 introduces no new external implementation source. Pinned llama.cpp
  revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` `llama-bench` is already
  the External same-GGUF authority. This increment freezes a local exclusive-
  RTX-5090 exact-4096 keep/reject protocol: three cold production
  `sync_tokens` walls with attribution null and graphs created, compared to
  same-sitting `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99`. The
  schema-1 contract, retained live fixture, and claim-labeled report are
  [`pins/opt021_oracle_contract.json`](../pins/opt021_oracle_contract.json),
  [`fixtures/opt021_oracle.json`](../fixtures/opt021_oracle.json), and
  [`evidence/optimization/opt021-4k-oracle/REPORT.md`](../evidence/optimization/opt021-4k-oracle/REPORT.md).
  The beginner explanations are
  [`docs/61-benchmark-harness.md`](61-benchmark-harness.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: 4K keep/reject oracle; envelopes unloosened; does not
  substitute for the 2K llama.cpp parity gate; attribution null; graphs
  created; scout sitting is not the retained fixture. Quartz ≥ llama.cpp is
  not this gate. No kernel implementation was copied or adapted.
- OPT-022 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors) Q8_0
  quality MMQ from `quantize.cu` (`quantize_mmq_q8_1` D4),
  `mmq-load-tiles.cuh` packed `ggml_cuda_mmq_load_tiles_q8_0`,
  `mmq-vec-dot.cuh` `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`, and Ampere
  `mmq-config-ampere.cuh`. Geometry is `MMQ_ITER_K=256`, packed load-tiles,
  D4 Y in existing `prompt_q8_`, and `dim3(32, 8)`. ds4 `cuda/mmq` is the
  ggml-free launcher pattern; Q8_0 numeric admission follows ds4
  `cuda/mmq/test/test_mmq_parity.cu` `run_q8_0` / `check_close` (CPU dequant
  GEMM; an element fails only when both `abs > 0.05*sqrt(K)` and
  `rel > 0.05`). This increment does not copy `../ds4/cuda/mmq/` and does
  not include ggml headers. Production mixer Q8_0 uses quality MMA behind
  `launch_q8_mmq_bf16` when `prompt_rows >= 8`; mixer GEMMs that share the
  residual activation share one D4 Y per layer. Production still does not
  route Q8_0 through `launch_quant_mmq`. The OPT-009 tiled kernel remains
  the unloosened byte-exact reference versus `launch_q8_mmq_bf16_reference`.
  Rank-1 `launch_q8_mmq_mma` is retained and is not production. **Measured,
  RTX 5090:** live exclusive cold exact-4096 keep Quartz mean 1680.38025
  tok/s versus the frozen oracle baseline 967.267761; `reverted` false;
  `successor_oracle` true; `production_q8` `quality_mma_shared_y`. Live
  numbers stay in the report; this ledger does not replace them. The
  schema-1 contract, measured fixture, and report are
  [`pins/opt022_mixer_q8_quality_contract.json`](../pins/opt022_mixer_q8_quality_contract.json),
  [`fixtures/opt022_mixer_q8_quality.json`](../fixtures/opt022_mixer_q8_quality.json),
  and
  [`evidence/optimization/opt022-mixer-q8-quality/REPORT.md`](../evidence/optimization/opt022-mixer-q8-quality/REPORT.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: mixer Q8 quality plus shared residual Y under the Q8
  association rule; 4K keep/reject versus the frozen oracle baseline;
  envelopes unloosened; OPT-009 Q8_0 reference remains byte-exact; OPT-016
  remains the parity gate owner; does not substitute for the 2K llama.cpp
  parity gate; Quartz ≥ llama.cpp is not this gate.
- OPT-023 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors) Ampere
  Q8_0 MMQ I/J geometry from `mmq-config-ampere.cuh` (I=128; J in
  `{8…128}`) as the reason not to treat small `output_rows` as llama.cpp
  MMVQ. `mmvq.cu` `ggml_cuda_should_use_mmvq` is small `ne11` (token
  batch), not small `nrows`; a 4096-token prefill is MMQ. Quartz therefore
  A/Bs small-I quality MMA (`mma_i32_j128` / `mma_i64_j128`) and prompt-MMV
  `launch_q8_mmq_bf16_variant` tile 1 against I=128 Fallback, and does not
  loop decode `q8_mmv_bf16` or vendor llama.cpp MMVQ. Quality MMA numeric
  admission remains ds4 `cuda/mmq/test/test_mmq_parity.cu` Q8 association
  (`abs > 0.05*sqrt(K)` AND `rel > 0.05`). This increment does not copy
  `../ds4/cuda/mmq/` and does not include ggml headers. Production mixer
  Q8_0 with `output_rows < 128` (GDN α/β) dispatches templated I=32 quality
  MMA on the shared residual D4 Y; large mixer GEMMs stay I=128 / J=128
  quality MMA with shared Y. Decode `q8_mmv_bf16` is unchanged. The OPT-009
  tiled kernel remains the unloosened byte-exact reference versus
  `launch_q8_mmq_bf16_reference`. **Measured, RTX 5090:** A/B winner
  `mma_i32_j128`; live exclusive cold exact-4096 keep versus the frozen
  post-OPT-022 oracle baseline 1680.80627; `reverted` false;
  `successor_oracle` true; `production_skinny` true. Live numbers stay in
  the report; this ledger does not replace them. The schema-1 contract,
  measured fixture, and report are
  [`pins/opt023_skinny_mixer_contract.json`](../pins/opt023_skinny_mixer_contract.json),
  [`fixtures/opt023_skinny_mixer.json`](../fixtures/opt023_skinny_mixer.json),
  and
  [`evidence/optimization/opt023-skinny-mixer/REPORT.md`](../evidence/optimization/opt023-skinny-mixer/REPORT.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: skinny-M mixer dispatch under the Q8 association rule; 4K
  keep/reject versus the post-OPT-022 oracle baseline; envelopes
  unloosened; OPT-009 Q8_0 reference remains byte-exact; OPT-016 remains
  the parity gate owner; does not substitute for the 2K llama.cpp
  parity gate; Quartz ≥ llama.cpp is not this gate.
- OPT-024 adapts ds4 MIT dense Q8 D2R / kind-5 aligned SoA technique
  (`proto_gemm_dense_q8_d2r.cu`, `ds4_mmq_d2r.cu` comments at inspected
  `../ds4` HEAD) as inspiration only: row-major `[half d][int8 qs]` with
  64-byte scale-plane pad, aligned int8 loads, `mma.m16n8k32.s8`, D4 Y,
  float fold `acc += C * d_w * d_y`. Quality MMA numeric admission remains
  ds4 `cuda/mmq/test/test_mmq_parity.cu` Q8 association
  (`abs > 0.05*sqrt(K)` AND `rel > 0.05`). This increment does not copy
  `../ds4/cuda/mmq/` and does not include ggml headers. Production large
  mixer Q8_0 (`output_rows >= 128`) remains I=128 / J=128 quality MMA on
  the shared residual D4 Y; skinny α/β stay `mma_i32_j128`; decode
  `q8_mmv_bf16` stays on GGUF 34-byte blocks. D2R launchers remain
  non-production symbols. The OPT-009 tiled kernel remains the unloosened
  byte-exact reference versus `launch_q8_mmq_bf16_reference`. **Measured,
  RTX 5090:** A/B winner `quality_mma` (`win=false`; D2R did not strictly
  beat quality MMA on every timed large shape); live exclusive cold
  exact-4096 reject
  versus the frozen successor-oracle baseline 1687.86169; `reverted`
  true; `successor_oracle` false; `production_d2r` false. Live numbers
  stay in the report; this ledger does not replace them. The schema-1
  contract, rejected fixture, report, and rejection are
  [`pins/opt024_mixer_q8_d2r_contract.json`](../pins/opt024_mixer_q8_d2r_contract.json),
  [`fixtures/opt024_mixer_q8_d2r.json`](../fixtures/opt024_mixer_q8_d2r.json),
  [`evidence/optimization/opt024-mixer-q8-d2r/REPORT.md`](../evidence/optimization/opt024-mixer-q8-d2r/REPORT.md),
  and
  [`evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md`](../evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: aligned-SoA D2R for large mixer Q8_0 under the Q8
  association rule when it beats quality MMQ; 4K keep/reject versus the
  current successor-oracle baseline; envelopes unloosened; OPT-009 Q8_0
  reference remains byte-exact; OPT-016 remains the parity gate owner;
  does not substitute for the 2K llama.cpp parity gate; Quartz ≥ llama.cpp
  is not this gate.
- OPT-025 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `quantize.cu` `quantize_mmq_q8_1` DS4 `block_q8_1_mmq` packing as the
  shared gate/up Y and SwiGLU-into-down Y layout. Q4_K MMA numeric
  admission remains ds4 `cuda/mmq/test/test_mmq_parity.cu` Q4_K parity
  association. This increment does not copy `../ds4/cuda/mmq/` and does
  not include ggml headers. Production prompt FFN Q4_K reuses one DS4
  Q8_1 of the FFN-norm activation for gate and up and writes down-leg Y
  from SwiGLU with a register BF16 round-trip and no global BF16 mid
  store. Mixer Q8_0 quality MMA, skinny `mma_i32_j128`, and the D2R
  reject stay. Decode FFN stays MMV. The OPT-009 tiled kernel remains the
  unloosened byte-exact reference versus `launch_q8_mmq_bf16_reference`.
  **Measured, RTX 5090:** A/B winner `shared_y_swiglu_q8`; live exclusive
  cold exact-4096 keep versus the frozen successor-oracle baseline
  1687.86169; `reverted` false; `successor_oracle` true;
  `production_ffn_optimized` true. Live numbers stay in the report; this
  ledger does not replace them. The schema-1 contract, measured fixture,
  and report are
  [`pins/opt025_ffn_shared_y_contract.json`](../pins/opt025_ffn_shared_y_contract.json),
  [`fixtures/opt025_ffn_shared_y.json`](../fixtures/opt025_ffn_shared_y.json),
  and
  [`evidence/optimization/opt025-ffn-shared-y/REPORT.md`](../evidence/optimization/opt025-ffn-shared-y/REPORT.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: dense FFN shared-Y and SwiGLU-into-down Q8 under unloosened
  Q4_K association; 4K keep/reject versus the then-current oracle
  baseline; envelopes unloosened; OPT-009 Q8_0 reference remains
  byte-exact; OPT-016 remains the parity gate owner; does not substitute
  for the 2K llama.cpp parity gate; Quartz ≥ llama.cpp is not this gate.
- OPT-026 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  Ada+ fattn stream-K occupancy from `fattn.cu` / `fattn-common.cuh`
  (`cc >= ADA_LOVELACE` stream-K plus occupancy-2 `__launch_bounds__` for
  Ampere DKQ=DV=256 ncols=32). This increment does not vendor
  `fattn-mma-f16.cuh`, does not copy `../ds4`, and does not include ggml
  headers. Production prompt attention keeps ncols1=16 and splits each
  query tile's KV range across `grid.z=2`, then combines online-softmax
  partials. Combine buffers alias existing `prompt_projected_bf16_` and
  `prompt_q8_`; score scratch stays untouched; no extra `cudaMalloc`.
  Occupancy-2 whole-tile was A/B-eligible and slower than stream-K.
  Mixer Q8 quality, skinny `mma_i32_j128`, and FFN `shared_y_swiglu_q8`
  stay. Decode attention stays one-token. Tiled
  `launch_attention_prepare_chunk_tiled` remains the unloosened OPT-005
  reference. **Measured, RTX 5090:** A/B winner `stream_k`; live exclusive
  cold exact-4096 keep versus the frozen successor-oracle baseline
  1709.21912; `reverted` false; `successor_oracle` true;
  `production_fattn_optimized` true; `ladder_exhausted` true. Live
  numbers stay in the report; this ledger does not replace them. The
  schema-1 contract, measured fixture, and report are
  [`pins/opt026_fattn_streamk_contract.json`](../pins/opt026_fattn_streamk_contract.json),
  [`fixtures/opt026_fattn_streamk.json`](../fixtures/opt026_fattn_streamk.json),
  and
  [`evidence/optimization/opt026-fattn-streamk/REPORT.md`](../evidence/optimization/opt026-fattn-streamk/REPORT.md).
  The beginner explanation is
  [`docs/44-cuda-attention-prefill.md`](44-cuda-attention-prefill.md).
  Proof limit: fattn occupancy / stream-K under unloosened OPT-005
  envelopes; 4K keep/reject versus the then-current oracle baseline;
  envelopes unloosened; tiled attention remains the reference; OPT-016
  remains the parity gate owner; does not substitute for the 2K llama.cpp
  parity gate; Quartz ≥ llama.cpp is not this gate.
- OPT-027 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  Ada+ persistent stream-K from `fattn-common.cuh` (`nblocks_stream_k`,
  5% occupancy-loss efficiency rounding, uniform/general fixup) and
  `fattn-mma-f16.cuh` linearized `kbc` / `kbc_stop` tiles. This increment
  does not vendor those llama.cpp files, does not copy `../ds4`, and does
  not include ggml headers. Persistent `dst_tmp_meta` would alias existing
  `prompt_q8_` / `prompt_projected_bf16_`; score scratch stays untouched;
  no extra `cudaMalloc`. Production prompt attention remains OPT-026
  Ada+ stream-K (`grid.z=2`) after the paired A/B loss. Persistent kernels
  remain as non-production symbols. `kSelectedFattnPath` stays
  `"stream_k"`; `kSelectedPersistentFattnPath` is `off`. Mixer Q8 quality,
  skinny `mma_i32_j128`, and FFN `shared_y_swiglu_q8` stay. Decode
  attention stays one-token. Tiled `launch_attention_prepare_chunk_tiled`
  remains the unloosened OPT-005 reference. **Measured, RTX 5090:** A/B
  winner `stream_k` (`win=false`; persistent did not strictly beat
  `stream_k`); live exclusive cold exact-4096 reject versus the frozen
  successor-oracle baseline 1746.71973; `reverted` true;
  `successor_oracle` false; `production_persistent_installed` false;
  `ladder_exhausted` false. Live numbers stay in the report; this ledger
  does not replace them. The schema-1 contract, rejected fixture, report,
  and rejection are
  [`pins/opt027_persistent_fattn_contract.json`](../pins/opt027_persistent_fattn_contract.json),
  [`fixtures/opt027_persistent_fattn.json`](../fixtures/opt027_persistent_fattn.json),
  [`evidence/optimization/opt027-persistent-fattn/REPORT.md`](../evidence/optimization/opt027-persistent-fattn/REPORT.md),
  and
  [`evidence/optimization/opt027-persistent-fattn/REJECTION.md`](../evidence/optimization/opt027-persistent-fattn/REJECTION.md).
  The beginner explanation is
  [`docs/44-cuda-attention-prefill.md`](44-cuda-attention-prefill.md).
  Proof limit: persistent Ada+ fattn stream-K under unloosened OPT-005
  envelopes; 4K keep/reject versus the then-current oracle baseline;
  envelopes unloosened; tiled attention remains the reference; OPT-016
  remains the parity gate owner; does not substitute for the 2K llama.cpp
  parity gate; Quartz ≥ llama.cpp is not this gate.
- OPT-028 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  Q4_K/Q6_K MMQ stream-K from `mmq.cuh` (`mul_mat_q` kbc walk,
  `mul_mat_q_stream_k_fixup`, 90% tile-efficiency launch rule). This
  increment does not vendor `mmq.cuh`, does not copy `../ds4/cuda/mmq/`,
  and does not include ggml headers. Production Q4_K/Q6_K quality MMA
  remains 2D tiling after the paired A/B loss. Stream-K kernels remain as
  non-production symbols. `kSelectedMmqStreamKPath` is `off`.
  `kSelectedFfnPath` stays `"shared_y_swiglu_q8"`. Mixer Q8 quality,
  skinny `mma_i32_j128`, and fattn Ada+ stream-K stay. Decode FFN stays
  MMV. J stays 128. Scheduler fixup aliases exist and are unused while
  the selected path is `off`; no extra `cudaMalloc`. Prompt FFN graphs
  were not recaptured. **Measured, RTX 5090:** A/B winner `off`
  (`win=false`; neither `stream_k` nor `stream_k_nsm` strictly beat 2D
  tiling); live exclusive cold exact-4096 reject versus the frozen
  successor-oracle baseline 1746.71973; `reverted` true;
  `successor_oracle` false; `production_mmq_stream_k_installed` false;
  `ladder_exhausted` false. Live numbers stay in the report; this ledger
  does not replace them. The schema-1 contract, rejected fixture, report,
  and rejection are
  [`pins/opt028_mmq_streamk_contract.json`](../pins/opt028_mmq_streamk_contract.json),
  [`fixtures/opt028_mmq_streamk.json`](../fixtures/opt028_mmq_streamk.json),
  [`evidence/optimization/opt028-mmq-streamk/REPORT.md`](../evidence/optimization/opt028-mmq-streamk/REPORT.md),
  and
  [`evidence/optimization/opt028-mmq-streamk/REJECTION.md`](../evidence/optimization/opt028-mmq-streamk/REJECTION.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: Q4_K/Q6_K MMQ stream-K under unloosened Q4_K association
  when a paired A/B wins; 4096 FFN graph recapture if nodes change; 4K
  keep/reject versus the then-current oracle baseline; envelopes
  unloosened; OPT-009 Q8_0 reference remains byte-exact; OPT-016 remains
  the parity gate owner; does not substitute for the 2K llama.cpp parity
  gate; Quartz ≥ llama.cpp is not this gate.
- OPT-037 reuses the already-cited llama.cpp Ampere MMQ I/J geometry
  (`mmq-config-ampere.cuh` / quality MMA `MMQ_ITER_K=256`, packed
  load-tiles, block `dim3(32, 8)`) at revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors).
  This increment does not vendor ggml headers and does not copy
  `../ds4/cuda/mmq/`. Production 4096-row Q4_K FFN quality MMA stays
  I=128 / J=128 after the independent gate/up/down sweep. Shared-Y /
  SwiGLU-into-Q8 stays. Stream-K stays `off`. Mixer skinny stays
  `mma_i32_j128`. Global `selected_mma_mmq_prompt_tile()` stays 128.
  Decode FFN stays MMV. Prompt FFN graphs were not recaptured. **Measured,
  RTX 5090:** A/B winner `i128_j128` on every projection (`any_win=false`);
  live tok/s sitting skipped; `reverted` true. Live numbers stay in the
  report; this ledger does not replace them. The schema-1 contract,
  rejected fixture, report, and rejection are
  [`pins/opt037_ffn_tile_contract.json`](../pins/opt037_ffn_tile_contract.json),
  [`fixtures/opt037_ffn_tiles.json`](../fixtures/opt037_ffn_tiles.json),
  [`evidence/optimization/opt037-ffn-tiles/REPORT.md`](../evidence/optimization/opt037-ffn-tiles/REPORT.md),
  and
  [`evidence/optimization/opt037-ffn-tiles/REJECTION.md`](../evidence/optimization/opt037-ffn-tiles/REJECTION.md).
  The beginner explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
  Proof limit: 4K FFN I/J per projection under unloosened Q4_K
  association; keep requires admitted component wins, improved P, frozen
  MMQ envelopes, and the cross-workload guard; envelopes unloosened;
  OPT-016 remains the parity gate owner; does not substitute for the 2K
  llama.cpp parity   gate; Quartz ≥ llama.cpp is not this gate.
- OPT-038 introduces no new external implementation source. Pinned llama.cpp
  revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` is already the External
  same-GGUF authority (`llama-bench` for P; public `llama.h` /
  `llama_time_us()` for D128/D2048). This increment freezes a local exclusive-
  RTX-5090 post-ladder P/D128/D2048 refresh on current production objects:
  exact-4096 production `sync_tokens` with attribution null and graphs created,
  plus prefix-128 and prefix-2048 decode (256 predetermined tokens, 3 warm + 30
  measured) matched by `qw38-llama-decode-oracle`, plus new OPT-038 attribution
  diagnostics with independent `raw_host_wall_ms` alongside unchanged
  `finish_*_attribution` adjusted reconstruction. Random-token `llama-bench`
  decode is informational. Matched-component experiment specifications are
  written but not run in this sitting. Exclusive decode categories live on
  opt-in `DecodeAttribution`; public `RuntimeTimings` stay composite. The
  schema-1 contract, retained live fixture, claim-labeled report, and component
  protocol are
  [`pins/opt038_post_ladder_gap_contract.json`](../pins/opt038_post_ladder_gap_contract.json),
  [`fixtures/opt038_post_ladder_gap.json`](../fixtures/opt038_post_ladder_gap.json),
  [`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](../evidence/optimization/opt038-post-ladder-gap/REPORT.md),
  and
  [`evidence/optimization/opt038-post-ladder-gap/COMPONENT-PROTOCOL.md`](../evidence/optimization/opt038-post-ladder-gap/COMPONENT-PROTOCOL.md).
  Live tok/s stay in that report; this ledger does not replace them. The
  beginner explanations are
  [`docs/61-benchmark-harness.md`](61-benchmark-harness.md),
  [`docs/51-runtime-timing-and-nvtx.md`](51-runtime-timing-and-nvtx.md),
  [`docs/06-system-optimization.md`](06-system-optimization.md), and
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  Proof limit: post-ladder P/D128/D2048 measurements; exclusive subsystem
  breakdowns; independent raw host-wall accounting; matched pinned llama.cpp;
  matched-component experiment specifications; recorded next-task order; claims
  no performance improvement; does not publish a successor oracle; accepted keep
  denominators remain unchanged; envelopes unloosened; does not substitute for
  the 2K llama.cpp parity gate; llama-bench random decode is informational.
  Quartz ≥ llama.cpp is not this gate. No kernel implementation was copied or
  adapted.
- OPT-039 is a local derivation over admitted 16/16 partitioned one-token decode.
  It A/B's the accepted CTA 16-partition kernel against a warp-owned query-head
  kernel at fixed partition count: grid `(24, 16)`, block `(32)`, eight
  dimensions per lane, warp-shuffle QK reduction, register VKQ, and the
  unchanged FP32 ascending-part merge. File-level provenance for register-
  resident Q/KQ/VKQ and warp reductions is already External: pinned llama.cpp
  revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `fattn-vec.cuh`. This increment does not vendor `fattn-vec.cuh` or
  `fattn-common.cuh`, does not include ggml headers, and does not copy
  `../ds4`. Prefill fattn and tiled prefill for `token_count >= 2` stay.
  Partials alias idle `prompt_projected_bf16_` / `prompt_q8_`; score scratch
  stays unused; no extra `cudaMalloc`. Keep denominators are the frozen
  then-current accepted P / D128 / D2048 means and p95s copied into the
  contract. **Measured, RTX 5090:** D2048 A/B winner `warp_query`; production
  pin `warp_query`; `reverted` false; `keep_sitting_skipped` false; `status`
  measured. Live tok/s stay in the report; this ledger does not replace them.
  The schema-1 contract, measured fixture, and report are
  [`pins/opt039_decode_warp_contract.json`](../pins/opt039_decode_warp_contract.json),
  [`fixtures/opt039_decode_warp.json`](../fixtures/opt039_decode_warp.json),
  and
  [`evidence/optimization/opt039-decode-warp/REPORT.md`](../evidence/optimization/opt039-decode-warp/REPORT.md).
  The beginner explanations are
  [`docs/43-cuda-attention-decode.md`](43-cuda-attention-decode.md) and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: frozen attention envelopes; exact candidate KV and state
  isolation; lower D2048 component time; improved D2048; cross-workload guard;
  then-current accepted P D128 D2048 are the keep denominators; does not
  substitute for the 2K llama.cpp parity gate; Quartz ≥ llama.cpp is not this
  gate. Envelopes unloosened; Nsight is not used.
- OPT-040 is a local derivation over admitted warp-column prompt GDN quality.
  It hoists per-(token, key_head) Q/K inverse normalization into existing
  `prompt_projected_bf16_` float overlay scratch between parallel conv and
  warp-column recurrence, using one shared device helper so association cannot
  drift from the repeated in-loop path. File-level provenance for pre-loop Q/K
  L2 is already External: pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `qwen35.cpp` `ggml_l2_norm` on `q_conv`/`k_conv` before recurrent attention.
  This increment does not vendor `qwen35.cpp` or `gated_delta_net.cu`, does not
  include ggml headers, and does not copy `../ds4`. Production prompt GDN on
  `GdnScanPath::kFusedTokenLoop` with fuse path `off` installs
  `prepare_gdn_shared_inverses` plus
  `prepare_recurrence_fused_warp_column_shared` when dispatch predicates hold;
  `prepare_recurrence_fused_warp_column` remains the A/B baseline and
  `token_count == 1` tails stay `repeated`. Decode sequential GDN,
  `kSelectedGdnFusePath` `off`, and sequential windows stay. No extra persistent
  `cudaMalloc`; the workspace byte formula is unchanged. Keep denominators are
  the frozen then-current accepted P and D128/D2048 means and p95s copied into
  the contract. **Measured, RTX 5090:** 4096 complete-GDN A/B winner `shared`;
  live exclusive sitting keep; production pin `shared`; `reverted` false;
  `keep_sitting_skipped` false; `status` measured. Live tok/s stay in the
  report; this ledger does not replace them. The schema-1 contract, measured
  fixture, and report are
  [`pins/opt040_gdn_shared_inverse_contract.json`](../pins/opt040_gdn_shared_inverse_contract.json),
  [`fixtures/opt040_gdn_shared_inverse.json`](../fixtures/opt040_gdn_shared_inverse.json),
  and
  [`evidence/optimization/opt040-gdn-shared-inverse/REPORT.md`](../evidence/optimization/opt040-gdn-shared-inverse/REPORT.md).
  The beginner explanations are
  [`docs/42-cuda-gdn-chunks.md`](42-cuda-gdn-chunks.md),
  [`docs/41-cuda-gdn-step.md`](41-cuda-gdn-step.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: byte-equal quality outputs/state; frozen sequential GDN gates;
  lower complete GDN component time; improved P; cross-workload guard; does not
  substitute for the 2K llama.cpp parity gate; Quartz ≥ llama.cpp is not this
  gate. Envelopes unloosened; Nsight is not used.
- OPT-052 is a local derivation over the production fused token-loop GDN scan
  and the measured OPT-040 shared-inverse keep. The fused warp-column loop
  still multiplied every value-column Q/K by the hoisted inverses and called
  `expf(log_decay)` per token; this increment A/B's a preprocessing sibling
  that writes L2-scaled Q/K once per `(token, key_head)` and decay once per
  `(token, value_head)` into the existing `prompt_projected_bf16_` overlay,
  then runs the same register recurrence. V stays on original convolved
  columns. FMA and approximate exp are measured as separate variants. An
  optional conversion-inclusive column-major state tile is also measured; the
  canonical session layout is unchanged. Sequential 64-token windows, decode
  GDN, and fuse `off` stay. **Measured, RTX 5090:** 4096 complete-GDN A/B
  winner `transpose`; live exclusive sitting keep; production pin `transpose`;
  `reverted` false; `keep_sitting_skipped` false; `status` measured. Live
  tok/s stay in the report; this ledger does not replace them. The schema-1
  contract, measured fixture, and report are
  [`pins/opt052_gdn_arithmetic_contract.json`](../pins/opt052_gdn_arithmetic_contract.json),
  [`fixtures/opt052_gdn_arithmetic.json`](../fixtures/opt052_gdn_arithmetic.json),
  and
  [`evidence/optimization/opt052-gdn-arithmetic/REPORT.md`](../evidence/optimization/opt052-gdn-arithmetic/REPORT.md).
  The beginner explanations are
  [`docs/42-cuda-gdn-chunks.md`](42-cuda-gdn-chunks.md),
  [`docs/41-cuda-gdn-step.md`](41-cuda-gdn-step.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: OPT-044 production-numerics budgets; like-arithmetic shared
  control remains OPT-040 hoisted inverses; conv+preprocessing+recurrence+gated
  complete cost; hoisted scaled Q/K and decay; admitted FMA and approximate
  exp as separate variants; optional conversion-inclusive state transpose;
  nonzero incoming state and outer-chunk restore; same-path
  transaction/restore exactness; 95% throughput floors versus OPT-051 keep;
  105% p95 ceilings versus OPT-051; does not substitute for the 2K llama.cpp
  parity gate. Envelopes unloosened; Nsight is not used.
- OPT-053 is a local derivation over production quality MMA
  (`quality_mma_process_tile`). Integer fragments stay. Explicit `__fmaf_rn`
  is used only in dequant-scale accumulation. Packed Y is already Q8_1-style
  ints; a two-stage `cp.async.cg.shared.global` 16-byte loader overlaps the
  next Y tile with MMA on admitted full tiles. SM120 encodes that PTX as
  `LDGSTS`. Tails and misalignment stay synchronous. Q4_K min corrections and
  Q6_K subscales stay. Stream-K stays `off`. FFN tiles stay I=128/J=128.
  Extra dynamic shared for the second Y tile is 18432 bytes at J=128.
  Occupancy remains 1. File-level provenance for FMA accumulation and Ampere
  `cp.async` tile loads is External: pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `mmq-vec-dot.cuh` / `mmq-load-tiles.cuh` techniques, not a wholesale ggml
  stub import. This increment does not vendor those headers and does not
  copy `../ds4/cuda/mmq/`. Keep denominators are the frozen prior keep P and
  D128/D2048 means and p95s copied into the contract. **Measured, RTX 5090:**
  4096 complete-FFN A/B winner `fma_async`; live exclusive sitting keep;
  production pin `fma_async`; `reverted` false; `keep_sitting_skipped` false;
  `status` measured. Live tok/s stay in the report; this ledger does not
  replace them. The schema-1 contract, measured fixture, and report are
  [`pins/opt053_mmq_pipeline_contract.json`](../pins/opt053_mmq_pipeline_contract.json),
  [`fixtures/opt053_mmq_pipeline.json`](../fixtures/opt053_mmq_pipeline.json),
  and
  [`evidence/optimization/opt053-mmq-pipeline/REPORT.md`](../evidence/optimization/opt053-mmq-pipeline/REPORT.md).
  The beginner explanations are
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: production-numerics budgets; like-arithmetic off control
  remains production quality MMA I=128/J=128; complete FFN staging plus
  gate/up/SwiGLU/down cost; explicit FMA scale accumulation and two-stage
  packed-Y cp.async; Q4_K min corrections and Q6_K subscales unchanged;
  synchronous fallback for tails and misalignment; no unchanged D2R or
  stream-K rerun; 95% throughput floors versus the prior keep; 105% p95
  ceilings versus the prior keep; does not substitute for the 2K llama.cpp
  parity gate. Envelopes unloosened; Nsight is not used.
- OPT-054 keeps the public 4096-token prompt transaction atomic. Internal
  physical batches of 512/1024/2048/4096 run all 64 layers per chronological
  microbatch. Candidate KV uses an outer staging range and
  `split_candidate_origin` so `start_position` is not the committed/uncommitted
  classifier. GDN convolution/recurrent state is carried in a private buffer
  and never swapped into the session until the outer publish. Graphs-off A/B
  isolates batch size; graphs-on 4096 is the shipping control. Keep 4096
  unless a smaller size wins complete cold P without regressing D. OPT-044
  admits cross-size arithmetic drift; isolation/restore/frontier stay exact.
  Keep denominators are the frozen OPT-053 P and D128/D2048 means and p95s.
  **Measured, RTX 5090 keep 4096:** graphs-off 512 2476.38477 tok/s, 1024
  2697.28589, 2048 2822.62891, 4096 2842.41333; graphs-on 4096 2840.70215.
  No smaller size beat 4096 eager or graphs-on shipping. Copied P 2895.42773,
  D128 37.5605927, D2048 35.7286987. Production pin
  `kSelectedPromptMicrobatchRows` remains 4096. The schema-1 contract, measured fixture, and report are
  [`pins/opt054_prefill_microbatch_contract.json`](../pins/opt054_prefill_microbatch_contract.json),
  [`fixtures/opt054_prefill_microbatch.json`](../fixtures/opt054_prefill_microbatch.json),
  and
  [`evidence/optimization/opt054-prefill-microbatch/REPORT.md`](../evidence/optimization/opt054-prefill-microbatch/REPORT.md).
  The beginner explanations are
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md) and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: atomic 4096-token transaction; internal 512/1024/2048/4096
  microbatches; no early commit; graphs-off isolation then graphs-on shipping;
  keep 4096 unless a smaller size wins complete P; D128/D2048 95% floors and
  p95 inside 105% versus OPT-053; OPT-044 admits cross-size arithmetic drift;
  does not substitute for the 2K llama.cpp parity gate. Envelopes unloosened;
  Nsight is not used.
- OPT-055 revisits remaining decode and prompt launch gaps after OPT-054.
  `SchedulerGraphs` gains node counts, a host `GraphLaunchParams` block, and
  empty layer-segment / prompt-mixer slots. Mixer/GDN/attention stay ordinary
  launches. Keep FFN-only unless remaining idle exceeds 0.5 ms decode / 20 ms
  prefill or 1% of wall on both a null poll and a non-null Session poll.
  Graph/eager equality, token/frontier changes, invalidation, partial tails,
  and cancellation before publication stay. Extra device allocation is zero;
  the measured 128-graph 128K reserve is unchanged. Keep denominators are the
  frozen OPT-054 P and D128/D2048 means and p95s.
  **Measured, RTX 5090 no-change:** remaining idle below noise; production pin
  `kSelectedExecutionGraphPath` remains `ffn_only`. Copied P 2895.42773, D128
  37.5605927, D2048 35.7286987. The schema-1 contract, measured fixture, and
  report are
  [`pins/opt055_execution_graphs_contract.json`](../pins/opt055_execution_graphs_contract.json),
  [`fixtures/opt055_execution_graphs.json`](../fixtures/opt055_execution_graphs.json),
  and
  [`evidence/optimization/opt055-execution-graphs/REPORT.md`](../evidence/optimization/opt055-execution-graphs/REPORT.md).
  The beginner explanations are
  [`docs/53-stable-address-cuda-graphs.md`](53-stable-address-cuda-graphs.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: stable-address decode and prompt FFN graphs remain the shipping
  path; layer-segment mixer/core capture stays unpopulated unless idle exceeds
  noise; graph/eager equality on the same arithmetic path; token change,
  frontier growth, invalidation, partial tails, and cancellation before
  publication; node and launch counts plus measured idle/waits including poll
  versus null; parameter uploads counted in host launch_params without extra
  device allocation; 128K post-graph reserve unchanged; does not substitute
  for the 2K llama.cpp parity gate. Envelopes unloosened; Nsight is not used.
- OPT-056 is the end-to-end outcome gate after OPT-045–055. Same-sitting
  exclusive RTX 5090 P/D128/D2048 versus pinned llama.cpp must exceed llama by
  at least 5% with confidence-supported means and decode p95 no worse;
  combined OPT-044 production-optimization quality on selected paths; original
  OPT-016 2K parity evidence. Candidate-task completion is insufficient.
  QLT-001 remains its own owner. Session TTFT does not replace the historical
  OPT-021/OPT-032 protocol. Measured sitting 2026-09-11T01:50:43Z is unpassed:
  P 2808.49609 vs 3263.516321, D128 37.4816246 vs 68.9318767, D2048 35.7208481
  vs 67.3394327 tok/s; decode p95 worse; greedy tasks failed; 2K 3012.69507 vs
  3169.571249. The schema-1 contract, fixture, and report are
  [`pins/opt056_performance_gate_contract.json`](../pins/opt056_performance_gate_contract.json),
  [`fixtures/opt056_performance_gate.json`](../fixtures/opt056_performance_gate.json),
  and
  [`evidence/optimization/opt056-performance-gate/REPORT.md`](../evidence/optimization/opt056-performance-gate/REPORT.md).
  Proof limit: same-sitting P/D128/D2048 versus pinned llama.cpp; at least 5%
  throughput margin; decode p95 no worse than llama; confidence-supported
  improvement; combined production quality on selected paths; original
  OPT-016 2K parity evidence; candidate-task completion alone is insufficient;
  does not redefine the 2K llama.cpp parity gate; QLT-001 remains its own
  owner; Session TTFT does not replace the historical workload protocol.
- OPT-041 is a local derivation over admitted Ada+ stream-K fattn with
  register-resident VKQ and dual-F16 probability×V MMA. It assigns each of
  eight 16×8 QK microtiles to one of four warps (`tile_id % 4`), lets the
  owning warp visit K in the current virtual-warp partial order, folds in the
  current FP32 order, and uses one block barrier after the microtile loop
  instead of shared `cparts` reduction with two barriers per microtile. The
  `cparts` shared slab stays allocated so occupancy and shared bytes are not
  mixed with ownership. File-level provenance for warp-owned KQ MMA fragments
  is already External: pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `fattn-mma-f16.cuh` `KQ_C` with `np` parallel warps per Q column. This
  increment does not vendor `fattn-mma-f16.cuh`, does not include ggml
  headers, and does not copy `../ds4`. Production prompt fattn installs
  `WarpQK=true` stream-K when dispatch predicates hold;
  `launch_attention_prepare_chunk_stream_k_qk` with `cparts` remains the A/B
  baseline. Softmax, rescale, P×V MMA, register VKQ, meta, and the stream-K
  combine kernel stay. Decode `warp_query` attention stays. No extra persistent
  `cudaMalloc`; the workspace byte formula is unchanged. Keep denominators are
  the frozen then-current accepted P and D128/D2048 means and p95s copied into
  the contract. **Measured, RTX 5090:** 4096 complete-attention A/B winner
  `warp_microtile`; live exclusive sitting keep; production pin
  `warp_microtile`; `reverted` false; `keep_sitting_skipped` false; `status`
  measured. Live tok/s stay in the report; this ledger does not replace them.
  The schema-1 contract, measured fixture, and report are
  [`pins/opt041_fattn_warp_qk_contract.json`](../pins/opt041_fattn_warp_qk_contract.json),
  [`fixtures/opt041_fattn_warp_qk.json`](../fixtures/opt041_fattn_warp_qk.json),
  and
  [`evidence/optimization/opt041-fattn-warp-qk/REPORT.md`](../evidence/optimization/opt041-fattn-warp-qk/REPORT.md).
  The beginner explanations are
  [`docs/44-cuda-attention-prefill.md`](44-cuda-attention-prefill.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: byte-equal quality attention outputs; frozen attention gates;
  lower complete component time; improved P; cross-workload guard; does not
  substitute for the 2K llama.cpp parity gate; Quartz ≥ llama.cpp is not this
  gate. Envelopes unloosened; Nsight is not used.
- OPT-042 adapts the pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `vecdotq.cuh` Q4_K `__dp4a` integer-dot technique (`vec_dot_q4_K_q8_1`)
  against Quartz FP32-scale `Q8Block` staging, not `block_q8_1`. This increment
  does not vendor `vecdotq.cuh` or `mmvq.cu`, does not include ggml headers,
  and does not add `integer` to `legal_mmv_load_path`. The integer-dot kernel
  lives only in `cuda/opt042_mmv_integer_study.cu`; production
  `launch_quant_mmv` and `kSelectedMmvLoadPath = packed` stay. **Measured
  diagnostic, RTX 5090:** paired complete and prequant A/B on CUD-001 probes,
  synthetic production FFN gate/up and down shapes, and real layer-0/3/63
  gate/up/down; synthetic weighted complete packed **0.0666723549** ms versus
  integer **0.0607566237** ms; real weighted complete packed **0.0675881952**
  ms versus integer **0.061906416** ms; `admissibility` `numeric_reject`
  because synthetic production shapes miss the frozen CUD-001 envelope for both
  candidates; `promote_to_production_ab` false; `claims_performance_improvement`
  false; `selected_mmv_load_path` `packed`; `staging_equal` true. Live numbers
  stay in the report; this ledger does not replace them. The schema-1 contract,
  measured fixture, and report are
  [`pins/opt042_mmv_integer_study_contract.json`](../pins/opt042_mmv_integer_study_contract.json),
  [`fixtures/opt042_mmv_integer_study.json`](../fixtures/opt042_mmv_integer_study.json),
  and
  [`evidence/optimization/opt042-mmv-integer-study/REPORT.md`](../evidence/optimization/opt042-mmv-integer-study/REPORT.md).
  The beginner explanations are
  [`docs/39-cuda-quant-mmv.md`](39-cuda-quant-mmv.md) and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: claims no performance improvement; fixed CUD-001 envelope;
  unchanged FP32-scale Q8 staging; production dispatch unchanged; accepted keep
  denominators remain unchanged; does not substitute for the 2K llama.cpp parity
  gate; integer-dot is diagnostic only. Envelopes unloosened; Nsight is not used.
- OPT-044 does not vendor llama.cpp or ds4. It independently reproduces the
  OPT-042 synthetic identities, measures Quartz Q8 and pinned llama Q8_1 against
  FP64 dequantized weights, and freezes production family/shape ceilings plus a
  1.01 production-optimization quality suite. Production still selects
  `kSelectedProductionNumericsPath = strict`; no unvalidated fast kernel is
  installed. The contract, fixture, host diagnostic, and report are
  [`pins/production_numerics_contract.json`](../pins/production_numerics_contract.json),
  [`fixtures/opt044_production_numerics.json`](../fixtures/opt044_production_numerics.json),
  [`src/opt044_production_numerics.cpp`](../src/opt044_production_numerics.cpp),
  and
  [`evidence/optimization/opt044-production-numerics/REPORT.md`](../evidence/optimization/opt044-production-numerics/REPORT.md).
  Beginner explanations are [`docs/04-numerics.md`](04-numerics.md) and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: claims no performance improvement; strict reference contracts
  retained; OPT-042 numeric_reject unchanged; structural/session exactness is
  not an accuracy compromise.
- OPT-029 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  warp-column GDN recurrence from `gated_delta_net.cu` (`S_v=128`,
  grid `(H, n_seqs, S_v/4)`, block `(32, 4)`, register `s_shard`).
  Quartz-owned causal conv and gated-output collapse into that token
  loop; llama.cpp keeps causal conv as a separate op (chunked-kernel
  TODO). This increment does not vendor `gated_delta_net.cu`, does not
  copy `../ds4`, and does not include ggml headers. Production prompt
  GDN remains split parallel conv + warp-column + `gdn_gated_output_rows`
  after the paired A/B loss. Fused kernels remain as non-production
  symbols. `kSelectedGdnFusePath` is `off`. `GdnScanPath::kFusedTokenLoop`
  stays the production scan enum. Mixer Q8 quality, skinny
  `mma_i32_j128`, FFN `shared_y_swiglu_q8`, and fattn Ada+ stream-K stay.
  Decode GDN stays tiled prepare + `launch_gdn_gated_output`. Sequential
  windows remain the unloosened GDN-002 reference. **Measured, RTX 5090:**
  A/B winner `off` (`win=false`; no fuse id strictly beat split conv +
  warp-column + gated-output); live exclusive cold exact-4096 reject
  versus the frozen successor-oracle baseline 1746.71973; `reverted`
  true; `successor_oracle` false; `production_gdn_fuse_installed` false;
  `ladder_exhausted` false. Live numbers stay in the report; this ledger
  does not replace them. The schema-1 contract, rejected fixture, report,
  and rejection are
  [`pins/opt029_gdn_fuse_contract.json`](../pins/opt029_gdn_fuse_contract.json),
  [`fixtures/opt029_gdn_fuse.json`](../fixtures/opt029_gdn_fuse.json),
  [`evidence/optimization/opt029-gdn-fuse/REPORT.md`](../evidence/optimization/opt029-gdn-fuse/REPORT.md),
  and
  [`evidence/optimization/opt029-gdn-fuse/REJECTION.md`](../evidence/optimization/opt029-gdn-fuse/REJECTION.md).
  The beginner explanation is
  [`docs/42-cuda-gdn-chunks.md`](42-cuda-gdn-chunks.md).
  Proof limit: GDN conv and/or gated-output collapsed into the
  warp-column token loop under unloosened GDN-002 envelopes when a
  paired A/B wins; 4K keep/reject versus the then-current oracle
  baseline; envelopes unloosened; sequential remains the reference;
  OPT-016 remains the parity gate owner; does not substitute for the 2K
  llama.cpp parity gate; Quartz ≥ llama.cpp is not this gate.
- OPT-030 adapts llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  Hopper/Blackwell programmatic dependent launch from
  `ggml/src/ggml-cuda/common.cuh` (`ggml_cuda_kernel_launch` /
  `ggml_cuda_pdl_config` / `ggml_cuda_kernel_can_use_pdl` /
  `ggml_cuda_pdl_sync` / `ggml_cuda_pdl_lc`): host
  `cudaLaunchKernelEx` with one
  `cudaLaunchAttributeProgrammaticStreamSerialization` attribute
  (`programmaticStreamSerializationAllowed = 1`) when `ptxVersion >=
  90`, plus device `cudaGridDependencySynchronize` /
  `cudaTriggerProgrammaticLaunchCompletion` for `__CUDA_ARCH__ >=
  900`. This increment does not vendor `common.cuh`, does not include
  ggml headers, and does not honor `GGML_CUDA_PDL`. Quartz uses a
  compile-time enumerator plus a paired A/B. PDL is forbidden while a
  stream is capturing; FFN graphs stay ordinary kernel nodes.
  Production stays ordinary `<<<>>>` after the 4K reject.
  `kSelectedPdlPath` is `off`. Mixer Q8 quality, skinny
  `mma_i32_j128`, FFN `shared_y_swiglu_q8`, fattn Ada+ stream-K, and
  split GDN core stay. Decode launches are unchanged. **Measured, RTX
  5090:** A/B winner `pdl` (`win=true`; `pdl` strictly beat `off`;
  `pdl_host` did not); live exclusive cold exact-4096 reject versus
  the frozen successor-oracle baseline 1746.71973; `reverted` true;
  `successor_oracle` false; `production_pdl_installed` false;
  `ladder_exhausted` false. Live numbers stay in the report; this
  ledger does not replace them. The schema-1 contract, rejected
  fixture, report, and rejection are
  [`pins/opt030_pdl_launches_contract.json`](../pins/opt030_pdl_launches_contract.json),
  [`fixtures/opt030_pdl_launches.json`](../fixtures/opt030_pdl_launches.json),
  [`evidence/optimization/opt030-pdl-launches/REPORT.md`](../evidence/optimization/opt030-pdl-launches/REPORT.md),
  and
  [`evidence/optimization/opt030-pdl-launches/REJECTION.md`](../evidence/optimization/opt030-pdl-launches/REJECTION.md).
  The beginner explanation is
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  Proof limit: Hopper/Blackwell PDL serialization of ungraphed
  mixer/GDN/attention prompt launches on sm_120 when a paired A/B
  wins; arithmetic and graph-vs-fused byte equality retained; FFN
  graphs stay non-PDL; 4K keep/reject versus the then-current oracle
  baseline; envelopes unloosened; OPT-016 remains the parity gate
  owner; does not substitute for the 2K llama.cpp parity gate; Quartz
  ≥ llama.cpp is not this gate.
- OPT-032 introduces no new external implementation source. Pinned llama.cpp
  revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` is already the External
  same-GGUF authority (`llama-bench` for P; public `llama.h` /
  `llama_time_us()` for D128/D2048). This increment freezes a local exclusive-
  RTX-5090 P/D128/D2048 oracle sitting: exact-4096 production `sync_tokens`
  with attribution null and graphs created, plus prefix-128 and prefix-2048
  decode (256 predetermined tokens, 3 warm + 30 measured) matched by
  `qw38-llama-decode-oracle`. Random-token `llama-bench` decode is
  informational. Exclusive decode categories live on opt-in
  `DecodeAttribution`; public `RuntimeTimings` stay composite. The schema-1
  contract, retained live fixture, and claim-labeled report are
  [`pins/opt032_decode_oracle_contract.json`](../pins/opt032_decode_oracle_contract.json),
  [`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json),
  and
  [`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md).
  Live tok/s stay in that report; this ledger does not replace them. The
  beginner explanations are
  [`docs/61-benchmark-harness.md`](61-benchmark-harness.md),
  [`docs/51-runtime-timing-and-nvtx.md`](51-runtime-timing-and-nvtx.md),
  [`docs/06-system-optimization.md`](06-system-optimization.md), and
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md).
  Proof limit: D128 and D2048 oracles; exclusive decode categories; matched
  pinned llama.cpp; recorded next-task order; claims no performance
  improvement; envelopes unloosened; does not substitute for the 2K llama.cpp
  parity gate; llama-bench random decode is informational. Quartz ≥ llama.cpp
  is not this gate. No kernel implementation was copied or adapted.
- OPT-033 is a local derivation over admitted Ada+ stream-K fattn. It holds
  the running value sum in per-thread registers across 32-row KV tiles
  instead of load/rescale/store through global `vkq` on every tile.
  File-level provenance for register-held `VKQ_C` is already External:
  pinned llama.cpp revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT,
  The ggml authors) `fattn-mma-f16.cuh`. This increment does not vendor
  that file, does not include ggml headers, and does not copy `../ds4`.
  MMA probability×V is out of scope. Production prompt attention keeps
  ncols1=16, ncols2=2, KV tile 32, dual-F16 Q, and `grid.z=2`. Scalar KV
  iteration and the stream-K combine stay. Score scratch stays untouched;
  no extra `cudaMalloc`. Combine buffers continue to alias existing
  `prompt_projected_bf16_` / `prompt_q8_`. Decode attention stays on its
  partitioned one-token path. Keep denominators are the frozen then-current
  accepted P / D128 / D2048 means and p95s copied into the contract, not
  historical 4K successor-oracle 1746.71973 and not stale sitting
  1637.58594. **Measured, RTX 5090:** A/B winner `registers`; production
  pin `registers`; `reverted` false; `keep_sitting_skipped` false;
  `status` measured. Live tok/s stay in the report; this ledger does not
  replace them. The schema-1 contract, measured fixture, and report are
  [`pins/opt033_register_vkq_contract.json`](../pins/opt033_register_vkq_contract.json),
  [`fixtures/opt033_register_vkq.json`](../fixtures/opt033_register_vkq.json),
  and
  [`evidence/optimization/opt033-register-vkq/REPORT.md`](../evidence/optimization/opt033-register-vkq/REPORT.md).
  The beginner explanations are
  [`docs/44-cuda-attention-prefill.md`](44-cuda-attention-prefill.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: frozen attention envelopes; byte-equal output; lower
  component time; improved P; cross-workload guard; then-current accepted
  P D128 D2048 are the keep denominators; does not substitute for the 2K
  llama.cpp parity gate; Quartz ≥ llama.cpp is not this gate. Envelopes
  unloosened; Nsight is not used.
- OPT-035 is a local derivation over admitted Ada+ stream-K fattn with
  register-resident value sums. It replaces the scalar register-resident
  probability×V inner product with dual-F16 probability × F16 V MMA into
  FP32 C while keeping FP32 online max/denominator. File-level provenance
  for VKQ MMA is already External: pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The ggml authors)
  `fattn-mma-f16.cuh`. This increment does not vendor that file, does not
  include ggml headers, and does not copy `../ds4`. Dual-F16 V is out of
  scope. Production prompt attention keeps ncols1=16, ncols2=2, KV tile 32,
  dual-F16 Q, `grid.z=2`, and register VKQ. The stream-K combine stays.
  Score scratch stays untouched; no extra `cudaMalloc`. Combine buffers
  continue to alias existing `prompt_projected_bf16_` / `prompt_q8_`. Decode
  attention stays on its partitioned one-token path. Keep denominators are
  the frozen then-current accepted P / D128 / D2048 means and p95s copied
  into the contract (P 1745.10315, D128 15.1528101, D2048 13.5596962), not
  historical 4K successor-oracle 1746.71973, not stale sitting 1637.58594,
  and not 1644.04822. **Measured, RTX 5090:** A/B winner `mma`; production
  pin `mma`; `reverted` false; `keep_sitting_skipped` false; `status`
  measured. Live tok/s stay in the report; this ledger does not replace
  them. The schema-1 contract, measured fixture, and report are
  [`pins/opt035_pv_mma_contract.json`](../pins/opt035_pv_mma_contract.json),
  [`fixtures/opt035_pv_mma.json`](../fixtures/opt035_pv_mma.json),
  and
  [`evidence/optimization/opt035-pv-mma/REPORT.md`](../evidence/optimization/opt035-pv-mma/REPORT.md).
  The beginner explanations are
  [`docs/44-cuda-attention-prefill.md`](44-cuda-attention-prefill.md),
  [`docs/62-cuda-full-prefill.md`](62-cuda-full-prefill.md), and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: frozen attention envelopes; lower component time; improved
  P; cross-workload guard; then-current accepted P D128 D2048 are the keep
  denominators; does not substitute for the 2K llama.cpp parity gate;
  Quartz ≥ llama.cpp is not this gate. Byte equality versus scalar is not
  this keep predicate. Envelopes unloosened; Nsight is not used.
- OPT-034 is a local derivation over admitted CUD-001 Q4_K/Q6_K CUDA
  vector-dot decoding. It loads packed fields and scales once per
  256-weight block, then consumes each lane's eight values in the existing
  column order with the existing FP32 products and five-step warp tree.
  File-level provenance for packed-field loads is already External: pinned
  llama.cpp revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT, The
  ggml authors) `vecdotq.cuh`. This increment does not vendor that file,
  does not include ggml headers, and does not copy `../ds4`. Integer-dot
  reassociation is out of scope. Q8_0 MMV stays on the per-column decoder.
  Transient Q8 staging and warp-count dispatch stay. Production decode MMV
  stays behind `launch_quant_mmv`. No extra `cudaMalloc`. Keep denominators
  are the frozen then-current accepted P / D128 / D2048 means and p95s
  copied into the contract (P 1865.21155, D128 15.0562878, D2048
  13.5411425), not historical 4K successor-oracle 1746.71973, not stale
  sitting 1637.58594, not 1745.10315, and not 1644.04822. **Measured, RTX
  5090:** A/B winner `packed`; production pin `packed`; `reverted` false;
  `keep_sitting_skipped` false; `status` measured. Live tok/s stay in the
  report; this ledger does not replace them. The schema-1 contract,
  measured fixture, and report are
  [`pins/opt034_packed_mmv_contract.json`](../pins/opt034_packed_mmv_contract.json),
  [`fixtures/opt034_packed_mmv.json`](../fixtures/opt034_packed_mmv.json),
  and
  [`evidence/optimization/opt034-packed-mmv/REPORT.md`](../evidence/optimization/opt034-packed-mmv/REPORT.md).
  The beginner explanations are
  [`docs/39-cuda-quant-mmv.md`](39-cuda-quant-mmv.md) and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: byte equality; lower weighted MMV time; improved D2048;
  cross-workload guard; then-current accepted P D128 D2048 are the keep
  denominators; does not substitute for the 2K llama.cpp parity gate;
  Quartz ≥ llama.cpp is not this gate. Envelopes unloosened; Nsight is not
  used.
- OPT-036 is a local derivation over admitted ATN-001 tiled one-token decode.
  It splits the legal KV range into contiguous partitions `{1, 4, 8, 16}`,
  keeps candidate `1` as today's tiled kernel, and merges FP32 partial max /
  denominator / numerator in deterministic ascending-part order. File-level
  provenance for the KV-split / combine technique is already External:
  pinned llama.cpp revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` (MIT,
  The ggml authors) `fattn-vec.cuh` and `fattn-common.cuh`
  (`flash_attn_combine_results`). This increment does not vendor those files,
  does not include ggml headers, and does not copy `../ds4`. Production
  decode attention installs independent compile-time pins below 2048 and at
  or above 2048. Prefill fattn and tiled prefill for `token_count >= 2` stay.
  Partials alias idle `prompt_projected_bf16_` / `prompt_q8_`; score scratch
  stays unused; no extra `cudaMalloc`. Keep denominators are the frozen
  decode-oracle P / D128 / D2048 means and p95s copied into the contract, not
  historical 4K successor-oracle 1746.71973. **Measured, RTX 5090:** both
  regimes selected 16; production pins 16/16; `reverted` false;
  `keep_sitting_skipped` false; `status` measured. Live tok/s stay in the
  report; this ledger does not replace them. The schema-1 contract, measured
  fixture, and report are
  [`pins/opt036_decode_kv_partition_contract.json`](../pins/opt036_decode_kv_partition_contract.json),
  [`fixtures/opt036_decode_kv_partition.json`](../fixtures/opt036_decode_kv_partition.json),
  and
  [`evidence/optimization/opt036-decode-kv-partition/REPORT.md`](../evidence/optimization/opt036-decode-kv-partition/REPORT.md).
  The beginner explanations are
  [`docs/43-cuda-attention-decode.md`](43-cuda-attention-decode.md) and
  [`docs/06-system-optimization.md`](06-system-optimization.md).
  Proof limit: frozen attention envelopes; lower component time; improved
  D2048; cross-workload guard; decode-oracle P D128 D2048 are the keep
  denominators; does not substitute for the 2K llama.cpp parity gate; Quartz
  ≥ llama.cpp is not this gate. Envelopes unloosened; Nsight is not used.
- CUD-003 introduces no new external implementation source. GGUF Q8_0 decoding
  follows the format already admitted by the pinned scalar decoder, and the
  pointwise/layout equations come from the pinned model contract and scalar
  layers. [`pins/cuda_scheduler_primitives_contract.json`](../pins/cuda_scheduler_primitives_contract.json)
  authenticates the narrow CUDA boundary; Q8_0, BF16 pointwise, embedding, and
  tiled/grouped measurements are retained in
  [`fixtures/cuda_scheduler_primitives.json`](../fixtures/cuda_scheduler_primitives.json).
- SCH-001 introduces no new external implementation source. It composes the
  already admitted local CUDA primitives under the pinned typed model and
  scalar schedule, and checks its complete logits against the pinned scalar and
  llama.cpp evidence. [`pins/cuda_scheduler_contract.json`](../pins/cuda_scheduler_contract.json)
  authenticates the source/model/authority boundary; selected taps, complete
  logit metrics, timings, allocations, and negative results are retained in
  [`fixtures/cuda_full_scheduler.json`](../fixtures/cuda_full_scheduler.json).
- SES-001 introduces no new external implementation source. It composes the
  admitted local CUDA scheduler with an exact host token history, reset/replay
  rule, and device byte-comparison diagnostic. The authenticated ownership and
  reuse boundary is frozen in
  [`pins/cuda_prefix_sync_contract.json`](../pins/cuda_prefix_sync_contract.json),
  while append/no-op reuse, divergent/shorter replay, and invalid-input
  preservation are retained in
  [`fixtures/cuda_prefix_sync.json`](../fixtures/cuda_prefix_sync.json).
- SES-002 introduces no new external implementation source. It changes the
  local CUDA scheduler's ownership and publication protocol without changing
  model arithmetic. The all-layer candidate, status-poll, frontier-last commit,
  and read-only greedy boundary are authenticated in
  [`pins/cuda_atomic_eval_contract.json`](../pins/cuda_atomic_eval_contract.json);
  cancellation, injected-error, exact-commit, and sampling-purity evidence is
  retained in [`fixtures/cuda_atomic_eval.json`](../fixtures/cuda_atomic_eval.json).
- SES-003 introduces no new external implementation source. Its little-endian
  framing, fixed layout identity, bounded SHA-256 prefix helper, logical-state
  sections, Linux `fsync`/rename publication, and validation order are local
  Quartz contracts authenticated in
  [`pins/cuda_checkpoint_contract.json`](../pins/cuda_checkpoint_contract.json).
  Exact round-trip/continuation and retained corrupt/incompatible failures are
  recorded in [`fixtures/cuda_checkpoint.json`](../fixtures/cuda_checkpoint.json).
- MEM-001's pre-graph increment introduces no external implementation source.
  It exercises existing local allocation owners and reconciles their exact byte
  counters with CUDA's runtime `cudaMemGetInfo` readings. The provisional
  contract and explicit OPT-003 dependency are authenticated in
  [`pins/cuda_memory_fit_contract.json`](../pins/cuda_memory_fit_contract.json),
  with raw owner, allocator-delta, host-RSS, and reserve values retained in
  [`fixtures/cuda_memory_fit_pre_graph.json`](../fixtures/cuda_memory_fit_pre_graph.json).
- OPT-001 uses NVIDIA's CUDA event and NVTX v3 interfaces supplied by the pinned
  CUDA 13.0.2 image; no external implementation code is copied. The local
  category/availability boundary and source identities are authenticated in
  [`pins/cuda_timing_contract.json`](../pins/cuda_timing_contract.json). One
  RTX 5090 attribution sample plus the retained unavailable-Nsight-Systems and
  denied-Nsight-Compute-counter results are recorded in
  [`fixtures/cuda_timing.json`](../fixtures/cuda_timing.json).
- OPT-002 uses Nsight Compute 2025.3.1 from the pinned CUDA image to profile
  local kernels; no external implementation code is copied. NVIDIA profiler
  metrics select the candidate, while Quartz's retained unfused path and paired
  CUDA-event samples decide admission. The source/fusion boundary is
  authenticated in [`pins/cuda_fusion_contract.json`](../pins/cuda_fusion_contract.json);
  raw A/B samples, profiler metrics, the balanced-MMV rejection, and the slower
  serialized-fusion negative are retained in
  [`fixtures/cuda_fusion.json`](../fixtures/cuda_fusion.json).
- OPT-003 uses CUDA 13.0.2 stream-capture, graph instantiate/upload, and graph
  launch APIs; no external implementation code is copied. The stable-address
  FFN boundary and source identities are authenticated in
  [`pins/cuda_graph_contract.json`](../pins/cuda_graph_contract.json), with raw
  paired replay samples, graph allocation, launch attribution, and exact state
  evidence retained in [`fixtures/cuda_graph.json`](../fixtures/cuda_graph.json).
- MEM-001's final increment reuses the local allocation arithmetic from its
  provisional gate and adds the live OPT-003 graph owner. OPT-012 later
  remeasures that same post-graph fixture after uploading 64 decode plus 64
  prompt executables. The live simultaneous owner/free-memory/RSS readings are
  retained in
  [`fixtures/cuda_memory_fit_post_graph.json`](../fixtures/cuda_memory_fit_post_graph.json);
  the earlier pre-graph fixture remains historical evidence rather than being
  overwritten.
- OPT-004 uses no copied implementation. Quartz specializes its existing MMV
  and MMQ kernels with compile-time launch candidates, then selects the checked-in
  SM120 dispatch table from the local RTX 5090 sweep. Source identities and the
  selection rule are authenticated in
  [`pins/cuda_dispatch_tuning_contract.json`](../pins/cuda_dispatch_tuning_contract.json);
  candidate means are retained in
  [`fixtures/cuda_dispatch_tuning.json`](../fixtures/cuda_dispatch_tuning.json)
  and individual samples in
  [`evidence/profiling/opt004-dispatch-sweep-raw.txt`](../evidence/profiling/opt004-dispatch-sweep-raw.txt).
  That fixture remains the MMV-bucket and historical ≤64-row Q4_K MMQ authority.
  OPT-009 supersedes production MMQ prompt-tile selection and does not mutate
  the OPT-004 contract or fixture.
- OPT-009 introduces no new external implementation source. It does not copy
  llama.cpp, DwarfStar, or cuBLAS GEMM. The tiled Q8_0 path keeps the
  local decode `q8_mmv_bf16` multiply, `__fadd_rn` / `__fmul_rn` column walk,
  and warp shuffle reduction, applied independently to each prompt row inside a
  CUD-002-style weight-reusing tile. Q4_K/Q6_K remain on the admitted
  `launch_quant_mmq` path with legal tiles `{1,2,4,8,16,32,64}`. Kind-specific
  SM120 winners, occupancy, Q8_0 byte equality to the retained row-wise
  reference, frozen CUD-002 envelopes, and component timing predicates are
  authenticated in
  [`pins/cuda_prompt_mmq_contract.json`](../pins/cuda_prompt_mmq_contract.json)
  and [`fixtures/cuda_prompt_mmq.json`](../fixtures/cuda_prompt_mmq.json), with
  every sweep sample in
  [`evidence/profiling/opt009-mmq-tile-sweep-raw.txt`](../evidence/profiling/opt009-mmq-tile-sweep-raw.txt).
  The proof is component-only: it is not an end-to-end speedup or 128K quality
  claim. After OPT-017, this fixture remains the tiled-versus-reference
  authority; production mixer Q8_0 MMA is a later increment. The beginner
  explanation is
  [`docs/40-cuda-prompt-mmq.md`](40-cuda-prompt-mmq.md).
- CLI-001 and EDU-041 use no copied implementation. The public CUDA runtime,
  inverse tokenizer byte map, incremental chat suffix, seeded sampler, and
  terminal loop are local Quartz code. Their ownership and command contract is
  authenticated in [`pins/cli_contract.json`](../pins/cli_contract.json), and
  the real-model generation/save/restore result is retained in
  [`fixtures/cli_smoke.json`](../fixtures/cli_smoke.json).
- MDL-003 dynamically uses OpenSSL 3's high-level
  [EVP digest interface](https://docs.openssl.org/3.0/man3/EVP_DigestInit/),
  whose implementation is supplied by the immutable CUDA base image; no
  OpenSSL code is copied. OpenSSL's
  [x86 capability documentation](https://docs.openssl.org/3.4/man3/OPENSSL_ia32cap/)
  identifies the SHA extension used by provider dispatch. The local wrapper,
  portable fallback, and measurement identity are authenticated in
  [`pins/sha256_acceleration_contract.json`](../pins/sha256_acceleration_contract.json),
  with the full-model result in
  [`fixtures/sha256_acceleration.json`](../fixtures/sha256_acceleration.json).
  The distinction between artifact identity and ZFS block integrity follows
  OpenZFS's [checksum documentation](https://openzfs.github.io/openzfs-docs/Basic%20Concepts/Data%20Storage/Checksums.html).
- SRV-001 and EDU-043 use Linux/POSIX socket, signal, and C++17 synchronization
  interfaces supplied by the pinned host/container toolchain; no HTTP server or
  queue implementation is copied. The exact parser/routes, FIFO ownership,
  cancellation, lifecycle, and source identities are authenticated in
  [`pins/server_core_contract.json`](../pins/server_core_contract.json), with
  native concurrency and real CUDA route evidence retained in
  [`fixtures/server_core.json`](../fixtures/server_core.json).
- API-002, SRV-002, and EDU-044 use an original bounded JSON grammar,
  OpenAI-shaped Chat Completions records, the Qwen tool format attributed above,
  POSIX socket polling, and the admitted Engine/Session API. The parser in
  [`src/server_json.cpp`](../src/server_json.cpp), protocol mapping in
  [`src/server_api.cpp`](../src/server_api.cpp), generation loop in
  [`src/server_generation.cpp`](../src/server_generation.cpp), and SRV-002
  additions to [`src/server.cpp`](../src/server.cpp) do not copy a JSON library
  or external serving-runtime implementation. Source identities and measured
  behavior are authenticated by
  [`pins/chat_completions_contract.json`](../pins/chat_completions_contract.json)
  and [`fixtures/chat_completions.json`](../fixtures/chat_completions.json).
- API-003, SES-004, SRV-003, and EDU-045 use original Responses-to-Chat mapping,
  exact incremental template rendering, response event serialization, and an
  atomic POSIX continuation-record implementation. No OpenAI SDK, protocol
  server, storage library, or serving-runtime code is copied. The public shape
  follows the Responses vocabulary, while the narrow supported boundary is
  authenticated in [`pins/responses_contract.json`](../pins/responses_contract.json)
  and measured behavior is retained in
  [`fixtures/responses.json`](../fixtures/responses.json).
- BEN-001 and EDU-046 use an original C++17 measurement harness around the
  admitted public Engine/Session boundary. Percentile arithmetic, result JSON,
  failure retention, cache-policy control, and POSIX atomic publication are
  local code. CUDA runtime version queries provide build/runtime identity;
  `nvidia-smi` supplies explicitly snapshot-based device telemetry. The protocol
  and proof limits are authenticated in
  [`pins/benchmark_contract.json`](../pins/benchmark_contract.json), with smoke
  evidence in [`fixtures/benchmark_harness.json`](../fixtures/benchmark_harness.json)
  and [`evidence/benchmark`](../evidence/benchmark). No benchmark-framework or
  competing-runtime implementation is copied.
- SCH-002, MEM-002, OPT-008, and EDU-047 introduce no new external implementation
  source. The full prompt path composes Quartz's already admitted MMQ, GDN scan,
  causal attention prefill, pointwise, session, and memory-ledger boundaries.
  SCH-002 retained a local Q8_0-by-BF16 prompt kernel specifically to avoid an
  extra activation requantization; that first kernel batched launches
  (`grid.y = prompt_rows`) rather than reusing weights. OPT-008 sets the
  4,096-row outer policy while retaining the internal 64-row GDN scan; its
  allocation is bounded by session capacity. OPT-009 later replaces production
  Q8_0 with weight-reusing tiles and remeasures Q4_K/Q6_K tiles through 64; that
  increment is documented separately and remains component-only. OPT-011 later
  fuses prompt embedding, residual-add-norm, all-layer scatter, and commit
  D2H/scatter overlap; that increment is also documented separately and remains
  component-only, with no Nsight Systems overlap claim. OPT-012 later captures
  4,096-row prompt FFN graphs beside the decode set; that increment is documented
  separately and remains component-only, not a whole-chunk graph. OPT-013 later
  replaces sequential prompt GDN windows with an associative scan when overlay
  scratch fits; that increment is documented separately and remains
  component-only. OPT-014 later adds opt-in live 2048-token prefill attribution
  from prompt-compute-stream CUDA events plus a host-wall remainder; that
  increment is documented separately and is instrumentation, not a throughput
  or Nsight claim. OPT-015 later cites that 2K attribution plus the scaling
  `llama-bench` 2K JSON as a host-tested recovery map; that increment is
  documented separately and is not a throughput gate or llama.cpp parity
  claim. OPT-017 later admits production mixer Q8_0 MMA behind
  `launch_q8_mmq_bf16` for `prompt_rows >= 8`, keeping the OPT-009 tiled kernel
  as the unloosened byte-exact reference; that increment is documented separately,
  is Measured component recovery, and is not the OPT-016 2K parity gate.
  OPT-020 later splits mixer-projection MMQ out of the composite GDN and
  attention buckets; that increment is documented separately and is
  instrumentation, not a throughput gate. OPT-021 later pins the exclusive
  exact-4096 keep/reject oracle against same-sitting `llama-bench` 4K; that
  increment is documented separately, is Measured protocol/baseline plus
  External `llama-bench` provenance, and is not the OPT-016 2K parity gate.
  OPT-022 later replaces production mixer Q8_0 Rank-1 MMA with quality MMA
  and shared residual Y, keeping the OPT-009 tiled kernel as the unloosened
  byte-exact reference; that increment is documented separately, is Measured
  4K keep plus External llama.cpp Q8_0 MMQ / ds4 Q8 association provenance,
  and is not the OPT-016 2K parity gate.
  OPT-023 later templates quality I=32 for mixer Q8_0 `output_rows < 128`
  when that A/B wins, keeping large mixer GEMMs on I=128 quality MMA and
  the OPT-009 tiled kernel as the unloosened byte-exact reference; that
  increment is documented separately, is Measured 4K keep plus External
  llama.cpp Ampere Q8_0 I/J and MMVQ-not-for-prefill provenance, and is
  not the OPT-016 2K parity gate.
  OPT-024 later A/Bs aligned-SoA D2R for mixer Q8_0 `output_rows >= 128`
  when that path beats quality MMA, keeping skinny α/β on I=32 quality MMA
  and the OPT-009 tiled kernel as the unloosened byte-exact reference; that
  increment is documented separately, is Measured 4K reject plus External
  ds4 kind-5 / dense Q8 D2R technique provenance (not a vendor), and is
  not the OPT-016 2K parity gate.
  Exact
  `[4096, 1]` differential, capacity fallback, cancellation, and memory
  evidence is authenticated in
  [`pins/cuda_prompt_scheduler_contract.json`](../pins/cuda_prompt_scheduler_contract.json)
  and [`fixtures/cuda_prompt_scheduler.json`](../fixtures/cuda_prompt_scheduler.json),
  with the live 128K owner/reserve measurement retained in
  [`fixtures/cuda_memory_fit_post_graph.json`](../fixtures/cuda_memory_fit_post_graph.json).
- TRC-004 introduces no new external implementation source. It composes the
  existing CUDA scheduler, backend-neutral diagnostic sink, scalar trace
  contract, and immutable three-authority tolerances. The exact five-filter
  subset, diagnostic-build isolation rule, and source identities are frozen in
  [`pins/cuda_trace_contract.json`](../pins/cuda_trace_contract.json); full
  token-42 metrics, greedy equality, and failure-path evidence are retained in
  [`fixtures/cuda_trace.json`](../fixtures/cuda_trace.json). The beginner
  explanation is [`docs/63-cuda-diagnostic-traces.md`](63-cuda-diagnostic-traces.md).
  No CUDA trace code or tolerance is copied from an external project.
- EVAL-001 introduces no new external implementation source. The typed request
  records, native command boundary, and logits evidence reader are local code;
  they reuse the public Engine/Session contract and existing trace-v1 reader.
  The shape is frozen in [`pins/eval_contract.json`](../pins/eval_contract.json),
  with retained RTX 5090 wiring and negative evidence in
  [`fixtures/eval_harness.json`](../fixtures/eval_harness.json) and focused
  tests in [`tests/test_eval.py`](../tests/test_eval.py). The beginner
  explanation is [`docs/64-eval-harness.md`](64-eval-harness.md). The fixture
  records complete harness-only logits/checkpoint/trace wiring and negative
  publication evidence; generated outputs remain uncommitted and nothing here
  is QLT-001 quality evidence.
- DOC-001 introduces no copied source or external dependency. Its human-readable
  coverage record is [`docs/65-documentation-audit.md`](65-documentation-audit.md),
  and its mechanical link/coverage gate is [`tests/test_documentation.py`](../tests/test_documentation.py).
  The audit links the implementation ledger, retained fixtures, contracts, and
  source locations; documentation claims remain bounded by those records.

## Specialization and hardware references

- [q27](https://github.com/signalnine/q27): attributed Qwen-on-5090 case study.
  Its reported formats and measurements are **External** and apply only to its
  artifacts, revisions, harnesses, and hardware.
- [RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
  [CUDA GPU compute capabilities](https://developer.nvidia.com/cuda-gpus),
  [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/),
  and [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/):
  **External**, version-sensitive hardware/toolchain facts.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  and [CUTLASS](https://github.com/NVIDIA/cutlass): **External primary** sources
  for CUDA execution, memory, and kernel-building interfaces.
- [DGX Spark product specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
  and [DGX Spark user guide](https://docs.nvidia.com/dgx/dgx-spark/): **External
  primary** sources for its GB10 platform, ARM host, and 128 GB coherent unified
  system memory. Exact toolchain and compute-target support remain
  version-sensitive.
- Apple's [Metal overview](https://developer.apple.com/metal/),
  and [resource storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus):
  **External primary** sources for Metal execution and Apple Silicon unified
  memory behavior.

## Arithmetic and proposed claims

- FFN elements/FLOPs, BF16/4-bit lower bounds, 144 MiB recurrence, 7.5 MiB
  convolution storage, and 64 KiB/token KV are **Estimated** from displayed
  formulas and official shapes.
- Unevidenced future results (including new weight, quality, context, throughput,
  or optimization claims) are **Proposed** until the milestone protocol emits
  named logs and fixtures. Named retained measurements in the chapters and
  fixtures keep their **Measured** label, including historical or partial
  results; they are not converted to Proposed by this general note.
- No q27 measurement is evidence for DwarfStar or for the proposed engine.

## Review checklist

- Re-pin and hash config, checkpoint, tokenizer/template, Transformers, llama.cpp, and converter.
- Re-inventory tensors and recalculate every shape/memory table.
- Run Markdown internal-link, heading/navigation, and Mermaid rendering checks.
- Run all scalar/CUDA/checkpoint/quant/context acceptance scenarios.
- Audit every performance and fit statement for an evidence label and raw record.
