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
DSpark are explicitly rejected as Qwen model semantics.

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
  tile selection remains profiler work rather than an externally borrowed
  performance claim.
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
- Weight artifact size, runtime allocations, quant quality, context capacity,
  RTX 5090 throughput, and every optimization result are **Proposed** until the
  milestone protocol emits named logs and fixtures.
- No q27 measurement is evidence for DwarfStar or for the proposed engine.

## Review checklist

- Re-pin and hash config, checkpoint, tokenizer/template, Transformers, llama.cpp, and converter.
- Re-inventory tensors and recalculate every shape/memory table.
- Run Markdown internal-link, heading/navigation, and Mermaid rendering checks.
- Run all scalar/CUDA/checkpoint/quant/context acceptance scenarios.
- Audit every performance and fit statement for an evidence label and raw record.
