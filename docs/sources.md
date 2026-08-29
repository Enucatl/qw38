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
  fixture is transparently labeled a scalar transcription; a direct eager trace
  remains ORA-001 work.
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
  [`fixtures/real_ffn_step.json`](../fixtures/real_ffn_step.json). The complete
  down projection is currently native regression evidence; direct semantic
  trace admission remains TRC-001, TRC-002, and ORA-001 work.

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
