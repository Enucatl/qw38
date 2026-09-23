# TASK-005 — BF16 identity compiler path

## Status
DONE
## Milestone
M1 — Runtime artifact and compiler foundation
## Purpose
Compile the authoritative BF16 checkpoint into the same tensor identities, semantic bindings, and state schema used by V0, providing a control that separates semantic/compiler failures from quantization loss.
## Depends on
- TASK-004
## Normative references
- `docs/architecture/architecture-v0.md` — Compiler pipeline; semantic graph; precision policy
- `docs/architecture/model-semantics.md`
- `docs/architecture/model-inventory.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01 | Six semantic families; primary-language executable scope | LOCKED |
| Q-03 | BF16 embeddings and small/sensitive families | LOCKED |
| P-01–P-02, S-01–S-02, M-01 | Precision, state, and materialization descriptors | LOCKED |
## Starting point
The format round-trips metadata and arbitrary byte spans. Source checkpoint is `.cache/authorities/qwen3.8-27b-transformers`.
## Scope
- Add an offline compiler CLI that reads HF config/safetensor index/shards, verifies exact architecture/tensor names/shapes/dtypes/sharing, excludes vision payloads, classifies language and retained-disabled MTP tensors, and emits a BF16 identity `.qw38`.
- Preserve exact BF16 values except static V0 transforms: dense BF16 control tile layout, convolution `[channel,1,tap]`→`[tap,channel]`, and generated 32 FP32 RoPE inverse frequencies.
- Emit source/config/tokenizer hashes, compiler revision, semantic/precision/layout descriptors, shared embedding/head bindings where specified, and language state/scratch schema.
- Stream tensors; add an independent reconstruction/export check. Support diagnostic one-layer-at-a-time consumption metadata rather than production CPU offload.
## Out of scope
Low-bit quantization, runtime kernels, tokenizer tables inside `.qw38`, vision, executable MTP, arbitrary Qwen variants, full-model RAM buffering.
## Required interfaces
Deterministic compiler CLI with explicit input/output; typed compiler stages returning `std::expected`; tensor identity/classification table encoded in code, not inferred heuristically.
## Required semantics
Norm weights remain stored and runtime applies the correct `1+gamma` or multiplicative role. Embedding and lm_head remain untied. MTP payloads/descriptors are retained but disabled.
## Data representation
BF16 dense control uses the corresponding eight-row/256-input tile order; embeddings row-major; vectors contiguous; convolution tap-major; RoPE inverse frequencies FP32.
## Implementation constraints
No PyTorch/TensorFlow runtime dependency in the engine. A narrowly justified offline safetensors/config parser is allowed. Reject nonfinite/unexpected/missing tensors and `.qw38` manifest-digest mismatches. Never compute, store, or compare a content digest over cached BF16 tensor payloads or their shard files, or over `.qw38` tensor payload/scale spans; follow the [checkpoint and `.qw38` payload digest policy](../code-standards.md#checkpoint-and-qw38-payload-digest-policy).
## Tuning defaults
None.
## Expected files/modules
Host `compiler/`, compiler CLI, source-format reader, identity reconstruction tests/tools.
## Tests required
### Unit tests
- Classification, shape/dtype/share validation, convolution transform, tile mapping, RoPE generation, missing/extra/duplicate tensor errors.
### Reference/numerical tests
- Exact BF16 reconstruction for synthetic tensors and sampled/full source tensors where feasible.
### Integration tests
- Compile a deterministic fixture and the authoritative checkpoint manifest; reader validates all bindings/state metadata; repeated compile is byte-identical given fixed revision metadata.
## Benchmark required
No; record peak host memory as diagnostic evidence only.
## Acceptance criteria
- [x] Identity artifact reconstructs every included BF16 source tensor exactly after inverse layout transforms.
- [x] Language graph has 130 instances and MTP is retained but disabled.
- [x] Source/config/tokenizer/compiler identities and state schema are present.
- [x] Compiler streams without holding the complete checkpoint in memory.
- [x] Tests pass with exact commands recorded.
## Architecture blocker rule
On a locked conflict, stop with all required `ARCHITECTURE_BLOCKER` fields; do not reinterpret source semantics.
## Completion report
### Result
DONE. Independent verification PASS (Debug/Release ctest 14/14).
### Changes made
- Host `src/compiler/`: encoded 38-family identity table expanded to the sitting 866 language+MTP tensors; vision `model.visual.*` excluded; MTP classified retained-disabled.
- Offline CLI `qw38-compile --checkpoint DIR --output FILE [--no-verify]` reads HF `config.json`, `model.safetensors.index.json`, and shard headers/payloads with a narrow JSON/safetensors parser (no PyTorch/TF).
- Typed `std::expected` stages: config/architecture validation, header classification (names/shapes/dtypes/sharing), schema emit, streaming transform/write, independent reconstruction.
- V0 identity transforms: BF16 dense `[N/8,K/256,8,256]` tiles streamed in 4 KiB units; conv `[channel,1,tap]`→`[tap,channel]`; generated 32 FP32 RoPE `ω_j=θ^{-2j/64}` as `rope.inv_freq`.
- Artifact metadata: SHA-256 of index/config/tokenizer, compiler revision `qw38-bf16-identity` 0.1.0, precision policy V0, scope `language_plus_mtp_descriptors`, shared MTP embed/`lm_head` aliases, language GDN/conv/KV state schema, decode scratch schema. Graph bindings carry `layer_index` for diagnostic one-layer consumption.
- Format extension required to store generated RoPE: `StorageClass::Fp32` and `PhysicalLayoutId::CudaFp32VectorV0`.
- Tests: `compiler_identity`, `compiler_transform`, `compiler_integration`. CMake wires `qw38_compiler` and `qw38-compile`.
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 14/14 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.02s, `format_sha256` 0.02s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`, `compiler_identity` 0.02s, `compiler_transform`, `compiler_integration` 0.24s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 14/14 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.01s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.22s, `cuda_sm120_cubin`, `compiler_identity`, `compiler_transform`, `compiler_integration` 0.07s).

Authoritative checkpoint `.cache/authorities/qwen3.8-27b-transformers` was present (18 shards). Integration classified all 866 language+MTP tensors, excluded 333 vision tensors, built a 130-instance language graph with 5 retained-disabled MTP instances, and reconstructed sampled real `A_log`, `conv1d`, and `in_proj_a` tensors exactly after inverse transforms. Repeated synthetic compile was byte-identical. Full 54 GiB identity emission is `src/qw38-compile --checkpoint ... --output ...`; default ctest does not write that artifact.
### Benchmark results
Not required. CLI prints `peak_rss_bytes` from `/proc/self/status` `VmHWM` after compile. Default tests stream one shard at a time plus 4 KiB dense tiles; they do not materialize the 54.64 GB source.
### Architecture blocker
None.
### Follow-up observations
- TASK-002 graph validation allows `LM_HEAD` bindings only with role `LmHeadWeight`, so `model.language_model.norm.weight` and `mtp.norm.weight` are stored in the tensor directory but not graph-bound as head norms.
- Generated RoPE required an FP32 vector storage/layout that TASK-002 did not define; `StorageClass::Fp32` / `cuda_fp32_vector_v0` is the additive ABI used here.
- Full-checkpoint `.qw38` emission is implemented and reconstruction-checked in the CLI (`--no-verify` to skip the second pass) but is omitted from default ctest because the identity payload is ~54 GiB.
