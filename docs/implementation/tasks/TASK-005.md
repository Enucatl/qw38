# TASK-005 — BF16 identity compiler path

## Status
TODO
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
No PyTorch/TensorFlow runtime dependency in the engine. A narrowly justified offline safetensors/config parser is allowed. Reject nonfinite/unexpected/missing tensors and hash mismatches.
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
- [ ] Identity artifact reconstructs every included BF16 source tensor exactly after inverse layout transforms.
- [ ] Language graph has 130 instances and MTP is retained but disabled.
- [ ] Source/config/tokenizer/compiler identities and state schema are present.
- [ ] Compiler streams without holding the complete checkpoint in memory.
- [ ] Tests pass with exact commands recorded.
## Architecture blocker rule
On a locked conflict, stop with all required `ARCHITECTURE_BLOCKER` fields; do not reinterpret source semantics.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Not required; include peak-memory diagnostic if measured.
### Architecture blocker
None or full report.
### Follow-up observations
Concrete only.

