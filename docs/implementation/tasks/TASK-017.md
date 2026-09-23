# TASK-017 — Complete primary-language decode

## Status
DONE
## Milestone
M6 — Integrated language decode
## Purpose
Reach the first complete batch-one autoregressive language-model execution through all 64 layers.
## Depends on
- TASK-016
## Normative references
- `docs/architecture/architecture-v0.md` — Decode schedule; end-to-end architecture
- `docs/architecture/model-semantics.md`
- `docs/architecture/model-inventory.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01–G-02 | 130-instance primary-language graph and decode schedule | LOCKED |
| Q-01–Q-03, P-01–P-02, S-01–S-02, L-01, M-01 | Full V0 storage/execution policy | LOCKED |
## Starting point
Both layer styles execute through shared runtime; artifact/compiler include complete language bindings.
## Scope
Implement model plan validation and decode CLI/API: BF16 embedding gather/widen; 64 layers in three-GDN/one-attention pattern repeated 16 times, each with MLP; all GDN/history/KV state; final zero-centered norm; Q8G32 lm_head producing all 248320 FP32 logits; deterministic argmax default while retaining logits. Add deliberately slow repeated-decode prompt-state setup for correctness only and BF16 one-layer-upload diagnostic if full BF16 control does not fit VRAM.
## Out of scope
Production prefill, sampling policy, tokenizer payload in artifact, MTP execution, vision, batching, paging, CPU offload production path.
## Required interfaces
Concrete compiled-model plan, session capacity, `decode_token(token_id, position)` returning FP32 logits/view and argmax; CLI clearly labels primary-language scope and slow prompt setup.
## Required semantics
Exactly 64 language layers; MTP disabled; embedding and lm_head untied; each token commits state in order; populated length/position remain coherent; logits are FP32. Per-layer progress remains pending until all layers and output work complete; one global token position commits at that boundary. A failure after any layer mutation poisons the session until reset or valid snapshot restore; it cannot be retried at the same position as if earlier layers had not advanced.
## Data representation
Use artifact bindings and session schema; no runtime repack/full dequantization; active state byte formula remains language-only.
## Implementation constraints
Stable allocations, eager one-stream schedule, no optimized prefill prerequisite, no model-wide BF16 residency requirement for diagnostic control.
## Tuning defaults
Inherited decode geometries.
## Expected files/modules
Runtime model plan/scheduler, decode CLI, full-model smoke/integration fixtures.
## Tests required
### Unit tests
Graph count/order/binding validation, token/position/capacity errors, MTP disabled, logits shape/dtype.
### Reference/numerical tests
Small deterministic full-stack fixture and selected authoritative-checkpoint checkpoints versus an independent source-model oracle. BF16 engine control must first agree with source logits and selected residual/state checkpoints within declared tolerances before V0-versus-BF16 differences are attributed to quantization.
### Integration tests
Reset, multi-token decode, slow prompt setup, save/restore continuation, deterministic repeated greedy output; complete authoritative model produces finite logits.
Inject failure after an earlier layer advanced; verify execution and snapshots reject while poisoned, then compare continuation after reset/restore with a clean control. Inspect persistent bytes and host metadata together.
## Benchmark required
No performance gate; record smoke latency only as diagnostic.
## Acceptance criteria
- [x] Complete language graph executes one and multiple tokens.
- [x] All state families update/continue correctly and reset works.
- [x] Final output is 248320 finite FP32 logits plus deterministic argmax.
- [x] MTP stays retained but disabled and output is labeled language-only.
- [x] No production-prefill shortcut is claimed.
- [x] Pin source weights/config/tokenizer, source software revision, token IDs, precision settings, and hashes for initial and continued-token logits plus selected residual/state checkpoints.
- [x] Budget the one-layer-resident BF16 diagnostic explicitly; report untested coverage and preserve exact equality only for deterministic replay of the same implementation/schedule.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not omit/reorder layers or enable unresolved MTP.
## Completion report
### Result
DONE. The original graph-binding failure came from a stale compiler executable. Current `src/compiler/identity.cpp` already assigns retained MTP layer-scoped bindings layer index 0. Rebuilding the pinned Debug compiler made both required artifacts compile without a compiler or schema source change.

### Changes made
Added the 64-layer language plan, embedding gather, final RMS, Q8G32/BF16 head, FP32 logits and deterministic lowest-index argmax; a primary-language decode CLI with slow repeated-decode prompt setup; one global token commit after output completion; and session snapshot/reset handling of that commit. A failed execution poisons the session until reset or valid restore. Added a BF16 diagnostic that uploads embedding, one layer, or output weights at a time, and an independent source-oracle comparison. MTP descriptors remain in the artifacts and are never executed.

### Artifact and oracle identity
Checkpoint: local `.cache/authorities/qwen3.8-27b-transformers`; `model.safetensors.index.json` SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`, `config.json` SHA-256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`, `tokenizer.json` SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. The compiler uses these metadata identities, not payload hashes. Production artifact manifest SHA-256 `ef59127793383c40e0887315c084180eef12907627ebf6bd1a0956cc7b8a0044` (`qw38-v0:0.1.1`, 17,095,659,090 bytes); BF16 identity manifest SHA-256 `ab2a56a43855c5f541d8ddbe1fd1238b95abb1795938719b5f4c4ecf4b9ee09c` (`qw38-bf16-identity:0.1.1`, 54,641,561,181 bytes). Both compile 130 language instances, retain 5 disabled MTP instances, include 866 tensors and exclude 333 vision tensors.

Source oracle: `scripts/task017_source_oracle.py`, Python 3.12, Transformers 5.17.0, PyTorch 2.9.0+cu128, Accelerate 1.12.0, Safetensors 0.8.0, NumPy 2.5.3; Qwen3.5 conditional-generation model, BF16 weights/activations, eager attention, TF32 disabled, cached single-token decode. GPU/CPU placement uses 26 GiB/20 GiB limits with disk offload. Inputs are token IDs `1, 2` at positions `0, 1`. A separate two-token source forward checks cached continuation. The engine uses BF16 identity or V0 quantized weights and FP32 logits.

| Checkpoint | Source SHA-256 | BF16 engine SHA-256 | V0 engine SHA-256 |
| --- | --- | --- | --- |
| Logits, position 0 | `f72f45f56028378fb7eefe65bf1b828f6bf90bc0c11c9097e088f8d93aa5d8cc` | `0b0c0f2e8f83d232ab8fa565bd22c9d055c78c5cd61955a998f16078667cddbc` | `e8c33da87c6c1d34eaaf90168b1f495db264d19ef79065ff35113d8bedaa3e66` |
| Logits, position 1 | `7ba92f9d29a5c79673942f22faccfdb2181c61386cdefb7c30d10c0ccc094516` | `398ec6ba11f6b2a8ac21d2d2167f442398eac038c233e213e14669f5553fed6a` | `9e12134a2b75e106572c62689fafc70e2853145aab71fa36f3333d3c59181b38` |
| Residual after layer 0, position 0 | `f65f093ed398ae01083e9977067c17d778b12b49439991c16bee771844518d62` | `86199685619a2f7556ad0837be56d9bf6500ac072cae1cddda3648d71c6089c4` | — |
| Recurrent state layer 0, position 0 | `1a25d41f0536aa77762eab5c53d4487269970eebe96c157508321afa788f6ef8` | `58e0c2ac9187e34ba716311c1effbbc7c3cfb9afb89cc87267fa8448bf46b58f` | — |
| Convolution history layer 0, position 0 | `bbbe1c3fcd6508846313b641ae28eb48c0b43d0ba2b597891b3ea4feebb313c6` | `a86bfb5c6d9d3d6382f5e68acc9332791a499f692c36c3e23b2db3bf7d6f51bd` | — |
| Attention key cache layer 3, position 0 | `0a02f1a6eb35dc7d6b39b268847732ac2f0fef874dbbbe91e7e3cfe67005a064` | `5c10000ebc93acba4990799cf621e3606ecaeff7a467ce75a73cd74ac01e2ba3` | — |
| Residual after layer 60, position 1 | `9b75244e0c096272c81df2e42dbe50f02d5fcc603060e9c5784828abe62d17f4` | `f896eee30348e58159b87bd43471baa2669720f9bbe4f1a6cbf862787bbb2274` | — |
| Recurrent state layer 60, position 1 | `c7eb6251ddbe6e1b506a7a56f93dab2b5ac91d011c9ea9ef2873fdffa894e3d5` | `68ef8ed3394b6ed797dc0324e1ca56d56fcc9057a6573194feb0660645c1ef01` | — |
| Convolution history layer 60, position 1 | `3f1c1a2047d0d9a35c5d451fa0af569c9d97506ae01533541261dc3cea6cf470` | `729b48315e2ae171200e24c33600ca5cf674f5f6cdcdd5e7a7d576e239b4faa4` | — |
| Attention key cache layer 3, position 1 | `030c96dcd13073a60790f08059e973bcc2bdb5f6152c4dbdb92ecc2b0d971de9` | `5c5975df05d2362eeac46ff18fb175acdbcedcaafedca259e5702016e1a4ea1d` | — |

State hashes above are for the normalized comparison views in `scripts/task017_compare_oracle.py` (source recurrent axes transposed, engine convolution ring reordered, BF16 cache widened to FP32). The script checks residual layers 0/28/60, their recurrent and convolution state, attention layer 3 keys and values, and both positions. BF16 versus source logit mean/max absolute errors are `0.0242/0.1427` and `0.0268/0.1404`, within declared `0.035/0.16` gates. Cached versus two-token source logits differ by `0.0183/0.125`, within `0.025/0.15`. V0 versus source logit mean/max errors are `0.2644/1.7073` and `0.2414/1.4474`; V0 argmax is `5328, 14955`, while source and BF16 argmax are `5328, 220`. This second-token V0 difference is recorded as quantization behavior; TASK-018 owns the quality baseline.

### Tests run
Rebuilt in `qw38-dev:cuda13.4.1-pinned`. Production and BF16 identity artifact builds passed. `ctest --test-dir build/pinned-debug --output-on-failure -R '^(language_model_plan|language_model_integration|runtime_session_integration)$'` passed 3/3 with `QW38_AUTHORITY_ARTIFACT` set to the production artifact; token IDs `1, 2` form the deterministic two-token full-stack fixture. The full-model test checks finite 248320-element logits, token and position errors, all state families, a late injected head failure, poison, restore, reset, exact replay logits, and every persistent byte/metadata field. Focused `language_layer_integration`, `runtime_session_integration`, and `compiler_integration` passed 3/3. The BF16 diagnostic completed both tokens; the production artifact was rejected by its identity-only guard. `uv run --script --python 3.12 scripts/task017_compare_oracle.py build/pinned-debug` passed every declared numerical gate. `git diff --check` passed.

### Benchmark results
No performance gate. Full authoritative integration including model load, multiple decodes, snapshot/recovery, and replay took 29.67 seconds in the pinned Debug build. The one-token CLI smoke took 31.25 seconds including container startup and artifact load, producing 248320 finite FP32 logits and argmax 5328. This is an eager correctness path, not production prefill.

### BF16 diagnostic budget and limits
The diagnostic peaks at 2,542,807,040 bytes of uploaded model weights, plus 154,075,136 bytes of persistent state, 256,901,120 bytes of arena scratch, and 10,485,760 bytes of residual buffers: 2,964,269,056 bytes of accounted GPU allocations, excluding CUDA/runtime overhead. Full 54.6 GB BF16 weight residency is unnecessary. It covers two sequential tokens, selected state checkpoints, and logits; it does not validate longer contexts, all per-layer checkpoints against the source, MTP, vision, batching, or sampling quality. Exact equality is required only for replay of the same engine schedule after reset/restore; source and BF16 comparison uses the declared numerical tolerances.

### Architecture blocker
None. The stale executable caused the original artifact failure; no locked Architecture V0 conflict was found.
