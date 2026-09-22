# Implementation review

This is the deduplicated review of the completed subagent reviews for TASK-001 through TASK-015. Findings are ordered by priority. Each item is written as an actionable work description for a fixing worker. The four TASK-015 findings supplied for this consolidation are included below.

There were no P0 findings. “Confirmed” means the reviewer found the issue directly in the current implementation. “Concern” means the issue needs a contract decision or additional reproduction before changing code.

## Repair verification policy

Each repair uses the smallest deterministic test that exercises its changed
behavior, plus a focused regression for any shared interface it affects. Use
one configured build by default: Debug for host-side format/compiler/validation
changes and Release for CUDA/runtime numerical changes. Do not run both full
suites unless the change is configuration-sensitive, changes the build or
toolchain, or the task contract explicitly requires it.

The default repair suite is:

```text
ctest --test-dir build/release --output-on-failure -LE extended
```

The `extended` label is reserved for authoritative-checkpoint scans and
full-vocabulary MMV integration. Run an extended test only when the change
affects that property or at a deliberate repair-batch/release checkpoint, and
state why. Long-context, maximum-capacity, sanitizer, benchmark, and full
artifact identity checks follow the same rule. Existing test directions below
describe the required property; they do not require the largest available
fixture when a smaller fixture establishes that property.

## P1 — high priority

### 1. Pin the complete TASK-001 build toolchain (confirmed; TASK-001)

`Dockerfile:21` installs `cmake`, `ninja-build`, `gcc-14`, and `g++-14` from mutable Ubuntu repositories after `apt-get update`, while `docs/implementation/dev-environment.md` claims exact versions. The CUDA base image is digest-pinned, but the host compiler, CMake, Ninja, and standard library are not. Rebuilding later can silently change compilation and binary output.

Fix by consuming exact package versions from an immutable repository snapshot, or by publishing and consuming the completed development image by immutable digest. Update the documented build command so it resolves to the same pinned toolchain. Re-run the documented Debug/Release builds and record the image and tool identities.

### 2. Make writer publication race-safe (confirmed; TASK-003)

`src/format/writer.cpp:434` always uses `<destination>.tmp`, removes that path at line 443, and later renames the pathname at line 706. Two concurrent writers can unlink one another’s open temporary file; writer A can then rename writer B’s incomplete file and report success. An unrelated pre-existing `.tmp` file is also deleted.

Fix by creating a unique same-directory temporary file atomically, retaining that exact private pathname through final rename, and deleting only the inode/path owned by the current writer. Add a concurrent-writer test and a test proving a pre-existing temporary file is never removed or published.

### 3. Enforce a canonical, acyclic shared-binding ownership graph (confirmed; TASK-002/TASK-003/TASK-004)

Schema validation only rejects duplicate directed pairs (`src/format/schema.cpp:2011`). It permits multiple owners, alias-as-owner chains, and cycles such as `A→B`, `B→C`, `A→C` or `A→B`, `B→A`. `writer.cpp:131` selects the first matching owner, physical spans are emitted only for non-alias tensors, and integrity records can therefore be omitted. The reader treats every alias endpoint as non-owning (`reader.cpp:155`, `:253`), allowing a cyclic alias to bypass file-range and overlap checks and potentially expose header bytes as payload.

Fix the schema contract to require one canonical non-alias root per alias, reject multiple owners, aliases that are owners, chains if unsupported, and all cycles. Resolve each alias to its canonical root before assigning spans, emitting integrity records, and validating ranges. Add encode, writer, reader, and round-trip tests for reverse pairs, multiple owners, chains, cycles, missing owner digests, and alias/header overlap.

### 4. Bound all wire-record counts before allocation and translate allocation failures (confirmed; TASK-002/TASK-004)

`decode_schema` reserves vectors directly from untrusted `uint32_t` counts (`src/format/schema.cpp:1758-1828`) before proving that the manifest can contain that many records. `Artifact::parse(span)` copies the entire input before validation (`src/format/reader.cpp:533`), and public parse/open paths do not translate `std::bad_alloc` or `std::length_error`.

Fix by preflighting each count against the remaining bytes and minimum encoded record size, imposing defensible V0 limits, and catching allocation/size exceptions at public `std::expected` boundaries. Validate bounded header/manifest sizes before copying input. Add tiny adversarial manifests with `UINT32_MAX` counts and verify typed errors, bounded resource use, and no process-terminating exceptions.

### 5. Make schema shape/rank helpers safe on malformed input (confirmed; TASK-002/TASK-004/TASK-005)

`TensorShape::logical_dims()` trusts `rank` (`src/format/schema.hpp:38`), `write_shape` loops to it (`schema.cpp:107`), and `encode(ArtifactSchema)` and `expected_payload_bytes` use the unchecked arrays. Ranks above eight can read beyond fixed storage. Safetensors shape dimensions are also converted from `double` to `uint64_t` without checking finiteness/integrality/range (`src/compiler/checkpoint.cpp:429`).

Fix by validating rank before every array access, using checked products and byte sizes, and parsing JSON integer fields with explicit finite, integral, nonnegative, and representable checks. Make `encode`, sizing helpers, checkpoint parsing, and compile paths return typed errors for malformed ranks/dimensions. Add tests through every public entry point, including rank 8, rank 9, zero, overflow, fractional, negative, NaN, and infinity cases.

### 6. Enforce the required state/scratch schema before runtime allocation (confirmed; TASK-002/TASK-004)

The state validator records observed state kinds but never requires the complete required set (`src/format/schema.cpp:1908`). `validate_scratch` accepts any known dtype and nonzero byte count (`:1246`), and KV state returns before the no-padding check (`:1219` versus `:1236`). Empty state schemas and one-record scratch fixtures are accepted by tests, while stricter checks happen only later in `Session::create`, after `Model::upload` has allocated/uploaded device buffers.

Fix by enforcing the exact language-only state kinds, fixed dimensions, no padding, required scratch kinds, kind-specific dtype/size/alignment, and precision-policy consistency during artifact validation. Perform this compatibility preflight before any device allocation. Add malformed artifact tests for missing states, wrong sizes, wrong dtypes, padded KV, invalid scratch byte counts, and contradictory precision policy.

### 7. Validate every behavior-defining architecture configuration field (confirmed; TASK-005)

`parse_text_config`/`validate_architecture` (`src/compiler/checkpoint.cpp:224`, `:335`) validate only selected dimensions and RoPE fields. They ignore `model_type`, `rms_norm_eps`, `attention_bias`, `attn_output_gate`, `hidden_act`, `output_gate_type`, and `rope_type`; `kTextModelType` is declared but unused (`src/compiler/identity.hpp:45`). A checkpoint with different model semantics but matching tensor names/shapes can therefore compile as V0.

Fix by representing and validating every behavior-defining language/MTP configuration field, including cross-field invariants and exact allowed values. Reject unsupported Qwen variants and config/dtype combinations with typed errors. Add mutation tests that change one field at a time and prove compilation is rejected.

### 8. Bind source and reconstruction verification to all artifact identities (confirmed; TASK-005)

`open_checkpoint` sets `source_hash` from only `model.safetensors.index.json` (`src/compiler/checkpoint.cpp:475`), excluding shard contents. `verify_compiled_artifact` (`src/compiler/compile.cpp:676`) compares payloads but does not compare source, config, tokenizer, or compiler identities.

Fix by defining a canonical source identity covering the index and every shard’s identity/digest, storing config/tokenizer/compiler identities, and requiring verification to compare all of them before reconstruction. Add tests that mutate shard bytes, config, tokenizer, and compiler revision while keeping filenames/indexes unchanged; each mismatch must fail with a typed diagnostic.

### 9. Make safetensors parsing reject malformed offsets, overlaps, and type errors (confirmed; TASK-005)

`parse_shard_header` passes offset fields through `Json::as_number()` (`src/compiler/checkpoint.cpp:436`), which can throw `std::bad_variant_access`. Identity validation rejects only identical `(shard, offset, length)` triples (`src/compiler/identity.cpp:272`), not partial/contained overlaps. Wrong numeric types can escape the `std::expected` error contract, and overlapping source tensors can be compiled as distinct tensors.

Fix by parsing unsigned integer fields without throwing, checking interval overflow, rejecting partial overlaps unless an explicit exact-sharing rule authorizes them, and translating all parser failures to `CompilerError`. Add malformed-header tests for wrong JSON types, fractional/negative values, overflow, out-of-file ranges, partial overlaps, and valid exact sharing.

### 10. Preserve complete semantic norm bindings (confirmed; TASK-005)

All additive residual/QK norms and the multiplicative GDN gated norm use `TensorRole::NormGamma` (`src/compiler/identity.cpp:43`, `:79`), so roles are ambiguous. `bindable()` (`src/compiler/compile.cpp:154`) omits `model.language_model.norm.weight` and `mtp.norm.weight` because their node is `LmHead` but their role is not `LmHeadWeight`.

Fix by defining distinct roles or binding slots for additive norms, QK norms, multiplicative GDN norms, final language norm, and MTP norm. Permit the required final/MTP norm bindings on the correct graph nodes and validate required multiplicity per node kind. Add artifact inspection tests proving every required norm is addressable without hard-coded logical names in consumers.

### 11. Keep compiler verification bounded and measure peak memory after verification (confirmed; TASK-006)

`verify_compiled_artifact` requantizes complete tensors (`src/compiler/compile.cpp:727-732`), while `quantize_bf16` allocates full FP32 values plus codes and scales (`src/compiler/quantization/quantizer.cpp:147-179`). `peak_rss_bytes` is captured before verification (`compile.cpp:656`). The default compiler path therefore defeats TASK-006’s eight-row bounded streaming promise and underreports memory, especially for the vocabulary head.

Fix by verifying payload and scale bytes incrementally in eight-row or physical-tile chunks, reusing bounded scratch, and recording peak RSS after all requested verification has completed. Add a large-tensor memory regression test or documented measurement proving the bound.

### 12. Validate packed quantizer domains before CUDA execution (confirmed; TASK-006/TASK-009)

The independent unpacker rejects Q4 code `-8` and Q8 code `-128` (`src/format/unpack.cpp:14`), but CUDA decoders interpret all nibble/byte values (`cuda/decode_mmv.cu:103-125`). CUDA also accepts invalid FP16 scale encodings. Artifact opening checks structure and hashes but does not validate these semantic domains before upload.

Fix by adding one-time artifact/weight validation for forbidden codes, zero padding, and scale encodings (zero or positive finite normal values with the required floor), preferably during the existing hash/scan pass. Return typed errors before model creation and add malformed-artifact tests proving reference and CUDA paths cannot disagree.

### 13. Restore the required QK precision boundary in TASK-008 (confirmed; TASK-008)

`cuda/activation.hpp:31` accepts FP32 Q/K, `cuda/activation.cu:121` stores normalized values to BF16, and `:175` widens that BF16 for RoPE before another BF16 store. The V0 contract requires BF16 projection staging, FP32 QK normalization and RoPE, and one BF16 store after RoPE. The integration test uses synthetic FP32 Q/K (`tests/activation_integration_test.cpp:95`) and therefore validates the wrong pipeline.

Fix the authoritative reference and CUDA comparison to consume BF16 projection output, retain normalized values in FP32 through rotation, and round only at the declared post-RoPE store. If rounded standalone primitives remain, mark them non-authoritative. Add bit-sensitive tests that distinguish one store from two.

### 14. Add complete typed operand descriptors to decode MMV launches (confirmed; TASK-009)

`DecodeMmvDesc` exposes bare pointers for input/output/residual (`cuda/decode_mmv.hpp:43-58`); `validate_geometry` checks only nullness (`cuda/decode_mmv.cu:289-312`). `DecodeMmvPairedDesc` has complete metadata only for A; B is validated only by byte counts derived from A (`:316-347`). Equal-sized but differently shaped/formatted B matrices can therefore pass.

Fix all operand views to carry memory space, dtype, layout, logical/padded shape, byte extent, alignment, and writability. Validate exact required sizes, epilogue output type, non-overlap, and Q4/Q8/BF16 compatibility before launch. Give paired B a complete descriptor and require it to match A’s required shape/layout/input/epilogue contract. Add negative tests using equal-size but incompatible matrices.

### 15. Implement the required MLP residual ping-pong (confirmed; TASK-010)

`MlpPlan` has only one residual view and documents in-place mutation (`src/runtime/mlp.hpp:35`). Binding selects `session.residual_h_mid()` only (`src/runtime/mlp.cpp:321`), and the down epilogue mutates it (`:379`); `session.residual_h()` remains unused. TASK-010 requires `h_mid` to remain live while the down projection produces the next FP32 residual.

Fix the plan to bind distinct input `h_mid` and output `next_h` views. Make the down epilogue write `next_h = h_mid + accumulator` without destroying `h_mid`. Add tests that verify source preservation, destination contents, repeated execution, and the exact session buffer consumed by the next layer.

### 16. Make MLP alias validation range-aware (confirmed; TASK-010)

`bind_mlp_plan` checks only three exact pointer combinations (`src/runtime/mlp.cpp:259`). It permits residual/swiglu, gamma/normalized, gamma/swiglu, and partial overlaps even though region 2 writes scratch before region 3 reads the residual and gamma is immutable.

Fix by constructing checked byte intervals for every bound view and rejecting all harmful overlap according to region lifetimes, including partial overlap and output/input overlap. Add negative tests for every exact and offset overlap combination.

### 17. Implement the mandated single-launch GDN schedule (confirmed; TASK-011/TASK-013)

The architecture requires each numbered GDN step to be one launch, including one ranged qkv/z projection and direct output projection/residual production. `region_qkvz` calls `launch_decode_mmv` twice (`src/runtime/gdn.cpp:248-263`), and `region_out_residual` copies residual before output MMV (`:1015`). This makes a mixer nine kernels plus a copy instead of eight launches. Both TASK-011 and TASK-013 completion reports acknowledge the deviation.

Fix or generalize the ranged MMV consumer so qkv and z are one launch, and make the output epilogue read the original residual while writing the next residual, eliminating the pre-copy. Add a launch-count/profiling test for Q4 and BF16-control. If the selected schedule cannot be preserved, record the required architecture blocker instead of marking the task complete.

### 18. Enforce the locked GDN recurrence state ABI and layer selection (confirmed; TASK-012)

`bind_gdn_recurrence_plan` (`src/runtime/gdn.cpp:704-785`) checks placement, FP32, writability, and element count but not `PhysicalLayoutId::CudaFp32GdnSHvKV0`, `StorageClass::Fp32`, rank, or `[layer,48,128,128]` shape. It computes `gdn_state_index(language_layer)` but does not require it to equal caller-supplied `s_layer`; execution uses `s_layer` while `gdn_layer` is unused.

Fix by requiring exact layout/storage/rank/extents with checked arithmetic, deriving the state layer from the language layer or rejecting mismatches, and adding tests for transposed state, wrong layer, wrong storage, wrong shape, and overflow.

### 19. Separate attention cache population ownership from per-layer cache writes (confirmed; TASK-014)

`Session` owns one `kv_populated_` scalar (`src/runtime/session.hpp:54`), every attention layer binds it (`src/runtime/attention.cpp:655`), and each layer requires `position == populated` then advances it (`:673`, `:731`). After one layer appends token 0, the next attention layer rejects token 0. This blocks full multi-layer decode even though the cache is layer-indexed.

Fix the ownership model: use per-layer population state, or let every layer append the same position while the model-level scheduler commits one token only after all layers complete. Ensure snapshots represent all layer caches consistently. Add a test that executes position 0 through at least two attention layers in one session and verifies all layer cache lengths.

### 20. Translate host allocation failures from runtime APIs (confirmed; TASK-007)

Fallible runtime APIs allocate without exception translation: `Session::save` resizes potentially multi-gigabyte snapshot vectors (`src/runtime/session.cpp:193`), while `Model::upload`, `Session::create`, and `plan_scratch_arena` allocate at `src/runtime/model.cpp:132`, `src/runtime/session.cpp:94`, and `src/runtime/arena.cpp:72`. A host memory failure escapes as `std::bad_alloc` instead of the declared typed runtime error.

Fix by containing `bad_alloc` and size-related exceptions at the public fallible boundaries, translating them to the existing allocation error type, and preserving RAII cleanup. Add failure-injection or bounded-resource tests for upload, arena creation, and full-state snapshot.

### 21. Make Runtime/Session stream lifetime safe (confirmed; TASK-007)

`Runtime` is freely movable (`src/runtime/runtime.hpp:17`), while `Session` stores a raw pointer to the Runtime’s `Stream` (`src/runtime/session.hpp:89`, assigned at `session.cpp:100`). Moving Runtime leaves sessions pointing at the moved-from stream object; destroying Runtime leaves dangling pointers used by reset, save, restore, and execution.

Fix the ownership contract by making Runtime immovable while sessions exist, or by giving sessions stable shared ownership of stream state. Add tests for Runtime move, Runtime destruction ordering, and all session operations after the permitted ownership transition.

### 22. Size attention scratch for the full configured context (confirmed; TASK-015)

`src/runtime/session.cpp:90-96` fixes the arena at 256 tokens, and `src/runtime/arena.cpp:249-252` provisions 78,016 × 256 = 19,972,096 attention bytes. Attention partial sizing uses KV capacity (`src/runtime/attention.cpp:509-513`), while the configured architecture context is 262,144 tokens (`docs/architecture/architecture-v0.md:278`) and requires 25,415,680 bytes. Binding therefore fails above approximately 205,824 tokens even though KV storage can be allocated.

Fix by sizing attention partial scratch from the requested KV capacity during session creation, with checked arithmetic and a clear memory-limit error. If V0 intentionally cannot support the configured context, reject that capacity during session creation before allocating KV and document the exact supported maximum. Add tests at 205,824, 205,825, and 262,144 tokens.

### 23. Use a stable CUDA sigmoid implementation for extreme attention gates (confirmed; TASK-015)

`cuda/attention.cu:333-336` computes `1 / (1 + expf(-gate))`, while the CPU reference uses a sign-dependent stable formulation (`src/reference/math.cpp:85-92`). For sufficiently negative gates, `expf(-gate)` overflows and CUDA produces exact zero even where stable FP32 sigmoid remains nonzero. Tests stop at -80 (`tests/attention_core_unit_test.cpp:59-61`, `:150-152`).

Fix the CUDA gate to use the same stable sign-dependent formula as the reference, preserve the declared FP32 gate semantics before BF16 output, and add tests below the FP32 exponential overflow threshold as well as positive extremes. Compare CUDA/reference gate and final gated output values.

### 24. Enforce complete attention typed-view, alignment, and overlap contracts (confirmed; TASK-015)

The attention binder (`src/runtime/attention.cpp:71-92`) checks pointer, memory space, dtype, writability, and element count but not physical layout, storage class, alignment, or interval overlap. Workspace binding (`:260-292`) and KV validation (`:456-468`) omit layout/storage requirements, and only two exact-pointer aliases are checked (`:456-458`). Q4 scales (`:137-148`) are not required to be FP16 with the expected layout/storage. A residual/output overlap can destroy projected input during the residual copy, and overlapping workspace/cache views can corrupt persistent state.

Fix by validating every attention operand’s exact dtype, rank/extents, physical layout, storage class, alignment, and checked byte interval. Explicitly document and permit only intentional aliases such as Q→Y reuse; reject all other exact and partial overlaps. Add malformed metadata, misalignment, and offset-overlap tests for preparation, cache, workspace, residual, and output views.

## P2 — moderate priority

### 20. Preserve complete integrity coverage for manifest metadata (confirmed; TASK-003)

`writer.cpp:611-622` hashes only the prefix before the complete 56-byte integrity record, leaving 24 bytes of manifest record metadata (kind, tensor ID, region fields) outside the digest while excluding the whole record. A corruption of manifest `tensor_id` can go undetected.

Fix the manifest hash protocol so only the self-referential digest bytes are excluded or zeroed; hash the remaining integrity-record metadata. Update reader verification and add a fixed-byte corruption test for every manifest record field.

### 21. Reject caller-supplied invalid span placements instead of normalizing them (confirmed; TASK-003)

`writer.cpp:294-306` validates lengths only when nonzero, skips alignment/overflow/overlap checks for aliases, and later overwrites placements in `:172` and `:207`. Empty spans with nonzero offsets and malformed alias spans are silently accepted.

Fix by validating every supplied span, including empty and alias spans, before deterministic placement: offset/length consistency, 256-byte alignment, checked end arithmetic, and permitted overlap. Reject bad caller metadata with typed errors instead of repairing it.

### 22. Add real writer failure injection and independent digest vectors (confirmed; TASK-003)

`tests/format_writer_test.cpp:338` tests destructor cleanup and semantic failure but not `pwrite`, manifest write, `fsync`, close, or rename failure. Emitted digest tests use the same SHA implementation under test (`tests/format_sha256_test.cpp:160`, `format_writer_integration_test.cpp:328`), so shared hash bugs can pass.

Fix by adding fault-injected filesystem tests for each publication stage and asserting temporary-file cleanup and destination preservation. Add externally generated/frozen SHA-256 vectors, including multi-chunk writes across block boundaries, and compare emitted records against the independent values.

### 23. Contain writer exceptions and preserve close/rename errors (confirmed; TASK-003)

Fallible writer APIs allocate at `writer.cpp:283`, `:430`, `:572`, and `:618` without translating allocation exceptions. `UniqueFd::reset` discards `close` errors (`:52`), including the pre-rename close at `:703`. Rename error handling reuses and clears the same `error_code` during cleanup (`:708-710`), so a failure can be reported as “Success” or as the cleanup error.

Fix by translating allocation/length failures at public `std::expected` boundaries, explicitly checking the publication close, preserving the original rename error, and reporting cleanup failure only as secondary context. Add tests for each error path.

### 24. Enforce complete schema overlap and enum validation (confirmed; TASK-002/TASK-004)

Schema validation checks individual span ends but only recognizes exact equal payload spans (`schema.cpp:1034`, `:2020`); partial payload overlap, payload/scale overlap, and scale/scale overlap pass. Several enum fields also lack rejecting defaults: state kind (`:1134`), scratch (`:1246`), graph bindings (`:1875`), shared roles (`:1987`), and integrity kind (`:2046`).

Fix by comparing all checked half-open intervals, allowing only explicitly authorized exact shared spans, and validating every enum-bearing field before switches. Add one negative test per enum family and overlap class; ensure encode and decode reject the same invalid values.

### 25. Enforce scratch precision and KV geometry invariants (confirmed; TASK-002/TASK-004)

`validate_scratch` accepts any known dtype and nonzero bytes, allowing contradictory BF16/FP32 scratch records. KV validation returns before the no-padding check, so padded `[4,256]` KV state can pass. These are manifestations of the broader state/scratch issue but require separate fixed geometry and precision rules.

Fix by mapping every scratch/state kind to its permitted dtype, byte count, rank, logical/padded dimensions, and precision policy. Apply no-padding checks to KV before returning. Add negative tests for each locked state/scratch representation.

### 26. Add complete manifest golden-byte ABI coverage (confirmed; TASK-002)

`tests/format_schema_test.cpp:187` has primitive/header golden bytes, but the manifest test at `:488` is only a self-roundtrip. Encoder and decoder could change together without detecting ABI drift.

Fix by freezing complete bytes, offsets, lengths, reserved fields, and field order for a compact manifest containing representative tensor, shared-binding, graph, state, scratch, and integrity records. Keep the golden bytes independent of the production encoder.

### 27. Preserve absolute error context and unsupported-version details (confirmed; TASK-002/TASK-004)

`decode_schema` validates with base offset zero (`src/format/schema.cpp:1833`), losing record positions. Reader passes a manifest-relative slice without its file base (`reader.cpp:432`), and validators omit tensor identity. Error enums are generic `UnknownEnum` and do not retain raw values (`src/format/constants.cpp:6`); there are no distinct quantizer/layout-version error codes.

Fix by carrying absolute manifest offsets and record start offsets through decoding, including tensor/state identity in semantic errors, retaining observed raw enum/version values, and distinguishing unsupported container, quantizer, and physical-layout versions. Add malformed-file assertions for field, tensor, and absolute offset context.

### 28. Validate graph topology and semantic scope, not only references (confirmed; TASK-004)

Graph validation (`src/format/schema.cpp:1875`) checks existence and a few node kinds but not layer ranges/patterns, node instance consistency, allowed roles for attention/GDN/MLP/MTP, graph completeness, tensor shape/layout compatibility, duplicate bindings, or actual MTP descriptors. Miniature fixtures make incomplete graphs look valid.

Fix by adding a V0 semantic validator keyed by scope and node kind, enforcing layer topology, role sets, uniqueness/completeness, shape/layout compatibility, and MTP descriptor requirements. If low-level schema fixtures must remain incomplete, separate low-level decoding from the production-ready artifact validation API.

### 29. Preserve convolution logical shape and transform metadata (confirmed; TASK-005)

`artifact_shape()` (`src/compiler/compile.cpp:78`) replaces source `[channel,1,tap]` with physical `[tap,channel]`, while `mapping_for()` (`:91`) reports `Identity`. The artifact therefore loses the source logical shape and the transpose/squeeze transform; reconstruction works only through an external hard-coded identity table.

Fix by retaining the source logical shape and encoding an explicit tap-major mapping/transform descriptor, then make reconstruction use artifact metadata rather than a separate hard-coded table. Add a round-trip assertion for logical and physical identities.

### 30. Validate excluded vision inventory exactly (confirmed; TASK-005)

`is_vision_tensor` accepts every `model.visual.*` name and only checks BF16 dtype (`src/compiler/identity.cpp:251`, `:336`). Exact vision/shard checks happen only when the total tensor count is already 1199 (`identity.cpp:352`, `checkpoint.cpp:563`). A stripped or arbitrary vision inventory can pass.

Fix by validating the complete authoritative inventory, including excluded vision tensor names, shapes, dtypes, shard placement, and occupancy, while omitting only vision payload emission. Add tests for missing, extra, wrong-shape, and wrong-dtype vision entries.

### 31. Exercise the real TASK-005 compiler path and record bounded-memory evidence (confirmed; TASK-005)

`tests/compiler_integration_test.cpp:190` uses `compile_synthetic`, bypassing config/index/shard parsing and artifact verification. The authority test opens a checkpoint and samples three tensor families (`:245`) but never invokes `compile_checkpoint`, `compile_identity`, `verify_identity_artifact`, or the CLI; peak RSS is printed but not asserted or recorded.

Fix by adding an identity-bound HF-format fixture that drives the real compiler and verifier across all included tensors, checks publication and exact reconstruction, and records peak RSS after completion. Keep the large authority test explicit/separately labeled if it cannot be part of the hermetic default suite.

### 32. Make RoPE generation independently exact and revision-stable (concern; TASK-005)

`generate_rope_inv_freq` uses implementation-dependent `std::pow` (`src/compiler/transforms.cpp:161`), while tests check only one word, monotonicity, and same-process repeatability (`tests/compiler_transform_test.cpp:105`). The full 32-word payload is not frozen.

Fix by deriving all expected FP32 bit patterns independently, freezing all 32 words, and tying the expected bytes to the compiler revision. Confirm the authoritative model’s exact convention and reject revision drift.

### 33. Add independent quantizer contraction oracles (confirmed; TASK-006)

`tests/quant_reference_test.cpp:89-95` compares explicit and packed paths that both call `dequantize_to_bf16`; `reference_gemv_packed` uses the same implementation (`src/compiler/quantization/reference.cpp:183`). Real-sample verification repeats the shared path (`tests/quant_compiler_integration_test.cpp:333`). Shared errors in scale decoding, code interpretation, BF16 rounding, or accumulation can pass.

Fix with a test-local oracle that directly indexes raw packed bytes, independently decodes FP16/BF16, and performs the declared accumulation. Freeze expected Q4, Q8, and BF16 results for asymmetric matrices and edge codes.

### 34. Add complete Q8 physical-layout golden coverage (confirmed; TASK-006)

Q4 has an independent multi-tile golden encoder (`tests/pack_layout_test.cpp:56`, `:171`), but Q8 coverage (`:216`) checks only three first-row bytes and otherwise round-trips production pack/unpack.

Fix with an independent Q8 golden encoder covering scale order, multiple N/K tiles, padding, complete code/scale bytes, and determinism, including a representative padded 16×512 case.

### 35. Make the normal TASK-006 integration test hermetic (confirmed; TASK-006)

`quant_compiler_integration` is registered unconditionally (`tests/CMakeLists.txt:149`) but hard-fails when `.cache/authorities/qwen3.8-27b-transformers` is absent (`tests/quant_compiler_integration_test.cpp:277`, `:344`). A clean checkout cannot reproduce the normal test command without an undocumented multi-gigabyte fixture.

Fix by committing a small identity-bound extracted fixture, or clearly separating/provisioning the authority test so the normal correctness suite remains hermetic and passes from a clean checkout.

### 36. Add checked geometry and exact payload checks to BF16/quantizer APIs (confirmed; TASK-006)

`pack_bf16_dense_tile_v0` and its inverse compute `elems * 2` without checked arithmetic (`src/format/pack.cpp:219`, `src/format/unpack.cpp:148`); `quantize_bf16` also multiplies unchecked (`src/compiler/quantization/quantizer.cpp:170`). `decode_bf16_payload` accepts any even payload length and ignores declared `n*k` (`src/compiler/quantization/reference.cpp:128`).

Fix all element/byte products with checked arithmetic before allocation/copy/loop, require payload length to equal the checked declared physical size, and return typed errors for overflow or mismatch. Add huge-dimension and short/long payload tests.

### 37. Make runtime session capacities and scratch sizing coherent (confirmed; TASK-007)

`Session::create` always plans attention scratch for 256 tokens (`src/runtime/session.cpp:90`) regardless of `kv_capacity`, while attention binding needs capacity-dependent segment partials (`src/runtime/attention.cpp:509`). Zero capacity can be accepted by session creation even though attention binding rejects it (`attention.cpp:308`).

Fix by validating capacity at session creation, rejecting zero/unsupported values, and sizing all capacity-dependent scratch from the actual requested capacity with checked arithmetic. Add boundary tests at minimum, maximum, and one-past-supported capacity.

### 38. Make runtime views const-correct and describe mixed workspaces truthfully (confirmed; TASK-007)

`TensorView` exposes mutable `void*` even for model views marked `writable=false` (`src/runtime/view.hpp:18`, `model.cpp:25`), so callers can still write model storage. GDN and attention workspaces are labeled uniformly FP32 (`sizes.cpp:241`) despite containing BF16 and FP32 subregions (`sizes.hpp:65`, `:110`); `Session::scratch()` reports one FP32 vector layout.

Fix by separating const model views from writable state views and representing mixed arenas as byte-region/composite typed workspace views with truthful subregion metadata. Add compile-time/API tests that model storage cannot be passed to writable operations.

### 39. Encapsulate session metadata slots and validate invariants (confirmed; TASK-007)

`kv_populated_slot()` (`src/runtime/session.hpp:60`) and `conv_cursor_slot()` (`session.cpp:345`) return unrestricted mutable pointers, bypassing checked setters (`:334`, `:354`). `save()` copies these values without invariant checks (`:202`), so callers can save `populated > capacity` or invalid cursor values.

Fix by keeping slots private to execution plans, exposing checked advance/commit operations, and validating all invariants before snapshot. Add tests for invalid mutation attempts and save/restore rejection.

### 40. Add full runtime lifetime, snapshot, capacity, and alignment coverage (confirmed; TASK-007)

The integration tests reject only a 15-byte non-artifact before allocation (`tests/runtime_session_integration_test.cpp:52-62`), sample only one S byte during snapshot/restore (`:156-163`, `:193-203`), and do not cover Runtime/Session move/destruction, zero/capacity-boundary, or actual 256-byte alignment.

Fix with full-byte state round-trip comparisons for S, history, and KV, runtime/session move and destruction-order tests, malformed-upload tests that prove no device allocation, capacity boundary tests, stable-address tests, and alignment assertions.

### 41. Reject foreign streams for session-bound GDN plans (confirmed; TASK-013)

`bind_gdn_plan(Model, Session, ...)` accepts any nonempty stream (`src/runtime/gdn.cpp:922`) without comparing it to the session’s owning stream. Session lifecycle operations use the recorded stream, so state/history updates can be queued on a foreign stream.

Fix by deriving the execution stream from the session or rejecting a stream whose native handle differs from the session’s owning stream. Add a foreign-stream rejection test and a same-stream success test.

### 42. Enforce GDN plan view types, exact shapes, and live-range disjointness (confirmed; TASK-011/TASK-013)

`as_vector` in `src/runtime/gdn.cpp:69` accepts any rank and sufficient flattened count, then overwrites shape/layout. Q4/BF16 weight and scale binding omits arithmetic dtype and scale dtype/layout/storage (`:93`, `:135`, `:149`); history accepts any sufficiently large BF16 view (`:533`). Alias checks compare only a few exact pointers (`:555`, `:881`), permitting partial overlap among residuals, workspaces, parameters, history, and output `u`.

Fix by validating exact rank/extents/layout/storage/dtype for every semantic view, checked byte counts, and all harmful byte-range overlaps. Preserve only explicitly authorized aliases. Add malformed metadata and offset-overlap tests.

### 43. Test GDN history reset and wrap semantics (confirmed; TASK-011)

TASK-011 requires a history wrap/reset test, but existing tests only inspect recurrence S (`tests/gdn_unit_test.cpp:666-729`) or alter a cursor without seeding/checking history bytes (`runtime_session_integration_test.cpp:172-185`). Stale qkv history after reset could therefore survive unnoticed.

Fix by seeding nonzero history through real front execution, advancing/wrapping the cursor, resetting the session, asserting all history bytes and cursor are zero, and verifying the next convolution observes zero-padding.

### 44. Use exact snapshot/restore comparisons for GDN state (confirmed; TASK-012)

`tests/gdn_integration_test.cpp:592-597` compares restored and uninterrupted S with floating tolerances. The lower-level session test fills only 256 bytes and checks one restored byte (`runtime_session_integration_test.cpp:125-162`, `:193-203`).

Fix by byte-comparing or hashing the complete selected GDN state after restore and requiring bitwise equality after identical deterministic continuation. Cover all heads, values, keys, and layers relevant to the plan.

### 45. Validate attention preparation views exactly and use checked arithmetic (confirmed; TASK-014)

Attention binding discards original shapes after minimum-size checks (`src/runtime/attention.cpp:71`), omits payload/scale/workspace dtype/layout/storage checks (`:95`, `:260`), and KV binding does not require `CudaBf16KvCacheV0`, BF16 storage, rank 5, or `[16,2,4,capacity,256]` (`:374`). `element_count` reads up to eight extents without bounding rank or checking multiplication overflow (`:37`), and KV byte arithmetic is unchecked (`:386`). Tests even pass rank-1 KV metadata (`tests/attention_unit_test.cpp:116`).

Fix exact rank/extents/layout/storage/dtype checks for every operand, reject rank > 8, use checked element/byte products, and add negative tests for every metadata field and overflow case.

### 46. Make attention alias validation range-aware (confirmed; TASK-014)

Runtime and CUDA preparation checks reject only exact pointer equality (`src/runtime/attention.cpp:394`, `cuda/attention.cu:377`). They do not detect partial overlap among workspace, normalized input, residual, cache, projection inputs, or weights.

Fix by deriving checked byte intervals for all simultaneously live operands and rejecting all unsafe exact and partial overlaps, with only explicitly documented aliases allowed. Add offset-overlap tests that would otherwise cause projection blocks to overwrite live preparation data or cache.

### 47. Complete full-vocabulary BF16 control coverage (confirmed; TASK-009)

The Q8 head integration runs the full `248320×5120` shape (`tests/decode_mmv_integration_test.cpp:266-285`), but BF16 control uses only `256×5120` and unrelated weights (`:287-293`).

Fix by running BF16 control at the full vocabulary shape against an independent CPU oracle using the same logical source family as the Q8 case. Verify output extent, FP32 logit storage, tile addressing, and memory use across all output tiles.

### 48. Make decode padding arithmetic checked (confirmed; TASK-009)

`decode_pad_n` performs 32-bit addition (`cuda/decode_mmv.hpp:78-84`) and `validate_geometry` trusts the wrapped result (`decode_mmv.cu:257-259`). `UINT32_MAX` can wrap to zero, allowing invalid or partially unwritten ranged launches.

Fix with checked 64-bit padding arithmetic and explicit supported-N limits. Reject overflow before launch and add single/ranged tests for maximum, overflow, and zero-tile cases.

### 49. Make benchmark and build evidence truthful and identity-bound (confirmed; TASK-001/TASK-003/TASK-009/TASK-013)

The grouped decode benchmark ignores event-record and event-synchronization errors (`benchmarks/decode_mmv_bench.cpp:234-242`). The GDN benchmark prints `launches=8` for eight mixer iterations even though each iteration currently contains nine kernels and a copy (`benchmarks/gdn_mixer_bench.cpp:202`, `:264`). Several completion reports identify mutable paths/tags rather than binary hashes or immutable source/container identities.

Fix by checking every CUDA timing operation, distinguishing iteration count from actual kernel/copy count, and recording source revision, executable SHA-256/build ID, and immutable container digest for reported measurements. Keep diagnostic benchmarks out of correctness tests.

### 50. Strengthen MLP binding metadata validation (confirmed; TASK-010)

`as_decode_vector` (`src/runtime/mlp.cpp:58`) checks pointer, memory space, arithmetic dtype, writability, and a minimum count but not physical layout or storage. Q4 scale binding (`:121`) does not require FP16 dtype or compatible layout/storage, and weight payload metadata is similarly incomplete.

Fix every MLP binding to validate exact dtype, rank/extents, physical layout, storage class, writability, and checked byte extent. Add negative tests for malformed scale dtype/layout/storage, wrong vector layout, and overlapping/undersized views.

### 51. Make attention/GDN token execution position-aware (confirmed; TASK-013)

`GdnPlan` contains no token position (`src/runtime/gdn.hpp:194`), and `execute_decode_gdn` accepts only a prebound plan (`:211`) while every invocation advances cursor/state (`src/runtime/gdn.cpp:287`). Duplicate execution for one logical token or skipped positions cannot be detected, violating the required position/session interface and once-per-token commit rule.

Fix the execution contract to carry or validate the absolute token position and session sequence. Reject repeated/skipped positions before state mutation, and commit history/recurrence exactly once after the position is accepted. Add tests for duplicate, skipped, reset, and continuation positions.

### 52. Preserve required vectorized packed-word loads or revise the contract (confirmed; TASK-009)

Although `cuda/decode_mmv.cu:91-123` uses 32/64-bit temporaries, Release `sm_120` disassembly shows four `LDG.E.U8` operations for Q4 and eight for Q8 rather than one 32/64-bit lane load. This violates TASK-009’s explicit generated-consumer requirement.

Fix the source/load forms and alignment guarantees so SASS contains the required word loads, or formally amend the contract if byte loads are the intended implementation. Add a targeted SASS assertion for the final binary and ensure the descriptor rejects unaligned spans.

### 53. Make per-head activation association tests discriminating (confirmed; TASK-008)

`tests/activation_integration_test.cpp:96` constructs every head as a positive scalar multiple of the same coordinate pattern, with identical gamma, position, and frequencies (`:103`). RMS normalization makes the heads nearly identical, and the test has no q/g tensor to detect a global-half split. A kernel that reuses or permutes heads can pass.

Fix the fixture with non-collinear, uniquely identifiable patterns and distinct q/g sentinels across head boundaries. Assert exact head-to-output mapping and both q/g halves independently.

### 54. Prove deterministic continuation across attention segment boundaries (confirmed; TASK-015)

Repeated-run and snapshot comparisons in `tests/attention_integration_test.cpp:477-480` and `:505-508` use tolerances instead of byte-exact equality. Full append→attention execution covers only lengths 1–4 (`:416-454`), while lengths 255/256/257 are tested only with a directly populated synthetic cache (`tests/attention_core_unit_test.cpp:85-136`).

Fix by asserting byte-exact repeated outputs and cache/state snapshots for deterministic inputs. Add an end-to-end append, attention, snapshot, restore, and replay sequence crossing 255→256→257, including continuation after each boundary and equality of all relevant output/cache bytes.

## P3 — minor priority and evidence improvements

### 50. Remove or correctly implement the host checked-overflow smoke path (confirmed; TASK-001)

`src/host_expected_smoke.cpp:28` performs signed addition before checking for overflow, which is undefined behavior; the overflow path is not exercised. Either reduce the smoke to a simple `std::expected` demonstration or perform a valid pre-addition bounds check and test the failure case.

### 51. Validate manifest span ordering and deterministic collection ordering (concern; TASK-003/TASK-004)

The reader sorts occupied spans before overlap checks (`src/format/reader.cpp:230`), so it accepts noncanonical directory order. The writer assigns spans in caller order (`src/format/writer.cpp:151`) and the repeat test only repeats the same vector order (`tests/format_writer_test.cpp:215`). Decide whether collection order is semantic. If not, canonicalize and validate payload/scale ordering and add equivalent-permuted-input byte identity tests.

### 52. Preserve useful semantic error offsets and identities (confirmed; TASK-002)

Semantic validation calls use offset zero (`src/format/schema.cpp:1833`), so errors for large manifests do not identify the actual record or tensor. Retain record start offsets and include tensor/state index or logical name. This is lower priority than correctness but important for repairability.

### 53. Add independent dense GEMV orientation vectors (concern; TASK-008)

CPU FP32 and double GEMV implementations use the same indexing structure (`src/reference/math.cpp:337`, `:359`), and tests compare only those implementations (`tests/reference_math_unit_test.cpp:274`). Add a hand-calculated asymmetric matrix with sentinel rows and expected outputs to detect shared transpose/stride mistakes.

### 54. Define consistent non-finite argmax behavior (confirmed; TASK-008)

CPU argmax initializes from `logits[0]` (`src/reference/math.cpp:1134`), while CUDA initializes to negative infinity and ordinary comparisons ignore NaNs (`cuda/activation.cu:203`). `[NaN,1]` yields different indices and all-NaN CUDA input can produce `UINT32_MAX`.

Fix with a typed rejection policy or explicit NaN-skipping/all-NaN error, implement it identically on CPU/CUDA, and test NaN, infinity, and all-nonfinite inputs.

### 55. Make RMS reductions numerically stable or bound their input domain (confirmed; TASK-008)

CPU and CUDA RMS paths accumulate unscaled `v*v` (`src/reference/math.cpp:57`, `:177`; `cuda/activation.cu:88`, `:109`, `:134`). Finite values near `1e20` overflow FP32 sum-of-squares, and hidden/QK multiplication can overflow before normalization. Existing adversarial tests reach only magnitude 80.

Fix with scaled sum-of-squares and an operation order that avoids unnecessary overflow, or explicitly validate/document a bounded input domain. Add threshold and finiteness tests for all RMS variants.

### 56. Make numerical comparison helpers reject NaN mismatches (confirmed; TASK-008)

`tests/activation_support.hpp:102-106` computes differences that become NaN and never updates the maximum, allowing a CUDA NaN to pass against a finite reference.

Fix the oracle to fail when finiteness differs, and only accept matching non-finite values if the operation explicitly permits them. Add finite-vs-NaN, NaN-vs-NaN, and infinity cases.

### 57. Make runtime device affinity explicit (concern; TASK-007)

`Runtime::create` records the current device (`src/runtime/runtime.cpp:7`), but later operations do not restore or verify it; `DeviceBuffer` and `Stream` store no device identity. On a multi-GPU process or changed current device, resources may belong to different devices.

Either enforce the runtime device around every resource operation and validate stream/device identity, or document and enforce a strict single-device precondition. Add a multi-device or simulated-device test if supported.

### 58. Clarify preallocation rejection and runtime compatibility boundaries (concern; TASK-007)

`Model::upload` preflights file spans before CUDA allocations (`src/runtime/model.cpp:137`), but exact language state/scratch compatibility is checked later in `Session::create` (`:75`). Decide whether runtime-incompatible but format-valid artifacts must be rejected before any device allocation. If yes, move the compatibility preflight earlier and test allocation counters.

### 59. Define long-horizon adversarial GDN coverage (concern; TASK-012)

The 128-step fixture uses moderate, decaying gates (`tests/gdn_reference_test.cpp:436-447`); near-unit gates, cancellation, sustained dynamic range, and large finite BF16 values are only isolated one-step cases (`:520-566`). Add long continuations with near-unit gates, cancellation-heavy q/k/v, and finiteness assertions if the model input domain permits them.

### 60. Make GDN Q/K normalization scale-safe or explicitly bounded (concern; TASK-012)

`cuda/gdn.cu:103-114` accumulates `v*v` directly. Large finite BF16 values can overflow FP32 and produce NaN during normalization. Use scaled normalization or define/test an input bound for prepared q/k.

### 61. Define asynchronous failure semantics for host/device continuation state (concern; TASK-011/TASK-013/TASK-014)

GDN advances its host cursor after an asynchronous convolution launch (`src/runtime/gdn.cpp:287`) without synchronization (`:685`, `:1134`). Attention restore/reset also updates host metadata before synchronization (`src/runtime/session.cpp:164-183`, `:246-259`). A deferred CUDA failure could leave host cursors/populated lengths inconsistent with device state.

Choose one explicit policy: commit metadata only after successful completion, or poison/invalidate the session after an asynchronous failure. Add injected-failure tests for reset, restore, GDN execution, and attention append.

### 62. Define Runtime/Session destruction and CUDA teardown error policy (concern; TASK-007)

`cudaFree`, `cudaStreamDestroy`, and `cudaEventDestroy` statuses are ignored (`cuda/buffer.cpp:26`, `cuda/stream.cpp:20`, `cuda/event.cpp:20`), and instrumentation records frees regardless of result. Destructors cannot return errors, so the project needs an explicit best-effort destructor policy and, if required, a fallible shutdown path with diagnostics.
