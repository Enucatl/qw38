# CPU laptop inference for Qwen3.5-2B

This chapter explains the additive Darwin/x86_64 host path that chats with one
pinned Qwen3.5-2B Q4_K_M artifact on the **Intel MacBook + AVX2** class. It
corresponds to MAC-001–MAC-014 and DOC-CPU-001–DOC-CPU-003 in the
[implementation ledger](../implementation_ledger.md). It does not replace v1
CUDA, the RTX 5090 gate, or the pinned Qwen3.8-27B identity. Apple Silicon is
not in this class.

## What this track admits

The public `Engine` / `Session` API is unchanged. On a host build without
`QW38_CUDA_RUNTIME`, `Engine::open` accepts the 2B pin after GGUF contract
validation, whole-file SHA-256, and an AVX2 CPUID check. Sessions run the
parameterized scalar GDN runtime. The 27B file still fails closed here with a
message that it needs the CUDA production build.

Default context is **4096** tokens (`--ctx`, hard cap 8192). GDN recurrence is
fixed-size; KV grows only on the six attention layers. Allocations that would
exceed the documented 10 GiB RSS budget fail closed.

## Geometry

| Field | 2B | 27B (unchanged CUDA pin) |
|---|---|---|
| Layers | 24 (18 GDN + 6 attention) | 64 (48 GDN + 16 attention) |
| Residual | 2048 | 5120 |
| FFN | 6144 | 17408 |
| Attention | 8 Q / 2 KV × 256 | 24 Q / 4 KV × 256 |
| GDN | 16/16 heads × 128 | 16/48 heads × 128 |
| Embeddings | tied `token_embd.weight` | separate `output.weight` |

The hybrid rule is shared: `layer % 4 == 3` is attention. CUDA kernels stay
27B-specialized in `cuda/*.cu`.

## Tied embeddings

Official Qwen3.5-2B sets `tie_word_embeddings: true`. If `output.weight` is
absent, logits use the token embedding matrix. 27B still requires a separate
output tensor.

## AVX2 and threads

Quant decode remains the scalar oracle used by `--check-quant`. Host matvec
rows ≥ 256 quantize the activation to Q8 and use AVX2 integer dots for
Q4_K/Q5_K/Q6_K/Q8_0. Darwin/x86_64 builds `#error` without AVX2. Host
`Engine::open` also fail-closes unless CPUID reports AVX2. The worker pool
spawns `hw.logicalcpu` threads, capped at 8, so a 4c/8t Intel MacBook stays at
eight workers and a dual-core Intel MacBook does not oversubscribe. Small
synthetic rows still use the decode-then-dot path so frozen 27B fixtures stay
bit-exact on Linux. Those kernels are a laptop speed path, not a bit-exact
substitute for the frozen 27B scalar-oracle blobs. Numeric claims for 2B must
be labeled **measured** on a named SKU.

## Measured on this SKU (MAC-010)

Hardware: Intel Core i7-8569U @ 2.80 GHz, 4 cores / 8 threads, AVX2, 16 GiB
LPDDR3. Compiler: Apple Clang 17.0.0. SHA-256 backend: `commoncrypto`. Artifact:
`Qwen3.5-2B-Q4_K_M.gguf`, 1280835840 bytes, SHA-256
`aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.

| Item | Measured |
|---|---|
| Load including whole-file SHA-256 | 8.57 s |
| TTFT, greedy, 64 prompt tokens, ctx 512 | 11.83 s |
| TTFT, greedy, 128 prompt tokens, ctx 512 | 25.49 s |
| Decode tok/s, greedy, prompt 64, 16 generated, ctx 512 | 5.61 |
| Peak RSS | 1 366 302 720 bytes (1.27 GiB) |
| Matvec workers | 8 |

The proposed ≥ 8 tok/s decode gate was not reached after AVX2 integer dots and
eight workers. The frozen laptop speed claim is the **measured** 5.61 tok/s
above, not the proposal. 512-token TTFT was not timed; 128-token TTFT scales
at about 0.20 s/token on this machine. These numbers are **this SKU only**, not
a portable SLO for every Intel MacBook + AVX2 machine.

## Proof boundary

This chapter does not claim Linux/Windows CPU product binaries, Apple Silicon,
Metal, 4B/9B, 27B-on-CPU, vision, MTP, or 128K laptop context. Those need a
later change note. The table above is **measured** on this SKU only. The
supported class is Intel MacBook + AVX2.

## 2B numeric proof ladder

Pytest on this laptop is **skip-if-missing** for the 1.28 GiB GGUF. Host pytest
stays green without the 27B file; 27B `real-*` tests still skip. Generate the
2B taps with `QW38_SCALAR_MATVEC=1` so dumps use the scalar decode-then-dot
matvec. Product `qw38` stays on AVX2.

| Rung | What it proves | What it does not prove |
|---|---|---|
| `--check-quant` Q5_K | Packed Q5_K decode/dot matches frozen hex | Product AVX2 integer dots |
| rows≥256 `--check-matvec` | AVX2 Q8-activation path vs scalar `tensor_row_dot` inside frozen max-abs/RMS | Bit-exact logits |
| Inventory row SHA + scalar dots | GGUF bytes for token_embd Q6_K, one Q5_K GDN row, one Q4_K FFN row | Full forward |
| `--check-weight-binding` | 320 tensors, tied `output.data == token_embd.data` | Numerics |
| `--tokenize-hex` | Token IDs from **this** 2B GGUF | 27B `tokenizer_authority.json` |
| `--check-real-gdn-step` / attention / FFN | Selected scalar taps for layer 0 GDN, layer 3 attention, layer 0 FFN | 27B tap hex |
| `--check-real-scalar-token` | 24 layers, 248320 logits, greedy id, structural sizes | llama.cpp |
| Darwin CPU llama.cpp same-GGUF | Independent GGUF decode/layout vs Quartz **scalar** logits on tokens `[42, 3649]` | Transformers-on-HF (cannot catch GGUF load bugs) |

The llama.cpp pin is `sources.llama_cpp.revision` `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`,
built on Darwin with `GGML_CUDA=OFF` via `tools/build_llama_cpu_oracle.sh`.
That oracle is not the CUDA 13 / sm_120 container in
`docker/llama-authority.Dockerfile`. Pytest reads
`fixtures/cpu_llama_scalar_authority.json`; it does not rebuild llama.cpp.
Logit blobs stay in gitignored `.cache/`.

Transformers eager on Hugging Face 2B is out of scope here. Frozen 27B llama
and CUDA fixtures are not regenerated.
