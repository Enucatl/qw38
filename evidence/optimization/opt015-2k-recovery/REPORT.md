# OPT-015 — 2K prefill recovery report

This is a source-and-citation recovery map for cold exact-2048-token prefill
on the pinned Qwen3.8-27B Q4_K_M GGUF and one RTX 5090. It is
**not a throughput gate** and **not llama.cpp parity**. OPT-016 owns the
later 2K parity gate. No production kernel, envelope, public API, `Makefile`,
or `plan.md` change is admitted here.

## Claim labels and proof limits

- **Measured**: copied timings from
  [`fixtures/cuda_prefill_attribution.json`](../../../fixtures/cuda_prefill_attribution.json)
  (`measurement_utc` 2026-09-08T10:46:11Z) and the first object of
  [`evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json`](../../../evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json)
  (`n_prompt` 2048, `build_commit` `cc83d7b`). Those GPU jobs are not re-run.
- **External**: pinned llama.cpp revision
  `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and local `../ds4` source
  inspection (MIT). Nsight Systems and Nsight Compute were `not_used` on
  OPT-014 and are `not_required` here.
- **Estimated**: the llama.cpp / Quartz tok/s ratio and the FFN-only ceiling.
- **Proposed**: the ranked recovery sequence. It is a map for OPT-016, not a
  claim that the sequence will pass that gate.

Proof limit: host-tested citation report; no new CUDA sample; no `nsys`/`ncu`;
ds4 cannot run this Qwen GGUF; no ds4 same-model baseline; this increment is
not a throughput gate and not llama.cpp parity.

## Measured 2K comparison

Primary Quartz number is OPT-014 exact-2048, not the scaling chat-rendered
~2052-token Quartz sample (49.25 tok/s).

| Engine | Source | Tokens | Time | tok/s | Claim |
|---|---|---:|---:|---:|---|
| Quartz production `sync_tokens` | `fixtures/cuda_prefill_attribution.json` | 2048 | 41963.8828 ms | 48.8038712 | **Measured** |
| llama.cpp `llama-bench` `-n 0` `--no-warmup` `-r 3` `-ngl 99` | scaling JSON first object | 2048 | 0.657690797 s (`avg_ns` 657690797) | 3114.049476 | **Measured** |

llama.cpp settings copied from that object: `n_ubatch` 512, `flash_attn` -1
(auto), `test_time` 2026-09-08T09:51:43Z, `build_commit` `cc83d7b`. Hardware
on both records is NVIDIA GeForce RTX 5090.

**Estimated** gap: `3114.049476 / 48.8038712 ≈ 63.8×`.

OPT-014 category shares of `wall_ms` 41963.8828 ms:

| Category | ms | Share |
|---|---:|---:|
| `ffn_mmq` | 32294.9512 | 77.0% |
| `gdn` | 5300.55371 | 12.6% |
| `attention` | 4364.3335 | 10.4% |
| `logits` | 2.86684799 | ~0.007% |
| `commit_sync` | 0.576767981 | ~0.001% |
| `embedding` | 0.076063998 | ~0.0002% |
| `graph` | 0 | 0% |
| `other_idle` | 0.5234375 | ~0.001% |

`prompt_graph_launches` is 0. `nsight_systems` / `nsight_compute` are
`not_used`.

## What pushes pinned llama.cpp into thousands of tok/s

Pinned llama.cpp `src/models/qwen35.cpp` at `cc83d7b` loads the same hybrid
Qwen3.5 contract: 64 layers, `full_attn_interval` 4, so 48 recurrent GDN
layers and 16 full-attention layers
([`qwen35.cpp:21-34`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/src/models/qwen35.cpp#L21-L34),
[`qwen35.cpp:153`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/src/models/qwen35.cpp#L153)).
The speed gap is kernel and dispatch quality on the identical GGUF, not a
different model policy. LICENSE at that revision is MIT, copyright 2023-2026
The ggml authors
([`LICENSE:1-3`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/LICENSE#L1-L3)).

1. **Quantized MMQ is MMA/tensor-core, not scalar FMA.** Ampere Q4_K MMQ tiles
   are 256 threads × 128 output rows × prompt-tile J in `{8…128}` with
   `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1` shared-memory staging
   ([`mmq-config-ampere.cuh:157-172`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/mmq-config-ampere.cuh#L157-L172);
   [`mmq.cuh:8-11,27-46`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/mmq.cuh#L8-L46)).
   Dots use tensor-core MMA PTX
   ([`mma.cuh:1-17`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/mma.cuh#L1-L17)).
   Blackwell Q4_K falls through to that Ampere table
   ([`mmq-config-blackwell.cuh:36`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/mmq-config-blackwell.cuh#L36)).
   Quartz FFN at 2048 rows selects tile 4 because `prompt_rows > 256` returns 4
   (`cuda/quant_mmv.cu:353-357`; `pins/cuda_prompt_mmq_contract.json:33-43`
   has no 2048 bucket). Production Q4_K/Q6_K still use scalar
   `__fmul_rn`/`__fadd_rn` weight-tile reuse (`docs/40-cuda-prompt-mmq.md`).
   This is the dominant reason llama.cpp reaches **thousands of tok/s**:
   `ffn_mmq` is 77.0% of Quartz 2K wall. **Estimated** FFN-only ceiling: if
   `gdn` and `attention` became free, Quartz would still be
   `2048 / (32294.9512 / 1000) ≈ 63.4` tok/s (`2048 / 32.295 ≈ 63` tok/s)
   versus llama.cpp 3114.049476 tok/s. Rank 1 is therefore mandatory before any
   other recovery can approach parity.

2. **Microbatch 512 still has enough rows for those MMA tiles.** The scaling
   `llama-bench` object uses `n_ubatch` 512. Quartz already evaluates a
   2048-row chunk (`kPromptChunkRows` 4096 in `cuda/full_scheduler.h:22`, one
   2048-token chunk in OPT-014). llama.cpp is not faster because it batches
   more prompt rows. Quartz has more rows than that ubatch and still loses on
   arithmetic intensity per weight byte (tile 4 versus J up to 128 plus MMA).

3. **Fused GDN kernel walks the ubatch in one launch per head.** llama.cpp
   `gated_delta_net_cuda` holds `S` in registers (`s_shard`) and loops
   `for (int t = 0; t < n_tokens; t++)`
   ([`gated_delta_net.cu:63`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/gated_delta_net.cu#L63)).
   Quartz still pays GDN-002 `kScanWindow = 64` windowed launches
   (`cuda/gdn_step.cu:15`) plus the OPT-013 intra/prefix/from-state overlay.
   `gdn` is 12.6% of Quartz 2K, so this is the second sink, not the first.

4. **Flash attention auto (MMA F16).** `llama-bench` `flash_attn: -1` selects
   auto. On Turing-class MMA and newer, `ggml_cuda_get_best_fattn_kernel`
   returns `BEST_FATTN_KERNEL_MMA_F16` for typical prefill batch sizes
   ([`fattn.cu:331-336,460-482`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/fattn.cu#L331-L482)),
   and `ggml_cuda_flash_attn_ext` dispatches `ggml_cuda_flash_attn_ext_mma_f16`
   ([`fattn.cu:570-583`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/fattn.cu#L570-L583)).
   Quartz attention is tiled causal GQA (OPT-005–OPT-007) without MMA.
   `attention` is 10.4% of Quartz 2K.

5. **CUDA graphs in ggml** amortize launch overhead on the ggml graph
   (`ggml_cuda_graph_evaluate_and_capture`, `cudaGraphLaunch` at
   [`ggml-cuda.cu:4012,4222`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/ggml-cuda.cu#L4012)).
   Quartz 2K `graph` is **Measured** 0 ms because 2048-row chunks do not replay
   4096-row prompt FFN graphs (`docs/62-cuda-full-prefill.md:198-200,310-311`).
   Graphs are not the present gap.

## ds4 license, cannot-run, and inspiration boundary

`../ds4` is inspected read-only as MIT technique inspiration. It is not
executed against `models/Qwen3.8-27B-Q4_K_M.gguf`.

| Identity | Value |
|---|---|
| PIN-002 / `pins/artifacts.lock.json` DwarfStar pin | `c1d4597a80e300b803dc642519718f2c999589da` |
| Inspected local HEAD (`git -C ../ds4 rev-parse HEAD`) | `c238077a87186381bf626cc531bccffe1fef79e7` |
| LICENSE blob (`git -C ../ds4 rev-parse HEAD:LICENSE`) | `5973a4c99a8b7c0f2e0a58fbcd7f93b230da6b78` |
| License text | MIT License (`../ds4/LICENSE:1-11`; copyright 2026 The ds4.c authors and 2023-2026 The ggml authors) |

The report uses the inspected HEAD as the snapshot and the PIN-002 pin as the
Quartz-locked revision. It does not require checking out the older pin.

**Cannot-run proof (source, not execution):**
`config_validate_model` accepts `glm-dsa` or else runs the DeepSeek4
validator (`../ds4/ds4.c:5809-5816`). Production GGUF architecture is `qwen35`
(`pins/artifacts.lock.json:13`; `pins/model_contract.json`). DwarfStar is a
narrow DeepSeek-V4-Flash / GLM 5.2 / DeepSeek-V4-PRO engine, not a general GGUF
runner (`../ds4/README.md:5-9`). Therefore **ds4 cannot run this Qwen GGUF** and
there is **no ds4 same-model baseline**.

ds4 `cuda/mmq/VENDOR.md` records MIT-vendored llama.cpp MMQ/MMA (upstream
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`). That is transferable kernel
technique, not a Qwen denominator. GB10 / 8xL40S / PRO 6000 ds4 numbers
remain a different model, hardware, and attention policy
(`tasks/QLT-001.md:610-664`). They may illustrate that vendored MMA MMQ moved a
*different* model's prefill; they must not be used as Quartz tok/s.

Role in this contract: `technique_inspiration_only`. `plan.md:66-68` allows
focused MIT-licensed llama.cpp and DwarfStar techniques with file-level
provenance; it forbids forking DwarfStar or translating its model-specific
machinery.

## Transferable methods

Adapt later under `plan.md:66-68` provenance, C++17/CUDA 13.0.2, `sm_120`,
visible unfused references (`plan.md:104-107`), and frozen envelopes. An
optimization cannot loosen the tolerance used to admit it (`plan.md:122-123`).
Local kernels with provenance; not a production cuBLAS/cuDNN dependency
(`plan.md:63-65`).

| Technique | Source | Quartz sink it addresses |
|---|---|---|
| MMA/shared-memory quantized MMQ (Q4_K/Q6_K, optionally Q8_0) | llama.cpp `ggml/src/ggml-cuda/mmq.cuh` + `mma.cuh` at `cc83d7b`; ds4 analog `../ds4/cuda/mmq/` (vendored llama.cpp, MIT `VENDOR.md`) | `ffn_mmq` first; mixer MMQ inside `gdn`/`attention` second |
| Fused per-head GDN token loop with register-held `S` | llama.cpp `gated_delta_net.cu` only (ds4 has no GDN) | `gdn` recurrence portion |
| Full-causal MMA/tiled multi-row attention | llama.cpp `fattn-mma-f16` (`fattn.cu`); ds4 multi-row online kernel *structure* only | `attention` history portion |
| Layer-major large-chunk prefill, allocation accounting, graph capture as launch amortization | already in Quartz (OPT-008/012) and named in `docs/sources.md` | `graph` / `other_idle` after kernel recovery |

## Non-transferable ds4 model policies

These are DeepSeek/GLM model policy or plan-forbidden product dependencies,
not Qwen3.5 semantics. Full causal attention remains mandatory. FP32 GDN
recurrence remains mandatory.

- Compressed Sparse Attention / Heavily Compressed Attention (CSA/HCA),
  time-axis KV pooling, indexer top-k (**compressed attention**)
- Sparse indexing (**sparse**)
- Sliding-window-only layers as a substitute for full KV
- MoE expert routing/streaming
- mHC
- DSpark equations
- Forking DwarfStar or copying its engine
- Treating ds4 throughput as a same-GGUF baseline
- Production cuBLAS/cuDNN dependency

`docs/03-deepseek-v4.md:178-180` already states this split: reuse
validation/ownership/allocation/hybrid scheduling; reject compressed attention,
sparse indexing, MoE routing/streaming, mHC, and DSpark.

## Quartz 2K category map

Exclusive mapping of every OPT-014 category. Percentages are of `wall_ms`
41963.8828 ms.

| Category | OPT-014 ms | Share | Faster path (or keep) | Rank |
|---|---:|---:|---|---:|
| `ffn_mmq` | 32294.9512 | 77.0% | Q4_K/Q6_K MMA MMQ with llama.cpp-style tiles (J ≫ 4), Q8_1/shared staging, SM120; keep CUD-002 unfused reference | 1 (`q4k_q6k_mma_mmq`) |
| `gdn` | 5300.55371 | 12.6% | After Rank 1, fused per-head GDN token loop (llama.cpp). Mixer Q8_0 MMQ inside this category may ride Rank 1 only if Q8_0 can be admitted without loosening the current byte-exact Q8_0 reference | 2 (`gdn_fused_token_loop`) |
| `attention` | 4364.3335 | 10.4% | Full-causal MMA flash/tiled attention; not sparse/compressed. Mixer QKV MMQ as in `gdn` | 3 (`causal_mma_attention`) |
| `logits` | 2.86684799 | ~0.007% | Keep; not a 2K recovery target | — |
| `commit_sync` | 0.576767981 | ~0.001% | Keep OPT-011 overlap | — |
| `embedding` | 0.076063998 | ~0.0002% | Keep | — |
| `graph` | 0 | 0% | Optional 2048-row prompt FFN graphs after Rank 1; not a present closer (`prompt_graph_launches == 0` because graphs are 4096-row) | 4 (`optional_2048_prompt_graphs`) |
| `other_idle` | 0.5234375 | ~0.001% | Keep as remainder; do not invent a ninth GPU interval | — |

**Estimated** FFN-only ceiling restated: if `gdn` and `attention` became
free, Quartz would still be ~63 tok/s versus llama.cpp 3114 tok/s. Rank 1 is
mandatory.

## Ranked recovery sequence

**Proposed** sequence for OPT-016. This report does not claim the sequence
will hit llama.cpp 2K tok/s; it ranks where the **Estimated** 63.8× lives.
OPT-016 remains the throughput gate: cold 2K Quartz tok/s ≥ pinned llama.cpp
`llama-bench` 2K on this GGUF and RTX 5090, envelopes unloosened.

1. **`q4k_q6k_mma_mmq`** — Replace production Q4_K/Q6_K prompt MMQ with
   llama.cpp-style MMA/shared-memory tiles under MIT file-level provenance
   (`mmq.cuh`, `mma.cuh` at `cc83d7b`; ds4 `cuda/mmq` as adapter-without-ggml
   inspiration, not wholesale vendoring in this task). Retain
   `launch_quant_mmq` / CUD-002 as the visible reference. Admit against the frozen
   CUD-002 envelope (max abs `5e-4`, RMS `2.5e-4`, exact Q8 staging, zero
   non-finites). Do not edit `Makefile` `--fmad=false` for the reference path.
   Q8_0 stays on the byte-exact OPT-009 tiled kernel unless a later increment
   freezes a new Q8_0 envelope *before* swapping; do not loosen the
   byte-exact gate in place.

2. **`gdn_fused_token_loop`** — Fuse the prompt GDN recurrence into a per-head
   token loop with register-held `S` (llama.cpp `gated_delta_net.cu`), still
   FP32, still compared to sequential GDN-002 windows at max abs `5e-8` /
   RMS `5e-9`. Keep OPT-013 overlay scan as an alternate until the fused loop
   wins a paired CUDA-event A/B. ds4 has no GDN analog.

3. **`causal_mma_attention`** — Replace the remaining attention-history cost with
   full-causal MMA/tiled multi-row attention (llama.cpp `fattn-mma-f16`). Keep
   ATN-002/OPT-005–007 unfused/tiled references and frozen envelopes. Reject
   CSA/HCA/top-k.

4. **`optional_2048_prompt_graphs`** — Only after Rank 1, and only if launch
   overhead reappears in `other_idle` or `graph`. Capture 2048-row FFN
   graphs beside the existing 4096-row executables. Not a 2K closer today
   (**Measured** `graph` 0 ms).
