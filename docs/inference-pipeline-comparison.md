# Quartz and llama.cpp: component-by-component inference review

Reviewed 2026-09-19. [Interactive structural map](inference-pipeline-comparison.html).

The engines implement the same hybrid decoder equations, but they do not execute the same finite-precision calculation or GPU schedule. The strongest remaining structural mismatch is long-context decode attention: Quartz's selected OPT-137 kernel computes the useful QK and PV results with scalar arithmetic, whereas llama.cpp's selected MMA path computes both with matrix instructions. Quartz's Q4 decode FFN, by contrast, already uses a llama-derived implementation. "Replace the matrix multiplication kernels" is therefore too broad a diagnosis.

This is a source review with retained benchmark evidence, not a new performance measurement or a proposed selector change. Differences below are established from the reachable code; their contribution to current end-to-end latency requires a matched measurement.

## Scope and identity

| Item | Reviewed identity |
| --- | --- |
| Quartz checkout | `24ab9bc0e9d6c76f125fd67d9786b6428e9b2989` |
| Neighboring llama.cpp checkout | `1945e092030f8668ff93382799502d01490e564d` |
| llama.cpp in retained optimization benchmarks and most adapters | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| Model | Pinned `Qwen3.8-27B-Q4_K_M.gguf`; GGUF architecture `qwen35` |
| GGUF SHA-256 | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| Intended runtime | One RTX 5090, CUDA; Quartz capacity 131,072 tokens |
| Comparison assumptions | Same GGUF and raw tokens, dense text trunk, no MTP/speculation, one sequence, resident GPU weights |

Both source trees were clean before this review. The artifacts added by the review are documentation only. The neighboring checkout is newer than the benchmark authority: historical rates must not be presented as measured rates for these two current revisions. llama.cpp backend choices are conditional on hardware, configuration, shapes, and fusion eligibility. References to its F16 KV describe its default configuration, also used in the cited native attention comparisons; they are not a restriction of llama.cpp.

Model identity comes from [artifacts.lock.json](../pins/artifacts.lock.json), [model_contract.json](../pins/model_contract.json), and [tensor_inventory.json](../pins/tensor_inventory.json). The applicable llama model is [qwen35.cpp](../../llama.cpp/src/models/qwen35.cpp), not the MoE `qwen3next` implementation.

## Shared structure and dimensions

```text
token IDs -> embedding -> FP32 residual x
  repeat [GDN layer, GDN layer, GDN layer, attention layer] 16 times:
    u = RMSNorm(x)
    r = x + Mixer(u)                     # GDN or full attention
    z = RMSNorm(r)
    x = r + Wdown(SiLU(Wgate z) * Wup z) # dense FFN
  -> final RMSNorm -> vocabulary projection -> logits -> sampling
```

The schematic omits rounding; the following sections restore those boundaries. Every one of the 64 layers has an FFN. The 48 GDN layers replace the attention mixer, not the FFN.

| Quantity | Shape / value |
| --- | --- |
| Residual | 5,120 values per token |
| FFN intermediate | 17,408 values per token |
| Full attention | 24 query heads, 4 KV heads, width 256; six Q heads share each KV head |
| Rotary dimensions | First 64 of 256 dimensions; 32 rotary pairs |
| GDN | 16 Q/K heads, 48 value heads, Q/K/V head width 128 |
| GDN convolution | 10,240 channels, depthwise width 4 |
| GDN recurrent matrix | 48 × 128 × 128 FP32 values per GDN layer |
| Vocabulary | 248,320 logits |
| Resident weight families | Q4_K embeddings and FFNs; Q8_0 GDN projections and attention Q/gate/K/V; Q6_K attention output and vocabulary head |

Decode processes one token using existing state. Prefill applies the same layer structure to many token rows: projections become matrix-matrix operations, attention is causal across the rows, and GDN still carries its recurrence forward in time. Quartz's prompt capacity and selected microbatch are now 4,096 rows, not the historical 64-row scheduler described in early documentation.

Structural anchors: Quartz [full_scheduler.cu](../cuda/full_scheduler.cu), `enqueue_decode_layer_eager` (line 3761), `execute_token` (5519), `execute_prompt_chunk` (6445); llama [qwen35.cpp](../../llama.cpp/src/models/qwen35.cpp), graph constructor (133), attention (254), GDN (335), FFN (470).

## Component comparison

### 1. Loading and converted parameter meaning

Both engines consume the same packed GGUF weights. The file already contains folded GDN `A = -exp(A_log)` and direct RMSNorm scales. Ordinary source RMSNorm offsets have already had one added by the converter; GDN gated-norm scales are already direct. Both runtimes multiply the stored scales and use the stored negative A. There is no evidence here of a second exponential or a second norm offset being applied in Quartz.

The model's GGUF value-head order is `[replica, key_head, lane]`. Quartz's canonical GDN state uses `[key_head, replica, key_lane, value_lane]`, with explicit mappings at gates, V selection, and gated output. llama's fused recurrence stays in the tiled head order and selects the Q/K head with `value_head % 16`. Quartz uses grouped `value_head / 3`, then maps the value head back to `replica * 16 + key_head` when reading projected V. These indexing expressions are equivalent only after the head permutation; raw state bytes cannot be compared directly.

Sources: [conversion.cpp](../src/conversion.cpp), [conversion explanation](22-gguf-conversion.md), [scheduler_primitives.cu](../cuda/scheduler_primitives.cu) `prepare_gates` / `gated_output`, and llama [gated_delta_net.cu](../../llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu) `iq1` and state offsets.

### 2. Embedding lookup: the first rounding difference

Quartz decodes the selected quantized embedding row into BF16, then widens it into the FP32 residual. llama's ordinary graph uses an FP32 embedding result from GGML row lookup. Widening BF16 to FP32 does not restore the bits discarded during the lookup.

```text
Quartz: dequantize(row) -> round_BF16 -> widen_FP32 -> residual
llama:  dequantize(row) -> FP32 graph activation
```

This is a numerical difference before layer 0. Consequently, replaying the same token ID is insufficient to isolate a projection kernel's error: feed both kernels the same captured activation as well.

Sources: [quant_mmv.cu](../cuda/quant_mmv.cu) `quant_row_decode` (564), [full_scheduler.cu](../cuda/full_scheduler.cu) embedding section (5653 onward), llama [getrows.cu](../../llama.cpp/ggml/src/ggml-cuda/getrows.cu) and [qwen35.cpp](../../llama.cpp/src/models/qwen35.cpp) `build_inp_embd`.

### 3. RMSNorm and activation staging

The shared real-number equation is `y_i = x_i * w_i / sqrt(mean(x²) + 1e-6)`. Quartz selects `parallel_fma`, 256 threads for residual norms and 32 for GDN output norms. It explicitly rounds the result to BF16 before projection staging. llama's ordinary norm/scaling output remains FP32; its CUDA backend can fuse norm, scale, and eligible RoPE/cache operations.

```text
Quartz decode: FP32 residual -> norm -> BF16 -> transient Q8 -> projection
llama decode:  FP32 residual -> norm -> FP32 -> transient Q8 -> projection
```

These transient Q8 activations are separate from the Q8_0 resident weights. Identical dot-product instructions will not make the pipelines numerically identical if the input was already rounded differently.

Quartz prefill already fuses mixer/FFN input norm with MMQ Q8 staging, preserving BF16 rounding in registers. Decode norm-to-Q8 fusion remains disabled. Thus an extra arithmetic rounding boundary need not imply an extra global-memory buffer in every phase.

Sources: [rms_norm.cuh](../cuda/rms_norm.cuh), [full_scheduler.cu](../cuda/full_scheduler.cu) `store_norm_bf16_and_q8` (1331), [full_scheduler.h](../cuda/full_scheduler.h) selected fusion flags, [opt149_norm_q8.cuh](../cuda/opt149_norm_q8.cuh) selected flag (50), llama [norm.cu](../../llama.cpp/ggml/src/ggml-cuda/norm.cu) and [ggml-cuda.cu](../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu) norm fusion matching (2692, 3218 onward).

The retained [OPT-149 screen](../evidence/optimization/opt149-norm-q8/REPORT.md) found fused decode staging slower for the complete mixer family: 2.1010 → 2.2556 ms. Removing a launch is not by itself proof of a win.

### 4. Quantized decode projections

| Family | Quartz selected calculation | llama CUDA counterpart | Remaining distinction |
| --- | --- | --- | --- |
| Mixer Q8_0 | `dp4a_q8_1`, grouped input projections, one row / four warps, raw GGUF weights | Quantized MMVQ with backend-selected geometry/fusions | BF16 input boundary, staging/reduction and launch organization |
| FFN Q4_K | `llama_q4k_mmvq` adapter; fused gate/up/SwiGLU, then down | Native Q4_K MMVQ and eligible GLU fusion | Quartz adapter consumes BF16, emits BF16 SwiGLU, derives from older llama revision |
| Attention output / vocabulary Q6_K | `integer_q8_1`, two warps per row, raw GGUF weights | Native Q6_K MMVQ | BF16 gated-attention or final-norm input, reduction tree and launch geometry |

Quartz groups four input projections for GDN (QKV, Z, alpha, beta) and three for attention (Q+gate, K, V), staging the input once. Output projections remain separate. Its selected Q4 path takes precedence over the older `paired_integer` FFN selector. Reading that latter selector alone gives the wrong picture of production execution.

The Q8 staging formats also need explicit identities:

```text
d = max(abs(x_i)) / 127; q_i = round(x_i / d)
Quartz general decode Q8_1: half(d), half(sum(q_i)), int8 q_i
llama native block_q8_1:   half(d), half(sum(x_i)), int8 q_i
Quartz OPT-110 adapter:    native block_q8_1 fields, but x has BF16 rounding
```

Equal 36-byte sizes do not make these representations interchangeable. However, the selected Q8_0 and Q6_K dot products use the scale and codes, not the sum field; the sum convention is not automatically an output error in those consumers. The selected Q4 adapter reconstructs the needed integer sums in its dot routine. Scope any diagnosis to the actual consumer.

Sources: [q8_decode_path.cuh](../cuda/q8_decode_path.cuh) (30), [q4k_decode_path.cuh](../cuda/q4k_decode_path.cuh) (79), [q6k_decode_path.cuh](../cuda/q6k_decode_path.cuh) (20), [full_scheduler.cu](../cuda/full_scheduler.cu) `launch_q8_mixer_input_group` (1862), `execute_ffn` (2175); [q4k_decode_dots.cuh](../cuda/q4k_decode_dots.cuh), [q8_decode_dots.cuh](../cuda/q8_decode_dots.cuh), [q6k_decode_dots.cuh](../cuda/q6k_decode_dots.cuh), [opt110_llama_q4_adapter.cu](../cuda/opt110_llama_q4_adapter.cu); llama [quantize.cu](../../llama.cpp/ggml/src/ggml-cuda/quantize.cu) (54), [mmvq.cu](../../llama.cpp/ggml/src/ggml-cuda/mmvq.cu), [vecdotq.cuh](../../llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh).

### 5. GDN gates and convolution

Both compute the following, using the stored negative A:

```text
g = A * softplus(alpha + dt_bias)
beta = sigmoid(beta_projection)
[q_raw, k_raw, v] = SiLU(depthwise_conv4(projected_QKV, history))
q = q_raw / sqrt(sum(q_raw²) + 1e-6)
k = k_raw / sqrt(sum(k_raw²) + 1e-6)
```

Quartz uses explicit rounded multiply/add operations in convolution, a thresholded stable softplus, and grouped gate ordering. llama builds these operations as FP32 graph nodes; its backend has SSM-convolution/SiLU fusion. Quartz also combines convolution and SiLU in its convolution kernel, so graph-node counts must not be mistaken for kernel counts.

Quartz stores four convolution-history entries per channel; llama's persistent convolution history is `kernel_size - 1`, or three entries, and constructs the convolution input with the new rows. This is a state representation difference, not a four-versus-three-tap convolution difference.

llama's `build_gdn_l2_norm` expresses L2 normalization using RMSNorm with `eps / width`, followed by `1 / sqrt(width)`. In real arithmetic it equals Quartz's `1 / sqrt(sum(x²) + eps)`; the different reduction/scaling order can change FP32 rounding.

Sources: [scheduler_primitives.cu](../cuda/scheduler_primitives.cu) `prepare_gates`, [gdn_step.cu](../cuda/gdn_step.cu) convolution kernels, llama [qwen35.cpp](../../llama.cpp/src/models/qwen35.cpp) (357 onward), [models.h](../../llama.cpp/src/models/models.h) `build_gdn_l2_norm` (14), [delta-net-base.cpp](../../llama.cpp/src/models/delta-net-base.cpp) `build_conv_state` (449).

### 6. GDN decode recurrence: same equation, different parallelism

For one head, let `S` be the 128 × 128 state, with key dimension first:

```text
a      = exp(g)
Sdecay = a * S
delta  = beta * (v - kᵀ Sdecay)
Snew   = Sdecay + k deltaᵀ
o      = qᵀ Snew / sqrt(128)
```

| Detail | Quartz shipping decode | llama fused CUDA GDN |
| --- | --- | --- |
| Persistent matrix layout | Row-major `S[key, value]` | Transposed `S[value, key]` |
| Work ownership | One thread per value column, serial loops over 128 keys | One warp per value column; four key values per lane |
| Grid for 48 heads / one sequence | 48 head CTAs, 128 active value lanes | 48 × 32 column-group CTAs, four warps each |
| Q/K inverse | Lane 0 serially computes sums for each value-head CTA | Separate graph normalization, parallel backend kernels |
| Prediction | Sum of `k_i * (a * S_ij)` | Warp sum of `k_i * S_ij`, then multiply by a |
| Query scale | Incorporated in query inverse before output dot | Applied after the output reduction |
| State update | Explicit multiply/add; reloads state across two source loops | Register-held state shards; multiply/add may contract |

The arithmetic rearrangements are equivalent over real numbers, not bitwise FP32 identities. The layout and work ownership differences are substantial, but do not establish how much of today's whole-engine gap they explain.

Quartz's transposed variants exist but are **not selected**: `kSelectedGdnDecodePath = "sequential"`. [OPT-109](../evidence/optimization/opt109-persistent-gdn-state/REPORT.md) is a useful warning: isolated recurrence improved from 0.02592 to 0.01102 ms, but the complete 48-layer enclosing path regressed from 13.190 to 13.993 ms in its older `ffn_only` graph setting. That result does not justify enabling the variant on today's graph stack without a new complete-family measurement.

Sources: [gdn_decode_path.cuh](../cuda/gdn_decode_path.cuh) (41), [gdn_step.cu](../cuda/gdn_step.cu) `prepare_recurrence_window` (103), [gdn_decode_recurrence.cuh](../cuda/gdn_decode_recurrence.cuh), llama [gated_delta_net.cu](../../llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu) (6).

### 7. GDN prefill and output gate

Do not extend the decode description to prompt execution. Quartz normally selects `kFusedTokenLoop`, with preprocessed Q/K and decay and a temporary transposed state (`kSelectedGdnPreprocPath = "transpose"`). It transposes the canonical row-major state into scratch, loops through token rows with the optimized recurrence, and transposes the final state back. Convolution and gated output are separate on the selected `kSelectedGdnFusePath = "off"` route. Fallbacks exist for shape/scratch constraints.

llama normally enables fused GDN for both single-token and multi-token execution, subject to backend support probing. Its fused kernel loads transposed state into registers, walks token time sequentially, and writes the final state or requested snapshots. It also has a separate graph-based chunked algorithm when fused chunk GDN is unavailable/disabled. It would be incorrect to describe all llama prefill as a parallel scan.

Both finish with `RMSNorm(o) * SiLU(z)`, then the mixer output projection. Quartz maps back into GGUF tiled head order and rounds the gated output to BF16; llama's corresponding graph activation is FP32.

Sources: [full_scheduler.h](../cuda/full_scheduler.h) prompt defaults (1173), [gdn_fused_quality.cuh](../cuda/gdn_fused_quality.cuh) (34), [gdn_step.cu](../cuda/gdn_step.cu) `launch_gdn_quality_fused` (1009), [full_scheduler.cu](../cuda/full_scheduler.cu) prompt GDN dispatch (6862); llama [delta-net-base.cpp](../../llama.cpp/src/models/delta-net-base.cpp) `build_delta_net` (426), [llama-context.cpp](../../llama.cpp/src/llama-context.cpp) fused-GDN initialization and probes (232, 562).

### 8. Attention projections, RoPE, and persistent KV

Both split each head's packed `[Q, gate]`, apply per-head Q/K RMSNorm, rotate the first 64 dimensions, and use causal grouped-query attention:

```text
p = softmax(Q Kᵀ / sqrt(256) + causal_mask)
o = (p V) * sigmoid(gate)
mixer_output = Wo o
```

Quartz uses its fixed text-only partial RoPE parameters (`theta = 10,000,000`). llama represents positions through configurable multi-section RoPE. For the matched text positions and model settings these encode the same intended rotary transformation; this review does not extend that equivalence to multimodal positions or arbitrary RoPE overrides.

Quartz persists dense BF16 K/V and reads the pending token from a separate candidate row. llama defaults to F16 K/V in its hybrid KV memory. These are both two-byte formats, but BF16 has fewer significand bits and a larger exponent range. Quartz long-decode/prefill attention converts BF16 cache operands to F16 tiles for the relevant kernels; that does not recover precision previously lost to BF16. Its short flash-vector path consumes BF16 directly. The corresponding state, loads, conversion cost, and operand arithmetic therefore differ.

The dense Quartz KV payload at capacity 131,072 is `16 layers * 2(K,V) * 4 heads * 256 * 131072 * 2 bytes = 8 GiB`, before candidates and scratch. This is allocated capacity, not the number of rows every attention invocation should scan.

Sources: [attention_decode.cu](../cuda/attention_decode.cu), [opt120_packed_kv.cuh](../cuda/opt120_packed_kv.cuh) `kDenseBf16` (46), [opt148_short_decode_flash.cuh](../cuda/opt148_short_decode_flash.cuh); llama [qwen35.cpp](../../llama.cpp/src/models/qwen35.cpp) attention (254), [llama-context.cpp](../../llama.cpp/src/llama-context.cpp) default cache types (3642).

### 9. Decode attention dispatch and useful arithmetic

| Region | Quartz selected route | llama route under the matched CUDA/F16/GQA6 conditions |
| --- | --- | --- |
| Zero-based position 0–7935 | OPT-148 flash vector, 16 partitions | Flash vector |
| Position 7936–8191 | OPT-148 flash vector, 16 partitions | F16 MMA: padded used KV length has reached 8192 |
| Position ≥8192 | OPT-137 `dense_bf16_tile_f16_mma_decode_v1` | F16 MMA |

Quartz attends over `position + 1` rows. llama's relevant selector examines padded **used** KV length, generally a multiple of 256 on this path, not allocated capacity. The numeric threshold 8192 therefore does not designate the same transition. OPT-152 extended flash-vector coverage to all Quartz positions below 8192; the older `verified_max = 4096` constant does not limit that selected coverage overlay.

The OPT-137 name overstates its similarity to llama. The actual output calculation includes:

```cpp
local = warp_sum(local);                         // scalar FP32 QK
mma_qk_eight(..., mma_scores);
local = __fadd_rn(local, 0.0F * mma_scores[0]);    // no useful MMA score
// online softmax updated for each KV row
vkq[i] = vkq[i] * rescale + weight * v_f;         // scalar PV
```

For finite intermediates the MMA term contributes zero. Its presence in compiled SASS is not evidence that tensor cores produce attention scores. llama's F16 MMA kernel uses the QK matrix-product accumulators for scores and the PV matrix-product accumulators for the output, with tiled softmax rescaling and shared KV across GQA heads. Quartz OPT-137 instead performs per-row rescaling, scalar QK/PV, BF16-to-F16 tile conversion, and partition merging. Those are concrete computational differences.

This is already corroborated by [OPT-150's dataflow audit](../evidence/optimization/opt150-attention-dataflow/REPORT.md). [OPT-151](../evidence/optimization/opt151-qk-pv-mma/REPORT.md) attempted useful MMA for both products but lost its screen: complete 16-layer attention time was 3.252 → 6.304 ms at D8192 and 13.804 → 38.466 ms at D32768. It remains disabled. The missing work is an efficient complete matrix-based attention pipeline, not simply making an MMA instruction appear or flipping OPT-151 on.

Sources: [attention_decode_path.cuh](../cuda/attention_decode_path.cuh) selected flags (300–327), [opt137_dense_mma_decode.cuh](../cuda/opt137_dense_mma_decode.cuh) useful dot/zero-weight helper/PV (236–261), [opt148_short_decode_flash.cuh](../cuda/opt148_short_decode_flash.cuh); llama [fattn.cu](../../llama.cpp/ggml/src/ggml-cuda/fattn.cu) selector (591), [fattn-mma-f16.cuh](../../llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh).

### 10. Prefill attention and projections

Quartz prompt attention selects OPT-147 `prefill_attention_8x8_v1`: eight query positions and eight GQA columns, with six real heads and two masked columns. It carries the OPT-111 arithmetic and BF16 state boundary. This prompt path is distinct from the scalar-result OPT-137 decode kernel; the long-decode finding must not be generalized to prompt attention.

Quartz prompt projections use packed integer MMA MMQ, with large Q8 mixers on `quality_mma`, skinny projections on a smaller tile, and Q4 FFN tiles selected at 128 × 128. Gate and up share staged activations but use separate selected prompt launches. SwiGLU writes down-projection Q8 staging directly, preserving BF16 rounding semantics. The selected pipeline is `fma_async`, asynchronous X loading enabled, MMQ stream-K disabled.

llama also stages quantized MMQ operands and uses integer matrix products, with backend/device-dependent tile and scheduling decisions. For these types, both use DS4 scale/sum layout for Q4_K prompt staging and D4 for Q8_0; the Quartz **decode** `sum(q)` convention must not be applied to its MMQ DS4 staging. The remaining comparison is input rounding, packing, tiling, asynchronous overlap, and complete consumer cost, not "tensor cores versus no tensor cores" for prompt MMQ.

Sources: [fattn_mma_f16.cuh](../cuda/fattn_mma_f16.cuh) pipeline pin (70), [fattn_mma_f16_pipeline.cuh](../cuda/fattn_mma_f16_pipeline.cuh), [quant_mmq_mma.cuh](../cuda/quant_mmq_mma.cuh) selected pipeline (2354), FFN tiles (3405), staging reuse (3569), [full_scheduler.cu](../cuda/full_scheduler.cu) `execute_prompt_ffn_projections` (1973); llama [mmq.cuh](../../llama.cpp/ggml/src/ggml-cuda/mmq.cuh) layout mapping (60), [quantize.cu](../../llama.cpp/ggml/src/ggml-cuda/quantize.cu) MMQ staging (458).

The [OPT-147 keep](../evidence/optimization/opt147-prefill-8x8/REPORT.md) reduced complete prompt-attention family time by about 20.7 ms, while whole-prefill throughput improved about 1.16%. Component and whole-engine gains have different denominators.

### 11. Residuals, FFN output, and compiler arithmetic

Both implement two residual additions per layer and dense SwiGLU. Quartz's main residual stream is FP32; describing the whole engine as BF16 would be wrong. Its extra BF16 boundaries are at embeddings, normalized projection inputs, mixer gated outputs, and FFN activated inputs to down projection. llama's normal graph keeps these intermediate activations FP32 before its backend quantization or attention operand conversions.

Quartz fuses an FFN residual add with the next layer's input norm on the eligible path. Its general CUDA build uses `-O2 --fmad=false`, with many explicit `__fmul_rn`/`__fadd_rn` operations. Selected helpers use explicit FMA, and the OPT-110 Q4 translation unit separately uses `-O3 --use_fast_math`. llama's CUDA CMake adds `-use_fast_math`; expressions can contract and transcendental implementations differ. This is a family-specific arithmetic policy, not one uniform "strict Quartz / fast llama" division.

Sources: [full_scheduler.cu](../cuda/full_scheduler.cu) residual and FFN epilogues, [scheduler_primitives.cu](../cuda/scheduler_primitives.cu) SwiGLU and gated output, [Makefile](../Makefile) (6, 559), llama [CUDA CMakeLists.txt](../../llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt) (195).

### 12. Final logits and sampling

Quartz applies final RMSNorm, rounds to BF16, stages the vector, then computes all 248,320 Q6_K logits. Prefill normally computes vocabulary logits only for the final prompt row. llama selects requested output rows around its final normalization/head; a benchmark requesting all prompt logits performs different work.

Quartz's selected lazy output policy keeps outputs resident and computes a device greedy argmax for the greedy route. Non-greedy sampling materializes logits for the host sampler. Full-logit D2H transfer is therefore not an unconditional per-token cost. llama also has configurable sampling/output behavior; compare identical output and sampling requirements, or isolate raw decode evaluation.

Sources: [full_scheduler.cu](../cuda/full_scheduler.cu) `commit_outputs` (5033), decode logits (6109 onward), final prompt row (7191), `greedy_sample` (7437); [engine.cpp](../src/engine.cpp) `Session::sample` (316), llama [qwen35.cpp](../../llama.cpp/src/models/qwen35.cpp) final output construction (202 onward).

### 13. Graph execution, state publication, and host work

| Concern | Quartz | llama.cpp |
| --- | --- | --- |
| Execution description | Explicit fixed-model scheduler | Model graph + backend scheduling/fusion |
| Decode replay | Eight captured segments, eight layers each; selected topology follows context bucket | Eligible CUDA backend graph replay; topology/property changes can require update/rebuild |
| Outside Quartz decode segments | Embedding, final norm/head, output handling, state commit | Depends on backend graph partition and output extraction |
| Live inputs | Device launch state carries token/position/state selection | Graph input tensors and backend graph update machinery |
| GDN publication | Separate candidate/committed buffers, slot swap after success | Hybrid recurrent memory and graph state-copy/snapshot operations |
| KV publication | Read candidate during evaluation, scatter across layers at commit | Graph KV write operations in hybrid memory |
| Cancellation | With a poll callback, selected code synchronizes device after each segment | Separate context/backend abort and scheduling semantics |

Quartz still creates timing events per token, synchronizes the stop event, scatters candidate KV, and synchronizes at commit. Additional segment poll synchronizations occur only when a non-null callback is supplied; `Session::eval` omits it when cancellation is not requested. These differences affect host/GPU overlap and service latency without changing model equations.

Do not count candidate buffers as proof that Quartz copies the full GDN state once more at each decode commit: the selected implementation swaps GDN slots. Conversely, do not assume llama gives exactly Quartz's cancellation/checkpoint transaction guarantee. A speed comparison must keep service requirements visible rather than silently dropping them.

Sources: [execution_graph_path.cuh](../cuda/execution_graph_path.cuh), [decode_launch_state.cuh](../cuda/decode_launch_state.cuh), [full_scheduler.cu](../cuda/full_scheduler.cu) poll (97), graph replay (5681), output/state commit (6180 onward), slot swap (4842); [engine.cpp](../src/engine.cpp) evaluation (444); llama [ggml-cuda.cu](../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu) capture/replay (4177, 4412), [delta-net-base.cpp](../../llama.cpp/src/models/delta-net-base.cpp) recurrent publication (527), [llama-memory-hybrid.cpp](../../llama.cpp/src/llama-memory-hybrid.cpp).

## What the retained timings establish

The [OPT-146 matched run](../evidence/optimization/opt146-batch-reconciliation/REPORT.md) measured the post-137 stack against the pinned older llama revision. It predates the OPT-147/148/152 keeps:

| Workload | Metric | Quartz tok/s | llama tok/s |
| --- | --- | ---: | ---: |
| D128 | decode only | 59.46 | 70.42 |
| D2048 | decode only | 50.06 | 69.56 |
| D8192 | decode only | 48.56 | 66.85 |
| D32768 | decode only | 33.98 | 62.34 |
| P4096 | prefill | 2996.22 | 3223.78 |

Later [OPT-152](../evidence/optimization/opt152-vector-coverage/REPORT.md) reports Quartz control/candidate gains at D512 (57.72 → 60.38 tok/s) and D6144 (39.20 → 58.19), with roughly 59.71 at D2048 and 34.05 at D32768 in its guards. These are coverage-overlay A/B results, not a fresh head-to-head comparison with the neighboring checkout. They do show why the older D2048 rate cannot stand in for today's selected stack.

OPT-150 contains native-kernel, BF16-adapter, complete-family, and whole-engine timings with different scopes, including a missing D8192 llama replay result after an illegal-memory-access failure. Those numbers cannot be combined into a single universal attention multiplier. OPT-151's report also lacks raw per-pair arrays for independently reconstructing its screen means. This review uses those reports to qualify existing decisions, not to manufacture a new causal attribution.

## A concrete structure for the next comparison

The useful unit is a complete producer-to-consumer boundary, with a common input and an explicit output representation. Existing [component replay](../cuda/optimization_component_replay.cu), [engine probe](../cuda/optimization_engine_probe.cu), and [kernel parity helpers](../cuda/kernel_parity.cuh) provide starting points.

| Priority | Boundary to compare | What must be held equal / recorded | Question it resolves |
| --- | --- | --- | --- |
| 1 | Long attention: prepared Q + persistent/candidate KV → gated output | Actual live and padded lengths; cache/operand types; prep, conversion, QK, softmax, PV, merge; output-producing instructions | Where Quartz's scalar-result pipeline loses, and why OPT-151 also lost |
| 2 | Short attention + dispatch transitions | Positions 7935/7936/8191/8192; all-short flash coverage; actual selected kernel and partition count | Cost of the remaining threshold/layout mismatch |
| 3 | Norm → staging → all mixer projections | Same FP32 input, then separate production-BF16 and matched-FP32 experiments; complete family timing | Rounding cost versus staging/launch cost; revisit OPT-149 only with a new mechanism |
| 4 | GDN convolution → recurrence → gated output → projection | Head permutation, matrix transpose, state layout lifetime, same graph mode, complete-family time | Whether current segment graphs change OPT-109's enclosing loss |
| 5 | P4096 prompt attention and MMQ | Same chunk rows, final-row logits policy, DS4/D4 representation, tails, conversion and synchronization | Remaining prefill cost after OPT-147 |
| 6 | Whole decode transaction | Exact binaries and token IDs; output mode, cancellation callback, graph bucket, commit and sampling | Whether component improvements survive service-level costs |

For each boundary report both numeric comparison and latency. Numeric comparison should align head permutations and transposes, distinguish error from input rounding versus kernel arithmetic, and include recurrent continuation, not just one isolated token. Timing should include an uninstrumented paired whole-engine run and a separately instrumented disjoint wall-time account; overlapping CPU waits and GPU kernels must not be added together.

No shipping selector should be changed on this review alone. The source supports a clear order of investigation, especially the useful arithmetic in long attention, but not an exact predicted speedup or a claim that a previously rejected candidate is now faster.

## Review artifacts and verification

The diagram is an architecture overview of the two paths. Its arrows show layer-stage order, with mixer and FFN repeating across 64 layers; detailed GDN/attention equations and dispatch exceptions live in this review. Archify was used to render and validate the paired structure.

The review was checked against source dispatch and call sites, not just kernel filenames or historical documentation. No inference code was changed and no new GPU benchmark was run. Diagram validation passed all nine showcase checks with zero composition errors or warnings. Delivery hashes and browser review are recorded in the [artifact receipt](inference-pipeline-comparison.receipt.json).
