#pragma once

// OPT-051 sibling of fattn_mma_f16.cuh. Prepared-Q prompt attention inner loop
// with staged F16 operands, register softmax, cp.async KV staging, and optional
// six-head GQA. Adapts llama.cpp fattn-mma-f16.cuh KQ_C / VKQ_C / nstages /
// flash_attn_ext_f16_load_K at cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT).
// Not a vendor of that file. SM120 uses Ampere cp.async; no SM100 TMA/tcgen.
// Included from fattn_mma_f16.cuh inside namespace qw38::cuda.

constexpr char kLegalAttentionPipelineOff[] = "off";
constexpr char kLegalAttentionPipelineF16[] = "f16";
constexpr char kLegalAttentionPipelineDualReg[] = "dual_reg";
constexpr char kLegalAttentionPipelineF16Reg[] = "f16_reg";
constexpr char kLegalAttentionPipelineDualAsync[] = "dual_async";
constexpr char kLegalAttentionPipelineF16Async[] = "f16_async";
constexpr char kLegalAttentionPipelineN64[] = "nbatch64";
constexpr char kLegalAttentionPipelineGqa6[] = "gqa6";

inline bool fattn_pipeline_is_off(const char* path) noexcept {
  return path == nullptr || std::strcmp(path, kLegalAttentionPipelineOff) == 0;
}

inline bool legal_attention_pipeline_path(const char* path) noexcept {
  if (fattn_pipeline_is_off(path)) return true;
  return path != nullptr &&
         (std::strcmp(path, kLegalAttentionPipelineF16) == 0 ||
          std::strcmp(path, kLegalAttentionPipelineDualReg) == 0 ||
          std::strcmp(path, kLegalAttentionPipelineF16Reg) == 0 ||
          std::strcmp(path, kLegalAttentionPipelineDualAsync) == 0 ||
          std::strcmp(path, kLegalAttentionPipelineF16Async) == 0 ||
          std::strcmp(path, kLegalAttentionPipelineN64) == 0 ||
          std::strcmp(path, kLegalAttentionPipelineGqa6) == 0);
}

inline const char* effective_attention_pipeline_path() noexcept {
  return g_attention_pipeline_path_override != nullptr
             ? g_attention_pipeline_path_override
             : kSelectedAttentionPipelinePath;
}

inline __device__ void fattn_cp_async16(void* dst, const void* src) {
  const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(src));
}

inline __device__ void fattn_cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

inline __device__ void fattn_cp_async_wait() {
  asm volatile("cp.async.wait_all;");
}

constexpr int fattn_pipeline_nthreads(int ncols1, int ncols2) {
  return (ncols1 <= 8 && ncols2 == 2) ? 64 : 128;
}

inline std::size_t fattn_pipeline_shared_bytes(int ncols1, bool dual_f16,
                                               bool register_softmax,
                                               int nbatch, int nstages,
                                               int ncols2) {
  const int ncols = ncols1 * ncols2;
  const int nthreads = fattn_pipeline_nthreads(ncols1, ncols2);
  const int nwarps = nthreads / 32;
  const std::size_t kv = static_cast<std::size_t>(nstages) * 2ull *
                         static_cast<std::size_t>(nbatch) * kFattnHeadWidth *
                         sizeof(__nv_bfloat16);
  const std::size_t q = static_cast<std::size_t>(ncols) * kFattnHeadWidth *
                        sizeof(__half) * (dual_f16 ? 2ull : 1ull);
  const std::size_t weights =
      register_softmax
          ? (static_cast<std::size_t>(ncols) * static_cast<std::size_t>(nbatch) *
             sizeof(__half) * (dual_f16 ? 2ull : 1ull))
          : (static_cast<std::size_t>(ncols) * static_cast<std::size_t>(nbatch) *
             sizeof(float));
  const std::size_t stats = static_cast<std::size_t>(ncols) * 3ull * sizeof(float);
  const std::size_t reduce =
      static_cast<std::size_t>(nwarps) * static_cast<std::size_t>(ncols) *
      sizeof(float);
  const std::size_t scratch = kFattnHeadWidth * sizeof(float);
  const std::size_t cparts =
      register_softmax
          ? 0ull
          : static_cast<std::size_t>(nwarps) * 32ull * 4ull * sizeof(float);
  return kv + q + weights + stats + reduce + scratch + cparts;
}

template <int Nbatch, int Width, int Nthreads, int Nstages, bool UseCpAsync>
inline __device__ void fattn_pipeline_load_kv(
    __nv_bfloat16* keys, __nv_bfloat16* values, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, std::size_t tile, int rows,
    std::size_t start_position, std::size_t kv_span, std::uint32_t kv_head,
    std::size_t row_values, std::size_t width, std::uint32_t capacity,
    int stage, int tid) {
  __nv_bfloat16* kdst = keys + stage * Nbatch * Width;
  __nv_bfloat16* vdst = values + stage * Nbatch * Width;
  for (int row = 0; row < Nbatch; ++row) {
    const bool in_tile = row < rows;
    const std::size_t absolute = tile + static_cast<std::size_t>(row);
    const __nv_bfloat16* ksrc = nullptr;
    const __nv_bfloat16* vsrc = nullptr;
    if (in_tile && absolute < kv_span) {
      if (absolute < start_position) {
        const std::size_t packed = attention_kv_physical_index(
            absolute, kv_head, 0, capacity, static_cast<std::uint32_t>(width));
        ksrc = committed_key + packed;
        vsrc = committed_value + packed;
      } else {
        ksrc = candidate_key + (absolute - start_position) * row_values +
               kv_head * width;
        vsrc = candidate_value + (absolute - start_position) * row_values +
               kv_head * width;
      }
    }
    if constexpr (UseCpAsync) {
      if (in_tile && ksrc != nullptr) {
        constexpr int kChunks = Width * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
        for (int chunk = tid; chunk < kChunks; chunk += Nthreads) {
          const int dim = chunk * 8;
          fattn_cp_async16(kdst + row * Width + dim, ksrc + dim);
          fattn_cp_async16(vdst + row * Width + dim, vsrc + dim);
        }
      } else {
        for (int dim = tid; dim < Width; dim += Nthreads) {
          kdst[row * Width + dim] = __float2bfloat16_rn(0.0F);
          vdst[row * Width + dim] = __float2bfloat16_rn(0.0F);
        }
      }
    } else {
      for (int dim = tid; dim < Width; dim += Nthreads) {
        kdst[row * Width + dim] =
            in_tile && ksrc != nullptr ? ksrc[dim] : __float2bfloat16_rn(0.0F);
        vdst[row * Width + dim] =
            in_tile && vsrc != nullptr ? vsrc[dim] : __float2bfloat16_rn(0.0F);
      }
    }
  }
  if constexpr (UseCpAsync) fattn_cp_async_commit();
}

template <int Ncols1, bool DualF16, bool RegisterSoftmax, int NbatchFa,
          int Nstages, int Ncols2, int Occupancy, int KvParts>
__global__ void __launch_bounds__(Ncols1 <= 8 && Ncols2 == 2 ? 64 : 128,
                                  Occupancy)
fattn_mma_pipeline_kernel(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, float* partial, float* meta,
    const __half* prepared_q, std::size_t kv_origin) {
  quartz_pdl_sync();
  constexpr int kNcols = Ncols1 * Ncols2;
  constexpr int kNthreads = Ncols1 <= 8 && Ncols2 == 2 ? 64 : 128;
  constexpr int kNwarps = kNthreads / 32;
  constexpr int kWidth = kFattnHeadWidth;
  constexpr int kSubgroups = 6 / Ncols2;
  constexpr bool kUseCpAsync = Nstages > 1;
  constexpr int kPvMTiles = kNcols / 16;
  constexpr int kPvNTiles = kWidth / (kNwarps * 8);
  constexpr int kOwnedStrips = NbatchFa / (8 * kNwarps);
  static_assert(kNcols % 16 == 0, "pipeline Q columns must be 16-wide");
  static_assert(NbatchFa % (8 * kNwarps) == 0, "pipeline KV strips must map warps");
  static_assert(Nstages == 1 || Nstages == 2, "pipeline nstages 1 or 2");
  static_assert(Ncols2 == 2 || Ncols2 == 6, "pipeline GQA pairs or all six");
  (void)query;
  (void)query_scale;

  const std::uint32_t kv_head = blockIdx.x;
  const std::size_t first_row =
      static_cast<std::size_t>(Ncols1) * blockIdx.y;
  if (first_row >= token_count) {
    quartz_pdl_lc();
    return;
  }
  const int active_rows = static_cast<int>(
      token_count - first_row < static_cast<std::size_t>(Ncols1)
          ? token_count - first_row
          : static_cast<std::size_t>(Ncols1));
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const std::uint32_t first_head =
      kv_head * (config.query_heads / config.kv_heads);
  const std::size_t width = config.head_width;
  const std::size_t row_values =
      static_cast<std::size_t>(config.kv_heads) * width;
  const std::size_t origin = attention_kv_origin(start_position, kv_origin);
  const float attn_scale = 1.0F / sqrtf(static_cast<float>(width));

  extern __shared__ unsigned char raw[];
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(raw);
  __nv_bfloat16* values = keys + Nstages * NbatchFa * kWidth;
  __half* q_f16 = reinterpret_cast<__half*>(values + Nstages * NbatchFa * kWidth);
  __half* q_lo = q_f16 + kNcols * kWidth;
  __half* p_f16 = DualF16 ? q_lo + kNcols * kWidth : q_lo;
  __half* p_lo = p_f16 + kNcols * NbatchFa;
  float* scores = reinterpret_cast<float*>(RegisterSoftmax ? p_lo : p_f16);
  float* qmax = RegisterSoftmax
                    ? reinterpret_cast<float*>(
                          DualF16 ? p_lo + kNcols * NbatchFa : p_f16 + kNcols * NbatchFa)
                    : scores + kNcols * NbatchFa;
  float* qden = qmax + kNcols;
  float* rescale = qden + kNcols;
  float* wred = rescale + kNcols;
  [[maybe_unused]] float* scratch = wred + kNwarps * kNcols;
  [[maybe_unused]] float* cparts = scratch + kWidth;

  const int kv_z = KvParts > 1 ? static_cast<int>(blockIdx.z) : 0;
  float* vkq = (KvParts > 1 && kv_z > 0) ? partial : output;
  const std::size_t last_position =
      start_position + first_row + static_cast<std::size_t>(active_rows) - 1U;
  const std::size_t kv_span = last_position + 1U;
  const std::size_t kv_tiles =
      (kv_span + static_cast<std::size_t>(NbatchFa) - 1U) /
      static_cast<std::size_t>(NbatchFa);
  const std::size_t tiles_per_part =
      KvParts > 1 ? (kv_tiles + static_cast<std::size_t>(KvParts) - 1U) /
                        static_cast<std::size_t>(KvParts)
                  : kv_tiles;
  const std::size_t kv_begin = tiles_per_part * static_cast<std::size_t>(kv_z) *
                               static_cast<std::size_t>(NbatchFa);
  const std::size_t kv_end =
      kv_begin + tiles_per_part * static_cast<std::size_t>(NbatchFa);
  const std::size_t kv_stop = kv_end < kv_span ? kv_end : kv_span;

  for (int subgroup = 0; subgroup < kSubgroups; ++subgroup) {
    float c_acc[kPvMTiles][kPvNTiles][4];
#pragma unroll
    for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
#pragma unroll
      for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
#pragma unroll
        for (int item = 0; item < 4; ++item) c_acc[m_tile][n_local][item] = 0.0F;
      }
    }
    for (int member = 0; member < Ncols2; ++member) {
      const std::uint32_t query_head =
          first_head +
          static_cast<std::uint32_t>(subgroup * Ncols2 + member);
      for (int row = 0; row < Ncols1; ++row) {
        const int qcol = member * Ncols1 + row;
        const bool live = row < active_rows;
        const std::size_t qbase =
            live ? (first_row + static_cast<std::size_t>(row)) *
                         config.query_heads * width +
                     query_head * width
                 : 0;
        const std::size_t nq =
            token_count * static_cast<std::size_t>(config.query_heads) * width;
        const __half* hi_src = prepared_q;
        const __half* lo_src = prepared_q + nq;
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          if (live) {
            if constexpr (DualF16) {
              q_f16[qcol * kWidth + dim] = hi_src[qbase + dim];
              q_lo[qcol * kWidth + dim] = lo_src[qbase + dim];
            } else {
              const float value = __half2float(hi_src[qbase + dim]) +
                                  __half2float(lo_src[qbase + dim]);
              q_f16[qcol * kWidth + dim] = __float2half_rn(value);
            }
            if (token_count == 1) {
              float value = __half2float(hi_src[qbase + dim]);
              value += __half2float(lo_src[qbase + dim]);
              normalized_query[query_head * width + dim] = value;
            }
          } else {
            q_f16[qcol * kWidth + dim] = __float2half_rn(0.0F);
            if constexpr (DualF16)
              q_lo[qcol * kWidth + dim] = __float2half_rn(0.0F);
          }
        }
      }
    }
    __syncthreads();

    for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
      qmax[qcol] = -INFINITY;
      qden[qcol] = 0.0F;
      rescale[qcol] = 0.0F;
    }
    __syncthreads();

    int buf = 0;
    if (kv_begin < kv_stop) {
      const int rows0 = static_cast<int>(
          (kv_span - kv_begin) < static_cast<std::size_t>(NbatchFa)
              ? (kv_span - kv_begin)
              : static_cast<std::size_t>(NbatchFa));
      fattn_pipeline_load_kv<NbatchFa, kWidth, kNthreads, Nstages, kUseCpAsync>(
          keys, values, committed_key, committed_value, candidate_key,
          candidate_value, kv_begin, rows0, origin, kv_span, kv_head,
          row_values, width, config.capacity, 0, tid);
      if constexpr (kUseCpAsync) fattn_cp_async_wait();
      __syncthreads();
    }

    for (std::size_t tile = kv_begin; tile < kv_stop; tile += NbatchFa) {
      const int rows = static_cast<int>(
          (kv_span - tile) < static_cast<std::size_t>(NbatchFa)
              ? (kv_span - tile)
              : static_cast<std::size_t>(NbatchFa));
      const std::size_t next = tile + static_cast<std::size_t>(NbatchFa);
      if constexpr (Nstages > 1) {
        if (next < kv_stop) {
          const int rows_next = static_cast<int>(
              (kv_span - next) < static_cast<std::size_t>(NbatchFa)
                  ? (kv_span - next)
                  : static_cast<std::size_t>(NbatchFa));
          fattn_pipeline_load_kv<NbatchFa, kWidth, kNthreads, Nstages, true>(
              keys, values, committed_key, committed_value, candidate_key,
              candidate_value, next, rows_next, origin, kv_span,
              kv_head, row_values, width, config.capacity, 1 - buf, tid);
        }
      } else if (tile != kv_begin) {
        fattn_pipeline_load_kv<NbatchFa, kWidth, kNthreads, Nstages, kUseCpAsync>(
            keys, values, committed_key, committed_value, candidate_key,
            candidate_value, tile, rows, origin, kv_span, kv_head,
            row_values, width, config.capacity, 0, tid);
        if constexpr (kUseCpAsync) fattn_cp_async_wait();
        __syncthreads();
      }
      const __nv_bfloat16* keys_tile = keys + buf * NbatchFa * kWidth;
      const __nv_bfloat16* values_tile = values + buf * NbatchFa * kWidth;

      if constexpr (RegisterSoftmax) {
        float kq[kOwnedStrips][kPvMTiles][4];
#pragma unroll
        for (int strip = 0; strip < kOwnedStrips; ++strip) {
          const int n0 = (warp + strip * kNwarps) * 8;
#pragma unroll
          for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
            const int m0 = m_tile * 16;
            float reduced[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            for (int vwarp = 0; vwarp < kNwarps; ++vwarp) {
              float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              for (int k0 = vwarp * 16; k0 < kWidth; k0 += kNwarps * 16) {
                fattn_qk_mma_k16<DualF16, kNcols, kWidth>(
                    cfrag, cfrag_lo, q_f16, q_lo, keys_tile, m0, n0, k0, rows,
                    lane);
              }
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                float sum = cfrag[item];
                if constexpr (DualF16) sum += cfrag_lo[item];
                reduced[item] += sum;
              }
            }
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int i = mma::tile16x8_i(lane, item);
              const int j = mma::tile16x8_j(lane, item);
              const int qcol = m0 + i;
              const int krow = n0 + j;
              const int qrow = qcol % Ncols1;
              const std::size_t qpos =
                  start_position + first_row + static_cast<std::size_t>(qrow);
              const std::size_t absolute = tile + static_cast<std::size_t>(krow);
              float score = reduced[item] * attn_scale;
              if (qrow >= active_rows || krow >= rows || absolute > qpos)
                score = -INFINITY;
              kq[strip][m_tile][item] = score;
            }
          }
        }

        for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
          for (int w = 0; w < kNwarps; ++w) wred[w * kNcols + qcol] = -INFINITY;
        }
        __syncthreads();
#pragma unroll
        for (int strip = 0; strip < kOwnedStrips; ++strip) {
#pragma unroll
          for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
            float mx0 = fmaxf(kq[strip][m_tile][0], kq[strip][m_tile][1]);
            float mx1 = fmaxf(kq[strip][m_tile][2], kq[strip][m_tile][3]);
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffU, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffU, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffU, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffU, mx1, 2));
            if (lane % 4 == 0) {
              const int q0 = m_tile * 16 + (lane / 4);
              const int q1 = q0 + 8;
              wred[warp * kNcols + q0] = fmaxf(wred[warp * kNcols + q0], mx0);
              wred[warp * kNcols + q1] = fmaxf(wred[warp * kNcols + q1], mx1);
            }
          }
        }
        __syncthreads();
        for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
          float tile_max = -INFINITY;
          for (int w = 0; w < kNwarps; ++w)
            tile_max = fmaxf(tile_max, wred[w * kNcols + qcol]);
          const float old_max = qmax[qcol];
          const float new_max = fmaxf(old_max, tile_max);
          const float factor =
              old_max == -INFINITY ? 0.0F : expf(old_max - new_max);
          qmax[qcol] = new_max;
          rescale[qcol] = factor;
          qden[qcol] *= factor;
        }
        __syncthreads();

#pragma unroll
        for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
#pragma unroll
          for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int qcol = m_tile * 16 + mma::tile16x8_i(lane, item);
              c_acc[m_tile][n_local][item] *= rescale[qcol];
            }
          }
        }

        for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
          for (int w = 0; w < kNwarps; ++w) wred[w * kNcols + qcol] = 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (int strip = 0; strip < kOwnedStrips; ++strip) {
          const int n0 = (warp + strip * kNwarps) * 8;
#pragma unroll
          for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
            const int m0 = m_tile * 16;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int i = mma::tile16x8_i(lane, item);
              const int j = mma::tile16x8_j(lane, item);
              const int qcol = m0 + i;
              const int krow = n0 + j;
              const float new_max = qmax[qcol];
              float weight = 0.0F;
              if (kq[strip][m_tile][item] != -INFINITY && new_max != -INFINITY)
                weight = expf(kq[strip][m_tile][item] - new_max);
              kq[strip][m_tile][item] = weight;
              if (qcol < kNcols && krow < NbatchFa) {
                p_f16[qcol * NbatchFa + krow] = __float2half_rn(weight);
                if constexpr (DualF16) {
                  const __half hi = p_f16[qcol * NbatchFa + krow];
                  p_lo[qcol * NbatchFa + krow] =
                      __float2half_rn(weight - __half2float(hi));
                }
              }
            }
            float s0 = kq[strip][m_tile][0] + kq[strip][m_tile][1];
            float s1 = kq[strip][m_tile][2] + kq[strip][m_tile][3];
            s0 += __shfl_xor_sync(0xffffffffU, s0, 1);
            s0 += __shfl_xor_sync(0xffffffffU, s0, 2);
            s1 += __shfl_xor_sync(0xffffffffU, s1, 1);
            s1 += __shfl_xor_sync(0xffffffffU, s1, 2);
            if (lane % 4 == 0) {
              const int q0 = m0 + (lane / 4);
              const int q1 = q0 + 8;
              wred[warp * kNcols + q0] += s0;
              wred[warp * kNcols + q1] += s1;
            }
          }
        }
        __syncthreads();
        for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
          float add = 0.0F;
          for (int w = 0; w < kNwarps; ++w) add += wred[w * kNcols + qcol];
          qden[qcol] += add;
        }
        __syncthreads();

#pragma unroll
        for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
          const int m0 = m_tile * 16;
#pragma unroll
          for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
            const int n0 = warp * 8 + n_local * (kNwarps * 8);
#pragma unroll
            for (int k0 = 0; k0 < NbatchFa; k0 += 16) {
              if (k0 >= rows) continue;
              std::uint32_t a[4];
              std::uint32_t a_lo[4];
              std::uint32_t b[2];
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                const int i = mma::tile16x8_half2_i(lane, item);
                const int j = mma::tile16x8_half2_j(lane, item);
                const int qcol = m0 + i;
                const int krow = k0 + j * 2;
                __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
                __half pair_lo[2] = {__float2half_rn(0.0F),
                                     __float2half_rn(0.0F)};
                if (qcol < kNcols && (qcol % Ncols1) < active_rows) {
                  if (krow < rows) pair[0] = p_f16[qcol * NbatchFa + krow];
                  if ((krow + 1) < rows)
                    pair[1] = p_f16[qcol * NbatchFa + krow + 1];
                  if constexpr (DualF16) {
                    if (krow < rows) pair_lo[0] = p_lo[qcol * NbatchFa + krow];
                    if ((krow + 1) < rows)
                      pair_lo[1] = p_lo[qcol * NbatchFa + krow + 1];
                  }
                }
                a[item] = *reinterpret_cast<std::uint32_t*>(pair);
                a_lo[item] = *reinterpret_cast<std::uint32_t*>(pair_lo);
              }
#pragma unroll
              for (int item = 0; item < 2; ++item) {
                const int n = mma::tile8x8_i(lane, item);
                const int j = mma::tile8x8_j(lane, item);
                const int dim = n0 + n;
                const int krow = k0 + j * 2;
                __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
                if (dim < kWidth) {
                  if (krow < rows)
                    pair[0] = __float2half_rn(
                        __bfloat162float(values_tile[krow * kWidth + dim]));
                  if ((krow + 1) < rows)
                    pair[1] = __float2half_rn(__bfloat162float(
                        values_tile[(krow + 1) * kWidth + dim]));
                }
                b[item] = *reinterpret_cast<std::uint32_t*>(pair);
              }
              mma::mma_m16n8k16_f16_f32(c_acc[m_tile][n_local], a, b);
              if constexpr (DualF16) {
                float c_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                mma::mma_m16n8k16_f16_f32(c_lo, a_lo, b);
#pragma unroll
                for (int item = 0; item < 4; ++item)
                  c_acc[m_tile][n_local][item] += c_lo[item];
              }
            }
          }
        }
      } else {
        for (int m0 = 0; m0 < kNcols; m0 += 16) {
          for (int n0 = 0; n0 < NbatchFa; n0 += 8) {
            const int tile_id = (m0 / 16) * (NbatchFa / 8) + (n0 / 8);
            if (warp != tile_id % kNwarps) continue;
            float reduced[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            for (int vwarp = 0; vwarp < kNwarps; ++vwarp) {
              float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              for (int k0 = vwarp * 16; k0 < kWidth; k0 += kNwarps * 16) {
                fattn_qk_mma_k16<DualF16, kNcols, kWidth>(
                    cfrag, cfrag_lo, q_f16, q_lo, keys_tile, m0, n0, k0, rows,
                    lane);
              }
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                float sum = cfrag[item];
                if constexpr (DualF16) sum += cfrag_lo[item];
                reduced[item] += sum;
              }
            }
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int i = mma::tile16x8_i(lane, item);
              const int j = mma::tile16x8_j(lane, item);
              const int qcol = m0 + i;
              const int krow = n0 + j;
              if (qcol < kNcols && krow < NbatchFa)
                scores[qcol * NbatchFa + krow] = reduced[item] * attn_scale;
            }
          }
        }
        __syncthreads();
        for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
          const int row = qcol % Ncols1;
          const std::size_t qpos =
              start_position + first_row + static_cast<std::size_t>(row);
          float tile_max = -INFINITY;
          for (int krow = 0; krow < rows; ++krow) {
            const std::size_t absolute = tile + static_cast<std::size_t>(krow);
            float score = scores[qcol * NbatchFa + krow];
            if (row >= active_rows || absolute > qpos) score = -INFINITY;
            scores[qcol * NbatchFa + krow] = score;
            tile_max = fmaxf(tile_max, score);
          }
          const float old_max = qmax[qcol];
          const float new_max = fmaxf(old_max, tile_max);
          const float factor =
              old_max == -INFINITY ? 0.0F : expf(old_max - new_max);
          float denominator = qden[qcol] * factor;
          for (int krow = 0; krow < rows; ++krow) {
            const float score = scores[qcol * NbatchFa + krow];
            const float weight =
                score == -INFINITY ? 0.0F : expf(score - new_max);
            scores[qcol * NbatchFa + krow] = weight;
            denominator += weight;
          }
          qmax[qcol] = new_max;
          qden[qcol] = denominator;
          rescale[qcol] = factor;
        }
        __syncthreads();
#pragma unroll
        for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
#pragma unroll
          for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int qcol = m_tile * 16 + mma::tile16x8_i(lane, item);
              c_acc[m_tile][n_local][item] *= rescale[qcol];
            }
          }
        }
#pragma unroll
        for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
          const int m0 = m_tile * 16;
#pragma unroll
          for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
            const int n0 = warp * 8 + n_local * (kNwarps * 8);
#pragma unroll
            for (int k0 = 0; k0 < NbatchFa; k0 += 16) {
              if (k0 >= rows) continue;
              std::uint32_t a[4];
              std::uint32_t a_lo[4];
              std::uint32_t b[2];
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                const int i = mma::tile16x8_half2_i(lane, item);
                const int j = mma::tile16x8_half2_j(lane, item);
                const int qcol = m0 + i;
                const int krow = k0 + j * 2;
                const int qrow = qcol % Ncols1;
                __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
                __half pair_lo[2] = {__float2half_rn(0.0F),
                                     __float2half_rn(0.0F)};
                if (qcol < kNcols && qrow < active_rows) {
                  const float w0 =
                      krow < rows ? scores[qcol * NbatchFa + krow] : 0.0F;
                  const float w1 = (krow + 1) < rows
                                       ? scores[qcol * NbatchFa + krow + 1]
                                       : 0.0F;
                  pair[0] = __float2half_rn(w0);
                  pair[1] = __float2half_rn(w1);
                  if constexpr (DualF16) {
                    pair_lo[0] = __float2half_rn(w0 - __half2float(pair[0]));
                    pair_lo[1] = __float2half_rn(w1 - __half2float(pair[1]));
                  }
                }
                a[item] = *reinterpret_cast<std::uint32_t*>(pair);
                a_lo[item] = *reinterpret_cast<std::uint32_t*>(pair_lo);
              }
#pragma unroll
              for (int item = 0; item < 2; ++item) {
                const int n = mma::tile8x8_i(lane, item);
                const int j = mma::tile8x8_j(lane, item);
                const int dim = n0 + n;
                const int krow = k0 + j * 2;
                __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
                if (dim < kWidth) {
                  if (krow < rows)
                    pair[0] = __float2half_rn(
                        __bfloat162float(values_tile[krow * kWidth + dim]));
                  if ((krow + 1) < rows)
                    pair[1] = __float2half_rn(__bfloat162float(
                        values_tile[(krow + 1) * kWidth + dim]));
                }
                b[item] = *reinterpret_cast<std::uint32_t*>(pair);
              }
              mma::mma_m16n8k16_f16_f32(c_acc[m_tile][n_local], a, b);
              if constexpr (DualF16) {
                float c_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                mma::mma_m16n8k16_f16_f32(c_lo, a_lo, b);
#pragma unroll
                for (int item = 0; item < 4; ++item)
                  c_acc[m_tile][n_local][item] += c_lo[item];
              }
            }
          }
        }
      }

      if constexpr (Nstages > 1) {
        __syncthreads();
        if (next < kv_stop) {
          fattn_cp_async_wait();
          __syncthreads();
        }
        buf = 1 - buf;
      } else {
        __syncthreads();
      }
    }

#pragma unroll
    for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
      const int m0 = m_tile * 16;
#pragma unroll
      for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
        const int n0 = warp * 8 + n_local * (kNwarps * 8);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          const int i = mma::tile16x8_i(lane, item);
          const int j = mma::tile16x8_j(lane, item);
          const int qcol = m0 + i;
          const int row = qcol % Ncols1;
          const int member = qcol / Ncols1;
          const int dim = n0 + j;
          if (row < active_rows && member < Ncols2 && dim < kWidth) {
            const std::uint32_t query_head =
                first_head +
                static_cast<std::uint32_t>(subgroup * Ncols2 + member);
            const std::size_t qbase =
                (first_row + static_cast<std::size_t>(row)) *
                    config.query_heads * width +
                query_head * width;
            vkq[qbase + static_cast<std::size_t>(dim)] =
                c_acc[m_tile][n_local][item];
          }
        }
      }
    }

    for (int member = 0; member < Ncols2; ++member) {
      const std::uint32_t query_head =
          first_head +
          static_cast<std::uint32_t>(subgroup * Ncols2 + member);
      for (int row = 0; row < active_rows; ++row) {
        const int qcol = member * Ncols1 + row;
        const std::size_t qbase =
            (first_row + static_cast<std::size_t>(row)) * config.query_heads *
                width +
            query_head * width;
        const float denominator = qden[qcol];
        if constexpr (KvParts == 1) {
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            const float g = gate[qbase + static_cast<std::size_t>(dim)];
            const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                            : expf(g) / (1.0F + expf(g));
            const float attn =
                denominator == 0.0F
                    ? 0.0F
                    : vkq[qbase + static_cast<std::size_t>(dim)] / denominator;
            output[qbase + static_cast<std::size_t>(dim)] = attn * sigmoid;
          }
        } else if (tid == 0) {
          const std::size_t meta_index =
              (static_cast<std::size_t>(kv_z) * token_count + first_row +
               static_cast<std::size_t>(row)) *
                  config.query_heads +
              query_head;
          meta[meta_index * 2U] = qmax[qcol];
          meta[meta_index * 2U + 1U] = denominator;
        }
      }
    }
    __syncthreads();
  }
  quartz_pdl_lc();
}

template <int Ncols1, bool DualF16, bool RegisterSoftmax, int NbatchFa,
          int Nstages, int Ncols2, int Occupancy, int KvParts>
inline cudaError_t launch_fattn_mma_pipeline_typed(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta, cudaStream_t stream,
    const __half* prepared_q,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept {
  if (prepared_q == nullptr) return cudaErrorInvalidValue;
  constexpr int kNthreads = fattn_pipeline_nthreads(Ncols1, Ncols2);
  const std::size_t shared = fattn_pipeline_shared_bytes(
      Ncols1, DualF16, RegisterSoftmax, NbatchFa, Nstages, Ncols2);
  dim3 grid(config.kv_heads,
            static_cast<unsigned>((token_count + Ncols1 - 1) / Ncols1),
            KvParts > 1 ? 2U : 1U);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_pipeline_kernel<Ncols1, DualF16, RegisterSoftmax, NbatchFa,
                                Nstages, Ncols2, Occupancy, KvParts>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  error = quartz_launch_kernel(
      fattn_mma_pipeline_kernel<Ncols1, DualF16, RegisterSoftmax, NbatchFa,
                                Nstages, Ncols2, Occupancy, KvParts>,
      grid, dim3(kNthreads), shared, stream, config, start_position,
      token_count, query, query_scale, gate, committed_key, committed_value,
      candidate_key, candidate_value, output, normalized_query, partial, meta,
      prepared_q, resolve_attention_kv_origin(start_position, kv_origin));
  if (error != cudaSuccess || KvParts == 1) return error;
  const std::size_t values =
      token_count * static_cast<std::size_t>(config.query_heads) *
      config.head_width;
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((values + threads - 1) / threads);
  return quartz_launch_kernel(
      fattn_stream_k_combine_kernel, dim3(blocks), dim3(threads), 0, stream,
      config, token_count, gate, output, partial, meta);
}

template <int Ncols1, bool DualF16, bool RegisterSoftmax, int NbatchFa,
          int Nstages, int Ncols2, int Occupancy, int KvParts>
inline int fattn_pipeline_occupancy_typed() noexcept {
  int occupancy = 0;
  const int threads = fattn_pipeline_nthreads(Ncols1, Ncols2);
  const std::size_t shared = fattn_pipeline_shared_bytes(
      Ncols1, DualF16, RegisterSoftmax, NbatchFa, Nstages, Ncols2);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_pipeline_kernel<Ncols1, DualF16, RegisterSoftmax, NbatchFa,
                                Nstages, Ncols2, Occupancy, KvParts>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error == cudaSuccess)
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        fattn_mma_pipeline_kernel<Ncols1, DualF16, RegisterSoftmax, NbatchFa,
                                  Nstages, Ncols2, Occupancy, KvParts>,
        threads, shared);
  return error == cudaSuccess ? occupancy : 0;
}

inline cudaError_t launch_fattn_mma_pipeline(
    const char* path, const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta, cudaStream_t stream,
    const __half* prepared_q,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept {
  if (path == nullptr) return cudaErrorInvalidValue;
  if (std::strcmp(path, kLegalAttentionPipelineF16) == 0) {
    return launch_fattn_mma_pipeline_typed<16, false, false, 32, 1, 2, 2, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (std::strcmp(path, kLegalAttentionPipelineDualReg) == 0) {
    return launch_fattn_mma_pipeline_typed<16, true, true, 32, 1, 2, 2, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (std::strcmp(path, kLegalAttentionPipelineF16Reg) == 0) {
    return launch_fattn_mma_pipeline_typed<16, false, true, 32, 1, 2, 2, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (std::strcmp(path, kLegalAttentionPipelineDualAsync) == 0) {
    return launch_fattn_mma_pipeline_typed<16, true, true, 32, 2, 2, 1, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (std::strcmp(path, kLegalAttentionPipelineF16Async) == 0) {
    return launch_fattn_mma_pipeline_typed<16, false, true, 32, 2, 2, 1, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (std::strcmp(path, kLegalAttentionPipelineN64) == 0) {
    return launch_fattn_mma_pipeline_typed<16, false, true, 64, 1, 2, 1, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (std::strcmp(path, kLegalAttentionPipelineGqa6) == 0) {
    return launch_fattn_mma_pipeline_typed<8, false, true, 32, 1, 6, 1, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  return cudaErrorInvalidValue;
}

inline int fattn_pipeline_occupancy_for(const char* path) noexcept {
  if (path == nullptr) return 0;
  if (std::strcmp(path, kLegalAttentionPipelineF16) == 0)
    return fattn_pipeline_occupancy_typed<16, false, false, 32, 1, 2, 2, 2>();
  if (std::strcmp(path, kLegalAttentionPipelineDualReg) == 0)
    return fattn_pipeline_occupancy_typed<16, true, true, 32, 1, 2, 2, 2>();
  if (std::strcmp(path, kLegalAttentionPipelineF16Reg) == 0)
    return fattn_pipeline_occupancy_typed<16, false, true, 32, 1, 2, 2, 2>();
  if (std::strcmp(path, kLegalAttentionPipelineDualAsync) == 0)
    return fattn_pipeline_occupancy_typed<16, true, true, 32, 2, 2, 1, 2>();
  if (std::strcmp(path, kLegalAttentionPipelineF16Async) == 0)
    return fattn_pipeline_occupancy_typed<16, false, true, 32, 2, 2, 1, 2>();
  if (std::strcmp(path, kLegalAttentionPipelineN64) == 0)
    return fattn_pipeline_occupancy_typed<16, false, true, 64, 1, 2, 1, 2>();
  if (std::strcmp(path, kLegalAttentionPipelineGqa6) == 0)
    return fattn_pipeline_occupancy_typed<8, false, true, 32, 1, 6, 1, 2>();
  return 0;
}
