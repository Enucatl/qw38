#pragma once

// Focused Ampere fattn-mma-f16 analog for Quartz OPT-019 (DKQ=DV=256, GQA 6).
// Adapted from llama.cpp ggml/src/ggml-cuda/fattn.cu, fattn-mma-f16.cuh, and
// mma.cuh at cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors).
// Not a vendor of those files and not a copy of ../ds4. ds4 is mentioned only
// for multi-row online structure. Explicit mma.sync PTX is not compiler FMA
// contraction; Makefile NVCCFLAGS --fmad=false stays unchanged.

#include "attention_decode.h"
#include "mma.cuh"
#include "pdl_launch.cuh"

#include <cstdint>
#include <cstring>

#include <cuda_fp16.h>

namespace qw38::cuda {

constexpr int kFattnNcols2 = 2;
constexpr int kFattnNbatchFa = 32;
constexpr int kFattnHeadWidth = 256;
constexpr int kFattnSubgroups = 3;
constexpr float kFattnRmsEpsilon = 1.0e-6F;
constexpr float kFattnRopeTheta = 10000000.0F;

// Pinned after the 2048-row ncols1 sweep. Legal values: 8, 16, 32.
constexpr int kSelectedAttentionMmaQueryRows = 16;

// OPT-026 A/B winner. Legal values: baseline, occ2, stream_k.
constexpr char kSelectedFattnPath[] = "stream_k";

// OPT-027 A/B winner. Legal values: off, persistent.
constexpr char kSelectedPersistentFattnPath[] = "off";

// OPT-033 A/B winner. Legal values: global, registers. Default until A/B: global.
// Register-resident VKQ sums adapt llama.cpp fattn-mma-f16.cuh VKQ_C at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT). MMA probability×V is OPT-035.
constexpr char kSelectedVkqAccum[] = "registers";

// OPT-035 A/B winner. Legal values: scalar, mma. Default until A/B: scalar.
// Dual-F16 probability × F16 V MMA adapts llama.cpp fattn-mma-f16.cuh VKQ MMA at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT). Not a vendor of that file.
constexpr char kSelectedPvPath[] = "mma";

// OPT-041 A/B winner. Legal values: cparts, warp_microtile. Default until A/B:
// cparts. Warp-owned 16×8 QK microtiles adapt llama.cpp fattn-mma-f16.cuh KQ_C
// ownership at cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT). Quartz still
// visits K in the current virtual-warp order and folds in the current FP32
// order. Not a vendor of that file.
constexpr char kSelectedQKPath[] = "warp_microtile";

// OPT-050 A/B winner. Legal values: in_kernel, hoisted, hoisted_fma. Default
// until A/B: in_kernel. Hoisted RMS/RoPE/dual-F16 Q preparation follows the
// pinned llama.cpp qwen35.cpp / fattn-mma-f16.cuh upstream-Q boundary at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT). Not a vendor of those files.
constexpr char kLegalQueryPrepareInKernel[] = "in_kernel";
constexpr char kLegalQueryPrepareHoisted[] = "hoisted";
constexpr char kLegalQueryPrepareHoistedFma[] = "hoisted_fma";
constexpr char kSelectedQueryPreparePath[] = "hoisted";

// OPT-051 A/B winner. Legal values: off, f16, dual_reg, f16_reg, dual_async,
// f16_async, nbatch64, gqa6. Default until A/B: off. Sibling kernel in
// fattn_mma_f16_pipeline.cuh. Not a vendor of llama.cpp fattn-mma-f16.cuh.
constexpr char kSelectedAttentionPipelinePath[] = "f16_async";

inline thread_local const char* g_query_prepare_path_override = nullptr;
inline thread_local const char* g_attention_pipeline_path_override = nullptr;

inline bool fattn_query_prepare_is_in_kernel(const char* path) noexcept {
  return path != nullptr && std::strcmp(path, kLegalQueryPrepareInKernel) == 0;
}

inline bool fattn_query_prepare_is_hoisted(const char* path) noexcept {
  return path != nullptr && (std::strcmp(path, kLegalQueryPrepareHoisted) == 0 ||
                             std::strcmp(path, kLegalQueryPrepareHoistedFma) == 0);
}

inline bool fattn_query_prepare_is_fma(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalQueryPrepareHoistedFma) == 0;
}

inline bool legal_query_prepare_path(const char* path) noexcept {
  return fattn_query_prepare_is_in_kernel(path) ||
         fattn_query_prepare_is_hoisted(path);
}

inline const char* effective_query_prepare_path() noexcept {
  return g_query_prepare_path_override != nullptr ? g_query_prepare_path_override
                                                 : kSelectedQueryPreparePath;
}

inline std::size_t fattn_prepared_query_halfs(
    const AttentionConfig& config, std::size_t token_count) noexcept {
  return token_count * static_cast<std::size_t>(config.query_heads) *
         config.head_width * 2ull;
}

inline std::size_t fattn_prepared_query_bytes(
    const AttentionConfig& config, std::size_t token_count) noexcept {
  return fattn_prepared_query_halfs(config, token_count) * sizeof(__half);
}

inline __device__ float fattn_block_sum(float local, float* scratch, int tid,
                                       int nthreads) {
  float value = local;
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffU, value, mask));
  }
  const int lane = tid % 32;
  const int warp = tid / 32;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0F;
    const int nwarps = nthreads / 32;
    for (int other = 0; other < nwarps; ++other)
      total = __fadd_rn(total, scratch[other]);
    scratch[0] = total;
  }
  __syncthreads();
  const float total = scratch[0];
  __syncthreads();
  return total;
}

constexpr int fattn_nthreads_for_ncols1(int ncols1) {
  return ncols1 <= 8 ? 64 : 128;
}

constexpr int fattn_occupancy_target_for_ncols1(int ncols1) {
  return ncols1 <= 8 ? 4 : 2;
}

constexpr std::size_t fattn_quality_shared_bytes(int ncols1, bool dual_f16) {
  const int ncols = ncols1 * kFattnNcols2;
  const int nwarps = fattn_nthreads_for_ncols1(ncols1) / 32;
  const std::size_t kv = 2ull * kFattnNbatchFa * kFattnHeadWidth *
                         sizeof(__nv_bfloat16);
  const std::size_t q = static_cast<std::size_t>(ncols) * kFattnHeadWidth *
                        sizeof(__half) * (dual_f16 ? 2ull : 1ull);
  const std::size_t scores = static_cast<std::size_t>(ncols) * kFattnNbatchFa *
                             sizeof(float);
  const std::size_t stats = static_cast<std::size_t>(ncols) * 2ull *
                            sizeof(float);
  const std::size_t scratch = kFattnHeadWidth * sizeof(float);
  const std::size_t cparts = static_cast<std::size_t>(nwarps) * 32ull * 4ull *
                             sizeof(float);
  const std::size_t rescale = static_cast<std::size_t>(ncols) * sizeof(float);
  return kv + q + scores + stats + scratch + cparts + rescale;
}

inline int fattn_persistent_ntiles_x(std::size_t token_count) noexcept {
  return static_cast<int>((token_count +
                           static_cast<std::size_t>(kSelectedAttentionMmaQueryRows) -
                           1U) /
                          static_cast<std::size_t>(kSelectedAttentionMmaQueryRows));
}

inline int fattn_persistent_ntiles_z_gqa(const AttentionConfig& config) noexcept {
  const int gqa = static_cast<int>(config.query_heads / config.kv_heads);
  return (gqa + kFattnNcols2 - 1) / kFattnNcols2;
}

inline int fattn_persistent_ntiles_dst(const AttentionConfig& config,
                                         std::size_t token_count) noexcept {
  return fattn_persistent_ntiles_x(token_count) *
         fattn_persistent_ntiles_z_gqa(config) *
         static_cast<int>(config.kv_heads);
}

inline int fattn_persistent_ntiles_kv(std::size_t start_position,
                                        std::size_t token_count) noexcept {
  const std::size_t kv_len = start_position + token_count;
  return static_cast<int>((kv_len + static_cast<std::size_t>(kFattnNbatchFa) -
                            1U) /
                          static_cast<std::size_t>(kFattnNbatchFa));
}

inline int fattn_compute_persistent_nblocks(int nsm, int occupancy,
                                              int ntiles_dst,
                                              int ntiles_kv) noexcept {
  if (nsm <= 0 || occupancy <= 0 || ntiles_dst <= 0 || ntiles_kv <= 0)
    return 0;
  const int max_blocks = occupancy * nsm;
  const int total = ntiles_kv * ntiles_dst;
  const int nblocks_raw = max_blocks < total ? max_blocks : total;
  const int nblocks_rounded = (nblocks_raw / ntiles_dst) * ntiles_dst;
  const int loss =
      nblocks_rounded > 0
          ? 100 * (nblocks_raw - nblocks_rounded) / nblocks_raw
          : 100;
  return loss <= 5 ? nblocks_rounded : nblocks_raw;
}

inline std::size_t fattn_compute_persistent_fixup_values(int nblocks) noexcept {
  constexpr int kNcols = kSelectedAttentionMmaQueryRows * kFattnNcols2;
  if (nblocks <= 0) return 0;
  return static_cast<std::size_t>(nblocks) * static_cast<std::size_t>(kNcols) *
         (4ull + static_cast<std::size_t>(kFattnHeadWidth));
}

inline __device__ float fattn_output_gate(float g) {
  return g >= 0.0F ? 1.0F / (1.0F + expf(-g)) : expf(g) / (1.0F + expf(g));
}

inline __device__ void fattn_persistent_decode(int index, int ntiles_kv,
                                                 int ntiles_x, int ntiles_z_gqa,
                                                 int* z_KV, int* zt_gqa,
                                                 int* jt) {
  const int dest = ntiles_kv == 0 ? 0 : index / ntiles_kv;
  const int span = ntiles_x * ntiles_z_gqa;
  *z_KV = span == 0 ? 0 : dest / span;
  const int rem = dest - (*z_KV) * span;
  *zt_gqa = ntiles_x == 0 ? 0 : rem / ntiles_x;
  *jt = rem - (*zt_gqa) * ntiles_x;
}

template <bool DualF16, int kNcols, int kWidth>
inline __device__ void fattn_qk_mma_k16(
    float cfrag[4], float cfrag_lo[4], const __half* q_f16, const __half* q_lo,
    const __nv_bfloat16* keys, int m0, int n0, int k0, int rows, int lane) {
  std::uint32_t a[4];
  std::uint32_t a_lo[4] = {0, 0, 0, 0};
  std::uint32_t b[2];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int i = mma::tile16x8_half2_i(lane, item);
    const int j = mma::tile16x8_half2_j(lane, item);
    const int qcol = m0 + i;
    __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
    __half pair_lo[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
    if (qcol < kNcols) {
      const int offset = qcol * kWidth + k0 + j * 2;
      pair[0] = q_f16[offset];
      pair[1] = q_f16[offset + 1];
      if constexpr (DualF16) {
        pair_lo[0] = q_lo[offset];
        pair_lo[1] = q_lo[offset + 1];
      }
    }
    a[item] = *reinterpret_cast<std::uint32_t*>(pair);
    if constexpr (DualF16) {
      a_lo[item] = *reinterpret_cast<std::uint32_t*>(pair_lo);
    }
  }
#pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int n = mma::tile8x8_i(lane, item);
    const int j = mma::tile8x8_j(lane, item);
    const int krow = n0 + n;
    __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
    if (krow < rows) {
      pair[0] = __float2half_rn(
          __bfloat162float(keys[krow * kWidth + k0 + j * 2]));
      pair[1] = __float2half_rn(
          __bfloat162float(keys[krow * kWidth + k0 + j * 2 + 1]));
    }
    b[item] = *reinterpret_cast<std::uint32_t*>(pair);
  }
  mma::mma_m16n8k16_f16_f32(cfrag, a, b);
  if constexpr (DualF16) {
    mma::mma_m16n8k16_f16_f32(cfrag_lo, a_lo, b);
  }
}

template <int Ncols1, bool DualF16, int Occupancy, int KvParts,
          bool RegisterAcc = false, bool MmaPv = false, bool WarpQK = false,
          bool PreparedQ = false>
__global__ void __launch_bounds__(Ncols1 <= 8 ? 64 : 128, Occupancy)
fattn_mma_quality_kernel(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, float* partial, float* meta,
    [[maybe_unused]] const __half* prepared_q, std::size_t kv_origin) {
  quartz_pdl_sync();
  constexpr int kNcols = Ncols1 * kFattnNcols2;
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  constexpr int kNwarps = kNthreads / 32;
  constexpr int kWidth = kFattnHeadWidth;
  static_assert(!MmaPv || RegisterAcc,
                "OPT-035 MMA P×V requires register-resident VKQ");
  static_assert(!MmaPv || (kNcols % 16 == 0 && kWidth % (kNwarps * 8) == 0),
                "OPT-035 MMA P×V needs 16-wide M tiles and warp N ownership");
  static_assert(!WarpQK || (RegisterAcc && MmaPv && Ncols1 == 16 && DualF16 &&
                            kNcols == 32 && kFattnNbatchFa == 32 &&
                            kNwarps == 4),
                "OPT-041 warp QK microtiles require production stream-K");
  constexpr int kPvMTiles = kNcols / 16;
  constexpr int kPvNTiles = kWidth / (kNwarps * 8);

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
  const std::uint32_t half = config.rotary_width / 2;
  const float attn_scale = 1.0F / sqrtf(static_cast<float>(width));
  if constexpr (PreparedQ) {
    (void)query;
    (void)query_scale;
  }

  extern __shared__ unsigned char raw[];
  float* scratch = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(scratch + kWidth);
  __nv_bfloat16* values = keys + kFattnNbatchFa * kWidth;
  __half* q_f16 = reinterpret_cast<__half*>(values + kFattnNbatchFa * kWidth);
  __half* q_lo = q_f16 + kNcols * kWidth;
  float* scores = reinterpret_cast<float*>(
      DualF16 ? reinterpret_cast<__half*>(q_lo + kNcols * kWidth)
              : q_lo);
  float* qmax = scores + kNcols * kFattnNbatchFa;
  float* qden = qmax + kNcols;
  float* rescale = qden + kNcols;
  [[maybe_unused]] float* cparts = rescale + kNcols;

  const int kv_z = KvParts > 1 ? static_cast<int>(blockIdx.z) : 0;
  float* vkq = (KvParts > 1 && kv_z > 0) ? partial : output;
  const std::size_t last_position =
      start_position + first_row + static_cast<std::size_t>(active_rows) - 1U;
  const std::size_t kv_span = last_position + 1U;
  const std::size_t kv_tiles =
      (kv_span + static_cast<std::size_t>(kFattnNbatchFa) - 1U) /
      static_cast<std::size_t>(kFattnNbatchFa);
  const std::size_t tiles_per_part =
      KvParts > 1 ? (kv_tiles + static_cast<std::size_t>(KvParts) - 1U) /
                        static_cast<std::size_t>(KvParts)
                  : kv_tiles;
  const std::size_t kv_begin =
      tiles_per_part * static_cast<std::size_t>(kv_z) *
      static_cast<std::size_t>(kFattnNbatchFa);
  const std::size_t kv_end = kv_begin + tiles_per_part *
                                           static_cast<std::size_t>(kFattnNbatchFa);
  const std::size_t kv_stop = kv_end < kv_span ? kv_end : kv_span;

  for (int subgroup = 0; subgroup < kFattnSubgroups; ++subgroup) {
    constexpr int kDimSlots = kWidth / kNthreads;
    constexpr int kAccCount = kFattnNcols2 * Ncols1 * kDimSlots;
    constexpr bool kUseScalarAcc = RegisterAcc && !MmaPv;
    float acc[kUseScalarAcc ? kAccCount : 1];
    float c_acc[MmaPv ? kPvMTiles : 1][MmaPv ? kPvNTiles : 1][MmaPv ? 4 : 1];
#pragma unroll
    for (int slot = 0; slot < (kUseScalarAcc ? kAccCount : 1); ++slot)
      acc[slot] = 0.0F;
    if constexpr (MmaPv) {
#pragma unroll
      for (int m_tile = 0; m_tile < kPvMTiles; ++m_tile) {
#pragma unroll
        for (int n_local = 0; n_local < kPvNTiles; ++n_local) {
#pragma unroll
          for (int item = 0; item < 4; ++item)
            c_acc[m_tile][n_local][item] = 0.0F;
        }
      }
    }
    for (int member = 0; member < kFattnNcols2; ++member) {
      const std::uint32_t query_head =
          first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                  member);
      for (int row = 0; row < Ncols1; ++row) {
        const int qcol = member * Ncols1 + row;
        const bool live = row < active_rows;
        const std::size_t qbase =
            live ? (first_row + static_cast<std::size_t>(row)) *
                         config.query_heads * width +
                     query_head * width
                 : 0;
        if constexpr (PreparedQ) {
          const std::size_t nq =
              token_count * static_cast<std::size_t>(config.query_heads) *
              width;
          const __half* hi_src = prepared_q;
          const __half* lo_src = prepared_q + nq;
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            if (live) {
              q_f16[qcol * kWidth + dim] = hi_src[qbase + dim];
              if constexpr (DualF16) {
                q_lo[qcol * kWidth + dim] = lo_src[qbase + dim];
              }
              if (token_count == 1) {
                float value = __half2float(hi_src[qbase + dim]);
                if constexpr (DualF16)
                  value += __half2float(lo_src[qbase + dim]);
                normalized_query[query_head * width + dim] = value;
              }
              if constexpr (!RegisterAcc)
                vkq[qbase + static_cast<std::size_t>(dim)] = 0.0F;
            } else {
              q_f16[qcol * kWidth + dim] = __float2half_rn(0.0F);
              if constexpr (DualF16)
                q_lo[qcol * kWidth + dim] = __float2half_rn(0.0F);
            }
          }
          __syncthreads();
        } else {
          float local = 0.0F;
          if (live) {
            for (int dim = tid; dim < static_cast<int>(width);
                 dim += kNthreads) {
              const float item = query[qbase + static_cast<std::size_t>(dim)];
              local = __fadd_rn(local, __fmul_rn(item, item));
            }
          }
          const float sum = fattn_block_sum(local, scratch, tid, kNthreads);
          const float inverse =
              1.0F / sqrtf(sum / static_cast<float>(width) + kFattnRmsEpsilon);
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            scratch[dim] =
                live ? query[qbase + static_cast<std::size_t>(dim)] * inverse *
                           query_scale[dim]
                     : 0.0F;
          }
          __syncthreads();
          if (tid < static_cast<int>(half)) {
            const float first = scratch[tid];
            const float second = scratch[half + tid];
            const float exponent = static_cast<float>(tid * 2) /
                                   static_cast<float>(config.rotary_width);
            const float angle =
                live ? static_cast<float>(start_position + first_row +
                                          static_cast<std::size_t>(row)) /
                           powf(kFattnRopeTheta, exponent)
                     : 0.0F;
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            scratch[tid] = first * cosine - second * sine;
            scratch[half + tid] = second * cosine + first * sine;
          }
          __syncthreads();
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            const float value = scratch[dim];
            const __half hi = __float2half_rn(value);
            q_f16[qcol * kWidth + dim] = hi;
            if constexpr (DualF16) {
              q_lo[qcol * kWidth + dim] =
                  __float2half_rn(value - __half2float(hi));
            }
            if (token_count == 1 && live)
              normalized_query[query_head * width + dim] = value;
            if constexpr (!RegisterAcc) {
              if (live)
                vkq[qbase + static_cast<std::size_t>(dim)] = 0.0F;
            }
          }
          __syncthreads();
        }
      }
    }

    for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
      qmax[qcol] = -INFINITY;
      qden[qcol] = 0.0F;
    }
    __syncthreads();

    for (std::size_t tile = kv_begin; tile < kv_stop; tile += kFattnNbatchFa) {
      const int rows = static_cast<int>(
          (kv_span - tile) < static_cast<std::size_t>(kFattnNbatchFa)
              ? (kv_span - tile)
              : static_cast<std::size_t>(kFattnNbatchFa));
      for (int row = 0; row < kFattnNbatchFa; ++row) {
        const bool in_tile = row < rows;
        const std::size_t absolute = tile + static_cast<std::size_t>(row);
        const __nv_bfloat16* ksrc = nullptr;
        const __nv_bfloat16* vsrc = nullptr;
        if (in_tile) {
          if (absolute < origin) {
            const std::size_t packed = attention_kv_physical_index(
                absolute, kv_head, 0, config.capacity, config.head_width);
            ksrc = committed_key + packed;
            vsrc = committed_value + packed;
          } else {
            ksrc = candidate_key + (absolute - origin) * row_values +
                   kv_head * width;
            vsrc = candidate_value + (absolute - origin) * row_values +
                   kv_head * width;
          }
        }
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          keys[row * kWidth + dim] =
              in_tile ? ksrc[dim] : __float2bfloat16_rn(0.0F);
          values[row * kWidth + dim] =
              in_tile ? vsrc[dim] : __float2bfloat16_rn(0.0F);
        }
      }
      __syncthreads();

      if constexpr (WarpQK) {
        for (int m0 = 0; m0 < kNcols; m0 += 16) {
          for (int n0 = 0; n0 < kFattnNbatchFa; n0 += 8) {
            const int tile_id = (m0 / 16) * 4 + (n0 / 8);
            if (warp != tile_id % kNwarps) continue;
            float reduced[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            for (int vwarp = 0; vwarp < kNwarps; ++vwarp) {
              float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              for (int k0 = vwarp * 16; k0 < kWidth; k0 += kNwarps * 16) {
                fattn_qk_mma_k16<DualF16, kNcols, kWidth>(
                    cfrag, cfrag_lo, q_f16, q_lo, keys, m0, n0, k0, rows,
                    lane);
              }
#pragma unroll
              for (int item = 0; item < 4; ++item)
                reduced[item] += cfrag[item] + cfrag_lo[item];
            }
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int i = mma::tile16x8_i(lane, item);
              const int j = mma::tile16x8_j(lane, item);
              const int qcol = m0 + i;
              const int krow = n0 + j;
              if (qcol < kNcols && krow < kFattnNbatchFa)
                scores[qcol * kFattnNbatchFa + krow] =
                    reduced[item] * attn_scale;
            }
          }
        }
        __syncthreads();
      } else {
        for (int m0 = 0; m0 < kNcols; m0 += 16) {
          for (int n0 = 0; n0 < kFattnNbatchFa; n0 += 8) {
            float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            for (int k0 = warp * 16; k0 < kWidth; k0 += kNwarps * 16) {
              fattn_qk_mma_k16<DualF16, kNcols, kWidth>(
                  cfrag, cfrag_lo, q_f16, q_lo, keys, m0, n0, k0, rows, lane);
            }
#pragma unroll
            for (int item = 0; item < 4; ++item)
              cparts[(warp * 32 + lane) * 4 + item] =
                  cfrag[item] + cfrag_lo[item];
            __syncthreads();
            if (warp == 0) {
              float reduced[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              for (int other = 0; other < kNwarps; ++other) {
#pragma unroll
                for (int item = 0; item < 4; ++item)
                  reduced[item] += cparts[(other * 32 + lane) * 4 + item];
              }
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                const int i = mma::tile16x8_i(lane, item);
                const int j = mma::tile16x8_j(lane, item);
                const int qcol = m0 + i;
                const int krow = n0 + j;
                if (qcol < kNcols && krow < kFattnNbatchFa)
                  scores[qcol * kFattnNbatchFa + krow] =
                      reduced[item] * attn_scale;
              }
            }
            __syncthreads();
          }
        }
      }

      for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
        const int row = qcol % Ncols1;
        const std::size_t qpos =
            start_position + first_row + static_cast<std::size_t>(row);
        float tile_max = -INFINITY;
        for (int krow = 0; krow < rows; ++krow) {
          const std::size_t absolute = tile + static_cast<std::size_t>(krow);
          float score = scores[qcol * kFattnNbatchFa + krow];
          if (row >= active_rows || absolute > qpos) score = -INFINITY;
          scores[qcol * kFattnNbatchFa + krow] = score;
          tile_max = fmaxf(tile_max, score);
        }
        const float old_max = qmax[qcol];
        const float new_max = fmaxf(old_max, tile_max);
        const float factor =
            old_max == -INFINITY ? 0.0F : expf(old_max - new_max);
        float denominator = qden[qcol] * factor;
        for (int krow = 0; krow < rows; ++krow) {
          const float score = scores[qcol * kFattnNbatchFa + krow];
          const float weight =
              score == -INFINITY ? 0.0F : expf(score - new_max);
          scores[qcol * kFattnNbatchFa + krow] = weight;
          denominator += weight;
        }
        qmax[qcol] = new_max;
        qden[qcol] = denominator;
        rescale[qcol] = factor;
      }
      __syncthreads();

      if constexpr (MmaPv) {
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
            for (int k0 = 0; k0 < kFattnNbatchFa; k0 += 16) {
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
                      krow < rows ? scores[qcol * kFattnNbatchFa + krow]
                                  : 0.0F;
                  const float w1 =
                      (krow + 1) < rows
                          ? scores[qcol * kFattnNbatchFa + krow + 1]
                          : 0.0F;
                  pair[0] = __float2half_rn(w0);
                  pair[1] = __float2half_rn(w1);
                  pair_lo[0] = __float2half_rn(w0 - __half2float(pair[0]));
                  pair_lo[1] = __float2half_rn(w1 - __half2float(pair[1]));
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
                  if (krow < rows) {
                    pair[0] = __float2half_rn(__bfloat162float(
                        values[krow * kWidth + dim]));
                  }
                  if ((krow + 1) < rows) {
                    pair[1] = __float2half_rn(__bfloat162float(
                        values[(krow + 1) * kWidth + dim]));
                  }
                }
                b[item] = *reinterpret_cast<std::uint32_t*>(pair);
              }
              float c_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
              mma::mma_m16n8k16_f16_f32(c_acc[m_tile][n_local], a, b);
              mma::mma_m16n8k16_f16_f32(c_lo, a_lo, b);
#pragma unroll
              for (int item = 0; item < 4; ++item)
                c_acc[m_tile][n_local][item] += c_lo[item];
            }
          }
        }
      } else if constexpr (RegisterAcc) {
#pragma unroll
        for (int member = 0; member < kFattnNcols2; ++member) {
#pragma unroll
          for (int row = 0; row < Ncols1; ++row) {
            const int qcol = member * Ncols1 + row;
            const float factor = rescale[qcol];
#pragma unroll
            for (int dim_slot = 0; dim_slot < kDimSlots; ++dim_slot) {
              const int idx = (member * Ncols1 + row) * kDimSlots + dim_slot;
              acc[idx] = acc[idx] * factor;
              if (row < active_rows) {
                const int dim = tid + dim_slot * kNthreads;
                for (int krow = 0; krow < rows; ++krow) {
                  acc[idx] += scores[qcol * kFattnNbatchFa + krow] *
                              __bfloat162float(values[krow * kWidth + dim]);
                }
              }
            }
          }
        }
      } else {
        for (int member = 0; member < kFattnNcols2; ++member) {
          const std::uint32_t query_head =
              first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                      member);
          for (int row = 0; row < active_rows; ++row) {
            const int qcol = member * Ncols1 + row;
            const std::size_t qbase =
                (first_row + static_cast<std::size_t>(row)) *
                    config.query_heads * width +
                query_head * width;
            const float factor = rescale[qcol];
            for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
              float total = vkq[qbase + static_cast<std::size_t>(dim)] * factor;
              for (int krow = 0; krow < rows; ++krow) {
                total += scores[qcol * kFattnNbatchFa + krow] *
                         __bfloat162float(values[krow * kWidth + dim]);
              }
              vkq[qbase + static_cast<std::size_t>(dim)] = total;
            }
          }
        }
      }
      __syncthreads();
    }
    if constexpr (MmaPv) {
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
            if (row < active_rows && member < kFattnNcols2 && dim < kWidth) {
              const std::uint32_t query_head =
                  first_head +
                  static_cast<std::uint32_t>(subgroup * kFattnNcols2 + member);
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
    } else if constexpr (RegisterAcc) {
#pragma unroll
      for (int member = 0; member < kFattnNcols2; ++member) {
        const std::uint32_t query_head =
            first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                  member);
#pragma unroll
        for (int row = 0; row < Ncols1; ++row) {
          if (row >= active_rows) continue;
          const std::size_t qbase =
              (first_row + static_cast<std::size_t>(row)) *
                  config.query_heads * width +
              query_head * width;
#pragma unroll
          for (int dim_slot = 0; dim_slot < kDimSlots; ++dim_slot) {
            const int dim = tid + dim_slot * kNthreads;
            const int idx = (member * Ncols1 + row) * kDimSlots + dim_slot;
            vkq[qbase + static_cast<std::size_t>(dim)] = acc[idx];
          }
        }
      }
    }

    for (int member = 0; member < kFattnNcols2; ++member) {
      const std::uint32_t query_head =
          first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                  member);
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

__global__ void fattn_stream_k_combine_kernel(
    AttentionConfig config, std::size_t token_count, const float* gate,
    float* output, const float* partial, const float* meta) {
  quartz_pdl_sync();
  const std::size_t width = config.head_width;
  const std::size_t values =
      token_count * static_cast<std::size_t>(config.query_heads) * width;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                static_cast<std::size_t>(blockDim.x) +
                            static_cast<std::size_t>(threadIdx.x);
  if (index >= values) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t row_values =
      static_cast<std::size_t>(config.query_heads) * width;
  const std::size_t row = index / row_values;
  const std::size_t in_row = index % row_values;
  const std::uint32_t head = static_cast<std::uint32_t>(in_row / width);
  const std::size_t meta0 =
      (row * static_cast<std::size_t>(config.query_heads) + head) * 2U;
  const std::size_t meta1 =
      ((token_count + row) * static_cast<std::size_t>(config.query_heads) +
       head) *
      2U;
  const float max0 = meta[meta0];
  const float den0 = meta[meta0 + 1U];
  const float max1 = meta[meta1];
  const float den1 = meta[meta1 + 1U];
  const float new_max = fmaxf(max0, max1);
  const float scale0 = max0 == -INFINITY ? 0.0F : expf(max0 - new_max);
  const float scale1 = max1 == -INFINITY ? 0.0F : expf(max1 - new_max);
  const float vkq = output[index] * scale0 + partial[index] * scale1;
  const float denominator = den0 * scale0 + den1 * scale1;
  const float g = gate[index];
  const float sigmoid =
      g >= 0.0F ? 1.0F / (1.0F + expf(-g)) : expf(g) / (1.0F + expf(g));
  const float attn = denominator == 0.0F ? 0.0F : vkq / denominator;
  output[index] = attn * sigmoid;
  quartz_pdl_lc();
}

// Packed dst_tmp_meta (floats): needs max/den, is_fixup max/den, then VKQ.
// Adapted from llama.cpp fattn-common.cuh / fattn-mma-f16.cuh at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT). Not a vendor of those files.
inline __device__ float* fattn_persist_needs_meta(float* meta, int nblocks,
                                                 int qcol) {
  constexpr int kNcols = kSelectedAttentionMmaQueryRows * kFattnNcols2;
  (void)nblocks;
  return meta + (static_cast<int>(blockIdx.x) * kNcols + qcol) * 2;
}

inline __device__ float* fattn_persist_is_meta(float* meta, int nblocks,
                                                int qcol) {
  constexpr int kNcols = kSelectedAttentionMmaQueryRows * kFattnNcols2;
  return meta + ((nblocks + static_cast<int>(blockIdx.x)) * kNcols + qcol) * 2;
}

inline __device__ float* fattn_persist_vkq_data(float* meta, int nblocks,
                                                int qcol) {
  constexpr int kNcols = kSelectedAttentionMmaQueryRows * kFattnNcols2;
  constexpr int kWidth = kFattnHeadWidth;
  return meta + nblocks * kNcols * 4 +
         (static_cast<int>(blockIdx.x) * kNcols + qcol) * kWidth;
}

template <int Ncols1, bool DualF16>
__device__ void fattn_persistent_process_tile(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, float* meta, int nblocks, int jt,
    int zt_gqa, int z_KV, int kb0_start, int kb0_stop, bool needs_fixup,
    bool is_fixup, float* scratch, __nv_bfloat16* keys, __nv_bfloat16* values,
    __half* q_f16, __half* q_lo, float* scores, float* qmax, float* qden,
    float* rescale, float* cparts) {
  constexpr int kNcols = Ncols1 * kFattnNcols2;
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  constexpr int kNwarps = kNthreads / 32;
  constexpr int kWidth = kFattnHeadWidth;
  const std::size_t first_row = static_cast<std::size_t>(Ncols1) * jt;
  if (first_row >= token_count) return;
  const int active_rows = static_cast<int>(
      token_count - first_row < static_cast<std::size_t>(Ncols1)
          ? token_count - first_row
          : static_cast<std::size_t>(Ncols1));
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const std::uint32_t kv_head = static_cast<std::uint32_t>(z_KV);
  const std::uint32_t first_head =
      kv_head * (config.query_heads / config.kv_heads);
  const std::size_t width = config.head_width;
  const std::size_t row_values =
      static_cast<std::size_t>(config.kv_heads) * width;
  const std::uint32_t half = config.rotary_width / 2;
  const float attn_scale = 1.0F / sqrtf(static_cast<float>(width));
  const std::size_t last_position =
      start_position + first_row + static_cast<std::size_t>(active_rows) - 1U;
  const std::size_t kv_span = last_position + 1U;
  const std::size_t kv_begin =
      static_cast<std::size_t>(kb0_start) * kFattnNbatchFa;
  const std::size_t kv_end =
      static_cast<std::size_t>(kb0_stop) * kFattnNbatchFa;
  const std::size_t kv_stop = kv_end < kv_span ? kv_end : kv_span;
  float* vkq = is_fixup ? nullptr : output;
  (void)vkq;
  const int subgroup = zt_gqa;

  for (int member = 0; member < kFattnNcols2; ++member) {
    const std::uint32_t query_head =
        first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                member);
    for (int row = 0; row < Ncols1; ++row) {
      const int qcol = member * Ncols1 + row;
      const bool live = row < active_rows;
      const std::size_t qbase =
          live ? (first_row + static_cast<std::size_t>(row)) *
                       config.query_heads * width +
                     query_head * width
               : 0;
      float local = 0.0F;
      if (live) {
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          const float item = query[qbase + static_cast<std::size_t>(dim)];
          local = __fadd_rn(local, __fmul_rn(item, item));
        }
      }
      const float sum = fattn_block_sum(local, scratch, tid, kNthreads);
      const float inverse =
          1.0F / sqrtf(sum / static_cast<float>(width) + kFattnRmsEpsilon);
      for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
        scratch[dim] =
            live ? query[qbase + static_cast<std::size_t>(dim)] * inverse *
                       query_scale[dim]
                 : 0.0F;
      }
      __syncthreads();
      if (tid < static_cast<int>(half)) {
        const float first = scratch[tid];
        const float second = scratch[half + tid];
        const float exponent = static_cast<float>(tid * 2) /
                               static_cast<float>(config.rotary_width);
        const float angle =
            live ? static_cast<float>(start_position + first_row +
                                      static_cast<std::size_t>(row)) /
                       powf(kFattnRopeTheta, exponent)
                 : 0.0F;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        scratch[tid] = first * cosine - second * sine;
        scratch[half + tid] = second * cosine + first * sine;
      }
      __syncthreads();
      for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
        const float value = scratch[dim];
        const __half hi = __float2half_rn(value);
        q_f16[qcol * kWidth + dim] = hi;
        if constexpr (DualF16) {
          q_lo[qcol * kWidth + dim] =
              __float2half_rn(value - __half2float(hi));
        }
        if (token_count == 1 && live)
          normalized_query[query_head * width + dim] = value;
        if (live) {
          if (is_fixup) {
            fattn_persist_vkq_data(meta, nblocks, qcol)[dim] = 0.0F;
          } else {
            vkq[qbase + static_cast<std::size_t>(dim)] = 0.0F;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
    qmax[qcol] = -INFINITY;
    qden[qcol] = 0.0F;
  }
  __syncthreads();

  for (std::size_t tile = kv_begin; tile < kv_stop; tile += kFattnNbatchFa) {
    const int rows = static_cast<int>(
        (kv_span - tile) < static_cast<std::size_t>(kFattnNbatchFa)
            ? (kv_span - tile)
            : static_cast<std::size_t>(kFattnNbatchFa));
    for (int row = 0; row < kFattnNbatchFa; ++row) {
      const bool in_tile = row < rows;
      const std::size_t absolute = tile + static_cast<std::size_t>(row);
      const __nv_bfloat16* ksrc = nullptr;
      const __nv_bfloat16* vsrc = nullptr;
      if (in_tile) {
        if (absolute < start_position) {
          const std::size_t packed = attention_kv_physical_index(
              absolute, kv_head, 0, config.capacity, config.head_width);
          ksrc = committed_key + packed;
          vsrc = committed_value + packed;
        } else {
          ksrc = candidate_key + (absolute - start_position) * row_values +
                 kv_head * width;
          vsrc = candidate_value + (absolute - start_position) * row_values +
                 kv_head * width;
        }
      }
      for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
        keys[row * kWidth + dim] =
            in_tile ? ksrc[dim] : __float2bfloat16_rn(0.0F);
        values[row * kWidth + dim] =
            in_tile ? vsrc[dim] : __float2bfloat16_rn(0.0F);
      }
    }
    __syncthreads();

    for (int m0 = 0; m0 < kNcols; m0 += 16) {
      for (int n0 = 0; n0 < kFattnNbatchFa; n0 += 8) {
        float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (int k0 = warp * 16; k0 < kWidth; k0 += kNwarps * 16) {
          std::uint32_t a[4];
          std::uint32_t a_lo[4] = {0, 0, 0, 0};
          std::uint32_t b[2];
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            const int i = mma::tile16x8_half2_i(lane, item);
            const int j = mma::tile16x8_half2_j(lane, item);
            const int qcol = m0 + i;
            __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
            __half pair_lo[2] = {__float2half_rn(0.0F),
                                 __float2half_rn(0.0F)};
            if (qcol < kNcols) {
              const int offset = qcol * kWidth + k0 + j * 2;
              pair[0] = q_f16[offset];
              pair[1] = q_f16[offset + 1];
              if constexpr (DualF16) {
                pair_lo[0] = q_lo[offset];
                pair_lo[1] = q_lo[offset + 1];
              }
            }
            a[item] = *reinterpret_cast<std::uint32_t*>(pair);
            if constexpr (DualF16) {
              a_lo[item] = *reinterpret_cast<std::uint32_t*>(pair_lo);
            }
          }
#pragma unroll
          for (int item = 0; item < 2; ++item) {
            const int n = mma::tile8x8_i(lane, item);
            const int j = mma::tile8x8_j(lane, item);
            const int krow = n0 + n;
            __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
            if (krow < rows) {
              pair[0] = __float2half_rn(__bfloat162float(
                  keys[krow * kWidth + k0 + j * 2]));
              pair[1] = __float2half_rn(__bfloat162float(
                  keys[krow * kWidth + k0 + j * 2 + 1]));
            }
            b[item] = *reinterpret_cast<std::uint32_t*>(pair);
          }
          mma::mma_m16n8k16_f16_f32(cfrag, a, b);
          if constexpr (DualF16) {
            mma::mma_m16n8k16_f16_f32(cfrag_lo, a_lo, b);
          }
        }
#pragma unroll
        for (int item = 0; item < 4; ++item)
          cparts[(warp * 32 + lane) * 4 + item] =
              cfrag[item] + cfrag_lo[item];
        __syncthreads();
        if (warp == 0) {
          float reduced[4] = {0.0F, 0.0F, 0.0F, 0.0F};
          for (int other = 0; other < kNwarps; ++other) {
#pragma unroll
            for (int item = 0; item < 4; ++item)
              reduced[item] += cparts[(other * 32 + lane) * 4 + item];
          }
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            const int i = mma::tile16x8_i(lane, item);
            const int j = mma::tile16x8_j(lane, item);
            const int qcol = m0 + i;
            const int krow = n0 + j;
            if (qcol < kNcols && krow < kFattnNbatchFa)
              scores[qcol * kFattnNbatchFa + krow] =
                  reduced[item] * attn_scale;
          }
        }
        __syncthreads();
      }
    }

    for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
      const int row = qcol % Ncols1;
      const std::size_t qpos =
          start_position + first_row + static_cast<std::size_t>(row);
      float tile_max = -INFINITY;
      for (int krow = 0; krow < rows; ++krow) {
        const std::size_t absolute = tile + static_cast<std::size_t>(krow);
        float score = scores[qcol * kFattnNbatchFa + krow];
        if (row >= active_rows || absolute > qpos) score = -INFINITY;
        scores[qcol * kFattnNbatchFa + krow] = score;
        tile_max = fmaxf(tile_max, score);
      }
      const float old_max = qmax[qcol];
      const float new_max = fmaxf(old_max, tile_max);
      const float factor =
          old_max == -INFINITY ? 0.0F : expf(old_max - new_max);
      float denominator = qden[qcol] * factor;
      for (int krow = 0; krow < rows; ++krow) {
        const float score = scores[qcol * kFattnNbatchFa + krow];
        const float weight =
            score == -INFINITY ? 0.0F : expf(score - new_max);
        scores[qcol * kFattnNbatchFa + krow] = weight;
        denominator += weight;
      }
      qmax[qcol] = new_max;
      qden[qcol] = denominator;
      rescale[qcol] = factor;
    }
    __syncthreads();

    for (int member = 0; member < kFattnNcols2; ++member) {
      const std::uint32_t query_head =
          first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                  member);
      for (int row = 0; row < active_rows; ++row) {
        const int qcol = member * Ncols1 + row;
        const std::size_t qbase =
            (first_row + static_cast<std::size_t>(row)) *
                config.query_heads * width +
            query_head * width;
        const float factor = rescale[qcol];
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          float* slot =
              is_fixup ? fattn_persist_vkq_data(meta, nblocks, qcol) + dim
                       : vkq + qbase + static_cast<std::size_t>(dim);
          float total = (*slot) * factor;
          for (int krow = 0; krow < rows; ++krow) {
            total += scores[qcol * kFattnNbatchFa + krow] *
                     __bfloat162float(values[krow * kWidth + dim]);
          }
          *slot = total;
        }
      }
    }
    __syncthreads();
  }

  for (int member = 0; member < kFattnNcols2; ++member) {
    const std::uint32_t query_head =
        first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                member);
    for (int row = 0; row < active_rows; ++row) {
      const int qcol = member * Ncols1 + row;
      const std::size_t qbase =
          (first_row + static_cast<std::size_t>(row)) * config.query_heads *
              width +
          query_head * width;
      const float denominator = qden[qcol];
      if (!needs_fixup && !is_fixup) {
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          const float g = gate[qbase + static_cast<std::size_t>(dim)];
          const float attn =
              denominator == 0.0F
                  ? 0.0F
                  : output[qbase + static_cast<std::size_t>(dim)] / denominator;
          output[qbase + static_cast<std::size_t>(dim)] =
              attn * fattn_output_gate(g);
        }
      } else if (tid == 0) {
        float* packed = needs_fixup
                            ? fattn_persist_needs_meta(meta, nblocks, qcol)
                            : fattn_persist_is_meta(meta, nblocks, qcol);
        packed[0] = qmax[qcol];
        packed[1] = denominator;
      }
    }
  }
  __syncthreads();
}

template <int Ncols1, bool DualF16, int Occupancy>
__global__ void __launch_bounds__(Ncols1 <= 8 ? 64 : 128, Occupancy)
fattn_mma_persistent_kernel(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, float* meta, int ntiles_kv,
    int ntiles_x, int ntiles_z_gqa) {
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  constexpr int kWidth = kFattnHeadWidth;
  constexpr int kNcols = Ncols1 * kFattnNcols2;
  constexpr int kNwarps = kNthreads / 32;
  (void)kNthreads;
  (void)kNwarps;
  const int nblocks = static_cast<int>(gridDim.x);
  const int total_work =
      ntiles_kv * ntiles_x * ntiles_z_gqa * static_cast<int>(config.kv_heads);
  int kbc = static_cast<int>(static_cast<long long>(blockIdx.x) * total_work /
                                 nblocks);
  const int kbc_stop = static_cast<int>(
      static_cast<long long>(blockIdx.x + 1) * total_work / nblocks);

  extern __shared__ unsigned char raw[];
  float* scratch = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(scratch + kWidth);
  __nv_bfloat16* values = keys + kFattnNbatchFa * kWidth;
  __half* q_f16 = reinterpret_cast<__half*>(values + kFattnNbatchFa * kWidth);
  __half* q_lo = q_f16 + kNcols * kWidth;
  float* scores = reinterpret_cast<float*>(
      DualF16 ? reinterpret_cast<__half*>(q_lo + kNcols * kWidth) : q_lo);
  float* qmax = scores + kNcols * kFattnNbatchFa;
  float* qden = qmax + kNcols;
  float* rescale = qden + kNcols;
  float* cparts = rescale + kNcols;
  (void)kNwarps;

  int kb0_start = ntiles_kv == 0 ? 0 : kbc % ntiles_kv;
  int kb0_stop = ntiles_kv < kb0_start + kbc_stop - kbc
                      ? ntiles_kv
                      : kb0_start + kbc_stop - kbc;

  while (kbc < kbc_stop && kb0_stop == ntiles_kv) {
    int z_KV = 0, zt_gqa = 0, jt = 0;
    fattn_persistent_decode(kbc, ntiles_kv, ntiles_x, ntiles_z_gqa, &z_KV,
                             &zt_gqa, &jt);
    const bool needs_fixup = kb0_start != 0;
    fattn_persistent_process_tile<Ncols1, DualF16>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, meta, nblocks, jt, zt_gqa, z_KV, kb0_start,
        kb0_stop, needs_fixup, false, scratch, keys, values, q_f16, q_lo,
        scores, qmax, qden, rescale, cparts);
    kbc += ntiles_kv;
    kbc -= kbc % ntiles_kv;
    kb0_start = 0;
    kb0_stop = ntiles_kv < kbc_stop - kbc ? ntiles_kv : kbc_stop - kbc;
  }

  if (kbc >= kbc_stop) return;
  int z_KV = 0, zt_gqa = 0, jt = 0;
  fattn_persistent_decode(kbc, ntiles_kv, ntiles_x, ntiles_z_gqa, &z_KV,
                           &zt_gqa, &jt);
  fattn_persistent_process_tile<Ncols1, DualF16>(
      config, start_position, token_count, query, query_scale, gate,
      committed_key, committed_value, candidate_key, candidate_value, output,
      normalized_query, meta, nblocks, jt, zt_gqa, z_KV, kb0_start, kb0_stop,
      false, true, scratch, keys, values, q_f16, q_lo, scores, qmax, qden,
      rescale, cparts);
}

__global__ void fattn_persistent_uniform_fixup(
    AttentionConfig config, std::size_t token_count, const float* gate,
    float* output, const float* meta, int nblocks, int ntiles_x,
    int ntiles_z_gqa) {
  constexpr int kNcols = kSelectedAttentionMmaQueryRows * kFattnNcols2;
  constexpr int kNcols1 = kSelectedAttentionMmaQueryRows;
  const std::size_t width = config.head_width;
  const std::size_t values =
      token_count * static_cast<std::size_t>(config.query_heads) * width;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                static_cast<std::size_t>(blockDim.x) +
                            static_cast<std::size_t>(threadIdx.x);
  if (index >= values) return;
  const std::size_t row_values =
      static_cast<std::size_t>(config.query_heads) * width;
  const std::size_t row = index / row_values;
  const std::size_t in_row = index % row_values;
  const int head = static_cast<int>(in_row / width);
  const int dim = static_cast<int>(in_row % width);
  const int gqa = static_cast<int>(config.query_heads / config.kv_heads);
  const int kv_head = head / gqa;
  const int gqa_off = head % gqa;
  const int zt_gqa = gqa_off / kFattnNcols2;
  const int member = gqa_off % kFattnNcols2;
  const int jt = static_cast<int>(row) / kNcols1;
  const int row_in = static_cast<int>(row) % kNcols1;
  const int qcol = member * kNcols1 + row_in;
  const int ntiles_dst = ntiles_x * ntiles_z_gqa * static_cast<int>(config.kv_heads);
  const int bpt = nblocks / ntiles_dst;
  const int dest_id =
      (kv_head * ntiles_z_gqa + zt_gqa) * ntiles_x + jt;
  const int b_first = dest_id * bpt;
  const int b_last = b_first + bpt - 1;
  float vkq = output[index];
  const float* last_meta = meta + (b_last * kNcols + qcol) * 2;
  float mx = last_meta[0];
  float den = last_meta[1];
  const float* data = meta + nblocks * kNcols * 4;
  for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
    const float add = data[(bidx * kNcols + qcol) * static_cast<int>(width) + dim];
    const float* add_meta = meta + ((nblocks + bidx) * kNcols + qcol) * 2;
    const float mx_add = add_meta[0];
    const float den_add = add_meta[1];
    const float new_max = fmaxf(mx, mx_add);
    const float scale0 = mx == -INFINITY ? 0.0F : expf(mx - new_max);
    const float scale1 = mx_add == -INFINITY ? 0.0F : expf(mx_add - new_max);
    vkq = vkq * scale0 + add * scale1;
    den = den * scale0 + den_add * scale1;
    mx = new_max;
  }
  const float attn = den == 0.0F ? 0.0F : vkq / den;
  output[index] = attn * fattn_output_gate(gate[index]);
}

__global__ void fattn_persistent_general_fixup(
    AttentionConfig config, std::size_t token_count, const float* gate,
    float* output, const float* meta, int ntiles_kv, int ntiles_x,
    int ntiles_z_gqa, int total_work) {
  constexpr int kNcols = kSelectedAttentionMmaQueryRows * kFattnNcols2;
  constexpr int kNcols1 = kSelectedAttentionMmaQueryRows;
  const int nblocks = static_cast<int>(gridDim.x);
  const int bidx0 = static_cast<int>(blockIdx.x);
  const int j = static_cast<int>(blockIdx.y);
  const int c = static_cast<int>(blockIdx.z);
  const int qcol = c * kNcols1 + j;
  const int tid = static_cast<int>(threadIdx.x);
  const int width = static_cast<int>(config.head_width);
  if (tid >= width) return;
  const int kbc0 = static_cast<int>(
      static_cast<long long>(bidx0) * total_work / nblocks);
  const int kbc0_stop = static_cast<int>(
      static_cast<long long>(bidx0 + 1) * total_work / nblocks);
  const bool did_not_have_any_data = kbc0 == kbc0_stop;
  const bool wrote_beginning_of_tile =
      ntiles_kv > 0 && kbc0 % ntiles_kv == 0;
  const bool did_not_write_last =
      ntiles_kv > 0 && kbc0 / ntiles_kv == kbc0_stop / ntiles_kv &&
      kbc0_stop % ntiles_kv != 0;
  if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last)
    return;
  const int dest = kbc0 / ntiles_kv;
  const int span = ntiles_x * ntiles_z_gqa;
  const int z_KV = dest / span;
  const int rem = dest - z_KV * span;
  const int zt_gqa = rem / ntiles_x;
  const int jt = rem - zt_gqa * ntiles_x;
  const int gqa = static_cast<int>(config.query_heads / config.kv_heads);
  if (jt * kNcols1 + j >= static_cast<int>(token_count) ||
      zt_gqa * kFattnNcols2 + c >= gqa)
    return;
  const int head = z_KV * gqa + zt_gqa * kFattnNcols2 + c;
  const int row = jt * kNcols1 + j;
  const std::size_t index =
      (static_cast<std::size_t>(row) *
           static_cast<std::size_t>(config.query_heads) +
       static_cast<std::size_t>(head)) *
          static_cast<std::size_t>(width) +
      static_cast<std::size_t>(tid);
  float vkq = output[index];
  const float* last_meta = meta + (bidx0 * kNcols + qcol) * 2;
  float mx = last_meta[0];
  float den = last_meta[1];
  const float* data = meta + nblocks * kNcols * 4;
  const int tile_kbc0 = kbc0 / ntiles_kv;
  int bidx = bidx0 - 1;
  int kbc_stop = kbc0;
  while (bidx >= 0) {
    const int kbc = static_cast<int>(
        static_cast<long long>(bidx) * total_work / nblocks);
    if (kbc == kbc_stop) {
      --bidx;
      kbc_stop = kbc;
      continue;
    }
    const float add =
        data[(bidx * kNcols + qcol) * width + tid];
    const float* add_meta = meta + ((nblocks + bidx) * kNcols + qcol) * 2;
    const float mx_add = add_meta[0];
    const float den_add = add_meta[1];
    const float new_max = fmaxf(mx, mx_add);
    const float scale0 = mx == -INFINITY ? 0.0F : expf(mx - new_max);
    const float scale1 = mx_add == -INFINITY ? 0.0F : expf(mx_add - new_max);
    vkq = vkq * scale0 + add * scale1;
    den = den * scale0 + den_add * scale1;
    mx = new_max;
    if (kbc % ntiles_kv == 0 || kbc / ntiles_kv < tile_kbc0) break;
    --bidx;
    kbc_stop = kbc;
  }
  const float attn = den == 0.0F ? 0.0F : vkq / den;
  output[index] = attn * fattn_output_gate(gate[index]);
}

template <int Ncols1, bool DualF16, int Occupancy>
inline int fattn_persistent_occupancy_typed() noexcept {
  int occupancy = 0;
  const int threads = fattn_nthreads_for_ncols1(Ncols1);
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_persistent_kernel<Ncols1, DualF16, Occupancy>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error == cudaSuccess)
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, fattn_mma_persistent_kernel<Ncols1, DualF16, Occupancy>,
        threads, shared);
  return error == cudaSuccess ? occupancy : 0;
}

inline cudaError_t launch_fattn_mma_persistent_impl(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* meta, cudaStream_t stream) noexcept {
  constexpr int Ncols1 = 16;
  constexpr bool DualF16 = true;
  constexpr int Occupancy = 2;
  constexpr int kNthreads = 128;
  if (meta == nullptr || token_count < 16) return cudaErrorInvalidValue;
  const int ntiles_x = fattn_persistent_ntiles_x(token_count);
  const int ntiles_z_gqa = fattn_persistent_ntiles_z_gqa(config);
  const int ntiles_dst = fattn_persistent_ntiles_dst(config, token_count);
  const int ntiles_kv = fattn_persistent_ntiles_kv(start_position, token_count);
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_persistent_kernel<Ncols1, DualF16, Occupancy>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  int occupancy = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, fattn_mma_persistent_kernel<Ncols1, DualF16, Occupancy>,
      kNthreads, shared);
  if (error != cudaSuccess || occupancy < 1) return cudaErrorInvalidValue;
  int device = 0;
  cudaDeviceProp prop{};
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  error = cudaGetDeviceProperties(&prop, device);
  if (error != cudaSuccess) return error;
  const int nblocks = fattn_compute_persistent_nblocks(
      prop.multiProcessorCount, occupancy, ntiles_dst, ntiles_kv);
  if (nblocks <= 0) return cudaErrorInvalidValue;
  fattn_mma_persistent_kernel<Ncols1, DualF16, Occupancy>
      <<<dim3(nblocks, 1, 1), kNthreads, shared, stream>>>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value, output,
          normalized_query, meta, ntiles_kv, ntiles_x, ntiles_z_gqa);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  const int total_work = ntiles_kv * ntiles_dst;
  if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
    const std::size_t values =
        token_count * static_cast<std::size_t>(config.query_heads) *
        config.head_width;
    const unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((values + threads - 1) / threads);
    fattn_persistent_uniform_fixup<<<blocks, threads, 0, stream>>>(
        config, token_count, gate, output, meta, nblocks, ntiles_x,
        ntiles_z_gqa);
  } else if (ntiles_dst % nblocks != 0) {
    fattn_persistent_general_fixup<<<dim3(nblocks, Ncols1, kFattnNcols2),
                                     256, 0, stream>>>(
        config, token_count, gate, output, meta, ntiles_kv, ntiles_x,
        ntiles_z_gqa, total_work);
  }
  return cudaPeekAtLastError();
}

template <int Ncols1, bool DualF16, int Occupancy, int KvParts,
          bool RegisterAcc = false, bool MmaPv = false, bool WarpQK = false,
          bool PreparedQ = false>
inline cudaError_t launch_fattn_mma_quality_typed(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta,
    cudaStream_t stream, const __half* prepared_q = nullptr,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept {
  if constexpr (PreparedQ) {
    if (prepared_q == nullptr) return cudaErrorInvalidValue;
  }
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  dim3 grid(config.kv_heads,
            static_cast<unsigned>((token_count + Ncols1 - 1) / Ncols1),
            KvParts > 1 ? 2U : 1U);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts, RegisterAcc,
                               MmaPv, WarpQK, PreparedQ>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  error = quartz_launch_kernel(
      fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts, RegisterAcc,
                               MmaPv, WarpQK, PreparedQ>,
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

#include "fattn_mma_f16_pipeline.cuh"

inline bool fattn_vkq_is_registers(const char* vkq_accum) noexcept {
  return vkq_accum != nullptr && std::strcmp(vkq_accum, "registers") == 0;
}

inline bool fattn_vkq_is_global(const char* vkq_accum) noexcept {
  return vkq_accum != nullptr && std::strcmp(vkq_accum, "global") == 0;
}

inline bool fattn_pv_is_mma(const char* pv_path) noexcept {
  return pv_path != nullptr && std::strcmp(pv_path, "mma") == 0;
}

inline bool fattn_pv_is_scalar(const char* pv_path) noexcept {
  return pv_path != nullptr && std::strcmp(pv_path, "scalar") == 0;
}

inline bool fattn_qk_is_cparts(const char* qk_path) noexcept {
  return qk_path != nullptr && std::strcmp(qk_path, "cparts") == 0;
}

inline bool fattn_qk_is_warp_microtile(const char* qk_path) noexcept {
  return qk_path != nullptr && std::strcmp(qk_path, "warp_microtile") == 0;
}

inline cudaError_t launch_fattn_mma_quality_path(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, int ncols1, const char* path, float* partial,
    float* meta, cudaStream_t stream,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept {
  const bool persistent = path != nullptr && path[0] == 'p';
  const bool stream_k = path != nullptr && path[0] == 's';
  const bool occ2 = stream_k || (path != nullptr && path[0] == 'o');
  if (persistent) {
    if (meta == nullptr || ncols1 != 16) return cudaErrorInvalidValue;
    return launch_fattn_mma_persistent_impl(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value,
        output, normalized_query, meta, stream);
  }
  if (stream_k && (partial == nullptr || meta == nullptr))
    return cudaErrorInvalidValue;
  if (ncols1 == 16 && stream_k) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, nullptr, kv_origin);
  }
  if (ncols1 == 16 && occ2) {
    return launch_fattn_mma_quality_typed<16, true, 2, 1>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, nullptr, nullptr, stream, nullptr, kv_origin);
  }
  switch (ncols1) {
    case 8:
      return launch_fattn_mma_quality_typed<8, true, 1, 1>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, nullptr, nullptr, stream, nullptr,
          kv_origin);
    case 16:
      return launch_fattn_mma_quality_typed<16, true, 1, 1>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, nullptr, nullptr, stream, nullptr,
          kv_origin);
    case 32:
      return launch_fattn_mma_quality_typed<32, false, 1, 1>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, nullptr, nullptr, stream, nullptr,
          kv_origin);
    default:
      return cudaErrorInvalidValue;
  }
}

inline cudaError_t launch_fattn_mma_quality(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, int ncols1, cudaStream_t stream,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept {
  const char* path = kSelectedFattnPath;
  if (path[0] == 's') path = "occ2";
  return launch_fattn_mma_quality_path(
      config, start_position, token_count, query, query_scale, gate,
      committed_key, committed_value, candidate_key, candidate_value, output,
      normalized_query, ncols1, path, nullptr, nullptr, stream, kv_origin);
}

inline cudaError_t launch_fattn_mma_stream_k_vkq(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta, const char* vkq_accum,
    cudaStream_t stream) noexcept {
  if (fattn_vkq_is_registers(vkq_accum)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, true>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  if (fattn_vkq_is_global(vkq_accum)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, false>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  return cudaErrorInvalidValue;
}

inline cudaError_t launch_fattn_mma_stream_k_pv(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta, const char* pv_path,
    cudaStream_t stream) noexcept {
  if (fattn_pv_is_mma(pv_path)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, true, true>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  if (fattn_pv_is_scalar(pv_path)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, true, false>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  return cudaErrorInvalidValue;
}

inline cudaError_t launch_fattn_mma_stream_k_qk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta, const char* qk_path,
    cudaStream_t stream) noexcept {
  if (fattn_qk_is_warp_microtile(qk_path)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, true, true, true>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  if (fattn_qk_is_cparts(qk_path)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, true, true, false>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  return cudaErrorInvalidValue;
}

inline cudaError_t launch_fattn_mma_stream_k(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta,
    cudaStream_t stream, const __half* prepared_q = nullptr,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept {
  if (kSelectedPersistentFattnPath[0] == 'p') {
    return launch_fattn_mma_persistent_impl(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, meta, stream);
  }
  const char* prep = effective_query_prepare_path();
  const char* pipe = effective_attention_pipeline_path();
  if (fattn_query_prepare_is_hoisted(prep) && prepared_q != nullptr &&
      fattn_vkq_is_registers(kSelectedVkqAccum) &&
      fattn_pv_is_mma(kSelectedPvPath) &&
      fattn_qk_is_warp_microtile(kSelectedQKPath) &&
      !fattn_pipeline_is_off(pipe) && legal_attention_pipeline_path(pipe)) {
    return launch_fattn_mma_pipeline(
        pipe, config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (fattn_query_prepare_is_hoisted(prep) && prepared_q != nullptr &&
      fattn_vkq_is_registers(kSelectedVkqAccum) &&
      fattn_pv_is_mma(kSelectedPvPath) &&
      fattn_qk_is_warp_microtile(kSelectedQKPath)) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2, true, true, true,
                                          true>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream, prepared_q, kv_origin);
  }
  if (fattn_vkq_is_registers(kSelectedVkqAccum)) {
    if (fattn_pv_is_mma(kSelectedPvPath)) {
      return launch_fattn_mma_stream_k_qk(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, partial, meta, kSelectedQKPath, stream);
    }
    return launch_fattn_mma_stream_k_pv(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, kSelectedPvPath, stream);
  }
  return launch_fattn_mma_quality_path(
      config, start_position, token_count, query, query_scale, gate,
      committed_key, committed_value, candidate_key, candidate_value, output,
      normalized_query, kSelectedAttentionMmaQueryRows, "stream_k", partial,
      meta, stream);
}

template <int Ncols1, bool DualF16, int Occupancy, int KvParts,
          bool RegisterAcc = false, bool MmaPv = false, bool WarpQK = false,
          bool PreparedQ = false>
inline int fattn_occupancy_typed() noexcept {
  int occupancy = 0;
  const int threads = fattn_nthreads_for_ncols1(Ncols1);
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts, RegisterAcc,
                               MmaPv, WarpQK, PreparedQ>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error == cudaSuccess)
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts, RegisterAcc,
                                 MmaPv, WarpQK, PreparedQ>,
        threads, shared);
  return error == cudaSuccess ? occupancy : 0;
}

inline int fattn_register_vkq_occupancy_typed() noexcept {
  return fattn_occupancy_typed<16, true, 2, 2, true>();
}

inline int fattn_pv_mma_occupancy_typed() noexcept {
  return fattn_occupancy_typed<16, true, 2, 2, true, true>();
}

inline int fattn_warp_qk_occupancy_typed() noexcept {
  return fattn_occupancy_typed<16, true, 2, 2, true, true, true>();
}

inline int fattn_prepared_query_occupancy_typed() noexcept {
  return fattn_occupancy_typed<16, true, 2, 2, true, true, true, true>();
}

inline int fattn_occupancy_for(int ncols1) noexcept {
  if (ncols1 == 8) return fattn_occupancy_typed<8, true, 1, 1>();
  if (ncols1 == 16) {
    if (kSelectedFattnPath[0] == 's' || kSelectedFattnPath[0] == 'o')
      return fattn_occupancy_typed<16, true, 2, 1>();
    return fattn_occupancy_typed<16, true, 1, 1>();
  }
  if (ncols1 == 32) return fattn_occupancy_typed<32, false, 1, 1>();
  return 0;
}

inline int fattn_occupancy_for_path(const char* path) noexcept {
  if (path != nullptr && path[0] == 'p')
    return fattn_persistent_occupancy_typed<16, true, 2>();
  if (path != nullptr && path[0] == 's')
    return fattn_occupancy_typed<16, true, 2, 2>();
  if (path != nullptr && path[0] == 'o')
    return fattn_occupancy_typed<16, true, 2, 1>();
  return fattn_occupancy_typed<16, true, 1, 1>();
}

}  // namespace qw38::cuda
