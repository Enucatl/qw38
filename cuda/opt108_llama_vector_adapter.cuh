#pragma once

// OPT-108 source-faithful NVIDIA adapter of pinned llama.cpp
// flash_attn_ext_vec<256,1> (MIT, The ggml authors, revision
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328). Namespaced Quartz adapter; does
// not vendor ggml tensor/runtime machinery. SM120 follows the float2 V
// accumulator branch: V_DOT2_F32_F16_AVAILABLE is AMD/HIP-only in the pin.

#include "attention_decode.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt108 {

constexpr char kCandidateId[] = "llama_vec_nvidia";
constexpr char kLaunchName[] = "opt108_llama_vec_nvidia";
constexpr int kD = 256;
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarps = 4;
constexpr int kCpyBytes = 16;
constexpr int kCpyNe = 4;  // 16-byte copy / sizeof(float)
constexpr int kNbatchFa = kD;
constexpr int kMaxParts = 64;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr bool kNvidiaFloat2VAccum = true;
constexpr bool kAmdHalf2VAccum = false;
constexpr bool kF16CacheMigration = false;
constexpr float kScale = 0.0625F;  // 1/sqrt(256)
constexpr float kKqMaxOffset = 3.0F * 0.6931F;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;

// NVIDIA BF16 path: nthreads_KQ = nthreads_V = 128 / 16 = 8.
constexpr int kNthreadsKq = 8;
constexpr int kNthreadsV = 8;
constexpr int kVRowsPerThread = 8;  // 2 * cpy_ne
constexpr int kVColsPerIter = 4;    // WARP_SIZE / nthreads_V
constexpr int kQRegs = (kD / 2) / kNthreadsKq;  // 16 float2
constexpr int kNeCombine = kWarps * kVColsPerIter * kD;  // 4096 floats

static_assert(kNvidiaFloat2VAccum && !kAmdHalf2VAccum,
              "OPT-108 is the NVIDIA float2 path, not AMD half2");
static_assert(kCpyBytes == 16, "pinned Volta+ copy width is 16 bytes");

struct LaunchInfo final {
  int n_parts = 1;
  int occupancy_per_sm = 0;
  int nsm = 0;
  int ntiles_kv = 0;
  int n_kv = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  std::size_t shared_bytes = 0;
  std::size_t staging_bytes = 0;
  bool prepared_q_once = true;
  bool bf16_kv = true;
  bool f16_cache_migration = false;
  bool nvidia_float2_v_accum = true;
  bool amd_half2_v_accum = false;
};

inline thread_local LaunchInfo g_last_launch{};

inline const LaunchInfo& last_launch() noexcept { return g_last_launch; }

inline int nsm_count() noexcept {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
  return prop.multiProcessorCount;
}

inline int parallel_blocks_for_kv(int n_kv, int occupancy_per_sm,
                                  int nsm) noexcept {
  if (n_kv < 1) n_kv = 1;
  const int ntiles_kv = (n_kv + kNbatchFa - 1) / kNbatchFa;
  const int ntiles_dst = kQueryHeads;
  const int occ = occupancy_per_sm > 0 ? occupancy_per_sm : 1;
  const int sm = nsm > 0 ? nsm : 1;
  int parallel_blocks = occ < ntiles_kv ? occ : ntiles_kv;
  if (parallel_blocks < 1) parallel_blocks = 1;
  const int blocks_per_wave = sm * occ;
  int nwaves_best = 0;
  int efficiency_best = 0;
  for (int test = parallel_blocks; test <= ntiles_kv; ++test) {
    const int nblocks_total = ntiles_dst * test;
    const int nwaves =
        (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
    const int efficiency =
        blocks_per_wave > 0 ? (100 * nblocks_total) / (nwaves * blocks_per_wave)
                            : 0;
    if (efficiency_best >= 95 && nwaves > nwaves_best) break;
    if (efficiency > efficiency_best) {
      nwaves_best = nwaves;
      efficiency_best = efficiency;
      parallel_blocks = test;
    }
  }
  if (parallel_blocks > kMaxParts) parallel_blocks = kMaxParts;
  return parallel_blocks;
}

__device__ __forceinline__ void cpy16(void* dst, const void* src) {
  *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
}

template <int Width>
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, Width);
  }
  return x;
}

template <int Width>
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, Width));
  }
  return x;
}

__device__ __forceinline__ float2 bf162_to_float2(nv_bfloat162 value) {
#if __CUDA_ARCH__ >= 800
  return __bfloat1622float2(value);
#else
  return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
#endif
}

__device__ __forceinline__ float mad_float2(float acc, float2 left,
                                            float2 right) {
  acc += left.x * right.x;
  acc += left.y * right.y;
  return acc;
}

__device__ __forceinline__ const __nv_bfloat16* kv_row(
    std::size_t token, std::size_t position, std::uint32_t kv_head,
    std::uint32_t capacity, const __nv_bfloat16* committed,
    const __nv_bfloat16* candidate) {
  if (token < position) {
    return committed + attention_kv_physical_index(token, kv_head, 0, capacity,
                                                   static_cast<std::uint32_t>(kD));
  }
  const std::size_t row =
      static_cast<std::size_t>(kKvHeads) * static_cast<std::size_t>(kD);
  return candidate + (token - position) * row +
         static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(kD);
}

// NVIDIA float2 / BF16 specialization of flash_attn_ext_vec<256,1>.
__global__ void __launch_bounds__(kThreads, 1)
flash_attn_ext_vec_nvidia_bf16(
    std::size_t position, std::uint32_t capacity, const float* prepared_q,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* dst, float2* dst_meta) {
  constexpr int nthreads_kq = kNthreadsKq;
  constexpr int nthreads_v = kNthreadsV;
  const int n_kv = static_cast<int>(position + 1);
  const int head = static_cast<int>(blockIdx.z);
  const int part = static_cast<int>(blockIdx.y);
  const int tid = kWarpSize * static_cast<int>(threadIdx.y) +
                  static_cast<int>(threadIdx.x);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(head) /
      (static_cast<std::uint32_t>(kQueryHeads) /
       static_cast<std::uint32_t>(kKvHeads));
  const float2* q_head = reinterpret_cast<const float2*>(
      prepared_q + static_cast<std::size_t>(head) * kD);

  float2 vkq[kQRegs];
  float2 q_reg[kQRegs];
#pragma unroll
  for (int i = 0; i < kQRegs; ++i) {
    vkq[i] = make_float2(0.0F, 0.0F);
    q_reg[i] = make_float2(0.0F, 0.0F);
  }
  float kq_max = -FLT_MAX / 2.0F;
  float kq_sum = 0.0F;

#pragma unroll
  for (int i0 = 0; i0 < kD / 2; i0 += nthreads_kq * kCpyNe) {
    const int i =
        i0 + (static_cast<int>(threadIdx.x) % nthreads_kq) * kCpyNe;
    cpy16(&q_reg[i0 / nthreads_kq], &q_head[i]);
    cpy16(&q_reg[i0 / nthreads_kq + kCpyNe / 2], &q_head[i + kCpyNe / 2]);
  }
#pragma unroll
  for (int k = 0; k < kQRegs; ++k) {
    q_reg[k].x *= kScale;
    q_reg[k].y *= kScale;
  }

  __shared__ float kq_shared[kNeCombine];
  const int k_vkq_max = n_kv;
  for (int k_vkq0 = part * kThreads; k_vkq0 < k_vkq_max;
       k_vkq0 += static_cast<int>(gridDim.y) * kThreads) {
    float kq_reg = 0.0F;
    float kq_max_new = kq_max;
#pragma unroll
    for (int i_kq0 = 0; i_kq0 < nthreads_kq; ++i_kq0) {
      const int i_kq =
          static_cast<int>(threadIdx.y) * kWarpSize +
          (static_cast<int>(threadIdx.x) & ~(nthreads_kq - 1)) + i_kq0;
      const int token = k_vkq0 + i_kq;
      float sum = 0.0F;
      if (token < n_kv) {
        const nv_bfloat162* k_row = reinterpret_cast<const nv_bfloat162*>(
            kv_row(static_cast<std::size_t>(token), position, kv_head, capacity,
                   committed_key, candidate_key));
#pragma unroll
        for (int k0 = 0; k0 < kD / 2; k0 += nthreads_kq * kCpyNe) {
          __align__(16) nv_bfloat162 tmp[kCpyNe];
          cpy16(tmp, k_row + k0 +
                         (static_cast<int>(threadIdx.x) % nthreads_kq) * kCpyNe);
#pragma unroll
          for (int k1 = 0; k1 < kCpyNe; ++k1) {
            sum = mad_float2(sum, bf162_to_float2(tmp[k1]),
                             q_reg[k0 / nthreads_kq + k1]);
          }
        }
      }
      sum = warp_sum<nthreads_kq>(sum);
      if (token < n_kv) {
        kq_max_new = fmaxf(kq_max_new, sum + kKqMaxOffset);
      }
      if ((static_cast<int>(threadIdx.x) % nthreads_kq) == i_kq0) {
        kq_reg = token < n_kv ? sum : 0.0F;
      }
    }
#pragma unroll
    for (int offset = nthreads_kq; offset < kWarpSize; offset <<= 1) {
      kq_max_new =
          fmaxf(kq_max_new, __shfl_xor_sync(0xffffffff, kq_max_new, offset,
                                            kWarpSize));
    }
    const float kq_max_scale = expf(kq_max - kq_max_new);
    kq_max = kq_max_new;
    const int my_token = k_vkq0 + tid;
    if (my_token < n_kv) {
      kq_reg = expf(kq_reg - kq_max);
    } else {
      kq_reg = 0.0F;
    }
    kq_sum = kq_sum * kq_max_scale + kq_reg;
    kq_shared[tid] = kq_reg;
#pragma unroll
    for (int i = 0; i < kQRegs; ++i) {
      vkq[i].x *= kq_max_scale;
      vkq[i].y *= kq_max_scale;
    }
    __syncwarp();
#pragma unroll
    for (int k0 = 0; k0 < kWarpSize; k0 += kVColsPerIter) {
      const int k =
          static_cast<int>(threadIdx.y) * kWarpSize + k0 +
          static_cast<int>(threadIdx.x) / nthreads_v;
      const int token = k_vkq0 + k;
      const float kq_k = kq_shared[k];
#pragma unroll
      for (int i0 = 0; i0 < kD / 2;
           i0 += nthreads_v * kVRowsPerThread / 2) {
        float2 tmp[kVRowsPerThread / 2];
#pragma unroll
        for (int t = 0; t < kVRowsPerThread / 2; ++t) {
          tmp[t] = make_float2(0.0F, 0.0F);
        }
        if (token < n_kv) {
          const __nv_bfloat16* v_row =
              kv_row(static_cast<std::size_t>(token), position, kv_head,
                     capacity, committed_value, candidate_value);
          const int i_v =
              2 * i0 +
              (static_cast<int>(threadIdx.x) % nthreads_v) * kVRowsPerThread;
          __align__(16) nv_bfloat162 packed[kVRowsPerThread / 2];
          cpy16(packed, v_row + i_v);
#pragma unroll
          for (int t = 0; t < kVRowsPerThread / 2; ++t) {
            tmp[t] = bf162_to_float2(packed[t]);
          }
        }
#pragma unroll
        for (int t = 0; t < kVRowsPerThread / 2; ++t) {
          vkq[i0 / nthreads_v + t].x += tmp[t].x * kq_k;
          vkq[i0 / nthreads_v + t].y += tmp[t].y * kq_k;
        }
      }
    }
  }

  __shared__ float kq_max_shared[kWarpSize];
  __shared__ float kq_sum_shared[kWarpSize];
  if (threadIdx.y == 0) {
    kq_max_shared[threadIdx.x] = -FLT_MAX / 2.0F;
    kq_sum_shared[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0) kq_max_shared[threadIdx.y] = kq_max;
  __syncthreads();

  float kqmax_new = kq_max_shared[threadIdx.x];
  kqmax_new = warp_max<kWarpSize>(kqmax_new);
  const float kqmax_scale = expf(kq_max - kqmax_new);
  kq_max = kqmax_new;
#pragma unroll
  for (int i = 0; i < kQRegs; ++i) {
    vkq[i].x *= kqmax_scale;
    vkq[i].y *= kqmax_scale;
  }
  float2* vkq_tmp =
      reinterpret_cast<float2*>(kq_shared) +
      static_cast<int>(threadIdx.y) * (kVColsPerIter * kD / 2) +
      (static_cast<int>(threadIdx.x) / nthreads_v) * (kD / 2);
#pragma unroll
  for (int i0 = 0; i0 < kD / 2; i0 += nthreads_v * kVRowsPerThread / 2) {
    const int i_vkq =
        i0 + (static_cast<int>(threadIdx.x) % nthreads_v) *
                 (kVRowsPerThread / 2);
    cpy16(vkq_tmp + i_vkq, &vkq[i0 / nthreads_v]);
    cpy16(vkq_tmp + i_vkq + kVRowsPerThread / 4,
          &vkq[i0 / nthreads_v + kVRowsPerThread / 4]);
  }
  kq_sum *= kqmax_scale;
  kq_sum = warp_sum<kWarpSize>(kq_sum);
  if (threadIdx.x == 0) kq_sum_shared[threadIdx.y] = kq_sum;
  __syncthreads();

  kq_sum = kq_sum_shared[threadIdx.x];
  kq_sum = warp_sum<kWarpSize>(kq_sum);
#pragma unroll
  for (int i0 = 0; i0 < kD; i0 += kThreads) {
    float dst_val = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
#pragma unroll
      for (int v = 0; v < kVColsPerIter; ++v) {
        dst_val += kq_shared[w * kVColsPerIter * kD + v * kD + i0 + tid];
      }
    }
    if (gridDim.y == 1) {
      dst_val = (kq_sum > 0.0F) ? dst_val / kq_sum : 0.0F;
    }
    dst[(static_cast<std::size_t>(head) * gridDim.y +
         static_cast<std::size_t>(part)) *
            kD +
        static_cast<std::size_t>(i0 + tid)] = dst_val;
  }
  if (gridDim.y != 1 && tid == 0) {
    dst_meta[static_cast<std::size_t>(head) * gridDim.y +
             static_cast<std::size_t>(part)] = make_float2(kq_max, kq_sum);
  }
}

__global__ void __launch_bounds__(kD, 1)
combine_and_gate(int n_parts, const float* parts, const float2* meta,
                 const float* gate, float* output) {
  const int head = static_cast<int>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  extern __shared__ float2 meta_shared[];
  for (int i = dim; i < n_parts; i += kD) meta_shared[i] = meta[head * n_parts + i];
  __syncthreads();
  float kqmax = meta_shared[0].x;
  for (int p = 1; p < n_parts; ++p) kqmax = fmaxf(kqmax, meta_shared[p].x);
  float numerator = 0.0F;
  float denominator = 0.0F;
  const float* head_parts =
      parts + static_cast<std::size_t>(head) * n_parts * kD;
  for (int p = 0; p < n_parts; ++p) {
    const float scale = expf(meta_shared[p].x - kqmax);
    numerator += scale * head_parts[p * kD + dim];
    denominator += scale * meta_shared[p].y;
  }
  const float ungated =
      (denominator == 0.0F || kqmax == -FLT_MAX / 2.0F) ? 0.0F
                                                        : numerator / denominator;
  const std::size_t qbase = static_cast<std::size_t>(head) * kD + dim;
  const float g = gate[qbase];
  const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                  : expf(g) / (1.0F + expf(g));
  output[qbase] = ungated * sigmoid;
}

__global__ void apply_gate(const float* ungated, const float* gate,
                           float* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * kD +
                            static_cast<std::size_t>(threadIdx.x);
  const float g = gate[index];
  const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                  : expf(g) / (1.0F + expf(g));
  output[index] = ungated[index] * sigmoid;
}

// Quartz candidate-row stage (RMS+RoPE on K, BF16 V). Counted as adapter
// staging, not a long-lived cache rewrite.
__global__ void stage_current_kv(
    AttentionConfig config, std::size_t position, const float* key,
    const float* value, const float* key_scale, __nv_bfloat16* candidate_key,
    __nv_bfloat16* candidate_value, float* normalized_key) {
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::size_t width = config.head_width;
  const std::size_t base =
      static_cast<std::size_t>(kv_head) * width;
  __shared__ float normalized[kD];
  __shared__ float inverse;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < config.head_width; ++i) {
      const float item = key[base + i];
      sum += item * item;
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
  }
  __syncthreads();
  if (lane < width)
    normalized[lane] = key[base + lane] * inverse * key_scale[lane];
  __syncthreads();
  const std::uint32_t half = config.rotary_width / 2;
  if (lane < half) {
    const float first = normalized[lane];
    const float second = normalized[half + lane];
    const float exponent =
        static_cast<float>(lane * 2) / static_cast<float>(config.rotary_width);
    const float angle = static_cast<float>(position) / powf(kRopeTheta, exponent);
    const float c = cosf(angle);
    const float s = sinf(angle);
    normalized[lane] = first * c - second * s;
    normalized[half + lane] = second * c + first * s;
  }
  __syncthreads();
  if (lane < width) {
    normalized_key[base + lane] = normalized[lane];
    candidate_key[base + lane] = __float2bfloat16_rn(normalized[lane]);
    candidate_value[base + lane] = __float2bfloat16_rn(value[base + lane]);
  }
}

inline int query_occupancy() noexcept {
  int blocks = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, flash_attn_ext_vec_nvidia_bf16, kThreads, 0);
  if (error != cudaSuccess) return 0;
  return blocks;
}

inline void kernel_attributes(int* registers, std::size_t* local_bytes,
                              int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, flash_attn_ext_vec_nvidia_bf16);
  if (registers != nullptr) *registers = attrs.numRegs;
  if (local_bytes != nullptr) {
    *local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  }
  if (occupancy != nullptr) *occupancy = query_occupancy();
}

inline std::size_t partial_vkq_values(int n_parts) noexcept {
  return static_cast<std::size_t>(kQueryHeads) * static_cast<std::size_t>(n_parts) *
         static_cast<std::size_t>(kD);
}

inline std::size_t partial_meta_values(int n_parts) noexcept {
  return static_cast<std::size_t>(kQueryHeads) * static_cast<std::size_t>(n_parts) *
         2U;
}

inline cudaError_t launch_llama_vec_stack(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* output,
    float* partial_vkq, float* meta, int n_parts_override,
    cudaStream_t stream) noexcept {
  if (config.query_heads != static_cast<std::uint32_t>(kQueryHeads) ||
      config.kv_heads != static_cast<std::uint32_t>(kKvHeads) ||
      config.head_width != static_cast<std::uint32_t>(kD) ||
      query == nullptr || key == nullptr || value == nullptr ||
      query_norm_scale == nullptr || key_norm_scale == nullptr ||
      output_gate == nullptr || committed.key == nullptr ||
      committed.value == nullptr || candidate_row.key == nullptr ||
      candidate_row.value == nullptr || normalized_query == nullptr ||
      normalized_key == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  dim3 staging(config.kv_heads, 1, 1);
  stage_current_kv<<<staging, 256, 0, stream>>>(
      config, position, key, value, key_norm_scale, candidate_row.key,
      candidate_row.value, normalized_key);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  error = launch_prepare_decode_query(config, position, query, query_norm_scale,
                                      normalized_query, stream);
  if (error != cudaSuccess) return error;
  int occupancy = 0;
  std::size_t local_bytes = 0;
  int registers = 0;
  kernel_attributes(&registers, &local_bytes, &occupancy);
  const int nsm = nsm_count();
  const int n_kv = static_cast<int>(position + 1);
  const int ntiles_kv = (n_kv + kNbatchFa - 1) / kNbatchFa;
  int n_parts = n_parts_override > 0
                    ? n_parts_override
                    : parallel_blocks_for_kv(n_kv, occupancy, nsm);
  if (n_parts < 1) n_parts = 1;
  if (n_parts > kMaxParts) n_parts = kMaxParts;
  if (n_parts > 1 && (partial_vkq == nullptr || meta == nullptr)) {
    return cudaErrorInvalidValue;
  }
  float* dst = n_parts > 1 ? partial_vkq : output;
  float2* dst_meta = n_parts > 1 ? reinterpret_cast<float2*>(meta) : nullptr;
  dim3 grid(1, static_cast<unsigned>(n_parts), config.query_heads);
  dim3 block(kWarpSize, kWarps, 1);
  flash_attn_ext_vec_nvidia_bf16<<<grid, block, 0, stream>>>(
      position, config.capacity, normalized_query, committed.key,
      committed.value, candidate_row.key, candidate_row.value, dst, dst_meta);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (n_parts > 1) {
    const std::size_t shared = static_cast<std::size_t>(n_parts) * sizeof(float2);
    combine_and_gate<<<config.query_heads, kD, shared, stream>>>(
        n_parts, partial_vkq, dst_meta, output_gate, output);
  } else {
    apply_gate<<<config.query_heads, kD, 0, stream>>>(output, output_gate,
                                                      output);
  }
  g_last_launch.n_parts = n_parts;
  g_last_launch.occupancy_per_sm = occupancy;
  g_last_launch.nsm = nsm;
  g_last_launch.ntiles_kv = ntiles_kv;
  g_last_launch.n_kv = n_kv;
  g_last_launch.registers = registers;
  g_last_launch.local_bytes = local_bytes;
  g_last_launch.shared_bytes = kNeCombine * sizeof(float);
  g_last_launch.staging_bytes =
      2 * static_cast<std::size_t>(kKvHeads) * kD * sizeof(__nv_bfloat16);
  g_last_launch.prepared_q_once = true;
  g_last_launch.bf16_kv = true;
  g_last_launch.f16_cache_migration = false;
  g_last_launch.nvidia_float2_v_accum = true;
  g_last_launch.amd_half2_v_accum = false;
  return cudaPeekAtLastError();
}

}  // namespace opt108
}  // namespace qw38::cuda
