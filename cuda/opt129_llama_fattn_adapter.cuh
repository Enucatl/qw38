#pragma once

// OPT-129 diagnostic F16 token-major flash_attn_ext_vec<256,1> consumer of
// pinned llama.cpp (MIT, The ggml authors, revision
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328). Does not vendor ggml
// tensor/runtime. Production hybrid_crossover@1024 is unchanged.

#include "attention_decode.h"
#include "opt108_llama_vector_adapter.cuh"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt129 {

constexpr char kLlamaRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr char kNativeQuartz[] = "quartz_hybrid_crossover";
constexpr char kMatchedLlamaVec[] = "opt108_llama_vec_nvidia_bf16";
constexpr char kNativeLlamaVec[] = "opt129_flash_attn_ext_vec_f16";
constexpr char kAdapterConvert[] = "opt129_bf16_physical_to_f16_token_major";
constexpr int kD = 256;
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarps = 4;
constexpr int kCpyBytes = 16;
constexpr int kCpyNe = 4;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kFattnStride = 256;
constexpr int kNthreadsKq = 8;
constexpr int kNthreadsV = 8;
constexpr int kVRowsPerThread = 8;
constexpr int kVColsPerIter = 4;
constexpr int kQRegs = 16;
constexpr int kNeCombine = kWarps * kVColsPerIter * kD;
constexpr float kScale = 0.0625F;
constexpr float kKqMaxOffset = 3.0F * 0.6931F;
constexpr int kGqaRatio = kQueryHeads / kKvHeads;
constexpr int kMmaSwitchNkv = 8192;

inline int llama_padded_nkv(int n_kv) noexcept {
  const int pad = kFattnStride;
  if (n_kv < pad) return pad;
  return ((n_kv + pad - 1) / pad) * pad;
}

inline const char* llama_selected_kernel(int padded_nkv) noexcept {
  if (padded_nkv % kFattnStride != 0) return "fattn-mma-f16";
  if (kGqaRatio > 4 && padded_nkv >= kMmaSwitchNkv) {
    return "fattn-mma-f16_ncols1=1_ncols2=8";
  }
  return "flash_attn_ext_vec<256,1>";
}

inline bool llama_vec_is_selected(int padded_nkv) noexcept {
  return std::strcmp(llama_selected_kernel(padded_nkv),
                     "flash_attn_ext_vec<256,1>") == 0;
}

inline const char* quartz_shipping_kernel(std::size_t position) noexcept {
  if (decode_attention_vec128_uses_online_at(position)) {
    return "vec128_online_decode_attention";
  }
  return "warp_query_decode_attention";
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

__device__ __forceinline__ float mad_float2(float acc, float2 left,
                                            float2 right) {
  acc += left.x * right.x;
  acc += left.y * right.y;
  return acc;
}

__device__ __forceinline__ const __half* llama_kv_row(std::size_t token,
                                                      std::uint32_t kv_head,
                                                      int padded_nkv,
                                                      const __half* packed) {
  (void)padded_nkv;
  const std::size_t row =
      static_cast<std::size_t>(kKvHeads) * static_cast<std::size_t>(kD);
  return packed + token * row +
         static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(kD);
}

__global__ void convert_physical_bf16_to_llama_f16(
    std::uint32_t kv_heads, std::uint32_t capacity, std::uint32_t width,
    std::size_t position, const __nv_bfloat16* committed,
    const __nv_bfloat16* candidate, __half* packed) {
  const std::size_t token = static_cast<std::size_t>(blockIdx.x);
  const std::uint32_t kv_head = blockIdx.y;
  const std::uint32_t dim = threadIdx.x;
  const std::size_t n_visible = position + 1;
  if (token >= n_visible || kv_head >= kv_heads || dim >= width) return;
  __nv_bfloat16 src_val;
  if (token < position) {
    src_val = committed[attention_kv_physical_index(token, kv_head, dim,
                                                    capacity, width)];
  } else {
    src_val = candidate[static_cast<std::size_t>(kv_head) * width + dim];
  }
  const std::size_t dst =
      token * static_cast<std::size_t>(kv_heads) * width +
      static_cast<std::size_t>(kv_head) * width + dim;
  packed[dst] = __float2half(__bfloat162float(src_val));
}

// NVIDIA float2 F16 specialization of flash_attn_ext_vec<256,1> on llama
// token-major packed KV (n_embd_k_gqa x n_kv underlying storage).
__global__ void __launch_bounds__(kThreads, 1)
flash_attn_ext_vec_f16(
    int n_kv, int padded_nkv, const float* prepared_q, const __half* packed_key,
    const __half* packed_value, float* dst, float2* dst_meta) {
  constexpr int nthreads_kq = kNthreadsKq;
  constexpr int nthreads_v = kNthreadsV;
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
        const half2* k_row = reinterpret_cast<const half2*>(llama_kv_row(
            static_cast<std::size_t>(token), kv_head, padded_nkv, packed_key));
#pragma unroll
        for (int k0 = 0; k0 < kD / 2; k0 += nthreads_kq * kCpyNe) {
          __align__(16) half2 tmp[kCpyNe];
          cpy16(tmp, k_row + k0 +
                         (static_cast<int>(threadIdx.x) % nthreads_kq) * kCpyNe);
#pragma unroll
          for (int k1 = 0; k1 < kCpyNe; ++k1) {
            sum = mad_float2(sum, __half22float2(tmp[k1]),
                             q_reg[k0 / nthreads_kq + k1]);
          }
        }
      }
      sum = warp_sum<nthreads_kq>(sum);
      if (token < n_kv) kq_max_new = fmaxf(kq_max_new, sum + kKqMaxOffset);
      if ((static_cast<int>(threadIdx.x) % nthreads_kq) == i_kq0) {
        kq_reg = token < n_kv ? sum : 0.0F;
      }
    }
#pragma unroll
    for (int offset = nthreads_kq; offset < kWarpSize; offset <<= 1) {
      kq_max_new = fmaxf(kq_max_new, __shfl_xor_sync(0xffffffff, kq_max_new,
                                                     offset, kWarpSize));
    }
    const float kq_max_scale = expf(kq_max - kq_max_new);
    kq_max = kq_max_new;
    const int my_token = k_vkq0 + tid;
    kq_reg = (my_token < n_kv) ? expf(kq_reg - kq_max) : 0.0F;
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
      const int k = static_cast<int>(threadIdx.y) * kWarpSize + k0 +
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
          const __half* v_row = llama_kv_row(static_cast<std::size_t>(token),
                                             kv_head, padded_nkv, packed_value);
          const int i_v =
              2 * i0 +
              (static_cast<int>(threadIdx.x) % nthreads_v) * kVRowsPerThread;
          __align__(16) half2 packed[kVRowsPerThread / 2];
          cpy16(packed, v_row + i_v);
#pragma unroll
          for (int t = 0; t < kVRowsPerThread / 2; ++t) {
            tmp[t] = __half22float2(packed[t]);
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
  float2* vkq_tmp = reinterpret_cast<float2*>(kq_shared) +
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

inline int f16_occupancy() noexcept {
  int blocks = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, flash_attn_ext_vec_f16, kThreads, 0);
  if (error != cudaSuccess) return 0;
  return blocks;
}

inline void f16_kernel_attributes(int* registers, std::size_t* local_bytes,
                                  int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, flash_attn_ext_vec_f16);
  if (registers != nullptr) *registers = attrs.numRegs;
  if (local_bytes != nullptr) {
    *local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  }
  if (occupancy != nullptr) *occupancy = f16_occupancy();
}

inline cudaError_t launch_convert_bf16_to_f16(
    std::uint32_t kv_heads, std::uint32_t capacity, std::uint32_t width,
    std::size_t position, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, __half* packed_key,
    __half* packed_value, cudaStream_t stream) noexcept {
  const std::size_t n_visible = position + 1;
  if (n_visible == 0) return cudaSuccess;
  dim3 grid(static_cast<unsigned>(n_visible), kv_heads, 1);
  convert_physical_bf16_to_llama_f16<<<grid, width, 0, stream>>>(
      kv_heads, capacity, width, position, committed_key, candidate_key,
      packed_key);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  convert_physical_bf16_to_llama_f16<<<grid, width, 0, stream>>>(
      kv_heads, capacity, width, position, committed_value, candidate_value,
      packed_value);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_llama_f16_vec(
    int n_kv, const float* prepared_q, const __half* packed_key,
    const __half* packed_value, float* output, float* partial_vkq, float* meta,
    const float* output_gate, int n_parts_override,
    cudaStream_t stream) noexcept {
  if (prepared_q == nullptr || packed_key == nullptr ||
      packed_value == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  int occupancy = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  f16_kernel_attributes(&registers, &local_bytes, &occupancy);
  const int nsm = opt108::nsm_count();
  const int padded = llama_padded_nkv(n_kv);
  int n_parts = n_parts_override > 0
                    ? n_parts_override
                    : opt108::parallel_blocks_for_kv(n_kv, occupancy, nsm);
  if (n_parts < 1) n_parts = 1;
  if (n_parts > opt108::kMaxParts) n_parts = opt108::kMaxParts;
  if (n_parts > 1 && (partial_vkq == nullptr || meta == nullptr)) {
    return cudaErrorInvalidValue;
  }
  float* dst = n_parts > 1 ? partial_vkq : output;
  float2* dst_meta = n_parts > 1 ? reinterpret_cast<float2*>(meta) : nullptr;
  dim3 grid(1, static_cast<unsigned>(n_parts), kQueryHeads);
  dim3 block(kWarpSize, kWarps, 1);
  flash_attn_ext_vec_f16<<<grid, block, 0, stream>>>(
      n_kv, padded, prepared_q, packed_key, packed_value, dst, dst_meta);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (output_gate == nullptr) return error;
  if (n_parts > 1) {
    const std::size_t shared = static_cast<std::size_t>(n_parts) * sizeof(float2);
    opt108::combine_and_gate<<<kQueryHeads, kD, shared, stream>>>(
        n_parts, partial_vkq, dst_meta, output_gate, output);
  } else {
    opt108::apply_gate<<<kQueryHeads, kD, 0, stream>>>(output, output_gate,
                                                       output);
  }
  return cudaPeekAtLastError();
}

}  // namespace opt129
}  // namespace qw38::cuda
