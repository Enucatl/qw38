// OPT-110 candidate TU. Compiled with pinned-llama CUDA flags
// (-O3 --use_fast_math --extended-lambda). Source-faithful Q4_K MMVQ
// and native block_q8_1 staging from identical BF16 activations.
//
// Technique from llama.cpp (MIT, The ggml authors, cc83d7b):
//   ggml/src/ggml-cuda/mmvq.cu::mul_mat_vec_q<Q4_K,1>
//   ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_q4_K_q8_1
//   ggml/src/ggml-cuda/quantize.cu::quantize_q8_1
//   ggml/src/ggml-cuda/unary.cuh::ggml_cuda_op_silu_single

#include "opt110_llama_q4_adapter.cuh"
#include "opt110_engine_hook.cuh"
#include "q4k_decode_path.cuh"

namespace qw38::cuda {
namespace opt110 {
namespace {

__device__ __forceinline__ int dp4a(int a, int b, int c) {
  return __dp4a(a, b, c);
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

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// vec_dot_q4_K_q8_1_impl_vmmq (vecdotq.cuh, cc83d7b)
__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int* v, const int* u, const std::uint8_t* sc, const std::uint8_t* m,
    const __half2& dm4, const float* d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < kQr4K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 =
        dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = __half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// vec_dot_q4_K_q8_1 (vecdotq.cuh, cc83d7b). iqs in {0,2,...,30}.
__device__ __forceinline__ float vec_dot_q4_K_q8_1(const void* vbq,
                                                   const BlockQ81* bq8_1,
                                                   const int kbx,
                                                   const int iqs) {
  const BlockQ4K* bq4_K = static_cast<const BlockQ4K*>(vbq) + kbx;
  int v[2];
  int u[2 * kQr4K];
  float d8[kQr4K];
  const int bq8_offset = kQr4K * ((iqs / 2) / (kQ8 / 8));
  const int* q4 = reinterpret_cast<const int*>(
      bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];
  const std::uint16_t* scales =
      reinterpret_cast<const std::uint16_t*>(bq4_K->scales);
  std::uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const std::uint8_t* sc = reinterpret_cast<const std::uint8_t*>(aux);
  const std::uint8_t* m = sc + 2;
#pragma unroll
  for (int i = 0; i < kQr4K; ++i) {
    const BlockQ81* bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

// quantize_q8_1 from BF16, llama GPU semantics: d=amax/127, s=sum(x) as half2.
__global__ void quantize_bf16_block_q8_1(const __nv_bfloat16* x, BlockQ81* y,
                                         std::size_t columns) {
  const std::size_t i0 =
      static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (i0 >= columns) return;
  const float xi = __bfloat162float(x[i0]);
  float amax = fabsf(xi);
  float sum = xi;
  amax = warp_max<kQ8>(amax);
  sum = warp_sum<kQ8>(sum);
  const float d = amax / 127.0f;
  const std::int8_t q =
      amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
  const std::size_t ib = i0 / static_cast<std::size_t>(kQ8);
  const int iqs = static_cast<int>(i0 % static_cast<std::size_t>(kQ8));
  y[ib].qs[iqs] = q;
  if (iqs > 0) return;
  y[ib].ds = make_half2(d, sum);
}

// mul_mat_vec_q<Q4_K, ncols_dst=1, has_fusion> GENERIC nwarps=4, rpb=1.
template <bool HasFusion>
__launch_bounds__(kNwarps* kWarp, 1) __global__
    void mul_mat_vec_q4_k(const BlockQ4K* vx, const BlockQ81* vy,
                          const BlockQ4K* vgate, float* dst_f,
                          __nv_bfloat16* dst_bf16, std::uint32_t ncols_x) {
  constexpr int nwarps = kNwarps;
  constexpr int warp_size = kWarp;
  constexpr int vdr = kVdr;
  constexpr int qi = kQi;
  constexpr int qk = kQk;
  constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
  const int tid = warp_size * static_cast<int>(threadIdx.y) +
                  static_cast<int>(threadIdx.x);
  const int row0 = static_cast<int>(blockIdx.x);
  const int blocks_per_row_x = static_cast<int>(ncols_x / qk);

  float tmp = 0.0f;
  float tmp_gate = 0.0f;
  const int kbx_offset = row0 * blocks_per_row_x;
  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / kQ8);
    const int kqs = vdr * (tid % (qi / vdr));
    tmp += vec_dot_q4_K_q8_1(vx, vy + kby, kbx_offset + kbx, kqs);
    if constexpr (HasFusion) {
      tmp_gate += vec_dot_q4_K_q8_1(vgate, vy + kby, kbx_offset + kbx, kqs);
    }
  }

  __shared__ float tmp_shared[nwarps - 1][warp_size];
  __shared__ float tmp_shared_gate[HasFusion ? (nwarps - 1) : 1][warp_size];
  if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    if constexpr (HasFusion) {
      tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) return;

#pragma unroll
  for (int l = 0; l < nwarps - 1; ++l) {
    tmp += tmp_shared[l][threadIdx.x];
    if constexpr (HasFusion) {
      tmp_gate += tmp_shared_gate[l][threadIdx.x];
    }
  }
  tmp = warp_sum<warp_size>(tmp);
  if constexpr (HasFusion) {
    tmp_gate = warp_sum<warp_size>(tmp_gate);
  }
  if (threadIdx.x == 0) {
    if constexpr (HasFusion) {
      const float result = tmp * silu(tmp_gate);
      dst_bf16[row0] = __float2bfloat16_rn(result);
    } else {
      dst_f[row0] = tmp;
    }
  }
}

void fill_launch(LaunchInfo* info, std::size_t rows, std::size_t columns,
                 bool fused) {
  *info = LaunchInfo{};
  info->grid_x = static_cast<int>(rows);
  info->fused_glu = fused;
  info->staging_bytes = q8_1_bytes(columns);
  info->shared_bytes =
      fused ? sizeof(float) * (kNwarps - 1) * kWarp * 2
            : sizeof(float) * (kNwarps - 1) * kWarp;
  info->launch_name = fused ? kLaunchMmvqGlu : kLaunchMmvq;
}

}  // namespace

thread_local LaunchInfo g_last_launch{};

const LaunchInfo& last_launch() noexcept { return g_last_launch; }

cudaError_t launch_quantize_bf16_block_q8_1(const __nv_bfloat16* activation,
                                            BlockQ81* q8, std::size_t columns,
                                            cudaStream_t stream) noexcept {
  if (activation == nullptr || q8 == nullptr || columns % kQ8 != 0) {
    return cudaErrorInvalidValue;
  }
  const unsigned int grid = static_cast<unsigned int>(
      (columns + static_cast<std::size_t>(kQuantizeBlock) - 1U) /
      static_cast<std::size_t>(kQuantizeBlock));
  quantize_bf16_block_q8_1<<<grid, kQuantizeBlock, 0, stream>>>(
      activation, q8, columns);
  return cudaPeekAtLastError();
}

cudaError_t launch_mmvq_q4k_q8_1(const BlockQ4K* weights, std::size_t rows,
                                 std::size_t columns, const BlockQ81* q8,
                                 float* output, cudaStream_t stream) noexcept {
  if (weights == nullptr || q8 == nullptr || output == nullptr || rows == 0 ||
      columns % kQk != 0) {
    return cudaErrorInvalidValue;
  }
  fill_launch(&g_last_launch, rows, columns, false);
  int occupancy = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, mul_mat_vec_q4_k<false>, kWarp * kNwarps, 0);
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, mul_mat_vec_q4_k<false>);
  g_last_launch.occupancy = occupancy;
  g_last_launch.registers = attrs.numRegs;
  g_last_launch.local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  dim3 block(kWarp, kNwarps);
  mul_mat_vec_q4_k<false><<<static_cast<unsigned int>(rows), block, 0, stream>>>(
      weights, q8, nullptr, output, nullptr,
      static_cast<std::uint32_t>(columns));
  return cudaPeekAtLastError();
}

cudaError_t launch_mmvq_q4k_q8_1_swiglu(const BlockQ4K* gate,
                                        const BlockQ4K* up, std::size_t rows,
                                        std::size_t columns, const BlockQ81* q8,
                                        __nv_bfloat16* output,
                                        cudaStream_t stream) noexcept {
  if (gate == nullptr || up == nullptr || q8 == nullptr || output == nullptr ||
      rows == 0 || columns % kQk != 0) {
    return cudaErrorInvalidValue;
  }
  fill_launch(&g_last_launch, rows, columns, true);
  int occupancy = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, mul_mat_vec_q4_k<true>, kWarp * kNwarps, 0);
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, mul_mat_vec_q4_k<true>);
  g_last_launch.occupancy = occupancy;
  g_last_launch.registers = attrs.numRegs;
  g_last_launch.local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  dim3 block(kWarp, kNwarps);
  // llama fusion.gate is the gate matrix; vx is up (result *= silu(gate)).
  mul_mat_vec_q4_k<true><<<static_cast<unsigned int>(rows), block, 0, stream>>>(
      up, q8, gate, nullptr, output, static_cast<std::uint32_t>(columns));
  return cudaPeekAtLastError();
}

void mmvq_kernel_attributes(int* registers, std::size_t* local_bytes,
                            int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  const cudaError_t error =
      cudaFuncGetAttributes(&attrs, mul_mat_vec_q4_k<false>);
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes =
        error == cudaSuccess ? static_cast<std::size_t>(attrs.localSizeBytes)
                             : 0;
  }
  if (occupancy != nullptr) {
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, mul_mat_vec_q4_k<false>, kWarp * kNwarps, 0) == cudaSuccess) {
      *occupancy = occ;
    } else {
      *occupancy = 0;
    }
  }
}

void mmvq_swiglu_kernel_attributes(int* registers, std::size_t* local_bytes,
                                   int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  const cudaError_t error =
      cudaFuncGetAttributes(&attrs, mul_mat_vec_q4_k<true>);
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes =
        error == cudaSuccess ? static_cast<std::size_t>(attrs.localSizeBytes)
                             : 0;
  }
  if (occupancy != nullptr) {
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, mul_mat_vec_q4_k<true>, kWarp * kNwarps, 0) == cudaSuccess) {
      *occupancy = occ;
    } else {
      *occupancy = 0;
    }
  }
}

void quantize_kernel_attributes(int* registers, std::size_t* local_bytes,
                                int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  const cudaError_t error =
      cudaFuncGetAttributes(&attrs, quantize_bf16_block_q8_1);
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes =
        error == cudaSuccess ? static_cast<std::size_t>(attrs.localSizeBytes)
                             : 0;
  }
  if (occupancy != nullptr) {
    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, quantize_bf16_block_q8_1, kQuantizeBlock, 0) ==
        cudaSuccess) {
      *occupancy = occ;
    } else {
      *occupancy = 0;
    }
  }
}

cudaError_t launch_engine_gate_up(const std::uint8_t* gate,
                                  const std::uint8_t* up, std::size_t rows,
                                  std::size_t columns, const void* activation,
                                  void* q81, void* output,
                                  cudaStream_t stream) {
  auto* staged = static_cast<BlockQ81*>(q81);
  auto* act = static_cast<const __nv_bfloat16*>(activation);
  auto* dst = static_cast<__nv_bfloat16*>(output);
  cudaError_t error =
      launch_quantize_bf16_block_q8_1(act, staged, columns, stream);
  if (error != cudaSuccess) return error;
  error = launch_mmvq_q4k_q8_1_swiglu(
      reinterpret_cast<const BlockQ4K*>(gate),
      reinterpret_cast<const BlockQ4K*>(up), rows, columns, staged, dst,
      stream);
  if (error == cudaSuccess) {
    record_q4_launch_variant(kQ4LaunchVariantLlamaMmvqGlu);
  }
  return error;
}

cudaError_t launch_engine_down(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns, const void* activation,
                               void* q81, float* output, cudaStream_t stream) {
  auto* staged = static_cast<BlockQ81*>(q81);
  auto* act = static_cast<const __nv_bfloat16*>(activation);
  cudaError_t error =
      launch_quantize_bf16_block_q8_1(act, staged, columns, stream);
  if (error != cudaSuccess) return error;
  error = launch_mmvq_q4k_q8_1(reinterpret_cast<const BlockQ4K*>(weights),
                               rows, columns, staged, output, stream);
  if (error == cudaSuccess) {
    record_q4_launch_variant(kQ4LaunchVariantLlamaMmvq);
  }
  return error;
}

namespace {
struct EngineHookInit {
  EngineHookInit() {
    g_engine_gate_up = &launch_engine_gate_up;
    g_engine_down = &launch_engine_down;
  }
};
EngineHookInit g_engine_hook_init;
}  // namespace

}  // namespace opt110
}  // namespace qw38::cuda
