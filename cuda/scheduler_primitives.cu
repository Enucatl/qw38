#include "scheduler_primitives.h"

#include <cmath>

namespace qw38::cuda {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kMaximumNormWidth = 17408;

__global__ void rms_norm(const __nv_bfloat16* input, const float* scale,
                         std::size_t count, __nv_bfloat16* output) {
  __shared__ float inverse;
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
      const float value = __bfloat162float(input[index]);
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    inverse = 1.0F / sqrtf(sum / static_cast<float>(count) + 1.0e-6F);
  }
  __syncthreads();
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    output[index] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(__bfloat162float(input[index]), inverse),
                   scale[index]));
  }
}

__global__ void fp32_to_bf16(const float* input, std::size_t count,
                             __nv_bfloat16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = __float2bfloat16_rn(input[index]);
}

__global__ void residual_add(const __nv_bfloat16* residual,
                             const float* correction, std::size_t count,
                             __nv_bfloat16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float2bfloat16_rn(
        __fadd_rn(__bfloat162float(residual[index]), correction[index]));
  }
}

__global__ void swiglu(const float* gate, const float* up, std::size_t count,
                       __nv_bfloat16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float activated = gate[index] / (1.0F + expf(-gate[index]));
    output[index] =
        __float2bfloat16_rn(__fmul_rn(activated, up[index]));
  }
}

__global__ void split_attention(const float* packed, std::size_t head_width,
                                std::size_t count, float* query, float* gate) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::size_t head = index / head_width;
  const std::size_t lane = index % head_width;
  const std::size_t packed_base = head * head_width * 2;
  query[index] = packed[packed_base + lane];
  gate[index] = packed[packed_base + head_width + lane];
}

__global__ void prepare_gates(const float* alpha, const float* beta,
                              const float* folded_a, const float* dt_bias,
                              std::size_t key_heads, std::size_t replicas,
                              float* log_decay, float* update) {
  const std::size_t grouped = threadIdx.x;
  const std::size_t count = key_heads * replicas;
  if (grouped >= count) return;
  const std::size_t key = grouped / replicas;
  const std::size_t replica = grouped % replicas;
  const std::size_t tiled = replica * key_heads + key;
  const float gate = alpha[tiled] + dt_bias[tiled];
  const float softplus =
      gate > 20.0F ? gate : gate < -20.0F ? expf(gate) : log1pf(expf(gate));
  log_decay[grouped] = __fmul_rn(folded_a[tiled], softplus);
  const float beta_value = beta[tiled];
  update[grouped] = beta_value >= 0.0F
                        ? 1.0F / (1.0F + expf(-beta_value))
                        : expf(beta_value) / (1.0F + expf(beta_value));
}

__global__ void gated_output(const float* recurrent, const float* gate_tiled,
                             const float* norm, std::size_t key_heads,
                             std::size_t replicas, std::size_t head_width,
                             __nv_bfloat16* output_tiled) {
  const std::size_t grouped_head = blockIdx.x;
  const std::size_t lane = threadIdx.x;
  __shared__ float inverse;
  const std::size_t grouped_base = grouped_head * head_width;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < head_width; ++index) {
      const float value = recurrent[grouped_base + index];
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    inverse = 1.0F /
              sqrtf(sum / static_cast<float>(head_width) + 1.0e-6F);
  }
  __syncthreads();
  if (lane < head_width) {
    const std::size_t key = grouped_head / replicas;
    const std::size_t replica = grouped_head % replicas;
    const std::size_t tiled_head = replica * key_heads + key;
    const float gate = gate_tiled[tiled_head * head_width + lane];
    const float silu = gate / (1.0F + expf(-gate));
    const float value = __fmul_rn(
        __fmul_rn(recurrent[grouped_base + lane], inverse), norm[lane]);
    output_tiled[tiled_head * head_width + lane] =
        __float2bfloat16_rn(__fmul_rn(value, silu));
  }
}

unsigned int blocks_for(std::size_t count) {
  return static_cast<unsigned int>((count + kThreads - 1) / kThreads);
}

}  // namespace

cudaError_t launch_rms_norm_bf16(const __nv_bfloat16* input,
                                 const float* scale, std::size_t count,
                                 __nv_bfloat16* output,
                                 cudaStream_t stream) noexcept {
  if (input == nullptr || scale == nullptr || output == nullptr || count == 0 ||
      count > kMaximumNormWidth) return cudaErrorInvalidValue;
  rms_norm<<<1, kThreads, 0, stream>>>(input, scale, count, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_fp32_to_bf16(const float* input, std::size_t count,
                                __nv_bfloat16* output,
                                cudaStream_t stream) noexcept {
  if (input == nullptr || output == nullptr || count == 0)
    return cudaErrorInvalidValue;
  fp32_to_bf16<<<blocks_for(count), kThreads, 0, stream>>>(input, count, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_residual_add_bf16(const __nv_bfloat16* residual,
                                     const float* correction,
                                     std::size_t count, __nv_bfloat16* output,
                                     cudaStream_t stream) noexcept {
  if (residual == nullptr || correction == nullptr || output == nullptr ||
      count == 0) return cudaErrorInvalidValue;
  residual_add<<<blocks_for(count), kThreads, 0, stream>>>(
      residual, correction, count, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_swiglu_bf16(const float* gate, const float* up,
                               std::size_t count, __nv_bfloat16* output,
                               cudaStream_t stream) noexcept {
  if (gate == nullptr || up == nullptr || output == nullptr || count == 0)
    return cudaErrorInvalidValue;
  swiglu<<<blocks_for(count), kThreads, 0, stream>>>(gate, up, count, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_split_attention_query_gate(
    const float* packed, std::size_t query_heads, std::size_t head_width,
    float* query, float* gate, cudaStream_t stream) noexcept {
  if (packed == nullptr || query == nullptr || gate == nullptr ||
      query_heads == 0 || head_width == 0) return cudaErrorInvalidValue;
  const std::size_t count = query_heads * head_width;
  split_attention<<<blocks_for(count), kThreads, 0, stream>>>(
      packed, head_width, count, query, gate);
  return cudaPeekAtLastError();
}

cudaError_t launch_prepare_gdn_gates(
    const float* alpha_tiled, const float* beta_tiled,
    const float* folded_a_tiled, const float* dt_bias_tiled,
    std::size_t key_heads, std::size_t replicas, float* log_decay_grouped,
    float* beta_grouped, cudaStream_t stream) noexcept {
  const std::size_t count = key_heads * replicas;
  if (alpha_tiled == nullptr || beta_tiled == nullptr ||
      folded_a_tiled == nullptr || dt_bias_tiled == nullptr ||
      log_decay_grouped == nullptr || beta_grouped == nullptr || count == 0 ||
      count > kThreads) return cudaErrorInvalidValue;
  prepare_gates<<<1, kThreads, 0, stream>>>(
      alpha_tiled, beta_tiled, folded_a_tiled, dt_bias_tiled, key_heads,
      replicas, log_decay_grouped, beta_grouped);
  return cudaPeekAtLastError();
}

cudaError_t launch_gdn_gated_output(
    const float* recurrent_grouped, const float* gate_tiled,
    const float* norm_scale, std::size_t key_heads, std::size_t replicas,
    std::size_t head_width, __nv_bfloat16* output_tiled,
    cudaStream_t stream) noexcept {
  const std::size_t heads = key_heads * replicas;
  if (recurrent_grouped == nullptr || gate_tiled == nullptr ||
      norm_scale == nullptr || output_tiled == nullptr || heads == 0 ||
      head_width == 0 || head_width > kThreads)
    return cudaErrorInvalidValue;
  gated_output<<<static_cast<unsigned int>(heads), kThreads, 0, stream>>>(
      recurrent_grouped, gate_tiled, norm_scale, key_heads, replicas,
      head_width, output_tiled);
  return cudaPeekAtLastError();
}

}  // namespace qw38::cuda
