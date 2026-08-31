#ifndef QW38_CUDA_SCHEDULER_PRIMITIVES_H_
#define QW38_CUDA_SCHEDULER_PRIMITIVES_H_

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

cudaError_t launch_rms_norm_bf16(const __nv_bfloat16* input,
                                 const float* scale, std::size_t count,
                                 __nv_bfloat16* output,
                                 cudaStream_t stream) noexcept;
cudaError_t launch_fp32_to_bf16(const float* input, std::size_t count,
                                __nv_bfloat16* output,
                                cudaStream_t stream) noexcept;
cudaError_t launch_residual_add_bf16(const __nv_bfloat16* residual,
                                     const float* correction,
                                     std::size_t count, __nv_bfloat16* output,
                                     cudaStream_t stream) noexcept;
cudaError_t launch_swiglu_bf16(const float* gate, const float* up,
                               std::size_t count, __nv_bfloat16* output,
                               cudaStream_t stream) noexcept;
cudaError_t launch_split_attention_query_gate(
    const float* packed, std::size_t query_heads, std::size_t head_width,
    float* query, float* gate, cudaStream_t stream) noexcept;
cudaError_t launch_prepare_gdn_gates(
    const float* alpha_tiled, const float* beta_tiled,
    const float* folded_a_tiled, const float* dt_bias_tiled,
    std::size_t key_heads, std::size_t replicas, float* log_decay_grouped,
    float* beta_grouped, cudaStream_t stream) noexcept;
cudaError_t launch_gdn_gated_output(
    const float* recurrent_grouped, const float* gate_tiled,
    const float* norm_scale, std::size_t key_heads, std::size_t replicas,
    std::size_t head_width, __nv_bfloat16* output_tiled,
    cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_SCHEDULER_PRIMITIVES_H_
