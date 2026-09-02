#ifndef QW38_CUDA_GDN_STEP_H_
#define QW38_CUDA_GDN_STEP_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace qw38::cuda {

struct GdnConfig {
  std::uint32_t key_heads;
  std::uint32_t value_heads;
  std::uint32_t key_width;
  std::uint32_t value_width;
  std::uint32_t convolution_width;
};

struct GdnState {
  float* convolution;
  float* recurrent;
};

std::size_t gdn_convolution_channels(const GdnConfig& config) noexcept;
std::size_t gdn_convolution_values(const GdnConfig& config) noexcept;
std::size_t gdn_recurrent_values(const GdnConfig& config) noexcept;
std::size_t gdn_output_values(const GdnConfig& config) noexcept;

cudaError_t launch_gdn_prepare(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, const GdnState& committed, const GdnState& candidate,
    float* convolution_output, float* recurrent_output,
    cudaStream_t stream) noexcept;

cudaError_t launch_gdn_prepare_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, const GdnState& committed, const GdnState& candidate,
    float* convolution_output, float* recurrent_output,
    cudaStream_t stream) noexcept;

cudaError_t launch_gdn_prepare_chunk(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream) noexcept;

cudaError_t launch_gdn_prepare_chunk_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream) noexcept;

cudaError_t launch_gdn_commit(const GdnConfig& config,
                              const GdnState& candidate,
                              const GdnState& committed,
                              std::uint64_t new_frontier,
                              std::uint64_t* committed_frontier,
                              cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_GDN_STEP_H_
