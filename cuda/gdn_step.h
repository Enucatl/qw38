#ifndef QW38_CUDA_GDN_STEP_H_
#define QW38_CUDA_GDN_STEP_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gdn_decode_path.cuh"

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

enum class GdnScanPath : std::uint8_t {
  kSequentialWindows = 0,
  kParallelAssociative = 1,
  kFusedTokenLoop = 2,
};

std::size_t gdn_convolution_channels(const GdnConfig& config) noexcept;
std::size_t gdn_convolution_values(const GdnConfig& config) noexcept;
std::size_t gdn_recurrent_values(const GdnConfig& config) noexcept;
std::size_t gdn_output_values(const GdnConfig& config) noexcept;
std::size_t gdn_scan_window_count(std::size_t token_count) noexcept;
std::size_t gdn_scan_operator_values(const GdnConfig& config) noexcept;
std::size_t gdn_scan_scratch_floats(const GdnConfig& config,
                                    std::size_t token_count) noexcept;

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
    float* recurrent_output, cudaStream_t stream,
    GdnScanPath path = GdnScanPath::kSequentialWindows,
    float* scratch = nullptr, std::size_t scratch_floats = 0) noexcept;

cudaError_t launch_gdn_prepare_chunk_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream,
    GdnScanPath path = GdnScanPath::kSequentialWindows,
    float* scratch = nullptr, std::size_t scratch_floats = 0) noexcept;

cudaError_t launch_gdn_fused_rank2(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream,
    bool value_is_tiled) noexcept;

int gdn_fused_quality_occupancy() noexcept;
const char* selected_gdn_fuse_path() noexcept;
bool gdn_fuses_conv() noexcept;
bool gdn_fuses_gated_output() noexcept;
int gdn_fuse_occupancy(const char* path) noexcept;
const char* selected_gdn_inverse_path() noexcept;
bool gdn_uses_shared_inverse() noexcept;
std::size_t gdn_shared_inverse_floats(std::size_t token_count,
                                     std::uint32_t key_heads) noexcept;
int gdn_shared_inverse_occupancy() noexcept;
int gdn_shared_recurrence_occupancy() noexcept;
const char* selected_gdn_preproc_path() noexcept;
bool gdn_uses_preproc() noexcept;
std::size_t gdn_preproc_floats(const GdnConfig& config,
                               std::size_t token_count) noexcept;
std::size_t gdn_preproc_transpose_floats(const GdnConfig& config,
                                         std::size_t token_count) noexcept;
int gdn_preproc_occupancy() noexcept;
int gdn_preproc_decay_occupancy() noexcept;
int gdn_preproc_recurrence_occupancy() noexcept;
int gdn_preproc_fma_occupancy() noexcept;
int gdn_preproc_transpose_occupancy() noexcept;
int gdn_decode_tiled_occupancy(unsigned int value_tile) noexcept;
void gdn_decode_tiled_attributes(unsigned int value_tile, int* registers,
                                 std::size_t* local_bytes,
                                 int* occupancy) noexcept;
int gdn_decode_transposed_occupancy() noexcept;
void gdn_decode_transposed_attributes(int* registers, std::size_t* local_bytes,
                                      int* occupancy) noexcept;

cudaError_t launch_gdn_recurrence_only(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, bool value_is_tiled,
    cudaStream_t stream) noexcept;

cudaError_t launch_gdn_convert_recurrent_layout(
    const GdnConfig& config, float* recurrent, bool to_col_major,
    cudaStream_t stream, GdnConversionBoundary boundary) noexcept;

cudaError_t launch_gdn_copy_converted_recurrent(
    const GdnConfig& config, const float* source, float* dest, bool to_col_major,
    cudaStream_t stream, GdnConversionBoundary boundary) noexcept;

cudaError_t launch_gdn_convert_session_recurrent(
    float* recurrent, std::size_t layers, bool to_col_major,
    cudaStream_t stream, GdnConversionBoundary boundary) noexcept;

cudaError_t launch_gdn_shared_inverses(
    const GdnConfig& config, const float* convolution_output,
    std::size_t token_count, float* inverses, cudaStream_t stream) noexcept;

cudaError_t launch_gdn_quality_fused(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, const float* gate_tiled, const float* norm,
    __nv_bfloat16* output_bf16, cudaStream_t stream, bool value_is_tiled,
    const char* path = nullptr, const char* inverse_path = nullptr,
    float* inverse_scratch = nullptr,
    std::size_t inverse_scratch_floats = 0,
    const char* preproc_path = nullptr) noexcept;

cudaError_t launch_gdn_gated_output_rows(
    const float* recurrent, const float* gate_tiled, const float* norm,
    std::size_t key_heads, std::size_t replicas, std::size_t head_width,
    std::size_t token_count, __nv_bfloat16* output_tiled,
    cudaStream_t stream) noexcept;

cudaError_t launch_gdn_commit(const GdnConfig& config,
                              const GdnState& candidate,
                              const GdnState& committed,
                              std::uint64_t new_frontier,
                              std::uint64_t* committed_frontier,
                              cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_GDN_STEP_H_
