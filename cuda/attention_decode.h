#ifndef QW38_CUDA_ATTENTION_DECODE_H_
#define QW38_CUDA_ATTENTION_DECODE_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

struct AttentionConfig {
  std::uint32_t query_heads;
  std::uint32_t kv_heads;
  std::uint32_t head_width;
  std::uint32_t rotary_width;
  std::uint32_t capacity;
};

struct AttentionCache {
  __nv_bfloat16* key;
  __nv_bfloat16* value;
};

std::size_t attention_query_values(const AttentionConfig& config) noexcept;
std::size_t attention_kv_row_values(const AttentionConfig& config) noexcept;
std::size_t attention_cache_values(const AttentionConfig& config) noexcept;
std::size_t attention_score_values(const AttentionConfig& config,
                                   std::size_t position) noexcept;

cudaError_t launch_attention_prepare(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_commit(
    const AttentionConfig& config, std::size_t position,
    const AttentionCache& candidate_row, const AttentionCache& committed,
    std::uint64_t new_frontier, std::uint64_t* committed_frontier,
    cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_ATTENTION_DECODE_H_
