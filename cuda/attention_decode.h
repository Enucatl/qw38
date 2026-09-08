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

constexpr std::uint32_t kAttentionKvTileRows = 32;

struct AttentionKvTileSpan {
  std::uint64_t first_address;
  std::uint64_t last_address;
  std::uint32_t rows;
  std::uint32_t kv_head;
  std::uint32_t tile_start;
  std::uint32_t coalesced;
};

inline __host__ __device__ std::size_t attention_kv_logical_index(
    std::size_t token, std::uint32_t kv_head, std::uint32_t lane,
    std::uint32_t kv_heads, std::uint32_t head_width) noexcept {
  return token * static_cast<std::size_t>(kv_heads) * head_width +
         static_cast<std::size_t>(kv_head) * head_width + lane;
}

inline __host__ __device__ std::size_t attention_kv_physical_index(
    std::size_t token, std::uint32_t kv_head, std::uint32_t lane,
    std::uint32_t capacity, std::uint32_t head_width) noexcept {
  return (static_cast<std::size_t>(kv_head) * capacity + token) * head_width +
         lane;
}

inline __host__ __device__ std::size_t attention_kv_tile_base_offset(
    std::uint32_t kv_head, std::size_t tile_start, std::uint32_t capacity,
    std::uint32_t head_width) noexcept {
  return attention_kv_physical_index(tile_start, kv_head, 0, capacity,
                                     head_width);
}

inline void attention_kv_copy_logical_to_physical(
    const __nv_bfloat16* logical, __nv_bfloat16* physical,
    std::uint32_t kv_heads, std::uint32_t capacity,
    std::uint32_t head_width) noexcept {
  for (std::size_t token = 0; token < capacity; ++token) {
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      for (std::uint32_t lane = 0; lane < head_width; ++lane) {
        physical[attention_kv_physical_index(token, kv_head, lane, capacity,
                                             head_width)] =
            logical[attention_kv_logical_index(token, kv_head, lane, kv_heads,
                                               head_width)];
      }
    }
  }
}

inline void attention_kv_copy_physical_to_logical(
    const __nv_bfloat16* physical, __nv_bfloat16* logical,
    std::uint32_t kv_heads, std::uint32_t capacity,
    std::uint32_t head_width) noexcept {
  for (std::size_t token = 0; token < capacity; ++token) {
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      for (std::uint32_t lane = 0; lane < head_width; ++lane) {
        logical[attention_kv_logical_index(token, kv_head, lane, kv_heads,
                                           head_width)] =
            physical[attention_kv_physical_index(token, kv_head, lane, capacity,
                                                 head_width)];
      }
    }
  }
}

inline void attention_kv_gather_logical_rows(
    const __nv_bfloat16* physical, __nv_bfloat16* logical,
    std::size_t token_begin, std::size_t token_count, std::uint32_t kv_heads,
    std::uint32_t capacity, std::uint32_t head_width) noexcept {
  for (std::size_t rel = 0; rel < token_count; ++rel) {
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      for (std::uint32_t lane = 0; lane < head_width; ++lane) {
        logical[attention_kv_logical_index(rel, kv_head, lane, kv_heads,
                                           head_width)] =
            physical[attention_kv_physical_index(token_begin + rel, kv_head,
                                                 lane, capacity, head_width)];
      }
    }
  }
}

inline void attention_kv_scatter_logical_rows(
    const __nv_bfloat16* logical, __nv_bfloat16* physical,
    std::size_t token_begin, std::size_t token_count, std::uint32_t kv_heads,
    std::uint32_t capacity, std::uint32_t head_width) noexcept {
  for (std::size_t rel = 0; rel < token_count; ++rel) {
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      for (std::uint32_t lane = 0; lane < head_width; ++lane) {
        physical[attention_kv_physical_index(token_begin + rel, kv_head, lane,
                                             capacity, head_width)] =
            logical[attention_kv_logical_index(rel, kv_head, lane, kv_heads,
                                               head_width)];
      }
    }
  }
}

std::size_t attention_query_values(const AttentionConfig& config) noexcept;
std::size_t attention_kv_row_values(const AttentionConfig& config) noexcept;
std::size_t attention_cache_values(const AttentionConfig& config) noexcept;
std::size_t attention_score_values(const AttentionConfig& config,
                                   std::size_t position) noexcept;
std::size_t attention_chunk_score_values(const AttentionConfig& config,
                                         std::size_t start_position,
                                         std::size_t token_count) noexcept;

cudaError_t launch_attention_prepare(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_tiled(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_reference(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_mma_rank3(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_mma_ncols1(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, int ncols1, cudaStream_t stream) noexcept;

int selected_attention_mma_query_rows() noexcept;
int attention_mma_quality_occupancy() noexcept;
int attention_mma_quality_occupancy_for(int ncols1) noexcept;
std::size_t attention_mma_quality_shared_bytes() noexcept;

cudaError_t launch_attention_prepare_chunk_grouped_instrumented(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, std::uint64_t* kv_load_values,
    cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_per_query_tiled_reference(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, std::uint64_t* kv_load_values,
    cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_tile_span_instrumented(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, AttentionKvTileSpan* tile_spans,
    std::uint32_t* tile_span_count, std::uint32_t tile_span_capacity,
    cudaStream_t stream) noexcept;

cudaError_t launch_pack_committed_kv(
    const AttentionConfig& config, std::size_t token_begin,
    std::size_t token_count, const __nv_bfloat16* physical,
    __nv_bfloat16* logical, cudaStream_t stream) noexcept;

cudaError_t launch_unpack_committed_kv(
    const AttentionConfig& config, std::size_t token_begin,
    std::size_t token_count, const __nv_bfloat16* logical,
    __nv_bfloat16* physical, cudaStream_t stream) noexcept;

cudaError_t launch_attention_commit(
    const AttentionConfig& config, std::size_t position,
    const AttentionCache& candidate_row, const AttentionCache& committed,
    std::uint64_t new_frontier, std::uint64_t* committed_frontier,
    cudaStream_t stream) noexcept;

cudaError_t launch_attention_commit_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const AttentionCache& candidate_rows,
    const AttentionCache& committed, std::uint64_t new_frontier,
    std::uint64_t* committed_frontier, cudaStream_t stream) noexcept;

cudaError_t launch_attention_scatter_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const AttentionCache& candidate_rows,
    const AttentionCache& committed, cudaStream_t stream) noexcept;

cudaError_t launch_attention_scatter_layers(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, std::size_t layer_count,
    const __nv_bfloat16* candidate_key_base,
    const __nv_bfloat16* candidate_value_base,
    std::size_t candidate_layer_stride, __nv_bfloat16* committed_key_base,
    __nv_bfloat16* committed_value_base, std::size_t committed_layer_stride,
    cudaStream_t stream) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_ATTENTION_DECODE_H_
