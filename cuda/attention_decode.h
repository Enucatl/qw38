#ifndef QW38_CUDA_ATTENTION_DECODE_H_
#define QW38_CUDA_ATTENTION_DECODE_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

struct AttentionConfig {
  std::uint32_t query_heads;
  std::uint32_t kv_heads;
  std::uint32_t head_width;
  std::uint32_t rotary_width;
  std::uint32_t capacity;
};

// Candidate scratch is indexed from origin. Pass origin == start_position
// (the unsplit default) to keep HEAD committed/candidate classification.
// Do not store origin on AttentionConfig: extra by-value struct members
// shift host/device kernel-argument layout.
inline __host__ __device__ std::size_t attention_kv_origin(
    std::size_t start_position, std::size_t candidate_origin) noexcept {
  return candidate_origin == static_cast<std::size_t>(-1) ? start_position
                                                          : candidate_origin;
}

inline __host__ __device__ std::size_t split_candidate_origin(
    std::size_t outer_frontier, std::size_t query_start,
    bool split_candidate) noexcept {
  return split_candidate ? outer_frontier : query_start;
}

inline std::size_t resolve_attention_kv_origin(
    std::size_t start_position, std::size_t kv_origin) noexcept {
  return attention_kv_origin(start_position, kv_origin);
}

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

int selected_decode_kv_parts_below_2048() noexcept;
int selected_decode_kv_parts_at_or_above_2048() noexcept;
int decode_kv_parts_for_position(std::size_t position) noexcept;
std::size_t decode_kv_partial_vkq_values(int n_parts) noexcept;
std::size_t decode_kv_partial_meta_values(int n_parts) noexcept;
int decode_kv_partition_occupancy(int n_parts) noexcept;
int decode_kv_merge_occupancy(int n_parts) noexcept;
const char* selected_decode_attention_vec() noexcept;
bool decode_uses_warp_query() noexcept;
int decode_kv_warp_query_occupancy() noexcept;

cudaError_t launch_attention_prepare(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_partitioned(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, float* partial_vkq, float* meta, int n_parts,
    cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_partitioned_vec(
    const AttentionConfig& config, std::size_t position, const float* query,
    const float* key, const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, float* partial_vkq, float* meta, int n_parts,
    const char* vec_path, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, cudaStream_t stream,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept;

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

cudaError_t launch_attention_prepare_chunk_fattn_path(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, const char* path, float* partial_vkq, float* meta,
    cudaStream_t stream) noexcept;

int selected_attention_mma_query_rows() noexcept;
int attention_mma_quality_occupancy() noexcept;
int attention_mma_quality_occupancy_for(int ncols1) noexcept;
int attention_mma_quality_occupancy_for_path(const char* path) noexcept;
std::size_t attention_mma_quality_shared_bytes() noexcept;
const char* selected_fattn_path() noexcept;
bool fattn_uses_occupancy2() noexcept;
bool fattn_uses_stream_k() noexcept;
std::size_t fattn_stream_k_partial_values(const AttentionConfig& config,
                                          std::size_t token_count) noexcept;
std::size_t fattn_stream_k_meta_values(const AttentionConfig& config,
                                       std::size_t token_count) noexcept;
const char* selected_persistent_fattn_path() noexcept;
bool fattn_uses_persistent_stream_k() noexcept;
int fattn_persistent_nsm() noexcept;
int fattn_persistent_nblocks(int nsm, int occupancy, int ntiles_dst,
                             int ntiles_kv) noexcept;
std::size_t fattn_persistent_fixup_values(int nblocks) noexcept;

cudaError_t launch_fattn_mma_persistent(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* meta, cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_stream_k(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, float* partial_vkq, float* meta,
    cudaStream_t stream, const __half* prepared_q = nullptr,
    std::size_t kv_origin = static_cast<std::size_t>(-1)) noexcept;

const char* selected_vkq_accum() noexcept;
bool fattn_uses_register_vkq() noexcept;
int fattn_register_vkq_occupancy() noexcept;

cudaError_t launch_attention_prepare_chunk_stream_k_vkq(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, float* partial_vkq, float* meta, const char* vkq_accum,
    cudaStream_t stream) noexcept;

const char* selected_pv_path() noexcept;
bool fattn_uses_pv_mma() noexcept;
int fattn_pv_mma_occupancy() noexcept;

cudaError_t launch_attention_prepare_chunk_stream_k_pv(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, float* partial_vkq, float* meta, const char* pv_path,
    cudaStream_t stream) noexcept;

const char* selected_qk_path() noexcept;
bool fattn_uses_warp_qk() noexcept;
int fattn_warp_qk_occupancy() noexcept;
const char* selected_query_prepare_path() noexcept;
bool fattn_uses_prepared_query() noexcept;
int fattn_prepared_query_occupancy() noexcept;
int attention_prepare_query_occupancy() noexcept;
std::size_t attention_prepared_query_bytes(const AttentionConfig& config,
                                           std::size_t token_count) noexcept;
bool query_prepare_is_hoisted(const char* path) noexcept;
void set_query_prepare_path_override(const char* path) noexcept;
void clear_query_prepare_path_override() noexcept;
const char* selected_attention_pipeline_path() noexcept;
bool fattn_uses_attention_pipeline() noexcept;
int fattn_pipeline_occupancy() noexcept;
int fattn_pipeline_occupancy_path(const char* path) noexcept;
void set_attention_pipeline_path_override(const char* path) noexcept;
void clear_attention_pipeline_path_override() noexcept;
bool attention_pipeline_is_off(const char* path) noexcept;
bool apply_attention_pipeline_ident(const char* path) noexcept;
const char* last_attention_pipeline_path() noexcept;
const char* last_attention_pipeline_launch() noexcept;
int last_attention_pipeline_convert_once() noexcept;
void fattn_pipeline_kv_once_attributes(int* regs, std::size_t* local_bytes,
                                      int* occupancy) noexcept;
void fattn_pipeline_f16_async_attributes(int* regs, std::size_t* local_bytes,
                                         int* occupancy) noexcept;

struct QueryPreparePathScope final {
  explicit QueryPreparePathScope(const char* path) noexcept {
    set_query_prepare_path_override(path);
  }
  ~QueryPreparePathScope() { clear_query_prepare_path_override(); }
  QueryPreparePathScope(const QueryPreparePathScope&) = delete;
  QueryPreparePathScope& operator=(const QueryPreparePathScope&) = delete;
};

struct AttentionPipelinePathScope final {
  explicit AttentionPipelinePathScope(const char* path) noexcept {
    set_attention_pipeline_path_override(path);
  }
  ~AttentionPipelinePathScope() { clear_attention_pipeline_path_override(); }
  AttentionPipelinePathScope(const AttentionPipelinePathScope&) = delete;
  AttentionPipelinePathScope& operator=(const AttentionPipelinePathScope&) =
      delete;
};

cudaError_t launch_attention_prepare_prompt_queries(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_norm_scale,
    __half* prepared_q, float* normalized_query, const char* prepare_path,
    cudaStream_t stream) noexcept;

cudaError_t launch_attention_prepare_chunk_stream_k_qk(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* key,
    const float* value, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    const AttentionCache& committed, const AttentionCache& candidate_rows,
    float* normalized_query, float* normalized_key, float* score_workspace,
    float* output, float* partial_vkq, float* meta, const char* qk_path,
    cudaStream_t stream) noexcept;

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
