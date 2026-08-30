#ifndef QW38_ATTENTION_H_
#define QW38_ATTENTION_H_

#include <cstddef>

#include "qw38/status.h"

namespace qw38::internal {

struct AttentionShape final {
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t head_width;
  std::size_t rotary_width;
  std::size_t capacity;
};

Status rms_norm(const float* input, const float* weight, std::size_t count,
                float* output) noexcept;

Status rms_norm_scale(const float* input, const float* scale,
                      std::size_t count, float* output) noexcept;

Status attention_decode_step(
    const AttentionShape& shape, std::size_t position, const float* query,
    std::size_t query_count, const float* key, std::size_t key_count,
    const float* value, std::size_t value_count, const float* query_norm_weight,
    const float* key_norm_weight, const float* output_gate,
    std::size_t gate_count, float* key_cache, std::size_t key_cache_count,
    float* value_cache, std::size_t value_cache_count, float* score_workspace,
    std::size_t score_count, float* output, std::size_t output_count) noexcept;

Status attention_decode_step_scale(
    const AttentionShape& shape, std::size_t position, const float* query,
    std::size_t query_count, const float* key, std::size_t key_count,
    const float* value, std::size_t value_count, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    std::size_t gate_count, float* key_cache, std::size_t key_cache_count,
    float* value_cache, std::size_t value_cache_count, float* score_workspace,
    std::size_t score_count, float* output, std::size_t output_count) noexcept;

#ifdef QW38_DIAGNOSTIC_TRACE
Status attention_decode_step_scale_traced(
    const AttentionShape& shape, std::size_t position, const float* query,
    std::size_t query_count, const float* key, std::size_t key_count,
    const float* value, std::size_t value_count, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    std::size_t gate_count, float* key_cache, std::size_t key_cache_count,
    float* value_cache, std::size_t value_cache_count, float* score_workspace,
    std::size_t score_count, float* output, std::size_t output_count,
    float* rope_query, std::size_t rope_query_count, float* rope_key,
    std::size_t rope_key_count) noexcept;
#endif

Status swiglu_ffn(
    const float* input, std::size_t hidden_width,
    const float* gate_weights, const float* up_weights,
    std::size_t intermediate_width, const float* down_weights,
    float* gate_projection, float* up_projection, float* activated,
    float* output) noexcept;

}  // namespace qw38::internal

#endif  // QW38_ATTENTION_H_
