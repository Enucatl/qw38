#ifndef QW38_GDN_H_
#define QW38_GDN_H_

#include <cstddef>

#include "qw38/status.h"

namespace qw38::internal {

struct GdnShape final {
  std::size_t key_heads;
  std::size_t value_heads;
  std::size_t key_width;
  std::size_t value_width;
};

Status gdn_gates_from_source(const float* a, const float* b,
                             const float* a_log, const float* dt_bias,
                             std::size_t count, float* log_decay,
                             float* beta) noexcept;

Status gdn_gates_from_gguf(const float* a, const float* b,
                           const float* folded_a, const float* dt_bias,
                           std::size_t count, float* log_decay,
                           float* beta) noexcept;

Status gdn_recurrent_step_precomputed(
    const GdnShape& shape, const float* query, std::size_t query_count,
    const float* key, std::size_t key_count, const float* value,
    std::size_t value_count, const float* log_decay, const float* beta,
    std::size_t gate_count, float* state, std::size_t state_count,
    float* output, std::size_t output_count) noexcept;

Status gdn_recurrent_step(
    const GdnShape& shape, const float* query, std::size_t query_count,
    const float* key, std::size_t key_count, const float* value,
    std::size_t value_count, const float* a, const float* b,
    const float* a_log, const float* dt_bias, std::size_t gate_count,
    float* state, std::size_t state_count, float* output,
    std::size_t output_count, float* log_decay, float* beta) noexcept;

Status causal_depthwise_conv_step(
    std::size_t channels, std::size_t kernel_width, const float* input,
    std::size_t input_count, const float* weights, std::size_t weight_count,
    float* state, std::size_t state_count, float* output,
    std::size_t output_count) noexcept;

}  // namespace qw38::internal

#endif  // QW38_GDN_H_
