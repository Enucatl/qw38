#ifndef QW38_PROJECTION_H_
#define QW38_PROJECTION_H_

#include <cstddef>

#include "qw38/status.h"

namespace qw38::internal {

Status split_gdn_qkv(const float* packed, std::size_t packed_count,
                     std::size_t key_heads, std::size_t key_width,
                     std::size_t value_heads, std::size_t value_width,
                     float* query, std::size_t query_count, float* key,
                     std::size_t key_count, float* value,
                     std::size_t value_count) noexcept;

Status split_attention_query_gate(
    const float* packed, std::size_t packed_count, std::size_t query_heads,
    std::size_t head_width, float* query, std::size_t query_count, float* gate,
    std::size_t gate_count) noexcept;

}  // namespace qw38::internal

#endif  // QW38_PROJECTION_H_
