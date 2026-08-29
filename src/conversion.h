#ifndef QW38_CONVERSION_H_
#define QW38_CONVERSION_H_

#include <cstddef>

#include "qw38/status.h"

namespace qw38::internal {

Status gdn_tiled_to_grouped(const float* tiled, std::size_t key_heads,
                            std::size_t replicas_per_key,
                            std::size_t head_width, float* grouped,
                            std::size_t count) noexcept;
Status gdn_grouped_to_tiled(const float* grouped, std::size_t key_heads,
                            std::size_t replicas_per_key,
                            std::size_t head_width, float* tiled,
                            std::size_t count) noexcept;

}  // namespace qw38::internal

#endif  // QW38_CONVERSION_H_
