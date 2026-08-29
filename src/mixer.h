#ifndef QW38_MIXER_H_
#define QW38_MIXER_H_

#include <cstddef>

#include "qw38/status.h"
#include "weights.h"

namespace qw38::internal {

constexpr std::size_t kResidualWidth = 5120;
constexpr std::size_t kGdnPackedQkvWidth = 10240;
constexpr std::size_t kGdnValueWidth = 6144;
constexpr std::size_t kGdnGateCount = 48;
constexpr std::size_t kAttentionPackedQueryGateWidth = 12288;
constexpr std::size_t kAttentionQueryWidth = 6144;
constexpr std::size_t kAttentionKvWidth = 1024;

struct GdnProjectionWorkspace final {
  float* packed_qkv;
  std::size_t packed_qkv_count;
  float* value_gate;
  std::size_t value_gate_count;
  float* alpha;
  std::size_t alpha_count;
  float* beta;
  std::size_t beta_count;
};

struct AttentionProjectionWorkspace final {
  float* packed_query_gate;
  std::size_t packed_query_gate_count;
  float* query;
  std::size_t query_count;
  float* gate;
  std::size_t gate_count;
  float* key;
  std::size_t key_count;
  float* value;
  std::size_t value_count;
};

Status project_gdn_mixer(const GdnLayerWeights& weights,
                         const float* activation,
                         std::size_t activation_count,
                         const GdnProjectionWorkspace& workspace) noexcept;

Status project_attention_mixer(
    const AttentionLayerWeights& weights, const float* activation,
    std::size_t activation_count,
    const AttentionProjectionWorkspace& workspace) noexcept;

}  // namespace qw38::internal

#endif  // QW38_MIXER_H_
