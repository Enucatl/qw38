#ifndef QW38_HOST_CHECKPOINT_H_
#define QW38_HOST_CHECKPOINT_H_

#include <string>
#include <vector>

#include "qw38/engine.h"
#include "qw38/status.h"
#include "scalar_runtime.h"

namespace qw38::internal {

struct HostSamplerState final {
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0;
  std::uint64_t rng_state = 0;
};

Status save_host_checkpoint(const std::string& path,
                            const ModelGeometry& geometry,
                            const ScalarSessionState& state,
                            const std::vector<Token>& tokens,
                            const std::vector<float>& logits,
                            const HostSamplerState& sampler) noexcept;

Status restore_host_checkpoint(const std::string& path,
                               const ModelGeometry& geometry,
                               ScalarSessionState* state,
                               std::vector<Token>* tokens,
                               std::vector<float>* logits,
                               HostSamplerState* sampler) noexcept;

}  // namespace qw38::internal

#endif  // QW38_HOST_CHECKPOINT_H_
