#pragma once

#include "cuda/prefill.hpp"
#include "runtime/error.hpp"
#include "runtime/model.hpp"

#include <cstdint>
#include <expected>

namespace qw38::runtime {

// Weight-only projection binding. Model and its uploaded buffers must outlive
// this plan. Mixer cores and state commits belong to TASK-024/025.
struct PrefillLayerProjectionPlan {
  bool gdn{};
  std::uint32_t layer{};
  qw38::cuda::PrefillWeight first{};   // GDN qkv or attention q/g
  qw38::cuda::PrefillWeight second{};  // GDN z or attention k
  qw38::cuda::PrefillWeight third{};   // GDN a or attention v
  qw38::cuda::PrefillWeight fourth{};  // GDN b; empty for attention
  qw38::cuda::PrefillWeight mixer_out{};
  qw38::cuda::PrefillWeight mlp_gate{};
  qw38::cuda::PrefillWeight mlp_up{};
  qw38::cuda::PrefillWeight mlp_down{};
  std::uint16_t const* mlp_gamma{};
};

[[nodiscard]] std::expected<PrefillLayerProjectionPlan, Error>
bind_prefill_layer_projections(Model const& model, std::uint32_t layer,
                               qw38::cuda::Stream const& stream);

[[nodiscard]] std::expected<qw38::cuda::PrefillWeight, Error>
bind_prefill_head(Model const& model, qw38::cuda::Stream const& stream);

// Complete prefill MLP projection chain, without mixer recurrence/attention.
[[nodiscard]] std::expected<void, Error> execute_prefill_mlp(
    PrefillLayerProjectionPlan const& plan, qw38::cuda::PrefillEngine& engine,
    float const* h_mid, float* next_h, std::uint32_t valid_tokens,
    std::uint64_t first_position, float eps = 1.0e-6f);

}  // namespace qw38::runtime
