#pragma once

#include "cuda/prefill.hpp"
#include "runtime/error.hpp"
#include "runtime/gdn.hpp"
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

// A layer plan borrows the uploaded model and session. The workspace and the
// projection engine are owned separately so one allocation serves every layer.
class PrefillGdnWorkspace;

class PrefillGdnLayerPlan {
 public:
  PrefillGdnLayerPlan(PrefillGdnLayerPlan const&) = default;
  PrefillGdnLayerPlan& operator=(PrefillGdnLayerPlan const&) = default;

 private:
  friend std::expected<PrefillGdnLayerPlan, Error> bind_prefill_gdn_layer(
      Model const&, Session&, std::uint32_t, qw38::cuda::Stream const&);
  friend std::expected<void, Error> execute_prefill_gdn_layer(
      PrefillGdnLayerPlan const&, PrefillGdnWorkspace&,
      qw38::cuda::PrefillEngine&, float const*, float*, float*,
      std::uint32_t, std::uint64_t, std::uint32_t);

  PrefillGdnLayerPlan(GdnPlan const& gdn,
                      PrefillLayerProjectionPlan const& projections)
      : gdn_(gdn), projections_(projections) {}
  GdnPlan gdn_{};
  PrefillLayerProjectionPlan projections_{};
};

[[nodiscard]] std::expected<PrefillGdnLayerPlan, Error>
bind_prefill_gdn_layer(Model const& model, Session& session,
                       std::uint32_t layer, qw38::cuda::Stream const& stream);

struct PrefillGdnSlices {
  std::uint16_t* normalized{};  // BF16 [M,5120]
  std::uint16_t* qkv{};         // BF16 [M,10240]
  std::uint16_t* z{};           // BF16 [M,6144]
  float* a{};                   // FP32 [M,48]
  float* b{};                   // FP32 [M,48]
  std::uint16_t* convolved{};   // BF16 [M,10240]
  float* q_hat{};               // FP32 [M,16,128]
  float* k_hat{};               // FP32 [M,16,128]
  float* alpha{};               // FP32 [M,48]
  float* beta{};                // FP32 [M,48]
  float* o{};                   // FP32 [M,48,128]
  std::uint16_t* u{};           // BF16 [M,48,128]
};

class PrefillGdnWorkspace {
 public:
  [[nodiscard]] static std::expected<PrefillGdnWorkspace, Error> create(
      std::uint32_t token_capacity, int device);
  [[nodiscard]] PrefillGdnSlices slices() noexcept;
  [[nodiscard]] std::uint32_t token_capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return storage_.bytes(); }
  [[nodiscard]] void const* data() const noexcept { return storage_.data(); }
  [[nodiscard]] int device() const noexcept { return storage_.device(); }

 private:
  qw38::cuda::DeviceBuffer storage_{};
  std::uint32_t capacity_{};
};

// Complete GDN mixer and post-mixer MLP for one contiguous valid chunk.
// Drains the stream before returning after a launch, including on error.
// Any failure after the first launch poisons the Session until reset or restore.
[[nodiscard]] std::expected<void, Error> execute_prefill_gdn_layer(
    PrefillGdnLayerPlan const& plan, PrefillGdnWorkspace& workspace,
    qw38::cuda::PrefillEngine& engine, float const* residual,
    float* h_mid, float* next_h, std::uint32_t valid_tokens,
    std::uint64_t first_position, std::uint32_t recurrence_interval = 64);

}  // namespace qw38::runtime
