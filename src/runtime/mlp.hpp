#pragma once

#include "runtime/error.hpp"
#include "runtime/model.hpp"
#include "runtime/session.hpp"
#include "runtime/sizes.hpp"
#include "runtime/view.hpp"

#include "cuda/stream.hpp"

#include <cstdint>
#include <expected>
#include <string>

namespace qw38::runtime {

inline constexpr std::uint32_t kMlpLayers = 64;
inline constexpr float kMlpRmsEps = 1.0e-6f;

// Bound decode MLP: three weight identities, post-mixer RMS gamma, distinct
// input/output residual views, two scratch views, and one ordered stream.
// execute_decode_mlp allocates nothing.
struct MlpWeightBinding {
  TensorView codes{};
  TensorView scales{};
  std::uint16_t layout{};
  std::uint16_t quantizer{};
  std::uint32_t n{};
  std::uint32_t k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  std::uint64_t codes_bytes{};
  std::uint64_t scales_bytes{};
};

struct MlpPlan {
  MlpWeightBinding gate{};
  MlpWeightBinding up{};
  MlpWeightBinding down{};
  TensorView gamma{};
  TensorView h_mid{};        // FP32 input, preserved through the down add
  TensorView next_h{};       // FP32 output: h_mid + down projection
  TensorView normalized{};   // BF16 [5120]
  TensorView swiglu{};       // BF16 [17408]; only globally materialized gate/up value
  qw38::cuda::Stream const* stream{nullptr};
  float eps{kMlpRmsEps};
};

struct MlpBindViews {
  TensorView gate{};
  TensorView gate_scales{};
  TensorView up{};
  TensorView up_scales{};
  TensorView down{};
  TensorView down_scales{};
  TensorView gamma{};
  TensorView h_mid{};
  TensorView next_h{};
  TensorView normalized{};
  TensorView swiglu{};
};

[[nodiscard]] std::string mlp_gate_name(std::uint32_t layer);
[[nodiscard]] std::string mlp_up_name(std::uint32_t layer);
[[nodiscard]] std::string mlp_down_name(std::uint32_t layer);
[[nodiscard]] std::string mlp_norm_name(std::uint32_t layer);

[[nodiscard]] std::expected<MlpPlan, Error> bind_mlp_plan(
    MlpBindViews const& views, qw38::cuda::Stream const& stream,
    float eps = kMlpRmsEps);

[[nodiscard]] std::expected<MlpPlan, Error> bind_mlp_plan(
    Model const& model, Session const& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps = kMlpRmsEps);

// Three launches: hidden RMS, paired Q4/BF16 gate/up + SwiGLU, down residual-add.
[[nodiscard]] std::expected<void, Error> execute_decode_mlp(MlpPlan const& plan);

}  // namespace qw38::runtime
