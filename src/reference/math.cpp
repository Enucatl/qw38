#include "reference/math.hpp"

#include "format/floatcvt.hpp"

#include <cmath>
#include <limits>

namespace qw38::reference {
namespace {

using qw38::format::bf16_to_fp32;
using qw38::format::fp32_to_bf16_rne;

Error arg_error(std::string_view field, std::string_view detail) {
  return make_error(ErrorCode::InvalidArgument, field, detail);
}

Error shape_error(std::string_view field, std::string_view detail) {
  return make_error(ErrorCode::InvalidShape, field, detail);
}

Error index_error(std::string_view field, std::string_view detail) {
  return make_error(ErrorCode::InvalidIndex, field, detail);
}

template <class T>
bool finite_val(T x) noexcept {
  return x == x && x <= std::numeric_limits<T>::max() &&
         x >= std::numeric_limits<T>::lowest();
}

std::expected<void, Error> require_span_size(std::size_t got, std::size_t want,
                                             std::string_view field) {
  if (got != want) {
    return std::unexpected(shape_error(field, "span length does not match shape"));
  }
  return {};
}

template <class Acc>
std::expected<void, Error> rms_norm_1p_gamma_impl(
    std::span<float const> x, std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out, std::uint32_t dim, std::string_view name) {
  if (!eps_ok(eps)) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (auto st = require_span_size(x.size(), dim, name); !st) {
    return st;
  }
  if (auto st = require_span_size(gamma.size(), dim, "gamma"); !st) {
    return st;
  }
  if (auto st = require_span_size(out.size(), dim, "out"); !st) {
    return st;
  }
  Acc sumsq = Acc{0};
  for (std::uint32_t i = 0; i < dim; ++i) {
    Acc const v = static_cast<Acc>(x[i]);
    sumsq += v * v;
  }
  Acc const rms = std::sqrt(sumsq / static_cast<Acc>(dim) + static_cast<Acc>(eps));
  Acc const inv_rms = Acc{1} / rms;
  for (std::uint32_t i = 0; i < dim; ++i) {
    Acc const g = static_cast<Acc>(bf16_to_fp32(gamma[i]));
    Acc const y = (Acc{1} + g) * static_cast<Acc>(x[i]) * inv_rms;
    out[i] = fp32_to_bf16_rne(static_cast<float>(y));
  }
  return {};
}

}  // namespace

bool eps_ok(float eps) noexcept { return finite_val(eps) && eps > 0.0f; }

std::array<float, kRopeFreqs> rope_inv_freq() {
  std::array<float, kRopeFreqs> out{};
  for (std::uint32_t j = 0; j < kRopeFreqs; ++j) {
    double const exponent =
        -2.0 * static_cast<double>(j) / static_cast<double>(kRotaryDim);
    out[j] = static_cast<float>(std::pow(kRopeTheta, exponent));
  }
  return out;
}

float sigmoid_fp32(float u) noexcept {
  if (u >= 0.0f) {
    float const e = std::exp(-u);
    return 1.0f / (1.0f + e);
  }
  float const e = std::exp(u);
  return e / (1.0f + e);
}

float silu_fp32(float z) noexcept { return z * sigmoid_fp32(z); }

double sigmoid_f64(double u) noexcept {
  if (u >= 0.0) {
    double const e = std::exp(-u);
    return 1.0 / (1.0 + e);
  }
  double const e = std::exp(u);
  return e / (1.0 + e);
}

double silu_f64(double z) noexcept { return z * sigmoid_f64(z); }

std::expected<void, Error> embed_gather_bf16_to_fp32(
    std::span<std::uint16_t const> table, std::uint32_t vocab,
    std::uint32_t token_id, std::span<float> residual) {
  if (vocab == 0) {
    return std::unexpected(arg_error("vocab", "vocab must be > 0"));
  }
  if (table.size() != static_cast<std::size_t>(vocab) * kHidden) {
    return std::unexpected(
        shape_error("table", "table must be [vocab, 5120] BF16"));
  }
  if (residual.size() != kHidden) {
    return std::unexpected(
        shape_error("residual", "residual must be [5120] FP32"));
  }
  if (token_id >= vocab) {
    return std::unexpected(index_error("token_id", "token_id >= vocab"));
  }
  std::size_t const row = static_cast<std::size_t>(token_id) * kHidden;
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    residual[i] = bf16_to_fp32(table[row + i]);
  }
  return {};
}

std::expected<void, Error> hidden_rms_norm_1p_gamma(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t> out_bf16) {
  return rms_norm_1p_gamma_impl<float>(residual, gamma, eps, out_bf16, kHidden,
                                       "residual");
}

std::expected<void, Error> hidden_rms_norm_1p_gamma_f64(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t> out_bf16) {
  return rms_norm_1p_gamma_impl<double>(residual, gamma, eps, out_bf16, kHidden,
                                        "residual");
}

std::expected<void, Error> qk_rms_norm_1p_gamma(
    std::span<float const> head, std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out_bf16) {
  return rms_norm_1p_gamma_impl<float>(head, gamma, eps, out_bf16, kHeadDim,
                                       "head");
}

std::expected<void, Error> qk_rms_norm_1p_gamma_f64(
    std::span<float const> head, std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out_bf16) {
  return rms_norm_1p_gamma_impl<double>(head, gamma, eps, out_bf16, kHeadDim,
                                        "head");
}

std::expected<void, Error> gdn_gated_rms_norm(
    std::span<float const> o, std::span<std::uint16_t const> z_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out_bf16) {
  if (!eps_ok(eps)) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (o.size() != kGdnHeadDim || z_bf16.size() != kGdnHeadDim ||
      gamma.size() != kGdnHeadDim || out_bf16.size() != kGdnHeadDim) {
    return std::unexpected(
        shape_error("gdn_gated_rms", "o/z/gamma/out must be [128]"));
  }
  float sumsq = 0.0f;
  for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
    sumsq += o[i] * o[i];
  }
  float const rms =
      std::sqrt(sumsq / static_cast<float>(kGdnHeadDim) + eps);
  float const inv_rms = 1.0f / rms;
  for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
    float const g = bf16_to_fp32(gamma[i]);
    float const z = bf16_to_fp32(z_bf16[i]);
    float const y = g * (o[i] * inv_rms) * silu_fp32(z);
    out_bf16[i] = fp32_to_bf16_rne(y);
  }
  return {};
}

std::expected<void, Error> gdn_gated_rms_norm_f64(
    std::span<float const> o, std::span<std::uint16_t const> z_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out_bf16) {
  if (!eps_ok(eps)) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (o.size() != kGdnHeadDim || z_bf16.size() != kGdnHeadDim ||
      gamma.size() != kGdnHeadDim || out_bf16.size() != kGdnHeadDim) {
    return std::unexpected(
        shape_error("gdn_gated_rms", "o/z/gamma/out must be [128]"));
  }
  double sumsq = 0.0;
  for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
    double const v = static_cast<double>(o[i]);
    sumsq += v * v;
  }
  double const rms =
      std::sqrt(sumsq / static_cast<double>(kGdnHeadDim) + static_cast<double>(eps));
  double const inv_rms = 1.0 / rms;
  for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
    double const g = static_cast<double>(bf16_to_fp32(gamma[i]));
    double const z = static_cast<double>(bf16_to_fp32(z_bf16[i]));
    double const y = g * (static_cast<double>(o[i]) * inv_rms) * silu_f64(z);
    out_bf16[i] = fp32_to_bf16_rne(static_cast<float>(y));
  }
  return {};
}

std::expected<void, Error> sigmoid_fp32(std::span<float const> in,
                                        std::span<float> out) {
  if (in.size() != out.size()) {
    return std::unexpected(shape_error("sigmoid", "in/out lengths must match"));
  }
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = sigmoid_fp32(in[i]);
  }
  return {};
}

std::expected<void, Error> silu_fp32(std::span<float const> in,
                                     std::span<float> out) {
  if (in.size() != out.size()) {
    return std::unexpected(shape_error("silu", "in/out lengths must match"));
  }
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = silu_fp32(in[i]);
  }
  return {};
}

std::expected<void, Error> sigmoid_f64(std::span<float const> in,
                                       std::span<double> out) {
  if (in.size() != out.size()) {
    return std::unexpected(shape_error("sigmoid", "in/out lengths must match"));
  }
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = sigmoid_f64(static_cast<double>(in[i]));
  }
  return {};
}

std::expected<void, Error> silu_f64(std::span<float const> in,
                                    std::span<double> out) {
  if (in.size() != out.size()) {
    return std::unexpected(shape_error("silu", "in/out lengths must match"));
  }
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = silu_f64(static_cast<double>(in[i]));
  }
  return {};
}

std::expected<void, Error> partial_rope(std::span<std::uint16_t const> head_bf16,
                                        std::span<float const> inv_freq,
                                        std::int32_t position,
                                        std::span<std::uint16_t> out_bf16) {
  if (position < 0) {
    return std::unexpected(arg_error("position", "position must be >= 0"));
  }
  if (head_bf16.size() != kHeadDim || out_bf16.size() != kHeadDim) {
    return std::unexpected(shape_error("rope", "head/out must be [256] BF16"));
  }
  if (inv_freq.size() != kRopeFreqs) {
    return std::unexpected(shape_error("inv_freq", "inv_freq must be 32 FP32"));
  }
  float const p = static_cast<float>(position);
  constexpr std::uint32_t kHalf = kRotaryDim / 2;
  for (std::uint32_t j = 0; j < kHalf; ++j) {
    float const x0 = bf16_to_fp32(head_bf16[j]);
    float const x1 = bf16_to_fp32(head_bf16[j + kHalf]);
    float const phase = p * inv_freq[j];
    float const c = std::cos(phase);
    float const s = std::sin(phase);
    out_bf16[j] = fp32_to_bf16_rne(x0 * c - x1 * s);
    out_bf16[j + kHalf] = fp32_to_bf16_rne(x1 * c + x0 * s);
  }
  for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
    out_bf16[i] = head_bf16[i];
  }
  return {};
}

std::expected<void, Error> partial_rope_f64(
    std::span<std::uint16_t const> head_bf16, std::span<float const> inv_freq,
    std::int32_t position, std::span<std::uint16_t> out_bf16) {
  if (position < 0) {
    return std::unexpected(arg_error("position", "position must be >= 0"));
  }
  if (head_bf16.size() != kHeadDim || out_bf16.size() != kHeadDim) {
    return std::unexpected(shape_error("rope", "head/out must be [256] BF16"));
  }
  if (inv_freq.size() != kRopeFreqs) {
    return std::unexpected(shape_error("inv_freq", "inv_freq must be 32 FP32"));
  }
  double const p = static_cast<double>(position);
  constexpr std::uint32_t kHalf = kRotaryDim / 2;
  for (std::uint32_t j = 0; j < kHalf; ++j) {
    double const x0 = static_cast<double>(bf16_to_fp32(head_bf16[j]));
    double const x1 = static_cast<double>(bf16_to_fp32(head_bf16[j + kHalf]));
    double const phase = p * static_cast<double>(inv_freq[j]);
    double const c = std::cos(phase);
    double const s = std::sin(phase);
    out_bf16[j] = fp32_to_bf16_rne(static_cast<float>(x0 * c - x1 * s));
    out_bf16[j + kHalf] =
        fp32_to_bf16_rne(static_cast<float>(x1 * c + x0 * s));
  }
  for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
    out_bf16[i] = head_bf16[i];
  }
  return {};
}

std::expected<void, Error> dense_gemv_bf16(
    std::span<std::uint16_t const> weight, std::span<std::uint16_t const> input,
    std::uint32_t n, std::uint32_t k, std::span<float> out) {
  if (n == 0 || k == 0) {
    return std::unexpected(arg_error("dense_gemv", "N and K must be > 0"));
  }
  if (weight.size() != static_cast<std::size_t>(n) * k || input.size() != k ||
      out.size() != n) {
    return std::unexpected(
        shape_error("dense_gemv", "W is [N,K], x is [K], y is [N]"));
  }
  for (std::uint32_t i = 0; i < n; ++i) {
    float acc = 0.0f;
    std::size_t const row = static_cast<std::size_t>(i) * k;
    for (std::uint32_t j = 0; j < k; ++j) {
      acc += bf16_to_fp32(weight[row + j]) * bf16_to_fp32(input[j]);
    }
    out[i] = acc;
  }
  return {};
}

std::expected<void, Error> dense_gemv_bf16_f64(
    std::span<std::uint16_t const> weight, std::span<std::uint16_t const> input,
    std::uint32_t n, std::uint32_t k, std::span<double> out) {
  if (n == 0 || k == 0) {
    return std::unexpected(arg_error("dense_gemv", "N and K must be > 0"));
  }
  if (weight.size() != static_cast<std::size_t>(n) * k || input.size() != k ||
      out.size() != n) {
    return std::unexpected(
        shape_error("dense_gemv", "W is [N,K], x is [K], y is [N]"));
  }
  for (std::uint32_t i = 0; i < n; ++i) {
    double acc = 0.0;
    std::size_t const row = static_cast<std::size_t>(i) * k;
    for (std::uint32_t j = 0; j < k; ++j) {
      acc += static_cast<double>(bf16_to_fp32(weight[row + j])) *
             static_cast<double>(bf16_to_fp32(input[j]));
    }
    out[i] = acc;
  }
  return {};
}

std::expected<std::uint32_t, Error> argmax_fp32(std::span<float const> logits) {
  if (logits.empty()) {
    return std::unexpected(arg_error("logits", "logits must be non-empty"));
  }
  std::uint32_t best_i = 0;
  float best = logits[0];
  for (std::uint32_t i = 1; i < static_cast<std::uint32_t>(logits.size()); ++i) {
    float const v = logits[i];
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  return best_i;
}

}  // namespace qw38::reference
