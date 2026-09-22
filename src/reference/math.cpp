#include "reference/math.hpp"

#include "format/floatcvt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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
struct ScaledRms {
  Acc scale;
  Acc inv_scaled_rms;
};

template <class Acc, class Value>
ScaledRms<Acc> scaled_rms(std::uint32_t dim, Value value, float eps) {
  Acc scale = Acc{0};
  for (std::uint32_t i = 0; i < dim; ++i) {
    scale = std::max(scale, std::abs(static_cast<Acc>(value(i))));
  }
  if (scale == Acc{0}) {
    return {Acc{1}, Acc{1} / std::sqrt(static_cast<Acc>(eps))};
  }

  Acc sumsq = Acc{0};
  for (std::uint32_t i = 0; i < dim; ++i) {
    Acc const scaled = static_cast<Acc>(value(i)) / scale;
    sumsq += scaled * scaled;
  }
  Acc const scaled_eps = std::sqrt(static_cast<Acc>(eps)) / scale;
  Acc const inv_scaled_rms =
      Acc{1} / std::sqrt(sumsq / static_cast<Acc>(dim) +
                         scaled_eps * scaled_eps);
  return {scale, inv_scaled_rms};
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
  auto const rms = scaled_rms<Acc>(
      dim, [&](std::uint32_t i) { return x[i]; }, eps);
  for (std::uint32_t i = 0; i < dim; ++i) {
    Acc const g = static_cast<Acc>(bf16_to_fp32(gamma[i]));
    Acc const normalized =
        (static_cast<Acc>(x[i]) / rms.scale) * rms.inv_scaled_rms;
    Acc const y = (Acc{1} + g) * normalized;
    out[i] = fp32_to_bf16_rne(static_cast<float>(y));
  }
  return {};
}

template <class Acc>
std::expected<void, Error> qk_rms_rope_1p_gamma_impl(
    std::span<std::uint16_t const> projected_head_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16) {
  if (!eps_ok(eps)) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (position < 0) {
    return std::unexpected(arg_error("position", "position must be >= 0"));
  }
  if (projected_head_bf16.size() != kHeadDim ||
      gamma.size() != kHeadDim || out_bf16.size() != kHeadDim) {
    return std::unexpected(shape_error(
        "qk_rms_rope", "projected head/gamma/out must be [256] BF16"));
  }
  if (inv_freq.size() != kRopeFreqs) {
    return std::unexpected(shape_error("inv_freq", "inv_freq must be 32 FP32"));
  }

  std::array<Acc, kHeadDim> normalized{};
  for (std::uint32_t i = 0; i < kHeadDim; ++i) {
    Acc const v = static_cast<Acc>(bf16_to_fp32(projected_head_bf16[i]));
    normalized[i] = v;
  }
  auto const rms = scaled_rms<Acc>(
      kHeadDim, [&](std::uint32_t i) { return normalized[i]; }, eps);
  for (std::uint32_t i = 0; i < kHeadDim; ++i) {
    Acc const g = static_cast<Acc>(bf16_to_fp32(gamma[i]));
    Acc const unit = (normalized[i] / rms.scale) * rms.inv_scaled_rms;
    normalized[i] = (Acc{1} + g) * unit;
  }

  Acc const p = static_cast<Acc>(position);
  constexpr std::uint32_t kHalf = kRotaryDim / 2;
  for (std::uint32_t j = 0; j < kHalf; ++j) {
    Acc const phase = p * static_cast<Acc>(inv_freq[j]);
    Acc const c = std::cos(phase);
    Acc const s = std::sin(phase);
    Acc const x0 = normalized[j];
    Acc const x1 = normalized[j + kHalf];
    out_bf16[j] = fp32_to_bf16_rne(static_cast<float>(x0 * c - x1 * s));
    out_bf16[j + kHalf] =
        fp32_to_bf16_rne(static_cast<float>(x1 * c + x0 * s));
  }
  for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
    out_bf16[i] = fp32_to_bf16_rne(static_cast<float>(normalized[i]));
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

float softplus_fp32(float x) noexcept {
  float const ax = std::fabs(x);
  return std::fmax(x, 0.0f) + std::log1p(std::exp(-ax));
}

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

std::expected<void, Error> qk_rms_rope_1p_gamma(
    std::span<std::uint16_t const> projected_head_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16) {
  return qk_rms_rope_1p_gamma_impl<float>(
      projected_head_bf16, gamma, eps, inv_freq, position, out_bf16);
}

std::expected<void, Error> qk_rms_rope_1p_gamma_f64(
    std::span<std::uint16_t const> projected_head_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16) {
  return qk_rms_rope_1p_gamma_impl<double>(
      projected_head_bf16, gamma, eps, inv_freq, position, out_bf16);
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
  auto const rms = scaled_rms<float>(
      kGdnHeadDim, [&](std::uint32_t i) { return o[i]; }, eps);
  for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
    float const g = bf16_to_fp32(gamma[i]);
    float const z = bf16_to_fp32(z_bf16[i]);
    float const normalized = (o[i] / rms.scale) * rms.inv_scaled_rms;
    float const y = g * normalized * silu_fp32(z);
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
  auto const rms = scaled_rms<double>(
      kGdnHeadDim, [&](std::uint32_t i) { return o[i]; }, eps);
  for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
    double const g = static_cast<double>(bf16_to_fp32(gamma[i]));
    double const z = static_cast<double>(bf16_to_fp32(z_bf16[i]));
    double const normalized =
        (static_cast<double>(o[i]) / rms.scale) * rms.inv_scaled_rms;
    double const y = g * normalized * silu_f64(z);
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

std::expected<void, Error> gdn_conv_history_step(
    std::span<std::uint16_t const> qkv, std::span<std::uint16_t const> taps,
    std::span<std::uint16_t> history, std::uint32_t& cursor,
    std::span<std::uint16_t> convolved) {
  if (cursor >= kConvHistoryTaps) {
    return std::unexpected(arg_error("cursor", "convolution cursor must be < 3"));
  }
  if (auto st = require_span_size(qkv.size(), kQkvWidth, "qkv"); !st) {
    return st;
  }
  if (auto st = require_span_size(taps.size(), kConvKernel * kQkvWidth, "taps");
      !st) {
    return st;
  }
  if (auto st =
          require_span_size(history.size(), kConvHistoryTaps * kQkvWidth, "history");
      !st) {
    return st;
  }
  if (auto st = require_span_size(convolved.size(), kQkvWidth, "convolved"); !st) {
    return st;
  }
  for (std::uint32_t c = 0; c < kQkvWidth; ++c) {
    float acc = 0.0f;
    for (std::uint32_t j = 0; j < kConvKernel; ++j) {
      std::uint16_t xbits = qkv[c];
      if (j < kConvHistoryTaps) {
        std::uint32_t const slot = (cursor + j) % kConvHistoryTaps;
        xbits = history[slot * kQkvWidth + c];
      }
      float const w = bf16_to_fp32(taps[j * kQkvWidth + c]);
      acc += w * bf16_to_fp32(xbits);
    }
    convolved[c] = fp32_to_bf16_rne(silu_fp32(acc));
    history[cursor * kQkvWidth + c] = qkv[c];
  }
  cursor = (cursor + 1u) % kConvHistoryTaps;
  return {};
}

std::expected<void, Error> gdn_qk_normalize(
    std::span<std::uint16_t const> convolved, float eps, std::span<float> q_hat,
    std::span<float> k_hat) {
  if (!eps_ok(eps)) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (auto st = require_span_size(convolved.size(), kQkvWidth, "convolved"); !st) {
    return st;
  }
  std::size_t const nqk =
      static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  if (auto st = require_span_size(q_hat.size(), nqk, "q_hat"); !st) {
    return st;
  }
  if (auto st = require_span_size(k_hat.size(), nqk, "k_hat"); !st) {
    return st;
  }
  auto normalize_heads = [&](std::uint32_t src_off, std::span<float> out) {
    for (std::uint32_t h = 0; h < kGdnKeyHeads; ++h) {
      float sumsq = 0.0f;
      std::uint32_t const base = src_off + h * kGdnHeadDim;
      for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
        float const v = bf16_to_fp32(convolved[base + i]);
        sumsq += v * v;
      }
      float const inv = 1.0f / std::sqrt(sumsq + eps);
      std::uint32_t const dst = h * kGdnHeadDim;
      for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
        out[dst + i] = bf16_to_fp32(convolved[base + i]) * inv;
      }
    }
  };
  normalize_heads(0, q_hat);
  normalize_heads(kGdnKeyHeads * kGdnHeadDim, k_hat);
  return {};
}

std::expected<void, Error> gdn_alpha_beta(
    std::span<float const> a, std::span<float const> b,
    std::span<std::uint16_t const> a_log, std::span<std::uint16_t const> dt_bias,
    std::span<float> alpha, std::span<float> beta) {
  if (auto st = require_span_size(a.size(), kGdnValueHeads, "a"); !st) {
    return st;
  }
  if (auto st = require_span_size(b.size(), kGdnValueHeads, "b"); !st) {
    return st;
  }
  if (auto st = require_span_size(a_log.size(), kGdnValueHeads, "A_log"); !st) {
    return st;
  }
  if (auto st = require_span_size(dt_bias.size(), kGdnValueHeads, "dt_bias");
      !st) {
    return st;
  }
  if (auto st = require_span_size(alpha.size(), kGdnValueHeads, "alpha"); !st) {
    return st;
  }
  if (auto st = require_span_size(beta.size(), kGdnValueHeads, "beta"); !st) {
    return st;
  }
  for (std::uint32_t i = 0; i < kGdnValueHeads; ++i) {
    float const alog = bf16_to_fp32(a_log[i]);
    float const dt = bf16_to_fp32(dt_bias[i]);
    float const sp = softplus_fp32(a[i] + dt);
    alpha[i] = std::exp(-std::exp(alog) * sp);
    beta[i] = sigmoid_fp32(b[i]);
  }
  return {};
}

std::expected<void, Error> gdn_prepare(
    std::span<std::uint16_t const> convolved, std::span<float const> a,
    std::span<float const> b, std::span<std::uint16_t const> a_log,
    std::span<std::uint16_t const> dt_bias, float eps, std::span<float> q_hat,
    std::span<float> k_hat, std::span<float> alpha, std::span<float> beta) {
  if (auto st = gdn_qk_normalize(convolved, eps, q_hat, k_hat); !st) {
    return st;
  }
  return gdn_alpha_beta(a, b, a_log, dt_bias, alpha, beta);
}

std::expected<void, Error> gdn_recurrence_step(
    std::span<float const> q_hat, std::span<float const> k_hat,
    std::span<float const> alpha, std::span<float const> beta,
    std::span<std::uint16_t const> v, std::span<float> s, std::span<float> o) {
  std::size_t const nqk =
      static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::size_t const ns = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim *
                         kGdnHeadDim;
  std::size_t const no =
      static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  std::size_t const nv = no;
  if (auto st = require_span_size(q_hat.size(), nqk, "q_hat"); !st) {
    return st;
  }
  if (auto st = require_span_size(k_hat.size(), nqk, "k_hat"); !st) {
    return st;
  }
  if (auto st = require_span_size(alpha.size(), kGdnValueHeads, "alpha"); !st) {
    return st;
  }
  if (auto st = require_span_size(beta.size(), kGdnValueHeads, "beta"); !st) {
    return st;
  }
  if (auto st = require_span_size(v.size(), nv, "v"); !st) {
    return st;
  }
  if (auto st = require_span_size(s.size(), ns, "s"); !st) {
    return st;
  }
  if (auto st = require_span_size(o.size(), no, "o"); !st) {
    return st;
  }

  float const inv_sqrt =
      1.0f / std::sqrt(static_cast<float>(kGdnHeadDim));
  for (std::uint32_t vh = 0; vh < kGdnValueHeads; ++vh) {
    std::uint32_t const kh = gdn_key_head(vh);
    float const a = alpha[vh];
    float const b = beta[vh];
    float const* q = q_hat.data() + static_cast<std::size_t>(kh) * kGdnHeadDim;
    float const* k = k_hat.data() + static_cast<std::size_t>(kh) * kGdnHeadDim;
    for (std::uint32_t j = 0; j < kGdnHeadDim; ++j) {
      float* row =
          s.data() + (static_cast<std::size_t>(vh) * kGdnHeadDim + j) * kGdnHeadDim;
      float const vj = bf16_to_fp32(v[static_cast<std::size_t>(vh) * kGdnHeadDim + j]);
      float p = 0.0f;
      for (std::uint32_t key = 0; key < kGdnHeadDim; ++key) {
        p += (a * row[key]) * k[key];
      }
      float const e = b * (vj - p);
      float oacc = 0.0f;
      for (std::uint32_t key = 0; key < kGdnHeadDim; ++key) {
        float const d = a * row[key];
        float const ns = d + k[key] * e;
        row[key] = ns;
        oacc += ns * q[key] * inv_sqrt;
      }
      o[static_cast<std::size_t>(vh) * kGdnHeadDim + j] = oacc;
    }
  }
  return {};
}

std::expected<GdnFrontReference, Error> gdn_front_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_qkv,
    std::span<std::uint16_t const> w_z, std::span<std::uint16_t const> w_a,
    std::span<std::uint16_t const> w_b, std::span<std::uint16_t const> taps,
    std::span<std::uint16_t const> a_log, std::span<std::uint16_t const> dt_bias,
    std::span<std::uint16_t const> history, std::uint32_t cursor) {
  if (residual.size() != kHidden) {
    return std::unexpected(
        shape_error("residual", "residual must be [5120] FP32"));
  }
  GdnFrontReference out;
  out.normalized.assign(kHidden, 0);
  if (auto st = hidden_rms_norm_1p_gamma(residual, gamma, eps, out.normalized);
      !st) {
    return std::unexpected(st.error());
  }

  std::vector<float> qkv_f(kQkvWidth, 0.0f);
  std::vector<float> z_f(kGdnZWidth, 0.0f);
  if (auto st = dense_gemv_bf16(w_qkv, out.normalized, kQkvWidth, kHidden, qkv_f);
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = dense_gemv_bf16(w_z, out.normalized, kGdnZWidth, kHidden, z_f);
      !st) {
    return std::unexpected(st.error());
  }
  out.qkv.resize(kQkvWidth);
  out.z.resize(kGdnZWidth);
  for (std::uint32_t i = 0; i < kQkvWidth; ++i) {
    out.qkv[i] = fp32_to_bf16_rne(qkv_f[i]);
  }
  for (std::uint32_t i = 0; i < kGdnZWidth; ++i) {
    out.z[i] = fp32_to_bf16_rne(z_f[i]);
  }

  out.a.assign(kGdnValueHeads, 0.0f);
  out.b.assign(kGdnValueHeads, 0.0f);
  if (auto st =
          dense_gemv_bf16(w_a, out.normalized, kGdnValueHeads, kHidden, out.a);
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st =
          dense_gemv_bf16(w_b, out.normalized, kGdnValueHeads, kHidden, out.b);
      !st) {
    return std::unexpected(st.error());
  }

  out.history.assign(history.begin(), history.end());
  out.cursor = cursor;
  out.convolved.assign(kQkvWidth, 0);
  if (auto st = gdn_conv_history_step(out.qkv, taps, out.history, out.cursor,
                                      out.convolved);
      !st) {
    return std::unexpected(st.error());
  }

  std::size_t const nqk =
      static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  out.q_hat.assign(nqk, 0.0f);
  out.k_hat.assign(nqk, 0.0f);
  out.alpha.assign(kGdnValueHeads, 0.0f);
  out.beta.assign(kGdnValueHeads, 0.0f);
  if (auto st = gdn_prepare(out.convolved, out.a, out.b, a_log, dt_bias, eps,
                            out.q_hat, out.k_hat, out.alpha, out.beta);
      !st) {
    return std::unexpected(st.error());
  }
  return out;
}

std::expected<GdnMixerReference, Error> gdn_mixer_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    std::span<std::uint16_t const> gated_gamma, float eps,
    std::span<std::uint16_t const> w_qkv, std::span<std::uint16_t const> w_z,
    std::span<std::uint16_t const> w_a, std::span<std::uint16_t const> w_b,
    std::span<std::uint16_t const> w_out, std::span<std::uint16_t const> taps,
    std::span<std::uint16_t const> a_log, std::span<std::uint16_t const> dt_bias,
    std::span<std::uint16_t const> history, std::uint32_t cursor,
    std::span<float const> s) {
  std::size_t const ns = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim *
                         kGdnHeadDim;
  std::size_t const no =
      static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  if (auto st = require_span_size(gated_gamma.size(), kGdnHeadDim, "gated_gamma");
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_span_size(w_out.size(), static_cast<std::size_t>(kHidden) *
                                                   kGdnZWidth,
                                  "w_out");
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_span_size(s.size(), ns, "s"); !st) {
    return std::unexpected(st.error());
  }

  auto front = gdn_front_reference(residual, gamma, eps, w_qkv, w_z, w_a, w_b,
                                   taps, a_log, dt_bias, history, cursor);
  if (!front) {
    return std::unexpected(front.error());
  }

  GdnMixerReference out;
  out.front = std::move(*front);
  out.s.assign(s.begin(), s.end());
  out.o.assign(no, 0.0f);
  auto v = std::span<std::uint16_t const>(out.front.convolved).subspan(
      static_cast<std::size_t>(kGdnKeyHeads) * 2u * kGdnHeadDim, no);
  if (auto st = gdn_recurrence_step(out.front.q_hat, out.front.k_hat,
                                    out.front.alpha, out.front.beta, v, out.s,
                                    out.o);
      !st) {
    return std::unexpected(st.error());
  }

  out.u.assign(no, 0);
  for (std::uint32_t vh = 0; vh < kGdnValueHeads; ++vh) {
    std::size_t const off = static_cast<std::size_t>(vh) * kGdnHeadDim;
    auto ost = gdn_gated_rms_norm(
        std::span<float const>(out.o.data() + off, kGdnHeadDim),
        std::span<std::uint16_t const>(out.front.z.data() + off, kGdnHeadDim),
        gated_gamma, eps,
        std::span<std::uint16_t>(out.u.data() + off, kGdnHeadDim));
    if (!ost) {
      return std::unexpected(ost.error());
    }
  }

  std::vector<float> mix(kHidden, 0.0f);
  if (auto st = dense_gemv_bf16(w_out, out.u, kHidden, kGdnZWidth, mix); !st) {
    return std::unexpected(st.error());
  }
  out.residual.resize(kHidden);
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    out.residual[i] = residual[i] + mix[i];
  }
  out.history = out.front.history;
  out.cursor = out.front.cursor;
  return out;
}

std::expected<DecodeMlpReference, Error> decode_mlp_reference(
    std::span<float const> h_mid, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_gate,
    std::span<std::uint16_t const> w_up, std::span<std::uint16_t const> w_down) {
  if (h_mid.size() != kHidden) {
    return std::unexpected(
        shape_error("h_mid", "residual must be [5120] FP32"));
  }
  DecodeMlpReference out;
  out.normalized.assign(kHidden, 0);
  if (auto st = hidden_rms_norm_1p_gamma(h_mid, gamma, eps, out.normalized);
      !st) {
    return std::unexpected(st.error());
  }
  std::vector<float> gate(kFfn, 0.0f);
  std::vector<float> up(kFfn, 0.0f);
  if (auto st = dense_gemv_bf16(w_gate, out.normalized, kFfn, kHidden, gate);
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = dense_gemv_bf16(w_up, out.normalized, kFfn, kHidden, up); !st) {
    return std::unexpected(st.error());
  }
  out.swiglu.assign(kFfn, 0);
  for (std::uint32_t i = 0; i < kFfn; ++i) {
    out.swiglu[i] = fp32_to_bf16_rne(silu_fp32(gate[i]) * up[i]);
  }
  std::vector<float> down(kHidden, 0.0f);
  if (auto st = dense_gemv_bf16(w_down, out.swiglu, kHidden, kFfn, down); !st) {
    return std::unexpected(st.error());
  }
  out.residual.resize(kHidden);
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    out.residual[i] = h_mid[i] + down[i];
  }
  return out;
}

std::expected<void, Error> attn_split_qg(std::span<std::uint16_t const> qg,
                                         std::span<std::uint16_t> q_raw,
                                         std::span<std::uint16_t> g) {
  std::size_t const nq =
      static_cast<std::size_t>(kQueryHeads) * kHeadDim;
  if (auto st = require_span_size(qg.size(), static_cast<std::size_t>(kQgWidth),
                                  "qg");
      !st) {
    return st;
  }
  if (auto st = require_span_size(q_raw.size(), nq, "q_raw"); !st) {
    return st;
  }
  if (auto st = require_span_size(g.size(), nq, "g"); !st) {
    return st;
  }
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    std::size_t const src = static_cast<std::size_t>(h) * (2u * kHeadDim);
    std::size_t const dst = static_cast<std::size_t>(h) * kHeadDim;
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      q_raw[dst + i] = qg[src + i];
      g[dst + i] = qg[src + kHeadDim + i];
    }
  }
  return {};
}

std::expected<void, Error> attn_qk_norm_rope(
    std::span<std::uint16_t const> head_bf16,
    std::span<std::uint16_t const> gamma, std::span<float const> inv_freq,
    std::int32_t position, float eps, std::span<std::uint16_t> out_bf16) {
  return qk_rms_rope_1p_gamma(
      head_bf16, gamma, eps, inv_freq, position, out_bf16);
}

std::expected<void, Error> attn_cache_append(
    std::span<std::uint16_t> kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t token,
    std::span<std::uint16_t const> k, std::span<std::uint16_t const> v) {
  std::size_t const nkv =
      static_cast<std::size_t>(kKvHeads) * kHeadDim;
  if (capacity == 0 || token >= capacity) {
    return std::unexpected(arg_error("token", "token must be < capacity"));
  }
  if (attn_layer >= 16u) {
    return std::unexpected(arg_error("attn_layer", "attn_layer must be < 16"));
  }
  std::size_t const layer_elems =
      static_cast<std::size_t>(2u) * kKvHeads * static_cast<std::size_t>(capacity) *
      kHeadDim;
  std::size_t const want = static_cast<std::size_t>(attn_layer + 1u) * layer_elems;
  if (kv.size() < want) {
    return std::unexpected(shape_error("kv", "cache is smaller than the addressed layer"));
  }
  if (auto st = require_span_size(k.size(), nkv, "k"); !st) {
    return st;
  }
  if (auto st = require_span_size(v.size(), nkv, "v"); !st) {
    return st;
  }
  std::size_t const head_stride = static_cast<std::size_t>(capacity) * kHeadDim;
  std::size_t const comp_stride = static_cast<std::size_t>(kKvHeads) * head_stride;
  std::size_t const layer_off = static_cast<std::size_t>(attn_layer) * 2u * comp_stride;
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    std::size_t const src = static_cast<std::size_t>(h) * kHeadDim;
    std::size_t const k_off = layer_off + static_cast<std::size_t>(h) * head_stride +
                              static_cast<std::size_t>(token) * kHeadDim;
    std::size_t const v_off = layer_off + comp_stride +
                              static_cast<std::size_t>(h) * head_stride +
                              static_cast<std::size_t>(token) * kHeadDim;
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      kv[k_off + i] = k[src + i];
      kv[v_off + i] = v[src + i];
    }
  }
  return {};
}

std::expected<AttnPrepReference, Error> attn_prep_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_qg,
    std::span<std::uint16_t const> w_k, std::span<std::uint16_t const> w_v,
    std::span<std::uint16_t const> gamma_q, std::span<std::uint16_t const> gamma_k,
    std::span<float const> inv_freq, std::int32_t position) {
  if (residual.size() != kHidden) {
    return std::unexpected(
        shape_error("residual", "residual must be [5120] FP32"));
  }
  AttnPrepReference out;
  out.normalized.assign(kHidden, 0);
  if (auto st = hidden_rms_norm_1p_gamma(residual, gamma, eps, out.normalized);
      !st) {
    return std::unexpected(st.error());
  }

  std::vector<float> qg_f(kQgWidth, 0.0f);
  std::vector<float> k_f(kAttnKvWidth, 0.0f);
  std::vector<float> v_f(kAttnKvWidth, 0.0f);
  if (auto st = dense_gemv_bf16(w_qg, out.normalized, kQgWidth, kHidden, qg_f);
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = dense_gemv_bf16(w_k, out.normalized, kAttnKvWidth, kHidden, k_f);
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = dense_gemv_bf16(w_v, out.normalized, kAttnKvWidth, kHidden, v_f);
      !st) {
    return std::unexpected(st.error());
  }
  out.qg.resize(kQgWidth);
  out.k_raw.resize(kAttnKvWidth);
  out.v_raw.resize(kAttnKvWidth);
  for (std::uint32_t i = 0; i < kQgWidth; ++i) {
    out.qg[i] = fp32_to_bf16_rne(qg_f[i]);
  }
  for (std::uint32_t i = 0; i < kAttnKvWidth; ++i) {
    out.k_raw[i] = fp32_to_bf16_rne(k_f[i]);
    out.v_raw[i] = fp32_to_bf16_rne(v_f[i]);
  }

  std::size_t const nq =
      static_cast<std::size_t>(kQueryHeads) * kHeadDim;
  std::vector<std::uint16_t> q_raw(nq, 0);
  out.g.assign(nq, 0);
  if (auto st = attn_split_qg(out.qg, q_raw, out.g); !st) {
    return std::unexpected(st.error());
  }
  out.q.assign(nq, 0);
  out.k.assign(kAttnKvWidth, 0);
  out.v = out.v_raw;
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    std::size_t const off = static_cast<std::size_t>(h) * kHeadDim;
    auto st = attn_qk_norm_rope(
        std::span<std::uint16_t const>(q_raw.data() + off, kHeadDim), gamma_q,
        inv_freq, position, eps,
        std::span<std::uint16_t>(out.q.data() + off, kHeadDim));
    if (!st) {
      return std::unexpected(st.error());
    }
  }
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    std::size_t const off = static_cast<std::size_t>(h) * kHeadDim;
    auto st = attn_qk_norm_rope(
        std::span<std::uint16_t const>(out.k_raw.data() + off, kHeadDim),
        gamma_k, inv_freq, position, eps,
        std::span<std::uint16_t>(out.k.data() + off, kHeadDim));
    if (!st) {
      return std::unexpected(st.error());
    }
  }
  return out;
}

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct OnlinePartial {
  float m{kNegInf};
  float l{0.0f};
  float num[kHeadDim]{};
};

void merge_partial(OnlinePartial& a, OnlinePartial const& b) {
  float const m_new = a.m > b.m ? a.m : b.m;
  if (!(m_new > kNegInf)) {
    return;
  }
  float const sa = (a.m > kNegInf) ? std::exp(a.m - m_new) : 0.0f;
  float const sb = (b.m > kNegInf) ? std::exp(b.m - m_new) : 0.0f;
  a.l = sa * a.l + sb * b.l;
  for (std::uint32_t d = 0; d < kHeadDim; ++d) {
    a.num[d] = sa * a.num[d] + sb * b.num[d];
  }
  a.m = m_new;
}

void online_key(OnlinePartial& p, float score, std::span<float const> v) {
  float const m_new = p.m > score ? p.m : score;
  float const alpha = (p.m > kNegInf) ? std::exp(p.m - m_new) : 0.0f;
  float const w = std::exp(score - m_new);
  p.l = p.l * alpha + w;
  for (std::uint32_t d = 0; d < kHeadDim; ++d) {
    p.num[d] = p.num[d] * alpha + w * v[d];
  }
  p.m = m_new;
}

float head_dot(std::span<float const> q, std::span<float const> k) {
  float s = 0.0f;
  for (std::uint32_t d = 0; d < kHeadDim; ++d) {
    s += q[d] * k[d];
  }
  return s * kAttnScale;
}

std::size_t cache_index(std::uint32_t attn_layer, std::uint32_t component,
                        std::uint32_t head, std::uint64_t token,
                        std::uint32_t dim, std::uint64_t capacity) {
  std::size_t const head_stride = static_cast<std::size_t>(capacity) * kHeadDim;
  std::size_t const comp_stride = static_cast<std::size_t>(kKvHeads) * head_stride;
  return (static_cast<std::size_t>(attn_layer) * 2u + component) * comp_stride +
         static_cast<std::size_t>(head) * head_stride +
         static_cast<std::size_t>(token) * kHeadDim + dim;
}

std::expected<void, Error> require_kv(std::span<std::uint16_t const> kv,
                                      std::uint32_t attn_layer,
                                      std::uint64_t capacity) {
  if (capacity == 0) {
    return std::unexpected(arg_error("capacity", "capacity must be > 0"));
  }
  if (attn_layer >= 16u) {
    return std::unexpected(arg_error("attn_layer", "attn_layer must be < 16"));
  }
  std::size_t const layer_elems =
      static_cast<std::size_t>(2u) * kKvHeads * static_cast<std::size_t>(capacity) *
      kHeadDim;
  std::size_t const want = static_cast<std::size_t>(attn_layer + 1u) * layer_elems;
  if (kv.size() < want) {
    return std::unexpected(shape_error("kv", "cache is smaller than the addressed layer"));
  }
  return {};
}

OnlinePartial scan_range(std::span<float const> q, std::span<std::uint16_t const> kv,
                         std::uint32_t attn_layer, std::uint32_t kv_h,
                         std::uint64_t capacity, std::uint64_t key0,
                         std::uint64_t key1, std::uint64_t populated) {
  OnlinePartial p;
  float v[kHeadDim];
  float k[kHeadDim];
  for (std::uint64_t t = key0; t < key1; ++t) {
    if (t >= populated || t >= capacity) {
      continue;
    }
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      k[d] = bf16_to_fp32(
          kv[cache_index(attn_layer, kKvComponentK, kv_h, t, d, capacity)]);
      v[d] = bf16_to_fp32(
          kv[cache_index(attn_layer, kKvComponentV, kv_h, t, d, capacity)]);
    }
    online_key(p, head_dot(q, std::span<float const>(k, kHeadDim)),
               std::span<float const>(v, kHeadDim));
  }
  return p;
}

}  // namespace

std::expected<AttnCoreReference, Error> attn_online_core(
    std::span<std::uint16_t const> q, std::span<std::uint16_t const> g,
    std::span<std::uint16_t const> kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t populated, bool segmented) {
  std::size_t const nq = static_cast<std::size_t>(kQueryHeads) * kHeadDim;
  if (auto st = require_span_size(q.size(), nq, "q"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_span_size(g.size(), nq, "g"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_kv(kv, attn_layer, capacity); !st) {
    return std::unexpected(st.error());
  }
  if (populated > capacity) {
    return std::unexpected(
        arg_error("populated", "populated length exceeds capacity"));
  }

  AttnCoreReference out;
  out.attn.assign(nq, 0.0f);
  out.y.assign(nq, 0);
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    std::uint32_t const kv_h = kv_head_for_query(h);
    float qf[kHeadDim];
    std::size_t const off = static_cast<std::size_t>(h) * kHeadDim;
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      qf[d] = bf16_to_fp32(q[off + d]);
    }
    OnlinePartial acc;
    if (segmented) {
      std::uint64_t const nseg =
          populated == 0 ? 0 : (populated + (kAttnSegmentKeys - 1u)) / kAttnSegmentKeys;
      for (std::uint64_t s = 0; s < nseg; ++s) {
        std::uint64_t const key0 = s * kAttnSegmentKeys;
        std::uint64_t const key1 = key0 + kAttnSegmentKeys;
        merge_partial(acc, scan_range(std::span<float const>(qf, kHeadDim), kv,
                                      attn_layer, kv_h, capacity, key0, key1,
                                      populated));
      }
    } else {
      acc = scan_range(std::span<float const>(qf, kHeadDim), kv, attn_layer, kv_h,
                       capacity, 0, populated, populated);
    }
    float const inv = (acc.l > 0.0f) ? (1.0f / acc.l) : 0.0f;
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      float const attn = acc.num[d] * inv;
      out.attn[off + d] = attn;
      float const gate = sigmoid_fp32(bf16_to_fp32(g[off + d]));
      out.y[off + d] = fp32_to_bf16_rne(attn * gate);
    }
  }
  return out;
}

std::expected<AttnMixerReference, Error> attn_mixer_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_qg,
    std::span<std::uint16_t const> w_k, std::span<std::uint16_t const> w_v,
    std::span<std::uint16_t const> w_o, std::span<std::uint16_t const> gamma_q,
    std::span<std::uint16_t const> gamma_k, std::span<float const> inv_freq,
    std::span<std::uint16_t const> kv_in, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t token, bool segmented) {
  if (token >= capacity) {
    return std::unexpected(arg_error("token", "token must be < capacity"));
  }
  auto prep = attn_prep_reference(residual, gamma, eps, w_qg, w_k, w_v, gamma_q,
                                  gamma_k, inv_freq, static_cast<std::int32_t>(token));
  if (!prep) {
    return std::unexpected(prep.error());
  }
  std::vector<std::uint16_t> kv(kv_in.begin(), kv_in.end());
  if (auto st = attn_cache_append(kv, attn_layer, capacity, token, prep->k, prep->v);
      !st) {
    return std::unexpected(st.error());
  }
  auto core = attn_online_core(prep->q, prep->g, kv, attn_layer, capacity, token + 1u,
                               segmented);
  if (!core) {
    return std::unexpected(core.error());
  }
  std::vector<float> mix(kHidden, 0.0f);
  if (auto st = dense_gemv_bf16(w_o, core->y, kHidden, kAttnOutWidth, mix); !st) {
    return std::unexpected(st.error());
  }
  AttnMixerReference out;
  out.prep = std::move(*prep);
  out.core = std::move(*core);
  out.kv = std::move(kv);
  out.residual.resize(kHidden);
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    out.residual[i] = residual[i] + mix[i];
  }
  return out;
}

std::expected<std::uint32_t, Error> argmax_fp32(std::span<float const> logits) {
  if (logits.empty()) {
    return std::unexpected(arg_error("logits", "logits must be non-empty"));
  }
  for (float const v : logits) {
    if (!std::isfinite(v)) {
      return std::unexpected(
          arg_error("logits", "logits must contain only finite values"));
    }
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
