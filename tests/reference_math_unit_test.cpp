#include "format/floatcvt.hpp"
#include "reference/math.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using qw38::format::bf16_to_fp32;
using qw38::format::fp32_to_bf16_rne;
using qw38::reference::ErrorCode;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kGdnHeadDim;
using qw38::reference::kHeadDim;
using qw38::reference::kHidden;
using qw38::reference::kRotaryDim;

namespace {

int g_failures = 0;

void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

void expect(bool cond, std::string_view what) {
  if (!cond) {
    fail(what);
  }
}

std::uint16_t bf16(float x) { return fp32_to_bf16_rne(x); }

std::vector<float> zeros_f(std::uint32_t n) {
  return std::vector<float>(n, 0.0f);
}

std::vector<std::uint16_t> zeros_h(std::uint32_t n) {
  return std::vector<std::uint16_t>(n, bf16(0.0f));
}

std::vector<std::uint16_t> filled_h(std::uint32_t n, float v) {
  return std::vector<std::uint16_t>(n, bf16(v));
}

std::vector<float> ramp_f(std::uint32_t n, float scale) {
  std::vector<float> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = scale * (static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f);
  }
  return out;
}

void test_embed() {
  std::uint32_t const vocab = 4;
  std::vector<std::uint16_t> table(vocab * kHidden);
  for (std::uint32_t t = 0; t < vocab; ++t) {
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      table[t * kHidden + i] = bf16(static_cast<float>(t + 1) + 0.01f * static_cast<float>(i % 50));
    }
  }
  std::vector<float> residual(kHidden, 99.0f);
  auto st = qw38::reference::embed_gather_bf16_to_fp32(table, vocab, 2, residual);
  expect(static_cast<bool>(st), "embed token 2");
  expect(residual[0] == bf16_to_fp32(table[2 * kHidden]), "embed widens BF16 exactly");
  expect(residual[100] == bf16_to_fp32(table[2 * kHidden + 100]), "embed row offset");

  auto bad = qw38::reference::embed_gather_bf16_to_fp32(table, vocab, 4, residual);
  expect(!bad && bad.error().code == ErrorCode::InvalidIndex, "embed invalid token");
  auto shapebad = qw38::reference::embed_gather_bf16_to_fp32(
      std::span<std::uint16_t const>(table.data(), 8), vocab, 0, residual);
  expect(!shapebad && shapebad.error().code == ErrorCode::InvalidShape,
         "embed invalid table shape");
}

void test_hidden_rms_zero_nonzero_gamma() {
  auto x0 = zeros_f(kHidden);
  auto g0 = zeros_h(kHidden);
  std::vector<std::uint16_t> out(kHidden, bf16(1.0f));
  auto st = qw38::reference::hidden_rms_norm_1p_gamma(x0, g0, kDefaultRmsEps, out);
  expect(static_cast<bool>(st), "zero hidden rms");
  for (auto h : out) {
    if (h != bf16(0.0f)) {
      fail("zero vector RMS store is 0");
      break;
    }
  }

  auto x = ramp_f(kHidden, 2.5f);
  auto gamma0 = zeros_h(kHidden);
  auto gamma1 = filled_h(kHidden, 1.0f);
  std::vector<std::uint16_t> y0(kHidden);
  std::vector<std::uint16_t> y1(kHidden);
  expect(static_cast<bool>(qw38::reference::hidden_rms_norm_1p_gamma(
             x, gamma0, kDefaultRmsEps, y0)),
         "rms gamma=0");
  expect(static_cast<bool>(qw38::reference::hidden_rms_norm_1p_gamma(
             x, gamma1, kDefaultRmsEps, y1)),
         "rms gamma=1");
  // 1+gamma: gamma=0 → scale 1; gamma=1 → scale 2. Not multiplicative.
  float a0 = bf16_to_fp32(y0[17]);
  float a1 = bf16_to_fp32(y1[17]);
  expect(std::fabs(a1 - 2.0f * a0) < 1.0e-2f, "1+gamma doubles when gamma=1 vs 0");
  expect(std::fabs(a1 - a0) > 0.1f * (std::fabs(a0) + 1.0e-3f),
         "1+gamma is not identity gamma");

  auto bad_eps = qw38::reference::hidden_rms_norm_1p_gamma(x, gamma0, 0.0f, y0);
  expect(!bad_eps && bad_eps.error().code == ErrorCode::InvalidArgument,
         "eps <= 0 rejected");
  std::vector<float> shortx(8);
  auto bad_shape = qw38::reference::hidden_rms_norm_1p_gamma(shortx, gamma0,
                                                            kDefaultRmsEps, y0);
  expect(!bad_shape && bad_shape.error().code == ErrorCode::InvalidShape,
         "hidden rms shape rejected");
}

void test_qk_and_gdn_gamma_roles() {
  auto q = ramp_f(kHeadDim, 1.25f);
  auto g0 = zeros_h(kHeadDim);
  auto g1 = filled_h(kHeadDim, 1.0f);
  std::vector<std::uint16_t> q0(kHeadDim);
  std::vector<std::uint16_t> q1(kHeadDim);
  expect(static_cast<bool>(
             qw38::reference::qk_rms_norm_1p_gamma(q, g0, kDefaultRmsEps, q0)),
         "qk gamma0");
  expect(static_cast<bool>(
             qw38::reference::qk_rms_norm_1p_gamma(q, g1, kDefaultRmsEps, q1)),
         "qk gamma1");
  expect(std::fabs(bf16_to_fp32(q1[3]) - 2.0f * bf16_to_fp32(q0[3])) < 2.0e-2f,
         "QK RMS uses 1+gamma");

  auto o = ramp_f(kGdnHeadDim, 0.75f);
  auto z = filled_h(kGdnHeadDim, 0.0f);  // SiLU(0)=0 → output 0
  auto gz0 = zeros_h(kGdnHeadDim);
  auto gz1 = filled_h(kGdnHeadDim, 1.0f);
  std::vector<std::uint16_t> u0(kGdnHeadDim);
  std::vector<std::uint16_t> u1(kGdnHeadDim);
  auto z_one = filled_h(kGdnHeadDim, 1.0f);
  expect(static_cast<bool>(qw38::reference::gdn_gated_rms_norm(
             o, z, gz1, kDefaultRmsEps, u0)),
         "gated silu0");
  for (auto h : u0) {
    if (bf16_to_fp32(h) != 0.0f) {
      fail("SiLU(0) zeros gated RMS");
      break;
    }
  }
  expect(static_cast<bool>(qw38::reference::gdn_gated_rms_norm(
             o, z_one, gz0, kDefaultRmsEps, u0)),
         "gated gamma=0");
  expect(static_cast<bool>(qw38::reference::gdn_gated_rms_norm(
             o, z_one, gz1, kDefaultRmsEps, u1)),
         "gated gamma=1");
  expect(bf16_to_fp32(u0[9]) == 0.0f, "multiplicative gamma=0 zeros GDN RMS");
  expect(std::fabs(bf16_to_fp32(u1[9])) > 1.0e-4f,
         "multiplicative gamma=1 is not 1+gamma zero-point");
}

void test_bf16_rounding() {
  auto x = ramp_f(kHeadDim, 3.0f);
  auto gamma = filled_h(kHeadDim, 0.25f);
  std::vector<std::uint16_t> out(kHeadDim);
  expect(static_cast<bool>(
             qw38::reference::qk_rms_norm_1p_gamma(x, gamma, kDefaultRmsEps, out)),
         "rounding rms");
  float sumsq = 0.0f;
  for (float v : x) {
    sumsq += v * v;
  }
  float const inv_rms =
      1.0f / std::sqrt(sumsq / static_cast<float>(kHeadDim) + kDefaultRmsEps);
  bool rounded = false;
  for (std::uint32_t i = 0; i < kHeadDim; ++i) {
    float const y = (1.0f + bf16_to_fp32(gamma[i])) * x[i] * inv_rms;
    expect(out[i] == fp32_to_bf16_rne(y), "declared BF16 store is RNE of FP32 y");
    if (bf16_to_fp32(fp32_to_bf16_rne(y)) != y) {
      rounded = true;
    }
  }
  expect(rounded, "at least one RMS coordinate is not a BF16 value before store");
}

void test_rope() {
  auto inv = qw38::reference::rope_inv_freq();
  expect(inv[0] == 1.0f, "ω_0 is 1");
  expect(inv[1] < inv[0] && inv[31] > 0.0f && inv[31] < inv[30],
         "inv_freq strictly decreasing");

  std::vector<std::uint16_t> head(kHeadDim);
  for (std::uint32_t i = 0; i < kHeadDim; ++i) {
    head[i] = bf16(0.05f * static_cast<float>(i) - 1.0f);
  }
  std::vector<std::uint16_t> out(kHeadDim);
  expect(static_cast<bool>(qw38::reference::partial_rope(head, inv, 0, out)),
         "rope pos 0");
  // pos 0 → identity rotation; BF16 store of the widened rotary pair.
  for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
    if (out[i] != head[i]) {
      fail("RoPE suffix unchanged at pos 0");
      break;
    }
  }

  expect(static_cast<bool>(
             qw38::reference::partial_rope(head, inv, 1234, out)),
         "rope pos 1234");
  for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
    if (out[i] != head[i]) {
      fail("RoPE suffix unchanged at pos 1234");
      break;
    }
  }
  bool rotary_changed = false;
  for (std::uint32_t i = 0; i < kRotaryDim; ++i) {
    if (out[i] != head[i]) {
      rotary_changed = true;
      break;
    }
  }
  expect(rotary_changed, "RoPE mutates first 64 coords at nonzero position");

  std::vector<std::uint16_t> large(kHeadDim);
  expect(static_cast<bool>(qw38::reference::partial_rope(head, inv, 262144, large)),
         "rope large position");
  for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
    if (large[i] != head[i]) {
      fail("RoPE suffix unchanged at 262144");
      break;
    }
  }

  auto neg = qw38::reference::partial_rope(head, inv, -1, out);
  expect(!neg && neg.error().code == ErrorCode::InvalidArgument,
         "negative position rejected");
  std::array<float, 4> short_freq{};
  auto badf = qw38::reference::partial_rope(head, short_freq, 1, out);
  expect(!badf && badf.error().code == ErrorCode::InvalidShape,
         "inv_freq length 32 required");
}

void test_silu_sigmoid_gemv_argmax() {
  std::vector<float> in{-8.0f, -1.0f, 0.0f, 1.0f, 8.0f};
  std::vector<float> sg(5);
  std::vector<float> sl(5);
  expect(static_cast<bool>(qw38::reference::sigmoid_fp32(in, sg)), "sigmoid span");
  expect(static_cast<bool>(qw38::reference::silu_fp32(in, sl)), "silu span");
  expect(sg[2] == 0.5f, "sigmoid(0)=0.5");
  expect(sl[2] == 0.0f, "silu(0)=0");
  expect(sg[0] > 0.0f && sg[0] < 0.01f, "sigmoid large negative saturates");
  expect(sg[4] > 0.99f && sg[4] <= 1.0f, "sigmoid large positive saturates");
  expect(std::fabs(sl[3] - in[3] * sg[3]) < 1.0e-6f, "silu is z*sigmoid(z)");

  std::uint32_t const n = 4;
  std::uint32_t const k = 8;
  std::vector<std::uint16_t> w(n * k);
  std::vector<std::uint16_t> x(k);
  for (std::uint32_t i = 0; i < n * k; ++i) {
    w[i] = bf16(0.125f * static_cast<float>(static_cast<int>(i % 5) - 2));
  }
  for (std::uint32_t j = 0; j < k; ++j) {
    x[j] = bf16(0.25f * static_cast<float>(static_cast<int>(j) - 3));
  }
  std::vector<float> y(n);
  std::vector<double> y64(n);
  expect(static_cast<bool>(qw38::reference::dense_gemv_bf16(w, x, n, k, y)),
         "gemv");
  expect(static_cast<bool>(qw38::reference::dense_gemv_bf16_f64(w, x, n, k, y64)),
         "gemv f64");
  for (std::uint32_t i = 0; i < n; ++i) {
    expect(std::fabs(y[i] - static_cast<float>(y64[i])) <=
               qw38::reference::tol::kGemvAbs +
                   qw38::reference::tol::kGemvRel * std::fabs(y[i]),
           "gemv fp32 vs double");
  }
  auto gemv_bad = qw38::reference::dense_gemv_bf16(w, x, n, k + 1, y);
  expect(!gemv_bad && gemv_bad.error().code == ErrorCode::InvalidShape,
         "gemv shape rejected");

  std::vector<float> logits{1.0f, 3.0f, 3.0f, 2.5f, -9.0f};
  auto am = qw38::reference::argmax_fp32(logits);
  expect(static_cast<bool>(am) && *am == 1, "argmax first max on ties");
  auto empty = qw38::reference::argmax_fp32({});
  expect(!empty && empty.error().code == ErrorCode::InvalidArgument,
         "empty argmax rejected");
}

}  // namespace

int main() {
  test_embed();
  test_hidden_rms_zero_nonzero_gamma();
  test_qk_and_gdn_gamma_roles();
  test_bf16_rounding();
  test_rope();
  test_silu_sigmoid_gemv_argmax();
  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
