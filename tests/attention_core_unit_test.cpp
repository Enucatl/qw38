#include "cuda/attention.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/stream.hpp"
#include "format/floatcvt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

constexpr std::uint32_t kHeads = qw38::cuda::kAttnQueryHeads;
constexpr std::uint32_t kKvHeads = qw38::cuda::kAttnKvHeads;
constexpr std::uint32_t kDim = qw38::cuda::kAttnHeadDim;
constexpr std::uint64_t kCapacity = 257;
constexpr std::uint64_t kKvElems =
    16ull * 2ull * kKvHeads * kCapacity * kDim;

std::size_t kv_index(std::uint32_t component, std::uint32_t head,
                     std::uint64_t token, std::uint32_t dim) {
  auto const head_stride = kCapacity * kDim;
  auto const component_stride = kKvHeads * head_stride;
  return component * component_stride + head * head_stride + token * kDim + dim;
}

float sigmoid_for_test(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  auto const e = std::exp(x);
  return e / (1.0f + e);
}

}  // namespace

int main() {
  using qw38::cuda::DeviceBuffer;
  using qw38::cuda::Stream;
  auto stream = Stream::create();
  auto q = DeviceBuffer::allocate(kHeads * kDim * 2ull);
  auto g = DeviceBuffer::allocate(kHeads * kDim * 2ull);
  auto kv = DeviceBuffer::allocate(kKvElems * 2ull);
  auto partials = DeviceBuffer::allocate(
      static_cast<std::uint64_t>(kHeads) * 2ull *
      qw38::cuda::kAttnPartialStride * sizeof(float));
  auto y = DeviceBuffer::allocate(kHeads * kDim * 2ull);
  if (!stream || !q || !g || !kv || !partials || !y) return 1;

  std::vector<std::uint16_t> q_h(kHeads * kDim, 0);
  std::vector<std::uint16_t> g_h(kHeads * kDim, 0);
  std::vector<std::uint16_t> kv_h(kKvElems, 0);
  // Zero Q/K gives uniform stable-softmax weights. Distinct KV-head values
  // make the six-query-head GQA mapping observable.
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    g_h[h * kDim] = qw38::format::fp32_to_bf16_rne(80.0f);
    g_h[h * kDim + (kDim - 1u)] =
        qw38::format::fp32_to_bf16_rne(-80.0f);
  }
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    for (std::uint64_t t = 0; t < kCapacity; ++t) {
      float const value = 0.01f * static_cast<float>(h + 1u) *
                          static_cast<float>(t + 1u);
      auto const bits = qw38::format::fp32_to_bf16_rne(value);
      for (std::uint32_t d = 0; d < kDim; ++d) {
        kv_h[kv_index(1, h, t, d)] = bits;
      }
    }
  }
  auto upload = [&](DeviceBuffer& d, std::span<std::uint16_t const> h) {
    return static_cast<bool>(qw38::cuda::copy_h2d(
        d.data(), std::as_bytes(h), *stream));
  };
  auto const negative_gate_bf16 = qw38::format::fp32_to_bf16_rne(
      sigmoid_for_test(-80.0f));
  if (negative_gate_bf16 == 0) {
    std::cerr << "sigmoid(-80) unexpectedly rounded to BF16 zero\n";
    return 1;
  }
  if (!upload(*q, q_h) || !upload(*g, g_h) || !upload(*kv, kv_h)) return 1;

  for (std::uint64_t length : {0ull, 1ull, 31ull, 32ull, 255ull, 256ull,
                               257ull}) {
    auto const nseg = static_cast<std::uint32_t>(
        length == 0 ? 0 : (length + 255u) / 256u);
    if (!qw38::cuda::zero(*partials, *stream) ||
        !qw38::cuda::zero(*y, *stream)) {
      return 1;
    }
    if (auto st = qw38::cuda::launch_attention_scan(
            static_cast<std::uint16_t const*>(q->data()),
            static_cast<std::uint16_t const*>(kv->data()), 0, kCapacity, length,
            static_cast<float*>(partials->data()), nseg, *stream);
        !st) {
      std::cerr << qw38::cuda::error_message(st.error()) << '\n';
      return 1;
    }
    if (auto st = qw38::cuda::launch_attention_merge(
            static_cast<float const*>(partials->data()),
            static_cast<std::uint16_t const*>(g->data()), nseg,
            static_cast<std::uint16_t*>(y->data()), *stream);
        !st) {
      std::cerr << qw38::cuda::error_message(st.error()) << '\n';
      return 1;
    }
    if (!stream->sync()) return 1;
    std::vector<std::uint16_t> got(kHeads * kDim);
    if (!qw38::cuda::copy_d2h(std::as_writable_bytes(std::span(got)), y->data(),
                              *stream) ||
        !stream->sync()) {
      return 1;
    }
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      auto const kv_h_id = h / 6u;
      float mean = 0.0f;
      for (std::uint64_t t = 0; t < length; ++t) {
        auto const bits = kv_h[kv_index(1, kv_h_id, t, 0)];
        mean += qw38::format::bf16_to_fp32(bits);
      }
      if (length != 0) mean /= static_cast<float>(length);
      for (std::uint32_t d = 0; d < kDim; ++d) {
        float const gate = d == 0 ? 80.0f : (d == kDim - 1u ? -80.0f : 0.0f);
        auto const expected = qw38::format::fp32_to_bf16_rne(
            mean * sigmoid_for_test(gate));
        if (got[h * kDim + d] != expected) {
          std::cerr << "attention core mismatch length=" << length
                    << " q_head=" << h << " dim=" << d << " got="
                    << got[h * kDim + d] << " expected=" << expected << '\n';
          return 1;
        }
      }
    }
  }

  // Use nonuniform, high-dynamic-range logits across both segments.  The
  // highest score arrives in the second segment, so this exercises online
  // rescaling in the scan and fixed-order rescaling in the merge rather than
  // only the uniform-score path above.
  constexpr std::uint64_t extreme_length = 257;
  constexpr std::uint32_t extreme_segments = 2;
  std::vector<std::uint16_t> q_extreme(kHeads * kDim, 0);
  std::vector<std::uint16_t> g_extreme(kHeads * kDim, 0);
  std::vector<std::uint16_t> kv_extreme(kKvElems, 0);
  std::vector<float> logits(extreme_length);
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    q_extreme[h * kDim] = qw38::format::fp32_to_bf16_rne(16.0f);
    g_extreme[h * kDim] = qw38::format::fp32_to_bf16_rne(80.0f);
    g_extreme[h * kDim + (kDim - 1u)] =
        qw38::format::fp32_to_bf16_rne(-80.0f);
  }
  for (std::uint64_t t = 0; t < extreme_length; ++t) {
    float score = -60.0f + static_cast<float>((t % 7u) * 4u);
    if (t == 0) score = -120.0f;
    if (t == 31) score = 70.0f;
    if (t == 63) score = -80.0f;
    if (t == 127) score = 100.0f;
    if (t == 255) score = -100.0f;
    if (t == 256) score = 120.0f;
    logits[t] = score;
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      kv_extreme[kv_index(0, h, t, 0)] =
          qw38::format::fp32_to_bf16_rne(score);
      float const base = 0.5f + 0.25f * static_cast<float>(h) +
                         0.01f * static_cast<float>(t);
      for (std::uint32_t d = 0; d < kDim; ++d) {
        kv_extreme[kv_index(1, h, t, d)] =
            qw38::format::fp32_to_bf16_rne(
                base + 0.001f * static_cast<float>(d));
      }
    }
  }
  if (!upload(*q, q_extreme) || !upload(*g, g_extreme) ||
      !upload(*kv, kv_extreme) || !qw38::cuda::zero(*partials, *stream) ||
      !qw38::cuda::zero(*y, *stream)) {
    return 1;
  }
  if (auto st = qw38::cuda::launch_attention_scan(
          static_cast<std::uint16_t const*>(q->data()),
          static_cast<std::uint16_t const*>(kv->data()), 0, kCapacity,
          extreme_length, static_cast<float*>(partials->data()),
          extreme_segments, *stream);
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  if (auto st = qw38::cuda::launch_attention_merge(
          static_cast<float const*>(partials->data()),
          static_cast<std::uint16_t const*>(g->data()), extreme_segments,
          static_cast<std::uint16_t*>(y->data()), *stream);
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  if (!stream->sync()) return 1;
  std::vector<std::uint16_t> extreme_got(kHeads * kDim);
  std::vector<float> extreme_partials(
      static_cast<std::size_t>(kHeads) * extreme_segments *
      qw38::cuda::kAttnPartialStride);
  if (!qw38::cuda::copy_d2h(
          std::as_writable_bytes(std::span(extreme_got)), y->data(), *stream) ||
      !qw38::cuda::copy_d2h(std::as_writable_bytes(std::span(extreme_partials)),
                            partials->data(), *stream) ||
      !stream->sync()) {
    return 1;
  }
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    auto const kv_h_id = h / 6u;
    float score_min = logits[0];
    float score_max = logits[0];
    for (float const score : logits) {
      score_min = std::fmin(score_min, score);
      score_max = std::fmax(score_max, score);
    }
    if (score_max - score_min < 200.0f) {
      std::cerr << "extreme logits fixture lost dynamic range\n";
      return 1;
    }
    for (std::uint32_t segment = 0; segment < extreme_segments; ++segment) {
      auto const off = (static_cast<std::size_t>(h) * extreme_segments +
                        segment) * qw38::cuda::kAttnPartialStride;
      if (!std::isfinite(extreme_partials[off]) ||
          !std::isfinite(extreme_partials[off + 1u]) ||
          extreme_partials[off + 1u] <= 0.0f) {
        std::cerr << "extreme logits produced invalid online partial\n";
        return 1;
      }
    }
    for (std::uint32_t d = 0; d < kDim; ++d) {
      double weighted = 0.0;
      double denom = 0.0;
      double max_score = static_cast<double>(logits[0]);
      for (float const score : logits) {
        max_score = std::max(max_score, static_cast<double>(score));
      }
      for (std::uint64_t t = 0; t < extreme_length; ++t) {
        double const weight = std::exp(static_cast<double>(logits[t]) - max_score);
        denom += weight;
        weighted += weight * static_cast<double>(qw38::format::bf16_to_fp32(
            kv_extreme[kv_index(1, kv_h_id, t, d)]));
      }
      weighted /= denom;
      float const gate = d == 0 ? 80.0f : (d == kDim - 1u ? -80.0f : 0.0f);
      auto const expected = qw38::format::fp32_to_bf16_rne(
          static_cast<float>(weighted) * sigmoid_for_test(gate));
      if (extreme_got[h * kDim + d] != expected) {
        std::cerr << "extreme logits mismatch q_head=" << h
                  << " dim=" << d << " got=" << extreme_got[h * kDim + d]
                  << " expected=" << expected << '\n';
        return 1;
      }
    }
    if (extreme_got[h * kDim + (kDim - 1u)] == 0) {
      std::cerr << "negative extreme gate incorrectly rounded to zero\n";
      return 1;
    }
  }
  std::cout << "attention core unit ok\n";
  return 0;
}
