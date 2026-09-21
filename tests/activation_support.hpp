#pragma once

#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/error.hpp"
#include "cuda/stream.hpp"
#include "format/floatcvt.hpp"

#include <cstdint>
#include <expected>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qw38::activation::test {

inline int g_failures = 0;

inline void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

inline void expect(bool cond, std::string_view what) {
  if (!cond) {
    fail(what);
  }
}

template <class T>
std::vector<std::byte> as_bytes(std::span<T const> src) {
  auto const p = reinterpret_cast<std::byte const*>(src.data());
  return std::vector<std::byte>(p, p + src.size_bytes());
}

template <class T>
std::expected<qw38::cuda::DeviceBuffer, qw38::cuda::Error> upload_vec(
    std::vector<T> const& host, qw38::cuda::Stream const& stream) {
  auto bytes = as_bytes(std::span<T const>(host));
  auto buf = qw38::cuda::DeviceBuffer::allocate(bytes.size());
  if (!buf) {
    return std::unexpected(buf.error());
  }
  auto st = qw38::cuda::copy_h2d(buf->data(), bytes.data(), bytes.size(), stream);
  if (!st) {
    return std::unexpected(st.error());
  }
  return buf;
}

template <class T>
std::expected<std::vector<T>, qw38::cuda::Error> download_vec(
    qw38::cuda::DeviceBuffer const& buf, std::size_t count,
    qw38::cuda::Stream const& stream) {
  std::vector<T> host(count);
  auto st = qw38::cuda::copy_d2h(host.data(), buf.data(),
                                 count * sizeof(T), stream);
  if (!st) {
    return std::unexpected(st.error());
  }
  auto sync = stream.sync();
  if (!sync) {
    return std::unexpected(sync.error());
  }
  return host;
}

inline std::uint16_t bf16(float x) {
  return qw38::format::fp32_to_bf16_rne(x);
}

inline float f32(std::uint16_t h) {
  return qw38::format::bf16_to_fp32(h);
}

inline bool almost_equal(float a, float b, float abs_tol, float rel_tol = 0.0f) {
  float const d = a > b ? a - b : b - a;
  float const scale = (a > 0 ? a : -a) > (b > 0 ? b : -b)
                          ? (a > 0 ? a : -a)
                          : (b > 0 ? b : -b);
  return d <= abs_tol || d <= rel_tol * (scale + 1.0e-12f);
}

inline float max_abs_diff(std::span<float const> a, std::span<float const> b) {
  float m = 0.0f;
  auto n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) {
    float d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
    if (d > m) {
      m = d;
    }
  }
  return m;
}

inline float max_abs_diff_bf16(std::span<std::uint16_t const> a,
                               std::span<std::uint16_t const> b) {
  float m = 0.0f;
  auto n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) {
    float fa = f32(a[i]);
    float fb = f32(b[i]);
    float d = fa > fb ? fa - fb : fb - fa;
    if (d > m) {
      m = d;
    }
  }
  return m;
}

}  // namespace qw38::activation::test
