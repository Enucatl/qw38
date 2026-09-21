#pragma once

#include "activation_support.hpp"
#include "compiler/quantization/quantizer.hpp"
#include "compiler/quantization/reference.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/stream.hpp"
#include "format/constants.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qw38::decode_mmv::test {

using qw38::activation::test::almost_equal;
using qw38::activation::test::as_bytes;
using qw38::activation::test::bf16;
using qw38::activation::test::download_vec;
using qw38::activation::test::expect;
using qw38::activation::test::f32;
using qw38::activation::test::fail;
using qw38::activation::test::g_failures;
using qw38::activation::test::upload_vec;
using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_pad_k;
using qw38::cuda::decode_pad_n;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeLayoutQ8G32V0;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ8G32V0;
using qw38::format::LogicalQuantizerId;
using qw38::format::LogicalWeightCodes;
using qw38::format::PackedMatrix;
using qw38::format::PhysicalLayoutId;
using qw38::format::fp32_to_bf16_rne;
using qw38::format::store_u16_le;

inline constexpr float kAbs = qw38::cuda::decode_mmv_tol::kFp32Abs;
inline constexpr float kRel = qw38::cuda::decode_mmv_tol::kFp32Rel;
inline constexpr float kBf16Abs = qw38::cuda::decode_mmv_tol::kBf16StoreAbs;

static_assert(static_cast<std::uint16_t>(PhysicalLayoutId::CudaQ4G64V0) ==
              kDecodeLayoutQ4G64V0);
static_assert(static_cast<std::uint16_t>(PhysicalLayoutId::CudaQ8G32V0) ==
              kDecodeLayoutQ8G32V0);
static_assert(static_cast<std::uint16_t>(PhysicalLayoutId::CudaBf16DenseTileV0) ==
              kDecodeLayoutBf16DenseTileV0);
static_assert(static_cast<std::uint16_t>(LogicalQuantizerId::Q4G64V0) ==
              kDecodeQuantizerQ4G64V0);
static_assert(static_cast<std::uint16_t>(LogicalQuantizerId::Q8G32V0) ==
              kDecodeQuantizerQ8G32V0);
static_assert(static_cast<std::uint16_t>(LogicalQuantizerId::None) ==
              kDecodeQuantizerNone);

inline std::vector<std::byte> bf16_matrix(std::uint32_t n, std::uint32_t k,
                                          float scale, std::uint32_t seed = 1) {
  std::vector<std::byte> out(static_cast<std::size_t>(n) * k * 2u);
  for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(n) * k; ++i) {
    float const v = scale * (static_cast<float>(static_cast<int>((i + seed) % 13) - 6) /
                             7.0f);
    store_u16_le(out.data() + i * 2, fp32_to_bf16_rne(v));
  }
  return out;
}

inline std::vector<std::uint16_t> bf16_vec(std::uint32_t k, float seed) {
  std::vector<std::uint16_t> out(k);
  for (std::uint32_t i = 0; i < k; ++i) {
    out[i] = fp32_to_bf16_rne(seed * (static_cast<float>(i % 7) - 3.0f) / 3.0f);
  }
  return out;
}

inline std::vector<float> residual_vec(std::uint32_t n, float seed) {
  std::vector<float> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = seed * (static_cast<float>(static_cast<int>(i % 5) - 2) / 2.0f);
  }
  return out;
}

inline DecodeMmvDesc desc_from_packed(PackedMatrix const& packed,
                                      DecodeEpilogue epi) {
  DecodeMmvDesc d;
  d.layout = static_cast<std::uint16_t>(packed.layout);
  d.quantizer = static_cast<std::uint16_t>(packed.quantizer);
  d.n = static_cast<std::uint32_t>(packed.logical_n);
  d.k = static_cast<std::uint32_t>(packed.logical_k);
  d.padded_n = static_cast<std::uint32_t>(packed.padded_n);
  d.padded_k = static_cast<std::uint32_t>(packed.padded_k);
  d.codes_bytes = packed.codes.size();
  d.scales_bytes = packed.scales.size();
  d.epilogue = epi;
  return d;
}

inline bool fp32_close(float a, float b, float abs_tol, float rel_tol) {
  return almost_equal(a, b, abs_tol, rel_tol);
}

inline void expect_fp32_close(std::span<float const> got, std::span<float const> ref,
                              std::string_view tag, float abs_tol = kAbs,
                              float rel_tol = kRel) {
  expect(got.size() == ref.size(), std::string(tag) + " size");
  if (got.size() != ref.size()) {
    return;
  }
  float max_d = 0.0f;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    float const d = got[i] > ref[i] ? got[i] - ref[i] : ref[i] - got[i];
    if (d > max_d) {
      max_d = d;
      worst = i;
    }
    if (!fp32_close(got[i], ref[i], abs_tol, rel_tol)) {
      fail(std::string(tag) + " idx " + std::to_string(i) + " got " +
           std::to_string(got[i]) + " ref " + std::to_string(ref[i]) + " d=" +
           std::to_string(d));
      return;
    }
  }
  (void)worst;
}

inline void expect_bf16_close(std::span<std::uint16_t const> got,
                              std::span<float const> ref_fp32, std::string_view tag) {
  expect(got.size() == ref_fp32.size(), std::string(tag) + " size");
  if (got.size() != ref_fp32.size()) {
    return;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    float const fa = f32(got[i]);
    float const fb = f32(fp32_to_bf16_rne(ref_fp32[i]));
    if (!fp32_close(fa, fb, kBf16Abs, kRel)) {
      fail(std::string(tag) + " idx " + std::to_string(i) + " got " +
           std::to_string(fa) + " ref " + std::to_string(fb));
      return;
    }
  }
}

inline LogicalWeightCodes make_logical(LogicalQuantizerId qid, std::uint32_t n,
                                       std::uint32_t k, std::int8_t code,
                                       std::uint16_t scale_bits) {
  LogicalWeightCodes logical;
  logical.quantizer = qid;
  logical.n = n;
  logical.k = k;
  logical.group_size = (qid == LogicalQuantizerId::Q4G64V0) ? 64u : 32u;
  logical.qmax = (qid == LogicalQuantizerId::Q4G64V0) ? 7 : 127;
  logical.codes.assign(static_cast<std::size_t>(n) * k, code);
  logical.scales.assign(static_cast<std::size_t>(n) * (k / logical.group_size),
                        scale_bits);
  return logical;
}

}  // namespace qw38::decode_mmv::test
