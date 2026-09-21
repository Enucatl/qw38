#pragma once

#include "mlp_support.hpp"
#include "runtime/gdn.hpp"

#include "cuda/gdn.hpp"

namespace qw38::gdn::test {

using qw38::mlp::test::DeviceBuffer;
using qw38::mlp::test::Stream;
using qw38::mlp::test::all_finite;
using qw38::mlp::test::as_bytes;
using qw38::mlp::test::bf16;
using qw38::mlp::test::bf16_vec;
using qw38::mlp::test::download_vec;
using qw38::mlp::test::expect;
using qw38::mlp::test::expect_bf16_close;
using qw38::mlp::test::expect_fp32_close;
using qw38::mlp::test::f32;
using qw38::mlp::test::fail;
using qw38::mlp::test::fill_logical_pattern;
using qw38::mlp::test::g_failures;
using qw38::mlp::test::make_logical;
using qw38::mlp::test::make_view;
using qw38::mlp::test::pack_q4_from_logical;
using qw38::mlp::test::residual_vec;
using qw38::mlp::test::upload_vec;
using qw38::mlp::test::vec_view;
using qw38::mlp::test::weight_view;
using qw38::format::ArithmeticDtype;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::format::fp32_to_bf16_rne;
using qw38::reference::kConvHistoryTaps;
using qw38::reference::kConvKernel;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kGdnHeadDim;
using qw38::reference::kGdnKeyHeads;
using qw38::reference::kGdnValueHeads;
using qw38::reference::kGdnZWidth;
using qw38::reference::kHidden;
using qw38::reference::kQkvWidth;
using qw38::reference::tol::kGdnConvBf16Abs;
using qw38::reference::tol::kGdnGateFp32Abs;
using qw38::reference::tol::kGdnGateFp32Rel;
using qw38::reference::tol::kGdnQkFp32Abs;
using qw38::reference::tol::kGdnQkFp32Rel;
using qw38::runtime::GdnFrontBindViews;
using qw38::runtime::MemorySpace;
using qw38::runtime::TensorView;

inline std::vector<std::uint16_t> filled_h(std::uint32_t n, float v) {
  return std::vector<std::uint16_t>(n, fp32_to_bf16_rne(v));
}

inline std::vector<std::uint16_t> zeros_h(std::uint32_t n) {
  return filled_h(n, 0.0f);
}

inline std::vector<std::uint16_t> pattern_h(std::uint32_t n, float seed) {
  std::vector<std::uint16_t> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = fp32_to_bf16_rne(
        seed * (static_cast<float>(static_cast<int>(i % 9) - 4) / 4.0f));
  }
  return out;
}

inline std::vector<std::byte> bf16_bytes(std::span<std::uint16_t const> src) {
  return as_bytes(src);
}

inline bool pack_bf16_tile(std::span<std::uint16_t const> row_major, std::uint32_t n,
                           std::uint32_t k, qw38::format::PackedMatrix& out,
                           std::string_view tag) {
  auto packed = qw38::format::pack_bf16_dense_tile_v0(bf16_bytes(row_major), n, k);
  if (!packed) {
    fail(std::string(tag) + " bf16 pack: " +
         qw38::format::error_message(packed.error()));
    return false;
  }
  out = std::move(*packed);
  return true;
}

}  // namespace qw38::gdn::test
