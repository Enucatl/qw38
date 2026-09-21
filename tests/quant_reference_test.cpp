#include "compiler/compiler.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"
#include "format/unpack.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

using qw38::compiler::dequantize_to_bf16;
using qw38::compiler::quantize_bf16;
using qw38::compiler::quantize_fp32;
using qw38::compiler::reference_gemv_bf16;
using qw38::compiler::reference_gemv_packed;
using qw38::format::bf16_to_fp32;
using qw38::format::fp32_to_bf16_rne;
using qw38::format::LogicalQuantizerId;
using qw38::format::pack_bf16_dense_tile_v0;
using qw38::format::pack_cuda_v0;
using qw38::format::PhysicalLayoutId;
using qw38::format::store_u16_le;
using qw38::format::unpack_bf16_dense_tile_v0;
using qw38::format::unpack_cuda_v0;

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

std::vector<std::byte> bf16_matrix(std::uint64_t n, std::uint64_t k,
                                   float scale) {
  std::vector<std::byte> out(static_cast<std::size_t>(n * k * 2));
  for (std::uint64_t i = 0; i < n * k; ++i) {
    float const v =
        scale * (static_cast<float>(static_cast<int>(i % 13) - 6) / 7.0f);
    store_u16_le(out.data() + i * 2, fp32_to_bf16_rne(v));
  }
  return out;
}

std::vector<std::uint16_t> bf16_vec(std::uint64_t k, float seed) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(k));
  for (std::uint64_t i = 0; i < k; ++i) {
    out[i] = fp32_to_bf16_rne(seed * (static_cast<float>(i % 5) - 2.0f));
  }
  return out;
}

void check_roundtrip(LogicalQuantizerId qid, PhysicalLayoutId layout,
                     std::uint64_t n, std::uint64_t k, float scale,
                     std::string_view tag) {
  auto src = bf16_matrix(n, k, scale);
  auto logical = quantize_bf16(qid, n, k, src);
  if (!logical) {
    fail(std::string(tag) + " quantize: " +
         qw38::compiler::error_message(logical.error()));
    return;
  }
  auto packed = pack_cuda_v0(qid, layout, *logical);
  if (!packed) {
    fail(std::string(tag) + " pack: " +
         qw38::format::error_message(packed.error()));
    return;
  }
  auto unpacked = unpack_cuda_v0(qid, layout, n, k, packed->codes, packed->scales);
  if (!unpacked) {
    fail(std::string(tag) + " unpack: " +
         qw38::format::error_message(unpacked.error()));
    return;
  }
  expect(unpacked->codes == logical->codes, std::string(tag) + " codes agree");
  expect(unpacked->scales == logical->scales, std::string(tag) + " scales agree");
  expect(unpacked->quantizer == qid && unpacked->n == n && unpacked->k == k,
         std::string(tag) + " geometry");

  auto dq = dequantize_to_bf16(*logical);
  auto x = bf16_vec(k, 0.25f);
  if (!dq) {
    fail(std::string(tag) + " dequant");
    return;
  }
  auto from_explicit = reference_gemv_bf16(*dq, x, n, k);
  auto from_packed = reference_gemv_packed(*packed, x);
  if (!from_explicit || !from_packed) {
    fail(std::string(tag) + " gemv");
    return;
  }
  expect(*from_explicit == *from_packed,
         std::string(tag) + " contraction matches explicit BF16 GEMV");
}

}  // namespace

int main() {
  check_roundtrip(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 8,
                  256, 1.0f, "q4-8x256");
  check_roundtrip(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 16,
                  512, 0.35f, "q4-16x512");
  check_roundtrip(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 24,
                  256, 2.5f, "q4-24x256");
  check_roundtrip(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0, 8,
                  256, 4.0f, "q8-8x256");
  check_roundtrip(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0, 16,
                  512, 0.9f, "q8-16x512");

  {
    auto src = bf16_matrix(8, 256, 1.25f);
    auto packed = pack_bf16_dense_tile_v0(src, 8, 256);
    if (!packed) {
      fail(qw38::format::error_message(packed.error()));
    } else {
      auto back = unpack_bf16_dense_tile_v0(packed->codes, 8, 256);
      expect(static_cast<bool>(back) && *back == src, "BF16 tile passthrough");
      auto x = bf16_vec(256, 0.5f);
      std::vector<std::uint16_t> w(8 * 256);
      for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = qw38::format::load_u16_le(src.data() + i * 2);
      }
      auto a = reference_gemv_bf16(w, x, 8, 256);
      auto b = reference_gemv_packed(*packed, x);
      expect(static_cast<bool>(a) && static_cast<bool>(b) && *a == *b,
             "BF16 control contraction");
    }
  }

  {
    std::vector<float> w(8 * 256, 0.0f);
    w[0] = 3.0f;
    auto q = quantize_fp32(LogicalQuantizerId::Q4G64V0, 8, 256, w);
    if (!q) {
      fail(qw38::compiler::error_message(q.error()));
    } else {
      auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                 PhysicalLayoutId::CudaQ4G64V0, *q);
      if (!packed) {
        fail(qw38::format::error_message(packed.error()));
      } else {
        auto unpacked = unpack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                       PhysicalLayoutId::CudaQ8G32V0, 8, 256,
                                       packed->codes, packed->scales);
        expect(!unpacked, "unpack rejects mismatched layout id");
        packed->codes[0] = std::byte{0x08};
        auto bad = unpack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                  PhysicalLayoutId::CudaQ4G64V0, 8, 256,
                                  packed->codes, packed->scales);
        expect(!bad, "unpack rejects forbidden Q4 nibble -8");
      }
    }
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
