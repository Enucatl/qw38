#pragma once

#include "activation_support.hpp"
#include "compiler/quantization/quantizer.hpp"
#include "compiler/quantization/reference.hpp"
#include "cuda/alloc.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/stream.hpp"
#include "format/constants.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"
#include "format/unpack.hpp"
#include "reference/math.hpp"
#include "runtime/error.hpp"
#include "runtime/mlp.hpp"
#include "runtime/view.hpp"

#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qw38::mlp::test {

using qw38::activation::test::almost_equal;
using qw38::activation::test::as_bytes;
using qw38::activation::test::bf16;
using qw38::activation::test::download_vec;
using qw38::activation::test::expect;
using qw38::activation::test::f32;
using qw38::activation::test::fail;
using qw38::activation::test::g_failures;
using qw38::activation::test::upload_vec;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::format::LogicalQuantizerId;
using qw38::format::PackedMatrix;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::format::fp32_to_bf16_rne;
using qw38::reference::kFfn;
using qw38::reference::kHidden;
using qw38::runtime::MlpBindViews;
using qw38::runtime::MlpPlan;
using qw38::runtime::TensorView;
using qw38::runtime::MemorySpace;

inline constexpr float kNormAbs = qw38::reference::tol::kMlpNormAbs;
inline constexpr float kSwigluAbs = qw38::reference::tol::kMlpSwigluAbs;
inline constexpr float kSwigluRel = qw38::reference::tol::kMlpSwigluRel;
inline constexpr float kResAbs = qw38::reference::tol::kMlpResidualAbs;
inline constexpr float kResRel = qw38::reference::tol::kMlpResidualRel;

inline TensorView make_view(void* ptr, qw38::format::ArithmeticDtype dtype,
                            PhysicalLayoutId layout, StorageClass storage,
                            bool writable, std::uint8_t rank, std::uint64_t e0,
                            std::uint64_t e1 = 0) {
  TensorView v{};
  v.pointer = ptr;
  v.dtype = dtype;
  v.layout = layout;
  v.storage = storage;
  v.space = MemorySpace::Device;
  v.writable = writable;
  v.rank = rank;
  v.extent[0] = e0;
  v.extent[1] = e1;
  return v;
}

inline TensorView weight_view(DeviceBuffer& buf, PhysicalLayoutId layout,
                              StorageClass storage, std::uint32_t n,
                              std::uint32_t k) {
  return make_view(buf.data(), qw38::format::ArithmeticDtype::Bf16, layout,
                   storage, false, 2, n, k);
}

inline TensorView scale_view(DeviceBuffer& buf, PhysicalLayoutId layout,
                             StorageClass storage, std::uint64_t n_scales) {
  return make_view(buf.data(), qw38::format::ArithmeticDtype::Fp16, layout,
                   storage, false, 1, n_scales);
}

inline TensorView vec_view(DeviceBuffer& buf, qw38::format::ArithmeticDtype dtype,
                           PhysicalLayoutId layout, StorageClass storage,
                           bool writable, std::uint64_t n) {
  return make_view(buf.data(), dtype, layout, storage, writable, 1, n);
}

inline std::vector<std::uint16_t> bf16_vec(std::uint32_t n, float seed) {
  std::vector<std::uint16_t> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = fp32_to_bf16_rne(seed * (static_cast<float>(i % 7) - 3.0f) / 3.0f);
  }
  return out;
}

inline std::vector<float> residual_vec(std::uint32_t n, float seed) {
  std::vector<float> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = seed * (static_cast<float>(static_cast<int>(i % 11) - 5) / 5.0f);
  }
  return out;
}

inline std::vector<std::byte> bf16_matrix(std::uint32_t n, std::uint32_t k,
                                          float scale, std::uint32_t seed) {
  std::vector<std::byte> out(static_cast<std::size_t>(n) * k * 2u);
  for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(n) * k; ++i) {
    float const v =
        scale * (static_cast<float>(static_cast<int>((i + seed) % 13) - 6) / 7.0f);
    qw38::format::store_u16_le(out.data() + i * 2, fp32_to_bf16_rne(v));
  }
  return out;
}

inline std::vector<std::uint16_t> bytes_to_u16(std::span<std::byte const> bytes) {
  std::vector<std::uint16_t> out(bytes.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = qw38::format::load_u16_le(bytes.data() + i * 2);
  }
  return out;
}

inline qw38::format::LogicalWeightCodes make_logical(LogicalQuantizerId qid,
                                                     std::uint32_t n,
                                                     std::uint32_t k,
                                                     std::int8_t code,
                                                     std::uint16_t scale_bits) {
  qw38::format::LogicalWeightCodes logical;
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

inline void fill_logical_pattern(qw38::format::LogicalWeightCodes& logical,
                                 std::int8_t seed) {
  for (std::uint64_t i = 0; i < logical.codes.size(); ++i) {
    int const v = static_cast<int>((i + static_cast<unsigned>(seed)) % 15) - 7;
    logical.codes[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(v);
  }
  for (std::uint64_t i = 0; i < logical.scales.size(); ++i) {
    logical.scales[static_cast<std::size_t>(i)] =
        (i % 11 == 0) ? std::uint16_t{0x3A00} : std::uint16_t{0x3800};
  }
}

inline bool fp32_close(float a, float b, float abs_tol, float rel_tol) {
  return almost_equal(a, b, abs_tol, rel_tol);
}

inline void expect_fp32_close(std::span<float const> got, std::span<float const> ref,
                              std::string_view tag, float abs_tol, float rel_tol) {
  expect(got.size() == ref.size(), std::string(tag) + " size");
  if (got.size() != ref.size()) {
    return;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!fp32_close(got[i], ref[i], abs_tol, rel_tol)) {
      fail(std::string(tag) + " idx " + std::to_string(i) + " got " +
           std::to_string(got[i]) + " ref " + std::to_string(ref[i]));
      return;
    }
  }
}

inline void expect_bf16_close(std::span<std::uint16_t const> got,
                              std::span<std::uint16_t const> ref,
                              std::string_view tag, float abs_tol, float rel_tol) {
  expect(got.size() == ref.size(), std::string(tag) + " size");
  if (got.size() != ref.size()) {
    return;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    float const fa = f32(got[i]);
    float const fb = f32(ref[i]);
    if (!fp32_close(fa, fb, abs_tol, rel_tol)) {
      fail(std::string(tag) + " idx " + std::to_string(i) + " got " +
           std::to_string(fa) + " ref " + std::to_string(fb));
      return;
    }
  }
}

inline std::expected<std::vector<std::uint16_t>, std::string> decoded_weights(
    PackedMatrix const& packed) {
  if (packed.quantizer == LogicalQuantizerId::None) {
    auto w = qw38::compiler::decode_bf16_payload(packed.layout, packed.codes,
                                                 packed.logical_n, packed.logical_k);
    if (!w) {
      return std::unexpected(qw38::compiler::error_message(w.error()));
    }
    return *w;
  }
  auto logical = qw38::format::unpack_cuda_v0(
      packed.quantizer, packed.layout, packed.logical_n, packed.logical_k,
      packed.codes, packed.scales);
  if (!logical) {
    return std::unexpected(qw38::format::error_message(logical.error()));
  }
  auto w = qw38::compiler::dequantize_to_bf16(*logical);
  if (!w) {
    return std::unexpected(qw38::compiler::error_message(w.error()));
  }
  return *w;
}

struct HostMlp {
  PackedMatrix gate;
  PackedMatrix up;
  PackedMatrix down;
  std::vector<std::uint16_t> gamma;
  std::vector<float> h_mid;
  std::vector<std::uint16_t> w_gate;
  std::vector<std::uint16_t> w_up;
  std::vector<std::uint16_t> w_down;
};

inline bool pack_q4_from_logical(qw38::format::LogicalWeightCodes const& logical,
                                 PackedMatrix& out, std::string_view tag) {
  auto packed = qw38::format::pack_cuda_v0(
      LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, logical);
  if (!packed) {
    fail(std::string(tag) + " pack: " + qw38::format::error_message(packed.error()));
    return false;
  }
  out = std::move(*packed);
  return true;
}

struct DeviceMlp {
  DeviceBuffer gate_codes;
  DeviceBuffer gate_scales;
  DeviceBuffer up_codes;
  DeviceBuffer up_scales;
  DeviceBuffer down_codes;
  DeviceBuffer down_scales;
  DeviceBuffer gamma;
  DeviceBuffer h_mid;
  DeviceBuffer next_h;
  DeviceBuffer normalized;
  DeviceBuffer swiglu;
  bool has_scales{true};
  PhysicalLayoutId layout{PhysicalLayoutId::CudaQ4G64V0};
  StorageClass storage{StorageClass::Int4Grouped};
};

inline bool upload_packed(DeviceBuffer& codes, DeviceBuffer& scales,
                          PackedMatrix const& packed, Stream const& stream,
                          std::string_view tag) {
  auto c = upload_vec(packed.codes, stream);
  if (!c) {
    fail(std::string(tag) + " codes upload");
    return false;
  }
  codes = std::move(*c);
  if (!packed.scales.empty()) {
    auto s = upload_vec(packed.scales, stream);
    if (!s) {
      fail(std::string(tag) + " scales upload");
      return false;
    }
    scales = std::move(*s);
  }
  return true;
}

inline bool upload_host_mlp(HostMlp const& host, DeviceMlp& dev, Stream const& stream) {
  if (!upload_packed(dev.gate_codes, dev.gate_scales, host.gate, stream, "gate") ||
      !upload_packed(dev.up_codes, dev.up_scales, host.up, stream, "up") ||
      !upload_packed(dev.down_codes, dev.down_scales, host.down, stream, "down")) {
    return false;
  }
  auto g = upload_vec(host.gamma, stream);
  auto h = upload_vec(host.h_mid, stream);
  auto next = DeviceBuffer::allocate(kHidden * sizeof(float));
  auto n = DeviceBuffer::allocate(kHidden * 2);
  auto s = DeviceBuffer::allocate(kFfn * 2);
  if (!g || !h || !next || !n || !s) {
    fail("activation upload");
    return false;
  }
  dev.gamma = std::move(*g);
  dev.h_mid = std::move(*h);
  dev.next_h = std::move(*next);
  dev.normalized = std::move(*n);
  dev.swiglu = std::move(*s);
  dev.has_scales = !host.gate.scales.empty();
  dev.layout = host.gate.layout;
  dev.storage = (host.gate.layout == PhysicalLayoutId::CudaQ4G64V0)
                    ? StorageClass::Int4Grouped
                    : StorageClass::Bf16;
  return true;
}

inline MlpBindViews bind_views(DeviceMlp& dev) {
  MlpBindViews v;
  v.gate = weight_view(dev.gate_codes, dev.layout, dev.storage, kFfn, kHidden);
  v.up = weight_view(dev.up_codes, dev.layout, dev.storage, kFfn, kHidden);
  v.down = weight_view(dev.down_codes, dev.layout, dev.storage, kHidden, kFfn);
  if (dev.has_scales) {
    auto const g_scales = dev.gate_scales.bytes() / 2;
    auto const u_scales = dev.up_scales.bytes() / 2;
    auto const d_scales = dev.down_scales.bytes() / 2;
    v.gate_scales = scale_view(dev.gate_scales, dev.layout, dev.storage, g_scales);
    v.up_scales = scale_view(dev.up_scales, dev.layout, dev.storage, u_scales);
    v.down_scales = scale_view(dev.down_scales, dev.layout, dev.storage, d_scales);
  }
  v.gamma = vec_view(dev.gamma, qw38::format::ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                     kHidden);
  v.h_mid = vec_view(dev.h_mid, qw38::format::ArithmeticDtype::Fp32,
                     PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                     true, kHidden);
  v.next_h = vec_view(dev.next_h, qw38::format::ArithmeticDtype::Fp32,
                      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                      true, kHidden);
  v.normalized =
      vec_view(dev.normalized, qw38::format::ArithmeticDtype::Bf16,
               PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true,
               kHidden);
  v.swiglu = vec_view(dev.swiglu, qw38::format::ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true,
                      kFfn);
  return v;
}

inline bool all_finite(std::span<float const> x, std::string_view tag) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!(x[i] == x[i]) || x[i] > 1.0e30f || x[i] < -1.0e30f) {
      fail(std::string(tag) + " nonfinite idx " + std::to_string(i));
      return false;
    }
  }
  return true;
}

}  // namespace qw38::mlp::test
