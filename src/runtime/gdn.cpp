#include "runtime/gdn.hpp"

#include "cuda/activation.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/event.hpp"
#include "cuda/gdn.hpp"

#include "format/constants.hpp"
#include "format/layout.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qw38::runtime {
namespace {

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeDtype;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DecodeMmvRangeDesc;
using qw38::cuda::decode_matrix_view;
using qw38::cuda::decode_vector_view;
using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_pad_k;
using qw38::cuda::decode_pad_n;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;

Error arg_error(std::string_view field, std::string_view detail) {
  return make_error(ErrorCode::InvalidArgument, field, detail);
}

template <typename Pointer>
std::expected<std::uint64_t, Error> element_count(
    BasicTensorView<Pointer> const& v, std::string_view field) {
  if (v.rank == 0 || v.rank > qw38::format::kMaxRank) {
    return std::unexpected(arg_error(field, "rank is invalid"));
  }
  std::uint64_t n = 1;
  for (std::uint8_t i = 0; i < v.rank; ++i) {
    auto product = qw38::format::checked_mul(n, v.extent[i], 0, field);
    if (!product) {
      return std::unexpected(make_error(ErrorCode::Overflow, product.error().field,
                                        product.error().detail));
    }
    n = *product;
  }
  return n;
}

template <typename Pointer>
std::expected<std::uint64_t, Error> view_bytes(
    BasicTensorView<Pointer> const& v, std::string_view field) {
  auto elements = element_count(v, field);
  if (!elements) {
    return std::unexpected(elements.error());
  }
  auto bytes = qw38::format::checked_mul(
      *elements, qw38::format::element_size(v.dtype), 0, field);
  if (!bytes) {
    return std::unexpected(make_error(ErrorCode::Overflow, bytes.error().field,
                                      bytes.error().detail));
  }
  return *bytes;
}

template <typename Pointer>
std::expected<BasicTensorView<Pointer>, Error> require_view(
    BasicTensorView<Pointer> v, ArithmeticDtype dtype, PhysicalLayoutId layout,
    StorageClass storage, std::initializer_list<std::uint64_t> extents,
    std::string_view field) {
  if (v.pointer == nullptr || v.space != MemorySpace::Device ||
      v.dtype != dtype || v.layout != layout || v.storage != storage ||
      v.rank != extents.size()) {
    return std::unexpected(arg_error(field, "typed view contract mismatch"));
  }
  std::size_t i = 0;
  for (auto extent : extents) {
    if (v.extent[i++] != extent) {
      return std::unexpected(arg_error(field, "typed view extent mismatch"));
    }
  }
  if (auto bytes = view_bytes(v, field); !bytes) {
    return std::unexpected(bytes.error());
  }
  return v;
}

struct ByteInterval {
  std::string_view name;
  std::uintptr_t begin;
  std::uintptr_t end;
};

std::expected<ByteInterval, Error> byte_interval(
    ConstTensorView const& view, std::uint64_t bytes, std::string_view name) {
  auto const begin = reinterpret_cast<std::uintptr_t>(view.pointer);
  if (view.pointer == nullptr || bytes == 0 ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return std::unexpected(arg_error(name, "byte interval is invalid"));
  }
  return ByteInterval{name, begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(ByteInterval const& a, ByteInterval const& b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

std::expected<void, Error> validate_intervals(
    std::initializer_list<
        std::pair<std::string_view,
                  std::pair<ConstTensorView, std::uint64_t>>> views) {
  std::vector<ByteInterval> intervals;
  intervals.reserve(views.size());
  for (auto const& [name, view_and_bytes] : views) {
    auto const& [view, bytes] = view_and_bytes;
    if (view.pointer == nullptr && bytes == 0) {
      continue;
    }
    auto interval = byte_interval(view, bytes, name);
    if (!interval) {
      return std::unexpected(interval.error());
    }
    for (auto const& prior : intervals) {
      if (overlaps(prior, *interval)) {
        return std::unexpected(arg_error(
            "alias", std::string(prior.name) + " overlaps " +
                         std::string(interval->name)));
      }
    }
    intervals.push_back(*interval);
  }
  return {};
}

ConstTensorView byte_arena_view(WorkspaceView const& workspace) {
  ConstTensorView view{};
  view.pointer = workspace.pointer;
  view.space = workspace.space;
  return view;
}

std::expected<void, Error> validate_gdn_state_view(TensorView const& s) {
  if (s.pointer == nullptr) {
    return std::unexpected(arg_error("s", "null view"));
  }
  if (s.space != MemorySpace::Device) {
    return std::unexpected(arg_error("s", "view must be device memory"));
  }
  if (s.dtype != ArithmeticDtype::Fp32) {
    return std::unexpected(arg_error("s", "S must be FP32"));
  }
  if (s.layout != PhysicalLayoutId::CudaFp32GdnSHvKV0) {
    return std::unexpected(arg_error("s", "S must use cuda_fp32_gdn_s_hvk_v0"));
  }
  if (s.storage != StorageClass::Fp32) {
    return std::unexpected(arg_error("s", "S storage must be fp32"));
  }
  auto elements = element_count(s, "s.extent");
  if (!elements) {
    return std::unexpected(elements.error());
  }
  if (s.rank != 4 || s.extent[0] != kGdnLayers ||
      s.extent[1] != kGdnValueHeads || s.extent[2] != kGdnValueDim ||
      s.extent[3] != kGdnKeyDim ||
      *elements != kGdnSBytes / qw38::format::kFp32Size) {
    return std::unexpected(
        arg_error("s", "S must be FP32 [48,48,128,128] in [layer,value_head,value,key] order"));
  }
  auto bytes = view_bytes(s, "s");
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  if (*bytes != kGdnSBytes) {
    return std::unexpected(arg_error("s", "S byte count mismatch"));
  }
  return {};
}

bool q4_or_bf16_tile(PhysicalLayoutId layout) noexcept {
  return layout == PhysicalLayoutId::CudaQ4G64V0 ||
         layout == PhysicalLayoutId::CudaBf16DenseTileV0;
}

std::uint16_t quantizer_for(PhysicalLayoutId layout) noexcept {
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    return kDecodeQuantizerQ4G64V0;
  }
  return kDecodeQuantizerNone;
}

std::expected<GdnWeightBinding, Error> bind_q4_or_bf16(ConstTensorView codes,
                                                       ConstTensorView scales,
                                                       std::uint32_t want_n,
                                                       std::uint32_t want_k,
                                                       std::string_view field) {
  if (codes.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null codes"));
  }
  if (codes.space != MemorySpace::Device ||
      codes.dtype != ArithmeticDtype::Bf16) {
    return std::unexpected(
        arg_error(field, "codes must be BF16 arithmetic device memory"));
  }
  if (codes.rank != 2 || codes.extent[0] != want_n || codes.extent[1] != want_k) {
    return std::unexpected(arg_error(field, "logical shape must be V0 GDN geometry"));
  }
  if (!q4_or_bf16_tile(codes.layout)) {
    return std::unexpected(
        arg_error(field, "layout must be cuda_q4g64_v0 or cuda_bf16_dense_tile_v0"));
  }
  bool const q4 = codes.layout == PhysicalLayoutId::CudaQ4G64V0;
  if (q4) {
    if (codes.storage != StorageClass::Int4Grouped) {
      return std::unexpected(arg_error(field, "Q4 payload storage must be int4_grouped"));
    }
  } else if (codes.storage != StorageClass::Bf16) {
    return std::unexpected(arg_error(field, "BF16 payload storage must be bf16"));
  }

  GdnWeightBinding b;
  b.codes = codes;
  b.layout = static_cast<std::uint16_t>(codes.layout);
  b.quantizer = quantizer_for(codes.layout);
  b.n = want_n;
  b.k = want_k;
  b.padded_n = decode_pad_n(want_n);
  b.padded_k = decode_pad_k(want_k);
  b.codes_bytes = decode_code_bytes(b.layout, b.padded_n, b.padded_k);
  b.scales_bytes = decode_scale_bytes(b.layout, b.padded_n, b.padded_k);
  if (b.codes_bytes == 0) {
    return std::unexpected(arg_error(field, "payload byte count is invalid"));
  }
  if (b.scales_bytes == 0) {
    if (scales.pointer != nullptr || scales.rank != 0) {
      return std::unexpected(arg_error(field, "BF16 dense tile must not supply scales"));
    }
  } else {
    if (scales.pointer == nullptr) {
      return std::unexpected(arg_error(field, "Q4 weight requires scales"));
    }
    if (scales.space != MemorySpace::Device ||
        scales.dtype != ArithmeticDtype::Fp16 ||
        scales.layout != codes.layout || scales.storage != codes.storage ||
        scales.rank != 1 ||
        scales.extent[0] != b.scales_bytes / qw38::format::kFp16Size) {
      return std::unexpected(
          arg_error(field, "scale typed view contract mismatch"));
    }
    auto scale_bytes = view_bytes(scales, field);
    if (!scale_bytes) {
      return std::unexpected(scale_bytes.error());
    }
    if (*scale_bytes != b.scales_bytes) {
      return std::unexpected(
          arg_error(field, "scale view length does not match layout"));
    }
    b.scales = scales;
  }
  return b;
}

std::expected<GdnWeightBinding, Error> bind_ab_bf16(ConstTensorView codes,
                                                    std::string_view field) {
  if (codes.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null codes"));
  }
  if (codes.space != MemorySpace::Device ||
      codes.dtype != ArithmeticDtype::Bf16) {
    return std::unexpected(
        arg_error(field, "codes must be BF16 arithmetic device memory"));
  }
  if (codes.rank != 2 || codes.extent[0] != kGdnValueHeads ||
      codes.extent[1] != kHidden) {
    return std::unexpected(arg_error(field, "a/b must be [48,5120]"));
  }
  if (codes.layout != PhysicalLayoutId::CudaBf16DenseTileV0 ||
      codes.storage != StorageClass::Bf16) {
    return std::unexpected(
        arg_error(field, "a/b must be cuda_bf16_dense_tile_v0 BF16"));
  }
  GdnWeightBinding b;
  b.codes = codes;
  b.layout = kDecodeLayoutBf16DenseTileV0;
  b.quantizer = kDecodeQuantizerNone;
  b.n = kGdnValueHeads;
  b.k = kHidden;
  b.padded_n = decode_pad_n(kGdnValueHeads);
  b.padded_k = decode_pad_k(kHidden);
  b.codes_bytes = decode_code_bytes(b.layout, b.padded_n, b.padded_k);
  return b;
}

DecodeMmvDesc mmv_from_weight(GdnWeightBinding const& w) {
  DecodeMmvDesc d;
  d.layout = w.layout;
  d.quantizer = w.quantizer;
  d.n = w.n;
  d.k = w.k;
  d.padded_n = w.padded_n;
  d.padded_k = w.padded_k;
  DecodeDtype const dtype =
      w.layout == qw38::cuda::kDecodeLayoutQ4G64V0 ? DecodeDtype::Q4
                                                   : DecodeDtype::Bf16;
  d.codes = decode_matrix_view(const_cast<void*>(w.codes.pointer), dtype,
                               w.layout, w.n, w.k,
                               w.padded_n, w.padded_k, w.codes_bytes, 16);
  if (w.scales_bytes != 0) {
    d.scales = decode_matrix_view(
        const_cast<void*>(w.scales.pointer), DecodeDtype::Fp16, w.layout,
        w.padded_n,
        w.padded_k / 64u, w.padded_n, w.padded_k / 64u,
        w.scales_bytes, 2);
  }
  return d;
}

std::expected<ConstTensorView, Error> require_payload(
    Model const& model, std::string const& name,
    qw38::format::TensorRole role, std::uint32_t layer) {
  auto id = model.resolve_tensor(name, qw38::format::SemanticNodeKind::GatedDeltaNet,
                                 role, layer);
  if (!id) return std::unexpected(id.error());
  auto v = model.payload(*id);
  if (!v) {
    return std::unexpected(v.error());
  }
  return *v;
}

std::expected<ConstTensorView, Error> optional_scales(
    Model const& model, std::string const& name, PhysicalLayoutId layout,
    std::uint32_t layer) {
  if (layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    return ConstTensorView{};
  }
  auto id = model.resolve_tensor(name, qw38::format::SemanticNodeKind::GatedDeltaNet,
                                 qw38::format::TensorRole::DenseWeight, layer);
  if (!id) return std::unexpected(id.error());
  auto v = model.scales(*id);
  if (!v) {
    return std::unexpected(v.error());
  }
  return *v;
}

TensorView overlay(std::byte* base, std::uint64_t off, ArithmeticDtype dtype,
                   PhysicalLayoutId layout, StorageClass storage, bool writable,
                   std::uint8_t rank, std::uint64_t e0, std::uint64_t e1 = 0) {
  TensorView v{};
  v.pointer = base + off;
  v.dtype = dtype;
  v.layout = layout;
  v.storage = storage;
  v.space = MemorySpace::Device;
  (void)writable;
  v.rank = rank;
  v.extent[0] = e0;
  v.extent[1] = e1;
  return v;
}

std::expected<void, Error> region_rms(GdnFrontPlan const& plan) {
  auto* residual = static_cast<float*>(plan.residual.pointer);
  auto* gamma = static_cast<std::uint16_t const*>(plan.gamma.pointer);
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  if (residual == nullptr || gamma == nullptr || normalized == nullptr) {
    return std::unexpected(arg_error("gdn", "plan views are null"));
  }
  if (auto st = qw38::cuda::launch_hidden_rms(residual, gamma, plan.eps, 1,
                                              normalized, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_qkvz(GdnFrontPlan const& plan) {
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  auto* qkv = static_cast<std::uint16_t*>(plan.scratch.qkv.pointer);
  auto* z = static_cast<std::uint16_t*>(plan.scratch.z.pointer);
  DecodeMmvDesc qkv_d = mmv_from_weight(plan.qkv);
  qkv_d.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      qkv_d.k, static_cast<std::uint64_t>(qkv_d.k) * 2u, 2, false);
  qkv_d.output = decode_vector_view(
      qkv, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      qkv_d.n, static_cast<std::uint64_t>(qkv_d.n) * 2u, 2, true);
  qkv_d.epilogue = DecodeEpilogue::StoreBf16;
  DecodeMmvDesc z_d = mmv_from_weight(plan.z);
  z_d.input = qkv_d.input;
  z_d.output = decode_vector_view(
      z, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      z_d.n, static_cast<std::uint64_t>(z_d.n) * 2u, 2, true);
  z_d.epilogue = DecodeEpilogue::StoreBf16;
  std::array<DecodeMmvDesc, 2> range_descs{qkv_d, z_d};
  DecodeMmvRangeDesc ranges{range_descs};
  if (auto st = qw38::cuda::launch_decode_mmv_ranges(ranges, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_ab(GdnFrontPlan const& plan) {
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  auto* a = static_cast<float*>(plan.scratch.a.pointer);
  auto* b = static_cast<float*>(plan.scratch.b.pointer);
  DecodeMmvPairedDesc ab;
  ab.a = mmv_from_weight(plan.a_proj);
  ab.a.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      ab.a.k, static_cast<std::uint64_t>(ab.a.k) * 2u, 2, false);
  ab.a.output = decode_vector_view(
      a, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      ab.a.n, static_cast<std::uint64_t>(ab.a.n) * 4u, 4, true);
  ab.a.epilogue = DecodeEpilogue::StoreFp32;
  ab.b = mmv_from_weight(plan.b_proj);
  ab.b.input = ab.a.input;
  ab.b.output = decode_vector_view(
      b, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      ab.b.n, static_cast<std::uint64_t>(ab.b.n) * 4u, 4, true);
  ab.b.epilogue = ab.a.epilogue;
  if (auto st = qw38::cuda::launch_decode_ab_bf16(ab, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<std::uint32_t, Error> region_conv(GdnFrontPlan const& plan) {
  auto current = plan.cursor.value();
  if (!current) {
    return std::unexpected(current.error());
  }
  auto* qkv = static_cast<std::uint16_t*>(plan.scratch.qkv.pointer);
  auto* convolved = static_cast<std::uint16_t*>(plan.scratch.convolved.pointer);
  auto* history = static_cast<std::uint16_t*>(plan.history.pointer);
  auto* taps = static_cast<std::uint16_t const*>(plan.taps.pointer);
  std::uint32_t const cursor = *current;
  if (auto st = qw38::cuda::launch_gdn_conv_silu(qkv, taps, history, cursor,
                                                 convolved, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return cursor;
}

std::expected<void, Error> region_prep(GdnFrontPlan const& plan) {
  auto* a = static_cast<float*>(plan.scratch.a.pointer);
  auto* b = static_cast<float*>(plan.scratch.b.pointer);
  auto* convolved = static_cast<std::uint16_t*>(plan.scratch.convolved.pointer);
  auto* q_hat = static_cast<float*>(plan.scratch.q_hat.pointer);
  auto* k_hat = static_cast<float*>(plan.scratch.k_hat.pointer);
  auto* alpha = static_cast<float*>(plan.scratch.alpha.pointer);
  auto* beta = static_cast<float*>(plan.scratch.beta.pointer);
  auto* a_log = static_cast<std::uint16_t const*>(plan.a_log.pointer);
  auto* dt_bias = static_cast<std::uint16_t const*>(plan.dt_bias.pointer);
  if (auto st = qw38::cuda::launch_gdn_prepare(convolved, a, b, a_log, dt_bias,
                                               plan.eps, q_hat, k_hat, alpha,
                                               beta, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

}  // namespace

std::expected<std::uint32_t, Error> gdn_state_index(std::uint32_t language_layer) {
  if (!is_gdn_language_layer(language_layer)) {
    return std::unexpected(arg_error("layer", "language layer is not a GDN mixer"));
  }
  return language_layer - language_layer / 4u;
}

std::string gdn_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".input_layernorm.weight";
  return out.str();
}

std::string gdn_gated_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.norm.weight";
  return out.str();
}

std::string gdn_qkv_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_qkv.weight";
  return out.str();
}

std::string gdn_z_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_z.weight";
  return out.str();
}

std::string gdn_a_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_a.weight";
  return out.str();
}

std::string gdn_b_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_b.weight";
  return out.str();
}

std::string gdn_out_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.out_proj.weight";
  return out.str();
}

std::string gdn_conv_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.conv1d.weight";
  return out.str();
}

std::string gdn_alog_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.A_log";
  return out.str();
}

std::string gdn_dt_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.dt_bias";
  return out.str();
}

std::expected<GdnWorkspaceViews, Error> bind_gdn_workspace(
    WorkspaceView workspace) {
  if (workspace.pointer == nullptr) {
    return std::unexpected(arg_error("workspace", "null view"));
  }
  if (workspace.space != MemorySpace::Device) {
    return std::unexpected(arg_error("workspace", "view must be device memory"));
  }
  if (workspace.bytes != kGdnWorkspaceBytesPerToken) {
    return std::unexpected(
        arg_error("workspace", "GdnWorkspace must be 107264 bytes per token"));
  }
  auto* base = workspace.pointer;
  GdnWorkspaceViews v;
  v.qkv = overlay(base, kGdnOffQkv, ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 1,
                  kConvChannels);
  v.z = overlay(base, kGdnOffZ, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kGdnValueHeads, kGdnValueDim);
  v.convolved = overlay(base, kGdnOffConvolved, ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true,
                        1, kConvChannels);
  v.q_hat = overlay(base, kGdnOffQHat, ArithmeticDtype::Fp32,
                    PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 2,
                    kGdnKeyHeads, kGdnKeyDim);
  v.k_hat = overlay(base, kGdnOffKHat, ArithmeticDtype::Fp32,
                    PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 2,
                    kGdnKeyHeads, kGdnKeyDim);
  v.a = overlay(base, kGdnOffA, ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                kGdnValueHeads);
  v.b = overlay(base, kGdnOffB, ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                kGdnValueHeads);
  v.alpha = overlay(base, kGdnOffAlpha, ArithmeticDtype::Fp32,
                    PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                    kGdnValueHeads);
  v.beta = overlay(base, kGdnOffBeta, ArithmeticDtype::Fp32,
                   PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                   kGdnValueHeads);
  v.v = overlay(base, kGdnOffConvolved + kGdnVOffset * qw38::format::kBf16Size,
                ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
                StorageClass::Bf16, true, 2, kGdnValueHeads, kGdnValueDim);
  v.o = overlay(base, kGdnOffO, ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 2,
                kGdnValueHeads, kGdnValueDim);
  v.u = overlay(base, kGdnOffU, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kGdnValueHeads, kGdnValueDim);
  return v;
}

std::expected<GdnFrontPlan, Error> bind_gdn_front_plan(
    GdnFrontBindViews const& views, qw38::cuda::Stream const& stream, float eps) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (auto cursor = views.cursor.value(); !cursor) {
    return std::unexpected(cursor.error());
  }
  auto gdn_i = gdn_state_index(views.language_layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }

  auto qkv = bind_q4_or_bf16(views.qkv, views.qkv_scales, kConvChannels, kHidden,
                             "qkv");
  if (!qkv) {
    return std::unexpected(qkv.error());
  }
  auto z = bind_q4_or_bf16(views.z, views.z_scales, kGdnZWidth, kHidden, "z");
  if (!z) {
    return std::unexpected(z.error());
  }
  if (qkv->layout != z->layout || qkv->quantizer != z->quantizer) {
    return std::unexpected(
        arg_error("layout", "qkv/z must share one Q4 or BF16-control family"));
  }
  auto a = bind_ab_bf16(views.a, "a");
  if (!a) {
    return std::unexpected(a.error());
  }
  auto b = bind_ab_bf16(views.b, "b");
  if (!b) {
    return std::unexpected(b.error());
  }

  auto gamma = require_view(
      views.gamma, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, {kHidden},
      "gamma");
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto residual = require_view(
      views.residual, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, {kHidden},
      "residual");
  if (!residual) {
    return std::unexpected(residual.error());
  }
  auto normalized = require_view(
      views.normalized, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, {kHidden},
      "normalized");
  if (!normalized) {
    return std::unexpected(normalized.error());
  }

  auto taps = require_view(
      views.taps, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16TapMajorV0, StorageClass::Bf16,
      {kConvKernel, kConvChannels}, "taps");
  if (!taps) {
    return std::unexpected(taps.error());
  }

  auto a_log = require_view(
      views.a_log, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
      {kGdnValueHeads}, "A_log");
  if (!a_log) {
    return std::unexpected(a_log.error());
  }
  auto dt = require_view(
      views.dt_bias, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
      {kGdnValueHeads}, "dt_bias");
  if (!dt) {
    return std::unexpected(dt.error());
  }

  auto history = require_view(
      views.history, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16ConvHistoryV0, StorageClass::Bf16,
      {kConvTaps, kConvChannels}, "history");
  if (!history) {
    return std::unexpected(history.error());
  }

  auto scratch = bind_gdn_workspace(views.workspace);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  if (auto st = validate_intervals(
          {{"qkv", {qkv->codes, qkv->codes_bytes}},
           {"qkv_scales", {qkv->scales, qkv->scales_bytes}},
           {"z", {z->codes, z->codes_bytes}},
           {"z_scales", {z->scales, z->scales_bytes}},
           {"a", {a->codes, a->codes_bytes}},
           {"b", {b->codes, b->codes_bytes}},
           {"gamma", {*gamma, kHidden * qw38::format::kBf16Size}},
           {"taps",
            {*taps, static_cast<std::uint64_t>(kConvKernel) *
                        kConvChannels * qw38::format::kBf16Size}},
           {"A_log",
            {*a_log, kGdnValueHeads * qw38::format::kBf16Size}},
           {"dt_bias", {*dt, kGdnValueHeads * qw38::format::kBf16Size}},
           {"residual",
            {*residual, kHidden * qw38::format::kFp32Size}},
           {"normalized",
            {*normalized, kHidden * qw38::format::kBf16Size}},
           {"workspace",
            {byte_arena_view(views.workspace), views.workspace.bytes}},
           {"history",
            {*history, static_cast<std::uint64_t>(kConvTaps) *
                           kConvChannels * qw38::format::kBf16Size}}});
      !st) {
    return std::unexpected(st.error());
  }

  GdnFrontPlan plan;
  plan.qkv = *qkv;
  plan.z = *z;
  plan.a_proj = *a;
  plan.b_proj = *b;
  plan.gamma = *gamma;
  plan.taps = *taps;
  plan.a_log = *a_log;
  plan.dt_bias = *dt;
  plan.residual = *residual;
  plan.normalized = *normalized;
  plan.scratch = *scratch;
  plan.history = *history;
  plan.cursor = views.cursor;
  plan.stream = &stream;
  plan.eps = eps;
  plan.language_layer = views.language_layer;
  plan.gdn_layer = *gdn_i;
  return plan;
}

std::expected<GdnFrontPlan, Error> bind_gdn_front_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps) {
  auto const* session_stream = detail::SessionPlanAccess::stream(session);
  if (session_stream == nullptr || session_stream->empty() || stream.empty() ||
      model.device() != stream.device() ||
      session_stream->native() != stream.native()) {
    return std::unexpected(arg_error(
        "stream", "model and GDN front plan must use the session device and stream"));
  }
  auto* session_state = detail::SessionPlanAccess::execution_state(session);
  if (session_state == nullptr || session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned or closed"));
  }
  auto gdn_i = gdn_state_index(layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }
  auto const qkv_n = gdn_qkv_name(layer);
  auto const z_n = gdn_z_name(layer);
  auto const a_n = gdn_a_name(layer);
  auto const b_n = gdn_b_name(layer);
  auto const conv_n = gdn_conv_name(layer);
  auto const alog_n = gdn_alog_name(layer);
  auto const dt_n = gdn_dt_name(layer);
  auto const norm_n = gdn_norm_name(layer);

  auto qkv = require_payload(model, qkv_n, qw38::format::TensorRole::DenseWeight, layer);
  auto z = require_payload(model, z_n, qw38::format::TensorRole::DenseWeight, layer);
  auto a = require_payload(model, a_n, qw38::format::TensorRole::DenseWeight, layer);
  auto b = require_payload(model, b_n, qw38::format::TensorRole::DenseWeight, layer);
  auto taps = require_payload(model, conv_n, qw38::format::TensorRole::ConvWeight, layer);
  auto alog = require_payload(model, alog_n, qw38::format::TensorRole::TimeParameter, layer);
  auto dt = require_payload(model, dt_n, qw38::format::TensorRole::TimeParameter, layer);
  auto gamma = require_payload(model, norm_n, qw38::format::TensorRole::AdditiveNorm, layer);
  if (!qkv) {
    return std::unexpected(qkv.error());
  }
  if (!z) {
    return std::unexpected(z.error());
  }
  if (!a) {
    return std::unexpected(a.error());
  }
  if (!b) {
    return std::unexpected(b.error());
  }
  if (!taps) {
    return std::unexpected(taps.error());
  }
  if (!alog) {
    return std::unexpected(alog.error());
  }
  if (!dt) {
    return std::unexpected(dt.error());
  }
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto qkv_s = optional_scales(model, qkv_n, qkv->layout, layer);
  auto z_s = optional_scales(model, z_n, z->layout, layer);
  if (!qkv_s) {
    return std::unexpected(qkv_s.error());
  }
  if (!z_s) {
    return std::unexpected(z_s.error());
  }

  auto cursor = detail::SessionPlanAccess::conv_cursor(session, *gdn_i);
  if (!cursor) {
    return std::unexpected(cursor.error());
  }
  auto off = conv_history_byte_offset(*gdn_i, 0, 0);
  if (!off) {
    return std::unexpected(off.error());
  }
  TensorView history = session.conv_history();
  history.pointer = static_cast<std::byte*>(history.pointer) + *off;
  history.rank = 2;
  history.extent = {};
  history.extent[0] = kConvTaps;
  history.extent[1] = kConvChannels;

  auto normalized = session.scratch(qw38::format::ScratchKind::NormalizedHidden);
  auto workspace = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  if (!workspace) {
    return std::unexpected(workspace.error());
  }

  GdnFrontBindViews views;
  views.qkv = *qkv;
  views.qkv_scales = *qkv_s;
  views.z = *z;
  views.z_scales = *z_s;
  views.a = *a;
  views.b = *b;
  views.gamma = *gamma;
  views.taps = *taps;
  views.taps.rank = 2;
  views.taps.extent = {};
  views.taps.extent[0] = kConvKernel;
  views.taps.extent[1] = kConvChannels;
  views.a_log = *alog;
  views.dt_bias = *dt;
  views.residual = session.residual_h();
  views.residual.rank = 1;
  views.residual.extent = {};
  views.residual.extent[0] = kHidden;
  views.normalized = normalized->region[0].tensor;
  views.normalized.rank = 1;
  views.normalized.extent = {};
  views.normalized.extent[0] = kHidden;
  workspace->bytes = kGdnWorkspaceBytesPerToken;
  views.workspace = *workspace;
  views.history = history;
  views.cursor = *cursor;
  views.language_layer = layer;
  auto plan = bind_gdn_front_plan(views, stream, eps);
  if (!plan) return std::unexpected(plan.error());
  plan->session_state = detail::SessionPlanAccess::execution_state(session);
  return plan;
}

std::expected<void, Error> execute_gdn_front(GdnFrontPlan const& plan) {
  if (plan.session_state != nullptr && plan.session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned"));
  }
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  auto run = [&]() -> std::expected<void, Error> {
    if (auto st = region_rms(plan); !st) return st;
    if (auto st = region_qkvz(plan); !st) return st;
    if (auto st = region_ab(plan); !st) return st;
    auto cursor = region_conv(plan);
    if (!cursor) return std::unexpected(cursor.error());
    if (auto st = region_prep(plan); !st) return st;
    if (auto st = plan.stream->sync(); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    return plan.cursor.commit_advance(*cursor);
  };
  auto result = run();
  if (!result && plan.session_state != nullptr) {
    plan.session_state->poison();
  }
  return result;
}

std::expected<GdnRecurrencePlan, Error> bind_gdn_recurrence_plan(
    GdnRecurrenceBindViews const& views, qw38::cuda::Stream const& stream) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  auto q = require_view(
      views.q_hat, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
      {kGdnKeyHeads, kGdnKeyDim}, "q_hat");
  if (!q) {
    return std::unexpected(q.error());
  }
  auto k = require_view(
      views.k_hat, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
      {kGdnKeyHeads, kGdnKeyDim}, "k_hat");
  if (!k) {
    return std::unexpected(k.error());
  }
  auto alpha = require_view(
      views.alpha, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
      {kGdnValueHeads}, "alpha");
  if (!alpha) {
    return std::unexpected(alpha.error());
  }
  auto beta = require_view(
      views.beta, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
      {kGdnValueHeads}, "beta");
  if (!beta) {
    return std::unexpected(beta.error());
  }
  auto v = require_view(
      views.v, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
      {kGdnValueHeads, kGdnValueDim}, "v");
  if (!v) {
    return std::unexpected(v.error());
  }
  auto o = require_view(
      views.o, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
      {kGdnValueHeads, kGdnValueDim}, "o");
  if (!o) {
    return std::unexpected(o.error());
  }
  if (auto state = validate_gdn_state_view(views.s); !state) {
    return std::unexpected(state.error());
  }
  if (auto st = validate_intervals(
          {{"q_hat", {*q, kGdnBytesQHat}},
           {"k_hat", {*k, kGdnBytesKHat}},
           {"alpha", {*alpha, kGdnBytesGate}},
           {"beta", {*beta, kGdnBytesGate}},
           {"v", {*v, kGdnBytesZ}},
           {"s", {views.s, kGdnSBytes}},
           {"o", {*o, kGdnBytesO}}});
      !st) {
    return std::unexpected(st.error());
  }

  auto idx = gdn_state_index(views.language_layer);
  if (!idx) {
    return std::unexpected(idx.error());
  }
  if (views.s_layer != *idx) {
    return std::unexpected(
        arg_error("s_layer", "must match the GDN state layer derived from language_layer"));
  }

  GdnRecurrencePlan plan;
  plan.q_hat = *q;
  plan.k_hat = *k;
  plan.alpha = *alpha;
  plan.beta = *beta;
  plan.v = *v;
  plan.s = views.s;
  plan.o = *o;
  plan.s_layer = *idx;
  plan.language_layer = views.language_layer;
  plan.gdn_layer = *idx;
  plan.stream = &stream;
  return plan;
}

std::expected<GdnRecurrencePlan, Error> bind_gdn_recurrence_plan(
    Session& session, std::uint32_t layer, qw38::cuda::Stream const& stream) {
  auto const* session_stream = detail::SessionPlanAccess::stream(session);
  if (session_stream == nullptr || session_stream->empty() || stream.empty() ||
      session_stream->native() != stream.native()) {
    return std::unexpected(arg_error(
        "stream", "GDN recurrence must use the session stream"));
  }
  auto* session_state = detail::SessionPlanAccess::execution_state(session);
  if (session_state == nullptr || session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned or closed"));
  }
  auto gdn_i = gdn_state_index(layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }
  auto workspace = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  if (!workspace) {
    return std::unexpected(workspace.error());
  }
  workspace->bytes = kGdnWorkspaceBytesPerToken;
  auto slices = bind_gdn_workspace(*workspace);
  if (!slices) {
    return std::unexpected(slices.error());
  }
  GdnRecurrenceBindViews views;
  views.q_hat = slices->q_hat;
  views.k_hat = slices->k_hat;
  views.alpha = slices->alpha;
  views.beta = slices->beta;
  views.v = slices->v;
  views.s = session.gdn_s();
  views.o = slices->o;
  views.s_layer = *gdn_i;
  views.language_layer = layer;
  auto plan = bind_gdn_recurrence_plan(views, stream);
  if (!plan) return std::unexpected(plan.error());
  plan->session_state = detail::SessionPlanAccess::execution_state(session);
  return plan;
}

std::expected<void, Error> execute_gdn_recurrence(GdnRecurrencePlan const& plan) {
  if (plan.session_state != nullptr && plan.session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned"));
  }
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  auto* q_hat = static_cast<float const*>(plan.q_hat.pointer);
  auto* k_hat = static_cast<float const*>(plan.k_hat.pointer);
  auto* alpha = static_cast<float const*>(plan.alpha.pointer);
  auto* beta = static_cast<float const*>(plan.beta.pointer);
  auto* v = static_cast<std::uint16_t const*>(plan.v.pointer);
  auto* s = static_cast<float*>(plan.s.pointer);
  auto* o = static_cast<float*>(plan.o.pointer);
  if (q_hat == nullptr || k_hat == nullptr || alpha == nullptr || beta == nullptr ||
      v == nullptr || s == nullptr || o == nullptr) {
    return std::unexpected(arg_error("gdn", "recurrence views are null"));
  }
  if (auto st = qw38::cuda::launch_gdn_recurrence(q_hat, k_hat, alpha, beta, v, s,
                                                  plan.s_layer, o, *plan.stream);
      !st) {
    if (plan.session_state != nullptr) plan.session_state->poison();
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<GdnPlan, Error> bind_gdn_plan(GdnBindViews const& views,
                                           qw38::cuda::Stream const& stream,
                                           float eps) {
  if (auto position = views.position.value(); !position) {
    return std::unexpected(position.error());
  }
  GdnFrontBindViews front;
  front.qkv = views.qkv;
  front.qkv_scales = views.qkv_scales;
  front.z = views.z;
  front.z_scales = views.z_scales;
  front.a = views.a;
  front.b = views.b;
  front.gamma = views.gamma;
  front.taps = views.taps;
  front.a_log = views.a_log;
  front.dt_bias = views.dt_bias;
  front.residual = views.residual;
  front.normalized = views.normalized;
  front.workspace = views.workspace;
  front.history = views.history;
  front.cursor = views.cursor;
  front.language_layer = views.language_layer;
  auto fp = bind_gdn_front_plan(front, stream, eps);
  if (!fp) {
    return std::unexpected(fp.error());
  }
  auto out = bind_q4_or_bf16(views.out, views.out_scales, kHidden, kGdnZWidth, "out");
  if (!out) {
    return std::unexpected(out.error());
  }
  if (out->layout != fp->qkv.layout || out->quantizer != fp->qkv.quantizer) {
    return std::unexpected(
        arg_error("layout", "qkv/z/out must share one Q4 or BF16-control family"));
  }
  auto gated = require_view(
      views.gated_gamma, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
      {kGdnValueDim}, "gated_gamma");
  if (!gated) {
    return std::unexpected(gated.error());
  }
  auto residual_out = require_view(
      views.residual_out, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, {kHidden},
      "residual_out");
  if (!residual_out) {
    return std::unexpected(residual_out.error());
  }
  if (auto state = validate_gdn_state_view(views.s); !state) {
    return std::unexpected(state.error());
  }
  if (fp->scratch.u.pointer == nullptr) {
    return std::unexpected(arg_error("workspace", "u overlay is required"));
  }
  if (auto st = validate_intervals(
          {{"qkv", {fp->qkv.codes, fp->qkv.codes_bytes}},
           {"qkv_scales", {fp->qkv.scales, fp->qkv.scales_bytes}},
           {"z", {fp->z.codes, fp->z.codes_bytes}},
           {"z_scales", {fp->z.scales, fp->z.scales_bytes}},
           {"a", {fp->a_proj.codes, fp->a_proj.codes_bytes}},
           {"b", {fp->b_proj.codes, fp->b_proj.codes_bytes}},
           {"out", {out->codes, out->codes_bytes}},
           {"out_scales", {out->scales, out->scales_bytes}},
           {"gamma",
            {fp->gamma, kHidden * qw38::format::kBf16Size}},
           {"gated_gamma",
            {*gated, kGdnValueDim * qw38::format::kBf16Size}},
           {"taps",
            {fp->taps, static_cast<std::uint64_t>(kConvKernel) *
                           kConvChannels * qw38::format::kBf16Size}},
           {"A_log",
            {fp->a_log, kGdnValueHeads * qw38::format::kBf16Size}},
           {"dt_bias",
            {fp->dt_bias, kGdnValueHeads * qw38::format::kBf16Size}},
           {"residual",
            {fp->residual, kHidden * qw38::format::kFp32Size}},
           {"residual_out",
            {*residual_out, kHidden * qw38::format::kFp32Size}},
           {"normalized",
            {fp->normalized, kHidden * qw38::format::kBf16Size}},
           {"workspace",
            {byte_arena_view(views.workspace), views.workspace.bytes}},
           {"history",
            {fp->history, static_cast<std::uint64_t>(kConvTaps) *
                              kConvChannels * qw38::format::kBf16Size}},
           {"s", {views.s, kGdnSBytes}}});
      !st) {
    return std::unexpected(st.error());
  }

  GdnPlan plan;
  plan.front = std::move(*fp);
  plan.out = *out;
  plan.gated_gamma = *gated;
  plan.residual_out = *residual_out;
  plan.s = views.s;
  plan.position = views.position;
  plan.s_layer = plan.front.gdn_layer;
  return plan;
}

std::expected<GdnPlan, Error> bind_gdn_plan(Model const& model, Session& session,
                                           std::uint32_t layer,
                                           qw38::cuda::Stream const& stream,
                                           float eps) {
  auto const* session_stream = detail::SessionPlanAccess::stream(session);
  if (session_stream == nullptr || session_stream->empty()) {
    return std::unexpected(
        make_error(ErrorCode::Internal, "session", "missing stream"));
  }
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (stream.native() != session_stream->native()) {
    return std::unexpected(
        arg_error("stream", "must match the session stream"));
  }
  if (model.device() != stream.device()) {
    return std::unexpected(arg_error("device", "model and session devices differ"));
  }

  auto gdn_i = gdn_state_index(layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }
  auto const out_n = gdn_out_name(layer);
  auto const gated_n = gdn_gated_norm_name(layer);
  auto out = require_payload(model, out_n, qw38::format::TensorRole::DenseWeight, layer);
  auto gated = require_payload(model, gated_n, qw38::format::TensorRole::GdnGatedNorm, layer);
  if (!out) {
    return std::unexpected(out.error());
  }
  if (!gated) {
    return std::unexpected(gated.error());
  }
  auto out_s = optional_scales(model, out_n, out->layout, layer);
  if (!out_s) {
    return std::unexpected(out_s.error());
  }

  auto front = bind_gdn_front_plan(model, session, layer, stream, eps);
  if (!front) {
    return std::unexpected(front.error());
  }
  auto position = detail::SessionPlanAccess::gdn_position(session, *gdn_i);
  if (!position) {
    return std::unexpected(position.error());
  }

  GdnBindViews views;
  views.qkv = front->qkv.codes;
  views.qkv_scales = front->qkv.scales;
  views.z = front->z.codes;
  views.z_scales = front->z.scales;
  views.a = front->a_proj.codes;
  views.b = front->b_proj.codes;
  views.out = *out;
  views.out_scales = *out_s;
  views.gamma = front->gamma;
  views.gated_gamma = *gated;
  views.taps = front->taps;
  views.a_log = front->a_log;
  views.dt_bias = front->dt_bias;
  views.residual = front->residual;
  views.residual_out = session.residual_h_mid();
  views.residual_out.rank = 1;
  views.residual_out.extent = {};
  views.residual_out.extent[0] = kHidden;
  views.normalized = front->normalized;
  auto workspace = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  if (!workspace) {
    return std::unexpected(workspace.error());
  }
  workspace->bytes = kGdnWorkspaceBytesPerToken;
  views.workspace = *workspace;
  views.history = front->history;
  views.s = session.gdn_s();
  views.cursor = front->cursor;
  views.position = *position;
  views.language_layer = layer;
  auto plan = bind_gdn_plan(views, stream, eps);
  if (!plan) return std::unexpected(plan.error());
  plan->session_state = front->session_state;
  plan->front.session_state = front->session_state;
  return plan;
}

std::expected<void, Error> region_recur(GdnPlan const& plan) {
  auto* q_hat = static_cast<float const*>(plan.front.scratch.q_hat.pointer);
  auto* k_hat = static_cast<float const*>(plan.front.scratch.k_hat.pointer);
  auto* alpha = static_cast<float const*>(plan.front.scratch.alpha.pointer);
  auto* beta = static_cast<float const*>(plan.front.scratch.beta.pointer);
  auto* v = static_cast<std::uint16_t const*>(plan.front.scratch.v.pointer);
  auto* s = static_cast<float*>(plan.s.pointer);
  auto* o = static_cast<float*>(plan.front.scratch.o.pointer);
  if (q_hat == nullptr || k_hat == nullptr || alpha == nullptr || beta == nullptr ||
      v == nullptr || s == nullptr || o == nullptr) {
    return std::unexpected(arg_error("gdn", "recurrence views are null"));
  }
  if (auto st = qw38::cuda::launch_gdn_recurrence(q_hat, k_hat, alpha, beta, v, s,
                                                  plan.s_layer, o, *plan.front.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_gated(GdnPlan const& plan) {
  auto* o = static_cast<float const*>(plan.front.scratch.o.pointer);
  auto* z = static_cast<std::uint16_t const*>(plan.front.scratch.z.pointer);
  auto* gamma = static_cast<std::uint16_t const*>(plan.gated_gamma.pointer);
  auto* u = static_cast<std::uint16_t*>(plan.front.scratch.u.pointer);
  if (o == nullptr || z == nullptr || gamma == nullptr || u == nullptr) {
    return std::unexpected(arg_error("gdn", "gated-RMS views are null"));
  }
  if (auto st = qw38::cuda::launch_gdn_output_transform(o, z, gamma, plan.front.eps,
                                                        u, *plan.front.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_out_residual(GdnPlan const& plan) {
  auto* residual = static_cast<float const*>(plan.front.residual.pointer);
  auto* residual_out = static_cast<float*>(plan.residual_out.pointer);
  auto* u = static_cast<std::uint16_t const*>(plan.front.scratch.u.pointer);
  if (residual == nullptr || residual_out == nullptr || u == nullptr) {
    return std::unexpected(arg_error("gdn", "residual views are null"));
  }
  qw38::cuda::DecodeMmvDesc out = mmv_from_weight(plan.out);
  out.input = decode_vector_view(
      const_cast<std::uint16_t*>(u), DecodeDtype::Bf16,
      qw38::cuda::kDecodeLayoutBf16VectorV0,
      out.k, static_cast<std::uint64_t>(out.k) * 2u, 2, false);
  out.residual = decode_vector_view(
      const_cast<float*>(residual), DecodeDtype::Fp32,
      qw38::cuda::kDecodeLayoutFp32VectorV0, out.n,
      static_cast<std::uint64_t>(out.n) * 4u, 4, false);
  out.output = decode_vector_view(
      residual_out, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      out.n, static_cast<std::uint64_t>(out.n) * 4u, 4, true);
  out.epilogue = DecodeEpilogue::ResidualAddFp32;
  if (auto st = qw38::cuda::launch_decode_mmv(out, *plan.front.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<TensorView, Error> execute_decode_gdn_impl(
    GdnPlan const& plan, std::uint64_t position, GdnRegionTimings* timings) {
  if (plan.session_state != nullptr && plan.session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned"));
  }
  if (plan.front.stream == nullptr || plan.front.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (auto st = plan.position.validate(position); !st) {
    return std::unexpected(st.error());
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (auto st = qw38::cuda::check(
          cudaStreamIsCapturing(plan.front.stream->native(), &capture_status),
          "cudaStreamIsCapturing");
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (capture_status == cudaStreamCaptureStatusActive) {
    return std::unexpected(arg_error(
        "stream.capture", "GDN decode does not support stream capture"));
  }

  using qw38::cuda::Event;
  using qw38::cuda::elapsed_ms;
  Event marks[kGdnMixerRegions + 1];
  if (timings != nullptr) {
    for (int i = 0; i <= kGdnMixerRegions; ++i) {
      auto e = Event::create_timing();
      if (!e) {
        return std::unexpected(from_cuda(e.error()));
      }
      marks[i] = std::move(*e);
    }
  }

  auto mark = [&](int i) -> std::expected<void, Error> {
    if (timings == nullptr) {
      return {};
    }
    if (auto st = marks[i].record(*plan.front.stream); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    return {};
  };

  if (auto st = mark(0); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_rms(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(1); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_qkvz(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(2); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_ab(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(3); !st) {
    return std::unexpected(st.error());
  }
  auto cursor = region_conv(plan.front);
  if (!cursor) {
    return std::unexpected(cursor.error());
  }
  if (auto st = mark(4); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_prep(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(5); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_recur(plan); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(6); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_gated(plan); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(7); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_out_residual(plan); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(8); !st) {
    return std::unexpected(st.error());
  }

  if (timings != nullptr) {
    if (auto st = marks[kGdnMixerRegions].sync(); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    for (int i = 0; i < kGdnMixerRegions; ++i) {
      auto ms = elapsed_ms(marks[i], marks[i + 1]);
      if (!ms) {
        return std::unexpected(from_cuda(ms.error()));
      }
      timings->ms[i] = *ms;
    }
  }
  if (auto st = plan.front.stream->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = plan.front.cursor.commit_advance(*cursor); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = plan.position.commit(position); !st) {
    return std::unexpected(st.error());
  }
  return plan.residual_out;
}

std::expected<TensorView, Error> execute_decode_gdn(GdnPlan const& plan,
                                                   std::uint64_t position) {
  if (plan.session_state != nullptr && plan.session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned"));
  }
  if (auto st = plan.position.validate(position); !st) {
    return std::unexpected(st.error());
  }
  if (plan.front.stream == nullptr || plan.front.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (auto st = qw38::cuda::check(
          cudaStreamIsCapturing(plan.front.stream->native(), &capture_status),
          "cudaStreamIsCapturing");
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (capture_status == cudaStreamCaptureStatusActive) {
    return std::unexpected(arg_error(
        "stream.capture", "GDN decode does not support stream capture"));
  }
  auto result = execute_decode_gdn_impl(plan, position, nullptr);
  if (!result && plan.session_state != nullptr &&
      result.error().field != "stream.capture") {
    plan.session_state->poison();
  }
  return result;
}

std::expected<TensorView, Error> execute_decode_gdn_timed(
    GdnPlan const& plan, std::uint64_t position, GdnRegionTimings& timings) {
  if (plan.session_state != nullptr && plan.session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned"));
  }
  if (auto st = plan.position.validate(position); !st) {
    return std::unexpected(st.error());
  }
  if (plan.front.stream == nullptr || plan.front.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (auto st = qw38::cuda::check(
          cudaStreamIsCapturing(plan.front.stream->native(), &capture_status),
          "cudaStreamIsCapturing");
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (capture_status == cudaStreamCaptureStatusActive) {
    return std::unexpected(arg_error(
        "stream.capture", "GDN decode does not support stream capture"));
  }
  auto result = execute_decode_gdn_impl(plan, position, &timings);
  if (!result && plan.session_state != nullptr &&
      result.error().field != "stream.capture") {
    plan.session_state->poison();
  }
  return result;
}

}  // namespace qw38::runtime
