#include "runtime/attention.hpp"

#include "cuda/activation.hpp"
#include "cuda/attention.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"

#include "format/constants.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

namespace qw38::runtime {
namespace {

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeDtype;
using qw38::cuda::DecodeMmvDesc;
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

std::expected<std::uint64_t, Error> element_count(TensorView const& v,
                                                  std::string_view field) {
  if (v.rank == 0) {
    return std::unexpected(arg_error(field, "rank must be nonzero"));
  }
  std::uint64_t n = 1;
  for (std::uint8_t i = 0; i < v.rank; ++i) {
    if (v.extent[i] == 0 || n > std::numeric_limits<std::uint64_t>::max() / v.extent[i]) {
      return std::unexpected(arg_error(field, "element count overflows or is empty"));
    }
    n *= v.extent[i];
  }
  return n;
}

std::expected<std::uint64_t, Error> view_bytes(TensorView const& v,
                                               std::string_view field) {
  auto n = element_count(v, field);
  if (!n) return std::unexpected(n.error());
  auto const size = qw38::format::element_size(v.dtype);
  if (size == 0 || *n > std::numeric_limits<std::uint64_t>::max() / size) {
    return std::unexpected(arg_error(field, "byte count overflows"));
  }
  return *n * size;
}

std::expected<void, Error> require_alignment(TensorView const& v,
                                             std::uint64_t alignment,
                                             std::string_view field) {
  auto const address = reinterpret_cast<std::uintptr_t>(v.pointer);
  if (alignment == 0 || address % alignment != 0) {
    return std::unexpected(arg_error(field, "view alignment is invalid"));
  }
  return {};
}

std::expected<TensorView, Error> require_view(
    TensorView v, ArithmeticDtype dtype, PhysicalLayoutId layout,
    StorageClass storage, bool writable, std::initializer_list<std::uint64_t> extents,
    std::uint64_t alignment, std::string_view field) {
  if (v.pointer == nullptr || v.space != MemorySpace::Device || v.dtype != dtype ||
      v.layout != layout || v.storage != storage || (writable && !v.writable) ||
      v.rank != extents.size()) {
    return std::unexpected(arg_error(field, "typed view contract mismatch"));
  }
  std::size_t i = 0;
  for (auto const extent : extents) {
    if (v.extent[i++] != extent) {
      return std::unexpected(arg_error(field, "typed view extent mismatch"));
    }
  }
  if (auto st = require_alignment(v, alignment, field); !st) {
    return std::unexpected(st.error());
  }
  if (auto bytes = view_bytes(v, field); !bytes) return std::unexpected(bytes.error());
  return v;
}

struct ByteInterval { std::string_view name; std::uintptr_t begin; std::uintptr_t end; };

std::expected<ByteInterval, Error> byte_interval(TensorView const& v,
                                                  std::uint64_t bytes,
                                                  std::string_view name) {
  auto const begin = reinterpret_cast<std::uintptr_t>(v.pointer);
  if (v.pointer == nullptr || bytes == 0 || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return std::unexpected(arg_error(name, "byte interval is invalid"));
  }
  return ByteInterval{name, begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(ByteInterval const& a, ByteInterval const& b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

std::expected<void, Error> validate_intervals(
    std::initializer_list<std::pair<std::string_view, std::pair<TensorView, std::uint64_t>>> views,
    bool permit_q_y = false) {
  std::vector<ByteInterval> intervals;
  intervals.reserve(views.size());
  for (auto const& [name, view] : views) {
    if (view.second == 0 && view.first.pointer == nullptr) continue;
    auto interval = byte_interval(view.first, view.second, name);
    if (!interval) return std::unexpected(interval.error());
    for (auto const& prior : intervals) {
      bool const allowed = permit_q_y &&
          ((prior.name == "q" && interval->name == "y") ||
           (prior.name == "y" && interval->name == "q")) &&
          prior.begin == interval->begin && prior.end == interval->end;
      if (overlaps(prior, *interval) && !allowed) {
        return std::unexpected(arg_error("alias", std::string(prior.name) +
            " overlaps " + std::string(interval->name)));
      }
    }
    intervals.push_back(*interval);
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

std::expected<AttnWeightBinding, Error> bind_q4_or_bf16(TensorView codes,
                                                        TensorView scales,
                                                        std::uint32_t want_n,
                                                        std::uint32_t want_k,
                                                        std::string_view field) {
  if (codes.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null codes"));
  }
  if (auto st = require_alignment(codes, 16, field); !st) return std::unexpected(st.error());
  if (codes.space != MemorySpace::Device || codes.dtype != ArithmeticDtype::Bf16) {
    return std::unexpected(arg_error(field, "codes must be device memory"));
  }
  if (codes.rank != 2 || codes.extent[0] != want_n || codes.extent[1] != want_k) {
    return std::unexpected(
        arg_error(field, "logical shape must be V0 attention geometry"));
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

  AttnWeightBinding b;
  b.codes = codes;
  b.layout = static_cast<std::uint16_t>(codes.layout);
  b.quantizer = quantizer_for(codes.layout);
  b.n = want_n;
  b.k = want_k;
  b.padded_n = decode_pad_n(want_n);
  b.padded_k = decode_pad_k(want_k);
  b.codes_bytes = decode_code_bytes(b.layout, b.padded_n, b.padded_k);
  b.scales_bytes = decode_scale_bytes(b.layout, b.padded_n, b.padded_k);
  if (b.scales_bytes == 0) {
    if (scales.pointer != nullptr || scales.rank != 0) {
      return std::unexpected(arg_error(field, "BF16 dense tile must not supply scales"));
    }
  } else {
    if (scales.pointer == nullptr) {
      return std::unexpected(arg_error(field, "Q4 weight requires scales"));
    }
    if (scales.space != MemorySpace::Device) {
      return std::unexpected(arg_error(field, "scales must be device memory"));
    }
    if (scales.dtype != ArithmeticDtype::Fp16 || scales.layout != codes.layout ||
        scales.storage != codes.storage || scales.rank != 1 ||
        scales.extent[0] != b.scales_bytes / qw38::format::kFp16Size) {
      return std::unexpected(arg_error(field, "scale typed view contract mismatch"));
    }
    if (auto st = require_alignment(scales, 2, field); !st) return std::unexpected(st.error());
    if (scales.extent[0] * qw38::format::kFp16Size != b.scales_bytes) {
      return std::unexpected(arg_error(field, "scale view length does not match layout"));
    }
    b.scales = scales;
  }
  return b;
}

DecodeMmvDesc mmv_from_weight(AttnWeightBinding const& w) {
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
  d.codes = decode_matrix_view(w.codes.pointer, dtype, w.layout, w.n, w.k,
                               w.padded_n, w.padded_k, w.codes_bytes, 16);
  if (w.scales_bytes != 0) {
    d.scales = decode_matrix_view(
        w.scales.pointer, DecodeDtype::Fp16, w.layout, w.padded_n,
        w.padded_k / 64u, w.padded_n, w.padded_k / 64u,
        w.scales_bytes, 2);
  }
  return d;
}

std::expected<TensorView, Error> require_payload(Model const& model,
                                                 std::string const& name) {
  auto v = model.payload(name);
  if (!v) {
    return std::unexpected(v.error());
  }
  return *v;
}

std::expected<TensorView, Error> optional_scales(Model const& model,
                                                 std::string const& name,
                                                 PhysicalLayoutId layout) {
  if (layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    return TensorView{};
  }
  auto v = model.scales(name);
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
  v.writable = writable;
  v.rank = rank;
  v.extent[0] = e0;
  v.extent[1] = e1;
  return v;
}

}  // namespace

std::expected<std::uint32_t, Error> attn_state_index(
    std::uint32_t language_layer) {
  if (!is_attention_language_layer(language_layer)) {
    return std::unexpected(
        arg_error("layer", "language layer is not a full-attention mixer"));
  }
  return language_layer / 4u;
}

std::string attn_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".input_layernorm.weight";
  return out.str();
}

std::string attn_q_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".self_attn.q_proj.weight";
  return out.str();
}

std::string attn_k_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".self_attn.k_proj.weight";
  return out.str();
}

std::string attn_v_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".self_attn.v_proj.weight";
  return out.str();
}

std::string attn_o_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".self_attn.o_proj.weight";
  return out.str();
}

std::string attn_q_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".self_attn.q_norm.weight";
  return out.str();
}

std::string attn_k_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".self_attn.k_norm.weight";
  return out.str();
}

std::expected<AttnWorkspaceViews, Error> bind_attention_workspace(
    TensorView workspace) {
  if (workspace.pointer == nullptr || workspace.space != MemorySpace::Device ||
      workspace.dtype != ArithmeticDtype::Fp32 ||
      workspace.layout != PhysicalLayoutId::CudaFp32VectorV0 ||
      workspace.storage != StorageClass::Fp32 || !workspace.writable || workspace.rank != 1) {
    return std::unexpected(arg_error("workspace", "typed view contract mismatch"));
  }
  if (auto st = require_alignment(workspace, 16, "workspace"); !st) return std::unexpected(st.error());
  auto workspace_bytes = view_bytes(workspace, "workspace");
  if (!workspace_bytes) return std::unexpected(workspace_bytes.error());
  if (*workspace_bytes < kAttentionWorkspaceBytesPerToken) {
    return std::unexpected(
        arg_error("workspace", "AttentionWorkspace must be 78016 bytes per token"));
  }
  auto* base = static_cast<std::byte*>(workspace.pointer);
  AttnWorkspaceViews v;
  v.qg = overlay(base, kAttnOffQg, ArithmeticDtype::Bf16,
                 PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                 kQueryHeads, 2u * kHeadDim);
  v.k = overlay(base, kAttnOffK, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kKvHeads, kHeadDim);
  v.v = overlay(base, kAttnOffV, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kKvHeads, kHeadDim);
  v.q = overlay(base, kAttnOffQ, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kQueryHeads, kHeadDim);
  v.g = overlay(base, kAttnOffG, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kQueryHeads, kHeadDim);
  return v;
}

std::expected<AttentionPrepPlan, Error> bind_attention_prep_plan(
    AttentionPrepBindViews const& views, qw38::cuda::Stream const& stream,
    float eps) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (views.host_populated == nullptr) {
    return std::unexpected(
        arg_error("populated", "host populated-length pointer is required"));
  }
  if (views.kv_capacity == 0) {
    return std::unexpected(
        make_error(ErrorCode::InvalidCapacity, "kv.capacity",
                   "capacity must be > 0"));
  }
  if (*views.host_populated > views.kv_capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                      "kv.populated",
                                      "populated length exceeds capacity"));
  }
  auto attn_i = attn_state_index(views.language_layer);
  if (!attn_i) {
    return std::unexpected(attn_i.error());
  }

  auto qg = bind_q4_or_bf16(views.qg, views.qg_scales, kQgWidth, kHidden, "qg");
  if (!qg) {
    return std::unexpected(qg.error());
  }
  auto k = bind_q4_or_bf16(views.k, views.k_scales, kAttnKvWidth, kHidden, "k");
  if (!k) {
    return std::unexpected(k.error());
  }
  auto v = bind_q4_or_bf16(views.v, views.v_scales, kAttnKvWidth, kHidden, "v");
  if (!v) {
    return std::unexpected(v.error());
  }
  if (qg->layout != k->layout || qg->quantizer != k->quantizer ||
      qg->layout != v->layout || qg->quantizer != v->quantizer) {
    return std::unexpected(
        arg_error("layout", "q/g, k, and v must share one Q4 or BF16-control family"));
  }

  auto gamma = require_view(views.gamma, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false, {kHidden}, 2, "gamma");
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto gamma_q =
      require_view(views.gamma_q, ArithmeticDtype::Bf16,
          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false, {kHeadDim}, 2, "gamma_q");
  if (!gamma_q) {
    return std::unexpected(gamma_q.error());
  }
  auto gamma_k =
      require_view(views.gamma_k, ArithmeticDtype::Bf16,
          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false, {kHeadDim}, 2, "gamma_k");
  if (!gamma_k) {
    return std::unexpected(gamma_k.error());
  }
  auto inv = require_view(views.inv_freq, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, false, {32}, 4, "inv_freq");
  if (!inv) {
    return std::unexpected(inv.error());
  }
  auto residual = require_view(views.residual, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, {1, kHidden}, 4, "residual");
  if (!residual) {
    return std::unexpected(residual.error());
  }
  auto normalized = require_view(views.normalized, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, {kHidden}, 2, "normalized");
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  auto scratch = bind_attention_workspace(views.workspace);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  auto kv = require_view(views.kv, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16KvCacheV0, StorageClass::Bf16, true,
      {kAttnLayers, kKvComponents, kKvHeads, views.kv_capacity, kHeadDim}, 2, "kv");
  if (!kv) return std::unexpected(kv.error());
  auto kv_bytes = view_bytes(*kv, "kv");
  if (!kv_bytes) return std::unexpected(kv_bytes.error());
  auto workspace_bytes = view_bytes(views.workspace, "workspace");
  if (!workspace_bytes) return std::unexpected(workspace_bytes.error());
  if (auto st = validate_intervals({
          {"qg", {qg->codes, qg->codes_bytes}}, {"qg_scales", {qg->scales, qg->scales_bytes}},
          {"k", {k->codes, k->codes_bytes}}, {"k_scales", {k->scales, k->scales_bytes}},
          {"v", {v->codes, v->codes_bytes}}, {"v_scales", {v->scales, v->scales_bytes}},
          {"gamma", {*gamma, kHidden * 2u}}, {"gamma_q", {*gamma_q, kHeadDim * 2u}},
          {"gamma_k", {*gamma_k, kHeadDim * 2u}}, {"inv_freq", {*inv, 32u * 4u}},
          {"residual", {*residual, kHidden * 4u}}, {"normalized", {*normalized, kHidden * 2u}},
          {"workspace", {views.workspace, *workspace_bytes}}, {"kv", {*kv, *kv_bytes}}}); !st) {
    return std::unexpected(st.error());
  }

  AttentionPrepPlan plan;
  plan.qg = *qg;
  plan.k = *k;
  plan.v = *v;
  plan.gamma = *gamma;
  plan.gamma_q = *gamma_q;
  plan.gamma_k = *gamma_k;
  plan.inv_freq = *inv;
  plan.residual = *residual;
  plan.normalized = *normalized;
  plan.scratch = *scratch;
  plan.kv = *kv;
  plan.host_populated = views.host_populated;
  plan.kv_capacity = views.kv_capacity;
  plan.stream = &stream;
  plan.eps = eps;
  plan.language_layer = views.language_layer;
  plan.attn_layer = *attn_i;
  return plan;
}

std::expected<AttentionCorePlan, Error> bind_attention_core_plan(
    AttentionCoreBindViews const& views, qw38::cuda::Stream const& stream) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (views.host_populated == nullptr) {
    return std::unexpected(arg_error("populated", "missing populated pointer"));
  }
  if (views.kv_capacity == 0 || *views.host_populated > views.kv_capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidCapacity, "kv.capacity",
                                      "invalid capacity/populated length"));
  }
  auto attn_i = attn_state_index(views.language_layer);
  if (!attn_i) return std::unexpected(attn_i.error());
  auto q = require_view(views.q, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, false,
      {kQueryHeads, kHeadDim}, 2, "q");
  auto g = require_view(views.g, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, false,
      {kQueryHeads, kHeadDim}, 2, "g");
  auto y = require_view(views.y, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true,
      {kQueryHeads, kHeadDim}, 2, "y");
  auto residual = require_view(views.residual, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, false,
      {1, kHidden}, 4, "residual");
  auto residual_out = require_view(views.residual_out, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
      {1, kHidden}, 4, "residual_out");
  auto required_workspace = attn_workspace_bytes_for_capacity(views.kv_capacity);
  if (!required_workspace) return std::unexpected(required_workspace.error());
  std::uint64_t const partial_elements =
      (*required_workspace - kAttnOffPartials) / qw38::format::kFp32Size;
  auto partials = require_view(views.partials, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
      {partial_elements}, 4, "partials");
  if (!q) return std::unexpected(q.error());
  if (!g) return std::unexpected(g.error());
  if (!y) return std::unexpected(y.error());
  if (!residual) return std::unexpected(residual.error());
  if (!residual_out) return std::unexpected(residual_out.error());
  if (!partials) return std::unexpected(partials.error());
  auto kv = require_view(views.kv, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16KvCacheV0, StorageClass::Bf16, true,
      {kAttnLayers, kKvComponents, kKvHeads, views.kv_capacity, kHeadDim}, 2, "kv");
  if (!kv) return std::unexpected(kv.error());
  auto out = bind_q4_or_bf16(views.out, views.out_scales, kHidden,
                             kAttnOutWidth, "out");
  if (!out) return std::unexpected(out.error());
  auto partial_bytes = view_bytes(*partials, "partials");
  auto kv_bytes = view_bytes(*kv, "kv");
  if (!partial_bytes || !kv_bytes) return std::unexpected(arg_error("core", "view byte count invalid"));
  if (auto st = validate_intervals({
          {"out", {out->codes, out->codes_bytes}}, {"out_scales", {out->scales, out->scales_bytes}},
          {"q", {*q, kAttnOutWidth * 2u}}, {"g", {*g, kAttnOutWidth * 2u}},
          {"kv", {*kv, *kv_bytes}}, {"partials", {*partials, *partial_bytes}},
          {"y", {*y, kAttnOutWidth * 2u}}, {"residual", {*residual, kHidden * 4u}},
          {"residual_out", {*residual_out, kHidden * 4u}}}, true); !st) {
    return std::unexpected(st.error());
  }

  AttentionCorePlan plan;
  plan.out = *out;
  plan.q = *q;
  plan.g = *g;
  plan.kv = *kv;
  plan.partials = *partials;
  plan.y = *y;
  plan.residual = *residual;
  plan.residual_out = *residual_out;
  plan.host_populated = views.host_populated;
  plan.kv_capacity = views.kv_capacity;
  plan.attn_layer = *attn_i;
  plan.stream = &stream;
  return plan;
}

std::expected<AttentionMixerPlan, Error> bind_attention_mixer_plan(
    AttentionMixerBindViews const& views, qw38::cuda::Stream const& stream,
    float eps) {
  auto prep = bind_attention_prep_plan(views.prep, stream, eps);
  if (!prep) return std::unexpected(prep.error());
  if (views.residual_out.pointer == nullptr) {
    return std::unexpected(arg_error("residual_out", "null view"));
  }
  auto out = bind_q4_or_bf16(views.out, views.out_scales, kHidden,
                             kAttnOutWidth, "out");
  if (!out) return std::unexpected(out.error());
  auto residual_out = require_view(views.residual_out, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
      {1, kHidden}, 4, "residual_out");
  if (!residual_out) return std::unexpected(residual_out.error());

  auto workspace_bytes = view_bytes(views.prep.workspace, "workspace");
  if (!workspace_bytes) return std::unexpected(workspace_bytes.error());
  auto required_workspace = attn_workspace_bytes_for_capacity(views.prep.kv_capacity);
  if (!required_workspace) return std::unexpected(required_workspace.error());
  if (*workspace_bytes != *required_workspace) {
    return std::unexpected(arg_error(
        "workspace", "workspace extent does not match KV-capacity contract"));
  }
  auto kv_bytes = view_bytes(prep->kv, "kv");
  if (!kv_bytes) return std::unexpected(kv_bytes.error());
  if (auto st = validate_intervals({
          {"qg", {prep->qg.codes, prep->qg.codes_bytes}},
          {"qg_scales", {prep->qg.scales, prep->qg.scales_bytes}},
          {"k", {prep->k.codes, prep->k.codes_bytes}},
          {"k_scales", {prep->k.scales, prep->k.scales_bytes}},
          {"v", {prep->v.codes, prep->v.codes_bytes}},
          {"v_scales", {prep->v.scales, prep->v.scales_bytes}},
          {"gamma", {prep->gamma, kHidden * 2u}},
          {"gamma_q", {prep->gamma_q, kHeadDim * 2u}},
          {"gamma_k", {prep->gamma_k, kHeadDim * 2u}},
          {"inv_freq", {prep->inv_freq, 32u * 4u}},
          {"residual", {prep->residual, kHidden * 4u}},
          {"normalized", {prep->normalized, kHidden * 2u}},
          {"workspace", {views.prep.workspace, *workspace_bytes}},
          {"kv", {prep->kv, *kv_bytes}},
          {"out", {out->codes, out->codes_bytes}},
          {"out_scales", {out->scales, out->scales_bytes}},
          {"residual_out", {*residual_out, kHidden * 4u}}}); !st) {
    return std::unexpected(st.error());
  }
  AttentionCoreBindViews core;
  core.q = prep->scratch.q;
  core.g = prep->scratch.g;
  core.kv = prep->kv;
  core.partials = overlay(static_cast<std::byte*>(views.prep.workspace.pointer),
                          kAttnOffPartials, ArithmeticDtype::Fp32,
                          PhysicalLayoutId::CudaFp32VectorV0,
                          StorageClass::Fp32, true, 1,
                          (*workspace_bytes - kAttnOffPartials) /
                              qw38::format::kFp32Size);
  core.y = prep->scratch.q;  // Q is dead after scan; merge materializes y in-place.
  core.residual = prep->residual;
  core.residual_out = *residual_out;
  core.out = views.out;
  core.out_scales = views.out_scales;
  core.host_populated = prep->host_populated;
  core.kv_capacity = prep->kv_capacity;
  core.language_layer = prep->language_layer;
  auto cp = bind_attention_core_plan(core, stream);
  if (!cp) return std::unexpected(cp.error());
  AttentionMixerPlan plan;
  plan.prep = std::move(*prep);
  plan.core = std::move(*cp);
  return plan;
}

std::expected<AttentionMixerPlan, Error> bind_attention_mixer_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps) {
  auto prep = bind_attention_prep_plan(model, session, layer, stream, eps);
  if (!prep) return std::unexpected(prep.error());
  auto out = require_payload(model, attn_o_name(layer));
  if (!out) return std::unexpected(out.error());
  auto scales = optional_scales(model, attn_o_name(layer), out->layout);
  if (!scales) return std::unexpected(scales.error());
  AttentionMixerBindViews views;
  views.prep.qg = prep->qg.codes;
  views.prep.qg_scales = prep->qg.scales;
  views.prep.k = prep->k.codes;
  views.prep.k_scales = prep->k.scales;
  views.prep.v = prep->v.codes;
  views.prep.v_scales = prep->v.scales;
  views.prep.gamma = prep->gamma;
  views.prep.gamma_q = prep->gamma_q;
  views.prep.gamma_k = prep->gamma_k;
  views.prep.inv_freq = prep->inv_freq;
  views.prep.residual = prep->residual;
  views.prep.normalized = prep->normalized;
  auto workspace = session.scratch(qw38::format::ScratchKind::AttentionWorkspace);
  if (!workspace) return std::unexpected(workspace.error());
  views.prep.workspace = *workspace;
  views.prep.kv = prep->kv;
  views.prep.host_populated = prep->host_populated;
  views.prep.kv_capacity = prep->kv_capacity;
  views.prep.language_layer = layer;
  views.out = *out;
  views.out_scales = *scales;
  views.residual_out = session.residual_h_mid();
  views.residual_out.rank = 2;
  views.residual_out.extent = {1, kHidden};
  return bind_attention_mixer_plan(views, stream, eps);
}

std::expected<AttentionPrepPlan, Error> bind_attention_prep_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps) {
  auto attn_i = attn_state_index(layer);
  if (!attn_i) {
    return std::unexpected(attn_i.error());
  }
  auto const q_n = attn_q_name(layer);
  auto const k_n = attn_k_name(layer);
  auto const v_n = attn_v_name(layer);
  auto const qn_n = attn_q_norm_name(layer);
  auto const kn_n = attn_k_norm_name(layer);
  auto const norm_n = attn_norm_name(layer);

  auto qg = require_payload(model, q_n);
  auto k = require_payload(model, k_n);
  auto v = require_payload(model, v_n);
  auto gamma = require_payload(model, norm_n);
  auto gamma_q = require_payload(model, qn_n);
  auto gamma_k = require_payload(model, kn_n);
  auto inv = require_payload(model, "rope.inv_freq");
  if (!qg) {
    return std::unexpected(qg.error());
  }
  if (!k) {
    return std::unexpected(k.error());
  }
  if (!v) {
    return std::unexpected(v.error());
  }
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  if (!gamma_q) {
    return std::unexpected(gamma_q.error());
  }
  if (!gamma_k) {
    return std::unexpected(gamma_k.error());
  }
  if (!inv) {
    return std::unexpected(inv.error());
  }
  auto qg_s = optional_scales(model, q_n, qg->layout);
  auto k_s = optional_scales(model, k_n, k->layout);
  auto v_s = optional_scales(model, v_n, v->layout);
  if (!qg_s) {
    return std::unexpected(qg_s.error());
  }
  if (!k_s) {
    return std::unexpected(k_s.error());
  }
  if (!v_s) {
    return std::unexpected(v_s.error());
  }

  auto normalized = session.scratch(qw38::format::ScratchKind::NormalizedHidden);
  auto workspace = session.scratch(qw38::format::ScratchKind::AttentionWorkspace);
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  if (!workspace) {
    return std::unexpected(workspace.error());
  }

  AttentionPrepBindViews views;
  views.qg = *qg;
  views.qg_scales = *qg_s;
  views.k = *k;
  views.k_scales = *k_s;
  views.v = *v;
  views.v_scales = *v_s;
  views.gamma = *gamma;
  views.gamma_q = *gamma_q;
  views.gamma_k = *gamma_k;
  views.inv_freq = *inv;
  views.residual = session.residual_h();
  views.residual.rank = 2;
  views.residual.extent = {1, kHidden};
  views.normalized = *normalized;
  views.normalized.rank = 1;
  views.normalized.extent = {kHidden};
  views.workspace = *workspace;
  views.kv = session.kv();
  auto populated = session.kv_populated_slot(*attn_i);
  if (!populated) {
    return std::unexpected(populated.error());
  }
  views.host_populated = *populated;
  views.kv_capacity = session.kv_capacity();
  views.language_layer = layer;
  return bind_attention_prep_plan(views, stream, eps);
}

std::expected<void, Error> execute_decode_attention_prep(
    AttentionPrepPlan const& plan, std::uint64_t position) {
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (plan.host_populated == nullptr) {
    return std::unexpected(arg_error("populated", "missing populated-length slot"));
  }
  if (position >= plan.kv_capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidCapacity, "position",
                                      "position exceeds KV capacity"));
  }
  if (position != *plan.host_populated) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength, "position",
                                      "position must equal the append/populated contract"));
  }
  if (position > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return std::unexpected(arg_error("position", "position exceeds int32 RoPE range"));
  }

  auto* residual = static_cast<float*>(plan.residual.pointer);
  auto* gamma = static_cast<std::uint16_t const*>(plan.gamma.pointer);
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  auto* qg = static_cast<std::uint16_t*>(plan.scratch.qg.pointer);
  auto* k = static_cast<std::uint16_t*>(plan.scratch.k.pointer);
  auto* v = static_cast<std::uint16_t*>(plan.scratch.v.pointer);
  auto* q = static_cast<std::uint16_t*>(plan.scratch.q.pointer);
  auto* g = static_cast<std::uint16_t*>(plan.scratch.g.pointer);
  auto* gamma_q = static_cast<std::uint16_t const*>(plan.gamma_q.pointer);
  auto* gamma_k = static_cast<std::uint16_t const*>(plan.gamma_k.pointer);
  auto* inv_freq = static_cast<float const*>(plan.inv_freq.pointer);
  auto* kv = static_cast<std::uint16_t*>(plan.kv.pointer);
  if (residual == nullptr || gamma == nullptr || normalized == nullptr ||
      qg == nullptr || k == nullptr || v == nullptr || q == nullptr ||
      g == nullptr || gamma_q == nullptr || gamma_k == nullptr ||
      inv_freq == nullptr || kv == nullptr) {
    return std::unexpected(arg_error("attn", "plan views are null"));
  }

  if (auto st = qw38::cuda::launch_hidden_rms(residual, gamma, plan.eps, 1,
                                              normalized, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }

  std::array<DecodeMmvDesc, 3> range_descs;
  auto& qg_range = range_descs[0];
  auto& k_range = range_descs[1];
  auto& v_range = range_descs[2];
  qg_range = mmv_from_weight(plan.qg);
  qg_range.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      qg_range.k, static_cast<std::uint64_t>(qg_range.k) * 2u, 2, false);
  qg_range.output = decode_vector_view(
      qg, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      qg_range.n, static_cast<std::uint64_t>(qg_range.n) * 2u, 2, true);
  qg_range.epilogue = DecodeEpilogue::StoreBf16;
  k_range = mmv_from_weight(plan.k);
  k_range.input = qg_range.input;
  k_range.output = decode_vector_view(
      k, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      k_range.n, static_cast<std::uint64_t>(k_range.n) * 2u, 2, true);
  k_range.epilogue = DecodeEpilogue::StoreBf16;
  v_range = mmv_from_weight(plan.v);
  v_range.input = qg_range.input;
  v_range.output = decode_vector_view(
      v, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      v_range.n, static_cast<std::uint64_t>(v_range.n) * 2u, 2, true);
  v_range.epilogue = DecodeEpilogue::StoreBf16;
  DecodeMmvRangeDesc ranges{range_descs};
  if (auto st = qw38::cuda::launch_decode_mmv_ranges(ranges, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }

  auto const pos = static_cast<std::int32_t>(position);
  if (auto st = qw38::cuda::launch_attention_prepare(
          qg, k, v, gamma_q, gamma_k, inv_freq, plan.eps, pos, q, g, kv,
          plan.attn_layer, plan.kv_capacity, position, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }

  if (auto st = plan.stream->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  *plan.host_populated = position + 1u;
  return {};
}

std::expected<TensorView, Error> execute_attention_core(
    AttentionCorePlan const& plan) {
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (plan.host_populated == nullptr || *plan.host_populated > plan.kv_capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength, "populated",
                                      "populated length exceeds capacity"));
  }
  std::uint64_t const populated = *plan.host_populated;
  std::uint64_t const nseg64 = attn_segment_count(populated);
  if (nseg64 > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(arg_error("populated", "segment count exceeds CUDA launch range"));
  }
  std::uint32_t const nseg = static_cast<std::uint32_t>(nseg64);
  auto* q = static_cast<std::uint16_t const*>(plan.q.pointer);
  auto* g = static_cast<std::uint16_t const*>(plan.g.pointer);
  auto* kv = static_cast<std::uint16_t const*>(plan.kv.pointer);
  auto* partials = static_cast<float*>(plan.partials.pointer);
  auto* y = static_cast<std::uint16_t*>(plan.y.pointer);
  auto* residual = static_cast<float const*>(plan.residual.pointer);
  auto* residual_out = static_cast<float*>(plan.residual_out.pointer);
  if (q == nullptr || g == nullptr || kv == nullptr || partials == nullptr ||
      y == nullptr || residual == nullptr || residual_out == nullptr) {
    return std::unexpected(arg_error("attention", "core views are null"));
  }
  if (auto st = qw38::cuda::launch_attention_scan(
          q, kv, plan.attn_layer, plan.kv_capacity, populated, partials, nseg,
          *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::launch_attention_merge(partials, g, nseg, y,
                                                     *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::copy_d2d(
          residual_out, residual, kHidden * qw38::format::kFp32Size,
          *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  DecodeMmvDesc out = mmv_from_weight(plan.out);
  out.input = decode_vector_view(
      y, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      out.k, static_cast<std::uint64_t>(out.k) * 2u, 2, false);
  out.residual = decode_vector_view(
      residual_out, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      out.n, static_cast<std::uint64_t>(out.n) * 4u, 4, true);
  out.epilogue = DecodeEpilogue::ResidualAddFp32;
  if (auto st = qw38::cuda::launch_decode_mmv(out, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return plan.residual_out;
}

std::expected<TensorView, Error> execute_decode_attention(
    AttentionMixerPlan const& plan, std::uint64_t position) {
  if (auto st = execute_decode_attention_prep(plan.prep, position); !st) {
    return std::unexpected(st.error());
  }
  return execute_attention_core(plan.core);
}

}  // namespace qw38::runtime
