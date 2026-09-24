#include "compiler/quantization/reference.hpp"

#include "compiler/quantization/quantizer.hpp"
#include "format/floatcvt.hpp"
#include "format/layout.hpp"
#include "format/unpack.hpp"

#include <cstring>

namespace qw38::compiler {
namespace {

using qw38::format::bf16_to_fp32;
using qw38::format::fp16_to_fp32;
using qw38::format::fp32_to_bf16_rne;
using qw38::format::load_u16_le;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;

CompilerError ref_err(CompilerErrorCode code, std::string_view detail) {
  return make_error(code, "reference", detail);
}

}  // namespace

bool family_uses_q4g64(TensorFamily family) noexcept {
  switch (family) {
    case TensorFamily::LinearAttnInProjQkv:
    case TensorFamily::LinearAttnInProjZ:
    case TensorFamily::LinearAttnOutProj:
    case TensorFamily::SelfAttnQProj:
    case TensorFamily::SelfAttnKProj:
    case TensorFamily::SelfAttnVProj:
    case TensorFamily::SelfAttnOProj:
    case TensorFamily::MlpGateProj:
    case TensorFamily::MlpUpProj:
    case TensorFamily::MlpDownProj:
    case TensorFamily::MtpFc:
    case TensorFamily::MtpSelfAttnQProj:
    case TensorFamily::MtpSelfAttnKProj:
    case TensorFamily::MtpSelfAttnVProj:
    case TensorFamily::MtpSelfAttnOProj:
    case TensorFamily::MtpMlpGateProj:
    case TensorFamily::MtpMlpUpProj:
    case TensorFamily::MtpMlpDownProj:
      return true;
    default:
      return false;
  }
}

bool family_uses_q8g32(TensorFamily family) noexcept {
  return family == TensorFamily::LmHead;
}

WeightFormat select_weight_format(TensorFamily family,
                                  PhysicalLayoutId identity_layout,
                                  WeightFormatPolicy policy) noexcept {
  WeightFormat fmt;
  fmt.layout = identity_layout;
  fmt.storage = StorageClass::Bf16;
  fmt.quantizer = LogicalQuantizerId::None;
  if (policy == WeightFormatPolicy::IdentityBf16) {
    return fmt;
  }
  if (policy == WeightFormatPolicy::CandidateV1) {
    bool const q8 = family == TensorFamily::LmHead ||
                    family == TensorFamily::LinearAttnInProjQkv ||
                    family == TensorFamily::LinearAttnInProjZ ||
                    family == TensorFamily::LinearAttnOutProj ||
                    family == TensorFamily::SelfAttnQProj ||
                    family == TensorFamily::SelfAttnKProj ||
                    family == TensorFamily::SelfAttnVProj ||
                    family == TensorFamily::SelfAttnOProj;
    if (q8) {
      fmt.storage = StorageClass::Int8Grouped;
      fmt.quantizer = LogicalQuantizerId::Q8G32CandidateV1;
      fmt.layout = PhysicalLayoutId::CudaQ8G32CandidateV1;
      return fmt;
    }
    bool const q4 = family == TensorFamily::MlpGateProj ||
                    family == TensorFamily::MlpUpProj ||
                    family == TensorFamily::MlpDownProj;
    if (q4) {
      fmt.storage = StorageClass::Int4Grouped;
      fmt.quantizer = LogicalQuantizerId::Q4G64CandidateV1;
      fmt.layout = PhysicalLayoutId::CudaQ4G64CandidateV1;
      return fmt;
    }
    // Inactive MTP weights keep their V0 representation until activation.
  }
  if (family_uses_q8g32(family)) {
    fmt.storage = StorageClass::Int8Grouped;
    fmt.quantizer = LogicalQuantizerId::Q8G32V0;
    fmt.layout = PhysicalLayoutId::CudaQ8G32V0;
    return fmt;
  }
  if (family_uses_q4g64(family)) {
    fmt.storage = StorageClass::Int4Grouped;
    fmt.quantizer = LogicalQuantizerId::Q4G64V0;
    fmt.layout = PhysicalLayoutId::CudaQ4G64V0;
    return fmt;
  }
  return fmt;
}

std::expected<std::vector<std::uint16_t>, CompilerError> dequantize_to_bf16(
    qw38::format::LogicalWeightCodes const& logical) {
  auto const n = logical.n;
  auto const k = logical.k;
  if (n == 0 || k == 0 || logical.group_size == 0 || k % logical.group_size != 0) {
    return std::unexpected(ref_err(CompilerErrorCode::ShapeMismatch,
                                   "logical geometry is invalid"));
  }
  if (logical.codes.size() != n * k) {
    return std::unexpected(ref_err(CompilerErrorCode::ShapeMismatch,
                                   "code count must equal N*K"));
  }
  auto const groups_row = k / logical.group_size;
  if (logical.scales.size() != n * groups_row) {
    return std::unexpected(ref_err(CompilerErrorCode::ShapeMismatch,
                                   "scale count must equal N*(K/group)"));
  }
  std::vector<std::uint16_t> out(static_cast<std::size_t>(n * k));
  for (std::uint64_t row = 0; row < n; ++row) {
    for (std::uint64_t g = 0; g < groups_row; ++g) {
      auto const scale = fp16_to_fp32(
          logical.scales[static_cast<std::size_t>(row * groups_row + g)]);
      for (std::uint32_t i = 0; i < logical.group_size; ++i) {
        auto const col = g * logical.group_size + i;
        auto const code =
            logical.codes[static_cast<std::size_t>(row * k + col)];
        float decoded = 0.0f;
        if (scale != 0.0f) {
          decoded = static_cast<float>(code) * scale;
        }
        out[static_cast<std::size_t>(row * k + col)] = fp32_to_bf16_rne(decoded);
      }
    }
  }
  return out;
}

std::expected<std::vector<std::uint16_t>, CompilerError> decode_bf16_payload(
    PhysicalLayoutId layout, std::span<std::byte const> payload, std::uint64_t n,
    std::uint64_t k) {
  auto elems = qw38::format::checked_mul(n, k, 0, "reference.bf16.elements");
  if (!elems) {
    return std::unexpected(from_format(elems.error()));
  }
  auto bytes = qw38::format::checked_mul(
      *elems, qw38::format::kBf16Size, 0, "reference.bf16.bytes");
  if (!bytes) {
    return std::unexpected(from_format(bytes.error()));
  }
  if (payload.size() != *bytes) {
    return std::unexpected(ref_err(
        CompilerErrorCode::ShapeMismatch,
        "BF16 payload length must equal the declared 2*N*K bytes"));
  }
  std::vector<std::byte> row_major;
  if (layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    auto unpacked = qw38::format::unpack_bf16_dense_tile_v0(payload, n, k);
    if (!unpacked) {
      return std::unexpected(from_format(unpacked.error()));
    }
    row_major = std::move(*unpacked);
  } else if (layout == PhysicalLayoutId::CudaBf16RowMajorV0 ||
             layout == PhysicalLayoutId::CudaBf16VectorV0) {
    row_major.assign(payload.begin(), payload.end());
  } else {
    return std::unexpected(ref_err(CompilerErrorCode::Internal,
                                   "unsupported BF16 decode layout"));
  }
  std::vector<std::uint16_t> out(static_cast<std::size_t>(*elems));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = load_u16_le(row_major.data() + i * 2);
  }
  return out;
}

std::expected<std::vector<float>, CompilerError> reference_gemv_bf16(
    std::span<std::uint16_t const> w_bf16, std::span<std::uint16_t const> x_bf16,
    std::uint64_t n, std::uint64_t k) {
  if (n == 0 || k == 0 || w_bf16.size() != n * k || x_bf16.size() != k) {
    return std::unexpected(ref_err(CompilerErrorCode::ShapeMismatch,
                                   "GEMV dimensions must match operand lengths"));
  }
  std::vector<float> y(static_cast<std::size_t>(n), 0.0f);
  for (std::uint64_t i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (std::uint64_t j = 0; j < k; ++j) {
      float const w = bf16_to_fp32(w_bf16[static_cast<std::size_t>(i * k + j)]);
      float const x = bf16_to_fp32(x_bf16[static_cast<std::size_t>(j)]);
      acc += w * x;
    }
    y[static_cast<std::size_t>(i)] = acc;
  }
  return y;
}

std::expected<std::vector<float>, CompilerError> reference_gemv_packed(
    qw38::format::PackedMatrix const& packed,
    std::span<std::uint16_t const> x_bf16) {
  if (x_bf16.size() != packed.logical_k) {
    return std::unexpected(ref_err(CompilerErrorCode::ShapeMismatch,
                                   "activation length must equal logical K"));
  }
  std::vector<std::uint16_t> w_bf16;
  if (packed.quantizer == LogicalQuantizerId::None) {
    auto decoded = decode_bf16_payload(packed.layout, packed.codes,
                                       packed.logical_n, packed.logical_k);
    if (!decoded) {
      return std::unexpected(decoded.error());
    }
    w_bf16 = std::move(*decoded);
  } else {
    auto logical = qw38::format::unpack_cuda_v0(
        packed.quantizer, packed.layout, packed.logical_n, packed.logical_k,
        packed.codes, packed.scales);
    if (!logical) {
      return std::unexpected(from_format(logical.error()));
    }
    auto dq = dequantize_to_bf16(*logical);
    if (!dq) {
      return std::unexpected(dq.error());
    }
    w_bf16 = std::move(*dq);
  }
  return reference_gemv_bf16(w_bf16, x_bf16, packed.logical_n, packed.logical_k);
}

}  // namespace qw38::compiler
