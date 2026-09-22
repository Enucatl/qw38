#include "format/constants.hpp"

#include <format>

namespace qw38::format {
namespace {

template <typename E>
std::expected<E, FormatError> decode_known(std::uint16_t raw, std::uint64_t offset,
                                           std::string_view field) {
  E const value{raw};
  if (!is_known(value)) {
    return std::unexpected(make_error(
        FormatErrorCode::UnknownEnum, offset, field,
        std::format("unsupported enumerator raw=0x{:04X}", raw)));
  }
  return value;
}

}  // namespace

bool is_known(StorageClass value) noexcept {
  switch (value) {
    case StorageClass::Int4Grouped:
    case StorageClass::Int8Grouped:
    case StorageClass::Bf16:
    case StorageClass::Fp32:
      return true;
  }
  return false;
}

bool is_known(LogicalQuantizerId value) noexcept {
  switch (value) {
    case LogicalQuantizerId::None:
    case LogicalQuantizerId::Q4G64V0:
    case LogicalQuantizerId::Q8G32V0:
      return true;
  }
  return false;
}

bool is_known(PhysicalLayoutId value) noexcept {
  switch (value) {
    case PhysicalLayoutId::CudaQ4G64V0:
    case PhysicalLayoutId::CudaQ8G32V0:
    case PhysicalLayoutId::CudaBf16DenseTileV0:
    case PhysicalLayoutId::CudaBf16RowMajorV0:
    case PhysicalLayoutId::CudaBf16VectorV0:
    case PhysicalLayoutId::CudaBf16TapMajorV0:
    case PhysicalLayoutId::CudaFp32VectorV0:
    case PhysicalLayoutId::CudaFp32GdnSHvKV0:
    case PhysicalLayoutId::CudaBf16ConvHistoryV0:
    case PhysicalLayoutId::CudaBf16KvCacheV0:
      return true;
  }
  return false;
}

bool is_known(SemanticNodeKind value) noexcept {
  switch (value) {
    case SemanticNodeKind::Embed:
    case SemanticNodeKind::GatedAttention:
    case SemanticNodeKind::GatedDeltaNet:
    case SemanticNodeKind::Mlp:
    case SemanticNodeKind::LmHead:
    case SemanticNodeKind::MtpMix:
      return true;
  }
  return false;
}

bool is_known(PrecisionPolicyId value) noexcept {
  return value == PrecisionPolicyId::V0;
}

bool is_known(SemanticScope value) noexcept {
  switch (value) {
    case SemanticScope::PrimaryLanguage:
    case SemanticScope::LanguagePlusMtpDescriptors:
      return true;
  }
  return false;
}

bool is_known(TensorRole value) noexcept {
  switch (value) {
    case TensorRole::DenseWeight:
    case TensorRole::EmbeddingTable:
    case TensorRole::LmHeadWeight:
    case TensorRole::NormGamma:
    case TensorRole::ConvWeight:
    case TensorRole::VectorWeight:
    case TensorRole::TimeParameter:
    case TensorRole::AdditiveNorm:
    case TensorRole::QkNorm:
    case TensorRole::GdnGatedNorm:
    case TensorRole::FinalLanguageNorm:
    case TensorRole::MtpNorm:
      return true;
  }
  return false;
}

bool is_known(MappingKind value) noexcept {
  switch (value) {
    case MappingKind::Identity:
    case MappingKind::DenseTileNK:
      return true;
  }
  return false;
}

bool is_known(IntegrityKind value) noexcept {
  switch (value) {
    case IntegrityKind::Sha256Manifest:
    case IntegrityKind::Sha256PayloadSpan:
    case IntegrityKind::Sha256ScaleSpan:
      return true;
  }
  return false;
}

bool is_known(SharedBindingRole value) noexcept {
  switch (value) {
    case SharedBindingRole::GenericAlias:
    case SharedBindingRole::MtpEmbeddingAlias:
    case SharedBindingRole::MtpLmHeadAlias:
      return true;
  }
  return false;
}

bool is_known(ArithmeticDtype value) noexcept {
  switch (value) {
    case ArithmeticDtype::Fp32:
    case ArithmeticDtype::Bf16:
    case ArithmeticDtype::Fp16:
      return true;
  }
  return false;
}

bool is_known(PrecisionDomain value) noexcept {
  switch (value) {
    case PrecisionDomain::ResidualStream:
    case PrecisionDomain::NormalizedActivation:
    case PrecisionDomain::ProjectionStaging:
    case PrecisionDomain::DotProductAccumulator:
    case PrecisionDomain::NonlinearIntermediate:
    case PrecisionDomain::GdnRecurrentState:
    case PrecisionDomain::ConvolutionHistory:
    case PrecisionDomain::KvCache:
    case PrecisionDomain::Logits:
    case PrecisionDomain::WeightScale:
      return true;
  }
  return false;
}

bool is_known(StateKind value) noexcept {
  switch (value) {
    case StateKind::GdnS:
    case StateKind::ConvolutionHistory:
    case StateKind::KvCache:
      return true;
  }
  return false;
}

bool is_known(ScratchKind value) noexcept {
  switch (value) {
    case ScratchKind::ResidualH:
    case ScratchKind::ResidualHMid:
    case ScratchKind::NormalizedHidden:
    case ScratchKind::GdnWorkspace:
    case ScratchKind::AttentionWorkspace:
    case ScratchKind::MlpSwiglu:
    case ScratchKind::Logits:
      return true;
  }
  return false;
}

char const* name_of(StorageClass value) noexcept {
  switch (value) {
    case StorageClass::Int4Grouped:
      return "int4_grouped";
    case StorageClass::Int8Grouped:
      return "int8_grouped";
    case StorageClass::Bf16:
      return "bf16";
    case StorageClass::Fp32:
      return "fp32";
  }
  return "unknown_storage";
}

char const* name_of(LogicalQuantizerId value) noexcept {
  switch (value) {
    case LogicalQuantizerId::None:
      return "none";
    case LogicalQuantizerId::Q4G64V0:
      return "q4g64_v0";
    case LogicalQuantizerId::Q8G32V0:
      return "q8g32_v0";
  }
  return "unknown_quantizer";
}

char const* name_of(PhysicalLayoutId value) noexcept {
  switch (value) {
    case PhysicalLayoutId::CudaQ4G64V0:
      return "cuda_q4g64_v0";
    case PhysicalLayoutId::CudaQ8G32V0:
      return "cuda_q8g32_v0";
    case PhysicalLayoutId::CudaBf16DenseTileV0:
      return "cuda_bf16_dense_tile_v0";
    case PhysicalLayoutId::CudaBf16RowMajorV0:
      return "cuda_bf16_row_major_v0";
    case PhysicalLayoutId::CudaBf16VectorV0:
      return "cuda_bf16_vector_v0";
    case PhysicalLayoutId::CudaBf16TapMajorV0:
      return "cuda_bf16_tap_major_v0";
    case PhysicalLayoutId::CudaFp32VectorV0:
      return "cuda_fp32_vector_v0";
    case PhysicalLayoutId::CudaFp32GdnSHvKV0:
      return "cuda_fp32_gdn_s_hvk_v0";
    case PhysicalLayoutId::CudaBf16ConvHistoryV0:
      return "cuda_bf16_conv_history_v0";
    case PhysicalLayoutId::CudaBf16KvCacheV0:
      return "cuda_bf16_kv_cache_v0";
  }
  return "unknown_layout";
}

char const* name_of(SemanticNodeKind value) noexcept {
  switch (value) {
    case SemanticNodeKind::Embed:
      return "embed";
    case SemanticNodeKind::GatedAttention:
      return "gated_attention";
    case SemanticNodeKind::GatedDeltaNet:
      return "gated_delta_net";
    case SemanticNodeKind::Mlp:
      return "mlp";
    case SemanticNodeKind::LmHead:
      return "lm_head";
    case SemanticNodeKind::MtpMix:
      return "mtp_mix";
  }
  return "unknown_semantic_node";
}

char const* name_of(PrecisionPolicyId value) noexcept {
  switch (value) {
    case PrecisionPolicyId::V0:
      return "precision_v0";
  }
  return "unknown_precision_policy";
}

char const* name_of(SemanticScope value) noexcept {
  switch (value) {
    case SemanticScope::PrimaryLanguage:
      return "primary_language";
    case SemanticScope::LanguagePlusMtpDescriptors:
      return "language_plus_mtp_descriptors";
  }
  return "unknown_scope";
}

std::expected<StorageClass, FormatError> decode_storage_class(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<StorageClass>(raw, offset, field);
}

std::expected<LogicalQuantizerId, FormatError> decode_logical_quantizer(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  auto value = decode_known<LogicalQuantizerId>(raw, offset, field);
  if (!value) {
    auto error = value.error();
    error.code = FormatErrorCode::UnsupportedQuantizerVersion;
    return std::unexpected(std::move(error));
  }
  return value;
}

std::expected<PhysicalLayoutId, FormatError> decode_physical_layout(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  auto value = decode_known<PhysicalLayoutId>(raw, offset, field);
  if (!value) {
    auto error = value.error();
    error.code = FormatErrorCode::UnsupportedPhysicalLayoutVersion;
    return std::unexpected(std::move(error));
  }
  return value;
}

std::expected<SemanticNodeKind, FormatError> decode_semantic_node(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<SemanticNodeKind>(raw, offset, field);
}

std::expected<PrecisionPolicyId, FormatError> decode_precision_policy_id(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<PrecisionPolicyId>(raw, offset, field);
}

std::expected<SemanticScope, FormatError> decode_semantic_scope(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<SemanticScope>(raw, offset, field);
}

std::expected<TensorRole, FormatError> decode_tensor_role(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<TensorRole>(raw, offset, field);
}

std::expected<MappingKind, FormatError> decode_mapping_kind(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<MappingKind>(raw, offset, field);
}

std::expected<IntegrityKind, FormatError> decode_integrity_kind(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<IntegrityKind>(raw, offset, field);
}

std::expected<SharedBindingRole, FormatError> decode_shared_binding_role(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<SharedBindingRole>(raw, offset, field);
}

std::expected<ArithmeticDtype, FormatError> decode_arithmetic_dtype(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<ArithmeticDtype>(raw, offset, field);
}

std::expected<PrecisionDomain, FormatError> decode_precision_domain(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<PrecisionDomain>(raw, offset, field);
}

std::expected<StateKind, FormatError> decode_state_kind(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<StateKind>(raw, offset, field);
}

std::expected<ScratchKind, FormatError> decode_scratch_kind(
    std::uint16_t raw, std::uint64_t offset, std::string_view field) {
  return decode_known<ScratchKind>(raw, offset, field);
}

std::uint64_t element_size(ArithmeticDtype dtype) noexcept {
  switch (dtype) {
    case ArithmeticDtype::Fp32:
      return kFp32Size;
    case ArithmeticDtype::Bf16:
      return kBf16Size;
    case ArithmeticDtype::Fp16:
      return kFp16Size;
  }
  return 0;
}

bool layout_is_weight(PhysicalLayoutId layout) noexcept {
  switch (layout) {
    case PhysicalLayoutId::CudaQ4G64V0:
    case PhysicalLayoutId::CudaQ8G32V0:
    case PhysicalLayoutId::CudaBf16DenseTileV0:
    case PhysicalLayoutId::CudaBf16RowMajorV0:
    case PhysicalLayoutId::CudaBf16VectorV0:
    case PhysicalLayoutId::CudaBf16TapMajorV0:
    case PhysicalLayoutId::CudaFp32VectorV0:
      return true;
    case PhysicalLayoutId::CudaFp32GdnSHvKV0:
    case PhysicalLayoutId::CudaBf16ConvHistoryV0:
    case PhysicalLayoutId::CudaBf16KvCacheV0:
      return false;
  }
  return false;
}

bool layout_is_state(PhysicalLayoutId layout) noexcept {
  switch (layout) {
    case PhysicalLayoutId::CudaFp32GdnSHvKV0:
    case PhysicalLayoutId::CudaBf16ConvHistoryV0:
    case PhysicalLayoutId::CudaBf16KvCacheV0:
      return true;
    default:
      return false;
  }
}

bool layout_is_tiled_dense(PhysicalLayoutId layout) noexcept {
  switch (layout) {
    case PhysicalLayoutId::CudaQ4G64V0:
    case PhysicalLayoutId::CudaQ8G32V0:
    case PhysicalLayoutId::CudaBf16DenseTileV0:
      return true;
    default:
      return false;
  }
}

}  // namespace qw38::format
