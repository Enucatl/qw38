#pragma once

#include "format/error.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <string_view>

namespace qw38::format {

// Container magic: ASCII "QW38FMT" plus a terminating NUL. Not a C++ object dump.
inline constexpr std::array<std::uint8_t, 8> kMagic{
    'Q', 'W', '3', '8', 'F', 'M', 'T', '\0'};

inline constexpr std::uint16_t kContainerVersionV0 = 1;
inline constexpr std::uint16_t kManifestVersionV0 = 1;
inline constexpr std::uint16_t kHeaderSizeV0 = 64;
inline constexpr std::uint64_t kSpanAlignment = 256;
inline constexpr std::uint8_t kMaxRank = 8;
inline constexpr std::uint16_t kMaxNameBytes = 1024;
inline constexpr std::uint32_t kHashBytes = 32;
inline constexpr std::uint32_t kIntegrityRecordBytes = 56;
inline constexpr std::uint32_t kNoLayerIndex = 0xFFFFFFFFu;

// V0 describes one fixed Qwen3.8-family model. These limits leave ample room
// for descriptors and aliases while keeping hostile wire counts and manifests
// from driving unbounded host allocations.
inline constexpr std::uint64_t kMaxManifestBytesV0 = 16 * 1024 * 1024;
inline constexpr std::uint16_t kMaxPrecisionBindingsV0 = 32;
inline constexpr std::uint32_t kMaxTensorRecordsV0 = 4096;
inline constexpr std::uint32_t kMaxSharedBindingsV0 = 4096;
inline constexpr std::uint32_t kMaxGraphBindingsV0 = 16384;
inline constexpr std::uint32_t kMaxStateAllocationsV0 = 16;
inline constexpr std::uint32_t kMaxScratchAllocationsV0 = 64;
inline constexpr std::uint32_t kMaxIntegrityRecordsV0 =
    2 * kMaxTensorRecordsV0 + 1;

inline constexpr std::uint64_t kBf16Size = 2;
inline constexpr std::uint64_t kFp16Size = 2;
inline constexpr std::uint64_t kFp32Size = 4;
inline constexpr std::uint32_t kDenseTileRows = 8;
inline constexpr std::uint32_t kDenseTileK = 256;
inline constexpr std::uint32_t kQ4GroupSize = 64;
inline constexpr std::uint32_t kQ4KGroupSize = 256;
inline constexpr std::uint32_t kQ4KMetadataBytes = 16;
inline constexpr std::uint32_t kQ8GroupSize = 32;
inline constexpr std::uint32_t kQ4PackedBytesPerTileRow = 128;
inline constexpr std::uint32_t kQ8PackedBytesPerTileRow = 256;
inline constexpr std::uint32_t kBf16PackedBytesPerTileRow = 512;

// Enumerators occupy disjoint wire ranges so a logical quantizer ID cannot be
// mistaken for a physical layout ID (A-02).

enum class StorageClass : std::uint16_t {
  Int4Grouped = 0x0001,
  Int8Grouped = 0x0002,
  Bf16 = 0x0003,
  Fp32 = 0x0004,
  NvFp4 = 0x0005,
};

enum class LogicalQuantizerId : std::uint16_t {
  None = 0x0100,
  Q4G64V0 = 0x0101,
  Q8G32V0 = 0x0102,
  Q4G64CandidateV1 = 0x0103,
  Q8G32CandidateV1 = 0x0104,
  Q4KCandidateV2 = 0x0105,
  NvFp4V1 = 0x0106,
};

enum class PhysicalLayoutId : std::uint16_t {
  CudaQ4G64V0 = 0x0201,
  CudaQ8G32V0 = 0x0202,
  CudaBf16DenseTileV0 = 0x0203,
  CudaBf16RowMajorV0 = 0x0204,
  CudaBf16VectorV0 = 0x0205,
  CudaBf16TapMajorV0 = 0x0206,
  CudaFp32VectorV0 = 0x020A,
  CudaFp32GdnSHvKV0 = 0x0207,
  CudaBf16ConvHistoryV0 = 0x0208,
  CudaBf16KvCacheV0 = 0x0209,
  CudaQ4G64CandidateV1 = 0x020B,
  CudaQ8G32CandidateV1 = 0x020C,
  CudaQ4KCandidateV2 = 0x020D,
  CudaNvFp4V1 = 0x020E,
};

enum class SemanticNodeKind : std::uint16_t {
  Embed = 0x0301,
  GatedAttention = 0x0302,
  GatedDeltaNet = 0x0303,
  Mlp = 0x0304,
  LmHead = 0x0305,
  MtpMix = 0x0306,
};

enum class PrecisionPolicyId : std::uint16_t {
  V0 = 0x0401,
  CandidateV1 = 0x0402,
  CandidateV2 = 0x0403,
  NvFp4MlpV1 = 0x0404,
};

enum class SemanticScope : std::uint16_t {
  PrimaryLanguage = 0x0501,
  LanguagePlusMtpDescriptors = 0x0502,
};

enum class TensorRole : std::uint16_t {
  DenseWeight = 0x0601,
  EmbeddingTable = 0x0602,
  LmHeadWeight = 0x0603,
  NormGamma = 0x0604,
  ConvWeight = 0x0605,
  VectorWeight = 0x0606,
  TimeParameter = 0x0607,
  AdditiveNorm = 0x0608,
  QkNorm = 0x0609,
  GdnGatedNorm = 0x060A,
  FinalLanguageNorm = 0x060B,
  MtpNorm = 0x060C,
};

enum class MappingKind : std::uint16_t {
  Identity = 0x0701,
  DenseTileNK = 0x0702,
  // Logical [channel, 1, tap] is stored as physical [tap, channel].
  TapMajorConvC1T = 0x0703,
  NvFp4TN = 0x0704,
};

enum class IntegrityKind : std::uint16_t {
  Sha256Manifest = 0x0801,
  Sha256PayloadSpan = 0x0802,
  Sha256ScaleSpan = 0x0803,
};

enum class SharedBindingRole : std::uint16_t {
  GenericAlias = 0x0901,
  MtpEmbeddingAlias = 0x0902,
  MtpLmHeadAlias = 0x0903,
};

enum class ArithmeticDtype : std::uint16_t {
  Fp32 = 0x0A01,
  Bf16 = 0x0A02,
  Fp16 = 0x0A03,
};

enum class PrecisionDomain : std::uint16_t {
  ResidualStream = 0x0B01,
  NormalizedActivation = 0x0B02,
  ProjectionStaging = 0x0B03,
  DotProductAccumulator = 0x0B04,
  NonlinearIntermediate = 0x0B05,
  GdnRecurrentState = 0x0B06,
  ConvolutionHistory = 0x0B07,
  KvCache = 0x0B08,
  Logits = 0x0B09,
  WeightScale = 0x0B0A,
};

enum class StateKind : std::uint16_t {
  GdnS = 0x0C01,
  ConvolutionHistory = 0x0C02,
  KvCache = 0x0C03,
};

enum class ScratchKind : std::uint16_t {
  ResidualH = 0x0D01,
  ResidualHMid = 0x0D02,
  NormalizedHidden = 0x0D03,
  GdnWorkspace = 0x0D04,
  AttentionWorkspace = 0x0D05,
  MlpSwiglu = 0x0D06,
  Logits = 0x0D07,
};

[[nodiscard]] bool is_known(StorageClass value) noexcept;
[[nodiscard]] bool is_known(LogicalQuantizerId value) noexcept;
[[nodiscard]] bool is_known(PhysicalLayoutId value) noexcept;
[[nodiscard]] bool is_known(SemanticNodeKind value) noexcept;
[[nodiscard]] bool is_known(PrecisionPolicyId value) noexcept;
[[nodiscard]] bool is_known(SemanticScope value) noexcept;
[[nodiscard]] bool is_known(TensorRole value) noexcept;
[[nodiscard]] bool is_known(MappingKind value) noexcept;
[[nodiscard]] bool is_known(IntegrityKind value) noexcept;
[[nodiscard]] bool is_known(SharedBindingRole value) noexcept;
[[nodiscard]] bool is_known(ArithmeticDtype value) noexcept;
[[nodiscard]] bool is_known(PrecisionDomain value) noexcept;
[[nodiscard]] bool is_known(StateKind value) noexcept;
[[nodiscard]] bool is_known(ScratchKind value) noexcept;

[[nodiscard]] char const* name_of(StorageClass value) noexcept;
[[nodiscard]] char const* name_of(LogicalQuantizerId value) noexcept;
[[nodiscard]] char const* name_of(PhysicalLayoutId value) noexcept;
[[nodiscard]] char const* name_of(SemanticNodeKind value) noexcept;
[[nodiscard]] char const* name_of(PrecisionPolicyId value) noexcept;
[[nodiscard]] char const* name_of(SemanticScope value) noexcept;

[[nodiscard]] std::expected<StorageClass, FormatError>
decode_storage_class(std::uint16_t raw, std::uint64_t offset,
                     std::string_view field);

[[nodiscard]] std::expected<LogicalQuantizerId, FormatError>
decode_logical_quantizer(std::uint16_t raw, std::uint64_t offset,
                         std::string_view field);

[[nodiscard]] std::expected<PhysicalLayoutId, FormatError>
decode_physical_layout(std::uint16_t raw, std::uint64_t offset,
                       std::string_view field);

[[nodiscard]] std::expected<SemanticNodeKind, FormatError>
decode_semantic_node(std::uint16_t raw, std::uint64_t offset,
                     std::string_view field);

[[nodiscard]] std::expected<PrecisionPolicyId, FormatError>
decode_precision_policy_id(std::uint16_t raw, std::uint64_t offset,
                           std::string_view field);

[[nodiscard]] std::expected<SemanticScope, FormatError>
decode_semantic_scope(std::uint16_t raw, std::uint64_t offset,
                      std::string_view field);

[[nodiscard]] std::expected<TensorRole, FormatError>
decode_tensor_role(std::uint16_t raw, std::uint64_t offset,
                   std::string_view field);

[[nodiscard]] std::expected<MappingKind, FormatError>
decode_mapping_kind(std::uint16_t raw, std::uint64_t offset,
                    std::string_view field);

[[nodiscard]] std::expected<IntegrityKind, FormatError>
decode_integrity_kind(std::uint16_t raw, std::uint64_t offset,
                      std::string_view field);

[[nodiscard]] std::expected<SharedBindingRole, FormatError>
decode_shared_binding_role(std::uint16_t raw, std::uint64_t offset,
                           std::string_view field);

[[nodiscard]] std::expected<ArithmeticDtype, FormatError>
decode_arithmetic_dtype(std::uint16_t raw, std::uint64_t offset,
                        std::string_view field);

[[nodiscard]] std::expected<PrecisionDomain, FormatError>
decode_precision_domain(std::uint16_t raw, std::uint64_t offset,
                        std::string_view field);

[[nodiscard]] std::expected<StateKind, FormatError>
decode_state_kind(std::uint16_t raw, std::uint64_t offset,
                  std::string_view field);

[[nodiscard]] std::expected<ScratchKind, FormatError>
decode_scratch_kind(std::uint16_t raw, std::uint64_t offset,
                    std::string_view field);

[[nodiscard]] std::uint64_t element_size(ArithmeticDtype dtype) noexcept;
[[nodiscard]] bool layout_is_weight(PhysicalLayoutId layout) noexcept;
[[nodiscard]] bool layout_is_state(PhysicalLayoutId layout) noexcept;
[[nodiscard]] bool layout_is_tiled_dense(PhysicalLayoutId layout) noexcept;

}  // namespace qw38::format
