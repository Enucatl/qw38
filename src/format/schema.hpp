#pragma once

#include "format/constants.hpp"
#include "format/error.hpp"
#include "format/layout.hpp"
#include "format/wire.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace qw38::format {

struct Hash256 {
  std::array<std::uint8_t, 32> bytes{};

  friend bool operator==(Hash256 const&, Hash256 const&) = default;
};

struct ByteSpan {
  std::uint64_t offset{};
  std::uint64_t length{};

  [[nodiscard]] bool empty() const noexcept { return length == 0; }

  friend bool operator==(ByteSpan const&, ByteSpan const&) = default;
};

struct TensorShape {
  std::uint8_t rank{};
  std::array<std::uint64_t, kMaxRank> logical{};
  std::array<std::uint64_t, kMaxRank> padded{};

  [[nodiscard]] std::span<std::uint64_t const> logical_dims() const noexcept {
    return std::span<std::uint64_t const>{logical.data(), rank};
  }
  [[nodiscard]] std::span<std::uint64_t const> padded_dims() const noexcept {
    return std::span<std::uint64_t const>{padded.data(), rank};
  }

  friend bool operator==(TensorShape const&, TensorShape const&) = default;
};

struct LogicalPhysicalMapping {
  MappingKind kind{MappingKind::Identity};
  std::uint32_t tile_rows{};
  std::uint32_t tile_k{};
  std::uint32_t group_size{};
  std::uint32_t packed_bytes_per_tile_row{};

  friend bool operator==(LogicalPhysicalMapping const&,
                         LogicalPhysicalMapping const&) = default;
};

struct ContainerHeader {
  std::array<std::uint8_t, 8> magic{kMagic};
  std::uint16_t container_version{kContainerVersionV0};
  std::uint16_t manifest_version{kManifestVersionV0};
  std::uint16_t header_bytes{kHeaderSizeV0};
  std::uint64_t manifest_offset{};
  std::uint64_t manifest_length{};

  friend bool operator==(ContainerHeader const&,
                         ContainerHeader const&) = default;
};

struct CompilerRevision {
  std::string ident;
  std::uint32_t major{};
  std::uint32_t minor{};
  std::uint32_t patch{};

  friend bool operator==(CompilerRevision const&,
                         CompilerRevision const&) = default;
};

struct PrecisionBinding {
  PrecisionDomain domain{};
  ArithmeticDtype dtype{};

  friend bool operator==(PrecisionBinding const&,
                         PrecisionBinding const&) = default;
};

struct PrecisionPolicyRecord {
  PrecisionPolicyId id{PrecisionPolicyId::V0};
  std::vector<PrecisionBinding> bindings;

  friend bool operator==(PrecisionPolicyRecord const&,
                         PrecisionPolicyRecord const&) = default;
};

struct TensorRecord {
  std::uint32_t tensor_id{};
  std::string logical_name;
  TensorShape shape{};
  StorageClass storage{StorageClass::Bf16};
  LogicalQuantizerId quantizer{LogicalQuantizerId::None};
  PhysicalLayoutId layout{PhysicalLayoutId::CudaBf16RowMajorV0};
  LogicalPhysicalMapping mapping{};
  ByteSpan payload{};
  ByteSpan scales{};

  friend bool operator==(TensorRecord const&, TensorRecord const&) = default;
};

struct SharedBinding {
  std::uint32_t owner_tensor_id{};
  std::uint32_t alias_tensor_id{};
  SharedBindingRole role{SharedBindingRole::GenericAlias};

  friend bool operator==(SharedBinding const&, SharedBinding const&) = default;
};

struct GraphBinding {
  std::uint32_t instance_id{};
  SemanticNodeKind kind{SemanticNodeKind::Embed};
  TensorRole role{TensorRole::DenseWeight};
  std::uint32_t layer_index{kNoLayerIndex};
  std::uint32_t tensor_id{};

  friend bool operator==(GraphBinding const&, GraphBinding const&) = default;
};

struct StateAllocation {
  StateKind kind{StateKind::GdnS};
  ArithmeticDtype dtype{ArithmeticDtype::Fp32};
  PhysicalLayoutId layout{PhysicalLayoutId::CudaFp32GdnSHvKV0};
  TensorShape shape_per_layer{};
  std::uint32_t layer_count{};
  std::uint32_t component_count{1};
  std::uint64_t declared_capacity{};
  std::uint64_t bytes_per_layer{};
  std::uint64_t bytes_per_token{};
  std::uint64_t total_bytes{};
  bool populated_length_distinct_from_capacity{};
  bool live_payload_present{};

  friend bool operator==(StateAllocation const&,
                         StateAllocation const&) = default;
};

struct ScratchAllocation {
  ScratchKind kind{ScratchKind::ResidualH};
  ArithmeticDtype dtype{ArithmeticDtype::Fp32};
  std::uint64_t bytes{};

  friend bool operator==(ScratchAllocation const&,
                         ScratchAllocation const&) = default;
};

struct IntegrityRecord {
  IntegrityKind kind{IntegrityKind::Sha256Manifest};
  std::uint32_t tensor_id{};
  ByteSpan region{};
  Hash256 digest{};

  friend bool operator==(IntegrityRecord const&,
                         IntegrityRecord const&) = default;
};

struct ArtifactSchema {
  std::uint16_t manifest_version{kManifestVersionV0};
  CompilerRevision compiler{};
  Hash256 source_hash{};
  Hash256 config_hash{};
  Hash256 tokenizer_hash{};
  PrecisionPolicyRecord precision{};
  SemanticScope scope{SemanticScope::PrimaryLanguage};
  std::vector<TensorRecord> tensors;
  std::vector<SharedBinding> shared_bindings;
  std::vector<GraphBinding> graph_bindings;
  std::vector<StateAllocation> state;
  std::vector<ScratchAllocation> scratch;
  std::vector<IntegrityRecord> integrity;

  friend bool operator==(ArtifactSchema const&,
                         ArtifactSchema const&) = default;
};

[[nodiscard]] PrecisionPolicyRecord v0_precision_policy();

[[nodiscard]] std::expected<std::uint64_t, FormatError> expected_payload_bytes(
    TensorRecord const& tensor, std::uint64_t offset = 0);
[[nodiscard]] std::expected<std::uint64_t, FormatError> expected_scale_bytes(
    TensorRecord const& tensor, std::uint64_t offset = 0);

// V0 shared bindings are a one-level ownership graph: every alias has exactly
// one canonical owner, and that owner must not itself be an alias.
[[nodiscard]] std::expected<void, FormatError>
validate_shared_binding_ownership(ArtifactSchema const& schema,
                                  std::uint64_t offset = 0);
[[nodiscard]] std::expected<std::uint32_t, FormatError>
canonical_owner_tensor_id(ArtifactSchema const& schema,
                          std::uint32_t tensor_id,
                          std::uint64_t offset = 0);

[[nodiscard]] std::expected<std::size_t, FormatError> encoded_size(
    ContainerHeader const& header);
[[nodiscard]] std::expected<void, FormatError> encode(
    ContainerHeader const& header, std::span<std::byte> out);
[[nodiscard]] std::expected<ContainerHeader, FormatError> decode_header(
    std::span<std::byte const> in);
[[nodiscard]] std::expected<void, FormatError> validate_header(
    ContainerHeader const& header, std::uint64_t offset = 0);

[[nodiscard]] std::expected<std::size_t, FormatError> encoded_size(
    ArtifactSchema const& schema);
[[nodiscard]] std::expected<void, FormatError> encode(
    ArtifactSchema const& schema, std::span<std::byte> out);
[[nodiscard]] std::expected<ArtifactSchema, FormatError> decode_schema(
    std::span<std::byte const> in);
[[nodiscard]] std::expected<void, FormatError> validate_schema(
    ArtifactSchema const& schema, std::uint64_t offset = 0);

}  // namespace qw38::format
