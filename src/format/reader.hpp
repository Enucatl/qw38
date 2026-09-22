#pragma once

#include "format/error.hpp"
#include "format/schema.hpp"
#include "format/writer.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace qw38::format {

// Host-only validated artifact. Construction succeeds only after structural
// and integrity checks; payload/scale views are never available before that.
class Artifact {
 public:
  Artifact(Artifact&&) noexcept;
  Artifact& operator=(Artifact&&) noexcept;
  ~Artifact();

  Artifact(Artifact const&) = delete;
  Artifact& operator=(Artifact const&) = delete;

  [[nodiscard]] static std::expected<Artifact, FormatError> open(
      std::filesystem::path path);
  [[nodiscard]] static std::expected<Artifact, FormatError> parse(
      std::vector<std::byte> const& bytes);
  [[nodiscard]] static std::expected<Artifact, FormatError> parse(
      std::vector<std::byte>&& bytes);
  [[nodiscard]] static std::expected<Artifact, FormatError> parse(
      std::span<std::byte const> bytes);

  [[nodiscard]] ArtifactIdentity const& identity() const noexcept;
  [[nodiscard]] ContainerHeader const& header() const noexcept;
  [[nodiscard]] ArtifactSchema const& schema() const noexcept;
  [[nodiscard]] CompilerRevision const& compiler() const noexcept;
  [[nodiscard]] PrecisionPolicyRecord const& precision() const noexcept;
  [[nodiscard]] SemanticScope scope() const noexcept;
  [[nodiscard]] std::span<StateAllocation const> state() const noexcept;
  [[nodiscard]] std::span<ScratchAllocation const> scratch() const noexcept;
  [[nodiscard]] std::span<GraphBinding const> graph_bindings() const noexcept;
  [[nodiscard]] std::span<SharedBinding const> shared_bindings() const noexcept;
  [[nodiscard]] std::span<TensorRecord const> tensors() const noexcept;

  [[nodiscard]] TensorRecord const* find_tensor(
      std::string_view logical_name) const noexcept;
  [[nodiscard]] TensorRecord const* find_tensor(
      std::uint32_t tensor_id) const noexcept;

  // Borrowed views are valid for the lifetime of this Artifact.
  [[nodiscard]] std::expected<std::span<std::byte const>, FormatError> payload(
      std::string_view logical_name) const;
  [[nodiscard]] std::expected<std::span<std::byte const>, FormatError> scales(
      std::string_view logical_name) const;
  [[nodiscard]] std::expected<std::span<std::byte const>, FormatError> span(
      std::string_view logical_name, SpanKind kind) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit Artifact(std::unique_ptr<Impl> impl) noexcept;

  [[nodiscard]] std::span<std::byte const> mapped_bytes() const noexcept;
  [[nodiscard]] std::span<std::byte const> view(ByteSpan region) const noexcept;
};

}  // namespace qw38::format
