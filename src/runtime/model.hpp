#pragma once

#include "runtime/error.hpp"
#include "runtime/view.hpp"

#include "cuda/buffer.hpp"
#include "format/reader.hpp"

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace qw38::cuda {
class Stream;
}

namespace qw38::runtime {

enum class DiagnosticWeights : std::uint8_t { Input, Layer, Output };

class Model {
 public:
  Model(Model&&) noexcept = default;
  Model& operator=(Model&&) noexcept = default;
  ~Model() = default;

  Model(Model const&) = delete;
  Model& operator=(Model const&) = delete;

  [[nodiscard]] qw38::format::ArtifactSchema const& schema() const noexcept {
    return schema_;
  }
  [[nodiscard]] std::span<qw38::format::TensorRecord const> tensors()
      const noexcept {
    return schema_.tensors;
  }
  [[nodiscard]] std::span<qw38::format::StateAllocation const> state()
      const noexcept {
    return schema_.state;
  }
  [[nodiscard]] std::span<qw38::format::ScratchAllocation const> scratch()
      const noexcept {
    return schema_.scratch;
  }
  [[nodiscard]] std::span<qw38::format::GraphBinding const> graph_bindings()
      const noexcept {
    return schema_.graph_bindings;
  }

  [[nodiscard]] qw38::format::TensorRecord const* find_tensor(
      std::string_view logical_name) const noexcept;
  // V0 compatibility boundary: resolve a canonical logical name to the
  // artifact tensor ID and require its exact graph role/layer membership.
  [[nodiscard]] std::expected<std::uint32_t, Error> resolve_tensor(
      std::string_view logical_name, qw38::format::SemanticNodeKind kind,
      qw38::format::TensorRole role, std::uint32_t layer) const;
  [[nodiscard]] std::expected<ConstTensorView, Error> payload(
      std::uint32_t tensor_id) const;
  [[nodiscard]] std::expected<ConstTensorView, Error> scales(
      std::uint32_t tensor_id) const;
  [[nodiscard]] std::expected<ConstTensorView, Error> payload(
      std::string_view logical_name) const;
  [[nodiscard]] std::expected<ConstTensorView, Error> scales(
      std::string_view logical_name) const;

  [[nodiscard]] std::uint64_t device_bytes() const noexcept { return device_bytes_; }
  [[nodiscard]] int device() const noexcept { return device_; }

 private:
  friend class Runtime;

  Model() = default;

  static std::expected<Model, Error> upload(qw38::format::Artifact const& artifact,
                                            qw38::cuda::Stream const& stream,
                                            std::optional<DiagnosticWeights> selection = {},
                                            std::uint32_t layer = 0);

  struct UploadedTensor {
    std::uint32_t tensor_id{};
    ConstTensorView payload{};
    ConstTensorView scales{};
  };

  qw38::format::ArtifactSchema schema_{};
  std::vector<qw38::cuda::DeviceBuffer> buffers_;
  std::vector<UploadedTensor> uploaded_;
  std::uint64_t device_bytes_{0};
  int device_{-1};
};

// Also usable by host-only schema tests before an artifact is uploaded.
[[nodiscard]] std::expected<std::uint32_t, Error> resolve_semantic_tensor(
    qw38::format::ArtifactSchema const& schema, std::string_view logical_name,
    qw38::format::SemanticNodeKind kind, qw38::format::TensorRole role,
    std::uint32_t layer);

}  // namespace qw38::runtime
