#pragma once

#include "runtime/arena.hpp"
#include "runtime/error.hpp"
#include "runtime/sizes.hpp"
#include "runtime/view.hpp"

#include "cuda/buffer.hpp"
#include "format/constants.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

namespace qw38::cuda {
class Stream;
}

namespace qw38::runtime {

class Model;

struct SessionSnapshot {
  std::vector<std::byte> gdn_s;
  std::vector<std::byte> conv_history;
  std::vector<std::byte> kv;
  std::array<std::uint32_t, kConvLayers> conv_cursor{};
  std::array<std::uint64_t, kAttnLayers> kv_populated{};
};

class Session {
 public:
  Session(Session&&) noexcept = default;
  Session& operator=(Session&&) noexcept = default;
  ~Session() = default;

  Session(Session const&) = delete;
  Session& operator=(Session const&) = delete;

  [[nodiscard]] std::expected<void, Error> reset();
  [[nodiscard]] std::expected<SessionSnapshot, Error> save() const;
  [[nodiscard]] std::expected<void, Error> restore(SessionSnapshot const& snap);

  [[nodiscard]] TensorView gdn_s() const noexcept;
  [[nodiscard]] TensorView conv_history() const noexcept;
  [[nodiscard]] TensorView kv() const noexcept;
  [[nodiscard]] TensorView residual_h() const noexcept;
  [[nodiscard]] TensorView residual_h_mid() const noexcept;
  [[nodiscard]] std::expected<WorkspaceView, Error> scratch(
      qw38::format::ScratchKind kind) const;

  [[nodiscard]] std::uint64_t kv_capacity() const noexcept { return kv_capacity_; }
  [[nodiscard]] std::uint64_t kv_populated(
      std::uint32_t attention_layer) const noexcept {
    return kv_populated_[attention_layer];
  }
  [[nodiscard]] std::expected<void, Error> set_populated_length(
      std::uint32_t attention_layer, std::uint64_t populated);
  [[nodiscard]] std::expected<std::uint64_t*, Error> kv_populated_slot(
      std::uint32_t attention_layer);

  [[nodiscard]] std::array<std::uint32_t, kConvLayers> const& conv_cursor()
      const noexcept {
    return conv_cursor_;
  }
  [[nodiscard]] std::expected<std::uint32_t*, Error> conv_cursor_slot(
      std::uint32_t gdn_layer);
  [[nodiscard]] std::expected<void, Error> set_conv_cursor(
      std::array<std::uint32_t, kConvLayers> cursor);

  [[nodiscard]] ArenaPlan const& arena_plan() const noexcept { return arena_; }
  [[nodiscard]] std::uint64_t persistent_bytes() const noexcept {
    return persistent_bytes_;
  }

 private:
  friend class Runtime;

  Session() = default;

  static std::expected<Session, Error> create(Model const& model,
                                              std::shared_ptr<qw38::cuda::Stream> stream,
                                              std::uint64_t kv_capacity);

  [[nodiscard]] std::expected<void, Error> zero_persistent();

  std::shared_ptr<qw38::cuda::Stream> stream_;
  qw38::cuda::DeviceBuffer gdn_s_;
  qw38::cuda::DeviceBuffer conv_history_;
  qw38::cuda::DeviceBuffer kv_;
  qw38::cuda::DeviceBuffer residual_h_;
  qw38::cuda::DeviceBuffer residual_h_mid_;
  qw38::cuda::DeviceBuffer scratch_;
  ArenaPlan arena_{};
  std::array<std::uint32_t, kConvLayers> conv_cursor_{};
  std::uint64_t kv_capacity_{0};
  std::array<std::uint64_t, kAttnLayers> kv_populated_{};
  std::uint64_t persistent_bytes_{0};
};

}  // namespace qw38::runtime
