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
class Session;

namespace detail {
struct SessionPlanAccess;
}

class KvPopulatedSlot {
 public:
  KvPopulatedSlot() = default;

  [[nodiscard]] static std::expected<KvPopulatedSlot, Error> bind(
      std::uint64_t* value, std::uint64_t capacity);
  [[nodiscard]] std::expected<std::uint64_t, Error> value() const;
  [[nodiscard]] std::expected<void, Error> commit_append(
      std::uint64_t position) const;

 private:
  std::uint64_t* value_{nullptr};
  std::uint64_t capacity_{0};
};

class ConvCursorSlot {
 public:
  ConvCursorSlot() = default;

  [[nodiscard]] static std::expected<ConvCursorSlot, Error> bind(
      std::uint32_t* value);
  [[nodiscard]] std::expected<std::uint32_t, Error> value() const;
  [[nodiscard]] std::expected<void, Error> commit_advance(
      std::uint32_t cursor) const;
  [[nodiscard]] std::expected<void, Error> commit_chunk(
      std::uint32_t cursor, std::uint32_t count) const;

 private:
  std::uint32_t* value_{nullptr};
};

class GdnPositionSlot {
 public:
  GdnPositionSlot() = default;

  [[nodiscard]] static std::expected<GdnPositionSlot, Error> bind(
      std::uint64_t* value);
  [[nodiscard]] std::expected<std::uint64_t, Error> value() const;
  [[nodiscard]] std::expected<void, Error> validate(
      std::uint64_t position) const;
  [[nodiscard]] std::expected<void, Error> commit(
      std::uint64_t position) const;
  [[nodiscard]] std::expected<void, Error> commit_chunk(
      std::uint64_t position, std::uint32_t count) const;

 private:
  std::uint64_t* value_{nullptr};
};

struct SessionSnapshot {
  std::vector<std::byte> gdn_s;
  std::vector<std::byte> conv_history;
  std::vector<std::byte> kv;
  std::array<std::uint32_t, kConvLayers> conv_cursor{};
  std::array<std::uint64_t, kGdnLayers> gdn_position{};
  std::array<std::uint64_t, kAttnLayers> kv_populated{};
  std::uint64_t token_position{};
};

// Host metadata borrowed by bound plans. Shared only so its address and array
// storage remain stable when Session ownership is moved; it does not own CUDA
// tensor storage.
class SessionExecutionState {
 public:
  [[nodiscard]] bool is_poisoned() const noexcept { return poisoned_; }
  void poison() noexcept { poisoned_ = true; }
  void recover() noexcept { poisoned_ = false; }
  [[nodiscard]] std::uint64_t token_position() const noexcept {
    return token_position_;
  }
  void commit_token() noexcept { ++token_position_; }
  [[nodiscard]] bool layers_at(std::uint64_t position) const noexcept {
    for (auto value : gdn_position) {
      if (value != position) return false;
    }
    for (auto value : kv_populated) {
      if (value != position) return false;
    }
    return true;
  }

 private:
  friend class Session;
  std::array<std::uint32_t, kConvLayers> conv_cursor{};
  std::array<std::uint64_t, kGdnLayers> gdn_position{};
  std::array<std::uint64_t, kAttnLayers> kv_populated{};
  std::uint64_t token_position_{};
  bool poisoned_{};
};

class Session {
 public:
  // Plans bound to a Session borrow its allocations and execution metadata.
  // They follow a moved-from source to the destination object. Move-assigning
  // over a destination invalidates its old plans; shutdown/destruction
  // invalidates every plan borrowed from that Session.
  Session(Session&&) noexcept = default;
  Session& operator=(Session&&) noexcept = default;
  ~Session() = default;

  Session(Session const&) = delete;
  Session& operator=(Session const&) = delete;

  // Synchronizes pending work, releases every owned CUDA allocation, and
  // returns the first teardown error. All resources are consumed even when an
  // error is returned. The destructor and move assignment perform the same
  // cleanup best-effort because they cannot report failures.
  [[nodiscard]] std::expected<void, Error> shutdown();

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
  [[nodiscard]] std::expected<std::uint64_t, Error> kv_populated(
      std::uint32_t attention_layer) const;
  [[nodiscard]] std::expected<void, Error> set_populated_length(
      std::uint32_t attention_layer, std::uint64_t populated);

  [[nodiscard]] std::array<std::uint32_t, kConvLayers> const& conv_cursor()
      const noexcept {
    return execution_state_->conv_cursor;
  }
  [[nodiscard]] std::expected<void, Error> set_conv_cursor(
      std::array<std::uint32_t, kConvLayers> cursor);

  [[nodiscard]] ArenaPlan const& arena_plan() const noexcept { return arena_; }
  [[nodiscard]] std::uint64_t persistent_bytes() const noexcept {
    return persistent_bytes_;
  }

 private:
  friend class Runtime;
  friend struct detail::SessionPlanAccess;

  Session() = default;

  static std::expected<Session, Error> create(Model const& model,
                                              std::shared_ptr<qw38::cuda::Stream> stream,
                                              std::uint64_t kv_capacity);

  [[nodiscard]] std::expected<void, Error> zero_persistent();
  [[nodiscard]] std::expected<void, Error> validate_metadata() const;
  [[nodiscard]] std::expected<KvPopulatedSlot, Error> kv_populated_slot(
      std::uint32_t attention_layer);
  [[nodiscard]] std::expected<ConvCursorSlot, Error> conv_cursor_slot(
      std::uint32_t gdn_layer);
  [[nodiscard]] std::expected<GdnPositionSlot, Error> gdn_position_slot(
      std::uint32_t gdn_layer);

  std::shared_ptr<qw38::cuda::Stream> stream_;
  qw38::cuda::DeviceBuffer gdn_s_;
  qw38::cuda::DeviceBuffer conv_history_;
  qw38::cuda::DeviceBuffer kv_;
  qw38::cuda::DeviceBuffer residual_h_;
  qw38::cuda::DeviceBuffer residual_h_mid_;
  qw38::cuda::DeviceBuffer scratch_;
  ArenaPlan arena_{};
  std::array<std::uint32_t, kConvLayers> conv_cursor_{};
  std::array<std::uint64_t, kGdnLayers> gdn_position_{};
  std::uint64_t kv_capacity_{0};
  std::uint64_t persistent_bytes_{0};
  std::shared_ptr<SessionExecutionState> execution_state_;
};

namespace detail {

struct SessionPlanAccess {
  [[nodiscard]] static qw38::cuda::Stream const* stream(
      Session const& session) noexcept;
  [[nodiscard]] static SessionExecutionState* execution_state(
      Session& session) noexcept;
  [[nodiscard]] static SessionExecutionState const* execution_state(
      Session const& session) noexcept;
  [[nodiscard]] static std::expected<KvPopulatedSlot, Error> kv_populated(
      Session& session, std::uint32_t attention_layer);
  [[nodiscard]] static std::expected<ConvCursorSlot, Error> conv_cursor(
      Session& session, std::uint32_t gdn_layer);
  [[nodiscard]] static std::expected<GdnPositionSlot, Error> gdn_position(
      Session& session, std::uint32_t gdn_layer);
};

}  // namespace detail

}  // namespace qw38::runtime
