#pragma once

#include "runtime/error.hpp"
#include "runtime/model.hpp"
#include "runtime/session.hpp"

#include "cuda/stream.hpp"
#include "format/reader.hpp"

#include <expected>
#include <filesystem>

namespace qw38::runtime {

class Runtime {
 public:
  Runtime(Runtime&&) noexcept = default;
  Runtime& operator=(Runtime&&) noexcept = default;
  ~Runtime() = default;

  Runtime(Runtime const&) = delete;
  Runtime& operator=(Runtime const&) = delete;

  [[nodiscard]] static std::expected<Runtime, Error> create();

  [[nodiscard]] std::expected<Model, Error> load(std::filesystem::path const& path);
  [[nodiscard]] std::expected<Model, Error> upload(
      qw38::format::Artifact const& artifact);
  [[nodiscard]] std::expected<Session, Error> create_session(
      Model const& model, std::uint64_t kv_capacity);

  [[nodiscard]] qw38::cuda::Stream const& stream() const noexcept {
    return stream_;
  }
  [[nodiscard]] int device() const noexcept { return device_; }

 private:
  Runtime() = default;

  qw38::cuda::Stream stream_;
  int device_{0};
};

}  // namespace qw38::runtime
