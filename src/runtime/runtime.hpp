#pragma once

#include "runtime/error.hpp"
#include "runtime/model.hpp"
#include "runtime/session.hpp"

#include "cuda/stream.hpp"
#include "format/reader.hpp"

#include <expected>
#include <filesystem>
#include <memory>

namespace qw38::runtime {

class Runtime {
 public:
  Runtime(Runtime&& other) noexcept
      : stream_(other.stream_), device_(other.device_) {}
  Runtime& operator=(Runtime&& other) noexcept {
    stream_ = other.stream_;
    device_ = other.device_;
    return *this;
  }
  ~Runtime() = default;

  Runtime(Runtime const&) = delete;
  Runtime& operator=(Runtime const&) = delete;

  [[nodiscard]] static std::expected<Runtime, Error> create();

  // Synchronizes and destroys the stream with error reporting. Shutdown is
  // rejected while a Session or moved Runtime still shares the stream.
  // Ordinary destruction only drops this owner's reference; final CUDA
  // teardown is best-effort and non-throwing.
  [[nodiscard]] std::expected<void, Error> shutdown();

  [[nodiscard]] std::expected<Model, Error> load(std::filesystem::path const& path);
  [[nodiscard]] std::expected<Model, Error> upload(
      qw38::format::Artifact const& artifact);
  [[nodiscard]] std::expected<Session, Error> create_session(
      Model const& model, std::uint64_t kv_capacity);

  [[nodiscard]] qw38::cuda::Stream const& stream() const noexcept {
    return *stream_;
  }
  [[nodiscard]] int device() const noexcept { return device_; }

 private:
  Runtime() = default;

  std::shared_ptr<qw38::cuda::Stream> stream_;
  int device_{0};
};

}  // namespace qw38::runtime
