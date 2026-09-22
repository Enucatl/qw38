#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::cuda {

class DeviceBuffer {
 public:
  DeviceBuffer() noexcept = default;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer();

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  [[nodiscard]] static std::expected<DeviceBuffer, Error> allocate(
      std::uint64_t bytes);
  [[nodiscard]] static std::expected<DeviceBuffer, Error> allocate(
      std::uint64_t bytes, int device);

  [[nodiscard]] void* data() noexcept { return ptr_; }
  [[nodiscard]] void const* data() const noexcept { return ptr_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr; }
  [[nodiscard]] int device() const noexcept { return device_; }

  [[nodiscard]] std::byte* as_bytes() noexcept {
    return static_cast<std::byte*>(ptr_);
  }
  [[nodiscard]] std::byte const* as_bytes() const noexcept {
    return static_cast<std::byte const*>(ptr_);
  }

 private:
  DeviceBuffer(void* ptr, std::uint64_t bytes, int device) noexcept
      : ptr_(ptr), bytes_(bytes), device_(device) {}
  void destroy() noexcept;

  void* ptr_{nullptr};
  std::uint64_t bytes_{0};
  int device_{-1};
};

[[nodiscard]] std::expected<void, Error> zero(DeviceBuffer& buffer,
                                              Stream const& stream);
[[nodiscard]] std::expected<void, Error> zero(void* ptr, std::uint64_t bytes,
                                              Stream const& stream);

}  // namespace qw38::cuda
