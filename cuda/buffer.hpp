#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"
#include "src/runtime/profiling.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>

namespace qw38::cuda {

class HostBuffer {
 public:
  HostBuffer() = default;
  HostBuffer(HostBuffer&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
  HostBuffer& operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) (void)cudaFreeHost(ptr_);
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }
  ~HostBuffer() { if (ptr_) (void)cudaFreeHost(ptr_); }
  HostBuffer(HostBuffer const&) = delete;
  HostBuffer& operator=(HostBuffer const&) = delete;
  static std::expected<HostBuffer, Error> allocate(std::size_t bytes) {
    qw38::runtime::profiling::ScopedRange range(
        "qw38:host name=pinned_buffer_allocate bytes=%llu",
        static_cast<unsigned long long>(bytes));
    HostBuffer result;
    if (auto st = check(cudaMallocHost(&result.ptr_, bytes), "cudaMallocHost"); !st)
      return std::unexpected(st.error());
    return result;
  }
  void* data() noexcept { return ptr_; }
 private:
  void* ptr_{};
};

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

  // Fallible teardown for callers that need diagnostics. Ownership is consumed
  // even on failure because CUDA may have released the allocation while
  // reporting an earlier asynchronous error. The destructor uses the same
  // teardown path but necessarily discards its status.
  [[nodiscard]] std::expected<void, Error> close();

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
  [[nodiscard]] cudaError_t destroy() noexcept;

  void* ptr_{nullptr};
  std::uint64_t bytes_{0};
  int device_{-1};
};

[[nodiscard]] std::expected<void, Error> zero(DeviceBuffer& buffer,
                                              Stream const& stream);
[[nodiscard]] std::expected<void, Error> zero(void* ptr, std::uint64_t bytes,
                                              Stream const& stream);

}  // namespace qw38::cuda
