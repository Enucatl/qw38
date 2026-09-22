#include "cuda/buffer.hpp"

#include "cuda/alloc.hpp"
#include "cuda/device.hpp"

namespace qw38::cuda {

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_), device_(other.device_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
  other.device_ = -1;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    (void)destroy();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    device_ = other.device_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
    other.device_ = -1;
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() { (void)destroy(); }

cudaError_t DeviceBuffer::destroy() noexcept {
  if (ptr_ == nullptr) {
    return cudaSuccess;
  }

  int previous = 0;
  bool const restore =
      cudaGetDevice(&previous) == cudaSuccess && previous != device_ &&
      cudaSetDevice(device_) == cudaSuccess;
  cudaError_t const status = cudaFree(ptr_);
  if (restore) {
    (void)cudaSetDevice(previous);
  }
  if (status == cudaSuccess) {
    record_free(bytes_);
  }
  ptr_ = nullptr;
  bytes_ = 0;
  device_ = -1;
  return status;
}

std::expected<void, Error> DeviceBuffer::close() {
  return check(destroy(), "cudaFree");
}

std::expected<DeviceBuffer, Error> DeviceBuffer::allocate(std::uint64_t bytes) {
  int device = 0;
  if (auto st = check(cudaGetDevice(&device), "cudaGetDevice"); !st) {
    return std::unexpected(st.error());
  }
  return allocate(bytes, device);
}

std::expected<DeviceBuffer, Error> DeviceBuffer::allocate(std::uint64_t bytes,
                                                           int device) {
  if (bytes == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "DeviceBuffer::allocate",
                                      "zero-byte device allocation"));
  }
  auto guard = DeviceGuard::activate(device, "cudaSetDevice(buffer.allocate)");
  if (!guard) {
    return std::unexpected(guard.error());
  }
  void* ptr = nullptr;
  if (auto st = check(cudaMalloc(&ptr, static_cast<std::size_t>(bytes)),
                      "cudaMalloc");
      !st) {
    return std::unexpected(st.error());
  }
  record_malloc(bytes);
  return DeviceBuffer{ptr, bytes, device};
}

std::expected<void, Error> zero(void* ptr, std::uint64_t bytes,
                                Stream const& stream) {
  if (bytes == 0) {
    return {};
  }
  if (ptr == nullptr || stream.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "zero",
                                      "empty pointer or stream"));
  }
  auto guard = stream.activate();
  if (!guard) {
    return std::unexpected(guard.error());
  }
  return check(cudaMemsetAsync(ptr, 0, static_cast<std::size_t>(bytes),
                               stream.native()),
               "cudaMemsetAsync");
}

std::expected<void, Error> zero(DeviceBuffer& buffer, Stream const& stream) {
  if (buffer.empty()) {
    return {};
  }
  if (buffer.device() != stream.device()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "zero",
                                      "buffer and stream devices differ"));
  }
  return zero(buffer.data(), buffer.bytes(), stream);
}

}  // namespace qw38::cuda
