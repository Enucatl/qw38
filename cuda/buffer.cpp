#include "cuda/buffer.hpp"

#include "cuda/alloc.hpp"

namespace qw38::cuda {

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    destroy();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() { destroy(); }

void DeviceBuffer::destroy() noexcept {
  if (ptr_ != nullptr) {
    cudaFree(ptr_);
    record_free(bytes_);
    ptr_ = nullptr;
    bytes_ = 0;
  }
}

std::expected<DeviceBuffer, Error> DeviceBuffer::allocate(std::uint64_t bytes) {
  if (bytes == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "DeviceBuffer::allocate",
                                      "zero-byte device allocation"));
  }
  void* ptr = nullptr;
  if (auto st = check(cudaMalloc(&ptr, static_cast<std::size_t>(bytes)),
                      "cudaMalloc");
      !st) {
    return std::unexpected(st.error());
  }
  record_malloc(bytes);
  return DeviceBuffer{ptr, bytes};
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
  return check(cudaMemsetAsync(ptr, 0, static_cast<std::size_t>(bytes),
                               stream.native()),
               "cudaMemsetAsync");
}

std::expected<void, Error> zero(DeviceBuffer& buffer, Stream const& stream) {
  if (buffer.empty()) {
    return {};
  }
  return zero(buffer.data(), buffer.bytes(), stream);
}

}  // namespace qw38::cuda
