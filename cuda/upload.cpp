#include "cuda/upload.hpp"
#include "src/runtime/profiling.hpp"

namespace qw38::cuda {

std::expected<DeviceBuffer, Error> upload(std::span<std::byte const> host,
                                          Stream const& stream) {
  if (host.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "upload",
                                      "empty host payload"));
  }
  if (stream.empty()) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "upload", "empty stream"));
  }
  qw38::runtime::profiling::ScopedRange allocation("qw38:host name=weight_allocate");
  auto buffer = DeviceBuffer::allocate(host.size(), stream.device());
  if (!buffer) {
    return std::unexpected(buffer.error());
  }
  allocation.close();
  qw38::runtime::profiling::ScopedRange transfer("qw38:host name=weight_upload");
  if (auto st = copy_h2d(buffer->data(), host, stream); !st) {
    return std::unexpected(st.error());
  }
  return buffer;
}

}  // namespace qw38::cuda
