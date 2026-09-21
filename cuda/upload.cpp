#include "cuda/upload.hpp"

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
  auto buffer = DeviceBuffer::allocate(host.size());
  if (!buffer) {
    return std::unexpected(buffer.error());
  }
  if (auto st = copy_h2d(buffer->data(), host, stream); !st) {
    return std::unexpected(st.error());
  }
  return buffer;
}

}  // namespace qw38::cuda
