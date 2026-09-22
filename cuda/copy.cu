#include "cuda/copy.hpp"

#include <algorithm>

namespace qw38::cuda {
namespace {

__global__ void fill_pattern_kernel(std::uint8_t* dst, std::uint64_t n,
                                    std::uint8_t seed) {
  std::uint64_t const stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
       i < n; i += stride) {
    dst[i] = static_cast<std::uint8_t>(seed + static_cast<std::uint8_t>(i));
  }
}

std::expected<void, Error> copy_kind(void* dst, void const* src,
                                     std::uint64_t bytes, Stream const& stream,
                                     cudaMemcpyKind kind,
                                     std::string_view op) {
  if (bytes == 0) {
    return {};
  }
  if (dst == nullptr || src == nullptr || stream.empty()) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "empty pointer or stream"));
  }
  auto guard = stream.activate();
  if (!guard) {
    return std::unexpected(guard.error());
  }
  return check(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(bytes), kind,
                               stream.native()),
               op);
}

}  // namespace

std::expected<void, Error> copy_h2d(void* dst, void const* src,
                                    std::uint64_t bytes, Stream const& stream) {
  return copy_kind(dst, src, bytes, stream, cudaMemcpyHostToDevice, "copy_h2d");
}

std::expected<void, Error> copy_d2h(void* dst, void const* src,
                                    std::uint64_t bytes, Stream const& stream) {
  return copy_kind(dst, src, bytes, stream, cudaMemcpyDeviceToHost, "copy_d2h");
}

std::expected<void, Error> copy_d2d(void* dst, void const* src,
                                    std::uint64_t bytes, Stream const& stream) {
  return copy_kind(dst, src, bytes, stream, cudaMemcpyDeviceToDevice, "copy_d2d");
}

std::expected<void, Error> copy_h2d(void* dst, std::span<std::byte const> src,
                                    Stream const& stream) {
  return copy_h2d(dst, src.data(), src.size(), stream);
}

std::expected<void, Error> copy_d2h(std::span<std::byte> dst, void const* src,
                                    Stream const& stream) {
  return copy_d2h(dst.data(), src, dst.size(), stream);
}

std::expected<void, Error> fill_pattern(void* dst, std::uint64_t bytes,
                                        std::uint8_t seed,
                                        Stream const& stream) {
  if (bytes == 0) {
    return {};
  }
  if (dst == nullptr || stream.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "fill_pattern",
                                      "empty pointer or stream"));
  }
  auto guard = stream.activate();
  if (!guard) {
    return std::unexpected(guard.error());
  }
  int const threads = 256;
  unsigned int const max_blocks = 65535u;
  auto blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(max_blocks, (bytes + threads - 1) / threads));
  if (blocks == 0) {
    blocks = 1;
  }
  fill_pattern_kernel<<<blocks, threads, 0, stream.native()>>>(
      static_cast<std::uint8_t*>(dst), bytes, seed);
  return check(cudaGetLastError(), "fill_pattern_kernel");
}

}  // namespace qw38::cuda
