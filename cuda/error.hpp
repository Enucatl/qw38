#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <expected>
#include <source_location>
#include <string>
#include <string_view>

namespace qw38::cuda {

enum class ErrorCode : std::uint16_t {
  Status = 1,
  InvalidArgument = 2,
  Overflow = 3,
};

struct Error {
  ErrorCode code{ErrorCode::Status};
  int cuda_status{static_cast<int>(cudaSuccess)};
  std::string operation;
  std::string detail;

  friend bool operator==(Error const&, Error const&) = default;
};

[[nodiscard]] char const* code_name(ErrorCode code) noexcept;

[[nodiscard]] Error from_status(
    cudaError_t status, std::string_view operation,
    std::source_location loc = std::source_location::current());

[[nodiscard]] Error make_error(ErrorCode code, std::string_view operation,
                               std::string_view detail);

[[nodiscard]] std::string error_message(Error const& error);

[[nodiscard]] inline std::expected<void, Error> check(
    cudaError_t status, std::string_view operation,
    std::source_location loc = std::source_location::current()) {
  if (status == cudaSuccess) {
    return {};
  }
  // Runtime APIs both return the status and store it as the sticky last error.
  // Consume it here so later kernel launches do not observe a stale failure.
  (void)cudaGetLastError();
  return std::unexpected(from_status(status, operation, loc));
}

}  // namespace qw38::cuda
